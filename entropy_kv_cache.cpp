// entropy_kv_cache.cpp - Entropy-Adaptive KV Cache Compression for BitNet/llama.cpp
//
// Implementation of per-head entropy-adaptive KV cache eviction.
// See entropy_kv_cache.h for API documentation.

#include "entropy_kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// We need llama.h for the public API (llama_kv_cache_seq_rm, etc.)
#include "llama.h"

// --------------------------------------------------------------------------
// JSON parsing (minimal, no external dependency)
// --------------------------------------------------------------------------

// Simple JSON value extraction - handles the specific format from calibrate_entropy.py
// This avoids adding a JSON library dependency to llama.cpp.

static std::string read_file_to_string(const char * path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        return "";
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Extract a quoted string value for a given key
static bool json_get_string(const std::string & json, const char * key, std::string & out) {
    std::string search = std::string("\"") + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;

    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return false;

    // skip whitespace and opening quote
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return false;
    pos++; // skip opening quote

    size_t end = json.find('"', pos);
    if (end == std::string::npos) return false;

    out = json.substr(pos, end - pos);
    return true;
}

// Extract an integer value for a given key
static bool json_get_int(const std::string & json, const char * key, int & out) {
    std::string search = std::string("\"") + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;

    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return false;
    pos++; // skip colon

    // skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) {
        pos++;
    }

    char * end = nullptr;
    long val = strtol(json.c_str() + pos, &end, 10);
    if (end == json.c_str() + pos) return false;

    out = (int)val;
    return true;
}

// Extract a float value for a given key
static bool json_get_float(const std::string & json, const char * key, float & out) {
    std::string search = std::string("\"") + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;

    pos = json.find(':', pos + search.length());
    if (pos == std::string::npos) return false;
    pos++; // skip colon

    // skip whitespace
    while (pos < json.length() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n' || json[pos] == '\r')) {
        pos++;
    }

    char * end = nullptr;
    double val = strtod(json.c_str() + pos, &end);
    if (end == json.c_str() + pos) return false;

    out = (float)val;
    return true;
}

// Extract a sub-object as a string for a given key
static bool json_get_object(const std::string & json, const char * key, std::string & out) {
    std::string search = std::string("\"") + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return false;

    pos = json.find('{', pos + search.length());
    if (pos == std::string::npos) return false;

    // find matching closing brace
    int depth = 1;
    size_t end = pos + 1;
    while (end < json.length() && depth > 0) {
        if (json[end] == '{') depth++;
        if (json[end] == '}') depth--;
        end++;
    }
    if (depth != 0) return false;

    out = json.substr(pos, end - pos);
    return true;
}

// Parse head_entropies object: {"L0_H0": 0.144, "L0_H1": 0.103, ...}
static bool parse_head_entropies(
    const std::string                        & obj,
    std::unordered_map<std::string, float>   & entropies)
{
    size_t pos = 0;
    while (pos < obj.length()) {
        // find key
        size_t key_start = obj.find('"', pos);
        if (key_start == std::string::npos) break;
        key_start++;
        size_t key_end = obj.find('"', key_start);
        if (key_end == std::string::npos) break;

        std::string key = obj.substr(key_start, key_end - key_start);

        // find value
        size_t colon = obj.find(':', key_end);
        if (colon == std::string::npos) break;

        char * end = nullptr;
        double val = strtod(obj.c_str() + colon + 1, &end);
        if (end == obj.c_str() + colon + 1) break;

        entropies[key] = (float)val;
        pos = (size_t)(end - obj.c_str());
    }

    return !entropies.empty();
}

// --------------------------------------------------------------------------
// Config initialization
// --------------------------------------------------------------------------

static float clamp_f(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

void entropy_kv_compute_budgets(entropy_kv_config & config) {
    const int n_kv_heads_total = config.n_layer * config.n_head_kv;

    if (config.kv_head_entropies.empty() || n_kv_heads_total == 0) {
        return;
    }

    // Compute entropy statistics
    float sum = 0.0f;
    config.entropy_min = config.kv_head_entropies[0];
    config.entropy_max = config.kv_head_entropies[0];

    for (int i = 0; i < n_kv_heads_total; i++) {
        float e = config.kv_head_entropies[i];
        sum += e;
        if (e < config.entropy_min) config.entropy_min = e;
        if (e > config.entropy_max) config.entropy_max = e;
    }

    config.entropy_mean = sum / (float)n_kv_heads_total;

    // Compute CV
    float var_sum = 0.0f;
    for (int i = 0; i < n_kv_heads_total; i++) {
        float diff = config.kv_head_entropies[i] - config.entropy_mean;
        var_sum += diff * diff;
    }
    float std_dev = sqrtf(var_sum / (float)n_kv_heads_total);
    config.entropy_cv = (config.entropy_mean > 1e-10f) ? (std_dev / config.entropy_mean) : 0.0f;

    // Compute per-head budgets
    // Budget formula: head_budget = base_budget * clamp(entropy / mean_entropy, scale_min, scale_max)
    // base_budget = n_ctx * keep_ratio
    const float base_budget = (float)config.n_ctx * config.keep_ratio;

    config.kv_head_budgets.resize(n_kv_heads_total);
    config.total_budget = 0;

    for (int i = 0; i < n_kv_heads_total; i++) {
        float scale = 1.0f;
        if (config.entropy_mean > 1e-10f) {
            scale = config.kv_head_entropies[i] / config.entropy_mean;
        }
        scale = clamp_f(scale, config.scale_min, config.scale_max);

        int32_t budget = (int32_t)(base_budget * scale);

        // Clamp to [1, n_ctx]: every head gets at least 1 entry (the sink)
        budget = std::max(budget, (int32_t)1);
        budget = std::min(budget, (int32_t)config.n_ctx);

        config.kv_head_budgets[i] = budget;
        config.total_budget += budget;
    }

    // For position-level eviction (since llama.cpp KV cache is flat per-layer),
    // we compute an effective per-position budget as the average across heads.
    // The actual eviction strategy uses position importance scoring.
}

bool entropy_kv_config_init(
    entropy_kv_config & config,
    int                 n_layer,
    int                 n_head,
    int                 n_head_kv,
    int                 n_ctx,
    const float       * entropies,
    int                 n_entropies,
    float               keep_ratio,
    bool                keep_sink,
    float               scale_min,
    float               scale_max)
{
    config.n_layer    = n_layer;
    config.n_head     = n_head;
    config.n_head_kv  = n_head_kv;
    config.n_ctx      = n_ctx;
    config.keep_ratio = keep_ratio;
    config.keep_sink  = keep_sink;
    config.scale_min  = scale_min;
    config.scale_max  = scale_max;
    config.loaded     = false;

    const int expected = n_layer * n_head_kv;
    if (n_entropies != expected) {
        fprintf(stderr, "entropy_kv: expected %d entropy values (n_layer=%d * n_head_kv=%d), got %d\n",
                expected, n_layer, n_head_kv, n_entropies);
        return false;
    }

    config.kv_head_entropies.assign(entropies, entropies + n_entropies);
    entropy_kv_compute_budgets(config);
    config.loaded = true;

    return true;
}

bool entropy_kv_config_load(
    entropy_kv_config & config,
    const char        * json_path,
    float               keep_ratio,
    bool                keep_sink,
    float               scale_min,
    float               scale_max)
{
    config.loaded = false;

    std::string json = read_file_to_string(json_path);
    if (json.empty()) {
        fprintf(stderr, "entropy_kv: failed to read config file: %s\n", json_path);
        return false;
    }

    // Parse model info
    int n_layer = 0, n_head = 0, n_head_kv = 0;
    if (!json_get_int(json, "n_layer", n_layer)) {
        // Try nested under model_info
        std::string model_info;
        if (json_get_object(json, "model_info", model_info)) {
            json_get_int(model_info, "num_hidden_layers", n_layer);
            json_get_int(model_info, "num_attention_heads", n_head);
            json_get_int(model_info, "num_key_value_heads", n_head_kv);
        }
    } else {
        json_get_int(json, "n_head", n_head);
        if (!json_get_int(json, "n_head_kv", n_head_kv)) {
            n_head_kv = n_head; // MHA: n_head_kv == n_head
        }
    }

    if (n_layer <= 0 || n_head <= 0 || n_head_kv <= 0) {
        fprintf(stderr, "entropy_kv: could not parse model geometry from %s\n", json_path);
        return false;
    }

    // Parse head entropies
    std::string entropies_obj;
    if (!json_get_object(json, "head_entropies", entropies_obj)) {
        fprintf(stderr, "entropy_kv: no 'head_entropies' object in %s\n", json_path);
        return false;
    }

    std::unordered_map<std::string, float> entropy_map;
    if (!parse_head_entropies(entropies_obj, entropy_map)) {
        fprintf(stderr, "entropy_kv: failed to parse head_entropies from %s\n", json_path);
        return false;
    }

    // Convert map to ordered vector
    // For GQA models, we need to map query head entropies to KV heads.
    // Strategy: for each KV head, take the MAX entropy among its grouped query heads.
    const int gqa_ratio = n_head / n_head_kv;
    const int n_kv_total = n_layer * n_head_kv;

    std::vector<float> kv_entropies(n_kv_total, 0.0f);

    for (int l = 0; l < n_layer; l++) {
        for (int kv_h = 0; kv_h < n_head_kv; kv_h++) {
            float max_entropy = 0.0f;
            bool found_any = false;

            // Check all query heads that map to this KV head
            for (int g = 0; g < gqa_ratio; g++) {
                int q_head = kv_h * gqa_ratio + g;
                char key[64];
                snprintf(key, sizeof(key), "L%d_H%d", l, q_head);

                auto it = entropy_map.find(key);
                if (it != entropy_map.end()) {
                    max_entropy = std::max(max_entropy, it->second);
                    found_any = true;
                }
            }

            if (!found_any) {
                // Try direct KV head key format
                char key[64];
                snprintf(key, sizeof(key), "L%d_H%d", l, kv_h);
                auto it = entropy_map.find(key);
                if (it != entropy_map.end()) {
                    max_entropy = it->second;
                    found_any = true;
                }
            }

            if (!found_any) {
                fprintf(stderr, "entropy_kv: missing entropy for layer %d, kv_head %d\n", l, kv_h);
                max_entropy = 1.0f; // fallback to moderate entropy
            }

            kv_entropies[l * n_head_kv + kv_h] = max_entropy;
        }
    }

    // Try to get n_ctx from the config, otherwise use a default
    int n_ctx = 2048;
    json_get_int(json, "n_ctx", n_ctx);

    // Initialize with parsed data
    bool ok = entropy_kv_config_init(
        config, n_layer, n_head, n_head_kv, n_ctx,
        kv_entropies.data(), n_kv_total,
        keep_ratio, keep_sink, scale_min, scale_max);

    if (ok) {
        fprintf(stderr, "entropy_kv: loaded config from %s\n", json_path);
        fprintf(stderr, "entropy_kv: %d layers, %d/%d heads (q/kv), GQA ratio %d\n",
                n_layer, n_head, n_head_kv, gqa_ratio);
        fprintf(stderr, "entropy_kv: entropy range [%.3f, %.3f], mean %.3f, CV %.3f\n",
                config.entropy_min, config.entropy_max, config.entropy_mean, config.entropy_cv);
        fprintf(stderr, "entropy_kv: keep_ratio=%.2f, total_budget=%d (%.1fx compression)\n",
                keep_ratio, config.total_budget,
                (float)(n_layer * n_head_kv * n_ctx) / (float)config.total_budget);
    }

    return ok;
}

void entropy_kv_config_uniform(
    entropy_kv_config & config,
    int                 n_layer,
    int                 n_head,
    int                 n_head_kv,
    int                 n_ctx,
    float               keep_ratio)
{
    const int n_total = n_layer * n_head_kv;
    std::vector<float> uniform_entropies(n_total, 1.0f); // all heads get entropy = 1.0

    entropy_kv_config_init(
        config, n_layer, n_head, n_head_kv, n_ctx,
        uniform_entropies.data(), n_total,
        keep_ratio, true, 0.3f, 2.5f);
}

int32_t entropy_kv_head_budget(const entropy_kv_config & config, int layer, int kv_head) {
    if (!config.loaded) return config.n_ctx;
    int idx = layer * config.n_head_kv + kv_head;
    if (idx < 0 || idx >= (int)config.kv_head_budgets.size()) return config.n_ctx;
    return config.kv_head_budgets[idx];
}

// --------------------------------------------------------------------------
// Eviction
// --------------------------------------------------------------------------

// The key architectural insight: llama.cpp's KV cache is FLAT per layer.
// Each cell stores KV for ALL heads at a given position. You cannot evict
// position P from head H while keeping it for head H'.
//
// This means we cannot do true per-head eviction. Instead, we use the
// entropy profile to compute a POSITION-LEVEL importance score that
// accounts for head heterogeneity:
//
//   importance(pos) = sum_over_heads[ head_entropy_weight * attention_to_pos ]
//
// Since we don't have access to attention weights in the pre-computed approach,
// we use the following heuristic:
//
//   - Position 0 (sink): always keep
//   - Recent positions: high importance (most heads need recent context)
//   - Other positions: importance decays, but the rate depends on the
//     entropy distribution. More high-entropy heads = slower decay.
//
// The total number of positions to keep is determined by the MINIMUM budget
// across all heads in each layer (since we can't evict per-head), adjusted
// upward by the entropy CV (more heterogeneous = keep more to satisfy
// high-entropy heads).
//
// For a more sophisticated approach, see Option B (online entropy measurement)
// in the integration guide, which would track actual attention weights.

struct position_score {
    int32_t pos;
    float   importance;
};

static bool importance_cmp(const position_score & a, const position_score & b) {
    return a.importance > b.importance; // descending
}

// Compute the effective position budget for a layer.
// Since the KV cache is flat, we must keep enough positions to satisfy
// the most demanding (highest-entropy) head in the layer.
static int32_t compute_layer_position_budget(
    const entropy_kv_config & config,
    int                       layer)
{
    if (!config.loaded || config.kv_head_budgets.empty()) {
        return (int32_t)(config.n_ctx * config.keep_ratio);
    }

    // Use the MAX budget among KV heads in this layer.
    // Rationale: since we can't evict per-head, we must keep enough positions
    // for the highest-entropy head to function.
    int32_t max_budget = 0;
    for (int h = 0; h < config.n_head_kv; h++) {
        int idx = layer * config.n_head_kv + h;
        if (idx < (int)config.kv_head_budgets.size()) {
            max_budget = std::max(max_budget, config.kv_head_budgets[idx]);
        }
    }

    return std::max(max_budget, (int32_t)1);
}

// Compute the global position budget (minimum across all layers).
// This is what we use for the flat KV cache eviction.
static int32_t compute_global_position_budget(const entropy_kv_config & config) {
    if (!config.loaded) {
        return (int32_t)(config.n_ctx * config.keep_ratio);
    }

    // For flat KV cache: use the AVERAGE of per-layer max budgets.
    // This is a compromise: some layers need more, some need less.
    // The average gives a reasonable middle ground.
    int64_t sum = 0;
    for (int l = 0; l < config.n_layer; l++) {
        sum += compute_layer_position_budget(config, l);
    }

    int32_t avg_budget = (int32_t)(sum / config.n_layer);
    return std::max(avg_budget, (int32_t)1);
}

bool entropy_kv_should_evict(
    struct llama_context      * ctx,
    const entropy_kv_config   & config)
{
    if (!config.loaded) return false;

    int32_t used = llama_get_kv_cache_used_cells(ctx);
    int32_t budget = compute_global_position_budget(config);

    return used > budget;
}

int entropy_kv_evict(
    struct llama_context      * ctx,
    const entropy_kv_config   & config,
    llama_seq_id                seq_id)
{
    if (!config.loaded) {
        fprintf(stderr, "entropy_kv: config not loaded, skipping eviction\n");
        return -1;
    }

    int32_t n_used = llama_get_kv_cache_used_cells(ctx);
    int32_t budget = compute_global_position_budget(config);

    if (n_used <= budget) {
        return 0; // nothing to evict
    }

    int32_t n_to_evict = n_used - budget;

    // Build position importance scores using entropy-weighted heuristic.
    //
    // For each occupied position, compute:
    //   importance = recency_score + sink_bonus
    //
    // Where recency_score reflects the entropy-weighted value of recent tokens.
    // High-entropy heads need more context, so we bias toward keeping more.
    //
    // The entropy information is encoded in the budget (which determines HOW MANY
    // positions to keep), while the importance scores determine WHICH ones.

    llama_pos pos_max = llama_kv_cache_seq_pos_max(ctx, seq_id);

    if (pos_max <= 0) {
        return 0; // nothing to evict
    }

    // Compute importance for each position
    std::vector<position_score> scores;
    scores.reserve(n_used);

    for (llama_pos p = 0; p <= pos_max; p++) {
        float importance = 0.0f;

        if (config.keep_sink && p == 0) {
            // Sink token: maximum importance, never evict
            importance = 1e10f;
        } else {
            // Recency-based importance with entropy-adaptive decay
            //
            // The decay rate is modulated by the entropy CV:
            // - High CV (diverse heads): slower decay, keep more positions
            //   because high-entropy heads need broader context
            // - Low CV (uniform heads): faster decay, recency dominates
            //
            // importance(p) = exp(-alpha * distance_from_current)
            // where alpha = base_rate / (1 + entropy_cv)

            float distance = (float)(pos_max - p);
            float alpha = 2.0f / ((float)pos_max * (1.0f + config.entropy_cv));
            importance = expf(-alpha * distance);

            // Small boost for very recent tokens (last 10%)
            if (distance < (float)pos_max * 0.1f) {
                importance += 0.5f;
            }
        }

        scores.push_back({p, importance});
    }

    // Sort by importance (descending) and mark the bottom positions for eviction
    std::sort(scores.begin(), scores.end(), importance_cmp);

    // The positions to KEEP are the top 'budget' entries
    // The positions to EVICT are the rest
    int n_evicted = 0;

    for (size_t i = (size_t)budget; i < scores.size(); i++) {
        llama_pos p = scores[i].pos;

        // Never evict the sink
        if (config.keep_sink && p == 0) continue;

        // Evict this position
        bool ok = llama_kv_cache_seq_rm(ctx, seq_id, p, p + 1);
        if (ok) {
            n_evicted++;
        }
    }

    // After eviction, shift positions to fill gaps and defrag
    // This is handled by llama.cpp's built-in defragmentation
    if (n_evicted > 0) {
        llama_kv_cache_defrag(ctx);
    }

    return n_evicted;
}

void entropy_kv_update_importance(
    entropy_kv_config & config,
    int                 current_pos,
    int                 n_ctx_used)
{
    // Resize importance array if needed
    if ((int)config.position_importance.size() < n_ctx_used) {
        config.position_importance.resize(n_ctx_used, 0.0f);
    }

    // Update importance using exponential moving average
    // Recent positions get boosted, older ones decay
    float decay = 0.95f;
    for (int i = 0; i < n_ctx_used; i++) {
        config.position_importance[i] *= decay;
    }

    // Current position gets full importance
    if (current_pos >= 0 && current_pos < n_ctx_used) {
        config.position_importance[current_pos] = 1.0f;
    }

    // Sink always has high importance
    if (config.keep_sink && n_ctx_used > 0) {
        config.position_importance[0] = 1e6f;
    }
}

// --------------------------------------------------------------------------
// Utility
// --------------------------------------------------------------------------

void entropy_kv_config_print(const entropy_kv_config & config) {
    fprintf(stderr, "\n=== Entropy-Adaptive KV Cache Config ===\n");
    fprintf(stderr, "Model: %d layers, %d q-heads, %d kv-heads (GQA ratio %d)\n",
            config.n_layer, config.n_head, config.n_head_kv,
            config.n_head / std::max(config.n_head_kv, 1));
    fprintf(stderr, "Context: %d tokens\n", config.n_ctx);
    fprintf(stderr, "Keep ratio: %.2f (%.1fx compression)\n",
            config.keep_ratio, 1.0f / config.keep_ratio);
    fprintf(stderr, "Keep sink: %s\n", config.keep_sink ? "yes" : "no");
    fprintf(stderr, "Scale bounds: [%.2f, %.2f]\n", config.scale_min, config.scale_max);
    fprintf(stderr, "\nEntropy statistics:\n");
    fprintf(stderr, "  Mean:  %.4f bits\n", config.entropy_mean);
    fprintf(stderr, "  Min:   %.4f bits\n", config.entropy_min);
    fprintf(stderr, "  Max:   %.4f bits\n", config.entropy_max);
    fprintf(stderr, "  CV:    %.4f\n", config.entropy_cv);
    fprintf(stderr, "  Range: %.1fx\n",
            config.entropy_min > 1e-10f ? config.entropy_max / config.entropy_min : 0.0f);
    fprintf(stderr, "\nBudgets (total: %d positions across all heads):\n", config.total_budget);

    // Print per-layer budget summary
    for (int l = 0; l < config.n_layer; l++) {
        int32_t layer_min = config.n_ctx;
        int32_t layer_max = 0;
        float   layer_sum = 0.0f;

        for (int h = 0; h < config.n_head_kv; h++) {
            int idx = l * config.n_head_kv + h;
            int32_t b = config.kv_head_budgets[idx];
            layer_min = std::min(layer_min, b);
            layer_max = std::max(layer_max, b);
            layer_sum += (float)b;
        }

        float layer_avg = layer_sum / (float)config.n_head_kv;
        int32_t pos_budget = compute_layer_position_budget(config, l);

        fprintf(stderr, "  Layer %2d: budget range [%4d, %4d], avg %6.1f, pos_budget %4d\n",
                l, layer_min, layer_max, layer_avg, pos_budget);
    }

    int32_t global_budget = compute_global_position_budget(config);
    fprintf(stderr, "\nGlobal position budget: %d (effective %.1fx compression)\n",
            global_budget, (float)config.n_ctx / (float)global_budget);
    fprintf(stderr, "========================================\n\n");
}

float entropy_kv_compression_ratio(
    const entropy_kv_config & config,
    int                       n_ctx_used)
{
    if (n_ctx_used <= 0) return 1.0f;
    int32_t budget = compute_global_position_budget(config);
    if (budget >= n_ctx_used) return 1.0f;
    return (float)n_ctx_used / (float)budget;
}
