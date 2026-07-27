// entropy_kv_cache.h - Entropy-Adaptive KV Cache Compression for BitNet/llama.cpp
//
// Per-head budget allocation based on pre-computed attention entropy profiles.
// Low-entropy heads (sink/focused) get fewer KV slots; high-entropy heads get more.
// Always preserves the attention sink (position 0).
//
// Usage:
//   1. Calibrate entropy with Python script (one-time): calibrate_entropy.py
//   2. Load config at startup: entropy_kv_config_load()
//   3. After each decode step: entropy_kv_evict()
//
// Reference: "Entropy-Adaptive KV Cache Compression" (2026)

#ifndef ENTROPY_KV_CACHE_H
#define ENTROPY_KV_CACHE_H

#include "llama.h"

#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------

struct entropy_kv_head_info {
    int      layer;          // layer index
    int      head;           // head index within layer
    float    entropy;        // measured attention entropy (bits)
    float    scale;          // computed scale factor (entropy / mean_entropy, clamped)
    int32_t  budget;         // max KV entries for this head at current compression
};

struct entropy_kv_config {
    // Model geometry (filled from config file or model params)
    int n_layer;
    int n_head;              // query heads per layer
    int n_head_kv;           // KV heads per layer (may differ for GQA)
    int n_ctx;               // max context length

    // Compression settings
    float    keep_ratio;     // global retention ratio (0.0-1.0), e.g. 0.5 for 2x compression
    bool     keep_sink;      // always keep position 0 (strongly recommended)
    float    scale_min;      // minimum scale factor (default 0.3)
    float    scale_max;      // maximum scale factor (default 2.5)

    // Per-KV-head entropy data
    // Indexed as: kv_head_entropies[layer * n_head_kv + kv_head]
    std::vector<float> kv_head_entropies;

    // Derived: per-KV-head budgets (computed from entropies + keep_ratio)
    // Indexed as: kv_head_budgets[layer * n_head_kv + kv_head]
    std::vector<int32_t> kv_head_budgets;

    // Derived: per-position importance scores for eviction decisions
    // These track cumulative attention importance per position
    // Indexed as: position_importance[pos]
    std::vector<float> position_importance;

    // Statistics
    float entropy_mean;
    float entropy_min;
    float entropy_max;
    float entropy_cv;        // coefficient of variation
    int   total_budget;      // sum of all per-head budgets

    // State
    bool loaded;
};

// --------------------------------------------------------------------------
// Core API
// --------------------------------------------------------------------------

// Load entropy config from a JSON file produced by calibrate_entropy.py.
// Returns true on success, false on error (logged to stderr).
//
// The JSON file format:
// {
//   "model": "model_name",
//   "n_layer": 24,
//   "n_head": 16,
//   "n_head_kv": 16,
//   "head_entropies": {
//     "L0_H0": 0.144, "L0_H1": 0.103, ...
//   },
//   "entropy_stats": {
//     "mean": 1.54, "min": 0.015, "max": 3.39, "cv": 0.55
//   }
// }
bool entropy_kv_config_load(
    entropy_kv_config & config,
    const char        * json_path,
    float               keep_ratio = 0.5f,
    bool                keep_sink  = true,
    float               scale_min  = 0.3f,
    float               scale_max  = 2.5f);

// Initialize config from raw entropy values (no file needed).
// Useful when entropy values are embedded in the application.
//
// entropies: array of size n_layer * n_head_kv, indexed [layer * n_head_kv + head]
bool entropy_kv_config_init(
    entropy_kv_config & config,
    int                 n_layer,
    int                 n_head,
    int                 n_head_kv,
    int                 n_ctx,
    const float       * entropies,
    int                 n_entropies,
    float               keep_ratio = 0.5f,
    bool                keep_sink  = true,
    float               scale_min  = 0.3f,
    float               scale_max  = 2.5f);

// Compute per-head budgets from entropies and compression settings.
// Called automatically by config_load/config_init, but can be called again
// if you change keep_ratio or scale bounds.
void entropy_kv_compute_budgets(entropy_kv_config & config);

// Get the budget for a specific KV head.
int32_t entropy_kv_head_budget(const entropy_kv_config & config, int layer, int kv_head);

// --------------------------------------------------------------------------
// Eviction API (position-level, works with llama.cpp's flat KV cache)
// --------------------------------------------------------------------------

// Perform entropy-adaptive eviction on the KV cache.
//
// This is the main integration point. Call after each llama_decode() when
// the number of used KV cells exceeds the desired budget.
//
// Strategy:
//   1. Compute per-position importance as the max entropy-weighted attention
//      across all heads (positions important to high-entropy heads are kept)
//   2. Sort positions by importance
//   3. Keep the top-N positions where N = total_budget
//   4. Always keep the sink position (pos=0)
//   5. Evict remaining positions using llama_kv_cache_seq_rm()
//
// Parameters:
//   ctx       - llama context
//   config    - entropy config with computed budgets
//   seq_id    - sequence ID to evict from (default 0)
//
// Returns the number of positions evicted, or -1 on error.
int entropy_kv_evict(
    struct llama_context      * ctx,
    const entropy_kv_config   & config,
    llama_seq_id                seq_id = 0);

// Check if eviction is needed (KV usage exceeds budget).
bool entropy_kv_should_evict(
    struct llama_context      * ctx,
    const entropy_kv_config   & config);

// --------------------------------------------------------------------------
// Position importance tracking (for advanced eviction strategies)
// --------------------------------------------------------------------------

// Update position importance scores based on attention patterns.
// This is optional - the basic eviction uses position-based heuristics.
//
// In a full implementation, this would be called with actual attention weights
// extracted from the model. For the pre-computed budget approach, we use
// position-based heuristics instead:
//   - Position 0: infinite importance (sink)
//   - Recent positions: high importance (recency bias)
//   - Other positions: moderate importance (decays with distance)
//
// The per-head entropy budgets determine HOW MANY positions to keep,
// while the importance scores determine WHICH positions to keep.
void entropy_kv_update_importance(
    entropy_kv_config         & config,
    int                         current_pos,
    int                         n_ctx_used);

// --------------------------------------------------------------------------
// Utility
// --------------------------------------------------------------------------

// Print config summary to stderr (for debugging).
void entropy_kv_config_print(const entropy_kv_config & config);

// Get the effective compression ratio given current usage and budget.
float entropy_kv_compression_ratio(
    const entropy_kv_config & config,
    int                       n_ctx_used);

// Generate a default config with uniform entropy (fallback when no calibration
// data is available). All heads get equal budgets.
void entropy_kv_config_uniform(
    entropy_kv_config & config,
    int                 n_layer,
    int                 n_head,
    int                 n_head_kv,
    int                 n_ctx,
    float               keep_ratio = 0.5f);

#endif // ENTROPY_KV_CACHE_H
