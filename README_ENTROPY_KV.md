# Entropy-Adaptive KV Cache Compression for BitNet

## What This Does

Per-head entropy-based KV cache budget allocation that reduces memory usage
by 2-5x with minimal quality loss. Allocates more cache to attention heads
that need it (high entropy / diffuse attention) and less to heads that don't
(low entropy / focused attention).

## Why It Matters

BitNet compresses weights to 1.58 bits, but stores KV cache activations in
float16. For a 0.7B BitNet model, the KV cache is **68.6% of total memory**.
This compression targets exactly that bottleneck.

## Results

### Quality vs Compression (BitNet-b1.58-large, 0.7B)

| Compression | `--kv-budget` | Exact Match | Memory Saved |
|-------------|---------------|-------------|--------------|
| 2x          | 0.5           | ~96.5%      | 34%          |
| 3.3x        | 0.3           | ~96%        | 48%          |
| 5x          | 0.2           | ~96%        | 55%          |
| 10x         | 0.1           | ~96%        | 62%          |

BitNet models show remarkably flat degradation curves (96.5% to 96.1% from
2x to 10x compression) because their attention patterns are inherently more
focused than standard float models.

### Memory Impact (BitNet-0.7B, 2048 context)

| Compression | KV Memory | Total Memory | Saving |
|-------------|-----------|--------------|--------|
| 1x (none)   | 302 MB    | 440 MB       | --     |
| 2x          | 151 MB    | 289 MB       | 34%    |
| 5x          | 60 MB     | 198 MB       | 55%    |
| 10x         | 30 MB     | 168 MB       | 62%    |

### Entropy Distribution (384 heads across 24 layers)

- **56 sink heads** (entropy < 0.5 bits): Need only 1-3 KV entries
- **135 focused heads** (0.5-1.5 bits): Need modest cache
- **179 moderate heads** (1.5-3.0 bits): Need proportionally more
- **14 mixed heads** (3.0+ bits): Get maximum allocation

## Quick Start

### Step 1: Calibrate (one-time, ~2 minutes)

```bash
python calibrate_entropy.py --model 1bitLLM/bitnet_b1_58-large --output entropy_config.json
```

This runs 20 diverse text sequences through the model and measures the
attention entropy of every head. The output is a JSON file consumed at
inference time.

A pre-generated config for `bitnet_b1_58-large` is included as
`entropy_config.json`.

### Step 2: Build

Add to `CMakeLists.txt` (after `add_subdirectory(src)`):

```cmake
# Entropy-adaptive KV cache compression
add_library(entropy_kv_cache STATIC entropy_kv_cache.cpp)
target_include_directories(entropy_kv_cache PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/llama.cpp/ggml/include
)
target_link_libraries(entropy_kv_cache PRIVATE llama)
```

Then build normally:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

### Step 3: Integrate into inference (4 lines)

```cpp
#include "entropy_kv_cache.h"

// After model/context creation:
entropy_kv_config ekv_config = {};
bool use_ekv = entropy_kv_config_load(ekv_config, "entropy_config.json", 0.5f);
if (use_ekv) {
    ekv_config.n_ctx = llama_n_ctx(ctx);
    entropy_kv_compute_budgets(ekv_config);
}

// In the generation loop, after each llama_decode():
if (use_ekv && entropy_kv_should_evict(ctx, ekv_config)) {
    entropy_kv_evict(ctx, ekv_config, 0);
}
```

### Step 4: Run with compression

```bash
./build/bin/llama-cli \
    -m models/bitnet_b1_58-large/ggml-model-i2_s.gguf \
    -p "The meaning of life is" \
    -n 256 \
    --entropy-config entropy_config.json \
    --kv-budget 0.5
```

## How It Works

### Calibration Phase

1. Feed diverse text through the model with `output_attentions=True`
2. For each head, compute Shannon entropy of the attention distribution:
   `H = -sum(p * log2(p))`
3. Sample at representative positions (25%, 50%, 75%, 100% of sequence)
4. Average across 20 calibration sequences
5. Save per-head entropy values to JSON

### Inference Phase

1. Load entropy config and compute per-head budgets:
   `budget(head) = base_budget * clamp(entropy / mean_entropy, 0.3, 2.5)`
2. When KV cache usage exceeds the global position budget, trigger eviction
3. Score each position by importance (recency-based with entropy-adaptive decay)
4. Always preserve position 0 (attention sink -- removing it crashes quality)
5. Evict lowest-importance positions via `llama_kv_cache_seq_rm()`
6. Defragment the cache via `llama_kv_cache_defrag()`

### Key Design Decisions

**Position-level eviction**: llama.cpp stores all heads at the same positions
(flat KV cache). We cannot evict per-head, so the budget is set by the most
demanding head in each layer. The entropy profile determines HOW MANY positions
to keep; importance scoring determines WHICH ones.

**Pre-computed entropy**: Entropy profiles are stable across inputs (r=0.975
correlation), so pre-computation adds zero runtime overhead. For very long
contexts (32K+) or out-of-distribution inputs, online entropy measurement
would be more adaptive.

**Sink preservation**: Removing position 0 crashes quality from 69% to 6.6%.
The sink token is always retained.

## Files

| File | Purpose |
|------|---------|
| `entropy_kv_cache.h` | C++ header with public API |
| `entropy_kv_cache.cpp` | Implementation (~690 lines) |
| `calibrate_entropy.py` | Python calibration script (~430 lines) |
| `entropy_config.json` | Pre-generated config for bitnet_b1_58-large |
| `README_ENTROPY_KV.md` | This file |

## Generating Configs for Other Models

```bash
# BitNet 2B (GQA, 20 query heads / 5 KV heads)
python calibrate_entropy.py --model microsoft/bitnet-b1.58-2B-4T

# Any HuggingFace model (for comparison)
python calibrate_entropy.py --model meta-llama/Llama-2-7b-hf --device cuda

# From existing calibration results
python calibrate_entropy.py --from-results results.json --output entropy_config.json
```

## API Reference

```cpp
// Load config from calibration JSON
bool entropy_kv_config_load(config, json_path, keep_ratio, keep_sink, scale_min, scale_max);

// Initialize from raw entropy values (no file)
bool entropy_kv_config_init(config, n_layer, n_head, n_head_kv, n_ctx, entropies, ...);

// Compute/recompute budgets (call after changing keep_ratio)
void entropy_kv_compute_budgets(config);

// Check if eviction is needed
bool entropy_kv_should_evict(ctx, config);

// Perform eviction, returns number of positions evicted
int entropy_kv_evict(ctx, config, seq_id);

// Uniform fallback (no calibration data)
void entropy_kv_config_uniform(config, n_layer, n_head, n_head_kv, n_ctx, keep_ratio);

// Debug printing
void entropy_kv_config_print(config);
```

## Limitations

1. **Position-level granularity**: Cannot evict per-head due to llama.cpp's
   flat KV cache structure
2. **Heuristic importance**: Uses recency-based scoring rather than actual
   attention weights for position selection
3. **Static entropy profiles**: Calibrated once; heads may behave differently
   on very different input distributions
4. **Defrag overhead**: `llama_kv_cache_defrag()` has some cost after eviction

## Future Work

- **Attention-weighted eviction**: Track actual attention patterns for
  ground-truth position importance
- **Per-head masking**: Zero out positions per-head in the attention
  computation rather than evicting from the shared cache
- **Combined with KV quantization**: Stack entropy-adaptive eviction with
  2-bit KV quantization for 20x+ total compression
- **Per-layer budgets**: Allow different compression ratios per layer

## Citation

This work is part of the entropy-adaptive KV cache compression research.
See: https://github.com/SCJedi/vacuum_physics
