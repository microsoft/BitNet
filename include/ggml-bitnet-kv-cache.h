/*
 * ggml-bitnet-kv-cache.h
 *
 * Per-(layer, kv_head) persistent K_i8 cache for tropical attention.
 *
 * Background:
 *   `tropical_attention` reads K as int8, but the KV cache stores K in F32.
 *   Re-quantizing all n_kv keys at every attention call is O(n_kv * d) per
 *   head per call — and n_kv grows by 1 per decode step. At context length
 *   256 this dominates the attention compute (3-pass K problem; see S2.4
 *   in SESSION_SUMMARY.md).
 *
 *   This cache makes quantization incremental: on the first call for a given
 *   (layer, kv_head), we quantize the full n_kv and lock the k_scale. On
 *   subsequent calls we only quantize the new entries using the locked scale.
 *
 * Design choices and trade-offs:
 *
 *   1. **Lock the scale at first call.** The relative ranking of dot
 *      products is preserved (all keys share the same scale), so top-K
 *      quality is unchanged for keys that don't saturate. New keys whose
 *      |value| > 127/k_scale saturate at ±127 — a small accuracy loss in
 *      exchange for skipping n_kv-1 re-quantizations per step.
 *
 *   2. **Process-lifetime, lazy-allocated.** No teardown on model swap;
 *      dimensions are re-checked on first use per session. Reset via
 *      `bitnet_kv_i8_cache_reset()` (env `BITNET_TROPICAL_KI8_RESET=1`).
 *
 *   3. **Single-writer per (il, h).** The tropical callback already assigns
 *      disjoint heads to disjoint threads (`for h = ith; h < n_head; h += nth`),
 *      so each (layer, head) slot has at most one writer per compute pass.
 *      No locking needed.
 *
 * Usage:
 *   bitnet_kv_i8_cache_set_layer(il);  // called from llama.cpp KQV site
 *   int8_t * K_i8 = bitnet_kv_i8_cache_get(
 *       il, kv_h, K_f32, n_kv, &k_scale, NULL, NULL);
 *   // K_i8 has n_kv * d int8 values; k_scale matches the locked scale.
 *
 *   The cache is no-op if `n_kv <= n_quantized` (all keys already cached).
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Configure cache dimensions. Idempotent: reallocates only if
 * (n_layer, n_head_kv, d) changed. Safe to call multiple times.
 *
 * @param n_layer   number of transformer layers
 * @param n_head_kv number of KV heads (GQA-aware; same for K and V)
 * @param d         head dimension
 * @param max_n_kv  max n_kv the cache can hold (typically n_ctx)
 */
void bitnet_kv_i8_cache_init(int n_layer, int n_head_kv, int d, int max_n_kv);

/*
 * Reset all cached data (next call re-quantizes from scratch with a fresh
 * scale). Does not free the slot memory; only sets n_quantized = 0.
 */
void bitnet_kv_i8_cache_reset(void);

/*
 * Free all memory. Call on process shutdown or before reinit.
 */
void bitnet_kv_i8_cache_free(void);

/*
 * Set the current layer index (for callers that don't pass il explicitly).
 * Must be called by llama.cpp's llm_build_kqv before each tropical call so
 * the callback knows which layer's cache to use.
 */
void bitnet_kv_i8_cache_set_layer(int il);

/*
 * Get the most recently set layer index. Returns -1 if unset.
 * Used by bitnet_op_tropical_attn to capture the layer into userdata.
 */
int  bitnet_kv_i8_current_layer(void);

/*
 * Get (or create + populate) the K_i8 buffer for the given (layer, kv_head),
 * quantizing only the new keys not already cached. Returns pointer to a
 * buffer of size n_kv * d.
 *
 * @param il            layer index (used as-is, not via g_current_layer)
 * @param kv_head       KV head index (0..n_head_kv-1)
 * @param K_f32         source float keys [n_kv * d]
 * @param n_kv          number of keys (must be >= last n_kv for this slot)
 * @param d             head dimension (must match the value used at init time;
 *                      triggers auto-reinit if the cache was built with a
 *                      different d — handles model-swap within a session)
 * @param k_scale_out   output: quantization scale used (locked after first call)
 * @param last_n_out    optional output: n_quantized BEFORE this call
 *                      (0 = first call, >0 = incremental)
 * @param n_new_out     optional output: n quantized in THIS call
 *                      (n_kv on first call, n_kv - last_n on subsequent)
 * @return              pointer to int8 buffer of size n_kv * d
 */
int8_t * bitnet_kv_i8_cache_get(
    int            il,
    int            kv_head,
    const float  * K_f32,
    int            n_kv,
    int            d,
    float        * k_scale_out,
    int          * last_n_out,
    int          * n_new_out);

#ifdef __cplusplus
}
#endif
