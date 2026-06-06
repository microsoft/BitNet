#pragma once

/*
 * ggml-bitnet-dispatch.h — Custom ggml ops for L3/L4/L5 math kernels
 *
 * These functions create ggml tensor nodes (via ggml_map_custom*) that
 * are executed during ggml_graph_compute.  Call them during graph
 * construction to replace standard ops with the research kernels:
 *
 *   L3 (ACDC)    — y = H(d ⊙ (H·x))        O(n log n) structured GEMV
 *   L4 (Tropical) — attention via (max,+)    O(n·d + K·d) top-K attention
 *   L5 (HRR)     — attention via circular    O(d log d) per-query retrieval
 *                   convolution memory
 *
 * All ops are single-threaded (n_tasks=1).  Multi-thread parallelism of
 * the surrounding graph is unaffected.
 *
 * Build requirements:
 *   -DBITNET_L3_ACDC=ON     enables bitnet_op_acdc
 *   -DBITNET_L4_TROPICAL=ON  enables bitnet_op_tropical_attn
 *   -DBITNET_L5_HRR=ON       enables bitnet_op_hrr_attn
 *
 * When the corresponding level is disabled, the functions return the first
 * source tensor unchanged (pass-through, no allocation).
 */

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * L3 — ACDC structured layer
 *
 * Computes y = H·(d ⊙ (H·x)) where H is the unnormalized WHT matrix.
 * Requires x->ne[0] to be a power of 2.
 *
 * @param ctx  ggml context
 * @param x    input activations  [n] or [n, batch]  (GGML_TYPE_F32)
 * @param d    learned diagonal   [n]                (GGML_TYPE_F32)
 * @return     output tensor, same shape as x        (GGML_TYPE_F32)
 *
 * Critical: ACDC only achieves energy recovery when the model was *trained*
 * with this architecture.  For random ternary W, ACDC captures only ~1/n
 * of the energy (see docs/theory/03-acdc-structured-layers.md).
 */
GGML_API struct ggml_tensor * bitnet_op_acdc(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    struct ggml_tensor  * d);

/*
 * L3 — ACDC GEMV (rectangular, K blocks + linear projection)
 *
 * Computes y[m] = proj · [H(d₀⊙(H·x)); H(d₁⊙(H·x)); ...; H(d_{K-1}⊙(H·x))]
 * where H is the unnormalized WHT.  Input x is zero-padded from n_orig to n
 * (must be next_pow2(n_orig)), and quantized to int8 inside the callback.
 *
 * Used for retangular projections (FFN up/down: 2560→6912, 6912→2560 in
 * BitNet 2B).  Pads:
 *   up:   n_orig=2560 → n=4096, m=6912, K=⌈6912/4096⌉=2
 *   down: n_orig=6912 → n=8192, m=2560, K=⌈2560/8192⌉=1
 *
 * The projection matrix and diagonals are statically allocated by the
 * callback (partial identity + zeros) on first use.  This produces
 * garbage output (P6: model wasn't trained with ACDC) but exercises
 * the kernel in the real dispatch path.  Use the env var
 * BITNET_ACDC_FFN=1 to activate.
 *
 * @param ctx    ggml context
 * @param x      input activations  [n_orig]  (F32)
 * @param m      output dim (the original model dim, not power-of-2)
 * @param n      ACDC block dim (power of 2 ≥ n_orig)
 * @param K      number of ACDC blocks (K*n ≥ m)
 * @param n_orig original input dim before padding to n
 * @return       output tensor [m]  (F32)
 */
GGML_API struct ggml_tensor * bitnet_op_acdc_gemv(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    int                   m,
    int                   n,
    int                   K,
    int                   n_orig);

/*
 * L4 — Tropical attention (max,+) semiring with top-K scan
 *
 * Replaces standard softmax attention:
 *   Standard: output = softmax(Q·Kᵀ/√d) · V    O(n²·d)
 *   Tropical:  output = softmax_topk(Q·Kᵀ) · V  O(n·d + K·d)
 *
 * Q and K are quantized to int8 internally before the tropical scan
 * (scores computed as integer dot products, zero multiplications).
 *
 * @param ctx   ggml context
 * @param q     query  [head_dim, n_queries]   (GGML_TYPE_F32)
 * @param k     keys   [head_dim, n_kv]        (GGML_TYPE_F32)
 * @param v     values [head_dim, n_kv]        (GGML_TYPE_F32)
 * @param topk  number of top-K keys to attend (K ≪ n_kv for speedup)
 * @param scale query scale factor (typically 1/√head_dim)
 * @return      output [head_dim, n_queries]   (GGML_TYPE_F32)
 */
GGML_API struct ggml_tensor * bitnet_op_tropical_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    int                   topk,
    float                 scale);

/*
 * L5 — HRR attention via holographic reduced representations
 *
 * Replaces standard attention with circular-convolution memory:
 *   Build:    M = Σᵢ kᵢ ⊛ vᵢ   (binding keys to values via ⊛)
 *   Retrieve: ṽ = M ⊛ q⁻¹       (unbinding with pseudo-inverse)
 *
 * Retrieval is O(d log d) per query, independent of context length.
 * Requires head_dim ≥ 10 × n_ctx for reliable retrieval (see CLAUDE.md).
 *
 * K is both provided as float (for the ternary approximation) and the
 * ternary version is derived internally from K_float by rounding.
 *
 * @param ctx  ggml context
 * @param q    queries [head_dim, n_queries]  (GGML_TYPE_F32)
 * @param k    keys    [head_dim, n_kv]       (GGML_TYPE_F32)
 * @param v    values  [head_dim, n_kv]       (GGML_TYPE_F32)
 * @return     output  [head_dim, n_queries]  (GGML_TYPE_F32)
 */
GGML_API struct ggml_tensor * bitnet_op_hrr_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v);

/*
 * bitnet_op_hrr_attn_with_cleanup: HRR attention + Frady 2021 iterative cleanup.
 *
 * Same as bitnet_op_hrr_attn but, after the unbind, runs hrr_cleanup_iter
 * (RESIDUAL mode) to identify the dominant values in the codebook (V) and
 * subtract their traces from a working copy of M. This recovers usable SNR
 * even when n_kv > d/10 (capacity limit of raw HRR retrieval).
 *
 * Complexity per head: O(n_kv·d·log d) build + n_tokens × O(max_iters × d·log d)
 * retrieve+cleanup. For d=128, n_kv=2048, max_iters=8: build ~17ms, retrieve
 * per token ~340µs (on a modern x86_64 with AVX2).
 *
 * @param ctx        ggml context
 * @param q          queries [head_dim, n_queries]  (GGML_TYPE_F32)
 * @param k          keys    [head_dim, n_kv]       (GGML_TYPE_F32)
 * @param v          values  [head_dim, n_kv]       (GGML_TYPE_F32) — also used as
 *                   the codebook for cleanup (each v_i is a candidate)
 * @param max_iters  iteration cap for cleanup (typ. 8-16); encoded as the
 *                   first 32 bits of an int userdata pointer.
 * @return           output  [head_dim, n_queries]  (GGML_TYPE_F32)
 */
GGML_API struct ggml_tensor * bitnet_op_hrr_attn_with_cleanup(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    int                  max_iters);

#ifdef __cplusplus
}
#endif
