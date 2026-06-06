#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ggml-bitnet-tropical.h — Tropical Attention API
 *
 * ─────────────────────────────────────────────────────────────────────────
 * MATHEMATICAL FOUNDATION: (max, +) SEMIRING
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Tropical algebra = semiring (ℝ ∪ {-∞}, ⊕, ⊗):
 *   a ⊕ b = max(a, b)         [tropical addition]
 *   a ⊗ b = a + b             [tropical multiplication]
 *
 * Tropical matrix product:
 *   (A ⊗ᵗʳᵒᵖ B)[i,k] = max_j (A[i,j] + B[j,k])
 *
 * Connection to Transformer attention (temperature limit):
 *   lim_{τ→0} softmax(QKᵀ/τ)[i,j] = 𝟙[j = argmax_k Q[i]·K[k]ᵀ]
 *
 * This IS the tropical matrix product. At low temperature, transformer
 * attention degenerates to nearest-neighbor lookup in (max,+) semiring.
 *
 * Complexity reduction:
 *   Standard attention:        O(n²·d) — all pairs
 *   Tropical hard attention:   O(n·d)  — argmax per query
 *   Tropical top-K attention:  O(n·d + K·d) — top-K retrieve + softmax
 *
 * For K=32, n=2048: 64× fewer operations than standard attention.
 * Keys are ternary {-1,0,+1}: dot product = additions only (Level 2).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * API OVERVIEW
 * ─────────────────────────────────────────────────────────────────────────
 *
 *  1. tropical_attn_scores    — compute all Q·K[j] scores (float output)
 *  2. tropical_attn_argmax    — find argmax_j Q·K[j] (hard attention)
 *  3. tropical_attn_topk      — find top-K indices + scores
 *  4. tropical_attention      — full attention: topK + softmax + V lookup
 *  5. tropical_gemv           — tropical matrix-vector product (max,+)
 */

/* ─── Score computation ───────────────────────────────────────────────── */

/*
 * tropical_attn_scores: compute all attention scores Q·K[j] / √d
 *
 * Uses ternary dot product (Level 2 kernel): zero multiplications.
 * The scale factor q_scale * k_scale / head_dim absorbs the 1/√d factor.
 *
 * @param scores    output [n_keys floats]
 * @param q         quantized query [head_dim int8]
 * @param K         ternary keys [n_keys × head_dim int8, values {-1,0,+1}]
 * @param n_keys    number of keys (sequence length)
 * @param head_dim  dimension per attention head
 * @param q_scale   quantization scale of query (absmax / 127)
 * @param k_scale   quantization scale of keys (absmax / 1, ternary)
 */
void tropical_attn_scores(
    float        * scores,
    const int8_t * q,
    const int8_t * K,
    int            n_keys,
    int            head_dim,
    float          q_scale,
    float          k_scale);

/* ─── Hard attention (argmax) ─────────────────────────────────────────── */

/*
 * tropical_attn_argmax: returns argmax_j Q·K[j]
 *
 * Pure (max,+) semiring — no softmax, no exp.
 * O(n·d) time, O(1) extra space.
 * For ternary K: dot product = additions only (Level 2).
 *
 * @return index of the key with maximum dot product score
 */
int tropical_attn_argmax(
    const int8_t * q,
    const int8_t * K,
    int            n_keys,
    int            head_dim);

/* ─── Top-K soft attention ────────────────────────────────────────────── */

/*
 * tropical_attn_topk: find top-K attention positions
 *
 * Step 1: O(n·d) scan — ternary dot products (additions only)
 * Step 2: O(n·log K) partial sort — comparisons only
 *
 * @param top_idx    output: indices of top-K keys [K ints]
 * @param top_scores output: scores of top-K keys [K floats]
 * @param q          quantized query [head_dim int8]
 * @param K          ternary keys [n_keys × head_dim int8]
 * @param n_keys     number of keys
 * @param head_dim   head dimension
 * @param K_top      number of top candidates to select
 * @param q_scale    query quantization scale
 * @param k_scale    key quantization scale
 */
void tropical_attn_topk(
    int          * top_idx,
    float        * top_scores,
    const int8_t * q,
    const int8_t * K,
    int            n_keys,
    int            head_dim,
    int            K_top,
    float          q_scale,
    float          k_scale);

/* ─── Full tropical attention ─────────────────────────────────────────── */

/*
 * tropical_attention: complete attention with tropical top-K + softmax
 *
 * Algorithm:
 *   1. Top-K via tropical max scan:  O(n·d) ternary dot products
 *   2. Softmax over K scores:        O(K) exponentials (K << n)
 *   3. Weighted sum of V[top_K]:     O(K·d) multiply-adds
 *
 * Total: O(n·d + K·d) vs O(n²·d) standard → speedup ≈ n/K
 *
 * @param output   output vector [head_dim floats]
 * @param q        quantized query [head_dim int8]
 * @param K        ternary keys [n_keys × head_dim int8]
 * @param V        float values [n_keys × head_dim floats]
 * @param n_keys   sequence length
 * @param head_dim head dimension
 * @param K_top    number of top keys to use in softmax
 * @param q_scale  query quantization scale
 * @param k_scale  key quantization scale
 */
void tropical_attention(
    float        * output,
    const int8_t * q,
    const int8_t * K,
    const float  * V,
    int            n_keys,
    int            head_dim,
    int            K_top,
    float          q_scale,
    float          k_scale);

/* ─── Float sparse attention ──────────────────────────────────────────── */

/*
 * sparse_attention_float: top-K attention with float32 scoring (no quantization)
 *
 * Computes attention restricting softmax to the K highest-scoring keys.
 * Uses standard float dot products (no ternary tricks) — single pass over K.
 *
 * This is faster than tropical_attention for current BitNet models because:
 *   - Eliminates float→int8 K quantization (the dominant memory bottleneck)
 *   - Single pass over K_f32 instead of 3 passes (F32→I8→score)
 *   - Compiler-vectorized float dot products
 *
 * Quality for K << n_keys: produces sparse attention approximation.
 * Quality is model-dependent — best when attention is naturally sparse
 * (validated empirically for trained LLMs, see Zhang et al. 2023).
 *
 * @param output    result [head_dim floats]
 * @param q         query vector [head_dim floats]
 * @param K         key matrix [n_keys × head_dim floats]
 * @param V         value matrix [n_keys × head_dim floats]
 * @param n_keys    number of available keys (KV cache size)
 * @param head_dim  dimension per attention head
 * @param K_top     maximum keys to include (clamped to n_keys if larger)
 */
void sparse_attention_float(
    float       * output,
    const float * q,
    const float * K,
    const float * V,
    int           n_keys,
    int           head_dim,
    int           K_top);

/* ─── Tropical GEMV ───────────────────────────────────────────────────── */

/*
 * tropical_gemv: tropical matrix-vector product (max,+)
 *
 * Computes: output[i] = max_j (A[i,j] + x[j])  for each row i
 * Also stores argmax_j in argmax_out[i].
 *
 * Pure (max,+) arithmetic — no standard multiplications needed.
 * A is ternary {-1,0,+1}: addition becomes conditional ±1.
 *
 * @param argmax_out  output: argmax index per row [m ints]
 * @param max_out     output: tropical max value per row [m floats]
 * @param A           ternary matrix [m × n int8, values {-1,0,+1}]
 * @param x           input vector [n floats]
 * @param m           number of rows
 * @param n           number of columns
 */
void tropical_gemv(
    int          * argmax_out,
    float        * max_out,
    const int8_t * A,
    const float  * x,
    int            m,
    int            n);

#ifdef __cplusplus
}
#endif
