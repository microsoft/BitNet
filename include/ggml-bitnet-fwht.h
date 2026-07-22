#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Fast Walsh-Hadamard Transform (FWHT) — CPU kernel
 *
 * ─────────────────────────────────────────────────────────────────────────
 * MATHEMATICAL FOUNDATION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The Hadamard matrix H_n (n = 2^k) is defined recursively:
 *
 *   H_1 = [1]
 *   H_{2k} = H_k ⊗ H_2  =  [ H_k   H_k ]
 *                            [ H_k  -H_k ]
 *
 * Properties:
 *   - All entries in {-1, +1}
 *   - H_n · H_n^T = n · I_n          (scaled orthogonal)
 *   - Inverse: H_n^{-1} = H_n / n    (self-inverse up to scale)
 *
 * The FWHT computes ŷ = H_n · y in O(n log n) using the butterfly:
 *
 *   for each stage s = 0, 1, ..., log₂(n)-1:
 *     len = 2^s
 *     for each block [i, i+2·len):
 *       for j = 0..len-1:
 *         a = v[i+j];  b = v[i+j+len]
 *         v[i+j]     = a + b   ← addition only
 *         v[i+j+len] = a - b   ← subtraction only
 *
 * ZERO multiplications. Only ± integer/float operations.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ACDC STRUCTURED LAYER
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Standard dense weight matrix W ∈ ℝ^{m×n}: cost O(mn)
 *
 * ACDC approximation (one block): W ≈ H_n · diag(d) · H_n
 *
 *   y = W·x  ≈  H_n · (d ⊙ (H_n · x))
 *
 *   Step 1: ẑ = H_n · x          — FWHT, O(n log n), zero multiplications
 *   Step 2: z = d ⊙ ẑ           — diagonal scaling, n multiplications
 *   Step 3: y = H_n · z          — FWHT, O(n log n), zero multiplications
 *
 * Total multiplications per layer: n (the diagonal d — irreducible minimum)
 * Total additions: 2 · n · log₂(n)
 *
 * For non-square W (m ≠ n): stack K = ⌈m/n⌉ ACDC blocks, each with its
 * own learned diagonal d_k, sharing the same Hadamard basis.
 *
 * Operation count comparison (n=2560, m=6912, one FFN layer):
 *   Dense ternary:   2560 × 6912 =  17.7M ops
 *   K=3 ACDC blocks: 3 × (2 × 2560 × log₂(4096) + 2560) ≈ 192K ops
 *   Speedup: ~92× in op count (empirical: 20-50× after memory effects)
 */

/* Padding: FWHT requires n = 2^k; round up */
int fwht_next_pow2(int n);

/* ── int8 → int32 WHT (first transform: activations) ─────────────────── */

/*
 * fwht_i8_to_i32: sign-extend int8 x to int32, then apply in-place FWHT.
 * Output lives in out[0..n-1] as unnormalized int32.
 * n must equal next_pow2(orig_n); zero-pad input if orig_n < n.
 * ZERO multiplications.
 */
void fwht_i8_to_i32(const int8_t * x, int32_t * out, int n);

/* ── float32 in-place WHT (second transform: after diagonal scaling) ──── */

/*
 * fwht_f32: in-place Fast WHT on float32 vector of length n (power of 2).
 * After this call: out[k] = Σⱼ (±1) · in[j]  (unnormalized).
 * Divide by n for the orthonormal transform.
 * ZERO multiplications.
 */
void fwht_f32(float * v, int n);

/*
 * fwht_f32_parallel: OpenMP-parallel variant for standalone tools.
 *
 * Semantically identical to fwht_f32(v, n); uses n_threads OMP threads for
 * the large butterfly stages (h ≥ 8).  DO NOT call from ggml thread-pool
 * callbacks — use fwht_f32() there to avoid CPU over-subscription.
 *
 * When compiled without BITNET_FWHT_OMP this is a no-op wrapper around fwht_f32.
 */
void fwht_f32_parallel(float * v, int n, int n_threads);

/* ── ACDC layer forward pass ──────────────────────────────────────────── */

/*
 * acdc_forward_i8: single ACDC block — int8 input, float output.
 *
 * @param y    output vector [n floats]
 * @param x    int8 activation input [n bytes], zero-padded to next_pow2
 * @param d    learned diagonal [n floats]
 * @param n    dimension (must be power of 2)
 */
void acdc_forward_i8(float * y, const int8_t * x, const float * d, int n);

/*
 * acdc_forward_f32: single ACDC block — float input, float output.
 * Used for stacked blocks (input of block k+1 = output of block k).
 */
void acdc_forward_f32(float * y, const float * x, const float * d, int n);

/*
 * acdc_gemv: ACDC approximation of W·x for non-square W using K stacked blocks.
 *
 * Approximates W ∈ ℝ^{m×n} as K blocks of size n×n with learned diagonals D[k].
 * Output y[m] produced by: stacking K WHT blocks, then linear projection to m.
 *
 * @param y      output [m floats]
 * @param x      int8 input [n bytes]
 * @param D      K learned diagonals, D[k*n .. (k+1)*n-1] is diagonal k [K*n floats]
 * @param proj   linear projection from K*n → m [m * K*n floats] (can be ternary)
 * @param m      output dimension
 * @param n      input dimension (padded to power of 2)
 * @param K      number of ACDC blocks
 */
void acdc_gemv(float * y, const int8_t * x, const float * D,
               const float * proj, int m, int n, int K);

/* ── Projection: find best ACDC approximation to a ternary matrix ─────── */

/*
 * acdc_project: given W ∈ {-1,0,+1}^{n×n}, find diagonal d that minimizes
 *   ||W - H·diag(d)·H||_F
 *
 * Closed-form solution: d[k] = (H^T · W · H)[k,k] / n²
 * Computed in O(n² log n) via two WHTs applied to each row.
 *
 * @param d  output diagonal [n floats]
 * @param W  input ternary matrix, row-major [n×n int8, values in {-1,0,+1}]
 * @param n  dimension (must be power of 2)
 */
void acdc_project(float * d, const int8_t * W, int n);

/* ── Approximation quality ────────────────────────────────────────────── */

/*
 * acdc_error: relative Frobenius error ||W - H·D·H||_F / ||W||_F
 * Returns value in [0, 1]; lower is better.
 */
float acdc_error(const int8_t * W, const float * d, int n);

/* ── Rectangular ACDC — Fase II ──────────────────────────────────────────
 *
 * Extends ACDC to rectangular weight matrices W ∈ ℝ^{m×n} (m ≠ n).
 *
 * Uses a single shared Hadamard size P = next_pow2(max(m,n)):
 *
 *   y[m] = first m elements of H_P · (d ⊙ (H_P · [x | 0]))
 *
 * The input x[n] is zero-padded to P before the first FWHT, and the
 * output is truncated from P to m after the second FWHT.
 *
 * For Falcon3-10B FFN (n=3072, m=23040):
 *   P = 32768
 *   Dense:     3072 × 23040 = 70.8M ops
 *   ACDC rect: 2 × 32768 × 15 = 983K ops → ~72× fewer
 * ────────────────────────────────────────────────────────────────────────── */

/*
 * acdc_forward_rect_f32: rectangular ACDC, float32 input.
 *
 * @param y  output [m floats]
 * @param m  output dimension
 * @param x  float input [n floats]
 * @param n  input dimension
 * @param d  diagonal [P floats], P = next_pow2(max(m,n))
 */
void acdc_forward_rect_f32(float * y, int m, const float * x, int n, const float * d);

/*
 * acdc_forward_rect_i8: rectangular ACDC, int8 pre-quantized input.
 *
 * @param y  output [m floats]
 * @param m  output dimension
 * @param x  int8 input [n bytes], values in [-128, 127]
 * @param n  input dimension
 * @param d  diagonal [P floats], P = next_pow2(max(m,n))
 */
void acdc_forward_rect_i8(float * y, int m, const int8_t * x, int n, const float * d);

/*
 * acdc_project_rect: best diagonal d for W ∈ {-1,0,+1}^{m×n}.
 *
 * Computes d[k] = (H_P · W_P · H_P)[k,k] / P² via XOR-convolution:
 *
 *   C[s] = Σ_{(i,j): i XOR j = s} W[i,j]    (accumulated in O(m·n))
 *   d* = FWHT(C) / P²                          (O(P log P))
 *
 * Memory O(P): 128 KB for P=32768 (vs 4 GB naive).
 * Cost O(m·n + P log P): ~71M ops for Falcon3-10B (vs 16G naive).
 * Run offline, not at inference time.
 *
 * @param d  output diagonal [P floats], P = next_pow2(max(m,n))
 * @param W  input ternary matrix [m×n int8], row-major, values in {-1,0,+1}
 * @param m  row dimension
 * @param n  column dimension
 */
void acdc_project_rect(float * d, const int8_t * W, int m, int n);

#ifdef __cplusplus
}
#endif
