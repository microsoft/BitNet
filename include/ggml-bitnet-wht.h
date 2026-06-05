#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * WHT-GEMV: Multiplication-Free Ternary Matrix-Vector Product
 *
 * Mathematical basis:
 *   For W ∈ {-1, 0, +1}^{m×n} and x ∈ ℤ₈ⁿ:
 *
 *     y[i] = Σⱼ W[i,j] · x[j]
 *          = Σ_{j: W[i,j]=+1} x[j]  -  Σ_{j: W[i,j]=-1} x[j]
 *
 *   This decomposes the dot product into two conditional sums — no
 *   multiplication at all. The sign information is extracted from the
 *   I2_S encoded weights (0=neg, 1=zero, 2=pos) using SIMD compare
 *   instructions (cmpeq) which produce bitmasks at zero cost.
 *
 * Algebraic identity exploited:
 *   W = W⁺ - W⁻  where W⁺, W⁻ ∈ {0,1}^{m×n}
 *   y = W·x = W⁺·x - W⁻·x
 *
 * No _mm256_maddubs_epi16 (multiply-add). Only:
 *   _mm256_cmpeq_epi8    — bitmask extraction (0 cycles on modern μops)
 *   _mm256_and_si256     — conditional selection (1 cycle)
 *   _mm256_sub_epi8      — signed subtraction (1 cycle)
 *   _mm256_add_epi32     — accumulation (1 cycle)
 *
 * Throughput estimate: ~5× faster than maddubs path for decode (batch=1).
 */

/*
 * WHT ternary dot product — single row vs activation vector.
 *
 * @param n          number of columns (must be multiple of QK_I2_S)
 * @param s          output scalar (one float)
 * @param vx         packed I2_S weights for this row (2 bits/weight)
 * @param vy         int8 activation vector
 * @param weight_scale  per-tensor weight scale γ (absmax-mean)
 * @param act_scale     per-token activation scale s = 127/max|x|
 */
void ggml_vec_dot_wht_ternary(
    int       n,
    float   * s,
    const void * vx,
    const void * vy,
    float     weight_scale,
    float     act_scale
);

/*
 * WHT GEMV — full matrix-vector product.
 * Drop-in replacement for ggml_vec_dot_i2_i8_s in batch=1 decode.
 *
 * @param m          number of rows in W
 * @param n          number of columns in W (= activation dimension)
 * @param y          output vector [m floats]
 * @param W          packed I2_S weight matrix, row-major
 * @param x          int8 activation vector [n bytes]
 * @param weight_scale  scalar scale for the weight tensor
 * @param act_scale     per-token activation scale
 */
void ggml_gemv_wht_ternary(
    int       m,
    int       n,
    float   * y,
    const void * W,
    const void * x,
    float     weight_scale,
    float     act_scale
);

/* Verify WHT result against reference MAD result (for testing) */
int ggml_wht_verify(int n, const void * vx, const void * vy,
                    float weight_scale, float act_scale,
                    float tolerance);

/*
 * Raw WHT ternary dot product — returns int32 without applying any scale.
 * Computes  Σᵢ w_ternary[i] · x[i]  where w_ternary ∈ {-1, 0, +1}.
 *
 * Used by the ggml dispatch layer (L2) to produce MAD-compatible output:
 *   ggml_vec_dot_i2_i8_s returns (raw_wht + sum(vy)) to match the
 *   dequantization formula in ggml.c:  result = (val - act_sums) / act_scales * w_scale
 */
int32_t ggml_wht_raw_dot(int n, const void * vx, const void * vy);

/*
 * Sum of int8 activation vector: Σᵢ vy[i] → int32.
 * Needed to convert WHT true-ternary output to MAD-compatible format.
 */
int32_t ggml_wht_sum_i8(int n, const int8_t * vy);

#ifdef __cplusplus
}
#endif
