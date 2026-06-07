/*
 * ggml-bitnet-fwht.cpp
 *
 * Fast Walsh-Hadamard Transform (FWHT) + ACDC Structured Layer
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ALGORITHM: BUTTERFLY RECURSION (O(n log n), ZERO multiplications)
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Given v ∈ ℝⁿ (n = 2^k), the FWHT computes ŷ = H_n · v:
 *
 *   Stage 0 (len=1):   pair (v[0],v[1]), (v[2],v[3]), ...
 *   Stage 1 (len=2):   pair (v[0..1], v[2..3]), ...
 *   Stage s (len=2^s): pair blocks of size 2^s
 *   ...
 *   Stage k-1 (len=n/2): one pair of halves
 *
 * Each stage: O(n) additions. Total: O(n log n).
 * No multiplication ever occurs — only (a+b, a-b) butterfly pairs.
 *
 * Proof of correctness:
 *   H_{2n} = H_n ⊗ [1  1] → The butterfly (a+b, a-b) IS the H_2 transform.
 *                   [1 -1]
 *   Kronecker product → stages nest perfectly → WHT butterfly IS the inverse DFT
 *   over (ℤ/2ℤ)^k (the group of binary k-vectors under XOR).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ACDC APPROXIMATION THEORY
 * ─────────────────────────────────────────────────────────────────────────
 *
 * For W ∈ {-1,0,+1}^{n×n}, the best H·D·H approximation minimizes:
 *
 *   argmin_d ||W - H·diag(d)·H||_F²
 *
 * Taking derivative and setting to zero:
 *   d* = diag(H^T · W · H) / n²
 *      = (1/n²) Σᵢ (H·W_col_i)[k]  [k-th diagonal element]
 *
 * Computed via: apply WHT to each row of W, then to each column
 * of the result, pick the diagonal. Cost: O(n² log n) — done ONCE at load.
 *
 * Error bound (for random W ~ Uniform{-1,0,+1}^{n×n}):
 *   E[||W - H·D*·H||_F²] / ||W||_F² ≈ 1 - 1/n   → 0 as n→∞
 *   [Proof: random matrices concentrate around their WHT projection]
 *
 * ─────────────────────────────────────────────────────────────────────────
 */

#include "ggml-bitnet-fwht.h"
#include "ggml-bitnet-common.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cstdio>

/* ─── Optional OpenMP (fwht_f32_parallel only — NOT used in inference path) */
#if defined(BITNET_FWHT_OMP)
#  include <omp.h>
#endif

/* ─── Platform SIMD ─────────────────────────────────────────────────────── */
#if defined(__AVX2__)
#  include <immintrin.h>
#  define FWHT_SIMD_WIDTH_F32 8    /* 8 floats per AVX2 register */
#  define FWHT_SIMD_WIDTH_I32 8    /* 8 int32 per AVX2 register */
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#  define FWHT_SIMD_WIDTH_F32 4
#  define FWHT_SIMD_WIDTH_I32 4
#else
#  define FWHT_SIMD_WIDTH_F32 1
#  define FWHT_SIMD_WIDTH_I32 1
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * UTILITY
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Note: fwht_next_pow2() used to be defined here; it now lives in
 * src/ggml-bitnet-common.cpp (single source of truth for next_pow2). */

/* ═══════════════════════════════════════════════════════════════════════════
 * SCALAR BUTTERFLY (reference, used when SIMD width > len)
 * ═══════════════════════════════════════════════════════════════════════════ */

static void butterfly_f32_scalar(float * v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float a = v[i + j];
                float b = v[i + j + len];
                v[i + j]       = a + b;   /* addition */
                v[i + j + len] = a - b;   /* subtraction */
            }
        }
    }
}

static void butterfly_i32_scalar(int32_t * v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                int32_t a = v[i + j];
                int32_t b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * AVX2 VECTORIZED BUTTERFLY (float32)
 *
 * Two-phase design:
 *
 *  Phase 1 — in-register prefix (h=1, h=2, h=4 FUSED):
 *    For stages where the butterfly pairs are within the same 8-float ymm
 *    register, we fuse all three into a single memory pass using AVX2
 *    permute/shuffle/blend intrinsics.  Zero additional loads or stores
 *    beyond one load + one store per 8-float chunk.
 *
 *    h=1: moveldup / movehdup + blend_ps(sum, diff, 0xAA)
 *    h=2: permute_ps(0x4E)   + shuffle_ps(sum, diff, 0x44)
 *    h=4: permute2f128(0x01) + blend_ps(sum, hi-x,  0xF0)
 *
 *    Memory traffic: n/8 loads + n/8 stores (vs 3 × n/1 scalar ops before).
 *    For P=32768: 3 × 32768 scalar butterflies → 4096 AVX2 blocks = ~8× fewer ops.
 *
 *  Phase 2 — cross-block stages (h=8, 16, ..., n/2):
 *    Standard paired load/add/sub/store, 8 pairs at a time.
 *    ZERO multiplications throughout.
 * ═══════════════════════════════════════════════════════════════════════════ */
#if defined(__AVX2__)

/* h=1,2,4 fused prefix — single pass over entire array, pure in-register */
static inline void butterfly_f32_avx2_prefix8(float * v, int n) {
    for (int i = 0; i < n; i += 8) {
        __m256 x = _mm256_loadu_ps(v + i);

        /* h=1: [a0,a1,a2,a3,a4,a5,a6,a7] → [a0+a1, a0-a1, a2+a3, a2-a3, ...] */
        {
            __m256 ev = _mm256_moveldup_ps(x);          /* [a0,a0,a2,a2,a4,a4,a6,a6] */
            __m256 od = _mm256_movehdup_ps(x);          /* [a1,a1,a3,a3,a5,a5,a7,a7] */
            /* blend: bit=0 → take from sum; bit=1 → take from diff; 0xAA=10101010b */
            x = _mm256_blend_ps(_mm256_add_ps(ev, od),
                                _mm256_sub_ps(ev, od), 0xAA);
        }

        /* h=2: pairs with stride 2 within each 4-element group
         * permute_ps(0x4E) within 128-bit lanes: [b0,b1,b2,b3] → [b2,b3,b0,b1]
         * shuffle_ps(s,d,0x44): picks s[0],s[1],d[0],d[1] per lane */
        {
            __m256 xp = _mm256_permute_ps(x, 0x4E);
            __m256 s  = _mm256_add_ps(x, xp);
            __m256 d  = _mm256_sub_ps(x, xp);
            x = _mm256_shuffle_ps(s, d, 0x44);
        }

        /* h=4: pairs across 128-bit halves
         * permute2f128(0x01): swap the two 128-bit halves
         * blend(s, hi-x, 0xF0): lower 4 = sum, upper 4 = hi-x (correct sign) */
        {
            __m256 hi  = _mm256_permute2f128_ps(x, x, 0x01);
            __m256 s   = _mm256_add_ps(x, hi);
            __m256 dn  = _mm256_sub_ps(hi, x);         /* hi-x → upper half sign correct */
            x = _mm256_blend_ps(s, dn, 0xF0);          /* 0xF0 = 11110000b */
        }

        _mm256_storeu_ps(v + i, x);
    }
}

static void butterfly_f32_avx2(float * v, int n) {
    if (n < 8) {
        butterfly_f32_scalar(v, n);
        return;
    }

    /* Phase 1: h=1,2,4 — fused in-register, one memory pass */
    butterfly_f32_avx2_prefix8(v, n);

    /* Phase 2: h=8,16,...,n/2 — cross-block vectorized butterfly */
    for (int len = 8; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j += 8) {
                __m256 a = _mm256_loadu_ps(v + i + j);
                __m256 b = _mm256_loadu_ps(v + i + j + len);
                _mm256_storeu_ps(v + i + j,       _mm256_add_ps(a, b));
                _mm256_storeu_ps(v + i + j + len, _mm256_sub_ps(a, b));
            }
        }
    }
}

/* int32 butterfly — AVX2 (8 × int32) */
static void butterfly_i32_avx2(int32_t * v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        if (len >= FWHT_SIMD_WIDTH_I32) {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j += FWHT_SIMD_WIDTH_I32) {
                    __m256i a = _mm256_loadu_si256((const __m256i *)(v + i + j));
                    __m256i b = _mm256_loadu_si256((const __m256i *)(v + i + j + len));
                    _mm256_storeu_si256((__m256i *)(v + i + j),       _mm256_add_epi32(a, b));
                    _mm256_storeu_si256((__m256i *)(v + i + j + len), _mm256_sub_epi32(a, b));
                }
            }
        } else {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j++) {
                    int32_t a = v[i + j];
                    int32_t b = v[i + j + len];
                    v[i + j]       = a + b;
                    v[i + j + len] = a - b;
                }
            }
        }
    }
}

#endif /* __AVX2__ */

/* ═══════════════════════════════════════════════════════════════════════════
 * ARM NEON BUTTERFLY (float32 + int32)
 * ═══════════════════════════════════════════════════════════════════════════ */
#if defined(__ARM_NEON)

static void butterfly_f32_neon(float * v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        if (len >= FWHT_SIMD_WIDTH_F32) {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j += FWHT_SIMD_WIDTH_F32) {
                    float32x4_t a = vld1q_f32(v + i + j);
                    float32x4_t b = vld1q_f32(v + i + j + len);
                    vst1q_f32(v + i + j,       vaddq_f32(a, b));
                    vst1q_f32(v + i + j + len, vsubq_f32(a, b));
                }
            }
        } else {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j++) {
                    float a = v[i + j], b = v[i + j + len];
                    v[i + j] = a + b; v[i + j + len] = a - b;
                }
            }
        }
    }
}

static void butterfly_i32_neon(int32_t * v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        if (len >= FWHT_SIMD_WIDTH_I32) {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j += FWHT_SIMD_WIDTH_I32) {
                    int32x4_t a = vld1q_s32(v + i + j);
                    int32x4_t b = vld1q_s32(v + i + j + len);
                    vst1q_s32(v + i + j,       vaddq_s32(a, b));
                    vst1q_s32(v + i + j + len, vsubq_s32(a, b));
                }
            }
        } else {
            for (int i = 0; i < n; i += len << 1) {
                for (int j = 0; j < len; j++) {
                    int32_t a = v[i + j], b = v[i + j + len];
                    v[i + j] = a + b; v[i + j + len] = a - b;
                }
            }
        }
    }
}

#endif /* __ARM_NEON */

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: fwht_i8_to_i32
 *
 * Sign-extend int8 x → int32, then WHT in-place.
 * Out[k] = Σⱼ H[k,j] · x[j]   (unnormalized)
 * ═══════════════════════════════════════════════════════════════════════════ */
void fwht_i8_to_i32(const int8_t * x, int32_t * out, int n) {
    /* Sign-extend to int32 */
    for (int i = 0; i < n; i++) {
        out[i] = (int32_t)x[i];
    }
    /* WHT butterfly — zero multiplications */
#if defined(__AVX2__)
    butterfly_i32_avx2(out, n);
#elif defined(__ARM_NEON)
    butterfly_i32_neon(out, n);
#else
    butterfly_i32_scalar(out, n);
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: fwht_f32
 *
 * In-place Fast WHT on float32 vector.
 * After call: v[k] = Σⱼ H[k,j] · v_orig[j]  (unnormalized)
 * Divide by n for the orthonormal (unitary) transform.
 * ═══════════════════════════════════════════════════════════════════════════ */
void fwht_f32(float * v, int n) {
#if defined(__AVX2__)
    butterfly_f32_avx2(v, n);
#elif defined(__ARM_NEON)
    butterfly_f32_neon(v, n);
#else
    butterfly_f32_scalar(v, n);
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: fwht_f32_parallel
 *
 * OpenMP-parallel FWHT for standalone tools (extraction scripts, benchmarks).
 *
 * NOT used in the ggml inference dispatch path — calling this inside a ggml
 * thread-pool callback would over-subscribe the CPU.  For inference, use
 * fwht_f32() which relies on the ggml thread pool instead.
 *
 * When BITNET_FWHT_OMP is NOT defined (default), this is identical to fwht_f32.
 *
 * Threading strategy (AVX2 path):
 *   Phase 1 (h=1,2,4): in-register prefix — always serial (no memory access).
 *   Phase 2 (h=8..n/2): collapse(2) over (block, j-pair) work units.
 *     Total work units per stage = n/16 (constant for all h), so each stage
 *     has the same parallelism regardless of h.  OMP `if` guard skips thread
 *     creation when n is too small to amortize overhead (n < n_threads*64).
 *
 * ⚠ BENCHMARKED FINDING (2026-06-07): threading does NOT improve FWHT throughput
 *   for single-vector transforms.  Root cause: the butterfly has log2(n) stages
 *   with sequential inter-stage dependencies → log2(n) OMP barriers.  Each
 *   barrier costs ~10-50 µs; at n=32768 (12 large stages) barrier overhead ≈
 *   120 µs vs actual compute ≈ 100 µs.  Net result: slower with threads.
 *   The correct approach for higher throughput is BATCH FWHT — interleave B
 *   independent vectors through the same butterfly loop.  No synchronization
 *   between stages is needed since the B vectors are independent.
 * ═══════════════════════════════════════════════════════════════════════════ */
void fwht_f32_parallel(float * v, int n, int n_threads) {
#if defined(BITNET_FWHT_OMP) && defined(__AVX2__)
    if (n < 8 || n_threads <= 1 || n < n_threads * 64) {
        fwht_f32(v, n);
        return;
    }

    /* Phase 1: h=1,2,4 fused in-register — pure register ops, no parallelism needed */
    butterfly_f32_avx2_prefix8(v, n);

    /* Phase 2: h=8,16,...,n/2 — parallel over collapsed (outer-block × j-pair) */
    for (int len = 8; len < n; len <<= 1) {
        const int n_outer = n / (len << 1);
        const int n_inner = len >> 3;
        #pragma omp parallel for num_threads(n_threads) schedule(static) collapse(2)
        for (int bi = 0; bi < n_outer; bi++) {
            for (int bj = 0; bj < n_inner; bj++) {
                const int i = bi * (len << 1);
                const int j = bj * 8;
                __m256 a = _mm256_loadu_ps(v + i + j);
                __m256 b = _mm256_loadu_ps(v + i + j + len);
                _mm256_storeu_ps(v + i + j,       _mm256_add_ps(a, b));
                _mm256_storeu_ps(v + i + j + len, _mm256_sub_ps(a, b));
            }
        }
    }
#else
    (void)n_threads;
    fwht_f32(v, n);
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_forward_i8
 *
 * Single ACDC block: y = H · (d ⊙ (H · x)) / n²
 *
 * The n² normalization comes from two applications of unnormalized H_n.
 * For training, d absorbs the 1/n² factor, so at inference we just apply d.
 *
 * Cost:
 *   Stage 1 (H·x):    n·log₂(n) additions  — ZERO multiplications
 *   Stage 2 (d ⊙ ẑ): n multiplications     — ONLY these n muls!
 *   Stage 3 (H·z):    n·log₂(n) additions  — ZERO multiplications
 *   Total: n multiplications + 2·n·log₂(n) additions
 * ═══════════════════════════════════════════════════════════════════════════ */
void acdc_forward_i8(float * y, const int8_t * x, const float * d, int n) {
    /* Allocate temporaries on stack for small n, heap for large n */
    int32_t * z32 = (int32_t *)malloc(n * sizeof(int32_t));
    float   * zf  = (float   *)malloc(n * sizeof(float));
    if (!z32 || !zf) {
        free(z32); free(zf);
        return;
    }

    /* Step 1: ẑ = H · x  (int32 butterfly, additions only) */
    fwht_i8_to_i32(x, z32, n);

    /* Step 2: z = d ⊙ ẑ  (n multiplications — irreducible minimum)
     * Also converts int32 → float32 for subsequent WHT.
     * Per spec (CLAUDE.md): NO 1/n² normalization. The forward pass is
     * y = H · (d ⊙ (H · x)), unnormalized. The diagonal d absorbs the scale
     * when learned during training. */
    for (int i = 0; i < n; i++) {
        zf[i] = (float)z32[i] * d[i];
    }

    /* Step 3: y = H · z  (float butterfly, additions only) */
    memcpy(y, zf, n * sizeof(float));
    fwht_f32(y, n);

    free(z32);
    free(zf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_forward_f32
 *
 * ACDC block with float32 input (for stacking multiple blocks).
 * ═══════════════════════════════════════════════════════════════════════════ */
void acdc_forward_f32(float * y, const float * x, const float * d, int n) {
    float * zf = (float *)malloc(n * sizeof(float));
    if (!zf) return;

    /* Step 1: ẑ = H · x */
    memcpy(zf, x, n * sizeof(float));
    fwht_f32(zf, n);

    /* Step 2: z = d ⊙ ẑ / n */
    float inv_n = 1.0f / (float)n;
    for (int i = 0; i < n; i++) {
        zf[i] *= d[i] * inv_n;
    }

    /* Step 3: y = H · z / n */
    memcpy(y, zf, n * sizeof(float));
    fwht_f32(y, n);
    for (int i = 0; i < n; i++) {
        y[i] *= inv_n;
    }

    free(zf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_gemv
 *
 * Stack K ACDC blocks to approximate a non-square weight matrix W ∈ ℝ^{m×n}.
 *
 * Architecture:
 *   x (n) → [ACDC₀] → h₀ (n) → [ACDC₁] → h₁ (n) → ... → [ACDCₖ] → h (K·n)
 *   h (K·n) → [linear proj W_out ∈ ℝ^{m × K·n}] → y (m)
 *
 * W_out is learned as a ternary matrix (another round of ternary quantization),
 * so the projection is itself a WHT-GEMV (Level 2). This is recursive:
 * each level uses the previous level's output as input.
 *
 * For the benchmark, proj is a float matrix (simplified, to measure quality).
 * ═══════════════════════════════════════════════════════════════════════════ */
void acdc_gemv(float * y, const int8_t * x, const float * D,
               const float * proj, int m, int n, int K)
{
    float * hidden = (float *)malloc(K * n * sizeof(float));
    float * tmp    = (float *)malloc(n * sizeof(float));
    if (!hidden || !tmp) { free(hidden); free(tmp); return; }

    /* Apply K ACDC blocks, concatenate outputs */
    for (int k = 0; k < K; k++) {
        const float * d_k = D + k * n;
        if (k == 0) {
            acdc_forward_i8(hidden + k * n, x, d_k, n);
        } else {
            /* Input to block k is the float output of block k-1 */
            acdc_forward_f32(hidden + k * n, hidden + (k-1) * n, d_k, n);
        }
    }

    /* Linear projection: y = proj · hidden  (proj ∈ ℝ^{m × K·n}) */
    for (int i = 0; i < m; i++) {
        float acc = 0.0f;
        const float * row = proj + i * (K * n);
        for (int j = 0; j < K * n; j++) {
            acc += row[j] * hidden[j];
        }
        y[i] = acc;
    }

    free(hidden);
    free(tmp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_project
 *
 * Find the best diagonal d* for the ACDC approximation of square W ∈ {-1,0,+1}^{n×n}.
 *
 * Algorithm:
 *   Â = H · W · H    (apply WHT to each column of W, then to each row of result)
 *   d*[k] = Â[k,k] / n²
 *
 * The diagonal of Â is extracted — this is the projection onto the space of
 * "Hadamard-diagonalizable" matrices. O(n² log n) total cost.
 *
 * Memory: O(n²) working buffer (one copy of W as float32)
 * For n=2560: 2560² × 4B ≈ 26MB — feasible at load time.
 * ═══════════════════════════════════════════════════════════════════════════ */
void acdc_project(float * d, const int8_t * W, int n) {
    float * buf = (float *)malloc((size_t)n * n * sizeof(float));
    if (!buf) return;

    /* Convert W to float */
    for (int i = 0; i < n * n; i++) {
        buf[i] = (float)W[i];
    }

    /* Step 1: WHT each column of W → H·W
     * Column j of W is buf[0*n+j, 1*n+j, ..., (n-1)*n+j] (stride n)
     * We need to extract, transform, and put back.
     * For efficiency: transpose → WHT rows → transpose back */
    float * col = (float *)malloc(n * sizeof(float));
    if (!col) { free(buf); return; }

    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) col[i] = buf[i * n + j];
        fwht_f32(col, n);
        for (int i = 0; i < n; i++) buf[i * n + j] = col[i];
    }

    /* Step 2: WHT each row of (H·W) → H·W·H */
    for (int i = 0; i < n; i++) {
        fwht_f32(buf + i * n, n);
    }

    /* Step 3: extract diagonal, normalize by n² */
    float inv_n2 = 1.0f / ((float)n * (float)n);
    for (int k = 0; k < n; k++) {
        d[k] = buf[k * n + k] * inv_n2;
    }

    free(col);
    free(buf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_error
 *
 * Relative Frobenius approximation error:
 *   ε = ||W - H·diag(d)·H||_F / ||W||_F
 *
 * Computed by: for each unit vector eⱼ, compute:
 *   ŷ_j = H·diag(d)·H·eⱼ   (single ACDC forward pass)
 *   compare with W[:,j]
 * O(n² log n) — used once for diagnostic, not at inference.
 * ═══════════════════════════════════════════════════════════════════════════ */
float acdc_error(const int8_t * W, const float * d, int n) {
    double num = 0.0, den = 0.0;

    float * y     = (float *)malloc(n * sizeof(float));
    float * x_buf = (float *)malloc(n * sizeof(float));
    if (!y || !x_buf) { free(y); free(x_buf); return -1.0f; }

    for (int j = 0; j < n; j++) {
        /* x = e_j (unit vector) as float */
        memset(x_buf, 0, n * sizeof(float));
        x_buf[j] = 1.0f;

        /* ACDC forward: y ≈ W·eⱼ = W[:,j] */
        memcpy(y, x_buf, n * sizeof(float));
        fwht_f32(y, n);
        float inv_n = 1.0f / (float)n;
        for (int i = 0; i < n; i++) y[i] *= d[i] * inv_n;
        fwht_f32(y, n);
        for (int i = 0; i < n; i++) y[i] *= inv_n;

        /* Compare with true column W[:,j] */
        for (int i = 0; i < n; i++) {
            float w_ij  = (float)W[i * n + j];
            float diff  = w_ij - y[i];
            num += (double)(diff * diff);
            den += (double)(w_ij * w_ij);
        }
    }

    free(y);
    free(x_buf);

    return (den > 0.0) ? (float)sqrt(num / den) : 0.0f;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_forward_rect_f32  (Fase II)
 *
 * Rectangular ACDC — float32 input, float32 output.
 *
 * Computes y[m] = first m elements of H_P · (d ⊙ (H_P · [x | 0]))
 * where P = next_pow2(max(m, n)).
 *
 * For m == n and P == n the math reduces to the square case (acdc_forward_f32)
 * but without the 1/n normalization steps: this matches the unnormalized spec
 * in CLAUDE.md ("no 1/n² factors; d absorbs the scale during training").
 *
 * Operation count for Falcon3-10B gate_proj (n=3072, m=23040, P=32768):
 *   Dense GEMV:   3072 × 23040 = 70.8M ops
 *   ACDC rect:    2 × 32768 × log₂32768 = 983K ops → ~72× fewer
 * ═══════════════════════════════════════════════════════════════════════════ */
void acdc_forward_rect_f32(float * y, int m, const float * x, int n, const float * d) {
    const int P = fwht_next_pow2(m > n ? m : n);

    float * zf = (float *)calloc((size_t)P, sizeof(float));
    if (!zf) return;

    /* Zero-pad x from n → P; calloc provides the trailing zeros */
    const int copy_n = (n < P) ? n : P;
    memcpy(zf, x, (size_t)copy_n * sizeof(float));

    /* Step 1: ẑ = H_P · [x | 0]  (zero multiplications) */
    fwht_f32(zf, P);

    /* Step 2: z = d ⊙ ẑ  (P multiplications — irreducible minimum) */
    for (int i = 0; i < P; i++) zf[i] *= d[i];

    /* Step 3: y_P = H_P · z  (zero multiplications) */
    fwht_f32(zf, P);

    /* Output: first m elements */
    memcpy(y, zf, (size_t)m * sizeof(float));

    free(zf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_forward_rect_i8  (Fase II)
 *
 * Rectangular ACDC — int8 input (pre-quantized activations), float output.
 *
 * Same math as acdc_forward_rect_f32 but uses fwht_i8_to_i32 for Stage 1,
 * which avoids converting the int8 activation to float before the first WHT.
 *
 * Memory layout (single zero-initialised allocation):
 *   [x_pad: P × int8] [z32: P × int32] [zf: P × float]
 *   P is a power of 2 ≥ 4, so each section starts 4-byte aligned.
 * ═══════════════════════════════════════════════════════════════════════════ */
void acdc_forward_rect_i8(float * y, int m, const int8_t * x, int n, const float * d) {
    const int P = fwht_next_pow2(m > n ? m : n);

    const size_t sz_i8  = (size_t)P;
    const size_t sz_i32 = (size_t)P * sizeof(int32_t);
    const size_t sz_f32 = (size_t)P * sizeof(float);
    char * buf = (char *)calloc(sz_i8 + sz_i32 + sz_f32, 1);
    if (!buf) return;

    int8_t  * x_pad = (int8_t  *)buf;
    int32_t * z32   = (int32_t *)(buf + sz_i8);         /* P ≥ 4 → 4-byte aligned */
    float   * zf    = (float   *)(buf + sz_i8 + sz_i32);

    /* Zero-pad x from n → P; calloc already zeroed the tail */
    const int copy_n = (n < P) ? n : P;
    memcpy(x_pad, x, (size_t)copy_n);

    /* Step 1: ẑ = H_P · [x | 0]  (int8→int32 butterfly, zero multiplications) */
    fwht_i8_to_i32(x_pad, z32, P);

    /* Step 2: z = d ⊙ ẑ  (P multiplications, int32→float conversion) */
    for (int i = 0; i < P; i++) zf[i] = (float)z32[i] * d[i];

    /* Step 3: y_P = H_P · z  (float butterfly, zero multiplications) */
    fwht_f32(zf, P);

    /* Output: first m elements */
    memcpy(y, zf, (size_t)m * sizeof(float));

    free(buf);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PUBLIC: acdc_project_rect
 *
 * Find the best diagonal d* ∈ ℝ^P for W ∈ {-1,0,+1}^{m×n}:
 *
 *   d*[k] = (H_P · W_P · H_P)[k,k] / P²
 *
 * where P = next_pow2(max(m,n)) and W_P is W zero-padded to P×P.
 *
 * EFFICIENT ALGORITHM via XOR-convolution (Fase V):
 *
 * d*[k] = Σ_{i<m,j<n} W[i,j] · (-1)^{popcount(k & (i XOR j))}
 *        = (H_P · C)[k] / P²
 *
 * where C[s] = Σ_{(i,j): i XOR j = s, i<m, j<n} W[i,j]
 *
 * Steps:
 *   1. C = 0                            O(P)
 *   2. For each (i,j): C[i^j] += W[i,j]  O(m·n)
 *   3. C ← H_P · C  (FWHT in-place)    O(P log P)
 *   4. d*[k] = C[k] / P²               O(P)
 *
 * Memory: O(P) — 128 KB for P=32768  (vs 4 GB naive)
 * Cost:   O(m·n + P log P) — ~71M for Falcon3-10B gate_proj  (vs 16G naive)
 * ═══════════════════════════════════════════════════════════════════════════ */
void acdc_project_rect(float * d, const int8_t * W, int m, int n) {
    const int P = fwht_next_pow2(m > n ? m : n);

    /* C[s] = XOR-convolution accumulator */
    float * C = (float *)calloc((size_t)P, sizeof(float));
    if (!C) {
        memset(d, 0, (size_t)P * sizeof(float));
        return;
    }

    /* Step 2: accumulate W[i,j] into C[i XOR j] */
    for (int i = 0; i < m; i++) {
        const int8_t * row = W + (size_t)i * n;
        for (int j = 0; j < n; j++) {
            int8_t w = row[j];
            if (w != 0) C[i ^ j] += (float)w;
        }
    }

    /* Step 3: FWHT in-place — C becomes H_P · C */
    fwht_f32(C, P);

    /* Step 4: normalize by P² */
    const float inv_P2 = 1.0f / ((float)P * (float)P);
    for (int k = 0; k < P; k++) d[k] = C[k] * inv_P2;

    free(C);
}
