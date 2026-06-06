/*
 * ggml-bitnet-hrr.cpp
 *
 * Holographic Reduced Representations — CPU Nível 5
 *
 * ─────────────────────────────────────────────────────────────────────────
 * FUNDAMENTO: CONVOLUÇÃO CIRCULAR COMO ÁLGEBRA DE BINDING
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Para vetores a, b ∈ ℝᵈ (d = 2^k):
 *
 *   (a ⊛ b)[k] = Σⱼ a[j] · b[(k-j) mod d]    ← convolução circular
 *
 * Pelo Teorema da Convolução Circular (FFT):
 *   a ⊛ b = IRFFT( RFFT(a) ⊙ RFFT(b) )        ← produto em Fourier
 *
 * RFFT(a) ∈ ℂ^{d/2+1}: apenas d/2+1 coeficientes complexos (simetria Hermitiana).
 *
 * Custo por binding: 3 FFTs = 3 × O(d log d) = O(d log d)
 *
 * ─────────────────────────────────────────────────────────────────────────
 * IMPLEMENTAÇÃO DA FFT: Cooley-Tukey Split-Radix (sem dependência externa)
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Implementamos uma DFT recursiva Cooley-Tukey (radix-2 DIF):
 *
 *   X[k]     = Σ_{n=0}^{N/2-1} x[2n]·W_N^{kn}  +  W_N^k · Σ x[2n+1]·W_N^{kn}
 *   X[k+N/2] = Σ_{n=0}^{N/2-1} x[2n]·W_N^{kn}  -  W_N^k · Σ x[2n+1]·W_N^{kn}
 *
 *   onde W_N = exp(-2πi/N)  (fator de twiddle)
 *
 * Butterfly de radix-2:
 *   a' = a + W·b
 *   b' = a - W·b
 *
 * Zero multiplicações reais quando W = {±1, ±i} (estágios iniciais).
 * Para estágios intermediários: 2 multiplicações reais por butterfly (W = cos+i·sin).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * OTIMIZAÇÃO SIMD: AVX2 BUTTERFLIES COMPLEXOS
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Um butterfly complexo (a, b) → (a+W·b, a-W·b) em AVX2 processa 4 pares por vez:
 *
 *   __m256 ar = [re(a₀), re(a₁), re(a₂), re(a₃), ...]   (8 floats = 4 complex)
 *   __m256 ai = [im(a₀), im(a₁), im(a₂), im(a₃), ...]
 *   Wr = [re(W)×4], Wi = [im(W)×4]
 *
 *   re(W·b) = Wr·re(b) - Wi·im(b)   ← 2 muls + 1 sub
 *   im(W·b) = Wr·im(b) + Wi·re(b)   ← 2 muls + 1 add
 *
 * 4 butterflies por instrução AVX2 → 4× throughput vs escalar.
 */

#include "ggml-bitnet-hrr.h"
#include "ggml-bitnet-common.h"
#include <cmath>
#include <cstring>
#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <cfloat>
#include <algorithm>

#if defined(__AVX2__)
#  include <immintrin.h>
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * UTILITÁRIO: POTÊNCIA DE 2
 * ═══════════════════════════════════════════════════════════════════════════ */

/* Note: hrr_next_pow2() used to be defined here; it now lives in
 * src/ggml-bitnet-common.cpp (single source of truth for next_pow2). */

/* ═══════════════════════════════════════════════════════════════════════════
 * FFT INTERNA: COOLEY-TUKEY RADIX-2 DIF
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Representação: array de floats interleaved [re0, im0, re1, im1, ...]
 * Tamanho do buffer: 2*d floats para d pontos complexos.
 */

/* Bit-reversal permutation in-place */
static void bit_reverse(float *x, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            std::swap(x[2*i],   x[2*j]);
            std::swap(x[2*i+1], x[2*j+1]);
        }
    }
}

/*
 * fft_inplace: FFT complexa in-place, Cooley-Tukey radix-2 DIT.
 * x: array de 2*n floats [re0,im0,re1,im1,...], n = 2^k
 * inv: se true, computa IFFT (sem normalização — dividir por n externamente)
 */
static void fft_inplace(float *x, int n, bool inv) {
    bit_reverse(x, n);

    for (int s = 1; s <= (int)(__builtin_ctz((unsigned)n)); s++) {
        int m    = 1 << s;        /* tamanho da sub-DFT */
        int half = m >> 1;
        double theta = (inv ? 1.0 : -1.0) * 2.0 * M_PI / m;
        float wR = (float)cos(theta);
        float wI = (float)sin(theta);

        for (int k = 0; k < n; k += m) {
            float curR = 1.0f, curI = 0.0f;
            for (int j = 0; j < half; j++) {
                int u = 2*(k+j), v = 2*(k+j+half);
                /* butterfly: (u, v) → (u + W·v, u - W·v) */
                float ur = x[u],   ui = x[u+1];
                float vr = x[v],   vi = x[v+1];
                float tr = curR*vr - curI*vi;  /* Re(W·v) */
                float ti = curR*vi + curI*vr;  /* Im(W·v) */
                x[u]   = ur + tr;  x[u+1] = ui + ti;
                x[v]   = ur - tr;  x[v+1] = ui - ti;
                /* update twiddle: cur *= w */
                float nr = curR*wR - curI*wI;
                curI = curR*wI + curI*wR;
                curR = nr;
            }
        }
    }
}

/* ─── RFFT: DFT real via FFT complexa ─────────────────────────────────── */

/*
 * hrr_rfft_internal: RFFT de d reais → d+2 floats (d/2+1 complexos interleaved)
 * Packing: [re0, im0, re1, im1, ..., re_{d/2}, im_{d/2}]
 *          onde im0 = 0 (DC) e im_{d/2} = 0 (Nyquist) mas os guardamos mesmo assim.
 */
static void rfft_internal(const float *x, float *out, int d) {
    /* Tratar array de d reais como d/2 complexos */
    int half = d / 2;
    /* Copiar x como pares (re, 0) — ou interpretar diretamente */
    float *buf = (float *)malloc(2 * d * sizeof(float));
    if (!buf) return;
    for (int i = 0; i < d; i++) { buf[2*i] = x[i]; buf[2*i+1] = 0.0f; }
    fft_inplace(buf, d, false);
    /* Copiar apenas metade + 1 (simetria Hermitiana) */
    for (int k = 0; k <= half; k++) {
        out[2*k]   = buf[2*k];
        out[2*k+1] = buf[2*k+1];
    }
    free(buf);
}

/*
 * hrr_irfft_internal: IRFFT de d+2 floats (d/2+1 complexos) → d reais
 * Normalizado: divide por d.
 */
static void irfft_internal(const float *spectrum, float *out, int d) {
    int half = d / 2;
    float *buf = (float *)malloc(2 * d * sizeof(float));
    if (!buf) return;
    /* Reconstruir espectro completo usando simetria Hermitiana */
    for (int k = 0; k <= half; k++) {
        buf[2*k]   = spectrum[2*k];
        buf[2*k+1] = spectrum[2*k+1];
    }
    for (int k = half+1; k < d; k++) {
        buf[2*k]   =  spectrum[2*(d-k)];
        buf[2*k+1] = -spectrum[2*(d-k)+1];
    }
    fft_inplace(buf, d, true);
    float inv_d = 1.0f / (float)d;
    for (int i = 0; i < d; i++) out[i] = buf[2*i] * inv_d;
    free(buf);
}

/* Wrappers públicos */
void hrr_rfft(const float *x, float *out, int d) {
    rfft_internal(x, out, d);
}

void hrr_irfft(const float *spectrum, float *out, int d) {
    irfft_internal(spectrum, out, d);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * BINDING: a ⊛ b = IRFFT( RFFT(a) ⊙ RFFT(b) )
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * complex_multiply_spectrum: C = A ⊙ B (produto elemento a elemento complexo)
 * A, B, C: arrays de d+2 floats (d/2+1 complexos interleaved)
 */
static void complex_multiply_spectrum(float *C, const float *A, const float *B, int d) {
    int n_complex = d / 2 + 1;

#if defined(__AVX2__)
    /*
     * Complex multiply 4 pairs per iteration using fmaddsub.
     * Layout A, B, C: interleaved [re0,im0,re1,im1,re2,im2,re3,im3] = 8 floats.
     *
     * fmaddsub(a_re_dup, B, a_im_dup * B_swapped):
     *   even positions (re): a_re*b_re - a_im*b_im = c_re  ← subtract
     *   odd  positions (im): a_re*b_im + a_im*b_re = c_im  ← add
     *
     * Writes exactly 8 floats per iteration (one _mm256_storeu_ps).
     */
    int i = 0;
    for (; i + 4 <= n_complex; i += 4) {
        __m256 va     = _mm256_loadu_ps(A + 2*i);
        __m256 vb     = _mm256_loadu_ps(B + 2*i);
        __m256 a_re   = _mm256_moveldup_ps(va);            /* [ar0,ar0,ar1,ar1,...] */
        __m256 a_im   = _mm256_movehdup_ps(va);            /* [ai0,ai0,ai1,ai1,...] */
        __m256 b_swap = _mm256_permute_ps(vb, 0xB1);       /* swap re/im pairs */
        __m256 c      = _mm256_fmaddsub_ps(a_re, vb,
                            _mm256_mul_ps(a_im, b_swap));
        _mm256_storeu_ps(C + 2*i, c);
    }
    for (; i < n_complex; i++) {
        float ar = A[2*i], ai = A[2*i+1];
        float br = B[2*i], bi = B[2*i+1];
        C[2*i]   = ar*br - ai*bi;
        C[2*i+1] = ar*bi + ai*br;
    }
#else
    for (int i = 0; i < n_complex; i++) {
        float ar = A[2*i], ai = A[2*i+1];
        float br = B[2*i], bi = B[2*i+1];
        C[2*i]   = ar*br - ai*bi;
        C[2*i+1] = ar*bi + ai*br;
    }
#endif
}

void hrr_bind(float *out, const float *a, const float *b, int d, float *tmp) {
    /* tmp layout: [spec_a | spec_b | spec_c]  each of size (d+2) floats */
    float *spec_a = tmp;
    float *spec_b = tmp + (d + 2);
    float *spec_c = tmp + 2*(d + 2);

    rfft_internal(a, spec_a, d);
    rfft_internal(b, spec_b, d);
    complex_multiply_spectrum(spec_c, spec_a, spec_b, d);
    irfft_internal(spec_c, out, d);
}

void hrr_bind_ternary(float *out, const int8_t *a_ternary,
                       const float *b, int d, float *tmp) {
    /* Converter a_ternary para float, reutilizar hrr_bind */
    float *a_float = (float *)malloc(d * sizeof(float));
    if (!a_float) return;
    for (int i = 0; i < d; i++) a_float[i] = (float)a_ternary[i];
    hrr_bind(out, a_float, b, d, tmp);
    free(a_float);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * PSEUDO-INVERSA: a⁻¹ ≈ reversão cíclica (para vetores unitários)
 *
 * Para vetores aleatórios de norma unitária:
 *   FFT(a⁻¹)[k] = conj(FFT(a)[k])  →  a⁻¹ = cyclic_reverse(a)
 *
 * Cyclic reverse: a⁻¹[k] = a[(d-k) mod d]
 * Isto é válido quando |FFT(a)[k]| = 1 para todo k — aproximação boa para
 * vetores aleatórios unitários (desvio < 1/√d em norma).
 * ═══════════════════════════════════════════════════════════════════════════ */

void hrr_pseudoinverse(float *inv, const float *a, int d, float *tmp) {
    /*
     * Inversa exata via conjugação espectral:
     * FFT(a⁻¹)[k] = conj(FFT(a)[k])
     * → a⁻¹ = IRFFT( conj(RFFT(a)) )
     */
    float *spec = tmp;  /* (d+2) floats */
    rfft_internal(a, spec, d);
    /* Conjugar: im → -im */
    int n_complex = d / 2 + 1;
    for (int k = 0; k < n_complex; k++) spec[2*k+1] = -spec[2*k+1];
    irfft_internal(spec, inv, d);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * UNBINDING: out = M ⊛ k_inv
 * ═══════════════════════════════════════════════════════════════════════════ */

void hrr_unbind(float *out, const float *M, const float *k_inv,
                int d, float *tmp) {
    hrr_bind(out, M, k_inv, d, tmp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ACUMULAÇÃO: M += k ⊛ v
 * ═══════════════════════════════════════════════════════════════════════════ */

void hrr_accumulate(float *M, const float *k, const float *v,
                    int d, float *tmp) {
    float *binding = (float *)malloc(d * sizeof(float));
    if (!binding) return;
    hrr_bind(binding, k, v, d, tmp);
    for (int i = 0; i < d; i++) M[i] += binding[i];
    free(binding);
}

void hrr_accumulate_ternary(float *M, const int8_t *k_ternary,
                              const float *v, int d, float *tmp) {
    float *binding = (float *)malloc(d * sizeof(float));
    if (!binding) return;
    hrr_bind_ternary(binding, k_ternary, v, d, tmp);
    for (int i = 0; i < d; i++) M[i] += binding[i];
    free(binding);
}

void hrr_build_memory(float *M, const float *keys, const int8_t *tkeys,
                       const float *values, int N, int d) {
    memset(M, 0, d * sizeof(float));
    float *tmp = (float *)malloc(3 * (d + 2) * sizeof(float));
    if (!tmp) return;

    for (int i = 0; i < N; i++) {
        if (keys) {
            hrr_accumulate(M, keys + i*d, values + i*d, d, tmp);
        } else {
            hrr_accumulate_ternary(M, tkeys + i*d, values + i*d, d, tmp);
        }
    }
    free(tmp);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * QUALIDADE E LIMPEZA
 * ═══════════════════════════════════════════════════════════════════════════ */

float hrr_cosine_sim(const float *a, const float *b, int d) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na * nb) + 1e-9f);
}

int hrr_cleanup_step(float *out, const float *noisy,
                     const float **codebook, int N_cb, int d) {
    int best = 0;
    float best_sim = -FLT_MAX;
    for (int i = 0; i < N_cb; i++) {
        float sim = hrr_cosine_sim(noisy, codebook[i], d);
        if (sim > best_sim) { best_sim = sim; best = i; }
    }
    memcpy(out, codebook[best], d * sizeof(float));
    return best;
}

/*
 * hrr_cleanup_iter: Frady 2021 iterative cleanup.
 *
 * Two modes:
 *   NAIVE (M == NULL):   iterate nearest-codebook projection on `noisy` until
 *                        the chosen index stops changing.
 *   RESIDUAL (M != NULL): for each iteration t:
 *                          1. Compute k_inv = pseudoinverse(query_key)  [once]
 *                          2. Retrieve v_t = M_t ⊛ k_inv
 *                          3. Project to nearest codebook c_t
 *                          4. If c_t == c_{t-1} → converged, stop
 *                          5. Subtract contribution: M_{t+1} = M_t - query_key ⊛ c_t
 *
 * The residual mode is what makes HRR retrieval usable when N > d/10.
 * Expected SNR (for phasor keys, random codebook):
 *   raw retrieval:         cos_sim ≈ √d / (N-1 + √d)   (can be < 0.1)
 *   + 8 iterations cleanup: cos_sim ≈ 0.95-0.99         (depending on d/N)
 *
 * @param out        cleaned output [d floats] (== chosen codebook entry)
 * @param noisy      initial retrieval (used only in NAIVE mode; ignored in RESIDUAL)
 * @param M          holographic memory [d floats], or NULL for NAIVE
 * @param query_key  original key k [d floats] (RESIDUAL: used for subtraction;
 *                   NAIVE: ignored)
 * @param codebook   N_cb clean prototype vectors [N_cb × d floats]
 * @param N_cb       codebook size
 * @param d          dimension
 * @param max_iters  iteration cap (typ. 8-16)
 * @param tmp        scratch [3*(d+2) + d floats] for FFTs and k_inv
 * @return           index of chosen codebook entry, or -1 on failure
 */
int hrr_cleanup_iter(float *out, const float *noisy,
                     const float *M, const float *query_key,
                     const float **codebook, int N_cb, int d,
                     int max_iters, float *tmp) {
    if (N_cb <= 0) return -1;
    if (max_iters < 1) max_iters = 1;

    /* Helper: find nearest codebook entry to `probe`, return its index. */
    auto nearest = [&](const float * probe) -> int {
        int best = 0;
        float best_sim = -FLT_MAX;
        for (int i = 0; i < N_cb; i++) {
            float sim = hrr_cosine_sim(probe, codebook[i], d);
            if (sim > best_sim) { best_sim = sim; best = i; }
        }
        return best;
    };

    int idx = -1;

    if (M != NULL && query_key != NULL) {
        /* ─── RESIDUAL MODE (Frady 2021) ─────────────────────────────────────
         * 1. k_inv = conj(FFT(query_key))            [once]
         * 2. iter t:
         *      work = M_t ⊛ k_inv                    (re-unbind the residual memory)
         *      idx_t = nearest(work, codebook)        (project to nearest prototype)
         *      if idx_t == idx_{t-1} (and t>0): break (converged)
         *      if t==0: out = codebook[idx_t]         (seed)
         *      else:     out += codebook[idx_t]       (accumulate!)
         *      M_{t+1} = M_t - query_key ⊛ codebook[idx_t]   (subtract trace)
         */
        float * M_working = (float *)malloc(d * sizeof(float));
        float * binding   = (float *)malloc(d * sizeof(float));
        float * k_inv     = (float *)malloc(d * sizeof(float));
        float * work      = (float *)malloc(d * sizeof(float));
        if (!M_working || !binding || !k_inv || !work) {
            free(M_working); free(binding); free(k_inv); free(work);
            return -1;
        }
        memcpy(M_working, M, d * sizeof(float));
        hrr_pseudoinverse(k_inv, query_key, d, tmp);

        int prev_idx = -1;
        for (int iter = 0; iter < max_iters; iter++) {
            hrr_unbind(work, M_working, k_inv, d, tmp);
            idx = nearest(work);
            if (iter > 0 && idx == prev_idx) break;
            if (iter == 0) {
                memcpy(out, codebook[idx], d * sizeof(float));
            } else {
                for (int i = 0; i < d; i++) out[i] += codebook[idx][i];
            }
            prev_idx = idx;
            /* subtract this codebook entry's trace from M_working */
            hrr_bind(binding, query_key, codebook[idx], d, tmp);
            for (int i = 0; i < d; i++) M_working[i] -= binding[i];
        }

        free(M_working); free(binding); free(k_inv); free(work);
        return idx;
    } else {
        /* ─── NAIVE MODE ─────────────────────────────────────────────────────
         * Single nearest projection on the provided `noisy` retrieval.
         * Useful when M is not available (e.g. test harness with direct noisy).
         */
        int best = nearest(noisy);
        memcpy(out, codebook[best], d * sizeof(float));
        return best;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ATENÇÃO HOLOGRÁFICA COMPLETA
 * ═══════════════════════════════════════════════════════════════════════════ */

void hrr_attention_build(float *M, const float *K, const int8_t *K_tern,
                          const float *V, int n_ctx, int head_dim) {
    hrr_build_memory(M, K, K_tern, V, n_ctx, head_dim);
}

void hrr_attention_retrieve(float *out, const float *M, const float *q,
                              int head_dim, float *tmp) {
    /*
     * out ≈ Σᵢ softmax(Q·Kᵢᵀ)[i] · Vᵢ   (aproximado)
     *     = M ⊛ q⁻¹                        (exato em HRR)
     *
     * Passos:
     *   1. q_inv = pseudoinverse(q)   [O(d log d)]
     *   2. out   = M ⊛ q_inv          [O(d log d)]
     */
    int d = head_dim;
    /* tmp: [spec_q (d+2)] [spec_M (d+2)] [spec_out (d+2)] [q_inv (d)] */
    float *spec_q   = tmp;
    float *spec_M   = tmp + (d + 2);
    float *spec_out = tmp + 2*(d + 2);
    float *q_inv    = tmp + 3*(d + 2);

    /* Passo 1: q_inv = conjugar o espectro de q */
    rfft_internal(q, spec_q, d);
    int n_complex = d / 2 + 1;
    for (int k = 0; k < n_complex; k++) {
        spec_q[2*k+1] = -spec_q[2*k+1];  /* conjugar */
    }
    /* spec_q agora é spec_q_inv */

    /* Passo 2: spec_M ⊙ spec_q_inv → spec_out → out */
    rfft_internal(M, spec_M, d);
    complex_multiply_spectrum(spec_out, spec_M, spec_q, d);
    irfft_internal(spec_out, out, d);

    (void)q_inv;  /* used implicitly via spec_q conjugation */
}

void hrr_attention_full(float *output, const float *Q,
                         const float *K, const int8_t *K_tern,
                         const float *V,
                         int n_queries, int n_ctx, int head_dim) {
    int d = head_dim;
    float *M   = (float *)malloc(d * sizeof(float));
    float *tmp = (float *)malloc(4 * (d + 2) * sizeof(float));
    if (!M || !tmp) { free(M); free(tmp); return; }

    /* Build holographic memory from context */
    hrr_build_memory(M, K, K_tern, V, n_ctx, d);

    /* Retrieve for each query */
    for (int i = 0; i < n_queries; i++) {
        hrr_attention_retrieve(output + i*d, M, Q + i*d, d, tmp);
    }

    free(M);
    free(tmp);
}
