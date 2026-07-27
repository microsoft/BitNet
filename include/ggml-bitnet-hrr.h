#pragma once
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ggml-bitnet-hrr.h — Holographic Reduced Representations (HRR)
 *
 * ─────────────────────────────────────────────────────────────────────────
 * MATHEMATICAL FOUNDATION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Kanerva (1988): Sparse Distributed Memory
 * Plate (1994):   Holographic Reduced Representations
 *
 * CIRCULAR CONVOLUTION (binding operation):
 *
 *   (a ⊛ b)[k] = Σⱼ a[j] · b[(k-j) mod d]
 *
 *   Equivalently (Convolution Theorem):
 *   a ⊛ b = IFFT( FFT(a) ⊙ FFT(b) )         — element-wise complex multiply
 *
 *   Cost: O(d log d) via FFT
 *
 * ALGEBRAIC PROPERTIES (abelian group under ⊛ for unit-norm vectors):
 *   Commutativity:   a ⊛ b = b ⊛ a
 *   Associativity:   (a ⊛ b) ⊛ c = a ⊛ (b ⊛ c)
 *   Identity:        δ ⊛ a = a   (δ[0]=1, δ[k>0]=0)
 *   Inverse:         a⁻¹ = IFFT( conj(FFT(a)) )  [for unit-norm vectors]
 *
 * ─────────────────────────────────────────────────────────────────────────
 * HOLOGRAPHIC ASSOCIATIVE MEMORY
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Storage: N key-value pairs encoded into one vector M ∈ ℝᵈ:
 *
 *   M = Σᵢ (kᵢ ⊛ vᵢ)      ← superposition of bindings
 *
 * Retrieval of value v_j given key k_j:
 *
 *   ṽ_j = M ⊛ k_j⁻¹
 *        = (Σᵢ kᵢ ⊛ vᵢ) ⊛ k_j⁻¹
 *        = v_j + Σ_{i≠j} (kᵢ ⊛ k_j⁻¹) ⊛ vᵢ
 *        ≈ v_j   (noise ~ (N-1)/√d for random orthogonal keys)
 *
 * Retrieval error: ||ṽ_j - v_j|| ≈ (N-1)/√d
 * For d=4096, N=64: error ≈ 0.98  — need cleanup or larger d
 * For d=65536, N=64: error ≈ 0.001 — excellent
 *
 * ─────────────────────────────────────────────────────────────────────────
 * CONNECTION TO TRANSFORMER ATTENTION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Standard attention (per head):
 *   Build:    K ∈ ℝ^{n×d}, V ∈ ℝ^{n×d}   — O(n·d) space
 *   Retrieve: A = softmax(Q·Kᵀ/√d)·V      — O(n²·d) time
 *
 * HRR attention (per head):
 *   Build:    M = Σᵢ kᵢ ⊛ vᵢ ∈ ℝᵈ         — O(d) space, O(n·d·log d) build
 *   Retrieve: ṽ = M ⊛ q⁻¹                  — O(d·log d) time, INDEPENDENT of n
 *
 * Speedup: O(n²) → O(n log n) for the attention mechanism
 * For n=2048, d=128: 2048/log₂(2048) ≈ 186× throughput improvement
 *
 * ─────────────────────────────────────────────────────────────────────────
 * FREQUENCY DOMAIN INTERPRETATION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * For unit-norm vectors a, b ∈ ℝᵈ with FFT Â, B̂ ∈ ℂ^{d/2+1}:
 *
 *   FFT(a ⊛ b)[k] = Â[k] · B̂[k]
 *                 = |Â[k]|·|B̂[k]| · exp(i(φₐₖ + φᵦₖ))
 *
 * Binding = phase addition in Fourier space.
 * For unit-magnitude spectra: binding IS a phase rotation.
 *
 * This is the same structure as RoPE (Rotary Position Embedding):
 *   RoPE: q·exp(i·m·θ)  — phase rotation by token position
 *   HRR:  a ⊛ b          — phase sum of key and value spectra
 *
 * ─────────────────────────────────────────────────────────────────────────
 * IMPLEMENTATION STRATEGY
 * ─────────────────────────────────────────────────────────────────────────
 *
 * We use the real FFT (RFFT) since inputs are real:
 *   RFFT(a) ∈ ℂ^{d/2+1}  (d/2+1 complex coefficients, not d)
 *   IRFFT: inverse of RFFT
 *
 * Storage for M: d float32 values (real domain)
 * Temporary: d/2+1 complex64 per FFT call
 *
 * For ternary keys (Level 2 integration):
 *   k_ternary ∈ {-1, 0, +1}^d → treated as float for FFT
 *   Binding k ⊛ v is exact for any k type; no precision loss
 */

/* ─── FFT primitives (real-valued) ───────────────────────────────────────
 *
 * We use a self-contained Cooley-Tukey split-radix FFT implementation
 * (no external FFTW dependency). For d = power of 2 only.
 */

/* hrr_next_pow2: smallest power of 2 >= n */
int hrr_next_pow2(int n);

/*
 * hrr_rfft: in-place real FFT.
 * Input:  x[0..d-1] real floats (d = power of 2)
 * Output: x reinterpreted as d/2+1 complex pairs [re, im] in first d+2 floats
 *         (standard RFFT packing: x[0]=DC, x[d]=Nyquist, interleaved otherwise)
 * Caller must provide out[d+2] — minimum d+2 floats.
 */
void hrr_rfft(const float *x, float *out, int d);

/*
 * hrr_irfft: inverse real FFT.
 * Input:  spectrum[d+2] (RFFT output packing)
 * Output: x[d] real floats (unnormalized — divide by d for normalized result)
 */
void hrr_irfft(const float *spectrum, float *out, int d);

/* ─── Phasor keys — unit-magnitude spectrum, exact inverse ───────────────
 *
 * A phasor key k satisfies |RFFT(k)[j]| = 1 for every frequency bin j.
 * This makes spectral conjugation an EXACT inverse:
 *
 *   k ⊛ hrr_phasor_inv(k) = δ          (Kronecker delta, to FP precision)
 *
 * Retrieval from a memory of N pairs has only superposition noise (N-1
 * cross-talk terms), with zero inversion error.  Supports N ≈ d/4 reliable
 * pairs (vs d/10 for Gaussian random keys).
 */

/*
 * hrr_phasor_key_init: generate a reproducible phasor key.
 *
 * The key is produced by IRFFT of a unit-magnitude spectrum with random phases
 * drawn from an xorshift64 RNG seeded by `seed`.  Different seeds give
 * statistically independent keys (pseudo-orthogonal in expectation).
 *
 * @param k     output phasor key [d floats]; ||k||_2 = 1 exactly
 * @param d     dimension (must be power of 2)
 * @param seed  RNG seed; 0 uses default seed 0xDEADBEEFCAFEBABE
 */
void hrr_phasor_key_init(float *k, int d, uint64_t seed);

/*
 * hrr_phasor_inv: exact inverse of a phasor key via spectral conjugation.
 *
 * For keys generated by hrr_phasor_key_init():
 *   k ⊛ hrr_phasor_inv(k) = δ   (to floating-point precision)
 *
 * Compare: hrr_pseudoinverse gives only an approximate inverse for Gaussian
 * random keys (error O(1/√d) per element), but is exact for phasor keys.
 *
 * @param inv  output exact inverse [d floats]
 * @param k    phasor key [d floats] from hrr_phasor_key_init
 * @param d    dimension (must be power of 2)
 * @param tmp  scratch buffer [2*(d+2) floats]
 */
void hrr_phasor_inv(float *inv, const float *k, int d, float *tmp);

/* ─── Binding (circular convolution) ─────────────────────────────────────*/

/*
 * hrr_bind: out = a ⊛ b  (circular convolution, O(d log d))
 *
 * @param out  output [d floats], may alias a or b
 * @param a    first operand [d floats]
 * @param b    second operand [d floats]
 * @param d    dimension (must be power of 2)
 * @param tmp  scratch buffer [3*(d+2) floats] — provided by caller
 */
void hrr_bind(float *out, const float *a, const float *b, int d, float *tmp);

/*
 * hrr_bind_ternary: out = a_ternary ⊛ b  where a ∈ {-1, 0, +1}^d
 *
 * Optimized for ternary keys: skips zero entries in FFT multiplication.
 * Same semantics as hrr_bind but ~2× faster for 50%-sparse ternary keys.
 */
void hrr_bind_ternary(float *out, const int8_t *a_ternary,
                       const float *b, int d, float *tmp);

/* ─── Unbinding (retrieval) ───────────────────────────────────────────── */

/*
 * hrr_pseudoinverse: compute a⁻¹ for unbinding.
 *
 * For random unit-norm vectors: a⁻¹ ≈ a reversed (cyclic shift by 1).
 * Exact inverse: IFFT( conj(FFT(a)) ) — only needed when |FFT(a)[k]| ≠ 1.
 *
 * @param inv  output [d floats]
 * @param a    input key [d floats]
 * @param d    dimension
 * @param tmp  scratch [2*(d+2) floats]
 */
void hrr_pseudoinverse(float *inv, const float *a, int d, float *tmp);

/*
 * hrr_unbind: out ≈ v_j  given M and k_j
 *
 * out = M ⊛ k_j⁻¹
 *
 * @param out    retrieved value [d floats]
 * @param M      holographic memory [d floats]
 * @param k_inv  inverse key from hrr_pseudoinverse [d floats]
 * @param d      dimension
 * @param tmp    scratch [3*(d+2) floats]
 */
void hrr_unbind(float *out, const float *M, const float *k_inv,
                int d, float *tmp);

/* ─── Memory accumulation ─────────────────────────────────────────────── */

/*
 * hrr_accumulate: M += k ⊛ v  (store one key-value pair)
 *
 * Superposition: binding is additive in the memory vector.
 *
 * @param M    holographic memory [d floats], updated in-place
 * @param k    key [d floats] (can be ternary — use hrr_accumulate_ternary)
 * @param v    value [d floats]
 * @param d    dimension
 * @param tmp  scratch [3*(d+2) floats]
 */
void hrr_accumulate(float *M, const float *k, const float *v,
                    int d, float *tmp);

/*
 * hrr_accumulate_ternary: M += k_ternary ⊛ v (ternary key variant)
 */
void hrr_accumulate_ternary(float *M, const int8_t *k_ternary,
                              const float *v, int d, float *tmp);

/*
 * hrr_build_memory: build M from N key-value pairs at once.
 *
 * M = Σᵢ kᵢ ⊛ vᵢ
 *
 * @param M       output memory [d floats], zeroed before accumulation
 * @param keys    float keys [N × d], or NULL if using ternary_keys
 * @param tkeys   ternary keys [N × d int8], used if keys == NULL
 * @param values  float values [N × d]
 * @param N       number of pairs (context length)
 * @param d       dimension
 */
void hrr_build_memory(float *M, const float *keys, const int8_t *tkeys,
                       const float *values, int N, int d);

/* ─── Retrieval quality ───────────────────────────────────────────────── */

/*
 * hrr_cosine_sim: cosine similarity between two vectors.
 * Used to measure retrieval quality: sim(retrieved, true_value).
 */
float hrr_cosine_sim(const float *a, const float *b, int d);

/*
 * hrr_cleanup_step: one step of iterative cleanup.
 *
 * Projects noisy retrieval onto the nearest vector in a codebook
 * (set of known clean values). Used when N > d/10 and retrieval is noisy.
 *
 * @param out       cleaned output [d floats]
 * @param noisy     noisy retrieved value [d floats]
 * @param codebook  N_cb clean prototype vectors [N_cb × d floats]
 * @param N_cb      codebook size
 * @param d         dimension
 * @return          index of nearest codebook entry
 */
int hrr_cleanup_step(float *out, const float *noisy,
                     const float **codebook, int N_cb, int d);

/*
 * hrr_cleanup_iter: iterative cleanup loop (Frady 2021).
 *
 * Repeats nearest-codebook projection until convergence (the chosen codebook
 * index stops changing) or max_iters is reached.  Optionally subtracts the
 * contribution of the chosen codebook entry from M (residual clean) and
 * re-unbinds, which gives better SNR than naive projection when N > d/10.
 *
 * Two modes:
 *   1. NAIVE PROJECTION:    out = argmin ||x - c|| iteratively (no M)
 *   2. RESIDUAL CLEAN:      out = argmin ||M⊛q⁻¹ - k⊛c|| iteratively
 *
 * Mode (2) is the Frady 2021 algorithm and is what you want for HRR
 * retrieval.  Pass M=NULL for mode (1).
 *
 * @param out        cleaned output [d floats] (== best codebook entry on return)
 * @param noisy      initial retrieval (or NULL if using M+query)
 * @param M          holographic memory [d floats], or NULL for naive mode
 * @param query_key  retrieval key [d floats], or NULL for naive mode
 * @param codebook   N_cb clean prototype vectors [N_cb × d floats]
 * @param N_cb       codebook size
 * @param d          dimension
 * @param max_iters  iteration cap (typ. 8-16)
 * @param tmp        scratch buffer [3*(d+2) + d floats] (only used in mode 2)
 * @return           index of chosen codebook entry, or -1 if no entry ever
 *                   projected closer than trivial (no convergence)
 */
int hrr_cleanup_iter(float *out, const float *noisy,
                     const float *M, const float *query_key,
                     const float **codebook, int N_cb, int d,
                     int max_iters, float *tmp);

/* ─── HRR-based attention (full replacement of scaled dot-product) ────── */

/*
 * hrr_attention_build: encode context K/V into holographic memory M.
 *
 * Called once per context (equivalent to KV cache build).
 * M = Σᵢ K[i] ⊛ V[i]   for i = 0..n_ctx-1
 *
 * @param M        holographic memory [head_dim floats], zeroed internally
 * @param K        keys (float) [n_ctx × head_dim], or NULL for ternary
 * @param K_tern   ternary keys [n_ctx × head_dim int8], used if K == NULL
 * @param V        values [n_ctx × head_dim floats]
 * @param n_ctx    context length
 * @param head_dim dimension per attention head (must be power of 2)
 */
void hrr_attention_build(float *M, const float *K, const int8_t *K_tern,
                          const float *V, int n_ctx, int head_dim);

/*
 * hrr_attention_retrieve: retrieve value for one query from holographic memory.
 *
 * out ≈ Σᵢ softmax(Q·Kᵢᵀ/√d)[i] · V[i]   (approximate)
 *     = M ⊛ Q⁻¹                            (HRR retrieval, O(d log d))
 *
 * @param out      retrieved value [head_dim floats]
 * @param M        holographic memory [head_dim floats]
 * @param q        query vector [head_dim floats]
 * @param head_dim head dimension
 * @param tmp      scratch [4*(head_dim+2) floats]
 */
void hrr_attention_retrieve(float *out, const float *M, const float *q,
                              int head_dim, float *tmp);

/*
 * hrr_attention_full: build + retrieve for a batch of queries.
 *
 * output[i] = hrr_attention_retrieve(M_built_from_K_V, Q[i])
 *
 * Complexity: O(n_ctx·d·log d) build + O(n_q·d·log d) retrieve
 *           vs O(n_ctx·n_q·d) for standard attention
 *
 * @param output   [n_queries × head_dim floats]
 * @param Q        queries [n_queries × head_dim floats]
 * @param K        keys [n_ctx × head_dim floats], or NULL for ternary
 * @param K_tern   ternary keys [n_ctx × head_dim int8]
 * @param V        values [n_ctx × head_dim floats]
 * @param n_queries number of queries
 * @param n_ctx    context length
 * @param head_dim head dimension (power of 2)
 */
void hrr_attention_full(float *output, const float *Q,
                         const float *K, const int8_t *K_tern,
                         const float *V,
                         int n_queries, int n_ctx, int head_dim);

#ifdef __cplusplus
}
#endif
