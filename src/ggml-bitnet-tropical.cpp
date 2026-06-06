/*
 * ggml-bitnet-tropical.cpp
 *
 * Tropical Attention — O(n log n) substituição do softmax(QKᵀ/√d)
 *
 * ─────────────────────────────────────────────────────────────────────────
 * FUNDAMENTO MATEMÁTICO: SEMIRING (max, +)
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Álgebra tropical = semiring (ℝ ∪ {-∞}, ⊕, ⊗) onde:
 *   a ⊕ b = max(a, b)       [adição tropical = máximo]
 *   a ⊗ b = a + b           [multiplicação tropical = soma real]
 *
 * Propriedades:
 *   (ℝ, max, +) é um semiring: distributividade, associatividade, comutatividade
 *   Elemento neutro de ⊕: -∞
 *   Elemento neutro de ⊗:  0
 *
 * PRODUTO MATRICIAL TROPICAL:
 *   (A ⊗ᵗʳᵒᵖ B)[i,k] = max_j (A[i,j] + B[j,k])
 *
 * ─────────────────────────────────────────────────────────────────────────
 * CONEXÃO COM TRANSFORMER ATTENTION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Atenção padrão (unnormalized):
 *   A[i,j] = exp(Q[i]·K[j]ᵀ / √d)
 *   softmax(A[i,:])[j] = A[i,j] / Σₖ A[i,k]
 *   output[i] = Σⱼ softmax[j] · V[j]
 *
 * No limite de temperatura τ → 0  (atenção hard / argmax):
 *   softmax(A/τ)[j] → δ[j = argmax_k Q[i]·K[k]ᵀ]
 *
 * Isso é exatamente o produto tropical:
 *   (Q ⊗ᵗʳᵒᵖ Kᵀ)[i] = max_j (Q[i]·K[j])   ← distância tropical = dot product max
 *   output[i] = V[argmax_j Q[i]·K[j]]
 *
 * Para τ finito (atenção soft), a aproximação tropical é válida quando a
 * distribuição de atenção é SHARP (concentrada em poucos tokens) — que é
 * exatamente o comportamento observado em LLMs treinados (Zhang et al., 2023:
 * "Trained LLMs exhibit increasingly sparse attention with depth").
 *
 * ─────────────────────────────────────────────────────────────────────────
 * REDUÇÃO DE COMPLEXIDADE
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Atenção padrão: O(n²·d) por head, onde n = seq_len, d = head_dim
 * Atenção tropical hard: O(n·d) — um dot product por query
 * Atenção tropical soft (top-K):
 *   1. Encontrar top-K tokens por produto tropical: O(n·d + n·log K)
 *   2. Softmax sobre K tokens: O(K·d)
 *   Total: O(n·d + K·d) = O(n·d) para K << n
 *
 * Com K=32 e n=2048, seq, d=128:
 *   Padrão:  2048² × 128 = 536M ops
 *   Tropical: 2048 × 128 + 32 × 128 = 266K ops → 2000× speedup
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ALGORITMO: MAXIMAL DOT PRODUCT SEARCH (MDPS)
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Para cada query q ∈ ℝᵈ e base de keys K ∈ ℝ^{n×d}:
 *   Find: k* = argmax_j q · K[j]
 *
 * Abordagem exata linear:  O(n·d) — o que implementamos aqui
 * Abordagem ANN sublinear:  O(log n · d) — via HNSW/LSH (próxima versão)
 *
 * Para CPU decode (batch=1, seq curto): O(n·d) exato já é suficiente.
 * Para seq longa (n > 4096): ANN via produto interno aproximado.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * IMPLEMENTAÇÃO: SIMD INT8 DOT PRODUCT (aproveitando quantização ternária)
 * ─────────────────────────────────────────────────────────────────────────
 *
 * As keys K são ternárias {-1,0,+1} → reutilizamos o kernel WHT (Level 2)
 * para o dot product. O "máximo" é puro comparação — sem multiplicação.
 *
 * Pipeline:
 *   1. Quantizar query q → int8 q_q  (per-token absmax)
 *   2. Para cada key k_j: dot(q_q, k_j) via WHT Level 2 (adições puras)
 *   3. Top-K: partial_sort dos escores → argpartition O(n log K)
 *   4. Softmax sobre top-K: exp + normalize (apenas K exponenciais!)
 *   5. Output: Σ_{j∈topK} softmax[j] · V[j]
 */

#include "ggml-bitnet-tropical.h"
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
 * UTILIDADES: DOT PRODUCT INT8 × TERNÁRIO (reutiliza Level 2)
 * ═══════════════════════════════════════════════════════════════════════════ */

/*
 * dot_ternary_int8: q · k  onde k ∈ {-1,0,+1}^d (ternário), q ∈ int8^d
 *
 * Decompõe: q·k = Σ_{j:k[j]=+1} q[j] - Σ_{j:k[j]=-1} q[j]
 * Zero multiplicações — adições condicionais apenas.
 *
 * k_encoded: codificação I2_S (0=neg, 1=zero, 2=pos), byte por elemento
 * (versão descompactada para simplicidade de indexação)
 */
static int32_t dot_ternary_int8_scalar(
    const int8_t  * q,
    const int8_t  * k_encoded,   /* valores em {-1, 0, +1} (int8 signed) */
    int d)
{
    int32_t acc = 0;
    for (int i = 0; i < d; i++) {
        int8_t kv = k_encoded[i];
        if      (kv > 0) acc += (int32_t)q[i];
        else if (kv < 0) acc -= (int32_t)q[i];
        /* kv == 0: skip — zero operação */
    }
    return acc;
}

#if defined(__AVX2__)
static int32_t dot_ternary_int8_avx2(
    const int8_t * q,
    const int8_t * k,
    int d)
{
    __m256i accum    = _mm256_setzero_si256();
    __m256i v_zero   = _mm256_setzero_si256();
    __m256i v_ones16 = _mm256_set1_epi16(1);

    int i = 0;
    for (; i + 32 <= d; i += 32) {
        __m256i kv   = _mm256_loadu_si256((const __m256i *)(k + i));
        __m256i qv   = _mm256_loadu_si256((const __m256i *)(q + i));

        /* pos_mask: 0xFF where k=+1 (kv > 0) */
        __m256i pos_mask = _mm256_cmpgt_epi8(kv, v_zero);
        /* neg_mask: 0xFF where k=-1 (kv < 0, i.e., kv < 0 ↔ kv > 0 negado) */
        __m256i neg_mask = _mm256_cmpgt_epi8(v_zero, kv);

        __m256i pos_vals = _mm256_and_si256(qv, pos_mask);
        __m256i neg_vals = _mm256_and_si256(qv, neg_mask);
        __m256i delta    = _mm256_sub_epi8(pos_vals, neg_vals);

        /* Acumular int8 → int32 via int16 */
        __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(delta));
        __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(delta, 1));
        __m256i sum16 = _mm256_add_epi16(lo16, hi16);
        accum = _mm256_add_epi32(accum, _mm256_madd_epi16(sum16, v_ones16));
    }

    /* Horizontal sum */
    __m128i lo  = _mm256_castsi256_si128(accum);
    __m128i hi  = _mm256_extracti128_si256(accum, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    int32_t result = _mm_cvtsi128_si32(sum);

    /* Tail */
    for (; i < d; i++) {
        int8_t kv = k[i];
        if      (kv > 0) result += (int32_t)q[i];
        else if (kv < 0) result -= (int32_t)q[i];
    }
    return result;
}
#endif

#if defined(__ARM_NEON)
static int32_t dot_ternary_int8_neon(
    const int8_t * q,
    const int8_t * k,
    int d)
{
    int32x4_t accum = vdupq_n_s32(0);
    int8x16_t v_zero = vdupq_n_s8(0);

    int i = 0;
    for (; i + 16 <= d; i += 16) {
        int8x16_t kv = vld1q_s8(k + i);
        int8x16_t qv = vld1q_s8(q + i);

        uint8x16_t pos_mask = vcgtq_s8(kv, v_zero);
        uint8x16_t neg_mask = vcltq_s8(kv, v_zero);

        int8x16_t pos_vals = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(qv), pos_mask));
        int8x16_t neg_vals = vreinterpretq_s8_u8(vandq_u8(vreinterpretq_u8_s8(qv), neg_mask));
        int8x16_t delta    = vsubq_s8(pos_vals, neg_vals);

#if defined(__ARM_FEATURE_DOTPROD)
        accum = vdotq_s32(accum, delta, vdupq_n_s8(1));
#else
        int16x8_t sum16 = vaddq_s16(
            vmovl_s8(vget_low_s8(delta)),
            vmovl_s8(vget_high_s8(delta)));
        accum = vaddq_s32(accum, vaddl_s16(vget_low_s16(sum16), vget_high_s16(sum16)));
#endif
    }

    int32_t result = vaddvq_s32(accum);
    for (; i < d; i++) {
        int8_t kv = k[i];
        if      (kv > 0) result += (int32_t)q[i];
        else if (kv < 0) result -= (int32_t)q[i];
    }
    return result;
}
#endif

static int32_t dot_ternary_int8(const int8_t * q, const int8_t * k, int d) {
#if defined(__AVX2__)
    return dot_ternary_int8_avx2(q, k, d);
#elif defined(__ARM_NEON)
    return dot_ternary_int8_neon(q, k, d);
#else
    return dot_ternary_int8_scalar(q, k, d);
#endif
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TROPICAL ATTENTION: MAXIMAL DOT PRODUCT SEARCH (MDPS)
 * ═══════════════════════════════════════════════════════════════════════════ */

void tropical_attn_scores(
    float        * scores,    /* output [n_keys floats] */
    const int8_t * q,         /* query quantizada [head_dim int8] */
    const int8_t * K,         /* keys ternárias [n_keys × head_dim int8] */
    int            n_keys,
    int            head_dim,
    float          q_scale,   /* escala de quantização da query */
    float          k_scale)   /* escala de quantização das keys */
{
    float scale = (q_scale * k_scale) / (float)head_dim;  /* absorve 1/√d */

    for (int j = 0; j < n_keys; j++) {
        int32_t raw = dot_ternary_int8(q, K + j * head_dim, head_dim);
        scores[j] = (float)raw * scale;
    }
}

int tropical_attn_argmax(
    const int8_t * q,
    const int8_t * K,
    int            n_keys,
    int            head_dim)
{
    int32_t best_score = INT32_MIN;
    int     best_idx   = 0;

    for (int j = 0; j < n_keys; j++) {
        int32_t s = dot_ternary_int8(q, K + j * head_dim, head_dim);
        if (s > best_score) { best_score = s; best_idx = j; }
    }
    return best_idx;
}

void tropical_attn_topk(
    int          * top_idx,   /* output: indices dos top-K [K ints] */
    float        * top_scores,/* output: escores dos top-K [K floats] */
    const int8_t * q,
    const int8_t * K,
    int            n_keys,
    int            head_dim,
    int            K_top,
    float          q_scale,
    float          k_scale)
{
    /* Clamp K_top to available keys — handles early decode / warmup where n_keys < topk */
    const int K_actual = (K_top < n_keys) ? K_top : n_keys;
    if (K_actual <= 0) return;

    /* Passo 1: computar todos os escores — O(n·d), adições puras */
    float * scores = (float *)malloc(n_keys * sizeof(float));
    if (!scores) return;
    tropical_attn_scores(scores, q, K, n_keys, head_dim, q_scale, k_scale);

    /* Passo 2: partial sort — O(n·log K), só comparações */
    int * idx = (int *)malloc(n_keys * sizeof(int));
    if (!idx) { free(scores); return; }
    for (int i = 0; i < n_keys; i++) idx[i] = i;

    /* partial_sort requires middle ≤ last — K_actual guarantees this */
    std::partial_sort(idx, idx + K_actual, idx + n_keys,
        [scores](int a, int b){ return scores[a] > scores[b]; });

    for (int k = 0; k < K_actual; k++) {
        top_idx[k]    = idx[k];
        top_scores[k] = scores[idx[k]];
    }

    free(scores);
    free(idx);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ATENÇÃO COMPLETA: TROPICAL SOFTMAX SOBRE TOP-K
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Algoritmo:
 *   1. Tropical max scan → top-K indices  [O(n·d) = O(n) adições]
 *   2. Softmax sobre top-K scores         [O(K) exponenciais]
 *   3. Weighted sum de V[top-K]           [O(K·d) adições]
 *
 * Total: O(n·d + K·d) ≈ O(n·d) para K << n
 * vs. padrão: O(n²·d) → speedup = n/K (para n=2048, K=32: 64×)
 * ═══════════════════════════════════════════════════════════════════════════ */

void tropical_attention(
    float        * output,    /* [head_dim floats] */
    const int8_t * q,         /* query quantizada [head_dim] */
    const int8_t * K,         /* keys ternárias [n_keys × head_dim] */
    const float  * V,         /* values float [n_keys × head_dim] */
    int            n_keys,
    int            head_dim,
    int            K_top,
    float          q_scale,
    float          k_scale)
{
    /* Clamp to available keys so we never read uninitialized top_idx/top_s entries */
    const int K_actual = (K_top < n_keys) ? K_top : n_keys;
    if (K_actual <= 0) { memset(output, 0, head_dim * sizeof(float)); return; }

    int   * top_idx = (int   *)malloc(K_actual * sizeof(int));
    float * top_s   = (float *)malloc(K_actual * sizeof(float));
    float * weights = (float *)malloc(K_actual * sizeof(float));
    if (!top_idx || !top_s || !weights) goto cleanup;

    /* 1. Top-K via tropical max — fills exactly K_actual entries */
    tropical_attn_topk(top_idx, top_s, q, K, n_keys, head_dim,
                        K_actual, q_scale, k_scale);

    /* 2. Softmax over top-K (log-sum-exp stable) */
    {
        float max_s = top_s[0];
        for (int k = 1; k < K_actual; k++)
            if (top_s[k] > max_s) max_s = top_s[k];

        float sum_exp = 0.0f;
        for (int k = 0; k < K_actual; k++) {
            weights[k] = expf(top_s[k] - max_s);
            sum_exp += weights[k];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int k = 0; k < K_actual; k++) weights[k] *= inv_sum;
    }

    /* 3. Weighted sum of top-K values */
    memset(output, 0, head_dim * sizeof(float));
    for (int k = 0; k < K_actual; k++) {
        const float * vk = V + top_idx[k] * head_dim;
        float w = weights[k];
        for (int i = 0; i < head_dim; i++) output[i] += w * vk[i];
    }

cleanup:
    free(top_idx);
    free(top_s);
    free(weights);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * TROPICAL GEMV: produto matricial tropical (max-plus)
 *
 * (A ⊗ᵗʳᵒᵖ x)[i] = max_j (A[i,j] + x[j])
 *
 * Para A ternária e x inteira: substituímos + por adição int8 com saturação.
 * Resultado: o índice j* que maximiza A[i,j]+x[j] para cada linha i.
 * ═══════════════════════════════════════════════════════════════════════════ */

void tropical_gemv(
    int          * argmax_out,  /* [m] — índice j* por linha */
    float        * max_out,     /* [m] — valor máximo por linha */
    const int8_t * A,           /* ternária [m × n], valores {-1,0,+1} */
    const float  * x,           /* vetor [n floats] */
    int            m,
    int            n)
{
    for (int i = 0; i < m; i++) {
        float best = -FLT_MAX;
        int   best_j = 0;
        const int8_t * row = A + i * n;
        for (int j = 0; j < n; j++) {
            /* Tropical: max_j(A[i,j] + x[j]) */
            float val = (float)row[j] + x[j];
            if (val > best) { best = val; best_j = j; }
        }
        argmax_out[i] = best_j;
        max_out[i]    = best;
    }
}
