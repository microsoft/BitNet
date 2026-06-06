/*
 * ggml-bitnet-dispatch.cpp — ggml custom ops for L3/L4/L5 math kernels
 *
 * Implements graph-node wrappers (ggml_map_custom*) that allow L3/L4/L5
 * research kernels to participate in ggml compute graphs without modifying
 * the ggml or llama.cpp core.
 *
 * Dispatch chain:
 *   graph build time:  bitnet_op_*(ctx, tensors...) → ggml tensor node
 *   graph compute time: ggml calls callback(dst, srcs..., ith, nth, ud)
 *   callback: calls kernel from ggml-bitnet-{fwht,tropical,hrr}.cpp
 */

#include "ggml-bitnet-dispatch.h"
#include "ggml.h"

#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <algorithm>

#if defined(BITNET_L3_ACDC)
#include "ggml-bitnet-fwht.h"
#endif

#if defined(BITNET_L4_TROPICAL)
#include "ggml-bitnet-tropical.h"
#endif

#if defined(BITNET_L5_HRR)
#include "ggml-bitnet-hrr.h"
#endif

/* ─── L3: ACDC structured layer ─────────────────────────────────────────── */

#if defined(BITNET_L3_ACDC)

static void acdc_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * a,
    const struct ggml_tensor * b,
    int ith, int nth, void * userdata)
{
    (void)nth; (void)userdata;
    if (ith != 0) return;

    /* a = input x [n, batch], b = diagonal d [n], dst = output [n, batch] */
    const int n     = (int)a->ne[0];
    const int batch = (int)(ggml_nelements(a) / n);

    const float * d = (const float *)b->data;

    for (int i = 0; i < batch; i++) {
        const float * x   = (const float *)a->data   + i * n;
        float       * out = (float *)dst->data + i * n;
        acdc_forward_f32(out, x, d, n);
    }
}

struct ggml_tensor * bitnet_op_acdc(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    struct ggml_tensor  * d)
{
    return ggml_map_custom2(ctx, x, d, acdc_callback, /*n_tasks=*/1, NULL);
}

/* ── ACDC GEMV (rectangular, K blocks + linear projection) ──────────────── */

struct acdc_gemv_ud {
    int     m;             /* output dim (original model dim)            */
    int     n;             /* ACDC block dim (power of 2)                */
    int     K;             /* number of ACDC blocks (K*n ≥ m)            */
    int     n_orig;        /* original input dim (first n_orig of x)     */
    float * D;             /* K*n learned diagonals (zero-initialized)   */
    float * proj;          /* m * K*n projection (partial identity)      */
    int8_t * x_i8;         /* scratch buffer for int8 quantized x [n]    */
    bool    initialized;   /* lazy init flag                             */
};

static void acdc_gemv_init_buffers(struct acdc_gemv_ud * p) {
    const int Kn = p->K * p->n;
    p->D     = (float *)calloc((size_t)Kn, sizeof(float));
    p->proj  = (float *)calloc((size_t)p->m * Kn, sizeof(float));
    p->x_i8  = (int8_t *)calloc((size_t)p->n, sizeof(int8_t));
    /*
     * Partial identity: proj[i * Kn + i] = 1.0 for i in [0, m).
     * Since Kn ≥ m (by K definition), this preserves the first m components
     * of the ACDC stacked output as-is, effectively truncating to m.
     * D is all zeros (model not trained with ACDC; P6 unvalidated).
     */
    for (int i = 0; i < p->m; i++) {
        p->proj[i * Kn + i] = 1.0f;
    }
    p->initialized = true;
}

static void acdc_gemv_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * a,
    int ith, int nth, void * userdata)
{
    (void)nth;
    if (ith != 0) return;

    struct acdc_gemv_ud * p = (struct acdc_gemv_ud *)userdata;
    if (!p->initialized) acdc_gemv_init_buffers(p);

    const int batch = (int)(ggml_nelements(a) / p->n_orig);
    const float * x = (const float *)a->data;
    float       * y = (float *)dst->data;

    for (int b = 0; b < batch; b++) {
        const float * xb = x + b * p->n_orig;

        /* Per-sample int8 quantization (per-row scale for tight range) */
        float mx = 1e-6f;
        for (int i = 0; i < p->n_orig; i++) mx = fmaxf(mx, fabsf(xb[i]));
        float s = 127.0f / mx;
        for (int i = 0; i < p->n_orig; i++) {
            float v = xb[i] * s;
            if (v >  127.0f) v =  127.0f;
            if (v < -128.0f) v = -128.0f;
            p->x_i8[i] = (int8_t)(int)v;
        }
        /* Positions [n_orig, n) remain zero (calloc-initialized) — padding */

        acdc_gemv(y + b * p->m, p->x_i8, p->D, p->proj, p->m, p->n, p->K);
    }
}

struct ggml_tensor * bitnet_op_acdc_gemv(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    int                   m,
    int                   n,
    int                   K,
    int                   n_orig)
{
    struct acdc_gemv_ud * ud = (struct acdc_gemv_ud *)malloc(sizeof(*ud));
    ud->m = m; ud->n = n; ud->K = K; ud->n_orig = n_orig;
    ud->D = NULL; ud->proj = NULL; ud->x_i8 = NULL;
    ud->initialized = false;
    return ggml_map_custom1(ctx, x, acdc_gemv_callback, /*n_tasks=*/1, ud);
}

#else /* BITNET_L3_ACDC not defined */

struct ggml_tensor * bitnet_op_acdc(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    struct ggml_tensor  * d)
{
    (void)ctx; (void)d;
    return x;
}

struct ggml_tensor * bitnet_op_acdc_gemv(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    int                   m,
    int                   n,
    int                   K,
    int                   n_orig)
{
    (void)ctx; (void)m; (void)n; (void)K; (void)n_orig;
    return x;
}

#endif /* BITNET_L3_ACDC */

/* ─── L4: Tropical attention ─────────────────────────────────────────────── */

#if defined(BITNET_L4_TROPICAL)

struct tropical_ud {
    int   topk;
    float scale;
};

/*
 * Quantize a float vector to int8 in-place.
 * Returns the scale s = 127 / max|x| used, so the caller can pass it to
 * tropical_attention as q_scale / k_scale.
 */
static float quantize_f32_to_i8(const float * src, int8_t * dst, int n) {
    float mx = 1e-6f;
    for (int i = 0; i < n; i++) mx = fmaxf(mx, fabsf(src[i]));
    float s = 127.0f / mx;
    for (int i = 0; i < n; i++) {
        float v = src[i] * s;
        if (v >  127.0f) v =  127.0f;
        if (v < -128.0f) v = -128.0f;
        dst[i] = (int8_t)(int)v;
    }
    return s;
}

static void tropical_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * q_t,
    const struct ggml_tensor * k_t,
    const struct ggml_tensor * v_t,
    int ith, int nth, void * userdata)
{
    (void)nth;
    if (ith != 0) return;

    const struct tropical_ud * p = (const struct tropical_ud *)userdata;

    /*
     * Tensor layout (after ggml_permute in llm_build_kqv, cast to F32):
     *   q:   [head_dim, n_tokens, n_head]     — F32 contiguous
     *   k:   [head_dim, n_kv,     n_head_kv]  — F32 contiguous
     *   v:   [head_dim, n_kv,     n_head_kv]  — F32 contiguous
     *   dst: same shape as q
     *
     * Within each head h, data layout is token-major:
     *   data[h * n_tok * d + tok * d + j] = value at (head=h, token=tok, dim=j)
     * This is exactly the [n_kv × d] row-major layout tropical_attention expects.
     *
     * GQA: n_head_q may be > n_head_kv; head h_q maps to kv head h_q / gqa_ratio.
     */
    const int d         = (int)q_t->ne[0];
    const int n_tokens  = (int)q_t->ne[1];
    const int n_head    = (int)(q_t->ne[2] > 0 ? q_t->ne[2] : 1);
    const int n_kv      = (int)k_t->ne[1];
    const int n_head_kv = (int)(k_t->ne[2] > 0 ? k_t->ne[2] : 1);
    const int gqa       = n_head / n_head_kv;

    const float * q_f = (const float *)q_t->data;
    const float * k_f = (const float *)k_t->data;
    const float * v_f = (const float *)v_t->data;
    float       * out = (float *)dst->data;

    /* Single int8 buffer per KV block; re-quantize K once per head. */
    int8_t * q_i8 = (int8_t *)malloc((size_t)d);
    int8_t * k_i8 = (int8_t *)malloc((size_t)n_kv * d);

    for (int h = 0; h < n_head; h++) {
        const int    kv_h    = h / gqa;
        const float *q_head  = q_f + (size_t)h    * n_tokens * d;
        const float *k_head  = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head  = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd  = out + (size_t)h    * n_tokens * d;

        /* Quantize the entire key block once per query head. */
        float k_scale = quantize_f32_to_i8(k_head, k_i8, n_kv * d);

        for (int qi = 0; qi < n_tokens; qi++) {
            /* Per-query quantization keeps scale tight for each token. */
            float q_scale = quantize_f32_to_i8(q_head + qi * d, q_i8, d);
            tropical_attention(
                out_hd  + qi * d,   /* output: dim vector for this query */
                q_i8,               /* one quantized query vector [d] */
                k_i8,               /* all n_kv key rows [n_kv × d] */
                v_head,             /* float values [n_kv × d] */
                n_kv,
                d,
                p->topk,
                q_scale,
                k_scale);
        }
    }

    free(q_i8);
    free(k_i8);
}

struct ggml_tensor * bitnet_op_tropical_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    int                   topk,
    float                 scale)
{
    (void)scale; /* stored in ud for future use */
    struct tropical_ud * ud = (struct tropical_ud *)malloc(sizeof(*ud));
    ud->topk  = topk;
    ud->scale = scale;
    return ggml_map_custom3(ctx, q, k, v, tropical_callback, /*n_tasks=*/1, ud);
}

#else /* BITNET_L4_TROPICAL not defined */

struct ggml_tensor * bitnet_op_tropical_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    int                   topk,
    float                 scale)
{
    (void)ctx; (void)k; (void)v; (void)topk; (void)scale;
    return q;
}

#endif /* BITNET_L4_TROPICAL */

/* ─── L5: HRR attention ──────────────────────────────────────────────────── */

#if defined(BITNET_L5_HRR)

/*
 * Derive ternary key approximation from float keys.
 * Rounds each element to the nearest value in {-1, 0, +1}.
 * Threshold: values with |x| < 0.5 * mean|K| → 0, else sign(x).
 */
static void derive_ternary_keys(const float * K_f, int8_t * K_tern, int n) {
    /* Threshold at half the mean absolute value */
    float mean_abs = 0.0f;
    for (int i = 0; i < n; i++) mean_abs += fabsf(K_f[i]);
    mean_abs /= (float)n;
    float thresh = 0.5f * mean_abs;

    for (int i = 0; i < n; i++) {
        float v = K_f[i];
        if (v >  thresh) K_tern[i] = 1;
        else if (v < -thresh) K_tern[i] = -1;
        else K_tern[i] = 0;
    }
}

static void hrr_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * q_t,
    const struct ggml_tensor * k_t,
    const struct ggml_tensor * v_t,
    int ith, int nth, void * userdata)
{
    (void)nth; (void)userdata;
    if (ith != 0) return;

    /*
     * Same 3D multi-head layout as tropical_callback (see comments there).
     * Tensor shapes after ggml_permute + cast to F32:
     *   q:   [head_dim, n_tokens, n_head]     contiguous
     *   k:   [head_dim, n_kv,     n_head_kv]  contiguous
     *   v:   [head_dim, n_kv,     n_head_kv]  contiguous
     *
     * hrr_attention_full expects row-major [n_tok × d] layout per head,
     * which matches since data[h*n*d + t*d + j] = (head=h, token=t, dim=j).
     *
     * HRR retrieval quality requires d ≥ 10·n_kv.  For d=128 n_kv=2048,
     * output is noisy — this is expected without HRR-trained weights.
     */
    const int d         = (int)q_t->ne[0];
    const int n_tokens  = (int)q_t->ne[1];
    const int n_head    = (int)(q_t->ne[2] > 0 ? q_t->ne[2] : 1);
    const int n_kv      = (int)k_t->ne[1];
    const int n_head_kv = (int)(k_t->ne[2] > 0 ? k_t->ne[2] : 1);
    const int gqa       = n_head / n_head_kv;

    const float * q_f = (const float *)q_t->data;
    const float * k_f = (const float *)k_t->data;
    const float * v_f = (const float *)v_t->data;
    float       * out = (float *)dst->data;

    /* Ternary key buffer — derived once per KV head */
    int8_t * k_tern = (int8_t *)malloc((size_t)n_kv * d);
    if (!k_tern) return;

    for (int h = 0; h < n_head; h++) {
        const int    kv_h    = h / gqa;
        const float *q_head  = q_f + (size_t)h    * n_tokens * d;
        const float *k_head  = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head  = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd  = out + (size_t)h    * n_tokens * d;

        /* Ternary approximation: threshold at 0.5 * mean|K| per head */
        derive_ternary_keys(k_head, k_tern, n_kv * d);

        /* hrr_attention_full: build holographic memory + retrieve all queries.
         * O(n_kv·d·log d) build  +  O(n_tokens·d·log d) retrieve. */
        hrr_attention_full(out_hd, q_head, k_head, k_tern, v_head,
                           n_tokens, n_kv, d);
    }

    free(k_tern);
}

struct ggml_tensor * bitnet_op_hrr_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v)
{
    return ggml_map_custom3(ctx, q, k, v, hrr_callback, /*n_tasks=*/1, NULL);
}

#else /* BITNET_L5_HRR not defined */

struct ggml_tensor * bitnet_op_hrr_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v)
{
    (void)ctx; (void)k; (void)v;
    return q;
}

#endif /* BITNET_L5_HRR */
