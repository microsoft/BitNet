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
     *
     * Thread parallelism: thread ith handles heads ith, ith+nth, ith+2*nth, ...
     * All head regions in q/dst are disjoint; k/v are read-only — no races.
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

    /* Per-thread scratch buffers (each thread allocates independently). */
    int8_t * q_i8 = (int8_t *)malloc((size_t)d);
    int8_t * k_i8 = (int8_t *)malloc((size_t)n_kv * d);
    if (!q_i8 || !k_i8) { free(q_i8); free(k_i8); return; }

    for (int h = ith; h < n_head; h += nth) {
        const int    kv_h    = h / gqa;
        const float *q_head  = q_f + (size_t)h    * n_tokens * d;
        const float *k_head  = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head  = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd  = out + (size_t)h    * n_tokens * d;

        float k_scale = quantize_f32_to_i8(k_head, k_i8, n_kv * d);

        for (int qi = 0; qi < n_tokens; qi++) {
            float q_scale = quantize_f32_to_i8(q_head + qi * d, q_i8, d);
            tropical_attention(
                out_hd  + qi * d,
                q_i8,
                k_i8,
                v_head,
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
    return ggml_map_custom3(ctx, q, k, v, tropical_callback, GGML_N_TASKS_MAX, ud);
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
    (void)userdata;

    /*
     * Same 3D multi-head layout as tropical_callback.
     * Thread ith handles heads ith, ith+nth, ith+2*nth, ... (no races).
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

    int8_t * k_tern = (int8_t *)malloc((size_t)n_kv * d);
    if (!k_tern) return;

    for (int h = ith; h < n_head; h += nth) {
        const int    kv_h    = h / gqa;
        const float *q_head  = q_f + (size_t)h    * n_tokens * d;
        const float *k_head  = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head  = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd  = out + (size_t)h    * n_tokens * d;

        derive_ternary_keys(k_head, k_tern, n_kv * d);
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
    return ggml_map_custom3(ctx, q, k, v, hrr_callback, GGML_N_TASKS_MAX, NULL);
}

/* ─── L5: HRR attention + Frady 2021 cleanup_iter ─────────────────────── */

struct hrr_cleanup_ud {
    int max_iters;   /* cleanup_iter iteration cap (typ. 8-16) */
};

static void hrr_cleanup_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * q_t,
    const struct ggml_tensor * k_t,
    const struct ggml_tensor * v_t,
    int ith, int nth, void * userdata)
{
    struct hrr_cleanup_ud * p = (struct hrr_cleanup_ud *)userdata;

    /* Same 3D layout as hrr_callback. Thread ith handles strided heads. */
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

    /* Per-thread scratch buffers. */
    int8_t       * k_tern   = (int8_t *)malloc((size_t)n_kv * d);
    float        * M        = (float  *)malloc((size_t)d * sizeof(float));
    float        * M_work   = (float  *)malloc((size_t)d * sizeof(float));
    float        * tmp      = (float  *)malloc((size_t)4 * (d + 2) * sizeof(float));
    const float ** codebook = (const float **)malloc((size_t)n_kv * sizeof(const float *));

    if (!k_tern || !M || !M_work || !tmp || !codebook) {
        free(k_tern); free(M); free(M_work); free(tmp); free(codebook);
        return;
    }

    for (int h = ith; h < n_head; h += nth) {
        const int    kv_h    = h / gqa;
        const float *q_head  = q_f + (size_t)h    * n_tokens * d;
        const float *k_head  = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head  = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd  = out + (size_t)h    * n_tokens * d;

        derive_ternary_keys(k_head, k_tern, n_kv * d);
        hrr_build_memory(M, nullptr, k_tern, v_head, n_kv, d);

        for (int i = 0; i < n_kv; i++) codebook[i] = v_head + (size_t)i * d;

        for (int t = 0; t < n_tokens; t++) {
            const float * q_tok = q_head + (size_t)t * d;
            float       * out_t = out_hd + (size_t)t * d;

            memcpy(M_work, M, (size_t)d * sizeof(float));
            hrr_cleanup_iter(out_t, /*noisy=*/nullptr,
                             M_work, q_tok,
                             codebook, n_kv, d,
                             p->max_iters, tmp);
        }
    }

    free(k_tern); free(M); free(M_work); free(tmp); free(codebook);
}

struct ggml_tensor * bitnet_op_hrr_attn_with_cleanup(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    int                   max_iters)
{
    struct hrr_cleanup_ud * ud = (struct hrr_cleanup_ud *)malloc(sizeof(*ud));
    if (!ud) return q;
    ud->max_iters = max_iters;
    return ggml_map_custom3(ctx, q, k, v, hrr_cleanup_callback, GGML_N_TASKS_MAX, ud);
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

struct ggml_tensor * bitnet_op_hrr_attn_with_cleanup(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    int                   max_iters)
{
    (void)ctx; (void)k; (void)v; (void)max_iters;
    return q;
}

#endif /* BITNET_L5_HRR */
