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

#else /* BITNET_L3_ACDC not defined */

struct ggml_tensor * bitnet_op_acdc(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    struct ggml_tensor  * d)
{
    (void)ctx; (void)d;
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

    const int d    = (int)q_t->ne[0];   /* head_dim */
    const int n_q  = (int)q_t->ne[1];   /* number of query tokens */
    const int n_kv = (int)k_t->ne[1];   /* number of key-value tokens */

    const float * q_f = (const float *)q_t->data;
    const float * k_f = (const float *)k_t->data;
    const float * v_f = (const float *)v_t->data;
    float       * out = (float *)dst->data;

    /* Quantize Q and K to int8 for the tropical scan (zero multiplications). */
    int8_t * q_i8 = (int8_t *)malloc((size_t)n_q  * d);
    int8_t * k_i8 = (int8_t *)malloc((size_t)n_kv * d);

    float q_scale = quantize_f32_to_i8(q_f, q_i8, n_q  * d);
    float k_scale = quantize_f32_to_i8(k_f, k_i8, n_kv * d);

    /*
     * tropical_attention processes ONE query vector against n_kv keys.
     * For multiple queries we loop; the query scale stays constant.
     */
    for (int qi = 0; qi < n_q; qi++) {
        tropical_attention(
            out + qi * d,          /* output for this query */
            q_i8 + qi * d,         /* one query row */
            k_i8,                  /* all n_kv key rows */
            v_f,                   /* float values */
            n_kv,                  /* number of keys */
            d,                     /* head_dim */
            p->topk,               /* top-K budget */
            q_scale,               /* query int8 scale */
            k_scale);              /* key int8 scale */
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

    const int d    = (int)q_t->ne[0];  /* head_dim (must be power of 2 ≥ 64) */
    const int n_q  = (int)q_t->ne[1];  /* query tokens */
    const int n_kv = (int)k_t->ne[1];  /* key-value tokens */

    const float * q_f = (const float *)q_t->data;
    const float * k_f = (const float *)k_t->data;
    const float * v_f = (const float *)v_t->data;
    float       * out = (float *)dst->data;

    /* Derive ternary key approximation (avoids needing a 4th tensor input) */
    int8_t * k_tern = (int8_t *)malloc((size_t)n_kv * d);
    derive_ternary_keys(k_f, k_tern, n_kv * d);

    /*
     * hrr_attention_full builds holographic memory M = Σᵢ kᵢ ⊛ vᵢ then
     * retrieves ṽq = M ⊛ q⁻¹ for each query.  Complexity O(n_kv·d + n_q·d)
     * with all convolutions done via FFT in O(d log d) each.
     *
     * Reliability requires d ≥ 10·n_kv (see docs/theory/05-holographic-memory.md).
     */
    hrr_attention_full(out, q_f, k_f, k_tern, v_f, n_q, n_kv, d);

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
