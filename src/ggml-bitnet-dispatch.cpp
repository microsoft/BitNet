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
#include <stdio.h>
#include <stdatomic.h>

#if defined(BITNET_L3_ACDC)
#include "ggml-bitnet-fwht.h"

/* ── Global ACDC diagonal store (loaded from BITNET_ACDC_FFN_RECT_DIAG) ──── */

/* Binary format:
 *   magic[8]:    b"ACDBD\x01\x00\x00"
 *   n_layers:    uint32
 *   n_proj:      uint32   (= 2: proj0=up, proj1=down)
 *   P:           uint32
 *   reserved:    uint32   (= 0)
 *   data:        float32[n_layers × n_proj × P]
 *                index:  layer * n_proj * P + proj * P + k
 *                proj 0 → up  (m=n_ff, n=n_embd)
 *                proj 1 → down (m=n_embd, n=n_ff)
 *
 * Populated by: utils/acdc_diag_to_bin.py (reads .acdc_diag.npz sidecar).
 * Env var: BITNET_ACDC_FFN_RECT_DIAG=path/to/file.bin
 */
static struct {
    float   * data;       /* flat float array [n_layers × n_proj × P] */
    uint32_t  n_layers;
    uint32_t  n_proj;
    uint32_t  P;
    bool      loaded;
} g_acdc_diag = { nullptr, 0, 2, 0, false };

/* Thread-safe call counter: tracks which (layer, proj) pair the next
 * acdc_ffn_rect_init_buffers call corresponds to.  Initialized lazily and
 * reset before each inference run via bitnet_acdc_diag_reset_counter(). */
static _Atomic int g_acdc_rect_call_count = 0;

static void acdc_diag_load_once(void) {
    if (g_acdc_diag.loaded) return;
    g_acdc_diag.loaded = true;  /* mark even on failure — no retry */

    const char * path = getenv("BITNET_ACDC_FFN_RECT_DIAG");
    if (!path || !path[0]) return;

    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ACDC] cannot open sidecar: %s\n", path); return; }

    /* Header */
    uint8_t magic[8];
    uint32_t nl, np, P, reserved;
    if (fread(magic, 1, 8, f) != 8 ||
        fread(&nl, 4, 1, f) != 1 ||
        fread(&np, 4, 1, f) != 1 ||
        fread(&P,  4, 1, f) != 1 ||
        fread(&reserved, 4, 1, f) != 1) {
        fprintf(stderr, "[ACDC] sidecar header read error: %s\n", path);
        fclose(f); return;
    }
    static const uint8_t EXPECTED_MAGIC[8] = {
        'A','C','D','B','D','\x01','\x00','\x00'
    };
    if (memcmp(magic, EXPECTED_MAGIC, 8) != 0) {
        fprintf(stderr, "[ACDC] sidecar bad magic: %s\n", path);
        fclose(f); return;
    }

    size_t n_floats = (size_t)nl * np * P;
    float * buf = (float *)malloc(n_floats * sizeof(float));
    if (!buf) { fclose(f); return; }
    if (fread(buf, sizeof(float), n_floats, f) != n_floats) {
        fprintf(stderr, "[ACDC] sidecar data read error (expected %zu floats)\n", n_floats);
        free(buf); fclose(f); return;
    }
    fclose(f);

    g_acdc_diag.data     = buf;
    g_acdc_diag.n_layers = nl;
    g_acdc_diag.n_proj   = np;
    g_acdc_diag.P        = P;
    fprintf(stderr, "[ACDC] loaded sidecar: %s (n_layers=%u n_proj=%u P=%u)\n",
            path, nl, np, P);
}

/* Call this before building/executing the compute graph for a new run. */
void bitnet_acdc_diag_reset_counter(void) {
    atomic_store_explicit(&g_acdc_rect_call_count, 0, memory_order_relaxed);
}

#endif /* BITNET_L3_ACDC */

#if defined(BITNET_L4_TROPICAL)
#include "ggml-bitnet-tropical.h"
#include "ggml-bitnet-kv-cache.h"
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

/* ── ACDC FFN rect (Fase II: H_P·diag(d)·H_P for rectangular FFN) ────────── */

struct acdc_ffn_rect_ud {
    int     m;           /* output dim */
    int     n;           /* input dim */
    float * d;           /* diagonal [P], P = next_pow2(max(m,n)) */
    int8_t *x_i8;        /* scratch [n] for per-sample quantization */
    bool    initialized;
};

static void acdc_ffn_rect_init_buffers(struct acdc_ffn_rect_ud * p) {
    const int P = fwht_next_pow2(p->m > p->n ? p->m : p->n);
    p->d   = (float  *)calloc((size_t)P,    sizeof(float));
    p->x_i8= (int8_t *)calloc((size_t)p->n, sizeof(int8_t));

    /* Priority 1: load real d* from sidecar binary (highest quality). */
    acdc_diag_load_once();
    if (g_acdc_diag.data && p->d) {
        int call_idx = atomic_fetch_add_explicit(&g_acdc_rect_call_count, 1,
                                                  memory_order_relaxed);
        /* call_idx layout: layer * n_proj + proj_idx
         *   proj 0 → up  (m > n, i.e. n_ff > n_embd)
         *   proj 1 → down (m < n, i.e. n_embd < n_ff)
         * Guard: only use sidecar data if P matches and we're in range. */
        uint32_t np = g_acdc_diag.n_proj;   /* = 2 */
        uint32_t nl = g_acdc_diag.n_layers;
        uint32_t sP = g_acdc_diag.P;
        uint32_t layer = (uint32_t)(call_idx / np);
        uint32_t proj  = (uint32_t)(call_idx % np);
        if ((uint32_t)P == sP && layer < nl) {
            size_t offset = ((size_t)layer * np + proj) * sP;
            memcpy(p->d, g_acdc_diag.data + offset, (size_t)P * sizeof(float));
            p->initialized = true;
            return;
        }
        /* P mismatch or out of range — fall through to default. */
    }

    /* Priority 2: randomize d for timing benchmarks (output is garbage). */
    const char * env = getenv("BITNET_ACDC_FFN_RECT_RAND");
    if (env && env[0] == '1' && p->d) {
        unsigned seed = 0xdeadbeef;
        float scale = 2.0f / (float)P;
        for (int i = 0; i < P; i++) {
            seed = seed * 1664525u + 1013904223u;
            float u = (float)((int)(seed >> 8) & 0xffffff) / (float)0xffffff - 0.5f;
            p->d[i] = u * scale;
        }
    }
    /* Priority 3 (default): d = all-zeros (calloc above). */
    p->initialized = true;
}

/*
 * custom2 callback: dst shape = [m, n_tokens] (from the shape template in src[0]).
 * src[0] = shape template tensor (not read — its only role is to set dst shape).
 * src[1] = actual input x [n, n_tokens].
 *
 * Using ggml_map_custom2 (not custom1) is required because the FFN up projection
 * changes the first dimension (n_embd → n_ff where n_ff ≠ n_embd).  custom1
 * would produce an output with the same shape as x, leading to a buffer overflow
 * when writing m > n output elements per batch item.
 */
static void acdc_ffn_rect_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * /* shape_t */,   /* src[0]: shape template, not read */
    const struct ggml_tensor * a,               /* src[1]: actual input x */
    int ith, int nth, void * userdata)
{
    (void)nth;
    if (ith != 0) return;

    struct acdc_ffn_rect_ud * p = (struct acdc_ffn_rect_ud *)userdata;
    if (!p->initialized) acdc_ffn_rect_init_buffers(p);
    if (!p->d || !p->x_i8) return;

    const int batch = (int)(ggml_nelements(a) / p->n);
    const float * x = (const float *)a->data;
    float       * y = (float *)dst->data;

    for (int b = 0; b < batch; b++) {
        const float * xb = x + b * p->n;

        /* Per-sample int8 quantization */
        float mx = 1e-6f;
        for (int i = 0; i < p->n; i++) mx = fmaxf(mx, fabsf(xb[i]));
        float s = 127.0f / mx;
        for (int i = 0; i < p->n; i++) {
            float v = xb[i] * s;
            if (v >  127.0f) v =  127.0f;
            if (v < -128.0f) v = -128.0f;
            p->x_i8[i] = (int8_t)(int)v;
        }

        acdc_forward_rect_i8(y + b * p->m, p->m, p->x_i8, p->n, p->d);
    }
}

struct ggml_tensor * bitnet_op_acdc_ffn_rect(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    int                   m,
    int                   n)
{
    struct acdc_ffn_rect_ud * ud =
        (struct acdc_ffn_rect_ud *)malloc(sizeof(*ud));
    if (!ud) return x;
    ud->m = m; ud->n = n;
    ud->d = NULL; ud->x_i8 = NULL;
    ud->initialized = false;

    /* Shape template: ggml_map_custom2 creates output with same shape as first arg.
     * We set first arg to a tensor of shape [m, n_tokens] so the output has the
     * correct dimensions for the FFN projection (m may be > n for up-projection). */
    int64_t n_tok = (x->ne[1] > 0) ? x->ne[1] : 1;
    struct ggml_tensor * shape_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t)m, n_tok);
    return ggml_map_custom2(ctx, shape_t, x, acdc_ffn_rect_callback, /*n_tasks=*/1, ud);
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

struct ggml_tensor * bitnet_op_acdc_ffn_rect(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,
    int                   m,
    int                   n)
{
    (void)ctx; (void)m; (void)n;
    return x;
}

void bitnet_acdc_diag_reset_counter(void) {}   /* no-op without L3_ACDC */

#endif /* BITNET_L3_ACDC */

/* ─── L4: Tropical attention ─────────────────────────────────────────────── */

#if defined(BITNET_L4_TROPICAL)

struct tropical_ud {
    int   topk;
    float scale;
    int   layer;   /* current transformer layer (set by KQV site via
                    * bitnet_kv_i8_cache_set_layer, captured at ggml_map_custom3
                    * time). Used to index the persistent K_i8 cache. */
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

    /* Q is per-thread (and small: d bytes); allocate per call as before.
     * K is now sourced from the persistent K_i8 cache (see
     * ggml-bitnet-kv-cache.h), indexed by (il, kv_head). The cache holds
     * an int8 buffer of n_kv * d entries with a locked scale computed on
     * the first call for that (il, kv_head); subsequent calls only
     * quantize the new keys appended to the KV cache. This eliminates
     * the O(n_kv * d) re-quantization on every decode step (the 3-pass K
     * problem from SESSION_SUMMARY.md §S2.4). */
    int8_t * q_i8 = (int8_t *)malloc((size_t)d);
    if (!q_i8) return;

    for (int h = ith; h < n_head; h += nth) {
        const int    kv_h    = h / gqa;
        const float *q_head  = q_f + (size_t)h    * n_tokens * d;
        const float *k_head  = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head  = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd  = out + (size_t)h    * n_tokens * d;

        /* Incremental K_i8: only the new keys get quantized. */
        float    k_scale = 0.0f;
        int      last_n  = 0;
        int      n_new   = 0;
        int8_t * k_i8 = bitnet_kv_i8_cache_get(p->layer, kv_h, k_head, n_kv, d,
                                                &k_scale, &last_n, &n_new);
        int k_i8_owned = (k_i8 != NULL);  /* 1 = cache owns, 0 = we malloc'd */

        if (!k_i8) {
            /* Cache miss (slot not allocated, or layer out of range):
             * fall back to per-call quant. We own this buffer. */
            k_i8 = (int8_t *)malloc((size_t)n_kv * d);
            if (!k_i8) continue;
            k_scale = quantize_f32_to_i8(k_head, k_i8, n_kv * d);
        }

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

        /* Free only the malloc'd fallback; cache-owned k_i8 stays. */
        if (!k_i8_owned) free(k_i8);
    }

    free(q_i8);
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
    ud->layer = bitnet_kv_i8_current_layer();  /* -1 if unset → cache miss */
    return ggml_map_custom3(ctx, q, k, v, tropical_callback, GGML_N_TASKS_MAX, ud);
}

/* ─── L4 variant: Float sparse top-K attention ───────────────────────────
 *
 * Uses float32 dot products for scoring — no ternary quantization.
 * Single pass over K (vs 3 passes in tropical_callback).
 * Activated by BITNET_SPARSE_TOPK env var.
 * Same thread-parallel head-strided layout as tropical_callback.
 */
static void sparse_float_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * q_t,
    const struct ggml_tensor * k_t,
    const struct ggml_tensor * v_t,
    int ith, int nth, void * userdata)
{
    const struct tropical_ud * p = (const struct tropical_ud *)userdata;

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

    /* Thread ith handles heads ith, ith+nth, ... No scratch buffers needed. */
    for (int h = ith; h < n_head; h += nth) {
        const int    kv_h   = h / gqa;
        const float *q_head = q_f + (size_t)h    * n_tokens * d;
        const float *k_head = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd = out + (size_t)h    * n_tokens * d;

        for (int qi = 0; qi < n_tokens; qi++) {
            sparse_attention_float(
                out_hd + qi * d,
                q_head + qi * d,
                k_head,
                v_head,
                n_kv,
                d,
                p->topk);
        }
    }
}

struct ggml_tensor * bitnet_op_sparse_attn(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    int                   topk,
    float                 scale)
{
    (void)scale;
    struct tropical_ud * ud = (struct tropical_ud *)malloc(sizeof(*ud));
    ud->topk  = topk;
    ud->scale = scale;
    return ggml_map_custom3(ctx, q, k, v, sparse_float_callback, GGML_N_TASKS_MAX, ud);
}

/* ─── L4 variant: Adaptive-K float sparse attention ─────────────────────
 *
 * Per-query dynamic K via cumulative softmax threshold.
 * Activated by BITNET_SPARSE_TOPK_ADAPTIVE=<coverage> (e.g. "0.90").
 */
struct sparse_adaptive_ud {
    float coverage;
    int   k_min;
    int   k_max;
};

static void sparse_float_adaptive_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * q_t,
    const struct ggml_tensor * k_t,
    const struct ggml_tensor * v_t,
    int ith, int nth, void * userdata)
{
    const struct sparse_adaptive_ud * p = (const struct sparse_adaptive_ud *)userdata;

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

    for (int h = ith; h < n_head; h += nth) {
        const int    kv_h   = h / gqa;
        const float *q_head = q_f + (size_t)h    * n_tokens * d;
        const float *k_head = k_f + (size_t)kv_h * n_kv     * d;
        const float *v_head = v_f + (size_t)kv_h * n_kv     * d;
        float       *out_hd = out + (size_t)h    * n_tokens * d;

        for (int qi = 0; qi < n_tokens; qi++) {
            sparse_attention_float_adaptive(
                out_hd + qi * d,
                q_head + qi * d,
                k_head,
                v_head,
                n_kv,
                d,
                p->coverage,
                p->k_min,
                p->k_max);
        }
    }
}

struct ggml_tensor * bitnet_op_sparse_attn_adaptive(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    float                 coverage,
    int                   k_min,
    int                   k_max)
{
    struct sparse_adaptive_ud * ud =
        (struct sparse_adaptive_ud *)malloc(sizeof(*ud));
    if (!ud) return q;
    ud->coverage = coverage;
    ud->k_min    = k_min;
    ud->k_max    = k_max;
    return ggml_map_custom3(ctx, q, k, v,
                            sparse_float_adaptive_callback,
                            GGML_N_TASKS_MAX, ud);
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

struct ggml_tensor * bitnet_op_sparse_attn(
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

struct ggml_tensor * bitnet_op_sparse_attn_adaptive(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v,
    float                 coverage,
    int                   k_min,
    int                   k_max)
{
    (void)ctx; (void)k; (void)v; (void)coverage; (void)k_min; (void)k_max;
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

/* ─── L5: HRR attention with phasor positional keys ───────────────────────
 *
 * Replaces the model's K projections with deterministic phasor keys
 * (one per position, seeded by head_index * MAX_KV + position).
 *
 * Advantage vs ternary-derived keys:
 *   k_phasor ⊛ k_phasor_inv = δ  (exact — zero inversion error)
 *   Gaussian/ternary: k ⊛ k_inv ≈ δ + O(1/√d) error
 *
 * The V values from the model are still used unchanged.
 * Memory layout: M = Σᵢ phasor_k[i] ⊛ V[i]
 * Retrieval: out ≈ M ⊛ argmin_k(‖Q - phasor_k[k]‖₂)⁻¹
 *
 * Enable at runtime: BITNET_HRR_PHASOR=1
 */
static void hrr_phasor_callback(
    struct ggml_tensor       * dst,
    const struct ggml_tensor * q_t,
    const struct ggml_tensor * k_t,
    const struct ggml_tensor * v_t,
    int ith, int nth, void * userdata)
{
    (void)userdata; (void)k_t;

    const int d         = (int)q_t->ne[0];
    const int n_tokens  = (int)q_t->ne[1];
    const int n_head    = (int)(q_t->ne[2] > 0 ? q_t->ne[2] : 1);
    const int n_kv      = (int)k_t->ne[1];
    const int n_head_kv = (int)(k_t->ne[2] > 0 ? k_t->ne[2] : 1);
    const int gqa       = n_head / n_head_kv;

    const float * q_f = (const float *)q_t->data;
    const float * v_f = (const float *)v_t->data;
    float       * out = (float *)dst->data;

    /* Per-thread scratch */
    float * M          = (float *)malloc((size_t)d * sizeof(float));
    float * tmp        = (float *)malloc((size_t)4 * (d + 2) * sizeof(float));
    /* All n_kv phasor keys + their exact inverses for one head */
    float * pk_all     = (float *)malloc((size_t)n_kv * d * sizeof(float));
    float * pk_inv_all = (float *)malloc((size_t)n_kv * d * sizeof(float));

    if (!M || !tmp || !pk_all || !pk_inv_all) {
        free(M); free(tmp); free(pk_all); free(pk_inv_all);
        return;
    }

    for (int h = ith; h < n_head; h += nth) {
        const int    kv_h   = h / gqa;
        const float *v_head = v_f + (size_t)kv_h * n_kv * d;
        float       *out_hd = out + (size_t)h    * n_tokens * d;

        /* 1. Generate phasor keys for all positions in this head.
         *    Seed: (head_index << 20) | position — unique per (head, pos). */
        for (int i = 0; i < n_kv; i++) {
            uint64_t seed = ((uint64_t)(kv_h + 1) << 20) | (uint64_t)i;
            float * pki     = pk_all     + (size_t)i * d;
            float * pki_inv = pk_inv_all + (size_t)i * d;
            hrr_phasor_key_init(pki, d, seed);
            hrr_phasor_inv(pki_inv, pki, d, tmp);
        }

        /* 2. Build holographic memory: M = Σᵢ phasor_k[i] ⊛ V[i] */
        memset(M, 0, (size_t)d * sizeof(float));
        for (int i = 0; i < n_kv; i++) {
            hrr_accumulate(M, pk_all + (size_t)i * d,
                           v_head   + (size_t)i * d, d, tmp);
        }

        /* 3. Retrieve for each query token.
         *    Strategy: find best-matching phasor key via dot product Q·phasor_k,
         *    then unbind with its exact inverse. */
        const float * q_head = q_f + (size_t)h * n_tokens * d;
        for (int t = 0; t < n_tokens; t++) {
            const float * q_tok = q_head + (size_t)t * d;
            float       * out_t = out_hd + (size_t)t * d;

            /* Find closest phasor key to query (cosine proxy = dot product,
             * all phasor keys have ||k||=1 exactly). */
            int   best_i   = 0;
            float best_dot = 0.0f;
            for (int i = 0; i < n_kv; i++) {
                const float * pki = pk_all + (size_t)i * d;
                float dot = 0.0f;
                for (int j = 0; j < d; j++) dot += q_tok[j] * pki[j];
                if (dot > best_dot) { best_dot = dot; best_i = i; }
            }

            /* Unbind: out ≈ M ⊛ phasor_k_inv[best_i] */
            hrr_unbind(out_t, M, pk_inv_all + (size_t)best_i * d, d, tmp);
        }
    }

    free(M); free(tmp); free(pk_all); free(pk_inv_all);
}

struct ggml_tensor * bitnet_op_hrr_attn_phasor(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v)
{
    return ggml_map_custom3(ctx, q, k, v, hrr_phasor_callback, GGML_N_TASKS_MAX, NULL);
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

struct ggml_tensor * bitnet_op_hrr_attn_phasor(
    struct ggml_context * ctx,
    struct ggml_tensor  * q,
    struct ggml_tensor  * k,
    struct ggml_tensor  * v)
{
    (void)ctx; (void)k; (void)v;
    return q;
}

#endif /* BITNET_L5_HRR */
