/*
 * ggml-bitnet-kv-cache.cpp
 *
 * Implementation of the per-(layer, kv_head) persistent K_i8 cache for
 * tropical attention. See ggml-bitnet-kv-cache.h for design rationale.
 *
 * Thread-safety contract: each (il, kv_head) slot has at most one writer
 * per compute pass (enforced by the tropical callback's strided head loop).
 * No internal locking. Safe to call from multiple threads as long as each
 * thread touches a different (il, kv_head).
 */

#include "ggml-bitnet-kv-cache.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <pthread.h>

/* ─── Per-slot state ────────────────────────────────────────────────────── */

struct kv_i8_slot {
    int8_t  * data;       /* quantized keys [capacity * d]                  */
    int       n_quantized;/* entries currently valid (0 = uninitialized)  */
    int       capacity;   /* allocated entries (always >= n_quantized)     */
    float     k_scale;    /* locked quantization scale (set on first call)*/
    pthread_mutex_t mtx;  /* per-slot mutex (GQA: multiple heads share kv_h)*/
};

static struct kv_i8_slot ** g_cache     = NULL;  /* [n_layer][n_head_kv]    */
static int                  g_n_layer   = 0;
static int                  g_n_head_kv = 0;
static int                  g_d         = 0;
static int                  g_max_n_kv  = 0;
static int                  g_cur_il    = -1;     /* current layer (set by setter) */

/* ─── Init / reset / free ───────────────────────────────────────────────── */

void bitnet_kv_i8_cache_init(int n_layer, int n_head_kv, int d, int max_n_kv) {
    if (n_layer <= 0 || n_head_kv <= 0 || d <= 0 || max_n_kv <= 0) return;

    /* If shape matches, no-op. The caller may call repeatedly with the same
     * shape (e.g. every forward pass); we don't want to realloc. */
    if (g_cache && g_n_layer == n_layer && g_n_head_kv == n_head_kv &&
        g_d == d && g_max_n_kv >= max_n_kv) {
        return;
    }

    /* Shape changed (model swap or first init with non-default args): free
     * and realloc. */
    bitnet_kv_i8_cache_free();

    g_cache = (struct kv_i8_slot **)calloc((size_t)n_layer, sizeof(*g_cache));
    if (!g_cache) return;
    for (int il = 0; il < n_layer; il++) {
        g_cache[il] = (struct kv_i8_slot *)calloc((size_t)n_head_kv,
                                                  sizeof(struct kv_i8_slot));
        if (!g_cache[il]) {
            /* Partial init: free everything and bail. */
            bitnet_kv_i8_cache_free();
            return;
        }
        for (int h = 0; h < n_head_kv; h++) {
            pthread_mutex_init(&g_cache[il][h].mtx, NULL);
        }
    }
    g_n_layer   = n_layer;
    g_n_head_kv = n_head_kv;
    g_d         = d;
    g_max_n_kv  = max_n_kv;
}

void bitnet_kv_i8_cache_reset(void) {
    if (!g_cache) return;
    for (int il = 0; il < g_n_layer; il++) {
        if (!g_cache[il]) continue;
        for (int h = 0; h < g_n_head_kv; h++) {
            pthread_mutex_lock(&g_cache[il][h].mtx);
            g_cache[il][h].n_quantized = 0;
            g_cache[il][h].k_scale     = 0.0f;
            pthread_mutex_unlock(&g_cache[il][h].mtx);
        }
    }
}

void bitnet_kv_i8_cache_free(void) {
    if (!g_cache) return;
    for (int il = 0; il < g_n_layer; il++) {
        if (!g_cache[il]) continue;
        for (int h = 0; h < g_n_head_kv; h++) {
            pthread_mutex_destroy(&g_cache[il][h].mtx);
            free(g_cache[il][h].data);
            g_cache[il][h].data       = NULL;
            g_cache[il][h].n_quantized = 0;
            g_cache[il][h].capacity    = 0;
        }
        free(g_cache[il]);
        g_cache[il] = NULL;
    }
    free(g_cache);
    g_cache     = NULL;
    g_n_layer   = 0;
    g_n_head_kv = 0;
    g_d         = 0;
    g_max_n_kv  = 0;
    g_cur_il    = -1;
}

/* ─── Setter for current layer (called by llama.cpp KQV site) ──────────── */

void bitnet_kv_i8_cache_set_layer(int il) {
    g_cur_il = il;
}

/*
 * Get the layer index most recently passed to bitnet_kv_i8_cache_set_layer.
 * The tropical dispatch captures this at ggml_map_custom3 time and stores
 * it in the userdata so the callback can index the cache without changing
 * the public bitnet_op_tropical_attn signature.
 *
 * Returns -1 if no layer has been set yet (caller should treat as a cache
 * miss and fall back to per-call quantization).
 */
int bitnet_kv_i8_current_layer(void) {
    return g_cur_il;
}

/* ─── Core: get (or quantize-incrementally) K_i8 buffer ────────────────── */

int8_t * bitnet_kv_i8_cache_get(
    int            il,
    int            kv_head,
    const float  * K_f32,
    int            n_kv,
    int            d,
    float        * k_scale_out,
    int          * last_n_out,
    int          * n_new_out)
{
    if (last_n_out) *last_n_out = 0;
    if (n_new_out)  *n_new_out  = 0;
    if (k_scale_out) *k_scale_out = 0.0f;
    if (d <= 0) return NULL;

    /* Auto-init or reinit when d doesn't match the current cache.
     * This handles: first call (g_cache==NULL), model swap (different
     * head_dim), and the original lazy-init that hardcoded d=128. */
    if (!g_cache || g_d != d) {
        int n_l = (g_n_layer   > 0) ? g_n_layer   : 64;
        int n_h = (g_n_head_kv > 0) ? g_n_head_kv : 64;
        int mx  = (g_max_n_kv  > 0) ? g_max_n_kv  : 4096;
        bitnet_kv_i8_cache_init(n_l, n_h, d, mx);
    }
    if (!g_cache) return NULL;
    if (il < 0 || il >= g_n_layer) return NULL;
    if (kv_head < 0 || kv_head >= g_n_head_kv) return NULL;
    if (n_kv <= 0) return NULL;

    struct kv_i8_slot * slot = &g_cache[il][kv_head];

    /* Lock the slot. GQA: multiple heads (h) may map to the same kv_head,
     * so multiple threads may reach this slot concurrently. The slot work
     * (max + quantize) is O(n_kv * d) — same as the work being parallelized
     * — so the mutex adds only one serial bottleneck per (il, kv_h), not
     * per token. */
    pthread_mutex_lock(&slot->mtx);

    /* Grow capacity if needed. */
    if (slot->capacity < n_kv) {
        int new_cap = slot->capacity > 0 ? slot->capacity * 2 : 64;
        while (new_cap < n_kv) new_cap *= 2;
        if (new_cap > g_max_n_kv) new_cap = g_max_n_kv;
        if (new_cap < n_kv) {
            /* Even the global cap is insufficient; bail to caller (alloc). */
            pthread_mutex_unlock(&slot->mtx);
            return NULL;
        }
        int8_t * new_data = (int8_t *)realloc(slot->data,
                                              (size_t)new_cap * g_d * sizeof(int8_t));
        if (!new_data) { pthread_mutex_unlock(&slot->mtx); return NULL; }
        slot->data     = new_data;
        slot->capacity = new_cap;
    }

    int last_n = slot->n_quantized;
    if (last_n_out) *last_n_out = last_n;
    if (last_n == 0) {
        /* First call for this slot: quantize everything, lock the scale. */
        float mx = 1e-6f;
        for (int i = 0; i < n_kv * g_d; i++) mx = fmaxf(mx, fabsf(K_f32[i]));
        float s = 127.0f / mx;
        int8_t * dst = slot->data;
        for (int i = 0; i < n_kv * g_d; i++) {
            float v = K_f32[i] * s;
            if (v >  127.0f) v =  127.0f;
            if (v < -128.0f) v = -128.0f;
            dst[i] = (int8_t)(int)v;
        }
        slot->k_scale     = s;
        slot->n_quantized = n_kv;
        if (k_scale_out) *k_scale_out = s;
        if (n_new_out)   *n_new_out   = n_kv;
    } else if (n_kv > last_n) {
        /* Incremental: quantize only the new entries with the locked scale. */
        const float s = slot->k_scale;
        int8_t * dst = slot->data + (size_t)last_n * g_d;
        const float * src = K_f32 + (size_t)last_n * g_d;
        const int n_new = n_kv - last_n;
        for (int i = 0; i < n_new * g_d; i++) {
            float v = src[i] * s;
            if (v >  127.0f) v =  127.0f;
            if (v < -128.0f) v = -128.0f;
            dst[i] = (int8_t)(int)v;
        }
        slot->n_quantized = n_kv;
        if (k_scale_out) *k_scale_out = s;
        if (n_new_out)   *n_new_out   = n_new;
    } else {
        /* No new keys (shouldn't happen if llama.cpp appends correctly).
         * Return current state. */
        if (k_scale_out) *k_scale_out = slot->k_scale;
    }

    pthread_mutex_unlock(&slot->mtx);
    return slot->data;
}
