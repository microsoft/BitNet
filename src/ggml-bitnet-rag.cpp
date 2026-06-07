/*
 * ggml-bitnet-rag.cpp — CPU-RAG flat-index retrieval engine (Level 6)
 *
 * Provides rag_store_t: a flat float32 embedding matrix that supports
 * O(n·d) brute-force ANN search via inner-product scoring + partial sort.
 *
 * Scoring: (query · doc) / sqrt(d)  — same convention as sparse_attention_float.
 * Adaptive K: cumulative softmax threshold — same algorithm as tropical_adaptive_k.
 *
 * No ggml runtime dependency. Can be linked as a standalone shared library
 * for Python ctypes (build with -DBITNET_RAG_SHARED=ON).
 */

#include "ggml-bitnet-rag.h"

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cfloat>

/* ─── Store internals ─────────────────────────────────────────────────── */

struct rag_store {
    float * embeddings;  /* [capacity × d] float32, row-major */
    int     n_docs;      /* number of documents currently stored */
    int     capacity;    /* maximum documents (static allocation) */
    int     d;           /* embedding dimension */
};

/* ─── Lifecycle ───────────────────────────────────────────────────────── */

rag_store_t * rag_store_create(int capacity, int d) {
    if (capacity <= 0 || d <= 0) return NULL;
    rag_store_t *s = (rag_store_t *)malloc(sizeof(rag_store_t));
    if (!s) return NULL;
    s->embeddings = (float *)malloc((size_t)capacity * (size_t)d * sizeof(float));
    if (!s->embeddings) { free(s); return NULL; }
    s->n_docs   = 0;
    s->capacity = capacity;
    s->d        = d;
    return s;
}

void rag_store_free(rag_store_t *store) {
    if (!store) return;
    free(store->embeddings);
    free(store);
}

void rag_store_reset(rag_store_t *store) {
    if (store) store->n_docs = 0;
}

/* ─── Insertion ───────────────────────────────────────────────────────── */

int rag_store_add(rag_store_t *store, const float *embedding) {
    if (!store || !embedding || store->n_docs >= store->capacity) return -1;
    int id = store->n_docs++;
    memcpy(store->embeddings + (size_t)id * (size_t)store->d,
           embedding, (size_t)store->d * sizeof(float));
    return id;
}

/* ─── Stats ───────────────────────────────────────────────────────────── */

int rag_store_n_docs(const rag_store_t *store) { return store ? store->n_docs : 0; }
int rag_store_dim(const rag_store_t *store)    { return store ? store->d      : 0; }

/* ─── Internal: score all documents against query ─────────────────────── */

/*
 * score_all: compute scores[i] = (query · doc[i]) / sqrt(d) for all i.
 * Compiler will auto-vectorize the inner dot product loop with AVX2/NEON.
 */
static void score_all(
    const rag_store_t * store,
    const float       * query,
    float             * scores)
{
    const int n   = store->n_docs;
    const int d   = store->d;
    const float inv_sqrt_d = 1.0f / sqrtf((float)d);
    const float *emb = store->embeddings;

    for (int i = 0; i < n; i++) {
        const float *doc = emb + (size_t)i * (size_t)d;
        float dot = 0.0f;
        for (int j = 0; j < d; j++) dot += query[j] * doc[j];
        scores[i] = dot * inv_sqrt_d;
    }
}

/* ─── Fixed-K retrieval ─────────────────────────────────────────────────── */

int rag_retrieve_topk(
    rag_store_t  * store,
    const float  * query,
    int            k,
    int          * out_ids,
    float        * out_scores)
{
    if (!store || !query || !out_ids || !out_scores || store->n_docs <= 0) return 0;
    const int n = store->n_docs;
    const int K = (k < n) ? k : n;
    if (K <= 0) return 0;

    float * scores = (float *)malloc((size_t)n * sizeof(float));
    int   * idx    = (int   *)malloc((size_t)n * sizeof(int));
    if (!scores || !idx) { free(scores); free(idx); return 0; }

    score_all(store, query, scores);
    for (int i = 0; i < n; i++) idx[i] = i;

    std::partial_sort(idx, idx + K, idx + n,
        [scores](int a, int b) { return scores[a] > scores[b]; });

    for (int i = 0; i < K; i++) {
        out_ids[i]    = idx[i];
        out_scores[i] = scores[idx[i]];
    }

    free(scores);
    free(idx);
    return K;
}

/* ─── Adaptive-K retrieval ────────────────────────────────────────────── */

int rag_retrieve_adaptive(
    rag_store_t  * store,
    const float  * query,
    float          coverage,
    int            k_min,
    int            k_max,
    int          * out_ids,
    float        * out_scores)
{
    if (!store || !query || !out_ids || !out_scores || store->n_docs <= 0) return 0;
    const int n = store->n_docs;

    int K_limit = (k_max < n) ? k_max : n;
    if (k_min < 1)       k_min = 1;
    if (k_min > K_limit) k_min = K_limit;

    float * scores = (float *)malloc((size_t)n       * sizeof(float));
    int   * idx    = (int   *)malloc((size_t)n       * sizeof(int));
    float * w      = (float *)malloc((size_t)K_limit * sizeof(float));
    if (!scores || !idx || !w) { free(scores); free(idx); free(w); return 0; }

    /* Step 1: score all docs O(n·d) */
    score_all(store, query, scores);
    for (int i = 0; i < n; i++) idx[i] = i;

    /* Step 2: partial sort to get top K_limit O(n·log K) */
    std::partial_sort(idx, idx + K_limit, idx + n,
        [scores](int a, int b) { return scores[a] > scores[b]; });

    /* Step 3: cumulative softmax → adaptive K O(K_limit) */
    float max_s = scores[idx[0]], sum_exp = 0.0f;
    for (int k = 0; k < K_limit; k++) {
        w[k]     = expf(scores[idx[k]] - max_s);
        sum_exp += w[k];
    }
    float inv_sum = 1.0f / sum_exp;
    float cum     = 0.0f;
    int   K_chosen = K_limit;
    if (coverage < 1.0f) {
        for (int k = 0; k < K_limit; k++) {
            cum += w[k] * inv_sum;
            if (cum >= coverage) { K_chosen = k + 1; break; }
        }
    }
    if (K_chosen < k_min) K_chosen = k_min;

    /* Step 4: copy results */
    for (int k = 0; k < K_chosen; k++) {
        out_ids[k]    = idx[k];
        out_scores[k] = scores[idx[k]];
    }

    free(scores);
    free(idx);
    free(w);
    return K_chosen;
}
