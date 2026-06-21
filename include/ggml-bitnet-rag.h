/*
 * ggml-bitnet-rag.h — CPU-RAG flat-index retrieval engine (Level 6)
 *
 * ─────────────────────────────────────────────────────────────────────────
 * DESIGN OVERVIEW
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Retrieval-Augmented Generation requires fast ANN (approximate nearest-
 * neighbor) search over a corpus of document embeddings.  This module
 * provides a flat-index brute-force ANN engine optimized for CPU:
 *
 *   - Score all documents: O(n·d) inner products (compiler-vectorized F32)
 *   - Select top-K:        O(n·log K) via partial_sort (std::partial_sort)
 *   - Adaptive K:          cumulative softmax threshold (Direção D, L4)
 *
 * Target: n ≤ 100K documents, d ≤ 4096.  On a 4-core laptop CPU:
 *   n=10K, d=768  → ~2ms per query (single-threaded, no SIMD intrinsics)
 *   n=100K, d=768 → ~20ms per query
 *
 * Connection to L4 / L5 kernels:
 *   - Scoring logic matches sparse_attention_float (L4) with V=identity
 *   - Adaptive K follows tropical_adaptive_k (L4, Direção D)
 *   - Optional: rag_fingerprint() uses hrr_phasor_key_init (L5) to
 *     generate compact 64-float fingerprints for dedup / fast pre-filter
 *
 * ─────────────────────────────────────────────────────────────────────────
 * API OVERVIEW
 * ─────────────────────────────────────────────────────────────────────────
 *
 *  LIFECYCLE:
 *    rag_store_t *s = rag_store_create(capacity, d);
 *    rag_store_add(s, embedding);          // returns doc_id
 *    rag_retrieve_topk(s, query, k, ...);  // fixed-K retrieval
 *    rag_retrieve_adaptive(s, query, ...); // coverage-based K
 *    rag_store_free(s);
 *
 *  CTYPES BRIDGE (Python):
 *    Build with -DBITNET_L6_RAG=ON -DBITNET_RAG_SHARED=ON
 *    Then in Python:
 *      import ctypes, numpy as np
 *      lib = ctypes.CDLL("build/lib/libbitnet_rag.so")
 *      # see utils/rag_demo.py for full wrappers
 *
 * ─────────────────────────────────────────────────────────────────────────
 * SCORING CONVENTION
 * ─────────────────────────────────────────────────────────────────────────
 *
 * Scores are (query · doc) / sqrt(d) — NOT cosine similarity.
 * For cosine similarity, normalize embeddings to unit length before insertion.
 * Higher score = better match.
 */

#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle — definition in ggml-bitnet-rag.cpp */
typedef struct rag_store rag_store_t;

/* ─── Lifecycle ───────────────────────────────────────────────────────── */

/*
 * rag_store_create: allocate a flat embedding store.
 *
 * @param capacity  maximum number of documents (static allocation)
 * @param d         embedding dimension (must match all subsequent calls)
 * @return          new store, or NULL on allocation failure
 */
rag_store_t * rag_store_create(int capacity, int d);

/*
 * rag_store_free: free all memory. Safe to call with NULL.
 */
void rag_store_free(rag_store_t *store);

/*
 * rag_store_reset: discard all documents, keep allocated memory.
 * Next rag_store_add() starts from doc_id = 0.
 */
void rag_store_reset(rag_store_t *store);

/* ─── Insertion ───────────────────────────────────────────────────────── */

/*
 * rag_store_add: add one document embedding.
 *
 * @param store      the RAG store
 * @param embedding  float array of length d (copied; caller may free)
 * @return           doc_id (0-based, monotonically increasing), or -1 if full
 */
int rag_store_add(rag_store_t *store, const float *embedding);

/* ─── Retrieval: fixed K ──────────────────────────────────────────────── */

/*
 * rag_retrieve_topk: retrieve the K highest-scoring documents.
 *
 * Scores all documents with inner-product scan, returns top-K in
 * descending score order.
 *
 * Complexity: O(n·d + n·log K)
 *
 * @param store      the RAG store
 * @param query      query embedding [d floats]
 * @param k          number of results requested (clamped to n_docs)
 * @param out_ids    output: doc ids [k ints] in descending score order
 * @param out_scores output: scores [k floats] in descending order
 * @return           actual number of results (min(k, n_docs))
 */
int rag_retrieve_topk(
    rag_store_t  * store,
    const float  * query,
    int            k,
    int          * out_ids,
    float        * out_scores);

/* ─── Retrieval: adaptive K (Direção D) ──────────────────────────────── */

/*
 * rag_retrieve_adaptive: retrieve with query-adaptive K.
 *
 * Selects the minimum K in [k_min, k_max] such that the top-K softmax
 * weights (normalized over top-k_max) cover ≥ `coverage` probability mass.
 * Concentrated queries (one dominant result) return K ≈ k_min; diffuse
 * queries return K ≈ k_max.
 *
 * Complexity: O(n·d + n·log k_max + k_max)
 *
 * @param store      the RAG store
 * @param query      query embedding [d floats]
 * @param coverage   target probability mass [0,1]; 0.90 is a good default
 * @param k_min      minimum K to return (floor; ≥ 1)
 * @param k_max      maximum K budget (≤ n_docs)
 * @param out_ids    output: doc ids [k_max ints] (allocate for k_max)
 * @param out_scores output: scores [k_max floats] (allocate for k_max)
 * @return           actual K chosen (in [k_min, min(k_max, n_docs)])
 */
int rag_retrieve_adaptive(
    rag_store_t  * store,
    const float  * query,
    float          coverage,
    int            k_min,
    int            k_max,
    int          * out_ids,
    float        * out_scores);

/* ─── Stats ───────────────────────────────────────────────────────────── */

/*
 * rag_store_n_docs: current number of documents (0 after reset).
 */
int rag_store_n_docs(const rag_store_t *store);

/*
 * rag_store_dim: embedding dimension passed to rag_store_create.
 */
int rag_store_dim(const rag_store_t *store);

#ifdef __cplusplus
}
#endif
