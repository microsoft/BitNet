// test_rag_retrieval.cpp
//
// Unit tests for the CPU-RAG flat-index retrieval engine (Level 6, Direção E).
//
// Verifies:
//   [1] exact_match       — query = doc[0] → retrieved id=0 with max score
//   [2] nn_ranking        — 8 docs at controlled distances → rank order correct
//   [3] adaptive_k        — concentrated query yields adaptive K = 1
//   [4] batch_accuracy    — 64 random docs; query=doc[i] → rank-0 is always i
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     -Iinclude -L/usr/lib/gcc/x86_64-linux-gnu/13 \
//     src/ggml-bitnet-rag.cpp test_rag_retrieval.cpp -lm -o build/test_rag_retrieval
//
// Convention: hand-rolled assert macros per T003 (no Catch2).

#include "ggml-bitnet-rag.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static int n_pass = 0, n_fail = 0;

static void report(const char *name, bool ok, const char *detail = "") {
    if (ok) { printf("  %-60s PASS ✓  %s\n", name, detail); n_pass++; }
    else     { printf("  %-60s FAIL ✗  %s\n", name, detail); n_fail++; }
}

/* ─── [1] exact_match: query = doc[0] → retrieved id=0 ─────────────────── */
static void test_exact_match() {
    printf("\n[1] Exact match: query = stored document → id=0\n");
    const int d = 64, N = 10;
    rag_store_t *s = rag_store_create(N, d);

    std::mt19937 rng(0xAABBCCDDu);
    std::normal_distribution<float> nd;

    std::vector<float> docs(N * d);
    for (auto &v : docs) v = nd(rng);

    for (int i = 0; i < N; i++)
        rag_store_add(s, docs.data() + i * d);

    /* query = exact copy of doc[0] */
    std::vector<int>   ids(N);
    std::vector<float> sc(N);
    int k_found = rag_retrieve_topk(s, docs.data(), 3, ids.data(), sc.data());

    bool ok_k   = (k_found == 3);
    bool ok_id  = (ids[0] == 0);
    bool ok_sc  = (sc[0] > 0.0f);      /* inner product with itself > 0 */

    char det[80];
    std::snprintf(det, sizeof(det), "k_found=%d, ids[0]=%d, score=%.4f",
                  k_found, ids[0], sc[0]);
    report("exact match → rank-0 is queried doc", ok_k && ok_id && ok_sc, det);
    rag_store_free(s);
}

/* ─── [2] nn_ranking: 8 docs at known inner products → rank order ───────── */
static void test_nn_ranking() {
    printf("\n[2] NN ranking: controlled inner products → deterministic rank order\n");
    const int d = 16, N = 8;
    rag_store_t *s = rag_store_create(N, d);

    /* Query = unit vector e_0 (first basis vector).
     * doc[i] = i * e_0 (scale i), so Q·doc[i] = i.
     * Expected rank: doc[7] > doc[6] > ... > doc[0]. */
    std::vector<float> query(d, 0.0f);
    query[0] = 1.0f;

    for (int i = 0; i < N; i++) {
        std::vector<float> doc(d, 0.0f);
        doc[0] = (float)i;
        rag_store_add(s, doc.data());
    }

    std::vector<int>   ids(N);
    std::vector<float> sc(N);
    int k_found = rag_retrieve_topk(s, query.data(), N, ids.data(), sc.data());

    /* Verify descending score order */
    bool ok_order = true;
    for (int i = 0; i < k_found - 1; i++)
        if (sc[i] < sc[i + 1]) { ok_order = false; break; }

    /* Top result must be doc[7] (highest scale = 7) */
    bool ok_top = (ids[0] == 7);

    /* Scores must be strictly decreasing (all distinct) */
    bool ok_distinct = true;
    for (int i = 0; i < k_found - 1; i++)
        if (sc[i] <= sc[i + 1] + 1e-6f) { ok_distinct = false; break; }

    char det[80];
    std::snprintf(det, sizeof(det), "top_id=%d, sc[0]=%.3f, sc[1]=%.3f, ordered=%d",
                  ids[0], sc[0], sc[1], ok_order);
    report("deterministic NN rank: top=doc[7], descending scores",
           ok_order && ok_top && ok_distinct, det);
    rag_store_free(s);
}

/* ─── [3] adaptive_k: one dominant doc → K=1 with coverage=0.90 ────────── */
/*
 * Design: query = e_0.  doc[0] = 50*e_0 → score = 50/√d ≈ 8.8.
 * doc[i>0]: zero first component → score = 0 exactly.
 * Softmax over k_max=16: w[0]/Σw = 1/(1+15·exp(-8.8)) ≈ 0.9978 ≥ 0.90.
 * So cumulative sum crosses 0.90 at K=1.
 */
static void test_adaptive_k() {
    printf("\n[3] Adaptive K: one dominant document → K=1 (coverage=0.90)\n");
    const int d = 32, N = 64;
    rag_store_t *s = rag_store_create(N, d);

    std::mt19937 rng(0x12345678u);
    std::normal_distribution<float> nd;

    /* query = e_0 */
    std::vector<float> query(d, 0.0f);
    query[0] = 1.0f;

    /* doc[0]: strong projection onto e_0, score = 50/sqrt(32) ≈ 8.84 */
    std::vector<float> doc0(d, 0.0f);
    doc0[0] = 50.0f;
    rag_store_add(s, doc0.data());

    /* doc[i>0]: zero first component → score = 0 (orthogonal to query) */
    for (int i = 1; i < N; i++) {
        std::vector<float> doc(d, 0.0f);
        for (int j = 1; j < d; j++) doc[j] = nd(rng);  /* j≥1: orthogonal */
        rag_store_add(s, doc.data());
    }

    std::vector<int>   ids(N);
    std::vector<float> sc(N);
    int K = rag_retrieve_adaptive(s, query.data(), 0.90f, 1, 16, ids.data(), sc.data());

    bool ok = (K == 1 && ids[0] == 0);
    char det[64];
    std::snprintf(det, sizeof(det), "K=%d, top_id=%d, score=%.3f", K, ids[0], sc[0]);
    report("concentrated → adaptive K=1, top=doc[0]", ok, det);
    rag_store_free(s);
}

/* ─── [4] batch_accuracy: query=doc[i] → always retrieved at rank 0 ─────── */
static void test_batch_accuracy() {
    printf("\n[4] Batch accuracy: query=doc[i] → always rank-0 (10 queries)\n");
    const int d = 128, N = 64, N_QUERIES = 10;
    rag_store_t *s = rag_store_create(N, d);

    std::mt19937 rng(0xDEADC0DEu);
    std::normal_distribution<float> nd;

    std::vector<float> corpus(N * d);
    for (auto &v : corpus) v = nd(rng);

    for (int i = 0; i < N; i++)
        rag_store_add(s, corpus.data() + i * d);

    int n_ok = 0;
    std::vector<int>   ids(5);
    std::vector<float> sc(5);
    for (int q = 0; q < N_QUERIES; q++) {
        /* Use a random doc as the query (exact match → should be rank-0) */
        int target = (q * 7) % N;   /* deterministic spread */
        int k_found = rag_retrieve_topk(s, corpus.data() + (size_t)target * d,
                                        5, ids.data(), sc.data());
        if (k_found > 0 && ids[0] == target) n_ok++;
    }

    bool ok = (n_ok == N_QUERIES);
    char det[64];
    std::snprintf(det, sizeof(det), "%d/%d queries rank-0 correct", n_ok, N_QUERIES);
    report("all exact-query retrievals return rank-0=target", ok, det);
    rag_store_free(s);
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  CPU-RAG Retrieval Engine — Direção E (Level 6)\n");
    printf("═══════════════════════════════════════════════════════════\n");

    test_exact_match();
    test_nn_ranking();
    test_adaptive_k();
    test_batch_accuracy();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Resultado: %d/%d %s\n", n_pass, n_pass + n_fail,
           n_fail == 0 ? "PASSARAM ✓" : "FALHARAM ✗");
    printf("═══════════════════════════════════════════════════════════\n");
    return n_fail == 0 ? 0 : 1;
}
