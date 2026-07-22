// test_adaptive_k.cpp
//
// Unit tests for tropical_adaptive_k and sparse_attention_float_adaptive.
//
// Verifies:
//   [1] Concentrated distribution → K = 1 (single dominant token)
//   [2] Uniform distribution → K = k_max (all tokens equally likely)
//   [3] coverage=1.0 → result equals sparse_attention_float(K=k_max)
//   [4] adaptive K is always ≤ fixed K for any distribution (coverage < 1)
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     -Iinclude -L/usr/lib/gcc/x86_64-linux-gnu/13 \
//     src/ggml-bitnet-tropical.cpp src/ggml-bitnet-common.cpp \
//     test_adaptive_k.cpp -o build/test_adaptive_k
//
// Convention: hand-rolled assert macros per T003 (no Catch2).

#include "ggml-bitnet-tropical.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

static int n_pass = 0, n_fail = 0;

static void report(const char *name, bool ok, const char *detail = "") {
    if (ok) { printf("  %-60s PASS ✓  %s\n", name, detail); n_pass++; }
    else     { printf("  %-60s FAIL ✗  %s\n", name, detail); n_fail++; }
}

static bool approx_eq(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) < tol;
}

static bool vec_eq(const float *a, const float *b, int n, float tol = 1e-3f) {
    for (int i = 0; i < n; i++) if (!approx_eq(a[i], b[i], tol)) return false;
    return true;
}

/* ─── [1] Concentrated distribution → K = 1 ───────────────────────────────
 * One key has a vastly higher score. Softmax is ≈ 1.0 on that key.
 * With coverage=0.95, tropical_adaptive_k should return K=1.                */
static void test_concentrated_gives_k1() {
    printf("\n[1] Concentrated distribution (one dominant key) → K=1\n");
    const int n_keys = 64;
    std::vector<float> scores(n_keys, -10.0f);
    scores[7] = 10.0f;   /* dominant key — softmax weight ≈ 1.0 */

    int k = tropical_adaptive_k(scores.data(), n_keys, 0.95f, /*k_min=*/1, /*k_max=*/32);
    char det[64]; std::snprintf(det, sizeof(det), "K=%d (expected 1)", k);
    report("concentrated → K=1", k == 1, det);
}

/* ─── [2] Uniform distribution → K = k_max ────────────────────────────────
 * All keys have the same score. Each softmax weight = 1/n_keys.
 * With coverage=0.95 and k_max=32, need ceil(0.95 × 32) = 31 tokens.        */
static void test_uniform_gives_large_k() {
    printf("\n[2] Uniform distribution → K close to k_max\n");
    const int n_keys = 64, k_max = 32;
    std::vector<float> scores(n_keys, 0.0f);  /* all equal */

    int k = tropical_adaptive_k(scores.data(), n_keys, 0.95f, /*k_min=*/1, k_max);
    /* Expected: need 95% of 32 equally-weighted tokens → K = ceil(0.95×32) = 31 */
    bool ok = (k >= 30 && k <= k_max);
    char det[64]; std::snprintf(det, sizeof(det), "K=%d (expected 30-32)", k);
    report("uniform → K close to k_max", ok, det);
}

/* ─── [3] coverage=1.0 → result equals sparse_attention_float(K=k_max) ────
 * When coverage=1.0, adaptive K is k_max. The aggregate result must match
 * sparse_attention_float with K=k_max exactly.                               */
static void test_coverage_one_matches_fixed() {
    printf("\n[3] coverage=1.0 → adaptive equals fixed K=k_max\n");
    const int d = 16, n_keys = 32, k_max = 32;
    std::mt19937 rng(0xC0FFEE42u);
    std::normal_distribution<float> nd;

    std::vector<float> q(d), K(n_keys * d), V(n_keys * d);
    for (auto &v : q)   v = nd(rng);
    for (auto &v : K)   v = nd(rng);
    for (auto &v : V)   v = nd(rng);

    std::vector<float> out_adaptive(d, 0.f), out_fixed(d, 0.f);

    sparse_attention_float_adaptive(out_adaptive.data(), q.data(), K.data(), V.data(),
                                    n_keys, d, /*coverage=*/1.0f, /*k_min=*/1, k_max);
    sparse_attention_float(out_fixed.data(), q.data(), K.data(), V.data(),
                           n_keys, d, /*K_top=*/k_max);

    bool ok = vec_eq(out_adaptive.data(), out_fixed.data(), d, 1e-4f);
    float max_diff = 0.f;
    for (int i = 0; i < d; i++)
        max_diff = std::max(max_diff, std::fabs(out_adaptive[i] - out_fixed[i]));
    char det[64]; std::snprintf(det, sizeof(det), "max_diff=%.2e", max_diff);
    report("coverage=1.0 matches sparse_attention_float(K=k_max)", ok, det);
}

/* ─── [4] Adaptive K ≤ fixed K for any distribution, 100 iters ────────────
 * By definition, adaptive K with coverage<1 selects ≤ k_max tokens.
 * Additionally, for any concentrated distribution, adaptive K < k_max.
 * We verify: over 100 random distributions, adaptive K is always ≤ k_max,
 * and on average noticeably less than k_max (distribution is not flat).       */
static void test_adaptive_le_fixed() {
    printf("\n[4] adaptive K ≤ fixed K (100 random distributions, coverage=0.90)\n");
    const int n_keys = 128, k_max = 32;
    const int ITERS = 100;
    std::mt19937 rng(0xBEEF1234u);
    std::normal_distribution<float> nd;

    int n_ok = 0;
    float sum_k = 0.f, max_k = 0.f;
    for (int it = 0; it < ITERS; it++) {
        /* Random scores — some concentrated, some diffuse */
        std::vector<float> scores(n_keys);
        if (it % 3 == 0) {
            /* Concentrated: 1-3 dominant keys */
            for (auto &v : scores) v = -5.0f + 0.1f * nd(rng);
            int peak = rng() % n_keys;
            scores[peak] = 5.0f + nd(rng);
        } else {
            /* Random */
            for (auto &v : scores) v = nd(rng);
        }
        int k = tropical_adaptive_k(scores.data(), n_keys, 0.90f, 1, k_max);
        if (k >= 1 && k <= k_max) n_ok++;
        sum_k += (float)k;
        if (k > max_k) max_k = (float)k;
    }
    float avg_k = sum_k / ITERS;
    bool ok = (n_ok == ITERS) && (avg_k < k_max);
    char det[96];
    std::snprintf(det, sizeof(det), "%d/%d in [1,%d], avg_K=%.1f, max_K=%.0f",
                  n_ok, ITERS, k_max, avg_k, max_k);
    report("adaptive K always ≤ k_max and avg < k_max", ok, det);
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Adaptive-K Tropical Attention — Direção D\n");
    printf("═══════════════════════════════════════════════════════════\n");

    test_concentrated_gives_k1();
    test_uniform_gives_large_k();
    test_coverage_one_matches_fixed();
    test_adaptive_le_fixed();

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Resultado: %d/%d %s\n", n_pass, n_pass + n_fail,
           n_fail == 0 ? "PASSARAM ✓" : "FALHARAM ✗");
    printf("═══════════════════════════════════════════════════════════\n");
    return n_fail == 0 ? 0 : 1;
}
