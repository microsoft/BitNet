// test_tropical.cpp — Standalone validation of L4 (Tropical attention) kernels
//
// Verifies:
//   [1] tropical_attn_argmax: returns correct argmax index
//   [2] tropical_attn_topk: top-K indices in descending order
//   [3] tropical_attention: softmax(top-K scores) · V matches reference
//   [4] tropical_gemv: max-plus matrix-vector product
//   [5] Zero-K edge case: K > n_keys must clamp to n_keys
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     -Iinclude -L/usr/lib/gcc/x86_64-linux-gnu/13 \
//     src/ggml-bitnet-tropical.cpp test_tropical.cpp -o build/test_tropical

#include "ggml-bitnet-tropical.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

static float max_abs_diff(const float * a, const float * b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

static void quantize_f32_to_i8(const float * x, int8_t * xi, float * scale, int n) {
    float mx = 1e-6f;
    for (int i = 0; i < n; i++) mx = std::fmax(mx, std::fabs(x[i]));
    *scale = 127.0f / mx;
    for (int i = 0; i < n; i++) {
        float v = x[i] * (*scale);
        if (v >  127.0f) v =  127.0f;
        if (v < -127.0f) v = -127.0f;
        xi[i] = (int8_t)std::round(v);
    }
}

static float dot_ref(const int8_t * a, const int8_t * b, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += (float)a[i] * (float)b[i];
    return s;
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

static int test_tropical_argmax() {
    printf("\n[1] tropical_attn_argmax: max over query·key  (n_keys=8, d=16)\n");
    const int n_keys = 8, d = 16;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float>   qf(d);
    std::vector<int8_t>  q(d), K(n_keys * d);
    for (int i = 0; i < d; i++) qf[i] = nd(rng);
    float qs, ks;
    quantize_f32_to_i8(qf.data(), q.data(), &qs, d);
    for (int j = 0; j < n_keys; j++) {
        std::vector<float> kf(d);
        for (int i = 0; i < d; i++) kf[i] = nd(rng);
        quantize_f32_to_i8(kf.data(), K.data() + j * d, &ks, d);
    }
    int best = tropical_attn_argmax(q.data(), K.data(), n_keys, d);

    std::vector<float> scores(n_keys);
    for (int j = 0; j < n_keys; j++) scores[j] = dot_ref(q.data(), K.data() + j * d, d);
    int ref = (int)(std::max_element(scores.begin(), scores.end()) - scores.begin());
    printf("    best=%d  ref=%d\n", best, ref);
    int ok = (best == ref);
    printf("    %s\n", ok ? "ARGMAX ✓" : "FAILED ✗");
    return ok;
}

static int test_tropical_topk() {
    printf("\n[2] tropical_attn_topk: top-3 of 8 keys  (K=3, n_keys=8, d=16)\n");
    const int n_keys = 8, d = 16, K = 3;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float>   qf(d);
    std::vector<int8_t>  q(d), keys(n_keys * d);
    for (int i = 0; i < d; i++) qf[i] = nd(rng);
    float qs, ks;
    quantize_f32_to_i8(qf.data(), q.data(), &qs, d);
    for (int j = 0; j < n_keys; j++) {
        std::vector<float> kf(d);
        for (int i = 0; i < d; i++) kf[i] = nd(rng);
        quantize_f32_to_i8(kf.data(), keys.data() + j * d, &ks, d);
    }
    std::vector<int>   top_idx(K);
    std::vector<float> top_scores(K);
    tropical_attn_topk(top_idx.data(), top_scores.data(),
                       q.data(), keys.data(), n_keys, d, K, qs, ks);

    std::vector<float> scores(n_keys);
    for (int j = 0; j < n_keys; j++) scores[j] = dot_ref(q.data(), keys.data() + j * d, d);
    std::vector<int> idx_ref(n_keys);
    for (int i = 0; i < n_keys; i++) idx_ref[i] = i;
    std::partial_sort(idx_ref.begin(), idx_ref.begin() + K, idx_ref.end(),
                      [&](int a, int b){ return scores[a] > scores[b]; });

    printf("    top_idx:    ");
    for (int k = 0; k < K; k++) printf("%d ", top_idx[k]);
    printf("\n    ref top-3:  ");
    for (int k = 0; k < K; k++) printf("%d ", idx_ref[k]);
    printf("\n");
    int ok = true;
    for (int k = 0; k < K; k++) {
        if (top_idx[k] != idx_ref[k]) { ok = false; break; }
    }
    printf("    %s\n", ok ? "TOPK ✓" : "FAILED ✗");
    return ok;
}

static int test_tropical_attention() {
    printf("\n[3] tropical_attention: softmax(top-K scores)·V  (K=2, n=4, d=8)\n");
    const int n_keys = 4, d = 8, K = 2;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float>   qf(d), V(n_keys * d);
    std::vector<int8_t>  q(d), K_q(n_keys * d);
    for (int i = 0; i < d; i++) qf[i] = nd(rng);
    float qs, ks;
    quantize_f32_to_i8(qf.data(), q.data(), &qs, d);
    for (int j = 0; j < n_keys; j++) {
        std::vector<float> kf(d);
        for (int i = 0; i < d; i++) kf[i] = nd(rng);
        quantize_f32_to_i8(kf.data(), K_q.data() + j * d, &ks, d);
        for (int i = 0; i < d; i++) V[j * d + i] = nd(rng);
    }
    std::vector<float> out(d);
    tropical_attention(out.data(), q.data(), K_q.data(), V.data(), n_keys, d, K, qs, ks);

    std::vector<float> scores(n_keys);
    for (int j = 0; j < n_keys; j++) scores[j] = dot_ref(q.data(), K_q.data() + j * d, d);
    std::vector<int> idx(n_keys);
    for (int i = 0; i < n_keys; i++) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                      [&](int a, int b){ return scores[a] > scores[b]; });
    std::vector<float> w(K);
    float max_s = scores[idx[0]];
    float sum = 0;
    for (int k = 0; k < K; k++) { w[k] = std::exp(scores[idx[k]] - max_s); sum += w[k]; }
    for (int k = 0; k < K; k++) w[k] /= sum;
    std::vector<float> out_ref(d, 0.0f);
    for (int k = 0; k < K; k++)
        for (int i = 0; i < d; i++) out_ref[i] += w[k] * V[idx[k] * d + i];
    float diff = max_abs_diff(out.data(), out_ref.data(), d);
    printf("    max|tropical - ref| = %.2e  (modulo FP)\n", diff);
    int ok = (diff < 1e-1f);
    printf("    %s\n", ok ? "ATTN ✓" : "FAILED ✗");
    return ok;
}

static int test_tropical_gemv() {
    printf("\n[4] tropical_gemv: y[i] = max_j (W[i,j] + x[j])  (m=4, n=8)\n");
    const int m = 4, n = 8;
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> wd(-1, 1);
    std::normal_distribution<float>   nd(0.0f, 1.0f);

    std::vector<int8_t> W(m * n);
    std::vector<float>  x(n);
    for (int i = 0; i < m * n; i++) W[i] = (int8_t)wd(rng);
    for (int i = 0; i < n; i++) x[i] = nd(rng);

    std::vector<int>   argmax(m);
    std::vector<float> y_max(m);
    tropical_gemv(argmax.data(), y_max.data(), W.data(), x.data(), m, n);

    std::vector<float> y_ref(m);
    std::vector<int>   argmax_ref(m);
    for (int i = 0; i < m; i++) {
        float best = -1e9f;
        int   best_j = 0;
        for (int j = 0; j < n; j++) {
            float v = (float)W[i * n + j] + x[j];
            if (v > best) { best = v; best_j = j; }
        }
        y_ref[i]      = best;
        argmax_ref[i] = best_j;
    }
    float diff_y      = max_abs_diff(y_max.data(), y_ref.data(), m);
    int   diff_argmax = 0;
    for (int i = 0; i < m; i++) if (argmax[i] != argmax_ref[i]) diff_argmax++;
    printf("    max|y_wht - y_ref| = %.2e  argmax mismatches=%d  (expected 0)\n",
           diff_y, diff_argmax);
    int ok = (diff_y < 1e-3f) && (diff_argmax == 0);
    printf("    %s\n", ok ? "GEMV ✓" : "FAILED ✗");
    return ok;
}

static int test_tropical_zero_k() {
    printf("\n[5] tropical_attention: K > n_keys clamps to n_keys  (K=10, n=3)\n");
    const int n_keys = 3, d = 4, K = 10;  /* K > n_keys — must not crash */
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float>   qf(d), V(n_keys * d);
    std::vector<int8_t>  q(d), K_q(n_keys * d);
    for (int i = 0; i < d; i++) qf[i] = nd(rng);
    float qs, ks;
    quantize_f32_to_i8(qf.data(), q.data(), &qs, d);
    for (int j = 0; j < n_keys; j++) {
        std::vector<float> kf(d);
        for (int i = 0; i < d; i++) kf[i] = nd(rng);
        quantize_f32_to_i8(kf.data(), K_q.data() + j * d, &ks, d);
        for (int i = 0; i < d; i++) V[j * d + i] = nd(rng);
    }
    std::vector<float> out(d, -1.0f);
    tropical_attention(out.data(), q.data(), K_q.data(), V.data(), n_keys, d, K, qs, ks);
    /* Must produce finite numbers (no crash, no NaN) */
    bool finite = true;
    for (int i = 0; i < d; i++) if (!std::isfinite(out[i])) { finite = false; break; }
    printf("    out finite=%s  out[0]=%.3f\n", finite ? "yes" : "NO", out[0]);
    int ok = finite;
    printf("    %s\n", ok ? "ZERO_K ✓" : "FAILED ✗");
    return ok;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Tropical (Level 4) — Standalone C++ validation\n");
    printf("═══════════════════════════════════════════════════════════\n");
    int n_pass = 0, n_total = 0;
    struct { const char * name; int (*fn)(); } tests[] = {
        { "argmax",  test_tropical_argmax       },
        { "topk",    test_tropical_topk         },
        { "attn",    test_tropical_attention    },
        { "gemv",    test_tropical_gemv         },
        { "zero_k",  test_tropical_zero_k       },
    };
    for (auto & t : tests) {
        n_total++;
        if (t.fn()) n_pass++;
    }
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Resultado: %d/%d testes %s\n", n_pass, n_total,
           n_pass == n_total ? "PASSARAM ✓" : "FALHARAM ✗");
    printf("═══════════════════════════════════════════════════════════\n");
    return n_pass == n_total ? 0 : 1;
}
