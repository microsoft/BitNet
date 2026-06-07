// test_acdc.cpp — Standalone validation of L3 (ACDC) kernels
//
// Verifica:
//   [1] fwht_f32 butterfly vs reference (H_n · v)
//   [2] acdc_forward_i8 ≈ H · diag(d) · H · x
//   [3] acdc_project on small W, reconstruction error below theoretical bound
//   [4] acdc_gemv (rectangular) vs naive (small d, m)
//   [5] acdc_error returns small for exact-match diagonal
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     -Iinclude -L/usr/lib/gcc/x86_64-linux-gnu/13 \
//     src/ggml-bitnet-fwht.cpp test_acdc.cpp -o build/test_acdc

#include "ggml-bitnet-fwht.h"
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

/* Reference Hadamard transform (n = 2^k): H_n · v */
static void hadamard_ref(float * v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                float a = v[i+j];
                float b = v[i+j+len];
                v[i+j]     = a + b;
                v[i+j+len] = a - b;
            }
        }
    }
}

static void random_ternary(int8_t * v, int n, std::mt19937 & rng) {
    std::uniform_int_distribution<int> d(-1, 1);
    for (int i = 0; i < n; i++) v[i] = (int8_t)d(rng);
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

static int test_fwht_f32() {
    printf("\n[1] fwht_f32: butterfly vs reference Hadamard  (n=64)\n");
    const int n = 64;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> v(n), v_ref(n);
    for (int i = 0; i < n; i++) { v[i] = nd(rng); v_ref[i] = v[i]; }

    fwht_f32(v.data(), n);
    hadamard_ref(v_ref.data(), n);
    float diff = max_abs_diff(v.data(), v_ref.data(), n);
    printf("    max|fwht - H·v_ref| = %.2e  (expected ≈0)\n", diff);
    int ok = (diff < 1e-4f);
    printf("    %s\n", ok ? "FWHT ✓" : "FAILED ✗");
    return ok;
}

static int test_fwht_i8_to_i32() {
    printf("\n[2] fwht_i8_to_i32: sign-extend + FWHT vs reference  (n=64)\n");
    const int n = 64;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> xd(-127, 127);
    std::vector<int8_t> x(n);
    std::vector<int32_t> out(n);
    for (int i = 0; i < n; i++) x[i] = (int8_t)xd(rng);
    fwht_i8_to_i32(x.data(), out.data(), n);
    /* Reference: sign-extend then FWHT */
    std::vector<float> v_ref(n);
    for (int i = 0; i < n; i++) v_ref[i] = (float)x[i];
    hadamard_ref(v_ref.data(), n);
    float diff = 0;
    for (int i = 0; i < n; i++) diff = std::max(diff, std::fabs((float)out[i] - v_ref[i]));
    printf("    max|fwht_i8 - H·x_ref| = %.2e  (expected ≈0)\n", diff);
    int ok = (diff < 1e-3f);
    printf("    %s\n", ok ? "FWHT_I8 ✓" : "FAILED ✗");
    return ok;
}

static int test_acdc_forward() {
    printf("\n[3] acdc_forward_i8: y = H·diag(d)·H·x vs naive (n=32)\n");
    const int n = 32;
    std::mt19937 rng(13);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> xd(-100, 100);
    std::vector<int8_t> x(n);
    std::vector<float> d(n);
    for (int i = 0; i < n; i++) { x[i] = (int8_t)xd(rng); d[i] = nd(rng); }
    std::vector<float> y(n);
    acdc_forward_i8(y.data(), x.data(), d.data(), n);
    /* Reference: H · (d ⊙ (H · x)) */
    std::vector<float> hx(n);
    for (int i = 0; i < n; i++) hx[i] = (float)x[i];
    hadamard_ref(hx.data(), n);
    for (int i = 0; i < n; i++) hx[i] *= d[i];
    hadamard_ref(hx.data(), n);
    float diff = max_abs_diff(y.data(), hx.data(), n);
    printf("    max|acdc_y - ref| = %.2e  (expected ≈0)\n", diff);
    int ok = (diff < 1e-2f);
    printf("    %s\n", ok ? "ACDC_FWD ✓" : "FAILED ✗");
    return ok;
}

static int test_acdc_project_roundtrip() {
    printf("\n[4] acdc_project: closed-form diagonal for W=I  (n=8)\n");
    const int n = 8;
    std::vector<int8_t> W(n * n);
    std::vector<float>  d(n);
    /* W = I → H·I·H = H·H^T = n·I (Hadamard is self-symmetric and orthogonal
     * up to n). So diag(H·I·H) = n, and d*[k] = n / n² = 1/n.
     * The diagonal d is "the spectral signature" of W in the Hadamard basis. */
    for (int i = 0; i < n; i++) W[i*n + i] = 1;
    acdc_project(d.data(), W.data(), n);
    float target = 1.0f / (float)n;
    float err = 0;
    for (int i = 0; i < n; i++) err = std::max(err, std::fabs(d[i] - target));
    printf("    max|d[k] - 1/n| = %.2e  (target=1/n=%.4f for W=I)\n", err, target);
    int ok = (err < 1e-4f);
    printf("    %s\n", ok ? "PROJECT ✓" : "FAILED ✗");
    return ok;
}

static int test_acdc_gemv_vs_naive() {
    printf("\n[5] acdc_gemv: K=2 stacked blocks, m=4, n=8 (small rectangle)\n");
    const int n = 8, K = 2, m = 4;
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_int_distribution<int> xd(-100, 100);
    std::vector<int8_t> x(n);
    std::vector<float>  D(K * n);
    std::vector<float>  proj(m * K * n);
    for (int i = 0; i < n; i++) x[i] = (int8_t)xd(rng);
    for (int i = 0; i < K*n; i++) D[i] = nd(rng);
    /* Identity projection: proj[i*Kn + i] = 1.0 (truncate to first m of K*n) */
    for (int i = 0; i < (int)proj.size(); i++) proj[i] = 0.0f;
    for (int i = 0; i < m; i++) proj[i * (K*n) + i] = 1.0f;
    std::vector<float> y(m);
    acdc_gemv(y.data(), x.data(), D.data(), proj.data(), m, n, K);
    /* Reference: for each k=0..K-1, compute h_k = H·(D[k] ⊙ H·x); then y[i] = proj·h. */
    std::vector<float> h(K * n);
    for (int k = 0; k < K; k++) {
        std::vector<float> hx(n);
        for (int i = 0; i < n; i++) hx[i] = (float)x[i];
        hadamard_ref(hx.data(), n);
        for (int i = 0; i < n; i++) hx[i] *= D[k*n + i];
        hadamard_ref(hx.data(), n);
        for (int i = 0; i < n; i++) h[k*n + i] = hx[i];
    }
    std::vector<float> y_ref(m, 0.0f);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < K*n; j++) y_ref[i] += proj[i*(K*n) + j] * h[j];
    float diff = max_abs_diff(y.data(), y_ref.data(), m);
    printf("    max|gemv_y - ref| = %.2e  (expected ≈0)\n", diff);
    int ok = (diff < 1e-2f);
    printf("    %s\n", ok ? "GEMV ✓" : "FAILED ✗");
    return ok;
}

/* AVX2 in-register prefix correctness: h=1,2,4 fused stages.
 * Tests n=8 (only the 3 in-register stages, no large-stage loop) and
 * n=16, n=4096 (in-register prefix + large stages together).
 * If butterfly_f32_avx2_prefix8 has wrong sign or permutation this detects it. */
static int test_fwht_avx2_prefix() {
    printf("\n[6] fwht_avx2_prefix: in-register h=1,2,4 stages (n=8,16,4096)\n");
    std::mt19937 rng(123);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int all_ok = 1;
    const int sizes[] = {8, 16, 32, 4096};
    for (int n : sizes) {
        std::vector<float> v(n), v_ref(n);
        for (int i = 0; i < n; i++) { v[i] = nd(rng); v_ref[i] = v[i]; }
        fwht_f32(v.data(), n);
        hadamard_ref(v_ref.data(), n);
        float diff = max_abs_diff(v.data(), v_ref.data(), n);
        int ok = (diff < 1e-3f * (float)n);
        printf("    n=%-5d  max|fwht - ref| = %.2e  %s\n", n, diff,
               ok ? "✓" : "FAILED ✗");
        if (!ok) all_ok = 0;
    }
    return all_ok;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  ACDC (Level 3) — Standalone C++ validation\n");
    printf("═══════════════════════════════════════════════════════════\n");
    int n_pass = 0, n_total = 0;
    struct { const char * name; int (*fn)(); } tests[] = {
        { "fwht_f32",         test_fwht_f32              },
        { "fwht_i8",          test_fwht_i8_to_i32        },
        { "acdc_forward",     test_acdc_forward          },
        { "acdc_project",     test_acdc_project_roundtrip },
        { "acdc_gemv",        test_acdc_gemv_vs_naive    },
        { "fwht_avx2_prefix", test_fwht_avx2_prefix      },
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
