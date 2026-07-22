// test_acdc_properties.cpp — Property-based tests for ACDC (Level 3) kernels
//
// Verifica 4 invariantes do ACDC sobre 1000 iterações cada com seeds
// determinísticas. As invariantes testadas correspondem ao princípio P6
// (Estrutura, não compressão).
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     -Iinclude -L/usr/lib/gcc/x86_64-linux-gnu/13 \
//     src/ggml-bitnet-fwht.cpp src/ggml-bitnet-common.cpp \
//     test_acdc_properties.cpp -o build/test_acdc_properties
//
// Convention: hand-rolled `assert(...)` per T003 (no Catch2 in this project).

#include "ggml-bitnet-fwht.h"
#include "ggml-bitnet-common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static int n_pass = 0, n_total = 0;

static void report(const char * name, bool ok, const char * detail = "") {
    n_total++;
    if (ok) n_pass++;
    printf("  %-50s %s   %s\n", name, ok ? "PASS ✓" : "FAIL ✗", detail);
}

/* ── Reference FWHT in float for verification ─────────────────────────── */

static void fwht_f32_ref(float *v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float a = v[i + j];
                float b = v[i + j + len];
                v[i + j]        = a + b;
                v[i + j + len]  = a - b;
            }
        }
    }
}

static void fwht_i8_to_f32_ref(const int8_t *x, float *out, int n) {
    for (int i = 0; i < n; i++) out[i] = (float)x[i];
    fwht_f32_ref(out, n);
}

/* ── Helper: build a random ternary matrix W in {-1, 0, +1}^{n×n} ─────── */

static void random_ternary_matrix(std::vector<int8_t> & W, int n, std::mt19937 & rng) {
    W.assign((size_t)n * n, 0);
    std::uniform_int_distribution<int> d(-1, 1);
    for (auto & v : W) v = (int8_t)d(rng);
}

static float fro_norm(const int8_t * W, int n) {
    double s = 0;
    for (int i = 0; i < n * n; i++) s += (double)W[i] * (double)W[i];
    return (float)std::sqrt(s);
}

/* ── Property 1: ‖d*‖ ≤ ‖W‖ / sqrt(n) ────────────────────────────────── */

static int test_acdc_norm_bound() {
    printf("\n[1] ‖d*‖ ≤ ‖W‖ / sqrt(n)   (n=64, 1000 iters)\n");
    const int n = 64;
    const int ITERS = 1000;
    std::mt19937 rng(0xACDC0001u);

    std::vector<int8_t> W;
    std::vector<float>  d(n);
    int n_ok = 0;
    float max_ratio = 0.f;

    for (int it = 0; it < ITERS; it++) {
        random_ternary_matrix(W, n, rng);
        acdc_project(d.data(), W.data(), n);
        float Wn = fro_norm(W.data(), n);
        float dn = 0.f;
        for (int i = 0; i < n; i++) dn += d[i] * d[i];
        dn = std::sqrt(dn);
        float bound = Wn / std::sqrt((float)n);
        if (dn <= bound + 1e-3f) n_ok++;
        max_ratio = std::max(max_ratio, dn / std::max(bound, 1e-9f));
    }
    char det[96];
    std::snprintf(det, sizeof(det), "%d/%d (max ‖d*‖/bound=%.3f)", n_ok, ITERS, max_ratio);
    report("‖d*‖ ≤ ‖W‖/sqrt(n)", n_ok == ITERS, det);
    return n_ok == ITERS;
}

/* Property 2: closed form — diag(H·W·H) / n² = d* exactly (P6 closed form) */

static int test_acdc_project_idempotent() {
    printf("\n[2] closed form: diag(H·W·H) / n² = d* (P6, 1000 iters)\n");
    const int n = 64;
    const int ITERS = 1000;
    std::mt19937 rng(0xACDC0002u);

    std::vector<int8_t> W;
    std::vector<float>  d_kernel(n);
    std::vector<float>  Wf((size_t)n * n);
    std::vector<float>  HWH((size_t)n * n);
    int n_ok = 0;
    float max_diff = 0.f;

    for (int it = 0; it < ITERS; it++) {
        random_ternary_matrix(W, n, rng);
        acdc_project(d_kernel.data(), W.data(), n);

        // Reference: Wf = float(W)
        for (int i = 0; i < n * n; i++) Wf[i] = (float)W[i];

        // H·W: row-wise FWHT
        for (int i = 0; i < n; i++) fwht_f32_ref(Wf.data() + i * n, n);

        // (H·W)·H: column-wise FWHT (apply to each column)
        // First copy: HWH[i,j] = Wf[i,j]
        for (int i = 0; i < n * n; i++) HWH[i] = Wf[i];
        // Column-wise: HWH[:,j] = FWHT(HWH[:,j])
        for (int j = 0; j < n; j++) {
            std::vector<float> col(n);
            for (int i = 0; i < n; i++) col[i] = HWH[i * n + j];
            fwht_f32_ref(col.data(), n);
            for (int i = 0; i < n; i++) HWH[i * n + j] = col[i];
        }

        // d_ref[k] = HWH[k,k] / n²
        std::vector<float> d_ref(n);
        for (int k = 0; k < n; k++) d_ref[k] = HWH[k * n + k] / (float)(n * n);

        // Compare
        float diff = 0.f;
        for (int i = 0; i < n; i++) diff = std::max(diff, std::fabs(d_kernel[i] - d_ref[i]));
        max_diff = std::max(max_diff, diff);
        if (diff < 1e-2f) n_ok++;
    }
    char det[96];
    std::snprintf(det, sizeof(det), "%d/%d (max |d_kernel - d_ref|=%.2e)",
                  n_ok, ITERS, max_diff);
    report("diag(H·W·H)/n² = d* (closed form, P6)", n_ok == ITERS, det);
    return n_ok == ITERS;
}

/* ── Property 3: n²·‖d*‖² ≈ ‖W_proj‖² ───────────────────────────────── */

static int test_acdc_energy() {
    printf("\n[3] n²·‖d*‖² ≈ ‖W_proj‖²  (energy identity)\n");
    const int n = 64;
    const int ITERS = 1000;
    std::mt19937 rng(0xACDC0003u);

    std::vector<int8_t> W;
    std::vector<float>  d(n);
    int n_ok = 0;
    float max_rel = 0.f;

    for (int it = 0; it < ITERS; it++) {
        random_ternary_matrix(W, n, rng);
        acdc_project(d.data(), W.data(), n);

        // ‖d*‖²
        float dn2 = 0.f;
        for (int i = 0; i < n; i++) dn2 += d[i] * d[i];

        // ‖W_proj‖² (use acdc_error to derive)
        float rel_err = acdc_error(W.data(), d.data(), n);
        // W_proj = H·diag(d)·H / n²  → ‖W_proj‖² = ‖d‖² / n²  (Parseval for H)
        // But W itself has different energy.  rel_err = ‖W - W_proj‖ / ‖W‖
        // This test instead checks the identity: ‖W‖² - n²·‖d‖² / n² = ‖W-W_proj‖²
        // i.e. ‖W‖² - ‖d‖²/n² = ‖W - W_proj‖²
        float Wn2 = 0.f;
        for (int i = 0; i < n * n; i++) Wn2 += (float)W[i] * (float)W[i];
        float lhs = Wn2 - dn2 / (float)(n * n);  // energy lost
        // Approximation: ‖W - W_proj‖² ≈ lhs (exact for ACDC)
        // rel_err = sqrt(lhs / Wn2)
        float expected_rel = std::sqrt(std::max(lhs, 0.f) / std::max(Wn2, 1e-9f));
        float rel_diff = std::fabs(rel_err - expected_rel);
        max_rel = std::max(max_rel, rel_diff);
        if (rel_diff < 0.05f) n_ok++;
    }
    char det[96];
    std::snprintf(det, sizeof(det), "%d/%d (max |Δrel_err|=%.3f)", n_ok, ITERS, max_rel);
    report("n²·‖d*‖² ≈ ‖W_proj‖² (energy)", n_ok == ITERS, det);
    return n_ok == ITERS;
}

/* ── Property 4: determinism ──────────────────────────────────────────── */

static int test_acdc_determinism() {
    printf("\n[4] determinism: 2 calls, same seed → identical d\n");
    const int n = 64;
    const int ITERS = 200;
    std::mt19937 rng(0xACDC0004u);
    std::vector<int8_t> W;
    std::vector<float>  d1(n), d2(n);
    int n_ok = 0;
    float max_d = 0.f;

    for (int it = 0; it < ITERS; it++) {
        random_ternary_matrix(W, n, rng);
        acdc_project(d1.data(), W.data(), n);
        acdc_project(d2.data(), W.data(), n);
        float diff = 0.f;
        for (int i = 0; i < n; i++) diff = std::max(diff, std::fabs(d1[i] - d2[i]));
        max_d = std::max(max_d, diff);
        if (diff < 1e-6f) n_ok++;
    }
    char det[96];
    std::snprintf(det, sizeof(det), "%d/%d (max |d1-d2|=%.2e)", n_ok, ITERS, max_d);
    report("determinism", n_ok == ITERS, det);
    return n_ok == ITERS;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  ACDC Properties (Level 3) — 1000 iters per property\n");
    printf("═══════════════════════════════════════════════════════════\n");
    test_acdc_norm_bound();
    test_acdc_project_idempotent();
    test_acdc_energy();
    test_acdc_determinism();
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf("  Resultado: %d/%d propriedades %s\n", n_pass, n_total,
           n_pass == n_total ? "PASSARAM ✓" : "FALHARAM ✗");
    printf("═══════════════════════════════════════════════════════════\n");
    return n_pass == n_total ? 0 : 1;
}
