/*
 * test_acdc_rect.cpp — Unit tests for Fase II rectangular ACDC kernel.
 *
 * Tests acdc_forward_rect_f32 and acdc_forward_rect_i8.  No model needed;
 * runtime < 5ms.  Follow hand-rolled assert convention (see tests/CMakeLists.txt
 * header note: no Catch2, no heavy deps).
 *
 * Gated by BITNET_ENABLE_ACDC_RECT=ON (D2 gate) in tests/CMakeLists.txt.
 */

#include "ggml-bitnet-fwht.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cfloat>
#include <vector>

/* ─── Helpers ───────────────────────────────────────────────────────────── */

static int g_fails = 0;

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL [line %d]: %s\n", __LINE__, (msg)); \
        g_fails++; \
    } else { \
        fprintf(stderr, "ok: %s\n", (msg)); \
    } \
} while (0)

#define EXPECT_NEAR(a, b, tol, msg) do { \
    float _a = (float)(a), _b = (float)(b), _t = (float)(tol); \
    if (fabsf(_a - _b) > _t * fmaxf(1.0f, fabsf(_b)) + _t) { \
        fprintf(stderr, "FAIL [line %d]: %s  (got %.6g, expected %.6g, tol %.2g)\n", \
                __LINE__, (msg), (double)_a, (double)_b, (double)_t); \
        g_fails++; \
    } else { \
        fprintf(stderr, "ok: %s\n", (msg)); \
    } \
} while (0)

/* Max absolute difference across a vector */
static float vec_max_diff(const float * a, const float * b, int n) {
    float d = 0.0f;
    for (int i = 0; i < n; i++) d = fmaxf(d, fabsf(a[i] - b[i]));
    return d;
}

static bool all_finite(const float * v, int n) {
    for (int i = 0; i < n; i++) if (!std::isfinite(v[i])) return false;
    return true;
}

/* ─── Test 1: square case — identity diagonal ────────────────────────────
 *
 * For m = n = P, d[i] = 1/P gives y = x (ACDC identity).
 *
 * Proof: H_P · (1/P · H_P · x) = (H_P · H_P / P) · x = I · x = x
 * ─────────────────────────────────────────────────────────────────────── */
static void test_square_identity() {
    fprintf(stderr, "\n--- test_square_identity ---\n");
    const int N = 16;
    const float inv_N = 1.0f / (float)N;

    std::vector<float> x(N), y(N), d(N, inv_N);
    for (int i = 0; i < N; i++) x[i] = (float)(i - N/2);

    acdc_forward_rect_f32(y.data(), N, x.data(), N, d.data());

    float diff = vec_max_diff(x.data(), y.data(), N);
    EXPECT_NEAR(diff, 0.0f, 1e-4f, "square identity: y ≈ x");
}

/* ─── Test 2: upscale — m > n ────────────────────────────────────────────
 *
 * m=32, n=16, P=32, d[i] = 1/32.
 * Input x[16], zero-padded to [x | 0..0_16].
 * Identity d: y_P = I · x_pad = [x | 0..0_16], output y[32] = x_pad.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_upscale() {
    fprintf(stderr, "\n--- test_upscale ---\n");
    const int M = 32, N = 16, P = 32;
    const float inv_P = 1.0f / (float)P;

    std::vector<float> x(N), y(M), d(P, inv_P);
    for (int i = 0; i < N; i++) x[i] = (float)(i + 1);

    acdc_forward_rect_f32(y.data(), M, x.data(), N, d.data());

    EXPECT(all_finite(y.data(), M), "upscale: all outputs finite");

    float diff_low = vec_max_diff(x.data(), y.data(), N);
    EXPECT_NEAR(diff_low, 0.0f, 1e-4f, "upscale: first n elements ≈ x");

    float max_high = 0.0f;
    for (int i = N; i < M; i++) max_high = fmaxf(max_high, fabsf(y[i]));
    EXPECT_NEAR(max_high, 0.0f, 1e-4f, "upscale: elements [n,m) ≈ 0");
}

/* ─── Test 3: downscale — m < n ──────────────────────────────────────────
 *
 * m=16, n=32, P=32, d[i] = 1/32.
 * y = first 16 elements of I · x = x[0..15].
 * ─────────────────────────────────────────────────────────────────────── */
static void test_downscale() {
    fprintf(stderr, "\n--- test_downscale ---\n");
    const int M = 16, N = 32, P = 32;
    const float inv_P = 1.0f / (float)P;

    std::vector<float> x(N), y(M), d(P, inv_P);
    for (int i = 0; i < N; i++) x[i] = (float)(i - N/2);

    acdc_forward_rect_f32(y.data(), M, x.data(), N, d.data());

    EXPECT(all_finite(y.data(), M), "downscale: all outputs finite");

    float diff = vec_max_diff(x.data(), y.data(), M);
    EXPECT_NEAR(diff, 0.0f, 1e-4f, "downscale: y[0..m-1] ≈ x[0..m-1]");
}

/* ─── Test 4: zero diagonal — output must be exactly zero ────────────────
 *
 * d = 0 → z = 0 → H·0 = 0 → y = 0.  No floating-point cancellation.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_zero_diagonal() {
    fprintf(stderr, "\n--- test_zero_diagonal ---\n");
    const int M = 24, N = 8, P = 32;

    std::vector<float> x(N, 1.0f), y(M, 99.0f), d(P, 0.0f);

    acdc_forward_rect_f32(y.data(), M, x.data(), N, d.data());

    float mx = 0.0f;
    for (int i = 0; i < M; i++) mx = fmaxf(mx, fabsf(y[i]));
    EXPECT_NEAR(mx, 0.0f, 1e-10f, "zero diagonal: y = 0");
}

/* ─── Test 5: linearity ──────────────────────────────────────────────────
 *
 * f(a·x + b·z) = a·f(x) + b·f(z) for random d.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_linearity() {
    fprintf(stderr, "\n--- test_linearity ---\n");
    const int M = 16, N = 8, P = 16;

    std::vector<float> x(N), z(N), xpz(N), d(P);
    std::vector<float> fx(M), fz(M), fxpz(M), expected(M);

    unsigned seed = 0xcafebabe;
    auto lcg = [&]() -> float {
        seed = seed * 1664525u + 1013904223u;
        return (float)((int)(seed >> 8) & 0xffffff) / (float)0xffffff - 0.5f;
    };

    for (int i = 0; i < N; i++) { x[i] = lcg(); z[i] = lcg(); }
    for (int i = 0; i < P; i++) d[i] = lcg() * 0.1f;

    const float a = 1.3f, b = -0.7f;
    for (int i = 0; i < N; i++) xpz[i] = a * x[i] + b * z[i];

    acdc_forward_rect_f32(fx.data(),   M, x.data(),   N, d.data());
    acdc_forward_rect_f32(fz.data(),   M, z.data(),   N, d.data());
    acdc_forward_rect_f32(fxpz.data(), M, xpz.data(), N, d.data());

    for (int i = 0; i < M; i++) expected[i] = a * fx[i] + b * fz[i];

    float diff = vec_max_diff(fxpz.data(), expected.data(), M);
    EXPECT_NEAR(diff, 0.0f, 5e-5f, "linearity: f(ax+bz) = a*f(x) + b*f(z)");
}

/* ─── Test 6: i8 vs f32 consistency ─────────────────────────────────────
 *
 * For integer-valued inputs that quantize exactly to int8, the i8 and f32
 * versions should give the same result up to quantization scale.
 *
 * Input: x[i] = i (small integers).
 * After quant: x_i8[i] = round(x[i] * 127 / max|x|) = round(x[i] * 127 / n)
 * The i8 path output is scaled by (max|x| / 127); compare after rescaling.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_i8_vs_f32() {
    fprintf(stderr, "\n--- test_i8_vs_f32 ---\n");
    const int M = 16, N = 8, P = 16;
    const float inv_P = 1.0f / (float)P;

    /* Use identity diagonal so f32 path gives y = x exactly */
    std::vector<float> d(P, inv_P);
    std::vector<float> x_f(N), y_f32(M);
    std::vector<int8_t> x_i8(N);
    std::vector<float> y_i8_f(M);

    /* Small integer inputs for exact int8 quantization */
    for (int i = 0; i < N; i++) x_f[i] = (float)(i);

    /* Float reference (identity) */
    acdc_forward_rect_f32(y_f32.data(), M, x_f.data(), N, d.data());

    /* Build int8 version: quantize with scale s = 127 / max|x| */
    float mx = 1e-6f;
    for (int i = 0; i < N; i++) mx = fmaxf(mx, fabsf(x_f[i]));
    float s = 127.0f / mx;
    for (int i = 0; i < N; i++) {
        float v = x_f[i] * s;
        if (v >  127.0f) v =  127.0f;
        if (v < -128.0f) v = -128.0f;
        x_i8[i] = (int8_t)(int)v;
    }

    acdc_forward_rect_i8(y_i8_f.data(), M, x_i8.data(), N, d.data());

    /* i8 output is scaled by s; rescale back */
    float inv_s = 1.0f / s;
    for (int i = 0; i < M; i++) y_i8_f[i] *= inv_s;

    EXPECT(all_finite(y_i8_f.data(), M), "i8 consistency: all finite");

    float diff = vec_max_diff(y_f32.data(), y_i8_f.data(), M);
    /* Quantization error: 1 LSB = 1/127 ≈ 0.8% per element.
     * After two FWHT passes accumulated over P=16 elements: tol = 5e-2. */
    EXPECT_NEAR(diff, 0.0f, 5e-2f, "i8 vs f32: max diff < 5e-2 (quant tol)");
}

/* ─── Test 7: Falcon3-10B FFN dimensions — no crash, finite output ───────
 *
 * gate_proj: m=23040, n=3072.  d = all zeros → y = all zeros.
 * This exercises the P=32768 code path under real model dimensions.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_falcon_ffn_dims() {
    fprintf(stderr, "\n--- test_falcon_ffn_dims ---\n");
    const int M = 23040, N = 3072;
    const int P = fwht_next_pow2(M > N ? M : N);   /* 32768 */

    std::vector<float> x(N, 1.0f), y(M, 0.0f), d(P, 0.0f);

    acdc_forward_rect_f32(y.data(), M, x.data(), N, d.data());

    EXPECT(P == 32768, "falcon dims: P = 32768");
    EXPECT(all_finite(y.data(), M), "falcon dims: all outputs finite");

    float mx = 0.0f;
    for (int i = 0; i < M; i++) mx = fmaxf(mx, fabsf(y[i]));
    EXPECT_NEAR(mx, 0.0f, 1e-10f, "falcon dims: d=0 → y=0");
}

/* ─── Test 8: down_proj reverse (m=3072, n=23040) ────────────────────────*/
static void test_falcon_down_proj_dims() {
    fprintf(stderr, "\n--- test_falcon_down_proj_dims ---\n");
    const int M = 3072, N = 23040;
    const int P = fwht_next_pow2(M > N ? M : N);   /* 32768 */

    std::vector<float> x(N, 0.5f), y(M, 0.0f), d(P, 0.0f);

    acdc_forward_rect_f32(y.data(), M, x.data(), N, d.data());

    EXPECT(all_finite(y.data(), M), "down_proj dims: all outputs finite");

    float mx = 0.0f;
    for (int i = 0; i < M; i++) mx = fmaxf(mx, fabsf(y[i]));
    EXPECT_NEAR(mx, 0.0f, 1e-10f, "down_proj dims: d=0 → y=0");
}

/* ─── Test 9: acdc_project_rect — square identity diagonal ──────────────
 *
 * For W = I_n (square identity, n=m=P), the XOR-convolution gives:
 *   C[s] = Σ_i δ(i XOR i, s) = Σ_i δ(0, s) = n·δ(s,0)
 *   FWHT([n, 0, ..., 0]) = [n, n, ..., n]
 *   d*[k] = n / n² = 1/n  for all k.
 * ─────────────────────────────────────────────────────────────────────── */
static void test_project_rect_square_identity() {
    fprintf(stderr, "\n--- test_project_rect_square_identity ---\n");
    const int N = 16;   /* square: m = n = P = 16 */

    std::vector<int8_t> W(N * N, 0);
    for (int i = 0; i < N; i++) W[i * N + i] = 1;   /* identity */

    std::vector<float> d(N, 0.0f);
    acdc_project_rect(d.data(), W.data(), N, N);

    const float expected = 1.0f / (float)N;
    float max_err = 0.0f;
    for (int k = 0; k < N; k++)
        max_err = fmaxf(max_err, fabsf(d[k] - expected));

    EXPECT_NEAR(max_err, 0.0f, 1e-5f, "project_rect square I: d[k] = 1/n");
}

/* ─── Test 10: acdc_project_rect — non-trivial W, XOR-conv by hand ──────
 *
 * W = 2×2 matrix embedded in m=4, n=2 (P=4):
 *   W = [[1, 0],
 *        [0, 1]]
 * C[0^0] += 1, C[1^1] += 1 → C = [2, 0, 0, 0]
 * FWHT([2,0,0,0]) = [2, 2, 2, 2]
 * d* = [2/16, 2/16, 2/16, 2/16] = [1/8, 1/8, 1/8, 1/8]
 * ─────────────────────────────────────────────────────────────────────── */
static void test_project_rect_known() {
    fprintf(stderr, "\n--- test_project_rect_known ---\n");
    const int M = 4, N = 2, P = 4;

    std::vector<int8_t> W(M * N, 0);
    W[0 * N + 0] = 1;   /* W[0,0] = 1 */
    W[1 * N + 1] = 1;   /* W[1,1] = 1 */

    std::vector<float> d(P, 0.0f);
    acdc_project_rect(d.data(), W.data(), M, N);

    const float expected = 2.0f / (float)(P * P);   /* 2/16 = 0.125 */
    float max_err = 0.0f;
    for (int k = 0; k < P; k++)
        max_err = fmaxf(max_err, fabsf(d[k] - expected));

    EXPECT_NEAR(max_err, 0.0f, 1e-5f, "project_rect known: d[k] = 1/8");
}

/* ─── Test 11: acdc_project_rect — sparse W, single nonzero ─────────────
 *
 * W[2,1] = 1 (only entry), m=4, n=4, P=4.
 * C[2 XOR 1] = C[3] = 1; rest zero.
 * FWHT of e_3 for H_4:
 *   H_4 = [[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]]
 *   H_4·e_3 = [1,-1,-1,1]
 * d* = [1,-1,-1,1] / 16
 * ─────────────────────────────────────────────────────────────────────── */
static void test_project_rect_sparse() {
    fprintf(stderr, "\n--- test_project_rect_sparse ---\n");
    const int M = 4, N = 4, P = 4;

    std::vector<int8_t> W(M * N, 0);
    W[2 * N + 1] = 1;   /* W[2,1] = 1 */

    std::vector<float> d(P, 0.0f);
    acdc_project_rect(d.data(), W.data(), M, N);

    /* Expected: H_4 · e_3 / 16 = [1,-1,-1,1] / 16 */
    float expected[4] = { 1.0f/16, -1.0f/16, -1.0f/16, 1.0f/16 };
    float max_err = 0.0f;
    for (int k = 0; k < P; k++)
        max_err = fmaxf(max_err, fabsf(d[k] - expected[k]));

    EXPECT_NEAR(max_err, 0.0f, 1e-5f, "project_rect sparse: d matches H_4·e_3/16");
}

/* ─── Test 12: acdc_project_rect — forward-project round-trip ───────────
 *
 * For square W=I (n=16), d* = 1/n all elements.
 * acdc_forward_rect_f32 with d=1/n on x=e_j should return e_j exactly:
 *   H·(1/n · H·e_j) = (H²/n)·e_j = (nI/n)·e_j = e_j
 * ─────────────────────────────────────────────────────────────────────── */
static void test_project_rect_forward_roundtrip() {
    fprintf(stderr, "\n--- test_project_rect_forward_roundtrip ---\n");
    const int N = 16;

    /* Build identity W and project */
    std::vector<int8_t> W(N * N, 0);
    for (int i = 0; i < N; i++) W[i * N + i] = 1;

    std::vector<float> d(N, 0.0f);
    acdc_project_rect(d.data(), W.data(), N, N);   /* d[k] = 1/N */

    /* Forward pass for x = e_3 */
    std::vector<float> x(N, 0.0f);
    x[3] = 1.0f;
    std::vector<float> y(N, 0.0f);
    acdc_forward_rect_f32(y.data(), N, x.data(), N, d.data());

    float max_err = 0.0f;
    for (int i = 0; i < N; i++)
        max_err = fmaxf(max_err, fabsf(y[i] - x[i]));

    EXPECT_NEAR(max_err, 0.0f, 1e-4f, "project_rect→forward: W=I roundtrip y=x");
}

/* ─── Driver ─────────────────────────────────────────────────────────────*/

int main(void) {
    test_square_identity();
    test_upscale();
    test_downscale();
    test_zero_diagonal();
    test_linearity();
    test_i8_vs_f32();
    test_falcon_ffn_dims();
    test_falcon_down_proj_dims();
    test_project_rect_square_identity();
    test_project_rect_known();
    test_project_rect_sparse();
    test_project_rect_forward_roundtrip();

    fprintf(stderr, "\n=== test_acdc_rect: %d failure(s) ===\n", g_fails);
    return g_fails == 0 ? 0 : 1;
}
