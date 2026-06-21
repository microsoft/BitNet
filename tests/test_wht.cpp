// test_wht.cpp — Standalone validation of L2 (WHT) kernels
//
// Verifica que o truque "WHT zero-multiplicação" produz o mesmo resultado
// que o caminho MAD de referência. 5/5 PASS esperado.
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     -Iinclude -L/usr/lib/gcc/x86_64-linux-gnu/13 \
//     src/ggml-bitnet-wht.cpp test_wht.cpp -o build/test_wht

#include "ggml-bitnet-wht.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

/* ── I2_S packing (BitNet strided layout, x86):
 *   Block of 128 weights = 32 bytes. Within a block:
 *     weight i → byte (i % 32), bits (3 - (i / 32) % 4) * 2 .. +1
 *   The bit order is INVERTED: bits [7:6] hold group 0 (positions 0..31),
 *   bits [1:0] hold group 3 (positions 96..127). Matches the AVX2 path
 *   and the library's own unpack_i2s_block. ── */
static void pack_ternary_i2s(const std::vector<int8_t> & src, std::vector<uint8_t> & dst) {
    size_t n_bytes = (src.size() + 3) / 4;
    dst.assign(n_bytes, 0);
    for (size_t i = 0; i < src.size(); i++) {
        int v = (src[i] > 0) ? 2 : (src[i] < 0 ? 0 : 1);
        size_t byte_idx = i % 32;
        size_t group    = (i / 32) % 4;
        size_t shift    = (3 - group) * 2;
        dst[byte_idx] |= (uint8_t)(v << shift);
    }
}

static int8_t unpack_i2s(const std::vector<uint8_t> & src, size_t i) {
    size_t byte_idx = i % 32;
    size_t group    = (i / 32) % 4;
    size_t shift    = (3 - group) * 2;
    int v = (src[byte_idx] >> shift) & 0x3;
    return (v == 2) ? 1 : (v == 0 ? -1 : 0);
}

static float max_abs_diff(const float * a, const float * b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
}

/* ── Tests ──────────────────────────────────────────────────────────────── */

static int test_wht_raw_dot() {
    printf("\n[1] ggml_wht_raw_dot: WHT path vs reference MAD  (n=128)\n");
    const int n = 128;
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> wd(-1, 1);
    std::uniform_int_distribution<int> xd(-127, 127);

    std::vector<int8_t> w(n);
    std::vector<int8_t> x(n);
    for (int i = 0; i < n; i++) { w[i] = wd(rng); x[i] = xd(rng); }
    std::vector<uint8_t> w_packed;
    pack_ternary_i2s(w, w_packed);

    int32_t wht = ggml_wht_raw_dot(n, w_packed.data(), x.data());

    /* Reference 1: Σᵢ w[i]·x[i]  (using unpacked ternary) */
    int32_t ref = 0;
    for (int i = 0; i < n; i++) ref += (int32_t)w[i] * (int32_t)x[i];

    /* Reference 2: Σᵢ unpacked_i2s(packed, i) · x[i]  (sanity check the pack) */
    int32_t ref2 = 0;
    for (int i = 0; i < n; i++) ref2 += (int32_t)unpack_i2s(w_packed, i) * (int32_t)x[i];

    int diff = std::abs(wht - ref);
    int diff2 = std::abs(wht - ref2);
    printf("    wht=%d  ref_unpacked(w)=%d  ref_via_pack=%d  |diff|=%d  |diff_pack|=%d\n",
           wht, ref, ref2, diff, diff2);
    int ok = diff == 0;
    printf("    %s\n", ok ? "WHT_RAW ✓" : "FAILED ✗");
    return ok;
}

static int test_wht_sum_i8() {
    printf("\n[2] ggml_wht_sum_i8: SIMD sum vs scalar  (n=128)\n");
    const int n = 128;
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> xd(-127, 127);
    std::vector<int8_t> x(n);
    for (int i = 0; i < n; i++) x[i] = xd(rng);

    int32_t s = ggml_wht_sum_i8(n, x.data());
    int32_t ref = 0;
    for (int i = 0; i < n; i++) ref += (int32_t)x[i];

    int diff = std::abs(s - ref);
    printf("    sum=%d  ref=%d  |diff|=%d\n", s, ref, diff);
    int ok = diff == 0;
    printf("    %s\n", ok ? "SUM ✓" : "FAILED ✗");
    return ok;
}

static int test_wht_verify() {
    printf("\n[3] ggml_wht_verify: ggml verify helper (n=128, tolerance=1e-5)\n");
    const int n = 128;
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> wd(-1, 1);
    std::uniform_int_distribution<int> xd(-100, 100);
    std::vector<int8_t> w(n), x(n);
    for (int i = 0; i < n; i++) { w[i] = wd(rng); x[i] = xd(rng); }
    std::vector<uint8_t> w_packed;
    pack_ternary_i2s(w, w_packed);
    /* Verify with non-zero scales — should still be exactly correct for raw dot. */
    int v = ggml_wht_verify(n, w_packed.data(), x.data(), 1.0f, 1.0f, 1e-5f);
    printf("    ggml_wht_verify → %d  (expected 1=match)\n", v);
    int ok = (v == 1);
    printf("    %s\n", ok ? "VERIFY ✓" : "FAILED ✗");
    return ok;
}

static int test_wht_gemv_single_row() {
    printf("\n[4] ggml_vec_dot_wht_ternary: single row vs unpacked reference  (n=128)\n");
    const int n = 128;
    std::mt19937 rng(13);
    std::uniform_int_distribution<int> wd(-1, 1);
    std::uniform_int_distribution<int> xd(-100, 100);
    std::vector<int8_t> w(n), x(n);
    for (int i = 0; i < n; i++) { w[i] = wd(rng); x[i] = xd(rng); }
    std::vector<uint8_t> w_packed;
    pack_ternary_i2s(w, w_packed);

    float s = 0.0f;
    ggml_vec_dot_wht_ternary(n, &s, w_packed.data(), x.data(), 1.0f, 1.0f);
    /* Reference (MAD dequantization): result = (raw - act_sum) * w_scale * act_scale
     * When scales=1, MAD returns (raw - 0) = raw. */
    int32_t ref = 0;
    for (int i = 0; i < n; i++) ref += (int32_t)w[i] * (int32_t)x[i];
    float diff = std::fabs(s - (float)ref);
    printf("    wht_dot=%.1f  ref=%d  |diff|=%.2e\n", s, ref, diff);
    int ok = (diff < 1e-3f);
    printf("    %s\n", ok ? "DOT ✓" : "FAILED ✗");
    return ok;
}

static int test_wht_identity_via_gemv() {
    printf("\n[5] ggml_gemv_wht_ternary: row dot + sum correction matches scalar\n");
    const int n = 128;
    const int m = 4;  /* 4 rows */
    std::mt19937 rng(2024);
    std::uniform_int_distribution<int> wd(-1, 1);
    std::uniform_int_distribution<int> xd(-100, 100);
    std::vector<int8_t> w(m * n), x(n);
    for (int i = 0; i < m * n; i++) w[i] = wd(rng);
    for (int i = 0; i < n; i++) x[i] = xd(rng);
    /* Each row of 128 weights packs to 32 bytes (strided I2_S). Rows in the
     * packed tensor are CONTIGUOUS: row i starts at offset i * (n/4) bytes.
     * We must pack each row independently, not the linear (m*n) array. */
    std::vector<uint8_t> w_packed(m * (n / 4), 0);
    for (int i = 0; i < m; i++) {
        std::vector<int8_t>   row_w(w.begin() + i*n, w.begin() + (i+1)*n);
        std::vector<uint8_t> row_p;
        pack_ternary_i2s(row_w, row_p);
        std::memcpy(w_packed.data() + i * (n / 4), row_p.data(), n / 4);
    }

    std::vector<float> y(m);
    ggml_gemv_wht_ternary(m, n, y.data(), w_packed.data(), x.data(), 1.0f, 1.0f);

    std::vector<float> y_ref(m);
    for (int i = 0; i < m; i++) {
        int32_t s = 0;
        for (int j = 0; j < n; j++) s += (int32_t)w[i*n+j] * (int32_t)x[j];
        y_ref[i] = (float)s;
    }
    float diff = max_abs_diff(y.data(), y_ref.data(), m);
    printf("    max|y_wht - y_ref| = %.2e  (m=%d)\n", diff, m);
    int ok = (diff < 1e-2f);  /* generous — sum correction can introduce FP noise */
    printf("    %s\n", ok ? "GEMV ✓" : "FAILED ✗");
    return ok;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  WHT (Level 2) — Standalone C++ validation\n");
    printf("═══════════════════════════════════════════════════════════\n");
    int n_pass = 0, n_total = 0;
    struct { const char * name; int (*fn)(); } tests[] = {
        { "raw_dot",   test_wht_raw_dot         },
        { "sum_i8",    test_wht_sum_i8          },
        { "verify",    test_wht_verify          },
        { "dot_row",   test_wht_gemv_single_row },
        { "gemv",      test_wht_identity_via_gemv },
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
