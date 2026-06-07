/* bench_fwht_avx2.cpp
 *
 * Measures throughput of the fwht_f32() butterfly implementations:
 *   - scalar reference (butterfly_f32_scalar, always compiled)
 *   - AVX2 optimized  (butterfly_f32_avx2 with fused h=1,2,4 prefix)
 *
 * Compares against expected sizes for ACDC rect workloads:
 *   BitNet-2B:    n_embd=2560 → P=4096
 *   Falcon3-3B:   n_ff=9216  → P=16384
 *   Falcon3-10B:  n_ff=23040 → P=32768
 *
 * Build:
 *   clang++-18 -O3 -mavx2 -mfma -std=c++17 \
 *     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
 *     -Iinclude \
 *     src/ggml-bitnet-fwht.cpp src/ggml-bitnet-common.cpp \
 *     benchmarks/bench_fwht_avx2.cpp \
 *     -L/usr/lib/gcc/x86_64-linux-gnu/13 -lm -o build/bench_fwht_avx2
 *
 * Run:
 *   ./build/bench_fwht_avx2
 */

#include "ggml-bitnet-fwht.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using hrc = std::chrono::high_resolution_clock;
using ns  = std::chrono::nanoseconds;

/* Scalar reference butterfly (used as baseline for speedup ratio) */
static void fwht_scalar_ref(float * v, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; j++) {
                float a = v[i + j];
                float b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
}

struct BenchResult {
    double ns_per_call_scalar;
    double ns_per_call_avx2;
    double speedup;
};

static BenchResult bench_fwht(int n, int warmup, int iters) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> v_scalar(n), v_avx2(n), v_init(n);
    for (int i = 0; i < n; i++) v_init[i] = nd(rng);

    /* Warmup */
    for (int i = 0; i < warmup; i++) {
        std::copy(v_init.begin(), v_init.end(), v_avx2.begin());
        fwht_f32(v_avx2.data(), n);
    }
    for (int i = 0; i < warmup; i++) {
        std::copy(v_init.begin(), v_init.end(), v_scalar.begin());
        fwht_scalar_ref(v_scalar.data(), n);
    }

    /* Time scalar */
    double scalar_ns = 0;
    for (int i = 0; i < iters; i++) {
        std::copy(v_init.begin(), v_init.end(), v_scalar.begin());
        auto t0 = hrc::now();
        fwht_scalar_ref(v_scalar.data(), n);
        auto t1 = hrc::now();
        scalar_ns += (double)std::chrono::duration_cast<ns>(t1 - t0).count();
    }
    scalar_ns /= iters;

    /* Time AVX2 (fwht_f32) */
    double avx2_ns = 0;
    for (int i = 0; i < iters; i++) {
        std::copy(v_init.begin(), v_init.end(), v_avx2.begin());
        auto t0 = hrc::now();
        fwht_f32(v_avx2.data(), n);
        auto t1 = hrc::now();
        avx2_ns += (double)std::chrono::duration_cast<ns>(t1 - t0).count();
    }
    avx2_ns /= iters;

    return {scalar_ns, avx2_ns, scalar_ns / avx2_ns};
}

int main() {
    const int WARMUP = 50;
    const int ITERS  = 500;

    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║  FWHT AVX2 in-register prefix benchmark (h=1,2,4 fused)    ║\n");
    printf("║  scalar vs fwht_f32 (AVX2 optimized)                        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    struct TestCase { int n; const char * label; };
    const TestCase cases[] = {
        {    8, "n=8        (prefix only, no large stages)"},
        {   16, "n=16       (prefix + 1 large stage)"},
        {   32, "n=32       (prefix + 2 large stages)"},
        {  128, "n=128      (test_acdc canonical size)"},
        {  256, "n=256      (½ Falcon head_dim)"},
        { 4096, "n=4096     (BitNet-2B P=next_pow2(2560))"},
        {16384, "n=16384    (Falcon3-3B P=next_pow2(9216))"},
        {32768, "n=32768    (Falcon3-10B P=next_pow2(23040))"},
    };

    printf("  %-40s  %9s  %9s  %6s\n", "Size", "Scalar ns", "AVX2 ns", "Speedup");
    printf("  %-40s  %9s  %9s  %6s\n",
           "----------------------------------------",
           "---------", "---------", "-------");

    for (auto & tc : cases) {
        auto r = bench_fwht(tc.n, WARMUP, ITERS);
        printf("  %-40s  %9.1f  %9.1f  %5.2f×\n",
               tc.label, r.ns_per_call_scalar, r.ns_per_call_avx2, r.speedup);
    }

    printf("\nNote: speedup > 1× means AVX2 is faster.\n");
    printf("Key: for ACDC rect (Falcon3-10B), FWHT called 2× per FFN layer per token.\n");
    printf("     n=32768 speedup directly translates to end-to-end throughput gain.\n\n");

    /* Verification: scalar and AVX2 agree */
    printf("Verification (scalar == AVX2):\n");
    std::mt19937 rng(99);
    std::normal_distribution<float> nd;
    bool all_ok = true;
    for (auto & tc : cases) {
        std::vector<float> vs(tc.n), va(tc.n);
        for (int i = 0; i < tc.n; i++) vs[i] = va[i] = nd(rng);
        fwht_scalar_ref(vs.data(), tc.n);
        fwht_f32(va.data(), tc.n);
        float mx = 0;
        for (int i = 0; i < tc.n; i++) mx = std::max(mx, std::fabs(vs[i] - va[i]));
        bool ok = (mx < 1e-3f * tc.n);
        printf("  n=%-6d  max_diff=%.2e  %s\n", tc.n, mx, ok ? "✓" : "FAILED ✗");
        if (!ok) all_ok = false;
    }
    return all_ok ? 0 : 1;
}
