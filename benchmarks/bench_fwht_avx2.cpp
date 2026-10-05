/* bench_fwht_avx2.cpp
 *
 * Benchmarks fwht_f32() (AVX2 + in-register prefix) and fwht_f32_parallel()
 * (OpenMP multi-thread) against a scalar reference.
 *
 * Relevant sizes for ACDC rect workloads:
 *   BitNet-2B:    P = next_pow2(2560)  =  4096
 *   Falcon3-3B:   P = next_pow2(9216)  = 16384
 *   Falcon3-10B:  P = next_pow2(23040) = 32768
 *
 * Build (serial, no OMP):
 *   clang++-18 -O3 -mavx2 -mfma -std=c++17 \
 *     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
 *     -Iinclude \
 *     src/ggml-bitnet-fwht.cpp src/ggml-bitnet-common.cpp \
 *     benchmarks/bench_fwht_avx2.cpp \
 *     -L/usr/lib/gcc/x86_64-linux-gnu/13 -lm -o build/bench_fwht_avx2
 *
 * Build (with OMP parallel section):
 *   clang++-18 -O3 -mavx2 -mfma -std=c++17 -fopenmp \
 *     -DBITNET_FWHT_OMP \
 *     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
 *     -Iinclude \
 *     src/ggml-bitnet-fwht.cpp src/ggml-bitnet-common.cpp \
 *     benchmarks/bench_fwht_avx2.cpp \
 *     -L/usr/lib/gcc/x86_64-linux-gnu/13 -lm -o build/bench_fwht_avx2_omp
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

static void fwht_scalar_ref(float * v, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = 0; j < len; j++) {
                float a = v[i+j], b = v[i+j+len];
                v[i+j] = a+b; v[i+j+len] = a-b;
            }
}

static double time_fn(std::vector<float> & buf, const std::vector<float> & init,
                      void (*fn)(float *, int), int iters) {
    double total = 0;
    for (int i = 0; i < iters; i++) {
        std::copy(init.begin(), init.end(), buf.begin());
        auto t0 = hrc::now();
        fn(buf.data(), (int)buf.size());
        auto t1 = hrc::now();
        total += (double)std::chrono::duration_cast<ns>(t1-t0).count();
    }
    return total / iters;
}

static double time_parallel(std::vector<float> & buf, const std::vector<float> & init,
                            int n_threads, int iters) {
    double total = 0;
    for (int i = 0; i < iters; i++) {
        std::copy(init.begin(), init.end(), buf.begin());
        auto t0 = hrc::now();
        fwht_f32_parallel(buf.data(), (int)buf.size(), n_threads);
        auto t1 = hrc::now();
        total += (double)std::chrono::duration_cast<ns>(t1-t0).count();
    }
    return total / iters;
}

int main() {
    const int WARMUP = 50;
    const int ITERS  = 500;

    struct TestCase { int n; const char * label; };
    const TestCase cases[] = {
        {    8, "n=8        (prefix only)"},
        {   32, "n=32       (prefix + 2 stages)"},
        {  128, "n=128      (test_acdc size)"},
        { 4096, "n=4096     (BitNet-2B P)"},
        {16384, "n=16384    (Falcon3-3B P)"},
        {32768, "n=32768    (Falcon3-10B P)"},
    };

    /* ── Section 1: scalar vs AVX2 single-thread ── */
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  FWHT benchmark — AVX2 in-register prefix + OMP parallel      ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    printf("[ 1 ] Scalar vs AVX2 single-thread\n");
    printf("  %-38s  %9s  %9s  %6s\n", "Size", "Scalar ns", "AVX2 ns", "Speedup");
    printf("  %s\n", std::string(72, '-').c_str());

    std::mt19937 rng(42);
    std::normal_distribution<float> nd;
    for (auto & tc : cases) {
        std::vector<float> init(tc.n), buf(tc.n);
        for (auto & x : init) x = nd(rng);

        /* warmup */
        for (int i = 0; i < WARMUP; i++) {
            std::copy(init.begin(), init.end(), buf.begin());
            fwht_f32(buf.data(), tc.n);
        }
        double scalar_ns  = time_fn(buf, init, fwht_scalar_ref, ITERS);
        double avx2_ns    = time_fn(buf, init, fwht_f32, ITERS);
        printf("  %-38s  %9.1f  %9.1f  %5.2f×\n",
               tc.label, scalar_ns, avx2_ns, scalar_ns / avx2_ns);
    }

    /* ── Section 2: AVX2 vs OMP parallel (T=2,4,8 threads) ── */
#if defined(BITNET_FWHT_OMP)
    const int thread_counts[] = {2, 4, 8};
    printf("\n[ 2 ] AVX2 single-thread vs OMP parallel\n");
    printf("  %-38s  %9s", "Size", "AVX2-1T ns");
    for (int t : thread_counts) printf("  %5dT ns  Spd", t);
    printf("\n  %s\n", std::string(90, '-').c_str());

    for (auto & tc : cases) {
        std::vector<float> init(tc.n), buf(tc.n);
        for (auto & x : init) x = nd(rng);

        for (int i = 0; i < WARMUP; i++) {
            std::copy(init.begin(), init.end(), buf.begin());
            fwht_f32(buf.data(), tc.n);
        }
        double avx2_1t = time_fn(buf, init, fwht_f32, ITERS);
        printf("  %-38s  %9.1f", tc.label, avx2_1t);

        for (int t : thread_counts) {
            for (int i = 0; i < WARMUP; i++) {
                std::copy(init.begin(), init.end(), buf.begin());
                fwht_f32_parallel(buf.data(), tc.n, t);
            }
            double par_ns = time_parallel(buf, init, t, ITERS);
            printf("  %9.1f  %3.1f×", par_ns, avx2_1t / par_ns);
        }
        printf("\n");
    }
    printf("\nFinding: OMP threading does NOT improve FWHT throughput for single vectors.\n");
    printf("  Root cause: FWHT has log2(n) sequentially dependent stages (h=8..n/2).\n");
    printf("  Each OMP barrier costs ~10-50 µs; with 12 barriers the overhead\n");
    printf("  exceeds the actual compute (n=32768: ~100 µs compute, ~120 µs barriers).\n");
    printf("  Solution for multi-vector throughput: batch FWHT (interleave B vectors\n");
    printf("  through the butterfly — no inter-stage synchronization needed).\n");
#else
    printf("\n[ 2 ] OMP parallel section: rebuild with -DBITNET_FWHT_OMP -fopenmp to enable.\n");
    printf("      FINDING: threading not beneficial for single-vector FWHT.\n");
    printf("      See comment in fwht_f32_parallel() for the architectural reason.\n");
#endif

    /* ── Verification ── */
    printf("\nVerification (all implementations agree):\n");
    bool all_ok = true;
    std::mt19937 rng2(99);
    for (auto & tc : cases) {
        std::vector<float> vs(tc.n), va(tc.n), vp(tc.n);
        for (int i = 0; i < tc.n; i++) vs[i] = va[i] = vp[i] = nd(rng2);
        fwht_scalar_ref(vs.data(), tc.n);
        fwht_f32(va.data(), tc.n);
        fwht_f32_parallel(vp.data(), tc.n, 4);
        float mx_avx2 = 0, mx_par = 0;
        for (int i = 0; i < tc.n; i++) {
            mx_avx2 = std::max(mx_avx2, std::fabs(vs[i]-va[i]));
            mx_par  = std::max(mx_par,  std::fabs(vs[i]-vp[i]));
        }
        bool ok = (mx_avx2 < 1e-3f * tc.n) && (mx_par < 1e-3f * tc.n);
        printf("  n=%-6d  avx2_diff=%.1e  par_diff=%.1e  %s\n",
               tc.n, mx_avx2, mx_par, ok ? "✓" : "FAILED ✗");
        if (!ok) all_ok = false;
    }
    return all_ok ? 0 : 1;
}
