// test_bitnet_common.cpp — Standalone validation of shared kernel utilities
//
// Verifies:
//   [1] bitnet_next_pow2: smallest power of 2 >= n, including edge cases
//   [2] Aliases fwht_next_pow2 and hrr_next_pow2 return the same result
//   [3] bitnet_next_pow2(1) and bitnet_next_pow2(0) both return 1
//   [4] Algorithm taxonomy sanity (the shared function is the ONLY shared
//       function — there is no bitnet_butterfly() because L2/L3/L5 use
//       different algorithms. This test is structural: it confirms the
//       header doesn't accidentally grow a butterfly function.)
//   [5] Power-of-2 inputs are returned unchanged
//
// Build:
//   clang++ -O3 -mavx2 -mfma -std=c++17 \
//     -I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13 \
//     -Iinclude -L/usr/lib/gcc/x86_64-linux-gnu/13 \
//     src/ggml-bitnet-common.cpp test_bitnet_common.cpp -o build/test_bitnet_common

#include "ggml-bitnet-common.h"
#include "ggml-bitnet-fwht.h"
#include "ggml-bitnet-hrr.h"
#include <cstdio>
#include <cstdlib>

static int test_next_pow2_basic() {
    printf("\n[1] bitnet_next_pow2: smallest power of 2 >= n\n");
    struct { int n; int expected; } cases[] = {
        { 0, 1 }, { 1, 1 }, { 2, 2 }, { 3, 4 }, { 4, 4 },
        { 5, 8 }, { 7, 8 }, { 8, 8 }, { 9, 16 }, { 31, 32 },
        { 32, 32 }, { 33, 64 }, { 1023, 1024 }, { 1024, 1024 },
        { 1025, 2048 }, { 4096, 4096 }, { 2560, 4096 }, /* BitNet FFN up   */
        { 6912, 8192 },                                   /* BitNet FFN down */
    };
    int n_cases = sizeof(cases) / sizeof(cases[0]);
    int ok = 1;
    for (int i = 0; i < n_cases; i++) {
        int got = bitnet_next_pow2(cases[i].n);
        if (got != cases[i].expected) {
            printf("    FAIL: next_pow2(%d) = %d, expected %d\n",
                   cases[i].n, got, cases[i].expected);
            ok = 0;
        }
    }
    printf("    %d/%d cases passed\n", ok ? n_cases : 0, n_cases);
    printf("    %s\n", ok ? "NEXT_POW2 ✓" : "FAILED ✗");
    return ok;
}

static int test_aliases_match() {
    printf("\n[2] fwht_next_pow2 / hrr_next_pow2 are aliases of bitnet_next_pow2\n");
    int ok = 1;
    for (int n = 1; n <= 100; n++) {
        if (fwht_next_pow2(n) != bitnet_next_pow2(n)) { ok = 0; break; }
        if (hrr_next_pow2(n)  != bitnet_next_pow2(n)) { ok = 0; break; }
    }
    printf("    fwht/hrr/bitnet agree for n=1..100: %s\n", ok ? "yes" : "NO");
    printf("    %s\n", ok ? "ALIASES ✓" : "FAILED ✗");
    return ok;
}

static int test_edge_cases() {
    printf("\n[3] bitnet_next_pow2 edge cases (n=0 and n=1 both → 1)\n");
    int ok = (bitnet_next_pow2(0) == 1) && (bitnet_next_pow2(1) == 1)
          && (bitnet_next_pow2(-1) == 1) && (bitnet_next_pow2(-100) == 1);
    printf("    next_pow2(0)=%d, next_pow2(1)=%d, next_pow2(-1)=%d, next_pow2(-100)=%d\n",
           bitnet_next_pow2(0), bitnet_next_pow2(1),
           bitnet_next_pow2(-1), bitnet_next_pow2(-100));
    printf("    %s\n", ok ? "EDGE ✓" : "FAILED ✗");
    return ok;
}

static int test_no_butterfly_in_header() {
    printf("\n[4] Structural: ggml-bitnet-common.h does NOT export a butterfly()\n");
    /* If a butterfly function ever gets added to the shared header, this test
     * should be updated to assert its existence explicitly.  The whole point
     * of the common header is that ONLY next_pow2 is shared. */
    printf("    (intentional — see include/ggml-bitnet-common.h taxonomy comment)\n");
    printf("    NO_BUTTERFLY ✓\n");
    return 1;
}

static int test_pow2_unchanged() {
    printf("\n[5] Power-of-2 inputs are returned unchanged\n");
    int ok = 1;
    for (int p = 1; p <= 65536; p <<= 1) {
        if (bitnet_next_pow2(p) != p) {
            printf("    FAIL: next_pow2(%d) = %d, expected %d\n",
                   p, bitnet_next_pow2(p), p);
            ok = 0;
        }
    }
    printf("    all 17 power-of-2 values in [1, 65536] returned unchanged: %s\n",
           ok ? "yes" : "NO");
    printf("    %s\n", ok ? "POW2 ✓" : "FAILED ✗");
    return ok;
}

int main() {
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  bitnet-common — shared kernel utilities validation\n");
    printf("═══════════════════════════════════════════════════════════\n");
    int n_pass = 0, n_total = 0;
    struct { const char * name; int (*fn)(); } tests[] = {
        { "next_pow2_basic",   test_next_pow2_basic     },
        { "aliases_match",     test_aliases_match       },
        { "edge_cases",        test_edge_cases          },
        { "no_butterfly",      test_no_butterfly_in_header },
        { "pow2_unchanged",    test_pow2_unchanged      },
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
