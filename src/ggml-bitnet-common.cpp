/*
 * ggml-bitnet-common.cpp — Implementation of shared utilities
 *
 * See include/ggml-bitnet-common.h for the algorithm taxonomy and
 * the rationale for why this file is intentionally small.
 */

#include "ggml-bitnet-common.h"

int bitnet_next_pow2(int n) {
    if (n <= 1) return 1;
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* Backward-compat thin wrappers.  We declare them extern "C" because
 * the historical headers (ggml-bitnet-fwht.h, ggml-bitnet-hrr.h) declare
 * them at file scope (no extern "C" wrapper), and standalone tests may
 * include those headers AFTER ggml-bitnet-common.h, which puts the test
 * in extern "C" context.  Matching linkage here keeps everyone happy. */
extern "C" {
int fwht_next_pow2(int n) { return bitnet_next_pow2(n); }
int hrr_next_pow2(int n)  { return bitnet_next_pow2(n); }
}
