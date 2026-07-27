/*
 * ggml-bitnet-common.h — Shared utilities across L2-L5 math kernels
 *
 * ─────────────────────────────────────────────────────────────────────────
 * WHY THIS HEADER IS SMALL
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The natural impulse when seeing three "butterfly" implementations
 * (L2 WHT, L3 FWHT, L5 FFT) is to extract a shared `butterfly_step()`
 * abstraction. After actually reading all three, that abstraction is
 * *not* a clean win — see the taxonomy below.
 *
 * The only piece that genuinely duplicates across kernels is the
 * "smallest power of 2 ≥ n" rounding utility (needed by L3 FWHT and
 * L5 FFT to pad their input vectors to a power of 2). Extracting
 * that, plus a few other small bits, is the right scope for a
 * "shared common" header. The butterfly operations themselves stay
 * per-kernel for clarity and to allow per-algorithm SIMD tricks
 * (e.g. L3 processes 8 float32 pairs at once in pure AVX2 add/sub;
 * L5 needs twiddle multiplications and complex number handling).
 *
 * ─────────────────────────────────────────────────────────────────────────
 * ALGORITHM TAXONOMY (L2 / L3 / L5)
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   L2 WHT (src/ggml-bitnet-wht.cpp)
 *       Algorithm: selection-mask dot product on I2_S packed bytes.
 *                 NOT a Cooley-Tukey butterfly. The "Hadamard domain"
 *                 trick is: H·x with H ∈ {±1} is computed via
 *                 `(w==+1 ? x : 0) − (w==−1 ? x : 0)` per element, with
 *                 32-wide AVX2 compare/select on packed bytes.
 *       Zero muls, no bit-reversal, in-place.
 *
 *   L3 FWHT (src/ggml-bitnet-fwht.cpp)
 *       Algorithm: in-order Cooley-Tukey radix-2 butterfly, real-valued.
 *       Twiddles are always ±1 (Hadamard matrix), so the inner operation
 *       is pure (a+b, a-b) — no multiplications.
 *       In-order (no bit-reversal — only the DIF variant of FFT
 *       needs it; L3 uses a DIT-like structure because the input
 *       order is the natural one for the final-form H matrix).
 *       Variants: f32 and i32, scalar + AVX2 + NEON.
 *
 *   L5 FFT (src/ggml-bitnet-hrr.cpp)
 *       Algorithm: Cooley-Tukey radix-2 DIF, complex-valued, with
 *       twiddle factors exp(−2πi·k/N). Bit-reversal permutation on
 *       input (Decimation In Frequency requires input in bit-reversed
 *       order for the output to be in natural order).
 *       Twiddles require complex multiplications (4 mults + 2 adds
 *       per butterfly, or 3 mults + 3 adds with the standard trick).
 *       The first log₂(N) stages have twiddles in {±1, ±i} and could
 *       avoid multiplications, but we don't bother (FMAs are cheap).
 *
 *   Conclusion: there is no common butterfly() to share. L2 is
 *   fundamentally different (selection mask, not butterfly), and L3/L5
 *   differ on twiddle handling, value type (real vs complex), and
 *   permutation (in-order vs bit-reversed). Forcing a shared API
 *   would obscure the math more than it would simplify the code.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * WHAT IS SHARED
 * ─────────────────────────────────────────────────────────────────────────
 *
 *   - bitnet_next_pow2: smallest power of 2 ≥ n (used by L3, L5 to pad)
 *   - BITNET_L* build-flag summary (re-exported here for convenience)
 *   - The taxonomy comment above (so future agents don't make the
 *     same "let's extract a butterfly" mistake)
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── bitnet_next_pow2 ────────────────────────────────────────────────────
 *
 * Returns the smallest power of 2 that is ≥ n. For n ≤ 1, returns 1.
 *
 * Used by:
 *   - L3 FWHT (src/ggml-bitnet-fwht.cpp): pads activation vectors
 *     to power-of-2 length before applying the butterfly.
 *   - L5 FFT  (src/ggml-bitnet-hrr.cpp): pads HRR vectors to power-of-2
 *     length for the radix-2 Cooley-Tukey FFT.
 *
 * L2 WHT does NOT use this (operates on fixed QK block size).
 * L4 tropical does NOT use this (operates per-token, not on fixed FFT blocks).
 */
int bitnet_next_pow2(int n);

#ifdef __cplusplus
}
#endif
