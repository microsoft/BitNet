#ifndef GGML_BITNET_STFMA_AVX512_H
#define GGML_BITNET_STFMA_AVX512_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Fully vectorized dense ternary FMA kernel (AVX-512)
 * 
 * This kernel is 100% SIMD with zero scalar fallbacks.
 * Processes 16 elements per iteration using AVX-512 instructions.
 * 
 * @param weights Pointer to STFMA-encoded ternary weights (2-bit packed)
 * @param activations Pointer to int32 activations
 * @param n Number of elements (must be multiple of 16 for optimal performance)
 * @return Dot product result
 * 
 * Requirements:
 * - weights must be aligned to 4-byte boundary
 * - activations must be aligned to 64-byte boundary for best performance
 * - n should be a multiple of 16 (tail version handles non-multiples)
 */
int32_t ggml_bitnet_stfma_dense_avx512(
    const uint8_t* weights,
    const int32_t* activations,
    size_t n
);

/**
 * Fully vectorized dense ternary FMA kernel with tail handling (AVX-512)
 * 
 * This version handles arrays that are not multiples of 16 using masked operations.
 * The tail is still processed using SIMD (not scalar fallback).
 * 
 * @param weights Pointer to STFMA-encoded ternary weights (2-bit packed)
 * @param activations Pointer to int32 activations
 * @param n Number of elements (any value)
 * @return Dot product result
 */
int32_t ggml_bitnet_stfma_dense_avx512_tail(
    const uint8_t* weights,
    const int32_t* activations,
    size_t n
);

#ifdef __cplusplus
}
#endif

#endif // GGML_BITNET_STFMA_AVX512_H
