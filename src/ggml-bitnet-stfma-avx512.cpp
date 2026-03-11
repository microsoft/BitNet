#include "ggml-bitnet-stfma.h"
#include <immintrin.h>
#include <stdint.h>

/**
 * Fully vectorized AVX-512 dense ternary FMA kernel
 * 
 * This implementation is 100% SIMD with zero scalar fallbacks.
 * All operations are performed using AVX-512 instructions.
 * 
 * Key optimizations:
 * 1. Process 16 trits per iteration (512-bit vectors)
 * 2. Branchless trit unpacking using variable shifts
 * 3. Direct SIMD ternary multiplication
 * 4. Horizontal reduction using AVX-512 instructions
 */

#if defined(__AVX512F__)

/**
 * Unpack 16 2-bit trits into 16 int32 values using AVX-512
 * Input: 32-bit packed value containing 16 trits
 * Output: __m512i containing 16 int32 values
 */
static inline __m512i unpack_trits_avx512(uint32_t packed) {
    // Broadcast packed value to all lanes
    __m512i packed_vec = _mm512_set1_epi32(packed);
    
    // Create shift amounts: 0, 2, 4, 6, ..., 30
    __m512i shift_amounts = _mm512_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28, 30
    );
    
    // Variable shift right per lane
    __m512i shifted = _mm512_srlv_epi32(packed_vec, shift_amounts);
    
    // Mask to 2 bits
    __m512i mask_2bits = _mm512_set1_epi32(0x3);
    __m512i trit_vec = _mm512_and_si512(shifted, mask_2bits);
    
    return trit_vec;
}

/**
 * Convert 2-bit encoded trits to signed values: 0→-1, 1→0, 2→+1
 * Input: __m512i with values in range [0, 2]
 * Output: __m512i with values in range [-1, +1]
 */
static inline __m512i decode_trits_avx512(__m512i encoded) {
    // Create constant vectors
    __m512i ones = _mm512_set1_epi32(1);
    
    // Subtract 1 to map: 0→-1, 1→0, 2→+1
    return _mm512_sub_epi32(encoded, ones);
}

/**
 * Horizontal sum of 16 int32 values in a __m512i vector
 * Uses AVX-512 reduction instructions for maximum performance
 */
static inline int32_t horizontal_sum_avx512(__m512i vec) {
    // Reduce to 256-bit
    __m256i low = _mm512_castsi512_si256(vec);
    __m256i high = _mm512_extracti64x4_epi64(vec, 1);
    __m256i sum256 = _mm256_add_epi32(low, high);
    
    // Reduce to 128-bit
    __m128i low128 = _mm256_castsi256_si128(sum256);
    __m128i high128 = _mm256_extracti128_si256(sum256, 1);
    __m128i sum128 = _mm_add_epi32(low128, high128);
    
    // Reduce to 64-bit
    __m128i high64 = _mm_unpackhi_epi64(sum128, sum128);
    __m128i sum64 = _mm_add_epi32(sum128, high64);
    
    // Reduce to 32-bit
    __m128i high32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i sum32 = _mm_add_epi32(sum64, high32);
    
    return _mm_cvtsi128_si32(sum32);
}

/**
 * Fully vectorized dense ternary FMA kernel (AVX-512)
 * 
 * @param weights Pointer to STFMA-encoded ternary weights (2-bit packed)
 * @param activations Pointer to int32 activations
 * @param n Number of elements (must be multiple of 16)
 * @return Dot product result
 */
int32_t ggml_bitnet_stfma_dense_avx512(
    const uint8_t* weights,
    const int32_t* activations,
    size_t n
) {
    __m512i accumulator = _mm512_setzero_si512();
    
    // Process 16 elements per iteration
    for (size_t i = 0; i < n; i += 16) {
        // Load 4 bytes (16 trits at 2 bits each)
        uint32_t packed = *(const uint32_t*)&weights[i / 4];
        
        // Unpack 16 trits to int32 (branchless, fully vectorized)
        __m512i trit_vec = unpack_trits_avx512(packed);
        
        // Decode to signed values: 0→-1, 1→0, 2→+1
        __m512i weight_vec = decode_trits_avx512(trit_vec);
        
        // Load 16 activations
        __m512i act_vec = _mm512_loadu_si512((const __m512i*)&activations[i]);
        
        // Multiply and accumulate (FMA)
        __m512i product = _mm512_mullo_epi32(weight_vec, act_vec);
        accumulator = _mm512_add_epi32(accumulator, product);
    }
    
    // Horizontal sum to get final result
    return horizontal_sum_avx512(accumulator);
}

/**
 * Fully vectorized dense ternary FMA kernel with tail handling
 * 
 * This version handles arrays that are not multiples of 16.
 * The tail is processed using masked operations (still vectorized).
 */
int32_t ggml_bitnet_stfma_dense_avx512_tail(
    const uint8_t* weights,
    const int32_t* activations,
    size_t n
) {
    __m512i accumulator = _mm512_setzero_si512();
    
    // Process full 16-element chunks
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        uint32_t packed = *(const uint32_t*)&weights[i / 4];
        __m512i trit_vec = unpack_trits_avx512(packed);
        __m512i weight_vec = decode_trits_avx512(trit_vec);
        __m512i act_vec = _mm512_loadu_si512((const __m512i*)&activations[i]);
        __m512i product = _mm512_mullo_epi32(weight_vec, act_vec);
        accumulator = _mm512_add_epi32(accumulator, product);
    }
    
    // Handle tail using masked operations (still vectorized!)
    if (i < n) {
        size_t remaining = n - i;
        __mmask16 mask = (__mmask16)((1 << remaining) - 1);
        
        // Load with mask
        uint32_t packed = *(const uint32_t*)&weights[i / 4];
        __m512i trit_vec = unpack_trits_avx512(packed);
        __m512i weight_vec = decode_trits_avx512(trit_vec);
        __m512i act_vec = _mm512_maskz_loadu_epi32(mask, &activations[i]);
        
        // Masked multiply and accumulate
        __m512i product = _mm512_maskz_mullo_epi32(mask, weight_vec, act_vec);
        accumulator = _mm512_add_epi32(accumulator, product);
    }
    
    return horizontal_sum_avx512(accumulator);
}

#else
// Fallback for non-AVX-512 systems
int32_t ggml_bitnet_stfma_dense_avx512(
    const uint8_t* weights,
    const int32_t* activations,
    size_t n
) {
    (void)weights;
    (void)activations;
    (void)n;
    return 0; // Should never be called
}

int32_t ggml_bitnet_stfma_dense_avx512_tail(
    const uint8_t* weights,
    const int32_t* activations,
    size_t n
) {
    (void)weights;
    (void)activations;
    (void)n;
    return 0; // Should never be called
}
#endif
