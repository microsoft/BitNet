/**
 * BitNet Sparse Ternary FMA Adapter - Implementation
 * 
 * Copyright 2025 HyperFold Technologies UK Ltd & BitNet Contributors
 * Licensed under the Apache License, Version 2.0
 */

#include "ggml-bitnet-stfma.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include <stdio.h>

/* ========================================================================== */
/* Thread-Local Buffer Management                                            */
/* ========================================================================== */

static thread_local struct stfma_thread_buffers tl_buffers = {
    nullptr, nullptr, nullptr, 0
};

void stfma_ensure_buffer_size(size_t required_size) {
    if (tl_buffers.buffer_size < required_size) {
        // Free old buffers
        free(tl_buffers.encoding_buffer);
        free(tl_buffers.int32_buffer);
        free(tl_buffers.accumulator_buffer);
        
        // Allocate new buffers with some headroom
        size_t alloc_size = required_size * 2;  // 2x for headroom
        tl_buffers.encoding_buffer = (uint8_t*)malloc(alloc_size / 4);
        tl_buffers.int32_buffer = (int32_t*)malloc(alloc_size * sizeof(int32_t));
        tl_buffers.accumulator_buffer = (int32_t*)malloc(alloc_size * sizeof(int32_t));
        tl_buffers.buffer_size = alloc_size;
    }
}

void stfma_free_buffers(void) {
    free(tl_buffers.encoding_buffer);
    free(tl_buffers.int32_buffer);
    free(tl_buffers.accumulator_buffer);
    tl_buffers.encoding_buffer = nullptr;
    tl_buffers.int32_buffer = nullptr;
    tl_buffers.accumulator_buffer = nullptr;
    tl_buffers.buffer_size = 0;
}

/* ========================================================================== */
/* Encoding Conversion Functions                                             */
/* ========================================================================== */

/**
 * Optimized Branchless Conversion
 * Replaces loop+switch with parallel bitwise logic.
 * 
 * Logic:
 * BitNet pairs:  00 (-1), 01 (0), 10 (+1), 11 (invalid)
 * STFMA pairs:   10 (-1), 00 (0), 01 (+1), 11 (invalid)
 *
 * Transformation per trit (2-bit pair):
 * Input 00 -> Output 10:  in_h=0, in_l=0 -> out_h=1, out_l=0
 * Input 01 -> Output 00:  in_h=0, in_l=1 -> out_h=0, out_l=0
 * Input 10 -> Output 01:  in_h=1, in_l=0 -> out_h=0, out_l=1
 * Input 11 -> Output 11:  in_h=1, in_l=1 -> out_h=1, out_l=1
 *
 * Bitwise logic:
 * out_low  = in_high (bit 1 of each pair)
 * out_high = ~(in_high XOR in_low)
 *
 * Performance: Zero branches, processes all 4 trits in parallel,
 * compiles to ~5 assembly instructions.
 */
uint8_t convert_bitnet_to_stfma_byte(uint8_t b) {
    // Mask for low bits of each pair: 01010101 = 0x55
    uint8_t low_bits = b & 0x55; 
    
    // Mask for high bits of each pair: 10101010 = 0xAA
    uint8_t high_bits = b & 0xAA;

    // STFMA Low Bit = BitNet High Bit (shifted right by 1)
    uint8_t out_low = (high_bits >> 1);

    // STFMA High Bit = ~(BitNet High Bit XOR BitNet Low Bit)
    // Need to align high_bits with low_bits for XOR
    uint8_t high_bits_shifted = (high_bits >> 1);
    uint8_t xor_result = high_bits_shifted ^ low_bits;
    uint8_t out_high = (~xor_result) & 0x55;
    out_high = out_high << 1;  // Shift back to high bit positions

    return out_high | out_low;
}

void convert_bitnet_to_stfma_array(
    const uint8_t* bitnet_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
) {
    for (size_t i = 0; i < num_bytes; i++) {
        stfma_packed[i] = convert_bitnet_to_stfma_byte(bitnet_packed[i]);
    }
}

#if defined(__AVX2__)

void convert_bitnet_to_stfma_avx2(
    const uint8_t* bitnet_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
) {
    // Lookup table for 4-bit nibble conversion
    // Each nibble contains 2 trits (4 bits)
    // We need to convert each 2-bit pair independently
    
    // For simplicity, we'll process byte by byte using scalar conversion
    // A full SIMD implementation would require a 256-entry lookup table
    // which is complex to set up efficiently
    
    size_t i = 0;
    
    // Process 32 bytes at a time (can be optimized further)
    for (; i + 32 <= num_bytes; i += 32) {
        for (size_t j = 0; j < 32; j++) {
            stfma_packed[i + j] = convert_bitnet_to_stfma_byte(bitnet_packed[i + j]);
        }
    }
    
    // Process remaining bytes
    for (; i < num_bytes; i++) {
        stfma_packed[i] = convert_bitnet_to_stfma_byte(bitnet_packed[i]);
    }
}

#endif

#if defined(__AVX512F__)

void convert_bitnet_to_stfma_avx512(
    const uint8_t* bitnet_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
) {
    // Similar to AVX2, use scalar conversion for now
    // Can be optimized with AVX-512 shuffle operations
    
    size_t i = 0;
    
    // Process 64 bytes at a time
    for (; i + 64 <= num_bytes; i += 64) {
        for (size_t j = 0; j < 64; j++) {
            stfma_packed[i + j] = convert_bitnet_to_stfma_byte(bitnet_packed[i + j]);
        }
    }
    
    // Process remaining bytes
    for (; i < num_bytes; i++) {
        stfma_packed[i] = convert_bitnet_to_stfma_byte(bitnet_packed[i]);
    }
}

#endif

/* ========================================================================== */
/* Type Conversion Functions                                                 */
/* ========================================================================== */

void convert_int8_to_int32_scalar(
    const int8_t* src,
    int32_t* dst,
    size_t n
) {
    for (size_t i = 0; i < n; i++) {
        dst[i] = (int32_t)src[i];
    }
}

#if defined(__AVX2__)

void convert_int8_to_int32_avx2(
    const int8_t* src,
    int32_t* dst,
    size_t n
) {
    size_t i = 0;
    
    // Process 8 elements at a time
    for (; i + 8 <= n; i += 8) {
        // Load 8 int8 values (64 bits)
        __m128i int8_vec = _mm_loadl_epi64((__m128i*)(src + i));
        
        // Sign-extend to int32 (256 bits)
        __m256i int32_vec = _mm256_cvtepi8_epi32(int8_vec);
        
        // Store 8 int32 values
        _mm256_storeu_si256((__m256i*)(dst + i), int32_vec);
    }
    
    // Process remaining elements
    for (; i < n; i++) {
        dst[i] = (int32_t)src[i];
    }
}

#endif

/* ========================================================================== */
/* int32 Sparse Ternary FMA Functions                                        */
/* ========================================================================== */

void sparse_ternary_fma_int32_scalar(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
) {
    for (size_t i = 0; i < N; i++) {
        // Extract 2-bit trit from packed array
        size_t byte_idx = i / 4;
        size_t trit_offset = (i % 4) * 2;
        uint8_t trit = (B_trit[byte_idx] >> trit_offset) & 0b11;
        
        // Decode and accumulate (STFMA encoding)
        if (trit == 0b01) {       // +1
            C[i] += A[i];
        } else if (trit == 0b10) { // -1
            C[i] -= A[i];
        }
        // else: trit == 0b00 (0), skip (no contribution)
    }
}

#if defined(__AVX2__)

void sparse_ternary_fma_int32_avx2(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i one = _mm256_set1_epi32(1);
    
    size_t i = 0;
    
    // Process 8 elements at a time
    for (; i + 8 <= N; i += 8) {
        // Load 8 coefficients
        __m256i a_vec = _mm256_loadu_si256((__m256i*)&A[i]);
        
        // Load 8 accumulators
        __m256i c_vec = _mm256_loadu_si256((__m256i*)&C[i]);
        
        // Load 2 bytes containing 8 trits (8 × 2 bits = 16 bits)
        size_t byte_idx = i / 4;
        uint16_t trit_packed = ((uint16_t)B_trit[byte_idx + 1] << 8) | B_trit[byte_idx];
        
        // Unpack 8 2-bit trits directly into 8 int32 lanes using SIMD
        // Broadcast the 16-bit value to all lanes as 32-bit
        __m256i packed_vec = _mm256_set1_epi32(trit_packed);
        
        // Create shift amounts for each lane: 0, 2, 4, 6, 8, 10, 12, 14
        __m256i shift_amounts = _mm256_setr_epi32(
            0, 2, 4, 6, 8, 10, 12, 14
        );
        
        // Shift each lane by its corresponding amount
        __m256i shifted = _mm256_srlv_epi32(packed_vec, shift_amounts);
        
        // Mask to extract only the lowest 2 bits from each lane
        __m256i mask_2bits = _mm256_set1_epi32(0b11);
        __m256i trit_vec = _mm256_and_si256(shifted, mask_2bits);
        
        // Create nonzero mask: true if trit != 0b00
        __m256i nonzero_cmp = _mm256_cmpgt_epi32(trit_vec, zero);
        
        // Extract sign bit (high bit of 2-bit trit)
        // For STFMA encoding: 0b01 (+1) has sign=0, 0b10 (-1) has sign=1
        __m256i sign_bit = _mm256_srli_epi32(trit_vec, 1);
        __m256i sign_bit_masked = _mm256_and_si256(sign_bit, one);
        __m256i sign_cmp = _mm256_cmpgt_epi32(sign_bit_masked, zero);
        
        // Compute contribution: 0 if trit=0, A if trit!=0
        __m256i contribution = _mm256_and_si256(a_vec, nonzero_cmp);
        
        // Conditionally negate if sign=1 (i.e., trit=0b10=-1)
        __m256i negated = _mm256_sub_epi32(zero, contribution);
        contribution = _mm256_blendv_epi8(contribution, negated, sign_cmp);
        
        // FMA: C += contribution
        c_vec = _mm256_add_epi32(c_vec, contribution);
        
        // Store result
        _mm256_storeu_si256((__m256i*)&C[i], c_vec);
    }
    
    // Process remaining elements
    for (; i < N; i++) {
        size_t byte_idx = i / 4;
        size_t trit_offset = (i % 4) * 2;
        uint8_t trit = (B_trit[byte_idx] >> trit_offset) & 0b11;
        
        if (trit == 0b01) {
            C[i] += A[i];
        } else if (trit == 0b10) {
            C[i] -= A[i];
        }
    }
}

#endif

#if defined(__AVX512F__)

void sparse_ternary_fma_int32_avx512(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
) {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi32(1);
    
    size_t i = 0;
    
    // Process 16 elements at a time
    for (; i + 16 <= N; i += 16) {
        // Load 16 coefficients
        __m512i a_vec = _mm512_loadu_si512(&A[i]);
        
        // Load 16 accumulators
        __m512i c_vec = _mm512_loadu_si512(&C[i]);
        
        // Load 4 bytes containing 16 trits (16 × 2 bits = 32 bits)
        size_t byte_idx = i / 4;
        uint32_t trit_packed = *(uint32_t*)&B_trit[byte_idx];
        
        // Unpack 16 2-bit trits directly into 16 int32 lanes using SIMD
        // Strategy: Broadcast the 32-bit value, then use shifts and masks
        // to extract each 2-bit pair into its own 32-bit lane
        
        // Broadcast the packed trits to all lanes
        __m512i packed_vec = _mm512_set1_epi32(trit_packed);
        
        // Create shift amounts for each lane: 0, 2, 4, 6, ..., 30
        // Lane i needs to shift right by (i * 2) bits
        __m512i shift_amounts = _mm512_setr_epi32(
            0, 2, 4, 6, 8, 10, 12, 14,
            16, 18, 20, 22, 24, 26, 28, 30
        );
        
        // Shift each lane by its corresponding amount
        __m512i shifted = _mm512_srlv_epi32(packed_vec, shift_amounts);
        
        // Mask to extract only the lowest 2 bits from each lane
        __m512i mask_2bits = _mm512_set1_epi32(0b11);
        __m512i trit_vec = _mm512_and_si512(shifted, mask_2bits);
        
        // Create nonzero mask
        __mmask16 nonzero_mask = _mm512_cmpneq_epi32_mask(trit_vec, zero);
        
        // Extract sign bit
        __m512i sign_bit = _mm512_srli_epi32(trit_vec, 1);
        __m512i sign_bit_masked = _mm512_and_si512(sign_bit, one);
        __mmask16 sign_mask = _mm512_cmpneq_epi32_mask(sign_bit_masked, zero);
        
        // Compute contribution: 0 if trit=0, A if trit!=0
        __m512i contribution = _mm512_maskz_mov_epi32(nonzero_mask, a_vec);
        
        // Conditionally negate if sign=1
        __m512i negated = _mm512_sub_epi32(zero, contribution);
        contribution = _mm512_mask_blend_epi32(sign_mask, contribution, negated);
        
        // FMA: C += contribution
        c_vec = _mm512_add_epi32(c_vec, contribution);
        
        // Store result
        _mm512_storeu_si512(&C[i], c_vec);
    }
    
    // Process remaining elements
    for (; i < N; i++) {
        size_t byte_idx = i / 4;
        size_t trit_offset = (i % 4) * 2;
        uint8_t trit = (B_trit[byte_idx] >> trit_offset) & 0b11;
        
        if (trit == 0b01) {
            C[i] += A[i];
        } else if (trit == 0b10) {
            C[i] -= A[i];
        }
    }
}

#endif

void sparse_ternary_fma_int32(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
) {
#if defined(__AVX512F__)
    if (N >= 16 && N % 16 == 0) {
        sparse_ternary_fma_int32_avx512(A, B_trit, C, N);
        return;
    }
#endif
    
#if defined(__AVX2__)
    if (N >= 8 && N % 8 == 0) {
        sparse_ternary_fma_int32_avx2(A, B_trit, C, N);
        return;
    }
#endif
    
    sparse_ternary_fma_int32_scalar(A, B_trit, C, N);
}

/* ========================================================================== */
/* BitNet Integration Functions                                              */
/* ========================================================================== */

void ggml_vec_dot_i2_i8_stfma(
    int n,
    float* s,
    size_t bs,
    const void* vx,
    size_t bx,
    const void* vy,
    size_t by,
    int nrc
) {
    const uint8_t* x = (uint8_t*)vx;
    const int8_t* y = (int8_t*)vy;
    
    // Ensure buffers are large enough
    stfma_ensure_buffer_size(n);
    
    // Get thread-local buffers
    uint8_t* x_stfma = tl_buffers.encoding_buffer;
    int32_t* y_int32 = tl_buffers.int32_buffer;
    int32_t* accumulator = tl_buffers.accumulator_buffer;
    
    // Clear accumulator
    memset(accumulator, 0, n * sizeof(int32_t));
    
    // Convert BitNet encoding to STFMA encoding
    size_t num_bytes = n / 4;
#if defined(__AVX512F__)
    convert_bitnet_to_stfma_avx512(x, x_stfma, num_bytes);
#elif defined(__AVX2__)
    convert_bitnet_to_stfma_avx2(x, x_stfma, num_bytes);
#else
    convert_bitnet_to_stfma_array(x, x_stfma, num_bytes);
#endif
    
    // Convert int8 to int32
#if defined(__AVX2__)
    convert_int8_to_int32_avx2(y, y_int32, n);
#else
    convert_int8_to_int32_scalar(y, y_int32, n);
#endif
    
    // Perform sparse ternary FMA
    sparse_ternary_fma_int32(y_int32, x_stfma, accumulator, n);
    
    // Sum accumulator
    int64_t sum = 0;
    for (int i = 0; i < n; i++) {
        sum += accumulator[i];
    }
    
    *s = (float)sum;
}
