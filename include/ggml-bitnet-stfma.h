/**
 * BitNet Sparse Ternary FMA Adapter
 * 
 * This file provides an adapter layer between BitNet and the sparse-ternary-fma library.
 * It handles encoding conversions, type conversions, and provides optimized implementations
 * of ternary arithmetic operations.
 * 
 * Copyright 2025 HyperFold Technologies UK Ltd & BitNet Contributors
 * Licensed under the Apache License, Version 2.0
 */

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
/* Configuration                                                              */
/* ========================================================================== */

/**
 * Threshold for using sparse-ternary-fma instead of original implementation.
 * Operations with n >= threshold will use sparse-ternary-fma.
 * Can be overridden at compile time.
 */
#ifndef GGML_BITNET_STFMA_THRESHOLD
#define GGML_BITNET_STFMA_THRESHOLD 1024
#endif

/* ========================================================================== */
/* Encoding Conversion Functions                                             */
/* ========================================================================== */

/**
 * Convert a single byte from BitNet encoding to sparse-ternary-fma encoding.
 * 
 * BitNet encoding:  0→-1, 1→0, 2→+1
 * STFMA encoding:   0b10→-1, 0b00→0, 0b01→+1
 * 
 * @param bitnet_byte Byte containing 4 trits in BitNet encoding
 * @return Byte containing 4 trits in STFMA encoding
 */
uint8_t convert_bitnet_to_stfma_byte(uint8_t bitnet_byte);

/**
 * Convert an array from BitNet encoding to sparse-ternary-fma encoding (scalar).
 * 
 * @param bitnet_packed Input array in BitNet encoding
 * @param stfma_packed Output array in STFMA encoding (must be pre-allocated)
 * @param num_bytes Number of bytes to convert
 */
void convert_bitnet_to_stfma_array(
    const uint8_t* bitnet_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
);

#if defined(__AVX2__)
/**
 * Convert an array from BitNet encoding to sparse-ternary-fma encoding (AVX2).
 * 
 * @param bitnet_packed Input array in BitNet encoding
 * @param stfma_packed Output array in STFMA encoding (must be pre-allocated)
 * @param num_bytes Number of bytes to convert (must be multiple of 32)
 */
void convert_bitnet_to_stfma_avx2(
    const uint8_t* bitnet_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
);
#endif

#if defined(__AVX512F__)
/**
 * Convert an array from BitNet encoding to sparse-ternary-fma encoding (AVX-512).
 * 
 * @param bitnet_packed Input array in BitNet encoding
 * @param stfma_packed Output array in STFMA encoding (must be pre-allocated)
 * @param num_bytes Number of bytes to convert (must be multiple of 64)
 */
void convert_bitnet_to_stfma_avx512(
    const uint8_t* bitnet_packed,
    uint8_t* stfma_packed,
    size_t num_bytes
);
#endif

/* ========================================================================== */
/* Type Conversion Functions                                                 */
/* ========================================================================== */

/**
 * Convert int8 array to int32 array (scalar).
 * 
 * @param src Source int8 array
 * @param dst Destination int32 array (must be pre-allocated)
 * @param n Number of elements
 */
void convert_int8_to_int32_scalar(
    const int8_t* src,
    int32_t* dst,
    size_t n
);

#if defined(__AVX2__)
/**
 * Convert int8 array to int32 array (AVX2).
 * 
 * @param src Source int8 array
 * @param dst Destination int32 array (must be pre-allocated)
 * @param n Number of elements (must be multiple of 8)
 */
void convert_int8_to_int32_avx2(
    const int8_t* src,
    int32_t* dst,
    size_t n
);
#endif

/* ========================================================================== */
/* int32 Sparse Ternary FMA Functions                                        */
/* ========================================================================== */

/**
 * Sparse Ternary FMA: C = A * B + C (int32 scalar implementation)
 * 
 * @param A Dense coefficient array [N]
 * @param B_trit Packed ternary array (STFMA encoding) [N/4 bytes]
 * @param C Accumulator array [N] (modified in-place)
 * @param N Array length (must be multiple of 4)
 */
void sparse_ternary_fma_int32_scalar(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
);

#if defined(__AVX2__)
/**
 * Sparse Ternary FMA: C = A * B + C (int32 AVX2 implementation)
 * 
 * @param A Dense coefficient array [N]
 * @param B_trit Packed ternary array (STFMA encoding) [N/4 bytes]
 * @param C Accumulator array [N] (modified in-place)
 * @param N Array length (must be multiple of 8)
 */
void sparse_ternary_fma_int32_avx2(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
);
#endif

#if defined(__AVX512F__)
/**
 * Sparse Ternary FMA: C = A * B + C (int32 AVX-512 implementation)
 * 
 * @param A Dense coefficient array [N]
 * @param B_trit Packed ternary array (STFMA encoding) [N/4 bytes]
 * @param C Accumulator array [N] (modified in-place)
 * @param N Array length (must be multiple of 16)
 */
void sparse_ternary_fma_int32_avx512(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
);
#endif

/**
 * Sparse Ternary FMA: Automatic dispatch (int32)
 * 
 * @param A Dense coefficient array [N]
 * @param B_trit Packed ternary array (STFMA encoding) [N/4 bytes]
 * @param C Accumulator array [N] (modified in-place)
 * @param N Array length
 */
void sparse_ternary_fma_int32(
    const int32_t* A,
    const uint8_t* B_trit,
    int32_t* C,
    size_t N
);

/* ========================================================================== */
/* BitNet Integration Functions                                              */
/* ========================================================================== */

/**
 * Vector dot product using sparse-ternary-fma (drop-in replacement).
 * 
 * This function is a drop-in replacement for ggml_vec_dot_i2_i8_s that uses
 * the sparse-ternary-fma library for improved performance on supported hardware.
 * 
 * @param n Vector length
 * @param s Output scalar (dot product result)
 * @param bs Stride for x (unused)
 * @param vx Packed 2-bit ternary vector (BitNet encoding)
 * @param bx Unused
 * @param vy Dense int8 vector
 * @param by Unused
 * @param nrc Unused
 */
void ggml_vec_dot_i2_i8_stfma(
    int n,
    float* s,
    size_t bs,
    const void* vx,
    size_t bx,
    const void* vy,
    size_t by,
    int nrc
);

/* ========================================================================== */
/* Buffer Management                                                          */
/* ========================================================================== */

/**
 * Thread-local buffer structure for temporary allocations.
 */
struct stfma_thread_buffers {
    uint8_t* encoding_buffer;
    int32_t* int32_buffer;
    int32_t* accumulator_buffer;
    size_t buffer_size;
};

/**
 * Ensure thread-local buffers are large enough for the given size.
 * 
 * @param required_size Required buffer size (in elements)
 */
void stfma_ensure_buffer_size(size_t required_size);

/**
 * Free thread-local buffers.
 */
void stfma_free_buffers(void);

#ifdef __cplusplus
}
#endif
