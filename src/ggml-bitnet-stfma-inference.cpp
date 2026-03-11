/**
 * BitNet Sparse Ternary FMA - Cached Inference Implementation
 * 
 * This file implements the inference path with load-time caching.
 * Weights are converted ONCE at model load and cached in memory.
 * During inference, we use zero-cost pointer lookups.
 * 
 * Copyright 2025 HyperFold Technologies UK Ltd
 * Licensed under the Apache License, Version 2.0
 */

#include "ggml-bitnet-stfma.h"
#include "ggml-bitnet-stfma-cache.h"
#include "ggml-bitnet-stfma-avx512.h"
#include <string.h>

/**
 * Cached inference function for ggml_vec_dot_i2_i8_s
 * 
 * This function assumes weights have been pre-converted and cached.
 * It performs zero conversions during inference.
 * 
 * @param n Number of elements
 * @param s Output: dot product result
 * @param vx Cached STFMA weights handle
 * @param vy int8 activations
 */
void ggml_vec_dot_i2_i8_s_stfma_cached(
    int n,
    float* s,
    ggml_bitnet_stfma_cache_handle vx_handle,
    const void* vy
) {
    // Get pre-converted weights (zero-cost pointer lookup)
    const uint8_t* stfma_weights = ggml_bitnet_stfma_get_cached_weights(vx_handle);
    if (!stfma_weights) {
        *s = 0.0f;
        return;
    }
    
    const int8_t* activations_i8 = (const int8_t*)vy;
    
    // Ensure we have buffer space for int32 activations
    stfma_ensure_buffer_size(n);
    
    // Convert activations from int8 to int32 (vectorized)
#if defined(__AVX2__)
    convert_i8_to_i32_avx2(activations_i8, tl_buffers.int32_buffer, n);
#else
    // Scalar fallback
    for (int i = 0; i < n; i++) {
        tl_buffers.int32_buffer[i] = (int32_t)activations_i8[i];
    }
#endif
    
    // Compute dot product using fully vectorized AVX-512 kernel
#if defined(__AVX512F__)
    int32_t result = ggml_bitnet_stfma_dense_avx512_tail(
        stfma_weights,
        tl_buffers.int32_buffer,
        n
    );
#elif defined(__AVX2__)
    // AVX2 fallback (to be implemented)
    int32_t result = 0;
    for (int i = 0; i < n; i++) {
        // Decode trit from packed format
        int byte_idx = i / 4;
        int trit_idx = i % 4;
        uint8_t byte = stfma_weights[byte_idx];
        uint8_t trit = (byte >> (trit_idx * 2)) & 0x3;
        
        // Convert to signed: 0→-1, 1→0, 2→+1
        int32_t weight = (int32_t)trit - 1;
        result += weight * tl_buffers.int32_buffer[i];
    }
#else
    // Scalar fallback
    int32_t result = 0;
    for (int i = 0; i < n; i++) {
        int byte_idx = i / 4;
        int trit_idx = i % 4;
        uint8_t byte = stfma_weights[byte_idx];
        uint8_t trit = (byte >> (trit_idx * 2)) & 0x3;
        int32_t weight = (int32_t)trit - 1;
        result += weight * tl_buffers.int32_buffer[i];
    }
#endif
    
    *s = (float)result;
}

/**
 * Hybrid inference function that supports both cached and non-cached paths
 * 
 * This function checks if weights are cached. If so, uses the cached path.
 * Otherwise, falls back to JIT conversion (for backward compatibility).
 * 
 * @param n Number of elements
 * @param s Output: dot product result
 * @param vx BitNet weights (either raw or cached handle)
 * @param vy int8 activations
 * @param use_cache Whether to use cached weights
 */
void ggml_vec_dot_i2_i8_s_stfma_hybrid(
    int n,
    float* s,
    const void* vx,
    const void* vy,
    bool use_cache
) {
    if (use_cache) {
        // Cast to cache handle and use cached path
        ggml_bitnet_stfma_cache_handle handle = 
            (ggml_bitnet_stfma_cache_handle)vx;
        ggml_vec_dot_i2_i8_s_stfma_cached(n, s, handle, vy);
    } else {
        // Fall back to JIT conversion (original implementation)
        // This path should rarely be used in production
        const uint8_t* bitnet_weights = (const uint8_t*)vx;
        const int8_t* activations_i8 = (const int8_t*)vy;
        
        // Ensure buffers
        stfma_ensure_buffer_size(n);
        
        // Convert weights (JIT - expensive!)
        size_t num_bytes = (n + 3) / 4;
        convert_bitnet_to_stfma_array(
            bitnet_weights,
            tl_buffers.encoding_buffer,
            num_bytes
        );
        
        // Convert activations
#if defined(__AVX2__)
        convert_i8_to_i32_avx2(activations_i8, tl_buffers.int32_buffer, n);
#else
        for (int i = 0; i < n; i++) {
            tl_buffers.int32_buffer[i] = (int32_t)activations_i8[i];
        }
#endif
        
        // Compute
#if defined(__AVX512F__)
        int32_t result = ggml_bitnet_stfma_dense_avx512_tail(
            tl_buffers.encoding_buffer,
            tl_buffers.int32_buffer,
            n
        );
#else
        int32_t result = 0;
        for (int i = 0; i < n; i++) {
            int byte_idx = i / 4;
            int trit_idx = i % 4;
            uint8_t byte = tl_buffers.encoding_buffer[byte_idx];
            uint8_t trit = (byte >> (trit_idx * 2)) & 0x3;
            int32_t weight = (int32_t)trit - 1;
            result += weight * tl_buffers.int32_buffer[i];
        }
#endif
        
        *s = (float)result;
    }
}

/**
 * Get cache statistics for monitoring
 */
void ggml_bitnet_stfma_get_cache_stats(
    size_t* num_cached_tensors,
    size_t* total_cached_bytes,
    float* memory_overhead_ratio
) {
    size_t num_entries, total_bytes;
    ggml_bitnet_stfma_cache_stats(&num_entries, &total_bytes);
    
    if (num_cached_tensors) *num_cached_tensors = num_entries;
    if (total_cached_bytes) *total_cached_bytes = total_bytes;
    
    // Memory overhead is 100% (we store both original and converted)
    if (memory_overhead_ratio) *memory_overhead_ratio = 1.0f;
}
