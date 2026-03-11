#ifndef GGML_BITNET_STFMA_CACHE_H
#define GGML_BITNET_STFMA_CACHE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle to a cached weight tensor
 */
typedef struct ggml_bitnet_stfma_cache_entry* ggml_bitnet_stfma_cache_handle;

/**
 * @brief Initialize the weight caching system
 * 
 * This should be called once during model loading.
 */
void ggml_bitnet_stfma_cache_init(void);

/**
 * @brief Convert and cache a weight tensor at load time
 * 
 * @param bitnet_weights Pointer to BitNet 2-bit encoded weights
 * @param n Number of elements
 * @return Handle to cached weights, or NULL on failure
 * 
 * This function:
 * 1. Converts BitNet encoding to STFMA encoding (branchless)
 * 2. Allocates persistent memory for the converted weights
 * 3. Returns a handle that can be used during inference
 * 
 * The conversion happens ONCE at load time, not per-inference.
 */
ggml_bitnet_stfma_cache_handle ggml_bitnet_stfma_cache_weights(
    const uint8_t* bitnet_weights,
    size_t n
);

/**
 * @brief Get pointer to cached STFMA-encoded weights
 * 
 * @param handle Handle returned by ggml_bitnet_stfma_cache_weights
 * @return Pointer to STFMA-encoded weights (read-only)
 */
const uint8_t* ggml_bitnet_stfma_get_cached_weights(
    ggml_bitnet_stfma_cache_handle handle
);

/**
 * @brief Free a cached weight tensor
 * 
 * @param handle Handle to free
 */
void ggml_bitnet_stfma_free_cached_weights(
    ggml_bitnet_stfma_cache_handle handle
);

/**
 * @brief Free all cached weights and shutdown the caching system
 * 
 * This should be called during model unloading.
 */
void ggml_bitnet_stfma_cache_shutdown(void);

/**
 * @brief Get statistics about the cache
 * 
 * @param num_entries Output: number of cached weight tensors
 * @param total_bytes Output: total memory used by cache
 */
void ggml_bitnet_stfma_cache_stats(size_t* num_entries, size_t* total_bytes);

#ifdef __cplusplus
}
#endif

#endif // GGML_BITNET_STFMA_CACHE_H
