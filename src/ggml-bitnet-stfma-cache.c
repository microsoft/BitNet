#include "ggml-bitnet-stfma-cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// Branchless conversion function (from previous optimization)
static inline uint8_t convert_bitnet_to_stfma_byte(uint8_t b) {
    uint8_t low_bits = b & 0x55;
    uint8_t high_bits = b & 0xAA;
    uint8_t out_low = (high_bits >> 1);
    uint8_t high_bits_shifted = (high_bits >> 1);
    uint8_t xor_result = high_bits_shifted ^ low_bits;
    uint8_t out_high = (~xor_result) & 0x55;
    out_high = out_high << 1;
    return out_high | out_low;
}

// Cache entry structure
struct ggml_bitnet_stfma_cache_entry {
    uint8_t* stfma_weights;
    size_t size_bytes;
    struct ggml_bitnet_stfma_cache_entry* next;
};

// Global cache state
static struct {
    struct ggml_bitnet_stfma_cache_entry* head;
    size_t num_entries;
    size_t total_bytes;
} g_cache = {NULL, 0, 0};

void ggml_bitnet_stfma_cache_init(void) {
    g_cache.head = NULL;
    g_cache.num_entries = 0;
    g_cache.total_bytes = 0;
}

ggml_bitnet_stfma_cache_handle ggml_bitnet_stfma_cache_weights(
    const uint8_t* bitnet_weights,
    size_t n
) {
    if (!bitnet_weights || n == 0) {
        return NULL;
    }
    
    // Allocate cache entry
    struct ggml_bitnet_stfma_cache_entry* entry = 
        malloc(sizeof(struct ggml_bitnet_stfma_cache_entry));
    if (!entry) {
        return NULL;
    }
    
    // Calculate size: n elements = n/4 bytes (2 bits per element)
    size_t size_bytes = (n + 3) / 4;
    
    // Allocate memory for converted weights
    entry->stfma_weights = malloc(size_bytes);
    if (!entry->stfma_weights) {
        free(entry);
        return NULL;
    }
    
    entry->size_bytes = size_bytes;
    
    // Convert all weights using branchless conversion
    // This happens ONCE at load time
    for (size_t i = 0; i < size_bytes; i++) {
        entry->stfma_weights[i] = convert_bitnet_to_stfma_byte(bitnet_weights[i]);
    }
    
    // Add to cache linked list
    entry->next = g_cache.head;
    g_cache.head = entry;
    g_cache.num_entries++;
    g_cache.total_bytes += size_bytes;
    
    return entry;
}

const uint8_t* ggml_bitnet_stfma_get_cached_weights(
    ggml_bitnet_stfma_cache_handle handle
) {
    if (!handle) {
        return NULL;
    }
    return handle->stfma_weights;
}

void ggml_bitnet_stfma_free_cached_weights(
    ggml_bitnet_stfma_cache_handle handle
) {
    if (!handle) {
        return;
    }
    
    // Remove from linked list
    struct ggml_bitnet_stfma_cache_entry** curr = &g_cache.head;
    while (*curr) {
        if (*curr == handle) {
            *curr = handle->next;
            g_cache.num_entries--;
            g_cache.total_bytes -= handle->size_bytes;
            break;
        }
        curr = &(*curr)->next;
    }
    
    // Free memory
    free(handle->stfma_weights);
    free(handle);
}

void ggml_bitnet_stfma_cache_shutdown(void) {
    struct ggml_bitnet_stfma_cache_entry* curr = g_cache.head;
    while (curr) {
        struct ggml_bitnet_stfma_cache_entry* next = curr->next;
        free(curr->stfma_weights);
        free(curr);
        curr = next;
    }
    
    g_cache.head = NULL;
    g_cache.num_entries = 0;
    g_cache.total_bytes = 0;
}

void ggml_bitnet_stfma_cache_stats(size_t* num_entries, size_t* total_bytes) {
    if (num_entries) {
        *num_entries = g_cache.num_entries;
    }
    if (total_bytes) {
        *total_bytes = g_cache.total_bytes;
    }
}
