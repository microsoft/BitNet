#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Minimal GGML tensor structure for our implementation
struct ggml_tensor {
    int64_t ne[4]; // dimensions
    void * data;   // data pointer
    void * extra;  // extra data
};

// Minimal GGML type enum
enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8 = 16,
    GGML_TYPE_I16 = 17,
    GGML_TYPE_I32 = 18,
    GGML_TYPE_COUNT = 19,
    GGML_TYPE_TL1 = 20,
};

// Minimal GGML API
void ggml_init(void);
int64_t ggml_nelements(const struct ggml_tensor * tensor);
size_t ggml_row_size(enum ggml_type type, int64_t n);

#ifdef __cplusplus
}
#endif
