#include "ggml.h"

void ggml_init(void) {
    // Stub implementation
}

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    if (!tensor) return 0;
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

size_t ggml_row_size(enum ggml_type type, int64_t n) {
    // Simple implementation for our needs
    switch (type) {
        case GGML_TYPE_F32: return n * sizeof(float);
        case GGML_TYPE_I8:  return n * sizeof(int8_t);
        default:            return n * sizeof(float); // Default to float
    }
}
