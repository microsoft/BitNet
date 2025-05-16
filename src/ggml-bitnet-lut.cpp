#include "ggml-backend.h"
#include <vector>
#include <type_traits>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "bitnet-lut-kernels.h"

#if defined(GGML_BITNET_ARM_TL1)

void ggml_bitnet_init(void) {
    // LOG(INFO) << "ggml_bitnet_init";

    if (initialized) {
        return;
    }
    initialized = true;

    // if (wrapper == nullptr) {
    //     wrapper = new BITNET::BITNETGeMMWrapper<bitnet_bitnet_float_type>();
    // }
    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    // LOG(INFO) << "ggml_bitnet_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    // delete wrapper;
    // wrapper = nullptr;
    for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
        // aligned_free(bitnet_tensor_extras[i].qweights);
        // aligned_free(bitnet_tensor_extras[i].scales);
    }
    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
        // Add additional args to decide if permuted I2 or naive I2
        return false;
    } else {
        return true;
    }
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        (!src0->buffer || ggml_backend_buft_is_host(ggml_backend_buffer_get_type(src0->buffer)))) {
        if (src1->ne[1] <= 1) {
            return true;
        }
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    const int bits = ggml_bitnet_get_type_bits(src0->type);
    
    size_t wsize = ne10 * ne11 * 15 * sizeof(int8_t) + 1 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL1:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}

#endif
#if defined(GGML_BITNET_X86_TL2)
void ggml_bitnet_init(void) {
    // LOG(INFO) << "ggml_bitnet_init";

    if (initialized) {
        return;
    }
    initialized = true;

    // if (wrapper == nullptr) {
    //     wrapper = new BITNET::BITNETGeMMWrapper<bitnet_bitnet_float_type>();
    // }
    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    // LOG(INFO) << "ggml_bitnet_free";

    if (!initialized) {
        return;
    }
    initialized = false;

    // delete wrapper;
    // wrapper = nullptr;
    for (size_t i = 0; i < bitnet_tensor_extras_index; i++) {
        // aligned_free(bitnet_tensor_extras[i].qweights);
        // aligned_free(bitnet_tensor_extras[i].scales);
    }
    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 &&
        (!src0->buffer || ggml_backend_buft_is_host(ggml_backend_buffer_get_type(src0->buffer)))) {
        return true;
    }
    return false;
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];
    
    size_t wsize = ne10 * ne11 * 11 * sizeof(int8_t) + 2 * ne11 * 2 * sizeof(bitnet_float_type);
    if (sizeof(bitnet_float_type) == 2) {
        // Need fp32 to fp16 conversion
        wsize += std::max(ne10, ne01) * ne11 * sizeof(bitnet_float_type);
    }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

int ggml_bitnet_get_type_bits(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_TL2:
            return 2;
        case GGML_TYPE_Q4_0:
            return 4;
        default:
            return 0;
    }
}
#endif

// Placeholder for ggml_bitnet_mul_mat_task_compute
// This function is critical for BitNet matrix multiplication and needs a real implementation.
void ggml_bitnet_mul_mat_task_compute(void * src0, void * scales, void * qlut, void * lut_scales, void * lut_biases, void * dst, int n, int k, int m, int bits) {
    (void)src0; (void)scales; (void)qlut; (void)lut_scales; (void)lut_biases; (void)dst; (void)n; (void)k; (void)m; (void)bits;
    // For now, it does nothing. A real implementation would perform matrix multiplication.
    // To prevent uninitialized data issues if this were called, one might zero out dst if its size is known.
    // However, without knowing the structure of dst (e.g., if it's a ggml_tensor or raw float array and its dimensions),
    // directly writing to it is risky. The parameters n, k, m, bits give clues.
    // Assuming dst is a float pointer and n, m are output dimensions (e.g. m rows, n cols)
    if (dst && m > 0 && n > 0) {
        float* out = (float*)dst;
        for (int i = 0; i < m * n; ++i) {
            out[i] = 0.0f; // Zero out destination as a safe default for a stub
        }
    }
    // Consider adding a printf here for debugging if this function gets called:
    // printf("Warning: ggml_bitnet_mul_mat_task_compute is a stub and has been called!\n");
}

// Placeholder for ggml_bitnet_transform_tensor
// This function is critical for preparing BitNet tensors and needs a real implementation.
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    (void)tensor;
    // For now, it does nothing. A real implementation would transform the tensor data/metadata.
    // Consider adding a printf here for debugging if this function gets called:
    // printf("Warning: ggml_bitnet_transform_tensor is a stub and has been called!\n");
}

