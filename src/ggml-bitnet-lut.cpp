#include <vector>
#include <type_traits>

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
        src0->backend == GGML_BACKEND_TYPE_CPU) {
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
        src0->backend == GGML_BACKEND_TYPE_CPU) {
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