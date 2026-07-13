#include <vector>
#include <type_traits>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __x86_64__
#include <immintrin.h>
#endif

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "ggml-cpu-impl.h"

#if defined(GGML_BITNET_ARM_TL1) || defined(GGML_BITNET_X86_TL2)
#include "bitnet-lut-kernels.h"
#endif

#if defined(GGML_BITNET_ARM_TL1)

void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

static bool do_permutate(enum ggml_type type) {
    if (type == GGML_TYPE_TL1) {
        return false;
    } else {
        return true;
    }
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
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

void ggml_bitnet_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const size_t ne00 = src0->ne[0];
    const size_t ne01 = src0->ne[1];
    const size_t ne10 = src1->ne[0];
    const size_t ne11 = src1->ne[1];

    const int ith = params->ith;
    const int nth = params->nth;
    const int bits = ggml_bitnet_get_type_bits(src0->type);

    struct bitnet_tensor_extra * extra = (struct bitnet_tensor_extra *)src0->extra;
    GGML_ASSERT(extra != nullptr);

    char * wdata = (char *)params->wdata;
    const size_t wsize_per_thread = ggml_bitnet_mul_mat_get_wsize(src0, src1, dst);

    int8_t  * qlut  = (int8_t  *)(wdata);
    bitnet_float_type * lut_scales = (bitnet_float_type *)(qlut + ne10 * ne11 * 15);
    bitnet_float_type * lut_biases = (bitnet_float_type *)(lut_scales + ne11);

    if (ith == 0) {
        ggml_bitnet_mul_mat_task_init(
            (void *)((char *)src1->data),
            (void *)qlut,
            (void *)lut_scales,
            (void *)lut_biases,
            ne10, ne00, ne11, bits);
    }

    // barrier
    if (nth > 1) {
        ggml_barrier(params->threadpool);
    }

    ggml_bitnet_mul_mat_task_compute(
        (void *)extra->qweights,
        (void *)extra->scales,
        (void *)qlut,
        (void *)lut_scales,
        (void *)lut_biases,
        (void *)((char *)dst->data),
        ne10, ne00, ne11, bits);
}

#endif
#if defined(GGML_BITNET_X86_TL2)
void ggml_bitnet_init(void) {
    if (initialized) {
        return;
    }
    initialized = true;

    if (bitnet_tensor_extras == nullptr) {
        bitnet_tensor_extras = new bitnet_tensor_extra[GGML_BITNET_MAX_NODES];
    }
    bitnet_tensor_extras_index = 0;
}

void ggml_bitnet_free(void) {
    if (!initialized) {
        return;
    }
    initialized = false;

    delete[] bitnet_tensor_extras;
    bitnet_tensor_extras = nullptr;
}

bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    if ((is_type_supported(src0->type)) &&
        src1->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
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
