import argparse
import os
from configparser import ConfigParser

def gen_ctor_code():
    kernel_code = "\n\
#include \"ggml-bitnet.h\"\n\
#include <cstring>\n\
#include <immintrin.h>\n\
#define GGML_BITNET_MAX_NODES 8192\n\
static bool initialized = false;\n\
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;\n\
static size_t bitnet_tensor_extras_index = 0;\n\
static void * aligned_malloc(size_t size) {\n\
#if defined(_WIN32)\n\
    return _aligned_malloc(size, 64);\n\
#else\n\
    void * ptr = nullptr;\n\
    posix_memalign(&ptr, 64, size);\n\
    return ptr;\n\
#endif\n\
}\n\
\n\
static void aligned_free(void * ptr) {\n\
#if defined(_WIN32)\n\
    _aligned_free(ptr);\n\
#else\n\
    free(ptr);\n\
#endif\n\
}\n\
#define BK2 32\n\
#if defined __AVX2__\n\
inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));\n\
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));\n\
    *vl = _mm256_unpacklo_epi32(va, vb);\n\
    *vh = _mm256_unpackhi_epi32(va, vb);\n\
}\n\
inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));\n\
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));\n\
    *vl = _mm256_unpacklo_epi64(va, vb);\n\
    *vh = _mm256_unpackhi_epi64(va, vb);\n\
}\n\
inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)\n\
{\n\
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));\n\
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));\n\
}\n\
inline void Transpose_8_8(\n\
    __m256i *v0,\n\
    __m256i *v1,\n\
    __m256i *v2,\n\
    __m256i *v3,\n\
    __m256i *v4,\n\
    __m256i *v5,\n\
    __m256i *v6,\n\
    __m256i *v7)\n\
{\n\
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;\n\
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;\n\
    _mm256_merge_epi32(*v0, *v1, &w0, &w1);\n\
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);\n\
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);\n\
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);\n\
    _mm256_merge_epi64(w0, w2, &x0, &x1);\n\
    _mm256_merge_epi64(w1, w3, &x2, &x3);\n\
    _mm256_merge_epi64(w4, w6, &x4, &x5);\n\
    _mm256_merge_epi64(w5, w7, &x6, &x7);\n\
    _mm256_merge_si128(x0, x4, v0, v1);\n\
    _mm256_merge_si128(x1, x5, v2, v3);\n\
    _mm256_merge_si128(x2, x6, v4, v5);\n\
    _mm256_merge_si128(x3, x7, v6, v7);\n\
}\n\
#endif\n\
inline int32_t per_tensor_quant(int k, void* lut_scales_, void* b_) {\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    bitnet_float_type* b = (bitnet_float_type*)b_;\n\
#if defined __AVX2__\n\
    __m256 max_vec = _mm256_set1_ps(0.f);\n\
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);\n\
    for (int i = 0; i < k / 8; i++) {\n\
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);\n\
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);\n\
        max_vec = _mm256_max_ps(vec_babs, max_vec);\n\
    }\n\
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));\n\
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));\n\
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));\n\
    float scales = 127 / _mm_cvtss_f32(max1);\n\
    *lut_scales = scales;\n\
#endif\n\
    return 0;\n\
}\n\
inline int32_t partial_max_reset(int32_t bs, void* lut_scales_) {\n\
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;\n\
    #pragma unroll\n\
    for (int i=0; i< bs; i++) {\n\
        lut_scales[i] = 0.0;\n\
    }\n\
    return 0;\n\
}\n\
template<int act_k>\n\
inline int32_t three_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {\n\
#if defined __AVX2__\n\
    __m256i vec_lut[16];\n\
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);\n\
    float scales = *lut_scales;\n\
    __m256i shuffle_mask = _mm256_set_epi8(\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00\n\
                                            );\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 24; ++k) {\n\
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 24 + 0, vec_bi, 1);\n\
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 24 + 1, vec_bi, 1);\n\
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 24 + 2, vec_bi, 1);\n\
\n\
        __m256i vec_b0i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        __m256i vec_b1i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        __m256i vec_b2i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b2, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
\n\
        vec_lut[15] = _mm256_setzero_si256();\n\
        vec_lut[14] = _mm256_setzero_si256();\n\
        vec_lut[13] = vec_b0i;\n\
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b1i);\n\
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b2i);\n\
        vec_lut[12] = vec_b0i;\n\
        vec_lut[12] = _mm256_add_epi32(vec_lut[12], vec_b1i);\n\
        vec_lut[11] = vec_b0i;\n\
        vec_lut[11] = _mm256_add_epi32(vec_lut[11], vec_b1i);\n\
        vec_lut[11] = _mm256_sub_epi32(vec_lut[11], vec_b2i);\n\
        vec_lut[10] = vec_b0i;\n\
        vec_lut[10] = _mm256_add_epi32(vec_lut[10], vec_b2i);\n\
        vec_lut[9] = vec_b0i;\n\
        vec_lut[8] = vec_b0i;\n\
        vec_lut[8] = _mm256_sub_epi32(vec_lut[8], vec_b2i);\n\
        vec_lut[7] = vec_b0i;\n\
        vec_lut[7] = _mm256_sub_epi32(vec_lut[7], vec_b1i);\n\
        vec_lut[7] = _mm256_add_epi32(vec_lut[7], vec_b2i);\n\
        vec_lut[6] = vec_b0i;\n\
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1i);\n\
        vec_lut[5] = vec_b0i;\n\
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b1i);\n\
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b2i);\n\
        vec_lut[4] = vec_b1i;\n\
        vec_lut[4] = _mm256_add_epi32(vec_lut[4], vec_b2i);\n\
        vec_lut[3] = vec_b1i;\n\
        vec_lut[2] = vec_b1i;\n\
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b2i);\n\
        vec_lut[1] = vec_b2i;\n\
        vec_lut[0] = _mm256_setzero_si256();\n\
        __m256i ix[16];\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 16; ++g) {\n\
            ix[g] = vec_lut[g];\n\
        }\n\
\n\
        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));\n\
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 8; ++g) {\n\
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
        }\n\
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);\n\
\n\
    }\n\
\n\
    *lut_scales = scales;\n\
#endif\n\
    return 0;\n\
}\n\
\n\
template<int act_k>\n\
inline int32_t two_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {\n\
#if defined __AVX2__\n\
    __m256i vec_lut[16];\n\
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);\n\
    float scales = *lut_scales;\n\
    __m256i shuffle_mask = _mm256_set_epi8(\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,\n\
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,\n\
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00\n\
                                            );\n\
#pragma unroll\n\
    for (int k = 0; k < act_k / 16; ++k) {\n\
        __m256 vec_b0f = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);\n\
        __m256 vec_b1f = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);\n\
\n\
        __m256i vec_b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        __m256i vec_b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));\n\
        vec_lut[15] = _mm256_setzero_si256();\n\
        vec_lut[14] = _mm256_setzero_si256();\n\
        vec_lut[13] = _mm256_setzero_si256();\n\
        vec_lut[12] = _mm256_setzero_si256();\n\
        vec_lut[11] = _mm256_setzero_si256();\n\
        vec_lut[10] = _mm256_setzero_si256();\n\
        vec_lut[9] = _mm256_setzero_si256();\n\
        vec_lut[8] = vec_b0;\n\
        vec_lut[8] = _mm256_add_epi32(vec_lut[8], vec_b1);\n\
        vec_lut[7] = vec_b0;\n\
        vec_lut[6] = vec_b0;\n\
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1);\n\
        vec_lut[5] = vec_b1;\n\
        vec_lut[4] = _mm256_setzero_si256();\n\
        vec_lut[3] = _mm256_setzero_si256();\n\
        vec_lut[3] = _mm256_sub_epi32(vec_lut[3], vec_b1);\n\
        vec_lut[2] = _mm256_setzero_si256();\n\
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b0);\n\
        vec_lut[2] = _mm256_add_epi32(vec_lut[2], vec_b1);\n\
        vec_lut[1] = _mm256_setzero_si256();\n\
        vec_lut[1] = _mm256_sub_epi32(vec_lut[1], vec_b0);\n\
        vec_lut[0] = _mm256_setzero_si256();\n\
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b0);\n\
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b1);\n\
\n\
        __m256i ix[16];\n\
#pragma unroll\n\
        for (int g = 0; g < 16; ++g) {\n\
            ix[g] = vec_lut[g];\n\
        }\n\
\n\
        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));\n\
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));\n\
\n\
#pragma unroll\n\
        for (int g = 0; g < 8; ++g) {\n\
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);\n\
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));\n\
        }\n\
\n\
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);\n\
\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);\n\
\n\
    }\n\
    *lut_scales = scales;\n\
#endif\n\
    return 0;\n\
}\n\
static bool is_type_supported(enum ggml_type type) {\n\
    if (type == GGML_TYPE_Q4_0 ||\n\
        type == GGML_TYPE_TL2) {\n\
        return true;\n\
    } else {\n\
        return false;\n\
    }\n\
}\n\
"
    return kernel_code

def gen_tbl_impl(pre, BM, BK, bm, k_list):

    kernel_code = "\
#include <immintrin.h>\n\
\n\
#define BM{0} {1}\n\
#define BBK{0} {2}\n\
template<int batch_size, int K3>\n\
inline void three_tbl_impl_{0}(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {{\n\
".format(pre, BM, BK)

    kernel_code = "".join([kernel_code, "\
#ifdef __AVX2__\n\
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);\n\
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);\n\
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);\n\
    const __m256i vec_one  = _mm256_set1_epi8(0xff);\n\
    const int KK = BBK{0} / 3;\n\
#pragma unroll\n\
        for (int i = 0; i < BM{0}; i += 32) {{\n\
        __m256i vec_as[KK / 2];\n\
        __m256i vec_signs[KK / 8];\n\
        #pragma unroll\n\
        for (int ai = 0; ai < KK / 2; ai++) {{\n\
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));\n\
        }}\n\
        #pragma unroll\n\
        for (int as = 0; as < KK / 8; as++) {{\n\
            vec_signs[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));\n\
        }}\n\
#pragma unroll\n\
    for (int bs = 0; bs < batch_size; bs++) {{\n\
        __m256i vec_c0 = _mm256_setzero_si256();\n\
        __m256i vec_c1 = _mm256_setzero_si256();\n\
#pragma unroll\n\
        for (int k = 0; k < KK / 8; k++) {{\n\
            __m256i vec_sign = vec_signs[k];\n\
                __m256i vec_a_0 = vec_as[k * 4 + 0];\n\
                __m128i vec_k1_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0)), 15);\n\
                __m256i vec_sign_left_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 1)), 15);\n\
                __m256i vec_v_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), vec_mask);\n\
                __m256i vec_v_top_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_0, vec_k1_0), vec_v_top_0);\n\
                __m256i vec_v_top_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_0, vec_k2_0), vec_v_top_0);\n\
                __m256i vec_sign_right_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 2)), 15);\n\
                __m256i vec_sign_right_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 3)), 15);\n\
                __m256i vec_v_bot_0 = _mm256_and_si256(vec_a_0, vec_mask);\n\
                __m256i vec_v_bot_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_0, vec_k3_0), vec_v_bot_0);\n\
                __m256i vec_v_bot_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_0, vec_k4_0), vec_v_bot_0);\n\
                __m256i vec_v_top_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_lo_0), vec_sign_left_lo_0);\n\
                __m256i vec_v_top_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_hi_0), vec_sign_left_hi_0);\n\
                __m256i vec_v_bot_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_lo_0), vec_sign_right_lo_0);\n\
                __m256i vec_v_bot_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_hi_0), vec_sign_right_hi_0);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_0);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_0);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_0);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_0);\n\
                __m256i vec_a_1 = vec_as[k * 4 + 1];\n\
                __m128i vec_k1_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1)), 15);\n\
                __m256i vec_sign_left_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 1)), 15);\n\
                __m256i vec_v_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), vec_mask);\n\
                __m256i vec_v_top_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_1, vec_k1_1), vec_v_top_1);\n\
                __m256i vec_v_top_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_1, vec_k2_1), vec_v_top_1);\n\
                __m256i vec_sign_right_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 2)), 15);\n\
                __m256i vec_sign_right_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 3)), 15);\n\
                __m256i vec_v_bot_1 = _mm256_and_si256(vec_a_1, vec_mask);\n\
                __m256i vec_v_bot_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_1, vec_k3_1), vec_v_bot_1);\n\
                __m256i vec_v_bot_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_1, vec_k4_1), vec_v_bot_1);\n\
                __m256i vec_v_top_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_lo_1), vec_sign_left_lo_1);\n\
                __m256i vec_v_top_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_hi_1), vec_sign_left_hi_1);\n\
                __m256i vec_v_bot_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_lo_1), vec_sign_right_lo_1);\n\
                __m256i vec_v_bot_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_hi_1), vec_sign_right_hi_1);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_1);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_1);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_1);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_1);\n\
                __m256i vec_a_2 = vec_as[k * 4 + 2];\n\
                __m128i vec_k1_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2)), 15);\n\
                __m256i vec_sign_left_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 1)), 15);\n\
                __m256i vec_v_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), vec_mask);\n\
                __m256i vec_v_top_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_2, vec_k1_2), vec_v_top_2);\n\
                __m256i vec_v_top_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_2, vec_k2_2), vec_v_top_2);\n\
                __m256i vec_sign_right_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 2)), 15);\n\
                __m256i vec_sign_right_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 3)), 15);\n\
                __m256i vec_v_bot_2 = _mm256_and_si256(vec_a_2, vec_mask);\n\
                __m256i vec_v_bot_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_2, vec_k3_2), vec_v_bot_2);\n\
                __m256i vec_v_bot_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_2, vec_k4_2), vec_v_bot_2);\n\
                __m256i vec_v_top_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_lo_2), vec_sign_left_lo_2);\n\
                __m256i vec_v_top_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_hi_2), vec_sign_left_hi_2);\n\
                __m256i vec_v_bot_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_lo_2), vec_sign_right_lo_2);\n\
                __m256i vec_v_bot_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_hi_2), vec_sign_right_hi_2);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_2);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_2);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_2);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_2);\n\
                __m256i vec_a_3 = vec_as[k * 4 + 3];\n\
                __m128i vec_k1_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 0  + K3 / 3 * 32 * bs));\n\
                __m128i vec_k2_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 16 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k3_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 32 + K3 / 3 * 32 * bs));\n\
                __m128i vec_k4_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 48 + K3 / 3 * 32 * bs));\n\
                __m256i vec_sign_left_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3)), 15);\n\
                __m256i vec_sign_left_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 1)), 15);\n\
                __m256i vec_v_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), vec_mask);\n\
                __m256i vec_v_top_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_3, vec_k1_3), vec_v_top_3);\n\
                __m256i vec_v_top_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_3, vec_k2_3), vec_v_top_3);\n\
                __m256i vec_sign_right_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 2)), 15);\n\
                __m256i vec_sign_right_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 3)), 15);\n\
                __m256i vec_v_bot_3 = _mm256_and_si256(vec_a_3, vec_mask);\n\
                __m256i vec_v_bot_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_3, vec_k3_3), vec_v_bot_3);\n\
                __m256i vec_v_bot_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_3, vec_k4_3), vec_v_bot_3);\n\
                __m256i vec_v_top_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_lo_3), vec_sign_left_lo_3);\n\
                __m256i vec_v_top_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_hi_3), vec_sign_left_hi_3);\n\
                __m256i vec_v_bot_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_lo_3), vec_sign_right_lo_3);\n\
                __m256i vec_v_bot_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_hi_3), vec_sign_right_hi_3);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_3);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_3);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_3);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_3);\n\
        }}\n\
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM{0} * bs));\n\
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{0} * bs));\n\
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{0} * bs));\n\
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{0} * bs));\n\
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n\
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n\
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));\n\
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM{0} * bs), vec_gc0);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{0} * bs), vec_gc1);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{0} * bs), vec_gc2);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{0} * bs), vec_gc3);\n\
    }}\n\
    }}\n\
#endif\n\
}}\n\
\n\
template<int batch_size, int K2>\n\
inline int32_t two_tbl_impl{0}(int32_t* c, int8_t* lut, uint8_t* a) {{\n\
#ifdef __AVX2__\n\
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);\n\
    const int KK = BK2 / 2;\n\
#pragma unroll\n\
    for (int i = 0; i < BM{0}; i += 32) {{\n\
        __m256i vec_as[KK / 2];\n\
        #pragma unroll\n\
        for (int ai = 0; ai < KK / 2; ai++) {{\n\
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));\n\
        }}\n\
#pragma unroll\n\
    for (int bs = 0; bs < batch_size; bs++) {{\n\
        __m256i vec_c0 = _mm256_setzero_si256();\n\
        __m256i vec_c1 = _mm256_setzero_si256();\n\
#pragma unroll\n\
        for (int k = 0; k < KK / 8; k++) {{\n\
            #pragma unroll\n\
            for (int j = 0; j < 4; j++) {{\n\
                __m256i vec_a = vec_as[k * 4 + j];\n\
\n\
                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));\n\
                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));\n\
                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));\n\
                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));\n\
\n\
                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);\n\
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);\n\
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);\n\
\n\
                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);\n\
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);\n\
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);\n\
\n\
                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);\n\
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);\n\
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);\n\
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);\n\
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);\n\
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); \n\
            }}\n\
        }}\n\
\n\
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM{0} * bs));\n\
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{0} * bs));\n\
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{0} * bs));\n\
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{0} * bs));\n\
\n\
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));\n\
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));\n\
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));\n\
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));\n\
\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM{0} * bs), vec_gc0);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM{0} * bs), vec_gc1);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM{0} * bs), vec_gc2);\n\
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM{0} * bs), vec_gc3);\n\
    }}\n\
    }}\n\
#endif\n\
    return 0;\n\
}}\n\
\n\
template<int BATCH_SIZE>\n\
int32_t three_qgemm_lut_{0}(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    alignas(32) uint32_t CBits[BATCH_SIZE * BM{0}];\n\
    memset(&(CBits[0]), 0, BATCH_SIZE * BM{0} * sizeof(int32_t));\n\
#pragma unroll\n\
    for (int32_t k_outer = 0; k_outer < {1} / BBK{0}; ++k_outer) {{\n\
        three_tbl_impl_{0}<BATCH_SIZE, {1}>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK{0} / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK{0} / 3 / 2 * BM{0})])), (&(((uint8_t*)sign)[(k_outer * BBK{0} / 3 / 8 * BM{0})])));\n\
    }}\n\
#pragma unroll\n\
    for (int bs = 0; bs < BATCH_SIZE; bs++) {{\n\
#pragma unroll\n\
        for (int i = 0; i < BM{0}; i++) {{\n\
            ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM{0}]);\n\
        }}\n\
  }}\n\
  return 0;\n\
}}\n\
\n\
template<int BATCH_SIZE>\n\
int32_t two_qgemm_lut_{0}(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    alignas(32) uint32_t CBits[BATCH_SIZE * BM{0}];\n\
    memset(&(CBits[0]), 0, BATCH_SIZE * BM{0} * sizeof(int32_t));\n\
#pragma unroll\n\
    for (int32_t k_outer = 0; k_outer < {2} / 32; ++k_outer) {{\n\
        two_tbl_impl{0}<BATCH_SIZE, {2}>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM{0})])));\n\
    }}\n\
#pragma unroll\n\
    for (int bs = 0; bs < BATCH_SIZE; bs++) {{\n\
#pragma unroll\n\
        for (int i = 0; i < BM{0}; i++) {{\n\
            ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BM{0}]);\n\
            ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];\n\
        }}\n\
    }}\n\
  return 0;\n\
}}\n\
\n\
".format(pre, k_list[1], k_list[0])])
    return kernel_code

def gen_top_api(kernel_shapes, k_list):

    kernel_code = "void ggml_preprocessor(int bs, int m, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT) {{\n\
    partial_max_reset(bs, (&(((float*)LUT_Scales)[0])));\n\
    if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + {2}])), (&(((float*)LUT_Scales)[b])));\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], k_list[0][0], k_list[0][1])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && two_k == {1} && three_k == {2}) {{\n\
        for (int32_t b = 0; b < bs; b++) {{\n\
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));\n\
            three_lut_ctor<{2}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));\n\
            two_lut_ctor<{1}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + {2}])), (&(((float*)LUT_Scales)[b])));\n\
        }}\n\
    }}\n".format(kernel_shapes[i][0], k_list[i][0], k_list[i][1])])
    kernel_code = "".join([kernel_code, "}\n"])


    kernel_code = "".join([kernel_code, "void ggml_qgemm_lut(int bs, int m, int k, int BK, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {{\n\
    if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 8) {{\n\
                two_qgemm_lut_{4}<8>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 32) {{\n\
                two_qgemm_lut_{4}<32>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 128) {{\n\
                two_qgemm_lut_{4}<128>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 256) {{\n\
                two_qgemm_lut_{4}<256>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 512) {{\n\
                two_qgemm_lut_{4}<512>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 8) {{\n\
                three_qgemm_lut_{4}<8>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 32) {{\n\
                three_qgemm_lut_{4}<32>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 128) {{\n\
                three_qgemm_lut_{4}<128>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 256) {{\n\
                three_qgemm_lut_{4}<256>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 512) {{\n\
                three_qgemm_lut_{4}<512>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[0][0], kernel_shapes[0][1], k_list[0][0], k_list[0][1], "{}_{}".format(kernel_shapes[0][0], kernel_shapes[0][1]))])
    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "    else if (m == {0} && k == {1}) {{\n\
        if (BK == {2}) {{\n\
            if (bs == 1) {{\n\
                two_qgemm_lut_{4}<1>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 8) {{\n\
                two_qgemm_lut_{4}<8>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 32) {{\n\
                two_qgemm_lut_{4}<32>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 128) {{\n\
                two_qgemm_lut_{4}<128>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 256) {{\n\
                two_qgemm_lut_{4}<256>(A, LUT, Scales, LUT_Scales, C);\n\
            }} else if (bs == 512) {{\n\
                two_qgemm_lut_{4}<512>(A, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
        else if (BK == {3}) {{\n\
            if (bs == 1) {{\n\
                three_qgemm_lut_{4}<1>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 8) {{\n\
                three_qgemm_lut_{4}<8>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 32) {{\n\
                three_qgemm_lut_{4}<32>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 128) {{\n\
                three_qgemm_lut_{4}<128>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 256) {{\n\
                three_qgemm_lut_{4}<256>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}else if (bs == 512) {{\n\
                three_qgemm_lut_{4}<512>(A, sign, LUT, Scales, LUT_Scales, C);\n\
            }}\n\
        }}\n\
    }}\n\
".format(kernel_shapes[i][0], kernel_shapes[i][1], k_list[i][0], k_list[i][1], "{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]))])
    kernel_code = "".join([kernel_code, "}\n"])
    return kernel_code

def gen_transform_code(kernel_shapes):
    kernel_code = "\n\
void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {\n\
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {\n\
        return;\n\
    }\n\
\n\
    int k = tensor->ne[0];\n\
    int m = tensor->ne[1];\n\
    const int lut_scales_size = 1;\n\
    int bk = 0;\n\
    int bm = 0;\n"

    kernel_code = "".join([kernel_code, "\n\
    if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[0][0], kernel_shapes[0][1])])

    for i in range(1, len(kernel_shapes)):
        kernel_code = "".join([kernel_code, "else if (m == {0} && k == {1}) {{\n\
        bm = BM{0}_{1};\n\
        bk = BBK{0}_{1};\n\
    }}\n".format(kernel_shapes[i][0], kernel_shapes[i][1])])

    kernel_code = "".join([kernel_code, "\n\
    const int n_tile_num = m / bm;\n\
    const int BK = bk;\n\
    uint8_t * qweights;\n\
    bitnet_float_type * scales;\n\
\n\
    scales = (bitnet_float_type *) aligned_malloc(sizeof(bitnet_float_type));\n\
    qweights = (uint8_t *) tensor->data;\n\
    int nbytes = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;\n\
    if (nbytes % 32 != 0) nbytes = 32 - nbytes % 32 + nbytes;\n\
    float * i2_scales = (float * )(qweights + nbytes);\n\
    scales[0] = (bitnet_float_type) i2_scales[0];\n\
\n\
    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;\n\
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {\n\
        /* .lut_scales_size = */ lut_scales_size,\n\
        /* .BK              = */ BK,\n\
        /* .n_tile_num      = */ n_tile_num,\n\
        /* .qweights        = */ qweights,\n\
        /* .scales          = */ scales\n\
    };\n\
}\n"])

    return kernel_code

def get_three_k_two_k(K, bk):
    bk_num = K // bk
    three_k = bk_num * bk
    two_k = K - three_k
    return two_k, three_k

if __name__ == "__main__":
    ModelShapeDict = {
        "bitnet_b1_58-large"                : [[1536, 4096],
                                               [1536, 1536],
                                               [4096, 1536]],
        "bitnet_b1_58-3B"                   : [[3200, 8640],
                                               [3200, 3200],
                                               [8640, 3200]],
        "Llama3-8B-1.58-100B-tokens"        : [[14336, 4096],
                                               [4096, 14336],
                                               [1024, 4096],
                                               [4096, 4096]] 
    }

    parser = argparse.ArgumentParser(description='gen impl')
    parser.add_argument('--model',default="input", type=str, dest="model", 
                        help="choose from bitnet_b1_58-large/bitnet_b1_58-3B/Llama3-8B-1.58-100B-tokens.")
    parser.add_argument('--BM',default="input", type=str,
                        help="block length when cutting one weight (M, K) into M / BM weights (BM, K).")
    parser.add_argument('--BK',default="input", type=str,
                        help="block length when cutting one weight (M, K) into K / BK weights (M, BK).")
    parser.add_argument('--bm',default="input", type=str,
                        help="using simd instructions to compute (bm, 192 / bm) in one block")
    args = parser.parse_args()

    kernel_shapes = ModelShapeDict[args.model]

    BM_list = [int(item) for item in args.BM.split(',')]
    BK_list = [int(item) for item in args.BK.split(',')]
    bm_list = [int(item) for item in args.bm.split(',')]

    tbl_impl_code = []
    k_list = []

    for i in range(len(kernel_shapes)):
        k_list.append(get_three_k_two_k(kernel_shapes[i][1], BK_list[i]))

    for i in range(len(kernel_shapes)):
        tbl_impl_code.append(
            gen_tbl_impl("{}_{}".format(kernel_shapes[i][0], kernel_shapes[i][1]), BM_list[i], BK_list[i], bm_list[i], k_list[i])
        )

    assert(len(BM_list) == len(BK_list) == len(bm_list) == len(kernel_shapes)), "number of BM / BK / bm shoud be {}".format(len(kernel_shapes))
    
    for i in range(len(kernel_shapes)):
        assert kernel_shapes[i][0] % BM_list[i] == 0, "M %% BM should be 0"
        assert (kernel_shapes[i][1] % BK_list[i]) % 32 == 0, "K %% BK %% 32 should be 0"
        assert bm_list[i] in [32], "choose bm from [32]"

    ctor_code = gen_ctor_code()
    api_code = gen_top_api(kernel_shapes, k_list)
    trans_code = gen_transform_code(kernel_shapes)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "include")

    with open(''.join([output_dir, "/bitnet-lut-kernels.h"]), 'w') as f:
        f.write(''.join("#if defined(GGML_BITNET_X86_TL2)"))
        f.write(''.join(ctor_code))
        for code in tbl_impl_code:
            f.write(''.join(code))
        f.write(''.join(api_code))
        f.write(''.join(trans_code))
        f.write(''.join("#endif"))

    config = ConfigParser()

    for i in range(len(kernel_shapes)):
        config.add_section('Kernels_{}'.format(i))
        config.set('Kernels_{}'.format(i), 'M'.format(i), str(kernel_shapes[i][0]))
        config.set('Kernels_{}'.format(i), 'K'.format(i), str(kernel_shapes[i][1]))
        config.set('Kernels_{}'.format(i), 'BM'.format(i), str(BM_list[i]))
        config.set('Kernels_{}'.format(i), 'BK'.format(i), str(BK_list[i]))
        config.set('Kernels_{}'.format(i), 'bmm'.format(i), str(bm_list[i]))

    with open(''.join([output_dir, "/kernel_config.ini"]), 'w') as configfile:
        config.write(configfile)