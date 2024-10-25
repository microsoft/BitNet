#include "ggml-bitnet.h"
#include <cstring>
#include <immintrin.h>
#define GGML_BITNET_MAX_NODES 8192
static bool initialized = false;
static bitnet_tensor_extra * bitnet_tensor_extras = nullptr;
static size_t bitnet_tensor_extras_index = 0;
static void * aligned_malloc(size_t size) {
#if defined(_WIN32)
    return _aligned_malloc(size, 64);
#else
    void * ptr = nullptr;
    posix_memalign(&ptr, 64, size);
    return ptr;
#endif
}

static void aligned_free(void * ptr) {
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}
#define BK2 32
#if defined __AVX2__
inline void _mm256_merge_epi32(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi32(va, vb);
    *vh = _mm256_unpackhi_epi32(va, vb);
}
inline void _mm256_merge_epi64(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    __m256i va = _mm256_permute4x64_epi64(v0, _MM_SHUFFLE(3, 1, 2, 0));
    __m256i vb = _mm256_permute4x64_epi64(v1, _MM_SHUFFLE(3, 1, 2, 0));
    *vl = _mm256_unpacklo_epi64(va, vb);
    *vh = _mm256_unpackhi_epi64(va, vb);
}
inline void _mm256_merge_si128(const __m256i v0, const __m256i v1, __m256i *vl, __m256i *vh)
{
    *vl = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 2, 0, 0));
    *vh = _mm256_permute2x128_si256(v0, v1, _MM_SHUFFLE(0, 3, 0, 1));
}
inline void Transpose_8_8(
    __m256i *v0,
    __m256i *v1,
    __m256i *v2,
    __m256i *v3,
    __m256i *v4,
    __m256i *v5,
    __m256i *v6,
    __m256i *v7)
{
    __m256i w0, w1, w2, w3, w4, w5, w6, w7;
    __m256i x0, x1, x2, x3, x4, x5, x6, x7;
    _mm256_merge_epi32(*v0, *v1, &w0, &w1);
    _mm256_merge_epi32(*v2, *v3, &w2, &w3);
    _mm256_merge_epi32(*v4, *v5, &w4, &w5);
    _mm256_merge_epi32(*v6, *v7, &w6, &w7);
    _mm256_merge_epi64(w0, w2, &x0, &x1);
    _mm256_merge_epi64(w1, w3, &x2, &x3);
    _mm256_merge_epi64(w4, w6, &x4, &x5);
    _mm256_merge_epi64(w5, w7, &x6, &x7);
    _mm256_merge_si128(x0, x4, v0, v1);
    _mm256_merge_si128(x1, x5, v2, v3);
    _mm256_merge_si128(x2, x6, v4, v5);
    _mm256_merge_si128(x3, x7, v6, v7);
}
#endif
inline int32_t per_tensor_quant(int k, void* lut_scales_, void* b_) {
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    bitnet_float_type* b = (bitnet_float_type*)b_;
#if defined __AVX2__
    __m256 max_vec = _mm256_set1_ps(0.f);
    const __m256 vec_sign = _mm256_set1_ps(-0.0f);
    for (int i = 0; i < k / 8; i++) {
        __m256 vec_b = _mm256_loadu_ps(b + i * 8);
        __m256 vec_babs = _mm256_andnot_ps(vec_sign, vec_b);
        max_vec = _mm256_max_ps(vec_babs, max_vec);
    }
    __m128 max1 = _mm_max_ps(_mm256_extractf128_ps(max_vec, 1), _mm256_castps256_ps128(max_vec));
    max1 = _mm_max_ps(max1, _mm_movehl_ps(max1, max1));
    max1 = _mm_max_ss(max1, _mm_movehdup_ps(max1));
    float scales = 127 / _mm_cvtss_f32(max1);
    *lut_scales = scales;
#endif
    return 0;
}
inline int32_t partial_max_reset(int32_t bs, void* lut_scales_) {
    bitnet_float_type* lut_scales = (bitnet_float_type*)lut_scales_;
    #pragma unroll
    for (int i=0; i< bs; i++) {
        lut_scales[i] = 0.0;
    }
    return 0;
}
template<int act_k>
inline int32_t three_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {
#if defined __AVX2__
    __m256i vec_lut[16];
    const __m256i vec_bi = _mm256_set_epi32(84, 72, 60, 48, 36, 24, 12, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 24; ++k) {
        __m256 vec_b0 = _mm256_i32gather_ps(b + k * 24 + 0, vec_bi, 1);
        __m256 vec_b1 = _mm256_i32gather_ps(b + k * 24 + 1, vec_bi, 1);
        __m256 vec_b2 = _mm256_i32gather_ps(b + k * 24 + 2, vec_bi, 1);

        __m256i vec_b0i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b2i = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b2, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));

        vec_lut[15] = _mm256_setzero_si256();
        vec_lut[14] = _mm256_setzero_si256();
        vec_lut[13] = vec_b0i;
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b1i);
        vec_lut[13] = _mm256_add_epi32(vec_lut[13], vec_b2i);
        vec_lut[12] = vec_b0i;
        vec_lut[12] = _mm256_add_epi32(vec_lut[12], vec_b1i);
        vec_lut[11] = vec_b0i;
        vec_lut[11] = _mm256_add_epi32(vec_lut[11], vec_b1i);
        vec_lut[11] = _mm256_sub_epi32(vec_lut[11], vec_b2i);
        vec_lut[10] = vec_b0i;
        vec_lut[10] = _mm256_add_epi32(vec_lut[10], vec_b2i);
        vec_lut[9] = vec_b0i;
        vec_lut[8] = vec_b0i;
        vec_lut[8] = _mm256_sub_epi32(vec_lut[8], vec_b2i);
        vec_lut[7] = vec_b0i;
        vec_lut[7] = _mm256_sub_epi32(vec_lut[7], vec_b1i);
        vec_lut[7] = _mm256_add_epi32(vec_lut[7], vec_b2i);
        vec_lut[6] = vec_b0i;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1i);
        vec_lut[5] = vec_b0i;
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b1i);
        vec_lut[5] = _mm256_sub_epi32(vec_lut[5], vec_b2i);
        vec_lut[4] = vec_b1i;
        vec_lut[4] = _mm256_add_epi32(vec_lut[4], vec_b2i);
        vec_lut[3] = vec_b1i;
        vec_lut[2] = vec_b1i;
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b2i);
        vec_lut[1] = vec_b2i;
        vec_lut[0] = _mm256_setzero_si256();
        __m256i ix[16];

#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }
        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }

    *lut_scales = scales;
#endif
    return 0;
}

template<int act_k>
inline int32_t two_lut_ctor(int8_t* qlut, bitnet_float_type* b, bitnet_float_type* lut_scales) {
#if defined __AVX2__
    __m256i vec_lut[16];
    const __m256i vec_bi = _mm256_set_epi32(56, 48, 40, 32, 24, 16, 8, 0);
    float scales = *lut_scales;
    __m256i shuffle_mask = _mm256_set_epi8(
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00,
                                            0x0f, 0x0d, 0x0b, 0x09, 0x07, 0x05, 0x03, 0x01,
                                            0x0e, 0x0c, 0x0a, 0x08, 0x06, 0x04, 0x02, 0x00
                                            );
#pragma unroll
    for (int k = 0; k < act_k / 16; ++k) {
        __m256 vec_b0f = _mm256_i32gather_ps(b + k * 16 + 0, vec_bi, 1);
        __m256 vec_b1f = _mm256_i32gather_ps(b + k * 16 + 1, vec_bi, 1);

        __m256i vec_b0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b0f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        __m256i vec_b1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(vec_b1f, _mm256_set1_ps(scales)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        vec_lut[15] = _mm256_setzero_si256();
        vec_lut[14] = _mm256_setzero_si256();
        vec_lut[13] = _mm256_setzero_si256();
        vec_lut[12] = _mm256_setzero_si256();
        vec_lut[11] = _mm256_setzero_si256();
        vec_lut[10] = _mm256_setzero_si256();
        vec_lut[9] = _mm256_setzero_si256();
        vec_lut[8] = vec_b0;
        vec_lut[8] = _mm256_add_epi32(vec_lut[8], vec_b1);
        vec_lut[7] = vec_b0;
        vec_lut[6] = vec_b0;
        vec_lut[6] = _mm256_sub_epi32(vec_lut[6], vec_b1);
        vec_lut[5] = vec_b1;
        vec_lut[4] = _mm256_setzero_si256();
        vec_lut[3] = _mm256_setzero_si256();
        vec_lut[3] = _mm256_sub_epi32(vec_lut[3], vec_b1);
        vec_lut[2] = _mm256_setzero_si256();
        vec_lut[2] = _mm256_sub_epi32(vec_lut[2], vec_b0);
        vec_lut[2] = _mm256_add_epi32(vec_lut[2], vec_b1);
        vec_lut[1] = _mm256_setzero_si256();
        vec_lut[1] = _mm256_sub_epi32(vec_lut[1], vec_b0);
        vec_lut[0] = _mm256_setzero_si256();
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b0);
        vec_lut[0] = _mm256_sub_epi32(vec_lut[0], vec_b1);

        __m256i ix[16];
#pragma unroll
        for (int g = 0; g < 16; ++g) {
            ix[g] = vec_lut[g];
        }

        Transpose_8_8(&(ix[0]), &(ix[1]), &(ix[2]), &(ix[3]), &(ix[4]), &(ix[5]),&(ix[6]), &(ix[7]));
        Transpose_8_8(&(ix[8]), &(ix[9]), &(ix[10]), &(ix[11]), &(ix[12]), &(ix[13]),&(ix[14]), &(ix[15]));

#pragma unroll
        for (int g = 0; g < 8; ++g) {
            ix[g] = _mm256_packs_epi32(ix[g], ix[g + 8]);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
            ix[g] = _mm256_shuffle_epi8(ix[g], shuffle_mask);
            ix[g] = _mm256_permute4x64_epi64(ix[g], _MM_SHUFFLE(3, 1, 2, 0));
        }

        int8_t* qlut_i8 = reinterpret_cast<int8_t*>(qlut);

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 0 * 32 + 0), ix[0]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 1 * 32 + 0), ix[1]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 2 * 32 + 0), ix[2]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 3 * 32 + 0), ix[3]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 4 * 32 + 0), ix[4]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 5 * 32 + 0), ix[5]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 6 * 32 + 0), ix[6]);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(qlut_i8 + k * 256 + 7 * 32 + 0), ix[7]);

    }
    *lut_scales = scales;
#endif
    return 0;
}
static bool is_type_supported(enum ggml_type type) {
    if (type == GGML_TYPE_Q4_0 ||
        type == GGML_TYPE_TL2) {
        return true;
    } else {
        return false;
    }
}
