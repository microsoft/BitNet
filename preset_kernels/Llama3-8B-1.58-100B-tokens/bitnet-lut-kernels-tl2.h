#if defined(GGML_BITNET_X86_TL2)
#include "ggml-bitnet.h"
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
    float scales = 127 / (_mm_cvtss_f32(max1) + 1e-10f);
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
    __m256 vec_lut[16];
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
    __m256 vec_lut[16];
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
#include <immintrin.h>

#define BM14336_4096 256
#define BBK14336_4096 96
template<int batch_size, int K3>
inline void three_tbl_impl_14336_4096(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);
    const __m256i vec_one  = _mm256_set1_epi8(0xff);
    const int KK = BBK14336_4096 / 3;
#pragma unroll
        for (int i = 0; i < BM14336_4096; i += 32) {
        __m256i vec_as[KK / 2];
        __m256i vec_signs[KK / 8];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
        #pragma unroll
        for (int as = 0; as < KK / 8; as++) {
            vec_signs[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            __m256i vec_sign = vec_signs[k];
                __m256i vec_a_0 = vec_as[k * 4 + 0];
                __m128i vec_k1_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0)), 15);
                __m256i vec_sign_left_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 1)), 15);
                __m256i vec_v_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), vec_mask);
                __m256i vec_v_top_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_0, vec_k1_0), vec_v_top_0);
                __m256i vec_v_top_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_0, vec_k2_0), vec_v_top_0);
                __m256i vec_sign_right_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 2)), 15);
                __m256i vec_sign_right_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 3)), 15);
                __m256i vec_v_bot_0 = _mm256_and_si256(vec_a_0, vec_mask);
                __m256i vec_v_bot_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_0, vec_k3_0), vec_v_bot_0);
                __m256i vec_v_bot_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_0, vec_k4_0), vec_v_bot_0);
                __m256i vec_v_top_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_lo_0), vec_sign_left_lo_0);
                __m256i vec_v_top_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_hi_0), vec_sign_left_hi_0);
                __m256i vec_v_bot_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_lo_0), vec_sign_right_lo_0);
                __m256i vec_v_bot_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_hi_0), vec_sign_right_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_0);
                __m256i vec_a_1 = vec_as[k * 4 + 1];
                __m128i vec_k1_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1)), 15);
                __m256i vec_sign_left_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 1)), 15);
                __m256i vec_v_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), vec_mask);
                __m256i vec_v_top_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_1, vec_k1_1), vec_v_top_1);
                __m256i vec_v_top_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_1, vec_k2_1), vec_v_top_1);
                __m256i vec_sign_right_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 2)), 15);
                __m256i vec_sign_right_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 3)), 15);
                __m256i vec_v_bot_1 = _mm256_and_si256(vec_a_1, vec_mask);
                __m256i vec_v_bot_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_1, vec_k3_1), vec_v_bot_1);
                __m256i vec_v_bot_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_1, vec_k4_1), vec_v_bot_1);
                __m256i vec_v_top_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_lo_1), vec_sign_left_lo_1);
                __m256i vec_v_top_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_hi_1), vec_sign_left_hi_1);
                __m256i vec_v_bot_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_lo_1), vec_sign_right_lo_1);
                __m256i vec_v_bot_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_hi_1), vec_sign_right_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_1);
                __m256i vec_a_2 = vec_as[k * 4 + 2];
                __m128i vec_k1_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2)), 15);
                __m256i vec_sign_left_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 1)), 15);
                __m256i vec_v_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), vec_mask);
                __m256i vec_v_top_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_2, vec_k1_2), vec_v_top_2);
                __m256i vec_v_top_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_2, vec_k2_2), vec_v_top_2);
                __m256i vec_sign_right_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 2)), 15);
                __m256i vec_sign_right_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 3)), 15);
                __m256i vec_v_bot_2 = _mm256_and_si256(vec_a_2, vec_mask);
                __m256i vec_v_bot_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_2, vec_k3_2), vec_v_bot_2);
                __m256i vec_v_bot_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_2, vec_k4_2), vec_v_bot_2);
                __m256i vec_v_top_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_lo_2), vec_sign_left_lo_2);
                __m256i vec_v_top_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_hi_2), vec_sign_left_hi_2);
                __m256i vec_v_bot_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_lo_2), vec_sign_right_lo_2);
                __m256i vec_v_bot_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_hi_2), vec_sign_right_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_2);
                __m256i vec_a_3 = vec_as[k * 4 + 3];
                __m128i vec_k1_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3)), 15);
                __m256i vec_sign_left_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 1)), 15);
                __m256i vec_v_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), vec_mask);
                __m256i vec_v_top_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_3, vec_k1_3), vec_v_top_3);
                __m256i vec_v_top_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_3, vec_k2_3), vec_v_top_3);
                __m256i vec_sign_right_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 2)), 15);
                __m256i vec_sign_right_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 3)), 15);
                __m256i vec_v_bot_3 = _mm256_and_si256(vec_a_3, vec_mask);
                __m256i vec_v_bot_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_3, vec_k3_3), vec_v_bot_3);
                __m256i vec_v_bot_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_3, vec_k4_3), vec_v_bot_3);
                __m256i vec_v_top_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_lo_3), vec_sign_left_lo_3);
                __m256i vec_v_top_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_hi_3), vec_sign_left_hi_3);
                __m256i vec_v_bot_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_lo_3), vec_sign_right_lo_3);
                __m256i vec_v_bot_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_hi_3), vec_sign_right_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_3);
        }
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM14336_4096 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM14336_4096 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM14336_4096 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM14336_4096 * bs));
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM14336_4096 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM14336_4096 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM14336_4096 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM14336_4096 * bs), vec_gc3);
    }
    }
#endif
}

template<int batch_size, int K2>
inline int32_t two_tbl_impl14336_4096(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const int KK = BK2 / 2;
#pragma unroll
    for (int i = 0; i < BM14336_4096; i += 32) {
        __m256i vec_as[KK / 2];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                __m256i vec_a = vec_as[k * 4 + j];

                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));
                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));
                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));
                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);

                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); 
            }
        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM14336_4096 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM14336_4096 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM14336_4096 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM14336_4096 * bs));

        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM14336_4096 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM14336_4096 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM14336_4096 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM14336_4096 * bs), vec_gc3);
    }
    }
#endif
    return 0;
}

template<int BATCH_SIZE>
int32_t three_qgemm_lut_14336_4096(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM14336_4096];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM14336_4096 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 4032 / BBK14336_4096; ++k_outer) {
        three_tbl_impl_14336_4096<BATCH_SIZE, 4032>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK14336_4096 / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK14336_4096 / 3 / 2 * BM14336_4096)])), (&(((uint8_t*)sign)[(k_outer * BBK14336_4096 / 3 / 8 * BM14336_4096)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM14336_4096; i++) {
            ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM14336_4096]);
        }
  }
  return 0;
}

template<int BATCH_SIZE>
int32_t two_qgemm_lut_14336_4096(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM14336_4096];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM14336_4096 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 64 / 32; ++k_outer) {
        two_tbl_impl14336_4096<BATCH_SIZE, 64>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM14336_4096)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM14336_4096; i++) {
            ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BM14336_4096]);
            ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
        }
    }
  return 0;
}

#include <immintrin.h>

#define BM4096_14336 128
#define BBK4096_14336 96
template<int batch_size, int K3>
inline void three_tbl_impl_4096_14336(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);
    const __m256i vec_one  = _mm256_set1_epi8(0xff);
    const int KK = BBK4096_14336 / 3;
#pragma unroll
        for (int i = 0; i < BM4096_14336; i += 32) {
        __m256i vec_as[KK / 2];
        __m256i vec_signs[KK / 8];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
        #pragma unroll
        for (int as = 0; as < KK / 8; as++) {
            vec_signs[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            __m256i vec_sign = vec_signs[k];
                __m256i vec_a_0 = vec_as[k * 4 + 0];
                __m128i vec_k1_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0)), 15);
                __m256i vec_sign_left_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 1)), 15);
                __m256i vec_v_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), vec_mask);
                __m256i vec_v_top_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_0, vec_k1_0), vec_v_top_0);
                __m256i vec_v_top_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_0, vec_k2_0), vec_v_top_0);
                __m256i vec_sign_right_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 2)), 15);
                __m256i vec_sign_right_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 3)), 15);
                __m256i vec_v_bot_0 = _mm256_and_si256(vec_a_0, vec_mask);
                __m256i vec_v_bot_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_0, vec_k3_0), vec_v_bot_0);
                __m256i vec_v_bot_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_0, vec_k4_0), vec_v_bot_0);
                __m256i vec_v_top_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_lo_0), vec_sign_left_lo_0);
                __m256i vec_v_top_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_hi_0), vec_sign_left_hi_0);
                __m256i vec_v_bot_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_lo_0), vec_sign_right_lo_0);
                __m256i vec_v_bot_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_hi_0), vec_sign_right_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_0);
                __m256i vec_a_1 = vec_as[k * 4 + 1];
                __m128i vec_k1_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1)), 15);
                __m256i vec_sign_left_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 1)), 15);
                __m256i vec_v_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), vec_mask);
                __m256i vec_v_top_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_1, vec_k1_1), vec_v_top_1);
                __m256i vec_v_top_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_1, vec_k2_1), vec_v_top_1);
                __m256i vec_sign_right_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 2)), 15);
                __m256i vec_sign_right_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 3)), 15);
                __m256i vec_v_bot_1 = _mm256_and_si256(vec_a_1, vec_mask);
                __m256i vec_v_bot_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_1, vec_k3_1), vec_v_bot_1);
                __m256i vec_v_bot_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_1, vec_k4_1), vec_v_bot_1);
                __m256i vec_v_top_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_lo_1), vec_sign_left_lo_1);
                __m256i vec_v_top_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_hi_1), vec_sign_left_hi_1);
                __m256i vec_v_bot_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_lo_1), vec_sign_right_lo_1);
                __m256i vec_v_bot_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_hi_1), vec_sign_right_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_1);
                __m256i vec_a_2 = vec_as[k * 4 + 2];
                __m128i vec_k1_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2)), 15);
                __m256i vec_sign_left_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 1)), 15);
                __m256i vec_v_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), vec_mask);
                __m256i vec_v_top_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_2, vec_k1_2), vec_v_top_2);
                __m256i vec_v_top_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_2, vec_k2_2), vec_v_top_2);
                __m256i vec_sign_right_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 2)), 15);
                __m256i vec_sign_right_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 3)), 15);
                __m256i vec_v_bot_2 = _mm256_and_si256(vec_a_2, vec_mask);
                __m256i vec_v_bot_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_2, vec_k3_2), vec_v_bot_2);
                __m256i vec_v_bot_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_2, vec_k4_2), vec_v_bot_2);
                __m256i vec_v_top_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_lo_2), vec_sign_left_lo_2);
                __m256i vec_v_top_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_hi_2), vec_sign_left_hi_2);
                __m256i vec_v_bot_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_lo_2), vec_sign_right_lo_2);
                __m256i vec_v_bot_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_hi_2), vec_sign_right_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_2);
                __m256i vec_a_3 = vec_as[k * 4 + 3];
                __m128i vec_k1_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3)), 15);
                __m256i vec_sign_left_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 1)), 15);
                __m256i vec_v_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), vec_mask);
                __m256i vec_v_top_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_3, vec_k1_3), vec_v_top_3);
                __m256i vec_v_top_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_3, vec_k2_3), vec_v_top_3);
                __m256i vec_sign_right_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 2)), 15);
                __m256i vec_sign_right_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 3)), 15);
                __m256i vec_v_bot_3 = _mm256_and_si256(vec_a_3, vec_mask);
                __m256i vec_v_bot_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_3, vec_k3_3), vec_v_bot_3);
                __m256i vec_v_bot_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_3, vec_k4_3), vec_v_bot_3);
                __m256i vec_v_top_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_lo_3), vec_sign_left_lo_3);
                __m256i vec_v_top_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_hi_3), vec_sign_left_hi_3);
                __m256i vec_v_bot_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_lo_3), vec_sign_right_lo_3);
                __m256i vec_v_bot_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_hi_3), vec_sign_right_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_3);
        }
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_14336 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_14336 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_14336 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_14336 * bs));
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_14336 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_14336 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_14336 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_14336 * bs), vec_gc3);
    }
    }
#endif
}

template<int batch_size, int K2>
inline int32_t two_tbl_impl4096_14336(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const int KK = BK2 / 2;
#pragma unroll
    for (int i = 0; i < BM4096_14336; i += 32) {
        __m256i vec_as[KK / 2];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                __m256i vec_a = vec_as[k * 4 + j];

                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));
                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));
                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));
                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);

                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); 
            }
        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_14336 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_14336 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_14336 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_14336 * bs));

        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_14336 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_14336 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_14336 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_14336 * bs), vec_gc3);
    }
    }
#endif
    return 0;
}

template<int BATCH_SIZE>
int32_t three_qgemm_lut_4096_14336(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM4096_14336];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM4096_14336 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 14304 / BBK4096_14336; ++k_outer) {
        three_tbl_impl_4096_14336<BATCH_SIZE, 14304>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK4096_14336 / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK4096_14336 / 3 / 2 * BM4096_14336)])), (&(((uint8_t*)sign)[(k_outer * BBK4096_14336 / 3 / 8 * BM4096_14336)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM4096_14336; i++) {
            ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM4096_14336]);
        }
  }
  return 0;
}

template<int BATCH_SIZE>
int32_t two_qgemm_lut_4096_14336(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM4096_14336];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM4096_14336 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 32 / 32; ++k_outer) {
        two_tbl_impl4096_14336<BATCH_SIZE, 32>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM4096_14336)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM4096_14336; i++) {
            ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BM4096_14336]);
            ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
        }
    }
  return 0;
}

#include <immintrin.h>

#define BM1024_4096 256
#define BBK1024_4096 96
template<int batch_size, int K3>
inline void three_tbl_impl_1024_4096(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);
    const __m256i vec_one  = _mm256_set1_epi8(0xff);
    const int KK = BBK1024_4096 / 3;
#pragma unroll
        for (int i = 0; i < BM1024_4096; i += 32) {
        __m256i vec_as[KK / 2];
        __m256i vec_signs[KK / 8];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
        #pragma unroll
        for (int as = 0; as < KK / 8; as++) {
            vec_signs[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            __m256i vec_sign = vec_signs[k];
                __m256i vec_a_0 = vec_as[k * 4 + 0];
                __m128i vec_k1_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0)), 15);
                __m256i vec_sign_left_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 1)), 15);
                __m256i vec_v_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), vec_mask);
                __m256i vec_v_top_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_0, vec_k1_0), vec_v_top_0);
                __m256i vec_v_top_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_0, vec_k2_0), vec_v_top_0);
                __m256i vec_sign_right_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 2)), 15);
                __m256i vec_sign_right_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 3)), 15);
                __m256i vec_v_bot_0 = _mm256_and_si256(vec_a_0, vec_mask);
                __m256i vec_v_bot_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_0, vec_k3_0), vec_v_bot_0);
                __m256i vec_v_bot_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_0, vec_k4_0), vec_v_bot_0);
                __m256i vec_v_top_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_lo_0), vec_sign_left_lo_0);
                __m256i vec_v_top_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_hi_0), vec_sign_left_hi_0);
                __m256i vec_v_bot_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_lo_0), vec_sign_right_lo_0);
                __m256i vec_v_bot_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_hi_0), vec_sign_right_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_0);
                __m256i vec_a_1 = vec_as[k * 4 + 1];
                __m128i vec_k1_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1)), 15);
                __m256i vec_sign_left_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 1)), 15);
                __m256i vec_v_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), vec_mask);
                __m256i vec_v_top_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_1, vec_k1_1), vec_v_top_1);
                __m256i vec_v_top_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_1, vec_k2_1), vec_v_top_1);
                __m256i vec_sign_right_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 2)), 15);
                __m256i vec_sign_right_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 3)), 15);
                __m256i vec_v_bot_1 = _mm256_and_si256(vec_a_1, vec_mask);
                __m256i vec_v_bot_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_1, vec_k3_1), vec_v_bot_1);
                __m256i vec_v_bot_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_1, vec_k4_1), vec_v_bot_1);
                __m256i vec_v_top_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_lo_1), vec_sign_left_lo_1);
                __m256i vec_v_top_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_hi_1), vec_sign_left_hi_1);
                __m256i vec_v_bot_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_lo_1), vec_sign_right_lo_1);
                __m256i vec_v_bot_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_hi_1), vec_sign_right_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_1);
                __m256i vec_a_2 = vec_as[k * 4 + 2];
                __m128i vec_k1_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2)), 15);
                __m256i vec_sign_left_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 1)), 15);
                __m256i vec_v_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), vec_mask);
                __m256i vec_v_top_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_2, vec_k1_2), vec_v_top_2);
                __m256i vec_v_top_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_2, vec_k2_2), vec_v_top_2);
                __m256i vec_sign_right_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 2)), 15);
                __m256i vec_sign_right_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 3)), 15);
                __m256i vec_v_bot_2 = _mm256_and_si256(vec_a_2, vec_mask);
                __m256i vec_v_bot_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_2, vec_k3_2), vec_v_bot_2);
                __m256i vec_v_bot_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_2, vec_k4_2), vec_v_bot_2);
                __m256i vec_v_top_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_lo_2), vec_sign_left_lo_2);
                __m256i vec_v_top_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_hi_2), vec_sign_left_hi_2);
                __m256i vec_v_bot_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_lo_2), vec_sign_right_lo_2);
                __m256i vec_v_bot_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_hi_2), vec_sign_right_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_2);
                __m256i vec_a_3 = vec_as[k * 4 + 3];
                __m128i vec_k1_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3)), 15);
                __m256i vec_sign_left_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 1)), 15);
                __m256i vec_v_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), vec_mask);
                __m256i vec_v_top_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_3, vec_k1_3), vec_v_top_3);
                __m256i vec_v_top_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_3, vec_k2_3), vec_v_top_3);
                __m256i vec_sign_right_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 2)), 15);
                __m256i vec_sign_right_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 3)), 15);
                __m256i vec_v_bot_3 = _mm256_and_si256(vec_a_3, vec_mask);
                __m256i vec_v_bot_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_3, vec_k3_3), vec_v_bot_3);
                __m256i vec_v_bot_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_3, vec_k4_3), vec_v_bot_3);
                __m256i vec_v_top_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_lo_3), vec_sign_left_lo_3);
                __m256i vec_v_top_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_hi_3), vec_sign_left_hi_3);
                __m256i vec_v_bot_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_lo_3), vec_sign_right_lo_3);
                __m256i vec_v_bot_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_hi_3), vec_sign_right_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_3);
        }
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM1024_4096 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM1024_4096 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM1024_4096 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM1024_4096 * bs));
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM1024_4096 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM1024_4096 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM1024_4096 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM1024_4096 * bs), vec_gc3);
    }
    }
#endif
}

template<int batch_size, int K2>
inline int32_t two_tbl_impl1024_4096(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const int KK = BK2 / 2;
#pragma unroll
    for (int i = 0; i < BM1024_4096; i += 32) {
        __m256i vec_as[KK / 2];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                __m256i vec_a = vec_as[k * 4 + j];

                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));
                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));
                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));
                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);

                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); 
            }
        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM1024_4096 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM1024_4096 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM1024_4096 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM1024_4096 * bs));

        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM1024_4096 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM1024_4096 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM1024_4096 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM1024_4096 * bs), vec_gc3);
    }
    }
#endif
    return 0;
}

template<int BATCH_SIZE>
int32_t three_qgemm_lut_1024_4096(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM1024_4096];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM1024_4096 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 4032 / BBK1024_4096; ++k_outer) {
        three_tbl_impl_1024_4096<BATCH_SIZE, 4032>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK1024_4096 / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK1024_4096 / 3 / 2 * BM1024_4096)])), (&(((uint8_t*)sign)[(k_outer * BBK1024_4096 / 3 / 8 * BM1024_4096)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM1024_4096; i++) {
            ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM1024_4096]);
        }
  }
  return 0;
}

template<int BATCH_SIZE>
int32_t two_qgemm_lut_1024_4096(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM1024_4096];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM1024_4096 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 64 / 32; ++k_outer) {
        two_tbl_impl1024_4096<BATCH_SIZE, 64>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM1024_4096)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM1024_4096; i++) {
            ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BM1024_4096]);
            ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
        }
    }
  return 0;
}

#include <immintrin.h>

#define BM4096_4096 128
#define BBK4096_4096 96
template<int batch_size, int K3>
inline void three_tbl_impl_4096_4096(int32_t* c, int8_t* lut, uint8_t* a, uint8_t* sign) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const __m256i vec_sign_mask  = _mm256_set1_epi16(0x8000);
    const __m256i vec_zero  = _mm256_set1_epi8(0x00);
    const __m256i vec_one  = _mm256_set1_epi8(0xff);
    const int KK = BBK4096_4096 / 3;
#pragma unroll
        for (int i = 0; i < BM4096_4096; i += 32) {
        __m256i vec_as[KK / 2];
        __m256i vec_signs[KK / 8];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
        #pragma unroll
        for (int as = 0; as < KK / 8; as++) {
            vec_signs[as] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(sign + i * KK / 8 + as * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            __m256i vec_sign = vec_signs[k];
                __m256i vec_a_0 = vec_as[k * 4 + 0];
                __m128i vec_k1_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 0 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0)), 15);
                __m256i vec_sign_left_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 1)), 15);
                __m256i vec_v_top_0 = _mm256_and_si256(_mm256_srli_epi16(vec_a_0, 4), vec_mask);
                __m256i vec_v_top_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_0, vec_k1_0), vec_v_top_0);
                __m256i vec_v_top_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_0, vec_k2_0), vec_v_top_0);
                __m256i vec_sign_right_hi_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 2)), 15);
                __m256i vec_sign_right_lo_0 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 0 + 3)), 15);
                __m256i vec_v_bot_0 = _mm256_and_si256(vec_a_0, vec_mask);
                __m256i vec_v_bot_fir_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_0, vec_k3_0), vec_v_bot_0);
                __m256i vec_v_bot_sec_0 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_0, vec_k4_0), vec_v_bot_0);
                __m256i vec_v_top_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_lo_0), vec_sign_left_lo_0);
                __m256i vec_v_top_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_0, vec_v_top_sec_0), vec_sign_left_hi_0), vec_sign_left_hi_0);
                __m256i vec_v_bot_lo_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_lo_0), vec_sign_right_lo_0);
                __m256i vec_v_bot_hi_0 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_0, vec_v_bot_sec_0), vec_sign_right_hi_0), vec_sign_right_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_0);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_0);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_0);
                __m256i vec_a_1 = vec_as[k * 4 + 1];
                __m128i vec_k1_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 1 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1)), 15);
                __m256i vec_sign_left_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 1)), 15);
                __m256i vec_v_top_1 = _mm256_and_si256(_mm256_srli_epi16(vec_a_1, 4), vec_mask);
                __m256i vec_v_top_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_1, vec_k1_1), vec_v_top_1);
                __m256i vec_v_top_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_1, vec_k2_1), vec_v_top_1);
                __m256i vec_sign_right_hi_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 2)), 15);
                __m256i vec_sign_right_lo_1 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 1 + 3)), 15);
                __m256i vec_v_bot_1 = _mm256_and_si256(vec_a_1, vec_mask);
                __m256i vec_v_bot_fir_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_1, vec_k3_1), vec_v_bot_1);
                __m256i vec_v_bot_sec_1 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_1, vec_k4_1), vec_v_bot_1);
                __m256i vec_v_top_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_lo_1), vec_sign_left_lo_1);
                __m256i vec_v_top_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_1, vec_v_top_sec_1), vec_sign_left_hi_1), vec_sign_left_hi_1);
                __m256i vec_v_bot_lo_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_lo_1), vec_sign_right_lo_1);
                __m256i vec_v_bot_hi_1 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_1, vec_v_bot_sec_1), vec_sign_right_hi_1), vec_sign_right_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_1);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_1);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_1);
                __m256i vec_a_2 = vec_as[k * 4 + 2];
                __m128i vec_k1_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 2 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2)), 15);
                __m256i vec_sign_left_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 1)), 15);
                __m256i vec_v_top_2 = _mm256_and_si256(_mm256_srli_epi16(vec_a_2, 4), vec_mask);
                __m256i vec_v_top_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_2, vec_k1_2), vec_v_top_2);
                __m256i vec_v_top_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_2, vec_k2_2), vec_v_top_2);
                __m256i vec_sign_right_hi_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 2)), 15);
                __m256i vec_sign_right_lo_2 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 2 + 3)), 15);
                __m256i vec_v_bot_2 = _mm256_and_si256(vec_a_2, vec_mask);
                __m256i vec_v_bot_fir_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_2, vec_k3_2), vec_v_bot_2);
                __m256i vec_v_bot_sec_2 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_2, vec_k4_2), vec_v_bot_2);
                __m256i vec_v_top_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_lo_2), vec_sign_left_lo_2);
                __m256i vec_v_top_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_2, vec_v_top_sec_2), vec_sign_left_hi_2), vec_sign_left_hi_2);
                __m256i vec_v_bot_lo_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_lo_2), vec_sign_right_lo_2);
                __m256i vec_v_bot_hi_2 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_2, vec_v_bot_sec_2), vec_sign_right_hi_2), vec_sign_right_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_2);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_2);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_2);
                __m256i vec_a_3 = vec_as[k * 4 + 3];
                __m128i vec_k1_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 0  + K3 / 3 * 32 * bs));
                __m128i vec_k2_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 16 + K3 / 3 * 32 * bs));
                __m128i vec_k3_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 32 + K3 / 3 * 32 * bs));
                __m128i vec_k4_3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + 3 * 64 + 48 + K3 / 3 * 32 * bs));
                __m256i vec_sign_left_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3)), 15);
                __m256i vec_sign_left_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 1)), 15);
                __m256i vec_v_top_3 = _mm256_and_si256(_mm256_srli_epi16(vec_a_3, 4), vec_mask);
                __m256i vec_v_top_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1_3, vec_k1_3), vec_v_top_3);
                __m256i vec_v_top_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2_3, vec_k2_3), vec_v_top_3);
                __m256i vec_sign_right_hi_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 2)), 15);
                __m256i vec_sign_right_lo_3 = _mm256_srai_epi16(_mm256_slli_epi16(vec_sign, (4 * 3 + 3)), 15);
                __m256i vec_v_bot_3 = _mm256_and_si256(vec_a_3, vec_mask);
                __m256i vec_v_bot_fir_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3_3, vec_k3_3), vec_v_bot_3);
                __m256i vec_v_bot_sec_3 = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4_3, vec_k4_3), vec_v_bot_3);
                __m256i vec_v_top_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_lo_3), vec_sign_left_lo_3);
                __m256i vec_v_top_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_top_fir_3, vec_v_top_sec_3), vec_sign_left_hi_3), vec_sign_left_hi_3);
                __m256i vec_v_bot_lo_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpackhi_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_lo_3), vec_sign_right_lo_3);
                __m256i vec_v_bot_hi_3 = _mm256_xor_si256(_mm256_add_epi16(_mm256_unpacklo_epi8(vec_v_bot_fir_3, vec_v_bot_sec_3), vec_sign_right_hi_3), vec_sign_right_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi_3);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo_3);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo_3);
        }
        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_4096 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_4096 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_4096 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_4096 * bs));
        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_4096 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_4096 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_4096 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_4096 * bs), vec_gc3);
    }
    }
#endif
}

template<int batch_size, int K2>
inline int32_t two_tbl_impl4096_4096(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __AVX2__
    const __m256i vec_mask = _mm256_set1_epi8(0x0f);
    const int KK = BK2 / 2;
#pragma unroll
    for (int i = 0; i < BM4096_4096; i += 32) {
        __m256i vec_as[KK / 2];
        #pragma unroll
        for (int ai = 0; ai < KK / 2; ai++) {
            vec_as[ai] = _mm256_loadu_si256(reinterpret_cast<__m256i*>(a + i * KK / 2 + ai * 32));
        }
#pragma unroll
    for (int bs = 0; bs < batch_size; bs++) {
        __m256i vec_c0 = _mm256_setzero_si256();
        __m256i vec_c1 = _mm256_setzero_si256();
#pragma unroll
        for (int k = 0; k < KK / 8; k++) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                __m256i vec_a = vec_as[k * 4 + j];

                __m128i vec_k1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 0  + K2 / 2 * 32 * bs));
                __m128i vec_k2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 16 + K2 / 2 * 32 * bs));
                __m128i vec_k3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 32 + K2 / 2 * 32 * bs));
                __m128i vec_k4 = _mm_loadu_si128(reinterpret_cast<__m128i*>(lut + k * 32 * 8 + j * 64 + 48 + K2 / 2 * 32 * bs));

                __m256i vec_v_top = _mm256_and_si256(_mm256_srli_epi16(vec_a, 4), vec_mask);
                __m256i vec_v_top_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k1, vec_k1), vec_v_top);
                __m256i vec_v_top_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k2, vec_k2), vec_v_top);

                __m256i vec_v_bot = _mm256_and_si256(vec_a, vec_mask);
                __m256i vec_v_bot_fir = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k3, vec_k3), vec_v_bot);
                __m256i vec_v_bot_sec = _mm256_shuffle_epi8(_mm256_set_m128i(vec_k4, vec_k4), vec_v_bot);

                __m256i vec_v_top_lo = _mm256_unpackhi_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_top_hi = _mm256_unpacklo_epi8(vec_v_top_fir, vec_v_top_sec);
                __m256i vec_v_bot_lo = _mm256_unpackhi_epi8(vec_v_bot_fir, vec_v_bot_sec);
                __m256i vec_v_bot_hi = _mm256_unpacklo_epi8(vec_v_bot_fir, vec_v_bot_sec);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_top_hi);
                vec_c0 = _mm256_add_epi16(vec_c0, vec_v_bot_hi);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_top_lo);
                vec_c1 = _mm256_add_epi16(vec_c1, vec_v_bot_lo); 
            }
        }

        __m256i vec_gc0 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_4096 * bs));
        __m256i vec_gc1 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_4096 * bs));
        __m256i vec_gc2 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_4096 * bs));
        __m256i vec_gc3 = _mm256_loadu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_4096 * bs));

        vec_gc0 = _mm256_add_epi32(vec_gc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c0)));
        vec_gc1 = _mm256_add_epi32(vec_gc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c0, 1)));
        vec_gc2 = _mm256_add_epi32(vec_gc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vec_c1)));
        vec_gc3 = _mm256_add_epi32(vec_gc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vec_c1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i      + BM4096_4096 * bs), vec_gc0);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 8  + BM4096_4096 * bs), vec_gc1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 16 + BM4096_4096 * bs), vec_gc2);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(c + i + 24 + BM4096_4096 * bs), vec_gc3);
    }
    }
#endif
    return 0;
}

template<int BATCH_SIZE>
int32_t three_qgemm_lut_4096_4096(void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM4096_4096];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM4096_4096 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 4032 / BBK4096_4096; ++k_outer) {
        three_tbl_impl_4096_4096<BATCH_SIZE, 4032>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK4096_4096 / 3 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK4096_4096 / 3 / 2 * BM4096_4096)])), (&(((uint8_t*)sign)[(k_outer * BBK4096_4096 / 3 / 8 * BM4096_4096)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM4096_4096; i++) {
            ((int32_t*)C)[i] = (int32_t)(((int32_t*)CBits)[i + bs * BM4096_4096]);
        }
  }
  return 0;
}

template<int BATCH_SIZE>
int32_t two_qgemm_lut_4096_4096(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas(32) uint32_t CBits[BATCH_SIZE * BM4096_4096];
    memset(&(CBits[0]), 0, BATCH_SIZE * BM4096_4096 * sizeof(int32_t));
#pragma unroll
    for (int32_t k_outer = 0; k_outer < 64 / 32; ++k_outer) {
        two_tbl_impl4096_4096<BATCH_SIZE, 64>((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BK2 / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BK2 / 2 / 2 * BM4096_4096)])));
    }
#pragma unroll
    for (int bs = 0; bs < BATCH_SIZE; bs++) {
#pragma unroll
        for (int i = 0; i < BM4096_4096; i++) {
            ((int32_t*)C)[i] += (int32_t)(((int32_t*)CBits)[i + bs * BM4096_4096]);
            ((float*)C)[i] = (float)(((int32_t*)C)[i]) / ((float*)LUT_Scales)[bs] * ((float*)Scales)[0];
        }
    }
  return 0;
}

void ggml_preprocessor(int bs, int m, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT) {
    partial_max_reset(bs, (&(((float*)LUT_Scales)[0])));
    if (m == 14336 && two_k == 64 && three_k == 4032) {
        for (int32_t b = 0; b < bs; b++) {
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));
            three_lut_ctor<4032>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
            two_lut_ctor<64>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + 4032])), (&(((float*)LUT_Scales)[b])));
        }
    }
    else if (m == 4096 && two_k == 32 && three_k == 14304) {
        for (int32_t b = 0; b < bs; b++) {
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));
            three_lut_ctor<14304>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
            two_lut_ctor<32>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + 14304])), (&(((float*)LUT_Scales)[b])));
        }
    }
    else if (m == 1024 && two_k == 64 && three_k == 4032) {
        for (int32_t b = 0; b < bs; b++) {
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));
            three_lut_ctor<4032>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
            two_lut_ctor<64>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + 4032])), (&(((float*)LUT_Scales)[b])));
        }
    }
    else if (m == 4096 && two_k == 64 && three_k == 4032) {
        for (int32_t b = 0; b < bs; b++) {
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));
            three_lut_ctor<4032>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
            two_lut_ctor<64>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + 4032])), (&(((float*)LUT_Scales)[b])));
        }
    }
}
void ggml_qgemm_lut(int bs, int m, int k, int BK, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    if (m == 14336 && k == 4096) {
        if (BK == 64) {
            if (bs == 1) {
                two_qgemm_lut_14336_4096<1>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 8) {
                two_qgemm_lut_14336_4096<8>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 32) {
                two_qgemm_lut_14336_4096<32>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 128) {
                two_qgemm_lut_14336_4096<128>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 256) {
                two_qgemm_lut_14336_4096<256>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 512) {
                two_qgemm_lut_14336_4096<512>(A, LUT, Scales, LUT_Scales, C);
            }
        }
        else if (BK == 4032) {
            if (bs == 1) {
                three_qgemm_lut_14336_4096<1>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 8) {
                three_qgemm_lut_14336_4096<8>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 32) {
                three_qgemm_lut_14336_4096<32>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 128) {
                three_qgemm_lut_14336_4096<128>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 256) {
                three_qgemm_lut_14336_4096<256>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 512) {
                three_qgemm_lut_14336_4096<512>(A, sign, LUT, Scales, LUT_Scales, C);
            }
        }
    }
    else if (m == 4096 && k == 14336) {
        if (BK == 32) {
            if (bs == 1) {
                two_qgemm_lut_4096_14336<1>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 8) {
                two_qgemm_lut_4096_14336<8>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 32) {
                two_qgemm_lut_4096_14336<32>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 128) {
                two_qgemm_lut_4096_14336<128>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 256) {
                two_qgemm_lut_4096_14336<256>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 512) {
                two_qgemm_lut_4096_14336<512>(A, LUT, Scales, LUT_Scales, C);
            }
        }
        else if (BK == 14304) {
            if (bs == 1) {
                three_qgemm_lut_4096_14336<1>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 8) {
                three_qgemm_lut_4096_14336<8>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 32) {
                three_qgemm_lut_4096_14336<32>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 128) {
                three_qgemm_lut_4096_14336<128>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 256) {
                three_qgemm_lut_4096_14336<256>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 512) {
                three_qgemm_lut_4096_14336<512>(A, sign, LUT, Scales, LUT_Scales, C);
            }
        }
    }
    else if (m == 1024 && k == 4096) {
        if (BK == 64) {
            if (bs == 1) {
                two_qgemm_lut_1024_4096<1>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 8) {
                two_qgemm_lut_1024_4096<8>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 32) {
                two_qgemm_lut_1024_4096<32>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 128) {
                two_qgemm_lut_1024_4096<128>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 256) {
                two_qgemm_lut_1024_4096<256>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 512) {
                two_qgemm_lut_1024_4096<512>(A, LUT, Scales, LUT_Scales, C);
            }
        }
        else if (BK == 4032) {
            if (bs == 1) {
                three_qgemm_lut_1024_4096<1>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 8) {
                three_qgemm_lut_1024_4096<8>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 32) {
                three_qgemm_lut_1024_4096<32>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 128) {
                three_qgemm_lut_1024_4096<128>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 256) {
                three_qgemm_lut_1024_4096<256>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 512) {
                three_qgemm_lut_1024_4096<512>(A, sign, LUT, Scales, LUT_Scales, C);
            }
        }
    }
    else if (m == 4096 && k == 4096) {
        if (BK == 64) {
            if (bs == 1) {
                two_qgemm_lut_4096_4096<1>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 8) {
                two_qgemm_lut_4096_4096<8>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 32) {
                two_qgemm_lut_4096_4096<32>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 128) {
                two_qgemm_lut_4096_4096<128>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 256) {
                two_qgemm_lut_4096_4096<256>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 512) {
                two_qgemm_lut_4096_4096<512>(A, LUT, Scales, LUT_Scales, C);
            }
        }
        else if (BK == 4032) {
            if (bs == 1) {
                three_qgemm_lut_4096_4096<1>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 8) {
                three_qgemm_lut_4096_4096<8>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 32) {
                three_qgemm_lut_4096_4096<32>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 128) {
                three_qgemm_lut_4096_4096<128>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 256) {
                three_qgemm_lut_4096_4096<256>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 512) {
                three_qgemm_lut_4096_4096<512>(A, sign, LUT, Scales, LUT_Scales, C);
            }
        }
    }
}

void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {
        return;
    }

    int k = tensor->ne[0];
    int m = tensor->ne[1];
    const int lut_scales_size = 1;
    int bk = 0;
    int bm = 0;

    if (m == 14336 && k == 4096) {
        bm = BM14336_4096;
        bk = BBK14336_4096;
    }
else if (m == 4096 && k == 14336) {
        bm = BM4096_14336;
        bk = BBK4096_14336;
    }
else if (m == 1024 && k == 4096) {
        bm = BM1024_4096;
        bk = BBK1024_4096;
    }
else if (m == 4096 && k == 4096) {
        bm = BM4096_4096;
        bk = BBK4096_4096;
    }

    const int n_tile_num = m / bm;
    const int BK = bk;
    uint8_t * qweights;
    bitnet_float_type * scales;

    scales = (bitnet_float_type *) aligned_malloc(sizeof(bitnet_float_type));
    qweights = (uint8_t *) tensor->data;
    float * i2_scales = (float * )(qweights + k * m / 4);
    scales[0] = (bitnet_float_type) i2_scales[0];

    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {
        /* .lut_scales_size = */ lut_scales_size,
        /* .BK              = */ BK,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    };
}
#endif