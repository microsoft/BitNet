#include <vector>
#include <type_traits>

#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include <cmath>
#include <cstring>

#define QK_I2_S 128
#define QK_I2 128

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#include <immintrin.h>
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}
#elif defined(__loongarch_asx)
// horizontally add 8 int32_t
static inline int hsum_i32_8(const __m256i a) {

    __m256i tmp1 = __lasx_xvpermi_q(a, a, 0x11);
    __m256i tmp2 = __lasx_xvpermi_q(a, a, 0x00);

    __m128i  tmp1_128 = lasx_extracti128_lo(tmp1);
    __m128i  tmp2_128 = lasx_extracti128_lo(tmp2);

    __m128i sum128 = __lsx_vadd_w(tmp1_128, tmp2_128);

    __m128i ev = __lsx_vpickev_w(sum128, sum128);
    __m128i od = __lsx_vpickod_w(sum128, sum128);
    __m128i sum64 = __lsx_vadd_w(ev, od);

    int sum64_1, sum64_2;
    sum64_1 = __lsx_vpickve2gr_w(sum64, 0);
    sum64_2 = __lsx_vpickve2gr_w(sum64, 1);

    return  sum64_1 + sum64_2;
}
#endif

size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    // 2 bits per weight

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int n = nrow * n_per_row;

    // f32 -> q8
    double max = 0;
    for (int i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n * sizeof(uint8_t) / 4);

    // q8 -> 0, 1, 2
    //       |  |  |
    //      -1, 0, 1

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int i = 0; i < n / QK_I2; i++) {
        for (int j = 0; j < QK_I2; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2 + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;            
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    // 32B for alignment
    return nrow * row_size / 4 + 32;
}

void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

#if defined(__AVX2__)

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i accu = _mm256_setzero_si256();

    for (int i=0; i < group32_num; i++){
        __m256i accu32 = _mm256_setzero_si256();
        for (int j=0; j < 32; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + i * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + i * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
        accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, _mm256_set1_epi16(1)), accu);
    }

    for (int i = 0; i < groupla_num; i++){
        __m256i accula = _mm256_setzero_si256();
        for (int j = 0; j < la_num; j++) {
        // 128 index
        __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(x + group32_num * 32 * 32 + j * 32));
        __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
        __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
        __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

        // each 32 index
        xq8_3 = _mm256_and_si256(xq8_3, mask);
        xq8_2 = _mm256_and_si256(xq8_2, mask);
        xq8_1 = _mm256_and_si256(xq8_1, mask);
        xq8_0 = _mm256_and_si256(xq8_0, mask);

        // each 32 index
        __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 0));
        __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 32));
        __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 64));
        __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(y + group32_num * 128 * 32 + j * 128 + 96));

        // 128 index accumulation add
        // split into 32 accumulation block
        // each block each 128 index accumulated 4index
        // each index maximum 256
        // each block maximum 4 * 256
        // each block accumulation maximum 127 * 256
        // each 32 group index (128 index in one group) needs cast to int32
        xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
        xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
        xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
        xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
        accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));
        }
        accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, _mm256_set1_epi16(1)));
    }
    int sumi = hsum_i32_8(accu);
    *s = (float)sumi;

#elif defined(__ARM_NEON)

    int32x4_t accu_0 = vdupq_n_s32(0);
    int32x4_t accu_1 = vdupq_n_s32(0);
    int32x4_t accu_2 = vdupq_n_s32(0);
    int32x4_t accu_3 = vdupq_n_s32(0);
    const uint8x16_t mask = vdupq_n_u8(3);

    for (int i=0; i < group32_num; i++) {

#if defined(__ARM_FEATURE_DOTPROD)

#else
        int16x8_t accu32_0 = vdupq_n_s16(0);
        int16x8_t accu32_1 = vdupq_n_s16(0);
        int16x8_t accu32_2 = vdupq_n_s16(0);
        int16x8_t accu32_3 = vdupq_n_s16(0);
#endif

        for (int j=0; j < 32; j++) {
            uint8x16_t xq8_6 = vld1q_u8(x + i * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + i * 32 * 32 + j * 32 + 16);
            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            const int8x16_t yq8_0 = vld1q_s8(y + i * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + i * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + i * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + i * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + i * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + i * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + i * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + i * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
            accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
            accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
            accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
            accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
            accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
            accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
            accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accu32_0 = vmlal_s8(accu32_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accu32_1 = vmlal_s8(accu32_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accu32_2 = vmlal_s8(accu32_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accu32_3 = vmlal_s8(accu32_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }

#if defined(__ARM_FEATURE_DOTPROD)

#else
        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accu32_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accu32_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accu32_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accu32_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accu32_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accu32_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accu32_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accu32_3));
#endif
    }

    for (int i = 0; i < groupla_num; i++){
#if defined(__ARM_FEATURE_DOTPROD)

#else
        int16x8_t accula_0 = vdupq_n_s16(0);
        int16x8_t accula_1 = vdupq_n_s16(0);
        int16x8_t accula_2 = vdupq_n_s16(0);
        int16x8_t accula_3 = vdupq_n_s16(0);
#endif
        for (int j = 0; j < la_num; j++) {
            uint8x16_t xq8_6 = vld1q_u8(x + group32_num * 32 * 32 + j * 32);
            uint8x16_t xq8_7 = vld1q_u8(x + group32_num * 32 * 32 + j * 32 + 16);
            uint8x16_t xq8_4 = vshrq_n_u8(xq8_6, 2);
            uint8x16_t xq8_5 = vshrq_n_u8(xq8_7, 2);
            uint8x16_t xq8_2 = vshrq_n_u8(xq8_6, 4);
            uint8x16_t xq8_3 = vshrq_n_u8(xq8_7, 4);
            uint8x16_t xq8_0 = vshrq_n_u8(xq8_6, 6);
            uint8x16_t xq8_1 = vshrq_n_u8(xq8_7, 6);

            int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
            int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
            int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
            int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
            int8x16_t q8_4 = vreinterpretq_s8_u8(vandq_u8(xq8_4, mask));
            int8x16_t q8_5 = vreinterpretq_s8_u8(vandq_u8(xq8_5, mask));
            int8x16_t q8_6 = vreinterpretq_s8_u8(vandq_u8(xq8_6, mask));
            int8x16_t q8_7 = vreinterpretq_s8_u8(vandq_u8(xq8_7, mask));

            const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 0);
            const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 16);
            const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 32);
            const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 48);
            const int8x16_t yq8_4 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 64);
            const int8x16_t yq8_5 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 80);
            const int8x16_t yq8_6 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 96);
            const int8x16_t yq8_7 = vld1q_s8(y + group32_num * 128 * 32 + j * 128 + 112);

#if defined(__ARM_FEATURE_DOTPROD)
            accu_0 = vdotq_s32(accu_0, q8_0, yq8_0);
            accu_1 = vdotq_s32(accu_1, q8_1, yq8_1);
            accu_2 = vdotq_s32(accu_2, q8_2, yq8_2);
            accu_3 = vdotq_s32(accu_3, q8_3, yq8_3);
            accu_0 = vdotq_s32(accu_0, q8_4, yq8_4);
            accu_1 = vdotq_s32(accu_1, q8_5, yq8_5);
            accu_2 = vdotq_s32(accu_2, q8_6, yq8_6);
            accu_3 = vdotq_s32(accu_3, q8_7, yq8_7);
#else
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_0), vget_low_s8(yq8_0));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_0), vget_high_s8(yq8_0));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_1), vget_low_s8(yq8_1));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_1), vget_high_s8(yq8_1));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_2), vget_low_s8(yq8_2));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_2), vget_high_s8(yq8_2));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_3), vget_low_s8(yq8_3));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_3), vget_high_s8(yq8_3));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_4), vget_low_s8(yq8_4));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_4), vget_high_s8(yq8_4));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_5), vget_low_s8(yq8_5));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_5), vget_high_s8(yq8_5));
            accula_0 = vmlal_s8(accula_0, vget_low_s8(q8_6), vget_low_s8(yq8_6));
            accula_1 = vmlal_s8(accula_1, vget_high_s8(q8_6), vget_high_s8(yq8_6));
            accula_2 = vmlal_s8(accula_2, vget_low_s8(q8_7), vget_low_s8(yq8_7));
            accula_3 = vmlal_s8(accula_3, vget_high_s8(q8_7), vget_high_s8(yq8_7));
#endif
        }
#if defined(__ARM_FEATURE_DOTPROD)

#else
        accu_0 = vaddq_s32(accu_0, vmovl_s16(vget_low_s16(accula_0)));
        accu_0 = vaddq_s32(accu_0, vmovl_high_s16(accula_0));
        accu_1 = vaddq_s32(accu_1, vmovl_s16(vget_low_s16(accula_1)));
        accu_1 = vaddq_s32(accu_1, vmovl_high_s16(accula_1));
        accu_2 = vaddq_s32(accu_2, vmovl_s16(vget_low_s16(accula_2)));
        accu_2 = vaddq_s32(accu_2, vmovl_high_s16(accula_2));
        accu_3 = vaddq_s32(accu_3, vmovl_s16(vget_low_s16(accula_3)));
        accu_3 = vaddq_s32(accu_3, vmovl_high_s16(accula_3));
#endif
    }
    accu_0 = vaddq_s32(accu_0, accu_1);
    accu_2 = vaddq_s32(accu_2, accu_3);
    accu_0 = vaddq_s32(accu_0, accu_2);
    int sumi = vaddlvq_s32(accu_0);
    *s = (float)sumi;

#endif
}