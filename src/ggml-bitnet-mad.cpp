#include <vector>
#include <type_traits>
#include <assert.h>
#include "ggml-bitnet.h"
#include "ggml-quants.h"
#include "gemm-config.h"
#include "ggml-cpu-impl.h"
#include <cmath>
#include <cstring>

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#define QK_I2_S 128
#endif

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
#elif defined(__ARM_NEON)
static inline int32x4_t bitnet_neon_dot_block128_loaded(
    int32x4_t accu,
    int8x16_t q0_lo, int8x16_t q0_hi,
    int8x16_t q1_lo, int8x16_t q1_hi,
    int8x16_t q2_lo, int8x16_t q2_hi,
    int8x16_t q3_lo, int8x16_t q3_hi,
    int8x16_t y0_lo, int8x16_t y0_hi,
    int8x16_t y1_lo, int8x16_t y1_hi,
    int8x16_t y2_lo, int8x16_t y2_hi,
    int8x16_t y3_lo, int8x16_t y3_hi) {
#if defined(__ARM_FEATURE_DOTPROD)
    accu = vdotq_s32(accu, q0_lo, y0_lo);
    accu = vdotq_s32(accu, q0_hi, y0_hi);
    accu = vdotq_s32(accu, q1_lo, y1_lo);
    accu = vdotq_s32(accu, q1_hi, y1_hi);
    accu = vdotq_s32(accu, q2_lo, y2_lo);
    accu = vdotq_s32(accu, q2_hi, y2_hi);
    accu = vdotq_s32(accu, q3_lo, y3_lo);
    accu = vdotq_s32(accu, q3_hi, y3_hi);
#else
    // Accumulate a single 128-element block in int16x8_t (max |lane| <= 16 * 3 * 128 = 6144 < 32767)
    // and widen to int32x4_t every block so large n never overflows int16_t.
    int16x8_t sum16 = vmull_s8(vget_low_s8(q0_lo), vget_low_s8(y0_lo));
    sum16 = vmlal_s8(sum16, vget_high_s8(q0_lo), vget_high_s8(y0_lo));
    sum16 = vmlal_s8(sum16, vget_low_s8(q0_hi),  vget_low_s8(y0_hi));
    sum16 = vmlal_s8(sum16, vget_high_s8(q0_hi), vget_high_s8(y0_hi));
    sum16 = vmlal_s8(sum16, vget_low_s8(q1_lo),  vget_low_s8(y1_lo));
    sum16 = vmlal_s8(sum16, vget_high_s8(q1_lo), vget_high_s8(y1_lo));
    sum16 = vmlal_s8(sum16, vget_low_s8(q1_hi),  vget_low_s8(y1_hi));
    sum16 = vmlal_s8(sum16, vget_high_s8(q1_hi), vget_high_s8(y1_hi));
    sum16 = vmlal_s8(sum16, vget_low_s8(q2_lo),  vget_low_s8(y2_lo));
    sum16 = vmlal_s8(sum16, vget_high_s8(q2_lo), vget_high_s8(y2_lo));
    sum16 = vmlal_s8(sum16, vget_low_s8(q2_hi),  vget_low_s8(y2_hi));
    sum16 = vmlal_s8(sum16, vget_high_s8(q2_hi), vget_high_s8(y2_hi));
    sum16 = vmlal_s8(sum16, vget_low_s8(q3_lo),  vget_low_s8(y3_lo));
    sum16 = vmlal_s8(sum16, vget_high_s8(q3_lo), vget_high_s8(y3_lo));
    sum16 = vmlal_s8(sum16, vget_low_s8(q3_hi),  vget_low_s8(y3_hi));
    sum16 = vmlal_s8(sum16, vget_high_s8(q3_hi), vget_high_s8(y3_hi));
    accu = vpadalq_s16(accu, sum16);
#endif
    return accu;
}

static inline int32x4_t bitnet_neon_dot_block128(
    int32x4_t accu,
    int8x16_t q0_lo, int8x16_t q0_hi,
    int8x16_t q1_lo, int8x16_t q1_hi,
    int8x16_t q2_lo, int8x16_t q2_hi,
    int8x16_t q3_lo, int8x16_t q3_hi,
    const int8_t * py) {
    return bitnet_neon_dot_block128_loaded(
        accu,
        q0_lo, q0_hi, q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi,
        vld1q_s8(py + 0),  vld1q_s8(py + 16),
        vld1q_s8(py + 32), vld1q_s8(py + 48),
        vld1q_s8(py + 64), vld1q_s8(py + 80),
        vld1q_s8(py + 96), vld1q_s8(py + 112));
}
#endif

size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__)
#if defined(ACT_PARALLEL)
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
    for (int i = 0; i < n / QK_I2_S; i++) {
        for (int j = 0; j < QK_I2_S; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2_S + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;            
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    // 32B for alignment
    return nrow * row_size / 4 + 32;
#else
    assert((nrow % 4) == 0 && "quantize_i2_s_1x4 requires nrow % 4 == 0");

    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
    int64_t n = nrow * n_per_row;

    double max = 0;
    for (int64_t i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc(n * sizeof(uint8_t));
    for (int64_t i=0; i<n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    uint8_t* out = (uint8_t*)dst;
    memset(out, 0, (size_t)(n / 4));

    // for each group of 4 rows, for each column, write one byte
    int64_t nrow4 = nrow / 4;
    for (int64_t rg = 0; rg < nrow4; rg++) {
        int64_t r0 = rg * 4 + 0;
        int64_t r1 = rg * 4 + 1;
        int64_t r2 = rg * 4 + 2;
        int64_t r3 = rg * 4 + 3;

        int64_t base = rg * n_per_row;

        for (int64_t col = 0; col < n_per_row; col++) {
            uint8_t q0 = q8[r0 * n_per_row + col];
            uint8_t q1 = q8[r1 * n_per_row + col];
            uint8_t q2 = q8[r2 * n_per_row + col];
            uint8_t q3 = q8[r3 * n_per_row + col];

            uint8_t packed = (uint8_t)((q0 << 6) | (q1 << 4) | (q2 << 2) | (q3 << 0));
            out[base + col] = packed;
        }
    }

    // store scale at the end of quantized data (same location pattern as quantize_i2_s)
    float* scale_ptr = (float*)((char*)out + n / 4);
    scale_ptr[0] = (float)i2_scale;

    free(q8);

    // return size (keep same formula as quantize_i2_s)
    return nrow * row_size / 4 + 32;
#endif
#elif defined(__ARM_NEON)
    (void)quant_weights;
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int64_t n = nrow * n_per_row;

    // f32 -> q8
    double max = 0;
    for (int64_t i = 0; i < n; ++i) {
        max = fmax(max, (double)fabs((double)src[i]));
    }
    double i2_scale = max;

    uint8_t* q8 = (uint8_t*)malloc((size_t)n * sizeof(uint8_t));
    for (int64_t i = 0; i < n; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, (size_t)(n / 4));

    // q8 -> 0, 1, 2
    //       |  |  |
    //      -1, 0, 1

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int64_t i = 0; i < n / QK_I2_S; i++) {
        for (int j = 0; j < QK_I2_S; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (uint8_t)(q8[i * QK_I2_S + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = (float)i2_scale;

    free(q8);

    // 32B for alignment
    return (size_t)nrow * row_size / 4 + 32;
#endif
}

void ggml_vec_dot_i2_i8_s_1x1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    
    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i one16 = _mm256_set1_epi16(1);

    // 处理多行，nrc表示要处理的行数
    for (int row = 0; row < nrc; row++) {
        __m256i accu = _mm256_setzero_si256();
        
        // 计算当前行的x指针偏移
        const uint8_t * x_row = x + row * bx / 4;
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x_row + i * 1024;     // 32 * 32
            const int8_t  *py = y + i * 4096;         // 32 * 128
            __m256i accu32 = _mm256_setzero_si256();
            
            for (int j = 0; j < 32; j++) {
                // 128 index
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                // each 32 index
                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                // each 32 index
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_0, xq8_1));
                accu32 = _mm256_add_epi16(accu32, _mm256_add_epi16(xq8_2, xq8_3));

                px += 32;
                py += 128;
            }
            accu = _mm256_add_epi32(_mm256_madd_epi16(accu32, one16), accu);
        }

        for (int i = 0; i < groupla_num; i++) {
            __m256i accula = _mm256_setzero_si256();
            const uint8_t *px = x_row + group32_num * 1024; // 32 * 32
            const int8_t  *py = y + group32_num * 4096;     // 32 * 128
            
            for (int j = 0; j < la_num; j++) {
                // 128 index
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                // each 32 index
                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);

                // each 32 index
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

                xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_0, xq8_1));
                accula = _mm256_add_epi16(accula, _mm256_add_epi16(xq8_2, xq8_3));

                px += 32;
                py += 128;
            }
            accu = _mm256_add_epi32(accu, _mm256_madd_epi16(accula, one16));
        }
        
        int sumi = hsum_i32_8(accu);
        s[row] = (float)sumi;
    }
#elif defined(__ARM_NEON)
    (void)bs;
    (void)by;
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t *)vy;

    const int nb = n / QK_I2_S;
    const uint8x16_t mask = vdupq_n_u8(3);

    for (int row = 0; row < nrc; row++) {
        int32x4_t accu = vdupq_n_s32(0);
        const uint8_t * x_row = x + (size_t)row * bx / 4;

        for (int j = 0; j < nb; j++) {
            const uint8_t * px = x_row + (size_t)j * 32;
            const int8_t  * py = y + (size_t)j * 128;

            const uint8x16_t x_lo = vld1q_u8(px);
            const uint8x16_t x_hi = vld1q_u8(px + 16);

            const int8x16_t q0_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 6), mask));
            const int8x16_t q0_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 6), mask));
            const int8x16_t q1_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 4), mask));
            const int8x16_t q1_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 4), mask));
            const int8x16_t q2_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 2), mask));
            const int8x16_t q2_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 2), mask));
            const int8x16_t q3_lo = vreinterpretq_s8_u8(vandq_u8(x_lo, mask));
            const int8x16_t q3_hi = vreinterpretq_s8_u8(vandq_u8(x_hi, mask));

            accu = bitnet_neon_dot_block128(
                accu, q0_lo, q0_hi, q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi, py);
        }

        s[row] = (float)vaddlvq_s32(accu);
    }
#endif
}

void ggml_vec_dot_i2_i8_s_1x4_32W(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    // 处理多行，nrc表示要处理的行数
    for (int row = 0; row < nrc; row+=4) {
        __m256i accu[4];
        for(int rb = 0; rb < 4; rb++) {
            accu[rb] = _mm256_setzero_si256();
        }
        const uint8_t * x_row = x + (row) * bx / 4;
        // 计算当前行的x指针偏移
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t * px = x_row + i * 1024 * 4;
            __m256i accu32[4];
            for(int rb = 0; rb < 4; rb++) {
                accu32[rb] = _mm256_setzero_si256();
            }
            const int8_t  *py = y + i * 4096; 
            
            for (int j = 0; j < 32 * 4; j++) {
                // each 32 index
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i xq8[4];
                xq8[3] = _mm256_loadu_si256((const __m256i*)(px));
                xq8[2] = _mm256_srli_epi16(xq8[3], 2);
                xq8[1] = _mm256_srli_epi16(xq8[3], 4);
                xq8[0] = _mm256_srli_epi16(xq8[3], 6);
                xq8[3] = _mm256_and_si256(xq8[3], mask);
                xq8[2] = _mm256_and_si256(xq8[2], mask);
                xq8[1] = _mm256_and_si256(xq8[1], mask);
                xq8[0] = _mm256_and_si256(xq8[0], mask);
                for (int rb = 0; rb < 4; rb++)
                {
                    xq8[rb] = _mm256_maddubs_epi16(xq8[rb], yq8_0);
                    accu32[rb] = _mm256_add_epi16(accu32[rb], xq8[rb]);
                }
                px += 32;
                py += 32;
            }
            for(int rb = 0; rb < 4; rb++) {
                accu[rb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[rb], one16), accu[rb]);
            } 
        }

        for (int i = 0; i < groupla_num; i++) {
            const int8_t  *py = y + group32_num * 4096;     // 32 * 128
            __m256i accula[4];
            for(int rb = 0; rb < 4; rb++) {
                accula[rb] = _mm256_setzero_si256();
            }
            const uint8_t * px = x_row + group32_num * 1024 * 4;
            
            for (int j = 0; j < la_num * 4; j++) {
                // each 32 index
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i xq8[4];
                xq8[3] = _mm256_loadu_si256((const __m256i*)(px));
                xq8[2] = _mm256_srli_epi16(xq8[3], 2);
                xq8[1] = _mm256_srli_epi16(xq8[3], 4);
                xq8[0] = _mm256_srli_epi16(xq8[3], 6);
                xq8[3] = _mm256_and_si256(xq8[3], mask);
                xq8[2] = _mm256_and_si256(xq8[2], mask);
                xq8[1] = _mm256_and_si256(xq8[1], mask);
                xq8[0] = _mm256_and_si256(xq8[0], mask);

                for (int rb = 0; rb < 4; rb++) {
                    xq8[rb] = _mm256_maddubs_epi16(xq8[rb], yq8_0);
                    accula[rb] = _mm256_add_epi16(accula[rb], xq8[rb]);
                }
                px += 32;
                py += 32;
            }
            for(int rb = 0; rb < 4; rb++) {
                accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(accula[rb], one16));
            } 
        }
        
        for(int rb = 0; rb < 4; rb++) {
            int sumi = hsum_i32_8(accu[rb]);
            s[row + rb] = (float)sumi;
        }
    }
#elif defined(__ARM_NEON)
    (void)n; (void)s; (void)bs; (void)vx; (void)bx; (void)vy; (void)by; (void)nrc;
#endif
}

void ggml_vec_dot_i2_i8_s_1xN(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    // 处理多行，nrc表示要处理的行数
    for (int row = 0; row < nrc; row+=PARALLEL_SIZE) {
        //__m256i accu = _mm256_setzero_si256();
        __m256i accu[PARALLEL_SIZE];
        const uint8_t * x_row[PARALLEL_SIZE];
        for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
            accu[rb] = _mm256_setzero_si256();
            x_row[rb] = x + (row + rb) * bx / 4;
        }
        // 计算当前行的x指针偏移
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t * px[PARALLEL_SIZE];
            __m256i accu32[PARALLEL_SIZE];
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 1024;     // 32 * 32
                accu32[rb] = _mm256_setzero_si256();
            }
            const int8_t  *py = y + i * 4096;         // 32 * 128
            
            for (int j = 0; j < 32; j++) {
                // each 32 index
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));
                for (int rb = 0; rb < PARALLEL_SIZE; rb++)
                {
                    __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px[rb]));
                    __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                    __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                    __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                    // each 32 index
                    xq8_3 = _mm256_and_si256(xq8_3, mask);
                    xq8_2 = _mm256_and_si256(xq8_2, mask);
                    xq8_1 = _mm256_and_si256(xq8_1, mask);
                    xq8_0 = _mm256_and_si256(xq8_0, mask);

                    xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                    xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                    xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                    xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                    accu32[rb] = _mm256_add_epi16(accu32[rb], _mm256_add_epi16(xq8_0, xq8_1));
                    accu32[rb] = _mm256_add_epi16(accu32[rb], _mm256_add_epi16(xq8_2, xq8_3));

                    px[rb] += 32;
                }
                py += 128;
            }
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[rb], one16), accu[rb]);
            } 
        }

        for (int i = 0; i < groupla_num; i++) {
            const int8_t  *py = y + group32_num * 4096;     // 32 * 128
            const uint8_t * px[PARALLEL_SIZE];
            __m256i accula[PARALLEL_SIZE];
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 1024;     // 32 * 32
                accula[rb] = _mm256_setzero_si256();
            }
            
            for (int j = 0; j < la_num; j++) {
                // each 32 index
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    // 128 index
                    __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px[rb]));
                    __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                    __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                    __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                    // each 32 index
                    xq8_3 = _mm256_and_si256(xq8_3, mask);
                    xq8_2 = _mm256_and_si256(xq8_2, mask);
                    xq8_1 = _mm256_and_si256(xq8_1, mask);
                    xq8_0 = _mm256_and_si256(xq8_0, mask);

                    

                    xq8_0 = _mm256_maddubs_epi16(xq8_0, yq8_0);
                    xq8_1 = _mm256_maddubs_epi16(xq8_1, yq8_1);
                    xq8_2 = _mm256_maddubs_epi16(xq8_2, yq8_2);
                    xq8_3 = _mm256_maddubs_epi16(xq8_3, yq8_3);

                    accula[rb] = _mm256_add_epi16(accula[rb], _mm256_add_epi16(xq8_0, xq8_1));
                    accula[rb] = _mm256_add_epi16(accula[rb], _mm256_add_epi16(xq8_2, xq8_3));

                    px[rb] += 32;
                }
                py += 128;
            }
            for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(accula[rb], one16));
            } 
        }
        
        for(int rb = 0; rb < PARALLEL_SIZE; rb++) {
            int sumi = hsum_i32_8(accu[rb]);
            s[row + rb] = (float)sumi;
        }
    }
#elif defined(__ARM_NEON)
    (void)bs;
    (void)by;
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t *)vy;

    const int nb = n / QK_I2_S;
    const uint8x16_t mask = vdupq_n_u8(3);

    int row = 0;
    for (; row + PARALLEL_SIZE <= nrc; row += PARALLEL_SIZE) {
        int32x4_t accu[PARALLEL_SIZE];
        const uint8_t * x_row[PARALLEL_SIZE];

        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            accu[rb] = vdupq_n_s32(0);
            x_row[rb] = x + (size_t)(row + rb) * bx / 4;
        }

        for (int j = 0; j < nb; j++) {
            const int8_t * py = y + (size_t)j * 128;
            const int8x16_t y0_lo = vld1q_s8(py + 0);
            const int8x16_t y0_hi = vld1q_s8(py + 16);
            const int8x16_t y1_lo = vld1q_s8(py + 32);
            const int8x16_t y1_hi = vld1q_s8(py + 48);
            const int8x16_t y2_lo = vld1q_s8(py + 64);
            const int8x16_t y2_hi = vld1q_s8(py + 80);
            const int8x16_t y3_lo = vld1q_s8(py + 96);
            const int8x16_t y3_hi = vld1q_s8(py + 112);

            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                const uint8_t * px = x_row[rb] + (size_t)j * 32;
                const uint8x16_t x_lo = vld1q_u8(px);
                const uint8x16_t x_hi = vld1q_u8(px + 16);

                const int8x16_t q0_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 6), mask));
                const int8x16_t q0_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 6), mask));
                const int8x16_t q1_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 4), mask));
                const int8x16_t q1_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 4), mask));
                const int8x16_t q2_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 2), mask));
                const int8x16_t q2_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 2), mask));
                const int8x16_t q3_lo = vreinterpretq_s8_u8(vandq_u8(x_lo, mask));
                const int8x16_t q3_hi = vreinterpretq_s8_u8(vandq_u8(x_hi, mask));

                accu[rb] = bitnet_neon_dot_block128_loaded(
                    accu[rb],
                    q0_lo, q0_hi, q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi,
                    y0_lo, y0_hi, y1_lo, y1_hi, y2_lo, y2_hi, y3_lo, y3_hi);
            }
        }

        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            s[row + rb] = (float)vaddlvq_s32(accu[rb]);
        }
    }

    for (; row < nrc; row++) {
        int32x4_t accu = vdupq_n_s32(0);
        const uint8_t * x_row = x + (size_t)row * bx / 4;

        for (int j = 0; j < nb; j++) {
            const uint8_t * px = x_row + (size_t)j * 32;
            const int8_t  * py = y + (size_t)j * 128;

            const uint8x16_t x_lo = vld1q_u8(px);
            const uint8x16_t x_hi = vld1q_u8(px + 16);

            const int8x16_t q0_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 6), mask));
            const int8x16_t q0_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 6), mask));
            const int8x16_t q1_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 4), mask));
            const int8x16_t q1_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 4), mask));
            const int8x16_t q2_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 2), mask));
            const int8x16_t q2_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 2), mask));
            const int8x16_t q3_lo = vreinterpretq_s8_u8(vandq_u8(x_lo, mask));
            const int8x16_t q3_hi = vreinterpretq_s8_u8(vandq_u8(x_hi, mask));

            accu = bitnet_neon_dot_block128(
                accu, q0_lo, q0_hi, q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi, py);
        }

        s[row] = (float)vaddlvq_s32(accu);
    }
#endif
}

void ggml_vec_dot_i2_i8_s_Nx1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i one16 = _mm256_set1_epi16(1);

    for (int col = 0; col < nrc; col += PARALLEL_SIZE) {
        __m256i accu[PARALLEL_SIZE];

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            accu[iy] = _mm256_setzero_si256();
        }

        const int8_t * y_col = y + col * by;
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x + i * 1024;
            const int8_t  *py = y_col + i * 4096;
            __m256i accu32[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu32[iy] = _mm256_setzero_si256();
            }

            for (int j = 0; j < 32; j++) {

                __m256i xq8   = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_3 = _mm256_and_si256(xq8, mask);
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8, 6), mask);

                for (int iy = 0; iy < PARALLEL_SIZE; iy++)
                {
                    accu32[iy] = _mm256_add_epi16(accu32[iy], _mm256_add_epi16(
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, _mm256_loadu_si256((const __m256i*)(py + 0 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_1, _mm256_loadu_si256((const __m256i*)(py + 1 * 32 + iy * by)))),
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, _mm256_loadu_si256((const __m256i*)(py + 2 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_3, _mm256_loadu_si256((const __m256i*)(py + 3 * 32 + iy * by))))));
                }

                px += 32;
                py += 128;
            }

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = _mm256_add_epi32(_mm256_madd_epi16(accu32[iy], one16), accu[iy]);
            }
        }

        for (int i = 0; i < groupla_num; i++) {
            const uint8_t *px = x + group32_num * 1024;
            const int8_t  *py = y_col + group32_num * 4096;
            __m256i accula[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accula[iy] = _mm256_setzero_si256();
            }
            
            for (int j = 0; j < la_num; j++) {
                
                __m256i xq8   = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_3 = _mm256_and_si256(xq8, mask);
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8, 6), mask);

                for (int iy = 0; iy < PARALLEL_SIZE; iy++)
                {
                    accula[iy] = _mm256_add_epi16(accula[iy], _mm256_add_epi16(
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, _mm256_loadu_si256((const __m256i*)(py + 0 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_1, _mm256_loadu_si256((const __m256i*)(py + 1 * 32 + iy * by)))),
                                    _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, _mm256_loadu_si256((const __m256i*)(py + 2 * 32 + iy * by))),
                                                    _mm256_maddubs_epi16(xq8_3, _mm256_loadu_si256((const __m256i*)(py + 3 * 32 + iy * by))))));
                }

                px += 32;
                py += 128;
            }

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = _mm256_add_epi32(_mm256_madd_epi16(accula[iy], one16), accu[iy]);
            }
        }

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            int sumi = hsum_i32_8(accu[iy]);
            s[(col + iy) * bs] = (float)sumi;
        }
    }
#elif defined(__ARM_NEON)
    (void)bx;
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t *)vy;

    const int nb = n / QK_I2_S;
    const uint8x16_t mask = vdupq_n_u8(3);

    int col = 0;
    for (; col + PARALLEL_SIZE <= nrc; col += PARALLEL_SIZE) {
        int32x4_t accu[PARALLEL_SIZE];
        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            accu[iy] = vdupq_n_s32(0);
        }

        const int8_t * y_col = y + (size_t)col * by;

        for (int j = 0; j < nb; j++) {
            const uint8_t * px = x + (size_t)j * 32;
            const int8_t  * py = y_col + (size_t)j * 128;

            const uint8x16_t x_lo = vld1q_u8(px);
            const uint8x16_t x_hi = vld1q_u8(px + 16);

            const int8x16_t q0_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 6), mask));
            const int8x16_t q0_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 6), mask));
            const int8x16_t q1_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 4), mask));
            const int8x16_t q1_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 4), mask));
            const int8x16_t q2_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 2), mask));
            const int8x16_t q2_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 2), mask));
            const int8x16_t q3_lo = vreinterpretq_s8_u8(vandq_u8(x_lo, mask));
            const int8x16_t q3_hi = vreinterpretq_s8_u8(vandq_u8(x_hi, mask));

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = bitnet_neon_dot_block128(
                    accu[iy], q0_lo, q0_hi, q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi,
                    py + (size_t)iy * by);
            }
        }

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            s[(size_t)(col + iy) * bs] = (float)vaddlvq_s32(accu[iy]);
        }
    }

    for (; col < nrc; col++) {
        int32x4_t accu = vdupq_n_s32(0);
        const int8_t * y_col = y + (size_t)col * by;

        for (int j = 0; j < nb; j++) {
            const uint8_t * px = x + (size_t)j * 32;
            const int8_t  * py = y_col + (size_t)j * 128;

            const uint8x16_t x_lo = vld1q_u8(px);
            const uint8x16_t x_hi = vld1q_u8(px + 16);

            const int8x16_t q0_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 6), mask));
            const int8x16_t q0_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 6), mask));
            const int8x16_t q1_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 4), mask));
            const int8x16_t q1_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 4), mask));
            const int8x16_t q2_lo = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_lo, 2), mask));
            const int8x16_t q2_hi = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(x_hi, 2), mask));
            const int8x16_t q3_lo = vreinterpretq_s8_u8(vandq_u8(x_lo, mask));
            const int8x16_t q3_hi = vreinterpretq_s8_u8(vandq_u8(x_hi, mask));

            accu = bitnet_neon_dot_block128(
                accu, q0_lo, q0_hi, q1_lo, q1_hi, q2_lo, q2_hi, q3_lo, q3_hi, py);
        }

        s[(size_t)col * bs] = (float)vaddlvq_s32(accu);
    }
#endif
}


void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__ARM_NEON)
    if (nrc == 1) {
        ggml_vec_dot_i2_i8_s_1x1(n, s, bs, vx, bx, vy, by, 1);
    } else {
#if defined(ACT_PARALLEL)
        ggml_vec_dot_i2_i8_s_Nx1(n, s, bs, vx, bx, vy, by, nrc);
#else
        ggml_vec_dot_i2_i8_s_1xN(n, s, bs, vx, bx, vy, by, nrc);
#endif
    }
#else
    if (nrc % PARALLEL_SIZE == 0)
    {
#if defined(ACT_PARALLEL)
        ggml_vec_dot_i2_i8_s_Nx1(n, s, bs, vx, bx, vy, by, nrc);
#else
        ggml_vec_dot_i2_i8_s_1xN(n, s, bs, vx, bx, vy, by, nrc);
#endif
    }
    else
    {
        ggml_vec_dot_i2_i8_s_1x1(n, s, bs, vx, bx, vy, by, nrc);
    }
#endif
}