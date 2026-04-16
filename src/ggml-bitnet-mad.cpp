#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include "ggml-bitnet.h"

__attribute__((constructor))
static void mad_init() {
    // Silencio soberano en producción, o telemetría mínima si se desea.
}

#define QK_I2_S 128

#if defined(__AVX2__)
#warning "AVX2 IS DEFINED IN GGML-BITNET-MAD"
#else
#warning "AVX2 IS NOT NOT NOT DEFINED IN GGML-BITNET-MAD"
#endif

extern "C" {

static inline float hsum_i32_8(__m256i x) {
    __m128i res = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extractf128_si256(x, 1));
    res = _mm_add_epi32(res, _mm_srli_si128(res, 8));
    res = _mm_add_epi32(res, _mm_srli_si128(res, 4));
    return (float)_mm_cvtsi128_si32(res);
}

size_t quantize_i2_s(const float * src, void * dst, int64_t nrow, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);
    uint8_t * out = (uint8_t *)dst;

    for (int64_t r = 0; r < nrow; r++) {
        const float * x = src + (size_t)r * n_per_row;
        uint8_t * y = out + (size_t)r * row_size;

        for (int64_t i = 0; i < n_per_row / QK_I2_S; i++) {
            const float * block_src = x + (size_t)i * QK_I2_S;
            uint8_t * block_dst = y + (size_t)i * 36;

            float max_val = 0.0f;
            for (int k = 0; k < QK_I2_S; k++) {
                max_val = fmaxf(max_val, fabsf(block_src[k]));
            }
            float scale = max_val;

            memset(block_dst, 0, 32);
            for (int k = 0; k < QK_I2_S; k++) {
                float val = block_src[k];
                uint8_t q = (fabsf(val) < 1e-6f) ? 1 : ((val > 0) ? 2 : 0);
                int byte_idx = k % 32;
                int bit_shift = 6 - 2 * (k / 32);
                block_dst[byte_idx] |= (q << bit_shift);
            }
            *(float *)(block_dst + 32) = scale;
        }
    }
    return (size_t)nrow * row_size;
}

void dequantize_row_i2_s_block(const void * vx, float * y, int64_t n) {
    const uint8_t * x = (const uint8_t *)vx;
    const int nb = (int)(n / QK_I2_S);
    const uint8_t mask = 0x03;

    for (int i = 0; i < nb; i++) {
        const uint8_t * block_data = x + (size_t)i * 36;
        const float scale = *(const float *)(block_data + 32);

        for (int k = 0; k < QK_I2_S; k++) {
            int byte_idx = k % 32;
            int bit_shift = 6 - 2 * (k / 32);
            uint8_t q = (block_data[byte_idx] >> bit_shift) & mask;
            y[(size_t)i * QK_I2_S + k] = ((float)q - 1.0f) * scale;
        }
    }
}

void ggml_vec_dot_i2_i8_s_1x1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t  *)vy;
    (void)by;
    const int nb = n / QK_I2_S;
    const __m256i mask  = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);
    const __m256i one8  = _mm256_set1_epi8(1);

    for (int row = 0; row < nrc; row++) {
        float row_sum = 0.0f;
        const uint8_t * x_row = x + (size_t)row * bx;
        const int8_t  * y_row = y + (size_t)row * n; 

        for (int i = 0; i < nb; i++) {
            const uint8_t * px = x_row + (size_t)i * 36;
            const int8_t  * py = y_row + (size_t)i * 132;
            const float w_scale = *(const float *)(px + 32);
            const float a_scale = *(const float *)(py + 128);
            const float scale = w_scale * a_scale;

            __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
            __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
            __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
            __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

            xq8_0 = _mm256_and_si256(xq8_0, mask);
            xq8_1 = _mm256_and_si256(xq8_1, mask);
            xq8_2 = _mm256_and_si256(xq8_2, mask);
            xq8_3 = _mm256_and_si256(xq8_3, mask);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

            __m256i accu32_y = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(one8, yq8_0), _mm256_maddubs_epi16(one8, yq8_1)),
                                                _mm256_add_epi16(_mm256_maddubs_epi16(one8, yq8_2), _mm256_maddubs_epi16(one8, yq8_3)));

            __m256i accu32 = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, yq8_0), _mm256_maddubs_epi16(xq8_1, yq8_1)),
                                               _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, yq8_2), _mm256_maddubs_epi16(xq8_3, yq8_3)));

            int bits = (int)hsum_i32_8(_mm256_madd_epi16(accu32, one16));
            int acts = (int)hsum_i32_8(_mm256_madd_epi16(accu32_y, one16));
            float diff = (float)(bits - acts) * scale;
            if (row == 0 && i < 2) {
                // Diagnostic probe
                printf("MAD DBG: i=%d w_s=%.4f a_s=%.4f bits=%d acts=%d diff=%.4f\n", i, w_scale, a_scale, bits, acts, diff);
            }
            row_sum += diff;
        }
        if (row == 0) printf("MAD DBG: row_sum=%.4f\n", row_sum);
        s[row * bs] = row_sum;
    }
#else
    (void)n; (void)s; (void)bs; (void)vx; (void)bx; (void)vy; (void)by; (void)nrc;
#endif
}

void ggml_vec_dot_i2_i8_s_1xN(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
#if defined(__AVX2__)
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t  *)vy;
    (void)by;
    static const int PAS = 8;
    const int nb = n / QK_I2_S;
    const __m256i mask  = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);
    const __m256i one8  = _mm256_set1_epi8(1);

    for (int row = 0; row < nrc; row += PAS) {
        int act = (row + PAS <= nrc) ? PAS : (nrc - row);
        float sums[PAS];
        for (int rb = 0; rb < PAS; rb++) sums[rb] = 0.0f;

        for (int i = 0; i < nb; i++) {
            const int8_t * py = y + (size_t)i * 132;
            const float a_scale = *(const float *)(py + 128);

            __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py + 0));
            __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
            __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
            __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));

            __m256i ysum_16 = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(one8, yq8_0), _mm256_maddubs_epi16(one8, yq8_1)),
                                                _mm256_add_epi16(_mm256_maddubs_epi16(one8, yq8_2), _mm256_maddubs_epi16(one8, yq8_3)));
            int acts = (int)hsum_i32_8(_mm256_madd_epi16(ysum_16, one16));

            for (int rb = 0; rb < act; rb++) {
                const uint8_t * px = x + (size_t)(row + rb) * bx + (size_t)i * 36;
                const float w_scale = *(const float *)(px + 32);
                const float scale = w_scale * a_scale;

                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);

                xq8_0 = _mm256_and_si256(xq8_0, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_3 = _mm256_and_si256(xq8_3, mask);

                __m256i accu32 = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, yq8_0), _mm256_maddubs_epi16(xq8_1, yq8_1)),
                                                   _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, yq8_2), _mm256_maddubs_epi16(xq8_3, yq8_3)));

                int bits = (int)hsum_i32_8(_mm256_madd_epi16(accu32, one16));
                float diff = (float)(bits - acts) * scale;
                sums[rb] += diff;
            }
        }
        for (int rb = 0; rb < act; rb++) s[(row + rb) * bs] = sums[rb];
    }
#else
    (void)n; (void)s; (void)bs; (void)vx; (void)bx; (void)vy; (void)by; (void)nrc;
#endif
}

void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    ggml_vec_dot_i2_i8_s_1xN(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_i2_i8_s_1x4_32W(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    ggml_vec_dot_i2_i8_s(n, s, bs, vx, bx, vy, by, nrc);
}

void ggml_vec_dot_i2_i8_s_Nx1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    for (int i = 0; i < nrc; i++) {
        ggml_vec_dot_i2_i8_s_1x1(n, (float *)((char *)s + (size_t)i * bs), bs, (const char *)vx + (size_t)i * bx, bx, vy, by, 1);
    }
}

// BRIDGE IMPLEMENTATION
bool ggml_bitnet_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    (void)dst;
    return (src0->type == GGML_TYPE_I2_S && (src1->type == GGML_TYPE_I8_S || src1->type == GGML_TYPE_F32));
}

size_t ggml_bitnet_mul_mat_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    (void)src0;
    (void)dst;
    if (src1->type == GGML_TYPE_F32) {
        // We need ne00 bytes per thread for quantization
        // We return the MAX size needed (assuming up to GGML_MAX_NTHREADS)
        // or just calculate it based on the current tensor size.
        // Usually GGML calls this once. We'll provide enough for many threads.
        return (size_t)src1->ne[0] * 128; // 128 threads max fallback safety
    }
    return 0;
}

// Internal utilities matching the original BitNet fork
static inline int nearest_int(float fval) {
    if (fabsf(fval) > 4194303.f) return (fval > 0) ? 4194303 : -4194303;
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

static void bitnet_float_act_quant(const int K, const float* B, int8_t* dst, float* act_scale) {
    double max_val = 0.00001;
    for (int i = 0; i < K; ++i) {
        double val = (double)fabs((double)B[i]);
        if (val > max_val) max_val = val;
    }
    float s = 127 / (float)max_val;
    act_scale[0] = s;
    for (int i = 0; i < K; ++i) {
        int v = nearest_int(B[i] * s);
        if (v >  127) v = 127;
        if (v < -128) v = -128;
        dst[i] = (int8_t)v;
    }
}

void ggml_bitnet_mul_mat(
    const struct ggml_tensor * src0,
    const struct ggml_tensor * src1,
    struct ggml_tensor * dst,
    void * wdata,
    int ith, int nth) {
    
    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne11 = src1->ne[1];

    const size_t nb01 = src0->nb[1];
    const size_t nb11 = src1->nb[1];
    const size_t nbd  = dst->nb[1];

    const void * src0_data = ggml_bitnet_get_data(src0);
    const void * src1_data = src1->data;

    // Use wdata if available, otherwise fallback (though wdata should be there if we asked)
    int8_t * thread_buffer = NULL;
    if (wdata && src1->type == GGML_TYPE_F32) {
        thread_buffer = (int8_t *)wdata + (size_t)ith * ne00;
    }

    float s_act;
    for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
        float * dst_col = (float *)((char *)dst->data + i11*nbd);
        const void * src1_col = (const char *)src1_data + i11*nb11;
        const void * final_src1_col = src1_col;

        if (src1->type == GGML_TYPE_F32) {
            if (thread_buffer == NULL) return; // Should not happen with proper wsize
            bitnet_float_act_quant(ne00, (const float *)src1_col, thread_buffer, &s_act);
            final_src1_col = thread_buffer;
            ggml_vec_dot_i2_i8_s_1xN(ne00, dst_col, sizeof(float), src0_data, nb01, final_src1_col, 0, ne01);
            for (int64_t i = 0; i < ne01; ++i) {
                dst_col[i] /= s_act;
            }
        } else {
            ggml_vec_dot_i2_i8_s_1xN(ne00, dst_col, sizeof(float), src0_data, nb01, final_src1_col, 0, ne01);
        }
    }
}

} // extern "C"