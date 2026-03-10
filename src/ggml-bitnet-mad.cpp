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
#define QK_I2_S 64
#elif defined(__VSX__) || defined(__ALTIVEC__) || defined(__powerpc64__) || defined(__powerpc__) || defined(__ppc__)
#define QK_I2_S 128
#include <altivec.h>

// Detect POWER8 VSX vs G5 AltiVec-only
#if defined(__VSX__)
#define BITNET_POWER8_VSX 1
#else
#define BITNET_G5_ALTIVEC 1
#endif

// Horizontal sum: reduce vector int32 to scalar
#if defined(BITNET_POWER8_VSX)
// POWER8 VSX: use vec_sld chain (fast on LE)
static inline __attribute__((always_inline)) int hsum_i32_4_ppc(vector signed int v) {
    vector signed int sum = vec_add(v, vec_sld(v, v, 8));
    sum = vec_add(sum, vec_sld(sum, sum, 4));
    return vec_extract(sum, 0);
}
#else
// G5 AltiVec (big-endian): use vec_sums
static inline __attribute__((always_inline)) int hsum_i32_4_ppc(vector signed int v) {
    vector signed int zero = vec_splat_s32(0);
    vector signed int sum = vec_sums(v, zero);
    // vec_sums places result in element 3 on big-endian
    return vec_extract(sum, 3);
}
#endif

// Prefetch macros - G5 supports basic dcbt only, POWER8 has TH hints
#if defined(BITNET_POWER8_VSX)
// L3 resident prefetch (POWER8 extended hints)
#define I2S_DCBT_RESIDENT(addr) __asm__ __volatile__("dcbt 16, %0, 0" : : "b"(addr) : "memory")
#define I2S_DCBT_L2_RESIDENT(addr) __asm__ __volatile__("dcbt 2, %0, 0" : : "b"(addr) : "memory")
#define I2S_DCBT(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr) : "memory")
#define I2S_DCBT_L2(addr) __asm__ __volatile__("dcbt 0,%0,8" : : "r"(addr) : "memory")
#else
// G5 AltiVec: basic dcbt only (no TH field)
#define I2S_DCBT_RESIDENT(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr) : "memory")
#define I2S_DCBT_L2_RESIDENT(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr) : "memory")
#define I2S_DCBT(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr) : "memory")
#define I2S_DCBT_L2(addr) __asm__ __volatile__("dcbt 0,%0" : : "r"(addr) : "memory")
#endif

#define PPC_CACHE_LINE 128

// Shift/mask constants for 2-bit weight unpacking
// Use vec_splat_u8() to generate constants in-register (vspltisb instruction)
// instead of loading from memory - avoids Mach-O alignment issues on old Darwin
#define ppc_mask03 ((vector unsigned char)vec_splat_u8(3))
#define ppc_shift2 ((vector unsigned char)vec_splat_u8(2))
#define ppc_shift4 ((vector unsigned char)vec_splat_u8(4))
#define ppc_shift6 ((vector unsigned char)vec_splat_u8(6))

// Vector load abstraction: VSX supports unaligned, AltiVec requires 16-byte alignment
#if defined(BITNET_POWER8_VSX)
#define I2S_VEC_LD_UC(off, ptr) vec_vsx_ld(off, (const unsigned char *)(ptr))
#define I2S_VEC_LD_SC(off, ptr) vec_vsx_ld(off, (const signed char *)(ptr))
#else
// G5 AltiVec: vec_ld requires 16-byte aligned addresses.
// BitNet weight tensors are allocated by ggml with sufficient alignment.
// For activations that may not be aligned, we use vec_ld which masks the
// low 4 bits of the effective address (loads from aligned boundary).
#define I2S_VEC_LD_UC(off, ptr) vec_ld(off, (const unsigned char *)(ptr))
#define I2S_VEC_LD_SC(off, ptr) vec_ld(off, (const signed char *)(ptr))
#endif

// Process one 16-byte half of an I2_S block using vec_msum (vmsummbm)
// Available on both G5 AltiVec and POWER8 VSX
// always_inline is critical on Mach-O: without it, every call generates
// VRsave save/restore (mfspr/mtspr ~20 cycles each), devastating for a
// function called in the inner loop of every dot product.
static inline __attribute__((always_inline)) vector signed int i2s_ppc_half(
    const uint8_t * __restrict__ px, int px_off,
    const int8_t  * __restrict__ py, int py_off,
    vector signed int accu)
{
    vector unsigned char packed = I2S_VEC_LD_UC(px_off, px);

    // Unpack 4 groups from 2-bit packed weights
    vector unsigned char w0 = vec_and(vec_sr(packed, ppc_shift6), ppc_mask03);
    vector unsigned char w1 = vec_and(vec_sr(packed, ppc_shift4), ppc_mask03);
    vector unsigned char w2 = vec_and(vec_sr(packed, ppc_shift2), ppc_mask03);
    vector unsigned char w3 = vec_and(packed, ppc_mask03);

    // Load 16 bytes of activations from each of the 4 groups
    vector signed char y0 = I2S_VEC_LD_SC(py_off,      py);
    vector signed char y1 = I2S_VEC_LD_SC(py_off + 32,  py);
    vector signed char y2 = I2S_VEC_LD_SC(py_off + 64,  py);
    vector signed char y3 = I2S_VEC_LD_SC(py_off + 96,  py);

    // vec_msum: signed char * unsigned char -> accumulate to int32
    // vmsummbm instruction: 16 multiplies + 4 adds per call
    // Available on G4+ AltiVec and POWER8 VSX
    accu = vec_msum(y0, w0, accu);
    accu = vec_msum(y1, w1, accu);
    accu = vec_msum(y2, w2, accu);
    accu = vec_msum(y3, w3, accu);

    return accu;
}

#else
// Scalar fallback
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
            int group_idx = j / 16;
            int group_pos = j % 16;
            uint8_t temp = (q8[i * QK_I2_S + j] << (6 - 2 * group_idx));
            i2_weight[i * 16 + group_pos] |= temp;            
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n / 4);
    scale_ptr[0] = i2_scale;

    free(q8);

    // 32B for alignment
    return nrow * row_size / 4 + 32;


#elif defined(__VSX__) || defined(__ALTIVEC__) || defined(BITNET_G5_ALTIVEC)
    // PowerPC quantization (POWER8 VSX + G5 AltiVec)
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int64_t n_total = (int64_t)nrow * n_per_row;

    double max_val = 0;
    for (int64_t i = 0; i < n_total; ++i) {
        max_val = fmax(max_val, (double)fabs((double)src[i]));
    }
    double i2_scale = max_val;

    uint8_t* q8 = (uint8_t*)malloc(n_total * sizeof(uint8_t));
    for (int64_t i = 0; i < n_total; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;
    }

    memset(dst, 0, n_total * sizeof(uint8_t) / 4);

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int64_t i = 0; i < n_total / QK_I2_S; i++) {
        for (int j = 0; j < QK_I2_S; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2_S + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n_total / 4);
    scale_ptr[0] = (float)i2_scale;

    free(q8);
    return nrow * row_size / 4 + 32;

#else
    // Scalar fallback (PowerPC / generic)
    // Uses same packing format as x86 ACT_PARALLEL: 128 elements -> 32 bytes
    // Each byte: bits 7-6 = group0, 5-4 = group1, 3-2 = group2, 1-0 = group3
    size_t row_size = ggml_row_size(GGML_TYPE_I2_S, n_per_row);

    int64_t n_total = (int64_t)nrow * n_per_row;

    // f32 -> q8 (ternary: 0=-1, 1=0, 2=+1)
    double max_val = 0;
    for (int64_t i = 0; i < n_total; ++i) {
        max_val = fmax(max_val, (double)fabs((double)src[i]));
    }
    double i2_scale = max_val;

    uint8_t* q8 = (uint8_t*)malloc(n_total * sizeof(uint8_t));
    for (int64_t i = 0; i < n_total; i++) {
        if (fabs((double)(src[i])) < 1e-6) {
            q8[i] = 1;  // zero -> 1
            continue;
        }
        q8[i] = (double)src[i] * i2_scale > 0 ? 2 : 0;  // +1 -> 2, -1 -> 0
    }

    memset(dst, 0, n_total * sizeof(uint8_t) / 4);

    uint8_t* i2_weight = (uint8_t*)dst;
    for (int64_t i = 0; i < n_total / QK_I2_S; i++) {
        for (int j = 0; j < QK_I2_S; j++) {
            int group_idx = j / 32;
            int group_pos = j % 32;
            uint8_t temp = (q8[i * QK_I2_S + j] << (6 - 2 * group_idx));
            i2_weight[i * 32 + group_pos] |= temp;
        }
    }

    float* scale_ptr = (float*)((char*)i2_weight + n_total / 4);
    scale_ptr[0] = (float)i2_scale;

    free(q8);

    return nrow * row_size / 4 + 32;

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
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const uint8x16_t mask = vdupq_n_u8(3);

    // 处理多列，nrc表示要处理的列数
    for (int row = 0; row < nrc; row++) {
        int32x4_t accu = vdupq_n_s32(0);

        // 计算当前行的x指针偏移
        const uint8_t * x_row = x + row * bx / 4;

        for (int i=0; i < group32_num; i++) {

#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accu32 = vdupq_n_s16(0);
#endif
            for (int j=0; j < 32; j++) {
                uint8x16_t xq8_3 = vld1q_u8(x_row + i * 32 * 16 + j * 16);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                const int8x16_t yq8_0 = vld1q_s8(y + i * 32 * 64 + j * 64 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + i * 32 * 64 + j * 64 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + i * 32 * 64 + j * 64 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + i * 32 * 64 + j * 64 + 48);

#if defined(__ARM_FEATURE_DOTPROD)
                accu = vdotq_s32(accu, q8_0, yq8_0);
                accu = vdotq_s32(accu, q8_1, yq8_1);
                accu = vdotq_s32(accu, q8_2, yq8_2);
                accu = vdotq_s32(accu, q8_3, yq8_3);
#else
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accu32 = vmlal_s8(accu32, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accu32 = vmlal_s8(accu32, vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accu32)));
            accu = vaddq_s32(accu, vmovl_high_s16(accu32));
#endif
        }

        for (int i = 0; i < groupla_num; i++){
#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accula = vdupq_n_s16(0);
#endif
            for (int j = 0; j < la_num; j++) {
                uint8x16_t xq8_3 = vld1q_u8(x_row + group32_num * 32 * 16 + j * 16);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 48);

#if defined(__ARM_FEATURE_DOTPROD)
                accu = vdotq_s32(accu, q8_0, yq8_0);
                accu = vdotq_s32(accu, q8_1, yq8_1);
                accu = vdotq_s32(accu, q8_2, yq8_2);
                accu = vdotq_s32(accu, q8_3, yq8_3);
#else
                accula = vmlal_s8(accula, vget_low_s8(q8_0), vget_low_s8(yq8_0));
                accula = vmlal_s8(accula, vget_high_s8(q8_0), vget_high_s8(yq8_0));
                accula = vmlal_s8(accula, vget_low_s8(q8_1), vget_low_s8(yq8_1));
                accula = vmlal_s8(accula, vget_high_s8(q8_1), vget_high_s8(yq8_1));
                accula = vmlal_s8(accula, vget_low_s8(q8_2), vget_low_s8(yq8_2));
                accula = vmlal_s8(accula, vget_high_s8(q8_2), vget_high_s8(yq8_2));
                accula = vmlal_s8(accula, vget_low_s8(q8_3), vget_low_s8(yq8_3));
                accula = vmlal_s8(accula, vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
            }
#if defined(__ARM_FEATURE_DOTPROD)

#else
            accu = vaddq_s32(accu, vmovl_s16(vget_low_s16(accula)));
            accu = vaddq_s32(accu, vmovl_high_s16(accula));
#endif
        }
        int sumi = vaddlvq_s32(accu);
        s[row] = (float)sumi;
    }

#elif defined(__VSX__) || defined(__ALTIVEC__) || defined(BITNET_G5_ALTIVEC)
    // PowerPC optimized - 1x1 kernel (POWER8 VSX + G5 AltiVec)
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;
    const int nb = n / QK_I2_S;

    for (int row = 0; row < nrc; row++) {
        vector signed int accu = vec_splat_s32(0);
        const uint8_t * x_row = x + row * bx / 4;

        for (int block = 0; block < nb; block++) {
            const uint8_t * px = x_row + block * 32;
            const int8_t  * py = y + block * 128;

            // Prefetch next weight block
            if (block + 1 < nb) I2S_DCBT_RESIDENT(px + 32);
            // Prefetch for activations
            if (block + 1 < nb) I2S_DCBT(py + 128);

            // Process 32 bytes of weights in 2 x 16-byte halves
            accu = i2s_ppc_half(px, 0,  py, 0,  accu);
            accu = i2s_ppc_half(px, 16, py, 16, accu);
        }

        s[row] = (float)hsum_i32_4_ppc(accu);
    }

#else
    // Scalar fallback (PowerPC / generic)
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;

    const int nb = n / QK_I2_S;

    for (int row = 0; row < nrc; row++) {
        int32_t sumi = 0;
        const uint8_t * x_row = x + row * bx / 4;

        for (int block = 0; block < nb; block++) {
            const uint8_t * px = x_row + block * 32;   // 128 elements / 4 per byte = 32 bytes
            const int8_t  * py = y + block * 128;       // 128 int8 activations

            for (int pos = 0; pos < 32; pos++) {
                uint8_t packed = px[pos];
                uint8_t w0 = (packed >> 6) & 0x03;  // group 0
                uint8_t w1 = (packed >> 4) & 0x03;  // group 1
                uint8_t w2 = (packed >> 2) & 0x03;  // group 2
                uint8_t w3 = (packed >> 0) & 0x03;  // group 3

                sumi += (int32_t)w0 * (int32_t)py[pos];
                sumi += (int32_t)w1 * (int32_t)py[32 + pos];
                sumi += (int32_t)w2 * (int32_t)py[64 + pos];
                sumi += (int32_t)w3 * (int32_t)py[96 + pos];
            }
        }

        s[row] = (float)sumi;
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


#elif defined(__VSX__) || defined(__ALTIVEC__) || defined(BITNET_G5_ALTIVEC)
    // PowerPC optimized - 1x4_32W kernel (POWER8 VSX + G5 AltiVec)
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;
    const int nb = n / QK_I2_S;

    for (int row = 0; row < nrc; row += 4) {
        vector signed int accu0 = vec_splat_s32(0);
        vector signed int accu1 = vec_splat_s32(0);
        vector signed int accu2 = vec_splat_s32(0);
        vector signed int accu3 = vec_splat_s32(0);
        const uint8_t * x_base = x + row * bx / 4;

        for (int block = 0; block < nb; block++) {
            for (int sub = 0; sub < 4; sub++) {
                const uint8_t * px = x_base + (block * 4 + sub) * 32;
                const int8_t  * py = y + block * 128 + sub * 32;

                // Prefetch next weight block
                I2S_DCBT_RESIDENT(px + 32);

                // Process 32 bytes in 2 halves of 16
                for (int half = 0; half < 2; half++) {
                    vector unsigned char packed = I2S_VEC_LD_UC(half * 16, px);
                    vector unsigned char w0 = vec_and(vec_sr(packed, ppc_shift6), ppc_mask03);
                    vector unsigned char w1 = vec_and(vec_sr(packed, ppc_shift4), ppc_mask03);
                    vector unsigned char w2 = vec_and(vec_sr(packed, ppc_shift2), ppc_mask03);
                    vector unsigned char w3 = vec_and(packed, ppc_mask03);

                    vector signed char yv = I2S_VEC_LD_SC(half * 16, py);
                    accu0 = vec_msum(yv, w0, accu0);
                    accu1 = vec_msum(yv, w1, accu1);
                    accu2 = vec_msum(yv, w2, accu2);
                    accu3 = vec_msum(yv, w3, accu3);
                }
            }
        }

        s[row + 0] = (float)hsum_i32_4_ppc(accu0);
        s[row + 1] = (float)hsum_i32_4_ppc(accu1);
        s[row + 2] = (float)hsum_i32_4_ppc(accu2);
        s[row + 3] = (float)hsum_i32_4_ppc(accu3);
    }

#else
    // Scalar fallback (PowerPC / generic)
    // Processes 4 rows at a time with 32-wide interleaved x layout
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;

    const int nb = n / QK_I2_S;

    for (int row = 0; row < nrc; row += 4) {
        int32_t sumi[4] = {0, 0, 0, 0};
        const uint8_t * x_base = x + row * bx / 4;

        for (int block = 0; block < nb; block++) {
            // In 32W layout, x has 4 sub-blocks of 32 bytes per block
            // y is accessed linearly, 32 bytes at a time
            for (int sub = 0; sub < 4; sub++) {
                const uint8_t * px = x_base + (block * 4 + sub) * 32;
                const int8_t  * py = y + block * 128 + sub * 32;

                for (int pos = 0; pos < 32; pos++) {
                    uint8_t packed = px[pos];
                    int8_t yval = py[pos];

                    sumi[0] += (int32_t)((packed >> 6) & 0x03) * (int32_t)yval;
                    sumi[1] += (int32_t)((packed >> 4) & 0x03) * (int32_t)yval;
                    sumi[2] += (int32_t)((packed >> 2) & 0x03) * (int32_t)yval;
                    sumi[3] += (int32_t)((packed >> 0) & 0x03) * (int32_t)yval;
                }
            }
        }

        for (int rb = 0; rb < 4; rb++) {
            s[row + rb] = (float)sumi[rb];
        }
    }

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
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    
    const uint8x16_t mask = vdupq_n_u8(3);

    // 处理多行，nrc表示要处理的行数
    for (int row = 0; row < nrc; row += PARALLEL_SIZE) {

        int32x4_t accu[PARALLEL_SIZE];
        const uint8_t * x_row[PARALLEL_SIZE];
        
        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            accu[rb] = vdupq_n_s32(0);
            x_row[rb] = x + (row + rb) * bx / 4;
        }

        for (int i = 0; i < group32_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accu32[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu32[rb] = vdupq_n_s16(0);
            }
#endif
            const uint8_t * px[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + i * 32 * 16;
            }

            for (int j = 0; j < 32; j++) {
                // 加载 y 数据（对所有行共享）
                const int8x16_t yq8_0 = vld1q_s8(y + i * 32 * 64 + j * 64 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + i * 32 * 64 + j * 64 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + i * 32 * 64 + j * 64 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + i * 32 * 64 + j * 64 + 48);

                // 处理每一行
                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    uint8x16_t xq8_3 = vld1q_u8(px[rb] + 0);
                    uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                    uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                    uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                    int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));

#if defined(__ARM_FEATURE_DOTPROD)
                    accu[rb] = vdotq_s32(accu[rb], q8_0, yq8_0);
                    accu[rb] = vdotq_s32(accu[rb], q8_1, yq8_1);
                    accu[rb] = vdotq_s32(accu[rb], q8_2, yq8_2);
                    accu[rb] = vdotq_s32(accu[rb], q8_3, yq8_3);
#else
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_3), vget_high_s8(yq8_3));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accu32[rb] = vmlal_s8(accu32[rb], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    
#endif
                    px[rb] += 16;
                }
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accu32[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accu32[rb]));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accula[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accula[rb] = vdupq_n_s16(0);
            }
#endif
            const uint8_t * px[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                px[rb] = x_row[rb] + group32_num * 32 * 16;
            }

            for (int j = 0; j < la_num; j++) {
                // 加载 y 数据（对所有行共享）
                const int8x16_t yq8_0 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 0);
                const int8x16_t yq8_1 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 16);
                const int8x16_t yq8_2 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 32);
                const int8x16_t yq8_3 = vld1q_s8(y + group32_num * 32 * 64 + j * 64 + 48);

                // 处理每一行
                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    uint8x16_t xq8_3 = vld1q_u8(px[rb] + 0);
                    uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                    uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                    uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                    int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));
                    int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                    int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                    int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                    
#if defined(__ARM_FEATURE_DOTPROD)
                    accu[rb] = vdotq_s32(accu[rb], q8_0, yq8_0);
                    accu[rb] = vdotq_s32(accu[rb], q8_1, yq8_1);
                    accu[rb] = vdotq_s32(accu[rb], q8_2, yq8_2);
                    accu[rb] = vdotq_s32(accu[rb], q8_3, yq8_3);
#else
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_3), vget_high_s8(yq8_3));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accula[rb] = vmlal_s8(accula[rb], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accula[rb] = vmlal_s8(accula[rb], vget_high_s8(q8_0), vget_high_s8(yq8_0));

#endif
                    px[rb] += 16;
                }
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                accu[rb] = vaddq_s32(accu[rb], vmovl_s16(vget_low_s16(accula[rb])));
                accu[rb] = vaddq_s32(accu[rb], vmovl_high_s16(accula[rb]));
            }
#endif
        }

        // 合并结果并写回
        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            int sumi = vaddlvq_s32(accu[rb]);
            s[row + rb] = (float)sumi;
        }
    }

#elif defined(__VSX__) || defined(__ALTIVEC__) || defined(BITNET_G5_ALTIVEC)
    // PowerPC optimized - 1xN kernel (POWER8 VSX + G5 AltiVec)
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;
    const int nb = n / QK_I2_S;

    for (int row = 0; row < nrc; row += PARALLEL_SIZE) {
        vector signed int accu[PARALLEL_SIZE];
        const uint8_t * x_rows[PARALLEL_SIZE];

        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            accu[rb] = vec_splat_s32(0);
            x_rows[rb] = x + (row + rb) * bx / 4;
        }

        for (int block = 0; block < nb; block++) {
            const int8_t * py = y + block * 128;

            // Prefetch for activations
            if (block + 1 < nb) I2S_DCBT(py + 128);

            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                const uint8_t * px = x_rows[rb] + block * 32;

                // Prefetch next weight block
                if (block + 1 < nb) I2S_DCBT_RESIDENT(px + 32);

                // 2 halves of 16 bytes each
                accu[rb] = i2s_ppc_half(px, 0,  py, 0,  accu[rb]);
                accu[rb] = i2s_ppc_half(px, 16, py, 16, accu[rb]);
            }
        }

        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            s[row + rb] = (float)hsum_i32_4_ppc(accu[rb]);
        }
    }

#else
    // Scalar fallback (PowerPC / generic)
    // Processes PARALLEL_SIZE rows at a time
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;

    const int nb = n / QK_I2_S;

    for (int row = 0; row < nrc; row += PARALLEL_SIZE) {
        int32_t sumi[PARALLEL_SIZE];
        const uint8_t * x_rows[PARALLEL_SIZE];

        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            sumi[rb] = 0;
            x_rows[rb] = x + (row + rb) * bx / 4;
        }

        for (int block = 0; block < nb; block++) {
            const int8_t * py = y + block * 128;

            for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                const uint8_t * px = x_rows[rb] + block * 32;
                for (int pos = 0; pos < 32; pos++) {
                    uint8_t packed = px[pos];
                    uint8_t w0 = (packed >> 6) & 0x03;
                    uint8_t w1 = (packed >> 4) & 0x03;
                    uint8_t w2 = (packed >> 2) & 0x03;
                    uint8_t w3 = (packed >> 0) & 0x03;

                    sumi[rb] += (int32_t)w0 * (int32_t)py[pos];
                    sumi[rb] += (int32_t)w1 * (int32_t)py[32 + pos];
                    sumi[rb] += (int32_t)w2 * (int32_t)py[64 + pos];
                    sumi[rb] += (int32_t)w3 * (int32_t)py[96 + pos];
                }
            }
        }

        for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
            s[row + rb] = (float)sumi[rb];
        }
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

        int8_t * y_col = y + col * by;
        
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
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;

    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;

    const uint8x16_t mask = vdupq_n_u8(3);

    for (int col = 0; col < nrc; col += PARALLEL_SIZE) {
        int32x4_t accu[PARALLEL_SIZE];

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            accu[iy] = vdupq_n_s32(0);
        }

        const int8_t * y_col = y + col * by;
        
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x + i * 512;     // i * 32 * 16
            const int8_t  *py = y_col + i * 2048; // i * 32 * 64

#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accu32[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu32[iy] = vdupq_n_s16(0);
            }
#endif
            for (int j = 0; j < 32; j++) {
                // 加载并解包 x 数据（对所有列共享）
                uint8x16_t xq8_3 = vld1q_u8(px + 0);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                // 处理每一列
                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    const int8x16_t yq8_0 = vld1q_s8(py + 0 * 16 + iy * by);
                    const int8x16_t yq8_1 = vld1q_s8(py + 1 * 16 + iy * by);
                    const int8x16_t yq8_2 = vld1q_s8(py + 2 * 16 + iy * by);
                    const int8x16_t yq8_3 = vld1q_s8(py + 3 * 16 + iy * by);

#if defined(__ARM_FEATURE_DOTPROD)
                    accu[iy] = vdotq_s32(accu[iy], q8_0, yq8_0);
                    accu[iy] = vdotq_s32(accu[iy], q8_1, yq8_1);
                    accu[iy] = vdotq_s32(accu[iy], q8_2, yq8_2);
                    accu[iy] = vdotq_s32(accu[iy], q8_3, yq8_3);
#else
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accu32[iy] = vmlal_s8(accu32[iy], vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
                }

                px += 16;
                py += 64;
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = vaddq_s32(accu[iy], vaddq_s32(vmovl_high_s16(accu32[iy]), vmovl_s16(vget_low_s16(accu32[iy]))));
            }
#endif
        }

        for (int i = 0; i < groupla_num; i++) {
            const uint8_t *px = x + group32_num * 512;
            const int8_t  *py = y_col + group32_num * 2048;

#if defined(__ARM_FEATURE_DOTPROD)

#else
            int16x8_t accula[PARALLEL_SIZE];

            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accula[iy] = vdupq_n_s16(0);
            }
#endif
            
            for (int j = 0; j < la_num; j++) {
                // 加载并解包 x 数据（对所有列共享）
                uint8x16_t xq8_3 = vld1q_u8(px + 0);
                uint8x16_t xq8_2 = vshrq_n_u8(xq8_3, 2);
                uint8x16_t xq8_1 = vshrq_n_u8(xq8_3, 4);
                uint8x16_t xq8_0 = vshrq_n_u8(xq8_3, 6);

                int8x16_t q8_0 = vreinterpretq_s8_u8(vandq_u8(xq8_0, mask));
                int8x16_t q8_1 = vreinterpretq_s8_u8(vandq_u8(xq8_1, mask));
                int8x16_t q8_2 = vreinterpretq_s8_u8(vandq_u8(xq8_2, mask));
                int8x16_t q8_3 = vreinterpretq_s8_u8(vandq_u8(xq8_3, mask));

                // 处理每一列
                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    const int8x16_t yq8_0 = vld1q_s8(py + 0 * 16 + iy * by);
                    const int8x16_t yq8_1 = vld1q_s8(py + 1 * 16 + iy * by);
                    const int8x16_t yq8_2 = vld1q_s8(py + 2 * 16 + iy * by);
                    const int8x16_t yq8_3 = vld1q_s8(py + 3 * 16 + iy * by);

#if defined(__ARM_FEATURE_DOTPROD)
                    accu[iy] = vdotq_s32(accu[iy], q8_0, yq8_0);
                    accu[iy] = vdotq_s32(accu[iy], q8_1, yq8_1);
                    accu[iy] = vdotq_s32(accu[iy], q8_2, yq8_2);
                    accu[iy] = vdotq_s32(accu[iy], q8_3, yq8_3);
#else
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_0), vget_low_s8(yq8_0));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_0), vget_high_s8(yq8_0));
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_1), vget_low_s8(yq8_1));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_1), vget_high_s8(yq8_1));
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_2), vget_low_s8(yq8_2));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_2), vget_high_s8(yq8_2));
                    accula[iy] = vmlal_s8(accula[iy], vget_low_s8(q8_3), vget_low_s8(yq8_3));
                    accula[iy] = vmlal_s8(accula[iy], vget_high_s8(q8_3), vget_high_s8(yq8_3));
#endif
                }

                px += 16;
                py += 64;
            }

#if defined(__ARM_FEATURE_DOTPROD)

#else
            for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                accu[iy] = vaddq_s32(accu[iy], vaddq_s32(vmovl_high_s16(accula[iy]), vmovl_s16(vget_low_s16(accula[iy]))));
            }
#endif
        }

        // 合并结果并写回
        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            int sumi = vaddlvq_s32(accu[iy]);
            s[(col + iy) * bs] = (float)sumi;
        }
    }

#elif defined(__VSX__) || defined(__ALTIVEC__) || defined(BITNET_G5_ALTIVEC)
    // PowerPC optimized - Nx1 kernel (POWER8 VSX + G5 AltiVec)
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;
    const int nb = n / QK_I2_S;

    for (int col = 0; col < nrc; col += PARALLEL_SIZE) {
        vector signed int accu[PARALLEL_SIZE];
        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            accu[iy] = vec_splat_s32(0);
        }

        for (int block = 0; block < nb; block++) {
            const uint8_t * px = x + block * 32;

            // Prefetch next weight block
            if (block + 1 < nb) I2S_DCBT_RESIDENT(px + 32);

            // Process 2 halves of 16 bytes
            for (int half = 0; half < 2; half++) {
                vector unsigned char packed = I2S_VEC_LD_UC(half * 16, px);
                vector unsigned char w0 = vec_and(vec_sr(packed, ppc_shift6), ppc_mask03);
                vector unsigned char w1 = vec_and(vec_sr(packed, ppc_shift4), ppc_mask03);
                vector unsigned char w2 = vec_and(vec_sr(packed, ppc_shift2), ppc_mask03);
                vector unsigned char w3 = vec_and(packed, ppc_mask03);

                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    const int8_t * py = (const int8_t *)vy + (col + iy) * by + block * 128;

                    vector signed char y0 = I2S_VEC_LD_SC(half * 16,      py);
                    vector signed char y1 = I2S_VEC_LD_SC(half * 16 + 32, py);
                    vector signed char y2 = I2S_VEC_LD_SC(half * 16 + 64, py);
                    vector signed char y3 = I2S_VEC_LD_SC(half * 16 + 96, py);

                    accu[iy] = vec_msum(y0, w0, accu[iy]);
                    accu[iy] = vec_msum(y1, w1, accu[iy]);
                    accu[iy] = vec_msum(y2, w2, accu[iy]);
                    accu[iy] = vec_msum(y3, w3, accu[iy]);
                }
            }
        }

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            s[(col + iy) * bs] = (float)hsum_i32_4_ppc(accu[iy]);
        }
    }

#else
    // Scalar fallback (PowerPC / generic)
    // Single x row, PARALLEL_SIZE y columns
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;

    const int nb = n / QK_I2_S;

    for (int col = 0; col < nrc; col += PARALLEL_SIZE) {
        int32_t sumi[PARALLEL_SIZE];
        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            sumi[iy] = 0;
        }

        for (int block = 0; block < nb; block++) {
            const uint8_t * px = x + block * 32;

            for (int pos = 0; pos < 32; pos++) {
                uint8_t packed = px[pos];
                uint8_t w0 = (packed >> 6) & 0x03;
                uint8_t w1 = (packed >> 4) & 0x03;
                uint8_t w2 = (packed >> 2) & 0x03;
                uint8_t w3 = (packed >> 0) & 0x03;

                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    const int8_t * py = (const int8_t *)vy + (col + iy) * by + block * 128;

                    sumi[iy] += (int32_t)w0 * (int32_t)py[pos];
                    sumi[iy] += (int32_t)w1 * (int32_t)py[32 + pos];
                    sumi[iy] += (int32_t)w2 * (int32_t)py[64 + pos];
                    sumi[iy] += (int32_t)w3 * (int32_t)py[96 + pos];
                }
            }
        }

        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            s[(col + iy) * bs] = (float)sumi[iy];
        }
    }

#endif
}


void ggml_vec_dot_i2_i8_s(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
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
}