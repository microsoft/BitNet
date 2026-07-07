// Regression test for issue #547: i2_s produces garbage on x86 CPUs without AVX2.
//
// The ggml_vec_dot_i2_i8_s_* kernels in src/ggml-bitnet-mad.cpp were gated on
// `#if defined(__AVX2__) ... #elif defined(__ARM_NEON)`, leaving an EMPTY body on
// x86 targets without AVX2 (e.g. built with -DGGML_AVX2=OFF). The output `s` was
// then never written, so inference degenerated into a garbage token loop.
//
// This pins the portable scalar fallback: it re-implements the exact AVX2 kernels
// as ground truth and asserts the scalar fallback (identical to the `#else` branch
// shipped in ggml-bitnet-mad.cpp) is BIT-EXACT for all four kernels.
//
// Build & run on any AVX2 host (AVX2 path is the reference):
//     g++ -O2 -mavx2 tests/test-i2s-scalar-fallback.cpp -o /tmp/t && /tmp/t
// Exit code 0 == all bit-exact.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <immintrin.h>

#define QK_I2_S 128
#define PARALLEL_SIZE 4

static inline int hsum_i32_8(const __m256i a) {
    const __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(a), _mm256_extractf128_si256(a, 1));
    const __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
    const __m128i sum64 = _mm_add_epi32(hi64, sum128);
    const __m128i hi32  = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
    return _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));
}

// ---- portable scalar helper (identical to ggml_i2_s_block_dot in the fix) ----
static inline int ggml_i2_s_block_dot(const uint8_t * px, const int8_t * py, int nb) {
    int acc = 0;
    for (int b = 0; b < nb; b++) {
        const uint8_t * xb = px + (size_t)b * 32;
        const int8_t  * yb = py + (size_t)b * 128;
        for (int k = 0; k < 32; k++) {
            const uint8_t w = xb[k];
            acc += (int)((w >> 6) & 3) * (int)yb[0 * 32 + k];
            acc += (int)((w >> 4) & 3) * (int)yb[1 * 32 + k];
            acc += (int)((w >> 2) & 3) * (int)yb[2 * 32 + k];
            acc += (int)((w >> 0) & 3) * (int)yb[3 * 32 + k];
        }
    }
    return acc;
}

void dot_1x1_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t *    x = (uint8_t *)vx;
    const int8_t  *    y = (int8_t *)vy;
    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    __m256i mask = _mm256_set1_epi8(0x03);
    __m256i one16 = _mm256_set1_epi16(1);
    for (int row = 0; row < nrc; row++) {
        __m256i accu = _mm256_setzero_si256();
        const uint8_t * x_row = x + row * bx / 4;
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x_row + i * 1024;
            const int8_t  *py = y + i * 4096;
            __m256i accu32 = _mm256_setzero_si256();
            for (int j = 0; j < 32; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);
                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);
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
            const uint8_t *px = x_row + group32_num * 1024;
            const int8_t  *py = y + group32_num * 4096;
            for (int j = 0; j < la_num; j++) {
                __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);
                xq8_3 = _mm256_and_si256(xq8_3, mask);
                xq8_2 = _mm256_and_si256(xq8_2, mask);
                xq8_1 = _mm256_and_si256(xq8_1, mask);
                xq8_0 = _mm256_and_si256(xq8_0, mask);
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
}

// GROUND TRUTH: exact copy of the AVX2 body of ggml_vec_dot_i2_i8_s_Nx1

void dot_Nx1_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
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
        for (int iy = 0; iy < PARALLEL_SIZE; iy++) accu[iy] = _mm256_setzero_si256();
        int8_t * y_col = (int8_t *)y + col * by;
        for (int i = 0; i < group32_num; i++) {
            const uint8_t *px = x + i * 1024;
            const int8_t  *py = y_col + i * 4096;
            __m256i accu32[PARALLEL_SIZE];
            for (int iy = 0; iy < PARALLEL_SIZE; iy++) accu32[iy] = _mm256_setzero_si256();
            for (int j = 0; j < 32; j++) {
                __m256i xq8   = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_3 = _mm256_and_si256(xq8, mask);
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8, 6), mask);
                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    accu32[iy] = _mm256_add_epi16(accu32[iy], _mm256_add_epi16(
                        _mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, _mm256_loadu_si256((const __m256i*)(py + 0 * 32 + iy * by))),
                                         _mm256_maddubs_epi16(xq8_1, _mm256_loadu_si256((const __m256i*)(py + 1 * 32 + iy * by)))),
                        _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, _mm256_loadu_si256((const __m256i*)(py + 2 * 32 + iy * by))),
                                         _mm256_maddubs_epi16(xq8_3, _mm256_loadu_si256((const __m256i*)(py + 3 * 32 + iy * by))))));
                }
                px += 32;
                py += 128;
            }
            for (int iy = 0; iy < PARALLEL_SIZE; iy++)
                accu[iy] = _mm256_add_epi32(_mm256_madd_epi16(accu32[iy], one16), accu[iy]);
        }
        for (int i = 0; i < groupla_num; i++) {
            const uint8_t *px = x + group32_num * 1024;
            const int8_t  *py = y_col + group32_num * 4096;
            __m256i accula[PARALLEL_SIZE];
            for (int iy = 0; iy < PARALLEL_SIZE; iy++) accula[iy] = _mm256_setzero_si256();
            for (int j = 0; j < la_num; j++) {
                __m256i xq8   = _mm256_loadu_si256((const __m256i*)(px));
                __m256i xq8_3 = _mm256_and_si256(xq8, mask);
                __m256i xq8_2 = _mm256_and_si256(_mm256_srli_epi16(xq8, 2), mask);
                __m256i xq8_1 = _mm256_and_si256(_mm256_srli_epi16(xq8, 4), mask);
                __m256i xq8_0 = _mm256_and_si256(_mm256_srli_epi16(xq8, 6), mask);
                for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
                    accula[iy] = _mm256_add_epi16(accula[iy], _mm256_add_epi16(
                        _mm256_add_epi16(_mm256_maddubs_epi16(xq8_0, _mm256_loadu_si256((const __m256i*)(py + 0 * 32 + iy * by))),
                                         _mm256_maddubs_epi16(xq8_1, _mm256_loadu_si256((const __m256i*)(py + 1 * 32 + iy * by)))),
                        _mm256_add_epi16(_mm256_maddubs_epi16(xq8_2, _mm256_loadu_si256((const __m256i*)(py + 2 * 32 + iy * by))),
                                         _mm256_maddubs_epi16(xq8_3, _mm256_loadu_si256((const __m256i*)(py + 3 * 32 + iy * by))))));
                }
                px += 32;
                py += 128;
            }
            for (int iy = 0; iy < PARALLEL_SIZE; iy++)
                accu[iy] = _mm256_add_epi32(_mm256_madd_epi16(accula[iy], one16), accu[iy]);
        }
        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            int sumi = hsum_i32_8(accu[iy]);
            s[(col + iy) * bs] = (float)sumi;
        }
    }
}

void dot_1xN_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;
    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (int row = 0; row < nrc; row += PARALLEL_SIZE) {
        __m256i accu[PARALLEL_SIZE];
        const uint8_t * x_row[PARALLEL_SIZE];
        for (int rb = 0; rb < PARALLEL_SIZE; rb++) { accu[rb] = _mm256_setzero_si256(); x_row[rb] = x + (row + rb) * bx / 4; }
        for (int i = 0; i < group32_num; i++) {
            const uint8_t * px[PARALLEL_SIZE];
            __m256i accu32[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) { px[rb] = x_row[rb] + i * 1024; accu32[rb] = _mm256_setzero_si256(); }
            const int8_t * py = y + i * 4096;
            for (int j = 0; j < 32; j++) {
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));
                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px[rb]));
                    __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                    __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                    __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);
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
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) accu[rb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[rb], one16), accu[rb]);
        }
        for (int i = 0; i < groupla_num; i++) {
            const int8_t * py = y + group32_num * 4096;
            const uint8_t * px[PARALLEL_SIZE];
            __m256i accula[PARALLEL_SIZE];
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) { px[rb] = x_row[rb] + group32_num * 1024; accula[rb] = _mm256_setzero_si256(); }
            for (int j = 0; j < la_num; j++) {
                __m256i yq8_0 = _mm256_loadu_si256((const __m256i*)(py));
                __m256i yq8_1 = _mm256_loadu_si256((const __m256i*)(py + 32));
                __m256i yq8_2 = _mm256_loadu_si256((const __m256i*)(py + 64));
                __m256i yq8_3 = _mm256_loadu_si256((const __m256i*)(py + 96));
                for (int rb = 0; rb < PARALLEL_SIZE; rb++) {
                    __m256i xq8_3 = _mm256_loadu_si256((const __m256i*)(px[rb]));
                    __m256i xq8_2 = _mm256_srli_epi16(xq8_3, 2);
                    __m256i xq8_1 = _mm256_srli_epi16(xq8_3, 4);
                    __m256i xq8_0 = _mm256_srli_epi16(xq8_3, 6);
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
            for (int rb = 0; rb < PARALLEL_SIZE; rb++) accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(accula[rb], one16));
        }
        for (int rb = 0; rb < PARALLEL_SIZE; rb++) { int sumi = hsum_i32_8(accu[rb]); s[row + rb] = (float)sumi; }
    }
}

// ---- AVX2 ref: ggml_vec_dot_i2_i8_s_1x4_32W (exact copy) ----

void dot_1x4_avx2(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t * x = (uint8_t *)vx;
    const int8_t  * y = (int8_t *)vy;
    const int nb = n / QK_I2_S;
    const int group32_num = nb / 32;
    const int la_num = nb % 32;
    const int groupla_num = nb % 32 != 0 ? 1 : 0;
    const __m256i mask = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);
    for (int row = 0; row < nrc; row += 4) {
        __m256i accu[4];
        for (int rb = 0; rb < 4; rb++) accu[rb] = _mm256_setzero_si256();
        const uint8_t * x_row = x + (row) * bx / 4;
        for (int i = 0; i < group32_num; i++) {
            const uint8_t * px = x_row + i * 1024 * 4;
            __m256i accu32[4];
            for (int rb = 0; rb < 4; rb++) accu32[rb] = _mm256_setzero_si256();
            const int8_t * py = y + i * 4096;
            for (int j = 0; j < 32 * 4; j++) {
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
                for (int rb = 0; rb < 4; rb++) { xq8[rb] = _mm256_maddubs_epi16(xq8[rb], yq8_0); accu32[rb] = _mm256_add_epi16(accu32[rb], xq8[rb]); }
                px += 32; py += 32;
            }
            for (int rb = 0; rb < 4; rb++) accu[rb] = _mm256_add_epi32(_mm256_madd_epi16(accu32[rb], one16), accu[rb]);
        }
        for (int i = 0; i < groupla_num; i++) {
            const int8_t * py = y + group32_num * 4096;
            __m256i accula[4];
            for (int rb = 0; rb < 4; rb++) accula[rb] = _mm256_setzero_si256();
            const uint8_t * px = x_row + group32_num * 1024 * 4;
            for (int j = 0; j < la_num * 4; j++) {
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
                for (int rb = 0; rb < 4; rb++) { xq8[rb] = _mm256_maddubs_epi16(xq8[rb], yq8_0); accula[rb] = _mm256_add_epi16(accula[rb], xq8[rb]); }
                px += 32; py += 32;
            }
            for (int rb = 0; rb < 4; rb++) accu[rb] = _mm256_add_epi32(accu[rb], _mm256_madd_epi16(accula[rb], one16));
        }
        for (int rb = 0; rb < 4; rb++) { int sumi = hsum_i32_8(accu[rb]); s[row + rb] = (float)sumi; }
    }
}

// ---- scalar candidates (identical to what goes in the #else branch) ----
void dot_1x1_scalar(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t  *)vy;
    const int nb = n / QK_I2_S;
    for (int row = 0; row < nrc; row++) {
        const uint8_t * x_row = x + row * bx / 4;
        s[row] = (float)ggml_i2_s_block_dot(x_row, y, nb);
    }
}

void dot_Nx1_scalar(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t  *)vy;
    const int nb = n / QK_I2_S;
    for (int col = 0; col < nrc; col += PARALLEL_SIZE) {
        for (int iy = 0; iy < PARALLEL_SIZE; iy++) {
            const int8_t * py = y + (size_t)(col + iy) * by;
            s[(col + iy) * bs] = (float)ggml_i2_s_block_dot(x, py, nb);
        }
    }
}

void dot_1xN_scalar(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t  *)vy;
    const int nb = n / QK_I2_S;
    for (int row = 0; row < nrc; row += PARALLEL_SIZE)
        for (int rb = 0; rb < PARALLEL_SIZE; rb++)
            s[row + rb] = (float)ggml_i2_s_block_dot(x + (row + rb) * bx / 4, y, nb);
}

// interleaved 4-row bit-plane packing: 4 outputs share one activation stream

void dot_1x4_scalar(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc) {
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t  *)vy;
    const int nb = n / QK_I2_S;
    for (int row = 0; row < nrc; row += 4) {
        const uint8_t * x_row = x + (row) * bx / 4;
        long acc[4] = {0, 0, 0, 0};
        for (int t = 0; t < nb * 4; t++) {
            const uint8_t * pxj = x_row + (size_t)t * 32;
            const int8_t  * pyj = y     + (size_t)t * 32;
            for (int k = 0; k < 32; k++) {
                const uint8_t w = pxj[k];
                acc[0] += (long)((w >> 6) & 3) * (int)pyj[k];
                acc[1] += (long)((w >> 4) & 3) * (int)pyj[k];
                acc[2] += (long)((w >> 2) & 3) * (int)pyj[k];
                acc[3] += (long)((w >> 0) & 3) * (int)pyj[k];
            }
        }
        for (int rb = 0; rb < 4; rb++) s[row + rb] = (float)acc[rb];
    }
}

int main() {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> byte_d(0, 255), act_d(-128, 127);
    int fails = 0, checks = 0;

    for (int trial = 0; trial < 300; trial++) {            // _1x1: nrc rows x 1 activation
        int nb = 1 + (rng() % 40), n = nb * QK_I2_S, nrc = 1 + (rng() % 4); size_t bx = n;
        std::vector<uint8_t> xb((size_t)nrc*(n/4)); std::vector<int8_t> yb(n);
        for (auto &e : xb) e = (uint8_t)byte_d(rng); for (auto &e : yb) e = (int8_t)act_d(rng);
        std::vector<float> sa(nrc,-1e9f), sc(nrc,-2e9f);
        dot_1x1_avx2(n,sa.data(),1,xb.data(),bx,yb.data(),0,nrc);
        dot_1x1_scalar(n,sc.data(),1,xb.data(),bx,yb.data(),0,nrc);
        for(int r=0;r<nrc;r++){checks++; if(sa[r]!=sc[r]){fails++; if(fails<=5)printf("1x1 mismatch t%d r%d %.1f/%.1f\n",trial,r,sa[r],sc[r]);}}
    }
    for (int trial = 0; trial < 300; trial++) {            // _Nx1: 1 row x ncol activations
        int nb = 1 + (rng() % 40), n = nb * QK_I2_S, ncol = PARALLEL_SIZE*(1+(rng()%3)); size_t by = n;
        std::vector<uint8_t> xb(n/4); std::vector<int8_t> yb((size_t)ncol*by);
        for (auto &e : xb) e = (uint8_t)byte_d(rng); for (auto &e : yb) e = (int8_t)act_d(rng);
        std::vector<float> sa(ncol,-1e9f), sc(ncol,-2e9f);
        dot_Nx1_avx2(n,sa.data(),1,xb.data(),0,yb.data(),by,ncol);
        dot_Nx1_scalar(n,sc.data(),1,xb.data(),0,yb.data(),by,ncol);
        for(int c=0;c<ncol;c++){checks++; if(sa[c]!=sc[c]){fails++; if(fails<=5)printf("Nx1 mismatch t%d c%d %.1f/%.1f\n",trial,c,sa[c],sc[c]);}}
    }
    for (int trial = 0; trial < 300; trial++) {            // _1xN: PARALLEL_SIZE rows x shared activation
        int nb = 1 + (rng() % 40), n = nb * QK_I2_S, nrc = PARALLEL_SIZE*(1+(rng()%3)); size_t bx = n;
        std::vector<uint8_t> xb((size_t)nrc*(n/4)); std::vector<int8_t> yb(n);
        for (auto &e : xb) e = (uint8_t)byte_d(rng); for (auto &e : yb) e = (int8_t)act_d(rng);
        std::vector<float> sa(nrc,-1e9f), sc(nrc,-2e9f);
        dot_1xN_avx2(n,sa.data(),1,xb.data(),bx,yb.data(),0,nrc);
        dot_1xN_scalar(n,sc.data(),1,xb.data(),bx,yb.data(),0,nrc);
        for(int r=0;r<nrc;r++){checks++; if(sa[r]!=sc[r]){fails++; if(fails<=5)printf("1xN mismatch t%d r%d %.1f/%.1f\n",trial,r,sa[r],sc[r]);}}
    }
    for (int trial = 0; trial < 300; trial++) {            // _1x4_32W: 4 interleaved rows
        int nb = 1 + (rng() % 40), n = nb * QK_I2_S, nrc = 4*(1+(rng()%2)); size_t bx = n;
        std::vector<uint8_t> xb((size_t)nrc*(n/4)); std::vector<int8_t> yb(n);
        for (auto &e : xb) e = (uint8_t)byte_d(rng); for (auto &e : yb) e = (int8_t)act_d(rng);
        std::vector<float> sa(nrc,-1e9f), sc(nrc,-2e9f);
        dot_1x4_avx2(n,sa.data(),1,xb.data(),bx,yb.data(),0,nrc);
        dot_1x4_scalar(n,sc.data(),1,xb.data(),bx,yb.data(),0,nrc);
        for(int r=0;r<nrc;r++){checks++; if(sa[r]!=sc[r]){fails++; if(fails<=5)printf("1x4 mismatch t%d r%d %.1f/%.1f\n",trial,r,sa[r],sc[r]);}}
    }
    printf("checks=%d fails=%d -> %s\n", checks, fails, fails==0?"ALL BIT-EXACT MATCH":"MISMATCH");
    return fails==0?0:1;
}
