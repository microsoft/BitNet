#include <arm_neon.h>

#define BM{{ pre }} {{ BM }}
#define BBK{{ pre }} {{ BK }}
inline void tbl_impl_{{ pre }}(int32_t* c, int8_t* lut, uint8_t* a) {
#ifdef __ARM_NEON
    const int KK = BBK{{ pre }} / 2;
    const uint8x16_t vec_mask = vdupq_n_u8(0x0f);
    const int8x16_t vec_zero = vdupq_n_s16(0x0000);
    int8x16_t vec_lut[2 * KK];
    int16x8_t vec_c[{{ bm // 8 }}];
    #pragma unroll
    for (int k = 0; k < 2 * KK; k++) {
        vec_lut[k] = vld1q_s8(lut + k * 16);
    }
