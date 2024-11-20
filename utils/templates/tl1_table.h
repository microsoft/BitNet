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

    #pragma unroll
    for (int i = 0; i < BM{{ pre }}; i += {{ bm }}) {
        #pragma unroll
        for (int i=0; i<{{ bm // 8 }}; i++) {
            vec_c[i] = vandq_s16(vec_c[i], vec_zero);
        }

        #pragma unroll
        for (int k = 0; k < KK / {{ 256 // bm // 2 }}; k++) {
{% for index in range(length) %}
            uint8x16_t vec_a_{{ index }} = vld1q_u8(a + i * KK / 2 + k * 32 * 2 + {{ index }} * 16);
            uint8x16_t vec_a{{ index }}_top = vshrq_n_u8(vec_a_{{ index }}, 4);
            uint8x16_t vec_a{{ index }}_bot = vandq_u8(vec_a_{{ index }}, vec_mask);
            int8x16_t  vec_v_{{ index }}_left_tmp0 = vqtbl1q_s8(vec_lut[{{ 2 * by // 2 }} * k + {{ (4 * index) % (2 * by // 2) }}], vec_a{{ index }}_top);
            int8x16_t  vec_v_{{ index }}_left_tmp1 = vqtbl1q_s8(vec_lut[{{ 2 * by // 2 }} * k + {{ (4 * index + 1) % (2 * by // 2) }}], vec_a{{ index }}_top);
            int8x16_t  vec_v_{{ index }}_right_tmp0 = vqtbl1q_s8(vec_lut[{{ 2 * by // 2 }} * k + {{ (4 * index + 2) % (2 * by // 2) }}], vec_a{{ index }}_bot);
            int8x16_t  vec_v_{{ index }}_right_tmp1 = vqtbl1q_s8(vec_lut[{{ 2 * by // 2 }} * k + {{ (4 * index + 3) % (2 * by // 2) }}], vec_a{{ index }}_bot);
            int8x16x2_t  vec_v_left_{{ index }} = vzipq_s8(vec_v_{{ index }}_left_tmp1, vec_v_{{ index }}_left_tmp0);
            int8x16x2_t  vec_v_right_{{ index }} = vzipq_s8(vec_v_{{ index }}_right_tmp1, vec_v_{{ index }}_right_tmp0);
            vec_c[{{ (index * 2) // (by // 2) * 2 + 0 }}] += vec_v_left_{{ index }}.val[0];
            vec_c[{{ (index * 2) // (by // 2) * 2 + 0 }}] += vec_v_right_{{ index }}.val[0];
            vec_c[{{ (index * 2) // (by // 2) * 2 + 1 }}] += vec_v_left_{{ index }}.val[1];
            vec_c[{{ (index * 2) // (by // 2) * 2 + 1 }}] += vec_v_right_{{ index }}.val[1];
{% endfor %}
        }
{% for index in range(bm // 8) %}
        int32x4_t vec_v_bot_low_low_{{ index }} = vmovl_s16(vget_low_s16(vec_c[{{ index }}]));
        int32x4_t vec_v_bot_low_high_{{ index }} = vmovl_high_s16(vec_c[{{ index }}]);
        vst1q_s32(c + i + {{ index * 8 }}, vld1q_s32(c + i + {{ index * 8 }}) + vec_v_bot_low_low_{{ index }});
        vst1q_s32(c + i + {{ index * 8 + 4 }}, vld1q_s32(c + i + {{ index * 8 + 4 }}) + vec_v_bot_low_high_{{ index }});
{% endfor %}
    }
#endif
}

int32_t qgemm_lut_{{ pre }}(void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
    alignas({{ min(32, BK) }}) uint32_t CBits[BM{{ pre }}];
    memset(&(CBits[0]), 0, BM{{ pre }} * sizeof(int32_t));
    #pragma unroll
    for (int32_t k_outer = 0; k_outer < {{ k }} / BBK{{ pre }}; ++k_outer) {
        tbl_impl_{{ pre }}((&(((int32_t*)CBits)[0])), (&(((int8_t*)LUT)[(k_outer * BBK{{ pre }} / 2 * 32)])), (&(((uint8_t*)A)[(k_outer * BBK{{ pre }} / 2 / 2 * BM{{ pre }})])));
    }
    #pragma unroll
    for (int i = 0; i < BM{{ pre }}; i++) {
        ((bitnet_float_type*)C)[i] = (((int32_t*)CBits)[i]) / ((bitnet_float_type*)LUT_Scales)[0] * ((bitnet_float_type*)Scales)[0];
    }
  return 0;
};

