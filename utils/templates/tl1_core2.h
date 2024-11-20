        int32x4_t vec_v_bot_low_low_{{ index }} = vmovl_s16(vget_low_s16(vec_c[{{ index }}]));
        int32x4_t vec_v_bot_low_high_{{ index }} = vmovl_high_s16(vec_c[{{ index }}]);
        vst1q_s32(c + i + {{ index * 8 }}, vld1q_s32(c + i + {{ index * 8 }}) + vec_v_bot_low_low_{{ index }});
        vst1q_s32(c + i + {{ index * 8 + 4 }}, vld1q_s32(c + i + {{ index * 8 + 4 }}) + vec_v_bot_low_high_{{ index }});
