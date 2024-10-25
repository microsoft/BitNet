void ggml_preprocessor(int bs, int m, int three_k, int two_k, void* B, void* LUT_Scales, void* Three_QLUT, void* Two_QLUT) {
    partial_max_reset(bs, (&(((float*)LUT_Scales)[0])));
{% for kernel_shape in kernel_shapes %}
    {% if loop.index0 > 0 %}else {% endif %}if (m == {{ kernel_shapes[loop.index0][0] }} && two_k == {{ k_list[loop.index0][0] }} && three_k == {{ k_list[loop.index0][1] }}) {
        for (int32_t b = 0; b < bs; b++) {
            per_tensor_quant(two_k + three_k, (&(((float*)LUT_Scales)[b])), (&(((float*)B)[b * (two_k + three_k)])));
            three_lut_ctor<{{ k_list[loop.index0][1] }}>((&(((int8_t*)Three_QLUT)[b * three_k / 3 * 32])), (&(((float*)B)[b * (three_k + two_k)])), (&(((float*)LUT_Scales)[b])));
            two_lut_ctor<{{ k_list[loop.index0][0] }}>((&(((int8_t*)Two_QLUT)[b * two_k / 2 * 32])), (&(((float*)B)[b * (three_k + two_k) + {{ k_list[loop.index0][1] }}])), (&(((float*)LUT_Scales)[b])));
        }
    }
{% endfor %}
}

void ggml_qgemm_lut(int bs, int m, int k, int BK, void* A, void* sign, void* LUT, void* Scales, void* LUT_Scales, void* C) {
{% for kernel_shape in kernel_shapes %}
    {% if loop.index0 > 0 %}else {% endif %}if (m == {{ kernel_shapes[loop.index0][0] }} && k == {{ kernel_shapes[loop.index0][1] }}) {
        if (BK == {{ k_list[loop.index0][0] }}) {
            if (bs == 1) {
                two_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<1>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 8) {
                two_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<8>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 32) {
                two_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<32>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 128) {
                two_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<128>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 256) {
                two_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<256>(A, LUT, Scales, LUT_Scales, C);
            } else if (bs == 512) {
                two_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<512>(A, LUT, Scales, LUT_Scales, C);
            }
        }
        else if (BK == {{ k_list[loop.index0][1] }}) {
            if (bs == 1) {
                three_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<1>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 8) {
                three_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<8>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 32) {
                three_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<32>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 128) {
                three_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<128>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 256) {
                three_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<256>(A, sign, LUT, Scales, LUT_Scales, C);
            }else if (bs == 512) {
                three_qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}<512>(A, sign, LUT, Scales, LUT_Scales, C);
            }
        }
    }
{% endfor %}
}
