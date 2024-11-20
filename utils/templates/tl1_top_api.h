void ggml_preprocessor(int m, int k, void* B, void* LUT_Scales, void* QLUT) {
{% for kernel_shape in kernel_shapes %}
    {% if loop.index0 > 0 %}else {% endif %}if (m == {{ kernel_shapes[loop.index0][0] }} && k == {{ kernel_shapes[loop.index0][1] }}) {
        preprocessor_k<{{ kernel_shapes[loop.index0][1] }}>(B, LUT_Scales, QLUT);
    }
{% endfor %}
}
void ggml_qgemm_lut(int m, int k, void* A, void* LUT, void* Scales, void* LUT_Scales, void* C) {
{% for kernel_shape in kernel_shapes %}
    {% if loop.index0 > 0 %}else {% endif %}if (m == {{ kernel_shapes[loop.index0][0] }} && k == {{ kernel_shapes[loop.index0][1] }}) {
        qgemm_lut_{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }}(A, LUT, Scales, LUT_Scales, C);
    }
{% endfor %}
}
