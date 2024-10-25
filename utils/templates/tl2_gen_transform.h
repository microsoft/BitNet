void ggml_bitnet_transform_tensor(struct ggml_tensor * tensor) {
    if (!(is_type_supported(tensor->type) && tensor->backend == GGML_BACKEND_TYPE_CPU && tensor->extra == nullptr)) {
        return;
    }

    int k = tensor->ne[0];
    int m = tensor->ne[1];
    const int lut_scales_size = 1;
    int bk = 0;
    int bm = 0;
    {% for kernel_shape in kernel_shapes %}
    {% if loop.index0 > 0 %}else {% endif %}if (m == {{ kernel_shapes[loop.index0][0] }} && k == {{ kernel_shapes[loop.index0][1] }}) {
        bm = BM{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }};
        bk = BBK{{ kernel_shapes[loop.index0][0] }}_{{ kernel_shapes[loop.index0][1] }};
    }
    {% endfor %}
    const int n_tile_num = m / bm;
    const int BK = bk;
    uint8_t * qweights;
    bitnet_float_type * scales;

    scales = (bitnet_float_type *) aligned_malloc(sizeof(bitnet_float_type));
    qweights = (uint8_t *) tensor->data;
    int nbytes = (k - 256) * m / 3 * 5 / 8 + 256 * m / 2 * 4 / 8;
    if (nbytes % 32 != 0) nbytes = 32 - nbytes % 32 + nbytes;
    float * i2_scales = (float * )(qweights + nbytes);
    scales[0] = (bitnet_float_type) i2_scales[0];

    tensor->extra = bitnet_tensor_extras + bitnet_tensor_extras_index;
    bitnet_tensor_extras[bitnet_tensor_extras_index++] = {
        /* .lut_scales_size = */ lut_scales_size,
        /* .BK              = */ BK,
        /* .n_tile_num      = */ n_tile_num,
        /* .qweights        = */ qweights,
        /* .scales          = */ scales
    };
}
