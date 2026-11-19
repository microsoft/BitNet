
// Initial full-GPU I2_S reference kernel, preserving the existing packed format.
kernel void kernel_mul_mv_i2s_reference(
        constant ulong4 & shape [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const char * input_bytes [[buffer(2)]],
        device char * output_bytes [[buffer(3)]],
        constant ulong2 & strides [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simdgroup [[simdgroup_index_in_threadgroup]]) {
    const uint width = uint(shape.x);
    const uint rows = uint(shape.y);
    const uint row = group.x * 4 + simdgroup;
    const uint token = group.y;
    if (row >= rows) return;
    device const float * input = (device const float *)(input_bytes + token * strides.x);
    float maximum = 0.0f;
    for (uint k = lane; k < width; k += 32) maximum = max(maximum, abs(input[k]));
    maximum = simd_max(maximum);
    const float scale = maximum > 0.0f ? 127.0f / maximum : 1.0f;
    int sum = 0;
    for (uint k = lane; k < width; k += 32) {
        const uint index = row * (width / 4) + (k / 128) * 32 + k % 32;
        const int code = (weights[index] >> (6 - 2 * ((k % 128) / 32))) & 3;
        const int activation = clamp(int(round(input[k] * scale)), -128, 127);
        sum += (code - 1) * activation;
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        const float weight_scale = *((device const float *)(weights + ulong(width) * rows / 4));
        device float * output = (device float *)(output_bytes + token * strides.y);
        output[row] = float(sum) * (weight_scale / scale);
    }
}
