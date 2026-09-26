// Local build fix (bitnet-pruebas), NOT upstream.
//
// ggml.c (ggml-base) references dequantize_row_i2_s via the I2_S type traits
// (.to_float), but no translation unit linked into ggml-base defines it: the
// only definition lives in the ggml-cpu backend (quants.c), which is a
// separate target. This file provides that exact scalar implementation
// (copied verbatim from 3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c) so
// ggml-base.dll links. It is only a .to_float fallback; the hot I2_S
// GEMV/GEMM path uses the native kernels in ggml-cpu-i2s.c.

#include <stdint.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

void dequantize_row_i2_s(const uint8_t * x, float * y, int64_t n, const float i2_scale) {
    static const float map2bit[4] = { -1.0f, 0.0f, 1.0f, 0.0f };
    int64_t done = 0;
    while (done < n) {
        int64_t cols0 = MIN(32, n - done - 0*32);
        int64_t cols1 = MIN(32, n - done - 1*32);
        int64_t cols2 = MIN(32, n - done - 2*32);
        int64_t cols3 = MIN(32, n - done - 3*32);
        for (int gp = 0; gp < 32; gp++) {
            uint8_t byte = x[(done/4) + gp];
            uint8_t c0 = (byte >> 6) & 0x03;
            uint8_t c1 = (byte >> 4) & 0x03;
            uint8_t c2 = (byte >> 2) & 0x03;
            uint8_t c3 = (byte >> 0) & 0x03;
            if (gp < cols0) y[done + 0*32 + gp] = i2_scale * map2bit[c0];
            if (gp < cols1) y[done + 1*32 + gp] = i2_scale * map2bit[c1];
            if (gp < cols2) y[done + 2*32 + gp] = i2_scale * map2bit[c2];
            if (gp < cols3) y[done + 3*32 + gp] = i2_scale * map2bit[c3];
        }
        done += 128;
    }
}
