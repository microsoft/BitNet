/*
 * test_ggml_bitnet_kinecat_fusion.cpp
 *
 * Self-contained C++17 AVX2 verification & benchmark for ggml_bitnet_kinecat_fusion.h
 * Compile: g++ -O3 -mavx2 -mfma test_ggml_bitnet_kinecat_fusion.cpp -o /tmp/test_bitnet_kinecat
 */

#include "../include/ggml-bitnet-kinecat.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cassert>

static uint8_t encode_tl2_triple(int8_t t0, int8_t t1, int8_t t2) {
    const int32_t raw_idx = (t0 + 1) * 9 + (t1 + 1) * 3 + (t2 + 1);
    if (raw_idx == 13) return 0u;
    if (raw_idx > 13) return (uint8_t)(raw_idx - 13);
    return (uint8_t)(0x10u | (uint8_t)(13 - raw_idx));
}

int main() {
    const int32_t rows = 64;
    const int32_t cols = 256;
    std::vector<int8_t> trits(rows * cols);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t c = 0; c < cols; ++c) {
            const int32_t m = (r * 19 + c * 37) % 3;
            trits[r * cols + c] = (m == 0) ? -1 : ((m == 1) ? 0 : 1);
        }
    }

    /* Pack TL1 (2-trits-in-4-bits, 2 pairs per byte) */
    const int32_t packed_bytes_per_row = cols / 4;
    std::vector<uint8_t> packed_tl1(rows * packed_bytes_per_row);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t b = 0; b < packed_bytes_per_row; ++b) {
            const int8_t t0 = trits[r * cols + 4 * b];
            const int8_t t1 = trits[r * cols + 4 * b + 1];
            const int8_t t2 = trits[r * cols + 4 * b + 2];
            const int8_t t3 = trits[r * cols + 4 * b + 3];
            const uint8_t idx_lo = (uint8_t)((t0 + 1) * 3 + (t1 + 1));
            const uint8_t idx_hi = (uint8_t)((t2 + 1) * 3 + (t3 + 1));
            packed_tl1[r * packed_bytes_per_row + b] = (idx_lo & 0x0Fu) | ((idx_hi & 0x0Fu) << 4);
        }
    }

    /* Pack TL2 (3-trits-in-5-bits signed half-space) */
    const int32_t triples_per_row = (cols + 2) / 3;
    std::vector<uint8_t> codes_tl2(rows * triples_per_row);
    for (int32_t r = 0; r < rows; ++r) {
        for (int32_t g = 0; g < triples_per_row; ++g) {
            const int32_t c0 = 3 * g;
            const int8_t t0 = trits[r * cols + c0];
            const int8_t t1 = (c0 + 1 < cols) ? trits[r * cols + c0 + 1] : 0;
            const int8_t t2 = (c0 + 2 < cols) ? trits[r * cols + c0 + 2] : 0;
            codes_tl2[r * triples_per_row + g] = encode_tl2_triple(t0, t1, t2);
        }
    }

    std::vector<float> sigma_in(cols);
    std::vector<float> tau_out(rows);
    std::vector<int32_t> t_scales_q16(rows);
    std::vector<int32_t> o_scales_q16(rows);
    std::vector<uint32_t> csr_row_ptr(rows + 1);
    std::vector<uint16_t> csr_col_idx(rows * 2);
    std::vector<int8_t> csr_vals(rows * 2);

    for (int32_t c = 0; c < cols; ++c) {
        sigma_in[c] = 0.85f + 0.30f * ((float)(c % 7) / 7.0f);
    }
    csr_row_ptr[0] = 0;
    for (int32_t r = 0; r < rows; ++r) {
        tau_out[r] = 0.90f + 0.20f * ((float)(r % 5) / 5.0f);
        t_scales_q16[r] = 32768 + r * 256;
        o_scales_q16[r] = 16384 + r * 128;
        csr_col_idx[2 * r]     = (uint16_t)((r * 11) % cols);
        csr_vals[2 * r]        = 49;
        csr_col_idx[2 * r + 1] = (uint16_t)((r * 23 + 7) % cols);
        csr_vals[2 * r + 1]    = -38;
        csr_row_ptr[r + 1]     = (uint32_t)((r + 1) * 2);
    }

    ggml_bitnet_kinecat_morph_t morph = {
        rows,
        cols,
        sigma_in.data(),
        tau_out.data(),
        t_scales_q16.data(),
        o_scales_q16.data(),
        csr_row_ptr.data(),
        csr_col_idx.data(),
        csr_vals.data()
    };

    std::vector<float> x_in(cols);
    for (int32_t c = 0; c < cols; ++c) {
        x_in[c] = std::sin((float)c * 0.17f) * 2.0f;
    }
    std::vector<int8_t> x_q8(cols);
    const float x_scale = ggml_bitnet_subln_and_quantize_i8(
        x_in.data(), sigma_in.data(), cols, 1e-5f, x_q8.data()
    );

    std::vector<float> y_tl1(rows, 0.0f);
    std::vector<float> y_tl2(rows, 0.0f);
    ggml_bitnet_tl1_fisher_csr_q16_matvec(packed_tl1.data(), &morph, x_q8.data(), x_scale, y_tl1.data());
    ggml_bitnet_tl2_fisher_csr_q16_matvec(codes_tl2.data(),  &morph, x_q8.data(), x_scale, y_tl2.data());

    float max_diff_tl1 = 0.0f;
    float max_diff_tl2 = 0.0f;
    for (int32_t r = 0; r < rows; ++r) {
        int32_t ref_t = 0;
        for (int32_t c = 0; c < cols; ++c) {
            ref_t += (int32_t)trits[r * cols + c] * (int32_t)x_q8[c];
        }
        int32_t ref_o = 0;
        for (uint32_t idx = csr_row_ptr[r]; idx < csr_row_ptr[r + 1]; ++idx) {
            ref_o += (int32_t)csr_vals[idx] * (int32_t)x_q8[csr_col_idx[idx]];
        }
        const float expected = (((float)ref_t * ((float)t_scales_q16[r] / 65536.0f))
                              + ((float)ref_o * ((float)o_scales_q16[r] / 65536.0f)))
                              * x_scale * tau_out[r];
        const float d1 = std::fabs(y_tl1[r] - expected);
        const float d2 = std::fabs(y_tl2[r] - expected);
        if (d1 > max_diff_tl1) max_diff_tl1 = d1;
        if (d2 > max_diff_tl2) max_diff_tl2 = d2;
    }

    std::printf(
        "[OK] ggml_bitnet_kinecat_fusion.h verified! max_diff_tl1=%.2e | max_diff_tl2=%.2e\n",
        max_diff_tl1, max_diff_tl2
    );
    assert(max_diff_tl1 < 1e-5f);
    assert(max_diff_tl2 < 1e-5f);
    return 0;
}
