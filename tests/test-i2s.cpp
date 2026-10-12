#include "ggml-quants.h"
#include "ggml-cpu-i2s.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static const float sentinel = -999999.0f;

static bool compare(const char * name, int n, int columns, int stride,
                    const std::vector<float> & actual, const std::vector<float> & expected) {
    for (size_t i = 0; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::printf("FAIL %s n=%d columns=%d stride=%d index=%zu actual=%g expected=%g\n",
                        name, n, columns, stride, i, actual[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool matrix_case(bool dot, int n, int columns, int padding, bool saturated) {
    const int rows = dot ? 1 : 5;
    const int stride = rows + padding;
    const int activation_stride = n + (dot ? padding * 32 : 0);
    // Extra initialized rows keep the known row-vs-column bug deterministic, not an out-of-bounds read.
    const int backing_rows = rows + columns - 1;
    std::vector<uint8_t> codes(backing_rows * n);
    std::vector<uint8_t> packed(backing_rows * n / 4, 0);
    std::vector<int8_t> activations(columns * activation_stride, 99);
    for (int row = 0; row < backing_rows; ++row) {
        for (int k = 0; k < n; ++k) {
            codes[row * n + k] = saturated ? 2 : (k * k + k / 32 + row * 2) % 3;
        }
        // Four groups of 32 codes share each 32-byte block; do not use the kernel's quantizer.
        for (int block = 0; block < n / 128; ++block) {
            for (int lane = 0; lane < 32; ++lane) {
                for (int group = 0; group < 4; ++group) {
                    packed[row * n / 4 + block * 32 + lane] |=
                        codes[row * n + block * 128 + group * 32 + lane] << (6 - 2 * group);
                }
            }
        }
    }
    for (int col = 0; col < columns; ++col) {
        for (int k = 0; k < n; ++k) {
            activations[col * activation_stride + k] =
                saturated ? 127 : (k * 17 + col * 31) % 255 - 127;
        }
    }

    std::vector<float> expected(columns * stride + 4, sentinel);
    for (int col = 0; col < columns; ++col) {
        for (int row = 0; row < rows; ++row) {
            int32_t sum = 0;
            for (int k = 0; k < n; ++k) {
                // The caller subtracts the activation sum: these kernels return unsigned-code dots.
                sum += codes[row * n + k] * activations[col * activation_stride + k];
            }
            expected[col * stride + row] = static_cast<float>(sum);
        }
    }
    std::vector<float> actual(expected.size(), sentinel);
    if (dot) {
        ggml_vec_dot_i2_i8_s(n, actual.data(), stride, packed.data(), n,
                            activations.data(), activation_stride, columns);
    } else {
        ggml_gemm_i2_i8_s(n, actual.data(), stride, packed.data(), activations.data(), columns, rows);
    }
    return compare(saturated ? "saturated" : "mixed", n, columns, stride, actual, expected);
}

static bool zero_case(int n) {
    std::vector<float> input(n, 0.0f);
    std::vector<int8_t> quantized(n, 99);
    float scale = -1.0f;
    int32_t sum = -1;
    quantize_row_i8_s(input.data(), quantized.data(), n, &scale, &sum);
    const bool all_zero = std::all_of(quantized.begin(), quantized.end(), [](int8_t v) { return v == 0; });
    if (!all_zero || sum != 0 || !(scale > 0.0f) || !std::isfinite(scale) ||
        !std::isfinite(static_cast<float>(sum) / scale)) {
        std::printf("FAIL zero n=%d all_zero=%d sum=%d scale=%g\n", n, all_zero, sum, scale);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 2 || (std::strcmp(argv[1], "dot") != 0 &&
                      std::strcmp(argv[1], "gemm") != 0 && std::strcmp(argv[1], "zero") != 0)) {
        std::fprintf(stderr, "Usage: %s dot|gemm|zero\n", argv[0]);
        return 2;
    }
    int failures = 0;
    int cases = 0;
    for (int n : {128, 256, 2560, 6912}) {
        if (std::strcmp(argv[1], "zero") == 0) {
            ++cases;
            failures += !zero_case(n);
            continue;
        }
        for (int columns : {1, 4, 8, 16, 32, 48, 64}) {
            for (int padding : {0, 3}) {
                for (bool saturated : {false, true}) {
                    ++cases;
                    failures += !matrix_case(std::strcmp(argv[1], "dot") == 0,
                                             n, columns, padding, saturated);
                }
            }
        }
    }
    std::printf("%s: %d/%d cases failed\n", argv[1], failures, cases);
    return failures == 0 ? 0 : 1;
}
