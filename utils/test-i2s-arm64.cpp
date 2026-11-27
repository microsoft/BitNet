// Standalone numerical equivalence unit test for ARM64 I2_S (QK_I2_S=128) kernels.
//
// Build with DOTPROD:
//   clang++ -O2 -std=c++17 -Wall -Wextra -march=armv8.2-a+dotprod \
//     -Iinclude -I3rdparty/llama.cpp/ggml/include \
//     -I3rdparty/llama.cpp/ggml/src -I3rdparty/llama.cpp/ggml/src/ggml-cpu \
//     utils/test-i2s-arm64.cpp src/ggml-bitnet-mad.cpp -o test-i2s-dotprod
//
// Build without DOTPROD (NEON fallback):
//   clang++ -O2 -std=c++17 -Wall -Wextra -march=armv8-a \
//     -Iinclude -I3rdparty/llama.cpp/ggml/include \
//     -I3rdparty/llama.cpp/ggml/src -I3rdparty/llama.cpp/ggml/src/ggml-cpu \
//     utils/test-i2s-arm64.cpp src/ggml-bitnet-mad.cpp -o test-i2s-fallback

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-quants.h"
#include "gemm-config.h"

// Minimal ggml_row_size stub so the test links standalone without libggml.
extern "C" size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    (void)type;
    return static_cast<size_t>(ne);
}

// Kernel entry points defined in src/ggml-bitnet-mad.cpp.
void ggml_vec_dot_i2_i8_s_1x1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_i2_i8_s_1xN(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);
void ggml_vec_dot_i2_i8_s_Nx1(int n, float * s, size_t bs, const void * vx, size_t bx, const void * vy, size_t by, int nrc);

namespace {

constexpr int kBlockSize = 128;
constexpr int kBlockBytes = 32;

// Independent scalar reference encoder matching the x86 / official GGUF I2_S format:
// 128 ternary weights per 32-byte block (4 groups of 32 packed MSB-to-LSB as 2-bit codes:
// -1 -> 0, 0 -> 1, +1 -> 2), followed by a trailing float32 scale.
void ref_quantize_i2_s(const float * src, std::vector<uint8_t> & out_packed, float & out_scale, int64_t nrow, int64_t n_per_row) {
    const int64_t n = nrow * n_per_row;
    double max_abs = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double a = std::fabs(static_cast<double>(src[i]));
        if (a > max_abs) {
            max_abs = a;
        }
    }
    out_scale = static_cast<float>(max_abs);

    std::vector<uint8_t> q8(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        if (std::fabs(static_cast<double>(src[i])) < 1e-6) {
            q8[static_cast<size_t>(i)] = 1;
        } else {
            q8[static_cast<size_t>(i)] = (static_cast<double>(src[i]) * max_abs > 0.0) ? 2 : 0;
        }
    }

    out_packed.assign(static_cast<size_t>(n / 4), 0);
    for (int64_t blk = 0; blk < n / kBlockSize; ++blk) {
        for (int j = 0; j < kBlockSize; ++j) {
            const int group_idx = j / 32;
            const int group_pos = j % 32;
            const uint8_t code = q8[static_cast<size_t>(blk * kBlockSize + j)];
            out_packed[static_cast<size_t>(blk * kBlockBytes + group_pos)] |=
                static_cast<uint8_t>(code << (6 - 2 * group_idx));
        }
    }
}

// Independent scalar reference dot product over packed QK_I2_S=128 weights and int8 activations.
int64_t ref_dot_i2_i8(const uint8_t * packed_row, const int8_t * y, int n) {
    int64_t sum = 0;
    const int nb = n / kBlockSize;
    for (int blk = 0; blk < nb; ++blk) {
        const uint8_t * px = packed_row + static_cast<size_t>(blk) * kBlockBytes;
        const int8_t  * py = y + static_cast<size_t>(blk) * kBlockSize;
        for (int p = 0; p < 32; ++p) {
            const uint8_t b = px[p];
            const int q0 = (b >> 6) & 0x3;
            const int q1 = (b >> 4) & 0x3;
            const int q2 = (b >> 2) & 0x3;
            const int q3 = b & 0x3;
            sum += static_cast<int64_t>(q0) * py[p + 0];
            sum += static_cast<int64_t>(q1) * py[p + 32];
            sum += static_cast<int64_t>(q2) * py[p + 64];
            sum += static_cast<int64_t>(q3) * py[p + 96];
        }
    }
    return sum;
}

bool check_close(float got, int64_t expected, const char * label, int n, int nrc, int idx) {
    const double exp_d = static_cast<double>(expected);
    const double got_d = static_cast<double>(got);
    const double abs_err = std::fabs(got_d - exp_d);
    const double rel_err = (std::fabs(exp_d) > 0.0) ? abs_err / std::fabs(exp_d) : abs_err;
    // Exact integer match is expected whenever |expected| <= 2^24 (always true for n <= 6912).
    const bool exact_ok = (std::llabs(expected) <= (1LL << 24)) ? (abs_err == 0.0) : (rel_err < 1e-5);
    if (!exact_ok) {
        std::fprintf(stderr,
            "FAIL [%s] n=%d nrc=%d idx=%d: got=%.1f expected=%lld abs_err=%.6g rel_err=%.6g\n",
            label, n, nrc, idx, got_d, static_cast<long long>(expected), abs_err, rel_err);
        return false;
    }
    return true;
}

bool test_quantize_i2_s(int & checks) {
    struct Shape {
        int64_t nrow;
        int64_t n_per_row;
    };
    const Shape shapes[] = {
        {1, 128},
        {2, 256},
        {4, 2560},
        {2, 6912},
    };
    const float ternary_vals[3] = {-1.75f, 0.0f, 1.75f};
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 2);

    for (const Shape & sh : shapes) {
        const int64_t n = sh.nrow * sh.n_per_row;
        for (int mode = 0; mode < 4; ++mode) {
            std::vector<float> src(static_cast<size_t>(n));
            for (int64_t i = 0; i < n; ++i) {
                if (mode == 0) {
                    src[static_cast<size_t>(i)] = 2.5f;
                } else if (mode == 1) {
                    src[static_cast<size_t>(i)] = -3.25f;
                } else if (mode == 2) {
                    // Position-dependent ternary pattern to detect any intra-block permutation.
                    src[static_cast<size_t>(i)] = ternary_vals[(i + (i / 32) + (i / 128)) % 3];
                } else {
                    src[static_cast<size_t>(i)] = ternary_vals[dist(rng)];
                }
            }

            std::vector<uint8_t> ref_packed;
            float ref_scale = 0.0f;
            ref_quantize_i2_s(src.data(), ref_packed, ref_scale, sh.nrow, sh.n_per_row);

            std::vector<uint8_t> dst(static_cast<size_t>(n / 4) + 64, 0xA5);
            const size_t ret = quantize_i2_s(src.data(), dst.data(), sh.nrow, sh.n_per_row, nullptr);
            const size_t expected_ret = static_cast<size_t>(n / 4) + 32;
            if (ret != expected_ret) {
                std::fprintf(stderr,
                    "FAIL [quantize_i2_s return size] (%lldx%lld mode=%d): got=%zu expected=%zu\n",
                    static_cast<long long>(sh.nrow), static_cast<long long>(sh.n_per_row), mode, ret, expected_ret);
                return false;
            }

            for (size_t b = 0; b < ref_packed.size(); ++b) {
                if (dst[b] != ref_packed[b]) {
                    std::fprintf(stderr,
                        "FAIL [quantize_i2_s byte] (%lldx%lld mode=%d) byte=%zu: got=0x%02x expected=0x%02x\n",
                        static_cast<long long>(sh.nrow), static_cast<long long>(sh.n_per_row),
                        mode, b, dst[b], ref_packed[b]);
                    return false;
                }
            }

            float got_scale = 0.0f;
            std::memcpy(&got_scale, dst.data() + static_cast<size_t>(n / 4), sizeof(float));
            if (std::fabs(got_scale - ref_scale) > 1e-6f) {
                std::fprintf(stderr,
                    "FAIL [quantize_i2_s scale] (%lldx%lld mode=%d): got=%g expected=%g\n",
                    static_cast<long long>(sh.nrow), static_cast<long long>(sh.n_per_row),
                    mode, got_scale, ref_scale);
                return false;
            }
            ++checks;
        }
    }
    return true;
}

bool test_vec_dot_kernels(int & checks) {
    const int n_list[] = {128, 256, 2560, 6912};
    const int nrc_list[] = {1, 2, 3, 4, 6, 8};
    const float w_choices[3] = {-1.0f, 0.0f, 1.0f};

    std::mt19937 rng(20261127);
    std::uniform_int_distribution<int> w_dist(0, 2);
    std::uniform_int_distribution<int> y_dist(-127, 127);

    enum Pattern {
        PAT_EXTREME_POS = 0, // weights=+1 (q8=2), y=+127 (stress int16 overflow)
        PAT_EXTREME_NEG = 1, // weights=+1 (q8=2), y=-127 (stress negative int16 overflow)
        PAT_ALL_NEG_W   = 2, // weights=-1 (q8=0), y=random
        PAT_ALL_ZERO_W  = 3, // weights=0  (q8=1), y=±127
        PAT_IMPULSE_POS = 4, // distinct per-position weights & activations
        PAT_RANDOM      = 5, // random {-1,0,+1} weights and [-127,127] activations
        PAT_COUNT       = 6
    };

    for (int n : n_list) {
        const size_t row_bytes = static_cast<size_t>(n / 4);
        for (int nrc : nrc_list) {
            for (int pat = 0; pat < PAT_COUNT; ++pat) {
                // Prepare nrc weight rows (used by 1x1 and 1xN; row 0 used by Nx1).
                std::vector<float> w_f32(static_cast<size_t>(nrc) * static_cast<size_t>(n));
                for (int r = 0; r < nrc; ++r) {
                    for (int k = 0; k < n; ++k) {
                        const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(n) + static_cast<size_t>(k);
                        if (pat == PAT_EXTREME_POS || pat == PAT_EXTREME_NEG) {
                            w_f32[idx] = 1.0f;
                        } else if (pat == PAT_ALL_NEG_W) {
                            w_f32[idx] = -1.0f;
                        } else if (pat == PAT_ALL_ZERO_W) {
                            w_f32[idx] = 0.0f;
                        } else if (pat == PAT_IMPULSE_POS) {
                            w_f32[idx] = w_choices[(k + r * 7 + (k / 32)) % 3];
                        } else {
                            w_f32[idx] = w_choices[w_dist(rng)];
                        }
                    }
                }

                std::vector<uint8_t> packed_x(static_cast<size_t>(nrc) * row_bytes + 64, 0);
                quantize_i2_s(w_f32.data(), packed_x.data(), nrc, n, nullptr);

                // Prepare nrc activation vectors (col 0 used by 1x1 and 1xN; all nrc cols used by Nx1).
                std::vector<int8_t> y_mat(static_cast<size_t>(nrc) * static_cast<size_t>(n));
                for (int c = 0; c < nrc; ++c) {
                    for (int k = 0; k < n; ++k) {
                        const size_t idx = static_cast<size_t>(c) * static_cast<size_t>(n) + static_cast<size_t>(k);
                        if (pat == PAT_EXTREME_POS) {
                            y_mat[idx] = 127;
                        } else if (pat == PAT_EXTREME_NEG) {
                            y_mat[idx] = -127;
                        } else if (pat == PAT_ALL_ZERO_W) {
                            y_mat[idx] = (k % 2 == 0) ? 127 : -127;
                        } else if (pat == PAT_IMPULSE_POS) {
                            y_mat[idx] = static_cast<int8_t>(((k + 13 * c) % 251) - 125);
                        } else {
                            y_mat[idx] = static_cast<int8_t>(y_dist(rng));
                        }
                    }
                }

                // 1) Test ggml_vec_dot_i2_i8_s_1x1 (nrc rows x 1 activation vector)
                {
                    std::vector<float> out(static_cast<size_t>(nrc), -999999.0f);
                    ggml_vec_dot_i2_i8_s_1x1(n, out.data(), 1, packed_x.data(), static_cast<size_t>(n), y_mat.data(), 0, nrc);
                    for (int r = 0; r < nrc; ++r) {
                        const int64_t exp = ref_dot_i2_i8(packed_x.data() + static_cast<size_t>(r) * row_bytes, y_mat.data(), n);
                        if (!check_close(out[static_cast<size_t>(r)], exp, "1x1", n, nrc, r)) {
                            return false;
                        }
                    }
                    ++checks;
                }

                // 2) Test ggml_vec_dot_i2_i8_s_1xN (nrc rows x 1 activation vector)
                {
                    std::vector<float> out(static_cast<size_t>(nrc), -999999.0f);
                    ggml_vec_dot_i2_i8_s_1xN(n, out.data(), 1, packed_x.data(), static_cast<size_t>(n), y_mat.data(), 0, nrc);
                    for (int r = 0; r < nrc; ++r) {
                        const int64_t exp = ref_dot_i2_i8(packed_x.data() + static_cast<size_t>(r) * row_bytes, y_mat.data(), n);
                        if (!check_close(out[static_cast<size_t>(r)], exp, "1xN", n, nrc, r)) {
                            return false;
                        }
                    }
                    ++checks;
                }

                // 3) Test ggml_vec_dot_i2_i8_s_Nx1 (1 weight row x nrc activation columns, stride bs=3)
                {
                    const size_t bs = 3;
                    std::vector<float> out(static_cast<size_t>(nrc) * bs, -999999.0f);
                    ggml_vec_dot_i2_i8_s_Nx1(
                        n, out.data(), bs, packed_x.data(), static_cast<size_t>(n),
                        y_mat.data(), static_cast<size_t>(n), nrc);
                    for (int c = 0; c < nrc; ++c) {
                        const int64_t exp = ref_dot_i2_i8(
                            packed_x.data(), y_mat.data() + static_cast<size_t>(c) * static_cast<size_t>(n), n);
                        if (!check_close(out[static_cast<size_t>(c) * bs], exp, "Nx1", n, nrc, c)) {
                            return false;
                        }
                    }
                    ++checks;
                }

                // 4) Test top-level dispatcher ggml_vec_dot_i2_i8_s for ALL nrc values
                //    (including non-multiples of PARALLEL_SIZE, e.g. prompt tails during GEMM prefill).
                {
                    const size_t bs = (nrc == 1) ? 1 : 2;
                    std::vector<float> out(static_cast<size_t>(nrc) * bs, -999999.0f);
                    ggml_vec_dot_i2_i8_s(
                        n, out.data(), bs, packed_x.data(), static_cast<size_t>(n),
                        y_mat.data(), static_cast<size_t>(n), nrc);
                    for (int c = 0; c < nrc; ++c) {
                        const int64_t exp = ref_dot_i2_i8(
                            packed_x.data(), y_mat.data() + static_cast<size_t>(c) * static_cast<size_t>(n), n);
                        if (!check_close(out[static_cast<size_t>(c) * bs], exp, "dispatch_Nx1", n, nrc, c)) {
                            return false;
                        }
                    }
                    ++checks;
                }
            }
        }
    }

    // 5) Bit-exact single-element impulse sweep across all 128 positions in a block.
    {
        const int n = 128;
        std::vector<float> w_f32(n, -1.0f); // all q8 = 0
        std::vector<int8_t> y(n, 0);
        for (int pos = 0; pos < n; ++pos) {
            std::fill(w_f32.begin(), w_f32.end(), -1.0f);
            std::fill(y.begin(), y.end(), 0);
            w_f32[static_cast<size_t>(pos)] = 1.0f; // q8 = 2 only at pos
            y[static_cast<size_t>(pos)] = 37;

            std::vector<uint8_t> packed(static_cast<size_t>(n / 4) + 64, 0);
            quantize_i2_s(w_f32.data(), packed.data(), 1, n, nullptr);

            float out_1x1 = 0.0f;
            float out_nx1 = 0.0f;
            ggml_vec_dot_i2_i8_s_1x1(n, &out_1x1, 1, packed.data(), n, y.data(), 0, 1);
            ggml_vec_dot_i2_i8_s_Nx1(n, &out_nx1, 1, packed.data(), n, y.data(), n, 1);
            if (!check_close(out_1x1, 74, "impulse_1x1", n, 1, pos) ||
                !check_close(out_nx1, 74, "impulse_Nx1", n, 1, pos)) {
                return false;
            }
            ++checks;
        }
    }

    return true;
}

} // namespace

int main() {
#if defined(__ARM_FEATURE_DOTPROD)
    const char * path_name = "ARM64 NEON + DOTPROD (__ARM_FEATURE_DOTPROD=1)";
#elif defined(__ARM_NEON)
    const char * path_name = "ARM64 NEON fallback (__ARM_FEATURE_DOTPROD=0)";
#else
    const char * path_name = "non-ARM";
#endif

    std::printf("Running ARM64 I2_S unit tests [%s, PARALLEL_SIZE=%d]\n", path_name, PARALLEL_SIZE);

    int quant_checks = 0;
    if (!test_quantize_i2_s(quant_checks)) {
        return 1;
    }
    int dot_checks = 0;
    if (!test_vec_dot_kernels(dot_checks)) {
        return 1;
    }

    std::printf("PASS: %d quantize_i2_s checks, %d vec_dot checks, 0 errors.\n", quant_checks, dot_checks);
    return 0;
}
