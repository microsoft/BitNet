#include <arm_neon.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if !defined(__ARM_FEATURE_MATMUL_INT8)
#error This benchmark requires ARM I8MM.
#endif

void ggml_gemm_i2_i8_s(int n, float * s, size_t bs, const void * vx,
                      const void * vy, int nr, int nc);
void ggml_gemv_i2_i8_s(int n, float * s, size_t bs, const void * vx,
                      const void * vy, int nr, int nc);

static inline int8x16_t join_weight_halves(uint8x16_t a, uint8x16_t b, int high) {
    const int64x2_t aa = vreinterpretq_s64_u8(a);
    const int64x2_t bb = vreinterpretq_s64_u8(b);
    return vreinterpretq_s8_s64(high ? vzip2q_s64(aa, bb) : vzip1q_s64(aa, bb));
}

static inline void accumulate_group(uint8x16_t a0, uint8x16_t a1, uint8x16_t a2, uint8x16_t a3,
                                    const int8_t * ap, int32x4_t * s01a, int32x4_t * s01b,
                                    int32x4_t * s23a, int32x4_t * s23b) {
    const int8x16_t a01lo = join_weight_halves(a0, a1, 0);
    const int8x16_t a23lo = join_weight_halves(a2, a3, 0);
    const int8x16_t b01lo = vld1q_s8(ap);
    const int8x16_t b23lo = vld1q_s8(ap + 16);
    *s01a = vmmlaq_s32(*s01a, a01lo, b01lo);
    *s01b = vmmlaq_s32(*s01b, a01lo, b23lo);
    *s23a = vmmlaq_s32(*s23a, a23lo, b01lo);
    *s23b = vmmlaq_s32(*s23b, a23lo, b23lo);
    const int8x16_t a01hi = join_weight_halves(a0, a1, 1);
    const int8x16_t a23hi = join_weight_halves(a2, a3, 1);
    const int8x16_t b01hi = vld1q_s8(ap + 32);
    const int8x16_t b23hi = vld1q_s8(ap + 48);
    *s01a = vmmlaq_s32(*s01a, a01hi, b01hi);
    *s01b = vmmlaq_s32(*s01b, a01hi, b23hi);
    *s23a = vmmlaq_s32(*s23a, a23hi, b01hi);
    *s23b = vmmlaq_s32(*s23b, a23hi, b23hi);
}

static inline void store_pair(float * s, size_t stride, int32x4_t acc) {
    const float32x4_t f = vcvtq_f32_s32(acc);
    s[0]          = vgetq_lane_f32(f, 0);
    s[stride]     = vgetq_lane_f32(f, 1);
    s[1]          = vgetq_lane_f32(f, 2);
    s[stride + 1] = vgetq_lane_f32(f, 3);
}

// Uses the same 2-bit GGUF weight layout and uncentered integer products as SDOT.
__attribute__((noinline)) static void i8mm_i2s(int n, float * s, size_t bs, const uint8_t * x,
                                               const int8_t * y, int nr, int nc, int8_t * scratch) {
    assert(n % 128 == 0);
    const uint8x16_t mask = vdupq_n_u8(3);
    const size_t row_bytes = n / 4;
    int c = 0;
    for (; c + 4 <= nr; c += 4) {
        const int8_t * y0 = y + (size_t)c * n;
        const int8_t * y1 = y0 + n;
        const int8_t * y2 = y1 + n;
        const int8_t * y3 = y2 + n;
        for (int i = 0; i < n; i += 8) {
            vst1q_s8(scratch + 4 * i, vcombine_s8(vld1_s8(y0 + i), vld1_s8(y1 + i)));
            vst1q_s8(scratch + 4 * i + 16, vcombine_s8(vld1_s8(y2 + i), vld1_s8(y3 + i)));
        }

        int r = 0;
        for (; r + 4 <= nc; r += 4) {
            const uint8_t * x0 = x + (size_t)r * row_bytes;
            const uint8_t * x1 = x0 + row_bytes;
            const uint8_t * x2 = x1 + row_bytes;
            const uint8_t * x3 = x2 + row_bytes;
            int32x4_t s01a = vdupq_n_s32(0);
            int32x4_t s01b = vdupq_n_s32(0);
            int32x4_t s23a = vdupq_n_s32(0);
            int32x4_t s23b = vdupq_n_s32(0);
            for (int i = 0; i < n; i += 128) {
                for (int h = 0; h < 32; h += 16) {
                    const int off = i / 4 + h;
                    const uint8x16_t p0 = vld1q_u8(x0 + off);
                    const uint8x16_t p1 = vld1q_u8(x1 + off);
                    const uint8x16_t p2 = vld1q_u8(x2 + off);
                    const uint8x16_t p3 = vld1q_u8(x3 + off);
                    const int8_t * ap = scratch + 4 * (i + h);
                    accumulate_group(vshrq_n_u8(p0, 6), vshrq_n_u8(p1, 6),
                                     vshrq_n_u8(p2, 6), vshrq_n_u8(p3, 6),
                                     ap, &s01a, &s01b, &s23a, &s23b);
                    accumulate_group(vandq_u8(vshrq_n_u8(p0, 4), mask), vandq_u8(vshrq_n_u8(p1, 4), mask),
                                     vandq_u8(vshrq_n_u8(p2, 4), mask), vandq_u8(vshrq_n_u8(p3, 4), mask),
                                     ap + 128, &s01a, &s01b, &s23a, &s23b);
                    accumulate_group(vandq_u8(vshrq_n_u8(p0, 2), mask), vandq_u8(vshrq_n_u8(p1, 2), mask),
                                     vandq_u8(vshrq_n_u8(p2, 2), mask), vandq_u8(vshrq_n_u8(p3, 2), mask),
                                     ap + 256, &s01a, &s01b, &s23a, &s23b);
                    accumulate_group(vandq_u8(p0, mask), vandq_u8(p1, mask),
                                     vandq_u8(p2, mask), vandq_u8(p3, mask),
                                     ap + 384, &s01a, &s01b, &s23a, &s23b);
                }
            }
            float * base = s + (size_t)c * bs + r;
            store_pair(base, bs, s01a);
            store_pair(base + 2 * bs, bs, s01b);
            store_pair(base + 2, bs, s23a);
            store_pair(base + 2 * bs + 2, bs, s23b);
        }
        if (r < nc) {
            for (int j = 0; j < 4; ++j) {
                ggml_gemv_i2_i8_s(n, s + (size_t)(c + j) * bs + r, bs,
                                  x + (size_t)r * row_bytes, y + (size_t)(c + j) * n, 1, nc - r);
            }
        }
    }
    for (; c < nr; ++c) {
        ggml_gemv_i2_i8_s(n, s + (size_t)c * bs, bs, x, y + (size_t)c * n, 1, nc);
    }
}

static uint32_t rng = 23;
static uint32_t next_rng(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static void fill(int n, int rows, int cols, uint8_t * w, int8_t * a, unsigned code_count) {
    for (size_t i = 0; i < (size_t)n / 4 * rows; ++i) {
        unsigned code = 0;
        for (int shift = 0; shift < 8; shift += 2) {
            code |= (next_rng() % code_count) << shift;
        }
        w[i] = (uint8_t)code;
    }
    for (size_t i = 0; i < (size_t)n * cols; ++i) {
        a[i] = (int8_t)((next_rng() % 256) - 128);
    }
    a[0] = -128;
    a[1] = 127;
}

static int32_t reference(int n, const uint8_t * w, const int8_t * a) {
    int32_t s = 0;
    for (int k = 0; k < n; ++k) {
        unsigned code = (w[(k / 128) * 32 + k % 32] >> (6 - 2 * ((k % 128) / 32))) & 3;
        s += (int32_t)code * a[k];
    }
    return s;
}

static void correctness_case(int n, int rows, int cols, unsigned code_count) {
    size_t bs = rows + 5;
    uint8_t * w = malloc((size_t)n / 4 * rows);
    int8_t * a = malloc((size_t)n * cols);
    int8_t * scratch = malloc((size_t)n * 4);
    float * old = malloc(bs * cols * sizeof(float));
    float * candidate = malloc(bs * cols * sizeof(float));
    assert(w && a && scratch && old && candidate);
    fill(n, rows, cols, w, a, code_count);
    for (size_t i = 0; i < bs * cols; ++i) old[i] = candidate[i] = -999.f;
    ggml_gemm_i2_i8_s(n, old, bs, w, a, cols, rows);
    i8mm_i2s(n, candidate, bs, w, a, cols, rows, scratch);
    for (int c = 0; c < cols; ++c) {
        for (int r = 0; r < rows; ++r) {
            const int32_t expected = reference(n, w + (size_t)r * n / 4, a + (size_t)c * n);
            const size_t index = (size_t)c * bs + r;
            if (old[index] != (float)expected || candidate[index] != (float)expected) {
                fprintf(stderr, "n=%d row=%d col=%d expected=%d SDOT=%.0f I8MM=%.0f\n",
                        n, r, c, expected, old[index], candidate[index]);
                exit(EXIT_FAILURE);
            }
        }
        for (size_t i = rows; i < bs; ++i) {
            assert(old[(size_t)c * bs + i] == -999.f && candidate[(size_t)c * bs + i] == -999.f);
        }
    }
    free(w); free(a); free(scratch); free(old); free(candidate);
}

static double timestamp(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static int compare(const void * a, const void * b) {
    const double x = *(const double *)a;
    const double y = *(const double *)b;
    return (x > y) - (x < y);
}

static volatile float sink;

static void bench_case(int n, int rows, int cols) {
    enum { trials = 11, iters = 12 };
    uint8_t * w = malloc((size_t)n / 4 * rows);
    int8_t * a = malloc((size_t)n * cols);
    int8_t * scratch = malloc((size_t)n * 4);
    float * old = malloc((size_t)rows * cols * sizeof(float));
    float * candidate = malloc((size_t)rows * cols * sizeof(float));
    assert(w && a && scratch && old && candidate);
    fill(n, rows, cols, w, a, 3);
    ggml_gemm_i2_i8_s(n, old, rows, w, a, cols, rows);
    i8mm_i2s(n, candidate, rows, w, a, cols, rows, scratch);
    assert(!memcmp(old, candidate, (size_t)rows * cols * sizeof(float)));
    double times[2][trials];
    for (int trial = 0; trial < trials; ++trial) {
        for (int phase = 0; phase < 2; ++phase) {
            const int variant = (trial + phase) % 2;
            const double start = timestamp();
            for (int iter = 0; iter < iters; ++iter) {
                if (variant == 0) ggml_gemm_i2_i8_s(n, old, rows, w, a, cols, rows);
                else i8mm_i2s(n, candidate, rows, w, a, cols, rows, scratch);
            }
            times[variant][trial] = 1e3 * (timestamp() - start) / iters;
        }
    }
    for (int v = 0; v < 2; ++v) qsort(times[v], trials, sizeof(double), compare);
    sink += old[0] + candidate[0];
    const double sd = times[0][trials / 2];
    const double mm = times[1][trials / 2];
    printf("%5d x %5d x %2d  SDOT %8.3f ms  I8MM %8.3f ms  speedup %.3fx\n",
           rows, n, cols, sd, mm, sd / mm);
    free(w); free(a); free(scratch); free(old); free(candidate);
}

int main(int argc, char ** argv) {
    correctness_case(128, 1, 1, 3);
    correctness_case(128, 4, 4, 3);
    correctness_case(256, 11, 9, 4);
    correctness_case(2560, 19, 6, 3);
    correctness_case(6912, 5, 5, 3);
    puts("Signed int8 inputs including -128/+127, all packed codes, partial I8MM tiles, and padded outputs match scalar and SDOT exactly.");
    if (argc == 2 && !strcmp(argv[1], "--bench")) {
        bench_case(2560, 640, 4);
        bench_case(2560, 6912, 4);
        bench_case(6912, 2560, 4);
        bench_case(2560, 1728, 16);
    }
    return 0;
}
