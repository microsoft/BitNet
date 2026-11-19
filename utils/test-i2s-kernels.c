#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
size_t quantize_i2_s(const float *, void *, int64_t, int64_t, const float *);
void dequantize_row_i2_s(const uint8_t *, float *, int64_t, float);
void quantize_row_i8_s(const float *, void *, int64_t, float *, int32_t *);
void ggml_gemm_i2_i8_s(int, float *, size_t, const void *, const void *, int,
                       int);
void ggml_gemv_i2_i8_s(int, float *, size_t, const void *, const void *, int,
                       int);

static void check(int N, int R, int C) {
  const int stride = R + 3;
  float w[R * N], back[N], out[C * stride];
  uint8_t packed[R * N / 4 + 32];
  int8_t y[C * N];
  for (int r = 0; r < R; r++)
    for (int i = 0; i < N; i++)
      w[r * N + i] = ((i * 7 + i / 17 + r * 11) % 3) - 1;
  assert(quantize_i2_s(w, packed, R, N, 0) == sizeof packed);
  float scale = *(float *)(packed + R * N / 4);
  for (int r = 0; r < R; r++) {
    dequantize_row_i2_s(packed + r * N / 4, back, N, scale);
    for (int i = 0; i < N; i++)
      assert(back[i] == w[r * N + i]);
  }
  for (int c = 0; c < C; c++)
    for (int i = 0; i < N; i++)
      y[c * N + i] = (i * 3 + c * 7) % 256 - 128;
  for (int i = 0; i < C * stride; ++i)
    out[i] = -12345.0f;
  ggml_gemm_i2_i8_s(N, out, stride, packed, y, C, R);
  for (int c = 0; c < C; c++)
    for (int r = 0; r < R; r++) {
      int expected = 0, sum = 0;
      for (int i = 0; i < N; i++) {
        expected += w[r * N + i] * y[c * N + i];
        sum += y[c * N + i];
      }
      assert(out[c * stride + r] - sum == expected);
    }
  for (int c = 0; c < C; ++c)
    for (int r = R; r < stride; ++r)
      assert(out[c * stride + r] == -12345.0f);
  float gemv[R];
  ggml_gemv_i2_i8_s(N, gemv, R, packed, y, 1, R);
  for (int r = 0; r < R; ++r)
    assert(gemv[r] == out[r]);
  float z[N], as;
  memset(z, 0, sizeof z);
  int32_t asum;
  int8_t qz[N];
  quantize_row_i8_s(z, qz, N, &as, &asum);
  assert(as > 0 && isfinite(as) && asum == 0);
  printf("PASS: I2_S roundtrip and GEMM N=%d rows=%d columns=%d; zero "
         "activations\n",
         N, R, C);
}

static void check_activation(int n) {
  float x[n], scale;
  int8_t actual[n];
  int32_t sum;
  for (int pattern = 0; pattern < 5; ++pattern) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) {
      x[i] = pattern == 0   ? 0.0f
             : pattern == 1 ? ((i * 17) % 509 - 254) * 0.125f
             : pattern == 2 ? ((i * 29) % 253 - 126) + 0.5f
                            : sinf(i * 0.731f) * 3.14f;
      if (fabsf(x[i]) > amax)
        amax = fabsf(x[i]);
    }
    if (pattern == 4) {
      x[0] = 127.0f;
      amax = 127.0f;
      for (int i = 1; i < n; ++i)
        x[i] = (i % 253 - 126) + 0.5f;
    }
    const float expected_scale = amax > 0.0f ? 127.0f / amax : 1.0f;
    quantize_row_i8_s(x, actual, n, &scale, &sum);
    assert(scale == expected_scale);
    int32_t expected_sum = 0;
    for (int i = 0; i < n; ++i) {
      int q = (int)roundf(x[i] * expected_scale);
      if (q > 127)
        q = 127;
      if (q < -128)
        q = -128;
      assert(actual[i] == q);
      expected_sum += q;
    }
    assert(sum == expected_sum);
  }
  printf("PASS: activation quantization matches scalar reference, n=%d\n", n);
}

int main(void) {
  const int lengths[] = {1, 15, 16, 17, 128, 2560, 6912};
  for (unsigned i = 0; i < sizeof lengths / sizeof lengths[0]; ++i)
    check_activation(lengths[i]);
  check(128, 1, 1);
  check(2560, 5, 4);
  check(2560, 8, 1);
  check(6912, 9, 7);
  return 0;
}
