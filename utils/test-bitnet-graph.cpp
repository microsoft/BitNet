#include "ggml.h"

#include "../3rdparty/llama.cpp/src/models/bitnet-cpu.h"

#include "ggml-cpu.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
static void check(int tokens, int threads) {
  auto *ctx = ggml_init({32 * 1024 * 1024, nullptr, false});
  assert(ctx);
  auto *input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, tokens);
  auto *x = (float *)input->data;
  for (int i = 0; i < 256 * tokens; i++)
    x[i] = (i % 127 - 63) * .013f;
  ggml_tensor *weights[3], *old[3];
  int rows[3] = {32, 12, 20};
  int total = 64;
  for (int j = 0; j < 3; j++) {
    weights[j] = ggml_new_tensor_2d(ctx, GGML_TYPE_I2_S, 256, rows[j]);
    std::vector<float> w(256 * rows[j]);
    for (int i = 0; i < (int)w.size(); i++)
      w[i] = (i % 3 - 1) * (.25f * (j + 1));
    quantize_i2_s(w.data(), weights[j]->data, rows[j], 256, nullptr);
    old[j] = ggml_mul_mat(ctx, weights[j], input);
  }
  ggml_tensor *args[] = {input};
  auto *prep =
      ggml_custom_4d(ctx, GGML_TYPE_I8, (256 + 8) * tokens, 1, 1, 1, args, 1,
                     bitnet_prepare_activation, GGML_N_TASKS_MAX, nullptr);
  ggml_tensor *args2[] = {prep, weights[0], weights[1], weights[2]};
  auto *out =
      ggml_custom_4d(ctx, GGML_TYPE_F32, total, tokens, 1, 1, args2, 4,
                     bitnet_group_projection, GGML_N_TASKS_MAX, nullptr);
  auto *graph = ggml_new_graph(ctx);
  for (auto *node : old)
    ggml_build_forward_expand(graph, node);
  ggml_build_forward_expand(graph, out);
  assert(ggml_graph_compute_with_ctx(ctx, graph, threads) ==
         GGML_STATUS_SUCCESS);
  int offset = 0;
  for (int j = 0; j < 3; j++) {
    for (int t = 0; t < tokens; t++)
      assert(!memcmp((float *)out->data + t * total + offset,
                     (float *)old[j]->data + t * rows[j],
                     rows[j] * sizeof(float)));
    offset += rows[j];
  }
  auto *pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
  for (int i = 0; i < tokens; i++)
    ((int *)pos->data)[i] = 117 + i;
  for (float scale : {1.0f, .7f})
    for (float ext : {0.0f, 1.0f}) {
      auto *in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 128, 5, tokens);
      for (int i = 0; i < 128 * 5 * tokens; i++)
        ((float *)in->data)[i] = (i % 123 - 61) * .01f;
      auto *ones = ggml_fill(
          ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 64, 1, tokens), 1.0f);
      auto *basis = ggml_pad(ctx, ones, 64, 0, 0, 0);
      auto *coeff =
          ggml_rope_ext(ctx, basis, pos, nullptr, 128, GGML_ROPE_TYPE_NEOX,
                        4096, 500000, scale, ext, 1.0f, 32, 1);
      auto *cached = ggml_map_custom2(ctx, in, coeff, bitnet_rotate_cached,
                                      GGML_N_TASKS_MAX, nullptr);
      auto *ref = ggml_rope_ext(ctx, in, pos, nullptr, 128, GGML_ROPE_TYPE_NEOX,
                                4096, 500000, scale, ext, 1.0f, 32, 1);
      graph = ggml_new_graph(ctx);
      ggml_build_forward_expand(graph, cached);
      ggml_build_forward_expand(graph, ref);
      assert(ggml_graph_compute_with_ctx(ctx, graph, threads) ==
             GGML_STATUS_SUCCESS);
      assert(!memcmp(cached->data, ref->data, ggml_nbytes(ref)));
    }
  ggml_free(ctx);
  printf("PASS tokens=%d threads=%d, QKV scales and RoPE scaling exact\n",
         tokens, threads);
}
int main() {
  for (int n : {1, 5, 16})
    for (int t : {1, 3, 8})
      check(n, t);
}
