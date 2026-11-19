#include "ggml.h"
#include "ggml-cpu.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { width = 128, layers = 30, ops = 2 * layers, samples = 61 };
static const float freq_base = 500000.0f;

static void rotate_from_cache(struct ggml_tensor * dst, const struct ggml_tensor * src,
        const struct ggml_tensor * coeff, int ith, int nth, void * userdata) {
    (void) userdata;
    const float * x = src->data;
    const float * coefficients = coeff->data;
    float * out = dst->data;
    const int heads = (int) src->ne[1];
    const int rows = (int) ggml_nrows(src);
    const int step = (rows + nth - 1) / nth;
    for (int row = ith * step; row < (ith + 1) * step && row < rows; ++row) {
        const float * c = coefficients + width * (row / heads);
        const float * in = x + width * row;
        float * y = out + width * row;
        for (int i = 0; i < width / 2; ++i) {
            const float c0 = c[2 * i];
            const float s0 = c[2 * i + 1];
            const float x0 = in[i];
            const float x1 = in[i + width / 2];
            y[i] = x0 * c0 - x1 * s0;
            y[i + width / 2] = x0 * s0 + x1 * c0;
        }
    }
}

static void prepare_cache(float * dest, const int32_t * positions, int tokens) {
    const float theta_scale = powf(freq_base, -2.0f / width);
    for (int token = 0; token < tokens; ++token) {
        float theta = (float) positions[token];
        for (int i = 0; i < width; i += 2) {
            dest[token * width + i] = cosf(theta);
            dest[token * width + i + 1] = sinf(theta);
            theta *= theta_scale;
        }
    }
}

static double monotonic_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int compare_double(const void * a, const void * b) {
    const double x = *(const double *) a;
    const double y = *(const double *) b;
    return (x > y) - (x < y);
}

static void check_equal(struct ggml_tensor * old[ops], struct ggml_tensor * cached[ops]) {
    for (int op = 0; op < ops; ++op) {
        const float * a = old[op]->data;
        const float * b = cached[op]->data;
        if (memcmp(a, b, ggml_nbytes(old[op])) != 0) {
            for (int64_t i = 0; i < ggml_nelements(old[op]); ++i) {
                if (memcmp(a + i, b + i, sizeof(float))) {
                    fprintf(stderr, "op %d, float %lld: %a vs %a\n", op, (long long) i, a[i], b[i]);
                    abort();
                }
            }
        }
    }
}

static void run_case(int tokens, int threads) {
    const size_t mem = 4 * 1024 * 1024 + 3 * sizeof(float) * (size_t) layers * (20 + 5) * width * tokens;
    struct ggml_context * ctx = ggml_init((struct ggml_init_params) { mem, NULL, false });
    assert(ctx);
    struct ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, tokens);
    struct ggml_tensor * coeff = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, tokens);
    struct ggml_cgraph * graphs[2] = { ggml_new_graph(ctx), ggml_new_graph(ctx) };
    struct ggml_tensor * old[ops], * cached[ops];
    for (int op = 0; op < ops; ++op) {
        const int heads = op % 2 == 0 ? 20 : 5;
        struct ggml_tensor * in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, heads, tokens);
        float * f = in->data;
        for (int i = 0; i < width * heads * tokens; ++i) {
            f[i] = ((i + 17 * op) % 123 - 61) * 0.011f;
        }
        old[op] = ggml_rope_ext(ctx, in, pos, NULL, width, GGML_ROPE_TYPE_NEOX, 4096,
                              freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
        cached[op] = ggml_map_custom2(ctx, in, coeff, rotate_from_cache, GGML_N_TASKS_MAX, NULL);
        ggml_build_forward_expand(graphs[0], old[op]);
        ggml_build_forward_expand(graphs[1], cached[op]);
    }
    struct ggml_threadpool_params pool_params = ggml_threadpool_params_default(threads);
    struct ggml_threadpool * pool = ggml_threadpool_new(&pool_params);
    assert(pool);
    struct ggml_cplan plans[2] = { ggml_graph_plan(graphs[0], threads, pool), ggml_graph_plan(graphs[1], threads, pool) };
    for (int mode = 0; mode < 2; ++mode) {
        plans[mode].work_data = plans[mode].work_size ? malloc(plans[mode].work_size) : NULL;
        assert(plans[mode].work_data || !plans[mode].work_size);
    }
    double elapsed[2][samples];
    for (int trial = -3; trial < samples; ++trial) {
        int32_t * positions = pos->data;
        for (int i = 0; i < tokens; ++i) {
            positions[i] = (trial + 3) * 23 + i + 1;
        }
        for (int pass = 0; pass < 2; ++pass) {
            const int mode = (trial + 3 + pass) % 2;
            const double start = monotonic_us();
            if (mode == 1) {
                prepare_cache(coeff->data, positions, tokens);
            }
            assert(ggml_graph_compute(graphs[mode], &plans[mode]) == GGML_STATUS_SUCCESS);
            if (trial >= 0) {
                elapsed[mode][trial] = monotonic_us() - start;
            }
        }
        if (trial == -1 || trial == samples - 1) {
            check_equal(old, cached);
        }
    }
    for (int mode = 0; mode < 2; ++mode) {
        qsort(elapsed[mode], samples, sizeof(double), compare_double);
        free(plans[mode].work_data);
    }
    printf("RoPE 30 layers, %d token(s), %d threads: baseline %.2f us, shared %.2f us (%.2fx), outputs bit-identical\n",
           tokens, threads, elapsed[0][samples / 2], elapsed[1][samples / 2], elapsed[0][samples / 2] / elapsed[1][samples / 2]);
    ggml_threadpool_free(pool);
    ggml_free(ctx);
}

int main(void) {
    run_case(1, 8);
    run_case(32, 4);
    return 0;
}
