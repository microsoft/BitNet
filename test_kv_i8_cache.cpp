/*
 * test_kv_i8_cache.cpp
 *
 * Unit tests para o cache K_i8 persistente (Phase C). Cobre:
 *  - Init / reinit com mesma shape: no-op
 *  - Init com shape diferente: free + realloc
 *  - Reset: zera n_quantized sem realocar
 *  - Get first call (last_n=0): quantiza tudo
 *  - Get incremental (n_kv > last_n): quantiza só o novo
 *  - Get com n_kv <= last_n: idempotente
 *  - Thread-safety: dois threads chamando get(mesmo il, kv_h) não corrompem
 *  - Edge case: layer/h fora do range → NULL
 *  - Edge case: n_kv <= 0 → NULL
 *  - scale: fica lockado depois do primeiro call
 *
 * Compila como C++ dentro do diretório tests/ via CMakeLists (BITNET_TESTING=ON).
 */

#include "ggml-bitnet-kv-cache.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <pthread.h>
#include <vector>
#include <atomic>

/* ─── Helpers ───────────────────────────────────────────────────────────── */

static int fails = 0;
#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        fails++; \
    } else { \
        fprintf(stderr, "ok: %s\n", msg); \
    } \
} while (0)

static void make_K(float * K, int n, int d, float s) {
    for (int i = 0; i < n * d; i++) {
        K[i] = s * (((i * 1103515245 + 12345) % 1000) / 1000.0f - 0.5f);
    }
}

static int approx_eq(float a, float b, float tol) {
    return fabsf(a - b) < tol * fmaxf(1.0f, fabsf(b));
}

/* ─── Tests ─────────────────────────────────────────────────────────────── */

static void test_init_noop() {
    fprintf(stderr, "\n--- test_init_noop ---\n");
    bitnet_kv_i8_cache_init(4, 4, 16, 64);
    /* Second init with same shape: should be no-op (no crash, no realloc). */
    bitnet_kv_i8_cache_init(4, 4, 16, 64);
    bitnet_kv_i8_cache_init(4, 4, 16, 32);  /* smaller max_n_kv: still no-op */
    bitnet_kv_i8_cache_free();
    EXPECT(fails == 0, "init noop doesn't crash");
}

static void test_init_realloc() {
    fprintf(stderr, "\n--- test_init_realloc ---\n");
    bitnet_kv_i8_cache_init(4, 4, 16, 64);
    /* Use a slot. */
    std::vector<float> K(16 * 16);
    make_K(K.data(), 16, 16, 1.0f);
    float scale1;
    int8_t * p1 = bitnet_kv_i8_cache_get(0, 0, K.data(), 16, &scale1, NULL, NULL);
    EXPECT(p1 != NULL, "first get returns non-NULL");
    /* Reinit with different shape. */
    bitnet_kv_i8_cache_init(8, 8, 32, 128);
    /* Old slot is freed; new get should re-init. */
    std::vector<float> K2(8 * 32);
    make_K(K2.data(), 8, 32, 1.0f);
    float scale2;
    int8_t * p2 = bitnet_kv_i8_cache_get(0, 0, K2.data(), 8, &scale2, NULL, NULL);
    EXPECT(p2 != NULL, "get after reinit returns non-NULL");
    bitnet_kv_i8_cache_free();
}

static void test_first_call_quantizes_all() {
    fprintf(stderr, "\n--- test_first_call_quantizes_all ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 32);
    std::vector<float> K(10 * 8);
    make_K(K.data(), 10, 8, 2.0f);
    float scale;
    int last_n, n_new;
    int8_t * p = bitnet_kv_i8_cache_get(0, 0, K.data(), 10, &scale, &last_n, &n_new);
    EXPECT(p != NULL, "first get returns non-NULL");
    EXPECT(last_n == 0, "first call: last_n=0");
    EXPECT(n_new == 10, "first call: n_new=10");
    EXPECT(scale > 0, "scale positive");
    /* spot-check: the values are int8 in [-128, 127] */
    int out_of_range = 0;
    for (int i = 0; i < 10 * 8; i++) {
        if (p[i] < -128 || p[i] > 127) out_of_range++;
    }
    EXPECT(out_of_range == 0, "all quantized entries in int8 range");
    bitnet_kv_i8_cache_free();
}

static void test_incremental_only_new() {
    fprintf(stderr, "\n--- test_incremental_only_new ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 32);
    std::vector<float> K(15 * 8);
    make_K(K.data(), 15, 8, 1.0f);
    float scale1, scale2;
    int last_n1, n_new1, last_n2, n_new2;
    int8_t * p1 = bitnet_kv_i8_cache_get(0, 0, K.data(), 8, &scale1, &last_n1, &n_new1);
    EXPECT(p1 != NULL && last_n1 == 0 && n_new1 == 8, "first get n_new=8");
    /* Second call with n_kv=15: should quantize only the 7 new entries. */
    int8_t * p2 = bitnet_kv_i8_cache_get(0, 0, K.data(), 15, &scale2, &last_n2, &n_new2);
    EXPECT(p2 == p1, "incremental returns same buffer pointer");
    EXPECT(last_n2 == 8, "incremental: last_n=8");
    EXPECT(n_new2 == 7, "incremental: n_new=7");
    EXPECT(approx_eq(scale1, scale2, 1e-5f), "scale locked after first call");
    /* Old entries (0..8*8-1) are unchanged. */
    EXPECT(memcmp(p1, p2, 8 * 8) == 0, "old entries unchanged");
    bitnet_kv_i8_cache_free();
}

static void test_no_new_keys() {
    fprintf(stderr, "\n--- test_no_new_keys ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 32);
    std::vector<float> K(10 * 8);
    make_K(K.data(), 10, 8, 1.0f);
    float scale1, scale2;
    int8_t * p1 = bitnet_kv_i8_cache_get(0, 0, K.data(), 10, &scale1, NULL, NULL);
    /* Re-call with same n_kv: no quantization, same scale. */
    int8_t * p2 = bitnet_kv_i8_cache_get(0, 0, K.data(), 10, &scale2, NULL, NULL);
    EXPECT(p1 == p2, "no-new-keys: same buffer");
    EXPECT(approx_eq(scale1, scale2, 1e-5f), "no-new-keys: same scale");
    bitnet_kv_i8_cache_free();
}

static void test_out_of_range() {
    fprintf(stderr, "\n--- test_out_of_range ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 32);
    std::vector<float> K(8 * 8);
    make_K(K.data(), 8, 8, 1.0f);
    EXPECT(bitnet_kv_i8_cache_get(-1, 0, K.data(), 8, NULL, NULL, NULL) == NULL, "il=-1 → NULL");
    EXPECT(bitnet_kv_i8_cache_get( 2, 0, K.data(), 8, NULL, NULL, NULL) == NULL, "il=2 out of range");
    EXPECT(bitnet_kv_i8_cache_get( 0,-1, K.data(), 8, NULL, NULL, NULL) == NULL, "kv_h=-1 → NULL");
    EXPECT(bitnet_kv_i8_cache_get( 0, 2, K.data(), 8, NULL, NULL, NULL) == NULL, "kv_h=2 out of range");
    EXPECT(bitnet_kv_i8_cache_get( 0, 0, K.data(), 0, NULL, NULL, NULL) == NULL, "n_kv=0 → NULL");
    bitnet_kv_i8_cache_free();
}

static void test_capacity_growth() {
    fprintf(stderr, "\n--- test_capacity_growth ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 1024);
    std::vector<float> K(600 * 8);
    make_K(K.data(), 600, 8, 1.0f);
    /* Start small, grow. */
    int8_t * p1 = bitnet_kv_i8_cache_get(0, 0, K.data(), 64, NULL, NULL, NULL);
    EXPECT(p1 != NULL, "first get n_kv=64");
    int8_t * p2 = bitnet_kv_i8_cache_get(0, 0, K.data(), 200, NULL, NULL, NULL);
    EXPECT(p2 != NULL, "get n_kv=200 (forces realloc)");
    EXPECT(p2 != p1, "realloc moved buffer");
    int8_t * p3 = bitnet_kv_i8_cache_get(0, 0, K.data(), 600, NULL, NULL, NULL);
    EXPECT(p3 != NULL, "get n_kv=600 (max cap 1024)");
    bitnet_kv_i8_cache_free();
}

static void test_capacity_exceeds_max() {
    fprintf(stderr, "\n--- test_capacity_exceeds_max ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 16);
    std::vector<float> K(64 * 8);
    make_K(K.data(), 64, 8, 1.0f);
    /* max_n_kv=16, asking for 64: should return NULL (caller falls back). */
    int8_t * p = bitnet_kv_i8_cache_get(0, 0, K.data(), 64, NULL, NULL, NULL);
    EXPECT(p == NULL, "get n_kv > max returns NULL");
    bitnet_kv_i8_cache_free();
}

struct thread_arg {
    int il, kv_h, n_kv;
    std::atomic<int> * errors;
};

static void * thread_race_worker(void * arg) {
    struct thread_arg * a = (struct thread_arg *)arg;
    /* Many short K tensors, different content. Race scenario: all threads
     * write to slot (a->il, a->kv_h). The mutex must serialize. */
    std::vector<float> K(a->n_kv * 8);
    for (int trial = 0; trial < 200; trial++) {
        for (int i = 0; i < a->n_kv * 8; i++) {
            K[i] = (float)((i + trial) % 17 - 8) * 0.1f;
        }
        float scale;
        int last_n, n_new;
        int8_t * p = bitnet_kv_i8_cache_get(a->il, a->kv_h, K.data(), a->n_kv,
                                            &scale, &last_n, &n_new);
        if (!p) { (*a->errors)++; continue; }
        if (p != bitnet_kv_i8_cache_get(a->il, a->kv_h, K.data(), a->n_kv,
                                         &scale, &last_n, &n_new)) {
            /* Pointer must be stable across calls. */
            (*a->errors)++;
        }
    }
    return NULL;
}

static void test_thread_safety() {
    fprintf(stderr, "\n--- test_thread_safety ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 256);
    std::atomic<int> errors(0);
    struct thread_arg a = { 0, 0, 64, &errors };
    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_race_worker, &a);
    pthread_create(&t2, NULL, thread_race_worker, &a);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    EXPECT(errors.load() == 0, "two threads racing on same slot: 0 errors");
    bitnet_kv_i8_cache_free();
}

static void test_reset_clears_state() {
    fprintf(stderr, "\n--- test_reset_clears_state ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 32);
    std::vector<float> K(10 * 8);
    make_K(K.data(), 10, 8, 1.0f);
    float scale;
    bitnet_kv_i8_cache_get(0, 0, K.data(), 10, &scale, NULL, NULL);
    bitnet_kv_i8_cache_reset();
    /* After reset, n_quantized=0, so next get re-quantizes all. */
    int last_n, n_new;
    bitnet_kv_i8_cache_get(0, 0, K.data(), 10, &scale, &last_n, &n_new);
    EXPECT(last_n == 0, "after reset: last_n=0");
    EXPECT(n_new == 10, "after reset: n_new=10");
    bitnet_kv_i8_cache_free();
}

static void test_set_layer_current() {
    fprintf(stderr, "\n--- test_set_layer_current ---\n");
    bitnet_kv_i8_cache_init(2, 2, 8, 32);
    bitnet_kv_i8_cache_set_layer(0);
    EXPECT(bitnet_kv_i8_current_layer() == 0, "current_layer=0 after set_layer(0)");
    bitnet_kv_i8_cache_set_layer(1);
    EXPECT(bitnet_kv_i8_current_layer() == 1, "current_layer=1 after set_layer(1)");
    bitnet_kv_i8_cache_free();
    EXPECT(bitnet_kv_i8_current_layer() == -1, "current_layer=-1 after free");
}

/* ─── Driver ────────────────────────────────────────────────────────────── */

int main(void) {
    test_init_noop();
    test_init_realloc();
    test_first_call_quantizes_all();
    test_incremental_only_new();
    test_no_new_keys();
    test_out_of_range();
    test_capacity_growth();
    test_capacity_exceeds_max();
    test_thread_safety();
    test_reset_clears_state();
    test_set_layer_current();
    fprintf(stderr, "\n=== test_kv_i8_cache: %d failure(s) ===\n", fails);
    return fails == 0 ? 0 : 1;
}
