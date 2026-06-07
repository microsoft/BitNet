// test_sparse_attention.cpp
//
// Testes unitários para sparse_attention_float (L4 alternativa de alta performance).
//
// Cobre:
//   1. K_top <= 0: saída zero (degenerate, sem softmax)
//   2. K_top >= n_keys: equivalente a softmax full sobre todos os keys
//   3. Top-1 selection: dot(q, K[i]) máximo determina saída
//   4. Top-K selection: partial_sort pega os K maiores scores
//   5. Float vs referência manual: pequeno d, comparação com implementação
//      ingênua escrita do zero
//
// Compila isolado contra src/ggml-bitnet-tropical.cpp + src/ggml-bitnet-common.cpp
// (mesma estratégia dos outros testes data-driven).
//
// Convenções:
//   - Erros são fatais (return 1)
//   - Saída no padrão "TEST N: <name> ... PASS/FAIL"

#include "ggml-bitnet-tropical.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static int n_fail = 0;
static int n_pass = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, msg); \
        n_fail++; return; \
    } \
} while (0)

#define PASS(name) do { \
    std::printf("TEST %d: %s ... PASS\n", n_pass + n_fail + 1, name); \
    n_pass++; \
} while (0)

static bool approx_eq(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) < tol;
}

static bool vector_approx_eq(const float * a, const float * b, int n, float tol = 1e-4f) {
    for (int i = 0; i < n; i++) {
        if (!approx_eq(a[i], b[i], tol)) return false;
    }
    return true;
}

/* ─── Test 1: K_top <= 0 → output zero ────────────────────────────────────── */
static void test_k_top_zero() {
    const int d = 8;
    const int n_keys = 16;
    std::vector<float> q(d, 0.0f);
    std::vector<float> K(n_keys * d, 0.0f);
    std::vector<float> V(n_keys * d, 1.0f);
    std::vector<float> out(d, 99.0f);  // sentinela: não-zero, deve virar zero

    sparse_attention_float(out.data(), q.data(), K.data(), V.data(),
                           n_keys, d, /*K_top=*/0);

    for (int i = 0; i < d; i++) {
        if (!approx_eq(out[i], 0.0f)) {
            std::fprintf(stderr, "  out[%d] = %f, esperado 0\n", i, out[i]);
            CHECK(false, "K_top=0 deveria zerar output");
        }
    }
    PASS("k_top_zero_returns_zero_output");
}

/* ─── Test 2: K_top >= n_keys → equivalente a full softmax ──────────────── */
static void test_k_top_full() {
    const int d = 4;
    const int n_keys = 4;
    std::vector<float> q = {1.0f, 0.5f, -0.3f, 0.0f};
    std::vector<float> K = {
        1.0f,  0.0f,  0.0f,  0.0f,
        0.0f,  1.0f,  0.0f,  0.0f,
        0.0f,  0.0f,  1.0f,  0.0f,
        0.0f,  0.0f,  0.0f,  1.0f,
    };
    std::vector<float> V = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f,10.0f,11.0f,12.0f,
       13.0f,14.0f,15.0f,16.0f,
    };

    // Referência: full softmax com 1/√d scaling.
    const float inv_sqrt_d = 1.0f / std::sqrt((float)d);
    std::vector<float> scores(n_keys);
    for (int i = 0; i < n_keys; i++) {
        float dot = 0.0f;
        for (int j = 0; j < d; j++) dot += q[j] * K[i * d + j];
        scores[i] = dot * inv_sqrt_d;
    }
    float max_s = *std::max_element(scores.begin(), scores.end());
    std::vector<float> w(n_keys);
    float sum = 0.0f;
    for (int i = 0; i < n_keys; i++) {
        w[i] = std::exp(scores[i] - max_s);
        sum += w[i];
    }
    for (int i = 0; i < n_keys; i++) w[i] /= sum;

    std::vector<float> expected(d, 0.0f);
    for (int i = 0; i < n_keys; i++) {
        for (int j = 0; j < d; j++) expected[j] += w[i] * V[i * d + j];
    }

    std::vector<float> out(d, 0.0f);
    sparse_attention_float(out.data(), q.data(), K.data(), V.data(),
                           n_keys, d, /*K_top=*/n_keys);

    CHECK(vector_approx_eq(out.data(), expected.data(), d),
          "K_top=n_keys deveria equivaler a full softmax");
    PASS("k_top_full_equals_full_softmax");
}

/* ─── Test 3: Top-1 selection — score máximo determina saída ───────────── */
static void test_top1_selection() {
    const int d = 4;
    const int n_keys = 8;
    // q alinhado com K[3]; K[0..2] tem dot ≤ 0, K[4..7] tem dot < K[3]
    std::vector<float> q = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> K(n_keys * d);
    std::vector<float> V(n_keys * d);
    for (int i = 0; i < n_keys; i++) {
        for (int j = 0; j < d; j++) {
            // K[3] = [1,1,1,1] (dot=q·K[3]=4, máximo)
            // K[i] para i≠3 tem dot ≤ 3
            K[i * d + j] = (i == 3) ? 1.0f : (j == 0 ? 0.7f : 0.0f);
            V[i * d + j] = (float)(i * 10 + j);
        }
    }

    std::vector<float> out(d, 0.0f);
    sparse_attention_float(out.data(), q.data(), K.data(), V.data(),
                           n_keys, d, /*K_top=*/1);

    // Com K_top=1, saída é V[3] (único selecionado, softmax de 1 = 1)
    std::vector<float> expected(d);
    for (int j = 0; j < d; j++) expected[j] = V[3 * d + j];  // [30,31,32,33]

    CHECK(vector_approx_eq(out.data(), expected.data(), d),
          "K_top=1 deveria selecionar V[índice_do_max_score]");
    PASS("top1_selection_picks_argmax_score");
}

/* ─── Test 4: Top-K selection — partial_sort pega os K maiores scores ──── */
static void test_topk_partial_sort() {
    const int d = 2;
    const int n_keys = 6;
    // q = [1, 0]. K[i] = [s_i, 0] (segunda dimensão 0 ⇒ dot = s_i).
    // Pontuações: s = [0.1, 0.5, 0.9, 0.3, 0.7, 0.2]
    // Top-2 esperado: índices {2, 4} (scores 0.9, 0.7).
    std::vector<float> q = {1.0f, 0.0f};
    std::vector<float> K = {
        0.1f, 0.0f,
        0.5f, 0.0f,
        0.9f, 0.0f,
        0.3f, 0.0f,
        0.7f, 0.0f,
        0.2f, 0.0f,
    };
    // V[2] = [a,b], V[4] = [c,d]
    std::vector<float> V = {
        0,0, 0,0, 1,2, 0,0, 3,4, 0,0,
    };

    std::vector<float> out(d, 0.0f);
    sparse_attention_float(out.data(), q.data(), K.data(), V.data(),
                           n_keys, d, /*K_top=*/2);

    // Espera: output = softmax(s[2]/√d, s[4]/√d) · [V[2]; V[4]]
    const float inv_sqrt_d = 1.0f / std::sqrt((float)d);
    const float s2 = 0.9f * inv_sqrt_d;
    const float s4 = 0.7f * inv_sqrt_d;
    const float m = std::max(s2, s4);
    const float w2 = std::exp(s2 - m);
    const float w4 = std::exp(s4 - m);
    const float sum = w2 + w4;
    std::vector<float> expected(d);
    expected[0] = (w2 * 1.0f + w4 * 3.0f) / sum;
    expected[1] = (w2 * 2.0f + w4 * 4.0f) / sum;

    CHECK(vector_approx_eq(out.data(), expected.data(), d),
          "K_top=2 deveria selecionar V[2] e V[4] (top scores)");
    PASS("topk_partial_sort_picks_correct_keys");
}

/* ─── Test 5: Float scoring vs implementação de referência ─────────────── */
static void test_vs_reference() {
    const int d = 16;
    const int n_keys = 32;
    const int K_top = 4;

    // Dados pseudo-aleatórios determinísticos (semente fixa)
    std::srand(42);
    std::vector<float> q(d);
    std::vector<float> K(n_keys * d);
    std::vector<float> V(n_keys * d);
    for (int j = 0; j < d; j++) q[j] = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    for (int i = 0; i < n_keys * d; i++) {
        K[i] = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
        V[i] = (std::rand() / (float)RAND_MAX) * 2.0f - 1.0f;
    }

    // Referência: reimplementação ingênua
    std::vector<float> ref(d, 0.0f);
    {
        const float inv_sqrt_d = 1.0f / std::sqrt((float)d);
        std::vector<float> scores(n_keys);
        for (int i = 0; i < n_keys; i++) {
            float dot = 0.0f;
            for (int j = 0; j < d; j++) dot += q[j] * K[i * d + j];
            scores[i] = dot * inv_sqrt_d;
        }
        // partial_sort descendente
        std::vector<int> idx(n_keys);
        for (int i = 0; i < n_keys; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + K_top, idx.end(),
            [&scores](int a, int b){ return scores[a] > scores[b]; });
        // softmax estável
        float max_s = scores[idx[0]];
        for (int k = 1; k < K_top; k++)
            if (scores[idx[k]] > max_s) max_s = scores[idx[k]];
        std::vector<float> w(K_top);
        float sum = 0.0f;
        for (int k = 0; k < K_top; k++) {
            w[k] = std::exp(scores[idx[k]] - max_s);
            sum += w[k];
        }
        for (int k = 0; k < K_top; k++) w[k] /= sum;
        // soma ponderada
        for (int k = 0; k < K_top; k++) {
            for (int j = 0; j < d; j++) ref[j] += w[k] * V[idx[k] * d + j];
        }
    }

    std::vector<float> out(d, 0.0f);
    sparse_attention_float(out.data(), q.data(), K.data(), V.data(),
                           n_keys, d, K_top);

    CHECK(vector_approx_eq(out.data(), ref.data(), d, 1e-3f),
          "sparse_attention_float deveria bater com referência ingênua");
    PASS("matches_manual_reference_implementation");
}

int main() {
    std::printf("=== test_sparse_attention: sparse_attention_float ===\n");
    test_k_top_zero();
    test_k_top_full();
    test_top1_selection();
    test_topk_partial_sort();
    test_vs_reference();
    std::printf("\n%d/%d PASS\n", n_pass, n_pass + n_fail);
    return n_fail == 0 ? 0 : 1;
}
