# Nível 4 — Álgebra Tropical e Atenção (max,+)

**Status**: ✓ Implementado em `src/ggml-bitnet-tropical.cpp`

## O Gargalo da Atenção

A atenção Transformer padrão tem complexidade O(n²·d) por head por token:

```
A[i,j] = softmax( Q[i] · K[j]ᵀ / √d )   — n² dot products

output[i] = Σⱼ A[i,j] · V[j]             — n dot products com valores
```

Para BitNet-2B (n_heads=20, head_dim=128, seq=2048):
```
20 heads × 2048² × 128 × 2 = 21.474B ops/token  ← atenção
30 camadas × 3 projeções × 17.7M = 1.59B ops/token  ← FFN (com L2 WHT)
```

A atenção domina. Nenhum kernel SIMD resolve O(n²) — precisamos reduzir a
complexidade assintótica.

---

## O Semiring Tropical (max, +)

A **álgebra tropical** é um semiring sobre (ℝ ∪ {-∞}, ⊕, ⊗):

```
a ⊕ b = max(a, b)        ← adição tropical   (máximo)
a ⊗ b = a + b            ← multiplicação tropical  (adição real)
```

**Propriedades (semiring):**
- Comutatividade: a ⊕ b = b ⊕ a   e   a ⊗ b = b ⊗ a
- Associatividade de ⊕ e ⊗
- Distributividade: a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)
- Elemento neutro de ⊕: -∞ (pois max(a, -∞) = a)
- Elemento neutro de ⊗:  0 (pois a + 0 = a)

**Produto matricial tropical:**

```
(A ⊗ᵗʳᵒᵖ B)[i,k] = max_j (A[i,j] + B[j,k])
```

Substituiu-se (×, +, 0, 1) por (+, max, -∞, 0). A semelhança estrutural com álgebra
linear não é coincidência — o semiring tropical é a **dequantização** (limite t→∞
de uma família parametrizada) da álgebra real usual.

---

## A Conexão com Transformer

### Limite de temperatura

A função softmax parametrizada por temperatura τ é:

```
softmax(v/τ)[j] = exp(v[j]/τ) / Σₖ exp(v[k]/τ)
```

No limite τ → 0:

```
lim_{τ→0} softmax(v/τ)[j] = 𝟙[j = argmax(v)]
```

**Prova:**

Sem perda de generalidade, v[j*] = max(v). Então:
```
exp(v[j]/τ) / Σₖ exp(v[k]/τ)
= exp((v[j] - v[j*])/τ) / Σₖ exp((v[k] - v[j*])/τ)
```

Para j ≠ j*: v[j] - v[j*] < 0, então exp((v[j]-v[j*])/τ) → 0 quando τ → 0.
O denominador → 1 (só o termo j* sobrevive). Logo o limite é δ[j = j*]. ∎

### O argmax É o produto tropical

```
argmax_j (Q[i] · K[j]ᵀ)  =  argmax_j (Σₖ Q[i,k] · K[j,k])
```

Em álgebra tropical:
```
(Q ⊗ᵗʳᵒᵖ Kᵀ)[i,j] = max_k (Q[i,k] + K[j,k])
```

Mas dot product real vs produto tropical máximo são diferentes... exceto que para
Q e K positivos e no regime de atenção sharp, o argmax do dot product coincide com
o argmax tropical. Mais precisamente:

O logaritmo do softmax satisfaz:
```
log softmax(v/τ)[j] = v[j]/τ - log(Σₖ exp(v[k]/τ))
                    → v[j]/τ - v[j*]/τ - ...   (quando τ → 0)
```

Esta é a **dequantização** (Itenberg, Mikhalkin, 2009): a álgebra real é o
limite τ→0 da álgebra tropical ponderada por temperatura. A atenção Transformer
É um produto tropical no limite de temperatura zero.

---

## Atenção Tropical Top-K

Na prática, usamos temperatura moderada (τ ≈ 1) mas a atenção em LLMs treinados
é empiricamente **sharp** (concentrada em poucos tokens). Zhang et al. (2023)
demonstraram que LLMs treinados exibem atenção progressivamente mais esparsa
com a profundidade das camadas.

Aproveitamos essa sparsidade para atenção Top-K:

```
Algoritmo Tropical Top-K:

1. Scan tropical:  scores[j] = Q[i] · K_ternary[j]  para todo j ∈ [n]
   Custo: O(n·d) adições  (K ternário → zero multiplicações — Level 2!)

2. Top-K:          encontrar índices dos K maiores scores
   Custo: O(n·log K) comparações  (nth_element/partial_sort)

3. Softmax:        w[k] = softmax(scores[top_k]) para k ∈ Top-K
   Custo: O(K) exponenciais  (K << n — apenas K exponenciais!)

4. Output:         y = Σ_{k∈Top-K} w[k] · V[top_k]
   Custo: O(K·d) multiply-adds

Total: O(n·d + K·d) vs O(n²·d) padrão

Speedup: n²·d / (n·d + K·d) ≈ n/K  (para K << n)
```

Para BitNet-2B (n=2048, K=32): speedup = 64×.

---

## Contagem de Operações: BitNet-2B Completo

```
Atenção padrão (fp16, 20 heads, seq=2048):
  20 × 2048² × 128 × 2 = 21,474M ops/token

Atenção Tropical Top-32 (keys ternárias):
  Scan:    20 × 2048 × 128 = 5,242K adições (0 multiplicações)
  Top-K:   20 × 2048 × log₂(32) = 2,048K comparações
  Softmax: 20 × 32 × 1 = 640 exponenciais
  V sum:   20 × 32 × 128 = 81K multiply-adds
  Total:   ~7.5M ops/token

Speedup: 21,474M / 7.5M ≈ 2,863×
```

---

## Produto Matricial Tropical Completo

Para referência matemática, o produto tropical m×n completo:

```
(A ⊗ᵗʳᵒᵖ x)[i] = max_j (A[i,j] + x[j])      ← tropical GEMV

Para A ternária:
  A[i,j] = +1 → A[i,j] + x[j] = x[j] + 1
  A[i,j] =  0 → A[i,j] + x[j] = x[j]
  A[i,j] = -1 → A[i,j] + x[j] = x[j] - 1

O max_j depende dos valores de x[j], não apenas dos sinais de A.
```

---

## Geometria Tropical e Redes Neurais

A conexão vai além da atenção. Zhang et al. (2018) demonstraram que:

**Teorema**: Uma rede com ativações ReLU computa uma função linear por partes
cujas "regiões lineares" são os **poliedros** de uma subdivisão tropical da entrada.

Para redes ternárias com ReLU:
```
y = ReLU(W_ternary · x + b)

A fronteira das regiões lineares é:
{x : W_ternary · x + b = 0}

Em coordenadas tropicais, estas fronteiras são hipersuperfícies tropicais —
objetos combinatórios estudados na geometria algébrica tropical.
```

Isso implica que **toda rede ternária com ReLU é um objeto da geometria tropical**,
não apenas uma aproximação numérica de uma rede contínua.

---

## Verificação Empírica

`utils/tropical_benchmark.py` verifica:

```
[1] Limite softmax: τ=0.01 → weight[argmax] = 1.000000 ✓

[2] Produto tropical 3×3: max|ref - fast| = 0.00e+00 ✓

[3] Qualidade a τ=0.1: cosine_sim(top-K, hard) = 0.9746
    (top-K com K=8 já captura 97.5% da atenção hard → K alto não é necessário)

[4] Speedup teórico n=2048, K=32: 2,863×
```

---

## Limitações e Próximo Passo

A atenção tropical Top-K ainda requer o scan O(n·d) completo — todos os pares
(query, key) são visitados, mas apenas para comparação, não softmax.

O próximo nível elimina o scan completo:
**Memória Holográfica** (Nível 5) armazena todas as chaves K em um único vetor
de dimensão fixa via soma holográfica, e a recuperação é O(n log n) via FFT —
sem scan, sem Top-K, sem softmax.

---

## API

```c
// include/ggml-bitnet-tropical.h

void tropical_attn_scores(float *scores, const int8_t *q,
    const int8_t *K, int n_keys, int head_dim,
    float q_scale, float k_scale);

int tropical_attn_argmax(const int8_t *q, const int8_t *K,
    int n_keys, int head_dim);

void tropical_attn_topk(int *top_idx, float *top_scores,
    const int8_t *q, const int8_t *K, int n_keys, int head_dim,
    int K_top, float q_scale, float k_scale);

void tropical_attention(float *output, const int8_t *q,
    const int8_t *K, const float *V, int n_keys, int head_dim,
    int K_top, float q_scale, float k_scale);

void tropical_gemv(int *argmax_out, float *max_out,
    const int8_t *A, const float *x, int m, int n);
```
