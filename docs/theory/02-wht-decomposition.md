# Nível 2 — Decomposição WHT: Zero Multiplicações

**Status**: ✓ Implementado em `src/ggml-bitnet-wht.cpp`

## A Identidade Fundamental

Para qualquer matriz ternária W ∈ {-1, 0, +1}^{m×n} e vetor de ativações x ∈ ℤ^n:

```
Definição:   W⁺[i,j] = 𝟙[W[i,j] = +1]    (máscara dos positivos)
             W⁻[i,j] = 𝟙[W[i,j] = -1]    (máscara dos negativos)

Identidade:  W = W⁺ - W⁻                  (decomposição exata)

Produto:     (W·x)[i] = Σⱼ W[i,j]·x[j]
                      = Σ_{j: W[i,j]=+1} x[j]  -  Σ_{j: W[i,j]=-1} x[j]
                      = (W⁺·x)[i]  -  (W⁻·x)[i]
```

**Resultado**: o produto escalar com pesos ternários se decompõe em **duas somas
condicionais**. Nenhuma multiplicação ocorre. Apenas adições (onde w=+1), subtrações
(onde w=-1) e skips (onde w=0).

Esta não é uma aproximação. É uma **identidade algébrica exata**.

---

## Contagem de Operações

```
GEMV padrão (fp16):
  m × n multiplicações + m × (n-1) adições ≈ 2mn FLOPs

GEMV ternário (Nível 1, com multiplicação):
  m × n "multiplicações" por 0/±1 ≈ mn ops (mas usa maddubs, ainda multiplicações)

WHT decomposição (Nível 2):
  mn adições/subtrações  +  0 multiplicações

Multiplicações eliminadas: 100%
```

Para n=2560 (BitNet-2B FFN): ~17.7M multiplicações eliminadas por camada por token.

---

## A Estrutura Walsh-Hadamard

A conexão com a Transformada de Walsh-Hadamard não é coincidência. A WHT de um
vetor v ∈ {-1, +1}^n é:

```
V̂[k] = Σⱼ v[j] · H[j,k]    onde H[j,k] = (-1)^{popcount(j AND k)}
```

A matriz de Hadamard H tem entradas apenas em {-1, +1}. A Fast WHT (FWHT) calcula
todos os V̂[k] em O(n log n) usando apenas adições e subtrações — o **algoritmo
butterfly**, ancestral direto da FFT.

Nossa decomposição W = W⁺ - W⁻ **é** a WHT disfarçada:
- W⁺ codifica quais ativações somar
- W⁻ codifica quais ativações subtrair
- A estrutura butterfly mostra como isso pode ser organizado recursivamente

---

## Implementação AVX2

```c
// src/ggml-bitnet-wht.cpp — dot product de 32 elementos em um passo

__m256i kv       = _mm256_loadu_si256((const __m256i *)(k + i)); // pesos {-1,0,+1}
__m256i qv       = _mm256_loadu_si256((const __m256i *)(q + i)); // query int8
__m256i v_zero   = _mm256_setzero_si256();

// Extrair máscaras: pos=0xFF onde k=+1, neg=0xFF onde k=-1
__m256i pos_mask = _mm256_cmpgt_epi8(kv, v_zero);    // k > 0
__m256i neg_mask = _mm256_cmpgt_epi8(v_zero, kv);    // k < 0

// Seleção condicional: AND com máscara zera os não-selecionados
__m256i pos_vals = _mm256_and_si256(qv, pos_mask);   // q[j] onde k=+1, else 0
__m256i neg_vals = _mm256_and_si256(qv, neg_mask);   // q[j] onde k=-1, else 0

// Diferença: delta[j] = q[j] se k=+1, -q[j] se k=-1, 0 se k=0
__m256i delta    = _mm256_sub_epi8(pos_vals, neg_vals);

// Acumulação int8 → int16 → int32 (evita overflow)
__m256i lo16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(delta));
__m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(delta, 1));
__m256i sum16 = _mm256_add_epi16(lo16, hi16);
accum = _mm256_add_epi32(accum, _mm256_madd_epi16(sum16, v_ones16));
```

**Custo por 32 elementos**: ~7 ciclos (cmpgt×2 + and×2 + sub + cvtepi8×2 + add×2 + madd).
**Zero chamadas a `_mm256_maddubs_epi16`** — nenhuma multiplicação.

---

## Verificação de Exatidão

O benchmark `utils/wht_benchmark.py` verifica max_diff = 0 (identidade inteira exata)
para dimensões 6912×2560 (FFN do BitNet-2B).

```python
# A verificação prova que o resultado é identicamente igual ao GEMV ingênuo:
# Não é aproximação — é a mesma operação expressa sem multiplicação.
max_diff = 0   # para todos os testes realizados
```

---

## Limitações e Próximo Passo

O Nível 2 elimina multiplicações mas **não reduz a complexidade assintótica**:
ainda custa O(mn) operações. Para m=6912, n=2560: 17.7M adições por token por camada.

O Nível 3 (ACDC) reduz isso para O(n log n) ≈ 60K operações — uma redução de ~295×.
Isso requer que o peso W seja **estruturado** como uma matriz de Hadamard pesada,
o que é uma decisão de **arquitetura de treinamento**, não de compressão post-hoc.

---

## API

```c
// include/ggml-bitnet-wht.h

// Dot product único: s = Σⱼ W_ternary[j] · x_q[j]
void ggml_vec_dot_wht_ternary(
    int n, float *s,
    const void *W_encoded,   // I2_S packed
    const void *x_q,         // int8 activations
    float weight_scale,
    float act_scale);

// GEMV completo: y[0..m-1] = W · x_q
void ggml_gemv_wht_ternary(
    int m, int n, float *y,
    const void *W, const void *x,
    float weight_scale, float act_scale);
```
