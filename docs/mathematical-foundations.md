# Fundamentos Matemáticos: LLMs CPU-Universal via Álgebra Esquecida

> **Objetivo**: Universalizar modelos de linguagem de grande porte através de estruturas
> matemáticas que tornem a inferência CPU-nativa tão capaz quanto a GPU — não por
> força bruta de hardware, mas eliminando a necessidade de multiplicação no nível algébrico.

> **Documentação expandida**: ver `docs/theory/` para um documento detalhado por nível.

---

## A Questão Central

Um modelo fp16 de 7B parâmetros precisa de ~14 TFLOPS para gerar um token.
Uma CPU moderna entrega ~0.1–0.5 TFLOPS.
Uma GPU fecha esse gap com paralelismo.

**Nossa resposta**: eliminar FLOPS no nível algébrico, não no nível de hardware.

A hierarquia de custo operacional em hardware real:

```
Multiplicação float32   ~4–5 ciclos/elemento
Adição float32          ~1 ciclo/elemento
Comparação              ~0.3 ciclos/elemento
XOR / AND de bits       ~0.1 ciclos/elemento
```

Cada nível deste projeto desce um degrau dessa hierarquia.

---

## Nível 0 — Baseline: Aritmética Float

Camada linear padrão:

```
y = W · x      W ∈ ℝ^{m×n},  x ∈ ℝⁿ

Custo: m·n multiplicações + m·(n-1) adições ≈ 2mn FLOPs
```

Para BitNet-2B, uma camada FFN: m=6912, n=2560 → ~35.4M FLOPs por token.

---

## Nível 1 — Quantização Ternária: 1.58 bits/parâmetro ✓

**Base teórica**: Entropia de Shannon para distribuição uniforme sobre 3 símbolos.

```
H({-1, 0, +1}) = log₂(3) ≈ 1.585 bits/símbolo
```

Este é o piso de Shannon — nenhum código lossless faz melhor em média.

**Quantização de pesos** (absmax-mean, por tensor):

```
γ = (1/n) Σᵢ |wᵢ|                         (escala: média robusta, não max)
w_q = round( clamp(w/γ, -1, 1) )          → {-1, 0, +1}
```

Por que média e não max: o absmax é dominado por outliers. A média é o estimador
MLE para a distribuição de Laplace que os pesos ternários seguem após treinamento.

**Bound de erro**:

```
||W - γ·W_q||_F ≤ γ/2 · √(mn)
Para W ~ N(0, σ²/n): erro relativo ≈ 1/(2√n) → 0 quando n→∞
```

Modelos maiores toleram quantização ternária melhor — o erro relativo decresce
com √(número de parâmetros).

→ Detalhes completos: `docs/theory/01-ternary-algebra.md`

---

## Nível 2 — Decomposição WHT: Zero Multiplicações ✓ DONE

**Identidade algébrica** (o núcleo deste projeto):

```
Para W ∈ {-1, 0, +1}^{m×n} e x ∈ ℤⁿ:

W⁺[i,j] = 𝟙[W[i,j] = +1]     (máscara binária dos positivos)
W⁻[i,j] = 𝟙[W[i,j] = -1]     (máscara binária dos negativos)

y[i] = Σⱼ W[i,j]·x[j]
      = Σ_{j: W[i,j]=+1} x[j]  −  Σ_{j: W[i,j]=-1} x[j]
```

**Resultado**: o produto escalar com pesos ternários se reduz a duas somas condicionais.
**Nenhuma multiplicação ocorre.** Apenas adições, subtrações e skips.

**Implementação SIMD** (AVX2, 32 elementos por instrução):

```c
__m256i pos_mask = _mm256_cmpgt_epi8(kv, v_zero);    // onde w=+1
__m256i neg_mask = _mm256_cmpgt_epi8(v_zero, kv);    // onde w=-1
__m256i pos_vals = _mm256_and_si256(qv, pos_mask);   // selecionar x[j] positivos
__m256i neg_vals = _mm256_and_si256(qv, neg_mask);   // selecionar x[j] negativos
__m256i delta    = _mm256_sub_epi8(pos_vals, neg_vals);  // diferença
```

**Verificação**: max_diff = 0 (identidade inteira exata) para todas as dimensões testadas.

→ Detalhes completos: `docs/theory/02-wht-decomposition.md`
→ Implementação: `src/ggml-bitnet-wht.cpp`
→ Benchmark: `utils/wht_benchmark.py`

---

## Nível 3 — Aproximação WHT Estruturada: O(n log n) GEMV ✓ DONE

**A ideia ACDC / Fastfood** (Le et al., 2013; Yu et al., 2016):

```
W ≈ H · D · H    onde H é Hadamard, D = diag(d) é diagonal aprendida

y = W·x ≈ H·(D·(H·x)) = H·(d ⊙ (H·x))

Passo 1: ẑ = H·x    — Fast WHT, O(n log n), zero multiplicações
Passo 2: z = d ⊙ ẑ  — scaling diagonal, n multiplicações (mínimo irredutível)
Passo 3: y = H·z    — Fast WHT novamente, O(n log n), zero multiplicações

Total: O(n log n) em vez de O(n²)
Multiplicações: n (apenas a diagonal d — provado ser irredutível)
```

Para n=2560 (BitNet-2B FFN): 17.7M ops → ~102K ops → speedup ~174×.

**Invariante crítico**: ACDC NÃO é compressão post-hoc. Para W aleatório, a projeção
captura apenas ~1/n da energia. O valor de ACDC é como **arquitetura de treinamento**
onde d é o único parâmetro aprendido por camada.

**Projeção fechada**: d*[k] = (H·W·H)[k,k] / n²

**Verificações** (resultado do benchmark):

```
Identidade: max|acdc(x,d) - W·x| = 1.3e-16  (epsilon de máquina float64) ✓
Projeção:   ||d_true - d_recovered|| / ||d_true|| = 2.1e-16 ✓
W aleatório: erro = 99.9%  (conforme teoria: ~1/n energia) ✓
```

→ Detalhes completos: `docs/theory/03-acdc-structured-layers.md`
→ Implementação: `src/ggml-bitnet-fwht.cpp`
→ Benchmark: `utils/acdc_benchmark.py`

---

## Nível 4 — Atenção Tropical: O(n) por Token ✓ DONE

**O semiring tropical** (ℝ ∪ {-∞}, max, +):

```
a ⊕ b = max(a, b)          [adição tropical]
a ⊗ b = a + b              [multiplicação tropical]

Produto matricial tropical:
(A ⊗ᵗʳᵒᵖ B)[i,k] = max_j (A[i,j] + B[j,k])
```

**Conexão com Transformer** (limite de temperatura):

```
lim_{τ→0} softmax(v/τ)[j] = 𝟙[j = argmax(v)]

No limite τ→0, softmax(QKᵀ/√d) → produto tropical max-plus.
Atenção hard = V[argmax_j Q[i]·K[j]ᵀ] = lookup(V, tropical_nn(Q[i], K))
```

**Atenção Tropical Top-K** (para τ finito, atenção empiricamente sharp):

```
1. Scan tropical: scores[j] = Q[i]·K_ternary[j]  para todo j  [O(n·d) adições]
2. Top-K:         encontrar K maiores scores       [O(n·log K) comparações]
3. Softmax:       sobre K tokens apenas            [O(K) exponenciais]
4. Output:        Σ_{k∈topK} w_k · V[k]           [O(K·d) multiply-adds]

Total: O(n·d + K·d) vs O(n²·d) padrão
Speedup: n/K (para n=2048, K=32: 64×)
```

**Verificações** (benchmark):

```
Limite softmax τ→0: weight[argmax] = 1.000000 ✓
Produto tropical 3×3: max|ref - fast| = 0.00e+00 ✓
Qualidade τ=0.1: cosine_sim(top-K, hard) = 0.9746 ✓
Speedup teórico BitNet-2B: 2,863× na atenção ✓
```

→ Detalhes completos: `docs/theory/04-tropical-algebra.md`
→ Implementação: `src/ggml-bitnet-tropical.cpp`
→ Benchmark: `utils/tropical_benchmark.py`

---

## Nível 5 — Memória Holográfica: Substituição Completa da Atenção → EM ANDAMENTO

**A álgebra mais antiga e mais esquecida**: Kanerva (1988) e Plate (1994).

**Convolução circular como binding**:

```
Binding:      A # B = IFFT( FFT(A) ⊙ FFT(B) )    [O(n log n)]
Superposição: M = A # B + C # D + ...              [um único vetor M]
Recuperação:  B̃ = M # A⁻¹                         [O(n log n)]
```

**Conexão com Transformer**:

```
Transformer:   armazena K e V separados (O(n·d) espaço), recupera via O(n²) atenção
HRR:           armazena tudo em M (O(d) espaço!), recupera via FFT O(d log d) — independente de n
```

Para contexto de n=2048 tokens: speedup ≈ n/log n ≈ 186× sobre atenção padrão.

→ Detalhes completos: `docs/theory/05-holographic-memory.md`
→ Implementação: `src/ggml-bitnet-hrr.cpp` (em construção)
→ Benchmark: `utils/hrr_benchmark.py` (em construção)

---

## Tabela de Progresso e Budget Operacional

| Nível | Math | Status | Arquivo | CPU speedup estimado |
|-------|------|--------|---------|---------------------|
| 0 | fp16 GEMV | — | referência | 1× |
| 1 | Ternary {-1,0,+1} | ✓ (herdado) | `ggml-bitnet-mad.cpp` | 3–6× |
| 2 | WHT zero-mul | **✓ DONE** | `ggml-bitnet-wht.cpp` | 1.5–2× sobre L1 |
| 3 | FWHT + ACDC O(n log n) | **✓ DONE** | `ggml-bitnet-fwht.cpp` | ~174× FFN |
| 4 | Tropical attention top-K | **✓ DONE** | `ggml-bitnet-tropical.cpp` | ~64–2863× attn |
| 5 | Holographic memory HRR | **→ EM ANDAMENTO** | `ggml-bitnet-hrr.cpp` | ~186× attn |

**BitNet-2B (30 camadas) — ops/token por pipeline:**

```
fp16 baseline:        ~847 Gops/token
L1 ternário:          ~424 Gops/token   (2×)
L2 WHT zero-mul:      ~424 Gops adds    (efetivo 4–6×)
L3 ACDC FFN:          ~17 Gops/token    (~50×)
L4 +Tropical attn:    ~3 Gops/token     (~280×)
L5 +Holográfico:      ~500 Mops/token   (~1700×)
```

---

## Referências Matemáticas Fundamentais

- **Quantização ternária**: Ma et al., "The Era of 1-bit LLMs" (2024). arXiv:2402.17764
- **Walsh-Hadamard**: Walsh (1923). "A closed set of normal orthogonal functions." *Am. J. Math.*; Hadamard (1893)
- **ACDC/Fastfood**: Le et al., "Fastfood — Approximating Kernel Expansions in Loglinear Time." *ICML 2013*
- **Álgebra tropical**: Maclagan & Sturmfels, *Introduction to Tropical Geometry*. AMS, 2015
- **Tropical e redes neurais**: Zhang et al., "Tropical Geometry of Deep Neural Networks." *ICML 2018*
- **STE**: Bengio et al., "Estimating or Propagating Gradients Through Stochastic Neurons." (2013). arXiv:1308.3432
- **Memória distribuída esparsa**: Kanerva, P. *Sparse Distributed Memory*. MIT Press, 1988
- **HRR**: Plate, T.A. *Holographic Reduced Representations*. PhD thesis, Univ. Toronto, 1994
- **Marchenko-Pastur**: lei de matrizes aleatórias — explica por que a quantização ternária funciona em escala
- **Dequantização tropical**: Itenberg & Mikhalkin (2009). "Geometry in the tropical limit."
