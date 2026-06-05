# Nível 3 — Camadas ACDC: O(n log n) via Fast Walsh-Hadamard Transform

**Status**: ✓ Implementado em `src/ggml-bitnet-fwht.cpp`

## O Problema com O(n²)

O Nível 2 eliminou multiplicações, mas o custo permanece O(mn) — linear no número
de parâmetros. Para uma camada FFN do BitNet-2B (m=6912, n=2560):

```
17.7M adições por camada por token
30 camadas × 3 projeções cada = 90 camadas FFN
Total FFN: ~1.6B adições por token
```

O Nível 3 reduz cada camada de O(mn) para **O(n log n)** — redução de ~295× para
n=2560.

---

## A Matriz de Hadamard

A matriz de Hadamard H_n (n = 2^k) é definida recursivamente:

```
H₁ = [1]

H_{2k} = H_k ⊗ H₂ = ⎡ H_k   H_k ⎤
                       ⎣ H_k  -H_k ⎦
```

**Propriedades fundamentais:**
1. Todas as entradas em {-1, +1}
2. H_n · H_n^T = n · I_n       (ortonormalidade escalada)
3. H_n⁻¹ = H_n / n             (auto-inversa até escala)
4. Os vetores coluna são mutuamente ortogonais com norma √n

---

## A Camada ACDC

A ideia central (Le et al., 2013; Fastfood) é parametrizar uma camada linear como:

```
W ≈ H · diag(d) · H          d ∈ ℝⁿ  (único vetor de parâmetros)

y = W · x  =  H · (d ⊙ (H · x))
```

Substituindo na definição:
- **Passo 1**: ẑ = H · x     — Fast WHT, O(n log n), **zero multiplicações**
- **Passo 2**: z = d ⊙ ẑ    — scaling diagonal, **n multiplicações** (mínimo irredutível)
- **Passo 3**: y = H · z     — Fast WHT novamente, O(n log n), **zero multiplicações**

**Total**: O(n log n) adições + n multiplicações.

Para n=2560 (próxima potência de 2: 4096):
```
2 × 4096 × log₂(4096) = 2 × 4096 × 12 = 98,304 adições
4096 multiplicações (diagonal d)
Total: ~102K ops  vs  17.7M ops do GEMV padrão → speedup ~174×
```

---

## Por que n multiplicações são o Mínimo Irredutível

A diagonal d é o único "grau de liberdade" da camada ACDC. Matematicamente:

```
W = H · D · H    onde D = diag(d)

H · W · H = H · (H · D · H) · H = n · D · n = n² · D

d = diag(H · W · H) / n²
```

Para recuperar d a partir de W, precisamos da combinação linear H·W·H, que
envolve exatamente n produtos escalares. Não existe parametrização equivalente
com menos de n parâmetros que preserve a expressividade desta classe de funções.

**Prova que as n multiplicações são irredutíveis:**
- A transformação x ↦ H·(d⊙(H·x)) é linear em x
- A dimensão do espaço de tais transformações é n (uma por componente de d)
- Qualquer base deste espaço requer n coeficientes
- Representar esses coeficientes requer pelo menos n multiplicações ∎

---

## O Algoritmo Butterfly (Fast WHT)

O FWHT implementa a multiplicação H·x em O(n log n) usando o padrão butterfly:

```
Para cada estágio s = 0, 1, ..., log₂(n)-1:
  len = 2^s
  Para cada bloco [i, i + 2·len):
    Para j = 0, ..., len-1:
      a = v[i+j]
      b = v[i+j+len]
      v[i+j]     = a + b    ← adição
      v[i+j+len] = a - b    ← subtração
```

**Zero multiplicações em todo o butterfly.**

Para n=4096: log₂(4096) = 12 estágios × 2048 butterfly pairs × 2 ops = 49,152 ops.

### Implementação AVX2

```c
// src/ggml-bitnet-fwht.cpp — butterfly de 8 floats em paralelo

static void butterfly_f32_avx2(float * v, int len, int n) {
    for (int i = 0; i < n; i += 2 * len) {
        float * a = v + i;
        float * b = v + i + len;
        for (int j = 0; j < len; j += 8) {
            __m256 va = _mm256_loadu_ps(a + j);
            __m256 vb = _mm256_loadu_ps(b + j);
            _mm256_storeu_ps(a + j, _mm256_add_ps(va, vb));  // a+b
            _mm256_storeu_ps(b + j, _mm256_sub_ps(va, vb));  // a-b
        }
    }
}
```

8 pares de butterfly por instrução AVX2 — 8× throughput vs escalar.

---

## Projeção: Encontrar o Melhor d para uma Matriz W

Dado um W arbitrário (ternário ou não), encontrar o d que minimiza:

```
min_d  ||W - H·diag(d)·H||_F²

Solução fechada:  d*[k] = (H·W·H)[k,k] / n²
```

**Derivação:**

```
F(d) = ||W - H·D·H||_F² = ||W||_F² - 2·⟨W, H·D·H⟩ + ||H·D·H||_F²

∂F/∂d[k] = -2·(H·W·H)[k,k] + 2·n²·d[k] = 0

d*[k] = (H·W·H)[k,k] / n²   ∎
```

Esta projeção é computada em `acdc_project()`:
1. Aplicar FWHT a cada coluna de W
2. Aplicar FWHT a cada linha do resultado
3. Extrair a diagonal e dividir por n²

---

## ACDC NÃO é Compressão Post-Hoc

Esta é a confusão mais comum. Para W aleatório (ternário), a projeção ACDC
captura apenas ~1/n da energia:

```
||H·D*·H||_F² / ||W||_F²  ≈  1/n

Para n=2560: energia capturada ≈ 0.04%
```

**Por que?** A matriz W aleatória tem seus valores singulares distribuídos
uniformemente (lei de Marchenko-Pastur). A representação H·D·H só tem n
graus de liberdade enquanto uma matriz n×n genérica tem n² — ela captura
apenas a "projeção diagonal" de W na base de Hadamard.

**O valor real de ACDC é como arquitetura de treinamento:**

```
Camada padrão: W ∈ ℝ^{m×n}, ~mn parâmetros → mn ops/token
Camada ACDC:   d ∈ ℝⁿ,      ~n parâmetros  → n log n ops/token

O modelo é TREINADO com d como parâmetro.
O backward é diferenciável:
  ∂L/∂d[k] = (H · ∂L/∂y)[k] · (H · x)[k]
```

Para uma camada BitNet-2B FFN (n=2560):
- Parâmetros padrão: 2560 × 6912 × 1.58 bits ≈ 27.8 Mbits
- Parâmetros ACDC:   2560 × 16 bits = 40 Kbits  →  **700× menos parâmetros**

Para manter capacidade expressiva com ACDC: usar K diagonais por camada
(K blocos WHT empilhados), conectados por uma projeção linear leve.

---

## Benchmark de Verificação

`utils/acdc_benchmark.py` verifica as identidades exatas:

```
[1] Identidade: acdc_forward(x,d) ≡ W_ACDC · x
    max|acdc(x,d) - W·x| = 1.3e-16  (epsilon de máquina float64)
    IDENTIDADE VERIFICADA ✓

[2] Projeção: acdc_project(W) recupera d exatamente
    ||d_true - d_recovered|| / ||d_true|| = 2.1e-16
    RECUPERAÇÃO EXATA ✓

[3] Projeção de W aleatório:
    Erro relativo da melhor projeção ACDC: 99.9%
    Energia capturada: ~0.04%   (≈ 1/n — conforme teoria)
```

---

## API

```c
// include/ggml-bitnet-fwht.h

void fwht_f32(float *v, int n);                    // FWHT in-place
void fwht_i8_to_i32(const int8_t *x, int32_t *out, int n);  // int8 → int32 WHT

void acdc_forward_i8(float *y, const int8_t *x, const float *d, int n);
void acdc_forward_f32(float *y, const float *x, const float *d, int n);
void acdc_gemv(float *y, const int8_t *x, const float *D,
               const float *proj, int m, int n, int K);

void acdc_project(float *d, const int8_t *W, int n);  // melhor projeção
float acdc_error(const int8_t *W, const float *d, int n);
```
