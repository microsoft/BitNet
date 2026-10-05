# Nível 5 — Memória Holográfica: Representações Holográficas Reduzidas

**Status**: → Em andamento — `src/ggml-bitnet-hrr.cpp` (implementação ativa)

## A Álgebra Esquecida: Kanerva (1988) e Plate (1994)

Pentti Kanerva publicou *Sparse Distributed Memory* em 1988 — dez anos antes dos
Transformers. Ele propunha um modelo de memória associativa de alta dimensão onde:

- Endereços são vetores binários aleatórios de alta dimensão (n ≥ 1000)
- A "distância" entre endereços é a distância de Hamming (XOR + popcount)
- Armazenamento e recuperação são operações sobre vetores inteiros

Tony Plate (1994) formalizou as **Holographic Reduced Representations (HRR)**,
introduzindo a **convolução circular** como operação de binding:

```
A # B = IFFT( FFT(A) ⊙ FFT(B) )    ← binding (associação)
M = A # B + C # D + E # F + ...     ← superposição (múltiplos pares)
B̃ = IFFT( FFT(M) ⊙ conj(FFT(A)) )  ← unbinding (recuperação)
```

A conexão com Transformers: a atenção **É** uma recuperação holográfica aproximada,
onde Q é a "chave de recuperação", K é o "endereço armazenado", e V é o "valor".

---

## Convolução Circular: A Operação Fundamental

Para dois vetores a, b ∈ ℝⁿ:

```
(a ⊛ b)[k] = Σⱼ a[j] · b[(k-j) mod n]       ← convolução circular

Em domínio de frequência (pelo Teorema da Convolução Circular):
  FFT(a ⊛ b) = FFT(a) ⊙ FFT(b)              ← multiplicação elemento a elemento

Logo: a ⊛ b = IFFT( FFT(a) ⊙ FFT(b) )
```

**Custo**: O(n log n) via FFT rápida.

### Propriedades algébricas da convolução circular

```
Comutatividade:   a ⊛ b = b ⊛ a
Associatividade:  (a ⊛ b) ⊛ c = a ⊛ (b ⊛ c)
Identidade:       δ ⊛ a = a  (onde δ[0]=1, δ[k>0]=0)
Inversa:          a⁻¹ = IFFT( 1 / FFT(a) )  (se FFT(a) ≠ 0)
```

A convolução circular torna o espaço ℝⁿ em um **grupo abeliano** sob ⊛
(para vetores de norma unitária com espectro não-nulo — vetores aleatórios
satisfazem isso com probabilidade 1).

---

## Memória Associativa Holográfica

### Armazenamento: superposição de bindings

Dado um dicionário de pares {(k₁, v₁), (k₂, v₂), ..., (kₙ, vₙ)}:

```
M = Σᵢ kᵢ ⊛ vᵢ      ← um único vetor M ∈ ℝᵈ armazena N pares
```

Para vetores aleatórios unitários em ℝᵈ com d >> N:
- O ruído de interferência entre pares é O(N/√d)
- Para d=1024 e N=100: SNR ≈ 10 → recuperação perfeita com decodificador simples

### Recuperação: unbinding por pseudo-inversa

```
B̃ = M ⊛ k₁⁻¹ = (Σᵢ kᵢ ⊛ vᵢ) ⊛ k₁⁻¹
   = v₁ + Σ_{i≠1} (kᵢ ⊛ k₁⁻¹) ⊛ vᵢ
   ≈ v₁  (ruído ≈ 0 para kᵢ aleatórios independentes)
```

O erro de recuperação é:
```
||B̃ - v₁|| ≈ (N-1)/√d    (N pares armazenados, d dimensões)
```

Para d=4096, N=64 (contexto típico de LLM): erro ≈ 63/64 ≈ 0.98 — inaceitável.
Mas com d=65536 e N=64: erro ≈ 0.012 — aceitável.

A solução real: usar **projeção iterativa** (Kanerva) ou **limpeza por manifold**
(Frady et al., 2021) para reduzir o ruído para zero em O(log N) iterações.

---

## Conexão com Transformer Attention

### Transformer padrão

```
Q, K, V ∈ ℝ^{n×d}   (n tokens, d dimensões por head)

A = softmax(Q·Kᵀ/√d)      (matriz de atenção n×n — O(n²))
output = A · V              (soma ponderada — O(n²d))
```

### Interpretação holográfica

Cada head de atenção pode ser reinterpretada como:

```
Armazenamento (forward):
  M_head = Σᵢ K[i] ⊛ V[i]   ← bindings de todos os (K, V) do contexto

Recuperação (por query):
  output[q] = M_head ⊛ Q[q]⁻¹   ← unbinding pelo query
```

**Diferença crítica com Transformer**:
- Transformer: armazena K e V separados, recupera via produto interno O(n²)
- HRR: armazena tudo em M (um vetor!), recupera via FFT O(n log n)

O custo de construção do M é O(n log n) — análogo ao "encode" do KV cache.
O custo de recuperação por token é O(d log d) — independente de n!

---

## A Álgebra das Frequências Complexas

### Representação polar em frequência

Para vetores unitários aleatórios a ∈ ℝⁿ, no domínio de Fourier:

```
Â = FFT(a) = {|Â[k]| · exp(iφₖ)}   (amplitude × fase)
```

O binding via convolução circular em domínio de frequência é:

```
FFT(a ⊛ b)[k] = Â[k] · B̂[k]
              = |Â[k]|·|B̂[k]| · exp(i(φₐₖ + φᵦₖ))
```

**A fase da combinação é a soma das fases** — o binding é uma **adição de fases**.

Para vetores de módulo unitário (|Â[k]| = 1 ∀k): a ⊛ é uma rotação de fase pura.
Este é o grupo U(1)ⁿ — o mesmo grupo que aparece no **RoPE** (Rotary Position Embedding)!

A generalização para espaço de Hilbert complexo (dimensão d) dá o **Vector Symbolic Architecture**
de alta capacidade, implementado eficientemente via FFT complexa.

---

## Por que "Holográfico"?

Em holograma óptico:
- A informação de uma imagem 2D é codificada em toda a superfície do holograma
- Cada parte pequena do holograma contém uma versão degradada da imagem inteira
- O dano parcial do holograma degrada a qualidade mas não destrói a informação

Em HRR:
- A informação de N pares (kᵢ, vᵢ) é distribuída em todos os d componentes de M
- Qualquer subconjunto dos componentes de M contém informação sobre todos os pares
- A remoção de componentes de M degrada a qualidade de recuperação uniformemente

Esta propriedade de **distribuição uniforme da informação** é o que torna as HRR
robustas ao ruído e adequadas para hardware com aritmética de baixa precisão
(int8, float16) — os erros de quantização são absorvidos pelo ruído de fundo
da memória holográfica.

---

## Complexidade de Tempo e Espaço

```
Transformers padrão:
  Armazenamento KV cache: O(n·d)  espaço
  Atenção por token:      O(n·d)  tempo (n dot products de tamanho d)
  Complexidade total:     O(n²·d) para n tokens

HRR como substituto de atenção:
  Armazenamento M:        O(d)    espaço (um vetor!)
  Construção M:           O(n·d·log d)  (n FFTs de tamanho d)
  Recuperação por token:  O(d·log d)    (uma FFT de tamanho d + produto)
  Complexidade total:     O(n·d·log d)  → O(n log n) para d constante
```

Speedup sobre Transformer: O(n²) → O(n log n) → speedup ≈ n/log n.
Para n=2048: 2048/11 ≈ 186× na atenção.

---

## Plano de Implementação (Nível 5)

### Fase 1: Primitivas FFT (C++, CPU)

```c
// include/ggml-bitnet-hrr.h (em construção)

// Convolução circular via FFT (binding)
void hrr_bind(float *out, const float *a, const float *b, int d);
// Alias: hrr_bind(M, K[i], V[i], d)

// Unbinding: recuperação de V dado K e M
void hrr_unbind(float *out, const float *M, const float *k_inv, int d);

// Pseudo-inversa para unbinding
void hrr_pseudoinverse(float *k_inv, const float *k, int d);

// Superposição: M += K[i] ⊛ V[i]
void hrr_accumulate(float *M, const float *k, const float *v, int d);

// Limpeza por manifold (reduce noise)
void hrr_cleanup(float *out, const float *noisy, const float **codebook,
                 int n_items, int d, int n_iters);
```

### Fase 2: Integração com atenção ternária

A chave K será quantizada (ternária), mas a memória M será em float32:

```
Para cada token i no contexto:
  k_i = quantize_ternary(K[i])        ← int8, Level 2
  v_i = V[i]                          ← float32
  hrr_accumulate(M, k_i, v_i, d)     ← M += k_i ⊛ v_i  (O(d log d))

Para cada query q:
  k_inv = hrr_pseudoinverse(q)        ← pseudo-inversa (O(d log d))
  v_retrieved = hrr_unbind(M, k_inv) ← recuperação (O(d log d))
```

### Fase 3: Limpeza iterativa

Para melhorar qualidade de recuperação quando N (contexto) é grande:

```
v_approx = M ⊛ q⁻¹                    ← estimativa inicial
Para t = 1..T:
  v_approx = arg_nearest(v_approx, codebook_V)  ← projeta no manifold
  v_approx = M ⊛ q⁻¹ · λ + v_approx · (1-λ)   ← mistura
```

---

## Referências Fundamentais

- Kanerva, P. (1988). *Sparse Distributed Memory*. MIT Press.
- Plate, T.A. (1994). *Holographic Reduced Representations*. PhD thesis, Univ. Toronto.
- Frady, E.P. et al. (2021). "Resonator Networks, 2: Error Statistics and Capacity of the Resonator Network." *Neural Computation.*
- Smolensky, P. (1990). "Tensor product variable binding." *Artificial Intelligence.*
- Gayler, R.W. (2004). "Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience." arXiv.
- Schlegel, K. et al. (2022). "A comparison of vector symbolic architectures." *Artificial Intelligence Review.*
