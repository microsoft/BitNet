# Nível 1 — Álgebra Ternária e Quantização 1.58 bits

## Por que 1.58 bits? Teoria da Informação

A resposta começa com Shannon. A entropia de uma variável aleatória uniforme sobre
três símbolos é:

```
H({-1, 0, +1}) = log₂(3) ≈ 1.58496 bits/símbolo
```

Este é o **piso de Shannon**: o número mínimo de bits necessários para codificar
um trit sem perda de informação. Qualquer código lossless precisa de pelo menos
1.585 bits por peso em média — não existe compressão ternária mais eficiente.

A densidade informacional comparada:

```
fp32  →  32.000 bits/param  (1× referência)
fp16  →  16.000 bits/param  (2×)
int8  →   8.000 bits/param  (4×)
int4  →   4.000 bits/param  (8×)
trit  →   1.585 bits/param  (20.2× sobre fp32)
```

---

## O Anel Ternário Balanceado

O sistema ternário balanceado usa o alfabeto **{T, 0, 1} = {-1, 0, +1}**.

**Operações aritméticas ternárias:**

```
Adição (mod 3 balanceada):
  +1 ⊕ +1 = -1    (overflow)
  +1 ⊕  0 = +1
  +1 ⊕ -1 =  0
   0 ⊕  0 =  0
  -1 ⊕ -1 = +1    (underflow)

Multiplicação (grupo, fechado):
  (+1) × (+1) = +1
  (+1) × (-1) = -1
  (-1) × (-1) = +1
   0   ×  w   =  0    (zero absorvente)
```

O subconjunto {-1, +1} forma o grupo multiplicativo **Z₂ = {±1}**.
O conjunto completo {-1, 0, +1} é isomorfo ao anel **Z/3Z** (inteiros módulo 3),
exceto que usamos a representação balanceada em vez de {0, 1, 2}.

**Propriedade central para redes neurais:**

Para w ∈ {-1, 0, +1} e x ∈ ℝ:
```
w · x = +x    se w = +1
w · x =  0    se w =  0
w · x = -x    se w = -1
```

**Multiplicação (4–5 ciclos) → adição condicional (1 ciclo) → skip (0 ciclos)**

---

## Quantização Ternária: Algoritmo Preciso

### Quantização de pesos (per-tensor, absmax-mean)

```
γ = (1/nm) · Σᵢⱼ |W[i,j]|        (média dos valores absolutos)

W_q[i,j] = round( clamp(W[i,j] / γ, -1, +1) )  → {-1, 0, +1}
```

**Por que a média e não o máximo?**

O absmax é dominado por outliers (valores extremos) que desperdiçam a faixa
dinâmica. A média é o estimador de máxima verossimilhança para a distribuição
de Laplace que os pesos ternários seguem após convergência do treinamento:

```
p(w) = (1/2b) · exp(-|w|/b)    (distribuição de Laplace com escala b)

E[|w|] = b  →  γ = b  →  quantização ótima
```

Empiricamente verificado no BitNet-2B: os pesos converge para uma distribuição
de Laplace com ~45-55% de zeros (sparsidade natural).

**Bound de erro de quantização (norma de Frobenius):**

```
||W - γ · W_q||_F ≤ γ/2 · √(nm)

Para W ~ N(0, σ²/n):  γ ≈ σ·√(2/π)
Erro relativo: ||erro||_F / ||W||_F ≈ 1/(2√n)  → 0 quando n → ∞
```

Isso explica por que **modelos maiores toleram quantização ternária melhor**:
o erro relativo decresce com a raiz quadrada do número de parâmetros por camada.

### Quantização de ativações (per-token, int8)

```
s_token = 127 / max_j |x[j]|        (escala por token, não por tensor)

x_q[j] = round(x[j] · s_token).clamp(-128, 127).to(int8)
```

Per-token (e não per-tensor) porque a distribuição de ativações varia
enormemente token a token — alguns tokens têm outliers localizados que
inflariam a escala global, desperdiçando precisão nos outros tokens.

### GEMM quantizado completo

```
y = (W_q · x_q) · (γ / s_token)      →  resultado em bfloat16
```

O produto escalar W_q · x_q opera inteiramente em int8 (ou int2 para I2_S),
e o reescalonamento (γ/s_token) restaura a grandeza correta.

---

## Codificação I2_S (CPU)

O formato I2_S empacota pesos ternários em 2 bits cada, 4 por byte:

```
Mapeamento: -1 → 00 (0), 0 → 01 (1), +1 → 10 (2)

Byte layout: [w3|w2|w1|w0]  (4 pesos de 2 bits cada)
Bits:         [7:6|5:4|3:2|1:0]
```

**Bloco de quantização (QK):**
- x86_64: QK = 128 elementos por bloco
- ARM64:  QK = 64 elementos por bloco

Um bloco de 128 pesos ocupa 32 bytes (256 bits) — cabe exatamente em um
registrador AVX2 de 256 bits.

---

## Straight-Through Estimator (STE)

A função `round()` tem gradiente zero quase em todo lugar — inútil para backprop.
O **STE** resolve isso na direção do gradiente:

```
Forward:   W_q = round(clamp(W/γ, -1, +1))    →  ternário
Backward:  ∂L/∂W = ∂L/∂W_q · 𝟙[|W/γ| ≤ 1]  (identidade dentro do clamp)
```

Matematicamente: substituímos o subgradiente da função escalão pelo gradiente
da função identidade restrita ao intervalo [-1, +1]. É um estimador **enviesado**
(o gradiente verdadeiro é zero), mas **consistente na prática** — o modelo aprende
a posicionar os pesos na borda das regiões de quantização onde o gradiente flui.

---

## Geometria da Quantização Ternária

### O politopo de quantização

O conjunto {-1, 0, +1}^n é o conjunto dos **vértices inteiros** do hipercubo
[-1,1]^n que possuem entradas em {-1,0,+1}. Durante o treinamento (QAT), os pesos
latentes vivem em ℝ^n e são projetados sobre este conjunto discreto.

A região de atração de cada configuração ternária forma uma **célula de Voronoi**,
e a coleção de todas as células é a decomposição de Delaunay do reticulado
Z^n ∩ [-1,1]^n.

### Esparsidade como regularização implícita

A fração de zeros tipicamente converge para 45–55% após treinamento. Isso age como
regularização L₀ implícita:

```
||W_q||₀ = #{i,j : W_q[i,j] ≠ 0}    (número de parâmetros não-nulos)
```

Essa esparsidade reduz adicionalmente o custo computacional: para 50% de zeros,
metade dos GEMV condicionais são skips — custo efetivo 0.

### Representação de funções ternárias

O espaço de todas as redes neurais ternárias de arquitetura fixa é finito e discreto.
Mas o espaço de **funções** realizáveis (input → output) é contínuo (pela composição
com as ativações não-lineares). Isso cria uma **estratificação** do espaço de funções:
diferentes configurações ternárias podem realizar a mesma função, definindo classes
de equivalência — **órbitas** sob o grupo de simetria da rede (permutações de neurônios,
reescalonamentos compatíveis).

---

## Implementação: Kernel I2_S AVX2

O kernel central em `src/ggml-bitnet-mad.cpp` usa `_mm256_maddubs_epi16`:

```c
// Desempacota 32 pesos de 2 bits → int8 no intervalo {0,1,2}
// Converte para {-1,0,+1} subtraindo 1
// Multiplica por ativações int8 usando maddubs
// Acumula em int32

__m256i weights = unpack_i2s_block(w_packed);   // {0,1,2} → {-1,0,+1}
__m256i acts    = _mm256_loadu_si256(x);
__m256i prod    = _mm256_maddubs_epi16(weights, acts);  // signed × unsigned
accum           = _mm256_add_epi32(accum, madd16(prod));
```

---

## Modelos Suportados

| Modelo | Params | Quant | Sparsidade |
|--------|--------|-------|-----------|
| BitNet-b1.58-2B-4T | 2.4B | ternário | ~50% |
| bitnet_b1_58-large | 0.7B | ternário | ~48% |
| bitnet_b1_58-3B | 3.3B | ternário | ~52% |
| Llama3-8B-1.58 | 8.0B | ternário | ~47% |
| Falcon3/Falcon-E | 1B–10B | ternário | ~50% |
