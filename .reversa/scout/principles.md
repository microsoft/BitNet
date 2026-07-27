# Princípios Fundamentais — BitNet CPU-Universal

> Síntese unificada dos 8 documentos em `./docs/`. Gerado em 2026-06-05 pelo `reversa-scout`.
> Fontes primárias: `docs/theory/00..05-*.md`, `docs/mathematical-foundations.md`, `docs/codegen.md`.
> Objetivo: servir de mapa conceitual para os próximos agentes (archaeologist, detective, forward).

---

## Tese Central

A inferência de LLMs de grande porte no CPU pode atingir a velocidade da GPU não por
paralelismo de hardware, mas por **eliminação algébrica das multiplicações de ponto
flutuante** — descendo a hierarquia de custo operacional até estruturas matemáticas
publicadas há mais de um século e esquecidas pela corrida ao hardware.

```
Multiplicação float32    ~4–5 ciclos/elemento
Adição float32           ~1  ciclo/elemento
Comparação               ~0.3 ciclos/elemento
XOR / AND de bits        ~0.1 ciclos/elemento
```

Cada um dos 5 níveis da pesquisa desce **exatamente um degrau** desta hierarquia,
substituindo a operação cara do nível anterior pela mais barata do nível seguinte.

---

## Os 7 Princípios Transversais

Estes princípios sustentam a coerência interna de todos os 5 níveis. Não são
"features" isoladas; são o **substrato teórico** que torna possível cada nível
e que conecta os níveis entre si.

### P1 — Shannon floor (limite teórico inferior)

> **Onde**: `docs/theory/01-ternary-algebra.md:5-24` · `docs/mathematical-foundations.md:46-74`

A entropia de Shannon para distribuição uniforme sobre 3 símbolos é:
```
H({-1, 0, +1}) = log₂(3) ≈ 1.585 bits/símbolo
```

Este é o **piso de Shannon**: nenhum código lossless pode codificar um trit com
menos de 1.585 bits em média. Qualquer quantização ternária é ótima neste
limite. Densidade informacional comparada:

| Codificação | bits/param | vs fp32 |
|-------------|-----------:|--------:|
| fp32        | 32.000     | 1×      |
| fp16        | 16.000     | 2×      |
| int8        |  8.000     | 4×      |
| int4        |  4.000     | 8×      |
| **trit**    |**1.585**   |**20.2×**|

**Manifestação no código**: packing I2_S (2 bits/peso, 4 por byte) com QK=128 (x86) /
QK=64 (ARM). O trit é codificado com 2 bits brutos (4 valores possíveis em 2 bits
são 0,1,2,3, dos quais 0,1,2 são usados). O "desperdício" de 2→1.585 é o
overhead do packing binário; seria eliminado com packing ternário dedicado.

### P2 — Identidade algébrica, não aproximação

> **Onde**: `docs/theory/00-index.md:44-72` (relação entre níveis) ·
> `docs/theory/02-wht-decomposition.md:6-26` (decomposição exata) ·
> `docs/theory/03-acdc-structured-layers.md:127-152` (projeção fechada)

Cada nível não é uma aproximação heurística. É uma **identidade algébrica exata**,
demonstrável formalmente e verificada empiricamente com `max_diff = 0` (ou epsilon
de máquina `~1e-16`):

- L1: `w · x ∈ {−x, 0, +x}` é exato por construção (3 casos do `clamp`/`round`)
- L2: `(W⁺ − W⁻) · x = W · x` é decomposição literal — verificada em `utils/wht_benchmark.py`
- L3: `H · D · H · x` = FWHT exato da matriz estruturada — `max|acdc − W·x| = 1.3e-16`
- L4: `softmax(v/τ) → δ[argmax]` no limite τ→0 é prova com `lim` matemático — verificado em `utils/tropical_benchmark.py`
- L5: `a ⊛ b = IFFT(FFT(a) ⊙ FFT(b))` é o Teorema da Convolução Circular

**Consequência operacional**: nenhum nível introduz erro numérico que precise ser
treinado, compensado ou documentado como "perda de qualidade". O resultado
bitnet_wht ≡ bitnet_mad no mesmo bit.

### P3 — Hierarquia de custo (descida algébrica de custo)

> **Onde**: `docs/theory/00-index.md:10-28` (tabela) · `docs/mathematical-foundations.md:18-28`

Cada nível troca uma operação cara por uma mais barata, **mantendo o resultado
idêntico** (por P2):

| Nível | Operação eliminada | Substituída por | Ganho |
|-------|-------------------|-----------------|-------|
| L1 | Float weights (4 B/param) | Trit packing (2 bits/param) | 16× memória |
| L2 | Multiplicação inteira (5c) | Adição/subtração (1c) | ~5× compute |
| L3 | O(mn) GEMV (n² ops) | O(n log n) FWHT | ~174× FFN |
| L4 | Exponenciais + scan O(n²) | Comparações + top-K | ~2863× atenção |
| L5 | O(n²) atenção inteira | FFT O(d log d) | ~186× atenção |

**Invariante crucial**: o ganho vem **da álgebra**, não do hardware. Mudar de CPU
para GPU não desfaz a vantagem — a GPU também paga mul caro, e as instruções
`_mm256_cmpgt_epi8` (cmp) são apenas ~3× mais rápidas que `_mm256_maddubs_epi16`
(mul) em hardware x86 moderno. A vantagem é arquitetural, não implementacional.

### P4 — Mínimo irredutível (piso de complexidade)

> **Onde**: `docs/theory/03-acdc-structured-layers.md:65-87` (prova) ·
> `docs/mathematical-foundations.md:127-132` (orçamento)

Toda redução algébrica tem um **piso irredutível** abaixo do qual é impossível
descer sem perder expressividade. ACDC prova que:

```
W = H · D · H    com D = diag(d) ∈ ℝⁿ

Transformação x ↦ H·(d⊙(H·x)) é linear em x
Dimensão do espaço de tais transformações = n (uma por componente de d)
Qualquer base deste espaço requer n coeficientes
Representar esses coeficientes requer ≥ n multiplicações
```

Logo, ACDC **não pode** ter menos de `n` multiplicações. A diagonal `d` é o
único grau de liberdade.

**Manifestação análoga nos outros níveis**:
- L1: precisa de 1.585 bits/peso (P1, Shannon) — packing pode desperdiçar mas não comprimir mais
- L2: precisa de 2 adições por peso (uma para `W⁺`, uma para `W⁻`) — não dá para fazer com 1
- L4: precisa de pelo menos `n·d` comparações (linear scan) — top-K é o piso abaixo do `n²` softmax
- L5: precisa de pelo menos `d log d` ops para o binding (FFT) — abaixo disso não há algoritmo

**Quem ignora este princípio erra**: tentar comprimir ACDC post-hoc resulta em
~1/n energia capturada (P6). Tentar fazer tropical sem o scan O(n·d) perde
precisão empírica. Tentar fazer HRR com `d < 10·N` viola o SNR mínimo.

### P5 — Dequantização tropical (limite contínuo → discreto)

> **Onde**: `docs/theory/04-tropical-algebra.md:56-105` (prova do limite) ·
> `docs/mathematical-foundations.md:154-185` (atenção top-K)

A álgebra usual `(ℝ, ×, +, 0, 1)` é o **limite τ→0** da álgebra tropical
ponderada por temperatura. Em outras palavras, **tropical não é aproximação
discreta do real — o real é o caso limite do tropical**.

Prova formal do limite softmax → argmax:
```
softmax(v/τ)[j] = exp(v[j]/τ) / Σₖ exp(v[k]/τ)

Sem perda de generalidade, v[j*] = max(v):
   = exp((v[j] - v[j*])/τ) / Σₖ exp((v[k] - v[j*])/τ)

Para j ≠ j*: v[j] − v[j*] < 0 → exp → 0 quando τ → 0
Denominador → 1 (só termo j* sobrevive)
Logo: lim_{τ→0} softmax(v/τ)[j] = δ[j = j*] = argmax  ∎
```

Isto conecta a atenção Transformer (real) com o produto tropical max-plus
(discreto) de forma **contínua**, não discreta. Para `τ` finito, ambas as
interpretações coexistem. Na prática, atenção em LLMs treinados é
empiricamente **sharp** (concentrada em poucos tokens) — o que torna a
aproximação tropical top-K válida com `K = 32` (captura 97.5% da atenção
hard, segundo `utils/tropical_benchmark.py`).

**Manifestação arquitetural**: a temperatura `τ` do softmax é um
**parâmetro de interpolação** entre real e tropical. Reduzir `τ` no
fine-tuning gradualmente equivale a fazer annealing para a versão tropical.

### P6 — Estrutura, não compressão (arquitetura de treinamento)

> **Onde**: `docs/theory/03-acdc-structured-layers.md:159-189` (aviso explícito) ·
> `docs/mathematical-foundations.md:134-146` (orçamento)

A intuição tentadora: "comprimir W pré-treinado em W = H·D·H" produz o mesmo
resultado. **Errado** — produz perda catastrófica:

```
Projeção ACDC de W ternário aleatório:
   ||H·D*·H||_F² / ||W||_F² ≈ 1/n

Para n=2560 (BitNet-2B FFN):
   Energia capturada ≈ 0.04%  ← 99.96% perdida
```

Por quê? Matrizes aleatórias `n×n` têm valores singulares distribuídos
uniformemente (lei de Marchenko-Pastur). A representação H·D·H tem apenas
`n` graus de liberdade — ela captura a "projeção diagonal" de W na base de
Hadamard, que é minúscula para W aleatório.

**O valor real de ACDC é como arquitetura de treinamento**:
```
Camada padrão:  W ∈ ℝ^{m×n},  ~mn parâmetros,  mn ops/token
Camada ACDC:    d ∈ ℝⁿ,      ~n  parâmetros,  n log n ops/token

O modelo é TREINADO com d como único parâmetro por camada.
Backward diferenciável: ∂L/∂d[k] = (H·∂L/∂y)[k] · (H·x)[k]
```

Para uma camada BitNet-2B FFN (n=2560):
- Parâmetros padrão: 2560 × 6912 × 1.58 bits ≈ 27.8 Mbits
- Parâmetros ACDC:   2560 × 16 bits = 40 Kbits  →  **700× menos parâmetros**

**Consequência**: a função `acdc_project()` em `ggml-bitnet-fwht.h` é uma
**ferramenta de validação** (mostra que a projeção fechada recupera `d`
exato), não uma ferramenta de produção. Em produção, `d` é aprendido
diretamente via STE + backprop, não ajustado post-hoc.

**Aplicação ao HRR (L5)**: o mesmo princípio se aplica. A "memória holográfica"
só é interessante se o modelo for **treinado** com o regime HRR. Aplicar
HRR post-hoc a um Transformer pré-treinado dá `||recuperado − original|| ≈
(N−1)/√d` (ruído dominante para d < 10N).

### P7 — FFT como cola (estrutura butterfly ancestral)

> **Onde**: `docs/theory/02-wht-decomposition.md:50-64` (WHT ancestral da FFT) ·
> `docs/theory/03-acdc-structured-layers.md:90-126` (butterfly AVX2) ·
> `docs/theory/05-holographic-memory.md:32-50` (convolução circular)

A operação **butterfly** (add/sub par a par) é ancestral comum de toda a
hierarquia:

```
WHT (L2/L3):  butterfly add/sub par a par, base {-1, +1}     → O(n log n)
FFT (L5):     butterfly add/sub com twiddle factor W_N = exp(-2πi/N/N)
              → quando W_N ∈ {±1, ±i}, butterfly é puro add/sub
              → para estágios intermediários, requer 2 muls reais
```

A linha histórica:
- 1867 — Maxwell equações (formulação FFT do calor)
- 1893 — Hadamard matrix (H_n = H_k ⊗ H_2, base de butterfly)
- 1923 — Walsh functions (mesma base, ordenação sequencial)
- 1965 — Cooley-Tukey FFT (algoritmo O(n log n) para FFT complexa)
- 2013 — Fastfood (Le et al.): reaproveita estrutura Hadamard para kernel expansions

**Para L5 (HRR)**: a FFT complexa é **a mesma estrutura** que a WHT real,
estendida com twiddle factors. A escolha de implementar Cooley-Tukey do
zero em `ggml-bitnet-hrr.cpp:81-100` (em vez de chamar FFTW ou similar)
é justificada por:
1. Zero dependências externas
2. Controle fino sobre butterflies complexos AVX2
3. Mesma estrutura algorítmica da WHT em L2/L3

**Consequência arquitetural**: L2, L3, L5 compartilham o **mesmo padrão de
otimização** — butterfly SIMD, in-place, sem alocação no loop quente. Os
3 kernels (`wht.cpp`, `fwht.cpp`, `hrr.cpp`) podem compartilhar um header
de butterflies em uma futura refatoração.

---

## Mapa Conceitual das 5 Álgebras

```
                 ┌─────────────────────────────────────────────┐
                 │  TESE:  Inferência CPU-universal via         │
                 │         descida na hierarquia de custo       │
                 │         (mul → add → cmp → bitwise)          │
                 └─────────────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        ▼                              ▼                              ▼
   ALGÉBRAS CLÁSSICAS            ALGÉBRAS "ESQUECIDAS"          CONEXÃO NEURAL
   ─────────────────             ─────────────────────           ───────────────
   L1: Anel Z/3Z (ternário)      L4: Semiring tropical           L1→3: BitNet paper
       └─ P1 Shannon                 (max, +)                       (Ma 2024)
       └─ Manifestação:               └─ P5 dequantização         L4: geometria
          I2_S packing                   (limite real)               tropical de
                                       └─ Conexão: argmax ≡          redes (Zhang
   L3: Matriz de Hadamard              tropical produto              2018)
       (Walsh 1923,                       (proof of P5)
        Hadamard 1893)                                          L5: Vector
       └─ P4 mínimo irredutível    L5: Convolução circular         Symbolic Arch
       └─ P7 butterfly                  (Kanerva 1988,               (Gayler 2004,
            add/sub como cola            Plate 1994)                 Schlegel 2022)
                                       └─ P7 FFT = butterfly
   L2: Decomposição W=W⁺-W⁻              complexa
       (álgebra de máscaras)          └─ P6 estrutura, não
       └─ P2 identidade exata            compressão (HRR precisa
                                          de treinamento)
```

## Árvore de Dependências Teóricas

```
L1: Quantização ternária
  └─ usa: Shannon (P1)
  └─ produz: alfabeto {-1, 0, +1}, codificação I2_S
  └─ pré-requisito de: L2, L3, L4 (todos assumem pesos ternários)

L2: WHT zero-multiplicação
  └─ depende de: L1
  └─ usa: P2 (identidade), P3 (add > mul), P7 (butterfly)
  └─ produz: GEMV ternário zero-mul
  └─ pré-requisito de: L3 (ACDC), L4 (tropical usa WHT no scan)

L3: FWHT + ACDC O(n log n)
  └─ depende de: L1, L2
  └─ usa: P2, P3, P4 (mínimo irredutível = n muls), P6, P7
  └─ produz: camada linear O(n log n) estruturada
  └─ independente de: L4, L5

L4: Tropical attention (max, +)
  └─ depende de: L1, L2
  └─ usa: P2, P3, P5 (dequantização), P7
  └─ produz: atenção top-K com scan O(n·d) zero-mul
  └─ independente de: L3, L5

L5: HRR — Memória holográfica
  └─ depende de: L1 (K ternárias)
  └─ usa: P2, P3, P4 (FFT é piso), P6, P7
  └─ produz: atenção O(d log d) independente de n
  └─ independente de: L3, L4 (mas conceitualmente complementar)
```

**Observação importante**: L3, L4, L5 são **ortogonais** — podem ser
aplicados em conjunto (e.g., FFN com ACDC + atenção com tropical ou
holográfica). A combinação completa está no orçamento teórico:
~1700× speedup no BitNet-2B (de 847 Gops/token para 500 Mops/token).

---

## Glossário de Referências Cruzadas

| Termo | Aparece em | Significado |
|-------|-----------|-------------|
| **FWHT** | L2, L3 | Fast Walsh-Hadamard Transform; butterfly O(n log n) na base {±1} |
| **ACDC** | L3 | Approximate Circulant/Diagonal/Circulant; W = H·D·H |
| **HRR** | L5 | Holographic Reduced Representations; memória via convolução circular |
| **STE** | L1 | Straight-Through Estimator; backprop para `round()` |
| **QK** | L1 | Quantization block size (128 x86, 64 ARM) |
| **max-diff** | L2, L3, L4 | Erro máximo absoluto entre duas implementações; deve ser 0 |
| **Shannon floor** | L1 | log₂(3) ≈ 1.585 bits/peso, limite lossless |
| **τ → 0** | L4 | Limite de temperatura zero no softmax → argmax |
| **SNR** | L5 | Signal-to-Noise Ratio; `d ≥ 10N` para HRR limpo |
| **noise floor** | L5 | `||recuperado − original|| ≈ (N−1)/√d` para HRR não-limpo |
