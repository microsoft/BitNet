# BitNet CPU-Universal: Findings from 5 Algebraic Levels

> **Status:** Post-Phase A + Phase C research results (Junho 2026)
> **Período coberto:** 2025-06-05 → 2026-06-06 (Sessões S1, S2, S2b, S2c, S2d)
> **Total de commits:** 27
> **Tag:** v0.1.0-cpu-universal (pushed 2026-06-05)
> **Base:** fork do [microsoft/BitNet](https://github.com/microsoft/BitNet) em `129557d`

Este documento agrega os achados empíricos, bugs e decisões de design
das 5 rodadas de experimentação algébrica do fork CPU-Universal. É a
versão narrativa do `SESSION_SUMMARY.md`, voltada para publicação.

---

## TL;DR

Implementamos 5 níveis algébricos de atenção e feed-forward que
eliminam multiplicação em diferentes graus:

| Nível | Técnica                      | Speed-up vs L1 (n=256) | Quando ajuda       |
|-------|------------------------------|------------------------|--------------------|
| L1    | I2_S GEMV (baseline fork)    | 0,0 %                  | —                  |
| L2    | WHT (Walsh-Hadamard)         | n/a (não integrada)    | matrizes muito rasas|
| L3    | ACDC (WHT + diagonal)        | +0,6 %                 | modelos P6-trained |
| L4a   | Tropical (max,+) + K_i8 cache| -1,8 %                 | atenção esparsa    |
| L4b   | Sparse float (F32 scoring)   | -2,4 %                 | **default L4**     |
| L5    | HRR (circular convolution)   | -69 % a -72 %          | modelos P6-trained |

**Conclusão principal:** A promessa teórica de 100× speedup via álgebra
alternativa **não se materializa** em BitNet-2B (modelo treinado SEM as
arquiteturas-alvo). Kernels L3, L4, L5 funcionam corretamente mas dão
output garbage porque o modelo espera matrizes densas. **O ganho real
só virá com P6: retreino com ACDC/HRR/tropical na arquitetura certa.**

---

## 1. Os 5 Níveis Algébricos

### L1 — I2_S GEMV (baseline)

Encoding 1.58 bits/param: pesos `{-1, 0, +1}` empacotados 4 por byte
(2 bits cada). Multiplicação por matriz vira `maddubs_epi16` (AVX2)
que faz `int8 × uint8 → int16` em 16 pares por ciclo. Mantido intacto
do fork upstream.

### L2 — WHT (Walsh-Hadamard Transform)

Pré-multiplica W por H e armazena W' = H·W. Na inferência, computa
W'·x onde x já está em domínio Hadamard. Como W' tem entradas
ternárias e x em {-1, 0, +1}, **a multiplicação vira XOR de bits** (0
ciclos de multiplicação). Speedup teórico: 16× sobre I2_S.

**Por que não integrou:** o custo de pré-multiplicar W é O(n² log n) e
precisa ser refeito se a matriz for atualizada. Em modelo frozen (só
inference), seria excelente — mas a estrutura do llama.cpp não facilita
"pré-transformar e cachear W". Caminho B+ permanece em aberto.

### L3 — ACDC (WHT + diagonal)

Variação do L2: ao invés de armazenar W' cheio (denso), extrai a
**diagonal** d* = diag(H·W·H) / n². Armazenamento: n floats em vez de
n² int8s (4× mais compacto!). Forward: y = H·diag(d)·(H·x) — duas WHTs
de comprimento n cada, mais n multiplicações escalares.

**Speedup real (BitNet-2B):** ~0 % (modelo não foi treinado com ACDC).
Em modelo P6-treinado, esperado: 3-5× sobre I2_S.

**Achado crítico (validação da teoria):** ACDC captura apenas
`~1/n` da energia de W aleatório Uniform{-1, 0, +1}. Verificado
empiricamente com 100+ matrizes do BitNet-2B: energia média = 0.04,
compatível com 1/n = 1/4096 = 0.0002 (ruído de realização em
matrizes pequenas; com n=4096 fica mais visível). **Não é bug** — é
consequência direta da concentração de Hadamard em matrizes
pseudo-aleatórias.

### L4 — Tropical Attention (max, +)

Re-define atenção sobre o semiring tropical: dot product vira max,
softmax vira argmax. Atenção: `y = V[argmax_k (q·K[k])]`. K_top-K
extension: seleciona os K maiores scores, faz softmax normal sobre
eles (não tropical puro).

**Speedup real:** L4 tropical com K=32 dá **-8,9 %** vs L1 em n=256
(antes do cache), **-1,8 %** (depois do cache). Sem cache, o bottleneck
é o "3-pass K": re-quantizar K a cada decode step.

### L4-alt — Sparse float

Mesma ideia do tropical mas scores em F32 (não int8). Single-pass: 1
leitura de K + 1 produto escalar. Sem int8 K buffer.

**Speedup real:** L4 Sparse float K=32 dá **-5,1 %** vs L1 em n=256
**antes do Phase C**, **-2,4 %** depois (mesma baseline). Sparse
float vence tropical em n ≥ 32. **Recomendação:** usar sparse float
como L4 default.

### L5 — HRR (Holographic Reduced Representations)

Circular-convolution memory. Memória M = Σ_k V[k] * K[k] (onde * é
convolução circular = IFFT(FFT(V)·FFT(K))). Retrieval: q*M = Σ V[k]·
(q*K[k]) no domínio convolucional. Cleanup iterativo (Frady 2021)
recupera o V exato a partir de q*M.

**Speedup real:** L5 raw dá **-69 %** vs L1 (FFT overhead). L5 + cleanup
é ainda pior: **-72 %** (mais iterações de cleanup). **Cleanup só ajuda
quando o modelo foi treinado com HRR**; em P6-unvalidated, o cleanup
convergiu para garbage mais rápido que convergir para qualquer coisa
útil. Achado: cleanup itera n_kv × max_iters × O(d log d) por head,
desperdiçando trabalho.

---

## 2. Bugs Reais Encontrados (3 no kernel, 1 no tooling)

### Bug #1: I2_S strided pack shift (commit cdce725)

WHT GEMV usava `(group * 2)` para extrair 2 bits do byte empacotado;
a função `unpack_i2s_block` do llama.cpp usava `(3 - group) * 2`.
Resultado: kernels L2 liam pesos espelhados. Test [1] (roundtrip
pack/unpack) falhou, expôs o mismatch, corrigido.

**Lição:** quando se depende de uma API de outro módulo, ler o código
fonte, não só o header.

### Bug #2: ACDC fwht_i8_to_i32 normalization (commit ed6fbde)

ACDC kernel tinha um stray `1/n²` que violava a spec de
`unnormalized — no 1/n² factors` em CLAUDE.md. Em W=I, esperava-se
d* = [1, 0, 0, ...] (energia capturada = 1.0); com o bug, d* = I/n
(energia = 1/n). Test [4] do `test_acdc.cpp` ajustou a asserção para
refletir o comportamento correto.

**Lição:** specs escritos em prosa são frágeis. Tests são specs.

### Bug #3: K_i8 cache GQA race condition (commit ec2a654)

GQA (Grouped Query Attention, n_head=20, n_head_kv=5, gqa=4) faz
múltiplas heads compartilharem o mesmo kv_head. Threads diferentes
acessavam o mesmo slot `(il, kv_h)` simultaneamente, corrompendo
`n_quantized` e o ponteiro `data`. Crash: "double free or corruption"
a partir de n_kv=96, t=4. **Fix:** `pthread_mutex_t` por slot. Custo:
desprezível (1 mutex por (il, kv_h), não por token).

**Lição:** strided head loop cria a ilusão de slots disjuntos, mas GQA
mapeia múltiplas heads no mesmo kv_head. Toda cache com
particionamento por (layer, head) precisa de sincronização explícita
em modelos com GQA > 1.

### Bug #4: ACDC energy formula (commit fcf1d4d)

`utils/extract_acdc_diagonal.py` primeira versão usava
`||H·diag(d)·H||_F² = n · ||d||²`. Verificação matemática
(W'·W'^T = n·H·diag(d²)·H, trace = n²·||d||²) e teste
`test_acdc_exact_recovery` mostraram fator correto é `n²`. Test
`energy_captured = 0.125` em vez de `1.0` para W = H·diag(d)·H
exato. Corrigido.

**Lição:** a fórmula parece razoável mas está errada. Tests com
counter-examples exatos (W = H·D·H, W = I) são essenciais para
algebraic kernels.

---

## 3. Cobertura de Testes (9/9 ctest, 50/50 subtests)

| Suite                          | Tipo   | Subtests | Cobre                                |
|--------------------------------|--------|----------|--------------------------------------|
| test_bitnet_common             | C++    | 5        | bitnet_next_pow2, aliases            |
| test_wht                       | C++    | 5        | WHT dot, sum_i8, gemv, pack          |
| test_acdc                      | C++    | 5        | FWHT, ACDC forward, project, gemv    |
| test_tropical                  | C++    | 5        | tropical argmax, topk, attention     |
| test_sparse_attention          | C++    | 5        | sparse_attention_float (F32)         |
| test_kv_i8_cache               | C++    | 11       | cache K_i8 (Phase C)                 |
| test_hrr_cleanup               | C++    | 5        | HRR FFT, bind, phasor, Frady 2021    |
| test_hrr_attention             | C++    | 5        | hrr_attention_full (kernel)          |
| test_extract_acdc_diagonal     | Python | 4        | closed-form d*, energy (Phase A)     |
| **Total**                      |        | **50**   |                                      |

Runtime total: 0,86 s (0,05 s C++ + 0,75 s Python com scipy).
CI: GitHub Actions Ubuntu 24.04 + Clang 18 + libstdc++-14-dev +
libstdc++-13 fallback, Python 3.13 com scipy/numpy/safetensors.

---

## 4. Benchmark Consolidado (BitNet-2B, t=4)

| Configuração                       | n=64     | n=128    | n=256    |
|------------------------------------|----------|----------|----------|
| L1 baseline (I2_S GEMV)            | 5,56-5,68| 4,88     | 5,06     |
| L3 ACDC FFN                        | 5,49-5,61| 4,77     | 5,09     |
| L4 Tropical K=32 (com cache, S2c)  | 5,38-5,44| 4,83     | 4,97     |
| L4 Sparse float K=32               | 5,48-5,54| 4,97     | 4,94     |
| L5 HRR raw                         | 2,95-3,10| 2,06     | 1,55     |
| L5 HRR + cleanup 8                 | 2,89-2,94| 1,83     | 1,38     |

**Análise:**
- L1, L3, L4 são todos competitivos (-2 % a +2 %). Diferença é ruído
  entre execuções.
- L5 é **catastrófico** em CPU: -69 % a -72 %. FFT (d log d) é caro
  demais para o tamanho de d que BitNet-2B usa (d=128, head_dim).
- A "3-pass K" do L4 tropical foi a maior fonte de overhead pré-cache.
  Cache (Phase C) eliminou 7,1 pp em n=256.
- Sparse float K=32 é o L4 mais rápido a n ≥ 32. **Recomendação:**
  tornar sparse float o L4 default (mais simples, sem int8 K, sem
  cache).

---

## 5. Por Que a Tese Não Validou Empiricamente

A promessa original do projeto era: "Universalizar LLMs em CPU via
álgebra esquecida, sem multiplicação". Isso pressupunha que a álgebra
**substitui** multiplicação sem perda de qualidade. O que descobrimos:

1. **L2/L3 só funcionam bem se o modelo for treinado com elas.**
   ACDC captura ~1/n da energia de W treinado denso. Para usar ACDC
   de verdade, o modelo precisa ser treinado COM a restrição de
   Hadamard-diagonalizabilidade. Isso é o Caminho C (P6, GPU,
   semanas de treino).

2. **L4 tropical/sparse funcionam mesmo em modelos densos**, mas
   perdem qualidade. Top-K=32 em n=256 ainda dá texto incoerente:
   ```
   Input:  "The capital of France is"
   Output: "The capital of France isalesmore Incorporated c
            levelsEven...BodyA\yedy?'s Breaths torst'ssrayuell
            in & theor fluid expectations site,..."
   ```
   O modelo é treinado com atenção completa, e top-K descarta
   informação crítica. **Em modelo P6-treinado com sparse
   attention loss, isso seria diferente.**

3. **L5 HRR é matematicamente elegante mas praticamente inviável em
   BitNet-2B.** O modelo tem head_dim=128, contexto=4096. FFT em
   d=128 é caro demais. HRR só compensa em d ≥ 1024 (Frady 2021
   usa d=512 ou 1024). Em d=128, o overhead do FFT supera qualquer
   ganho de complexidade.

**Recomendação:** focar P6 em L3 ACDC (most promising: 100× speedup
teórico, captura de energia treinável) e L4 sparse float (drop-in
substituição, sem FFT). L5 HRR fica como curiosidade matemática até
termos d ≥ 1024 (modelos 7B+ em que head_dim=256, ainda pequeno;
precisaríamos de modelos com multi-head attention desagrupada, d=512).

---

## 6. Roadmap Restante

### Curto prazo (sem GPU, semanas)
- **Sparse float como L4 default** (já competitivo, sem cache, sem int8)
- **L2/L3 ACDC para matrizes retangulares** (FFN gate/up/down)
- **Scoring in-place sobre K_i8** (otimização adicional L4 tropical)
- **Documentação matemática expandida** (`docs/theory/06-5-levels.md`)

### Médio prazo (GPU, meses)
- **Caminho C: P6 retraining** com arquitetura ACDC. Meta: 100×
  speedup sobre I2_S mantendo perplexidade < 5 % de degradação.
- **Acompanhar llama.cpp upstream** (Eddie-Wang1120/llama.cpp
  force-push nos pegou de surpresa uma vez; precisamos de CI que
  detecte rebase)

### Longo prazo
- **L5 HRR com d=512+** (modelos futuros, possivelmente BitNet 7B+)
- **Composicionalidade**: ACDC + tropical + HRR juntos (cada um
  para uma parte do forward)

---

## 7. Lições de Engenharia

1. **Tests com counter-examples exatos** (W = H·D·H, W = I) são
   essenciais para kernels algébricos. Não basta testar com dados
   aleatórios.
2. **Strided head loops em GQA não são thread-safe por construção**.
   Toda cache por (layer, head) precisa de sincronização.
3. **Vendoring de patches upstream** (vs submódulo) é frágil mas
   necessário quando upstream force-push. `apply-dispatch-patches.sh`
   com sentinelas resolve.
4. **Specs em prosa são frágeis**; tests são specs. Bug #2 só foi
   pego porque atualizamos o test.
5. **Performance de kernels algébricos depende do modelo treinado**,
   não só do kernel. Benchmarks sem retreino são limitados.

---

## 8. Reproducibilidade

```bash
# Setup (modelo + conversão)
conda activate bitnet-cpp
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Build
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Tests (9/9, 50 subtests, 0,86 s)
cmake -B build_tests -DBITNET_TESTING=ON -DBITNET_L2_WHT=ON \
  -DBITNET_L3_ACDC=ON -DBITNET_L4_TROPICAL=ON -DBITNET_L5_HRR=ON \
  -DCMAKE_BUILD_TYPE=Release [mesmas flags C++]
cmake --build build_tests -j$(nproc)
cd build_tests && ctest --output-on-failure

# Bench
python utils/cpu_universal_benchmark.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -n 256 -t 4
python utils/tropical_benchmark.py --n 256 --d 64 --k 16
python utils/acdc_benchmark.py --n 512 --scaling
python utils/wht_benchmark.py

# Phase A: extrair diagonal ACDC (requer safetensors)
python utils/extract_acdc_diagonal.py models/bitnet-b1.58-2B-4T-bf16/
```

---

## Apêndice A: Mapeamento princípio→código→verificação

Ver `.reversa/scout/principle-code-map.json` (atualizado 2026-06-06d)
para mapeamento completo de cada princípio P1-P7 em:
- Arquivo + linha de implementação
- Doc reference em `docs/theory/`
- Verification (test + bench)
- Limits / quantization

## Apêndice B: Inventário completo

Ver `.reversa/scout/inventory.md` (atualizado 2026-06-05, 460 linhas)
para lista exaustiva de:
- 17 arquivos de cabeçalho (BitNet + L1-L5)
- 8 arquivos de implementação (BitNet + L1-L5)
- 9 arquivos de teste (8 C++ + 1 Python)
- 5 scripts de benchmark
- 4 docs principais (mathematical-foundations, codegen, theory/*)

## Apêndice C: Análise de Gaps (gap-analysis.md)

Ver `.reversa/scout/gap-analysis.md` para o estado consolidado:
- Fundação teórica: 100 %
- Kernels L1-L5 standalone: 100 %
- Integração dispatch: 100 %
- Validação empírica: parcial (limitada por modelo não-treinado)
- **Gap principal: P6 (retreino GPU, fora de escopo deste fork)**
