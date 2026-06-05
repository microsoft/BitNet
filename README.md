# CPU Universal LLM — Inferência sem GPU via Álgebra Esquecida

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CPU Only](https://img.shields.io/badge/compute-CPU%20only-orange.svg)]()
[![Math Level](https://img.shields.io/badge/math%20level-4%2F5-green.svg)]()

> **Hipótese central**: a inferência de LLMs de grande porte no CPU pode atingir
> a velocidade da GPU não por paralelismo de hardware, mas por eliminação algébrica
> das multiplicações de ponto flutuante — descendo a hierarquia de custo operacional
> até estruturas matemáticas publicadas há mais de um século e esquecidas pela
> corrida ao hardware.

---

## O Problema e a Resposta

Um modelo de 7B parâmetros em fp16 precisa de ~14 TFLOPS para gerar um token.
Uma CPU entrega ~0.5 TFLOPS. A GPU fecha esse gap com paralelismo bruto.

**Nossa abordagem**: ao invés de paralelismo, eliminamos as operações pelo lado matemático.

```
Hierarquia de custo real por elemento:
  Multiplicação float32  ~4–5 ciclos
  Adição float32         ~1 ciclo
  Comparação             ~0.3 ciclos
  XOR/AND de bits        ~0.1 ciclos
```

Cada nível deste projeto troca operações caras por mais baratas:

| Nível | O que eliminamos | Substituímos por | Status |
|-------|-----------------|-----------------|--------|
| 1 | Pesos float | Ternário {-1,0,+1} | ✓ herdado |
| 2 | Multiplicações GEMV | Adições condicionais (WHT) | ✓ feito |
| 3 | Complexidade O(n²) GEMV | O(n log n) FWHT + diagonal | ✓ feito |
| 4 | O(n²) atenção + exponenciais | Comparações top-K (tropical) | ✓ feito |
| 5 | Atenção O(n²) completa | Memória holográfica O(n log n) | → em andamento |

---

## Base: Inferência Ternária CPU (bitnet.cpp)

Este repositório é baseado no framework `bitnet.cpp` para inferência de LLMs com
pesos ternários {-1, 0, +1} (1.58 bits/parâmetro). Herda três formatos de kernel:

- **I2_S** — 2 bits por peso, packed (x86_64 + ARM64)
- **TL1** — Lookup-table GEMM otimizada (ARM64)
- **TL2** — Lookup-table GEMM otimizada (x86_64)

Os kernels são compilados via CMake + Clang e integrados ao backend `llama.cpp`.

---

## Modelos Suportados

| Modelo | Parâmetros | Kernel x86 | Kernel ARM |
|--------|-----------|-----------|-----------|
| [BitNet-b1.58-2B-4T](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) | 2.4B | I2_S, TL2 | I2_S, TL1 |
| [bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large) | 0.7B | I2_S, TL2 | I2_S, TL1 |
| [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B) | 3.3B | TL2 | TL1 |
| [Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens) | 8.0B | I2_S, TL2 | I2_S, TL1 |
| [Falcon3 Family](https://huggingface.co/tiiuae) | 1B–10B | I2_S, TL2 | I2_S, TL1 |
| [Falcon-E Family](https://huggingface.co/tiiuae) | 1B–3B | I2_S, TL2 | I2_S, TL1 |

---

## Instalação

### Requisitos

```
python >= 3.9
cmake  >= 3.22
clang  >= 18   (obrigatório — SIMD kernels requerem Clang)
conda  (recomendado)
```

### Setup completo

```bash
# Clone com submodules
git clone --recursive https://github.com/peder1981/BitNet.git
cd BitNet

# Ambiente conda
conda create -n bitnet python=3.9 -y
conda activate bitnet
pip install -r requirements.txt

# Download + conversão + compilação (BitNet-b1.58-2B-4T, x86_64)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
  --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# ARM64: usar -q tl1
# Para TL2 (x86_64, melhor performance):
# python setup_env.py -md models/BitNet-b1.58-2B-4T -q tl2
```

---

## Uso

### Inferência

```bash
# Geração de texto
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Explique álgebra ternária" \
  -n 200 -t 4

# Modo conversacional
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Você é um assistente especializado em matemática" \
  -cnv
```

### Benchmark de throughput

```bash
python utils/e2e_benchmark.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -n 128 -p 512 -t 4
```

### Conversão de modelos

```bash
# De safetensors (pesos bf16) para GGUF ternário
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 \
  --local-dir ./models/bitnet-b1.58-2B-4T-bf16
python utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16

# Com quantização de embeddings (Q6_K — melhor trade-off velocidade/qualidade)
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s --quant-embd
```

---

## Extensões Matemáticas (nossa contribuição)

### Nível 2 — Zero multiplicações (WHT)

A identidade W = W⁺ - W⁻ decompõe qualquer GEMV ternário em duas somas condicionais,
eliminando 100% das multiplicações. Verificação: max_diff = 0 (identidade inteira exata).

```bash
python utils/wht_benchmark.py
```

### Nível 3 — O(n log n) GEMV (ACDC)

Camadas estruturadas como H·diag(d)·H computadas via Fast Walsh-Hadamard Transform.
Custo: 2 FWHTs + n multiplicações (mínimo irredutível). Speedup: ~174× sobre GEMV padrão.

```bash
python utils/acdc_benchmark.py --n 512 --scaling
```

### Nível 4 — Atenção tropical (max,+)

O semiring tropical (max,+) substitui o softmax no limite de temperatura → 0.
Top-K via scan ternário (zero multiplicações) + softmax sobre apenas K tokens.
Speedup teórico: ~2863× na atenção do BitNet-2B para K=32.

```bash
python utils/tropical_benchmark.py --n 512 --d 64 --k 32 --scaling
```

### Nível 5 — Memória holográfica (em andamento)

Convolução circular via FFT (binding) substitui a atenção Transformer completamente.
Cada head de atenção se torna um único vetor M que armazena todos os pares (K, V)
do contexto. Recuperação por query: O(d log d) independente do comprimento de sequência.

---

## Documentação Teórica

- [`docs/theory/00-index.md`](docs/theory/00-index.md) — Índice e conexões entre níveis
- [`docs/theory/01-ternary-algebra.md`](docs/theory/01-ternary-algebra.md) — Álgebra ternária, quantização, STE
- [`docs/theory/02-wht-decomposition.md`](docs/theory/02-wht-decomposition.md) — WHT, zero multiplicações
- [`docs/theory/03-acdc-structured-layers.md`](docs/theory/03-acdc-structured-layers.md) — FWHT, ACDC, O(n log n)
- [`docs/theory/04-tropical-algebra.md`](docs/theory/04-tropical-algebra.md) — Semiring (max,+), atenção tropical
- [`docs/theory/05-holographic-memory.md`](docs/theory/05-holographic-memory.md) — HRR, convolução circular, Kanerva

---

## Arquitetura

```
src/
  ggml-bitnet-mad.cpp      ← Kernel I2_S (AVX2 + NEON), L1
  ggml-bitnet-lut.cpp      ← Kernels TL1/TL2 lookup-table, L1
  ggml-bitnet-wht.cpp      ← WHT zero-multiplicação, L2
  ggml-bitnet-fwht.cpp     ← FWHT + ACDC O(n log n), L3
  ggml-bitnet-tropical.cpp ← Atenção tropical (max,+), L4
  ggml-bitnet-hrr.cpp      ← Memória holográfica, L5 [em construção]

include/
  ggml-bitnet.h            ← API principal (L1)
  ggml-bitnet-wht.h        ← API WHT (L2)
  ggml-bitnet-fwht.h       ← API FWHT/ACDC (L3)
  ggml-bitnet-tropical.h   ← API tropical (L4)
  ggml-bitnet-hrr.h        ← API holográfica (L5) [em construção]
  gemm-config.h            ← Parâmetros de kernel (ROW/COL_BLOCK_SIZE, PARALLEL_SIZE)

utils/
  wht_benchmark.py         ← Verifica e benchmarka L2
  acdc_benchmark.py        ← Verifica e benchmarka L3
  tropical_benchmark.py    ← Verifica e benchmarka L4
  hrr_benchmark.py         ← Verifica e benchmarka L5 [em construção]
  codegen_tl1.py           ← Gerador de kernels TL1 (ARM64)
  codegen_tl2.py           ← Gerador de kernels TL2 (x86_64)
  e2e_benchmark.py         ← Benchmark end-to-end de throughput

3rdparty/llama.cpp         ← Backend de inferência CPU (submodule)
preset_kernels/            ← Configs GEMM pré-tunadas por modelo
```

---

## Configuração de Performance

O arquivo `include/gemm-config.h` controla os parâmetros do kernel I2_S:

```c
#define ROW_BLOCK_SIZE  4    // linhas processadas por bloco
#define COL_BLOCK_SIZE  128  // colunas por bloco (x86)
#define PARALLEL_SIZE   4    // grau de paralelismo
```

Para auto-tuning no seu hardware:
```bash
python utils/tune_gemm_config.py
```

---

## Licença

MIT — ver `LICENSE`.

Os modelos HuggingFace referenciados têm suas próprias licenças.
