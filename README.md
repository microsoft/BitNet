# BitNet CPU-Universal — Inferência 1.58-bit local-first

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CPU Only](https://img.shields.io/badge/compute-CPU%20only-orange.svg)]()
[![No CUDA](https://img.shields.io/badge/no%20CUDA-required-red.svg)]()
[![No Cloud](https://img.shields.io/badge/no%20cloud-required-lightgrey.svg)]()
[![Air-Gapped](https://img.shields.io/badge/air--gapped-tested-success.svg)]()
[![Math Levels](https://img.shields.io/badge/math%20levels-5%2F5-blueviolet.svg)]()

> **Inferência 1.58-bit local-first, sem CUDA, sem cloud, sem telemetria.**
> Para a persona **D4** — Desenvolvedores de Privacidade e Soberania de Dados.
>
> **Fork de [`microsoft/BitNet`](https://github.com/microsoft/BitNet)** que
> estende o framework com 5 níveis algébricos (L1 I2_S, L2 WHT, L3 ACDC,
> L4 tropical, L5 HRR) demonstrando a tese de "inferência CPU via álgebra
> esquecida".

---

## O que é este fork

BitNet CPU-Universal é uma engine de **inferência de LLM 100% local**,
otimizada para **CPU-only** (x86_64 com AVX2+ ou ARM64 com NEON) e
**auditada para uso em ambientes air-gapped** (sem rede, sem telemetria,
sem cloud).

**Para quem é:** Profissionais e organizações de **setores regulamentados**
(saúde, jurídico, financeiro) que precisam rodar LLMs em laptops
corporativos ou hardware legado, **sem enviar dados para a nuvem**.

**Diferencial:** ao contrário de forks que apenas **removem** o suporte a
GPU, este fork **adiciona** 4 níveis algébricos (L2-L5) que demonstram
que estruturas matemáticas publicadas há mais de um século (Walsh-Hadamard,
ACDC, semiring tropical, HRR) podem acelerar a inferência CPU eliminando
operações caras, **não** adicionando paralelismo.

---

## TL;DR (3 comandos)

```bash
# 1. Setup (uma vez, online)
git clone --recursive https://github.com/peder1981/BitNet.git && cd BitNet
conda create -n bitnet-cpp python=3.10 -y && conda activate bitnet-cpp
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# 2. Inferência (offline, sem rede)
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Resuma este prontuário:" -n 200 -t 4

# 3. Validar air-gapped (AC-11, NO-06, NO-07)
bash tests/test_air_gapped_boot.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# esperado: "AC-11 air-gapped boot: PASS ✓"
```

---

## Casos de uso (persona D4)

Documentação detalhada em `examples/`:

| Persona | Caso de uso | Documentação |
|---------|-------------|--------------|
| **Médico** | Analisa prontuário em laptop de consultório (LGPD/HIPAA) | [`examples/medical_offline.md`](examples/medical_offline.md) |
| **Advogado** | Resume petição inicial em escritório (sigilo OAB) | [`examples/legal_offline.md`](examples/legal_offline.md) |
| **Analista financeiro** | Categoriza despesas em workstation bancária restrita (BCB/GLBA) | [`examples/finance_offline.md`](examples/finance_offline.md) |
| **Pesquisador** | Roda BitNet-2B em máquina institucional bloqueada | Mesmo setup de `medical_offline.md` (substituir prompt) |
| **Entusiasta** | Roda em laptop de 2018 (hardware legado) | Baseline em [`docs/hardware-compatibility.md`](docs/hardware-compatibility.md) |

**Por que BitNet CPU-Universal atende:** inferência **1.58-bit/param**
(elimina dependência de CUDA), execução **nativa em CPU** (sem GPU,
sem cloud), modelo **inteiro off-line** após download inicial, **sem
telemetria** (NO-06), **sem cloud** (NO-07), footprint de RAM
**previsível** (BitNet-2B + KV cache 4-bit cabe em 4-5 GB).

---

## Os 5 Níveis Algébricos (L1-L5)

| Nível | Operação | Elimina | Substituída por | Ganho | Status |
|-------|----------|---------|-----------------|-------|--------|
| **L1 I2_S** | Ternary GEMM (x86/ARM) | FP32 weights (32 bits) | `quant(W) ∈ {-1,0,+1}` packed 4/byte | **20× menos memória** (1.58 bits/param) | ✅ Baseline |
| **L2 WHT** | Walsh-Hadamard decomposition | Multiplicação por W | `W = H·D·H` (3 matrizes esparsas) + XOR/add | **Zero multiplicações** | ✅ Pronto (pesquisa) |
| **L3 ACDC** | Adaptive Circulant Diagonal Conv | GEMM denso O(n²) | FWHT em circulant: `W·x = H·(d·(H·x))` | **O(n log n)** (vs O(n²)) | ✅ Pronto (requer retreino P6) |
| **L4 sparse float** | Tropical (max,+) | Softmax completo | `argmax` top-K + softmax sobre K tokens | **O(n·d + K·d)** (vs O(n²·d)) | ✅ Pronto (opt-in) |
| **L5 HRR** | Holographic Reduced Reps | Attention densa | `bind(q,k) = q ⊛ k` (FFT circular) + cleanup | **O(n·log d)** binding/unbinding | ✅ Pronto (requer retreino P6) |

**Decisão crítica (P6 — Estrutura, não compressão):** L3 ACDC e L5 HRR
são **arquiteturas de treinamento**, não compressões. Aplicar essas
arquiteturas a BitNet-2B (que foi treinado com arquitetura clássica) dá
output garbage. Para funcionar, o modelo precisa ser **treinado do zero**
com a arquitetura. Esse retreino é **reserva técnica** (Q4 2029, ver
`ROADMAP.md#2`).

**Recomendação (v0.2, dados empíricos):**

| Modelo | Modo recomendado | Env var | Speedup |
|--------|-----------------|---------|--------|
| BitNet-2B | L1 baseline | — | — |
| Falcon3-3B | L3 ACDC rect auto | `BITNET_ACDC_FFN_RECT=auto` | +144% |
| Falcon3-10B | L3 ACDC rect auto | `BITNET_ACDC_FFN_RECT=auto` | +267% |
| Qualquer (n_ff/n_embd 2-5) | L4 adaptive-K | `BITNET_SPARSE_TOPK_ADAPTIVE=0.90` | +29% |

Ver `docs/decision-matrix.md` para critérios completos.

---

## Instalação

### Requisitos

```
python >= 3.9
cmake  >= 3.22
clang  >= 18   (obrigatório — SIMD kernels requerem Clang)
conda  (recomendado)
```

**Hardware mínimo:** x86_64 com AVX2 (post-2013, ex: Intel Haswell) ou
ARM64 com NEON (ex: Apple M1, Cortex-A76). Ver
[`docs/hardware-compatibility.md`](docs/hardware-compatibility.md) para
matriz completa de CPUs e modos suportados.

### Setup completo

```bash
# Clone com submodules
git clone --recursive https://github.com/peder1981/BitNet.git
cd BitNet

# Ambiente conda
conda create -n bitnet-cpp python=3.10 -y
conda activate bitnet-cpp
pip install -r requirements.txt

# Build (Clang 18)
conda install -c conda-forge llvmdev=18 -y
cmake -B build -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Download + conversão + compilação (BitNet-b1.58-2B-4T, x86_64)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf \
  --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# ARM64: usar -q tl1
# Para TL2 (x86_64, melhor performance):
# python setup_env.py -md models/BitNet-b1.58-2B-4T -q tl2
```

Após este setup, **o laptop está pronto para uso offline permanente**.

---

## Uso

### Inferência básica

```bash
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Resuma este prontuário:" -n 200 -t 4
```

### Modo conversacional

```bash
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Você é um assistente especializado em matemática" \
  -cnv
```

### FFN acelerada: ACDC rect auto (L3, recomendado para Falcon3)

```bash
# Ativa automaticamente quando n_ff/n_embd >= 3.0.
# Falcon3-3B: +144%; Falcon3-10B: +267%. Zero configuração extra.
BITNET_ACDC_FFN_RECT=auto build/bin/llama-cli \
  -m models/Falcon3-10B-Instruct-1.58bit-GGUF/ggml-model-i2_s.gguf \
  -p "..." -n 64 -t 4
```

### Atenção esparsa adaptativa (L4 adaptive-K, opt-in)

```bash
# Adaptive-K cov=0.90: seleciona K por head via threshold softmax.
# +28.8% Falcon3-3B; quase neutro (-1.3%) BitNet-2B.
BITNET_SPARSE_TOPK_ADAPTIVE=0.90 build/bin/llama-cli \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "..." -n 64 -t 4

# Sparse float K fixo (legado, superado por adaptive-K):
BITNET_SPARSE_TOPK=32 build/bin/llama-cli \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "..." -n 200 -t 4
```

### Benchmark end-to-end

```bash
python utils/e2e_benchmark.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -n 128 -p 512 -t 4
```

### Benchmark sistemático (RF-07)

```bash
# Gera JSON canônico (source of truth) + Markdown derivado
python utils/bench_publish.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  --json benchmarks/v0.1.0/bench.json \
  --md benchmarks/v0.1.0/bench.md
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

## Validação air-gapped (AC-11, NO-06, NO-07)

```bash
# Smoke test: binário roda sem rede
bash tests/test_air_gapped_boot.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# esperado: "AC-11 air-gapped boot: PASS ✓"
```

O script usa `unshare -rn` (preferido) ou `strace -e network` (fallback)
para detectar qualquer syscall de rede. Valida também que o log não
contém "telemetry", "upload" ou "error" relacionado a GPU/network.

**Auditoria NO-06 (sem telemetria):**
```bash
grep -rn "telemetry\|upload_data\|send_metrics" src/ utils/ run_inference*.py
# esperado: 0 hits
```

**Auditoria NO-07 (sem cloud):**
```bash
grep -rn "http://\|https://" src/ 3rdparty/llama.cpp/ggml/src/ 2>/dev/null | grep -v "//.*comment" | grep -v "ggml-cuda\|ggml-opencl"
# esperado: 0 hits em código de produção
```

---

## Testes (RNF-01)

```bash
cd build && ctest --output-on-failure
# esperado: 15/15 PASS (default CI, test_acdc_rect opt-in via D2 gate)
# ou 16/16 com -DBITNET_ENABLE_ACDC_RECT=ON
```

Cobre: kernel L1-L5 (WHT, FWHT, ACDC, tropical, HRR, K_i8 cache),
property-based tests com 100-1000 iters cada, e análise estática do
dispatch. Ver `tests/test_*_properties.cpp` (T005-T008) e
`docs/invariants.md` (P1-P7).

---

## Documentação

### Decisão e arquitetura

- [`ROADMAP.md`](ROADMAP.md) — Roadmap público (Atual / Reserva / Fora de escopo)
- [`docs/decision-matrix.md`](docs/decision-matrix.md) — Quando usar L1/L3/L4/L5
- [`docs/hardware-compatibility.md`](docs/hardware-compatibility.md) — Tabela CPU → modo
- [`docs/invariants.md`](docs/invariants.md) — P1-P7 canônicas (invariantes matemáticas)
- [`docs/findings-cpu-universal.md`](docs/findings-cpu-universal.md) — Validação empírica consolidada

### Teoria (referência acadêmica)

- [`docs/theory/00-index.md`](docs/theory/00-index.md) — Índice
- [`docs/theory/01-ternary-algebra.md`](docs/theory/01-ternary-algebra.md) — Quantização ternária, Shannon floor
- [`docs/theory/02-wht-decomposition.md`](docs/theory/02-wht-decomposition.md) — WHT, zero multiplicações
- [`docs/theory/03-acdc-structured-layers.md`](docs/theory/03-acdc-structured-layers.md) — FWHT, ACDC
- [`docs/theory/04-tropical-algebra.md`](docs/theory/04-tropical-algebra.md) — Semiring (max,+)
- [`docs/theory/05-holographic-memory.md`](docs/theory/05-holographic-memory.md) — HRR, convolução circular
- [`docs/theory/06-5-levels.md`](docs/theory/06-5-levels.md) — Sumário canônico de 1 página

### Walkthroughs (persona D4)

- [`examples/medical_offline.md`](examples/medical_offline.md) — Médico em consultório
- [`examples/legal_offline.md`](examples/legal_offline.md) — Advogado em escritório
- [`examples/finance_offline.md`](examples/finance_offline.md) — Analista em workstation bancária

### Análise reversa (imutável)

- `_reversa_sdd/` — Specs do legado (análise reversa original)
- `.reversa/scout/` — Síntese de princípios e gaps

---

## Arquitetura do código

```
src/
  ggml-bitnet-mad.cpp      ← Kernel I2_S (AVX2 + NEON), L1
  ggml-bitnet-lut.cpp      ← Kernels TL1/TL2 lookup-table, L1
  ggml-bitnet-wht.cpp      ← WHT zero-multiplicação, L2
  ggml-bitnet-fwht.cpp     ← FWHT + ACDC O(n log n), L3
  ggml-bitnet-tropical.cpp ← Atenção tropical (max,+), L4
  ggml-bitnet-hrr.cpp      ← Memória holográfica, L5
  ggml-bitnet-dispatch.cpp ← Dispatch L3-L5 + integração llama.cpp
  ggml-bitnet-kv-cache.cpp ← K_i8 cache (P5, P6)
  ggml-bitnet-common.cpp   ← Utilitários compartilhados (next_pow2)

include/
  ggml-bitnet.h            ← API principal (L1)
  ggml-bitnet-wht.h        ← API WHT (L2)
  ggml-bitnet-fwht.h       ← API FWHT/ACDC (L3)
  ggml-bitnet-tropical.h   ← API tropical (L4)
  ggml-bitnet-hrr.h        ← API holográfica (L5)
  gemm-config.h            ← Parâmetros de kernel (ROW/COL_BLOCK_SIZE, PARALLEL_SIZE)

tests/
  test_bitnet_common.cpp, test_wht.cpp, test_acdc.cpp,
  test_tropical.cpp, test_sparse_attention.cpp, test_kv_i8_cache.cpp,
  test_hrr_cleanup.cpp, test_hrr_attention.cpp,
  test_acdc_properties.cpp, test_l4_sparse_properties.cpp,
  test_hrr_properties.cpp, test_dense_is_default.cpp  ← T005-T008
  test_air_gapped_boot.sh  ← AC-11 air-gapped
  cross_validation.py      ← C ↔ Python cross-validação
  snapshots/               ← Snapshots canônicos v0.1.0

utils/
  wht_benchmark.py         ← Verifica e benchmarka L2
  acdc_benchmark.py        ← Verifica e benchmarka L3
  tropical_benchmark.py    ← Verifica e benchmarka L4
  hrr_benchmark.py         ← Verifica e benchmarka L5
  cpu_universal_benchmark.py ← Benchmark sistemático L1-L5
  bench_publish.py         ← JSON canônico + Markdown derivado (T020)
  codegen_tl1.py, codegen_tl2.py ← Lookup-table kernel generators

examples/                  ← Walkthroughs persona D4 (T021-T023)
docs/                      ← Documentação canônica (theory, decisions, hardware)
ROADMAP.md                 ← Roadmap público (T014)
LICENSE                    ← MIT
```

---

## Restrições fundadoras

- **CPU only.** GPU kernels são proibidos (NO-02, decisão fundadora).
- **Sem cloud, sem servidor, sem multi-tenant** (NO-07). Persona D4 é
  incompatível com deployment cloud.
- **Sem telemetria** (NO-06). Qualquer instrumentação nova deve ser
  opt-in e justificada.
- **Sem mudança no formato GGUF** (NO-03). O fork consome GGUF, não
  produz variante.
- **Patches vendored** (RNF-04, NO-04). `3rdparty/llama.cpp/` permanece
  read-only; mudanças vão em `patches/llama.cpp/0N-*.patch` com
  sentinel idempotente em `scripts/apply-dispatch-patches.sh`.

---

## Configuração de performance

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

## Contribuindo

1. Leia `docs/invariants.md` (P1-P7 são **invariantes**; quebrar uma = bloquear o PR).
2. Cada kernel algébrico novo/modificado precisa de **test de contra-exemplo exato** (P7).
3. `ctest` é a especificação executável (P2): a prosa explica, o test valida.
4. Persona D4 governa produto e marketing; `docs/theory/` permanece como
   referência acadêmica intacta.

---

## Licença

MIT — ver [`LICENSE`](LICENSE).

Os modelos HuggingFace referenciados têm suas próprias licenças. Este
software é fornecido **como está**, sem garantias. Em particular, as
limitações conhecidas em `docs/findings-cpu-universal.md#5` e em cada
`examples/*.md` devem ser lidas antes de uso em produção.

---

*v2.0 — README reescrito por T028 (Fase 4: Integração) em 2026-06-06.*
*v1 → v2: persona D4 adicionada, 5 níveis promovidos no TL;DR, exemplos
promovidos, validação air-gapped no fluxo padrão. Preserva `docs/theory/`
como referência acadêmica intacta. v1 preservado em git history.*
