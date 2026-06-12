# BitNet CPU-Universal — Inferência 1.58-bit local-first + Tool-Calling PT-BR

[![CI](https://github.com/peder1981/BitNet/actions/workflows/ci.yml/badge.svg)](https://github.com/peder1981/BitNet/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![CPU Only](https://img.shields.io/badge/compute-CPU%20only-orange.svg)]()
[![No CUDA](https://img.shields.io/badge/no%20CUDA-required-red.svg)]()
[![No Cloud](https://img.shields.io/badge/no%20cloud-required-lightgrey.svg)]()
[![Air-Gapped](https://img.shields.io/badge/air--gapped-tested-success.svg)]()
[![Math Levels](https://img.shields.io/badge/math%20levels-5%2F5-blueviolet.svg)]()
[![Fine-Tuned](https://img.shields.io/badge/fine--tuned-Falcon3--3B--PTBR--tools-blue.svg)]()

> **Inferência 1.58-bit local-first, sem CUDA, sem cloud, sem telemetria.**
> Agora com **fine-tuning local CPU-only** para tool-calling em português
> via MCP, **parser robusto de JSON truncado**, e **memória cross-agent**.
>
> **Fork de [`microsoft/BitNet`](https://github.com/microsoft/BitNet)** +
> **BitNet Studio** (server Python) com adapter QLoRA Falcon3-3B-Instruct
> fine-tuned para 10 ferramentas Protheus-RAG em PT-BR.

---
## O que é este projeto

BitNet CPU-Universal é uma stack completa de **inferência de LLM 100% local**
que evoluiu de um fork C++ de pesquisa para um sistema produtivo com:

1. **BitNet C++** — Engine de inferência 1.58-bit com 5 níveis algébricos (L1-L5)
2. **BitNet Studio** — Server Python com MCP bridge, fine-tuning local, e tool-calling
3. **Falcon3-3B Adapter** — Modelo fine-tuned CPU-only para responder em PT-BR e
   invocar 10 ferramentas Protheus-RAG via `<tool_call>`

**Para quem é:** Desenvolvedores e organizações que precisam de LLM
**offline, privado e soberano** — especialmente no ecossistema TOTVS Protheus
(AdvPL/TLPP), com acesso a RAG interno, dicionário de dados, e memória
persistente entre sessões.

---
## TL;DR (4 comandos)

```bash
# 1. Clone e setup
git clone --recursive https://github.com/peder1981/BitNet.git && cd BitNet
conda create -n bitnet python=3.10 -y && conda activate bitnet
pip install -r bitnet-studio/pyproject.toml  # ou requirements.txt

# 2. Fine-tune local CPU (Falcon3-3B, 150 steps, ~34 min)
cd bitnet-studio
python finetune_local.py  # gera adapter em adapters/f3b-ptbr-tools-local/

# 3. Testar tool-calling (72 testes exaustivos, ~3h)
python test_50x_file.py  # valida extração de JSON truncado/multiline

# 4. Inferência C++ air-gapped (BitNet-2B, sem rede)
cd ..
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Resuma este prontuário:" -n 200 -t 4
```

---
## Stack atual (2026-06-12)

### BitNet C++ (núcleo de pesquisa)

Engine de inferência 1.58-bit com 5 níveis algébricos demonstrando
"inferência CPU via álgebra esquecida":

| Nível | Operação | Ganho | Status |
|-------|----------|-------|--------|
| **L1 I2_S** | Quantização ternária `{-1,0,+1}` | 20× menos memória | ✅ Produção |
| **L2 WHT** | Walsh-Hadamard `W = H·D·H` | Zero multiplicações | ✅ Pesquisa |
| **L3 ACDC** | FWHT em circulant O(n log n) | +144% Falcon3-3B | ✅ Produção |
| **L4 Tropical** | Atenção esparsa (max,+) | +29% adaptive-K | ✅ Produção |
| **L5 HRR** | Memória holográfica | O(n log d) binding | 🔄 Reserva |

Ver `docs/theory/` para fundamentação matemática completa.

### BitNet Studio (novo — server Python + fine-tuning)

```
bitnet-studio/
├── studio/
│   └── server/
│       ├── tool_engine.py      ← Parser robusto de tool_call (JSON truncado/multiline)
│       ├── mcp_bridge.py       ← Bridge MCP para 10 tools protheus-rag
│       └── inference.py        ← Geração com adapter QLoRA
├── finetune_local.py           ← Fine-tune 100% CPU (Falcon3-3B, QLoRA)
├── test_50x_file.py            ← Teste exaustivo 72 rodadas (6×12 perguntas)
└── adapters/
    └── f3b-ptbr-tools-local/   ← Adapter 150 steps (~13s/step, 34 min total)
```

**Ferramentas disponíveis (MCP — protheus-rag):**

| Tool | Função | Exemplo de uso |
|------|--------|----------------|
| `consultar_base_direta` | Busca direta no RAG AdvPL/TLPP | "Como funciona MaFisCalc?" |
| `consultar_base_interna` | Consulta interpretada via LLM | "Como funciona o faturamento?" |
| `consultar_dicionario_direto` | Dicionário de dados Protheus | "Quais campos tem SA1?" |
| `buscar_reversa_direto` | Busca no framework Reversa | "Como usar reversa-scout?" |
| `consultar_reversa_rag` | Consulta interpretada Reversa | "Como criar REST endpoint TLPP?" |
| `mem0_search` | Busca memórias do usuário | "O que sabemos sobre cliente João?" |
| `mem0_add` | Adiciona memória | "Anote: cliente prefere e-mail" |
| `mem0_list` | Lista todas memórias | "Liste memórias salvas" |
| `mem0_stats` | Estatísticas da base | "Quantas memórias temos?" |
| `mem0_delete` | Remove memória | "Apague memória sobre teste" |

**Parser de tool_call (robustez):**

- Extrai JSON de `<tool_call>...</tool_call>` completo
- Captura `<tool_call>` truncado (sem `</tool_call>`)
- Suporta JSON multiline com balanced braces
- Fallback para regex de nome isolado em texto corrido
- 6 níveis de fallback progressivos

### Protocolo mem0 (cross-agent)

Memória persistente compartilhada entre agentes (Claude, OpenCode, Windsurf,
Devin) via namespace `default`. Regra mandatória: **RAG local primeiro** —
consultar `mem0_search` antes de qualquer busca externa.

Configurado em `AGENTS.md` e `CLAUDE.md`.

---
## Fine-tuning local (100% CPU)

### Setup de dados

```bash
cd bitnet-studio
# Dataset: 162 exemplos de tool-calling em PT-BR
# Formato: <|user|>pergunta<|assistant|><tool_call>{"name":..., "arguments":...}
```

### Treinamento

```bash
# Falcon3-3B-Instruct + QLoRA (r=16, alpha=32, target_modules=all linear)
# 150 steps, batch_size=2, gradient_accumulation=4
# ~13s/step = ~34 min total em CPU (Ryzen 9, 12 cores)
python finetune_local.py
```

### Resultados do adapter

| Métrica | Valor |
|---------|-------|
| Base model | `tiiuae/Falcon3-3B-Instruct` |
| Adapter path | `adapters/f3b-ptbr-tools-local/` |
| Steps | 150 |
| Tempo total | ~34 min |
| Tempo/step | ~13s |
| Hardware | CPU-only (12 threads) |

### Validação exaustiva

```bash
# 72 testes = 12 perguntas × 6 iterações
# Verifica: extração correta, JSON truncado, multiline, sem </tool_call>
python test_50x_file.py
```

Resultado esperado (com parser robusto): **>80% acerto** na extração de
tool calls, mesmo com respostas truncadas pelo modelo.

---
## Uso

### Inferência C++ (air-gapped)

```bash
# Setup (uma vez)
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Uso offline permanente
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Resuma este prontuário:" -n 200 -t 4
```

### Tool-calling com Falcon3 (Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from studio.server.tool_engine import parse_tool_call

# Carregar base + adapter
base = AutoModelForCausalLM.from_pretrained("tiiuae/Falcon3-3B-Instruct")
model = PeftModel.from_pretrained(base, "adapters/f3b-ptbr-tools-local")

# Gerar resposta
prompt = "<|user|>\nComo funciona MaFisCalc?\n<|assistant|>\n"
output = model.generate(**tokenizer(prompt, return_tensors="pt"), max_new_tokens=180)
response = tokenizer.decode(output[0])

# Extrair tool call (6 fallbacks, tolerante a truncamento)
tc = parse_tool_call(response, TOOLS)
if tc:
    print(f"Tool: {tc.name}, Args: {tc.arguments}")
```

---
## Testes

### C++ (kernels algébricos)

```bash
cd build && ctest --output-on-failure
# esperado: 15/15 PASS (default CI)
# ou 16/16 com -DBITNET_ENABLE_ACDC_RECT=ON
```

Cobre: kernel L1-L5 (WHT, FWHT, ACDC, tropical, HRR, K_i8 cache),
property-based tests com 100-1000 iters cada.

### Python (tool-calling)

```bash
cd bitnet-studio

# Teste rápido (12 testes, ~10 min)
python test_3x.py

# Teste exaustivo (72 testes, ~3h) — salva progresso em arquivo
python test_50x_file.py
# Resultado: test_50x_progress.log + test_50x_results.json
```

---
## Documentação

### Decisão e arquitetura

- [`ROADMAP.md`](ROADMAP.md) — Roadmap público
- [`docs/decision-matrix.md`](docs/decision-matrix.md) — Quando usar L1/L3/L4/L5
- [`docs/hardware-compatibility.md`](docs/hardware-compatibility.md) — Matriz CPU → modo
- [`docs/invariants.md`](docs/invariants.md) — P1-P7 canônicas
- [`docs/findings-cpu-universal.md`](docs/findings-cpu-universal.md) — Validação empírica

### Teoria (referência acadêmica)

- [`docs/theory/00-index.md`](docs/theory/00-index.md) — Índice
- [`docs/theory/01-ternary-algebra.md`](docs/theory/01-ternary-algebra.md) — Quantização ternária
- [`docs/theory/02-wht-decomposition.md`](docs/theory/02-wht-decomposition.md) — WHT
- [`docs/theory/03-acdc-structured-layers.md`](docs/theory/03-acdc-structured-layers.md) — ACDC
- [`docs/theory/04-tropical-algebra.md`](docs/theory/04-tropical-algebra.md) — Semiring (max,+)
- [`docs/theory/05-holographic-memory.md`](docs/theory/05-holographic-memory.md) — HRR
- [`docs/theory/06-5-levels.md`](docs/theory/06-5-levels.md) — Sumário 1 página

### Walkthroughs

- [`examples/medical_offline.md`](examples/medical_offline.md) — Médico
- [`examples/legal_offline.md`](examples/legal_offline.md) — Advogado
- [`examples/finance_offline.md`](examples/finance_offline.md) — Financeiro

---
## Arquitetura do código

### C++ (inferência 1.58-bit)

```
src/
  ggml-bitnet-mad.cpp      ← Kernel I2_S (AVX2 + NEON), L1
  ggml-bitnet-lut.cpp      ← Kernels TL1/TL2 lookup-table, L1
  ggml-bitnet-wht.cpp      ← WHT zero-multiplicação, L2
  ggml-bitnet-fwht.cpp     ← FWHT + ACDC O(n log n), L3
  ggml-bitnet-tropical.cpp ← Atenção tropical (max,+), L4
  ggml-bitnet-hrr.cpp      ← Memória holográfica, L5
  ggml-bitnet-dispatch.cpp ← Dispatch L3-L5
  ggml-bitnet-kv-cache.cpp ← K_i8 cache
  ggml-bitnet-common.cpp   ← Utilitários

include/                   ← Headers L1-L5
utils/                     ← Benchmarks L1-L5
```

### Python (BitNet Studio)

```
bitnet-studio/
├── studio/server/
│   ├── tool_engine.py     ← Parser 6 fallbacks de tool_call
│   ├── mcp_bridge.py      ← Integração MCP (protheus-rag)
│   └── inference.py       ← Geração com adapter
├── finetune_local.py      ← Fine-tune QLoRA CPU
├── test_*.py              ← Testes de extração e acurácia
└── adapters/              ← Checkpoints QLoRA
```

---
## Restrições fundadoras

- **CPU only** — GPU kernels proibidos (NO-02)
- **Sem cloud, sem telemetria** (NO-06, NO-07)
- **Sem mudança no formato GGUF** (NO-03)
- **Patches vendored** — `3rdparty/llama.cpp/` read-only

## Licença

MIT — ver [`LICENSE`](LICENSE).

---

*v3.0 — README reescrito em 2026-06-12.*
*v2 → v3: adicionado BitNet Studio, Falcon3 adapter, tool-calling PT-BR,
parser robusto de JSON truncado, protocolo mem0 cross-agent.*
