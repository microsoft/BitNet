# Scout — Dependências (BitNet)

> Snapshot de dependências declaradas. Gerado em 2026-06-05 pelo `reversa-scout`.

## Python (top-level)

Arquivo: `requirements.txt` (11 linhas, todas re-exports)

```python
-r 3rdparty/llama.cpp/requirements/requirements-convert_legacy_llama.txt
-r 3rdparty/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
-r 3rdparty/llama.cpp/requirements/requirements-convert_hf_to_gguf_update.txt
-r 3rdparty/llama.cpp/requirements/requirements-convert_llama_ggml_to_gguf.txt
-r 3rdparty/llama.cpp/requirements/requirements-convert_lora_to_gguf.txt
```

**Não há `pyproject.toml`, `setup.py`, `Pipfile` ou `poetry.lock`** — a única fonte de deps Python é o requirements.txt (que delega ao 3rdparty).

### Dependências efetivas (importadas em `utils/*.py` e top-level)

| Módulo | Usado em | Função |
|--------|----------|--------|
| `subprocess` | run_inference, run_inference_server, setup_env | Spawn de llama-cli/llama-server/CMake |
| `argparse` | run_inference, run_inference_server, setup_env, e2e_benchmark, vários utils | CLI parsing |
| `platform` | run_inference | Detecção arm64/x86_64 |
| `shutil` | setup_env | file ops |
| `pathlib` (Path) | setup_env | paths |
| `json` | vários utils | config |
| `os` / `sys` | todos | env, exit |
| `numpy` (implícito) | convert.py, codegen_tl*, *_benchmark | tensores, matrizes |
| `torch` (implícito, herdado) | — | não usado neste fork (GPU removida) |
| `huggingface-cli` (externo) | setup_env | download modelos |

**Nota**: o ambiente conda recomendado no README é `conda create -n bitnet python=3.9 -y`, mas as deps concretas precisam ser instaladas via `pip install -r requirements.txt` que herda tudo do llama.cpp (numpy, sentencepiece, transformers, gguf-python, safetensors, etc.).

## C/C++

Nenhum gerenciador de pacotes C/C++ (vcpkg, conan, hunter). Dependências são resolvidas via CMake.

### Externas (ligadas via CMake)

| Dependência | Fonte | Uso |
|-------------|-------|-----|
| Threads (pthread) | `find_package(Threads REQUIRED)` (CMakeLists.txt:44) | paralelismo |
| llama.cpp (fork) | `add_subdirectory(3rdparty/llama.cpp)` | backend de inferência |
| ggml (do llama.cpp) | propagado via llama.cpp | tensor library + SIMD intrinsics |

### SIMD intrinsics usadas (em `src/ggml-bitnet-*.cpp`)

| Intrinsic | Arquitetura | Kernel |
|-----------|-------------|--------|
| `_mm256_*` (AVX2) | x86_64 | I2_S MAD, TL2 LUT |
| `_mm512_*` (AVX-512) | x86_64 | I2_S MAD (opcional) |
| NEON intrinsics | ARM64 | I2_S MAD, TL1 LUT |
| (genéricos) | qualquer | L2-L5 (não-SIMD no nível de operação; WHT/ACDC/tropical/HRR são bitwise) |

## Submódulos Git

| Path | URL | Branch |
|------|-----|--------|
| `3rdparty/llama.cpp` | https://github.com/Eddie-Wang1120/llama.cpp.git | `merge-dev` |

**Importante**: este llama.cpp é um fork customizado de `Eddie-Wang1120/llama.cpp` (não upstream). Tratar como dependência patcheável.

## Binários externos invocados (via subprocess)

| Binário | Quem chama | Função |
|---------|------------|--------|
| `build/bin/llama-cli` | `run_inference.py` | inferência CLI |
| `build/bin/llama-server` | `run_inference_server.py` | servidor HTTP |
| `huggingface-cli` | `setup_env.py` | download modelos |
| `cmake`, `clang`, `clang++` | `setup_env.py` | compilação |
| `llama-quantize` | `setup_env.py` | quantização GGUF |
| `python utils/codegen_tl*.py` | `setup_env.py` | geração de kernels LUT |
| `python utils/convert-helper-bitnet.py` | `setup_env.py` | conversão safetensors → GGUF |
| `python utils/quantize_embeddings.py` | `setup_env.py` | quantização Q6_K de embeddings |

## Modelos HuggingFace suportados (`setup_env.py`)

(segundo CLAUDE.md — `SUPPORTED_HF_MODELS` = 16 modelos; lista parcial documentada no README)

| Modelo | Parâmetros |
|--------|-----------:|
| BitNet-b1.58-2B-4T | 2.4B |
| bitnet_b1_58-large | 0.7B |
| bitnet_b1_58-3B | 3.3B |
| Llama3-8B-1.58-100B-tokens | 8.0B |
| Falcon3-1B/3B/7B/10B-Instruct/Base | 1B–10B |
| Falcon-E-1B/3B-Instruct/Base | 1B–3B |

## Formatos de quantização suportados

| Formato | Plataforma | Ativação |
|---------|------------|----------|
| `i2_s` (2-bit packed) | x86_64 + ARM64 | default |
| `tl1` (LUT) | ARM64 only | flag `-DBITNET_ARM_TL1=ON` |
| `tl2` (LUT) | x86_64 only | flag `-DBITNET_X86_TL2=ON` |
| Q6_K (embeddings) | qualquer | flag `--quant-embd` |

## Flags de build disponíveis (`CMakeLists.txt`)

| Flag | Padrão | Função |
|------|--------|--------|
| `BITNET_ARM_TL1` | OFF | Ativa kernel TL1 (ARM64) |
| `BITNET_X86_TL2` | OFF | Ativa kernel TL2 (x86_64) |
| `BITNET_L2_WHT` | ON | WHT zero-multiplicação (Level 2) |
| `BITNET_L3_ACDC` | ON | FWHT + ACDC (Level 3) |
| `BITNET_L4_TROPICAL` | ON | Atenção tropical (Level 4) |
| `BITNET_L5_HHR` | ON | Memória holográfica (Level 5) |

## Versões mínimas

| Componente | Versão | Fonte |
|------------|--------|-------|
| CMake | 3.14 (root) / 3.22 (CLAUDE.md) | `CMakeLists.txt:1` / `CLAUDE.md` |
| Clang | ≥ 18 | `CLAUDE.md` |
| Python | ≥ 3.9 | `README.md` |
| GCC | qualquer (com `-fpermissive`) | `CMakeLists.txt:40-42` |
| MSVC | **proibido** | `src/CMakeLists.txt` |
