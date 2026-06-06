# Scout — Inventário de Superfície (BitNet)

> Mapeamento superficial do projeto. Gerado em 2026-06-05 pelo agente `reversa-scout`.
> Não toca em artefatos pré-existentes (`_reversa_sdd/`, `.reversa/context/`).

## Identidade

| Campo | Valor |
|-------|-------|
| Nome | BitNet (fork) |
| Origem | microsoft/BitNet |
| Remoto | https://github.com/peder1981/BitNet.git |
| Licença | MIT |
| Branch atual | main |
| Último commit | `a884036` — *build(tests): wire all 4 kernel unit tests into CMake + CI* (2026-06-05 22:45) |
| Working tree | clean (único untracked: `_reversa_sdd/session-2025-06-05-tropical-attn.md`, imutável) |
| Propósito | Inferência CPU de LLMs com pesos ternários {-1, 0, +1} + extensões algébricas (WHT, ACDC, tropical, HRR) |

## Estrutura de pastas (top-level)

```
BitNet/
├── 3rdparty/llama.cpp/        # 65M — submódulo (fork, branch merge-dev)
├── build_test/                 # 4,2M — artefatos locais de build (não versionado)
├── docs/                       # 1.508 linhas — teoria matemática (5 níveis)
├── include/                    # 921 linhas — headers públicos (8 arquivos)
├── preset_kernels/             # 5.807 linhas — kernels LUT pré-gerados (3 modelos)
├── src/                        # ~2.585 linhas — kernels C++ (7 arquivos + README + CMakeLists)
├── utils/                      # 8.189 linhas — 19 scripts Python + 2 .sh
├── _reversa_sdd/               # artefatos Reversa (não modificar)
├── .reversa/                   # working dir Reversa (não modificar)
├── CLAUDE.md                   # guia do projeto para agentes
├── CMakeLists.txt              # 99 linhas — build root
├── README.md                   # 247 linhas
├── requirements.txt            # 11 linhas — só re-exporta do 3rdparty
├── run_inference.py            # 55 linhas — CLI CPU
├── run_inference_server.py     # 64 linhas — HTTP server
├── setup_env.py                # 244 linhas — orquestrador de setup
└── SECURITY.md
```

## Contagem de arquivos (excluindo `.git`, `3rdparty/`, `build_test/`, `.reversa/`, `_reversa_sdd/`, `__pycache__/`)

| Extensão | Contagem | Categoria |
|----------|---------:|-----------|
| `.py`    | 19 | Scripts Python (utils + entry points + setup) |
| `.h`     | 13 | Headers C/C++ (include + preset_kernels) |
| `.md`    | 12 | Documentação (docs, README, SECURITY, CLAUDE) |
| `.ini`   | 6 | (não significativos — provavelmente placeholders) |
| `.cpp`   | 6 | Kernels C++ CPU |
| `.txt`   | 3 | LICENSE, requirements, etc. |
| `.sh`    | 2 | Scripts de teste (utils/) |
| outros   | 3 | .gitignore, .gitmodules, LICENSE |
| **Total**| **64** | |

## Linguagens detectadas (linhas de código — top 10)

| Linguagem | LOC (aprox.) | Onde |
|-----------|------------:|------|
| C++       | ~2.270 | `src/ggml-bitnet-*.cpp` |
| C/C++ header | ~6.622 | `include/`, `preset_kernels/` |
| Python    | ~8.504 | `utils/*.py`, `run_inference*.py`, `setup_env.py` |
| Markdown  | ~1.755 | `docs/`, `README.md`, `CLAUDE.md`, `SECURITY.md` |
| CMake     | ~140 | `CMakeLists.txt`, `src/CMakeLists.txt` |
| Shell     | ~724 | `utils/test_gemm_kernel.sh`, `utils/test_power.sh` |
| **Total estimado** | **~20.015** | (sem contar 3rdparty/llama.cpp) |

## Frameworks e bibliotecas

| Camada | Tecnologia | Origem | Versão |
|--------|------------|--------|--------|
| Build | CMake | `CMakeLists.txt` | ≥ 3.14 (root) / ≥ 3.22 (CLAUDE.md) |
| Compilador | Clang (obrigatório para SIMD) | doc | ≥ 18 |
| Compilador alt. | GCC com `-fpermissive` | `CMakeLists.txt:40-42` | tolerado |
| Compilador proibido | MSVC | `src/CMakeLists.txt` | nunca |
| Backend CPU | llama.cpp (fork) | submódulo | branch `merge-dev` |
| Conversão | `llama-quantize` | herdado do llama.cpp | — |
| Tokenizer | tiktoken (Llama 3) | herdado | — |
| HuggingFace | `huggingface-cli` | `setup_env.py` | — |
| Gerenciador recomendado | conda | `README.md` | — |
| Python mínimo | — | `README.md` | 3.9 |

## Pontos de entrada

| Caminho | Tipo | Descrição |
|---------|------|-----------|
| `run_inference.py` | app_entry | CLI CPU; monta `llama-cli` via subprocess com `-ngl 0 -b 1` hardcoded |
| `run_inference_server.py` | app_entry | Servidor HTTP OpenAI-compatible; monta `llama-server` |
| `setup_env.py` | setup_entry | Orquestrador: download HF → convert → codegen → compile |

## Configuração e build

| Arquivo | Função |
|---------|--------|
| `CMakeLists.txt` | Top-level: define flags `BITNET_ARM_TL1`, `BITNET_X86_TL2`, `BITNET_L2_WHT`, `BITNET_L3_ACDC`, `BITNET_L4_TROPICAL`, `BITNET_L5_HRR` (defaults L2-L5 ON) |
| `src/CMakeLists.txt` | Compila L2-L5 como `bitnet_math` OBJECT library; L1 fica hardcoded dentro de 3rdparty/llama.cpp |
| `include/gemm-config.h` | Parâmetros de tuning I2_S (ROW_BLOCK_SIZE, COL_BLOCK_SIZE, PARALLEL_SIZE, ACT_PARALLEL) |
| `requirements.txt` | Apenas re-exporta 5 arquivos do `3rdparty/llama.cpp/requirements/` |
| `.gitmodules` | 1 submódulo: `3rdparty/llama.cpp` (url, branch) |
| `.gitignore` | Exclui `models/`, `build/`, `*.gguf`, `*.o`, etc. |
| `preset_kernels/<model>/` | Headers pré-gerados por modelo (3 modelos: 3B, large, Llama3-8B) |

## CI/CD

`.github/workflows/ci.yml` — kernel-ci workflow (commit b536d83, estendido em a884036):
- Trigger: push main, PR, manual dispatch
- Runner: ubuntu-24.04 + clang-18 + libstdc++-14-dev + ninja
- Build: Release com L2-L5 + tests=ON
- ctest: roda 4 suites (test_wht, test_acdc, test_tropical, test_hrr_cleanup)
- Sem smoke/perplexity (modelo é 1.18GB, downloads fora do escopo)

## Docker

Nenhum `Dockerfile` ou `docker-compose.yml` foi encontrado.

## Schema de banco de dados

Nenhum DDL, migration, schema ORM, model SQLAlchemy/Prisma/Django presente. O projeto não usa banco de dados — modelo é carregado de arquivos `.gguf` estáticos.

## Cobertura de testes

| Sinal | Valor |
|-------|-------|
| Framework de teste | **sem framework formal** — testes C++ standalone via compilação direta (não gtest) |
| **Testes unitários C++ (novo)** | **4 arquivos** — `test_wht.cpp`, `test_acdc.cpp`, `test_tropical.cpp`, `test_hrr_cleanup.cpp` (20/20 subtests PASS) |
| Test runner | **ctest** (CMake), wired em `tests/CMakeLists.txt` |
| Arquivos `test_*.py` | 1 — `utils/test_perplexity.py` |
| Scripts `.sh` de teste | 2 — `utils/test_gemm_kernel.sh`, `utils/test_power.sh` |
| Benchmarks | 7 — `acdc_benchmark.py`, `e2e_benchmark.py`, `hrr_benchmark.py`, `tropical_benchmark.py`, `tune_gemm_config.py`, `wht_benchmark.py`, `test_perplexity.py` |
| Cobertura estimada | **boa para kernels C++** (5/5 por nível, L2-L5); mínima para dispatch end-to-end (smoke manual) |

## Módulos identificados (nível superficial)

### CLI / Setup (Python top-level)
- `run_inference` — CLI de inferência CPU
- `run_inference_server` — Servidor HTTP
- `setup_env` — Orquestrador de setup completo

### Kernels C++ CPU (`src/`)
| Módulo | Arquivo | LOC | Função |
|--------|---------|----:|--------|
| L1 I2_S MAD | `ggml-bitnet-mad.cpp` | 1.055 | Kernel SIMD AVX2/NEON para 2-bit packed |
| L1 LUT (TL1/TL2) | `ggml-bitnet-lut.cpp` | (não contado) | LUT para ARM64/x86_64 |
| L2 WHT | `ggml-bitnet-wht.cpp` | 467 | Decomposição WHT zero-multiplicação (AVX2) + I2_S packing |
| L3 ACDC | `ggml-bitnet-fwht.cpp` | 481 | FWHT + diagonal O(n log n) + `acdc_gemv` retangular + `acdc_project` |
| L4 Tropical | `ggml-bitnet-tropical.cpp` | 391 | Atenção (max,+) semiring |
| L5 HRR | `ggml-bitnet-hrr.cpp` | (incl. header 326) | FFT Cooley-Tukey radix-2 + HRR bind/unbind + Frady 2021 cleanup_iter (NAIVE + RESIDUAL) |
| **Dispatch (L2-L5)** | `ggml-bitnet-dispatch.cpp` | 408 | Wrappers `bitnet_op_*` via `ggml_map_custom1/2/3`; ACDC GEMV (lazy proj init), tropical 3D GQA, HRR 3D GQA com Frady 2021 cleanup opcional |

### Headers (`include/`)
- `ggml-bitnet.h` (49) — API principal L1
- `ggml-bitnet-wht.h` (84) — API L2
- `ggml-bitnet-fwht.h` (148) — API L3
- `ggml-bitnet-tropical.h` (180) — API L4
- `ggml-bitnet-hrr.h` (326) — API L5 (incl. `hrr_cleanup_iter` Frady 2021)
- **`ggml-bitnet-dispatch.h` (106) — wrappers `bitnet_op_acdc/tropical_attn/hrr_attn`** (NOVO commit 129557d)
- `bitnet-lut-kernels.h` (25) — placeholder
- `gemm-config.h` (35) — tuning

### Utils (`utils/`)
- **Conversão** (4): `convert.py`, `convert-helper-bitnet.py`, `convert-hf-to-gguf-bitnet.py`, `convert-ms-to-gguf-bitnet.py`
- **Codegen** (2): `codegen_tl1.py`, `codegen_tl2.py`
- **Benchmarks** (7): `acdc`, `e2e`, `hrr`, `tropical`, `tune_gemm_config`, `wht`, `test_perplexity`
- **Embeddings** (1): `quantize_embeddings.py`
- **Preprocess** (2): `preprocess-huggingface-bitnet.py`, `generate-dummy-bitnet-model.py`
- **Testes shell** (2): `test_gemm_kernel.sh`, `test_power.sh`

## Sinais para organização das specs

| Sinal | Encontrado? | Evidência |
|-------|-------------|-----------|
| Roteamento centralizado (URLs/Routes) | Não | projeto CLI, não servidor web de aplicação |
| Pastas top-level com nomes de domínio | Parcial | `src/`, `include/`, `utils/`, `preset_kernels/`, `docs/` (organização por papel técnico, não domínio) |
| Specs Gherkin/BDD | Não | nenhum `*.feature`, `cypress/`, `playwright/` |
| Múltiplos sinais dominantes | Não | — |
| **Sugestão de organização** | **`module`** | organização por papel técnico, sem domínio de negócio explícito |

## Notas para próximos agentes

1. **Submódulo 3rdparty/llama.cpp** é fork customizado (branch `merge-dev`). Tratar como read-only a menos que patch intencional.
2. **L2-L5 estão agora completamente conectados ao dispatch do llama.cpp**: L4 tropical e L5 HRR via `bitnet_op_tropical_attn`/`bitnet_op_hrr_attn` em `llm_build_kqv` (`3rdparty/llama.cpp/src/llama.cpp:9797-9857`); L3 ACDC via `bitnet_op_acdc_gemv` em `llm_build_ffn_acdc_bitnet` (env `BITNET_ACDC_FFN=1`); L2 WHT patched em `ggml_vec_dot_i2_i8_s` (Hadamard no lugar de maddubs). L5 também tem `bitnet_op_hrr_attn_with_cleanup` (Frady 2021 RESIDUAL, `BITNET_HRR_ATTN_CLEANUP=N` iters). Dispatch chain L2-L5 **completo**.
3. **GPU foi removida** deste fork — não há `gpu/`, mas o contexto Reversa herdado (gerado em 2026-05-03) menciona `gpu/model.py`, `gpu/generate.py` etc. Esses módulos **não existem mais** — a análise arqueológica prévia pode estar obsoleta. Lacuna a validar.
4. **Testes unitários C++** — suíte completa. 4 arquivos (`test_wht.cpp`, `test_acdc.cpp`, `test_tropical.cpp`, `test_hrr_cleanup.cpp`) cobrem 5/5 subtests cada = **20/20 PASS**. Wired em `tests/CMakeLists.txt` + `.github/workflows/ci.yml`. Benchmarks Python (`utils/*_benchmark.py`) verificam corretude numérica independente.
5. **2 bugs reais encontrados nos kernels** (commits e7edb21, ed6fbde):
   - `wht_dot_avx2` em `src/ggml-bitnet-wht.cpp:186-189` tinha labels `g0..g3` invertidas vs `unpack_i2s_block` da própria lib. Os testes do próprio arquivo (ggml_wht_verify) também falhavam — bug latente.
   - `acdc_forward_i8` em `src/ggml-bitnet-fwht.cpp:291-303` tinha stray 1/n² que violava a spec do CLAUDE.md ("unnormalized — no 1/n² factors"). A diagonal d absorve o scale quando aprendida no treino.
6. **`build_test/`** é artefato de build local não versionado (4,2M) — ignorar. Adicionado `build_tests/` ao `.gitignore` (a884036) para quick-iteration builds.
7. **Idioma do projeto**: comentários e docs majoritariamente em **português-BR** (CLAUDE.md, README, commits). Mensagens de log também em PT-BR.
