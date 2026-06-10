# Contributing to BitNet CPU-Universal

Obrigado pelo interesse em contribuir! Este documento cobre:
- Como configurar o ambiente de desenvolvimento
- Como rodar os testes
- Política de PR
- Restrições do projeto (§3 do ROADMAP)

---

## Configuração do ambiente

### Requisitos

- Linux x86_64 (testado em Ubuntu 22.04 / 24.04) ou macOS ARM64
- CMake ≥ 3.20, Ninja, Clang-18 (preferido) ou GCC ≥ 13
- Python 3.10+ com `numpy`, `scipy`, `safetensors`
- Sem CUDA — é CPU-only por design

### Build de desenvolvimento

```bash
git clone --recursive https://github.com/peder1981/BitNet.git
cd BitNet

# Aplicar patch do dispatch no submodule
bash scripts/apply-dispatch-patches.sh

# Criar venv Python
python3 -m venv .venv
.venv/bin/pip install numpy scipy safetensors

# Configurar e compilar
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBITNET_L2_WHT=ON \
  -DBITNET_L3_ACDC=ON \
  -DBITNET_L4_TROPICAL=ON \
  -DBITNET_L5_HRR=ON \
  -DBITNET_L6_RAG=ON \
  -DBITNET_BUILD_TESTS=ON \
  -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python3
cmake --build build -j$(nproc)
```

---

## Rodando os testes

```bash
# Todos os testes (16/16 esperado)
ctest --test-dir build --output-on-failure -j$(nproc) --timeout 120

# Um teste específico
ctest --test-dir build -R test_acdc --output-on-failure

# Cross-validação C++ ↔ Python (kernels L3/L4/L5)
.venv/bin/python3 tests/cross_validation.py --all --build-dir build/tests
```

Os testes devem passar **sem modelo** — são unit tests dos kernels matemáticos, cada um termina em < 1s.

---

## Política de PR

### O que aceitamos

- Correções de bugs com test de regressão
- Melhorias de performance nos kernels L1-L5 (com benchmark antes/depois)
- Documentação e exemplos
- Portabilidade: ARM64 NEON, RISC-V, etc.
- Novos kernels algébricos (L6+) com motivação matemática clara

### O que NÃO aceitamos (§3 do ROADMAP)

| Restrição | Motivo |
|-----------|--------|
| **Sem CUDA/ROCm/Metal** | Persona D4 — CPU-only por design |
| **Sem telemetria** | Persona D4 — zero coleta de dados |
| **Sem chamadas de rede** | Air-gapped por contrato |
| **Sem cloud inference** | Soberania de dados |
| **Sem retreino online** | Sem GPU no target deployment |

### Fluxo de PR

1. Fork → branch `feat/nome-da-feature` ou `fix/nome-do-bug`
2. Rodar `ctest` localmente (16/16 obrigatório)
3. Rodar `cross_validation.py --all` se tocar L3/L4/L5
4. Abrir PR com:
   - Descrição do problema que resolve
   - Benchmark ou test demonstrando a melhora
   - Confirmação de que nenhuma restrição §3 é violada
5. CI deve passar (GitHub Actions) antes do merge

### Commits

Seguir [Conventional Commits](https://www.conventionalcommits.org/):
```
feat(l3): descrição
fix(ci): descrição
docs: descrição
test(l4): descrição
perf(l2): descrição
```

---

## Estrutura do projeto

```
src/                  # Kernels C++ (L1-L6)
  ggml-bitnet-*.cpp   # Um arquivo por nível
  ggml-bitnet-dispatch.cpp  # Dispatcher de ops customizadas
tests/                # Unit tests (sem modelo, < 1s cada)
utils/                # Scripts Python de extração/benchmark
patches/llama.cpp/    # Patch combinado do dispatch
scripts/              # apply-dispatch-patches.sh
include/              # Headers públicos
3rdparty/llama.cpp    # Submodule (base: Eddie-Wang1120/llama.cpp@merge-dev)
```

---

## Issues

- **Bugs:** abrir issue com output do `ctest --output-on-failure` e `cmake --version`
- **Feature requests:** descrever o caso de uso D4 que motiva
- **Good first issues:** marcados com `good first issue` no GitHub
  - Benchmark em novo hardware (AMD Ryzen, Intel Celeron, ARM64)
  - Documentação em inglês
  - ARM64 NEON path para WHT/ACDC

---

## Código de conduta

Este projeto segue o princípio de que **hardware acessível = IA acessível**.
Contribuições que aumentam barreiras de hardware (exigem GPU, CUDA, cloud)
serão recusadas independentemente da qualidade técnica.
