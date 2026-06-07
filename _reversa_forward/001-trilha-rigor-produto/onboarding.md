# Onboarding — `001-trilha-rigor-produto`

> Passo a passo executável para um humano (ou agente) que vai **testar a feature pela primeira vez**.
> Foco em **privacidade/soberania** (persona D4): o usuário roda tudo local, sem internet, em laptop corporativo padrão.
>
> **Versão:** v1 (gerado por reversa-plan em 2026-06-06)
> **Audiência:** contribuidor novo, usuário piloto de saúde/jurídico/financeiro, agente de IA em `/reversa-coding`
> **Pré-requisito:** Linux x86_64 (idealmente com AVX2), 8 GB RAM mínimo, 30 GB de disco livre

---

## 1. Antes de Começar

### 1.1. Leia primeiro (15 min)

Em ordem:

1. `CLAUDE.md` (raiz do projeto) — restrições, build, convenções.
2. `requirements.md` v2 (`_reversa_forward/001-trilha-rigor-produto/requirements.md`) — especialmente seção `## 9. Persona Alvo`.
3. `roadmap.md` v1 (`_reversa_forward/001-trilha-rigor-produto/roadmap.md`) — decisões, deltas, riscos.
4. `docs/findings-cpu-universal.md` — writeup técnico de 5 níveis, 4 bugs, bench.
5. `.reversa/scout/principles.md` — 7 princípios transversais.

### 1.2. Verifique seu hardware

```bash
# 1. CPU suporta AVX2?
grep -o 'avx2' /proc/cpuinfo | head -1
# Esperado: 'avx2' (qualquer Intel/AMD de 2013+)

# 2. Memória disponível
free -h | grep 'Mem:' | awk '{print "RAM total: " $2 ", disponível: " $7}'
# Esperado: ≥ 8 GB total, ≥ 4 GB disponível

# 3. Disco livre
df -h . | tail -1 | awk '{print "Livre: " $4}'
# Esperado: ≥ 30 GB (modelo + build artifacts)

# 4. Clang ≥ 18?
clang++ --version | head -1
# Esperado: 'clang version 18.x' ou superior
# Se menor: instalar via 'sudo apt install clang-18' (Ubuntu) ou equivalente

# 5. CMake?
cmake --version | head -1
# Esperado: ≥ 3.20

# 6. Python 3.10+?
python3 --version
# Esperado: 'Python 3.10.x' ou superior

# 7. Tem rede? (decida: online ou air-gapped)
ping -c 1 huggingface.co >/dev/null 2>&1 && echo "ONLINE" || echo "OFFLINE"
# Se OFFLINE: só teste build, não baixe modelo
```

### 1.3. Clone e submodule

```bash
# Se ainda não tem o repo
git clone https://github.com/peder1981/BitNet.git
cd BitNet

# Submodule (llama.cpp fork)
git submodule update --init --recursive
# Demora 1-2 min; sem isso build falha
```

---

## 2. Primeiro Build (10-30 min, online)

### 2.1. Setup ambiente conda

```bash
# Criar env conda (BitNet usa 'bitnet-cpp' por convenção)
conda create -n bitnet-cpp python=3.10 -y
conda activate bitnet-cpp

# Dependências Python
pip install numpy safetensors huggingface_hub

# Dependências C++ (build essentials; Ubuntu)
sudo apt install build-essential cmake libstdc++-13-dev clang-18
# Em outros distros: equivalente
```

### 2.2. Setup automatizado (recomendado para v0.1)

```bash
# Baixa modelo + converte + codegen + compila (passo único)
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# Se ARM64 (Apple Silicon, AWS Graviton): troque -q para tl1
# Se x86_64 com kernel LUT: troque para tl2
```

**Tempo esperado:**
- Download do modelo: 5-10 min (1.5 GB safetensors + 700 MB GGUF)
- Conversão safetensors → GGUF: 2-3 min
- Codegen dos kernels: 10-30 s
- Compilação: 5-15 min (mais longo na primeira vez)

**Verificação:**

```bash
# Binário principal existe?
ls -la build/bin/llama-cli
# Esperado: arquivo ELF executável, ~10-30 MB

# Binário do servidor?
ls -la build/bin/llama-server
# Esperado: similar
```

### 2.3. Setup manual (avançado, para reproduzir kernel headers)

```bash
# 1. Aplicar patches vendored (se ainda não aplicados)
bash scripts/apply-dispatch-patches.sh
# Deve reportar: '3 patches applied (L3, L5, L4)'

# 2. cmake
cmake -B build \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release
# Se GCC < 14 (Ubuntu 24.04 com libstdc++-13-dev), adicionar:
#   -DCMAKE_CXX_FLAGS="-I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13" \
#   -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13" \
#   -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13"

# 3. Build
cmake --build build --config Release -j$(nproc)
```

---

## 3. Primeiro Teste (5 min)

### 3.1. CTest (testes de unidade)

```bash
# Configurar testes (passo separado, requer Python3 com Interpreter)
cmake -B build_tests -S tests \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DBITNET_TEST_USE_BITNET_LIBS=ON
cmake --build build_tests -j$(nproc)

# Rodar todos
cd build_tests && ctest --output-on-failure && cd ..
# Esperado: '100% tests passed, 0 tests failed' (≥ 9/9, ≥ 60 subtests após M1)
```

**Testes individuais para investigar:**

```bash
# Property-based tests (após M1)
./build_tests/test_acdc_properties --success
# Esperado: roda 1000+ iterações sem falhar

# Cross-validation C ↔ Python (após M2)
cd tests && python3 cross_validation.py
# Esperado: 'all close' em todos os kernels
```

### 3.2. Inferência end-to-end (BitNet-2B)

```bash
# CPU-only (default; -ngl 0 hardcoded)
python run_inference.py \
  -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "The capital of France is" \
  -n 50 -t 4

# Esperado: 
#   - Gera 50 tokens em ~10-20 segundos
#   - Texto coerente ("...Paris, which is...")
#   - Final linha: 'total time = X.XX s, X.XX tokens per second'
```

### 3.3. Atenção esparsa (D-T-01: opt-in)

```bash
# SEM env var: usa attention denso
python run_inference.py -m models/.../ggml-model-i2_s.gguf \
  -p "Once upon a time" -n 50 -t 4

# COM env var: opt-in para L4 sparse
BITNET_SPARSE_TOPK=32 python run_inference.py -m models/.../ggml-model-i2_s.gguf \
  -p "Once upon a time" -n 50 -t 4

# Comparar: o segundo deve ser mais rápido (5-15% em n_ctx=512) mas pode
# ter qualidade marginalmente inferior em prompts específicos.
```

---

## 4. Testes de Persona D4 (após M5)

### 4.1. Air-gapped boot (AC-11)

```bash
# Ativa namespace de rede sem interfaces
# Tudo que tente connect() falha silenciosamente
unshare -rn bash -c './build/bin/llama-cli -m models/.../ggml-model-i2_s.gguf \
  -p "Test" -n 10 2>&1' | tee /tmp/air_gapped_log.txt

# Verificação:
grep -iE 'error|warning|telemetry|upload' /tmp/air_gapped_log.txt
# Esperado: NENHUMA linha (sem erros, sem warnings de rede, sem telemetria)

# Se aparecer: investigar com strace
strace -e network -f -o /tmp/strace.log \
  ./build/bin/llama-cli -m models/.../ggml-model-i2_s.gguf -p "Test" -n 10
grep -E 'connect|sendto|getaddrinfo' /tmp/strace.log | head -20
# Esperado: vazio (ou só DNS lookup de libc init, aceitável)
```

### 4.2. Compatibilidade de hardware (AC-13)

```bash
# Listar CPUs suportadas e modo recomendado
cat docs/hardware-compatibility.md
# (a criar em M5)

# Teste em CPU pré-AVX2 (opcional, se disponível)
# Se você tem um laptop de 2012-2013 sem AVX2:
cmake -B build_noavx2 -DCMAKE_CXX_FLAGS="-mno-avx2" ...
cmake --build build_noavx2 -j$(nproc)
# Esperado: build com warning, mas funcional; performance degradada
```

### 4.3. Cenários de uso D4 (após M5)

Cada cenário é um script de smoke test. Não são automatizados; são **walkthroughs manuais** para validar que o produto atende a persona:

```bash
# examples/medical_offline.md
# "Dr. Silva, médico, analisa prontuário em laptop de consultório"
# 1. Desconecta Wi-Fi
sudo nmcli networking off
# 2. Roda inferência
python run_inference.py -m models/.../ggml-model-i2_s.gguf \
  -p "Resuma o seguinte prontuário: paciente com diabetes tipo 2..." \
  -n 200 -t 4
# 3. Verifica: texto coerente, sem requests de rede
# 4. Reconecta
sudo nmcli networking on
```

```bash
# examples/legal_offline.md
# "Dra. Oliveira, advogada, resume petição em escritório"
# Similar: sem rede, inferência local, ~30s para 200 tokens
```

```bash
# examples/finance_offline.md
# "Carlos, analista financeiro, categoriza despesas"
# Similar: sem rede, inferência local
```

---

## 5. Sanity Checks Comuns (Troubleshooting)

### 5.1. Build falha com "GCC 14 stdlib not found"

Sintoma:
```
fatal error: 'bits/stdc++.h' file not found
```

Causa: Clang 18 no Ubuntu 24.04 padrão usa headers GCC 14; se só tem `libstdc++-13-dev` instalado.

Fix (do CLAUDE.md):
```bash
cmake -B build \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13" \
  -DCMAKE_BUILD_TYPE=Release
```

### 5.2. Patches não aplicam

Sintoma: `git apply` falha ou build não tem kernels L3/L4/L5.

Fix:
```bash
# Verificar se patches estão aplicados
bash scripts/apply-dispatch-patches.sh --check

# Se não: aplicar
bash scripts/apply-dispatch-patches.sh

# Se já estão: --reverse e re-apply
bash scripts/apply-dispatch-patches.sh --reverse
bash scripts/apply-dispatch-patches.sh
```

### 5.3. ctest reporta 0 testes

Sintoma: `ctest` não encontra nada, retorna "No tests were found".

Causa: `build_tests/` não foi configurado. Ver seção 3.1.

### 5.4. Inferência gera texto incoerente

Sintoma: perplexity > 100 ou texto repetitivo.

**Não é regressão desta feature** (v0.1 não muda inferência). Mas investigar:

```bash
# Validar modelo
python utils/test_perplexity.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
# Esperado: perplexity ~ 10-15 para BitNet-2B (não 100+)

# Se perplexity alta: modelo corrompido, re-baixar
rm -rf models/BitNet-b1.58-2B-4T
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
```

### 5.5. Property-based test falha (após M1)

Sintoma: `test_acdc_properties` reporta falha em alguma invariante.

Diagnóstico:
```bash
# Rodar com verbose
./build_tests/test_acdc_properties --success 2>&1 | tee /tmp/prop_test.log
# Imprime o seed da iteração que falhou

# Investigar: o kernel mudou, ou a invariante está errada?
# Ver P7 em requirements.md: invariante é ctest; revisar test, não kernel
```

---

## 6. Estrutura de Artefatos (Onde está cada coisa)

```
BitNet/
├── CLAUDE.md                        # Restrições, build, convenções
├── README.md                        # Será reescrito em M5 (persona D4)
├── ROADMAP.md                       # A criar em M1 (reserva técnica, etc.)
│
├── _reversa_sdd/                    # IMUTÁVEL: análise reversa original
│   ├── adrs/                        # 7 ADRs
│   ├── domain.md                    # 16 domain rules
│   └── ...
│
├── _reversa_forward/                # Features forward
│   └── 001-trilha-rigor-produto/
│       ├── requirements.md          # v2 (pós-clarify)
│       ├── roadmap.md               # v1 (este ciclo)
│       ├── investigation.md         # v1
│       ├── data-delta.md            # v1
│       └── onboarding.md            # v1 (este arquivo)
│
├── .reversa/                        # Configuração Reversa
│   ├── config.toml                  # feature-folder, pt-BR
│   └── active-requirements.json     # feature ativa: 001
│
├── docs/                            # Documentação canônica
│   ├── theory/                      # 5 níveis algébricos (intocado)
│   ├── findings-cpu-universal.md    # Writeup S2
│   ├── decision-matrix.md           # A criar M2
│   ├── invariants.md                # A criar M1
│   └── hardware-compatibility.md    # A criar M5
│
├── src/                             # Kernels C++
│   ├── ggml-bitnet-mad.cpp          # L1 (não modificado)
│   ├── ggml-bitnet-wht.cpp          # L2 (não modificado)
│   ├── ggml-bitnet-fwht.cpp         # L3 (não modificado em v0.1)
│   ├── ggml-bitnet-tropical.cpp     # L4 (não modificado, só doc)
│   ├── ggml-bitnet-hrr.cpp          # L5 (não modificado)
│   ├── ggml-bitnet-kv-cache.cpp     # L4 cache (não modificado em v0.1)
│   └── ggml-bitnet-dispatch.cpp     # Dispatch (não modificado)
│
├── include/                         # Headers públicos
│   └── ggml-bitnet-*.h              # Não modificados
│
├── tests/                           # Testes
│   ├── test_*.cpp                   # 9 existentes (não modificados)
│   ├── test_*_properties.cpp        # A criar M1
│   ├── test_acdc_rect.cpp           # A criar M3 (condicional)
│   ├── test_air_gapped_boot.sh      # A criar M5
│   ├── cross_validation.py          # A criar M2
│   ├── snapshots/                   # Snapshots versionados (a criar M2)
│   └── CMakeLists.txt               # Modificado M1/M3
│
├── utils/                           # Scripts Python
│   ├── cpu_universal_benchmark.py   # Existente
│   ├── extract_acdc_diagonal.py     # Existente (Phase A)
│   ├── finetune_acdc.py             # NÃO IMPLEMENTAR v0.1 (reserva D3)
│   └── bench_publish.py             # A criar M5
│
├── 3rdparty/llama.cpp/              # IMUTÁVEL (submodule)
├── patches/llama.cpp/               # 3 patches vendored (L3, L5, L4)
├── scripts/apply-dispatch-patches.sh
├── .github/workflows/ci.yml         # Modificado M5
└── examples/                        # A criar M5 (D4 scenarios)
```

---

## 7. Próximos Passos para o Onboarder

Depois de completar as seções 1-4, o onboarder tem 3 opções:

### 7.1. Contribuir com código

Ir para `_reversa_forward/001-trilha-rigor-produto/actions.md` (a criar em `/reversa-to-do`) e pegar uma ação atômica de M1.

### 7.2. Validar empiricamente (D2 trigger)

Sub-tarefa de M1: baixar Llama-2-7B GGUF, rodar inferência fim-a-fim, medir perplexity. Documentar resultado em `investigation-d2-result.md` (a criar).

### 7.3. Reportar bug ou improvement

Abrir issue em `https://github.com/peder1981/BitNet/issues` com template:

```markdown
## Contexto
- Persona: [saúde | jurídico | financeiro | hobbyista | pesquisador]
- Hardware: [CPU, RAM, OS]
- BitNet build: [commit hash ou 'main']

## Comando executado
\`\`\`bash
[paste exato]
\`\`\`

## Esperado
[o que você esperava]

## Atual
[o que aconteceu]

## Logs
\`\`\`
[relevant log output]
\`\`\`
```

---

## 8. Recursos Externos (para quem quer ir além)

- **Documentação completa do BitNet upstream**: `https://github.com/microsoft/BitNet`
- **Paper original**: Ma et al. 2024, "The Era of 1-bit LLMs"
- **llama.cpp**: `https://github.com/ggerganov/llama.cpp`
- **Tutorial CTest**: `https://cmake.org/cmake/help/latest/manual/ctest.1.html`
- **Catch2 v3 GENERATE**: `https://github.com/catchorg/Catch2/blob/devel/docs/generators.md`

---

*onboarding.md v1 — gerado por reversa-plan em 2026-06-06*
