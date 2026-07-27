# Spec Impact Matrix — BitNet CPU-Universal

> Gerado pelo Reversa Architect | 2026-06-06 | doc_level: completo
>
> **Como ler**: cada linha mapeia um **componente / container / decisão** para as **especificações que ele impacta** (RNs, ADRs, Princípios, ACs do forward, Dívidas). Use para responder "se eu mudar X, o que quebra?".

---

## 1. Matriz: Componentes → Especificações

| Componente | RNs impactadas | ADRs | Princípios | Dívidas |
|------------|---------------|------|------------|---------|
| **L1 I2_S MAD** (ggml-bitnet-mad) | RN-001, RN-004, RN-010, RN-013 | ADR-001, ADR-002, ADR-005 | P1, P3 | D-10 |
| **L1 I2_S LUT** (ggml-bitnet-lut) | RN-001, RN-002, RN-012 | ADR-001, ADR-005, ADR-006 | P1, P3 | D-10 |
| **L2 WHT** (ggml-bitnet-wht) | — | ADR-001, ADR-005, ADR-006 | P2, P3, P7 | D-09 |
| **L3 ACDC** (ggml-bitnet-fwht) | RN-001 | ADR-001, ADR-006 | P2, P3, P4, P6, P7 | D-01, D-06, D-09 |
| **L4 Tropical** (ggml-bitnet-tropical) | — | ADR-001, ADR-006 | P2, P3, P5, P7 | D-04, D-05 |
| **L4 Sparse Float** (em tropical.cpp) | — | — | P2, P3 | D-04 |
| **L5 HRR** (ggml-bitnet-hrr) | — | ADR-001, ADR-006 | P2, P3, P4, P6, P7 | D-01, D-02, D-09 |
| **L5 KV Cache K_i8** (ggml-bitnet-kv-cache) | — | — | P3 | D-08 |
| **Dispatch** (ggml-bitnet-dispatch) | RN-008, RN-009 | ADR-001, ADR-006 | (orquestra L1-L5) | D-04, D-07 |
| **Common** (ggml-bitnet-common) | — | — | P7 | D-09 |
| **CLI** (run_inference.py) | RN-008, RN-009 | — | — | D-04 |
| **Server** (run_inference_server.py) | RN-009 | — | — | — |
| **Setup** (setup_env.py) | RN-002, RN-003, RN-007 | ADR-002, ADR-005, ADR-006 | — | D-10, D-11 |
| **Conversion utils** (convert-hf-to-gguf-bitnet.py) | RN-001, RN-002, RN-010, RN-012, RN-013, RN-016 | ADR-005, ADR-006 | P1 | D-10, D-11 |
| **Codegen** (codegen_tl{1,2}.py) | — | ADR-002, ADR-006 | P1, P3 | D-10 |
| **Patches vendored** (patches/llama.cpp/*) | — | ADR-001 | — | D-07 |
| **CI** (.github/workflows/ci.yml) | — | ADR-002 | — | D-12 |
| **Submodule** (3rdparty/llama.cpp) | — | ADR-001 | — | D-07 |

🟢 CONFIRMADO para todos os mapeamentos (cruzamento de gap-analysis.md P2-P7 + domain.md RN-001..016 + adrs/001-007 + code-analysis.md).

---

## 2. Matriz Inversa: Especificações → Componentes

### 2.1 Regras de Negócio (RNs)

| RN | Componentes que a implementam | Componentes que a violariam se modificados |
|----|------------------------------|---------------------------------------------|
| **RN-001** (tensores protegidos) | `convert-hf-to-gguf-bitnet.py:795`, `convert_checkpoint.py` (legado) | L1 MAD, L1 LUT, L2 WHT, L3 ACDC — se aceitarem norm/lm_head/embed |
| **RN-002** (embed F16 com TL) | `setup_env.py:129-130` | `convert-hf-to-gguf-bitnet.py --quant-embd` |
| **RN-003** (arch → formats) | `setup_env.py:SUPPORTED_QUANT_TYPES` | L1 LUT (precisa compilar com arch certa) |
| **RN-004** (nrow % 4) | `ggml-bitnet-mad.cpp:98` (assert) | (n/a — é invariante) |
| **RN-007** (Clang obrigatório) | `.github/workflows/ci.yml`, `setup_env.py:214` | Build system inteiro |
| **RN-008** (ngl 0 hardcoded) | `run_inference.py` | llama.cpp CLI args |
| **RN-009** (b 1 hardcoded) | `run_inference.py` | llama.cpp CLI args |
| **RN-010** (ternário {0,1,2}) | `ggml-bitnet-mad.cpp`, `convert_checkpoint.py` (legado) | L1 packing, GPU packing |
| **RN-012** (base-3 TL1/TL2) | `convert-hf-to-gguf-bitnet.py` | L1 LUT |
| **RN-013** (absmax médio) | `convert_checkpoint.py:quant_weight_int8` (legado) | TENSOR.scale |

### 2.2 ADRs

| ADR | Componentes que o seguem | Componentes que o violariam |
|-----|--------------------------|------------------------------|
| **ADR-001** (llama.cpp) | 3rdparty/llama.cpp, dispatch, todos os kernels | (substituir backend quebraria o sistema inteiro) |
| **ADR-002** (Clang) | `setup_env.py`, `.github/workflows/ci.yml` | GCC build path |
| **ADR-003** (dual-model GPU) | N/A (fork sem GPU) | (legado upstream) |
| **ADR-004** (CUDA Graphs) | N/A (fork sem GPU) | (legado upstream) |
| **ADR-005** (3 formatos) | `setup_env.py:SUPPORTED_QUANT_TYPES`, `convert-hf-to-gguf-bitnet.py`, L1 LUT, L1 MAD | (qualquer novo formato requer novo kernel + conversão) |
| **ADR-006** (codegen) | `utils/codegen_tl{1,2}.py`, `preset_kernels/`, `setup_env.py:gen_code` | (kernel sem codegen = reparametrização runtime, sem otimização) |
| **ADR-007** (weights_only) | N/A (fork sem GPU); upstream `gpu/generate.py`, `gpu/convert_checkpoint.py` | (qualquer torch.load sem flag = CWE-502) |

### 2.3 Princípios Transversais

| Princípio | Componentes que o materializam | Lacuna |
|-----------|-------------------------------|--------|
| **P1** (Shannon floor) | L1 I2_S MAD/LUT packing | n/a |
| **P2** (identidade algébrica) | L2, L3, L4, L5 (todos verificados com max_diff = 0) | n/a |
| **P3** (hierarquia de custo) | L1 (memória), L2 (mul→add), L3 (n²→n log n), L4 (n²→top-K), L5 (n²→d log d) | n/a |
| **P4** (mínimo irredutível) | L3 ACDC (n muls), L5 FFT (d log d) | n/a |
| **P5** (dequantização tropical) | L4 tropical_attention (τ→0 + top-K) | P5 annealing τ finito (D-05) |
| **P6** (estrutura, não compressão) | `acdc_project` (validação), `hrr_pseudoinverse` | **Modelo treinado com ACDC/HRR (D-01)** |
| **P7** (FFT como cola) | L2, L3, L5 butterflies + L5 FFT | DRY refactor (D-09) |

---

## 3. Matriz: Mudanças → Impacto

### 3.1 Se mudar `ggml-bitnet-mad.cpp` (L1 I2_S MAD)

| Impacto | Severidade |
|---------|-----------|
| Quebra build inteiro | 🔴 CRÍTICA |
| Quebra todos os 9 testes ctest | 🔴 CRÍTICA |
| Muda baseline de todos os benchmarks | 🟡 IMPORTANTE |
| Pode violar RN-001, RN-004, RN-010, RN-013 | 🟡 IMPORTANTE |

### 3.2 Se mudar `ggml-bitnet-fwht.cpp` (L3 ACDC)

| Impacto | Severidade |
|---------|-----------|
| Quebra L3 dispatch (BITNET_ACDC_FFN=1) | 🟡 IMPORTANTE |
| Pode introduzir bug P6 (1/n² stray) — ver ed6fbde | 🟡 IMPORTANTE |
| Quebra `test_acdc.cpp` 5/5 | 🟡 IMPORTANTE |
| Não afeta L1, L2, L4, L5 (ortogonal) | — |

### 3.3 Se mudar `ggml-bitnet-tropical.cpp` (L4 Tropical)

| Impacto | Severidade |
|---------|-----------|
| Quebra L4 dispatch (BITNET_TROPICAL_TOPK) | 🟡 IMPORTANTE |
| Quebra `test_tropical.cpp` + `test_sparse_attention.cpp` | 🟡 IMPORTANTE |
| Não afeta L1, L2, L3, L5 (ortogonal) | — |
| Se mudar sparse_attention_float, afeta opt-in path | 🟢 MENOR |

### 3.4 Se mudar `ggml-bitnet-hrr.cpp` (L5 HRR)

| Impacto | Severidade |
|---------|-----------|
| Quebra L5 dispatch (BITNET_HRR_ATTN=1) | 🟡 IMPORTANTE |
| Quebra `test_hrr_cleanup.cpp` + `test_hrr_attention.cpp` | 🟡 IMPORTANTE |
| Regressão de performance esperada d=128 (D-02) | 🟡 IMPORTANTE |
| Não afeta L1-L4 (ortogonal) | — |

### 3.5 Se mudar `ggml-bitnet-kv-cache.cpp` (K_i8 cache)

| Impacto | Severidade |
|---------|-----------|
| Quebra L4 tropical cache (se GQA) | 🟡 IMPORTANTE |
| Quebra `test_kv_i8_cache.cpp` 11/11 | 🟡 IMPORTANTE |
| Se mudar mutex, reintroduz race GQA | 🔴 CRÍTICA |
| Não afeta L1, L2, L3, L5 HRR (mas L5 pode usar no futuro) | — |

### 3.6 Se mudar `ggml-bitnet-dispatch.cpp` (Dispatch)

| Impacto | Severidade |
|---------|-----------|
| Quebra TODOS os dispatch L2-L5 | 🔴 CRÍTICA |
| Requer atualizar 3 patches vendored (D-07) | 🟡 IMPORTANTE |
| Pode violar compat ABI com llama.cpp | 🔴 CRÍTICA |

### 3.7 Se atualizar `3rdparty/llama.cpp` (submodule)

| Impacto | Severidade |
|---------|-----------|
| 3 patches vendored podem falhar (D-07) | 🟡 IMPORTANTE |
| Requer `scripts/apply-dispatch-patches.sh --check` | — |
| Se patches não aplicam, dispatch L2-L5 quebra | 🔴 CRÍTICA |
| Pode introduzir novo upstream que conflita | 🟡 IMPORTANTE |

### 3.8 Se mudar `setup_env.py`

| Impacto | Severidade |
|---------|-----------|
| Quebra pipeline completo de setup | 🔴 CRÍTICA |
| Pode violar RN-002, RN-003, RN-007 | 🟡 IMPORTANTE |
| Pode quebrar D-10 (2B reusa config 3B) | 🟢 MENOR |

---

## 4. Matriz: ACs do Forward 001 → Componentes

O forward `001-trilha-rigor-produto` (em `_reversa_forward/001-trilha-rigor-produto/`) tem 13 ACs que mapeiam para:

| AC | Descrição resumida | Componentes que satisfazem |
|----|--------------------|-----------------------------|
| **AC-01** | Smoke benchmark L1-L5 (n=64/128/256) | `utils/cpu_universal_benchmark.py` |
| **AC-02** | Subtest PASS de cada kernel | `tests/test_*.cpp` (9 arquivos) |
| **AC-03** | Ctest 9/9 PASS | `tests/CMakeLists.txt` + CI |
| **AC-04** | Build com Clang 18 OK | `.github/workflows/ci.yml` |
| **AC-05** | BitNet-2B GGUF gerado | `setup_env.py:prepare_model` |
| **AC-06** | CLI inference end-to-end | `run_inference.py` |
| **AC-07** | ACDC d* extraído de modelo treinado | `utils/extract_acdc_diagonal.py` |
| **AC-08** | ACDC FFN retangular 2560×6912 funcional | L3 ACDC + dispatch (acdc_gemv) |
| **AC-09** | HRR cleanup d≥10N verificado | L5 HRR + `test_hrr_cleanup.cpp` |
| **AC-10** | Documentação L2-L5 atualizada | `docs/findings-cpu-universal.md` |
| **AC-11** | Air-gapped boot verificado | (manual, fora de testes) |
| **AC-12** | Single-user inference example | `onboarding.md` |
| **AC-13** | Hardware compatibility table | `onboarding.md` |

🟢 CONFIRMADO (forward 001 requirements.md v2).

---

## 5. Matriz: Dívidas Técnicas → Componentes / Decisões

| Dívida | Componente | Ação sugerida | Esforço |
|--------|------------|---------------|---------|
| **D-01** P6 não validado | L3, L5 + novo modelo | Treinar modelo com ACDC e/ou HRR (escopo GPU 2-6 sem) | XL |
| **D-02** L5 regressão d=128 | L5 HRR | Usar L5 apenas d≥256; documentar | S |
| **D-03** RNs obsoletas (GPU) | `_reversa_sdd/domain.md` | Marcar como `[LEGACY — UPSTREAM ONLY]` | XS |
| **D-04** L4 via env, não flag | `3rdparty/llama.cpp/src/llama.cpp:9797-9857` | Adicionar flag `--attn sparse/tropical/hrr` | M |
| **D-05** P5 τ finito | L4 tropical | Tornar τ treinável em fine-tuning | L |
| **D-06** ACDC FFN garbage | L3 dispatch | Documentar como esperado; medir P6 só com modelo treinado | XS |
| **D-07** 3 patches vendored | `patches/llama.cpp/*` | Refatorar para hook em runtime (substituir patches) | L |
| **D-08** K_i8 scale lock | `ggml-bitnet-kv-cache.cpp` | Adicionar teste de regressão | XS |
| **D-09** DRY butterflies | L2, L3, L5 | Extrair `bitnet_butterfly.h` comum | M |
| **D-10** 2B reusa config 3B | `setup_env.py` | Adicionar linha dedicada no `SUPPORTED_HF_MODELS` | XS |
| **D-11** quant-embd impacto | `convert-hf-to-gguf-bitnet.py:795-797` | Adicionar benchmark perplexidade com/sem | S |
| **D-12** CI sem smoke | `.github/workflows/ci.yml` | Adicionar nightly workflow com model download | M |

🟢 CONFIRMADO (gap-analysis.md, code-analysis.md, context-summary).

---

## 6. Matriz: 9 Testes CTest → Componentes

| Teste | Arquivo | Componente alvo | LOC | Subtests |
|-------|---------|-----------------|----:|---------:|
| `test_bitnet_common` | `tests/test_bitnet_common.cpp` | common (bitnet_next_pow2) | ~80 | 5/5 |
| `test_wht` | `tests/test_wht.cpp` | L2 WHT (wht_dot_avx2) | ~200 | 5/5 |
| `test_acdc` | `tests/test_acdc.cpp` | L3 ACDC (fwht + acdc_forward + acdc_project + acdc_gemv) | ~250 | 5/5 |
| `test_tropical` | `tests/test_tropical.cpp` | L4 tropical (tropical_attn + topk + argmax) | ~200 | 5/5 |
| `test_sparse_attention` | `tests/test_sparse_attention.cpp` | L4 sparse float (sparse_attention_float) | ~150 | 5/5 |
| `test_kv_i8_cache` | `tests/test_kv_i8_cache.cpp` | L4/L5 K_i8 cache (mutex, scale lock, GQA) | ~300 | 11/11 |
| `test_hrr_cleanup` | `tests/test_hrr_cleanup.cpp` | L5 HRR (FFT roundtrip + bind + phasor + RESIDUAL + NAIVE) | ~250 | 5/5 |
| `test_hrr_attention` | `tests/test_hrr_attention.cpp` | L5 HRR attention (dispatch kernel) | ~200 | 5/5 |
| `test_extract_acdc_diagonal` | `tests/test_extract_acdc_diagonal.py` | `utils/extract_acdc_diagonal.py` (Python) | ~150 | 4/4 |
| **Total** | 9 arquivos | 7 componentes C++ + 1 Python | ~1.780 | **50/50** |

🟢 CONFIRMADO (inventory.md, gap-analysis.md P2/P7, `ctest --output-on-failure`).

---

## 7. Matriz: 7 Princípios × 5 Níveis × 9 Testes

| Princípio | L1 | L2 | L3 | L4 | L5 | Teste que valida |
|-----------|:--:|:--:|:--:|:--:|:--:|------------------|
| P1 (Shannon) | ✓ | — | — | — | — | (paper BitNet) |
| P2 (identidade) | ✓ | ✓ | ✓ | ✓ | ✓ | `test_wht` 5/5, `test_acdc` 5/5, `test_tropical` 5/5, `test_hrr_cleanup` 5/5 |
| P3 (hierarquia) | ✓ | ✓ | ✓ | ✓ | ✓ | `utils/cpu_universal_benchmark.py` |
| P4 (mínimo) | ✓ | — | ✓ | ✓ | ✓ | (prova teórica) |
| P5 (tropical) | — | — | — | ◐ | — | `test_tropical` 5/5 (τ→0 só) |
| P6 (estrutura) | — | — | ✗ | — | ✗ | `test_extract_acdc_diagonal` 4/4 (validação, não treinamento) |
| P7 (FFT) | — | ✓ | ✓ | — | ✓ | L2/L3/L5 ctest |

🟢 CONFIRMADO (gap-analysis.md matriz 7×4 + este spec impact).

---

## 8. Traceability End-to-End (Exemplo: Smoke n=64)

Trace de um único smoke benchmark "BitNet-2B, n=64, L4 Tropical":

```
1. run_inference.py -m .../ggml-model-i2_s.gguf -p "..." -n 64 -t 4
   └── CLI: run_inference.py
       └── subprocess.run llama-cli -ngl 0 -b 1
           └── llama.cpp:build KQV
               └── patch 03: bitnet_kv_i8_cache_set_layer(il)
                   └── ggml-bitnet-kv-cache.cpp:set_layer
                       └── ggml-bitnet-dispatch.cpp:bitnet_op_tropical_attn
                           └── ggml-bitnet-tropical.cpp:tropical_attention
                               ├── quantize K → ternary {-1, 0, +1}
                               │   └── ggml-bitnet-common.cpp:bitnet_next_pow2
                               ├── cache.get(layer, kv_h) → K_i8
                               │   └── ggml-bitnet-kv-cache.cpp:get
                               ├── scan O(n·d) zero-mul
                               ├── top-K (K=32)
                               └── softmax over K

Testes que validam: test_tropical.cpp 5/5, test_kv_i8_cache.cpp 11/11
Princípio: P2 (max_diff = 0), P3 (speedup medido), P5 (top-K)
AC forward: AC-01 (smoke bench), AC-02 (subtest PASS)
RN: nenhuma diretamente; -ngl 0 (RN-008), -b 1 (RN-009)
ADR: ADR-001 (llama.cpp), ADR-005 (I2_S), ADR-006 (codegen)
Dívida: D-04 (env var ao invés de flag)
```

🟢 CONFIRMADO (cruzamento de state-machines.md fluxo 2 + gap-analysis.md + context-summary Phase C).

---

**Fim do Spec Impact Matrix.** Use este documento para responder perguntas de impacto durante refatorações, code review, e planejamento de novas features.
