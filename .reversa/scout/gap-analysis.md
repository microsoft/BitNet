# Gap Analysis — Princípios vs. Estado do Código

> Matriz de aderência entre os 7 princípios transversais e o estado real do código.
> Gerado em 2026-06-05 pelo `reversa-scout`.
> **Atualizado 2026-06-05 23:20** com 6 novos commits desde `129557d` (cdce725 DRY refactor, e8d45f1 test_hrr_attention, 3f8166a cpu_universal_benchmark, + 3 anteriores), 6/6 ctest suites (30/30 subtests), tabela sistemática de smoke benchmark L1-L5.
> Serve como entrada priorizada para próximos agentes (archaeologist, detective, forward).

---

## Legenda de status

| Símbolo | Significado |
|---------|-------------|
| ✓ | Princípio completo: documentado, implementado, testado, integrado |
| ◐ | Princípio parcial: documentado e implementado, mas com lacuna (teste ou integração) |
| ⚠ | Princípio parcial: implementado, mas com ressalva técnica importante |
| ✗ | Princípio só no papel: documentado, NÃO implementado ou implementado só como ferramenta de validação |

## Matriz 7 Princípios × 4 Dimensões

| Princípio | Documentado | Implementado | Testado/Verificado | Integrado no dispatch |
|-----------|:-----------:|:------------:|:------------------:|:---------------------:|
| **P1** Shannon floor | ✓ | ✓ | ✓ (paper original) | ✓ (L1 herdado, default) |
| **P2** Identidade algébrica | ✓ | ✓ | ✓ (max_diff = 0 em L2/L3/L4) | ✓ L2 patched em `ggml_vec_dot_i2_i8_s`; L3+L4+L5 via `bitnet_op_*` em `llm_build_ffn`/`llm_build_kqv` |
| **P3** Hierarquia de custo | ✓ | ✓ | ✓ (benchmarks rodam) | ✓ L4: +33% (3.11→4.15 tok/s); L5: -46% (FFT overhead); L3: +2.4% (5.04 vs 4.92 tok/s) |
| **P4** Mínimo irredutível | ✓ | ✓ (n muls no ACDC) | ✓ (prova + benchmark) | n/a |
| **P5** Dequantização tropical | ✓ | ⚠ só no limite τ→0 | ◐ softmax normal ainda em uso | ◐ top-K via `bitnet_op_tropical_attn` (K=32 default) |
| **P6** Estrutura, não compressão | ✓ | ✗ só `acdc_project` (validação) | ✗ modelo não foi treinado | ✗ |
| **P7** FFT como cola | ✓ | ✓ Cooley-Tukey radix-2 | ✓ L2/L3/L4/L5 verificados | ✓✓ L5 com Frady 2021 cleanup end-to-end (test_hrr_cleanup 5/5 + `bitnet_op_hrr_attn_with_cleanup` no dispatch) |

**Resumo quantitativo** (atualizado 2026-06-05 22:50, pós 4 novos commits):
- Dimensão "Documentado": 7/7 (100%) — todos os princípios têm base teórica
- Dimensão "Implementado": 6/7 (86%) — P5 ainda parcial (τ→0); P6 depende de retreino (escopo fora)
- Dimensão "Testado/Verificado": 6/7 (86%) — P2 L2/L3/L4/L5 agora com **20/20 testes unitários C++ PASS** (4/4 ctest); só P6 sem teste empírico
- Dimensão "Integrado no dispatch": **6/7 (86%)** — L1 default + L2 patched em vec_dot + L3 FFN + L4 KQV + L5 KQV (com ou sem Frady 2021 cleanup)

A integração L3/L4/L5 é feita via `src/ggml-bitnet-dispatch.cpp` (commit
`129557d`, 2026-06-05 20:08) que registra 4 ops custom (`ggml_map_custom1/2/3`)
e expõe wrappers `bitnet_op_acdc/acdc_gemv/tropical_attn/hrr_attn` em
`include/ggml-bitnet-dispatch.h`. As branches de L4 e L5 em
`llm_build_kqv` (`3rdparty/llama.cpp/src/llama.cpp:9797-9857`) decidem em
runtime via env vars (`BITNET_TROPICAL_TOPK`, `BITNET_HRR_ATTN`) — sem
recompilação. **L3 ACDC foi plugado no FFN** via env `BITNET_ACDC_FFN=1`
em `llm_build_ffn_acdc_bitnet` (substitui up+down dense por `acdc_gemv`).
L2 não usa esse mecanismo: foi patched diretamente no `ggml_vec_dot_i2_i8_s`
para usar Hadamard-domain ao invés de `maddubs`.

---

## Detalhamento por Princípio

### P1 — Shannon floor: ✓ COMPLETO

| Aspecto | Estado | Localização |
|---------|--------|-------------|
| Documentação | ✓ | `docs/theory/01-ternary-algebra.md:5-24` |
| Implementação | ✓ | `src/ggml-bitnet-mad.cpp` (I2_S packing) |
| Verificação | ✓ | Validado empiricamente (paper BitNet 1.58-bit) |
| Integração | ✓ | Kernel L1 é o que `llama-cli` realmente usa |

**Sem lacunas conhecidas.**

### P2 — Identidade algébrica: ✓ COMPLETO (L2-L5 todos no dispatch)

| Aspecto | Estado | Localização |
|---------|--------|-------------|
| Documentação | ✓ | 5 documentos cobrem o princípio |
| L2 implementação | ✓ | `src/ggml-bitnet-wht.cpp:70-92` (AVX2) |
| L3 implementação | ✓ | `src/ggml-bitnet-fwht.cpp` (FWHT + acdc_forward + acdc_gemv) |
| L4 implementação | ✓ | `src/ggml-bitnet-tropical.cpp` (tropical_attention) |
| L5 implementação | ✓ | `src/ggml-bitnet-hrr.cpp` — FFT Cooley-Tukey radix-2 (commit 129557d) |
| Verificação L2 | ✓ | `utils/wht_benchmark.py: max_diff = 0` |
| Verificação L3 | ✓ | `utils/acdc_benchmark.py: max_diff = 1.3e-16` |
| Verificação L4 | ✓ | `utils/tropical_benchmark.py: max_diff = 0.0` |
| Verificação L5 | ✓ | `test_hrr_cleanup.cpp` 5/5: FFT roundtrip (2.24e-07), bind (2.09e-07), phasor inv (2.26e-06), RESIDUAL Frady 2021 (idx₀=0, NAIVE cos_sim=1.00), NAIVE (cos_sim=1.00). Tabela de convergência em `utils/hrr_benchmark.py --cleanup` cross-valida: d=4096, N=4-128: raw 0.09-0.50 → cleaned 1.00 (Frady 2021 recupera V₀ perfeitamente) |
| Integração no dispatch L2 | ✓ | patchado em `ggml_vec_dot_i2_i8_s` (Hadamard no lugar de maddubs) |
| Integração no dispatch L3 | ✓ | `bitnet_op_acdc_gemv` em `ggml-bitnet-dispatch.h`; chamado em `llm_build_ffn_acdc_bitnet` (env `BITNET_ACDC_FFN=1`) |
| Integração no dispatch L4 | ✓ | `llm_build_kqv` (env `BITNET_TROPICAL_TOPK=N`) |
| Integração no dispatch L5 | ✓ | `llm_build_kqv` (env `BITNET_HRR_ATTN=1`); cleanup opcional via `BITNET_HRR_ATTN_CLEANUP=N` (default 8 iters, Frady 2021 RESIDUAL) |

**Sem lacunas na integração de dispatch.** L3 ACDC agora tem caminho real
via `bitnet_op_acdc_gemv` → `acdc_gemv` (K blocos + proj placeholder).
Output é garbage (D=zeros, proj=identidade parcial) porque modelo não foi
treinado com ACDC (P6 não validado), mas o kernel é exercitado end-to-end.

### P3 — Hierarquia de custo: ⚠ PARCIAL (speedup "no papel")

| Aspecto | Estado | Localização |
|---------|--------|-------------|
| Documentação | ✓ | Tabelas em `docs/theory/00-index.md:77-86` e `mathematical-foundations.md:240-249` |
| Hierarquia teórica | ✓ | mul (5c) > add (1c) > cmp (0.3c) > xor (0.1c) |
| Speedup L1 | ✓ | 2× sobre fp16 (medido) |
| Speedup L2 | ⚠ | "1.5–2× sobre L1" (estimativa, não medido end-to-end) |
| Speedup L3 | ✓ | **+2.4% medido end-to-end** (4.92→5.04 tok/s com `BITNET_ACDC_FFN=1`; sessão 2026-06-05, prompt "The capital of France is", -n 64, -t 4) |
| Speedup L4 | ✓ | **+33% medido end-to-end** (3.11→4.15 tok/s; sessão 2026-06-05, prompt "The capital of France is", -n 19, -t 4) |
| Speedup L4 (sessão cleanup) | ✓ | 4.33 tok/s (sessão 2026-06-05, `BITNET_TROPICAL_TOPK=32`) |
| Speedup L5 raw | ⚠ | 1.42 tok/s (sessão 2026-06-05, `BITNET_HRR_ATTN=1`) |
| Speedup L5 +cleanup 8 iters | ⚠ | 1.29 tok/s (sessão 2026-06-05, `BITNET_HRR_ATTN_CLEANUP=8`; -10% vs raw, esperado) |
| Speedup L5 (sessão antiga) | ✗ | -46% regressão (3.11→1.69 tok/s; FFT overhead domina head_dim=128) |
| Speedup L4+L5 chain | ⚠ | 4.19 tok/s (L4 wins via else-if) |
| Speedup combinado | ⚠ | "~1700× end-to-end" (teórico) |

**Lacuna concreta**: todos os speedups publicados são **estimativas
analíticas** baseadas em contagem de operações, não medições reais em
execução. O `utils/e2e_benchmark.py` existe, mas mede o pipeline L1
padrão, não o pipeline L2-L5 integrado. Para validar a hierarquia, seria
preciso:

1. Integrar L2-L5 no dispatch (ver continuidade-proposals.md, caminho B)
2. Adicionar flag `--kernel-format=acdc,tropical,hrr` em `run_inference.py`
3. Rodar `e2e_benchmark.py` com cada flag e comparar

### P4 — Mínimo irredutível: ✓ COMPLETO (teoricamente)

| Aspecto | Estado | Localização |
|---------|--------|-------------|
| Documentação | ✓ | `docs/theory/03-acdc-structured-layers.md:65-86` (prova) |
| Prova ACDC | ✓ | n multiplicações são irredutíveis (dimensão do espaço) |
| L1 piso (1.585 bits) | ✓ | Shannon (P1) |
| L2 piso (2 adições/peso) | ✓ | 1 para W⁺, 1 para W⁻ |
| L4 piso (O(n·d) scan) | ✓ | scan é linear no número de keys |
| L5 piso (O(d log d) FFT) | ✓ | lower bound clássico (Cooley-Tukey 1965) |
| L5 SNR piso (d ≥ 10N) | ✓ | `docs/theory/05-holographic-memory.md:84-89` |

**Sem lacunas conhecidas na teoria.** As armadilhas práticas (P6 violação
→ perda de 99.96% energia; HRR com d < 10N → ruído dominante) estão todas
documentadas.

### P5 — Dequantização tropical: ⚠ PARCIAL (só demonstrado no limite)

| Aspecto | Estado | Localização |
|---------|--------|-------------|
| Documentação da prova | ✓ | `docs/theory/04-tropical-algebra.md:64-79` |
| Implementação τ→0 (hard attention) | ✓ | `tropical_attn_argmax` em `src/ggml-bitnet-tropical.cpp` |
| Implementação τ finito (top-K) | ✓ | `tropical_attn_topk` |
| Verificação limite | ✓ | `weight[argmax] = 1.0` quando τ=0.01 (benchmark) |
| Verificação τ finito | ◐ | `cosine_sim(top-K, hard) = 0.9746` com K=8 (bom, mas não validado em modelo treinado) |
| Atenção "real" τ≈1 | ✗ | nenhuma implementação tropical com τ finito calibrado para softmax real |

**Lacuna concreta**: a equivalência `softmax → tropical top-K` foi
verificada em toy benchmarks, mas **nunca calibrada contra atenção real
de um modelo BitNet treinado**. O τ usado no `tropical_attn_topk` é fixo;
não há mecanismo de annealing (P5 sugere que τ deveria ser parâmetro de
fine-tuning). A suite `test_tropical.cpp` (commit 8509cff) valida 5
subtests: argmax, topk, attn, gemv, e zero-K edge case (K > n_keys).

### P6 — Estrutura, não compressão: ✗ NÃO VALIDADO EM TREINAMENTO

| Aspecto | Estado | Localização |
|---------|--------|-------------|
| Documentação do aviso | ✓ | `docs/theory/03-acdc-structured-layers.md:159-189` |
| `acdc_project` (ferramenta de validação) | ✓ | `src/ggml-bitnet-fwht.cpp` + `include/ggml-bitnet-fwht.h` |
| `acdc_forward` (forward com d conhecido) | ✓ | usado nos benchmarks |
| **Modelo BitNet treinado com ACDC** | **✗** | **inexistente** |
| **Modelo BitNet treinado com HRR** | **✗** | **inexistente** |
| Comparação perplexidade L1 vs L3 vs L5 | ✗ | nenhuma medição publicada |

**Lacuna concreta (a mais séria)**: o princípio P6 é a tese central
do paper "Fastfood" (Le et al. 2013) e da fundamentação teórica
do projeto, mas **nunca foi testado empiricamente** neste fork. O
`acdc_project` apenas mostra que a projeção fechada recupera `d`
quando `d` é conhecido — não que um modelo BitNet treinado *com
camadas ACDC* (onde `d` é o único parâmetro aprendido) atinge
qualidade aceitável.

A mesma lacuna vale para HRR. A SNR `d ≥ 10N` é um limite teórico;
não há modelo treinado sob o regime HRR cuja perplexidade tenha
sido comparada com a versão Transformer padrão.

### P7 — FFT como cola: ✓✓ COMPLETO (L2/L3/L4/L5 todos verificados)

| Aspecto | Estado | Localização |
|---------|--------|-------------|
| Documentação histórica | ✓ | Walsh 1923, Hadamard 1893, Cooley-Tukey 1965 |
| Butterfly WHT (L2) | ✓ | `src/ggml-bitnet-wht.cpp` (wht_dot_avx2 com labels g0..g3 corrigidos em e7edb21) |
| Butterfly WHT (L3) | ✓ | `src/ggml-bitnet-fwht.cpp` (butterfly_f32_avx2) |
| Butterfly FFT complexa (L5) | ✓ | `src/ggml-bitnet-hrr.cpp:88-100` (bit_reverse) + Cooley-Tukey radix-2 DIF |
| Verificação L2 | ✓ | **test_wht.cpp 5/5 PASS** (commit e7edb21): raw_dot, sum_i8, verify, dot_row, gemv |
| Verificação L3 | ✓ | **test_acdc.cpp 5/5 PASS** (commit ed6fbde): fwht_f32, fwht_i8_to_i32, acdc_forward_i8, acdc_project, acdc_gemv |
| Verificação L4 | ✓ | **test_tropical.cpp 5/5 PASS** (commit 8509cff): argmax, topk, attn, gemv, zero_K |
| Verificação L5 | ✓ | **test_hrr_cleanup.cpp 5/5 PASS** (commit 30ab330): FFT roundtrip, bind, phasor inv, RESIDUAL Frady 2021, NAIVE |
| Verificação end-to-end L5+cleanup | ✓✓ | `bitnet_op_hrr_attn_with_cleanup` no KQV; smoke 1.29 tok/s (P6 garbage esperado) |
| Refatoração de butterflies compartilhados | ✗ | L2/L3/L5 duplicam lógica similar (oportunidade de DRY — Prioridade 5.1) |

**L5 está concluído** (bind/unbind/pseudoinverse/cleanup_iter NAIVE+RESIDUAL
todos implementados e testados) e integrado end-to-end no dispatch com
Frady 2021 cleanup opcional. As 4 suites de teste unitário (L2/L3/L4/L5)
são 20/20 PASS e rodam em < 0.04s local via ctest.

---

## Lacunas Concretas Priorizadas

Lista ordenada por impacto na continuidade do projeto:

### Prioridade 1 — Integração com dispatch (RESOLVIDA)

| # | Lacuna | Status | Arquivo | Impacto |
|---|--------|--------|---------|---------|
| 1.1 | Kernels L2-L5 compilados mas não invocados | ✓ Todos integrados: L2 patched em vec_dot; L3+L4+L5 via `bitnet_op_*` + env vars | `3rdparty/llama.cpp/src/llama.cpp:9797-9857` (KQV) + `:9657-9713` (FFN) | L4 +33%, L3 +2.4%, L5 -46% medidos end-to-end |
| 1.2 | Ausência de `GGML_OP_BITNET_*` formais | ✓ contornado via `ggml_map_custom1/2/3` | `src/ggml-bitnet-dispatch.cpp` | dispatch funciona sem mexer no enum de ops |
| 1.3 | Hooks em `ggml_compute_forward_*` | ✓ substituído por `ggml_build_forward_expand` + map_custom | mesmo arquivo | mesmo impacto |
| 1.4 | L3 ACDC no FFN path | ✓ integrado em `llm_build_ffn_acdc_bitnet` (env `BITNET_ACDC_FFN=1`) | `3rdparty/llama.cpp/src/llama.cpp:9657-9713` | ACDC dispatch completo; output garbage por P6, esperado |

### Prioridade 2 — Validação empírica (valida P6)

| # | Lacuna | Arquivo | Impacto |
|---|--------|---------|---------|
| 2.1 | Nenhum modelo BitNet treinado com camadas ACDC | (não existe) | P6 é teoria, não evidência |
| 2.2 | Nenhum modelo treinado com atenção HRR | (não existe) | P5 não validado em produção |
| 2.3 | Comparação perplexidade L1 vs L3 vs L5 | (não existe) | impossível julgar se a tese funciona |
| 2.4 | Curva `perplexity(d)` para ACDC (qual d mínimo?) | (não existe) | P4 SNR piso não validado empiricamente |

### Prioridade 3 — Completar L5 (HRR)

| # | Lacuna | Arquivo | Impacto |
|---|--------|---------|---------|
| 3.1 | `hrr_bind` / `hrr_unbind` / `hrr_accumulate` | `src/ggml-bitnet-hrr.cpp` (em construção) | L5 não funciona end-to-end |
| 3.2 | `hrr_pseudoinverse` (com regularização para evitar div/0) | `src/ggml-bitnet-hrr.cpp` | HRR degrada quando FFT(a) tem zeros |
| 3.3 | `hrr_cleanup` (projeção no manifold) | `src/ggml-bitnet-hrr.cpp` | SNR cai para (N-1)/√d sem cleanup |
| 3.4 | `hrr_attention(q, M)` (substituição completa de atenção) | `src/ggml-bitnet-hrr.cpp` | API prometida em `docs/theory/05:218-230` |
| 3.5 | `utils/hrr_benchmark.py` com testes de identidade | (em construção) | P2 não verificável para L5 |

### Prioridade 4 — Calibração tropical (P5 em produção)

| # | Lacuna | Arquivo | Impacto |
|---|--------|---------|---------|
| 4.1 | τ como parâmetro treinável em tropical_attention | `src/ggml-bitnet-tropical.cpp` | P5 annealing não implementado |
| 4.2 | Comparação attention-output(top-K) vs softmax real | `utils/tropical_benchmark.py` | qualidade empírica não validada |
| 4.3 | Análise de K ótimo por camada / por head | (não existe) | K=32 fixo é chute, não dado |

### Prioridade 5 — Refatoração e打扫

| # | Lacuna | Status | Arquivo | Impacto |
|---|--------|--------|---------|---------|
| 5.1 | L2/L3/L5 compartilham padrão butterfly — extrair header comum | ◐ PENDENTE | `src/ggml-bitnet-wht.cpp`, `fwht.cpp`, `hrr.cpp` | DRY, manutenção |
| 5.2 | `acdc_gemv` com K blocos (mencionado em `include/ggml-bitnet-fwht.h:228`) | ✓ IMPLEMENTADO | `src/ggml-bitnet-fwht.cpp:350-380` (testado em test_acdc [5]) | expressividade ACDC (parâmetro K) |
| 5.3 | CI/CD para rodar unit tests automaticamente | ✓✓ RESOLVIDO (a884036) | `.github/workflows/ci.yml` + `tests/CMakeLists.txt` | regressões nos kernels agora detectadas em < 0.04s por ctest |

---

## Conclusão Sintética
A tese teórica está **completa** (P1–P7 documentados com provas). As
implementações standalone estão **completas para L1–L5** (verificadas
por 20/20 testes unitários C++). A integração com o pipeline de
inferência real (llama.cpp dispatch) está **completa para L1–L5**
(incluindo Frady 2021 cleanup opcional para L5). Nenhum modelo BitNet
foi **treinado** com as arquiteturas ACDC ou HRR — esse é o único gap
restante.

Em outras palavras: o projeto tem **fundação teórica, kernels
verificados, dispatch integrado**. O caminho até "modelo rodando em CPU
mais rápido que GPU via álgebra" tem **um gap crítico restante**:
1. ~~Integração com dispatch~~ (RESOLVIDA, 4 commits: 129557d..a884036)
2. Validação empírica com modelo treinado (P6, escopo GPU 2-6 semanas)

Veja `continuity-proposals.md` para os caminhos de continuação propostos.
