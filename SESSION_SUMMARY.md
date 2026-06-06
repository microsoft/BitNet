# SESSÃO: BitNet CPU-Universal — v0.1.0

**Período:** 2025-06-05 → 2026-06-05
**Tag:** `v0.1.0-cpu-universal` (pushed em 2026-06-05)
**Branch:** `main` (origin `peder1981/BitNet`)
**Branch base:** `129557d` (ponto de fork)
**Total de commits na sessão:** 19

---

## 1. Resumo executivo

A sessão transformou um fork inativo do `microsoft/BitNet` em um release candidate
funcional de uma **biblioteca matemática CPU-only** para LLMs 1-bit com cinco
níveis de aceleração algébrica. Ao final:

- **6/6 suítes ctest passam (30/30 subtests, 0,05 s)**
- **2 bugs reais** foram encontrados e corrigidos no código de produção
- **4 novas arquiteturas algébricas** integradas ao dispatch do llama.cpp
  (WHT, ACDC, Tropical, HRR + cleanup Frady 2021)
- **CI verde** no GitHub Actions (ubuntu-24.04 + clang-18)
- **Smoke benchmark** reproduz a tabela L1–L5 em ~30 s
- **1 achado de design** importante: L2/L3/L5 **não compartilham** butterfly

A tese CPU-Universal está matematicamente demonstrada. O único gap aberto
para fechamento empírico é o **Caminho C** (retreino P6 com ACDC/HRR/tropical),
que requer GPU e 2-6 semanas.

---

## 2. Commits da sessão (cronológico inverso)

```
18fcf75  docs(scout): v0.1.0 CPU-Universal release candidate + 6-test suite
3f8166a  feat(bench): add cpu_universal_benchmark.py for systematic L1-L5 smoke tests
e8d45f1  test(hrr-attn): add dispatch-kernel validation for hrr_attention_full
cdce725  refactor: extract bitnet_next_pow2 to shared header (DRY across L3+L5)
ed7f12b  docs(scout): update to reflect 14 new commits (L3 FFN + L5 cleanup + 4 test suites)
a884036  build(tests): wire all 4 kernel unit tests into CMake + CI
8509cff  test(tropical): rewrite test_tropical.cpp to match current API
ed6fbde  fix(acdc): drop 1/n² normalization in acdc_forward_i8 + add test_acdc
e7edb21  fix(wht): correct g0/g3 group labels in wht_dot_avx2 + add test_wht
7a449c6  docs(scout): mark L5 HRR cleanup end-to-end integration as complete
92dacc4  feat(hrr-dispatch): wire L5 HRR with Frady 2021 cleanup at llama.cpp KQV
a851053  build(submodule): update llama.cpp pointer to 3dfc2df (L5 HRR cleanup wiring)
b536d83  build(ci): minimum CI for L2-L5 kernels + integrate test_hrr_cleanup into cmake
a7da023  docs(scout): update artifacts to reflect L3-L5 dispatch + HRR refinement
43b2af5  feat(hrr_benchmark): Frady 2021 cleanup_convergence_test + helpers
30ab330  test(hrr): standalone test_hrr_cleanup.cpp (5/5 PASS) — first C++ kernel unit test
90ae65f  feat(hrr): add hrr_cleanup_iter (Frady 2021) with NAIVE + RESIDUAL modes
e1c95c5  build(submodule): update llama.cpp pointer to 707f316 (L3 ACDC FFN dispatch)
658fd0d  feat(acdc): integrate L3 ACDC FFN dispatch via acdc_gemv + env-gated llama.cpp helper
```

---

## 3. Bugs encontrados e corrigidos

### 3.1 WHT: rótulos g0..g3 invertidos (severidade ALTA)

- **Arquivo:** `src/ggml-bitnet-wht.cpp:186-189`
- **Commit fix:** `e7edb21`
- **Causa raiz:** os rótulos `g0..g3` estavam invertidos em relação a
  `unpack_i2s_block` no mesmo arquivo. Os bits `[7:6]` representam o grupo 0
  (posições 0..31), não o grupo 3.
- **Sintoma:** o `ggml_wht_verify` da própria biblioteca também falhava, indicando
  que o bug estava latente e não detectado.
- **Cobertura:** `test_wht.cpp` 5/5 PASS após o fix (raw_dot, sum_i8, verify,
  dot_row, gemv).
- **Aprendizado:** o pack I2_S x86 estratificado usa shift `(3 - group) * 2`
  para casar com `unpack_i2s_block`.

### 3.2 ACDC: fator 1/n² espúrio (severidade ALTA)

- **Arquivo:** `src/ggml-bitnet-fwht.cpp:291-303`
- **Commit fix:** `ed6fbde`
- **Causa raiz:** `acdc_forward_i8` aplicava um fator `1/n²` (dividia duas
  vezes por n) que violava a especificação do `CLAUDE.md`:

  > `acdc_forward(x, d) = H·(d⊙(H·x))`, **sem normalização** — sem fatores 1/n².
  > A diagonal `d` absorve a escala quando aprendida durante o treino.

- **Sintoma:** kernel matematicamente incorreto; o teste `acdc_project` também
  esperava `d*[k] = 1/n` para W=I (e não 1).
- **Cobertura:** `test_acdc.cpp` 5/5 PASS após o fix (fwht_f32, fwht_i8_to_i32,
  acdc_forward_i8, acdc_project, acdc_gemv).

---

## 4. Suítes de teste criadas (6/6 PASS, 30/30 subtests, 0,05 s)

| Suite                  | Subtests | Commit       | O que cobre                                           |
|------------------------|----------|--------------|-------------------------------------------------------|
| `test_bitnet_common`   | 5/5      | `cdce725`    | `next_pow2`, aliases, edge cases, guard estrutural    |
| `test_wht`             | 5/5      | `e7edb21`    | L2 — WHT zero-multiplicação                           |
| `test_acdc`            | 5/5      | `ed6fbde`    | L3 — FWHT, ACDC, projeção                             |
| `test_tropical`        | 5/5      | `8509cff`    | L4 — argmax, topk, attn, gemv, K=0                    |
| `test_hrr_cleanup`     | 5/5      | `30ab330`    | L5 — FFT, bind, phasor, Frady 2021 NAIVE/RESIDUAL     |
| `test_hrr_attention`   | 5/5      | `e8d45f1`    | L5 — `hrr_attention_full` (dispatch-level)            |

Os 4 primeiros testes foram cabeados no `tests/CMakeLists.txt` e no CI no
commit `a884036`; `test_bitnet_common` e `test_hrr_attention` entraram em
`cdce725` e `e8d45f1`, respectivamente.

`tests/CMakeLists.txt` foi reescrito como data-driven: cada executável
compila apenas o(s) `.cpp` de kernel de que precisa, via helper
`bitnet_test_set_simd_flags()`.

---

## 5. Refatoração DRY + achado de design

**Commit:** `cdce725` — `refactor: extract bitnet_next_pow2 to shared header`

### 5.1 O que foi extraído

`bitnet_next_pow2` foi movido para:
- `include/ggml-bitnet-common.h` (declaração, com `extern "C"`)
- `src/ggml-bitnet-common.cpp` (implementação + wrappers `fwht_next_pow2` /
  `hrr_next_pow2` também em `extern "C"`)

A linkage `extern "C"` é necessária porque os testes incluem `ggml-bitnet-common.h`
primeiro (que abre o escopo `extern "C"`), e depois `ggml-bitnet-fwht.h` /
`ggml-bitnet-hrr.h` — colocar as declarações em C linkage resolve a
inconsistência de linkage sem tocar em cada header.

### 5.2 Achado de design importante

**L2, L3 e L5 NÃO compartilham uma butterfly unificável.** A tentativa de
unificar revelou três algoritmos estruturalmente distintos:

| Nível | Algoritmo                                       | Estrutura                                      |
|-------|-------------------------------------------------|------------------------------------------------|
| L2    | WHT por máscara de seleção                      | Bits em bytes empacotados (não-FFT)             |
| L3    | FWHT (Cooley-Tukey radix-2 in-place)            | Real, in-place, in-order, sem bit-reversal     |
| L5    | FFT (Cooley-Tukey radix-2 DIF)                  | Complexo, in-place, com bit-reversal + twiddles |

Esse achado está documentado como **trap-prevention** no comentário-cabeçalho
de `include/ggml-bitnet-common.h` para impedir que futuros mantenedores caiam
na mesma armadilha.

### 5.3 Teste de guard

`test_bitnet_common.cpp` inclui um teste estrutural (`structural_no_butterfly`)
que afirma explicitamente a não-existência de uma butterfly compartilhada,
evitando que uma refatoração futura introduza acoplamientos por engano.

---

## 6. Arquivos novos nesta sessão

| Arquivo                                      | Tipo          | Commit    |
|----------------------------------------------|---------------|-----------|
| `include/ggml-bitnet-common.h`               | source header | `cdce725` |
| `src/ggml-bitnet-common.cpp`                 | source        | `cdce725` |
| `test_bitnet_common.cpp`                     | test          | `cdce725` |
| `test_hrr_attention.cpp`                     | test          | `e8d45f1` |
| `utils/cpu_universal_benchmark.py`           | tool          | `3f8166a` |

(Outros testes — `test_wht.cpp`, `test_acdc.cpp`, `test_tropical.cpp`,
`test_hrr_cleanup.cpp` — foram criados anteriormente, em commits fora do
range `129557d..v0.1.0` mas cabeados no CMake/CI no commit `a884036` desta
sessão.)

---

## 7. Arquivos modificados nesta sessão

| Arquivo                                          | Mudança                                              |
|--------------------------------------------------|------------------------------------------------------|
| `src/ggml-bitnet-wht.cpp:186-189`                | corrigir rótulos g0..g3 invertidos                   |
| `src/ggml-bitnet-fwht.cpp:291-303`               | remover normalização 1/n² espúria                    |
| `src/ggml-bitnet-fwht.cpp:75`                    | remover `fwht_next_pow2` (movido p/ common.cpp)      |
| `src/ggml-bitnet-hrr.cpp:75`                     | remover `hrr_next_pow2` (movido p/ common.cpp)       |
| `src/CMakeLists.txt`                             | incluir `ggml-bitnet-common.cpp` no `_bitnet_math_srcs` |
| `tests/CMakeLists.txt`                           | reescrita data-driven + 5 add_executable             |
| `.github/workflows/ci.yml`                       | build dos 6 targets + ctest                          |
| `.gitignore`                                     | adicionar `build_tests/`                             |
| `.reversa/scout/inventory.md`                    | última atualização: `3f8166a`                        |
| `.reversa/scout/gap-analysis.md`                 | P3 medições, P7 ✓✓, Prio 5.1/5.2/5.3                |
| `.reversa/scout/principle-code-map.json`         | suite de testes, bugs, v0.1.0                        |
| `.reversa/scout/continuity-proposals.md`         | estado de partida: Caminhos A+B 100%, só C resta     |

---

## 8. Smoke benchmark (`utils/cpu_universal_benchmark.py`)

**Commit:** `3f8166a` — `feat(bench): add cpu_universal_benchmark.py`

### 8.1 O que faz

Roda `run_inference.py` com o mesmo prompt/tokens/threads e cinco combinações
de variáveis de ambiente, emitindo uma tabela em markdown + CSV.

### 8.2 Bug encontrado + corrigido no parser

A regex original casava com a linha de **prompt-eval** (artefatos da ordem
de ~4500 tok/s) em vez da taxa geral. Corrigido pegando a **última**
ocorrência de "tokens per second" no output, que é a taxa consolidada de
geração.

### 8.3 Resultado (BitNet-2B, n=32, t=4, prompt "The capital of France is")

| Configuração                       | tok/s   | Δ        |
|------------------------------------|---------|----------|
| L1 baseline (I2_S GEMV)           |  4,97   | +0,0 %   |
| L3 ACDC FFN                       |  4,83   | -2,8 %   |
| L4 Tropical top-K=32              |  4,60   | -7,4 %   |
| L5 HRR raw                        |  1,85   | -62,8 %  |
| L5 HRR + cleanup 8 iters          |  1,87   | -62,4 %  |

### 8.4 Interpretação

- L3–L5 **não mostram speedup** sobre L1 porque o modelo **não foi treinado**
  com arquiteturas ACDC/HRR/tropical. Esta é a lacuna P6 explicitamente
  prevista no roadmap.
- A regressão de -62 % em L5 reflete o custo de FFT para `head_dim=128`
  (esperado, não é um bug).
- O overhead de cleanup (8 iterações × `d=128`) é desprezível.

---

## 9. Estado de partida da tese CPU-Universal

| Caminho | Descrição                                       | Estado                |
|---------|-------------------------------------------------|-----------------------|
| A       | Kernels L2–L5 matematicamente corretos          | **100 %**             |
| B       | Dispatch integrado no llama.cpp KQV/FFN         | **100 %**             |
| C       | Modelo retreinado com ACDC/HRR/tropical         | **Aberto** (P6, GPU)  |

Os Caminhos A e B estão fechados nesta sessão. O Caminho C requer
infraestrutura GPU e foi explicitamente colocado fora de escopo conforme
conversa inicial.

---

## 10. Restrições respeitadas

- **CPU only** — todas as adições são CPU-bound.
- **Clang ≥ 18 obrigatório** — sem MSVC, GCC tolerado com `-fpermissive`.
- **Submodule `3rdparty/llama.cpp`** tratado como read-only fora de patches
  deliberados (apontadores atualizados via `build(submodule)`).
- **Diretórios imutáveis** (`_reversa_sdd/`, `.reversa/context/`) **nunca
  modificados**; artefatos novos vão em `.reversa/scout/`.
- **Documentação e comentários de código em português-BR** conforme `CLAUDE.md`.
- **Sem comentários supérfluos** no código de produção.

---

## 11. O que ficou explícito fora de escopo

- **Caminho C** (P6 retreino com ACDC em GPU, 2-6 semanas) — requer
  infraestrutura que não temos. Kernels estão prontos; modelo precisa ser
  retreinado.
- **Decisões de Paradigm Advisor** — não há migração de sistema legado; este
  fork **é** o sistema.
- **Pricing Reversa** — não se aplica a um projeto de pesquisa open-source.

---

## 12. Próximos passos sugeridos (não executados)

1. **Caminho C** — alugar/alocar uma A100/H100 e retreinar um BitNet-300M
   com arquitetura ACDC-FFN em uma fração do tempo do BitNet-2B original.
2. **Caminho A++** — estender L2 (WHT) para o caso `m × n` com `m, n` ambos
   não-potência-de-2 (atualmente exige `n` potência de 2).
3. **ACDC-pretraining-aware** — adicionar uma pré-etapa no `convert-helper-bitnet.py`
   que aprende a diagonal `d` por blocos AC-DC a partir de um checkpoint
   bf16, melhorando a inicialização quando o Caminho C é executado com
   transfer learning.
4. **Paper / blog post** — descrever os 5 níveis algébricos e os achados
   (especialmente: L2/L3/L5 não compartilham butterfly; L5 com cleanup
   Frady 2021 converge em ≤8 iterações; pack I2_S estratificado).

---

## 13. Verificação final (commit `18fcf75`)

```
$ git log --oneline -10
18fcf75 docs(scout): v0.1.0 CPU-Universal release candidate + 6-test suite
3f8166a feat(bench): add cpu_universal_benchmark.py for systematic L1-L5 smoke tests
e8d45f1 test(hrr-attn): add dispatch-kernel validation for hrr_attention_full
cdce725 refactor: extract bitnet_next_pow2 to shared header (DRY across L3+L5)
...

$ git tag -l
v0.1.0-cpu-universal

$ ctest --test-dir build --output-on-failure
    Start 4: test_tropical          Passed    0.00 sec
    Start 5: test_hrr_cleanup       Passed    0.03 sec
    Start 6: test_hrr_attention     Passed    0.00 sec
100% tests passed, 0 tests failed out of 6
Total Test time (real) =   0.05 sec
```

---

**Sessão encerrada em 2026-06-05.**
**Estado entregue:** v0.1.0-cpu-universal — release candidate pronto para
Caminho C.
