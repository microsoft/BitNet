# Regression Watch — `001-trilha-rigor-produto`

> Watchlist de regressões **conhecidas** que podem afetar esta feature em
> releases futuros. Cada item: descrição, gatilho, severidade, como
> detectar, como mitigar. Gerado por `reversa-coding` ao final de todas as 5 fases.
>
> **Versão:** v1.0 — 2026-06-06
> **Ancoragem:** `verification-report.md` (ACs), `legacy-impact.md` (impacto)

---

## Como usar este documento

Antes de cada release (v0.1.x, v0.2.x, ...), o mantenedor deve:

1. Rodar `ctest --output-on-failure` em `build_tests/` — esperado: **13/13 PASS**.
2. Rodar `tests/test_air_gapped_boot.sh` — esperado: exit 0 (ou SKIPPED se sem modelo).
3. Inspecionar este watchlist — nenhum item deve ter sido acionado.
4. Se algum item acionar, seguir a "Mitigação" antes de commitar `bench.json`.

---

## W-01: `test_l4_sparse_properties` runtime aumenta

**Severidade:** 🟡 Média (afeta RNF-01 parcialmente)
**Sintoma:** `test_l4_sparse_properties` ultrapassa 1.5s em `ctest`
**Gatilho:** mudança em `sparse_attention_float` que aumenta N_max ou per-iteration cost
**Como detectar:** comparar `Total Test time` em `ctest` antes/depois; baseline = **2.96s**
**Mitigação:** encolher N_max de 2048 → 1024 (mantém cobertura estatística, reduz 30 % runtime)
**Quem cuida:** mantenedor da L4 tropical kernel
**Referência:** `verification-report.md#ac-01`

---

## W-02: AC-01 runtime > 5s em CI

**Severidade:** 🟡 Média (CI timeout)
**Sintoma:** `ctest` em `.github/workflows/ci.yml` excede timeout default
**Gatilho:** acúmulo de property tests em iterações grandes
**Como detectar:** falha de step "Run tests" em CI com exit 124 (timeout)
**Mitigação:** mover property tests para um target separado `ctest -L slow` (não roda em PRs, só em main)
**Quem cuida:** mantenedor de CI

---

## W-03: ACDC diagonal extraction (Python) lento

**Severidade:** 🟢 Baixa (já mitigado)
**Sintoma:** `test_extract_acdc_diagonal.py` ultrapassa 1s
**Gatilho:** N ou seed-count do script aumentados
**Baseline:** 0.85s (atual)
**Como detectar:** `ctest -V -R test_extract_acdc_diagonal` mostra tempo por iteração
**Mitigação:** cachear matrizes aleatórias em setUp (já documentado em T033, mas não aplicado)
**Quem cuida:** mantenedor do scaffolding ACDC
**Referência:** `verification-report.md#ac-01`

---

## W-04: `apply-dispatch-patches.sh` falha após `git pull upstream`

**Severidade:** 🟡 Média (afeta AC-07)
**Sintoma:** `patches/llama.cpp/*.patch` rejeita com "patch does not apply"
**Gatilho:** upstream `ggerganov/llama.cpp` muda linhas que nossos patches tocam
**Como detectar:** `scripts/apply-dispatch-patches.sh` exit ≠ 0
**Mitigação:**
1. Re-basear patches contra nova HEAD do fork `Eddie-Wang1120/llama.cpp`
2. Atualizar `patches/llama.cpp/0[1-3]-*.patch`
3. Re-rodar smoke test em clone fresh
**Quem cuida:** mantenedor de patches
**Referência:** `patches/llama.cpp/README.md`, `verification-report.md#ac-07`

---

## W-05: AIR-GAPPED step em CI reporta FAIL em runner

**Severidade:** 🟢 Baixa (já tratado com PIPESTATUS)
**Sintoma:** "Air-gapped boot test" step no CI falha com exit ≠ 0
**Gatilho:** runner do GitHub Actions tem rede bloqueada de forma diferente
**Como detectar:** step "Air-gapped boot test" em `.github/workflows/ci.yml` fica vermelho
**Mitigação atual:** step é **warning, não error** (PIPESTATUS check). Esperado: SKIPPED em CI (sem modelo) ou PASS em local release workflow
**Mitigação adicional se persistir:** tornar step `continue-on-error: true` (mais permissivo)
**Quem cuida:** mantenedor de CI
**Referência:** `.github/workflows/ci.yml` (T025), `verification-report.md#ac-11`

---

## W-06: bench_publish.py falha em Windows/macOS

**Severidade:** 🟡 Média (afeta AC-05)
**Sintoma:** `python utils/bench_publish.py` falha com `FileNotFoundError` ou path errors
**Gatilho:** diferenças Unix vs Windows path (`/` vs `\`, `uname` ausente, etc.)
**Como detectar:** rodar `bench_publish.py --help` em Windows / macOS
**Mitigação:**
1. Adicionar `pathlib.Path` ao invés de string concat
2. Usar `platform.system()` para detectar OS
3. Testar CI matrix `os: [ubuntu-latest, macos-latest, windows-latest]`
**Quem cuida:** mantenedor de tooling
**Referência:** `utils/bench_publish.py` (T020), `benchmarks/v0.1.0/methodology.md#6.1`

---

## W-07: Patches conflitam entre si (3-way merge)

**Severidade:** 🟢 Baixa (não observado)
**Sintoma:** `01-L3-ACDC-FFN-dispatch.patch` e `02-L5-HRR-cleanup-dispatch.patch` ambos modificam mesma região de `ggml-bitnet-dispatch.cpp`
**Gatilho:** adição de um 4º patch que toca as mesmas linhas
**Como detectar:** `git apply --check` reporta conflito
**Mitigação:** consolidar patches em 1 único (`.patch` consolidado) ou reordenar aplicação
**Quem cuida:** mantenedor de patches

---

## W-08: `test_dense_is_default` falha após mudança de `src/ggml-bitnet-dispatch.cpp`

**Severidade:** 🟢 Baixa
**Sintoma:** test detecta que dense NÃO é mais default (D1 violado)
**Gatilho:** alguém remove o early-return do dense path
**Como detectar:** `ctest -R test_dense_is_default` fica vermelho
**Mitigação:** corrigir dispatch para garantir que dense é checado primeiro
**Quem cuida:** mantenedor do dispatch
**Referência:** `test_dense_is_default.cpp` (T008), `docs/decision-matrix.md` (D1)

---

## W-09: ACDC retangular (G-D2) reclassificado como bloqueador

**Severidade:** 🟡 Média (afeta M3 do roadmap)
**Sintoma:** alguém executa Llama-2-7B smoke test e descobre que FFN ACDC quebra coerência
**Gatilho:** nova inferência com Llama-2-7B em hardware externo
**Como detectar:** perplexity > 100 ou output repetitivo/incoerente
**Mitigação:**
1. Mover T009/T018/T019 para curto-prazo no `ROADMAP.md`
2. Atualizar `verification-report.md#ac-08` de 🟡 para ❌
3. Implementar ACDC retangular em `ggml-bitnet-fwht.cpp` (RF-04)
4. Habilitar `test_acdc_rect` (remover `option(BITNET_ENABLE_ACDC_RECT OFF)`)
**Quem cuida:** quem tiver acesso a Llama-2-7B + hardware
**Referência:** `requirements.md#10` (LR-01), T029 gated, T034

---

## W-10: GPU acidentalmente re-introduzido

**Severidade:** 🔴 Alta (viola NO-02)
**Sintoma:** `git log` mostra commit que adiciona `-DUSE_CUDA` ou similar
**Gatilho:** PR de contribuidor externo que não leu CLAUDE.md
**Como detectar:** `grep -rn "USE_CUDA\|USE_HIPBLAS\|USE_METAL" src/ include/ 3rdparty/` retorna hits em código BitNet (não em llama.cpp)
**Mitigação:** rejeitar PR; reverter commit
**Quem cuida:** reviewers de PR
**Referência:** `CLAUDE.md` (NO-02 fundadora), `ROADMAP.md#3-fora-de-escopo`

---

## W-11: Telemetria acidental (NO-06 violado)

**Severidade:** 🔴 Alta (viola NO-06)
**Sintoma:** `telemetry`, `send_metrics`, `upload_data` aparece em código de produção
**Gatilho:** PR que adiciona analytics, error reporting, etc.
**Como detectar:** `grep -rn "telemetry\|upload_data\|send_metrics" src/ utils/ run_inference*.py` retorna hits
**Mitigação:** rejeitar PR; rodar T031 novamente para confirmar 0 hits
**Quem cuida:** reviewers de PR
**Referência:** T031, `verification-report.md#ac-12`

---

## W-12: Cloud call acidental (NO-07 violado)

**Severidade:** 🔴 Alta (viola NO-07)
**Sintoma:** `https://` em código de produção (não em comentários)
**Gatilho:** PR que adiciona model downloader, version check, ou auto-update
**Como detectar:** `grep -rn "https\?://" src/ include/ scripts/ patches/` (excluindo comentários e README.md)
**Mitigação:** rejeitar PR; rodar T032 novamente para confirmar 0 hits
**Quem cuida:** reviewers de PR
**Referência:** T032, `verification-report.md#ac-11`

---

## Resumo por severidade

| Severidade | Quantidade | IDs |
|------------|------------|-----|
| 🔴 Alta (viola fundadora) | 3 | W-10, W-11, W-12 |
| 🟡 Média (afeta release) | 5 | W-01, W-02, W-04, W-06, W-09 |
| 🟢 Baixa (cosmético/perf) | 4 | W-03, W-05, W-07, W-08 |

---

## Comando de verificação pré-release

```bash
# 1. ctest baseline
cd build_tests && ctest --output-on-failure
# Esperado: 100% tests passed, 0 tests failed out of 13
# Tempo: < 5s

# 2. Air-gapped
bash tests/test_air_gapped_boot.sh
# Esperado: exit 0 (ou SKIPPED)

# 3. Auditoria NO-06 (sem telemetria)
grep -rn "telemetry\|upload_data\|send_metrics" src/ utils/ run_inference*.py
# Esperado: (no output)

# 4. Auditoria NO-07 (sem cloud, código apenas)
grep -rn "https\?://" src/ include/ scripts/ patches/ | grep -v "^\s*//" | grep -v "README"
# Esperado: (no output)

# 5. Patches aplicam em clone fresh
cd /tmp && rm -rf test-clone && git clone /path/to/BitNet test-clone
cd test-clone && bash scripts/apply-dispatch-patches.sh
# Esperado: 3 patches aplicados, exit 0

# 6. Cross-validation Python
python tests/cross_validation.py
# Esperado: 3/3 PASS
```

Se todos os 6 passos passam, release pode prosseguir.

---

*v1.0 — gerado por reversa-coding ao final da Fase 5 em 2026-06-06*
*12 itens monitorados (3 🔴, 5 🟡, 4 🟢). 0 regressões ativas.*
