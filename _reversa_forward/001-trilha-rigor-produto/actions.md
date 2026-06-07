# Actions — `001-trilha-rigor-produto`

> Decomposição executável do `roadmap.md` em ações atômicas com IDs estáveis.
> **Versão:** v1.5 (pós Fases 1+2+3+4+5 — T001-T008, T010-T017, T020-T028, T030-T035, T036 ✅ em 2026-06-07T00:30:00Z; T009, T018, T019, T029 gated by D2)
> **Ancoragem:** `roadmap.md` v1.5, `data-delta.md` v1, `requirements.md` v2
> **Outputs finais:** `legacy-impact.md` v1.0, `regression-watch.md` v1.0

---

## Resumo Executivo

| Métrica | Valor |
|---------|-------|
| **Total de ações** | 36 |
| **Ações paralelizáveis `[//]`** | 20 (56%) |
| **Maior cadeia de dependência** | T005 → T024 → T033 (3 níveis); T011 → T033 (2 níveis); T018 → T019 → T034 (3 níveis); T036 → T033 (2 níveis) |
| **Ações por fase** | Preparação: 4 · Testes: 8 · Núcleo: 12 · Integração: 7 · Polimento: 5 |
| **Ações condicionais (gate D2)** | 1 (T009, T018, T019, T034 dependem do trigger D2) |
| **Ações em reserva (gate D3 Q4 2029)** | 0 (M4 é só doc, não código) |

**Gates (bloqueios condicionais):**
- **G-D2**: T009, T018, T019, T034 executam **apenas se** a investigação D2 (T029) confirmar "ACDC retangular vira bloqueador". Caso contrário, ficam pausadas em `requirements.md#11` (LR-01).
- **G-M3**: T015 (decision-matrix) menciona ACDC retangular; se D2 não dispara, T015 marca como "diferencial, não bloqueador".

---

## Fase 1: Preparação

> Setup, scaffolding, verificações iniciais. Tudo é pre-requisito das fases seguintes.

| ID | Descrição | Dependências | Paralelismo | Arquivo alvo | Confiança | Status |
|----|-----------|--------------|-------------|--------------|-----------|--------|
| T001 | Verificar baseline: `cd build_tests && ctest --output-on-failure` deve passar 9/9, ≥50 subtests | — | — | `build_tests/` | 🟢 | [X] |
| T002 | Criar diretórios novos: `mkdir -p examples/ tests/snapshots/ benchmarks/` | — | [//] | `examples/`, `tests/snapshots/`, `benchmarks/` | 🟢 | [X] |
| T003 | Verificar se Catch2 v3 já está disponível via `find_package(Catch2 REQUIRED)` no `tests/CMakeLists.txt`; se não, documentar a inclusão | T001 | [//] | `tests/CMakeLists.txt` | 🟢 | [X] |
| T004 | Criar esqueleto de `docs/invariants.md` com 7 seções P1-P7 (placeholders) | T002 | [//] | `docs/invariants.md` | 🟢 | [X] |

**Subtotal fase 1:** 4 ações (2 paralelizáveis, 0 condicionais).

---

## Fase 2: Testes (TDD)

> Testes são escritos **antes** do núcleo. Esta fase produz os tests que vão falhar até a fase 3 (núcleo) ser executada.

| ID | Descrição | Dependências | Paralelismo | Arquivo alvo | Confiança | Status |
|----|-----------|--------------|-------------|--------------|-----------|--------|
| T005 | [//] `tests/test_acdc_properties.cpp` com 4 invariantes: (1) `\|\|d*\|\| ≤ \|\|W\|\|/sqrt(n)`, (2) `H·diag(d*)·H = W_proj` exato, (3) energia `n²·\|\|d*\|\|² ≈ \|\|W_proj\|\|²`, (4) determinismo: 2 chamadas com mesma seed dão mesmo resultado. 1000 iterações cada. | T003 | [//] | `tests/test_acdc_properties.cpp` | 🟢 | [X] |
| T006 | [//] `tests/test_l4_sparse_properties.cpp` com 3 invariantes: (1) `argmax(sparse_topK(W·x, K=32)) ⊆ argmax(W·x)`, (2) `len(topK_indices) == K`, (3) `sum(weights_topK) ≤ sum(weights_full)`. | T003 | [//] | `tests/test_l4_sparse_properties.cpp` | 🟢 | [X] |
| T007 | [//] `tests/test_hrr_properties.cpp` com 3 invariantes: (1) `unbind(bind(a, b), b) ≈ a` com `rtol=1e-3` (HRR tem ruído por P6), (2) `\|FFT(x)\| = \|x\|` (Parseval), (3) `hrr_cleanup_iter(M, q, cb, N) ∈ cb` após convergência. | T003 | [//] | `tests/test_hrr_properties.cpp` | 🟢 | [X] |
| T008 | [//] `tests/test_dense_is_default.cpp` (D-T-01): verifica que **sem** env var `BITNET_SPARSE_TOPK`, o dispatch em `src/ggml-bitnet-dispatch.cpp` NÃO invoca `sparse_attention_float()`. Usa mock ou inspeção de call count. | T003 | [//] | `tests/test_dense_is_default.cpp` | 🟢 | [X] |
| T009 | `tests/test_acdc_rect.cpp` para ACDC retangular (2560×6912, 6912×2560, 32×48, 64×128). Verifica que `acdc_project_rect(W)` retorna `d ∈ ℝ^{min(m,n)}` com energia ≥ 1/n. **Gate D2**: só executar este test se T029 confirmar trigger. | T003, T029, G-D2 | — | `tests/test_acdc_rect.cpp` | 🟡 | [ ] |
| T010 | `tests/test_air_gapped_boot.sh` (AC-11): shell script que roda `unshare -rn ./build/bin/llama-cli -m ... -p "Test" -n 10` e valida que exit code = 0 e log não contém "telemetry" / "upload" / "error". | T002, T011 | — | `tests/test_air_gapped_boot.sh` | 🟢 | [X] |
| T011 | `tests/cross_validation.py`: orquestra C test + Python reference com seeds idênticas; compara com `np.testing.assert_allclose(rtol=1e-5, atol=1e-7)`. Suporta ACDC, sparse, HRR. | T002 | [//] | `tests/cross_validation.py` | 🟡 | [X] |
| T012 | `tests/snapshots/<kernel>_v0.1.0.txt`: 1 snapshot por kernel (ACDC, sparse, HRR). Gerado por `tests/snapshots/generate.py` (helper) a partir de seeds fixas. | T002 | [//] | `tests/snapshots/` | 🟢 | [X] |

**Subtotal fase 2:** 8 ações (6 paralelizáveis, 1 condicional [T009]). **Status pós-Fase 2:** 7/8 [X] (T005-T008, T010, T011, T012; T009 pendente gate D2).

---

## Fase 3: Núcleo

> Lógica central: implementações, documentações, scripts. Esta fase faz os tests da fase 2 passarem.

| ID | Descrição | Dependências | Paralelismo | Arquivo alvo | Confiança | Status |
|----|-----------|--------------|-------------|--------------|-----------|--------|
| T013 | `docs/invariants.md` (versão final): lista canônica P1-P7 com referência a `docs/theory/`, `.reversa/scout/principles.md`, e ao test que verifica cada invariante (cross-link para `tests/test_*`). | T004 | — | `docs/invariants.md` | 🟢 | [X] |
| T014 | `ROADMAP.md` (raiz do projeto) com 3 seções: **Atual** (v0.1), **Reserva técnica** (RF-06 com data Q4 2029), **Fora de escopo** (GPU kernels, P6 real, cloud). | T002 | [//] | `ROADMAP.md` | 🟢 | [X] |
| T015 | `docs/decision-matrix.md` (RF-02): tabela "Cenário → Kernel" com 5 linhas (BitNet-2B denso, sparse opt-in, FFN P6-ACDC, edge d≥256 P6-HRR, pesquisa L2). Referência a `requirements.md#9` para persona D4. | T013, T014 | — | `docs/decision-matrix.md` | 🟢 | [X] |
| T016 | `docs/hardware-compatibility.md` (AC-13): tabela CPU → modo suportado (L1 baseline OK, L2/L3/L4 com flag, L5 só com d ≥ 256), com testes em hardware mínimo documentados. | T013 | [//] | `docs/hardware-compatibility.md` | 🟢 | [X] |
| T017 | `src/ggml-bitnet-tropical.cpp`: adicionar bloco de comentário Doxygen acima de `sparse_attention_float()` declarando que é opt-in via `BITNET_SPARSE_TOPK`. Sem mudança de comportamento. | T008 | — | `src/ggml-bitnet-tropical.cpp` | 🟢 | [X] |
| T018 | `src/ggml-bitnet-fwht.cpp`: implementar `acdc_project_rect(W, m, n)` para matrizes m×n com m ≠ n. Usa Kronecker `H_m ⊗ H_n` (D-T-07). Padding zero para próxima power-of-2. **Gate D2**: só commitar se T029 confirmar. | T013, T029, G-D2 | — | `src/ggml-bitnet-fwht.cpp` | 🟡 | [ ] |
| T019 | `utils/extract_acdc_diagonal.py`: estender para shapes retangulares (FFN gate/up 2560×6912, down 6912×2560). Salva sidecar `.npz` (D-T-07, data-delta v0.2). **Gate D2**: depende de T018. | T018, G-D2 | — | `utils/extract_acdc_diagonal.py` | 🟡 | [ ] |
| T020 | `utils/bench_publish.py` (RF-07): CLI com 2 modos. Modo 1: roda `utils/cpu_universal_benchmark.py` e gera JSON canônico + Markdown derivado. Modo 2: lê JSON e renderiza Markdown. Argumentos: `--json`, `--md`, `--from-json`. | T012 | [//] | `utils/bench_publish.py` | 🟡 | [X] |
| T021 | [//] `examples/medical_offline.md`: walkthrough persona D4 — médico analisa prontuário em laptop de consultório. Comandos exatos (sem rede, inferência local, ~30s para 200 tokens). | T015 | [//] | `examples/medical_offline.md` | 🟢 | [X] |
| T022 | [//] `examples/legal_offline.md`: walkthrough persona D4 — advogado resume petição em escritório. | T015 | [//] | `examples/legal_offline.md` | 🟢 | [X] |
| T023 | [//] `examples/finance_offline.md`: walkthrough persona D4 — analista financeiro categoriza despesas. | T015 | [//] | `examples/finance_offline.md` | 🟢 | [X] |
| T036 | `docs/theory/06-5-levels.md` (AC-10): resumo canônico de 1 página dos 5 níveis algébricos (L1 I2_S, L2 WHT, L3 ACDC, L4 tropical, L5 HRR) com tabela "Nível → Operação eliminada → Substituída por → Ganho". Conteúdo **consolidado** a partir de `docs/mathematical-foundations.md` (que já cobre os 5 níveis) e `docs/findings-cpu-universal.md#1`. **NÃO** substitui os docs primários; é um sumário. | T013 | [//] | `docs/theory/06-5-levels.md` | 🟢 | [X] |

**Subtotal fase 3:** 12 ações (7 paralelizáveis, 2 condicionais [T018, T019]). **Status pós-Fase 3:** 10/12 [X] (T018, T019 gated by D2).

---

## Fase 4: Integração

> Conectar o núcleo ao build, CI, e fluxo de release. Inclui a investigação D2 (T029), que é o **gate** para ACDC retangular.

| ID | Descrição | Dependências | Paralelismo | Arquivo alvo | Confiança | Status |
|----|-----------|--------------|-------------|--------------|-----------|--------|
| T024 | `tests/CMakeLists.txt`: adicionar 4 alvos novos (`test_acdc_properties`, `test_l4_sparse_properties`, `test_hrr_properties`, `test_dense_is_default`) + 1 alvo condicional (`test_acdc_rect`) com `if(G-D2)`. Cada alvo herda flags SIMD de `bitnet_test_set_simd_flags()`. | T005, T006, T007, T008 | — | `tests/CMakeLists.txt` | 🟢 | [X] |
| T025 | `.github/workflows/ci.yml`: adicionar step "Air-gapped boot test" que executa `tests/test_air_gapped_boot.sh` em um job separado. Tempo esperado: ~1 min. | T010, T024 | — | `.github/workflows/ci.yml` | 🟢 | [X] |
| T026 | `tests/test_air_gapped_boot.sh` (script final): usar `unshare -rn` + `strace -e network -f` se primeira tentativa falhar. Exit code 0 = pass. | T010 | [//] | `tests/test_air_gapped_boot.sh` | 🟢 | [X] |
| T027 | `docs/findings-cpu-universal.md`: adicionar seção "## Persona Alvo" com cross-link para `requirements.md#9`. | T015, T016 | [//] | `docs/findings-cpu-universal.md` | 🟢 | [X] |
| T028 | `README.md` (reescrita persona D4): headline "Inferência 1.58-bit local-first, sem CUDA, sem cloud", casos de uso D4, instalação, build, link para `examples/`. Preserva `docs/theory/` como referência. | T015, T021, T022, T023, T027 | — | `README.md` | 🟢 | [X] |
| T029 | `investigation-d2-result.md` (gate D2): documento que registra o resultado do smoke test com Llama-2-7B. Estrutura: comando executado, output (perplexity, sample de texto), conclusão ("bloqueador" ou "diferencial"). Atualiza `requirements.md#11` (LR-01) com o resultado. | T001 | — | `investigation-d2-result.md` (na raiz) | 🟡 | [ ] |
| T030 | `benchmarks/v0.1.0/`: executar `utils/bench_publish.py --json > benchmarks/v0.1.0/bench.json && python utils/bench_publish.py --from-json benchmarks/v0.1.0/bench.json --md > benchmarks/v0.1.0/bench.md`. Commitar ambos + `methodology.md`. | T020 | — | `benchmarks/v0.1.0/` | 🟡 | [X] |

**Subtotal fase 4:** 7 ações (2 paralelizáveis, 1 condicional via T029). **Status pós-Fase 4:** 6/7 [X] (T029 gated by D2, requer Llama-2-7B + horas de inferência fora do escopo CPU-only).

---

## Fase 5: Polimento

> Auditorias finais, validação de critérios de aceitação, NO-06/NO-07 enforcement.

| ID | Descrição | Dependências | Paralelismo | Arquivo alvo | Confiança | Status |
|----|-----------|--------------|-------------|--------------|-----------|--------|
| T031 | [//] Auditoria NO-06 (sem telemetria): `grep -rn "telemetry\|upload_data\|send_metrics\|POST.*http" src/ utils/ run_inference*.py 2>&1 | tee /tmp/no06.log`. Esperado: 0 hits (ou apenas comentários em código explicando por que é desabilitado). | T001 | [//] | `/tmp/no06.log` | 🟢 | [X] |
| T032 | [//] Auditoria NO-07 (sem cloud): `grep -rn "http://\|https://" src/llama.cpp 3rdparty/llama.cpp 2>&1 | grep -v 'comment\|//' | tee /tmp/no07.log`. Esperado: 0 hits em código de produção (apenas `patches/llama.cpp/README.md` e `docs/`). | T001 | [//] | `/tmp/no07.log` | 🟢 | [X] |
| T033 | Validar AC-01 a AC-13: rodar ctest, verificar cada critério na tabela `requirements.md#6`, gerar `verification-report.md` com tabela `AC-XX | status (verde/vermelho) | evidência (arquivo:linha) | nota`. Verde só se a evidência for concreta. | T005, T006, T007, T008, T011, T012, T015, T018, T024, T025, T027, T028, T030, T036 | — | `verification-report.md` | 🟢 | [X] |
| T034 | Avaliar gate D2: se T029 confirmou "bloqueador", mover M3 (T009, T018, T019) para curto-prazo. Se "diferencial", manter T009/T018/T019 como pausa (LR-01). Atualizar `requirements.md#11`. | T029, T033 | — | `requirements.md#11` (edição) | 🟢 | [X] |
| T035 | Adicionar reminder Q4 2029 ao `ROADMAP.md`: seção "## Reavaliações agendadas" com data e gatilho. Tornar visível na abertura do ROADMAP. | T014, T033 | — | `ROADMAP.md` (edição) | 🟢 | [X] |

**Subtotal fase 5:** 5 ações (2 paralelizáveis, 1 condicional via T034). **Status pós-Fase 5:** 5/5 [X] (T034 resolveu D2 gate: pausa mantida por falta de Llama-2-7B; gate é hardware-side, não código-side).

---

## Mapa de Dependências (visual)

```
                              T001 (baseline)
                                 │
                                 ├──> T002 ──┐
                                 │            │
                                 │            ├──> T010 (air-gapped script)
                                 │            ├──> T011 (cross-val)
                                 │            ├──> T012 (snapshots)
                                 │            ├──> T014 (ROADMAP)
                                 │            ├──> T020 (bench_publish)
                                 │            └──> T031, T032 (audits)
                                 │
                                 ├──> T003 (Catch2 check) ──┐
                                 │                          │
                                 │                          ├──> T005 (ACDC props)
                                 │                          ├──> T006 (L4 props)
                                 │                          ├──> T007 (HRR props)
                                 │                          ├──> T008 (dense default)
                                 │                          └──> T009 (ACDC rect, G-D2)
                                 │
                                 └──> T004 (invariants skeleton) ──> T013 (invariants full)
                                 
T013 ──> T015 (decision-matrix) ──> T021, T022, T023 (examples) ──> T028 (README)
                                                                  ▲
T013 ──> T016 (hardware-compat) ───────────────────────────────────┤
                                                                  │
T013 ──> T036 (5-levels summary) ───────────────────────────────> T033
                                                                  │
T005,T006,T007,T008 ──> T024 (CMakeLists) ──> T025 (CI step)       │
                                                                  │
T010 ──> T026 (air-gapped script final) ────────────────────────> T028
                                                                  │
T011 ────────────────────────────────────────────────────────────> T028

T015, T016 ──> T027 (findings update) ──────────────────────────> T028

T020 ──> T030 (benchmarks v0.1.0) ──────────────────────────────> T033

T001 ──> T029 (D2 investigation Llama-2-7B) ──> G-D2 ──> T009, T018, T019
                                                   │
                                                   └──> T034 (avaliar gate)

T005,T006,T007,T008,T011,T012,T015,T018,T024,T025,T027,T028,T030,T036
                  │
                  └─────────────────────────────────────> T033 (valida AC-01..13, gera verification-report.md)
                                                              │
                                                              ├──> T034 (D2 gate)
                                                              │
                                                              └──> T035 (Q4 2029 reminder)
```

---

## Critérios de Quebra (Anti-fragmentação)

Ações seguem a regra "atômico = 1 turno, 1 agente, 1 assunto". Verificações:

- ✅ Cada ação tem ≤ 5 sub-pontos lógicos
- ✅ Cada ação toca ≤ 3 arquivos não-relacionados
- ✅ Nenhuma ação usa "e também" / "depois" / "em seguida" (exceto onde a sequência é parte da definição)
- ✅ IDs estáveis (não reciclados)

---

## Notas de Execução

### Para `/reversa-coding`:

- Comece por **Fase 1** (T001-T004) em ordem; nada depende de nada exceto T001 → T002/T003/T004
- Fase 2 (T005-T012) tem 6 ações `[//]` que podem ser feitas em paralelo por múltiplos agentes
- **NÃO execute T009, T018, T019 antes de T029** (gate D2). T029 (Llama-2-7B smoke test) é a primeira coisa da Fase 4 e bloqueia/desbloqueia M3
- Fase 3 (T013-T023) tem 7 ações `[//]` mas todas requerem T013/T015 como ancestrais
- **NÃO execute T028 (README) antes de T015, T021, T022, T023, T027** (dependências explícitas)
- Fase 5 (T031-T035) é onde tudo converge: T033 é a "validação de DoD"

### Para contribuidores externos (persona D4):

- Pegue uma ação `[//]` da Fase 2 ou Fase 3; são as mais isoladas
- Cada ação tem **arquivo alvo** explícito, então é fácil de localizar
- Status `[ ]` vira `[X]` quando concluída (pelo `/reversa-coding`)

---

*actions.md v1 — gerado por reversa-to-do em 2026-06-06*
