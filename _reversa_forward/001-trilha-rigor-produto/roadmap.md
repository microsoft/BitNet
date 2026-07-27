# Roadmap — `001-trilha-rigor-produto`

> **Feature:** Trilha de rigor teórico e fundamental para BitNet CPU-Universal como produto (privacy/sovereignty persona)
>
> **Versão:** v1 (gerado por reversa-plan em 2026-06-06)
> **Ancoragem:** `requirements.md` v2 (pós-clarify) + `_reversa_sdd/` + `.reversa/scout/principles.md`
> **Idioma:** pt-BR

---

## 1. Resumo da Abordagem

Esta feature é **meta**: ela não implementa kernels novos. Ela estabelece a trilha de governança, decisão e validação que permite ao BitNet CPU-Universal evoluir da posição atual (5 kernels algébricos comprovados, 9/9 ctest, 50/50 subtests, sem integração em produção) até a posição de **produto viável para a persona D4** (Desenvolvedores de Privacidade e Soberania de Dados — ver `requirements.md#9`).

A abordagem é **delta incremental em 5 marcos** (M1-M5), cada um com entregas concretas e testáveis. Não há "big bang": cada marco pode ser shippado em produção de forma independente.

A persona D4 (privacidade/soberania) governa todas as decisões daqui em diante: o fork é posicionado como **ferramenta de inferência local para usuários que não podem ou não querem enviar dados para a nuvem**. Esta não é uma restrição técnica; é uma restrição de design que afeta marketing, exemplos, e o que entra/não entra no produto.

---

## 2. Princípios Aplicados (Verificação)

Cada um dos 7 princípios transversais em `.reversa/scout/principles.md` foi avaliado contra esta feature:

| Princípio | Status | Notas |
|-----------|--------|-------|
| **P1 — Shannon floor** | 🟢 Compatível | Não alteramos a codificação ternária. |
| **P2 — Identidade algébrica** | 🟢 Reforçado | RF-01 (property-based tests) verifica identidades algébricas automaticamente, fortalecendo o ctest como especificação executável. |
| **P3 — Hierarquia de custo** | 🟢 Compatível | RF-04 (ACDC retangular) mantém O(n log n); não compromete a hierarquia. |
| **P4 — Mínimo irredutível** | 🟢 Compatível | Não tentamos comprimir ACDC post-hoc (P6). RF-04 só faz sentido com modelo P6-treinado (reserva). |
| **P5 — Dequantização tropical** | 🟢 Compatível | L4 sparse é opt-in (D1); τ finito do softmax preservado. |
| **P6 — Estrutura, não compressão** | 🟢 Reforçado | RF-06 explicitamente classificada como "reserva técnica" (D3); AC-08 é "bloqueador condicional" (D2); persona D4 reforça a restrição. |
| **P7 — FFT como cola** | 🟢 Compatível | Header `ggml-bitnet-common.h` disciplina "sem compartilhamento de butterflies" (já existente; manter). |
| **Restrição fundadora CPU-only** (CLAUDE.md) | 🟢 Reforçado | Persona D4 (privacidade/soberania) é incompatível com GPU; alinhamento natural. |
| **Privacy/sovereignty (D4)** | 🟢 Novo | Persona governa AC-11 (air-gapped boot), AC-12 (exemplos single-user), NO-06 (sem telemetria), NO-07 (sem cloud). |

**Sem conflitos.** Nenhum princípio precisa ser reescrito ou atenuado. Esta feature é puramente aditiva em governança e produto.

---

## 3. Decisões Técnicas (Marcadas com Confiança)

### 3.1. Decisões de alto impacto

#### D-T-01: L4 sparse float é opt-in, não default
- **Fonte**: Esclarecimento D1 em `requirements.md#10`
- **Confiança**: 🟢 CONFIRMADO (decisão do usuário)
- **Implementação**: `src/ggml-bitnet-tropical.cpp` mantém `sparse_attention_float()` atrás de env var `BITNET_SPARSE_TOPK` ou flag CLI `--attn sparse`. Default = attention denso (comportamento atual preservado).
- **Risco**: Nenhum. Mantém compatibilidade com BitNet-2B e modelos similares. Usuário que quiser opt-in tem caminho claro.
- **Teste**: AC-06 (já existe, manter); adicionar `test_dense_is_default.cpp` que verifica que sem env var, sparse não é invocado.

#### D-T-02: AC-08 (ACDC retangular) é bloqueador condicional, não bloqueador imediato
- **Fonte**: Esclarecimento D2 em `requirements.md#10`
- **Confiança**: 🟢 CONFIRMADO (decisão do usuário, com trigger empírico)
- **Implementação**: M3 inicial é "médio prazo, 1-2 meses, diferencial". Trigger de reclassificação: executar inferência fim-a-fim com Llama-2-7B (fp16) através do pipeline BitNet; se FFN falhar, RF-04 vira bloqueador e M3 movido para curto-prazo.
- **Risco**: Decisão pode reverter. Se Llama-2-7B for bloqueado por FFN, recursos do M3 precisam ser realocados.
- **Mitigação**: O trigger D2 é uma **tarefa de investigação** de baixo custo, não um PR de feature. Pode ser feita como sub-feature de M1.

#### D-T-03: RF-06 (finetune_acdc.py) é reserva técnica com reavaliação Q4 2029
- **Fonte**: Esclarecimento D3 em `requirements.md#10`
- **Confiança**: 🟢 CONFIRMADO
- **Implementação**: Não criar `utils/finetune_acdc.py` em v0.1. Em vez disso, documentar em `ROADMAP.md` (a criar) que a reserva existe conceitualmente, sem código. Reavaliação: Q4 2029.
- **Risco**: Documentação sem código é mais fácil de esquecer que código documentado. Risco aceito: melhor explicitar que escrever código que ninguém vai usar.
- **Mitigação**: ROADMAP.md é vinculado do README.md principal; revisado em cada release.

#### D-T-04: Persona D4 (Privacidade/Soberania) governa produto
- **Fonte**: Esclarecimento D4 em `requirements.md#9`
- **Confiança**: 🟢 CONFIRMADO
- **Implementação**: 
  - `README.md` reescrito com headline "Inferência 1.58-bit local-first, sem CUDA, sem cloud"
  - `examples/` adicionado com cenários single-user, single-laptop, sem rede
  - `docs/decision-matrix.md` (RF-02) usa persona D4 como vetor de decisão
  - `tests/test_air_gapped_boot.sh` (AC-11) verifica que binário roda sem rede
  - NO-06 (sem telemetria) e NO-07 (sem cloud) documentados no `requirements.md#12`
- **Risco**: Reposicionamento de produto pode alienar contribuidores que vieram pelo lado "pesquisa acadêmica". Mitigação: manter `docs/theory/` intocado; a persona D4 é adicional, não substituta.
- **Confiança na execução**: 🟡 INFERIDO — assumimos que a persona D4 é estável até Q4 2029 (reavaliação em LR-03).

### 3.2. Decisões de médio impacto

#### D-T-05: Property-based tests usam Catch2 GENERATE macro, não biblioteca externa
- **Fonte**: RF-01, AC-02
- **Confiança**: 🟡 INFERIDO (Catch2 já é dependência; GENERATE é nativo)
- **Implementação**: `tests/test_*_properties.cpp` usando `GENERATE` do Catch2 v3. Sem dependência nova (sem QuickCheck, sem RapidCheck). 1000 inputs por run é `GENERATE(range(0, 1000))`.
- **Risco**: Catch2 GENERATE tem performance pior que bibliotecas dedicadas. Aceitável: 9 testes × 1000 inputs × runtime < 1s é factível.
- **Alternativa rejeitada**: RapidCheck (adiciona dep, conflitos com versão Clang 18); hand-rolled (mais código para manter).

#### D-T-06: Cross-validação C ↔ Python usa `numpy.testing.assert_allclose` com `rtol=1e-5`
- **Fonte**: RF-03
- **Confiança**: 🟡 INFERIDO (escolha de tolerância)
- **Implementação**: Script `tests/cross_validation.py` orquestra C test + Python reference; compara com `np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)`.
- **Risco**: 1e-5 é folgado para float32 mas apertado o suficiente para catch bugs reais. ACDC tem `max_diff ≈ 1.3e-16` (do princípio P2) mas cross-language summation order pode degradar para `1e-6`. 1e-5 dá margem.
- **Alternativa rejeitada**: bit-exact (1e-15) — falha em cross-language por ordem de soma.

#### D-T-07: ACDC retangular (RF-04) usa FWHT 2D via Kronecker, não por bloco
- **Fonte**: RF-04 (condicional, M3)
- **Confiança**: 🟡 INFERIDO (a confirmar com protótipo)
- **Implementação proposta**: Para W ∈ ℝ^{m×n} com m ≠ n, usar W = H_m · D · H_n com H_m Hadamard (m próximo de power-of-2) e H_n similar. A diagonal D ∈ ℝ^{min(m,n)} captura a "essência diagonal". Para BitNet-2B: gate/up são 2560×6912, H_2560 ⊗ H_6912 (não são quadrados perfeitos, requer padding zero).
- **Risco**: Performance de H_m ⊗ H_n pode regredir vs ACDC quadrado (P3). Decisão final após prototipagem.
- **Alternativa rejeitada**: SVD (não atende P3 — O(mn²)); H-only-horizontal (perde simetria).

#### D-T-08: Bench publish (RF-07) usa formato JSON canônico + renderizador Markdown
- **Fonte**: RF-07
- **Confiança**: 🟡 INFERIDO
- **Implementação**: `utils/bench_publish.py --json > benchmarks/v0.1.0.json`; `utils/bench_publish.py --from-json benchmarks/v0.1.0.json --md > benchmarks/v0.1.0.md`. O JSON é o source of truth; o Markdown é derivado.
- **Risco**: Dois formatos para manter sincronizados. Mitigação: Markdown é gerado a partir do JSON, nunca editado manualmente.

### 3.3. Decisões de baixo impacto (táticas)

| ID | Decisão | Confiança |
|----|---------|-----------|
| D-T-09 | `tests/test_air_gapped_boot.sh` usa `unshare -rn` (network namespace) para isolar | 🟢 CONFIRMADO (padrão Linux) |
| D-T-10 | `docs/hardware-compatibility.md` é uma tabela CPU → modo de operação (L1 OK, L2/L3/L4 com flag, L5 só com d ≥ 256) | 🟢 CONFIRMADO |
| D-T-11 | `ROADMAP.md` separa "Atual", "Reserva técnica" e "Fora de escopo" em 3 seções | 🟡 INFERIDO (a refinar com feedback) |
| D-T-12 | README é reescrito com persona D4 mas mantém `docs/theory/` como referência canônica | 🟢 CONFIRMADO |

---

## 4. Delta Arquitetural

### 4.1. Componentes NOVOS

| Componente | Função | Arquivo (proposto) | Marco |
|------------|--------|--------------------|-------|
| `tests/test_<kernel>_properties.cpp` | Property-based tests com Catch2 GENERATE | `tests/test_acdc_properties.cpp` (1º), depois L4, L5 | M1 |
| `utils/bench_publish.py` | Bench sistemático + JSON/MD output | `utils/bench_publish.py` | M5 |
| `docs/decision-matrix.md` | Quando usar L1/L3/L4/L5 | `docs/decision-matrix.md` | M2 |
| `docs/hardware-compatibility.md` | Tabela CPU → modo | `docs/hardware-compatibility.md` | M5 |
| `docs/invariants.md` | Lista canônica de invariantes P1-P7 com referência ao test | `docs/invariants.md` | M1 |
| `ROADMAP.md` | Roadmap público com Atual/Reserva/Fora | `ROADMAP.md` (raiz) | M1 |
| `tests/test_air_gapped_boot.sh` | Smoke test air-gapped | `tests/test_air_gapped_boot.sh` | M5 |
| `tests/cross_validation.py` | Cross-validação C ↔ Python | `tests/cross_validation.py` | M2 |

### 4.2. Componentes MODIFICADOS

| Componente | Mudança | Marco |
|------------|---------|-------|
| `README.md` | Reescrito com persona D4 (privacidade/soberania) | M5 |
| `src/ggml-bitnet-tropical.cpp` | Documentar que `sparse_attention_float` é opt-in (já é, falta doc) | M2 |
| `examples/` | Adicionar `examples/medical_offline.md`, `examples/legal_offline.md`, `examples/finance_offline.md` (cenários D4) | M5 |
| `tests/CMakeLists.txt` | Adicionar targets para `test_acdc_properties` etc. | M1 |
| `.github/workflows/ci.yml` | Adicionar step `air-gapped boot` | M5 |
| `docs/findings-cpu-universal.md` | Adicionar seção "Pessoa Alvo" (cross-link com `requirements.md#9`) | M2 |

### 4.3. Componentes NÃO TOCADOS (explicitamente)

- `3rdparty/llama.cpp/` — patches vendored permanecem em `patches/llama.cpp/`
- `_reversa_sdd/` — imutável
- `.reversa/context/` — imutável
- `docs/theory/` — teoria canônica; não duplicar
- `src/ggml-bitnet-*.cpp` (kernels) — não modificar comportamento de produção, só adicionar testes e docs

---

## 5. Delta de Dados

**Não há mudança no modelo de dados para v0.1.**

O modelo BitNet (GGUF) é lido pela pipeline existente. Esta feature:
- Não introduz novos campos no GGUF
- Não introduz novos formatos de checkpoint
- Não requer migração de modelos existentes
- Não requer migração de dados de usuário

**Para v0.2 (ACDC retangular, se D2 trigger disparar)**: introduz-se um sidecar `.npz` ao lado do GGUF, contendo a diagonal `d*` por matriz. Formato: `{layer_name: array(d)}` salvo como NumPy savez. Análogo a `utils/extract_acdc_diagonal.py` (existente, commit `fcf1d4d`).

**Para v0.3 (finetune ACDC, se D3 reativar)**: novo formato GGUF extendido com seção `acdc.diagonals`. Não retrocompatível (P6 — estrutura, não compressão, exige treinamento).

Ver `data-delta.md` para detalhes.

---

## 6. Delta de Contratos

**Não há.** Esta feature não toca contratos externos (HTTP, fila, gRPC, GraphQL). 

- `run_inference_server.py` (HTTP OpenAI-compat) não é modificado.
- Não há clientes externos do BitNet além do CLI e do servidor.
- Persona D4 explicitamente **exclui** cloud deployment (NO-07), então novos endpoints HTTP estão fora do escopo.

A única "interface" nova é o flag CLI `--attn sparse` (já existente, documentado em D-T-01).

---

## 7. Plano de Migração (Ordem de Marcos)

```
M1 (curto prazo, 2-3 semanas) ──────────────────────── Hardening matemático
  ├── RF-01: test_acdc_properties.cpp (1000 inputs)
  ├── test_l4_sparse_properties.cpp
  ├── test_hrr_properties.cpp
  ├── docs/invariants.md (P1-P7)
  ├── ROADMAP.md (raiz) com seção Reserva técnica
  └── Investigação D2 (sub-tarefa): testar Llama-2-7B → resultado determina M3

M2 (curto prazo, 1 semana) ──────────────────────────── Decision matrix
  ├── RF-02: docs/decision-matrix.md
  ├── RF-05: documentar L4 sparse opt-in (já é comportamento)
  ├── RF-03: tests/cross_validation.py
  └── atualizar docs/findings-cpu-universal.md

[gate] Se M1 investigação D2 disparou "bloqueador", M3 é movido para M3' (curto prazo)

M3 (médio prazo, 1-2 meses) ─────────────────────────── ACDC retangular (condicional)
  ├── RF-04: src/ggml-bitnet-fwht.cpp#acdc_project_rect
  ├── tests/test_acdc_rect.cpp
  ├── Atualizar AC-08 para "bloqueador" se trigger D2 disparou
  └── RNF-02: bench antes/depois, performance não regride

M3' (apenas se D2 trigger) ──────────────────────────── M3 movido para curto prazo
  └── (mesmo conteúdo de M3, mas com deadline apertado)

M4 (reserva, reavaliação Q4 2029) ───────────────────── Validação empírica (futuro)
  └── RF-06: utils/finetune_acdc.py — NÃO IMPLEMENTAR em v0.1
      Apenas documentar em ROADMAP.md

M5 (médio prazo, paralelo a M1-M3) ──────────────────── Produto
  ├── AC-11: tests/test_air_gapped_boot.sh
  ├── AC-12: examples/medical_offline.md, legal_offline.md, finance_offline.md
  ├── AC-13: docs/hardware-compatibility.md
  ├── RF-07: utils/bench_publish.py
  ├── README.md reescrito (persona D4)
  └── PR upstream aberto em ggerganov/llama.cpp (com sparse opt-in + patches)
```

### 7.1. Dependências entre marcos

```
M1 ──(investiga D2)──> decisão M3 ou M3'
M1 ─> M2 ─> M3
M1 ─> M5 (paralelo)
M2 ─> M5 (paralelo)
M3 ─> M5
M4 ─> (futuro, sem dependência atual)
```

### 7.2. Marcos intermediários (sub-marcos)

- **S1.1** (1ª semana de M1): `test_acdc_properties.cpp` com 4 propriedades (energia, exatidão, ortogonalidade, determinismo)
- **S1.2** (2ª semana de M1): `test_l4_sparse_properties.cpp` + `test_hrr_properties.cpp`; `docs/invariants.md`
- **S1.3** (3ª semana de M1): investigação D2 (Llama-2-7B smoke test); `ROADMAP.md`

---

## 8. Riscos

| # | Risco | Probabilidade | Impacto | Mitigação |
|---|-------|---------------|---------|-----------|
| R-01 | Persona D4 aliena contribuidores que vieram pelo lado "pesquisa pura" | Média | Médio | Manter `docs/theory/` intocado; posicionar D4 como "caso de uso primário", não "exclusivo" |
| R-02 | Property-based tests revelam bug latente em kernel (rollback necessário) | Baixa | Alto | Property tests em M1 antes de qualquer otimização; se falharem, abrir issue de follow-up antes de avançar |
| R-03 | Trigger D2 (Llama-2-7B) dispara e exige mover M3 para curto-prazo sem recursos | Média | Médio | M1 já inclui a investigação; recursos são realocados, não criados |
| R-04 | ACDC retangular (M3) tem performance pior que o quadrático em BitNet-2B | Média | Alto | Prototipar antes de comprometer; RNF-02 garante que performance não regride |
| R-05 | Air-gapped boot test (AC-11) falha em CI por dependência oculta (ex: DNS lookup em libc init) | Baixa | Médio | Investigar com `strace -e network`; documentar dependências se necessário |
| R-06 | Bench publish (RF-07) tem variância alta entre runs, números não comparáveis | Média | Baixo | Fixar seed, t, n; documentar metodologia; publicar histogramas, não só médias |
| R-07 | Reavaliação Q4 2029 (LR-02) é esquecida | Alta | Baixo | Adicionar reminder no CI (cron job); revisões de release checam |
| R-08 | Reposicionamento para "privacidade/soberania" atrai escrutínio regulatório (LGPD, EU AI Act) | Baixa | Médio | Documentar compliance no README; consultar jurídico se necessário |

---

## 9. Critério de Pronto (Definition of Done)

A feature `001-trilha-rigor-produto` está **pronta** quando:

1. ✅ **M1 verde**: ctest passa 9+/9+, ≥ 60 subtests (4 property tests × 4-5 props + 5 existentes), `docs/invariants.md` existe, `ROADMAP.md` existe, investigação D2 concluída (resultado documentado).
2. ✅ **M2 verde**: `docs/decision-matrix.md` existe, `tests/cross_validation.py` passa, sparse opt-in documentado.
3. ✅ **M3 verde** OU **dispensado**: se D2 trigger disparou, `tests/test_acdc_rect.cpp` passa; senão, M3 fica para próximo ciclo (registrado em lacunas).
4. ✅ **M5 verde (parcial)**: AC-11 (air-gapped), AC-12 (exemplos), AC-13 (hardware-compat) verdes; RF-07 (bench publish) gera JSON+MD; README.md reescrito.
5. ✅ **AC-01 a AC-07 verdes** (limiar mínimo de "produto viável").
6. ✅ **Documentação cruzada**: `requirements.md`, `roadmap.md`, `investigation.md`, `data-delta.md`, `onboarding.md`, `ROADMAP.md` (raiz) linkados entre si.
7. ✅ **Lacunas residuais documentadas**: LR-01, LR-02, LR-03 referenciadas em `requirements.md#11`.

**Não-pronto** se:
- Algum AC-01 a AC-07 vermelho
- Alguma invariante P1-P7 violada em PR
- Telemetria inadvertidamente introduzida (NO-06 violado)
- Cloud endpoint inadvertidamente introduzido (NO-07 violado)

---

## 10. Premissas Adotadas (a partir de Dúvidas Resolvidas)

As 4 dúvidas resolvidas em `/reversa-clarify` (D1-D4) viraram **decisões de produto** documentadas em `requirements.md#10`. Não são mais premissas; são compromissos. Reproduzidas aqui para rastreabilidade:

- **PREM-D1**: L4 sparse é opt-in (D1). Premissa implícita: "compatibilidade tem prioridade sobre performance".
- **PREM-D2**: AC-08 (ACDC retangular) é bloqueador condicional com trigger empírico (D2). Premissa implícita: "decisão empírica > decisão arquitetural antecipada".
- **PREM-D3**: RF-06 é reserva técnica com reavaliação Q4 2029 (D3). Premissa implícita: "explícito > implícito; reavaliação periódica > ambição imediata".
- **PREM-D4**: Persona D4 (privacidade/soberania) governa o produto (D4). Premissa implícita: "hardware-alvo (laptop corporativo) > hardware-alvo (cluster GPU)".

---

## 11. Próximos Passos (Operacional)

Após este `roadmap.md`:

1. **`/reversa-to-do`**: decompor M1 em ações atômicas (`actions.md`) com IDs, dependências, e marcador de paralelismo.
2. **`/reversa-audit`** (opcional): cross-check entre `requirements.md`, `roadmap.md` e `actions.md` por inconsistências.
3. **`/reversa-quality`** (opcional): auditoria de clareza textual do `requirements.md`.
4. **`/reversa-coding`**: executar `actions.md` em código.

A próxima etapa natural é `/reversa-to-do`.

---

*roadmap.md v1 — gerado por reversa-plan em 2026-06-06*
