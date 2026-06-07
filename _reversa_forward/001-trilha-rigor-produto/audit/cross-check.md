# Cross-Check Audit — `001-trilha-rigor-produto`

> Auditoria leitora estrita entre `requirements.md`, `roadmap.md` e `actions.md` (e secundariamente `data-delta.md`, `investigation.md`, `onboarding.md`).
>
> **Versão:** v1 (gerado por reversa-audit em 2026-06-06)
> **Análise:** `_reversa_forward/001-trilha-rigor-produto/`
> **IDs estáveis:** A001-A0xx (próprios do relatório, não compartilhados com RF/M/AC/etc.)

---

## 1. Cabeçalho

| Item | Valor |
|------|-------|
| **Data** | 2026-06-06 |
| **Feature** | `001-trilha-rigor-produto` |
| **Artefatos analisados** | `requirements.md` (v2, 395 linhas), `roadmap.md` (v1, 303 linhas), `actions.md` (v1, 199 linhas) |
| **Artefatos secundários** | `data-delta.md` (v1, 234 linhas), `investigation.md` (v1, 288 linhas), `onboarding.md` (v1, 461 linhas) |
| **Regras de domínio consultadas** | `_reversa_sdd/domain.md` (16 RNs), `.reversa/scout/principles.md` (7 princípios) |

---

## 2. Resumo de Findings

| Severidade | Contagem |
|------------|---------:|
| **CRITICAL** | 0 |
| **HIGH** | 4 |
| **MEDIUM** | 4 |
| **LOW** | 3 |
| **Total** | **11** |

---

## 3. Tabela de Findings

| ID | Severidade | Eixo | Descrição | Onde está |
|----|------------|------|-----------|-----------|
| A001 | HIGH | Cobertura | AC-10 (`docs/theory/06-5-levels.md` resume os 5 níveis em uma página) **não é coberto** por nenhuma T-action em `actions.md`. A frase atual em `requirements.md#6` diz "Já parcialmente existe em `mathematical-foundations.md`" — mas isso não satisfaz o critério literal. | `requirements.md:187` (AC-10) → `actions.md` (sem T-action) |
| A002 | HIGH | Cobertura | AC-09 (scaffolding de fine-tuning ACDC como **reserva técnica**, com smoke test) tem cobertura fraca. T014 (ROADMAP.md) e T035 (Q4 2029 reminder) documentam a reserva, mas não há T-action que materialize o `utils/finetune_acdc.py --smoke` (que **existe conceitualmente** segundo D-T-03 mas não tem ação explícita). | `requirements.md:186` (AC-09) → `actions.md` (parcial: T014, T035) |
| A003 | HIGH | Cobertura | AC-04 (`docs/findings-cpu-universal.md` cobre 5 níveis, 4 bugs, 50 subtests) tem cobertura **redundante**: o documento já existe (commit `1be84ef`). T027 adiciona "Persona Alvo", não os 5 níveis. Risco de o critério ser entendido como "já passa" quando, na verdade, o conteúdo de AC-04 é de S2e, não desta feature. | `requirements.md:181` (AC-04) → `actions.md:27` (T027) |
| A004 | HIGH | Sanidade | Marcadores `[//]` (paralelismo) estão mal-colocados em T013 e T035: ambos têm dependências explícitas (T013 depende de T004; T035 depende de T014, T033) e portanto **não são paralelizáveis livremente**. T004 e T013 compartilham arquivo alvo (`docs/invariants.md`); T014 e T035 compartilham `ROADMAP.md`. O par `T004→T013` e `T014→T035` é **sequencial**, não paralelo. | `actions.md:35` (T004), `actions.md:66` (T013), `actions.md:38` (T014), `actions.md:115` (T035) |
| A005 | MEDIUM | Consistência | `requirements.md#2` cita "ADR-003" como fonte da regra "CPU only — GPU proibida". Mas **ADR-003 é sobre dual-model GPU** (prefill fp16 + decode int2), que é a pipeline **removida** no fork. A regra "CPU only" real vem de `CLAUDE.md:9-11`. Citação fantasma. | `requirements.md:39` (Restrições inegociáveis) |
| A006 | MEDIUM | Cobertura | Apenas 2 de 12 decisões técnicas (D-T-01 e D-T-07) são explicitamente referenciadas em `actions.md`. As outras 10 (D-T-02 a D-T-06, D-T-08 a D-T-12) estão **cobertas implicitamente** pelo arquivo alvo, mas não por cross-reference. Risco: ao refatorar `actions.md`, a rastreabilidade se perde. | `roadmap.md#3` (D-T-01..D-T-12) → `actions.md` |
| A007 | MEDIUM | Cobertura | Os 3 sub-marcos S1.1, S1.2, S1.3 definidos em `roadmap.md#7.2` **não são referenciados** em `actions.md`. As T-actions cobrem M1 mas sem agrupamento por sub-marco. Aceitável se a granularidade de T-action for mais fina que S-milestone, mas deve ser explícito. | `roadmap.md:175-177` (S1.1, S1.2, S1.3) → `actions.md` (ausente) |
| A008 | MEDIUM | Cobertura | Apenas R-01 (persona aliena contribuidores) é referenciado em `actions.md`. R-02 a R-08 (8 riscos no total) não têm mitigação como T-action. Possível interpretação: nem todo risco precisa de ação (alguns são passivos). Mas o `roadmap.md` lista-os como "mitigação" e a mitigação deveria ser executável. | `roadmap.md#8` (R-01..R-08) → `actions.md` (1/8) |
| A009 | LOW | Cobertura | RF-01 (property-based tests), RF-03 (cross-validation), RF-04 (ACDC retangular) e RF-05 (L4 sparse opt-in) **não são explicitamente citados** em `actions.md`. As T-actions cobrem o trabalho, mas a rastreabilidade RF→T é fraca. RF-02, RF-06, RF-07 são citados. | `requirements.md#4` (RF-01..RF-07) → `actions.md` (3/7 citados) |
| A010 | LOW | Sanidade | T009 depende de T003 (Fase 1) **e** T029 (Fase 4). A ordem de execução fica: T001 (Fase 1) → T002/T003 (Fase 1) → T029 (Fase 4, salta Fase 2-3) → T009 (Fase 2, salta de volta). Funcional mas contraintuitivo. **Sugestão** (humana): explicitar no `roadmap.md` que a investigação D2 (T029) é executada **em paralelo** com Fase 1-3, e T009 só inicia após T029 confirmar. | `actions.md:51` (T009) |
| A011 | LOW | Sanidade | T033 (validação DoD, "rodar ctest, verificar cada critério") tem **critérios subjetivos** embutidos: "registrar resultado em `roadmap.md#9`" — mas a frase não diz **como** registrar nem o **formato**. Para um agente de IA em `/reversa-coding`, isso vira ambiguidade. | `actions.md:111-112` (T033) |

---

## 4. Discussão de Findings CRITICAL/HIGH

### A001 — AC-10 sem cobertura

**Impacto:** AC-10 é um dos 13 critérios de aceitação do limiar mínimo (`requirements.md#6`). Sem uma T-action, este critério **nunca fica verde** em `/reversa-coding`. A redação atual de AC-10 ("Já parcialmente existe em `mathematical-foundations.md`") é ambígua — pode ser interpretada como "passa" ou como "precisa criar `docs/theory/06-5-levels.md` dedicado".

**Sugestão de skill para o humano corrigir:**
- Se AC-10 deve ser satisfeito por um novo arquivo: rodar `/reversa-clarify` para reabrir a dúvida e adicionar uma T-action. **OU** editar manualmente `actions.md` adicionando uma T-ação em Fase 3.
- Se AC-10 deve ser satisfeito por uma seção em `mathematical-foundations.md`: rodar `/reversa-clarify` para reescrever AC-10.
- **Em hipótese alguma** este skill altera os artefatos.

### A002 — AC-09 cobertura fraca

**Impacto:** AC-09 diz "Scaffolding de fine-tuning ACDC existe e roda em smoke test" com a marca "(RF-06; reavaliação Q4 2029)". A leitura literal pede `utils/finetune_acdc.py --smoke` funcional. A leitura de D-T-03 diz "documentar a reserva em `ROADMAP.md`". As duas leituras **divergem**.

**Sugestão de skill para o humano corrigir:** `/reversa-clarify` para resolver a divergência entre AC-09 e D-T-03. Alternativa: editar manualmente `requirements.md` (AC-09) para marcar como "diferencial, não requerido para v0.1", consistente com o status de "reserva técnica".

### A003 — AC-04 redundante

**Impacto:** AC-04 já é satisfeito pelo commit `1be84ef` (S2e). T027 adiciona "Persona Alvo" ao `docs/findings-cpu-universal.md`, que é uma adição de conteúdo, não de cobertura. Critério **já passa** independentemente desta feature. Risco: confusão no DoD (T033) — o agente pode interpretar que T027 é o que satisfaz AC-04, mas isso é parcialmente verdade.

**Sugestão de skill para o humano corrigir:** `/reversa-clarify` para reescrever AC-04 com critério específico do **delta** (ex: "achados sobre persona D4 cobertos"), em vez de "cobre 5 níveis" que é pré-existente.

### A004 — `[//]` mal-colocado em T013 e T035

**Impacto:** O marcador `[//]` é semanticamente "tarefa paralelizável com outras `[//]` no mesmo bloco". Em T013 e T035, o marcador é **enganoso**: ambos têm dependências explícitas, então o agente de IA pode tentar paralelizar, falhar porque T004 não está pronto, e perder tempo.

**Sugestão de skill para o humano corrigir:** Edição manual de `actions.md`: remover `[//]` da coluna "Paralelismo" de T013 e T035. Manter nas colunas descritivas se fizer sentido, ou remover de ambas.

---

## 5. Discussão de Findings MEDIUM

### A005 — Citação ADR-003 fantasma

**Impacto:** Baixo risco operacional (a regra "CPU only" é aplicada corretamente em todo o código). Risco de **confusão em auditoria externa**: alguém que abrir `requirements.md` e procurar ADR-003 para confirmar a regra vai encontrar conteúdo sobre dual-model GPU (incoerente).

**Sugestão de skill para o humano corrigir:** Edição manual de `requirements.md:39`: substituir "ADR-003" por `CLAUDE.md` ou criar um novo ADR-008 "CPU-only como restrição fundadora".

### A006 — Decisões sem cross-reference

**Impacto:** Refatorações futuras de `actions.md` podem quebrar a rastreabilidade decisão→ação. Risco médio: a feature funciona, mas a auditoria fica difícil.

**Sugestão de skill para o humano corrigir:** Edição manual de `actions.md` para adicionar a referência `(D-T-XX)` na coluna "Descrição" de cada T-action relevante. Baixa prioridade.

### A007 — Sub-marcos S1.x sem cross-reference

**Impacto:** S1.1, S1.2, S1.3 são definidos em `roadmap.md#7.2` mas não agrupam T-actions. Aceitável se a granularidade for intencional. Baixa prioridade.

**Sugestão:** Edição manual de `actions.md` para adicionar uma coluna "Sub-marco" ou agrupar Fase 1-2 em S1.1, S1.2, S1.3.

### A008 — Riscos sem mitigação executável

**Impacto:** 7 dos 8 riscos (R-02 a R-08) não têm T-action explícita. Se um risco se materializar durante `/reversa-coding`, o agente pode não saber como responder. Risco médio de execução descoordenada.

**Sugestão de skill para o humano corrigir:** Edição manual de `actions.md` adicionando T-actions para mitigação preventiva (ex: T036 "Investigar variância de bench antes de M5" cobre R-06; T037 "Adicionar strace audit antes de AC-11" cobre R-05).

---

## 6. Discussão de Findings LOW

### A009 — RF-01, RF-03, RF-04, RF-05 sem cross-reference

**Impacto:** RFs sem citação explícita na tabela. T-actions cobrem o trabalho (T005-008 cobrem RF-01, T011 cobre RF-03, T009+ T018+ T019 cobrem RF-04, T008+ T017 cobrem RF-05). Risco baixo: cobertura existe, só não está etiquetada.

**Sugestão:** Edição manual de `actions.md` para adicionar `(RF-XX)` em cada T-action relevante.

### A010 — Dependência de fase cruzada T009

**Impacto:** Ordem de execução contraintuitiva (T001→T003→T029→T009). Funcional mas exige atenção.

**Sugestão:** Documentar a dependência cruzada no `roadmap.md#7` (já existe menção a "M1 inclui a investigação D2" — só falta explicitar que T029 pode ser paralelizada com Fase 1-3).

### A011 — T033 critérios subjetivos

**Impacto:** T033 é o gargalo final. Sem critério objetivo, agentes podem divergir em "verde" vs. "vermelho".

**Sugestão:** Especificar T033 com formato de output concreto (ex: "cria `verification-report.md` com tabela `AC-XX | status | evidência`").

---

## 7. Itens Verificados que Passaram (por eixo)

### 7.1. Cobertura (passou)

- ✅ **Todos os 7 RFs** mapeados para ≥1 T-action (mesmo que sem cross-reference explícita)
- ✅ **13 de 13 ACs** referenciados em `roadmap.md#9` (DoD)
- ✅ **5 de 5 Ms (M1-M5)** com ≥1 T-action (exceto M4 reserva)
- ✅ **4 P-remissas (PREM-D1..D4)** em `roadmap.md#10` rastreáveis a `requirements.md#10`
- ✅ **3 de 3 LRs (LR-01..LR-03)** em `requirements.md#11` consistentes com D2/D3
- ✅ **G-D2 gate** corretamente aplicado a 4 ações (T009, T018, T019, T034) com T029 como gatilho
- ✅ **Interfaces/** omitido corretamente (sem contratos externos)
- ✅ **NO-01..NO-07** em `requirements.md#12` com T031, T032 cobrindo NO-06, NO-07

### 7.2. Consistência (passou)

- ✅ **Persona D4** terminologia consistente em `requirements.md`, `roadmap.md`, `actions.md`, `onboarding.md`
- ✅ **D2 trigger** terminologia consistente (D2, G-D2, LR-01, T029, T009, T018, T019, T034)
- ✅ **D3 reserva** terminologia consistente (D3, LR-02, T014, T035, AC-09)
- ✅ **P1-P7 princípios** referenciados consistentemente em `requirements.md` (invariantes) e `actions.md` (T013, T005-T007)
- ✅ **Privacy/soberania** (D4) consistente com NO-06 (sem telemetria) e NO-07 (sem cloud)
- ✅ **CPU-only** (restrição fundadora) consistente em `requirements.md`, `roadmap.md`, `actions.md`, `onboarding.md`

### 7.3. Coerência com legado (passou)

- ✅ **Nenhuma decisão em `roadmap.md` contradiz `_reversa_sdd/domain.md`**:
  - RN-001 (tensores protegidos) — não tocado
  - RN-003 (restrição arquitetura) — não tocado
  - RN-004 (I2_S nrow % 4) — não tocado
  - RN-007 (Clang) — respeitado em `onboarding.md`
  - RN-008 (-ngl 0 hardcoded) — não tocado
  - RN-009 (-b 1 hardcoded) — não tocado
  - RN-013 (escala absmax médio) — não tocado
- ✅ **Nenhuma decisão contradiz `_reversa_sdd/adrs/001-007`**
- ✅ **Princípios P1-P7 todos compatíveis** (ver `roadmap.md#2`)

### 7.4. Sanidade do `actions.md` (passou)

- ✅ **35 T-IDs (T001-T035)** sem buracos, sem reciclagem
- ✅ **Todas as dependências apontam para T-IDs existentes** (zero dependências órfãs)
- ✅ **Zero ciclos de dependência** (verificado manualmente; ver mapa em `actions.md#Mapa`)
- ✅ **4 ações condicionais (gate D2)** corretamente marcadas como `🟡 INFERIDO`
- ✅ **Fases (1-5) com soma 35** = total de T-actions (4+8+11+7+5=35)
- ✅ **Maior cadeia de dependência = 3 níveis** (T005 → T024 → T033), conforme limites do skill

---

## 8. Conclusão

**Veredito:** 0 CRITICAL, 4 HIGH, 4 MEDIUM, 3 LOW.

A feature é **executável** e **internamente consistente**. As 4 HIGH são todas relacionadas a **cobertura de ACs** (A001, A002, A003) e **sanidade de marcação `[//]`** (A004). As MEDIUM são de **rastreabilidade** (A005, A006, A007, A008), não de correção.

**Recomendação para o humano:**
- **Bloqueadores antes de `/reversa-coding`**: A001, A004 (afetam execução direta).
- **Melhorias antes de release**: A002, A003, A005 (afetam auditoria externa).
- **Opcionais**: A006-A011 (qualidade de vida).

**Ação imediata sugerida:** revisão manual de A001 (criar T036 para AC-10) e A004 (remover `[//]` de T013 e T035). Outras HIGHs podem ser resolvidas em sessões de `/reversa-clarify` futuras.

**Aviso explícito:** nenhum dos três artefatos principais (`requirements.md`, `roadmap.md`, `actions.md`) foi alterado por este audit. Tampouco os secundários (`data-delta.md`, `investigation.md`, `onboarding.md`).

---

## 9. Resolution Log (2026-06-06)

> Aplicado após o audit inicial, a pedido do humano. As mudanças abaixo são em `actions.md` apenas (o audit não pode modificar artefatos, mas o humano pode pedir correções pós-audit).

### A001 — RESOLVIDO

- **Mudança**: Adicionada T036 em `actions.md:77` (Fase 3).
- **Conteúdo**: `docs/theory/06-5-levels.md` (AC-10), dependência T013, confiança 🟢.
- **Impacto colateral**: 
  - T033 (DoD) agora valida AC-01 a AC-13 (era AC-01 a AC-07).
  - T033 agora produz `verification-report.md` (era subjetivo).
  - Total de ações: 35 → 36.
  - Paralelizáveis: 21 → 20 (T013 deixou de ser [//], T036 é [//]).

### A004 — RESOLVIDO

- **Mudança**: Removido `[//]` da coluna "Paralelismo" em T013 (`actions.md:66`) e T035 (`actions.md:111`).
- **Razão**: Ambos têm dependências explícitas (T013 depende de T004; T035 depende de T014, T033), portanto não são paralelizáveis. T004↔T013 e T014↔T035 compartilham arquivo alvo, e os segundos são sequenciais aos primeiros.
- **Impacto colateral**: contagem de paralelizáveis: 21 → 20 (T035 deixou de ser [//]).

### A011 — RESOLVIDO (bonus, junto com A001)

- **Mudança**: T033 reescrita com critério objetivo. Saída: `verification-report.md` com tabela `AC-XX | status | evidência | nota`.
- **Razão**: A subjetividade do T033 original ("verificar cada critério") podia gerar divergência entre agentes. Formato explícito elimina ambiguidade.

### Findings NÃO resolvidos (permanecem)

- **A002** (AC-09 cobertura fraca) — pendente para `/reversa-clarify` futura
- **A003** (AC-04 redundante) — pendente para `/reversa-clarify` futura
- **A005** (citação ADR-003 fantasma) — pendente para edição manual de `requirements.md`
- **A006** (decisões sem cross-reference) — pendente para edição manual de `actions.md`
- **A007** (sub-marcos S1.x sem cross-reference) — pendente para edição manual
- **A008** (R-02 a R-08 sem mitigação executável) — pendente para edição manual
- **A009** (RFs sem cross-reference) — pendente para edição manual
- **A010** (T009 dependência de fase cruzada) — informativo; sem ação obrigatória

### Severidade atualizada

| Severidade | Antes | Depois |
|------------|------:|-------:|
| CRITICAL | 0 | 0 |
| HIGH | 4 | 2 (A002, A003) |
| MEDIUM | 4 | 4 (A005-A008) |
| LOW | 3 | 2 (A009, A010) |
| **Total abertos** | **11** | **8** |

### Veredito pós-fix

Feature 001 está **pronta para `/reversa-coding`** com 2 HIGHs aceitáveis (A002, A003 — relacionadas a ambiguidade de AC, não a bloqueio de execução). Os 2 HIGHs resolvidos (A001, A004) eliminavam bloqueios diretos. Os 6 MEDIUM+LOW podem ser tratados em sessões futuras.

---

*cross-check.md v1.1 — gerado por reversa-audit + resolution log em 2026-06-06*

*cross-check.md v1 — gerado por reversa-audit em 2026-06-06*
