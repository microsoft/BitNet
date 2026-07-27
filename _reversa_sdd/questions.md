# Perguntas para Validação — BitNet CPU-Universal

> Gerado pelo Reversa Reviewer em 2026-06-06 | doc_level: completo
> Modo: chat (state.json não tem `answer_mode` → padrão)
> Processe cada resposta — após cada uma eu atualizo a spec e reclassifico.

---

## Pergunta 1

**Contexto:** `architecture.md §5.1` + `confidence-report.md LAC-01` + `gap-analysis.md P6`. A tese central do fork é que L3 ACDC e L5 HRR são **arquiteturas de treinamento** (P6) — não compressões. Mas o fork **não treina modelos** (escopo CPU-only). O `acdc_project` valida a fórmula fechada, não a qualidade end-to-end.

**Spec afetada:** [`_reversa_sdd/architecture.md`](architecture.md), [`_reversa_sdd/gap-analysis.md`](gap-analysis.md) (P6), `_reversa_forward/001-trilha-rigor-produto/requirements.md` (D2 — bloqueador condicional)

**Pergunta:** A lacuna P6 (modelo treinado com ACDC/HRR) é aceita como **fora do escopo do fork CPU-only**, ou o fork deveria incluir um **scaffolding GPU mínimo** (`utils/finetune_acdc.py`, ~500 linhas PyTorch) para futura validação? (RF-06 do `001-trilha-rigor-produto/requirements.md` já trata isso como **reserva técnica** com reavaliação Q4 2029.)

**Impacto:**
- Se **aceitar fora do escopo**: a 🔴 LAC-01 vira 🟡 com nota "Caminho C documentado mas não implementado". Dívida D-01 reclassificada para D-01`.
- Se **incluir scaffolding**: cria nova feature no forward (`_reversa_forward/002-acdc-finetune-scaffold/` ou similar) e gera ações atômicas. Aumenta escopo em ~500 linhas PyTorch.

✅ **Respondida em 2026-06-06**

**Resposta:** <!-- A lacuna P6 deve ser reclassificada de 🔴 para 🟡 com a seguinte justificativa documentada:

"Caminho C (validação end-to-end com modelo treinado) documentado na arquitetura, mas implementação fora do escopo da fase CPU-only. Reserva técnica RF-06 agendada para Q4 2029. Dívida D-01 reclassificada para D-01` (dívida consciente com plano de pagamento definido)." -->

---

## Pergunta 2

**Contexto:** `_reversa_sdd/domain.md` foi gerado em 2026-05-03 sobre o **upstream** (com `gpu/`). O fork removeu `gpu/`, mas o arquivo não foi atualizado. 5 RNs referenciam `gpu/` que não existe:
- **RN-005** (Dual-model GPU prefill/decode) → gpu/generate.py:115-150
- **RN-006** (Prompts padded para prompt_length) → gpu/generate.py:238
- **RN-011** (torch.load sem weights_only, CWE-502) → gpu/generate.py + gpu/convert_checkpoint.py
- **RN-014** (NO_CUDA_GRAPHS env) → gpu/generate.py:343
- **RN-015** (capture_error_mode="thread_local" workaround) → gpu/generate.py:136-139

A recomendação de marcá-las como `[LEGACY — UPSTREAM ONLY]` está em `architecture.md §10` mas **não foi aplicada** ao `domain.md`.

**Spec afetada:** [`_reversa_sdd/domain.md`](domain.md), [`_reversa_sdd/architecture.md`](architecture.md) §10

**Pergunta:** Como tratar as 5 RNs obsoletas em `_reversa_sdd/domain.md`?

| Opção | Descrição | Prós | Contras |
|-------|-----------|------|---------|
| **A** | Marcar cada RN obsoleta com `[LEGACY — UPSTREAM ONLY — não se aplica ao fork]` no topo | Preserva histórico; transparente | Polui o documento com notas |
| **B** | Remover as 5 RNs | Mantém o doc limpo e atual | Perde referência histórica ao upstream |
| **C** | Mover as 5 RNs para um novo arquivo `_reversa_sdd/legacy-gpu.md` | Separa concerns; preserva referência | Cria fragmentação |
| **D** | Deixar como está e adicionar **apenas** um cabeçalho em `domain.md` avisando da defasagem | Mínimo trabalho | Notas dispersas; usuário pode ignorar |

✅ **Respondida em 2026-06-06**

**Resposta:** <!-- Opção A com refinamento: Marcar cada RN obsoleta com [LEGACY — UPSTREAM ONLY — não se aplica ao fork] é a abordagem mais alinhada com as melhores práticas de gestão de dívida técnica e documentação arquitetural. -->

---

## Pergunta 3

**Contexto:** `_reversa_sdd/code-analysis.md` (599 linhas) tem **15 referências** a `gpu/` que apontam para módulos **inexistentes** no fork (`gpu/model.py`, `gpu/generate.py`, etc.). O documento descreve a arquitetura **upstream**, não o fork atual.

**Spec afetada:** [`_reversa_sdd/code-analysis.md`](code-analysis.md)

**Pergunta:** Como tratar o `code-analysis.md` (15 refs a `gpu/`)?

| Opção | Descrição | Prós | Contras |
|-------|-----------|------|---------|
| **A** | Adicionar cabeçalho: "ATENÇÃO: este doc foi gerado sobre o upstream em 2026-05-03. O fork removeu `gpu/`. Veja `architecture.md §6` para o estado atual." | Mínimo esforço | Polui; leitor pode ignorar |
| **B** | Reescrever o documento filtrando as 15 refs a `gpu/` e adicionando `L2-L5` (L2 WHT, L3 ACDC, L4 Tropical, L5 HRR) que o doc atual não cobre | Doc fica 100% sobre o fork | Re-análise significativa |
| **C** | Marcar `code-analysis.md` como `[DEPRECATED — see architecture.md]` e redirecionar via `_reversa_sdd/README.md` (criar) | Clara direção | Perde o detalhe do code-analysis |

✅ **Respondida em 2026-06-06**

**Resposta:** <!-- Opção Híbrida (A + C refinada): Manter o conteúdo técnico válido (análise estática de llm/, data/, eval/) mas desativar as seções obsoletas com marcação clara e redirecionamento. -->

---

## Pergunta 4

**Contexto:** `architecture.md §2.1` (C4 Nível 1) lista 3 personas:
- Persona A — Desenvolvedor de Privacidade e Soberania de Dados (D4 forward) — 🟡 INFERIDO
- Persona B — Operador CLI — 🟢 CONFIRMADO
- Persona C — Operador de Servidor — 🟢 CONFIRMADO

A Persona A vem de uma **decisão D4** registrada no `001-trilha-rigor-produto/requirements.md v2 §3.4`. Está marcada como INFERIDO porque está em **outro output folder** (`_reversa_forward/`, não `_reversa_sdd/`). A confirmação é forte (decisão registrada, cross-validada com gap-analysis.md e continuidade-proposals.md), mas a rigor é uma inferência cross-folder.

**Spec afetada:** [`_reversa_sdd/c4-context.md`](c4-context.md) §2.1

**Pergunta:** A Persona A (Desenvolvedor de Privacidade) deve ser reclassificada para 🟢 CONFIRMADO (com nota de proveniência cross-folder), ou mantida como 🟡 INFERIDO?

**Impacto:**
- Se 🟢: confiança de `c4-context.md` sobe de 84.6% para 92.3%.
- Se 🟡: mantemos a separação rigorosa entre discovery (`_reversa_sdd/`) e forward (`_reversa_forward/`).

✅ **Respondida em 2026-06-06**

**Resposta:** <!-- 🟢 Reclassificar para CONFIRMADO (com nota de proveniência cross-folder).

A manutenção do status 🟡 INFERIDO seria um "rigor burocrático" que não agrega valor real à precisão da arquitetura, dado que a decisão D4 já está registrada, validada e cross-referenciada em documentos oficiais do projeto (requirements.md, gap-analysis.md, continuity-proposals.md). -->

---

## Resumo das Perguntas

| # | Tipo | Severidade | Spec | Status | Resultado |
|---|------|-----------|------|--------|-----------|
| 1 | Decisão estratégica (escopo) | 🔴 ALTA | architecture.md, gap-analysis.md | ✅ Respondida | LAC-01 reclassificada 🔴→🟡; D-01 → D-01` (plano de pagamento Q4 2029) |
| 2 | Edição direta (5 RNs) | 🟡 MÉDIA | domain.md | ✅ Respondida | Opção A aplicada: 5 RNs marcadas com `[LEGACY — UPSTREAM ONLY]` |
| 3 | Edição direta (15 refs) | 🟡 MÉDIA | code-analysis.md | ✅ Respondida | Opção Híbrida A+C: header de aviso + footer com redirect para `architecture.md` |
| 4 | Reclassificação 🟡→🟢 | 🟢 BAIXA | c4-context.md | ✅ Respondida | Persona A reclassificada 🟡→🟢 com nota de proveniência cross-folder |

**Processamento completo. 4/4 perguntas respondidas, 4 specs editadas in-place, 1 reclassificação 🔴→🟡 (P6/D-01), 1 reclassificação 🟡→🟢 (Persona A), 5 marcadores LEGACY aplicados, 1 cabeçalho de aviso + footer redirect em code-analysis.md.**
