# Lacunas — BitNet CPU-Universal

> Gerado pelo Reversa Reviewer em 2026-06-06 | doc_level: completo
> Lista de gaps que **permaneceram sem resposta** após a revisão.
> Categorizado por severidade (doc_level: completo → categorização recomendada mas não obrigatória).

---

## 🔴 Crítico (bloqueiam reuso do discovery para o fork)

### ~~GAP-01: P6 (ACDC/HRR como arquitetura de treinamento) não validado empiricamente~~ ✅ RESOLVIDO 2026-06-06
- **Spec**: `architecture.md §1.1, §5.1, §6`, `gap-analysis.md P6`
- **Status original**: 🔴 LACUNA conhecida — fora do escopo do fork CPU-only
- **Resolução aplicada** (decisão D-Reviewer-1, opção "aceitar fora do escopo"):
  - Dívida D-01 em `architecture.md §5.1` reclassificada de 🔴 CRÍTICA para 🟡 IMPORTANTE com nota "Caminho C documentado + escopo CPU-only + RF-06 Q4 2029 reserva técnica"
  - Dívida renomeada para D-01` (dívida consciente com plano de pagamento definido)
  - `gap-analysis.md P6` atualizado com nota de reclassificação 2026-06-06
  - LAC-01 no `confidence-report.md` marcada como RESOLVIDA
- **Status final**: 🟡 INFERIDO (reclassificado, não resolvido tecnicamente; P6 continua `✗ NÃO VALIDADO EM TREINAMENTO` como observação histórica)
- **Pergunta**: `questions.md#pergunta-1` ✅ Respondida

### ~~GAP-02: 5 RNs obsoletas em `domain.md` referenciam `gpu/` inexistente~~ ✅ RESOLVIDO 2026-06-06
- **Spec**: `_reversa_sdd/domain.md` (RN-005, RN-006, RN-011, RN-014, RN-015)
- **Resolução aplicada** (decisão D-Reviewer-2, **Opção A**):
  - 5 RNs receberam o marcador `[LEGACY — UPSTREAM ONLY — não se aplica ao fork]` logo após o título
  - Cada uma com nota de fork explicando o que era no upstream, por que não se aplica, e a referência para o estado atual
  - As RNs permanecem 🟢 CONFIRMADO **para o contexto upstream** (preservação histórica)
  - O cabeçalho impede interpretação errada por leitores do fork
- **Status final**: ✅ resolvido (preservação histórica + transparência)
- **Pergunta**: `questions.md#pergunta-2` ✅ Respondida

### ~~GAP-03: `code-analysis.md` (599 linhas) tem 15 referências a `gpu/` inexistente~~ ✅ RESOLVIDO 2026-06-06
- **Spec**: `_reversa_sdd/code-analysis.md`
- **Resolução aplicada** (decisão D-Reviewer-3, **Opção Híbrida A+C**):
  - **Parte A (cabeçalho)**: Adicionado bloco `> ## ⚠️ ATENÇÃO — Documento parcial (2026-06-06)` logo após o header original, listando os 8 módulos `gpu/*` inexistentes no fork, o que permanece válido, e os novos kernels L2-L5 adicionados após 2026-05-03
  - **Parte C (footer)**: Apontador para `architecture.md`, `c4-containers.md`, `c4-components.md`, `erd-complete.md` e `gap-analysis.md` para o estado atual
  - Conteúdo técnico válido (Módulos 1, 2, 3, 12, 13, 14, 15) preservado intacto
- **Status final**: ✅ resolvido (híbrido A+C: cabeçalho + footer)
- **Pergunta**: `questions.md#pergunta-3` ✅ Respondida

---

## 🟡 Moderado (melhoria da qualidade, não bloqueia)

### ~~GAP-04: Persona A (Desenvolvedor de Privacidade) classificada como 🟡~~ ✅ RESOLVIDO 2026-06-06
- **Spec**: `c4-context.md §2.1`
- **Resolução aplicada** (decisão D-Reviewer-4): 🟡 → 🟢 CONFIRMADO
  - Adicionada nota de proveniência cross-folder: "decisão D4 do `001-trilha-rigor-produto/requirements.md v2 §3.4` (2026-06-06), cross-validada com `gap-analysis.md` e `continuity-proposals.md`"
  - Justificativa do usuário: "seria um rigor burocrático que não agrega valor real à precisão da arquitetura, dado que a decisão D4 já está registrada, validada e cross-referenciada em documentos oficiais do projeto"
- **Status final**: 🟢 CONFIRMADO
- **Pergunta**: `questions.md#pergunta-4` ✅ Respondida

### GAP-05: L5 HRR com regressão -46% end-to-end em d=128
- **Spec**: `architecture.md §1.1, §6`, `gap-analysis.md P3` (linha "Speedup L5 (sessão antiga)")
- **Status**: 🟡 — speedup analítico é positivo, mas a medição é negativa para d=128
- **Ação pendente**: Documentar limitação (L5 só é útil para d ≥ 256); sem decisão de design
- **Workaround atual**: gap-analysis.md §L5 já documenta
- **Custo de fechar**: 1 minuto (anotar em `architecture.md §1.1`)

### GAP-06: L4 sparse_attention_float consolidada em l4_tropical (não é container próprio)
- **Spec**: `c4-containers.md §1` e `c4-components.md §3.4`
- **Status**: 🟡 — decisão de design (consolidação em tropical.cpp)
- **Ação pendente**: Validar com o usuário; se preferir separação, mover para `ggml-bitnet-sparse-float.cpp` próprio
- **Workaround atual**: ambos os componentes compartilham arquivo
- **Custo de fechar**: 1-2 horas (refactor de extração)

### GAP-07: 3 patches vendored (L3, L5, L4) sem teste de regressão
- **Spec**: `patches/llama.cpp/01-03`, `scripts/apply-dispatch-patches.sh`
- **Status**: 🟡 — idempotência verificada por sentinel-grep, mas sem teste automatizado
- **Ação pendente**: Adicionar `tests/test_patches_idempotent.sh` que rode após `apply-dispatch-patches.sh --check`
- **Custo de fechar**: 30 min - 1 hora

---

## 🟢 Cosmético (não impacta funcionalidade)

### GAP-08: `spec-impact-matrix.md §1` mapeia L1 I2_S MAD → D-10 (impreciso)
- **Spec**: `traceability/spec-impact-matrix.md §1`
- **Status**: 🟢 — D-10 ("2B reusa config 3B") é em `setup_env.py`, não no kernel
- **Ação pendente**: Mover D-10 para o container `setup_env` na matriz
- **Custo de fechar**: 1 minuto

### GAP-09: `c4-containers.md §1` (Mermaid) usa nome `setup_gguf` ao invés de `setup_env`
- **Spec**: `c4-containers.md §1` (diagrama Mermaid)
- **Status**: 🟢 — confusão de nomenclatura (substep de `setup_env.py:setup_gguf()`)
- **Ação pendente**: Renomear para `setup_env` no Mermaid para consistência com a tabela §2
- **Custo de fechar**: 30 segundos

### GAP-10: `erd-complete.md §5` marca RN-014 como "⚠ legacy" mas ENV_VAR §2.10 lista ela
- **Spec**: `erd-complete.md §5` (linha RN-014)
- **Status**: 🟢 — pequena inconsistência de redação
- **Ação pendente**: Mudar "⚠ legacy" para "🟢 ativo (escape hatch legado para GPU)"
- **Custo de fechar**: 30 segundos

### GAP-11: Architecture.md Anexo B vs §8 (métricas) tinham referência stale a `4b7816a`
- **Spec**: `architecture.md §8`
- **Status**: ✅ **JÁ CORRIGIDO** durante esta revisão (commit `68971e2` é o last)
- **Custo de fechar**: aplicado

---

## Resumo por Severidade

| Severidade | Total | Resolvidos 2026-06-06 | Pendentes | Bloqueia reuso? | Bloqueia produção? |
|------------|------:|:---------------------:|:---------:|:---------------:|:------------------:|
| 🔴 Crítico | 3 | 3 (GAP-01, 02, 03) | 0 | — | — |
| 🟡 Moderado | 4 | 1 (GAP-04) | 3 (GAP-05, 06, 07) | não | não |
| 🟢 Cosmético | 4 | 0 | 4 (GAP-08, 09, 10) | não | não |
| **Total** | **11** | **4** | **7** | — | — |

> **Atualização 2026-06-06**: 4/11 gaps resolvidos após processamento das 4 respostas do `questions.md`. Os 7 restantes são:
> - 3 🟡 moderados (GAP-05, 06, 07) — sem decisão do usuário, trabalho mecânico futuro
> - 4 🟢 cosméticos (GAP-08, 09, 10) — < 5 min total de edição

**Status final do ciclo Reviewer (2026-06-06)**: 11 lacunas identificadas, 5 corrigidas in-place (GAP-11 + 4 resolvidas pós-respostas), 7 com trabalho mecânico futuro.

---

## Próximos Passos Recomendados

1. **Curto prazo (1 sessão) — ✅ CONCLUÍDO 2026-06-06**:
   - ~~Responder `questions.md` (4 perguntas)~~ ✅
   - ~~Aplicar GAP-02, GAP-03 (escolher opção A/B/C/D)~~ ✅ (Opção A + Opção Híbrida A+C)
   - ~~Reclassificar Persona A (GAP-04)~~ ✅
   - Corrigir GAP-08, GAP-09, GAP-10 (cosméticos, < 5 min total) — **ainda pendente, trabalho mecânico**

2. **Médio prazo (1-2 sprints)**:
   - Re-executar Detective filtrando `gpu/` upstream (refinamento do `code-analysis.md` se GAP-03 opção B for desejada no futuro)
   - Adicionar teste de regressão dos patches (GAP-07)
   - Documentar limitação L5 d=128 em `architecture.md §1.1` (GAP-05)

3. **Longo prazo (escopo Caminho C, Q4 2029 reserva técnica)**:
   - Validar P6 com modelo treinado (Caminho C, escopo RF-06 — reclassificado em 2026-06-06)
   - Avaliar separação sparse_attention_float em arquivo próprio (GAP-06)
