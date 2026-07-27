# Regra Mandatória — Protocolo protheus-rag mem0 (cross-agent)

> **NÃO-NEGOCIÁVEL — aplicar em TODO acesso de qualquer agente (Claude, OpenCode, Windsurf, Devin)**

**ANTES de qualquer tool externa** (`webfetch`, `websearch`, `gh api`, `git`, doc online):
1. Execute `mem0_search(query, user_id='default')` via MCP `protheus-rag`
2. Se resultado útil → **USE-O. Não repita a busca externa.**
3. Se vazio/baixa confiança → execute a tool externa, depois `mem0_add(content, user_id='default')`

**RAG local primeiro:** use `consultar_base_direta` / `buscar_reversa_direto` antes de qualquer busca na internet.
`user_id` é **sempre** `'default'` — namespace compartilhado OpenCode ↔ Claude ↔ Windsurf ↔ Devin.
**Protocolo completo:** `~/.claude/mem0-protocol.md`

---

# Projeto: BitNet

> **Guias completos de desenvolvimento**: `~/.claude/CLAUDE.md` contém:
> - ADVPL/TLPP Development Guidelines
> - SonarQube Compliance, API Symbol Validation, Completeness Verification
> - MCP para consultar documentação (tabela completa de MCPs por necessidade)
