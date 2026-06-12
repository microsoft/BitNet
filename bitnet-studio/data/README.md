# Dataset de Fine-tune PT-BR Tool-calling

## Visão Geral

Dataset para fine-tune de modelos Falcon3-Instruct em português brasileiro com habilidades de tool-calling via MCP (Model Context Protocol).

## Estrutura

Formato OpenAI Chat: conversas de 4 turnos (user → assistant tool call → user tool result → assistant answer).

```json
{
  "messages": [
    {"role": "user", "content": "pergunta do usuário"},
    {"role": "assistant", "content": "<tool_call>\n{\"name\": \"tool-name\", \"arguments\": {...}}\n</tool_call>"},
    {"role": "user", "content": "Resultado da ferramenta `tool-name`:\n```\n...\n```\nUse esse resultado para responder ao usuário em português."},
    {"role": "assistant", "content": "resposta final em português"}
  ]
}
```

## Arquivos

- `ptbr_tools_train.jsonl` — 61 exemplos (dataset inicial)
- `ptbr_tools_train_large.jsonl` — 162 exemplos (dataset expandido com variações)
- `ptbr_tools_train_premium.jsonl` — 19 exemplos (dataset premium com exemplos realistas)

## Tools Cobertas

1. **protheus-rag__consultar_base_direta** — Busca código-fonte AdvPL/Protheus
2. **protheus-rag__consultar_dicionario_direto** — Consulta dicionário de dados (SX3, SIX, SX6)
3. **protheus-rag__consultar_base_interna** — RAG interpretado com síntese
4. **protheus-rag__buscar_reversa_direto** — Busca skills do Reversa Framework
5. **protheus-rag__consultar_reversa_rag** — Consulta interpretada Reversa
6. **protheus-rag__mem0_search** — Busca memórias persistentes
7. **protheus-rag__mem0_add** — Adiciona memória
8. **protheus-rag__mem0_list** — Lista memórias
9. **protheus-rag__mem0_stats** — Estatísticas de memórias
10. **protheus-rag__mem0_delete** — Remove memória

## Uso

### Treino CPU (lento, ~10 min/step)

```bash
python3 finetune_cpu.py
```

### Treino CPU Piloto (rápido, 5 steps)

```bash
python3 finetune_cpu_mini.py
```

### Treino GPU (Google Colab T4)

Abrir `colab_finetune.ipynb` no Google Colab e executar todas as células.

## Validação

```bash
bitnet-studio dataset validate data/ptbr_tools_train.jsonl
```

## Merge e Quantize

```bash
python3 merge_and_quantize.py
# Ou via CLI:
bitnet-studio merge --base tiiuae/Falcon3-3B-Instruct \
    --adapter adapters/f3b-ptbr-tools-cpu \
    --name falcon3-3b-ptbr-tools
```
