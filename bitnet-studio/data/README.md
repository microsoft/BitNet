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

### Treino GPU — Falcon3-3B (Google Colab T4)

Abrir `colab_finetune.ipynb` no Google Colab e executar todas as células.

---

## Escalando para Falcon3-10B-Instruct

### Comparação de Hardware

| Opção | VRAM/RAM | Tempo 300 steps | Custo |
|-------|----------|----------------|-------|
| **Seu CPU** (35GB RAM) | ~20GB RAM | ~5-8 dias | Apenas energia |
| **Colab T4** (16GB) | ~12-14GB VRAM | ~30-40 min | Gratuito |
| **Colab Pro A100** (40GB) | ~12-14GB VRAM | ~10-15 min | $10/mês |
| **RunPod RTX 3090** (24GB) | ~12-14GB VRAM | ~15-20 min | ~$0.20/hora |

### ⚠️ Limitações do T4 Gratuito

O Falcon 10B em QLoRA 4-bit consome **~12-14GB VRAM**. No Colab T4 (16GB):
- Use `seq_len=128` (não 256)
- Use `batch_size=1` obrigatoriamente
- Use apenas camadas de atenção no LoRA (target_modules=`q_proj,k_proj,v_proj,o_proj`)
- Ative `gradient_checkpointing=True`
- **Risco de OOM**: se ocorrer, reduza para Falcon3-3B

### Scripts Falcon 10B

- `finetune_falcon10b_cpu.py` — CPU local (~20GB RAM, ~50min/step)
- `finetune_falcon10b_gpu.py` — GPU paga (RTX 3090/A100, ~15-20min total)
- `colab_finetune_falcon10b.ipynb` — Colab T4 (otimizado com seq=128)

### Recomendação

Para Falcon 10B, use **RunPod/Vast.ai com RTX 3090** (~$0.20/hora) ou **Colab Pro A100**. O T4 gratuito funciona mas é instável com 10B.

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
