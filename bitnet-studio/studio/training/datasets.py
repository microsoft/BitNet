"""Preparação de datasets PT-BR + tool-calling para QLoRA.

Formato de entrada: JSONL com conversas no estilo OpenAI:
  {"messages": [{"role": "system|user|assistant", "content": "..."}]}

Tool calls no dataset usam o MESMO formato que o tool_engine ensina em
runtime (<tool_call>{json}</tool_call>) — assim o fine-tune reforça
exatamente o comportamento que o servidor espera.

Inclui gerador de dataset sintético de tool-calling a partir das tools
reais de um MCP conectado (ex: protheus-rag) para bootstrap rápido.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

CHATML_TEMPLATE = "<|im_start|>{role}\n{content}<|im_end|>\n"


def conversation_to_text(messages: list[dict[str, str]]) -> str:
    """Converte mensagens OpenAI em texto chatml puro para o trainer."""
    return "".join(
        CHATML_TEMPLATE.format(role=m["role"], content=m["content"])
        for m in messages
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_dataset(path: Path) -> tuple[int, list[str]]:
    """Valida o JSONL: retorna (n_validas, lista_de_erros)."""
    errors: list[str] = []
    valid = 0
    for i, row in enumerate(load_jsonl(path)):
        msgs = row.get("messages")
        if not isinstance(msgs, list) or not msgs:
            errors.append(f"linha {i+1}: sem 'messages'")
            continue
        roles = {m.get("role") for m in msgs}
        if not roles <= {"system", "user", "assistant"}:
            errors.append(f"linha {i+1}: roles inválidos {roles}")
            continue
        if not any(m.get("role") == "assistant" for m in msgs):
            errors.append(f"linha {i+1}: sem resposta do assistant")
            continue
        valid += 1
    return valid, errors


# ── Gerador sintético de tool-calling ───────────────────────────────────────

_PTBR_QUESTION_TEMPLATES = [
    "Consulte {tool} e me diga: {ask}",
    "Preciso de uma informação: {ask}. Use a ferramenta adequada.",
    "{ask}",
    "Por favor, verifique {ask} no sistema.",
    "Me ajuda com isso: {ask}?",
]

_PTBR_ANSWER_TEMPLATES = [
    "Com base na consulta, encontrei o seguinte:\n\n{result}",
    "Aqui está o que o sistema retornou:\n\n{result}",
    "Consultei a ferramenta e o resultado foi:\n\n{result}",
]


def synth_tool_examples(
    tools: list[dict[str, Any]],
    asks_per_tool: list[str],
    n_per_tool: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Gera exemplos sintéticos de tool-calling em PT-BR.

    tools: [{"name": ..., "description": ..., "parameters": {...}}]
    asks_per_tool: perguntas-modelo (uma lista geral reutilizada por tool)
    """
    rng = random.Random(seed)
    examples = []
    for tool in tools:
        name = tool["name"]
        for _ in range(n_per_tool):
            ask = rng.choice(asks_per_tool)
            q = rng.choice(_PTBR_QUESTION_TEMPLATES).format(tool=name, ask=ask)
            args = {"pergunta": ask} if "pergunta" in json.dumps(
                tool.get("parameters", {})) else {"query": ask}
            call = json.dumps({"name": name, "arguments": args},
                              ensure_ascii=False)
            fake_result = f"[resultado simulado da tool {name} para: {ask}]"
            answer = rng.choice(_PTBR_ANSWER_TEMPLATES).format(
                result=fake_result)
            examples.append({"messages": [
                {"role": "user", "content": q},
                {"role": "assistant",
                 "content": f"<tool_call>\n{call}\n</tool_call>"},
                {"role": "user",
                 "content": f"Resultado da ferramenta `{name}`:\n"
                            f"```\n{fake_result}\n```\n"
                            f"Use esse resultado para responder ao usuário "
                            f"em português."},
                {"role": "assistant", "content": answer},
            ]})
    return examples


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
