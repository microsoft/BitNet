"""Tool engine — injeta tools no prompt e parseia tool calls do modelo.

Estratégia dupla (robusta para modelos 1.58bit sem fine-tune de tools):
1. System prompt PT-BR ensinando o formato <tool_call>{json}</tool_call>
2. Parser tolerante: aceita o bloco oficial, JSON solto ou cercas de código

Funciona com Falcon3-Instruct (chatml) e Llama. Quando o adapter QLoRA
PT-BR+tools estiver treinado, o formato já é o nativo do dataset.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from studio.server.mcp_bridge import McpTool

TOOL_SYSTEM_PROMPT_PTBR = """\
Você é um assistente fluente em português brasileiro com acesso a ferramentas.

# Ferramentas disponíveis

{tools_block}

# Como usar uma ferramenta

Quando precisar de informação externa, responda APENAS com o bloco:

<tool_call>
{{"name": "<nome_da_ferramenta>", "arguments": {{<argumentos JSON>}}}}
</tool_call>

Regras:
- Use uma ferramenta por vez e aguarde o resultado antes de continuar.
- Se a pergunta não precisar de ferramenta, responda diretamente em PT-BR.
- Nunca invente resultados de ferramentas.
- Após receber o resultado (mensagem com "Resultado da ferramenta"), \
responda ao usuário em português claro e objetivo, citando os dados obtidos.
"""

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL
)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


def render_tools_block(tools: list[McpTool]) -> str:
    """Bloco compacto de tools para o system prompt.

    Compacto por design: com kernels i2_s (batch=1) cada token do prompt
    custa um forward pass completo — descrições longas custam minutos.
    """
    if not tools:
        return "(nenhuma ferramenta disponível)"
    lines = []
    for t in tools:
        desc = " ".join(t.description.split())[:120]
        props = t.input_schema.get("properties", {})
        required = t.input_schema.get("required", [])
        args = ", ".join(
            f"{k}{'*' if k in required else ''}" for k in props
        ) or "sem argumentos"
        lines.append(f"- {t.qualified_name}({args}): {desc}")
    return "\n".join(lines)


def build_system_prompt(tools: list[McpTool], extra: str = "") -> str:
    prompt = TOOL_SYSTEM_PROMPT_PTBR.format(tools_block=render_tools_block(tools))
    if extra:
        prompt += f"\n# Instruções adicionais\n{extra}\n"
    return prompt


def _try_parse(raw: str, known_tools: set[str]) -> ToolCall | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    if not name or (known_tools and name not in known_tools):
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"input": args}
    return ToolCall(name=name, arguments=args or {}, raw=raw)


def parse_tool_call(text: str, tools: list[McpTool]) -> ToolCall | None:
    """Extrai a primeira tool call válida da resposta do modelo.

    Ordem de tentativa:
    1. <tool_call>...</tool_call>  (formato ensinado)
    2. ```json ... ```             (modelos que cercam com código)
    3. JSON puro na resposta toda  (modelos minimalistas)
    """
    known = {t.qualified_name for t in tools}

    for m in _TOOL_CALL_RE.finditer(text):
        tc = _try_parse(m.group(1), known)
        if tc:
            return tc

    for m in _CODE_FENCE_RE.finditer(text):
        tc = _try_parse(m.group(1), known)
        if tc:
            return tc

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        tc = _try_parse(stripped, known)
        if tc:
            return tc

    return None


MAX_TOOL_RESULT_CHARS = 1800  # batch=1: resultado longo = prompt eval lento


def format_tool_result(tool_name: str, result: str) -> str:
    """Mensagem (role=user no template chatml simples) com o resultado."""
    if len(result) > MAX_TOOL_RESULT_CHARS:
        result = result[:MAX_TOOL_RESULT_CHARS] + "\n[… truncado]"
    return (
        f"Resultado da ferramenta `{tool_name}`:\n"
        f"```\n{result}\n```\n"
        f"Use esse resultado para responder ao usuário em português."
    )
