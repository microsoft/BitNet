"""API do BitNet Studio — OpenAI-compatible + gestão de modelos e MCPs.

Endpoints:
  POST /v1/chat/completions   — chat com loop agentic de tools (OpenAI-compat)
  GET  /v1/models             — lista modelos do registry
  POST /v1/models/{name}/load — carrega um modelo no backend
  GET  /mcp                   — lista MCPs conectados e suas tools
  POST /mcp                   — hot-plug de um MCP em runtime
  DELETE /mcp/{name}          — remove um MCP
  GET  /                      — Web UI local (sem cloud)

Tudo roda em localhost. Nenhuma telemetria, nenhuma chamada externa.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from studio.config import McpServerConfig, StudioConfig, load_config
from studio.server.inference import LlamaServerBackend
from studio.server.mcp_bridge import McpRegistry
from studio.server.tool_engine import (
    build_system_prompt,
    format_tool_result,
    parse_tool_call,
)

log = logging.getLogger("studio.api")

MAX_TOOL_ITERATIONS = 8
WEBUI_DIR = Path(__file__).resolve().parents[1] / "webui"

# ── estado global (single-process por design) ───────────────────────────────
cfg: StudioConfig
backend: LlamaServerBackend
mcps: McpRegistry


@asynccontextmanager
async def lifespan(app: FastAPI):
    global cfg, backend, mcps
    cfg = load_config()
    backend = LlamaServerBackend(cfg.llama_bin_dir)
    mcps = McpRegistry()
    mcps.start_from_config(cfg.mcps)
    log.info("studio pronto: %d modelos, %d MCPs",
             len(cfg.models), len(mcps.clients))
    yield
    mcps.shutdown()
    backend.unload()


app = FastAPI(title="BitNet Studio", version="0.1.0", lifespan=lifespan)


# ── Schemas ─────────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = Field(default=1024, le=8192)
    use_tools: bool = True
    system_extra: str = ""


class McpAddRequest(BaseModel):
    name: str
    command: str
    args: list[str] = []
    env: dict[str, str] = {}


# ── Chat com loop agentic ───────────────────────────────────────────────────

@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest) -> dict[str, Any]:
    entry = cfg.models.get(req.model)
    if not entry:
        raise HTTPException(404, f"modelo '{req.model}' não está no registry")
    backend.load(entry)

    tools = mcps.all_tools() if req.use_tools else []
    messages: list[dict] = []
    has_system = any(m.role == "system" for m in req.messages)
    if tools and not has_system:
        messages.append({"role": "system",
                         "content": build_system_prompt(tools, req.system_extra)})
    messages.extend(m.model_dump() for m in req.messages)

    tool_trace: list[dict] = []
    answer = ""
    for _ in range(MAX_TOOL_ITERATIONS):
        answer = backend.chat(
            messages, temperature=req.temperature, max_tokens=req.max_tokens
        )
        tc = parse_tool_call(answer, tools) if tools else None
        if tc is None:
            break
        log.info("tool call: %s(%s)", tc.name, tc.arguments)
        result = mcps.call(tc.name, tc.arguments)
        tool_trace.append({"tool": tc.name, "arguments": tc.arguments,
                           "result": result[:2000]})
        messages.append({"role": "assistant", "content": answer})
        messages.append({"role": "user",
                         "content": format_tool_result(tc.name, result)})
    else:
        answer += "\n\n[aviso: limite de iterações de tools atingido]"

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "tool_trace": tool_trace,
    }


# ── Modelos ─────────────────────────────────────────────────────────────────

@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": name,
                "object": "model",
                "gguf": str(m.path),
                "exists": m.path.exists(),
                "chat_template": m.chat_template,
                "n_ctx": m.n_ctx,
                "description": m.description,
                "loaded": backend.model is not None and backend.model.name == name,
            }
            for name, m in cfg.models.items()
        ],
    }


@app.post("/v1/models/{name}/load")
def load_model(name: str) -> dict[str, str]:
    entry = cfg.models.get(name)
    if not entry:
        raise HTTPException(404, f"modelo '{name}' não está no registry")
    backend.load(entry)
    return {"status": "loaded", "model": name}


# ── MCPs (declarativo + hot-plug) ───────────────────────────────────────────

@app.get("/mcp")
def list_mcps() -> dict[str, Any]:
    return {
        name: {
            "alive": c.alive,
            "tools": [
                {"name": t.qualified_name, "description": t.description}
                for t in c.tools
            ],
        }
        for name, c in mcps.clients.items()
    }


@app.post("/mcp")
def add_mcp(req: McpAddRequest) -> dict[str, Any]:
    cfg_mcp = McpServerConfig(name=req.name, command=req.command,
                              args=req.args, env=req.env)
    try:
        client = mcps.add(cfg_mcp)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"falha ao conectar MCP: {e}") from e
    return {"status": "connected", "name": req.name,
            "tools": [t.qualified_name for t in client.tools]}


@app.delete("/mcp/{name}")
def remove_mcp(name: str) -> dict[str, str]:
    if not mcps.remove(name):
        raise HTTPException(404, f"MCP '{name}' não encontrado")
    return {"status": "removed", "name": name}


# ── Web UI ──────────────────────────────────────────────────────────────────

@app.get("/")
def webui() -> FileResponse:
    index = WEBUI_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"error": "webui não encontrada"}, status_code=404)  # type: ignore[return-value]
    return FileResponse(index)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": backend.model.name if backend.model else None,
        "mcps": {n: c.alive for n, c in mcps.clients.items()},
    }
