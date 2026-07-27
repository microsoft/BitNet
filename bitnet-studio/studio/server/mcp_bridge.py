"""Bridge MCP — cliente stdio JSON-RPC 2.0 para Model Context Protocol.

Conecta a qualquer servidor MCP "plugável" (ex: protheus-rag), descobre as
tools via tools/list e executa via tools/call. Tudo local via subprocess —
nenhuma conexão de rede é aberta pelo bridge em si (persona D4: o MCP em si
pode ser auditado separadamente).

Protocolo: https://spec.modelcontextprotocol.io (JSON-RPC 2.0 sobre stdio,
mensagens delimitadas por newline).
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any

from studio.config import McpServerConfig

log = logging.getLogger("studio.mcp")

PROTOCOL_VERSION = "2024-11-05"
CLIENT_INFO = {"name": "bitnet-studio", "version": "0.1.0"}
DEFAULT_TIMEOUT = 60.0


@dataclass
class McpTool:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def qualified_name(self) -> str:
        """Nome exposto ao modelo: servidor__tool (evita colisão entre MCPs)."""
        return f"{self.server}__{self.name}"

    def to_openai(self) -> dict[str, Any]:
        """Formato de tool da API OpenAI (function calling)."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description or "",
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


class McpClient:
    """Cliente de um único servidor MCP via stdio."""

    def __init__(self, cfg: McpServerConfig):
        self.cfg = cfg
        self.proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()
        self.tools: list[McpTool] = []

    # ── lifecycle ───────────────────────────────────────────────────────────
    def start(self) -> None:
        import os

        env = {**os.environ, **self.cfg.env}
        self.proc = subprocess.Popen(
            [self.cfg.command, *self.cfg.args],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            bufsize=1,
        )
        self._handshake()
        self._discover_tools()
        log.info("MCP '%s': %d tools", self.cfg.name, len(self.tools))

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ── JSON-RPC ────────────────────────────────────────────────────────────
    def _next_id(self) -> int:
        self._id += 1
        return self._id

    def _send(self, payload: dict[str, Any]) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()

    def _recv(self, want_id: int, timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        """Lê linhas até achar a resposta com o id esperado (ignora notifications)."""
        assert self.proc and self.proc.stdout
        import select

        deadline_fd = self.proc.stdout.fileno()
        remaining = timeout
        while remaining > 0:
            ready, _, _ = select.select([deadline_fd], [], [], min(remaining, 1.0))
            remaining -= 1.0
            if not ready:
                continue
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP '{self.cfg.name}': stdout fechado")
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == want_id:
                if "error" in msg:
                    raise RuntimeError(f"MCP '{self.cfg.name}': {msg['error']}")
                return msg.get("result", {})
        raise TimeoutError(f"MCP '{self.cfg.name}': timeout esperando id={want_id}")

    def _request(self, method: str, params: dict[str, Any] | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
        with self._lock:
            rid = self._next_id()
            self._send({"jsonrpc": "2.0", "id": rid, "method": method,
                        "params": params or {}})
            return self._recv(rid, timeout)

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    # ── MCP protocol ───────────────────────────────────────────────────────
    def _handshake(self) -> None:
        self._request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        }, timeout=30.0)
        self._notify("notifications/initialized")

    def _discover_tools(self) -> None:
        result = self._request("tools/list")
        self.tools = [
            McpTool(
                server=self.cfg.name,
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            for t in result.get("tools", [])
        ]

    def call_tool(self, name: str, arguments: dict[str, Any],
                  timeout: float = DEFAULT_TIMEOUT) -> str:
        """Executa a tool e devolve o conteúdo textual concatenado."""
        result = self._request("tools/call",
                               {"name": name, "arguments": arguments},
                               timeout=timeout)
        parts: list[str] = []
        for item in result.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        if result.get("isError"):
            return f"[tool error] {' '.join(parts)}"
        return "\n".join(parts) if parts else "(sem conteúdo)"


class McpRegistry:
    """Registry de MCPs: declarativos (mcp.json) + hot-plug em runtime."""

    def __init__(self):
        self.clients: dict[str, McpClient] = {}
        self._lock = threading.Lock()

    def start_from_config(self, cfgs: dict[str, McpServerConfig]) -> None:
        for name, cfg in cfgs.items():
            if not cfg.enabled:
                continue
            try:
                self.add(cfg)
            except Exception as e:  # noqa: BLE001 — boot não pode morrer por 1 MCP
                log.error("MCP '%s' falhou no boot: %s", name, e)

    def add(self, cfg: McpServerConfig) -> McpClient:
        with self._lock:
            if cfg.name in self.clients:
                self.clients[cfg.name].stop()
            client = McpClient(cfg)
            client.start()
            self.clients[cfg.name] = client
            return client

    def remove(self, name: str) -> bool:
        with self._lock:
            client = self.clients.pop(name, None)
            if client:
                client.stop()
                return True
            return False

    def all_tools(self) -> list[McpTool]:
        return [t for c in self.clients.values() if c.alive for t in c.tools]

    def call(self, qualified_name: str, arguments: dict[str, Any]) -> str:
        """qualified_name = servidor__tool."""
        server, _, tool = qualified_name.partition("__")
        client = self.clients.get(server)
        if not client or not client.alive:
            return f"[erro] MCP '{server}' não está conectado"
        return client.call_tool(tool, arguments)

    def shutdown(self) -> None:
        for c in self.clients.values():
            c.stop()
        self.clients.clear()
