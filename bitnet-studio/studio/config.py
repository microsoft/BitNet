"""Configuração central do BitNet Studio.

Carrega:
- configs/models.yaml  — registry de modelos GGUF locais
- configs/mcp.json     — MCPs declarativos (estilo Claude Desktop)

Tudo offline: nenhum download automático, nenhuma URL remota.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = STUDIO_ROOT / "configs"

DEFAULT_MODELS_DIR = Path(
    os.environ.get("BITNET_MODELS_DIR", "/media/peder/DATA/BitNet/models")
)
DEFAULT_LLAMA_BIN_DIR = Path(
    os.environ.get("BITNET_LLAMA_BIN", str(REPO_ROOT / "build" / "bin"))
)


@dataclass
class ModelEntry:
    name: str
    gguf: str                      # caminho do .gguf
    chat_template: str = "chatml"  # chatml | falcon | llama2
    n_ctx: int = 4096
    description: str = ""

    @property
    def path(self) -> Path:
        p = Path(self.gguf)
        return p if p.is_absolute() else DEFAULT_MODELS_DIR / p


@dataclass
class McpServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class StudioConfig:
    models: dict[str, ModelEntry] = field(default_factory=dict)
    mcps: dict[str, McpServerConfig] = field(default_factory=dict)
    llama_bin_dir: Path = DEFAULT_LLAMA_BIN_DIR
    models_dir: Path = DEFAULT_MODELS_DIR


def load_models(path: Path | None = None) -> dict[str, ModelEntry]:
    path = path or CONFIGS_DIR / "models.yaml"
    if not path.exists():
        return {}
    data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict[str, ModelEntry] = {}
    for name, spec in (data.get("models") or {}).items():
        out[name] = ModelEntry(
            name=name,
            gguf=spec["gguf"],
            chat_template=spec.get("chat_template", "chatml"),
            n_ctx=int(spec.get("n_ctx", 4096)),
            description=spec.get("description", ""),
        )
    return out


def load_mcps(path: Path | None = None) -> dict[str, McpServerConfig]:
    path = path or CONFIGS_DIR / "mcp.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, McpServerConfig] = {}
    for name, spec in (data.get("mcpServers") or {}).items():
        out[name] = McpServerConfig(
            name=name,
            command=spec["command"],
            args=list(spec.get("args", [])),
            env=dict(spec.get("env", {})),
            enabled=bool(spec.get("enabled", True)),
        )
    return out


def load_config() -> StudioConfig:
    return StudioConfig(models=load_models(), mcps=load_mcps())
