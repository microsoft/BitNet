"""Exporters — empacota modelos do Studio para outras plataformas.

Alvos:
  gguf    — cópia/validação do GGUF (llama.cpp, LM Studio, Jan, KoboldCpp)
  hf      — diretório HuggingFace (safetensors + tokenizer + config)
  ollama  — Modelfile + instruções de `ollama create`

Tudo offline: o export gera artefatos locais; o push para HF Hub ou
registry do Ollama é decisão (e ação manual) do usuário.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("studio.export")

OLLAMA_TEMPLATE_CHATML = '''FROM {gguf_path}

TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER temperature 0.7
PARAMETER num_ctx {n_ctx}

SYSTEM """{system}"""
'''

DEFAULT_SYSTEM_PTBR = (
    "Você é um assistente fluente em português brasileiro, "
    "preciso e objetivo."
)


def export_gguf(gguf: Path, out_dir: Path, name: str) -> Path:
    """Copia o GGUF com nome canônico + gera SHA256 para verificação."""
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{name}.gguf"
    if dest.resolve() != gguf.resolve():
        log.info("copiando %s → %s", gguf, dest)
        shutil.copy2(gguf, dest)
    sha = hashlib.sha256()
    with dest.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)
    (out_dir / f"{name}.gguf.sha256").write_text(
        f"{sha.hexdigest()}  {dest.name}\n", encoding="utf-8"
    )
    return dest


def export_hf(merged_dir: Path, out_dir: Path, name: str) -> Path:
    """Empacota o diretório HF merged (safetensors) com README mínimo."""
    dest = out_dir / name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(merged_dir, dest)
    readme = dest / "README.md"
    if not readme.exists():
        readme.write_text(
            f"# {name}\n\n"
            f"Modelo exportado pelo BitNet Studio.\n\n"
            f"- Formato: HuggingFace safetensors\n"
            f"- Uso: `AutoModelForCausalLM.from_pretrained('{name}')`\n"
            f"- Compatível com: transformers, vLLM, TGI\n",
            encoding="utf-8",
        )
    log.info("export HF: %s", dest)
    return dest


def export_ollama(gguf: Path, out_dir: Path, name: str,
                  n_ctx: int = 4096,
                  system: str = DEFAULT_SYSTEM_PTBR) -> Path:
    """Gera Modelfile pronto para `ollama create <name> -f Modelfile`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(
        OLLAMA_TEMPLATE_CHATML.format(
            gguf_path=gguf.resolve(), n_ctx=n_ctx, system=system
        ),
        encoding="utf-8",
    )
    instructions = out_dir / "INSTALL.md"
    instructions.write_text(
        f"# Instalar '{name}' no Ollama\n\n"
        f"```bash\n"
        f"ollama create {name} -f {modelfile.resolve()}\n"
        f"ollama run {name}\n"
        f"```\n",
        encoding="utf-8",
    )
    log.info("export Ollama: %s", modelfile)
    return modelfile
