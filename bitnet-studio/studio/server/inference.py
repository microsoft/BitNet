"""Backend de inferência — gerencia llama-server (build BitNet L2-L5) como subprocess.

CPU-only por design (persona D4). O llama-server do nosso build expõe
/completion em localhost; este módulo sobe/derruba o processo e formata
prompts com o chat template correto.
"""

from __future__ import annotations

import logging
import socket
import subprocess
import time
from pathlib import Path

import httpx

from studio.config import ModelEntry

log = logging.getLogger("studio.inference")

DEFAULT_PORT = 8089
STARTUP_TIMEOUT = 120.0  # 10B 1.58bit pode demorar a carregar


# ── Chat templates ──────────────────────────────────────────────────────────

def _chatml(messages: list[dict]) -> str:
    out = []
    for m in messages:
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
    out.append("<|im_start|>assistant\n")
    return "\n".join(out)


def _falcon(messages: list[dict]) -> str:
    # Falcon3-Instruct: <|system|>, <|user|>, <|assistant|>
    out = []
    for m in messages:
        if m["role"] == "system":
            out.append(f"<|system|>\n{m['content']}")
        elif m["role"] == "user":
            out.append(f"<|user|>\n{m['content']}")
        else:
            out.append(f"<|assistant|>\n{m['content']}")
    out.append("<|assistant|>\n")
    return "\n".join(out)


def _llama2(messages: list[dict]) -> str:
    system = ""
    convo = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            convo.append(("user", m["content"]))
        else:
            convo.append(("assistant", m["content"]))
    out = ""
    first = True
    for role, content in convo:
        if role == "user":
            if first and system:
                out += f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{content} [/INST]"
                first = False
            else:
                out += f"<s>[INST] {content} [/INST]"
        else:
            out += f" {content} </s>"
    return out


TEMPLATES = {"chatml": _chatml, "falcon": _falcon, "llama2": _llama2}

STOP_TOKENS = {
    "chatml": ["<|im_end|>", "<|im_start|>"],
    "falcon": [],
    "llama2": ["</s>", "[INST]"],
}


def _free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


class LlamaServerBackend:
    """Sobe um llama-server por modelo carregado. Um modelo ativo por vez."""

    def __init__(self, llama_bin_dir: Path, threads: int | None = None):
        self.bin = llama_bin_dir / "llama-server"
        if not self.bin.exists():
            raise FileNotFoundError(
                f"llama-server não encontrado em {self.bin}. "
                "Compile com: cmake --build build -j"
            )
        self.threads = threads
        self.proc: subprocess.Popen | None = None
        self.port: int = DEFAULT_PORT
        self.model: ModelEntry | None = None
        # Read timeout alto: prompt eval de system prompt longo (tools) em
        # CPU pode passar de 5min na primeira chamada (sem KV cache ainda).
        self._client = httpx.Client(timeout=httpx.Timeout(900.0, connect=10.0))

    # ── lifecycle ───────────────────────────────────────────────────────────
    def load(self, model: ModelEntry) -> None:
        if self.model and self.model.name == model.name and self.alive:
            return
        self.unload()
        gguf = model.path
        if not gguf.exists():
            raise FileNotFoundError(f"GGUF não encontrado: {gguf}")
        self.port = _free_port(DEFAULT_PORT)
        cmd = [
            str(self.bin),
            "-m", str(gguf),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(model.n_ctx),
            "--log-disable",
        ]
        # CRÍTICO: kernels ternários i2_s (BitNet/Falcon 1.58bit) só suportam
        # batch=1. Com o batch default (2048) o kernel produz output vazio
        # silenciosamente. Modelos Q4/fp16 usam o default (mais rápido).
        if "i2_s" in gguf.name.lower():
            cmd += ["-b", "1", "-ub", "1"]
        if self.threads:
            cmd += ["-t", str(self.threads)]
        else:
            import os
            cmd += ["-t", str(max(1, (os.cpu_count() or 4) - 2))]
        log.info("carregando %s (porta %d)...", model.name, self.port)
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        self._wait_ready()
        self.model = model
        log.info("modelo %s pronto", model.name)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        url = f"http://127.0.0.1:{self.port}/health"
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("llama-server morreu durante o load")
            try:
                r = self._client.get(url)
                if r.status_code == 200:
                    return
            except httpx.TransportError:
                pass
            time.sleep(1.0)
        raise TimeoutError("llama-server não ficou pronto a tempo")

    def unload(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        self.model = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ── inference ───────────────────────────────────────────────────────────
    def chat(self, messages: list[dict], *, temperature: float = 0.7,
             max_tokens: int = 1024, stop: list[str] | None = None) -> str:
        if not self.model or not self.alive:
            raise RuntimeError("nenhum modelo carregado")
        template = TEMPLATES.get(self.model.chat_template, _chatml)
        prompt = template(messages)
        stops = list(stop or []) + STOP_TOKENS.get(self.model.chat_template, [])
        r = self._client.post(
            f"http://127.0.0.1:{self.port}/completion",
            json={
                "prompt": prompt,
                "temperature": temperature,
                "n_predict": max_tokens,
                "stop": stops,
                "cache_prompt": True,
            },
        )
        r.raise_for_status()
        text = r.json().get("content", "").strip()
        # Falcon3 gera a continuação da conversa (\nuser\n...) sem stop
        # adequado em GGUF. Truncar na primeira ocorrência do próximo turno.
        if self.model and self.model.chat_template == "falcon":
            for marker in ["\n<|user|>", "\nuser\n", "\n<|system|>"]:
                if marker in text:
                    text = text.split(marker)[0].strip()
                    break
        return text
