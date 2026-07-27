"""CLI do BitNet Studio.

Comandos:
  serve      — sobe API + Web UI (CPU-only, D4)
  models     — lista modelos do registry
  dataset    — valida/gera datasets PT-BR + tools
  finetune   — QLoRA 4-bit em GPU modesta
  merge      — adapter → HF merged → GGUF quantizado
  export     — GGUF / HF / Ollama
  mcp        — testa conexão com um MCP do mcp.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("studio.cli")


# ── comandos ────────────────────────────────────────────────────────────────

def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "studio.server.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    from studio.config import load_models

    models = load_models()
    if not models:
        print("nenhum modelo no registry (configs/models.yaml)")
        return 1
    for name, m in models.items():
        status = "OK " if m.path.exists() else "AUSENTE"
        print(f"[{status}] {name:40s} {m.path}")
    return 0


def cmd_dataset(args: argparse.Namespace) -> int:
    from studio.training.datasets import (
        synth_tool_examples,
        validate_dataset,
        write_jsonl,
    )

    if args.action == "validate":
        valid, errors = validate_dataset(Path(args.path))
        print(f"válidas: {valid}, erros: {len(errors)}")
        for e in errors[:20]:
            print(f"  - {e}")
        return 0 if not errors else 1

    if args.action == "synth":
        tools = json.loads(Path(args.tools_json).read_text(encoding="utf-8"))
        asks = [
            a.strip()
            for a in Path(args.asks).read_text(encoding="utf-8").splitlines()
            if a.strip()
        ]
        rows = synth_tool_examples(tools, asks, n_per_tool=args.n)
        write_jsonl(rows, Path(args.path))
        print(f"{len(rows)} exemplos sintéticos → {args.path}")
        return 0
    return 1


def cmd_finetune(args: argparse.Namespace) -> int:
    from studio.training.qlora import QLoraConfig, run_qlora

    cfg = QLoraConfig(
        base_model=args.base,
        dataset_path=args.dataset,
        output_dir=args.out,
        max_seq_len=args.max_seq,
        epochs=args.epochs,
        lora_r=args.lora_r,
        local_files_only=args.offline,
    )
    run_qlora(cfg)
    print(f"adapter pronto: {args.out}")
    print("próximo passo: bitnet-studio merge "
          f"--base {args.base} --adapter {args.out} "
          f"--name <nome> --workdir work/")
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    from studio.training.merge_quantize import full_pipeline

    final = full_pipeline(
        base_model=args.base,
        adapter_dir=Path(args.adapter),
        work_dir=Path(args.workdir),
        model_name=args.name,
        quant=args.quant,
        local_files_only=args.offline,
    )
    print(f"GGUF final: {final}")
    print(f"registre em configs/models.yaml:\n"
          f"  {args.name}:\n"
          f"    gguf: {final}\n"
          f"    chat_template: chatml")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from studio.export.exporters import export_gguf, export_hf, export_ollama

    out = Path(args.out)
    if args.target == "gguf":
        dest = export_gguf(Path(args.source), out, args.name)
    elif args.target == "hf":
        dest = export_hf(Path(args.source), out, args.name)
    elif args.target == "ollama":
        dest = export_ollama(Path(args.source), out, args.name,
                             n_ctx=args.n_ctx)
    else:
        print(f"target desconhecido: {args.target}")
        return 1
    print(f"export {args.target}: {dest}")
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from studio.config import load_mcps
    from studio.server.mcp_bridge import McpClient

    mcps = load_mcps()
    cfg = mcps.get(args.name)
    if not cfg:
        print(f"MCP '{args.name}' não está em configs/mcp.json")
        print(f"disponíveis: {list(mcps)}")
        return 1
    client = McpClient(cfg)
    try:
        client.start()
        print(f"conectado: {cfg.name} ({len(client.tools)} tools)")
        for t in client.tools:
            print(f"  - {t.qualified_name}: {t.description[:80]}")
        if args.call:
            result = client.call_tool(args.call,
                                      json.loads(args.args or "{}"))
            print(f"\nresultado de {args.call}:\n{result[:2000]}")
    finally:
        client.stop()
    return 0


# ── parser ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bitnet-studio",
        description="BitNet Studio — treino, serve, MCP e export "
                    "(CPU-only D4 na inferência)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="sobe API + Web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8080)
    s.set_defaults(fn=cmd_serve)

    s = sub.add_parser("models", help="lista modelos do registry")
    s.set_defaults(fn=cmd_models)

    s = sub.add_parser("dataset", help="valida/gera datasets")
    s.add_argument("action", choices=["validate", "synth"])
    s.add_argument("path", help="caminho do JSONL")
    s.add_argument("--tools-json", help="JSON com tools (para synth)")
    s.add_argument("--asks", help="arquivo .txt com perguntas (para synth)")
    s.add_argument("-n", type=int, default=5, help="exemplos por tool")
    s.set_defaults(fn=cmd_dataset)

    s = sub.add_parser("finetune", help="QLoRA 4-bit (GPU)")
    s.add_argument("--base", default="tiiuae/Falcon3-10B-Instruct")
    s.add_argument("--dataset", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--max-seq", type=int, default=1024)
    s.add_argument("--epochs", type=float, default=3.0)
    s.add_argument("--lora-r", type=int, default=16)
    s.add_argument("--offline", action="store_true",
                   help="local_files_only (air-gapped)")
    s.set_defaults(fn=cmd_finetune)

    s = sub.add_parser("merge", help="adapter → GGUF quantizado")
    s.add_argument("--base", required=True)
    s.add_argument("--adapter", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--workdir", default="work")
    s.add_argument("--quant", default="Q4_K_M")
    s.add_argument("--offline", action="store_true")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("export", help="exporta para outras plataformas")
    s.add_argument("target", choices=["gguf", "hf", "ollama"])
    s.add_argument("--source", required=True,
                   help="GGUF (gguf/ollama) ou dir HF merged (hf)")
    s.add_argument("--name", required=True)
    s.add_argument("--out", default="exports")
    s.add_argument("--n-ctx", type=int, default=4096)
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("mcp", help="testa um MCP do mcp.json")
    s.add_argument("name", help="nome do MCP em configs/mcp.json")
    s.add_argument("--call", help="tool para invocar (teste)")
    s.add_argument("--args", help="argumentos JSON da tool")
    s.set_defaults(fn=cmd_mcp)

    return p


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
