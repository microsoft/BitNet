#!/usr/bin/env python3
"""
bench_publish.py — Publish BitNet-CPU kernel benchmarks as JSON + Markdown

CLI with 2 modes:
  Mode 1 (--json): runs `utils/cpu_universal_benchmark.py` and emits a
    canonical JSON file with hardware/methodology/rows.
  Mode 2 (--from-json <file> --md): reads a JSON file and renders the
    derived Markdown report.

The JSON is the source of truth; the Markdown is generated from it.
This avoids the "two formats to maintain" risk (R-06 do roadmap.md).

Usage:
  # Mode 1: run bench and emit JSON
  python utils/bench_publish.py \\
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \\
    --json > benchmarks/v0.1.0/bench.json

  # Mode 2: render Markdown from JSON
  python utils/bench_publish.py \\
    --from-json benchmarks/v0.1.0/bench.json \\
    --md > benchmarks/v0.1.0/bench.md

  # Mode 1 with --md in one go (composes the two):
  python utils/bench_publish.py \\
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \\
    --json benchmarks/v0.1.0/bench.json \\
    --md benchmarks/v0.1.0/bench.md

AC-05 (do requirements.md#6): "Bench sistemático commitado em
benchmarks/v0.1.0/ mostra baseline L1 vs L3 vs L4 com números."
"""
import argparse
import csv
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "0.1.0"


def detect_hardware():
    """Collect hardware metadata: CPU model, cores, RAM, OS, etc."""
    hw = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }
    # CPU model on Linux from /proc/cpuinfo
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                m = re.match(r"model name\s*:\s*(.*)", line)
                if m:
                    hw["cpu_model"] = m.group(1).strip()
                    break
    except (FileNotFoundError, PermissionError):
        hw["cpu_model"] = "unknown (non-Linux or no /proc/cpuinfo)"
    # Core count
    hw["cpu_count_logical"] = os.cpu_count()
    # RAM (Linux: /proc/meminfo)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                m = re.match(r"MemTotal:\s*(\d+)\s*kB", line)
                if m:
                    hw["ram_mb"] = int(m.group(1)) // 1024
                    break
    except (FileNotFoundError, PermissionError):
        hw["ram_mb"] = None
    return hw


def run_with_env(model, prompt, n_tokens, threads, env_extra, run_inference_py):
    """Run run_inference.py with extra env vars; return tok/s or None."""
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [
        sys.executable, run_inference_py,
        "-m", model, "-p", prompt, "-n", str(n_tokens), "-t", str(threads),
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    if result.returncode != 0:
        return None, f"exit={result.returncode}"
    text = (result.stdout.decode("utf-8", errors="replace") + "\n" +
            result.stderr.decode("utf-8", errors="replace"))
    matches = re.findall(r"(\d+[.,]\d+)\s*tokens per second", text)
    if matches:
        return float(matches[-1].replace(",", ".")), None
    return None, "no t/s in output"


CONFIGURATIONS = [
    ("L1_baseline_I2S_GEMV",      "L1 baseline (I2_S GEMV)",        {}),
    ("L3_ACDC_FFN",               "L3 ACDC FFN (env BITNET_ACDC_FFN=1)", {"BITNET_ACDC_FFN": "1"}),
    ("L4_Tropical_topK_32",       "L4 Tropical top-K=32 (env BITNET_TROPICAL_TOPK=32)",
                                   {"BITNET_TROPICAL_TOPK": "32"}),
    ("L4_SparseFloat_topK_32",    "L4 Sparse float top-K=32 (env BITNET_SPARSE_TOPK=32)",
                                   {"BITNET_SPARSE_TOPK": "32"}),
    ("L5_HRR_raw",                "L5 HRR raw (env BITNET_HRR_ATTN=1)",
                                   {"BITNET_HRR_ATTN": "1", "BITNET_HRR_ATTN_CLEANUP": "0"}),
    ("L5_HRR_cleanup_8",          "L5 HRR + cleanup 8 (env BITNET_HRR_ATTN=1, CLEANUP=8)",
                                   {"BITNET_HRR_ATTN": "1", "BITNET_HRR_ATTN_CLEANUP": "8"}),
]


def run_bench(model, prompt, n_tokens, threads, keep_running=False):
    """Run the full benchmark suite. Return list of dicts (one per config)."""
    run_inference_py = str(Path(__file__).parent.parent / "run_inference.py")
    if not os.path.exists(run_inference_py):
        raise FileNotFoundError(f"{run_inference_py} not found")

    rows = []
    for slug, name, env_extra in CONFIGURATIONS:
        toks, err = run_with_env(model, prompt, n_tokens, threads,
                                 env_extra, run_inference_py)
        if toks is None:
            status = err or "no parse"
            if not keep_running:
                rows.append({"id": slug, "name": name, "tok_per_sec": None,
                             "status": status, "env": env_extra})
                return rows
        else:
            status = "ok"
        rows.append({"id": slug, "name": name, "tok_per_sec": toks,
                     "status": status, "env": env_extra})
    return rows


def emit_json(model, prompt, n_tokens, threads, rows, out_path):
    """Emit canonical JSON to out_path. Returns the dict for chaining."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "methodology": {
            "tool": "utils/cpu_universal_benchmark.py (and bench_publish.py wrapper)",
            "model": model,
            "prompt": prompt,
            "n_tokens": n_tokens,
            "threads": threads,
            "configurations": [
                {"id": s, "name": n, "env": e} for s, n, e in CONFIGURATIONS
            ],
            "notes": [
                "All numbers are tok/s on a single CPU (no GPU offload).",
                "L2 WHT is patched in vec_dot (always on); L1 baseline includes it.",
                "L3/L5 may produce garbage output because BitNet-2B was not trained",
                "with those architectures (P6 — estrutura, não compressão).",
                "Numbers reflect kernel overhead only, not output quality.",
            ],
        },
        "hardware": detect_hardware(),
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=False)
        f.write("\n")
    return data


def render_markdown(data, out_path):
    """Render Markdown report from canonical JSON. Markdown is generated, never edited."""
    md = []
    md.append(f"# Benchmarks v{data['schema_version']}\n")
    md.append(f"**Gerado em:** {data['timestamp_utc']}\n")
    md.append("## Hardware\n")
    hw = data["hardware"]
    md.append(f"- **CPU:** {hw.get('cpu_model', 'unknown')}")
    md.append(f"- **Cores (lógicos):** {hw.get('cpu_count_logical', 'unknown')}")
    md.append(f"- **RAM:** {hw.get('ram_mb', 'unknown')} MB" if hw.get("ram_mb") else "- **RAM:** unknown")
    md.append(f"- **Platform:** {hw.get('platform', 'unknown')}")
    md.append(f"- **Python:** {hw.get('python_version', 'unknown')}\n")

    md.append("## Methodology\n")
    m = data["methodology"]
    md.append(f"- **Modelo:** `{m['model']}`")
    md.append(f"- **Prompt:** `{m['prompt']}`")
    md.append(f"- **Tokens gerados:** {m['n_tokens']}")
    md.append(f"- **Threads:** {m['threads']}")
    md.append("- **Métrica:** tokens/second (wall-clock do `llama-cli`)")
    md.append("- **Configurações:** 6 (L1 baseline + 5 kernels algébricos)")
    md.append("")
    for note in m.get("notes", []):
        md.append(f"> {note}")
    md.append("")

    md.append("## Resultados\n")
    md.append("| Configuração | tok/s | Δ vs L1 | Status | Env |")
    md.append("|--------------|------:|--------:|--------|-----|")
    base = next((r["tok_per_sec"] for r in data["rows"]
                 if r["id"] == "L1_baseline_I2S_GEMV"), None)
    for r in data["rows"]:
        if r["tok_per_sec"] is None:
            md.append(f"| {r['name']} | — | — | {r['status']} | `{r['env']}` |")
        else:
            if base and base > 0:
                pct = 100.0 * r["tok_per_sec"] / base - 100.0
                sign = "+" if pct >= 0 else ""
                delta = f"{sign}{pct:.1f}%"
            else:
                delta = "—"
            md.append(f"| {r['name']} | {r['tok_per_sec']:.2f} | {delta} | {r['status']} | `{r['env']}` |")
    md.append("")

    md.append("## Anotações\n")
    md.append("- **L1 baseline** é o comportamento padrão (atenção densa, GEMM I2_S).")
    md.append("- **L4 sparse float** é opt-in (D1, AC-06); usuário assume risco.")
    md.append("- **L3 ACDC FFN** e **L5 HRR** são arquiteturas de treinamento (P6);")
    md.append("  com BitNet-2B (não treinado com ACDC/HRR) o output é garbage —")
    md.append("  números acima medem só overhead, não qualidade.")
    md.append("- Veja `ROADMAP.md#2` para a reserva técnica (Q4 2029) que reativaria")
    md.append("  o scaffolding de fine-tuning ACDC.\n")

    md.append("---\n")
    md.append(f"*Gerado por `utils/bench_publish.py` v{SCHEMA_VERSION} em "
              f"{data['timestamp_utc']} a partir de JSON canônico. "
              f"Não edite este Markdown manualmente.*\n")

    with open(out_path, "w") as f:
        f.write("\n".join(md))
    return data


def render_markdown_to_stdout(data):
    """Print Markdown to stdout (for piping)."""
    import io
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    try:
        render_markdown(data, "/dev/null")
    finally:
        sys.stdout = old
    # Re-render: redirect to stdout directly
    md = []
    md.append(f"# Benchmarks v{data['schema_version']}\n")
    md.append(f"**Gerado em:** {data['timestamp_utc']}\n")
    md.append("## Hardware\n")
    hw = data["hardware"]
    md.append(f"- **CPU:** {hw.get('cpu_model', 'unknown')}")
    md.append(f"- **Cores (lógicos):** {hw.get('cpu_count_logical', 'unknown')}")
    md.append(f"- **RAM:** {hw.get('ram_mb', 'unknown')} MB" if hw.get("ram_mb") else "- **RAM:** unknown")
    md.append(f"- **Platform:** {hw.get('platform', 'unknown')}")
    md.append(f"- **Python:** {hw.get('python_version', 'unknown')}\n")
    md.append("## Methodology\n")
    m = data["methodology"]
    md.append(f"- **Modelo:** `{m['model']}`")
    md.append(f"- **Prompt:** `{m['prompt']}`")
    md.append(f"- **Tokens gerados:** {m['n_tokens']}")
    md.append(f"- **Threads:** {m['threads']}\n")
    md.append("## Resultados\n")
    md.append("| Configuração | tok/s | Δ vs L1 | Status | Env |")
    md.append("|--------------|------:|--------:|--------|-----|")
    base = next((r["tok_per_sec"] for r in data["rows"]
                 if r["id"] == "L1_baseline_I2S_GEMV"), None)
    for r in data["rows"]:
        if r["tok_per_sec"] is None:
            md.append(f"| {r['name']} | — | — | {r['status']} | `{r['env']}` |")
        else:
            if base and base > 0:
                pct = 100.0 * r["tok_per_sec"] / base - 100.0
                sign = "+" if pct >= 0 else ""
                delta = f"{sign}{pct:.1f}%"
            else:
                delta = "—"
            md.append(f"| {r['name']} | {r['tok_per_sec']:.2f} | {delta} | {r['status']} | `{r['env']}` |")
    md.append("")
    print("\n".join(md))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-m", "--model", help="path to .gguf model (Mode 1)")
    p.add_argument("-p", "--prompt", default="The capital of France is",
                   help="prompt (default: %(default)s)")
    p.add_argument("-n", "--n-tokens", type=int, default=64,
                   help="tokens to generate (default: %(default)s)")
    p.add_argument("-t", "--threads", type=int, default=4,
                   help="threads (default: %(default)s)")
    p.add_argument("--keep-running", action="store_true",
                   help="continue even if a config fails")
    p.add_argument("--json", metavar="FILE",
                   help="Mode 1: run bench and write JSON to FILE")
    p.add_argument("--md", metavar="FILE",
                   help="Mode 2: render Markdown to FILE (or stdout if '-')")
    p.add_argument("--from-json", metavar="FILE",
                   help="Mode 2: read JSON from FILE instead of running bench")
    args = p.parse_args()

    if args.from_json:
        # Mode 2: render Markdown from existing JSON
        with open(args.from_json) as f:
            data = json.load(f)
        if args.md and args.md != "-":
            render_markdown(data, args.md)
            print(f"Markdown written to {args.md}", file=sys.stderr)
        else:
            render_markdown_to_stdout(data)
    elif args.json:
        # Mode 1: run bench, emit JSON
        if not args.model:
            p.error("Mode 1 (--json) requires -m/--model")
        rows = run_bench(args.model, args.prompt, args.n_tokens, args.threads,
                         keep_running=args.keep_running)
        data = emit_json(args.model, args.prompt, args.n_tokens, args.threads,
                         rows, args.json)
        print(f"JSON written to {args.json}", file=sys.stderr)
        if args.md and args.md != "-":
            render_markdown(data, args.md)
            print(f"Markdown written to {args.md}", file=sys.stderr)
    else:
        p.error("Specify --json FILE (Mode 1) or --from-json FILE (Mode 2)")


if __name__ == "__main__":
    main()
