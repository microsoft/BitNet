#!/usr/bin/env python3
"""
cpu_universal_benchmark.py — Systematic smoke benchmark of L1-L5 CPU kernels

Runs the same prompt/tokens/threads configuration with each kernel level
enabled (via env vars), measures tok/s, and emits a markdown table.

Unlike utils/e2e_benchmark.py (which uses llama-bench and only measures the
default L1 kernel), this script exercises the per-level dispatch paths:
  L1 baseline       : no env var (default I2_S GEMV)
  L3 ACDC FFN       : BITNET_ACDC_FFN=1
  L4 Tropical attn  : BITNET_TROPICAL_TOPK=32
  L4 Sparse float   : BITNET_SPARSE_TOPK=32   (single-pass float scoring, no int8 K buffer)
  L5 HRR raw        : BITNET_HRR_ATTN=1, BITNET_HRR_ATTN_CLEANUP=0
  L5 HRR + cleanup  : BITNET_HRR_ATTN=1, BITNET_HRR_ATTN_CLEANUP=8

L2 WHT is patched in vec_dot (always on); the L1 baseline already includes it.

Output is markdown table printed to stdout. With --csv FILE, also writes CSV.
With --keep-running, continues even if a configuration fails (e.g. output is
garbage, which is expected for L3/L5 because the model wasn't trained with
those architectures — see CLAUDE.md P6).

Usage:
  python utils/cpu_universal_benchmark.py \\
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \\
    -p "The capital of France is" -n 64 -t 4
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


def run_with_env(model, prompt, n_tokens, threads, env_extra, run_inference):
    """Run run_inference.py with extra env vars; return tok/s (or None)."""
    env = os.environ.copy()
    env.update(env_extra)
    cmd = [
        sys.executable, run_inference,
        "-m", model, "-p", prompt, "-n", str(n_tokens), "-t", str(threads),
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    if result.returncode != 0:
        return None, f"exit={result.returncode}"
    # Parse tok/s from llama.cpp output.  llama-cli prints:
    #   "eval time =    6202,83 ms /    31 runs   (  200,09 ms per token,     5,00 tokens per second)"
    # followed by:
    #   "       total time = ... (    4,89 tokens per second)"  <-- this is what we want
    # (note: European decimal comma on pt_BR locale).  We want the LAST
    # "tokens per second" in the output (that's the overall rate).
    # Use errors="replace" to handle non-UTF8 escape sequences from llama-cli.
    text = (result.stdout.decode("utf-8", errors="replace") + "\n" +
            result.stderr.decode("utf-8", errors="replace"))
    matches = re.findall(r"(\d+[.,]\d+)\s*tokens per second", text)
    if matches:
        # Last match is the overall rate
        return float(matches[-1].replace(",", ".")), None
    return None, "no t/s in output"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True, help="path to .gguf model")
    parser.add_argument("-p", "--prompt", default="The capital of France is",
                        help="prompt to feed (default: %(default)s)")
    parser.add_argument("-n", "--n-tokens", type=int, default=64,
                        help="number of tokens to generate (default: %(default)s)")
    parser.add_argument("-t", "--threads", type=int, default=4,
                        help="number of CPU threads (default: %(default)s)")
    parser.add_argument("--csv", help="also write CSV to this file")
    parser.add_argument("--keep-running", action="store_true",
                        help="continue even if a configuration fails")
    args = parser.parse_args()

    run_inference = str(Path(__file__).parent.parent / "run_inference.py")
    if not os.path.exists(run_inference):
        print(f"ERROR: {run_inference} not found", file=sys.stderr)
        sys.exit(1)

    configurations = [
        ("L1 baseline (I2_S GEMV)",        {}),
        ("L3 ACDC FFN square (BITNET_ACDC_FFN=1)",
                                          {"BITNET_ACDC_FFN": "1"}),
        ("L3 ACDC FFN rect (BITNET_ACDC_FFN_RECT=1)",
                                          {"BITNET_ACDC_FFN_RECT": "1"}),
        ("L3 ACDC FFN rect auto (BITNET_ACDC_FFN_RECT=auto)",
                                          {"BITNET_ACDC_FFN_RECT": "auto"}),
        ("L4 Tropical top-K=32 (BITNET_TROPICAL_TOPK=32)",
                                          {"BITNET_TROPICAL_TOPK": "32"}),
        ("L4 Sparse float top-K=32 (BITNET_SPARSE_TOPK=32)",
                                          {"BITNET_SPARSE_TOPK": "32"}),
        ("L4 Adaptive-K cov=0.90 kmax=32 (BITNET_SPARSE_TOPK_ADAPTIVE=0.90)",
                                          {"BITNET_SPARSE_TOPK_ADAPTIVE": "0.90"}),
        ("L4 Adaptive-K cov=0.99 kmax=32 (BITNET_SPARSE_TOPK_ADAPTIVE=0.99)",
                                          {"BITNET_SPARSE_TOPK_ADAPTIVE": "0.99"}),
        ("L5 HRR raw (BITNET_HRR_ATTN=1)",
                                          {"BITNET_HRR_ATTN": "1",
                                           "BITNET_HRR_ATTN_CLEANUP": "0"}),
        ("L5 HRR + cleanup 8 (BITNET_HRR_ATTN=1, CLEANUP=8)",
                                          {"BITNET_HRR_ATTN": "1",
                                           "BITNET_HRR_ATTN_CLEANUP": "8"}),
        ("L5 HRR phasor keys (BITNET_HRR_ATTN=1, PHASOR=1)",
                                          {"BITNET_HRR_ATTN": "1",
                                           "BITNET_HRR_PHASOR": "1"}),
    ]

    print(f"CPU-Universal smoke benchmark")
    print(f"  model:   {args.model}")
    print(f"  prompt:  {args.prompt!r}")
    print(f"  tokens:  {args.n_tokens}")
    print(f"  threads: {args.threads}")
    print()
    print(f"{'Configuration':<60} {'tok/s':>10}  {'status':<20}")
    print(f"{'-'*60} {'-'*10}  {'-'*20}")

    rows = []
    for name, env_extra in configurations:
        toks, err = run_with_env(args.model, args.prompt, args.n_tokens,
                                 args.threads, env_extra, run_inference)
        if toks is None:
            status = err or "no parse"
            toks_str = "—"
            if not args.keep_running:
                print(f"{name:<60} {toks_str:>10}  {status:<20}")
                print(f"\nAborted (use --keep-running to continue on failure).")
                sys.exit(1)
        else:
            status = "ok"
            toks_str = f"{toks:.2f}"
        print(f"{name:<60} {toks_str:>10}  {status:<20}")
        rows.append((name, toks, status))

    if not any(r[1] for r in rows):
        print("\nNo successful runs; nothing to compare.")
        sys.exit(1)

    base = rows[0][1]
    if base and base > 0:
        print()
        print(f"Relative to L1 baseline ({base:.2f} tok/s):")
        for name, t, status in rows:
            if t and t > 0:
                pct = 100.0 * t / base
                sign = "+" if pct >= 100 else ""
                print(f"  {name:<60} {sign}{pct-100:+.1f}%  ({t:.2f} tok/s)")
            else:
                print(f"  {name:<60} —     ({status})")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["configuration", "tok_per_sec", "status", "delta_vs_L1_pct"])
            base = rows[0][1] or 0
            for name, t, status in rows:
                pct = (100.0 * t / base - 100.0) if (t and base) else ""
                w.writerow([name, t or "", status, f"{pct:+.1f}" if pct != "" else ""])
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
