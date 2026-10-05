#!/usr/bin/env python3
"""
tropical_sweep.py — Characterize L4 Tropical attention throughput vs K and context length.

Hypothesis: tropical attention is faster than standard only when K < n_kv (actual
key filtering occurs). When K >= n_kv the scoring still runs but no keys are dropped,
so the ternary-quantization overhead dominates.

The sweep varies:
  - BITNET_TROPICAL_TOPK  : 0 (=standard), 4, 8, 16, 32, 64, 128, 256
  - prompt length          : short (1 tok), medium (6 tok), long (≈50 tok)

For each cell, reports tok/s and delta vs K=0 (standard).

Usage:
  python utils/tropical_sweep.py \\
    -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \\
    -n 64 -t 4

Notes:
  - n_kv at decode step i = (prompt_tokens + i).  Mid-decode n_kv ≈ n_prompt + n/2.
  - All runs use the same -n tokens so total wallclock is proportional.
  - K=0 disables tropical and uses the standard flash_attn path (baseline).
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


# Short prompts to control expected n_kv range during decode
PROMPT_CONFIGS = [
    ("ctx≈1-n",  "Hi"),                                                       # ~1 prompt tok
    ("ctx≈6-n",  "The capital of France is"),                                  # ~6 prompt tok
    ("ctx≈50-n", "In mathematics, the Walsh-Hadamard transform is a generalization "
                 "of the Fourier transform to functions over binary vectors. It "
                 "decomposes a function into a sum of Walsh functions. The key"),  # ~50 prompt tok
]

K_VALUES = [0, 4, 8, 16, 32, 64, 128, 256]


def run_one(model, prompt, n_tokens, threads, k_val, run_inference, timeout=300):
    env = os.environ.copy()
    if k_val > 0:
        env["BITNET_TROPICAL_TOPK"] = str(k_val)
    else:
        env.pop("BITNET_TROPICAL_TOPK", None)

    cmd = [sys.executable, run_inference,
           "-m", model, "-p", prompt, "-n", str(n_tokens), "-t", str(threads)]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None

    text = r.stdout.decode("utf-8", errors="replace") + "\n" + \
           r.stderr.decode("utf-8", errors="replace")
    matches = re.findall(r"(\d+[.,]\d+)\s*tokens per second", text)
    if matches:
        return float(matches[-1].replace(",", "."))
    return None


def estimate_prompt_tokens(prompt):
    """Very rough: split on spaces, add 1 for BOS."""
    return len(prompt.split()) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-n", "--n-tokens", type=int, default=64)
    parser.add_argument("-t", "--threads",  type=int, default=4)
    parser.add_argument("--k-values", nargs="+", type=int, default=K_VALUES,
                        help="K values to sweep (0 = standard baseline)")
    args = parser.parse_args()

    run_inference = str(Path(__file__).parent.parent / "run_inference.py")
    if not os.path.exists(run_inference):
        sys.exit(f"ERROR: {run_inference} not found")

    print(f"Tropical sweep — model: {args.model}")
    print(f"  n_tokens={args.n_tokens}  threads={args.threads}")
    print()

    for prompt_label, prompt in PROMPT_CONFIGS:
        n_prompt = estimate_prompt_tokens(prompt)
        mid_nkv  = n_prompt + args.n_tokens // 2
        print(f"── {prompt_label}  (prompt≈{n_prompt} tok, mid-decode n_kv≈{mid_nkv}) ──")
        print(f"  {'K':>6}  {'tok/s':>8}  {'Δ vs K=0':>10}  {'note'}")
        print(f"  {'─'*6}  {'─'*8}  {'─'*10}  {'─'*30}")

        baseline = None
        for k in args.k_values:
            tps = run_one(args.model, prompt, args.n_tokens, args.threads,
                          k, run_inference)
            if tps is None:
                print(f"  {k:>6}  {'—':>8}  {'—':>10}  FAILED")
                continue
            if k == 0:
                baseline = tps
                print(f"  {k:>6}  {tps:>8.2f}  {'baseline':>10}")
            else:
                delta_pct = 100.0 * (tps - baseline) / baseline if baseline else float("nan")
                filtering = k < mid_nkv
                note = f"filters ({k}/{mid_nkv} keys)" if filtering else f"no-filter ({k}>={mid_nkv})"
                sign = "+" if delta_pct >= 0 else ""
                print(f"  {k:>6}  {tps:>8.2f}  {sign}{delta_pct:>+8.1f}%  {note}")
        print()

    print("Done.")
    print()
    print("Key insight to look for:")
    print("  - When K < mid_nkv (filtering): tropical should approach speedup")
    print("  - When K >= mid_nkv (no filtering): tropical slower due to quant overhead")
    print("  - Crossover K value identifies the optimal operating point")


if __name__ == "__main__":
    main()
