"""Compare optional CPU graph optimizations using the same executable."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("docs/benchmarks/arm-rope-qkv"))
parser.add_argument("--repeats", type=int, default=5)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
args.output.mkdir(parents=True, exist_ok=True)
variants = {
    "baseline": {},
    "rope": {"BITNET_ENABLE_ROPE_CACHE": "1"},
    "qkv": {"BITNET_ENABLE_GROUP_QKV": "1"},
    "both": {"BITNET_ENABLE_ROPE_CACHE": "1", "BITNET_ENABLE_GROUP_QKV": "1"},
}
passage = "Neural networks use layers of interconnected units. Training adjusts their weights using examples and feedback. "
cases = {
    "short": "Explain how a neural network works in three sentences.",
    "long": passage * 24 + "Summarize the passage in three sentences.",
}
rows = []
for case, prompt in cases.items():
    for repeat in range(args.repeats):
        names = list(variants)
        offset = repeat % len(names)
        names = names[offset:] + names[:offset]
        for name in names:
            env = dict(os.environ)
            env.pop("BITNET_ENABLE_ROPE_CACHE", None)
            env.pop("BITNET_ENABLE_GROUP_QKV", None)
            env.update(variants[name])
            command = [str(root / "build/bin/llama-cli"), "-m",
                       str(root / "models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf"),
                       "-p", prompt, "-n", "96", "-t", "8", "-tb", "4", "-c", "2048",
                       "--single-turn", "--simple-io", "--temp", "0", "--no-warmup"]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                    text=True, check=True, timeout=90)
            match = re.search(r"Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+)", result.stdout)
            if not match:
                raise RuntimeError(result.stderr[-2000:])
            response = result.stdout.split("\n> ", 1)[-1].split("\n\n", 1)[-1].split("[ Prompt:", 1)[0].strip()
            row = dict(case=case, variant=name, repeat=repeat + 1,
                       prompt_tps=float(match[1]), generation_tps=float(match[2]),
                       response=response, hash=hashlib.sha256(response.encode()).hexdigest())
            rows.append(row)
            print({key: value for key, value in row.items() if key not in ("response", "hash")}, flush=True)
            (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
summary = {}
for case in cases:
    summary[case] = {
        name: {metric: statistics.median(row[metric] for row in rows
               if row["case"] == case and row["variant"] == name)
               for metric in ("prompt_tps", "generation_tps")}
        for name in variants
    }
    summary[case]["identical"] = len({row["hash"] for row in rows if row["case"] == case}) == 1
(args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
