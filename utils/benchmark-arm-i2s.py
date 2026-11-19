"""Alternate saved-baseline and candidate runs; write reproducible JSON results."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--baseline', required=True)
parser.add_argument('--candidate', default='build/bin/llama-cli')
parser.add_argument('--model', default='models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf')
parser.add_argument('--output', required=True)
parser.add_argument('--repeats', type=int, default=3)
args = parser.parse_args()
paragraph = 'Neural networks use layers of interconnected units. Training adjusts their weights using examples and feedback. '
cases = {
    'short': 'Explain how a neural network works in three sentences.',
    'long': paragraph * 24 + 'Summarize the passage in three sentences.',
}
rows = []
for case, prompt in cases.items():
    for repeat in range(args.repeats):
        variants = [('baseline', args.baseline), ('candidate', args.candidate)]
        if repeat % 2:
            variants.reverse()
        for label, executable in variants:
            command = [str(Path(executable).resolve()), '-m', args.model, '-p', prompt,
                       '-n', '128', '-t', '8', '-tb', '4', '-c', '2048',
                       '--single-turn', '--simple-io', '--temp', '0', '--no-warmup']
            # A fresh wrapper measures this child alone, without cumulative RSS.
            wrapper = ("import resource, subprocess, sys; "
                       "p = subprocess.run(sys.argv[1:]); "
                       "r = resource.getrusage(resource.RUSAGE_CHILDREN); "
                       "print('__BENCH_RSS__', r.ru_maxrss, file=sys.stderr); "
                       "sys.exit(p.returncode)")
            command = [sys.executable, '-c', wrapper] + command
            started = time.monotonic()
            result = subprocess.run(command, text=True, capture_output=True, timeout=120, check=True)
            elapsed = time.monotonic() - started
            resident = re.search(r'__BENCH_RSS__ (\d+)', result.stderr)
            match = re.search(r'Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+)', result.stdout)
            if not match:
                raise RuntimeError(result.stderr[-2000:])
            response = result.stdout.split('\n> ', 1)[-1].split('\n\n', 1)[-1].split('[ Prompt:', 1)[0].strip()
            row = dict(case=case, repeat=repeat + 1, variant=label,
                       prompt_tps=float(match[1]), generation_tps=float(match[2]),
                       wall_seconds=elapsed,
                       peak_resident_bytes=int(resident[1]) * (1 if sys.platform == 'darwin' else 1024) if resident else None,
                       response_sha256=hashlib.sha256(response.encode()).hexdigest(), response=response)
            rows.append(row)
            checkpoint = Path(args.output)
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_text(json.dumps(dict(results=rows, complete=False), indent=2) + '\n')
            print({k: v for k, v in row.items() if not k.startswith('response')}, flush=True)
summary = {}
for case in cases:
    summary[case] = {}
    for label in ['baseline', 'candidate']:
        subset = [row for row in rows if row['case'] == case and row['variant'] == label]
        summary[case][label] = {metric: statistics.median(row[metric] for row in subset)
                                for metric in ['prompt_tps', 'generation_tps']}
    summary[case]['identical_responses'] = len({row['response_sha256'] for row in rows if row['case'] == case}) == 1
output = Path(args.output)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(dict(settings=dict(threads=8, threads_batch=4, context=2048,
                    max_output_tokens=128, temperature=0), results=rows, summary=summary), indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
