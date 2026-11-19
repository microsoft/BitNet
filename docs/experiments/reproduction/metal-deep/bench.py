import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--repeats', type=int, default=5)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--suite', choices=('tied', 'head', 'full'), default='tied')
parser.add_argument('--head-sampler', action='store_true')
args = parser.parse_args()
base = Path('/Users/antoniociolino/bitnet/BitNet')
scratch = Path(__file__).parent.resolve()
cpu_binary = base / 'build/bin/llama-cli'
metal_binary = scratch / 'build/bin/llama-cli'
model = base / 'models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf'
no_other_offload = ['-ngl', '0', '--no-kv-offload', '--no-op-offload']
variants = {
    'existing_cpu': [str(cpu_binary), '--device', 'none', *no_other_offload],
    'metal_binary_cpu': [str(metal_binary), '--device', 'none', *no_other_offload],
    'metal_f16': [str(metal_binary), '--device', 'MTL0', *no_other_offload,
                  '-ot', r'^token_embd\.weight$=MTL0'],
}
if args.suite == 'head':
    head = str(scratch / 'llama-cli-head-only')
    variants = {
        'head_binary_cpu': [head, '--device', 'none', *no_other_offload],
        'metal_head_only': [head, '--device', 'MTL0', '-ngl', '1', '--no-kv-offload', '--no-op-offload'],
    }
    if args.head_sampler:
        variants = {name: command + ['--samplers', 'temperature'] for name, command in variants.items()}
        variants['metal_head_backend_sampling'] = [*variants['metal_head_only'], '--backend-sampling']
if args.suite == 'full':
    full = str(scratch.parent / 'full-metal/llama-cli-full-metal')
    head = str(scratch / 'llama-cli-head-only')
    variants = {
        'cpu': [str(cpu_binary), '--device', 'none', *no_other_offload, '--samplers', 'temperature'],
        'hybrid': [head, '--device', 'MTL0', '-ngl', '1', '--no-kv-offload', '--no-op-offload', '--backend-sampling', '--samplers', 'temperature'],
        'full_gpu': [full, '--device', 'MTL0', '-ngl', 'all', '--no-op-offload', '-ot', r'^token_embd\.weight$=MTL0', '--backend-sampling', '--samplers', 'temperature'],
    }
passage = ('Training adjusts the parameters inside a neural network. '
           'Different layers transform inputs into progressively more useful representations. ')
cases = {
    'short': 'Explain how a neural network works in three sentences.',
    'long': passage * 24 + 'Summarize that passage in three sentences.',
}
settings = ['-m', str(model), '-n', '64', '-t', '8', '-tb', '4', '-c', '2048',
            '--temp', '0', '--seed', '42', '--ignore-eos', '--single-turn', '--simple-io',
            '--no-warmup', '--fit', 'off']
wrapper = ("import resource, subprocess, sys; "
           "p = subprocess.run(sys.argv[1:]); "
           "r = resource.getrusage(resource.RUSAGE_CHILDREN); "
           "print('__BENCH_RSS__', r.ru_maxrss, file=sys.stderr); "
           "sys.exit(p.returncode)")
rows = []
names = list(variants)
clean_env = dict(os.environ)
for name in ('GGML_SCHED_DEBUG', 'GGML_OP_OFFLOAD_MIN_BATCH', 'GGML_METAL_DEVICES'):
    clean_env.pop(name, None)

def run(case, name, repeat):
    started = time.monotonic()
    proc = subprocess.run([sys.executable, '-c', wrapper, *variants[name], *settings,
                           '-p', cases[case]], cwd=base, capture_output=True, text=True,
                          timeout=120, env=clean_env)
    elapsed = time.monotonic() - started
    match = re.search(r'Prompt: ([0-9.]+) t/s \| Generation: ([0-9.]+)', proc.stdout)
    if proc.returncode or not match:
        (scratch / ('failed-' + name + '.log')).write_text(proc.stdout + '\n' + proc.stderr)
        raise RuntimeError(f'{name}, {case}, returncode={proc.returncode}, stderr={proc.stderr[-1000:]}')
    response = proc.stdout.split('\n> ', 1)[-1].split('\n\n', 1)[-1].split('[ Prompt:', 1)[0].strip()
    rss = re.search(r'__BENCH_RSS__ (\d+)', proc.stderr)
    return dict(case=case, variant=name, repeat=repeat, prompt_tps=float(match[1]),
                generation_tps=float(match[2]), wall_seconds=elapsed,
                peak_resident_bytes=int(rss[1]) if rss else None, response=response,
                response_sha256=hashlib.sha256(response.encode()).hexdigest())

for case in cases:
    for i in range(args.repeats):
        order = [names[(i + j) % len(names)] for j in range(len(names))]
        if i % 2:
            order.reverse()
        for name in order:
            row = run(case, name, i + 1)
            rows.append(row)
            print({k: v for k, v in row.items() if not k.startswith('response')}, flush=True)
            args.output.write_text(json.dumps(dict(complete=False, results=rows), indent=2) + '\n')

summary = {}
for case in cases:
    group = [row for row in rows if row['case'] == case]
    summary[case] = {
        'throughput': {
            name: {metric: statistics.median(row[metric] for row in group if row['variant'] == name)
                   for metric in ['prompt_tps', 'generation_tps', 'wall_seconds', 'peak_resident_bytes']}
            for name in variants
        },
        'response_hashes': {name: sorted({row['response_sha256'] for row in group if row['variant'] == name})
                            for name in variants},
    }
report = dict(complete=True, hardware='Apple M2 Pro, 16 GB', settings=settings[2:],
              variants=variants, results=rows, summary=summary)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(summary, indent=2), flush=True)
