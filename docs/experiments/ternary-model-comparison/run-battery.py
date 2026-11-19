#!/usr/bin/env python3
"""Compare two ternary models. Save generated code for review; never execute it."""
import argparse
import datetime
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
ENGINE = ROOT / 'build/bin/llama-cli'
MODELS = {
    'falcon3-1b': ROOT / 'models/Falcon3-1B-Instruct-1.58bit/ggml-model-i2_s.gguf',
        'bitnet2b': ROOT / 'models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf',
    'falcon3': ROOT / 'models/Falcon3-7B-Instruct-1.58bit/ggml-model-i2_s.gguf',
}
TESTS = {
    'explain': ('Explain how a neural network works in exactly three sentences.', 192),
    'factorial': ('Write a complete standalone Python program that calculates 10 factorial using a loop and asserts the result equals 3628800. Output one Python code block only.', 512),
    'pi': ('Write a complete standalone Python app that calculates pi to 100 decimal places using decimal arithmetic and independently verifies the result. Output code only.', 2048),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', choices=TESTS)
parser.add_argument('--model', choices=MODELS)
parser.add_argument('--temp', type=float, default=0.0)
args = parser.parse_args()
out = Path(__file__).resolve().parent / ('battery-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
out.mkdir()
print('Results:', out, flush=True)
print('Both models use the same CPU engine, 8 generation threads, 4 prompt threads, 4096 context.', flush=True)
env = os.environ.copy()
for key in ('BITNET_ENABLE_ROPE_CACHE', 'BITNET_ENABLE_GROUP_QKV'):
    env.pop(key, None)
for name, model in MODELS.items():
    if args.model and name != args.model:
        continue
    for test, (prompt, count) in TESTS.items():
        if args.only and test != args.only:
            continue
        print('\n=== ' + name + ' / ' + test + ' ===', flush=True)
        cmd = [str(ENGINE), '-m', str(model), '-p', prompt, '-n', str(count),
               '-t', '8', '-tb', '4', '-c', '4096', '--temp', str(args.temp),
               '--seed', '42', '--no-display-prompt', '--single-turn']
        stem = out / (name + '-' + test)
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, env=env, timeout=600)
        except subprocess.TimeoutExpired as error:
            data = error.stdout or b''
            if isinstance(data, bytes):
                data = data.decode(errors='replace')
            stem.with_suffix('.log').write_text(data)
            print('Timed out after 10 minutes; partial output saved.', flush=True)
            continue
        stem.with_suffix('.log').write_text(result.stdout)
        print(result.stdout, flush=True)
        if result.returncode:
            print('Engine exit code:', result.returncode, flush=True)
        blocks = re.findall(r'```(?:python)?\s*\n(.*?)```', result.stdout, re.S)
        if blocks and test != 'explain':
            stem.with_suffix('.py').write_text(blocks[0])
            print('Saved code for review; it has NOT been executed:', stem.with_suffix('.py'), flush=True)
print('\nDone. Inspect pi code: it must calculate pi, print 100 decimal places, and independently verify it. A long literal alone fails. Speed is reported in each log.', flush=True)
