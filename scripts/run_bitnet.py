#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_CLI = os.path.join(ROOT, "build", "bin", "llama-cli")


def main():
    p = argparse.ArgumentParser(description="Run BitNet model via llama-cli")
    p.add_argument("--model", "-m", default="models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
    p.add_argument("--prompt", "-p", default="You are a helpful assistant. Say hello.")
    p.add_argument("--threads", "-t", type=int, default=4)
    p.add_argument("--tokens", "-n", type=int, default=128)
    args = p.parse_args()

    model_path = os.path.join(ROOT, args.model) if not os.path.isabs(args.model) else args.model
    if not os.path.isfile(LLAMA_CLI):
        print(f"Error: llama-cli not found at {LLAMA_CLI}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    cmd = [LLAMA_CLI, "-m", model_path, "-p", args.prompt, "-t", str(args.threads), "-n", str(args.tokens)]
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    main()
