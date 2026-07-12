#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_bitnet.sh [MODEL_PATH] [PROMPT] [THREADS] [TOKENS]
# Defaults: MODEL_PATH=models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf}"
PROMPT="${2:-You are a helpful assistant. Say hello.}"
THREADS="${3:-4}"
TOKENS="${4:-128}"

CLI_PATH="${ROOT_DIR}/build/bin/llama-cli"

if [ ! -x "$CLI_PATH" ]; then
  echo "Error: llama-cli not found or not executable at $CLI_PATH"
  exit 1
fi

MODEL_FULLPATH="$ROOT_DIR/$MODEL_PATH"
if [ ! -f "$MODEL_FULLPATH" ]; then
  echo "Error: model file not found: $MODEL_FULLPATH"
  exit 1
fi

echo "Running BitNet inference"
echo "Model: $MODEL_FULLPATH"
echo "Threads: $THREADS, Tokens: $TOKENS"
echo "Prompt: $PROMPT"

exec "$CLI_PATH" -m "$MODEL_FULLPATH" -p "$PROMPT" -t "$THREADS" -n "$TOKENS"
