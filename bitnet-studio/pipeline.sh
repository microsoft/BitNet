#!/bin/bash
# Pipeline completo: treino → merge → quantize → teste

set -e

echo "=== BitNet Studio — Pipeline de Fine-tune ==="
echo ""

# 1. Treino
if [ -f "data/ptbr_tools_train.jsonl" ]; then
    echo "[1/4] Iniciando fine-tune..."
    python3 finetune_cpu.py
else
    echo "[1/4] Dataset não encontrado!"
    exit 1
fi

# 2. Merge
if [ -d "adapters/f3b-ptbr-tools-cpu" ]; then
    echo "[2/4] Fazendo merge do adapter..."
    python3 merge_and_quantize.py
else
    echo "[2/4] Adapter não encontrado!"
    exit 1
fi

# 3. Quantize (se llama.cpp disponível)
if [ -f "../build/bin/llama-quantize" ]; then
    echo "[3/4] Quantizando para Q4_K_M..."
    # Converter HF para GGUF primeiro
    echo "Conversão requer python3 convert_hf_to_gguf.py"
else
    echo "[3/4] llama-quantize não encontrado, pulando quantização"
fi

# 4. Teste
echo "[4/4] Testando tool-calling..."
python3 test_tool_calling.py

echo ""
echo "=== Pipeline completo! ==="
