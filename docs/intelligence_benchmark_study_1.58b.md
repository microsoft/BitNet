# BitNet 1.58b Intelligence Benchmark: Technical Study & Post-Mortem

**Date:** 2026-03-31
**Environment:** Sovereign Pod / MSI Titanium (Remote) -> NVIDIA RTX 5070 (8GB VRAM) / Ryzen AI (50 TOPS)
**Framework:** `bitnet.cpp` (LLaMA.cpp fork optimized for Ternary Weights `i2_s`)

## 1. Executive Summary
This technical documentation summarizes the isolation, compilation, and evaluation of 1.58-bit ternary Large Language Models (LLMs) executing natively on a local workstation. The core objective was to validate intelligence capabilities (Zero-Shot reasoning, Structured JSON extraction, and Code Generation) to select the optimal cognition engine for the `red-pill` multi-agent architecture without polluting the core database.

## 2. Infrastructure & Compilation Pipeline
The models were processed and executed within an explicitly isolated minion script (`minion_benchmark.py`) under `sharing/experimental/BitNet/`.

### 2.1 Weight Conversion (`setup_env.py`)
HuggingFace `safetensors` were locally transpiled and quantized into the `i2_s` GGUF schema:
- **Baseline:** `microsoft/BitNet-b1.58-2B-4T`
- **Candidate A:** `HF1BitLLM/Llama3-8B-1.58-100B-tokens` (Base)
- **Candidate B:** `tiiuae/Falcon3-10B-Instruct-1.58bit` (Instruct-Tuned)

*Conversion Overhead:* The 8B/10B models peaked at ~16GB RAM during `fp16` translation before folding into `i2_s` (occupying ~3.8GB on disk, allowing complete offload into the RTX 5070's 8GB VRAM envelope).

### 2.2 Execution Engine (`llama-cli`)
Inference was orchestrated using the `bitnet.cpp` static binary.
**Parameters:** `-c 1024` (Context Width), `-n 256` (Max Tokens), `-t 8` (CPU Threads), `--temp 0.2` (Low creativity).

## 3. The Evaluation Matrix (Disciplines)
The Minion subjected each model to three single-turn (Zero-Shot) deterministic prompts:
1. **Razonamiento Lógico:** *Chain-of-thought* math puzzle regarding apples. Expected behavioral output: Numerical resolution "1".
2. **Extracción Estructurada JSON:** Natural language to JSON parsing. Expected behavioral output: Strict `{"name": "...", "age": ...}` schema without markdown dialogue.
3. **Generación Python:** Implementation of `reverse_string`. Expected behavioral output: A pure function block.

## 4. Empirical Benchmark Results

| Model | Architecture Type | Logic Math | JSON Extraction | Python Generation | TCO (RAM Delta / Inference Speed) | Final Score |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **BitNet-2B-4T** | Base (Pre-Trained) | 0/100 | 0/100 | 60/100 | ~118 MB / Fast | **20/100** (Fail) |
| **Llama3-8B-1.58** | Base (Pre-Trained) | 0/100 | 0/100 | 70/100 | ~300 MB / Med | **23/100** (Fail) |
| **Falcon3-10B** | Instruct (Post-Trained) | 95/100 | 100/100 | 100/100 | ~264 MB / Med | **98/100** (Pass) |

### 4.1 Observations on "Base" vs "Instruct" Topologies
The empirical failure of LLaMA3-8B and 2B-4T is strictly architectural. Base models operate as probabilistic next-token predictors. Without *Instruction Tuning*, they fail to recognize the boundary between the semantic prompt and their response, leading to severe zero-shot hallucinations (e.g., LLaMA-3 generating news about "a new president", 2B-4T generating a recipe).

In contrast, **Falcon3-10B-Instruct** contains the necessary alignment tokens to halt generation cleanly (`<|eot_id|>`) and adhere to strict schemas, achieving perfect JSON extraction.

## 5. Architectural Conclusion
**Falcon3-10B-Instruct-1.58bit** is formally certified for ingestion into the `red-pill` production standard. Its zero-shot cognitive fidelity perfectly matches our operational requirements (Sovereignty, Speed, and Structured Output), comfortably residing within the 8GB VRAM hardware constraint while freeing up system RAM entirely.
