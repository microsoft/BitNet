# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Purpose

This is a fork of Microsoft's **bitnet.cpp** — a CPU-only inference framework for 1-bit LLMs (ternary weights {-1, 0, +1}, 1.58 bits/param). The GPU pipeline has been removed. The fork extends the project with a mathematical research roadmap aimed at universalizing LLMs on CPU through forgotten algebraic structures.

**Primary constraint**: CPU only. Never GPU. All new kernels must remain CPU-bound.

---

## Build and Setup

**Full setup** (download model + convert + codegen + compile):
```bash
conda activate bitnet-cpp
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s
# ARM64: use -q tl1 instead; x86_64: use -q tl2 for LUT kernels
```

**Manual cmake build** (after kernel headers are generated):
```bash
# Standard build (requires libstdc++-14-dev; or use the flags below)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

**Compiler requirement**: Clang ≥ 18 is required for SIMD kernels. GCC is tolerated but requires `-fpermissive`. Never use MSVC.

**Ubuntu 24.04 workaround** — Clang 18 defaults to GCC 14 headers; if only `libstdc++-13-dev` is installed (no `libstdc++-14-dev`), add these flags:
```bash
cmake -B build \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_FLAGS="-I/usr/include/c++/13 -I/usr/include/x86_64-linux-gnu/c++/13" \
  -DCMAKE_EXE_LINKER_FLAGS="-L/usr/lib/gcc/x86_64-linux-gnu/13" \
  -DCMAKE_BUILD_TYPE=Release
```

**Submodule**: `3rdparty/llama.cpp` (fork, branch `merge-dev`) is the inference backend. Initialize with `git submodule update --init --recursive`.

---

## Running Inference and Benchmarks

```bash
# CPU inference (hardcoded -ngl 0, -b 1)
python run_inference.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Your prompt" -n 200 -t 4

# Conversational mode
python run_inference.py -m models/.../ggml-model-i2_s.gguf -p "System prompt" -cnv

# End-to-end throughput benchmark
python utils/e2e_benchmark.py -m models/.../ggml-model-i2_s.gguf -n 128 -p 512 -t 4

# Perplexity evaluation
python utils/test_perplexity.py -m models/.../ggml-model-i2_s.gguf
```

**Math kernel benchmarks** (Level 2/3/4 research, no model required):
```bash
python utils/wht_benchmark.py                    # Level 2: WHT zero-multiplication
python utils/acdc_benchmark.py --n 512           # Level 3: FWHT+ACDC O(n log n)
python utils/acdc_benchmark.py --n 512 --scaling # show operation count scaling table
python utils/tropical_benchmark.py --n 256 --d 64 --k 16  # Level 4: tropical attention
python utils/tropical_benchmark.py --scaling     # show speedup vs seq_len table
```

---

## Kernel Architecture

There are three CPU kernel families, selected at build time:

| Format | Platform | Build flag | Generator |
|--------|----------|-----------|-----------|
| **I2_S** | x86_64 + ARM | default (no flag) | `src/ggml-bitnet-mad.cpp` |
| **TL1** | ARM64 only | `-DBITNET_ARM_TL1=ON` | `utils/codegen_tl1.py` |
| **TL2** | x86_64 only | `-DBITNET_X86_TL2=ON` | `utils/codegen_tl2.py` |

**I2_S encoding**: weights {-1→0, 0→1, +1→2}, packed 4 per byte. QK block size = 128 (x86) / 64 (ARM). Main SIMD path uses `_mm256_maddubs_epi16` (AVX2).

**TL1/TL2** are lookup-table kernels. The `.h` files in `preset_kernels/<model>/` are pre-generated for known models. For new models, run `utils/codegen_tl1.py` or `codegen_tl2.py` to regenerate, then recompile.

**Kernel performance tuning**: Edit `include/gemm-config.h` before building. Controls `ROW_BLOCK_SIZE`, `COL_BLOCK_SIZE`, `PARALLEL_SIZE`, and the `ACT_PARALLEL` mode (activation-parallel vs weight-parallel). Activation parallel (`ACT_PARALLEL` defined) is recommended for I2_S. Run `python utils/tune_gemm_config.py` to auto-tune for your hardware.

---

## Mathematical Research Extensions (this fork)

The fork adds experimental kernels under a 5-level algebraic roadmap:

| Level | Math | Files | Status |
|-------|------|-------|--------|
| 2 | WHT decomposition — zero multiplications | `src/ggml-bitnet-wht.cpp`, `include/ggml-bitnet-wht.h` | Done |
| 3 | FWHT + ACDC layer — O(n log n) GEMV | `src/ggml-bitnet-fwht.cpp`, `include/ggml-bitnet-fwht.h` | Done |
| 4 | Tropical attention — (max,+) semiring | `src/ggml-bitnet-tropical.cpp`, `include/ggml-bitnet-tropical.h` | Done |
| 5 | Holographic Reduced Representations (HRR) | `src/ggml-bitnet-hrr.cpp`, `include/ggml-bitnet-hrr.h` | Done |

Full mathematical theory: `docs/mathematical-foundations.md`.

**Critical ACDC invariant**: ACDC is not a post-hoc compression method. For random ternary W, ACDC projection captures only ~1/n energy. ACDC only achieves exact recovery when the model is *trained* with the ACDC architecture (d is the learned diagonal, optimized during training, not fitted afterward).

**Level 3 kernel**: `acdc_forward(x, d)` = H·(d⊙(H·x)), unnormalized — no 1/n² factors. The projection formula `acdc_project`: d* = diag(H·W·H) / n².

**Level 4 kernel**: `tropical_attention()` scans all keys with ternary dot products (zero multiplications), selects top-K, applies softmax only over K tokens. Complexity O(n·d + K·d) vs O(n²·d) standard attention.

These Level 2–5 kernels are **wired into CMakeLists.txt** as a `bitnet_math` OBJECT library (linked into the `ggml` target) via `-DBITNET_L2_WHT=ON -DBITNET_L3_ACDC=ON -DBITNET_L4_TROPICAL=ON -DBITNET_L5_HRR=ON`. The build is verified (all four `.cpp` files compile with AVX2 flags on x86_64). They are not yet connected to the **llama.cpp tensor dispatch path** (that integration is the next step).

**HRR operating regime** (critical): retrieval quality requires d ≥ 10·N (d = head_dim, N = context tokens). At d=64, N=32 → capacity limit, noisy retrieval (mathematically expected — see `docs/theory/05-holographic-memory.md`). For practical attention replacement: d ≥ 640 for N=64, or use phasor keys (exact inverse) instead of Gaussian random keys.

---

## Model Conversion

```bash
# From HuggingFace GGUF (pre-quantized)
huggingface-cli download microsoft/BitNet-b1.58-2B-4T-gguf --local-dir models/BitNet-b1.58-2B-4T
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s

# From safetensors (bf16 checkpoint)
huggingface-cli download microsoft/bitnet-b1.58-2B-4T-bf16 --local-dir ./models/bitnet-b1.58-2B-4T-bf16
python utils/convert-helper-bitnet.py ./models/bitnet-b1.58-2B-4T-bf16

# With embedding quantization (Q6_K format, recommended for speed+quality tradeoff)
python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s --quant-embd
```

Conversion pipeline: safetensors → `convert-helper-bitnet.py` → `ggml-model-f32.gguf` → `llama-quantize` → `ggml-model-i2_s.gguf`.

---

## Repository Conventions

- `_reversa_sdd/` — Reversa framework analysis artifacts. **Never modify these files.**
- `.reversa/` — Reversa working directory. **Never modify these files.**
- `preset_kernels/` — Pre-tuned kernel configs for known models. Only regenerate via codegen scripts.
- The `3rdparty/llama.cpp` submodule is a fork (not upstream). Treat it as read-only unless deliberately patching the backend.
- `run_inference.py` hardcodes `-ngl 0` (no GPU offload) and `-b 1` (decode batch size 1). This is intentional — CPU-only decode mode.

---

## Remotes

- `origin` → `https://github.com/peder1981/BitNet.git` (this fork)
- `upstream` → `https://github.com/microsoft/BitNet.git`
