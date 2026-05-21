# BitNet Embeddings (Qwen3) GGUF Conversion Implementation

## 1. Background

`bitnet-embeddings-0.6b` is a Qwen3-based embedding model with BitNet per-projection RMSNorm (`BitLinear`). Each linear projection (q/k/v/o/gate/up/down) has a `.norm.weight` that applies RMSNorm to the input **before** the matmul:

```
x → RMSNorm(x, norm.weight) → activation_quant(8bit) → matmul(weight_quant(ternary))
```

This pattern does **not** exist in any standard llama.cpp architecture:
- Standard Qwen3: no per-projection norms
- Standard BitNet: has `attn_sub_norm`/`ffn_sub_norm` at different positions (after attention/gate*up, not before each projection)

### Model Config

- Architecture: `Qwen3Model`
- hidden_size: 1024, num_attention_heads: 16, num_key_value_heads: 8
- head_dim: 128 (note: != hidden_size/num_heads = 64)
- intermediate_size: 3072, num_hidden_layers: 28
- tie_word_embeddings: true
- rope_theta: 1000000, rms_norm_eps: 1e-06

### Per-Layer Tensors (7 extra norm tensors per layer)

| Tensor | Shape |
|--------|-------|
| `self_attn.q_proj.norm.weight` | [1024] |
| `self_attn.k_proj.norm.weight` | [1024] |
| `self_attn.v_proj.norm.weight` | [1024] |
| `self_attn.o_proj.norm.weight` | [2048] |
| `mlp.gate_proj.norm.weight` | [1024] |
| `mlp.up_proj.norm.weight` | [1024] |
| `mlp.down_proj.norm.weight` | [3072] |

---

## 2. GGUF Tensor Name Mapping

| HF Name | GGUF Name | Notes |
|----------|-----------|-------|
| `embed_tokens.weight` | `token_embd.weight` | |
| `norm.weight` | `output_norm.weight` | |
| `layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | |
| `layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` | |
| `layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` | |
| `layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` | |
| `layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` | |
| `layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_output.weight` | |
| `layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` | QK head norm |
| `layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` | QK head norm |
| `layers.{i}.self_attn.q_proj.norm.weight` | `blk.{i}.attn_q_norm_in.weight` | BitNet per-projection |
| `layers.{i}.self_attn.k_proj.norm.weight` | `blk.{i}.attn_k_norm_in.weight` | BitNet per-projection |
| `layers.{i}.self_attn.v_proj.norm.weight` | `blk.{i}.attn_v_norm_in.weight` | BitNet per-projection |
| `layers.{i}.self_attn.o_proj.norm.weight` | `blk.{i}.attn_output_norm_in.weight` | BitNet per-projection |
| `layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` | |
| `layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` | |
| `layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` | |
| `layers.{i}.mlp.gate_proj.norm.weight` | `blk.{i}.ffn_gate_norm_in.weight` | BitNet per-projection |
| `layers.{i}.mlp.up_proj.norm.weight` | `blk.{i}.ffn_up_norm_in.weight` | BitNet per-projection |
| `layers.{i}.mlp.down_proj.norm.weight` | `blk.{i}.ffn_down_norm_in.weight` | BitNet per-projection |

---

## 3. Conversion Script

### `utils/convert-bitnet-embedding-to-gguf.py`

Standalone conversion script (safetensors → GGUF). Key features:

- Hardcoded HF→GGUF tensor name mapping (no dependency on llama.cpp's Python converter)
- Supports three output types:
  - `--outtype f32`: all weights in float32
  - `--outtype f16`: 2D weights and embeddings as float16, norms as float16
  - `--outtype i2_s`: ternary weights packed in I2_S layout, non-ternary weights as float16
- Writes `key_length` and `value_length` metadata for head_dim=128 (critical: default calculation would give wrong value 64)
- GPT-2 BPE tokenizer handling with pre-tokenizer hash verification
- Pooling type auto-detection from `modules.json` / `1_Pooling/config.json` (sentence-transformers convention)
- EOS token override: uses `<|endoftext|>` (151643) for correct last-token pooling
- Architecture string: `"qwen3"`

### I2_S Ternary Packing

The I2_S format packs ternary weights {-1, 0, +1} into 2-bit representation:

- Quantization: `scale = 1/mean(|w|)`, `q = round(w * scale).clamp(-1, 1)`
- Encoding: `-1 → 0`, `0 → 1`, `+1 → 2`
- Every 128 values form a block, packed into 32 bytes
- Each byte stores 4 values: `byte = (c0 << 6) | (c1 << 4) | (c2 << 2) | c3`
- Scale (float32) is appended at the end of the packed data buffer

### Tensor Type Assignment

| Tensor Type | f16 mode | i2_s mode |
|-------------|----------|-----------|
| 2D linear weights | float16 | I2_S ternary packed |
| Embedding weights | float16 | float16 |
| Norm weights (1D) | float16 | float16 |

Note: `output.weight` (lm_head) is skipped for embedding models — it is not needed (no token generation).

---

## 4. C++ Modifications (`3rdparty/llama.cpp/src/llama.cpp`)

### 4.1 New Tensor Enums

Added 7 new entries after `LLM_TENSOR_FFN_SUB_NORM`:

```cpp
LLM_TENSOR_ATTN_Q_NORM_IN,
LLM_TENSOR_ATTN_K_NORM_IN,
LLM_TENSOR_ATTN_V_NORM_IN,
LLM_TENSOR_ATTN_OUT_NORM_IN,
LLM_TENSOR_FFN_GATE_NORM_IN,
LLM_TENSOR_FFN_UP_NORM_IN,
LLM_TENSOR_FFN_DOWN_NORM_IN,
```

### 4.2 Tensor Name Mappings

Added to `LLM_ARCH_QWEN3` tensor name map:

```cpp
{ LLM_TENSOR_ATTN_Q_NORM_IN,   "blk.%d.attn_q_norm_in" },
{ LLM_TENSOR_ATTN_K_NORM_IN,   "blk.%d.attn_k_norm_in" },
{ LLM_TENSOR_ATTN_V_NORM_IN,   "blk.%d.attn_v_norm_in" },
{ LLM_TENSOR_ATTN_OUT_NORM_IN, "blk.%d.attn_output_norm_in" },
{ LLM_TENSOR_FFN_GATE_NORM_IN, "blk.%d.ffn_gate_norm_in" },
{ LLM_TENSOR_FFN_UP_NORM_IN,   "blk.%d.ffn_up_norm_in" },
{ LLM_TENSOR_FFN_DOWN_NORM_IN, "blk.%d.ffn_down_norm_in" },
```

### 4.3 Layer Struct Fields

Added to `struct llama_layer`:

```cpp
struct ggml_tensor * attn_q_norm_in;
struct ggml_tensor * attn_k_norm_in;
struct ggml_tensor * attn_v_norm_in;
struct ggml_tensor * attn_out_norm_in;
struct ggml_tensor * ffn_gate_norm_in;
struct ggml_tensor * ffn_up_norm_in;
struct ggml_tensor * ffn_down_norm_in;
```

### 4.4 load_tensors (LLM_ARCH_QWEN3)

Added optional loading with `TENSOR_NOT_REQUIRED`:

```cpp
layer.attn_q_norm_in   = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM_IN,   "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
layer.attn_k_norm_in   = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM_IN,   "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
layer.attn_v_norm_in   = create_tensor(tn(LLM_TENSOR_ATTN_V_NORM_IN,   "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
layer.attn_out_norm_in = create_tensor(tn(LLM_TENSOR_ATTN_OUT_NORM_IN, "weight", i), {n_embd_head_k * n_head},    TENSOR_NOT_REQUIRED);
layer.ffn_gate_norm_in = create_tensor(tn(LLM_TENSOR_FFN_GATE_NORM_IN, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
layer.ffn_up_norm_in   = create_tensor(tn(LLM_TENSOR_FFN_UP_NORM_IN,   "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
layer.ffn_down_norm_in = create_tensor(tn(LLM_TENSOR_FFN_DOWN_NORM_IN, "weight", i), {n_ff},   TENSOR_NOT_REQUIRED);
```

Note: `o_proj.norm` input dimension is `n_embd_head_k * n_head` (=2048), `down_proj.norm` input dimension is `n_ff` (=3072).

### 4.5 build_qwen3() Graph Modifications

The `build_qwen3()` function was modified to conditionally apply per-projection RMSNorm. The logic is fully backward compatible — when no `*_norm_in` tensors exist, behavior is identical to original.

**Attention per-projection norms:**
```
// Before Q/K/V matmul:
if (layer.attn_q_norm_in) {
    cur_q = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    cur_q = ggml_mul(ctx, cur_q, layer.attn_q_norm_in);
} else {
    cur_q = cur;
}
Qcur = ggml_mul_mat(ctx, layer.wq, cur_q);
// Similarly for K, V
```

**O_proj norm** requires special handling because `llm_build_kv()` normally applies `wo` internally. Solution: pass `wo=NULL` to `llm_build_kv()`, then apply norm + wo manually:

```
cur = llm_build_kv(..., wo=NULL, ...);  // returns attention output without o_proj
if (layer.attn_out_norm_in) {
    cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, layer.attn_out_norm_in);
}
cur = ggml_mul_mat(ctx, layer.wo, cur);
```

**FFN per-projection norms:**
```
// Instead of llm_build_ffn(), manually:
if (layer.ffn_gate_norm_in) {
    tmp_gate = rms_norm(cur) * gate_norm_in;
} else {
    tmp_gate = cur;
}
tmp_gate = matmul(gate_proj, tmp_gate);
// Similarly for up_proj
tmp = silu(tmp_gate) * tmp_up;

if (layer.ffn_down_norm_in) {
    tmp = rms_norm(tmp) * down_norm_in;
}
cur = matmul(down_proj, tmp);
```

---

## 5. GGUF Conversion Process

There are two GGUF files to produce, from **two different source models**:

| GGUF Output | Source Model | Description |
|-------------|-------------|-------------|
| `embeddings-0.6b-f16.gguf` | `multilingual-e5-0.6b` (standard Qwen3) | F16 baseline, standard float16 weights |
| `bitnet-embeddings-0.6b-f16-i2_s.gguf` | `bitnet-embeddings-0.6b` (BitNet ternary) | I2_S ternary packed weights |

### 5.1 F16 GGUF: from multilingual-e5-0.6b

The F16 GGUF is converted from the **standard (non-BitNet) model** `multilingual-e5-0.6b`, which has normal float weights and no per-projection RMSNorm. This uses llama.cpp's standard converter since it is a vanilla Qwen3 model:

```bash
python3 /path/to/llama.cpp/convert_hf_to_gguf.py \
  /path/to/multilingual-e5-0.6b \
  --outtype f16 \
  --outfile embeddings-0.6b-f16.gguf
```

**What happens:**
1. Load `model.safetensors` (standard Qwen3 weights, bfloat16)
2. Convert all 2D weights (projections, embeddings) to float16
3. Convert norm weights to float32
4. Write GGUF with `qwen3` architecture metadata and tokenizer

**Output:** ~1.11 GiB (595.78M params)

### 5.2 I2_S GGUF: from bitnet-embeddings-0.6b

The I2_S GGUF is converted from the **BitNet ternary model** `bitnet-embeddings-0.6b`, which has ternary weights {-1, 0, +1} and 7 extra per-projection RMSNorm tensors per layer. This uses the custom converter because the standard llama.cpp converter does not handle per-projection norms or I2_S quantization:

```bash
python3 utils/convert-bitnet-embedding-to-gguf.py \
  /path/to/bitnet-embeddings-0.6b \
  --outfile bitnet-embeddings-0.6b-f16-i2_s.gguf --outtype i2_s
```

**What happens:**
1. Load `model.safetensors` (BitNet ternary weights, bfloat16)
2. Map HF tensor names to GGUF names, including 7 extra `*_norm_in` tensors per layer (see Section 2)
3. For each 2D linear weight (q/k/v/o/gate/up/down projections):
   - Compute scale: `scale = 1 / mean(|w|)`
   - Quantize: `q = round(w * scale).clamp(-1, 1)`
   - Encode: `-1 -> 0`, `0 -> 1`, `+1 -> 2`
   - Pack every 128 values into 32 bytes (4 values per byte, 2 bits each)
   - Append per-row float32 scale
4. Keep embeddings (`token_embd.weight`) in float16 (not ternary)
5. Keep all norm weights in float16
6. Skip `output.weight` (lm_head, not needed for embedding models)
7. Write GGUF with `I2_S` type tag for quantized tensors

**Output:** ~699 MiB (~50% of F16 size)

### 5.3 Why Two Different Source Models?

- `multilingual-e5-0.6b` is the **teacher/baseline model** with standard float weights, used as the F16 performance reference
- `bitnet-embeddings-0.6b` is the **1-bit quantized student model** with ternary weights and per-projection BitLinear norms, converted to I2_S for efficient CPU inference
- Benchmarking compares both to measure the throughput gain and quality trade-off of ternary quantization

### 5.4 Tensor Type Summary

| Tensor | F16 (from e5-0.6b) | I2_S (from bitnet-0.6b) |
|--------|---------------------|-------------------------|
| Linear projections (q/k/v/o/gate/up/down) | float16 | I2_S (2-bit packed + float32 scale) |
| Embedding (`token_embd.weight`) | float16 | float16 |
| Per-projection norms (`*_norm_in`) | N/A (not present) | float16 |
| Layer norms (`attn_norm`, `ffn_norm`) | float32 | float16 |
| QK head norms (`attn_q_norm`, `attn_k_norm`) | float32 | float32 |
| `output.weight` (lm_head) | present | skipped |

---

## 6. Build and Run

```bash
# Build with BitNet repo (includes I2_S support)
cmake -S /path/to/BitNet -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-embedding llama-bench -j$(nproc)

# Run embedding inference
build/bin/llama-embedding -m bitnet-embeddings-0.6b-f16-i2_s.gguf \
  -p "hello world" --embd-normalize 2 --embd-output-format array

# Benchmark: F16 vs I2_S
build/bin/llama-bench -m embeddings-0.6b-f16.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0

build/bin/llama-bench -m bitnet-embeddings-0.6b-f16-i2_s.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0
```
