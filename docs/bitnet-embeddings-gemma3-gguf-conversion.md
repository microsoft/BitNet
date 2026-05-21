# BitNet Embeddings (Gemma3) GGUF Conversion Implementation

## 1. Background

`bitnet-embeddings-270m` is a Gemma3-based embedding model with BitNet per-projection RMSNorm (`BitLinear`). Each linear projection (q/k/v/o/gate/up/down) has a `.norm.weight` that applies RMSNorm to the input **before** the matmul:

```
x → RMSNorm(x, norm.weight) → activation_quant(8bit) → matmul(weight_quant(ternary))
```

This pattern does **not** exist in any standard llama.cpp architecture:
- Standard Gemma3: no per-projection norms
- Standard BitNet: has `attn_sub_norm`/`ffn_sub_norm` at different positions (after attention/gate*up, not before each projection)

### Model Config

- Architecture: `Gemma3TextModel`
- hidden_size: 640, num_attention_heads: 4, num_key_value_heads: 1
- head_dim: 256 (note: != hidden_size/num_heads = 160)
- intermediate_size: 2048, num_hidden_layers: 18
- hidden_activation: gelu_pytorch_tanh
- vocab_size: 262144
- rope_theta: 10000.0, rms_norm_eps: 1e-06
- query_pre_attn_scalar: 256
- tie_word_embeddings: true (implied, no separate output.weight)

### Gemma3 vs Gemma2 Key Differences

| Feature | Gemma2 | Gemma3 |
|---------|--------|--------|
| QK head norms | No | Yes (`q_norm`, `k_norm`) |
| Pre-FFW norm | `ffn_norm` | `pre_feedforward_layernorm` → `ffn_norm` |
| Post-FFW norm | `post_ffw_norm` | `post_feedforward_layernorm` → `post_ffw_norm` |
| Post-attn norm | `post_attention_norm` | Same |
| Activation | GELU | GELU |
| Embedding scaling | sqrt(n_embd) | sqrt(n_embd) |

### Per-Layer Tensors (7 extra norm tensors per layer)

| Tensor | Shape |
|--------|-------|
| `self_attn.q_proj.norm.weight` | [640] |
| `self_attn.k_proj.norm.weight` | [640] |
| `self_attn.v_proj.norm.weight` | [640] |
| `self_attn.o_proj.norm.weight` | [1024] |
| `mlp.gate_proj.norm.weight` | [640] |
| `mlp.up_proj.norm.weight` | [640] |
| `mlp.down_proj.norm.weight` | [2048] |

---

## 2. GGUF Tensor Name Mapping

| HF Name | GGUF Name | Notes |
|----------|-----------|-------|
| `embed_tokens.weight` | `token_embd.weight` | |
| `norm.weight` | `output_norm.weight` | |
| `layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | |
| `layers.{i}.post_attention_layernorm.weight` | `blk.{i}.post_attention_norm.weight` | |
| `layers.{i}.pre_feedforward_layernorm.weight` | `blk.{i}.ffn_norm.weight` | |
| `layers.{i}.post_feedforward_layernorm.weight` | `blk.{i}.post_ffw_norm.weight` | |
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

### `utils/convert-bitnet-embedding-270m-to-gguf.py`

Standalone conversion script (safetensors → GGUF). Key features:

- Hardcoded HF→GGUF tensor name mapping (no dependency on llama.cpp's Python converter)
- Supports three output types:
  - `--outtype f32`: all weights in float32
  - `--outtype f16`: 2D weights and embeddings as float16, norms as float16
  - `--outtype i2_s`: ternary weights packed in I2_S layout, non-ternary weights as float16
- Writes `key_length` and `value_length` metadata for head_dim=256
- Writes `query_pre_attn_scalar = 256` for correct attention scaling
- GemmaTokenizerFast (BPE) tokenizer handling with pre-tokenizer hash verification
- Pooling type auto-detection from `modules.json` / `1_Pooling/config.json` (sentence-transformers convention)
- EOS token auto-set by SpecialVocab from tokenizer_config.json (eos_token_id=1)
- Architecture string: `"gemma3"`

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

### 4.1 New Architecture: `LLM_ARCH_GEMMA3`

Added after `LLM_ARCH_GEMMA2` in the `llm_arch` enum with name mapping `"gemma3"`.

### 4.2 Tensor Enums (shared with Qwen3)

Reuses the 7 per-projection norm tensor enums added for Qwen3:

```cpp
LLM_TENSOR_ATTN_Q_NORM_IN,
LLM_TENSOR_ATTN_K_NORM_IN,
LLM_TENSOR_ATTN_V_NORM_IN,
LLM_TENSOR_ATTN_OUT_NORM_IN,
LLM_TENSOR_FFN_GATE_NORM_IN,
LLM_TENSOR_FFN_UP_NORM_IN,
LLM_TENSOR_FFN_DOWN_NORM_IN,
```

### 4.3 Tensor Name Mappings for `LLM_ARCH_GEMMA3`

```cpp
{ LLM_TENSOR_TOKEN_EMBD,          "token_embd" },
{ LLM_TENSOR_OUTPUT_NORM,         "output_norm" },
{ LLM_TENSOR_ATTN_NORM,           "blk.%d.attn_norm" },
{ LLM_TENSOR_ATTN_Q,              "blk.%d.attn_q" },
{ LLM_TENSOR_ATTN_K,              "blk.%d.attn_k" },
{ LLM_TENSOR_ATTN_V,              "blk.%d.attn_v" },
{ LLM_TENSOR_ATTN_OUT,            "blk.%d.attn_output" },
{ LLM_TENSOR_ATTN_Q_NORM,         "blk.%d.attn_q_norm" },
{ LLM_TENSOR_ATTN_K_NORM,         "blk.%d.attn_k_norm" },
{ LLM_TENSOR_ATTN_Q_NORM_IN,      "blk.%d.attn_q_norm_in" },
{ LLM_TENSOR_ATTN_K_NORM_IN,      "blk.%d.attn_k_norm_in" },
{ LLM_TENSOR_ATTN_V_NORM_IN,      "blk.%d.attn_v_norm_in" },
{ LLM_TENSOR_ATTN_OUT_NORM_IN,    "blk.%d.attn_output_norm_in" },
{ LLM_TENSOR_ATTN_POST_NORM,      "blk.%d.post_attention_norm" },
{ LLM_TENSOR_FFN_NORM,            "blk.%d.ffn_norm" },
{ LLM_TENSOR_FFN_GATE,            "blk.%d.ffn_gate" },
{ LLM_TENSOR_FFN_DOWN,            "blk.%d.ffn_down" },
{ LLM_TENSOR_FFN_UP,              "blk.%d.ffn_up" },
{ LLM_TENSOR_FFN_GATE_NORM_IN,    "blk.%d.ffn_gate_norm_in" },
{ LLM_TENSOR_FFN_UP_NORM_IN,      "blk.%d.ffn_up_norm_in" },
{ LLM_TENSOR_FFN_DOWN_NORM_IN,    "blk.%d.ffn_down_norm_in" },
{ LLM_TENSOR_FFN_POST_NORM,       "blk.%d.post_ffw_norm" },
```

### 4.4 load_tensors (LLM_ARCH_GEMMA3)

Based on Gemma2's tensor loading with additions:

- QK head norms: `attn_q_norm`, `attn_k_norm`
- All 7 BitNet per-projection norm_in tensors (TENSOR_NOT_REQUIRED)

```cpp
layer.attn_q_norm_in   = create_tensor(tn(...), {n_embd}, TENSOR_NOT_REQUIRED);
layer.attn_k_norm_in   = create_tensor(tn(...), {n_embd}, TENSOR_NOT_REQUIRED);
layer.attn_v_norm_in   = create_tensor(tn(...), {n_embd}, TENSOR_NOT_REQUIRED);
layer.attn_out_norm_in = create_tensor(tn(...), {n_embd_head_k * n_head}, TENSOR_NOT_REQUIRED);
layer.ffn_gate_norm_in = create_tensor(tn(...), {n_embd}, TENSOR_NOT_REQUIRED);
layer.ffn_up_norm_in   = create_tensor(tn(...), {n_embd}, TENSOR_NOT_REQUIRED);
layer.ffn_down_norm_in = create_tensor(tn(...), {n_ff}, TENSOR_NOT_REQUIRED);
```

### 4.5 build_gemma3() Graph Function

Combines Gemma2's structure with Qwen3's per-projection norm pattern:

**Key features:**
- Embedding scaling by `sqrt(n_embd)` (Gemma convention)
- GELU activation (gelu_pytorch_tanh)
- QK head norms after Q/K projection
- Conditional per-projection RMSNorm (backward compatible)
- Post-attention and post-FFN layer norms
- `wo=NULL` pattern for `attn_out_norm_in` (same as Qwen3)
- `query_pre_attn_scalar` for attention scaling

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
// QK head norms applied after projection
Qcur = ggml_rms_norm(ctx, Qcur, hparams.f_norm_rms_eps);
Qcur = ggml_mul(ctx, Qcur, layer.attn_q_norm);
```

**O_proj norm** with `wo=NULL` pattern:
```
cur = llm_build_kv(..., wo=NULL, ...);
if (layer.attn_out_norm_in) {
    cur = ggml_rms_norm(ctx, cur, hparams.f_norm_rms_eps);
    cur = ggml_mul(ctx, cur, layer.attn_out_norm_in);
}
cur = ggml_mul_mat(ctx, layer.wo, cur);
```

**FFN per-projection norms with GELU:**
```
if (layer.ffn_gate_norm_in) {
    tmp_gate = rms_norm(cur) * gate_norm_in;
} else {
    tmp_gate = cur;
}
tmp_gate = matmul(gate_proj, tmp_gate);
tmp_gate = gelu(tmp_gate);  // GELU, not SILU
// ...
```

---

## 5. GGUF Conversion Process

There are two GGUF files to produce, from **two different source models**:

| GGUF Output | Source Model | Description |
|-------------|-------------|-------------|
| `multilingual-e5-270m-f16.gguf` | `multilingual-e5-270m-260311` (standard Gemma3) | F16 baseline, standard float16 weights |
| `bitnet-embeddings-270m-i2_s.gguf` | `bitnet-embeddings-270m` (BitNet ternary) | I2_S ternary packed weights |

### 5.1 F16 GGUF: from multilingual-e5-270m-260311

```bash
python3 utils/convert-bitnet-embedding-270m-to-gguf.py \
  /path/to/multilingual-e5-270m-260311 \
  --outtype f16
```

**What happens:**
1. Load `model.safetensors` (standard Gemma3 weights, bfloat16)
2. Convert all 2D weights (projections, embeddings) to float16
3. Convert norm weights to float16
4. Write GGUF with `gemma3` architecture metadata and tokenizer

### 5.2 I2_S GGUF: from bitnet-embeddings-270m

```bash
python3 utils/convert-bitnet-embedding-270m-to-gguf.py \
  /path/to/bitnet-embeddings-270m \
  --outtype i2_s
```

**What happens:**
1. Load `model.safetensors` (BitNet ternary weights, bfloat16)
2. Map HF tensor names to GGUF names, including 7 extra `*_norm_in` tensors per layer
3. For each 2D linear weight: quantize to I2_S ternary packed format
4. Keep embeddings (`token_embd.weight`) in float16
5. Keep all norm weights in float16
6. Skip `output.weight` (lm_head, not needed for embedding models)
7. Write GGUF with `I2_S` type tag for quantized tensors

### 5.3 Why Two Different Source Models?

- `multilingual-e5-270m-260311` is the **teacher/baseline model** with standard float weights, used as the F16 performance reference
- `bitnet-embeddings-270m` is the **1-bit quantized student model** with ternary weights and per-projection BitLinear norms, converted to I2_S for efficient CPU inference
- Benchmarking compares both to measure the throughput gain and quality trade-off of ternary quantization

### 5.4 Tensor Type Summary

| Tensor | F16 (from e5-270m) | I2_S (from bitnet-270m) |
|--------|---------------------|-------------------------|
| Linear projections (q/k/v/o/gate/up/down) | float16 | I2_S (2-bit packed + float32 scale) |
| Embedding (`token_embd.weight`) | float16 | float16 |
| Per-projection norms (`*_norm_in`) | N/A (not present) | float16 |
| Layer norms (attn_norm, ffn_norm, etc.) | float16 | float16 |
| QK head norms (`attn_q_norm`, `attn_k_norm`) | float16 | float16 |
| `output.weight` (lm_head) | skipped | skipped |

---

## 6. Additional Changes

### 6.1 ggml.c: F16 Norm Weight Support

Added `ggml_compute_forward_mul_f32_f16()` function to support element-wise multiplication where norm weights are stored in float16. Modified `ggml_compute_forward_mul()` to dispatch based on `src1->type`.

### 6.2 gguf-py: I2_S Type

Added `I2_S = 36` to `GGMLQuantizationType` enum and `(4, 1)` quant size in `constants.py`.

### 6.3 CMakeLists.txt: BitNet LUT Kernels Guard

Guarded `bitnet-lut-kernels.h` include with `if (GGML_BITNET_ARM_TL1 OR GGML_BITNET_X86_TL2)` to prevent build errors when LUT kernels are not available.

### 6.4 ggml-bitnet-mad.cpp: AVX512 SIMD

Added AVX512BW SIMD paths for I2_S dot product functions:
- `ggml_vec_dot_i2_i8_s_1x1`
- `ggml_vec_dot_i2_i8_s_1xN`
- `ggml_vec_dot_i2_i8_s_Nx1`

---

## 7. Build and Run

```bash
# Build with BitNet repo (includes I2_S support)
cmake -S /path/to/BitNet -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-embedding llama-bench -j$(nproc)

# Run embedding inference
build/bin/llama-embedding -m bitnet-embeddings-270m-i2_s.gguf \
  -p "hello world" --embd-normalize 2 --embd-output-format array

# Benchmark: F16 vs I2_S
build/bin/llama-bench -m multilingual-e5-270m-f16.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0

build/bin/llama-bench -m bitnet-embeddings-270m-i2_s.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0
```
