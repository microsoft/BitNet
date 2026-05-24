# BitNet Embeddings GGUF Conversion Implementation

## 1. Background

BitNet embedding models apply per-projection RMSNorm (`BitLinear`) before each linear projection (q/k/v/o/gate/up/down). Each projection has a `.norm.weight` that applies RMSNorm to the input **before** the matmul:

```
x → RMSNorm(x, norm.weight) → activation_quant(8bit) → matmul(weight_quant(ternary))
```

This pattern does **not** exist in any standard llama.cpp architecture:
- Standard Qwen3/Gemma3: no per-projection norms
- Standard BitNet: has `attn_sub_norm`/`ffn_sub_norm` at different positions (after attention/gate*up, not before each projection)

Currently two base architectures are supported:

| | bitnet-embeddings-0.6b (Qwen3) | bitnet-embeddings-270m (Gemma3) |
|---|---|---|
| Architecture | `Qwen3Model` | `Gemma3TextModel` |
| hidden_size | 1024 | 640 |
| num_attention_heads | 16 | 4 |
| num_key_value_heads | 8 | 1 |
| head_dim | 128 (note: != hidden_size/num_heads = 64) | 256 (note: != hidden_size/num_heads = 160) |
| intermediate_size | 3072 | 2048 |
| num_hidden_layers | 28 | 18 |
| hidden_activation | SiLU | gelu_pytorch_tanh |
| vocab_size | 151936 | 262144 |
| rope_theta | 1000000 | 10000.0 |
| rms_norm_eps | 1e-06 | 1e-06 |
| query_pre_attn_scalar | N/A | 256 |
| tie_word_embeddings | true | true |

### Gemma3 vs Qwen3 Key Differences

| Feature | Qwen3 | Gemma3 |
|---------|-------|--------|
| Post-attn norm | No | Yes (`post_attention_norm`) |
| Post-FFW norm | No | Yes (`post_ffw_norm`) |
| Pre-FFW norm naming | `post_attention_layernorm` → `ffn_norm` | `pre_feedforward_layernorm` → `ffn_norm` |
| QK head norms | Yes | Yes |
| Activation | SiLU | GELU |
| Embedding scaling | No | sqrt(n_embd) |
| EOS token override | Yes (`<\|endoftext\|>` 151643) | No (auto from tokenizer) |

### Per-Layer Tensors (7 extra norm tensors per layer)

| Tensor | Qwen3 Shape | Gemma3 Shape |
|--------|-------------|--------------|
| `self_attn.q_proj.norm.weight` | [1024] | [640] |
| `self_attn.k_proj.norm.weight` | [1024] | [640] |
| `self_attn.v_proj.norm.weight` | [1024] | [640] |
| `self_attn.o_proj.norm.weight` | [2048] | [1024] |
| `mlp.gate_proj.norm.weight` | [1024] | [640] |
| `mlp.up_proj.norm.weight` | [1024] | [640] |
| `mlp.down_proj.norm.weight` | [3072] | [2048] |

---

## 2. GGUF Tensor Name Mapping

### Common Tensors (both architectures)

| HF Name | GGUF Name | Notes |
|----------|-----------|-------|
| `embed_tokens.weight` | `token_embd.weight` | |
| `norm.weight` | `output_norm.weight` | |
| `layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | |
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

### Architecture-Specific Tensors

**Qwen3:**

| HF Name | GGUF Name |
|----------|-----------|
| `layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` |

**Gemma3 (additional):**

| HF Name | GGUF Name |
|----------|-----------|
| `layers.{i}.post_attention_layernorm.weight` | `blk.{i}.post_attention_norm.weight` |
| `layers.{i}.pre_feedforward_layernorm.weight` | `blk.{i}.ffn_norm.weight` |
| `layers.{i}.post_feedforward_layernorm.weight` | `blk.{i}.post_ffw_norm.weight` |

---

## 3. Conversion Script

### `utils/convert-bitnet-embedding-to-gguf.py`

Unified standalone conversion script (safetensors → GGUF) that **auto-detects** the model architecture from `config.json`'s `model_type` field (`qwen3` or `gemma3_text`). Key features:

- Hardcoded HF→GGUF tensor name mapping (no dependency on llama.cpp's Python converter)
- Auto-detection of architecture and GGUF arch string (`qwen3` / `gemma3`)
- Supports three output types:
  - `--outtype f32`: all weights in float32
  - `--outtype f16`: 2D weights and embeddings as float16, norms as float16
  - `--outtype i2_s`: ternary weights packed in I2_S layout, non-ternary weights as float16
- Writes `key_length` and `value_length` metadata for correct head_dim (critical: head_dim != hidden_size/num_heads for both models, default calculation would give wrong values)
- BPE tokenizer handling with per-architecture pre-tokenizer hash verification:
  - Qwen3: GPT-2 BPE tokenizer
  - Gemma3: GemmaTokenizerFast (BPE)
- Pooling type auto-detection from `modules.json` / `1_Pooling/config.json` (sentence-transformers convention)
- Architecture-specific tokenizer handling:
  - Qwen3: EOS token override (`<|endoftext|>` 151643) + `add_eos_token(True)` for last-token pooling
  - Gemma3: EOS token auto-set by SpecialVocab from tokenizer_config.json (eos_token_id=1)
- Gemma3: writes `query_pre_attn_scalar = 256` for correct attention scaling

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

Added after `LLM_ARCH_GEMMA2` in the `llm_arch` enum with name mapping `"gemma3"`. Qwen3 (`LLM_ARCH_QWEN3`) was added by the 0.6b adaptation.

### 4.2 New Tensor Enums (shared across architectures)

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

### 4.4 Tensor Name Mappings

Both `LLM_ARCH_QWEN3` and `LLM_ARCH_GEMMA3` include the 7 per-projection norm tensor mappings plus standard tensors (see Section 2 for full mapping). Key differences:

- Qwen3 includes `LLM_TENSOR_OUTPUT` (`"output"`); Gemma3 does not (uses tied embeddings directly)
- Gemma3 additionally includes `LLM_TENSOR_ATTN_POST_NORM` (`"blk.%d.post_attention_norm"`) and `LLM_TENSOR_FFN_POST_NORM` (`"blk.%d.post_ffw_norm"`)

### 4.5 load_tensors

Both architectures load the 7 per-projection norm tensors as optional (`TENSOR_NOT_REQUIRED`):

```cpp
layer.attn_q_norm_in   = create_tensor(tn(...), {n_embd},              TENSOR_NOT_REQUIRED);
layer.attn_k_norm_in   = create_tensor(tn(...), {n_embd},              TENSOR_NOT_REQUIRED);
layer.attn_v_norm_in   = create_tensor(tn(...), {n_embd},              TENSOR_NOT_REQUIRED);
layer.attn_out_norm_in = create_tensor(tn(...), {n_embd_head_k * n_head}, TENSOR_NOT_REQUIRED);
layer.ffn_gate_norm_in = create_tensor(tn(...), {n_embd},              TENSOR_NOT_REQUIRED);
layer.ffn_up_norm_in   = create_tensor(tn(...), {n_embd},              TENSOR_NOT_REQUIRED);
layer.ffn_down_norm_in = create_tensor(tn(...), {n_ff},                TENSOR_NOT_REQUIRED);
```

Note: `o_proj.norm` input dimension is `n_embd_head_k * n_head` (Qwen3: 2048, Gemma3: 1024), `down_proj.norm` input dimension is `n_ff` (Qwen3: 3072, Gemma3: 2048).

Both graph functions use the same per-projection norm pattern. The logic is fully backward compatible — when no `*_norm_in` tensors exist, behavior is identical to the original.

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
tmp_gate = activation(tmp_gate);  // SiLU for Qwen3, GELU for Gemma3
// Similarly for up_proj
tmp = tmp_gate * tmp_up;

if (layer.ffn_down_norm_in) {
    tmp = rms_norm(tmp) * down_norm_in;
}
cur = matmul(down_proj, tmp);
```

**Gemma3-specific differences:**
- Embedding scaling by `sqrt(n_embd)` (Gemma convention)
- GELU activation instead of SiLU
- Post-attention and post-FFN layer norms
- `query_pre_attn_scalar` for attention scaling

---

## 5. GGUF Conversion Process

Each model variant requires two GGUF files from **two different source models**:

### 5.1 Qwen3 (0.6b)

| GGUF Output | Source Model | Description |
|-------------|-------------|-------------|
| `embeddings-0.6b-f16.gguf` | `multilingual-e5-0.6b` (standard Qwen3) | F16 baseline |
| `bitnet-embeddings-0.6b-f16-i2_s.gguf` | `bitnet-embeddings-0.6b` (BitNet ternary) | I2_S ternary packed |

**F16 (from standard Qwen3 model):**
```bash
python3 utils/convert-bitnet-embedding-to-gguf.py \
  /path/to/multilingual-e5-0.6b \
  --outtype f16 \
  --outfile embeddings-0.6b-f16.gguf
```

What happens:
1. Load `model.safetensors` (standard Qwen3 weights, bfloat16)
2. Convert all 2D weights (projections, embeddings) to float16
3. Convert norm weights to float16
4. Write GGUF with `qwen3` architecture metadata and tokenizer

**Output:** ~1.11 GiB (595.78M params)

**I2_S (from BitNet model):**
```bash
python3 utils/convert-bitnet-embedding-to-gguf.py \
  /path/to/bitnet-embeddings-0.6b \
  --outfile bitnet-embeddings-0.6b-f16-i2_s.gguf --outtype i2_s
```

What happens:
1. Load `model.safetensors` (BitNet ternary weights, bfloat16)
2. Map HF tensor names to GGUF names, including 7 extra `*_norm_in` tensors per layer
3. For each 2D linear weight: quantize to I2_S ternary packed format
4. Keep embeddings (`token_embd.weight`) in float16
5. Keep all norm weights in float16
6. Skip `output.weight` (lm_head, not needed for embedding models)
7. Write GGUF with `I2_S` type tag for quantized tensors

**Output:** ~699 MiB (~50% of F16 size)

### 5.2 Gemma3 (270m)

| GGUF Output | Source Model | Description |
|-------------|-------------|-------------|
| `multilingual-e5-270m-f16.gguf` | `multilingual-e5-270m-260311` (standard Gemma3) | F16 baseline |
| `bitnet-embeddings-270m-i2_s.gguf` | `bitnet-embeddings-270m` (BitNet ternary) | I2_S ternary packed |

**F16 (from standard Gemma3 model):**
```bash
python3 utils/convert-bitnet-embedding-to-gguf.py \
  /path/to/multilingual-e5-270m-260311 \
  --outtype f16
```

What happens:
1. Load `model.safetensors` (standard Gemma3 weights, bfloat16)
2. Convert all 2D weights (projections, embeddings) to float16
3. Convert norm weights to float16
4. Write GGUF with `gemma3` architecture metadata and tokenizer

**I2_S (from BitNet model):**
```bash
python3 utils/convert-bitnet-embedding-to-gguf.py \
  /path/to/bitnet-embeddings-270m \
  --outtype i2_s
```

What happens:
1. Load `model.safetensors` (BitNet ternary weights, bfloat16)
2. Map HF tensor names to GGUF names, including 7 extra `*_norm_in` tensors per layer
3. For each 2D linear weight: quantize to I2_S ternary packed format
4. Keep embeddings (`token_embd.weight`) in float16
5. Keep all norm weights in float16
6. Skip `output.weight` (lm_head, not needed for embedding models)
7. Write GGUF with `I2_S` type tag for quantized tensors

### 5.3 Why Two Different Source Models?

- `multilingual-e5-*` is the **teacher/baseline model** with standard float weights, used as the F16 performance reference
- `bitnet-embeddings-*` is the **1-bit quantized student model** with ternary weights and per-projection BitLinear norms, converted to I2_S for efficient CPU inference
- Benchmarking compares both to measure the throughput gain and quality trade-off of ternary quantization

### 5.4 Tensor Type Summary

| Tensor | F16 (baseline) | I2_S (BitNet) |
|--------|----------------|---------------|
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

# Run embedding inference (Qwen3 example)
build/bin/llama-embedding -m bitnet-embeddings-0.6b-f16-i2_s.gguf \
  -p "hello world" --embd-normalize 2 --embd-output-format array

# Run embedding inference (Gemma3 example)
build/bin/llama-embedding -m bitnet-embeddings-270m-i2_s.gguf \
  -p "hello world" --embd-normalize 2 --embd-output-format array

# Benchmark: F16 vs I2_S (Qwen3)
build/bin/llama-bench -m embeddings-0.6b-f16.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0

build/bin/llama-bench -m bitnet-embeddings-0.6b-f16-i2_s.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0

# Benchmark: F16 vs I2_S (Gemma3)
build/bin/llama-bench -m multilingual-e5-270m-f16.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0

build/bin/llama-bench -m bitnet-embeddings-270m-i2_s.gguf \
  -t 8 -p 128,256,512,1024,2048 -n 32,64 -r 3 -ngl 0
```
