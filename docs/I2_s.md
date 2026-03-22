# I2_S Quantization Format

I2_S is the quantization format used to store BitNet b1.58 ternary weights in GGUF files.
It packs 4 ternary values per byte using a block-interleaved layout.

This document is intended for developers building alternative inference runtimes
(WebGPU, Vulkan, Metal, etc.) who need to load and dequantize BitNet weights directly.

### Ternary encoding

Each weight is one of three values, stored in 2 bits:

| Bits | Value |
|------|-------|
| 00 | 0     |
| 01 | +1    |
| 10 | -1    |
| 11 | unused |

### Block layout

Weights are stored in blocks of 128 elements (32 bytes each).
Within a block, the 128 elements are split into 4 groups of 32.
Each byte encodes one element from each group:
```
bits [7:6] → element at position gp        (group 0, offset 0)
bits [5:4] → element at position 32 + gp   (group 1, offset 32)
bits [3:2] → element at position 64 + gp   (group 2, offset 64)
bits [1:0] → element at position 96 + gp   (group 3, offset 96)
```

To extract the element at logical index `k`:
```python
block       = k // 128
pos         = k % 128
group       = pos // 32
gp          = pos % 32
byte_offset = block * 32 + gp
shift       = 6 - 2 * group
value       = (byte >> shift) & 0x03
```

### Scale factor

The total byte size per tensor is:
```
ceil(num_elements / 4) + 32
```

The trailing 32 bytes store a single `float32` scale value, replicated 8 times.

### GGUF notes

- Type ID is **36** in the Eddie-Wang1120/llama.cpp fork (not type 27, which is I64 in upstream ggml)
- GGUF metadata uses architecture prefix `bitnet-25`, not `bitnet` or `llama`
- `token_embd.weight` is stored as F16 (type 1) — embeddings are not quantized to I2_S
- There is no `output.weight` tensor — the model uses tied embeddings (`lm_head` reuses `token_embd`)
