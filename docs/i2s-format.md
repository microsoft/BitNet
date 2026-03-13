# I2_S Quantization Format

This note documents the `I2_S` weight layout used by `bitnet.cpp` so alternative runtimes and tooling can parse the format without reverse-engineering the kernels.

The description below is derived from the packing logic in `src/ggml-bitnet-mad.cpp`, especially `quantize_i2_s()`.

## What `I2_S` stores

`I2_S` is the ternary weight format used by the CPU inference path.

- Logical weight values are ternary: `-1`, `0`, or `+1`
- Each logical value is encoded into 2 bits
- A full tensor also stores one trailing `float32` scale value
- The serialized buffer reserves 32 extra bytes so the scale region stays aligned

At packing time the implementation first maps floating-point values into 2-bit symbols:

- `0` means `-1`
- `1` means `0`
- `2` means `+1`

Zero is detected with a small epsilon check, and non-zero values are converted by sign.

## CPU-dependent packing granularity

`QK_I2_S` depends on the active CPU backend in `src/ggml-bitnet-mad.cpp`:

- x86 / AVX / SSSE3 paths use `QK_I2_S = 128`
- ARM NEON paths use `QK_I2_S = 64`

The packing pattern is the same on both backends: each output byte stores four 2-bit symbols from different groups. The only difference is whether the groups are 32-wide (`128 = 4 x 32`) or 16-wide (`64 = 4 x 16`).

## x86 layout (`QK_I2_S = 128`)

For x86, one 32-byte block stores 128 ternary values split into 4 groups of 32.

For logical index `j` inside a 128-value block:

```text
group_idx = j / 32
group_pos = j % 32
```

The packer writes:

```text
byte_index = block_base + group_pos
shift = 6 - 2 * group_idx
packed_byte |= value << shift
```

So byte `group_pos` contains values from:

```text
[group_pos, 32 + group_pos, 64 + group_pos, 96 + group_pos]
```

with the bit layout:

```text
bits[7:6] -> element at offset 0
bits[5:4] -> element at offset 32
bits[3:2] -> element at offset 64
bits[1:0] -> element at offset 96
```

## ARM layout (`QK_I2_S = 64`)

For ARM NEON, one 16-byte block stores 64 ternary values split into 4 groups of 16.

For logical index `j` inside a 64-value block:

```text
group_idx = j / 16
group_pos = j % 16
```

The packer writes:

```text
byte_index = block_base + group_pos
shift = 6 - 2 * group_idx
packed_byte |= value << shift
```

So byte `group_pos` contains values from:

```text
[group_pos, 16 + group_pos, 32 + group_pos, 48 + group_pos]
```

with the same bit ordering:

```text
bits[7:6], bits[5:4], bits[3:2], bits[1:0]
```

## Scale storage

After the packed 2-bit payload, `quantize_i2_s()` stores one `float32` scale:

```text
scale_ptr = (float *)((char *)packed_weights + n / 4)
scale_ptr[0] = i2_scale
```

The function then returns:

```text
nrow * row_size / 4 + 32
```

That final `+ 32` keeps the serialized tensor aligned. If you are building a parser, treat the packed payload as `n / 4` bytes followed by a scale region that starts immediately after that payload, with extra alignment space reserved by the buffer size calculation.

## Practical decoding recipe

To decode a logical element:

1. Choose the backend block size (`128` for x86, `64` for ARM NEON).
2. Compute the block-local group and position.
3. Read the corresponding byte.
4. Extract the 2-bit symbol with the appropriate shift.
5. Map the symbol back to ternary:
   - `0 -> -1`
   - `1 -> 0`
   - `2 -> +1`

For x86:

```text
block = k / 128
pos = k % 128
group = pos / 32
lane = pos % 32
byte_offset = block * 32 + lane
shift = 6 - 2 * group
```

For ARM:

```text
block = k / 64
pos = k % 64
group = pos / 16
lane = pos % 16
byte_offset = block * 16 + lane
shift = 6 - 2 * group
```

## Related source files

- `src/ggml-bitnet-mad.cpp`
- `include/ggml-bitnet.h`
- `utils/convert-hf-to-gguf-bitnet.py`

If this format changes, update this document alongside the packing implementation.
