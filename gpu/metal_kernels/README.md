# BitNet Metal Backend

Metal (Apple GPU) implementation for BitNet inference on macOS and Apple Silicon devices.

## Overview

This directory contains the Metal backend implementation for BitNet inference, enabling high-performance quantized neural network execution on Apple GPUs (M1, M2, M3 series).

## Architecture

### Components

1. **Metal Shaders** (`bitnet_kernels.metal`)
   - `bitlinear_int8xint2`: Matrix multiplication kernel for int8 activations × int2 weights
   - `bitlinear_int8xint2_simd`: SIMD-optimized variant with threadgroup caching
   - `quantize_input`: Per-row activation quantization
   - 2-bit weight decompression with ternary mapping (-1, 0, +1)

2. **Objective-C++ Wrapper** (`metal_backend.mm`)
   - PyTorch extension binding
   - Metal device management and pipeline state caching
   - Buffer management and command encoding

3. **Python Model** (`model.py`)
   - PyTorch model wrapper for Metal backend
   - `BitLinearMetal`: Metal-accelerated linear layer
   - `pack_weight_int8_to_int2`: Weight packing utility
   - Falls back to MPS operations when custom kernels unavailable

4. **Setup Script** (`setup.py`)
   - Build configuration for Metal extension
   - Links against Metal and Foundation frameworks

## Performance Characteristics

### Expected Speedups (vs CPU SIMD)

Based on similar int8×int2 workloads:

- **M1 Pro/Max**: 2-4x faster than optimized CPU SIMD (Neon)
- **M2/M3**: 3-6x faster than CPU SIMD
- **M3 Max/Ultra**: 5-8x faster with unified memory benefits

### Comparison to CUDA

Metal performance is typically:
- 30-60% of equivalent NVIDIA GPU (A100/RTX 4090) for pure compute
- Similar or better for memory-bound workloads due to unified memory

## Building

### Prerequisites

- macOS 12.0+ (Monterey)
- Xcode Command Line Tools
- Python 3.8+
- PyTorch with MPS support

### Build Steps

```bash
cd gpu/metal_kernels

# Build Metal extension
python setup.py build_ext --inplace

# Or install
pip install -e .
```

## Usage

### Basic Usage

```python
import torch
from metal_kernels.model import Transformer, ModelArgs, BitLinearMetal

# Check Metal availability
if torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

# Create model with Metal backend
args = ModelArgs(use_kernel=True)
model = Transformer(args).to(device)

# Run inference
with torch.no_grad():
    output = model(tokens, cache)
```

### Profiling

```bash
# Profile Metal vs CPU
python utils/profile_inference.py --backend all --batch-sizes 1,8,16

# Specific backend
python utils/profile_inference.py --backend metal --batch-sizes 1,8
```

## Technical Details

### Quantization Format

- **Weights**: 2-bit packed (4 values per byte)
  - Mapping: -1 → 00, 0 → 01, +1 → 10
  - Stored as uint8, unpacked to int8 in kernel
  
- **Activations**: int8 with per-row scaling
  - Scale: `127 / max(abs(row))`
  - Range: [-128, 127]

### Memory Layout

```
Input [M, K] int8 → Quantize → Metal Buffer → Kernel
Weights [N, K/4] uint8 packed → Metal Buffer → Decode in kernel
Output [M, N] bfloat16 → Metal Buffer → PyTorch Tensor
```

### Kernel Design

The Metal kernels use:
- **Tile-based processing**: 8×32 tiles for efficient cache usage
- **Threadgroup memory**: For weight caching and reduction
- **SIMD groups**: 32 threads for warp-level operations
- **BFloat16 output**: Native Apple GPU format support

## Limitations

1. **No Tensor Cores**: Metal doesn't expose int8×int2 tensor operations like CUDA
2. **Kernel Compilation**: Shaders compiled at runtime (first use has overhead)
3. **Memory**: Unified memory is beneficial but still limited by system RAM
4. **Precision**: BFloat16 output may have slight accuracy differences vs FP32

## Future Optimizations

1. **Pre-compiled Metal library**: Ship `.metallib` instead of source compilation
2. **Persistent buffers**: Reuse Metal buffers across inference calls
3. **Graph capture**: Metal Performance Shaders graphs for reduced overhead
4. **SIMD shuffle**: More aggressive use of SIMD-scoped operations
5. **Half-precision accumulation**: Explore fp16 vs bf16 tradeoffs

## References

- [BitNet Paper](https://arxiv.org/abs/2310.11453)
- [Metal Shading Language Guide](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders)
