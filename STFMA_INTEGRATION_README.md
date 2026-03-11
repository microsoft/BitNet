# Sparse-Ternary-FMA Integration for BitNet

This document describes the integration of the sparse-ternary-fma library into BitNet for improved performance of ternary matrix operations.

## Overview

The sparse-ternary-fma library provides highly optimized implementations of ternary arithmetic operations using 2-bit encoding and SIMD instructions (AVX2/AVX-512). This integration replaces BitNet's matrix multiplication operations with sparse-ternary-fma implementations for improved performance on supported hardware.

## Features

- **2-bit Ternary Encoding:** Efficient storage of ternary values {-1, 0, +1}
- **SIMD Acceleration:** AVX2 and AVX-512 implementations for maximum throughput
- **Automatic Dispatch:** Automatically selects the best implementation based on hardware and operation size
- **Backward Compatible:** Falls back to original BitNet implementation for small operations
- **Zero Overhead:** Uses thread-local buffer pooling to minimize memory allocations

## Performance Benefits

- **2.38× throughput improvement** on AVX-512 systems (from sparse-ternary-fma benchmarks)
- **26.12× latency improvement** for sparse operations
- **4× memory density** compared to 8-bit representation
- **Better cache utilization** due to smaller memory footprint

## Architecture

### Layer 1: Core sparse-ternary-fma Library

Located in `3rdparty/sparse-ternary-fma/`

Provides the base ternary FMA operations with int64 support.

### Layer 2: BitNet Adapter Layer

**Files:**
- `include/ggml-bitnet-stfma.h` - Header file with API declarations
- `src/ggml-bitnet-stfma.cpp` - Implementation of adapter functions

**Functions:**
- Encoding conversion (BitNet ↔ sparse-ternary-fma)
- Type conversion (int8 ↔ int32)
- int32 variants of sparse ternary FMA
- BitNet integration function `ggml_vec_dot_i2_i8_stfma()`

### Layer 3: BitNet API Integration

**Modified Files:**
- `src/ggml-bitnet-mad.cpp` - Added automatic dispatch to `ggml_vec_dot_i2_i8_s()`

**Changes:**
- Added conditional compilation for sparse-ternary-fma
- Added threshold-based dispatch logic
- Maintains backward compatibility

### Layer 4: Build System

**Modified Files:**
- `CMakeLists.txt` - Added sparse-ternary-fma build configuration
- `src/CMakeLists.txt` - Added adapter source files

**Options:**
- `BITNET_USE_STFMA` - Enable/disable sparse-ternary-fma integration (default: ON)
- `GGML_BITNET_STFMA_THRESHOLD` - Threshold for using sparse-ternary-fma (default: 1024)

## Building

### Standard Build (with sparse-ternary-fma)

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Build without sparse-ternary-fma

```bash
mkdir build
cd build
cmake -DBITNET_USE_STFMA=OFF ..
make -j$(nproc)
```

### Custom Threshold

```bash
mkdir build
cd build
cmake -DGGML_BITNET_STFMA_THRESHOLD=2048 ..
make -j$(nproc)
```

## Testing

A test program is provided to verify the correctness of the integration:

```bash
# Build the test program
g++ -o test_stfma_integration test_stfma_integration.cpp \
    src/ggml-bitnet-stfma.cpp \
    3rdparty/sparse-ternary-fma/src/sparse_ternary_fma.c \
    -I include \
    -I 3rdparty/sparse-ternary-fma/include \
    -std=c++11 -mavx2 -mavx512f -O3

# Run the test
./test_stfma_integration
```

Expected output:
```
========================================
Sparse-Ternary-FMA Integration Test
========================================

Testing with n = 128...
  Reference result: 1234.0
  STFMA result:     1234.0
  Absolute error:   0.0
  Relative error:   0.0
  ✓ Test PASSED

...

========================================
Results: 6/6 tests passed
========================================
```

## Encoding Differences

### BitNet Encoding

| Value | Encoding | Binary |
|-------|----------|--------|
| -1    | 0        | 00     |
| 0     | 1        | 01     |
| +1    | 2        | 10     |

### sparse-ternary-fma Encoding

| Value | Encoding | Binary |
|-------|----------|--------|
| -1    | 2 (0b10) | 10     |
| 0     | 0 (0b00) | 00     |
| +1    | 1 (0b01) | 01     |

The adapter layer handles conversion between these encodings transparently.

## Performance Tuning

### Threshold Selection

The `GGML_BITNET_STFMA_THRESHOLD` parameter controls when to use sparse-ternary-fma vs. the original implementation. 

**Guidelines:**
- **AVX-512 systems:** 512-1024 (default: 1024)
- **AVX2 systems:** 1024-2048
- **Older systems:** Consider disabling (`BITNET_USE_STFMA=OFF`)

### Profiling

To profile the integration:

```bash
# Build with profiling enabled
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j$(nproc)

# Run with perf
perf record -g ./your_bitnet_application
perf report
```

Look for:
- Time spent in `ggml_vec_dot_i2_i8_stfma`
- Time spent in conversion functions
- Cache miss rates

### Optimization Tips

1. **Increase threshold** if conversion overhead is significant
2. **Decrease threshold** if you have AVX-512 and want maximum SIMD usage
3. **Disable integration** if your workload consists mostly of small operations
4. **Enable AVX-512** compilation flags for maximum performance

## Implementation Details

### Buffer Management

The adapter uses thread-local storage to minimize memory allocations:

```cpp
static thread_local struct stfma_thread_buffers {
    uint8_t* encoding_buffer;
    int32_t* int32_buffer;
    int32_t* accumulator_buffer;
    size_t buffer_size;
} tl_buffers;
```

Buffers are allocated once per thread and reused across multiple calls.

### Encoding Conversion

Conversion is performed using lookup tables and SIMD operations:

```cpp
uint8_t convert_bitnet_to_stfma_byte(uint8_t bitnet_byte) {
    // Convert 4 trits in a single byte
    // BitNet: 0→-1, 1→0, 2→+1
    // STFMA:  0b10→-1, 0b00→0, 0b01→+1
}
```

### SIMD Implementations

Three SIMD implementations are provided:

1. **Scalar:** Reference implementation for all platforms
2. **AVX2:** Processes 8 int32 elements per iteration
3. **AVX-512:** Processes 16 int32 elements per iteration

Automatic dispatch selects the best implementation at runtime.

## Troubleshooting

### Compilation Errors

**Error:** `undefined reference to sparse_ternary_fma_*`

**Solution:** Ensure `BITNET_USE_STFMA` is enabled and sparse-ternary-fma source files are included in the build.

**Error:** `AVX-512 instructions not supported`

**Solution:** Your CPU doesn't support AVX-512. The code will fall back to AVX2 or scalar implementations automatically.

### Runtime Issues

**Issue:** Results don't match original implementation

**Solution:** Run the test program to verify correctness. If tests pass but your application fails, there may be an issue with data alignment or encoding.

**Issue:** Performance regression

**Solution:** Try increasing `GGML_BITNET_STFMA_THRESHOLD` or disabling the integration for your workload.

### Debugging

Enable debug output:

```cpp
// Add to ggml-bitnet-stfma.cpp
#define STFMA_DEBUG 1

#ifdef STFMA_DEBUG
#define STFMA_LOG(...) fprintf(stderr, __VA_ARGS__)
#else
#define STFMA_LOG(...)
#endif
```

## Limitations

1. **Encoding Conversion Overhead:** Conversion between BitNet and sparse-ternary-fma encodings adds overhead
2. **Type Conversion Overhead:** Converting int8 to int32 adds overhead
3. **AVX-512 Availability:** Maximum performance requires AVX-512 support
4. **Threshold Sensitivity:** Performance depends on proper threshold tuning

## Future Improvements

1. **Native Encoding:** Modify BitNet to use sparse-ternary-fma encoding natively
2. **int8 Variant:** Create int8 variant of sparse-ternary-fma to eliminate type conversion
3. **Sparse Processing:** Leverage sparse index format for very sparse weights
4. **ARM NEON Support:** Add NEON implementations for ARM platforms
5. **Batch Processing:** Process multiple vectors in parallel

## References

- [sparse-ternary-fma GitHub Repository](https://github.com/HyperFoldUK/sparse-ternary-fma)
- [BitNet GitHub Repository](https://github.com/HyperFoldUK/BitNet)
- [sparse-ternary-fma Technical Documentation](../3rdparty/sparse-ternary-fma/TECHNICAL.md)

## License

This integration is licensed under the Apache License 2.0, consistent with both BitNet and sparse-ternary-fma.

## Contributing

Contributions are welcome! Please submit pull requests to the BitNet repository with:

1. Clear description of changes
2. Performance benchmarks
3. Test results
4. Documentation updates

## Contact

For questions or issues related to this integration:

- BitNet Issues: https://github.com/HyperFoldUK/BitNet/issues
- sparse-ternary-fma Issues: https://github.com/HyperFoldUK/sparse-ternary-fma/issues

## Acknowledgments

- **sparse-ternary-fma:** Maurice Wilson, HyperFold Technologies UK Ltd
- **BitNet:** Microsoft Research and contributors
- **Integration:** Community contributors
