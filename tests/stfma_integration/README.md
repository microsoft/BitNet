# Sparse-Ternary-FMA Integration Tests

This directory contains test programs and artifacts created during the integration of the sparse-ternary-fma library into BitNet.

## Test Programs

### Branchless Conversion Tests

- **`test_branchless_conversion.cpp`** - Initial comprehensive test for branchless byte conversion
- **`test_branchless_conversion_v2.cpp`** - Simplified verification test
- **`test_final.cpp`** - Final verification test (compact version)

These tests verify that the optimized branchless conversion function produces identical results to the original branching implementation for all 256 possible input bytes.

**Compile and run:**
```bash
g++ -o test_final test_final.cpp -O3 && ./test_final
```

### AVX-512 Unpacking Test

- **`test_avx512_unpack.cpp`** - Verifies SIMD unpacking of 2-bit trits into int32 lanes

Tests the AVX-512 optimization that eliminates stack round-trips by unpacking trits directly in SIMD registers.

**Compile and run:**
```bash
g++ -o test_avx512_unpack test_avx512_unpack.cpp -mavx512f -O3 && ./test_avx512_unpack
```

### Pattern Analysis

- **`analyze_pattern.cpp`** - Analyzes the bit patterns to derive the correct branchless conversion formula

This program helped derive the XOR-based formula used in the branchless conversion.

**Compile and run:**
```bash
g++ -o analyze_pattern analyze_pattern.cpp && ./analyze_pattern
```

### Integration Test

- **`test_stfma_integration.cpp`** - End-to-end integration test for the complete sparse-ternary-fma integration

Tests the full integration including encoding conversion, SIMD operations, and result verification.

## Backup Files

- **`CMakeLists.txt.backup`** - Original root CMakeLists.txt before modification
- **`CMakeLists_modified.txt`** - Modified root CMakeLists.txt (reference)
- **`src/CMakeLists.txt.backup`** - Original src/CMakeLists.txt before modification
- **`src/CMakeLists_modified.txt`** - Modified src/CMakeLists.txt (reference)

These backup files are kept for reference and can be used to revert changes if needed.

## Optimizations Verified

1. **Branchless Byte Conversion** - Eliminates branching in `convert_bitnet_to_stfma_byte()`
2. **SIMD Trit Unpacking** - Eliminates stack round-trips in AVX2/AVX-512 implementations
3. **Integration Correctness** - Verifies end-to-end functionality

## Running All Tests

```bash
# Compile all tests
g++ -o test_final test_final.cpp -O3
g++ -o test_avx512_unpack test_avx512_unpack.cpp -mavx512f -O3
g++ -o analyze_pattern analyze_pattern.cpp

# Run all tests
./test_final && echo "✓ Branchless conversion passed"
./test_avx512_unpack && echo "✓ AVX-512 unpacking passed"
./analyze_pattern && echo "✓ Pattern analysis completed"
```

## Notes

- All tests have been verified to pass on the development system
- The compiled binaries (executables without extensions) are included for convenience
- Tests can be safely deleted after verification if desired
