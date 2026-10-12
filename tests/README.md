# I2_S numerical regressions

These model-free tests check the CPU kernels linked into `ggml-cpu`, not a copied
implementation or the quality of generated text. They make no network requests
and require no model weights, server, Python packages, or test framework.

This is a **draft regression contribution**: the tests intentionally expose
existing failures on the revisions below. They return nonzero on failure; there
are no expected-failure overrides. `BITNET_BUILD_TESTS` defaults to `OFF`, leaving
normal builds unchanged. This contribution does not contain a runtime fix.

## Build and run on Apple Silicon

From the repository root, with the pinned llama.cpp submodule initialized:

```sh
git submodule update --init --depth 1 3rdparty/llama.cpp
python utils/codegen_tl1.py --model bitnet_b1_58-3B \
  --BM 160,320,320 --BK 64,128,64 --bm 32,64,32
cmake -S . -B build-i2s-tests \
  -DBITNET_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF \
  -DBITNET_ARM_TL1=OFF -DBITNET_X86_TL2=OFF \
  -DGGML_METAL=OFF -DGGML_BLAS=OFF -DGGML_OPENMP=OFF \
  -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a \
  -DLLAMA_CURL=OFF
cmake --build build-i2s-tests --target test-i2s -j 4
ctest --test-dir build-i2s-tests -R '^bitnet-i2s-' --output-on-failure
```

The existing build requires the generated TL1 header even with TL1 execution
disabled. The baseline `armv8-a` target avoids the separate `src1_cont` build
failure on i8mm-enabled targets
([microsoft/BitNet#618](https://github.com/microsoft/BitNet/issues/618)).
Static linking exposes the internal numerical entry points. Dynamic/multiple CPU
backends and TL1/TL2 configurations are rejected when these tests are enabled.

For AddressSanitizer and UndefinedBehaviorSanitizer, use a separate build directory
and add these flags to the same configure command:

```sh
-DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
-DCMAKE_EXE_LINKER_FLAGS="-fsanitize=address,undefined"
```

## Contract and coverage

- Pack independently generated codes `0, 1, 2` into four 32-element groups per
  128-element / 32-byte block. Do not use the production quantizer as the oracle.
- Compare against integer dot products of the original, unpacked codes. These
  low-level kernels return **unsigned-code** dot products; the CPU matmul caller
  subtracts the activation sum separately.
- Exercise inner dimensions 128, 256, 2560, and 6912; activation-column counts
  1, 4, 8, 16, 32, 48, and 64; tight and padded output strides.
- Check both mixed-sign activations and all-127 activations with all-2 codes.
  The largest exact sum is 1,755,648, representable in both int32 and float32.
- The dot test uses one weight row and multiple activation columns, including
  padded activation strides. The GEMM test uses five weight rows to exercise
  non-square matrices and row tails. Output gaps and trailing canaries must stay
  unchanged.
- Extra initialized backing weight rows let the known row/column confusion fail
  numerically without an out-of-allocation read. This is not a memory-safety suite.
- Zero activations must quantize to zero with a zero sum and a positive, finite
  scale, so the caller's division by that scale remains defined.

There are 112 dot cases, 112 GEMM cases, and four zero-activation cases, registered
as three CTest tests. Checks remain active in Release builds.

## Reproduced results

Apple M4 Max, arm64 macOS, Apple clang 21.0.0, CMake 3.31.10, with the configuration
above:

| Source under test | Dot failures | GEMM failures | Zero-scale failures |
|---|---:|---:|---:|
| Main `0b341e582afbf9e1011f24744b554c96a3477eb5` | 112/112 | 112/112 | 4/4 |
| microsoft/BitNet#642 at `9dcd80d20a5125b61eeea70472db9553dba378d9`, including its supplied llama.cpp patch | 84/112 | 84/112 | 4/4 |
| Local diagnostic scalar correction, not included in this PR | 0/112 | 0/112 | 0/4 |

Both upstream revisions pin llama.cpp
`390c307752ab78fd8189f359d6954c9ba1be74af`. On main, the active ARM dot entry point
comes from `3rdparty/llama.cpp/ggml/src/ggml-cpu/quants.c`; the top-level
`src/ggml-bitnet-mad.cpp` is not in that CPU target. The supplied patch in
[microsoft/BitNet#642](https://github.com/microsoft/BitNet/pull/642) adds it.
The comparison used that PR's top-level MAD changes **and** applied its
`patches/0001-fix-arm64-2b-inference.patch` inside the pinned submodule.
These results concern only the stated kernel configuration, not that PR's
end-to-end chat claims.

For example, main's first mixed dot case produces `461` instead of `-1090`;
the same case with microsoft/BitNet#642 produces `-1039`.
Main returns scale `0` for a zero activation vector.

The diagnostic correction changes the scalar fallback to the 128/32 packing,
unsigned codes, activation-column addressing and output stride, and uses a
nonzero scale for zero activations. It is a test-validation control, not a
performance-qualified implementation. Main and this control were also run with
ASan/UBSan: the final fixtures produced the numerical results above without
sanitizer diagnostics. No x86, GPU, full-model, or end-to-end inference
certification is implied.

Related reports and proposed fixes:
[microsoft/BitNet#585](https://github.com/microsoft/BitNet/issues/585),
[microsoft/BitNet#600](https://github.com/microsoft/BitNet/issues/600),
[microsoft/BitNet#633](https://github.com/microsoft/BitNet/pull/633), and
[microsoft/BitNet#642](https://github.com/microsoft/BitNet/pull/642).
Only the explicitly listed revisions/configurations were tested here.
