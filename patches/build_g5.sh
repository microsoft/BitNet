#!/bin/bash
# build_g5.sh - Build BitNet for Power Mac G5 (big-endian PowerPC AltiVec)
#
# Requirements:
#   - Mac OS X 10.5 Leopard (or Linux ppc64be)
#   - GCC 10+ with C++17 support
#   - Model file: bitnet_b1_58-large converted to GGUF I2_S format
#
# Three optimization levels:
#   1. Dot-product kernels (ggml-bitnet-mad.cpp) - vec_msum, vec_ld, vec_splat_u8
#   2. Framework vectorization (ggml.c GGML_SIMD + ggml-quants.c quantize_row_i8_s)
#      - vec_ld/vec_st, vec_madd, vec_abs, vec_round, vec_packs
#      - Applied via g5-altivec-framework.patch
#   3. OpenMP multi-threading (-fopenmp) - uses both G5 CPUs for 1.45x speedup
#      CRITICAL: -fopenmp must be in MK_CFLAGS/MK_CXXFLAGS (not just GGML_NO_OPENMP=)
#      because make command-line variables override Makefile's += appends
#
# Usage:
#   ./patches/build_g5.sh [GCC_PREFIX]
#
# Example:
#   ./patches/build_g5.sh /usr/local/gcc-10/bin
#   ./patches/build_g5.sh   # uses gcc/g++ from PATH

set -e

GCC_PREFIX="${1:-}"
if [ -n "$GCC_PREFIX" ]; then
    CC="${GCC_PREFIX}/gcc"
    CXX="${GCC_PREFIX}/g++"
else
    CC="gcc"
    CXX="g++"
fi

echo "=== BitNet G5 AltiVec Build ==="
echo "CC:  $CC"
echo "CXX: $CXX"
echo ""

# Step 1: Apply big-endian patches to llama.cpp submodule
echo ">>> Step 1: Applying big-endian patches..."
cd 3rdparty/llama.cpp
if git diff --quiet HEAD 2>/dev/null; then
    git apply ../../patches/g5-big-endian.patch
    echo "    Applied g5-big-endian.patch"
    git apply ../../patches/g5-altivec-framework.patch
    echo "    Applied g5-altivec-framework.patch"
    git apply ../../patches/g5-altivec-scale.patch
    echo "    Applied g5-altivec-scale.patch"
else
    echo "    Submodule already has local changes, skipping patch"
fi

# Step 2: Copy regex compatibility header
echo ">>> Step 2: Installing regex-ppc.h..."
cp ../../patches/regex-ppc.h common/regex-ppc.h
echo "    Installed common/regex-ppc.h"

# Step 3: Build using Makefile with G5 AltiVec flags
# C code uses -O3 (safe on PPC with GCC 10). C++ uses -Os because GCC 10.5
# has miscompile bugs at -O2/-O3 on PPC that cause Bus errors in arg.cpp,
# llama.cpp, and llama-vocab.cpp (aggressive vector register spills hit
# Mach-O ABI stack alignment issues).
# -include common/regex-ppc.h replaces broken std::regex on PPC BE
# -lm required for roundf() in AltiVec quantize path
echo ">>> Step 3: Building llama-cli with AltiVec + OpenMP flags..."
echo "    (This takes several minutes on dual G5)"
echo "    Use -t 2 for dual G5 (1.45x speedup via OpenMP)"

make -j2 \
    CC="$CC" \
    CXX="$CXX" \
    GGML_NO_METAL=1 \
    LLAMA_NO_ACCELERATE=1 \
    LLAMA_NO_LLAMAFILE=1 \
    "GGML_NO_OPENMP=" \
    MK_CFLAGS="-mcpu=970 -maltivec -fopenmp -O3 -I ggml/include" \
    MK_CXXFLAGS="-mcpu=970 -maltivec -fopenmp -Os -std=gnu++17 -I ggml/include -include common/regex-ppc.h" \
    MK_LDFLAGS="-L$(dirname $CC)/../lib -lgomp -lm" \
    llama-cli

echo ""
echo "=== Build complete ==="
echo ""
echo "Run inference with:"
echo "  ./3rdparty/llama.cpp/llama-cli \\"
echo "    -m <model>.gguf \\"
echo "    -p \"Once upon a time\" \\"
echo "    -n 30 -t 2 --no-warmup --no-mmap"
echo ""
echo "Performance (Dual 2.0 GHz G5, BitNet 700M I2_S):"
echo "  -t 1: ~721 ms/token (~1.4 t/s)"
echo "  -t 2: ~498 ms/token (~2.0 t/s)  [1.45x speedup]"
echo ""
echo "Optimization stack:"
echo "  - Dot product kernels (ggml-bitnet-mad.cpp): AltiVec vec_msum 16x raw"
echo "  - Framework ops (ggml_vec_scale/add/mul/dot/mad): ~4x via GGML_SIMD"
echo "  - Activation quantize (quantize_row_i8_s): ~4x via vec_abs/vec_packs"
echo "  - OpenMP threading: 1.45x on dual G5 (both CPUs active)"
