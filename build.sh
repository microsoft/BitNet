#!/bin/bash

# Navigate to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

echo "Building BitNet-WASM..."

# Define source files and include directories
BITNET_SOURCES="src/ggml-bitnet-lut.cpp"
INCLUDE_DIRS="-Iinclude"

# Output WASM file name
OUTPUT_FILE="bitnet.wasm"
OUTPUT_JS_FILE="bitnet.js" # Emscripten generates a JS loader

# Emscripten compiler flags
EMCC_FLAGS="-O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] -s EXPORTED_FUNCTIONS=['_ggml_bitnet_init','_ggml_bitnet_free','_ggml_bitnet_mul_mat_task_compute','_ggml_bitnet_transform_tensor','_malloc','_free'] -s ALLOW_MEMORY_GROWTH=1"

# Prepare bitnet-lut-kernels.h by copying a preset one
PRESET_KERNEL_HEADER="preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h"
TARGET_KERNEL_HEADER="include/bitnet-lut-kernels.h"

if [ ! -f "$PRESET_KERNEL_HEADER" ]; then
    echo "Error: Preset kernel header $PRESET_KERNEL_HEADER not found."
    exit 1
fi

echo "Copying $PRESET_KERNEL_HEADER to $TARGET_KERNEL_HEADER"
cp "$PRESET_KERNEL_HEADER" "$TARGET_KERNEL_HEADER"

# Create a simple ggml.h stub for our implementation
echo "Creating ggml.h stub..."
cat > include/ggml.h << 'EOF'
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Minimal GGML tensor structure for our implementation
struct ggml_tensor {
    int64_t ne[4]; // dimensions
    void * data;   // data pointer
    void * extra;  // extra data
};

// Minimal GGML type enum
enum ggml_type {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_I8 = 16,
    GGML_TYPE_I16 = 17,
    GGML_TYPE_I32 = 18,
    GGML_TYPE_COUNT = 19,
    GGML_TYPE_TL1 = 20,
};

// Minimal GGML API
void ggml_init(void);
int64_t ggml_nelements(const struct ggml_tensor * tensor);
size_t ggml_row_size(enum ggml_type type, int64_t n);

#ifdef __cplusplus
}
#endif
EOF

# Create a simple ggml-backend.h stub
echo "Creating ggml-backend.h stub..."
cat > include/ggml-backend.h << 'EOF'
#pragma once

#define GGML_API

#ifdef __cplusplus
extern "C" {
#endif

// Empty stub for ggml-backend.h

#ifdef __cplusplus
}
#endif
EOF

# Create a simple ggml-alloc.h stub
echo "Creating ggml-alloc.h stub..."
cat > include/ggml-alloc.h << 'EOF'
#pragma once

// Empty stub for ggml-alloc.h
EOF

# Create a simple ggml-quants.h stub
echo "Creating ggml-quants.h stub..."
cat > include/ggml-quants.h << 'EOF'
#pragma once

// Empty stub for ggml-quants.h
EOF

# Create a simple implementation of the required GGML functions
echo "Creating ggml.c stub..."
cat > src/ggml.c << 'EOF'
#include "ggml.h"

void ggml_init(void) {
    // Stub implementation
}

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    if (!tensor) return 0;
    return tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
}

size_t ggml_row_size(enum ggml_type type, int64_t n) {
    // Simple implementation for our needs
    switch (type) {
        case GGML_TYPE_F32: return n * sizeof(float);
        case GGML_TYPE_I8:  return n * sizeof(int8_t);
        default:            return n * sizeof(float); // Default to float
    }
}
EOF

echo "Compiling with Emscripten..."
echo "Executing: emcc $EMCC_FLAGS $INCLUDE_DIRS $BITNET_SOURCES src/ggml.c -o $OUTPUT_JS_FILE"
emcc $EMCC_FLAGS $INCLUDE_DIRS $BITNET_SOURCES src/ggml.c -o $OUTPUT_JS_FILE > emcc_stdout.log 2> emcc_stderr.log
EMCC_EXIT_CODE=$?
echo "emcc exit code: $EMCC_EXIT_CODE"
echo "--- emcc stderr ---"
cat emcc_stderr.log
echo "--- end emcc stderr ---"
echo "--- emcc stdout ---"
cat emcc_stdout.log
echo "--- end emcc stdout ---"

if [ $EMCC_EXIT_CODE -eq 0 ] && [ -f "$OUTPUT_FILE" ] && [ -f "$OUTPUT_JS_FILE" ]; then
    echo "Build successful!"
    echo "Output files: $OUTPUT_JS_FILE, $OUTPUT_FILE"
else
    echo "Build failed. Please check the output from emcc."
    exit 1
fi

echo "Build script finished."
