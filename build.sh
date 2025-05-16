#!/bin/bash

# Ensure Emscripten environment is set up
# If you haven't already, run:
# cd /path/to/emsdk
# ./emsdk install latest
# ./emsdk activate latest
# source ./emsdk_env.sh
# ---
# It's better to have emsdk_env.sh sourced in your .bashrc or .zshrc

# Navigate to the script's directory to ensure relative paths work
cd "$(dirname "$0")"

echo "Building BitNet-WASM..."

# Define source files and include directories
BITNET_SOURCES="src/ggml-bitnet-mad.cpp src/ggml-bitnet-lut.cpp"
INCLUDE_DIRS="-Iinclude -I3rdparty/llama.cpp/ggml/include -I3rdparty/llama.cpp/ggml/src"

# Output WASM file name
OUTPUT_FILE="bitnet.wasm"
OUTPUT_JS_FILE="bitnet.js" # Emscripten generates a JS loader

# Emscripten compiler flags
# -O3 for optimization
# -s WASM=1 to specify WASM output
# -s MODULARIZE=1 to create a modular instance
# -s EXPORT_ES6=1 for ES6 module output (optional, good for modern JS)
# -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] to allow calling C functions
# -s ALLOW_MEMORY_GROWTH=1 if dynamic memory is needed
EMCC_FLAGS="-O3 -msimd128 -fno-rtti -DNDEBUG -flto=full -s WASM=1 -s MODULARIZE=1 -s EXPORT_ES6=1 -s EXPORT_ALL=1 -s EXPORTED_RUNTIME_METHODS=['ccall','cwrap'] -s EXPORTED_FUNCTIONS=['ggml_init','ggml_bitnet_init','ggml_nelements','ggml_bitnet_transform_tensor','ggml_bitnet_mul_mat_task_compute','ggml_bitnet_free'] -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=1GB -s MAXIMUM_MEMORY=4GB -s FORCE_FILESYSTEM=1 -s NO_EXIT_RUNTIME=1"

# We need to compile the relevant ggml.c from llama.cpp as well.
# For simplicity, let's assume llama.cpp/ggml/ggml.c is the main one needed.
# This is a guess and might need adjustment.
LLAMA_GGML_SOURCES="3rdparty/llama.cpp/ggml/src/ggml.c 3rdparty/llama.cpp/ggml/src/ggml-alloc.c 3rdparty/llama.cpp/ggml/src/ggml-backend.cpp 3rdparty/llama.cpp/ggml/src/ggml-quants.c"

# Check if llama.cpp submodule is initialized and primary ggml source file exists
# (Checking the first file in the list as a proxy for all)
FIRST_GGML_SOURCE=$(echo $LLAMA_GGML_SOURCES | cut -d' ' -f1)
if [ ! -f "$FIRST_GGML_SOURCE" ]; then
    echo "Error: llama.cpp submodule not found or $FIRST_GGML_SOURCE is missing."
    echo "Please ensure the submodule is initialized: git submodule update --init --recursive"
    echo "And that the required ggml source files are present in 3rdparty/llama.cpp/ggml/src/"
    exit 1
fi


# Prepare bitnet-lut-kernels.h by copying a preset one
# For this example, we use the tl1 kernel from bitnet_b1_58-3B preset.
# Adjust this path if you want to use a different kernel.
PRESET_KERNEL_HEADER="preset_kernels/bitnet_b1_58-3B/bitnet-lut-kernels-tl1.h"
TARGET_KERNEL_HEADER="include/bitnet-lut-kernels.h"

if [ ! -f "$PRESET_KERNEL_HEADER" ]; then
    echo "Error: Preset kernel header $PRESET_KERNEL_HEADER not found."
    exit 1
fi

echo "Copying $PRESET_KERNEL_HEADER to $TARGET_KERNEL_HEADER"
cp "$PRESET_KERNEL_HEADER" "$TARGET_KERNEL_HEADER"

echo "Compiling with Emscripten..."
emcc $EMCC_FLAGS $INCLUDE_DIRS $BITNET_SOURCES $LLAMA_GGML_SOURCES -o $OUTPUT_JS_FILE > emcc_stdout.log 2> emcc_stderr.log
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
    echo "To use it, you'll typically load $OUTPUT_JS_FILE in your HTML/JavaScript."
    echo "You'll need to define which C functions are callable from JavaScript using EXPORTED_FUNCTIONS."
    echo "Example: -s EXPORTED_FUNCTIONS=['_my_function', '_another_function']"
    echo "(Remember to add the underscore prefix for C functions)"
else
    echo "Build failed. Please check the output from emcc."
    exit 1
fi

echo "Build script finished."
