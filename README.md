# BitNet-WASM: WebAssembly Package for BitNet Operations

This project provides a build process to compile parts of the BitNet-wasm C++ code into a WebAssembly (WASM) package.
The build uses Docker and the Emscripten SDK.

## Build Process

The build is orchestrated by the `build.sh` script.

### Prerequisites

*   Docker
*   A Linux-like environment with bash (the script uses `sudo dockerd` if Docker is not running, which might require passwordless sudo or manual Docker daemon startup).

### Steps

1.  **Clone the Repository (if not already done for this session):**
    ```bash
    # git clone https://github.com/jerfletcher/BitNet-wasm.git
    # cd BitNet-wasm
    # git submodule update --init --recursive 3rdparty/llama.cpp
    # (Note: The build script currently uses llama.cpp commit 5eb47b72 from the submodule)
    ```

2.  **Run the Build Script:**
    From the root of the `BitNet-wasm` directory:
    ```bash
    ./build.sh
    ```
    This script performs the following actions:
    *   Sets up the Emscripten Docker environment (`emscripten/emsdk:4.0.8`).
    *   Copies necessary kernel header files.
    *   Compiles the C/C++ sources (`src/*.cpp`, `3rdparty/llama.cpp/ggml/src/*.c`, `3rdparty/llama.cpp/ggml/src/*.cpp`) using `emcc`.
    *   Generates `bitnet.wasm` (the WASM module) and `bitnet.js` (the JavaScript loader/glue code) in the root directory.
    *   Logs stdout and stderr from `emcc` to `emcc_stdout.log` and `emcc_stderr.log` respectively.

## WASM Module Interface

The compiled WASM module (`bitnet.wasm` loaded via `bitnet.js`) exports the following C functions.
You can call these from JavaScript using `Module.ccall` or `Module.cwrap` after the module is loaded. Remember to prefix C function names with an underscore when using `EXPORTED_FUNCTIONS` or directly accessing them on the `Module` object (e.g., `Module._ggml_init`).

### Exported Functions

*   `void ggml_init(struct ggml_init_params params)`
    *   **C Signature:** `void ggml_init(struct ggml_init_params params);` (from `ggml.h`)
    *   **JS Access Example:** `Module._ggml_init(0);` (passing NULL for params)
    *   **Note:** `struct ggml_init_params` would need to be created in WASM memory if non-default initialization is needed. `params_ptr` would be a pointer to this structure.
    *   **Status:** Available.

*   `int64_t ggml_nelements(const struct ggml_tensor * tensor)`
    *   **C Signature:** `int64_t ggml_nelements(const struct ggml_tensor * tensor);` (from `ggml.h`)
    *   **JS Access Example:** `Module._ggml_nelements(tensor_ptr)`
    *   **Note:** Requires a valid `ggml_tensor` pointer. Creating and managing tensors is currently problematic (see "Limitations").
    *   **Status:** Available (but of limited use without tensor creation).

*   `void ggml_bitnet_init(void)`
    *   **C Signature:** `void ggml_bitnet_init(void);` (from `ggml-bitnet.h`, defined in `ggml-bitnet-lut.cpp`)
    *   **JS Access Example:** `Module._ggml_bitnet_init()`
    *   **Status:** Available. Initializes BitNet specific internal states (related to LUT kernels).

*   `void ggml_bitnet_free(void)`
    *   **C Signature:** `void ggml_bitnet_free(void);` (from `ggml-bitnet.h`, defined in `ggml-bitnet-lut.cpp`)
    *   **JS Access Example:** `Module._ggml_bitnet_free()`
    *   **Status:** Available. Frees resources allocated by `ggml_bitnet_init`.

### Limitations and Missing Functions

The current WASM build has significant limitations due to issues within the `BitNet-wasm` source code and its `llama.cpp` submodule (commit `5eb47b72`) when targeting WebAssembly:

1.  **Core BitNet Functionality Missing:**
    *   `ggml_bitnet_mul_mat`: This crucial function for BitNet matrix multiplication is **declared but not defined** in the provided C++ sources. It cannot be exported or used.
    *   `ggml_bitnet_transform_tensor`: This function, also declared in `ggml-bitnet.h`, is **not defined** in the provided sources and cannot be used.

2.  **General `ggml` Functionality Issues:**
    *   Many standard `ggml` functions essential for practical use (e.g., `ggml_new_context`, `ggml_free` (for context), `ggml_new_tensor_1d/2d/3d`, `ggml_get_data`, `ggml_set_f32_flat`) fail to link when included in `EXPORTED_FUNCTIONS`.
    *   These failures appear to be due to two main reasons:
        *   Linker errors: "symbol exported via --export not found" for the functions themselves, suggesting they are not being correctly picked up from the compiled `ggml.c` object file or are being optimized out before the final link stage for WASM under the current flags.
        *   Linker errors: "undefined symbol" for various architecture-specific quantization kernels (e.g., `quantize_mat_q8_0`, `ggml_gemv_q4_0_4x4_q8_0`). These are pulled in as dependencies by the more complex `ggml` functions. The `llama.cpp` commit `5eb47b72` does not seem to provide or enable generic C fallbacks for these kernels when compiling for the WASM target with the current build configuration.

**As a result, the current WASM package can only offer very basic initialization and de-initialization functions. It cannot create or manage tensors, nor can it perform BitNet computations.**

To achieve a functional WASM package for `BitNet-wasm`, the underlying C++ source code would need to be addressed:
*   Provide complete definitions for `ggml_bitnet_mul_mat` and `ggml_bitnet_transform_tensor` within the `BitNet-wasm/src` files.
*   Resolve the linking issues for standard `ggml` functions. This might involve:
    *   Investigating symbol visibility and linkage of functions in `ggml.c` when compiled to WASM.
    *   Modifying `ggml` (e.g., `ggml-quants.c`) to include or enable generic C implementations for the missing quantization kernels, or stubbing them if they are not strictly necessary for the desired BitNet operations on WASM.

## Example Usage

An example demonstrating how to load the module and call the few available functions is provided in the `example/` directory:
*   `example/index.html`
*   `example/main.js`

Due to the limitations mentioned above, this example is minimal and primarily shows successful module loading and calls to init/free functions.
