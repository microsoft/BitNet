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

## Current Status & Limitations

The build process has been updated:
*   The Emscripten SDK (currently v4.0.8) is now expected to be installed in the environment (e.g., in `/tmp/emsdk`). The `build.sh` script sources `emsdk_env.sh` and calls `emcc` directly. The Docker setup is still available in `build.sh` but commented out.
*   The `build.sh` script now uses the `-s EXPORT_ALL=1` Emscripten flag. This means most C functions from the compiled sources (including many `ggml_*` functions) are exported and available on the `Module` object in JavaScript (prefixed with an underscore).

**Build Artifacts:**
*   `bitnet.js` (JavaScript glue code)
*   `bitnet.wasm` (WebAssembly module)

**Key Available Functions (callable from JavaScript via `Module._functionName`):**
*   `ggml_init`
*   `ggml_bitnet_init`
*   `ggml_bitnet_free`
*   `ggml_nelements`
*   `ggml_bitnet_transform_tensor` (**STUBBED** - see below)
*   `ggml_bitnet_mul_mat_task_compute` (**STUBBED** - see below)
*   Many other `ggml_*` functions (due to `EXPORT_ALL=1`).

**Critical Limitations:**
1.  **Core BitNet Functions are STUBBED:**
    *   The essential C++ functions `ggml_bitnet_mul_mat_task_compute` and `ggml_bitnet_transform_tensor` (defined in `src/ggml-bitnet-lut.cpp`) are currently **placeholders (stubs)**.
    *   They were added to allow the project to compile and link successfully.
    *   **These stubs DO NOT perform any actual BitNet computations.** They will not produce correct results for inference.
2.  **No High-Level Inference API:**
    *   There isn't a simple C or JavaScript API yet to load a model, prepare input, run inference, and get output.

**Conclusion:** The WASM module can be built, loaded, and basic initialization/stub functions can be called. However, **it CANNOT perform any meaningful BitNet inference in its current state.**

## Next Steps for Full Functionality

To make this project fully capable of BitNet inference in WebAssembly, the following steps are crucial:

1.  **Implement Core BitNet C++ Functions:**
    *   The placeholder C++ implementations for `ggml_bitnet_mul_mat_task_compute` and `ggml_bitnet_transform_tensor` (located in `src/ggml-bitnet-lut.cpp`) **MUST** be replaced with their correct algorithmic logic based on the BitNet paper and `ggml` integration. This is the most critical step.

2.  **Develop/Expose High-Level Inference API:**
    *   Define and implement higher-level C/C++ functions that orchestrate the inference process. This would typically involve:
        *   Loading a model (e.g., from a GGUF file if leveraging `llama.cpp`'s `ggml` loading capabilities).
        *   Creating a `ggml_context` and managing `ggml_tensor` objects for weights, inputs, and computations.
        *   Building a `ggml_cgraph` (computation graph) that uses the BitNet-specific functions and other `ggml` operations.
        *   Executing the graph.
        *   Providing functions to set input data and retrieve output data.
    *   Ensure these high-level functions are exported for JavaScript access.

3.  **Update JavaScript Example:**
    *   Modify `example/main.js` to use the new high-level C/WASM API to:
        *   Load a BitNet model.
        *   Prepare sample input.
        *   Run inference.
        *   Display the results.

4.  **Model Preparation:**
    *   Obtain or convert a BitNet model into a format compatible with the chosen loading mechanism (e.g., GGUF if using `ggml`'s standard model loading).

5.  **Thorough Testing:**
    *   Verify the correctness of the implemented BitNet operations and the end-to-end inference pipeline against known test cases or reference implementations if available.

## Example Usage

An example demonstrating how to load the module and call some of the available functions (including the STUBBED BitNet functions) is provided in the `example/` directory:
*   `example/index.html`
*   `example/main.js` (updated to reflect current status and call stubs)

To run the example:
1.  Ensure `bitnet.js` and `bitnet.wasm` are built using `./build.sh`.
2.  Start a simple HTTP server in the `BitNet-wasm` root directory (e.g., `python3 -m http.server`).
3.  Open `http://localhost:PORT/example/index.html` in your browser.

**Note:** This example primarily shows that the WASM module loads and that the exported C functions (including the stubs) are callable from JavaScript. It does not perform any real inference.

