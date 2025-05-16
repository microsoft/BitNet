// example/main.js

// This function will be called when the WASM module is loaded
function onRuntimeInitialized() {
    console.log('BitNet WASM Module Loaded.');
    const outputElement = document.getElementById('output');
    outputElement.innerHTML += 'BitNet WASM Module Loaded.<br>';

    // --- Demonstrate calling available functions ---

    // 1. ggml_init
    // C Signature: void ggml_init(struct ggml_init_params params);
    // We\'ll pass 0 (null) for params for default initialization.
    try {
        console.log('Calling Module._ggml_init(0)...');
        outputElement.innerHTML += 'Calling Module._ggml_init(0)...<br>';
        Module._ggml_init(0); // Pass 0 for NULL params
        console.log('Module._ggml_init successfully called.');
        outputElement.innerHTML += 'Module._ggml_init successfully called.<br>';
    } catch (e) {
        console.error('Error calling Module._ggml_init:', e);
        outputElement.innerHTML += `Error calling Module._ggml_init: \${e}<br>`;
    }

    // 2. ggml_bitnet_init
    // C Signature: void ggml_bitnet_init(void);
    try {
        console.log('Calling Module._ggml_bitnet_init()...');
        outputElement.innerHTML += 'Calling Module._ggml_bitnet_init()...<br>';
        Module._ggml_bitnet_init();
        console.log('Module._ggml_bitnet_init successfully called.');
        outputElement.innerHTML += 'Module._ggml_bitnet_init successfully called.<br>';
    } catch (e) {
        console.error('Error calling Module._ggml_bitnet_init:', e);
        outputElement.innerHTML += `Error calling Module._ggml_bitnet_init: \${e}<br>\`;
    }

    // 3. ggml_nelements (demonstrative, as we can\'t create a real tensor)
    // C Signature: int64_t ggml_nelements(const struct ggml_tensor * tensor);
    // Since we cannot create a ggml_tensor due to missing ggml_new_tensor* functions,
    // calling this with a dummy pointer (e.g., 0) will likely crash or error.
    // This is just to show how it *would* be called if tensor creation was possible.
    console.log('Attempting to call Module._ggml_nelements(0) (expected to fail or be meaningless)...');
    outputElement.innerHTML += 'Attempting to call Module._ggml_nelements(0) (expected to fail or be meaningless)...<br>';
    try {
        const num_elements = Module._ggml_nelements(0); // Passing 0 for a NULL tensor pointer
        console.log('Module._ggml_nelements(0) returned:', num_elements, '(This value is likely meaningless as no valid tensor was passed)');
        outputElement.innerHTML += \`Module._ggml_nelements(0) returned: \${num_elements} (This value is likely meaningless as no valid tensor was passed)<br>\`;
    } catch (e) {
        console.error('Error calling Module._ggml_nelements (as expected with NULL tensor):', e);
        outputElement.innerHTML += \`Error calling Module._ggml_nelements (as expected with NULL tensor): \${e}<br>\`;
    }

    // 4. ggml_bitnet_transform_tensor (calling STUB)
    console.log('Attempting to call Module._ggml_bitnet_transform_tensor(0) (STUBBED FUNCTION)...');
    outputElement.innerHTML += 'Attempting to call Module._ggml_bitnet_transform_tensor(0) (STUBBED FUNCTION)...<br>';
    try {
        Module._ggml_bitnet_transform_tensor(0); // Pass 0 for NULL tensor
        console.log('Module._ggml_bitnet_transform_tensor(0) called (STUB).');
        outputElement.innerHTML += 'Module._ggml_bitnet_transform_tensor(0) called (STUB).<br>';
    } catch (e) {
        console.error('Error calling Module._ggml_bitnet_transform_tensor:', e);
        outputElement.innerHTML += \`Error calling Module._ggml_bitnet_transform_tensor: \${e}<br>\`;
    }

    // 5. ggml_bitnet_mul_mat_task_compute (calling STUB)
    console.log('Attempting to call Module._ggml_bitnet_mul_mat_task_compute (STUBBED FUNCTION)...');
    outputElement.innerHTML += 'Attempting to call Module._ggml_bitnet_mul_mat_task_compute (STUBBED FUNCTION)...<br>';
    try {
        // Call with dummy parameters: (src0, scales, qlut, lut_scales, lut_biases, dst, n, k, m, bits)
        Module._ggml_bitnet_mul_mat_task_compute(0, 0, 0, 0, 0, 0, 1, 1, 1, 2);
        console.log('Module._ggml_bitnet_mul_mat_task_compute called (STUB).');
        outputElement.innerHTML += 'Module._ggml_bitnet_mul_mat_task_compute called (STUB).<br>';
    } catch (e) {
        console.error('Error calling Module._ggml_bitnet_mul_mat_task_compute:', e);
        outputElement.innerHTML += \`Error calling Module._ggml_bitnet_mul_mat_task_compute: \${e}<br>\`;
    }

    // --- Status Note ---
    const statusMessage = \`
        <p><strong>CURRENT STATUS & NEXT STEPS:</strong></p>
        <p>The BitNet WASM module (bitnet.js & bitnet.wasm) has been successfully built.</p>
        <ul>
            <li>The build system was updated, and Emscripten SDK is used for compilation.</li>
            <li>Key C++ functions <code>ggml_bitnet_mul_mat_task_compute</code> and <code>ggml_bitnet_transform_tensor</code>, which were previously undefined, have been added as STUBS.</li>
            <li><strong>IMPORTANT:</strong> These stubs allow the module to load and call these functions, but they DO NOT perform the actual BitNet computations. They mostly do nothing or zero out memory.</li>
            <li>Many ggml functions should now be available for use from JavaScript due to the <code>-s EXPORT_ALL=1</code> flag used during compilation.</li>
        </ul>
        <p><strong>To achieve functional BitNet inference:</strong></p>
        <ol>
            <li>The C++ stub implementations for <code>ggml_bitnet_mul_mat_task_compute</code> and <code>ggml_bitnet_transform_tensor</code> in <code>src/ggml-bitnet-lut.cpp</code> MUST be replaced with their correct algorithmic logic.</li>
            <li>A higher-level C/C++ API for model loading, context management, and inference execution needs to be defined and then called from this JavaScript example. This would involve using various <code>ggml_*</code> functions to construct and evaluate a computation graph.</li>
            <li>An example BitNet model in GGUF format (or a compatible format) would be needed for testing.</li>
        </ol>
        <p>The calls below demonstrate that the initialization/deinitialization functions and the newly stubbed functions are callable.</p>
    \`;
    console.info(statusMessage.replace(/<[^>]*>/g, '\\n').replace(/\\n\\n+/g, '\\n')); // Log plain text version
    outputElement.innerHTML += statusMessage;


    // 4. ggml_bitnet_free
    // C Signature: void ggml_bitnet_free(void);
    try {
        console.log('Calling Module._ggml_bitnet_free()...');
        outputElement.innerHTML += 'Calling Module._ggml_bitnet_free()...<br>';
        Module._ggml_bitnet_free();
        console.log('Module._ggml_bitnet_free successfully called.');
        outputElement.innerHTML += 'Module._ggml_bitnet_free successfully called.<br>';
    } catch (e) {
        console.error('Error calling Module._ggml_bitnet_free:', e);
        outputElement.innerHTML += \`Error calling Module._ggml_bitnet_free: \${e}<br>\`;
    }

    console.log('--- End of WASM function demonstration ---');
    outputElement.innerHTML += '--- End of WASM function demonstration ---<br>';
    // Update status on the page
    const statusElement = document.getElementById('status');
    if (statusElement) {
        statusElement.textContent = 'WASM module loaded and example functions called. See details below and in console.';
    }
}

// Emscripten module configuration
// The \`bitnet.js\` glue code (generated by Emscripten) will look for a global \`Module\` object.
var Module = {
    onRuntimeInitialized: onRuntimeInitialized,
    print: function(text) {
        console.log('[WASM stdout]', text);
        const outputElement = document.getElementById('output');
        if (outputElement) outputElement.innerHTML += \`[WASM stdout] \${text}<br>\`;
    },
    printErr: function(text) {
        console.error('[WASM stderr]', text);
        const outputElement = document.getElementById('output');
        if (outputElement) outputElement.innerHTML += \`[WASM stderr] \${text}<br>\`;
    }
};

// The index.html will load bitnet.js, which in turn loads bitnet.wasm
// and then calls Module.onRuntimeInitialized.
console.log('main.js loaded. Waiting for bitnet.js to initialize the WASM module...');
const initialStatus = document.getElementById('status');
if (initialStatus) initialStatus.textContent = 'main.js loaded. Loading bitnet.js and bitnet.wasm...';
const initialOutput = document.getElementById('output');
if (initialOutput) initialOutput.innerHTML += 'main.js loaded. Waiting for bitnet.js to initialize the WASM module...<br>';
