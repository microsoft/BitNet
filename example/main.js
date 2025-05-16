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
        outputElement.innerHTML += \`Error calling Module._ggml_init: \${e}<br>\`;
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
        outputElement.innerHTML += \`Error calling Module._ggml_bitnet_init: \${e}<br>\`;
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

    // --- Limitations Note ---
    const limitationsMessage = \`
        <p><strong>IMPORTANT LIMITATIONS:</strong></p>
        <p>The current WASM module is severely limited due to issues in the underlying C++ source
        and its llama.cpp submodule (commit 5eb47b72) when targeting WebAssembly.</p>
        <ol>
            <li><strong>Missing Core BitNet Functions:</strong>
                <ul>
                    <li><code>ggml_bitnet_mul_mat</code>: Declared but NOT DEFINED. Cannot be called.</li>
                    <li><code>ggml_bitnet_transform_tensor</code>: Declared but NOT DEFINED. Cannot be called.</li>
                </ul>
            </li>
            <li><strong>GGML Function Linker Issues:</strong>
                <ul>
                    <li>Functions like <code>ggml_new_context</code>, <code>ggml_free</code> (context), <code>ggml_new_tensor_1d/2d/3d</code>,
                        <code>ggml_get_data</code>, etc., could NOT be successfully exported.</li>
                    <li>This is due to:
                        <ol type="a">
                            <li>"symbol exported via --export not found" errors for the functions themselves.</li>
                            <li>"undefined symbol" errors for quantization kernels (e.g., <code>quantize_mat_q8_0</code>)
                                that are dependencies of these ggml functions. The llama.cpp version used
                                does not seem to provide generic C fallbacks for these when building for WASM.</li>
                        </ol>
                    </li>
                </ul>
            </li>
        </ol>
        <p><strong>CONCLUSION:</strong></p>
        <p>This WASM module can initialize and free some BitNet-related internal structures
        (<code>_ggml_bitnet_init</code>, <code>_ggml_bitnet_free</code>) and call the basic <code>_ggml_init</code>.
        However, it CANNOT:
        <ul>
            <li>Create or manage <code>ggml_context</code> or <code>ggml_tensor</code> objects.</li>
            <li>Perform any actual BitNet computations (like matrix multiplication).</li>
            <li>Be used for any meaningful inference tasks in its current state.</li>
        </ul>
        </p>
        <p>Resolving these issues would require significant C++ development to define the missing
        BitNet functions and to fix the ggml linkage problems for the WASM target.</p>
    \`;
    console.warn(limitationsMessage.replace(/<[^>]*>/g, \'\\n\').replace(/\\n\\n+/g, \'\\n\')); // Log plain text version
    outputElement.innerHTML += limitationsMessage;


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
