// example/main.js
import ModuleFactory from '../bitnet.js';

// This function will be called when the WASM module is loaded and initialized
function onWasmInitialized(wasmModuleInstance) {
    console.log('BitNet WASM Module Initialized and Ready.');
    const outputElement = document.getElementById('output');
    outputElement.innerHTML = 'BitNet WASM Module Initialized and Ready.<br>'; // Clear previous messages

    // --- Demonstrate calling available functions ---
    // (Rest of the function is the same as before, using wasmModuleInstance)
    // 1. ggml_init
    try {
        console.log('Calling wasmModuleInstance._ggml_init(0)...');
        outputElement.innerHTML += 'Calling wasmModuleInstance._ggml_init(0)...<br>';
        wasmModuleInstance._ggml_init(0); // Pass 0 for NULL params
        console.log('wasmModuleInstance._ggml_init successfully called.');
        outputElement.innerHTML += 'wasmModuleInstance._ggml_init successfully called.<br>';
    } catch (e) {
        console.error('Error calling wasmModuleInstance._ggml_init:', e);
        outputElement.innerHTML += `Error calling wasmModuleInstance._ggml_init: ${e}<br>`;
    }

    // 2. ggml_bitnet_init
    try {
        console.log('Calling wasmModuleInstance._ggml_bitnet_init()...');
        outputElement.innerHTML += 'Calling wasmModuleInstance._ggml_bitnet_init()...<br>';
        wasmModuleInstance._ggml_bitnet_init();
        console.log('wasmModuleInstance._ggml_bitnet_init successfully called.');
        outputElement.innerHTML += 'wasmModuleInstance._ggml_bitnet_init successfully called.<br>';
    } catch (e) {
        console.error('Error calling wasmModuleInstance._ggml_bitnet_init:', e);
        outputElement.innerHTML += `Error calling wasmModuleInstance._ggml_bitnet_init: ${e}<br>`;
    }

    // 3. ggml_nelements
    console.log('Attempting to call wasmModuleInstance._ggml_nelements(0) (expected to fail or be meaningless)...');
    outputElement.innerHTML += 'Attempting to call wasmModuleInstance._ggml_nelements(0) (expected to fail or be meaningless)...<br>';
    try {
        const num_elements = wasmModuleInstance._ggml_nelements(0);
        console.log('wasmModuleInstance._ggml_nelements(0) returned:', num_elements, '(This value is likely meaningless as no valid tensor was passed)');
        outputElement.innerHTML += `wasmModuleInstance._ggml_nelements(0) returned: ${num_elements} (This value is likely meaningless as no valid tensor was passed)<br>`;
    } catch (e) {
        console.error('Error calling wasmModuleInstance._ggml_nelements (as expected with NULL tensor):', e);
        outputElement.innerHTML += `Error calling wasmModuleInstance._ggml_nelements (as expected with NULL tensor): ${e}<br>`;
    }

    // 4. ggml_bitnet_transform_tensor (calling STUB)
    console.log('Attempting to call wasmModuleInstance._ggml_bitnet_transform_tensor(0) (STUBBED FUNCTION)...');
    outputElement.innerHTML += 'Attempting to call wasmModuleInstance._ggml_bitnet_transform_tensor(0) (STUBBED FUNCTION)...<br>';
    try {
        wasmModuleInstance._ggml_bitnet_transform_tensor(0); // Pass 0 for NULL tensor
        console.log('wasmModuleInstance._ggml_bitnet_transform_tensor(0) called (STUB).');
        outputElement.innerHTML += 'wasmModuleInstance._ggml_bitnet_transform_tensor(0) called (STUB).<br>';
    } catch (e) {
        console.error('Error calling wasmModuleInstance._ggml_bitnet_transform_tensor:', e);
        outputElement.innerHTML += `Error calling wasmModuleInstance._ggml_bitnet_transform_tensor: ${e}<br>`;
    }

    // 5. ggml_bitnet_mul_mat_task_compute (calling STUB)
    console.log('Attempting to call wasmModuleInstance._ggml_bitnet_mul_mat_task_compute (STUBBED FUNCTION)...');
    outputElement.innerHTML += 'Attempting to call wasmModuleInstance._ggml_bitnet_mul_mat_task_compute (STUBBED FUNCTION)...<br>';
    try {
        wasmModuleInstance._ggml_bitnet_mul_mat_task_compute(0, 0, 0, 0, 0, 0, 1, 1, 1, 2);
        console.log('wasmModuleInstance._ggml_bitnet_mul_mat_task_compute called (STUB).');
        outputElement.innerHTML += 'wasmModuleInstance._ggml_bitnet_mul_mat_task_compute called (STUB).<br>';
    } catch (e) {
        console.error('Error calling wasmModuleInstance._ggml_bitnet_mul_mat_task_compute:', e);
        outputElement.innerHTML += `Error calling wasmModuleInstance._ggml_bitnet_mul_mat_task_compute: ${e}<br>`;
    }
    
    // --- Status Note ---
    const statusMessage = `
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
    `;
    console.info(statusMessage.replace(/<[^>]*>/g, '\n').replace(/\n\n+/g, '\n'));
    outputElement.innerHTML += statusMessage;

    // 6. ggml_bitnet_free
    try {
        console.log('Calling wasmModuleInstance._ggml_bitnet_free()...');
        outputElement.innerHTML += 'Calling wasmModuleInstance._ggml_bitnet_free()...<br>';
        wasmModuleInstance._ggml_bitnet_free();
        console.log('wasmModuleInstance._ggml_bitnet_free successfully called.');
        outputElement.innerHTML += 'wasmModuleInstance._ggml_bitnet_free successfully called.<br>';
    } catch (e) {
        console.error('Error calling wasmModuleInstance._ggml_bitnet_free:', e);
        outputElement.innerHTML += `Error calling wasmModuleInstance._ggml_bitnet_free: ${e}<br>`;
    }

    console.log('--- End of WASM function demonstration ---');
    outputElement.innerHTML += '--- End of WASM function demonstration ---<br>';
    const statusElement = document.getElementById('status');
    if (statusElement) {
        statusElement.textContent = 'WASM module initialized and example functions called. See details below and in console.';
    }
}

// Emscripten module configuration object
const moduleConfig = {
    print: function(text) {
        console.log('[WASM stdout]', text);
        const outputElement = document.getElementById('output');
        if (outputElement) outputElement.innerHTML += `[WASM stdout] ${text}<br>`;
    },
    printErr: function(text) {
        console.error('[WASM stderr]', text);
        const outputElement = document.getElementById('output');
        if (outputElement) outputElement.innerHTML += `[WASM stderr] ${text}<br>`;
    }
    // No 'locateFile' needed if .wasm is in the same directory as .js,
    // or if using SINGLE_FILE=1 (which we are not here).
    // Emscripten will try to fetch 'bitnet.wasm' relative to 'bitnet.js'
};

const initialStatus = document.getElementById('status');
const initialOutput = document.getElementById('output');

if (initialStatus) initialStatus.textContent = 'main.js (ESM) loaded. Importing Module factory from bitnet.js...';
if (initialOutput) initialOutput.innerHTML = 'main.js (ESM) loaded. Importing Module factory from bitnet.js...<br>';


function initializeWasm() {
    if (typeof ModuleFactory === 'function') {
        console.log('Module factory imported successfully. Initializing WASM...');
        if (initialOutput) initialOutput.innerHTML += 'Module factory imported successfully. Initializing WASM...<br>';
        
        ModuleFactory(moduleConfig).then((initializedInstance) => {
            onWasmInitialized(initializedInstance);
        }).catch(e => {
            console.error("Error initializing WASM module:", e);
            if (initialOutput) initialOutput.innerHTML += `Error initializing WASM module: ${e}<br>`;
            if (initialStatus) initialStatus.textContent = 'Error initializing WASM module.';
        });
    } else {
        // This case should ideally not be hit if ES6 imports work as expected
        // and bitnet.js correctly exports the factory.
        const currentTypeOfModuleFactory = typeof ModuleFactory;
        console.error(`Module factory not available or not a function (current type: ${currentTypeOfModuleFactory}). Check import and bitnet.js export.`);
        if (initialOutput) {
             initialOutput.innerHTML += `Module factory not available or not a function (current type: ${currentTypeOfModuleFactory}). Check import and bitnet.js export.<br>`;
        }
        if (initialStatus) initialStatus.textContent = 'Error: Module factory not found.';
    }
}

// Since main.js is a module, top-level await is not universally supported without specific server/browser configs.
// We rely on the static import to make ModuleFactory available.
// DOMContentLoaded is still useful to ensure elements are ready before manipulation.
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeWasm);
} else {
    initializeWasm();
}
