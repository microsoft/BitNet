// example/main.js
import ModuleFactory from '../bitnet.js';

// Global variable to store the WASM module instance
let wasmModule = null;

// Helper function to parse comma-separated values into a Float32Array
function parseFloatArray(text) {
    return new Float32Array(
        text.split(',')
            .map(s => s.trim())
            .filter(s => s.length > 0)
            .map(s => parseFloat(s))
    );
}

// Helper function to create a matrix from a Float32Array
function createMatrix(data, rows, cols) {
    if (data.length !== rows * cols) {
        throw new Error(`Data length ${data.length} does not match dimensions ${rows}x${cols}`);
    }
    
    return {
        data: data,
        rows: rows,
        cols: cols
    };
}

// Helper function to allocate memory in the WASM heap
function allocateFloat32Array(array) {
    const bytes = array.length * Float32Array.BYTES_PER_ELEMENT;
    const ptr = wasmModule._malloc(bytes);
    const heap = new Float32Array(wasmModule.HEAPF32.buffer, ptr, array.length);
    heap.set(array);
    return { ptr, length: array.length, free: () => wasmModule._free(ptr) };
}

// Helper function to allocate memory for int8 array in the WASM heap
function allocateInt8Array(array) {
    const bytes = array.length * Int8Array.BYTES_PER_ELEMENT;
    const ptr = wasmModule._malloc(bytes);
    const heap = new Int8Array(wasmModule.HEAP8.buffer, ptr, array.length);
    heap.set(array);
    return { ptr, length: array.length, free: () => wasmModule._free(ptr) };
}

// Helper function to read a Float32Array from the WASM heap
function readFloat32Array(ptr, length) {
    return new Float32Array(wasmModule.HEAPF32.buffer.slice(ptr, ptr + length * Float32Array.BYTES_PER_ELEMENT));
}

// Helper function to read an Int8Array from the WASM heap
function readInt8Array(ptr, length) {
    return new Int8Array(wasmModule.HEAP8.buffer.slice(ptr, ptr + length * Int8Array.BYTES_PER_ELEMENT));
}

// Function to perform matrix multiplication using BitNet
function performMatrixMultiplication(matrixA, matrixB) {
    // Allocate memory for input matrices
    const inputPtr = allocateFloat32Array(matrixA.data);
    const weightsPtr = allocateFloat32Array(matrixB.data);
    
    // Allocate memory for output matrix
    const outputData = new Float32Array(matrixA.rows * matrixB.cols);
    const outputPtr = allocateFloat32Array(outputData);
    
    // Allocate memory for quantized weights
    const qWeightsData = new Int8Array(matrixB.rows * matrixB.cols);
    const qWeightsPtr = allocateInt8Array(qWeightsData);
    
    // Allocate memory for scales
    const scalesData = new Float32Array(matrixB.rows);
    scalesData.fill(1.0); // Default scale
    const scalesPtr = allocateFloat32Array(scalesData);
    
    // Allocate memory for LUT scales
    const lutScalesData = new Float32Array(1);
    lutScalesData[0] = 127.0; // Default LUT scale
    const lutScalesPtr = allocateFloat32Array(lutScalesData);
    
    // Allocate memory for LUT biases (not used in this implementation)
    const lutBiasesData = new Float32Array(1);
    lutBiasesData[0] = 0.0;
    const lutBiasesPtr = allocateFloat32Array(lutBiasesData);
    
    try {
        // Call the BitNet matrix multiplication function
        wasmModule._ggml_bitnet_mul_mat_task_compute(
            inputPtr.ptr,
            scalesPtr.ptr,
            qWeightsPtr.ptr,
            lutScalesPtr.ptr,
            lutBiasesPtr.ptr,
            outputPtr.ptr,
            matrixA.rows,
            matrixA.cols,
            matrixB.cols,
            2 // 2-bit quantization
        );
        
        // Read the result
        const result = readFloat32Array(outputPtr.ptr, matrixA.rows * matrixB.cols);
        return createMatrix(result, matrixA.rows, matrixB.cols);
    } finally {
        // Free allocated memory
        inputPtr.free();
        weightsPtr.free();
        outputPtr.free();
        qWeightsPtr.free();
        scalesPtr.free();
        lutScalesPtr.free();
        lutBiasesPtr.free();
    }
}

// Function to transform a tensor using BitNet quantization
function transformTensor(tensorData) {
    // Create a dummy tensor structure
    const rows = 1;
    const cols = tensorData.length;
    
    // Allocate memory for the tensor data
    const dataPtr = allocateFloat32Array(tensorData);
    
    // Create a simple tensor structure in WASM memory
    const tensorPtr = wasmModule._malloc(32); // Simplified tensor structure
    
    // Set tensor dimensions
    wasmModule.HEAP32[tensorPtr/4] = cols; // ne[0]
    wasmModule.HEAP32[tensorPtr/4 + 1] = rows; // ne[1]
    wasmModule.HEAP32[tensorPtr/4 + 2] = 1; // ne[2]
    wasmModule.HEAP32[tensorPtr/4 + 3] = 1; // ne[3]
    
    // Set tensor data pointer
    wasmModule.HEAP32[tensorPtr/4 + 4] = dataPtr.ptr;
    
    try {
        // Call the BitNet tensor transformation function
        wasmModule._ggml_bitnet_transform_tensor(tensorPtr);
        
        // For demonstration purposes, we'll just return the original data
        // In a real implementation, we would access the quantized data from the tensor
        return {
            original: Array.from(tensorData),
            message: "Tensor transformed successfully. In a real implementation, we would return the quantized data."
        };
    } finally {
        // Free allocated memory
        dataPtr.free();
        wasmModule._free(tensorPtr);
    }
}

// This function will be called when the WASM module is loaded and initialized
function onWasmInitialized(wasmModuleInstance) {
    console.log('BitNet WASM Module Initialized and Ready.');
    wasmModule = wasmModuleInstance;
    
    const outputElement = document.getElementById('output');
    const statusElement = document.getElementById('status');
    
    outputElement.innerHTML = 'BitNet WASM Module Initialized and Ready.<br>';
    
    if (statusElement) {
        statusElement.textContent = 'WASM module initialized. Ready to use BitNet functions.';
        statusElement.classList.add('success');
    }

    // Initialize BitNet
    try {
        console.log('Initializing BitNet...');
        outputElement.innerHTML += 'Initializing BitNet...<br>';
        
        wasmModule._ggml_init(0);
        wasmModule._ggml_bitnet_init();
        
        console.log('BitNet initialized successfully.');
        outputElement.innerHTML += 'BitNet initialized successfully.<br>';
    } catch (e) {
        console.error('Error initializing BitNet:', e);
        outputElement.innerHTML += `Error initializing BitNet: ${e}<br>`;
        if (statusElement) {
            statusElement.textContent = 'Error initializing BitNet.';
            statusElement.classList.add('error');
        }
        return;
    }
    
    // Set up event listeners for the demo buttons
    setupMatrixMultiplicationDemo();
    setupTensorTransformationDemo();
}

// Set up the matrix multiplication demo
function setupMatrixMultiplicationDemo() {
    const runButton = document.getElementById('run-matmul');
    if (!runButton) return;
    
    runButton.addEventListener('click', () => {
        const matrixAInput = document.getElementById('matrix-a').value;
        const matrixBInput = document.getElementById('matrix-b').value;
        const resultElement = document.getElementById('matmul-result');
        
        try {
            // Parse input matrices
            const matrixAData = parseFloatArray(matrixAInput);
            const matrixBData = parseFloatArray(matrixBInput);
            
            // Determine matrix dimensions (assuming square matrices for simplicity)
            const size = Math.sqrt(matrixAData.length);
            if (!Number.isInteger(size) || !Number.isInteger(Math.sqrt(matrixBData.length))) {
                throw new Error('Input matrices must be square for this demo');
            }
            
            const matrixA = createMatrix(matrixAData, size, size);
            const matrixB = createMatrix(matrixBData, size, size);
            
            // Perform matrix multiplication
            const result = performMatrixMultiplication(matrixA, matrixB);
            
            // Display the result
            let resultHTML = '<h4>Result Matrix:</h4><pre>';
            for (let i = 0; i < result.rows; i++) {
                for (let j = 0; j < result.cols; j++) {
                    resultHTML += result.data[i * result.cols + j].toFixed(4) + '\t';
                }
                resultHTML += '\n';
            }
            resultHTML += '</pre>';
            
            resultElement.innerHTML = resultHTML;
        } catch (e) {
            console.error('Error in matrix multiplication demo:', e);
            resultElement.innerHTML = `<span class="error">Error: ${e.message}</span>`;
        }
    });
}

// Set up the tensor transformation demo
function setupTensorTransformationDemo() {
    const runButton = document.getElementById('run-transform');
    if (!runButton) return;
    
    runButton.addEventListener('click', () => {
        const tensorInput = document.getElementById('tensor-data').value;
        const resultElement = document.getElementById('transform-result');
        
        try {
            // Parse input tensor
            const tensorData = parseFloatArray(tensorInput);
            
            // Transform the tensor
            const result = transformTensor(tensorData);
            
            // Display the result
            let resultHTML = '<h4>Original Tensor:</h4><pre>';
            resultHTML += result.original.map(v => v.toFixed(4)).join(', ');
            resultHTML += '</pre>';
            resultHTML += `<p>${result.message}</p>`;
            
            resultElement.innerHTML = resultHTML;
        } catch (e) {
            console.error('Error in tensor transformation demo:', e);
            resultElement.innerHTML = `<span class="error">Error: ${e.message}</span>`;
        }
    });
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
};

const initialStatus = document.getElementById('status');
const initialOutput = document.getElementById('output');

if (initialStatus) initialStatus.textContent = 'Loading BitNet WASM module...';
if (initialOutput) initialOutput.innerHTML = 'Loading BitNet WASM module...<br>';

function initializeWasm() {
    if (typeof ModuleFactory === 'function') {
        console.log('Module factory imported successfully. Initializing WASM...');
        if (initialOutput) initialOutput.innerHTML += 'Module factory imported successfully. Initializing WASM...<br>';
        
        ModuleFactory(moduleConfig).then((initializedInstance) => {
            onWasmInitialized(initializedInstance);
        }).catch(e => {
            console.error("Error initializing WASM module:", e);
            if (initialOutput) initialOutput.innerHTML += `Error initializing WASM module: ${e}<br>`;
            if (initialStatus) {
                initialStatus.textContent = 'Error initializing WASM module.';
                initialStatus.classList.add('error');
            }
        });
    } else {
        const currentTypeOfModuleFactory = typeof ModuleFactory;
        console.error(`Module factory not available or not a function (current type: ${currentTypeOfModuleFactory}). Check import and bitnet.js export.`);
        if (initialOutput) {
             initialOutput.innerHTML += `Module factory not available or not a function (current type: ${currentTypeOfModuleFactory}). Check import and bitnet.js export.<br>`;
        }
        if (initialStatus) {
            initialStatus.textContent = 'Error: Module factory not found.';
            initialStatus.classList.add('error');
        }
    }
}

// Initialize the WASM module when the document is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeWasm);
} else {
    initializeWasm();
}
