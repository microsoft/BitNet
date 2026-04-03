// Copyright (c) Microsoft. All rights reserved.
// Objective-C++ wrapper for Metal backend

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <torch/extension.h>
#include <vector>
#include <cstring>

// Metal device and pipeline state
static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_commandQueue = nil;
static id<MTLLibrary> g_library = nil;

// Pipeline states for each kernel
static id<MTLComputePipelineState> g_matmulPipeline = nil;
static id<MTLComputePipelineState> g_quantizePipeline = nil;

// Initialize Metal
bool metal_init() {
    if (g_device != nil) return true;
    
    // Get default Metal device
    g_device = MTLCreateSystemDefaultDevice();
    if (g_device == nil) return false;
    
    // Create command queue
    g_commandQueue = [g_device newCommandQueue];
    
    // Load Metal library from default shader file
    NSError* error = nil;
    NSString* shaderPath = [[NSBundle mainBundle] pathForResource:@"bitnet_kernels" ofType:@"metallib"];
    
    if (shaderPath == nil) {
        // Try to compile from source
        NSString* sourcePath = [[NSBundle mainBundle] pathForResource:@"bitnet_kernels" ofType:@"metal"];
        if (sourcePath != nil) {
            NSString* source = [NSString stringWithContentsOfFile:sourcePath encoding:NSUTF8StringEncoding error:&error];
            if (source != nil) {
                g_library = [g_device newLibraryWithSource:source options:nil error:&error];
            }
        }
    } else {
        g_library = [g_device newLibraryWithURL:[NSURL fileURLWithPath:shaderPath] error:&error];
    }
    
    if (g_library == nil) {
        // Compile default shaders inline
        const char* defaultShaders = R"(
#include <metal_stdlib>
using namespace metal;

kernel void bitlinear_int8xint2(
    device const int8_t* A [[buffer(0)]],
    device const uint8_t* B [[buffer(1)]],
    device bfloat16_t* C [[buffer(2)]],
    device const bfloat16_t* s [[buffer(3)]],
    device const bfloat16_t* ws [[buffer(4)]],
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int m_idx = tid.y;
    const int n_idx = tid.x;
    
    if (m_idx >= M || n_idx >= N) return;
    
    int32_t acc = 0;
    const int k_blocks = (K + 15) / 16;
    
    for (int kb = 0; kb < k_blocks; kb++) {
        int k_start = kb * 16;
        int k_end = min(k_start + 16, K);
        
        for (int k = k_start; k < k_end; k++) {
            uint8_t packed = B[n_idx * (K / 4) + k / 4];
            int shift = (k % 4) * 2;
            int8_t w = (int8_t)(((packed >> shift) & 0x03) - 1);
            int8_t a = A[m_idx * K + k];
            acc += (int32_t)a * (int32_t)w;
        }
    }
    
    float result = (float)acc;
    result = result / (float)s[m_idx] * (float)ws[n_idx];
    C[m_idx * N + n_idx] = bfloat16_t(result);
}
)";
        NSString* source = [NSString stringWithUTF8String:defaultShaders];
        g_library = [g_device newLibraryWithSource:source options:nil error:&error];
    }
    
    if (g_library == nil) return false;
    
    // Create pipeline states
    id<MTLFunction> matmulFunction = [g_library newFunctionWithName:@"bitlinear_int8xint2"];
    if (matmulFunction != nil) {
        g_matmulPipeline = [g_device newComputePipelineStateWithFunction:matmulFunction error:&error];
    }
    
    return g_device != nil && g_commandQueue != nil && g_matmulPipeline != nil;
}

// Execute matrix multiplication
void metal_matmul(
    int64_t M, int64_t N, int64_t K,
    void* A_ptr,      // int8 [M, K]
    void* B_ptr,      // uint8 packed [N, K/4]
    void* C_ptr,      // bfloat16 [M, N]
    void* s_ptr,      // bfloat16 [M]
    void* ws_ptr      // bfloat16 [N]
) {
    if (!metal_init()) {
        throw std::runtime_error("Metal initialization failed");
    }
    
    @autoreleasepool {
        // Create command buffer and encoder
        id<MTLCommandBuffer> commandBuffer = [g_commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        
        // Set pipeline
        [encoder setComputePipelineState:g_matmulPipeline];
        
        // Calculate buffer sizes
        size_t A_size = M * K * sizeof(int8_t);
        size_t B_size = N * (K / 4) * sizeof(uint8_t);
        size_t C_size = M * N * sizeof(bfloat16_t);
        size_t s_size = M * sizeof(bfloat16_t);
        size_t ws_size = N * sizeof(bfloat16_t);
        
        // Create or reuse buffers (in production, use a buffer pool)
        id<MTLBuffer> A_buffer = [g_device newBufferWithBytes:A_ptr length:A_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> B_buffer = [g_device newBufferWithBytes:B_ptr length:B_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> C_buffer = [g_device newBufferWithBytesNoCopy:C_ptr length:C_size options:MTLResourceStorageModeShared deallocator:nil];
        id<MTLBuffer> s_buffer = [g_device newBufferWithBytes:s_ptr length:s_size options:MTLResourceStorageModeShared];
        id<MTLBuffer> ws_buffer = [g_device newBufferWithBytes:ws_ptr length:ws_size options:MTLResourceStorageModeShared];
        
        // Set buffers
        [encoder setBuffer:A_buffer offset:0 atIndex:0];
        [encoder setBuffer:B_buffer offset:0 atIndex:1];
        [encoder setBuffer:C_buffer offset:0 atIndex:2];
        [encoder setBuffer:s_buffer offset:0 atIndex:3];
        [encoder setBuffer:ws_buffer offset:0 atIndex:4];
        
        // Set constants
        struct Constants {
            int M, N, K;
        } constants = {(int)M, (int)N, (int)K};
        [encoder setBytes:&constants length:sizeof(constants) atIndex:5];
        
        // Dispatch threads with 256-thread configuration (32x8)
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize threadgroupSize = MTLSizeMake(32, 8, 1);  // 256 threads per group
        
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        
        // Commit and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
}

// PyTorch binding
void bitlinear_metal(
    int64_t M, int64_t N, int64_t K,
    uintptr_t A,
    uintptr_t B,
    uintptr_t C,
    uintptr_t s,
    uintptr_t ws
) {
    metal_matmul(M, N, K,
        reinterpret_cast<void*>(A),
        reinterpret_cast<void*>(B),
        reinterpret_cast<void*>(C),
        reinterpret_cast<void*>(s),
        reinterpret_cast<void*>(ws)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitlinear_metal", &bitlinear_metal, "BitNet linear layer on Metal",
        py::arg("M"), py::arg("N"), py::arg("K"),
        py::arg("A"), py::arg("B"), py::arg("C"),
        py::arg("s"), py::arg("ws"));
    m.def("metal_init", &metal_init, "Initialize Metal device");
}
