#include <metal_stdlib>
using namespace metal;

// Decode 2-bit packed weights to int8
// Packed format: 4 weights per byte (2 bits each: 00=-1, 01=0, 10=+1)
inline void decode_i2s_to_i8s(uint32_t i2s, thread int8_t* i8s) {
    // Extract 4 values from each byte
    // i2s = packed 2-bit values
    // 0 -> -1, 1 -> 0, 2 -> +1, 3 -> (unused/reserved)
    
    const uint32_t mask = 0x03030303;  // 0b11 mask for each byte
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint32_t val = (i2s >> (2 * i)) & mask;
        // Map: 0->-1, 1->0, 2->1
        i8s[i] = (int8_t)(val - 1);
    }
}

// Optimized version that decodes 16 values from a 32-bit word
inline void decode_i2s_to_i8s_16(uint32_t i2s, thread int8_t* i8s) {
    const uint32_t mask = 0x03;
    
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint32_t val = (i2s >> (2 * i)) & mask;
        i8s[i] = (int8_t)(val - 1);
    }
}

// Int8 x Int2 matrix multiplication kernel
// A: int8_t [M, K] - input activations
// B: packed int2 [N, K/4] - weights (4 values packed per byte)
// C: bfloat16_t [M, N] - output
// s: bfloat16_t [M] - input scales (per-row quantization)
// ws: bfloat16_t [N] - weight scales (per-column)
kernel void bitlinear_int8xint2(
    device const int8_t* A [[buffer(0)]],           // [M, K]
    device const uint8_t* B [[buffer(1)]],            // [N, K/4] packed
    device bfloat16_t* C [[buffer(2)]],               // [M, N]
    device const bfloat16_t* s [[buffer(3)]],         // [M] input scales
    device const bfloat16_t* ws [[buffer(4)]],        // [N] weight scales  
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]]
) {
    const int m_idx = tid.y;  // row
    const int n_idx = tid.x;  // column
    
    if (m_idx >= M || n_idx >= N) return;
    
    // Each thread computes one output element
    int32_t acc = 0;
    
    // Process K dimension in chunks of 16 (for SIMD efficiency)
    const int k_per_thread = 16;
    const int k_blocks = (K + k_per_thread - 1) / k_per_thread;
    
    for (int kb = 0; kb < k_blocks; kb++) {
        int k_start = kb * k_per_thread;
        int k_end = min(k_start + k_per_thread, K);
        int k_len = k_end - k_start;
        
        // Decode 16 weights from 4 bytes (4 weights per byte)
        int8_t weights[16];
        
        for (int i = 0; i < k_len; i += 4) {
            // Load 4 bytes of packed weights (16 2-bit values)
            int k_global = k_start + i;
            if (k_global + 3 < K) {
                uint32_t packed = *(device const uint32_t*)&B[n_idx * (K / 4) + k_global / 4];
                
                // Decode each byte
                for (int j = 0; j < 4 && (i + j) < k_len; j++) {
                    uint8_t byte = (packed >> (8 * j)) & 0xFF;
                    
                    // Extract 4 2-bit values from this byte
                    weights[i + j] = (int8_t)((byte & 0x03) - 1);
                    if (i + j + 4 < k_len) weights[i + j + 4] = (int8_t)(((byte >> 2) & 0x03) - 1);
                    if (i + j + 8 < k_len) weights[i + j + 8] = (int8_t)(((byte >> 4) & 0x03) - 1);
                    if (i + j + 12 < k_len) weights[i + j + 12] = (int8_t)(((byte >> 6) & 0x03) - 1);
                }
            }
        }
        
        // Dot product with activations
        for (int i = 0; i < k_len; i++) {
            int k_global = k_start + i;
            int8_t a_val = A[m_idx * K + k_global];
            acc += (int32_t)a_val * (int32_t)weights[i];
        }
    }
    
    // Apply scales and write output
    // C[m, n] = acc / s[m] * ws[n]
    float result = (float)acc;
    result = result / (float)s[m_idx] * (float)ws[n_idx];
    
    C[m_idx * N + n_idx] = bfloat16_t(result);
}

// Optimized version using SIMD groups for reduction
// Each SIMD group (32 threads) processes a tile of the matrix
kernel void bitlinear_int8xint2_simd(
    device const int8_t* A [[buffer(0)]],           // [M, K]
    device const uint8_t* B [[buffer(1)]],            // [N, K/4] packed
    device bfloat16_t* C [[buffer(2)]],               // [M, N]
    device const bfloat16_t* s [[buffer(3)]],         // [M] input scales
    device const bfloat16_t* ws [[buffer(4)]],        // [N] weight scales
    constant int& M [[buffer(5)]],
    constant int& N [[buffer(6)]],
    constant int& K [[buffer(7)]],
    uint2 tid [[thread_position_in_grid]],
    uint2 bid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]]
) {
    // Tile-based processing
    // Each threadgroup processes a tile of the output matrix
    const int tile_m = 8;  // rows per tile
    const int tile_n = 32; // columns per tile (one per SIMD thread)
    
    const int tile_m_idx = bid.y * tile_m;
    const int tile_n_idx = bid.x * tile_n;
    
    const int local_m = lid / tile_n;  // row within tile
    const int local_n = lid % tile_n;  // column within tile
    
    const int m_idx = tile_m_idx + local_m;
    const int n_idx = tile_n_idx + local_n;
    
    if (m_idx >= M || n_idx >= N) return;
    
    // Each thread accumulates its dot product
    int32_t acc = 0;
    
    // Process K in blocks that fit in threadgroup cache
    const int k_block_size = 64;
    
    for (int k_base = 0; k_base < K; k_base += k_block_size) {
        int k_end = min(k_base + k_block_size, K);
        
        // Load and decode weights for this column
        threadgroup int8_t weights_cache[32 * 64];  // tile_n x k_block_size
        
        // Collaborative loading: each thread loads some weights
        int weights_per_thread = (k_block_size + tile_n - 1) / tile_n;
        for (int i = 0; i < weights_per_thread; i++) {
            int k_local = lid * weights_per_thread + i;
            if (k_local < k_block_size && k_base + k_local < K) {
                // Load packed byte
                int k_packed = (k_base + k_local) / 4;
                uint8_t packed = B[n_idx * (K / 4) + k_packed];
                
                // Decode one value
                int shift = (k_local % 4) * 2;
                int8_t val = (int8_t)(((packed >> shift) & 0x03) - 1);
                weights_cache[local_n * k_block_size + k_local] = val;
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product
        for (int k = 0; k < k_end - k_base; k++) {
            int8_t a_val = A[m_idx * K + k_base + k];
            int8_t w_val = weights_cache[local_n * k_block_size + k];
            acc += (int32_t)a_val * (int32_t)w_val;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Apply scales and write
    float result = (float)acc;
    result = result / (float)s[m_idx] * (float)ws[n_idx];
    C[m_idx * N + n_idx] = bfloat16_t(result);
}

// Input quantization kernel: FP16/BF16 -> INT8 with per-row scaling
kernel void quantize_input(
    device const bfloat16_t* input [[buffer(0)]],     // [M, K]
    device int8_t* output [[buffer(1)]],               // [M, K]
    device bfloat16_t* scales [[buffer(2)]],           // [M]
    constant int& M [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int m_idx = tid.y;
    const int k_idx = tid.x;
    
    if (m_idx >= M || k_idx >= K) return;
    
    // First thread in each row computes scale
    threadgroup float row_max[1];
    
    if (k_idx == 0) {
        float max_val = 0.0f;
        for (int k = 0; k < K; k++) {
            float val = fabs((float)input[m_idx * K + k]);
            max_val = fmax(max_val, val);
        }
        row_max[0] = max_val;
        scales[m_idx] = bfloat16_t(127.0f / fmax(max_val, 1e-5f));
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Quantize: round(input * scale) clamped to [-128, 127]
    float scale = row_max[0];
    float val = (float)input[m_idx * K + k_idx];
    int32_t qval = (int32_t)(val * scale);
    qval = clamp(qval, -128, 127);
    output[m_idx * K + k_idx] = (int8_t)qval;
}
