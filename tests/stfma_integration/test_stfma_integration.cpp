/**
 * Test program for sparse-ternary-fma integration with BitNet
 * 
 * This program tests the correctness of the integration by comparing
 * the output of the original BitNet implementation with the sparse-ternary-fma
 * implementation.
 */

#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

// Include both implementations
extern "C" {
    #include "ggml-bitnet-stfma.h"
}

// Forward declare the original function
extern "C" void ggml_vec_dot_i2_i8_s(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

// Helper function to generate random ternary values
void generate_random_ternary(std::vector<int8_t>& trits, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 2);
    
    for (size_t i = 0; i < n; i++) {
        int val = dis(gen);
        trits[i] = (val == 0) ? -1 : (val == 1 ? 0 : 1);
    }
}

// Helper function to pack ternary values in BitNet format
void pack_bitnet_format(const std::vector<int8_t>& trits, std::vector<uint8_t>& packed) {
    size_t n = trits.size();
    size_t num_bytes = n / 4;
    packed.resize(num_bytes);
    
    for (size_t i = 0; i < n; i++) {
        size_t byte_idx = i / 4;
        size_t bit_offset = (i % 4) * 2;
        
        uint8_t encoded;
        if (trits[i] == -1) encoded = 0;
        else if (trits[i] == 0) encoded = 1;
        else encoded = 2;
        
        packed[byte_idx] |= (encoded << bit_offset);
    }
}

// Helper function to generate random int8 values
void generate_random_int8(std::vector<int8_t>& values, size_t n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-128, 127);
    
    for (size_t i = 0; i < n; i++) {
        values[i] = dis(gen);
    }
}

// Test function
bool test_integration(size_t n) {
    std::cout << "Testing with n = " << n << "..." << std::endl;
    
    // Generate random data
    std::vector<int8_t> trits(n);
    std::vector<int8_t> activations(n);
    generate_random_ternary(trits, n);
    generate_random_int8(activations, n);
    
    // Pack ternary values
    std::vector<uint8_t> packed_trits(n / 4, 0);
    pack_bitnet_format(trits, packed_trits);
    
    // Compute reference result (manual calculation)
    int64_t reference_sum = 0;
    for (size_t i = 0; i < n; i++) {
        reference_sum += (int64_t)trits[i] * (int64_t)activations[i];
    }
    float reference_result = (float)reference_sum;
    
    // Compute result using sparse-ternary-fma
    float stfma_result = 0.0f;
    ggml_vec_dot_i2_i8_stfma(n, &stfma_result, 0, packed_trits.data(), 0, activations.data(), 0, 0);
    
    // Compare results
    float diff = std::abs(stfma_result - reference_result);
    float rel_error = (reference_result != 0.0f) ? (diff / std::abs(reference_result)) : diff;
    
    std::cout << "  Reference result: " << reference_result << std::endl;
    std::cout << "  STFMA result:     " << stfma_result << std::endl;
    std::cout << "  Absolute error:   " << diff << std::endl;
    std::cout << "  Relative error:   " << rel_error << std::endl;
    
    // Check if results match (allowing for small floating-point errors)
    bool passed = (diff < 1e-3f) || (rel_error < 1e-6f);
    
    if (passed) {
        std::cout << "  ✓ Test PASSED" << std::endl;
    } else {
        std::cout << "  ✗ Test FAILED" << std::endl;
    }
    
    std::cout << std::endl;
    return passed;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Sparse-Ternary-FMA Integration Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    // Test various sizes
    std::vector<size_t> test_sizes = {128, 256, 512, 1024, 2048, 4096};
    
    int passed = 0;
    int total = test_sizes.size();
    
    for (size_t n : test_sizes) {
        if (test_integration(n)) {
            passed++;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Results: " << passed << "/" << total << " tests passed" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Clean up thread-local buffers
    stfma_free_buffers();
    
    return (passed == total) ? 0 : 1;
}
