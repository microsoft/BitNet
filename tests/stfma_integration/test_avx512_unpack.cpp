#include <iostream>
#include <cstdint>
#include <immintrin.h>

// Test the AVX-512 unpacking logic
void test_unpack() {
    // Test case: pack 16 trits into 4 bytes
    uint32_t trit_packed = 0;
    int32_t expected[16];
    
    // Create test pattern: 0, 1, 2, 3, 0, 1, 2, 3, ...
    for (int i = 0; i < 16; i++) {
        int32_t trit = i % 4;
        expected[i] = trit;
        trit_packed |= (trit << (i * 2));
    }
    
    std::cout << "Test packed value: 0x" << std::hex << trit_packed << std::dec << std::endl;
    std::cout << "Expected trits: ";
    for (int i = 0; i < 16; i++) {
        std::cout << expected[i] << " ";
    }
    std::cout << std::endl;
    
    // Unpack using AVX-512
    __m512i packed_vec = _mm512_set1_epi32(trit_packed);
    __m512i shift_amounts = _mm512_setr_epi32(
        0, 2, 4, 6, 8, 10, 12, 14,
        16, 18, 20, 22, 24, 26, 28, 30
    );
    __m512i shifted = _mm512_srlv_epi32(packed_vec, shift_amounts);
    __m512i mask_2bits = _mm512_set1_epi32(0b11);
    __m512i trit_vec = _mm512_and_si512(shifted, mask_2bits);
    
    // Extract results
    int32_t result[16];
    _mm512_storeu_si512(result, trit_vec);
    
    std::cout << "Unpacked trits:  ";
    for (int i = 0; i < 16; i++) {
        std::cout << result[i] << " ";
    }
    std::cout << std::endl;
    
    // Verify
    bool pass = true;
    for (int i = 0; i < 16; i++) {
        if (result[i] != expected[i]) {
            std::cout << "MISMATCH at index " << i << ": expected " << expected[i] 
                     << ", got " << result[i] << std::endl;
            pass = false;
        }
    }
    
    if (pass) {
        std::cout << "✓ AVX-512 unpacking test PASSED" << std::endl;
    } else {
        std::cout << "✗ AVX-512 unpacking test FAILED" << std::endl;
    }
}

int main() {
    std::cout << "Testing AVX-512 2-bit trit unpacking" << std::endl;
    std::cout << "=====================================" << std::endl;
    test_unpack();
    return 0;
}
