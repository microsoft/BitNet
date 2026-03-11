/**
 * Test program to verify the branchless conversion function
 * 
 * This program tests that the optimized branchless conversion produces
 * identical results to the original branching implementation.
 */

#include <iostream>
#include <iomanip>
#include <cstdint>

// Original branching implementation (for reference)
uint8_t convert_bitnet_to_stfma_byte_original(uint8_t bitnet_byte) {
    uint8_t result = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t trit = (bitnet_byte >> (i * 2)) & 0b11;
        uint8_t stfma_trit;
        switch (trit) {
            case 0: stfma_trit = 0b10; break; // -1
            case 1: stfma_trit = 0b00; break; // 0
            case 2: stfma_trit = 0b01; break; // +1
            default: stfma_trit = 0b11; break; // Invalid
        }
        result |= (stfma_trit << (i * 2));
    }
    return result;
}

// Optimized branchless implementation
uint8_t convert_bitnet_to_stfma_byte_branchless(uint8_t b) {
    // Mask for low bits of each pair: 01010101 = 0x55
    uint8_t low_bits = b & 0x55; 
    
    // Mask for high bits of each pair: 10101010 = 0xAA
    uint8_t high_bits = b & 0xAA;

    // STFMA Low Bit is simply the BitNet High Bit shifted right by 1
    uint8_t out_low = (high_bits >> 1);

    // STFMA High Bit is 1 ONLY if input was 00 (-1)
    uint8_t input_or = low_bits | (high_bits >> 1);
    uint8_t is_zero_zero = (~input_or) & 0x55;
    uint8_t out_high = is_zero_zero << 1;

    return out_high | out_low;
}

// Helper to print binary representation
void print_binary(uint8_t val) {
    for (int i = 7; i >= 0; i--) {
        std::cout << ((val >> i) & 1);
        if (i % 2 == 0) std::cout << " ";
    }
}

// Helper to decode a trit
const char* decode_trit(uint8_t trit) {
    switch (trit) {
        case 0b00: return " 0";
        case 0b01: return "+1";
        case 0b10: return "-1";
        case 0b11: return "??";
        default: return "??";
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Branchless Conversion Verification Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    int passed = 0;
    int failed = 0;
    
    // Test all possible byte values (0-255)
    for (int input = 0; input <= 255; input++) {
        uint8_t original = convert_bitnet_to_stfma_byte_original(input);
        uint8_t branchless = convert_bitnet_to_stfma_byte_branchless(input);
        
        if (original == branchless) {
            passed++;
        } else {
            failed++;
            std::cout << "MISMATCH for input " << std::hex << std::setw(2) << std::setfill('0') << input << std::dec << ":" << std::endl;
            std::cout << "  Input:      ";
            print_binary(input);
            std::cout << " (";
            for (int i = 0; i < 4; i++) {
                uint8_t trit = (input >> (i * 2)) & 0b11;
                std::cout << decode_trit(trit);
                if (i < 3) std::cout << ", ";
            }
            std::cout << ")" << std::endl;
            
            std::cout << "  Original:   ";
            print_binary(original);
            std::cout << std::endl;
            
            std::cout << "  Branchless: ";
            print_binary(branchless);
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Passed: " << passed << "/256" << std::endl;
    std::cout << "  Failed: " << failed << "/256" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (failed == 0) {
        std::cout << "✓ All tests PASSED! Branchless conversion is correct." << std::endl;
        
        // Show some example conversions
        std::cout << std::endl;
        std::cout << "Example conversions:" << std::endl;
        std::cout << "-------------------" << std::endl;
        
        uint8_t examples[] = {
            0b00000000,  // All -1
            0b01010101,  // All 0
            0b10101010,  // All +1
            0b00011000,  // Mixed: -1, 0, +1, 0
            0b10010100   // Mixed: +1, +1, 0, -1
        };
        
        for (uint8_t ex : examples) {
            std::cout << "Input:  ";
            print_binary(ex);
            std::cout << " -> Output: ";
            uint8_t out = convert_bitnet_to_stfma_byte_branchless(ex);
            print_binary(out);
            std::cout << std::endl;
        }
    } else {
        std::cout << "✗ FAILED! Branchless conversion has errors." << std::endl;
    }
    
    return (failed == 0) ? 0 : 1;
}
