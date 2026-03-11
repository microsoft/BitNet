#include <iostream>
#include <iomanip>
#include <cstdint>

uint8_t convert_bitnet_to_stfma_byte_original(uint8_t bitnet_byte) {
    uint8_t result = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t trit = (bitnet_byte >> (i * 2)) & 0b11;
        uint8_t stfma_trit;
        switch (trit) {
            case 0: stfma_trit = 0b10; break;
            case 1: stfma_trit = 0b00; break;
            case 2: stfma_trit = 0b01; break;
            default: stfma_trit = 0b11; break;
        }
        result |= (stfma_trit << (i * 2));
    }
    return result;
}

uint8_t convert_bitnet_to_stfma_byte_branchless(uint8_t b) {
    uint8_t low_bits = b & 0x55; 
    uint8_t high_bits = b & 0xAA;
    uint8_t out_low = (high_bits >> 1);
    uint8_t out_high = (~low_bits | high_bits) & 0xAA;
    return out_high | out_low;
}

void print_binary(uint8_t val) {
    for (int i = 7; i >= 0; i--) {
        std::cout << ((val >> i) & 1);
        if (i % 2 == 0) std::cout << " ";
    }
}

int main() {
    std::cout << "Branchless Conversion Test\n" << std::endl;
    
    int passed = 0, failed = 0;
    
    for (int input = 0; input <= 255; input++) {
        uint8_t original = convert_bitnet_to_stfma_byte_original(input);
        uint8_t branchless = convert_bitnet_to_stfma_byte_branchless(input);
        
        if (original == branchless) {
            passed++;
        } else {
            failed++;
            if (failed <= 10) {
                std::cout << "FAIL " << std::hex << input << ": ";
                print_binary(input);
                std::cout << " -> orig: ";
                print_binary(original);
                std::cout << " vs branch: ";
                print_binary(branchless);
                std::cout << std::dec << std::endl;
            }
        }
    }
    
    std::cout << "\nResults: " << passed << "/256 passed, " << failed << "/256 failed" << std::endl;
    
    if (failed == 0) {
        std::cout << "✓ SUCCESS! All conversions match." << std::endl;
        std::cout << "\nExample conversions:" << std::endl;
        uint8_t examples[] = {0x00, 0x55, 0xAA, 0x1B, 0xE4};
        for (uint8_t ex : examples) {
            std::cout << "  0x" << std::hex << std::setw(2) << std::setfill('0') << (int)ex << " -> 0x" 
                     << std::setw(2) << (int)convert_bitnet_to_stfma_byte_branchless(ex) << std::dec << std::endl;
        }
    }
    
    return (failed == 0) ? 0 : 1;
}
