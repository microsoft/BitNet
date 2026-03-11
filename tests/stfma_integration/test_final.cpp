#include <iostream>
#include <cstdint>

uint8_t orig(uint8_t b) {
    uint8_t r = 0;
    for (int i = 0; i < 4; i++) {
        uint8_t t = (b >> (i * 2)) & 3;
        uint8_t o = (t == 0) ? 2 : (t == 1) ? 0 : (t == 2) ? 1 : 3;
        r |= (o << (i * 2));
    }
    return r;
}

uint8_t branchless(uint8_t b) {
    uint8_t low_bits = b & 0x55;
    uint8_t high_bits = b & 0xAA;
    uint8_t out_low = (high_bits >> 1);
    uint8_t high_bits_shifted = (high_bits >> 1);
    uint8_t xor_result = high_bits_shifted ^ low_bits;
    uint8_t out_high = (~xor_result) & 0x55;
    out_high = out_high << 1;
    return out_high | out_low;
}

int main() {
    int pass = 0, fail = 0;
    for (int i = 0; i <= 255; i++) {
        if (orig(i) == branchless(i)) pass++;
        else { fail++; if (fail <= 5) std::cout << "FAIL: " << i << std::endl; }
    }
    std::cout << "Pass: " << pass << "/256, Fail: " << fail << "/256" << std::endl;
    if (fail == 0) std::cout << "✓ SUCCESS!" << std::endl;
    return (fail == 0) ? 0 : 1;
}
