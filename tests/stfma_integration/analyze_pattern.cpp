#include <iostream>
#include <iomanip>

int main() {
    std::cout << "BitNet -> STFMA Mapping Analysis\n" << std::endl;
    std::cout << "Input | In_H In_L | Out_H Out_L | Output" << std::endl;
    std::cout << "------|-----------|-------------|-------" << std::endl;
    
    // 00 -> 10
    std::cout << "  00  |  0    0   |   1    0    |   10" << std::endl;
    // 01 -> 00  
    std::cout << "  01  |  0    1   |   0    0    |   00" << std::endl;
    // 10 -> 01
    std::cout << "  10  |  1    0   |   0    1    |   01" << std::endl;
    // 11 -> 11
    std::cout << "  11  |  1    1   |   1    1    |   11" << std::endl;
    
    std::cout << "\nFormula derivation:" << std::endl;
    std::cout << "Out_L = In_H (simple copy)" << std::endl;
    std::cout << "Out_H = ?" << std::endl;
    
    std::cout << "\nTruth table for Out_H:" << std::endl;
    std::cout << "In_H In_L | Out_H" << std::endl;
    std::cout << "----------|------" << std::endl;
    std::cout << " 0    0   |  1    (NOT In_L)" << std::endl;
    std::cout << " 0    1   |  0    (NOT In_L)" << std::endl;
    std::cout << " 1    0   |  0    (In_H)" << std::endl;
    std::cout << " 1    1   |  1    (In_H)" << std::endl;
    
    std::cout << "\nPattern: Out_H = In_H XOR (NOT In_L)" << std::endl;
    std::cout << "Or: Out_H = In_H XOR (~In_L)" << std::endl;
    std::cout << "Simplified: Out_H = ~(In_H XOR In_L)" << std::endl;
    
    // Verify
    std::cout << "\nVerification:" << std::endl;
    for (int h = 0; h <= 1; h++) {
        for (int l = 0; l <= 1; l++) {
            int out_h = ~(h ^ l) & 1;
            std::cout << "In_H=" << h << " In_L=" << l << " -> Out_H=" << out_h << std::endl;
        }
    }
    
    return 0;
}
