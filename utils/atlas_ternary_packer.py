"""
ATLAS Ternary Packer - For BitNet GGUF Converter
=========================================
Add to utils/convert-hf-to-gguf-bitnet.py

This module provides 4x compression for ternary BitNet weights.
Insert at line ~159 in write_tensors() method.
"""

import numpy as np


def detect_ternary(values: np.ndarray, threshold: float = 0.1) -> bool:
    """Detect if tensor contains only ternary values {-1, 0, +1}"""
    unique = np.unique(np.round(values, 1))
    ternary_vals = {-1.0, 0.0, 1.0}
    return all(abs(v) < threshold or v in ternary_vals for v in unique)


def quantize_ternary(values: np.ndarray) -> np.ndarray:
    """Quantize floating point to ternary {-1, 0, +1}"""
    return np.where(values > 0.1, 1, np.where(values < -0.1, -1, 0))


def pack_ternary(ternary_arr: np.ndarray) -> np.ndarray:
    """Pack ternary values into 2-bit representation (4x compression)
    
    Encoding:
      -1 → 0b00
       0 → 0b01
      +1 → 0b10
    """
    TERNAMAP = {-1: 0b00, 0: 0b01, 1: 0b10}
    
    if len(ternary_arr) % 4 != 0:
        pad = 4 - (len(ternary_arr) % 4)
        ternary_arr = np.pad(ternary_arr, (0, pad), constant_values=0)
    
    packed = []
    for i in range(0, len(ternary_arr), 4):
        bits = 0
        for j, val in enumerate(ternary_arr[i:i+4]):
            bits |= (TERNAMAP.get(int(val), 0b11) << (j * 2))
        packed.append(bits)
    
    return np.array(packed, dtype=np.uint8)


def unpack_ternary(packed: np.ndarray, length: int) -> np.ndarray:
    """Unpack 2-bit representation back to ternary"""
    INVERSE = {0b00: -1, 0b01: 0, 0b10: 1}
    result = []
    
    for bits in packed:
        for j in range(4):
            if len(result) < length:
                code = (bits >> (j * 2)) & 0b11
                result.append(INVERSE.get(code, 0))
    
    return np.array(result[:length], dtype=np.int8)


# Integration point: Add to convert-hf-to-gguf-bitnet.py
# In write_tensors() method, around line 159:
#
# def write_tensors(self):
#     # [INSERT ATLAS PACKER HERE]
#     if detect_ternary(data):
#         packed = pack_ternary(quantize_ternary(data))
#         self.gguf_writer.add_tensor(new_name, packed, tensor_type=gguf.GGMLQuantizationType.TL1)
#     else:
#         self.gguf_writer.add_tensor(new_name, data)


# Test
if __name__ == "__main__":
    # Test with 8 values = 2 bytes packed (2-bit per value)
    test = np.array([-1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 1.0, -1.0], dtype=np.float32)
    print(f"Original: {test.nbytes} bytes (8 x float32)")
    
    ternary = quantize_ternary(test)
    packed = pack_ternary(ternary)
    print(f"Packed:   {packed.nbytes} bytes (8 x 2-bit)")
    print(f"Compression: {test.nbytes / packed.nbytes:.1f}x")
    
    restored = unpack_ternary(packed, len(test))
    print(f"Round-trip OK: {np.array_equal(ternary, restored)}")