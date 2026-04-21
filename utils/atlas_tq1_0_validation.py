import numpy as np
import os

# --- TQ1_0 Ternary Packing Logic ---

def quantize_ternary(values):
    """Quantizes float values to ternary scale (-1, 0, 1)."""
    return np.where(values > 0.1, 1, np.where(values < -0.1, -1, 0))

def pack_ternary(ternary_arr):
    """Packs 4 ternary values into a single uint8 (2 bits per value)."""
    TERNAMAP = {-1: 0b00, 0: 0b01, 1: 0b10}
    original_len = len(ternary_arr)
    
    # Padding to ensure length is a multiple of 4
    if original_len % 4 != 0:
        pad = 4 - (original_len % 4)
        ternary_arr = np.pad(ternary_arr, (0, pad), constant_values=0)
    
    packed = []
    for i in range(0, len(ternary_arr), 4):
        bits = 0
        for j, val in enumerate(ternary_arr[i:i+4]):
            # Default to 0b01 (ternary 0) if value is unexpected
            bits |= (TERNAMAP.get(int(val), 0b01) << (j * 2))
        packed.append(bits)
    return np.array(packed, dtype=np.uint8)

def unpack_ternary(packed, length):
    """Unpacks uint8 back to ternary values (-1, 0, 1)."""
    INVERSE = {0b00: -1, 0b01: 0, 0b10: 1, 0b11: 0}
    result = []
    for bits in packed:
        for j in range(4):
            if len(result) < length:
                code = (bits >> (j * 2)) & 0b11
                result.append(INVERSE.get(code, 0))
    return np.array(result, dtype=np.int8)

# --- Performance & Integrity Audit ---

def run_audit(sample_size=1_000_000):
    print(f"--- TQ1_0 VALIDATION AUDIT (Reference: Float16) ---")
    print(f"Sample Size: {sample_size} weights")

    # 1. Generate random weights in FP16
    raw_data = np.random.uniform(-1, 1, sample_size).astype(np.float16)

    # 2. Process: Quantize and Pack
    ternary = quantize_ternary(raw_data)
    packed_data = pack_ternary(ternary)

    # 3. I/O Simulation: Write to disk
    tmp_file = "temp_weights.bin"
    packed_data.tofile(tmp_file)
    
    # 4. Read back and Unpack
    disk_data = np.fromfile(tmp_file, dtype=np.uint8)
    restored = unpack_ternary(disk_data, sample_size)

    # 5. Calculate Metrics
    fp16_size_kb = raw_data.nbytes / 1024
    packed_size_kb = os.path.getsize(tmp_file) / 1024
    reduction = (1 - (packed_size_kb / fp16_size_kb)) * 100

    print(f"Original Size (FP16): {fp16_size_kb:.2f} KB")
    print(f"Packed Size (TQ1_0):   {packed_size_kb:.2f} KB")
    print(f"Memory Reduction:      {reduction:.2f}%")

    # 6. Integrity Check
    if np.array_equal(ternary, restored):
        print("\n✅ STATUS: BIT-EXACT RECONSTRUCTION SUCCESSFUL.")
    else:
        print("\n❌ STATUS: INTEGRITY CHECK FAILED.")

    # Cleanup
    if os.path.exists(tmp_file):
        os.remove(tmp_file)

if __name__ == "__main__":
    run_audit()