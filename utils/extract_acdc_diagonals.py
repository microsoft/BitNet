#!/usr/bin/env python3
#
# extract_acdc_diagonals.py
#
# Reads a BitNet GGUF file (I2_S quantized), extracts the ACDC rectangular
# diagonal d* for each FFN projection (gate, up, down), and saves a sidecar
# .npz file for use at inference time.
#
# ═══ Algorithm ═══
#
# For a rectangular weight W ∈ {-1,0,+1}^{m×n}, the best ACDC diagonal is:
#   d*[k] = diag(H_P · W_pad · H_P)[k] / P²
# where P = next_pow2(max(m, n)) and W_pad is W zero-padded to P×P.
#
# Computing diag(H_P·W_pad·H_P) directly costs O(P²) memory.
# XOR-convolution reduces this to O(m·n + P·log P) time and O(P) memory:
#
#   C[s] = Σ_{i,j: i⊕j=s} W[i,j]        (XOR-convolution, O(m·n))
#   diag(H_P·W_pad·H_P)[k] = (H_P · C)[k]  (WHT, O(P·log P))
#   d*[k] = (H_P · C)[k] / P²
#
# Derivation: H[k,i]·H[j,k] = (-1)^{popcount(k&(i XOR j))} = H[k, i XOR j].
# So diag(HWH)[k] = Σ_{i,j} W[i,j]·H[k,i]·H[j,k] = Σ_{i,j} W[i,j]·H[k,i⊕j]
#                 = (H · C)[k] where C[s] = Σ_{i⊕j=s} W[i,j].
#
# ═══ I2_S encoding ═══
#
# GGUF type 36 (GGML_TYPE_I2_S). Shape in GGUF: [n_cols, n_rows] (reversed).
# Each row uses n_cols/4 bytes. For each block of 128 values = 32 bytes:
#   byte gp (0..31) encodes 4 values at positions 0*32+gp, 1*32+gp, 2*32+gp, 3*32+gp
#   bits 7:6 → pos 0*32+gp,  bits 5:4 → pos 1*32+gp
#   bits 3:2 → pos 2*32+gp,  bits 1:0 → pos 3*32+gp
#   map: 0→-1, 1→0, 2→+1, 3→0
# One global float32 scale at offset n_cols*n_rows/4 bytes from tensor start.
#
# ═══ Uso ═══
#
#   python utils/extract_acdc_diagonals.py <model.gguf> [--out sidecar.npz]
#   python utils/extract_acdc_diagonals.py <model.gguf> --layer 0 --proj gate
#
#   Layers in GGUF named: blk.{layer}.ffn_gate.weight / ffn_up / ffn_down
#
# ═══ Saída ═══
#
#   sidecar.npz: numpy archive com chaves:
#     blk.{L}.ffn_gate.d_star  → float32 [P]
#     blk.{L}.ffn_up.d_star    → float32 [P]
#     blk.{L}.ffn_down.d_star  → float32 [P]
#   Plus "model_path", "n_layers", "P" metadata in a JSON sidecar.
#
# ═══ Exemplo de uso ═══
#
#   $ python utils/extract_acdc_diagonals.py \
#       models/Falcon3-10B-Instruct-1.58bit-GGUF/ggml-model-i2_s.gguf
#   [INFO] Falcon3-10B: 40 layers, n_ff=23040, n_embd=3072, P=32768
#   [INFO] Processing 120 tensors (40 layers × 3 projections)...
#   [INFO] Layer 0: gate=OK, up=OK, down=OK (5.4s)
#   ...
#   [OK] Saved: ggml-model-i2_s.acdc_diag.npz (15.0 MB)
#

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Minimal GGUF parser (handles type 36 = GGML_TYPE_I2_S)
# ──────────────────────────────────────────────────────────────────────────────

GGUF_MAGIC = b"GGUF"
GGML_TYPE_I2_S = 36

GGUF_VALUE_TYPES = {
    0: ("B", 1),    # UINT8
    1: ("b", 1),    # INT8
    2: ("H", 2),    # UINT16
    3: ("h", 2),    # INT16
    4: ("I", 4),    # UINT32
    5: ("i", 4),    # INT32
    6: ("f", 4),    # FLOAT32
    7: ("?", 1),    # BOOL
    8: None,        # STRING (special)
    9: None,        # ARRAY (special)
    10: ("Q", 8),   # UINT64
    11: ("q", 8),   # INT64
    12: ("d", 8),   # FLOAT64
}


class GGUFMeta:
    """Lightweight GGUF metadata extractor (no tensor data loading)."""

    def __init__(self, path: Path):
        self.path = path
        self._data = open(path, "rb").read()
        self._pos = 0
        self.tensors = {}  # name → {shape, type, offset}
        self._parse()

    def _read(self, fmt: str):
        size = struct.calcsize(fmt)
        val = struct.unpack_from(fmt, self._data, self._pos)
        self._pos += size
        return val[0] if len(val) == 1 else val

    def _read_str(self):
        length = self._read("<Q")
        s = self._data[self._pos:self._pos + length].decode("utf-8", errors="replace")
        self._pos += length
        return s

    def _skip_value(self, vtype: int):
        if vtype in GGUF_VALUE_TYPES and GGUF_VALUE_TYPES[vtype] is not None:
            _, size = GGUF_VALUE_TYPES[vtype]
            self._pos += size
        elif vtype == 8:  # STRING
            length = self._read("<Q")
            self._pos += length
        elif vtype == 9:  # ARRAY
            elem_type = self._read("<I")
            count = self._read("<Q")
            for _ in range(count):
                self._skip_value(elem_type)
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")

    def _parse(self):
        data = self._data
        magic = data[:4]
        if magic != GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={magic!r})")
        self._pos = 4
        version = self._read("<I")
        if version not in (2, 3):
            raise ValueError(f"Unsupported GGUF version: {version}")
        n_tensors = self._read("<Q")
        n_kv = self._read("<Q")

        # Skip KV pairs
        for _ in range(n_kv):
            self._read_str()          # key
            vtype = self._read("<I")  # value type
            self._skip_value(vtype)

        # Read tensor info
        tensor_infos = []
        for _ in range(n_tensors):
            name = self._read_str()
            n_dims = self._read("<I")
            dims = [self._read("<Q") for _ in range(n_dims)]
            ttype = self._read("<I")
            offset = self._read("<Q")
            tensor_infos.append((name, dims, ttype, offset))

        # Tensor data starts after alignment (GGUF aligns to 32 bytes by default)
        data_start = self._pos
        alignment = 32
        remainder = data_start % alignment
        if remainder != 0:
            data_start += alignment - remainder

        for name, dims, ttype, offset in tensor_infos:
            self.tensors[name] = {
                "dims": dims,        # in GGUF order: [n_cols, n_rows, ...]
                "type": ttype,
                "file_offset": data_start + offset,
            }

        self._data_start = data_start

    def get_tensor_raw(self, name: str) -> tuple:
        """Return (raw_bytes, dims, type) for a named tensor."""
        info = self.tensors[name]
        dims = info["dims"]
        ttype = info["type"]
        offset = info["file_offset"]

        if ttype == GGML_TYPE_I2_S:
            n_elems = 1
            for d in dims:
                n_elems *= d
            n_data_bytes = n_elems // 4 + 32  # packed + scale + alignment
        else:
            raise NotImplementedError(f"Tensor type {ttype} not supported (only I2_S=36)")

        raw = self._data[offset:offset + n_data_bytes]
        return raw, dims, ttype


# ──────────────────────────────────────────────────────────────────────────────
# I2_S decoding
# ──────────────────────────────────────────────────────────────────────────────

def decode_i2s_matrix(raw: bytes, n_rows: int, n_cols: int) -> np.ndarray:
    """Decode I2_S packed bytes to int8 ternary matrix {-1, 0, +1}.

    Layout: n_rows × (n_cols/4) bytes, organized in blocks of 128 values = 32 bytes.
    Within each 32-byte block, byte gp encodes 4 values at positions:
      0*32+gp, 1*32+gp, 2*32+gp, 3*32+gp  (from bits 7:6, 5:4, 3:2, 1:0).
    Map: 0→-1, 1→0, 2→+1, 3→0.
    """
    assert n_cols % 128 == 0, f"n_cols={n_cols} must be multiple of 128 for I2_S"
    n_blocks_per_row = n_cols // 128
    bytes_per_row = n_cols // 4

    raw_arr = np.frombuffer(raw, dtype=np.uint8)[:n_rows * bytes_per_row]
    raw_2d = raw_arr.reshape(n_rows, n_blocks_per_row, 32)  # [n_rows, n_blocks, 32]

    # Extract 4 groups from each byte
    g0 = (raw_2d >> 6) & 0x3  # [n_rows, n_blocks, 32] → positions 0*32+gp
    g1 = (raw_2d >> 4) & 0x3  # positions 1*32+gp
    g2 = (raw_2d >> 2) & 0x3  # positions 2*32+gp
    g3 = (raw_2d >> 0) & 0x3  # positions 3*32+gp

    # Stack groups: [n_rows, n_blocks, 4, 32] → [n_rows, n_cols]
    packed = np.stack([g0, g1, g2, g3], axis=2)  # [n_rows, n_blocks, 4, 32]
    packed = packed.reshape(n_rows, n_cols)

    # Map {0→-1, 1→0, 2→+1, 3→0}
    result = np.where(packed == 0, np.int8(-1),
             np.where(packed == 2, np.int8(1), np.int8(0)))
    return result.astype(np.int8)


def get_i2s_scale(raw: bytes, n_rows: int, n_cols: int) -> float:
    """Extract the global float32 scale from I2_S tensor data."""
    scale_offset = n_rows * n_cols // 4
    return struct.unpack_from("<f", raw, scale_offset)[0]


# ──────────────────────────────────────────────────────────────────────────────
# FWHT (Fast Walsh-Hadamard Transform)
# ──────────────────────────────────────────────────────────────────────────────

def fwht_inplace(a: np.ndarray):
    """In-place unnormalized Fast Walsh-Hadamard Transform. len(a) must be power of 2."""
    n = len(a)
    h = 1
    while h < n:
        a_view = a.reshape(-1, 2 * h)
        lo = a_view[:, :h].copy()
        hi = a_view[:, h:].copy()
        a_view[:, :h] = lo + hi
        a_view[:, h:] = lo - hi
        h *= 2


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ──────────────────────────────────────────────────────────────────────────────
# ACDC rectangular diagonal extraction
# ──────────────────────────────────────────────────────────────────────────────

def acdc_project_rect_numpy(W: np.ndarray, chunk_rows: int = 512) -> np.ndarray:
    """Compute d*[k] = (H_P · C)[k] / P² via XOR-convolution.

    W: int8 array [m, n], values {-1, 0, +1}
    P = next_pow2(max(m, n))
    chunk_rows: rows to process per NumPy call (controls memory use)
    Returns: float32 array [P]
    """
    m, n = W.shape
    P = next_pow2(max(m, n))
    C = np.zeros(P, dtype=np.float64)

    cols = np.arange(n, dtype=np.int32)

    for start in range(0, m, chunk_rows):
        end = min(start + chunk_rows, m)
        K = end - start
        rows = np.arange(start, end, dtype=np.int32)

        # XOR indices: [K, 1] ^ [1, n] → [K, n]
        xor_idx = (rows[:, None] ^ cols[None, :]).ravel()  # [K*n] int32
        w_flat = W[start:end].ravel().astype(np.float64)   # [K*n]

        np.add.at(C, xor_idx, w_flat)

    fwht_inplace(C)
    C /= (float(P) * float(P))
    return C.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# GGUF tensor → d* pipeline
# ──────────────────────────────────────────────────────────────────────────────

FFN_PROJ_NAMES = ("ffn_gate", "ffn_up", "ffn_down")


def discover_layers(gguf: GGUFMeta) -> dict:
    """Find all FFN projection tensors, return {layer_idx: {proj: tensor_name}}."""
    layers = {}
    for name in gguf.tensors:
        for proj in FFN_PROJ_NAMES:
            if f".{proj}.weight" in name and name.startswith("blk."):
                parts = name.split(".")
                layer = int(parts[1])
                layers.setdefault(layer, {})[proj] = name
    return layers


def process_tensor(gguf: GGUFMeta, tensor_name: str,
                   verbose: bool = True) -> tuple:
    """Decode I2_S tensor, compute d*, return (d_star, scale, shape, P)."""
    raw, dims, ttype = gguf.get_tensor_raw(tensor_name)
    if ttype != GGML_TYPE_I2_S:
        raise ValueError(f"{tensor_name}: type={ttype}, expected I2_S=36")

    # GGUF dims are [n_cols, n_rows, ...] (reversed from numpy)
    n_cols = int(dims[0])
    n_rows = int(dims[1]) if len(dims) > 1 else 1
    P = next_pow2(max(n_rows, n_cols))

    if verbose:
        print(f"      shape=[{n_rows},{n_cols}] P={P}", end=" ", flush=True)

    scale = get_i2s_scale(raw, n_rows, n_cols)

    # Decode ternary weights — skip if n_cols not multiple of 128
    if n_cols % 128 != 0:
        # Pad n_cols to next multiple of 128 for decoding
        pad_cols = (n_cols + 127) // 128 * 128
        W = np.zeros((n_rows, pad_cols), dtype=np.int8)
        W_partial = decode_i2s_matrix_unaligned(raw, n_rows, n_cols)
        W[:, :n_cols] = W_partial
        W = W  # keep padded
    else:
        W = decode_i2s_matrix(raw, n_rows, n_cols)

    t0 = time.time()
    d_star = acdc_project_rect_numpy(W)  # [P] float32
    elapsed = time.time() - t0

    if verbose:
        nnz = int(np.count_nonzero(W))
        total = n_rows * n_cols
        print(f"nnz={nnz/total:.2f} scale={scale:.4f} d*range=[{d_star.min():.4f},{d_star.max():.4f}] ({elapsed:.1f}s)")

    return d_star * scale, scale, (n_rows, n_cols), P


def decode_i2s_matrix_unaligned(raw: bytes, n_rows: int, n_cols: int) -> np.ndarray:
    """Decode I2_S for n_cols not multiple of 128 (pad last block)."""
    pad_cols = (n_cols + 127) // 128 * 128
    W = np.zeros((n_rows, pad_cols), dtype=np.int8)
    bytes_per_row = n_cols // 4
    n_blocks_per_row = (n_cols + 127) // 128
    raw_arr = np.frombuffer(raw, dtype=np.uint8)

    for r in range(n_rows):
        row_bytes = raw_arr[r * bytes_per_row:(r + 1) * bytes_per_row]
        for b in range(n_blocks_per_row):
            block_start = b * 32
            block_bytes = row_bytes[block_start:block_start + 32]
            n_in_block = min(128, n_cols - b * 128)
            n_bytes_in_block = (n_in_block + 3) // 4
            block_bytes = block_bytes[:n_bytes_in_block]
            for gp, byte_val in enumerate(block_bytes):
                for g in range(4):
                    pos = b * 128 + g * 32 + gp
                    if pos >= n_cols:
                        break
                    bits = (byte_val >> (6 - 2 * g)) & 0x3
                    W[r, pos] = np.int8(-1) if bits == 0 else (np.int8(1) if bits == 2 else np.int8(0))
    return W[:, :n_cols]


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract ACDC rectangular diagonals d* from BitNet I2_S GGUF.")
    parser.add_argument("gguf_path", type=Path, help="Path to .gguf model file")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output .npz path (default: <gguf>.acdc_diag.npz)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Process only this layer (debug)")
    parser.add_argument("--proj", type=str, default=None,
                        choices=list(FFN_PROJ_NAMES),
                        help="Process only this projection (debug)")
    parser.add_argument("--chunk-rows", type=int, default=512,
                        help="Rows per XOR-conv chunk (memory vs speed tradeoff, default=512)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    gguf_path = args.gguf_path.resolve()
    if not gguf_path.exists():
        print(f"[ERROR] File not found: {gguf_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or gguf_path.with_suffix(".acdc_diag.npz")
    out_path = out_path.resolve()

    print(f"[INFO] Reading GGUF metadata from {gguf_path.name}...")
    meta = GGUFMeta(gguf_path)

    layers = discover_layers(meta)
    if not layers:
        print("[ERROR] No FFN weight tensors found (expected blk.*.ffn_gate/up/down.weight)")
        sys.exit(1)

    layer_indices = sorted(layers.keys())
    if args.layer is not None:
        layer_indices = [args.layer]

    n_layers = max(layers.keys()) + 1
    print(f"[INFO] Found {n_layers} layers, {len(layers)} with FFN projections")

    # Peek at first tensor to get P
    first_layer = layer_indices[0]
    first_proj = next(iter(layers[first_layer]))
    first_name = layers[first_layer][first_proj]
    first_dims = meta.tensors[first_name]["dims"]
    P_example = next_pow2(max(int(first_dims[0]), int(first_dims[1])))
    print(f"[INFO] Sample: {first_name} dims={first_dims} P={P_example}")

    total = len(layer_indices) * len(FFN_PROJ_NAMES if args.proj is None else [args.proj])
    done = 0
    t_total = time.time()

    results = {}   # key → d_star array
    meta_info = {}

    for layer_idx in layer_indices:
        if layer_idx not in layers:
            print(f"  [SKIP] Layer {layer_idx}: no FFN tensors")
            continue

        projs_to_process = [args.proj] if args.proj else list(FFN_PROJ_NAMES)

        for proj in projs_to_process:
            tensor_name = layers[layer_idx].get(proj)
            if tensor_name is None:
                print(f"  [SKIP] Layer {layer_idx} {proj}: not found")
                continue

            done += 1
            if not args.quiet:
                print(f"  [{done}/{total}] {tensor_name}:", end=" ", flush=True)

            try:
                d_star, scale, shape, P = process_tensor(meta, tensor_name,
                                                          verbose=not args.quiet)
                key = tensor_name.replace(".weight", ".d_star")
                results[key] = d_star
                meta_info[tensor_name] = {
                    "shape": list(shape),
                    "P": P,
                    "scale": float(scale),
                    "d_star_norm": float(np.linalg.norm(d_star)),
                }
            except Exception as exc:
                print(f"\n  [ERROR] {tensor_name}: {exc}", file=sys.stderr)
                import traceback; traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"[INFO] Processed {len(results)} tensors in {elapsed:.1f}s")

    if not results:
        print("[ERROR] No results to save", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] Saving {out_path.name}...")
    np.savez_compressed(out_path, **results)

    meta_path = out_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({
            "model": str(gguf_path),
            "n_layers": n_layers,
            "P_example": P_example,
            "n_tensors": len(results),
            "elapsed_sec": elapsed,
            "tensors": meta_info,
        }, f, indent=2)

    size_mb = out_path.stat().st_size / 1e6
    print(f"[OK] Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"[OK] Meta: {meta_path}")


if __name__ == "__main__":
    main()
