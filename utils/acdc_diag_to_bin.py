#!/usr/bin/env python3
#
# acdc_diag_to_bin.py
#
# Converts the .acdc_diag.npz sidecar (from extract_acdc_diagonals.py) to a
# simple flat binary file that the C dispatch can mmap at inference time.
#
# Binary format (little-endian):
#   magic:    uint8[8]   = b"ACDBD\x01\x00\x00"
#   n_layers: uint32
#   n_proj:   uint32     = 2 (up, down per layer)
#   P:        uint32
#   reserved: uint32     = 0
#   data:     float32[n_layers × n_proj × P]
#             indexed: [layer * n_proj * P + proj * P + k]
#             proj 0 = ffn_up (or gate approximation)
#             proj 1 = ffn_down
#
# Usage:
#   python utils/acdc_diag_to_bin.py ggml-model-i2_s.acdc_diag.npz
#   → writes ggml-model-i2_s.acdc_diag.bin

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"ACDBD\x01\x00\x00"


def main():
    ap = argparse.ArgumentParser(
        description="Convert ACDC diag NPZ sidecar to flat binary for C dispatch")
    ap.add_argument("npz", type=Path, help="Input .acdc_diag.npz file")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .bin path (default: replace .npz → .bin)")
    ap.add_argument("--proj", nargs=2, default=["ffn_up", "ffn_down"],
                    metavar=("PROJ0", "PROJ1"),
                    help="Projection names to embed (default: ffn_up ffn_down)")
    args = ap.parse_args()

    npz_path = args.npz.resolve()
    if not npz_path.exists():
        print(f"[ERROR] Not found: {npz_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out or npz_path.with_suffix(".bin")
    out_path = out_path.resolve()

    data = np.load(npz_path)
    keys = [k for k in data.files if k != "_metadata_arr"]
    print(f"[INFO] Loaded {len(keys)} arrays from {npz_path.name}")

    # Find n_layers and P
    # Keys look like: blk.0.ffn_up.d_star, blk.0.ffn_down.d_star, ...
    layers = {}
    for k in keys:
        parts = k.split(".")
        if len(parts) < 3 or parts[0] != "blk":
            continue
        layer = int(parts[1])
        proj = parts[2]  # e.g. "ffn_up"
        layers.setdefault(layer, {})[proj] = k

    if not layers:
        print("[ERROR] No blk.*.ffn_*.d_star keys found", file=sys.stderr)
        sys.exit(1)

    n_layers = max(layers.keys()) + 1
    proj_names = args.proj  # e.g. ["ffn_up", "ffn_down"]
    n_proj = len(proj_names)

    # Determine P from first available array
    P = None
    for layer_idx in sorted(layers.keys()):
        for proj in proj_names:
            key = layers[layer_idx].get(proj)
            if key and key in data:
                P = data[key].shape[0]
                break
        if P is not None:
            break

    if P is None:
        print("[ERROR] Could not determine P", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] n_layers={n_layers}, n_proj={n_proj} {proj_names}, P={P}")

    # Build flat array [n_layers, n_proj, P]
    flat = np.zeros((n_layers, n_proj, P), dtype=np.float32)

    missing = 0
    for layer_idx in range(n_layers):
        for pi, proj in enumerate(proj_names):
            key = layers.get(layer_idx, {}).get(proj)
            if key and key in data:
                arr = data[key].astype(np.float32)
                if arr.shape[0] != P:
                    print(f"  [WARN] {key}: P={arr.shape[0]} ≠ expected {P}; skipping")
                    missing += 1
                    continue
                flat[layer_idx, pi, :] = arr
            else:
                print(f"  [WARN] Missing: blk.{layer_idx}.{proj}.d_star")
                missing += 1

    if missing:
        print(f"[WARN] {missing} missing/mismatched tensors (filled with zeros)")

    # Write binary
    header = struct.pack("<8sIIII",
                         MAGIC,
                         n_layers,
                         n_proj,
                         P,
                         0)  # reserved
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(flat.tobytes())

    size_mb = out_path.stat().st_size / 1e6
    print(f"[OK] Written: {out_path} ({size_mb:.2f} MB)")
    print(f"     Format: n_layers={n_layers}, n_proj={n_proj}, P={P}")
    print(f"     Set env: BITNET_ACDC_FFN_RECT_DIAG={out_path}")


if __name__ == "__main__":
    main()
