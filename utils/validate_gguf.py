#!/usr/bin/env python3
"""Validate a GGUF file for expected BitNet fields and tensor shapes.

Usage: utils/validate_gguf.py path/to/model.gguf
Exits 0 on success, non-zero on validation failure.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Any

def fail(msg: str) -> None:
    print("ERROR:", msg)
    sys.exit(2)

def info(msg: str) -> None:
    print(" -", msg)

def get_kv(reader, key: str):
    f = reader.fields.get(key)
    if f is None:
        return None
    try:
        return f.parts[f.data[0]]
    except Exception:
        return None

def main() -> int:
    try:
        from gguf import GGUFReader
    except Exception as e:
        fail(f"Could not import gguf: {e}")

    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    p = Path(sys.argv[1])
    if not p.exists():
        fail(f"File not found: {p}")

    print(f"Validating GGUF: {p}")
    try:
        r = GGUFReader(str(p))
    except Exception as e:
        fail(f"Failed to open GGUF: {e}")

    # required keys for BitNet models
    required = [
        "general.architecture",
        "bitnet.block_count",
        "bitnet.context_length",
        "bitnet.embedding_length",
        "bitnet.vocab_size",
    ]

    missing = []
    for k in required:
        v = get_kv(r, k)
        if v is None:
            missing.append(k)
    if missing:
        fail(f"Missing required GGUF keys: {missing}")

    vocab_size = int(get_kv(r, "bitnet.vocab_size"))
    emb_len = int(get_kv(r, "bitnet.embedding_length"))
    block_count = int(get_kv(r, "bitnet.block_count"))

    info(f"vocab_size={vocab_size}, embedding_length={emb_len}, block_count={block_count}")

    # find embedding tensor
    emb_tensor = None
    for t in r.tensors:
        name = getattr(t, "name", None) or getattr(t, "tensor_name", None) or ""
        # common names: token_embd.weight, model.embed_tokens.weight
        if "token_embd" in name or "embed_tokens" in name or "token_emb" in name:
            emb_tensor = t
            break

    if emb_tensor is None:
        fail("Embedding tensor not found (expected token_embd or embed_tokens)")

    shape = tuple(getattr(emb_tensor, "shape", getattr(emb_tensor, "tensor_shape", [])))
    # normalize remove trailing ones
    shape = tuple(x for x in shape if x != 1)
    info(f"Found embedding tensor '{emb_tensor.name}' shape={shape}")

    # check that embedding length and vocab size appear in shape
    if not (emb_len in shape and vocab_size in shape):
        fail(f"Embedding tensor shape {shape} does not include embedding_length {emb_len} and vocab_size {vocab_size}")

    # basic tensor count sanity: at least one tensor per block expected
    tensors_count = len(r.tensors)
    info(f"Total tensors: {tensors_count}")
    if tensors_count < 10:
        fail(f"Too few tensors in GGUF ({tensors_count}) — likely corrupted or incomplete")

    print("Validation passed")
    return 0

if __name__ == '__main__':
    sys.exit(main())
