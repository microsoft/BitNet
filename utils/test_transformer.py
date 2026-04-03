#!/usr/bin/env python3
"""Simple test of Transformer model"""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)

import torch
from model import Transformer, ModelArgs, make_cache

# Create small model
args = ModelArgs(
    dim=512,
    n_layers=1,
    n_heads=8,
    n_kv_heads=2,
    vocab_size=1000,
    ffn_dim=1024,
    use_kernel=False,
    use_mps_fallback=False,
)

print("Creating model...")
model = Transformer(args)
print(f"Model created: {sum(p.numel() for p in model.parameters()):,} parameters")

# Create test input
batch_size = 1
seq_len = 4
tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
print(f"\nInput tokens shape: {tokens.shape}")

# Create cache
print("Creating cache...")
cache = make_cache(args, length=batch_size * seq_len)
print(f"Cache length: {len(cache)} layers")

# Forward pass
print("\nRunning forward pass...")
try:
    with torch.no_grad():
        output = model(tokens, cache)
    print(f"✓ Success! Output shape: {output.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback

    traceback.print_exc()
