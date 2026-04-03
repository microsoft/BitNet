#!/usr/bin/env python3
"""Debug test for transformer shapes"""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)

import torch
from model import Transformer, ModelArgs, make_cache
import torch.nn as nn

# Patch Attention to add debugging
original_forward = None


def debug_forward(self, x, cache):
    print(f"\n=== Attention Debug ===")
    print(f"Input x shape: {x.shape}")

    xqkv = self.wqkv(x)
    print(f"xqkv shape: {xqkv.shape}")

    xq = xqkv[:, : (self.n_local_heads * self.head_dim)]
    xkv = xqkv[:, (self.n_local_heads * self.head_dim) :]
    xk, xv = xkv.chunk(2, 1)

    print(f"xq shape: {xq.shape}")
    print(f"xk shape: {xk.shape}")
    print(f"xv shape: {xv.shape}")

    xq_rs = xq.view(-1, self.n_local_heads, self.head_dim)
    xk_rs = xk.view(-1, self.n_local_kv_heads, self.head_dim)
    xv_rs = xv.view(-1, self.n_local_kv_heads, self.head_dim)

    print(f"xq reshaped: {xq_rs.shape}")
    print(f"xk reshaped: {xk_rs.shape}")
    print(f"xv reshaped: {xv_rs.shape}")

    heads_per_group = self.n_local_heads // self.n_local_kv_heads
    print(f"heads_per_group: {heads_per_group}")

    xq_grouped = xq_rs.view(-1, self.n_local_kv_heads, heads_per_group, self.head_dim)
    print(f"xq_grouped: {xq_grouped.shape}")

    xk_expanded = xk_rs.unsqueeze(2).expand(-1, -1, heads_per_group, -1)
    print(f"xk_expanded: {xk_expanded.shape}")

    xk_T = xk_expanded.transpose(-2, -1)
    print(f"xk_expanded transposed: {xk_T.shape}")

    print(f"\nAttempting matmul...")
    print(f"  xq_grouped: {xq_grouped.shape}")
    print(f"  xk_T: {xk_T.shape}")

    try:
        scores = torch.matmul(xq_grouped, xk_T)
        print(f"scores: {scores.shape}")
    except Exception as e:
        print(f"Error in matmul: {e}")
        raise

    raise RuntimeError("Debug stop")


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

# Replace attention forward
from model import Attention

original_forward = Attention.forward
Attention.forward = debug_forward

# Create test input
batch_size = 1
seq_len = 4
tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))
cache = make_cache(args, length=batch_size * seq_len)

print(f"Input tokens shape: {tokens.shape}")

# Forward pass
try:
    with torch.no_grad():
        output = model(tokens, cache)
except RuntimeError as e:
    if "Debug stop" in str(e):
        print("\nDebug completed")
    else:
        raise
