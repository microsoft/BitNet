#!/usr/bin/env python3
"""Debug test for attention shapes"""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)

import torch
from model import Attention, ModelArgs

# Test attention with known shapes
args = ModelArgs(
    dim=512,
    n_layers=1,
    n_heads=8,
    n_kv_heads=2,
    vocab_size=1000,
)

head_dim = args.dim // args.n_heads  # 512 / 8 = 64
print(f"dim={args.dim}, n_heads={args.n_heads}, head_dim={head_dim}")
print(f"n_kv_heads={args.n_kv_heads}")
print(f"heads_per_group={args.n_heads // args.n_kv_heads}")

attn = Attention(
    dim=args.dim,
    head_dim=head_dim,
    n_heads=args.n_heads,
    n_kv_heads=args.n_kv_heads,
    rope_theta=args.rope_theta,
    norm_eps=args.norm_eps,
    use_kernel=False,
)

# Create test input: batch=1, seq=16, tokens flattened
batch_size = 1
seq_len = 16
total_tokens = batch_size * seq_len
x = torch.randn(total_tokens, args.dim)

print(f"\nInput shape: {x.shape}")

# Check wqkv output shape
xqkv = attn.wqkv(x)
print(f"xqkv shape: {xqkv.shape}")
print(f"Expected: [{total_tokens}, {(args.n_heads + 2 * args.n_kv_heads) * head_dim}]")

xq = xqkv[:, : (attn.n_local_heads * attn.head_dim)]
xkv = xqkv[:, (attn.n_local_heads * attn.head_dim) :]
xk, xv = xkv.chunk(2, 1)

print(f"\nxq shape: {xq.shape}")
print(f"xk shape: {xk.shape}")
print(f"xv shape: {xv.shape}")

print(f"\nReshaping xq: {-1}, {attn.n_local_heads}, {attn.head_dim}")
xq_reshaped = xq.view(-1, attn.n_local_heads, attn.head_dim)
print(f"xq reshaped: {xq_reshaped.shape}")

print(f"\nReshaping xk: {-1}, {attn.n_local_kv_heads}, {attn.head_dim}")
xk_reshaped = xk.view(-1, attn.n_local_kv_heads, attn.head_dim)
print(f"xk reshaped: {xk_reshaped.shape}")

# Try forward
print("\n\nTrying forward pass...")
try:
    cache = (torch.zeros(1, 1, 1, 1, 1), torch.zeros(1, 1, 1, 1, 1))
    output = attn(x, cache)
    print(f"Success! Output shape: {output.shape}")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
