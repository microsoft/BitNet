#!/usr/bin/env python3
"""Quick real model test - smaller model only"""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)

import torch
import time
from model import Transformer, ModelArgs, make_cache

print("Quick Real Model Test")
print("=" * 60)

# Use the smallest real model configuration
config = {
    "dim": 1280,  # bitnet_b1_58-large
    "n_layers": 4,  # Reduced for faster testing
    "n_heads": 20,
    "n_kv_heads": 5,
    "vocab_size": 128256,
    "ffn_dim": 3584,
    "norm_eps": 1e-5,
    "rope_theta": 500000.0,
}

for device_type in ["cpu", "mps"]:
    if device_type == "mps" and not torch.backends.mps.is_available():
        print(f"\n⚠ Skipping Metal - not available")
        continue

    device = torch.device(device_type)
    print(f"\nDevice: {device}")
    print(f"Config: {config['dim']} dim, {config['n_layers']} layers")

    # Create model
    args = ModelArgs(**config, use_kernel=True)
    print(f"Creating model...")
    model = Transformer(args).to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")

    # Test with batch=1, seq=128
    batch_size, seq_len = 1, 128
    print(f"\nTesting batch={batch_size}, seq={seq_len}...")

    tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
    cache = make_cache(args, length=batch_size * seq_len, device=device)

    # Single forward pass
    print("Running forward pass...")
    start = time.perf_counter()
    with torch.no_grad():
        output = model(tokens, cache)

    if device.type == "mps":
        torch.mps.synchronize()

    elapsed = (time.perf_counter() - start) * 1000
    throughput = (batch_size * seq_len) / (elapsed / 1000)

    print(f"✓ Success!")
    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Throughput: {throughput:.2f} tok/s")
    print(f"  Output: {output.shape}")

print("\n" + "=" * 60)
print("Test Complete")
