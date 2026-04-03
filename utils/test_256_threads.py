#!/usr/bin/env python3
"""Test Metal backend with 256-thread configuration"""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)

import torch
import time
from model import Transformer, ModelArgs, make_cache

print("=" * 70)
print("BitNet Metal Backend - 256 Thread Configuration Test")
print("=" * 70)

# Test configurations designed to exercise 256 threads
configs = [
    (
        "Small",
        dict(dim=256, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=1000),
        1,
        128,
    ),
    (
        "Medium",
        dict(dim=512, n_layers=4, n_heads=8, n_kv_heads=4, vocab_size=10000),
        4,
        256,
    ),
    (
        "Large",
        dict(dim=1024, n_layers=4, n_heads=16, n_kv_heads=8, vocab_size=50000),
        8,
        256,
    ),
    (
        "256-Thread Test",
        dict(dim=2560, n_layers=1, n_heads=32, n_kv_heads=8, vocab_size=128256),
        16,
        256,
    ),
]

# Check device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"\n✓ Using Metal (MPS) on: {device}")
else:
    device = torch.device("cpu")
    print(f"\n⚠ Metal not available, using: {device}")

print(f"\nPyTorch version: {torch.__version__}")

# Run tests
for name, model_config, batch_size, seq_len in configs:
    print(f"\n{'-' * 70}")
    print(f"Test: {name}")
    print(
        f"Model: dim={model_config['dim']}, layers={model_config['n_layers']}, heads={model_config['n_heads']}"
    )
    print(f"Input: batch={batch_size}, seq={seq_len}")
    print(f"{'-' * 70}")

    try:
        # Create model
        args = ModelArgs(**model_config, use_kernel=True)
        model = Transformer(args).to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")

        # Create input
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
        cache = make_cache(args, length=batch_size * seq_len, device=device)

        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = model(tokens, cache)

        if device.type == "mps":
            torch.mps.synchronize()

        # Benchmark
        print("Benchmarking...")
        times = []
        iterations = 10

        for i in range(iterations):
            start = time.perf_counter()
            with torch.no_grad():
                output = model(tokens, cache)

            if device.type == "mps":
                torch.mps.synchronize()

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        total_tokens = batch_size * seq_len
        throughput = total_tokens / (mean_time / 1000)

        print(f"\nResults:")
        print(f"  Mean time: {mean_time:.2f} ms")
        print(f"  Min/Max: {min_time:.2f} / {max_time:.2f} ms")
        print(f"  Throughput: {throughput:.2f} tokens/sec")
        print(f"  Output shape: {output.shape}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()

print("\n" + "=" * 70)
print("256 Thread Configuration Test Complete")
print("=" * 70)
