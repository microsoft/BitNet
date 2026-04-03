#!/usr/bin/env python3
"""Quick 256-thread performance test"""

import sys
import os

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)

import torch
import time
import statistics
from model import Transformer, ModelArgs, make_cache

print("=" * 70)
print("BitNet Metal Backend - 256 Thread Performance Test")
print("=" * 70)

# Configuration optimized for 256 threads
args = ModelArgs(
    dim=2560, n_layers=1, n_heads=32, n_kv_heads=8, vocab_size=128256, use_kernel=True
)

# Test with different batch sizes to exercise 256 threads
configs = [
    ("256 threads (batch=16, seq=16)", 16, 16),
    ("512 threads (batch=32, seq=16)", 32, 16),
    ("1024 threads (batch=32, seq=32)", 32, 32),
]

for device_type in ["cpu", "mps"]:
    if device_type == "mps" and not torch.backends.mps.is_available():
        print(f"\n⚠ Skipping Metal - not available")
        continue

    device = torch.device(device_type)
    print(f"\n{'=' * 70}")
    print(f"Testing on: {device}")
    print(f"{'=' * 70}")

    model = Transformer(args).to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} parameters")

    for name, batch_size, seq_len in configs:
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len), device=device)
        cache = make_cache(args, length=batch_size * seq_len, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(tokens, cache)

        if device.type == "mps":
            torch.mps.synchronize()

        # Benchmark
        times = []
        for _ in range(10):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(tokens, cache)
            if device.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - start) * 1000)

        mean_time = statistics.mean(times)
        total_tokens = batch_size * seq_len
        throughput = total_tokens / (mean_time / 1000)

        print(f"\n  {name}:")
        print(f"    Time: {mean_time:.2f} ms")
        print(f"    Throughput: {throughput:.2f} tokens/sec")
        print(f"    Total tokens processed: {total_tokens}")

print("\n" + "=" * 70)
print("256 Thread Test Complete")
print("=" * 70)
print("\nNote: The Metal backend uses 256 threads per threadgroup")
print("(configured as 32x8 in the dispatch for better memory access patterns)")
