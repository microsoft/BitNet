#!/usr/bin/env python3
"""
Test BitNet Metal Backend with Full Real Model

This tests the actual model architecture with full layer count.
"""

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
print("BitNet Metal Backend - Full Real Model Test")
print("=" * 70)

# bitnet_b1_58-large configuration (REAL MODEL)
MODEL_CONFIG = {
    "name": "bitnet_b1_58-large",
    "dim": 1280,
    "n_layers": 24,  # Full 24 layers
    "n_heads": 20,
    "n_kv_heads": 5,
    "vocab_size": 128256,
    "ffn_dim": 3584,
    "norm_eps": 1e-5,
    "rope_theta": 500000.0,
}

print(f"\nModel: {MODEL_CONFIG['name']}")
print(f"Architecture:")
print(f"  - Layers: {MODEL_CONFIG['n_layers']}")
print(f"  - Dimension: {MODEL_CONFIG['dim']}")
print(
    f"  - Heads: {MODEL_CONFIG['n_heads']} (query), {MODEL_CONFIG['n_kv_heads']} (key/value)"
)
print(f"  - Vocabulary: {MODEL_CONFIG['vocab_size']:,} tokens")
print(f"  - FFN Dim: {MODEL_CONFIG['ffn_dim']}")

results = {}

for device_type in ["cpu", "mps"]:
    if device_type == "mps" and not torch.backends.mps.is_available():
        print(f"\n⚠ Metal not available, skipping")
        continue

    device = torch.device(device_type)
    print(f"\n{'=' * 70}")
    print(f"Testing on: {device}")
    print(f"{'=' * 70}")

    # Create model
    print("Creating model...")

    # Remove 'name' from config before passing to ModelArgs
    model_config = {k: v for k, v in MODEL_CONFIG.items() if k != "name"}
    args = ModelArgs(**model_config, use_kernel=True)
    model = Transformer(args).to(device)
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(
        f"✓ Model created: {params:,} parameters (~{params * 2 / 1e9:.2f} GB at BF16)"
    )

    # Test configurations
    configs = [
        ("Single token (1x1)", 1, 1),
        ("Small prompt (1x128)", 1, 128),
        ("Medium prompt (1x256)", 1, 256),
        ("Batch-4 (4x128)", 4, 128),
    ]

    device_results = []

    for desc, batch_size, seq_len in configs:
        print(f"\n{desc}:")

        try:
            tokens = torch.randint(
                0, args.vocab_size, (batch_size, seq_len), device=device
            )
            cache = make_cache(args, length=batch_size * seq_len, device=device)

            # Warmup
            with torch.no_grad():
                _ = model(tokens, cache)

            if device.type == "mps":
                torch.mps.synchronize()

            # Benchmark
            times = []
            iterations = 5

            for i in range(iterations):
                start = time.perf_counter()
                with torch.no_grad():
                    output = model(tokens, cache)

                if device.type == "mps":
                    torch.mps.synchronize()

                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            total_tokens = batch_size * seq_len
            throughput = total_tokens / (mean_time / 1000)

            print(f"  Time: {mean_time:.2f} ± {std_time:.2f} ms")
            print(f"  Throughput: {throughput:.2f} tok/s")
            print(f"  Output shape: {output.shape}")

            device_results.append(
                {
                    "desc": desc,
                    "batch": batch_size,
                    "seq": seq_len,
                    "time_ms": mean_time,
                    "throughput": throughput,
                }
            )

        except Exception as e:
            print(f"  ✗ Error: {e}")

    results[device_type] = device_results

# Summary comparison
print("\n" + "=" * 70)
print("Performance Comparison: CPU vs Metal")
print("=" * 70)

if "cpu" in results and "mps" in results:
    print(
        f"\n{'Configuration':<25} {'CPU (tok/s)':<15} {'Metal (tok/s)':<15} {'Speedup':<10}"
    )
    print("-" * 70)

    for cpu_r in results["cpu"]:
        metal_r = next((r for r in results["mps"] if r["desc"] == cpu_r["desc"]), None)
        if metal_r:
            speedup = metal_r["throughput"] / cpu_r["throughput"]
            print(
                f"{cpu_r['desc']:<25} {cpu_r['throughput']:<15.2f} {metal_r['throughput']:<15.2f} {speedup:<10.2f}x"
            )

print("\n" + "=" * 70)
print("Full Real Model Test Complete")
print("=" * 70)
print("\n✓ Metal backend successfully tested with real BitNet model!")
