#!/usr/bin/env python3
"""
Test BitNet Metal Backend with Actual Model Configuration

This script tests the Metal backend with real BitNet model configurations
to ensure it works correctly with actual model architectures.
"""

import sys
import os
import time
import statistics

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)

import torch
from model import Transformer, ModelArgs, make_cache

# Real BitNet model configurations from the repository
MODEL_CONFIGS = {
    "bitnet_b1_58-large": {
        "dim": 1280,
        "n_layers": 24,
        "n_heads": 20,
        "n_kv_heads": 5,
        "vocab_size": 128256,
        "ffn_dim": 3584,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "bitnet_b1_58-3B": {
        "dim": 2560,
        "n_layers": 30,
        "n_heads": 20,
        "n_kv_heads": 5,
        "vocab_size": 128256,
        "ffn_dim": 6912,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "Llama3-8B-1.58-100B": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 128256,
        "ffn_dim": 14336,
        "norm_eps": 1e-5,
        "rope_theta": 500000.0,
    },
    "Falcon3-1B-1.58bit": {
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 32,
        "n_kv_heads": 8,
        "vocab_size": 131072,
        "ffn_dim": 8192,
        "norm_eps": 1e-5,
        "rope_theta": 10000.0,
    },
}


def test_model_config(
    name, config, device_type="mps", batch_sizes=[1, 4, 8], seq_lengths=[128, 256, 512]
):
    """Test a specific model configuration."""

    if device_type == "mps" and not torch.backends.mps.is_available():
        print(f"\n⚠ Skipping {name} on Metal - not available")
        return None

    device = torch.device(device_type)
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"Device: {device}")
    print(f"{'=' * 70}")

    try:
        # Create model with real configuration
        args = ModelArgs(**config, use_kernel=True)
        model = Transformer(args).to(device)
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        param_size_mb = params * 4 / (1024 * 1024)  # Assuming float32

        print(f"Parameters: {params:,} ({param_size_mb:.1f} MB estimated)")
        print(
            f"Architecture: {config['n_layers']} layers, {config['dim']} dim, {config['n_heads']} heads"
        )

        results = []

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                # Skip very large configurations
                if batch_size * seq_len > 4096:
                    continue

                print(f"\n  Batch={batch_size}, Seq={seq_len}:")

                # Create input
                tokens = torch.randint(
                    0, args.vocab_size, (batch_size, seq_len), device=device
                )
                cache = make_cache(args, length=batch_size * seq_len, device=device)

                # Warmup
                with torch.no_grad():
                    for _ in range(2):
                        _ = model(tokens, cache)

                if device.type == "mps":
                    torch.mps.synchronize()

                # Benchmark
                times = []
                iterations = 5

                for _ in range(iterations):
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

                print(f"    Time: {mean_time:.2f} ± {std_time:.2f} ms")
                print(f"    Throughput: {throughput:.2f} tok/s")
                print(f"    Output: {output.shape}")

                results.append(
                    {
                        "batch": batch_size,
                        "seq": seq_len,
                        "time_ms": mean_time,
                        "throughput": throughput,
                        "tokens": total_tokens,
                    }
                )

        return {
            "name": name,
            "params": params,
            "config": config,
            "device": device_type,
            "results": results,
        }

    except Exception as e:
        print(f"\n✗ Error testing {name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    print("=" * 70)
    print("BitNet Metal Backend - Real Model Testing")
    print("=" * 70)
    print(f"PyTorch: {torch.__version__}")
    print(f"Metal Available: {torch.backends.mps.is_available()}")

    # Test on both CPU and Metal
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    all_results = []

    # Test smaller models first
    test_models = ["bitnet_b1_58-large", "Falcon3-1B-1.58bit"]

    for device in devices:
        print(f"\n{'=' * 70}")
        print(f"Testing on {device.upper()}")
        print(f"{'=' * 70}")

        for model_name in test_models:
            if model_name in MODEL_CONFIGS:
                result = test_model_config(
                    model_name,
                    MODEL_CONFIGS[model_name],
                    device_type=device,
                    batch_sizes=[1, 4],
                    seq_lengths=[128, 256],
                )
                if result:
                    all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("Real Model Test Summary")
    print("=" * 70)

    if all_results:
        for result in all_results:
            print(f"\n{result['name']} ({result['device']}):")
            print(f"  Parameters: {result['params']:,}")
            for r in result["results"]:
                print(
                    f"  Batch={r['batch']}, Seq={r['seq']}: {r['time_ms']:.2f} ms, {r['throughput']:.2f} tok/s"
                )

        # Compare CPU vs Metal
        print("\n" + "=" * 70)
        print("Performance Comparison (CPU vs Metal)")
        print("=" * 70)

        for model_name in test_models:
            cpu_result = next(
                (
                    r
                    for r in all_results
                    if r["name"] == model_name and r["device"] == "cpu"
                ),
                None,
            )
            metal_result = next(
                (
                    r
                    for r in all_results
                    if r["name"] == model_name and r["device"] == "mps"
                ),
                None,
            )

            if cpu_result and metal_result:
                print(f"\n{model_name}:")
                for cpu_r, metal_r in zip(
                    cpu_result["results"], metal_result["results"]
                ):
                    if (
                        cpu_r["batch"] == metal_r["batch"]
                        and cpu_r["seq"] == metal_r["seq"]
                    ):
                        speedup = metal_r["throughput"] / cpu_r["throughput"]
                        print(
                            f"  Batch={cpu_r['batch']}, Seq={cpu_r['seq']}: {speedup:.2f}x faster on Metal"
                        )
    else:
        print("No results collected.")

    print("\n" + "=" * 70)
    print("Real Model Testing Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
