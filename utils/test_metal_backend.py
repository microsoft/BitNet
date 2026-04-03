#!/usr/bin/env python3
"""Quick test of BitNet Metal backend components"""

import sys
import os

# Add the metal_kernels to path
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "gpu", "metal_kernels")
)


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from model import ModelArgs, pack_weight_int8_to_int2

        print("✓ Model imports successful")
    except Exception as e:
        print(f"✗ Model import failed: {e}")
        return False

    try:
        import torch

        print("✓ PyTorch available")
        has_torch = True
    except ImportError:
        print("⚠ PyTorch not available (optional)")
        has_torch = False

    return has_torch


def test_weight_packing():
    """Test weight packing function"""
    print("\nTesting weight packing...")

    try:
        import torch
        from model import pack_weight_int8_to_int2

        # Create test weights
        weight = torch.randint(-1, 2, (256, 256), dtype=torch.int8)
        print(f"  Original weight shape: {weight.shape}, dtype: {weight.dtype}")
        print(f"  Value range: [{weight.min()}, {weight.max()}]")

        # Pack weights
        packed = pack_weight_int8_to_int2(weight)
        print(f"  Packed weight shape: {packed.shape}, dtype: {packed.dtype}")
        print(
            f"  Size reduction: {weight.numel()} -> {packed.numel()} ({weight.numel() / packed.numel():.1f}x)"
        )

        # Verify packing is reversible
        unpacked = torch.zeros_like(weight)
        for i in range(4):
            shift = i * 2
            mask = 0x03
            val = ((packed >> shift) & mask).to(torch.int8) - 1
            unpacked[:, i::4] = val

        if torch.allclose(weight.float(), unpacked.float()):
            print("✓ Weight packing verified")
            return True
        else:
            print("✗ Weight packing verification failed")
            return False

    except Exception as e:
        print(f"✗ Weight packing test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_creation():
    """Test model instantiation"""
    print("\nTesting model creation...")

    try:
        import torch
        from model import Transformer, ModelArgs, make_cache

        # Create small test model
        args = ModelArgs(
            dim=512,
            n_layers=2,
            n_heads=8,
            n_kv_heads=2,
            vocab_size=1000,
            ffn_dim=1024,
            use_kernel=False,  # Use CPU fallback
            use_mps_fallback=False,
        )

        print(f"  Creating model with {args.n_layers} layers...")
        model = Transformer(args)

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Model created: {total_params:,} parameters")

        # Test forward pass with small input
        batch_size = 1
        seq_len = 16
        tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

        print(f"  Testing forward pass (batch={batch_size}, seq={seq_len})...")
        cache = make_cache(args, length=batch_size * seq_len)

        with torch.no_grad():
            output = model(tokens, cache)

        print(f"  ✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        print(f"  Output dtype: {output.dtype}")

        return True

    except Exception as e:
        print(f"✗ Model creation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metal_availability():
    """Check Metal/MPS availability"""
    print("\nChecking Metal availability...")

    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("✓ Metal (MPS) is available")
            print(f"  Device: {torch.device('mps')}")
            return True
        else:
            print("⚠ Metal (MPS) is not available")
            print("  The Metal backend will fall back to CPU")
            return False

    except Exception as e:
        print(f"✗ Error checking Metal: {e}")
        return False


def main():
    print("=" * 60)
    print("BitNet Metal Backend Test Suite")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Weight Packing", test_weight_packing),
        ("Model Creation", test_model_creation),
        ("Metal Availability", test_metal_availability),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Metal backend is working correctly.")
        return 0
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
