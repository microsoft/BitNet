#!/usr/bin/env python3
"""
BitNet Inference Profiler

Compares performance across CPU SIMD, Metal, and CUDA backends.
Usage:
    python utils/profile_inference.py --model <path> --backend metal --batch-sizes 1,8,16
    python utils/profile_inference.py --model <path> --backend all --profile
"""

import argparse
import sys
import time
import json
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
import platform

import torch
import numpy as np

# Try importing different backends
try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "gpu"))
    import gpu.model as cuda_model

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False

try:
    sys.path.insert(0, str(Path(__file__).parent.parent / "gpu" / "metal_kernels"))
    import metal_kernels.model as metal_model

    METAL_AVAILABLE = (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
except ImportError:
    METAL_AVAILABLE = False

# CPU is always available
CPU_AVAILABLE = True


@dataclass
class ProfileResult:
    """Results from a single profiling run."""

    backend: str
    batch_size: int
    seq_length: int
    model_dim: int

    # Timing (in milliseconds)
    warmup_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float

    # Throughput
    tokens_per_sec: float

    # Memory (if available)
    memory_mb: Optional[float] = None

    # Additional metrics
    iterations: int = 100


class InferenceProfiler:
    """Profiles BitNet inference across different backends."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results: List[ProfileResult] = []
        self.device_info = self._get_device_info()

    def _get_device_info(self) -> Dict:
        """Get system and device information."""
        info = {
            "platform": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cpu_count": torch.get_num_threads(),
        }

        if CUDA_AVAILABLE:
            info["cuda_available"] = True
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
        else:
            info["cuda_available"] = False

        if METAL_AVAILABLE:
            info["metal_available"] = True
        else:
            info["metal_available"] = False

        return info

    def _create_test_input(
        self, batch_size: int, seq_length: int, device: str
    ) -> torch.Tensor:
        """Create test input tensor."""
        return torch.randint(
            0, self.args.vocab_size, (batch_size, seq_length), device=device
        )

    def _profile_backend(
        self,
        backend: str,
        batch_size: int,
        seq_length: int,
        warmup_iters: int = 10,
        test_iters: int = 100,
    ) -> Optional[ProfileResult]:
        """Profile a specific backend."""

        print(f"\nProfiling {backend} backend - Batch: {batch_size}, Seq: {seq_length}")
        print("-" * 60)

        try:
            # Setup device and model
            if backend == "cuda":
                if not CUDA_AVAILABLE:
                    print(f"  SKIPPED: CUDA not available")
                    return None
                device = torch.device("cuda:0")
                model_args = cuda_model.ModelArgs(use_kernel=True)
                model = cuda_model.Transformer(model_args).to(device)
                dtype = torch.bfloat16

            elif backend == "metal":
                if not METAL_AVAILABLE:
                    print(f"  SKIPPED: Metal/MPS not available")
                    return None
                device = torch.device("mps")
                model_args = metal_model.ModelArgs(use_kernel=True)
                model = metal_model.Transformer(model_args).to(device)
                dtype = torch.bfloat16

            elif backend == "cpu":
                device = torch.device("cpu")
                # Use Metal model but with CPU fallback
                model_args = metal_model.ModelArgs(
                    use_kernel=False, use_mps_fallback=False
                )
                model = metal_model.Transformer(model_args).to(device)
                dtype = torch.float32
                # Optimize for CPU
                torch.set_num_threads(self.args.threads)
            else:
                print(f"  ERROR: Unknown backend {backend}")
                return None

            # Set model to eval mode
            model.eval()

            # Create cache
            cache = metal_model.make_cache(
                model_args, length=batch_size * seq_length, device=device, dtype=dtype
            )

            # Warmup
            print(f"  Warming up ({warmup_iters} iterations)...")
            warmup_start = time.perf_counter()

            for _ in range(warmup_iters):
                tokens = self._create_test_input(batch_size, seq_length, device)
                with torch.no_grad():
                    _ = model(tokens, cache)

                if backend == "cuda":
                    torch.cuda.synchronize()
                elif backend == "metal":
                    torch.mps.synchronize()

            warmup_time = (time.perf_counter() - warmup_start) * 1000
            print(f"  Warmup time: {warmup_time:.2f} ms")

            # Profile
            print(f"  Running {test_iters} iterations...")
            times = []

            for i in range(test_iters):
                tokens = self._create_test_input(batch_size, seq_length, device)

                if backend == "cuda":
                    torch.cuda.synchronize()
                elif backend == "metal":
                    torch.mps.synchronize()

                start = time.perf_counter()

                with torch.no_grad():
                    _ = model(tokens, cache)

                if backend == "cuda":
                    torch.cuda.synchronize()
                elif backend == "metal":
                    torch.mps.synchronize()

                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            # Calculate statistics
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0
            min_time = min(times)
            max_time = max(times)

            # Calculate throughput
            total_tokens = batch_size * seq_length
            tokens_per_sec = total_tokens / (mean_time / 1000)

            # Get memory usage
            memory_mb = None
            if backend == "cuda":
                memory_mb = torch.cuda.max_memory_allocated() / 1e6
            elif backend == "metal":
                # MPS doesn't expose memory stats directly
                memory_mb = None

            result = ProfileResult(
                backend=backend,
                batch_size=batch_size,
                seq_length=seq_length,
                model_dim=self.args.dim,
                warmup_time_ms=warmup_time,
                mean_time_ms=mean_time,
                std_time_ms=std_time,
                min_time_ms=min_time,
                max_time_ms=max_time,
                tokens_per_sec=tokens_per_sec,
                memory_mb=memory_mb,
                iterations=test_iters,
            )

            self._print_result(result)
            return result

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _print_result(self, result: ProfileResult):
        """Print profiling result."""
        print(f"\n  Results for {result.backend}:")
        print(f"    Mean time: {result.mean_time_ms:.3f} ± {result.std_time_ms:.3f} ms")
        print(f"    Min/Max: {result.min_time_ms:.3f} / {result.max_time_ms:.3f} ms")
        print(f"    Throughput: {result.tokens_per_sec:.2f} tokens/sec")
        if result.memory_mb:
            print(f"    Memory: {result.memory_mb:.2f} MB")

    def run(self):
        """Run profiling for all specified backends and configurations."""
        print("=" * 70)
        print("BitNet Inference Profiler")
        print("=" * 70)
        print(f"\nDevice Information:")
        for key, value in self.device_info.items():
            print(f"  {key}: {value}")

        # Determine which backends to test
        if self.args.backend == "all":
            backends = []
            if CPU_AVAILABLE:
                backends.append("cpu")
            if METAL_AVAILABLE:
                backends.append("metal")
            if CUDA_AVAILABLE:
                backends.append("cuda")
        else:
            backends = [self.args.backend]

        # Parse batch sizes
        batch_sizes = [int(x) for x in self.args.batch_sizes.split(",")]
        seq_lengths = [int(x) for x in self.args.seq_lengths.split(",")]

        print(f"\nBackends to test: {backends}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"Sequence lengths: {seq_lengths}")
        print("=" * 70)

        # Run profiling
        for backend in backends:
            for batch_size in batch_sizes:
                for seq_length in seq_lengths:
                    result = self._profile_backend(
                        backend,
                        batch_size,
                        seq_length,
                        self.args.warmup_iterations,
                        self.args.test_iterations,
                    )
                    if result:
                        self.results.append(result)

        # Generate report
        self._generate_report()

    def _generate_report(self):
        """Generate and save profiling report."""
        print("\n" + "=" * 70)
        print("Profiling Summary")
        print("=" * 70)

        if not self.results:
            print("No results to report.")
            return

        # Group results by configuration
        configs = {}
        for result in self.results:
            key = (result.batch_size, result.seq_length)
            if key not in configs:
                configs[key] = []
            configs[key].append(result)

        # Print comparison table
        for (batch, seq), results in configs.items():
            print(f"\nConfiguration: Batch={batch}, Seq={seq}")
            print("-" * 70)
            print(
                f"{'Backend':<12} {'Time (ms)':<15} {'Tokens/sec':<15} {'Speedup':<12}"
            )
            print("-" * 70)

            # Find baseline (CPU) for speedup calculation
            baseline_time = None
            for r in results:
                if r.backend == "cpu":
                    baseline_time = r.mean_time_ms
                    break

            for r in sorted(results, key=lambda x: x.mean_time_ms):
                speedup = ""
                if baseline_time and r.backend != "cpu":
                    speedup = f"{baseline_time / r.mean_time_ms:.2f}x"
                elif r.backend == "cpu":
                    speedup = "(baseline)"

                print(
                    f"{r.backend:<12} {r.mean_time_ms:<15.3f} {r.tokens_per_sec:<15.2f} {speedup:<12}"
                )

        # Save to file
        if self.args.output:
            output_data = {
                "device_info": self.device_info,
                "results": [asdict(r) for r in self.results],
                "args": vars(self.args),
            }

            with open(self.args.output, "w") as f:
                json.dump(output_data, f, indent=2)
            print(f"\nResults saved to: {self.args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile BitNet inference across different backends"
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["cpu", "metal", "cuda", "all"],
        default="all",
        help="Backend to profile (default: all)",
    )

    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,8",
        help="Comma-separated batch sizes to test (default: 1,8)",
    )

    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="128,512",
        help="Comma-separated sequence lengths to test (default: 128,512)",
    )

    parser.add_argument(
        "--dim", type=int, default=2560, help="Model hidden dimension (default: 2560)"
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=128256,
        help="Vocabulary size (default: 128256)",
    )

    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )

    parser.add_argument(
        "--test-iterations",
        type=int,
        default=100,
        help="Number of test iterations (default: 100)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of CPU threads (default: auto)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="profile_results.json",
        help="Output JSON file for results (default: profile_results.json)",
    )

    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run detailed profiling (implies --test-iterations=1 for GPU profiling)",
    )

    args = parser.parse_args()

    if args.threads is None:
        args.threads = torch.get_num_threads()

    profiler = InferenceProfiler(args)
    profiler.run()


if __name__ == "__main__":
    main()
