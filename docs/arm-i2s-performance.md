# ARM I2_S performance

The I2_S dot product now uses ARM NEON SDOT instructions when both
`__ARM_NEON` and `__ARM_FEATURE_DOTPROD` are available. Other ARM targets retain
the scalar fallback. The kernel reads the existing 128-weight/32-byte packing
and returns uncentered code products, matching the caller's activation-sum
correction. The model file does not need conversion.

On Apple silicon, the CLI and Python launchers default to up to eight generation
threads and four prompt-processing threads. Explicit `-t` and `-tb` override
these settings. This tuning was measured on a 12-core M2 Pro with 16 GB RAM;
other machines and workloads can have different optimal settings.

## Measurements

Three sequential runs per setting, greedy sampling, context 2048, up to 128
output tokens, using the neural-network explanation prompt:

| Generation threads | Prompt threads | Median generated tokens/s |
|---:|---:|---:|
| 6 | 4 | 63.6 |
| 8 | 4 | 67.2 |

These are short local measurements, not a sustained-load or long-context
benchmark. Earlier scalar measurements ran at approximately 3-4 tokens/s, but
were not controlled for concurrent user activity and are not a precise speedup
baseline.

## Validation

`utils/test-i2s-kernels.c` checks quantization roundtrips, signed-byte activation
extremes, multiple weight rows and activation columns, and zero activations at
inner dimensions 128, 2560 and 6912. Build and run it after building the CLI:

```sh
clang utils/test-i2s-kernels.c \
  build/3rdparty/llama.cpp/ggml/src/libggml-cpu.a \
  build/3rdparty/llama.cpp/ggml/src/libggml-base.a \
  -L/opt/homebrew/opt/llvm/lib -lomp -lc++ -framework Accelerate \
  -o /tmp/bitnet-i2s-regression
/tmp/bitnet-i2s-regression
```

The library paths above assume the current static-library build and Homebrew
LLVM installation. Rebuild before testing so the test links the latest kernel.

## Tiled ARM kernels checkpoint

Tracking branch: `codex/arm-i2s-performance` in both the outer repository and
its llama.cpp submodule. Changes are local and uncommitted. Existing edits were
preserved; baseline patches and the pre-tiling executable were saved separately.
No remote was added and nothing was published.

The first four-row GEMV implementation regressed generation speed. Splitting
its dependent accumulators restored parity. The retained implementation uses
four-row GEMV and 4x4 GEMM, reusing activations and decoded weights.

Three alternating baseline/candidate runs per case produced these medians:

| Case | Metric | Baseline | Tiled |
|---|---|---:|---:|
| Short prompt | Prompt tokens/s | 81.4 | 187.6 |
| Short prompt | Generation tokens/s | 63.0 | 63.7 |
| Longer prompt | Prompt tokens/s | 88.3 | 201.0 |
| Longer prompt | Generation tokens/s | 56.0 | 59.8 |

Greedy response text was identical between versions in all comparisons. The
small generation differences should not be treated as a robust speedup claim.
The prompt improvement is about 2.3x. Raw measurements and settings are in
`docs/benchmarks/arm-i2s-tiled.json`; `utils/benchmark-arm-i2s.py` reproduces the
alternating comparison given a saved baseline executable.

The longer test exposed a pre-existing crash: BLAS accepted I2_S tensors even
though its generic conversion cannot supply their scale or packed row layout.
Both compared executables include the same fix, keeping I2_S on the CPU kernel
path. The original pre-fix executable crashed on this input.

Regression checks cover GEMV, GEMM, partial tiles, padded output strides,
quantization roundtrips, signed-byte extremes and zero activations.

### Zero weights and Metal

A full scan counted 42.21% zero weights, but only 397,706 of 16,281,600
128-weight blocks (2.44%) were entirely zero. Packed bytes containing four
zeros accounted for 5.81%. Skipping whole blocks has limited headroom; a sparse
format would need a measured benefit after index and decoding costs.

Metal is not enabled by this checkpoint. The checkout lacks I2_S Metal kernels.
Testing F16 vocabulary-head offload is a separate next experiment, including
CPU/GPU synchronization overhead and output consistency checks.

## CPU activation and coordination checkpoint

Retained: ARM NEON activation quantization. Vector maximum reduction and
16-value conversion/packing replace scalar work, with FCVTAS rounding matching
`roundf` ties away from zero. Scalar tails handle arbitrary lengths; non-ARM
targets retain the scalar implementation. Tests compare every output byte,
activation scale and sum against the scalar reference for zero, fractional,
positive/negative half-integer and varying-scale inputs.

Three alternating comparisons after the user paused competing tests:

| Case | Metric | Previous tiled build | Activation SIMD |
|---|---|---:|---:|
| Short | Prompt tokens/s | 171.7 | 191.8 |
| Short | Generation tokens/s | 65.3 | 65.9 |
| Longer | Prompt tokens/s | 202.3 | 210.7 |
| Longer | Generation tokens/s | 58.5 | 60.1 |

Responses were identical. Prompt gains are about 12% and 4%, respectively;
generation gains are small and not a strong decoding-speed claim. An earlier
pass showed roughly 11% generation gains, but the quieter repeat is the more
conservative result. The isolated activation microbenchmark showed 1.02x at
2560 elements and 1.12x at 6912 elements; do not extrapolate either result to
whole-model speed.

Peak resident memory in the final comparison was approximately 1.42-1.46 GB
in both versions. This is the model process's peak resident footprint, not
whole-system memory pressure or private allocated memory. The machine was
holding about 5.9 GB of compressed memory when the user reported pressure;
cumulative swap counters cannot establish swapping during a specific run.

Rejected experiments: direct output-buffer writes had mixed results and were
reverted; the native CPU thread pool also gave mixed results and was reverted.
Final build configuration keeps `GGML_OPENMP=ON` and `GGML_METAL=OFF`.

Raw results are in `docs/benchmarks/arm-i2s-activation-isolated.json`, with the
other experiment files explicitly marked retained, reverted or inconclusive.
The benchmark runner now records per-child peak resident memory using a fresh
Python wrapper around each model process.

## F16 output layer and CPU synchronization

The retained CPU fusion evaluates adjacent F32 ReLU and square operations in one
pass when the intermediate has a single consumer. It removes one intermediate
write and one synchronization barrier per eligible pair. A synthetic graph
with 30 pairs of 6,912 values took a median 20 microseconds with fusion versus
37 microseconds without it. A separate graph test checked bit-for-bit outputs
with one, three and eight threads, including signed zero, NaN and infinities.
It also confirmed the intermediate is still computed when exposed as an output.

Each end-to-end comparison used five alternating runs per case, eight
generation threads, four prompt threads and greedy sampling. Responses were
identical. The numbers below are median generation tokens per second.

| Experiment | Short before | Short after | Longer before | Longer after |
|---|---:|---:|---:|---:|
| Four-row F16 output tile | 60.9 | 61.3 | 51.4 | 51.3 |
| Worker-local I2_S activations | 52.3 | 54.3 | 55.9 | 55.6 |
| ReLU and square fusion | 56.6 | 54.0 | 50.4 | 55.2 |
| Larger F16 work chunks | 55.1 | 47.4 | 54.9 | 55.1 |

In isolated F16 projection tests at the model's 128,256 by 2,560 dimensions,
four-row tiling went from 5.87 to 6.66 ms. Two-row tiling was effectively tied
at 5.44 versus 5.48 ms. Larger chunks improved the isolated GGML projection
from 5.70 to 5.51 ms, but the CLI comparison showed no consistent gain. The
existing F16 projection path was restored, as was the shared I2_S activation
path. The fusion is retained for the measured reduction in its own work.
End-to-end variability leaves its effect on generation throughput unclear.

Raw results are in `docs/benchmarks/arm-i2s-f16-and-fusion.json`.
