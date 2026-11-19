# BitNet optimization history

This records the repairs, experiments, and retained changes made during the investigation of the local BitNet checkout on an Apple M2 Pro with 16 GB RAM. The local tracking branch is `codex/arm-i2s-performance` in BitNet and its nested llama.cpp repository. Changes are uncommitted.

## Reading the measurements

The comparisons below belong to individual experiments. Context length, output limits, prompts, sampler options and concurrent activity differed across stages. Ratios from separate stages should not be multiplied into a cumulative speedup. Most timings are medians of alternating executable runs. Early tests suffered from concurrent user tests and memory pressure. Later tests included peak resident memory and quieter repeats. Greedy text agreement describes only the prompts actually tested.

## Correctness and build repairs

| Change | Evidence and effect | Decision |
|---|---|---|
| Standard-library includes in llama.cpp | The checkout already had additions for `cstdlib`, `cerrno`, and `memory` in context, graph, mmap, vocabulary, GGUF, Metal and the deprecation warning. They were preserved. No isolated speed measurement. | Existing build compatibility edits retained |
| Declare `src1_cont` in I2_S multiplication | Existing local compilation fix preserved. No isolated speed measurement. | Retained |
| Generated lookup headers and kernel configuration | These local edits existed before the investigation. The active build has TL1/TL2 disabled. No speed claim for these files. | Preserved |
| Q1_0 ARM kernel audit | One numerical case returned 93 in both ARM and generic paths. Model tensors are I2_S. | No kernel change |
| I2_S scalar packed-weight addressing | Corrected consecutive-value decoding to the 128-weight, 32-byte interleaved format. A scalar reference dot expected 11 while the old kernel returned 8 before caller correction. | Retained |
| I2_S centering contract | The kernel now returns uncentered code products. The caller performs the single activation-sum correction. Zero weights previously produced −128 for 128 unit activations after the erroneous second correction. | Retained |
| I2_S fallback batched indexing | Replaced incorrect use of the vector-dot row count for activation columns with explicit row/column addressing. | Retained |
| I2_S quantizer packing | Corrected the C quantizer to emit the same interleaved format that the reader consumes. Existing GGUF was preserved. | Retained |
| Zero activation scale | Uses a finite nonzero scale for all-zero activations, preventing division by zero during output rescaling. | Retained |
| Quantization metadata | Converter now writes I2_S file type 41. Loader recognizes legacy file type 40 when actual tensors are I2_S. | Retained |
| BitNet-b1.58 FFN activation | Selected squared ReLU for this architecture instead of inherited SiLU. Short factual answers alone had failed to expose the mismatch. | Retained |
| Legacy chat template | Replaced the legacy `Human`/`BITNETAssistant` template at load time with the published User/Assistant and end-of-turn format for this model. The neural-network explanation then became coherent and stopped normally. | Retained |
| BLAS I2_S capability guard | Generic BLAS conversion cannot handle I2_S packed row layout and scale arguments. Rejecting that route fixed the reproduced longer-prompt crash. | Retained |

The initial greedy France prompt produced twelve `@` characters. The combined correctness repairs restored coherent instruction responses. Individual repairs were not assigned isolated token-rate improvements.

## CPU performance experiments

| Change or experiment | Recorded result | Decision |
|---|---|---|
| ARM NEON SDOT I2_S dot product | Standalone prototype was about 9.3× faster than the scalar inner loop in its original microbenchmark. The integrated build subsequently produced roughly 50–67 generated tokens/s in repeated short tests, compared with early scalar observations around 3–4. The scalar baseline was affected by uncontrolled activity. | Retained |
| Generation and prompt thread defaults | Clean comparison with four prompt threads gave median generation of 63.6 tokens/s at six generation threads and 67.2 at eight. | Eight generation and four prompt threads on Apple Silicon, bounded by available CPUs |
| First four-row I2_S GEMV tile | Dependent accumulators slowed generation. | Replaced |
| Revised four-row I2_S GEMV | Independent accumulator chains restored approximate parity. | Retained with GEMM changes |
| 4×4 ARM I2_S GEMM | Short prompt processing 81.4→187.6 tokens/s. Longer prompt processing 88.3→201.0. Generation changed 63.0→63.7 and 56.0→59.8. Responses matched. | Retained, about 2.3× prompt processing |
| ARM activation quantization | Quieter comparison gave prompt rates 171.7→191.8 and 202.3→210.7 tokens/s. Generation changed 65.3→65.9 and 58.5→60.1. Output bytes, scales, sums and rounding ties matched the scalar reference. | Retained, approximately 4–12% prompt gain |
| Direct multiplication output writes | Short generation 69.1→72.0, longer generation 67.9→65.5. | Reverted after mixed results |
| Native CPU thread pool | Quieter repeat gave short generation 65.2→63.7 and longer generation 55.2→63.0. Earlier runs had severe interference. | OpenMP restored |
| Fuse ReLU and square | Isolated graph with 30 pairs fell from 37→20 µs. Whole-model short generation 56.6→54.0 and longer generation 50.4→55.2. Exact graph tests include exposed intermediate outputs. | Retained for reduced intermediate work, no established overall speedup |
| Four-row F16 output tile | Isolated projection 5.866→6.660 ms with identical logits. Whole-model rates were roughly unchanged. | Reverted |
| Two-row F16 output tile | Isolated projection 5.437→5.475 ms with identical logits. | Prototype discarded |
| Larger F16 work chunks | Isolated graph 5.703→5.509 ms. Whole-model short generation 55.1→47.4 and longer 54.9→55.1. | Reverted |
| Private I2_S activations per worker | Removed an internal barrier but repeated quantization. Short generation 52.3→54.3 and longer 55.9→55.6. | Reverted after mixed results |
| I8MM GEMM prototype | Exact against SDOT and scalar references. SDOT was 1.77–1.86× faster on four tested shapes, with activation packing included. | Prototype discarded |
| Zero-weight skipping investigation | 42.21% of individual weights are zero, but only 2.44% of full 128-weight blocks are entirely zero. | No sparse-format or skipping change adopted |
| Shared RoPE scratch graph | Four timing repeats gave median 127→92.5 µs at one token and 380→253 µs at 32 tokens. Rotations matched bit for bit. | Advanced to normal model-path experiment |

## Metal experiments

The Metal executable was built separately. Scheduler traces verified which operations ran on each device. CPU mode in the same executable served as a control. The model file was preserved.

| Configuration | Short generation | Longer generation | Decision |
|---|---:|---:|---|
| Tied-weight experiment, CPU mode | 58.0 tokens/s | 54.6 tokens/s | Control |
| Tied embedding lookup and vocabulary projection on Metal | 50.0 | 52.1 | Experimental; no overall gain, longer response changed |
| Output-only experiment, CPU mode | 56.45 | 46.4 | Control |
| Vocabulary output on Metal with CPU sampling | 67.25 | 47.65 | Experimental |
| Vocabulary output and greedy sampling on Metal | 73.35 | 41.9 | Experimental, short-workload gain only |

The output-only build duplicated the tied tensor descriptor using the loader's existing `TENSOR_DUPLICATED` mechanism and kept embedding lookup on the CPU. Tested output-only modes produced identical greedy text. The checkout's Metal backend advertises I2_S matrix multiplication without an implemented I2_S shader, so general layer offload needs a capability fix and kernel work. GPU sampling also has tie-breaking considerations for a future general correctness claim.

## Pending or unadopted ideas

Q8_0 vocabulary weights would reduce output-weight traffic while changing model scores and memory layout. That experiment has not been performed. Grouping FFN gate/up, plan-template caching, wider prefill tiles, shared K/V loads for grouped-query attention, and sampling-table shortcuts remain candidates. No speed claim is attached to them.

## Validation and memory

Numerical tests cover interleaved quantization, signed int8 extremes, zero activations, partial GEMM tiles, padded outputs, and exact activation rounding. Model comparisons used greedy responses and fixed prompts. CPU resident-memory measurements were approximately 1.42–1.46 GB for the tested workloads. Whole-system memory pressure and compressed memory affected some runs. These tests do not establish sustained thermal behavior, model quality across a dataset, or performance across every context length.

## RoPE and Q/K/V model-path checkpoint

Implemented two optional CPU graph paths. Shared RoPE creates coefficients once using the existing RoPE implementation, then applies the same rotation arithmetic to each layer. Grouped Q/K/V prepares one activation buffer and feeds the three matrices while preserving each scale trailer, reducing three activation preparations to one. Neither experiment quantizes model weights further.

The first three-run comparison was promising for Q/K/V. A second rotating comparison with five runs per configuration gave these medians in tokens per second.

| Configuration | Short prompt | Short generation | Longer prompt | Longer generation |
|---|---:|---:|---:|---:|
| Current baseline | 180.9 | 56.4 | 207.8 | 55.7 |
| Grouped Q/K/V | 186.6 | 53.9 | 208.2 | 53.7 |
| Grouped Q/K/V plus shared RoPE | 188.1 | 49.1 | 209.2 | 57.7 |

All tested greedy responses matched. Independent graph tests compared the grouped outputs to three ordinary matrix multiplications and the cached rotations to ordinary RoPE bit for bit. Cases covered one, five and sixteen tokens, one, three and eight threads, distinct Q/K/V scales, frequency scaling and YaRN extrapolation.

The mixed end-to-end results do not support enabling either experiment by default. The normal CPU path remains the default. Set `BITNET_ENABLE_GROUP_QKV=1` to test grouped projections and `BITNET_ENABLE_ROPE_CACHE=1` to test shared coefficients. These are experimental CPU paths. The implementations and regression test live in `src/models/bitnet-cpu.h` and `utils/test-bitnet-graph.cpp`. The experiment runner is `utils/benchmark-bitnet-graph.py`.

## Full Metal status at the previous checkpoint

The completed Metal experiments offloaded the F16 vocabulary path while keeping I2_S layers on the CPU. They measure a hybrid architecture with CPU/GPU handoffs. A full GPU implementation could have different performance and avoid these handoffs. It still requires I2_S shaders, proper backend capability checks and full numerical and performance validation. None of the hybrid timings settled that question. The following checkpoint adds a running full-GPU prototype.

## First full-Metal checkpoint

A separate executable now implements I2_S multiplication on the GPU. The shader
reads the existing packed GGUF weights, quantizes F32 activations, and performs
integer ternary dot products before rescaling the output. A Metal capability
guard restricts the path to supported matrix layouts and dimensions. Scheduler
traces show the model compute on MTL0, including Q/K/V, attention, FFN and the
vocabulary projection. Host tokenization and process control remain on the CPU.

| Same-session median generation | Short prompt | Longer prompt |
|---|---:|---:|
| CPU-only | 53.0 tokens/s | 54.1 tokens/s |
| Hybrid output-only Metal | 67.6 | 58.5 |
| Full-GPU reference shader | 30.9 | 30.9 |

Three rotated runs per case used a fixed 64-token decode, eight CPU generation
threads, four prompt threads, a 2,048-token context and temperature-only greedy
sampling. The first full-GPU shader is a correctness baseline, with activation
quantization repeated for each output row. Its prompt processing was about
42–43 tokens/s in these tests.

A second shader shared quantized activations across four rows using threadgroup
memory. It passed the same scalar checks but slowed generation to medians of
24.4 and 24.0 tokens/s, with prompt processing around 33–35 tokens/s. It was
rejected as the default full-GPU experiment. The first shader is preserved as
the faster full-GPU baseline.

Both shaders passed six scalar-reference checks covering widths 128, 2560 and
6912, multiple output columns, partial row groups, and zero activations. The
largest observed output difference was 2.4e-7. This validates the tested
arithmetic but does not replace model-quality evaluation. Full-GPU generation
matched CPU on the short prompt but differed on the longer one. The full model
uses GPU floating-point operations throughout, so exact CPU text parity is not
established across workloads.

The GPU experiment remains separate from the normal CPU build. The preserved
executables are in `bitnet-builds` beside this report, with their command-line
settings in that directory's README. `bitnet-full-metal.patch` contains the
shader and dispatch changes. `bitnet-full-metal-benchmarks.json` contains raw
runs and scalar-test results. All current changes remain local and uncommitted.
