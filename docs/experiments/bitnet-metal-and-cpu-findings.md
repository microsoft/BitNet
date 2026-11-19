# BitNet Metal and CPU performance findings

The output-only Metal experiment improved short-prompt decoding on this M2 Pro. The longer workload showed no end-to-end improvement. In the CPU investigation, shared RoPE coefficients passed bitwise checks and showed a small isolated gain. The current I8MM prototype was slower than SDOT.

## Metal on the M2 Pro

Generation medians in tokens per second. Every run used the local I2_S GGUF, eight generation threads, four prompt threads, a 2,048-token context and a fixed 64-token decode. The tied test used five runs per case. The output-only test used four runs per case with the same temperature-only greedy sampler across variants.

| Experiment | Short prompt | Longer prompt |
|---|---:|---:|
| Existing CPU-only binary | 50.9 | 53.3 |
| Tied experiment, same binary in CPU mode | 58.0 | 54.6 |
| Tied experiment, F16 lookup and projection on Metal | 50.0 | 52.1 |
| Output-only experiment, same binary in CPU mode | 56.5 | 46.4 |
| Output-only Metal, CPU sampling | 67.3 | 47.7 |
| Output-only Metal, GPU sampling | 73.4 | 41.9 |

Compare each Metal configuration with the corresponding same-binary CPU row. In the output-only test, GPU sampling improved short-prompt generation by about 30%. Its median wall time fell from 1.66 to 1.43 seconds. On the longer input, median wall time went from 4.20 to 4.40 seconds. The output-only experiment produced identical greedy text for each tested prompt in all modes. Moving the entire tied tensor to Metal changed the longer prompt's response.

The scheduler trace confirmed the F16 vocabulary projection on Metal with I2_S and attention on the CPU. In the output-only build, embedding lookup stayed on the CPU, and in GPU sampling mode the argmax was on Metal. The trace did not show the 626 MiB tied weight being copied as a graph input each token. CPU and GPU still need synchronization at their execution boundary. [Apple Metal storage modes](https://developer.apple.com/documentation/metal/choosing-a-resource-storage-mode-for-apple-gpus)

Full-layer Metal remains a separate implementation task. This checkout advertises I2_S matrix multiplication as supported by Metal even though it has no matching shader.

The existing checkout and normal CPU build were used as-is. The small [output-only patch](bitnet-metal-output-only.patch) applies from the nested llama.cpp checkout. It was compiled only into the experimental build. That build can run the output layer on Metal with `--device MTL0 -ngl 1 --no-kv-offload --no-op-offload`. Add `--backend-sampling --samplers temperature --temp 0` to reproduce the greedy GPU sampling variant.

## CPU findings and next experiments

| Opportunity | Evidence and next measurement |
|---|---|
| Group Q/K/V and gate/up | The projections in each group consume the same normalized activation. The current decoder quantizes it separately for each projection. Grouping would reduce seven activation preparations to four per layer. Preserve each matrix's own scale and integer arithmetic, then measure projection and full decode time. |
| Reduce vocabulary projection bandwidth | The F16 head touches about 657 MB of weights per token. A separately quantized Q8_0 head would reduce this to about 349 MB while keeping F16 input embeddings. It changes model scores. Compare logits, held-out perplexity, generation and resident memory before adopting it. |
| Share RoPE coefficients | Positions and coefficients are the same across Q, K and all 30 layers. A scratch graph using the model's dimensions produced bit-identical rotations. Across four repeats, medians went from 127 to 92.5 µs for one token and 380 to 253 µs for 32 tokens. This isolates RoPE and includes cache preparation. |
| Improve larger CPU batches | The I2_S caller hands the kernel four tokens at a time. A wider schedule that keeps packed weight tiles near the worker deserves a separate prefill test. The exact I8MM prototype was 1.77–1.86× slower across four shapes including activation packing, so it should stay out of the production path. [Arm I8MM implementation notes](https://developer.arm.com/community/arm-community-blogs/b/ai-blog/posts/optimize-llama-cpp-with-arm-i8mm-instruction) |
| Measure planning and sampling overhead | The CPU backend replans each compute even when the scheduler reuses its graph. Measure the planning cost before adding a cache keyed by graph ID and thread count. The host sampler also builds a vocabulary-sized candidate table. An exact fast path should preserve tie-breaking and handle penalties or grammar constraints explicitly. |

The CPU tests were isolated prototypes. The branch and CPU executable still use the previous kernels and settings. [Raw timings and placement results](bitnet-metal-and-cpu-benchmarks.json)
