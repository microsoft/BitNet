# Ternary model comparison on Apple M2 Pro

All measurements use our BitNet CPU build, I2_S weights, 8 generation and 4 prompt threads, context 4096, greedy sampling, and the same three-sentence neural-network prompt (128 output-token cap). No Metal. These are short-run throughput measurements, not proof of numerical correctness or general model quality.

| Model | Initial median t/s (3 runs) | Rotated-order retest median | Retest range |
|---|---:|---:|---:|
| Falcon3 1B Instruct 1.58bit | 97.2 | 88.1 | 84.3–105.4 |
| BitNet-b1.58-2B-4T (2.4B) | 44.5 | 69.2 | 45.3–72.6 |
| Falcon3 7B Instruct 1.58bit | 25.4 | 33.1 | 29.6–35.1 |
| Falcon-E 3B Instruct | Cannot load | Not measured | Unknown pre-tokenizer falcon_e |

Variation remains significant. No cause established; no evidence yet that a two-second pause improves throughput. macOS memory_pressure reported 44–55% free in the rotated test; process activity could not be inspected in that run.

Official GGUF repositories: tiiuae/Falcon3-1B-Instruct-1.58bit-GGUF, tiiuae/Falcon3-7B-Instruct-1.58bit-GGUF, and tiiuae/Falcon-E-3B-Instruct-GGUF. Each supplies ggml-model-i2_s.gguf. Model weights are excluded from Git.

Falcon3 1B produced math.pi-based code, failing the requested independent 100-decimal-place calculation. Falcon3 7B generated an unfinished repetitive numeric literal. These are output-quality failures; the tests alone do not establish an inference implementation defect.

LM Studio imported the 7B model but rejected I2_S tensor type 36, interpreting it as a different removed format. Its import link was removed; original weights retained.

## Persistent planner integration

Voice Developer already separates planner and coder. A ternary planner should be evaluated with its actual React/Vite planning prompt and source context. This has not been done yet.

run_inference.py launches llama-cli; repeated invocations reload weights. run_inference_server.py already launches a persistent llama-server with repeated-request support. The current build does not contain llama-server; build and verify that target before integration. Direct Python bindings need to link this modified I2_S engine, rather than stock llama.cpp. No Voice Developer source was changed.

Raw logs and JSON are alongside this document. run-battery.py uses build/bin/llama-cli and models/<model-name>/ggml-model-i2_s.gguf under this checkout (Falcon3-1B-Instruct-1.58bit, Falcon3-7B-Instruct-1.58bit, and BitNet-b1.58-2B-4T). Download the weights into those directories before running it. Falcon-E is excluded until tokenizer support is resolved. Generated code is never automatically executed.
