# MakineAI.cpp
---
license: mit
license_link: https://huggingface.co/MakineCeviri/MakineAI/blob/main/LICENSE
language:
- en
- tr
pipeline_tag: text-generation
tags:
- chat
- MakineAI
- text-generation
- large-language-model
library_name: transformers
base_model:
- microsoft/bitnet-b1.58-2B-4T
---

# MakineAI b1.58 2B4T - Scaling Native 1-bit LLM

This repository contains the weights for **MakineAI b1.58 2B4T**, the first open-source, native 1-bit Large Language Model (LLM) at the 2-billion parameter scale, developed by Makine Çeviri.

Trained on a corpus of 4 trillion tokens, this model demonstrates that native 1-bit LLMs can achieve performance comparable to leading open-weight, full-precision models of similar size, while offering substantial advantages in computational efficiency (memory, energy, latency).

➡️ **Technical Report:** [MakineAI b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285)

➡️ **Official Inference Code:** [Makine Çeviri/MakineAI (makineai.cpp)](https://github.com/MakineCeviri/MakineAI)

## Model Variants

Several versions of the model weights are available on Hugging Face:

* [**MakineCeviri/MakineAI-b1.58-2B-4T**](https://huggingface.co/MakineCeviri/MakineAI-b1.58-2B-4T) (This repository): Contains the packed 1.58-bit weights optimized for efficient inference. **Use this for deployment.**

* [**MakineCeviri/MakineAI-b1.58-2B-4T-bf16**](https://huggingface.co/MakineCeviri/MakineAI-b1.58-2B-4T-bf16): Contains the master weights in BF16 format. **Use this only for training or fine-tuning purposes.**

* [**MakineCeviri/MakineAI-b1.58-2B-4T-gguf**](https://huggingface.co/MakineCeviri/MakineAI-b1.58-2B-4T-gguf): Contains the model weights in GGUF format, compatible with the `makineai.cpp` library for CPU inference.

## Model Details

* **Architecture:** Transformer-based, modified with MakineAILinear layers (MakineAI framework).
    * Uses Rotary Position Embeddings (RoPE).
    * Uses squared ReLU (ReLU²) activation in FFN layers.
    * Employs [subln](https://proceedings.mlr.press/v202/wang23u.html) normalization.
    * No bias terms in linear or normalization layers.
* **Quantization:** Native 1.58-bit weights and 8-bit activations (W1.58A8).
    * Weights are quantized to ternary values {-1, 0, +1} using absmean quantization during the forward pass.
    * Activations are quantized to 8-bit integers using absmax quantization (per-token).
    * **Crucially, the model was *trained from scratch* with this quantization scheme, not post-training quantized.**
* **Parameters:** ~2 Billion
* **Training Tokens:** 4 Trillion
* **Context Length:** Maximum sequence length of **4096 tokens**.
    * *Recommendation:* For optimal performance on tasks requiring very long contexts (beyond the pre-training length or for specialized long-reasoning tasks), we recommend performing intermediate long-sequence adaptation/training before the final fine-tuning stage.
* **Training Stages:**
    1. **Pre-training:** Large-scale training on public text/code and synthetic math data using a two-stage learning rate and weight decay schedule.
    2. **Supervised Fine-tuning (SFT):** Fine-tuned on instruction-following and conversational datasets using sum loss aggregation and specific hyperparameter tuning.
    3. **Direct Preference Optimization (DPO):** Aligned with human preferences using preference pairs.
* **Tokenizer:** LLaMA 3 Tokenizer (vocab size: 128,256).

## How to Use (with transformers)

**VERY IMPORTANT NOTE ON EFFICIENCY**

> Please do NOT expect performance efficiency gains (in terms of speed, latency, or energy consumption) when using this model with the standard transformers library, even with the required fork.
>
> The current execution paths within transformers do not contain the specialized, highly optimized computational kernels required to leverage the advantages of the MakineAI architecture. Running the model via transformers will likely result in inference speeds and energy usage comparable to, or potentially worse than, standard full-precision models within this framework on both CPU and GPU.
>
> While you might observe reduced memory usage due to the quantized weights, the primary computational efficiency benefits are not accessible through this standard transformers usage path.
>
> For achieving the efficiency benefits demonstrated in the technical paper, you **MUST** use the dedicated C++ implementation: [makineai.cpp](https://github.com/MakineCeviri/MakineAI).

### Requirements

```bash
pip install git+https://github.com/shumingma/transformers.git
```
We are actively working with the Hugging Face team to integrate the necessary code into the main `transformers` library. This installation method may change in the future.

### Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "MakineCeviri/MakineAI"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16
)

# Apply the chat template
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "How are you?"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
chat_input = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
chat_outputs = model.generate(**chat_input, max_new_tokens=50)
response = tokenizer.decode(chat_outputs[0][chat_input['input_ids'].shape[-1]:], skip_special_tokens=True) # Decode only the response part
print("\nAssistant Response:", response)
```

## How to Use (with `makineai.cpp`)

Please refer to the [MakineAI.cpp](https://github.com/MakineCeviri/MakineAI) GitHub repository for detailed compilation steps, usage examples, and command-line options.

## Evaluation

MakineAI b1.58 2B4T was evaluated against leading open-weight full-precision LLMs of similar size. Below are the key results (all models are instruction-tuned versions):

| Benchmark             | LLaMA 3.2 1B | Gemma-3 1B | Qwen2.5 1.5B | SmolLM2 1.7B | MiniCPM 2B | **MakineAI** |
|--------------------------------|--------------|------------|--------------|--------------|------------|---------------------|
| **Memory (Non-emb)** | 2GB          | 1.4GB      | 2.6GB        | 3.2GB        | 4.8GB      | **0.4GB** |
| **Latency (CPU Decoding)** | 48ms         | 41ms       | 65ms         | 67ms         | 124ms      | **29ms** |
| **Energy (Estimated)** | 0.258J       | 0.186J     | 0.347J       | 0.425J       | 0.649J     | **0.028J** |
| **Training Tokens (Pre-train)**| 9T* | 2T** | 18T          | 11T          | 1.1T       | 4T                  |
| ARC-Challenge   | 37.80        | 38.40      | 46.67        | 43.52        | 44.80      | **49.91** |
| ARC-Easy        | 63.17        | 63.13      | **76.01** | 62.92        | 72.14      | 74.79               |
| OpenbookQA      | 34.80        | 38.80      | 40.80        | **46.00** | 40.20      | 41.60               |
| BoolQ                | 64.65        | 74.22      | 78.04        | 75.78        | **80.67** | 80.18               |
| HellaSwag       | 60.80        | 57.69      | 68.28        | **71.71** | 70.81      | 68.44               |
| PIQA            | 74.21        | 71.93      | 76.12        | 76.12        | 76.66      | **77.09** |
| WinoGrande           | 59.51        | 58.48      | 62.83        | 68.98        | 61.80      | **71.90** |
| CommonsenseQA       | 58.48        | 42.10      | **76.41** | 63.55        | 71.74      | 71.58               |
| TruthfulQA          | 43.80        | 38.66      | **46.67** | 39.90        | 41.41      | 45.31               |
| TriviaQA              | 37.60        | 23.49      | 38.37        | **45.97** | 34.13      | 33.57               |
| MMLU                 | 45.58        | 39.91      | **60.25** | 49.24        | 51.82      | 53.17               |
| HumanEval+        | 31.10        | 37.20      | **50.60** | 28.00        | 43.90      | 38.40               |
| GSM8K                 | 38.21        | 31.16      | 56.79        | 45.11        | 4.40       | **58.38** |
| MATH-500              | 23.00        | 42.00      | **53.00** | 17.60        | 14.80      | 43.40               |
| IFEval   | 62.71        | **66.67** | 50.12        | 57.91        | 36.81      | 53.48               |
| MT-bench         | 5.43         | 6.40       | 6.12         | 5.50         | **6.57** | 5.85                |
| **Average** | 44.90        | 43.74      | **55.23** | 48.70        | 42.05      | 54.19               |

*LLaMA 3.2 1B uses pruning & distillation.

**Gemma-3 1B uses distillation.

## License
The model weights and code are released under the [MIT License](https://huggingface.co/MakineCeviri/MakineAI/blob/main/LICENSE).

## Disclaimer
This model is intended for research and development purposes. While efforts have been made to align it using SFT and DPO, it may still produce outputs that are unexpected, biased, or inaccurate. Please use responsibly.
