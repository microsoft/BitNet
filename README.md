# bitnet.cpp

## Introduction
bitnet.cpp is the officially supported inference framework specifically designed for Bitnet ternary models, optimized for efficient CPU-based inference. bitnet.cpp offers a suite of optimized kernels, I2_S, TL1 ( Ternary Lookup 1 ) and TL2 ( Ternary Lookup 2 ), that support lossless inference of BitNet b1.58 models across both x86 and
ARM architectures. Below is a demo of bitnet.cpp runing 3.8B model on Apple M2. 

https://github.com/user-attachments/assets/a4b389c1-1b26-441c-8049-f0357217c5cf


## Timeline
- 10/17/2024 bitnet.cpp supports lossless inference on x86 and ARM CPUs.
- 02/27/2024 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- 10/17/2023 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Acknowledgements
This project is based on the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. We would like to thank all the authors for their contributions to the open-source community.
We also thank [T-MAC]([https://github.com/ggerganov/llama.cpp](https://github.com/microsoft/T-MAC/) team for indroducing LUT method for low-bit LLM inference.

## Supported Models
bitnet.cpp supports a list of models available on [Hugging Face](https://huggingface.co/)

|       Model                                                                                              | Parameters |   CPU    |              | Kernel       |              |              |
| :----------------:                                                                                       | :-------:  | :------: | :----------: |:----------:  |:----------:  |:----------:  |
|                                                                                                          |            |   x86    |     ARM      |     I2_S     |     TL1      |      TL2     |
| [bitnet_b1_58-large](https://huggingface.co/1bitLLM/bitnet_b1_58-large)                                  |    729M    | &#10004; |   &#10004;   |   &#10004;   |   &#10004;   |   &#10004;   |
| [bitnet_b1_58-3B](https://huggingface.co/1bitLLM/bitnet_b1_58-3B)                                        |    3.32B   | &#10004; |   &#10004;   |   &#10008;   |   &#10004;   |   &#10004;   |
| [Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)                |    8B      | &#10004; |   &#10004;   |   &#10004;   |   &#10004;   |   &#10004;   |

## Installation
### Requirements
- conda
- cmake>=3.22
- clang(if using Windows, Visual Studio is needed for clang support)

### Build from source
> if you are using Windows, please make sure you have installed Visual Studio with clang support, and run the following commands within the Developer PowerShell
1. Clone the repo
```bash
git clone --recursive https://github.com/microsoft/BitNet.git
cd BitNet
```
2. Install the dependencies
```bash
# (Recommended) Create a new conda environment
conda create -n bitnet-cpp python=3.9
conda activate bitnet-cpp

pip install -r requirements.txt
```
3. Build the project
```bash
# Download the model from Hugging Face, convert it to quantized gguf format, and build the project
python setup_env.py --hf-repo 1bitLLM/bitnet_b1_58-large -q i2_s

# Or you can manually download the model and run with local path
huggingface-cli download 1bitLLM/bitnet_b1_58-large --local-dir models/bitnet_b1_58-large 
python setup_env.py -md models/bitnet_b1_58-large -q i2_s
```
<pre>
usage: setup_env.py [-h] [--hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B}] [--model-dir MODEL_DIR] [--log-dir LOG_DIR] [--quant-type {i2_s,tl1}] [--quant-embd]
                    [--use-pretuned]

Setup the environment for running inference

optional arguments:
  -h, --help            show this help message and exit
  --hf-repo {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B}, -hr {1bitLLM/bitnet_b1_58-large,1bitLLM/bitnet_b1_58-3B}
                        Model used for inference
  --model-dir MODEL_DIR, -md MODEL_DIR
                        Directory to save/load the model
  --log-dir LOG_DIR, -ld LOG_DIR
                        Directory to save the logging info
  --quant-type {i2_s,tl1}, -q {i2_s,tl1}
                        Quantization type
  --quant-embd          Quantize the embeddings to f16
  --use-pretuned, -p    Use the pretuned kernel parameters
</pre>
## Usage
### Basic usage
```bash
# Run inference with the quantized model
python run_inference.py -m models/bitnet_b1_58-large/ggml-model-i2_s.gguf -p "Microsoft Corporation is"
```
<pre>
usage: run_inference.py [-h] [-m MODEL] [-n N_PREDICT] -p PROMPT [-t THREADS] [-c CTX_SIZE]

Run inference

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to model file
  -n N_PREDICT, --n-predict N_PREDICT
                        Number of tokens to predict when generating text
  -p PROMPT, --prompt PROMPT
                        Prompt to generate text from
  -t THREADS, --threads THREADS
                        Number of threads to use
  -c CTX_SIZE, --ctx-size CTX_SIZE
                        Size of the prompt context
</pre>

Example output:  
>Microsoft Corporation is an American software company headquartered in Redmond, Washington, United States. The company is a subsidiary of Microsoft Corporation. Microsoft is an American software company that designs and develops computer software and services. The company was founded in 1975 by Bill Gates. Microsoft has its headquarters in Redmond, Washington, United States.
Microsoft Corporation's business is mainly focused on software development and application development. The company is the largest software company in the world and a member of the Microsoft Group. The company was founded by Bill Gates, Paul Allen, and Steve Ballmer.






### Advanced usage
// TODO
We provide a series of tools that allow you to manually tune the kernel for your own device.

We also provide scipts to generate fake bitnet models with different sizes, making it easier to test the performance of the framework on your machine.

```bash
python utils/generate-fake-bitnet-model.py models/bitnet_b1_58-large --outfile models/fake-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M

# Run benchmark with the generated model, use -m to specify the model path, -p to specify the prompt processed, -n to specify the number of token to generate
python utils/e2e_benchmark.py -m models/fake-bitnet-125m.tl1.gguf -p 512 -n 128
```
Example output:
![alt text](media/benchmark.png)

