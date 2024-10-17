# bitnet.cpp

bitnet.cpp is the officially supported inference framework specifically designed for BitNet models (e.g., BitNet b1.58), optimized for efficient CPU-based inference. bitnet.cpp offers a suite of optimized kernels, that support lossless inference of BitNet b1.58 models on both x86 and ARM architectures. 

## Demo

A demo of bitnet.cpp runing 3.8B model on Apple M2:

https://github.com/user-attachments/assets/96bfd877-73a4-4471-8af6-25af7da39ab7

## Timeline

- 10/17/2024 bitnet.cpp is public.
- 02/27/2024 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- 10/17/2023 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

## Acknowledgements

This project is based on the [llama.cpp](https://github.com/ggerganov/llama.cpp) framework. We would like to thank all the authors for their contributions to the open-source community. We also thank [T-MAC](https://github.com/microsoft/T-MAC/) team for the helpful discussion on the LUT method for low-bit LLM inference.

## Supported Models

bitnet.cpp supports a list of 1-bit models available on [Hugging Face](https://huggingface.co/)

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
- clang (if using Windows, Visual Studio is needed for clang support)

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
python run_inference.py -m models/bitnet_b1_58-large/ggml-model-i2_s.gguf -p "Water is"
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
> Water is one of the most crucial resources on the planet. When it comes to water, we need a lot. It's the most precious commodity we have, and we don't get enough. We need to do more to conserve water, and with the help of our water conservation experts, we can do it.


### Benchmark
We provide scripts to run the inference benchmark providing a model.

```  
usage: e2e_benchmark.py -m MODEL [-n N_TOKEN] [-p N_PROMPT] [-t THREADS]  
   
Setup the environment for running the inference  
   
required arguments:  
  -m MODEL, --model MODEL  
                        Path to the model file. 
   
optional arguments:  
  -h, --help  
                        Show this help message and exit. 
  -n N_TOKEN, --n-token N_TOKEN  
                        Number of generated tokens. 
  -p N_PROMPT, --n-prompt N_PROMPT  
                        Prompt to generate text from. 
  -t THREADS, --threads THREADS  
                        Number of threads to use. 
```  
   
Here's a brief explanation of each argument:  
   
- `-m`, `--model`: The path to the model file. This is a required argument that must be provided when running the script.  
- `-n`, `--n-token`: The number of tokens to generate during the inference. It is an optional argument with a default value of 128.  
- `-p`, `--n-prompt`: The number of prompt tokens to use for generating text. This is an optional argument with a default value of 512.  
- `-t`, `--threads`: The number of threads to use for running the inference. It is an optional argument with a default value of 2.  
- `-h`, `--help`: Show the help message and exit. Use this argument to display usage information.  
   
For example:  
   
```sh  
python utils/e2e_benchmark.py -m /path/to/model -n 200 -p 256 -t 4  
```  
   
This command would run the inference benchmark using the model located at `/path/to/model`, generating 200 tokens from a 256 token prompt, utilizing 4 threads.  

For the model layout that do not supported by any public model, we provide scripts to generate a dummy model with the given model layout, and run the benchmark on your machine:

```bash
python utils/generate-fake-bitnet-model.py models/bitnet_b1_58-large --outfile models/fake-bitnet-125m.tl1.gguf --outtype tl1 --model-size 125M

# Run benchmark with the generated model, use -m to specify the model path, -p to specify the prompt processed, -n to specify the number of token to generate
python utils/e2e_benchmark.py -m models/fake-bitnet-125m.tl1.gguf -p 512 -n 128
```
