# bitnet.cpp

## Introduction
bitnet.cpp is the officially supported inference framework specifically designed for Bitnet ternary models, optimized for efficient CPU-based inference. bitnet.cpp offers a suite of optimized kernels, I2_S, TL1 ( Ternary Lookup 1 ) and TL2 ( Ternary Lookup 2 ), that support lossless inference of BitNet b1.58 models across both x86 and
ARM architectures. Below is a demo of bitnet.cpp runing 3.8B model on Apple M2. 

https://github.com/user-attachments/assets/96bfd877-73a4-4471-8af6-25af7da39ab7



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

#### Generate fake bitnet model
This script is designed to generate a fake Bitnet model in GGUF format.
   
```  
usage: generate-fake-bitnet-model.py [-h] [--vocab-only] [--outfile OUTFILE]  
                                     [--outtype {f32,f16,tl1,tl2}] [--bigendian]  
                                     [--use-temp-file] [--model-name MODEL_NAME]  
                                     [--model-size MODEL_SIZE] [--verbose]  
                                     model  
   
Generate a fake bitnet model with GGUF format  
   
positional arguments:  
  model                 directory containing model file  
   
optional arguments:  
  -h, --help            show this help message and exit  
  --vocab-only          extract only the vocab  
  --outfile OUTFILE     path to write to; default: based on input  
  --outtype {f32,f16,tl1,tl2}  
                        output format - use f32 for float32, f16 for float16  
  --bigendian           model is executed on big endian machine  
  --use-temp-file       use the tempfile library while processing (helpful when  
                        running out of memory, process killed)  
  --model-name MODEL_NAME  
                        name of the model  
  --model-size MODEL_SIZE  
                        size of the model  
  --verbose             increase output verbosity  
```  
   
Here's a brief explanation of each argument:  
   
- `model`: The directory containing the model file. This is a required positional argument.  
- `--vocab-only`: If specified, only the vocabulary will be extracted.  
- `--outfile`: The path to the output file. If not specified, the default is based on the input.  
- `--outtype`: The output format. Options are `f32` for float32, `f16` for float16, `tl1`, and `tl2`. The default is `f16`.  
- `--bigendian`: If specified, indicates that the model is executed on a big-endian machine.  
- `--use-temp-file`: If specified, the script will use the tempfile library while processing. This is helpful when running out of memory or if the process is killed.  
- `--model-name`: The name of the model. This is optional.  
- `--model-size`: The size of the model, such as "7B". The default is "7B".  
- `--verbose`: If specified, increases the output verbosity for debugging purposes.  
- `-h`, `--help`: Show the help message and exit.  
   
For example:  
   
```sh  
python utils/generate-fake-bitnet-model.py --outfile /path/to/output.gguf --model-size 1B /path/to/model/dir  
```  
   
This command would generate a fake Bitnet model with a size of 1B and write the output to `/path/to/output.gguf` using the model files located in `/path/to/model/dir`.
#### Benchmark
This script is designed to set up the environment for running the inference benchmark.  

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

