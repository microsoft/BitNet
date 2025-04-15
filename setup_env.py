import subprocess
import signal
import sys
import os
import platform
import argparse
import logging
import shutil
from pathlib import Path

logger = logging.getLogger("setup_env")

SUPPORTED_HF_MODELS = {
    "1bitLLM/bitnet_b1_58-large": {
        "model_name": "bitnet_b1_58-large",
    },
    "1bitLLM/bitnet_b1_58-3B": {
        "model_name": "bitnet_b1_58-3B",
    },
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens": {
        "model_name": "Llama3-8B-1.58-100B-tokens",
    },
    "tiiuae/Falcon3-7B-Instruct-1.58bit": {
        "model_name": "Falcon3-7B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-7B-1.58bit": {
        "model_name": "Falcon3-7B-1.58bit",
    },
    "tiiuae/Falcon3-10B-Instruct-1.58bit": {
        "model_name": "Falcon3-10B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-10B-1.58bit": {
        "model_name": "Falcon3-10B-1.58bit",
    },
    "tiiuae/Falcon3-3B-Instruct-1.58bit": {
        "model_name": "Falcon3-3B-Instruct-1.58bit",
    },
    "tiiuae/Falcon3-3B-1.58bit": {
        "model_name": "Falcon3-3B-1.58bit",
    },
    "tiiuae/Falcon3-1B-Instruct-1.58bit": {
        "model_name": "Falcon3-1B-Instruct-1.58bit",
    },
    "microsoft/BitNet-b1.58-2B-4T": {
        "model_name": "BitNet-b1.58-2B-4T",
    },
}

SUPPORTED_QUANT_TYPES = {
    "arm64": ["i2_s", "tl1"],
    "x86_64": ["i2_s", "tl2"]
}

COMPILER_EXTRA_ARGS = {
    "arm64": ["-DBITNET_ARM_TL1=ON"],
    "x86_64": ["-DBITNET_X86_TL2=ON"]
}

OS_EXTRA_ARGS = {
    "Windows":["-T", "ClangCL"],
}

ARCH_ALIAS = {
    "AMD64": "x86_64",
    "x86": "x86_64",
    "x86_64": "x86_64",
    "aarch64": "arm64",
    "arm64": "arm64",
    "ARM64": "arm64",
}

def system_info():
    return platform.system(), ARCH_ALIAS[platform.machine()]

def get_model_name():
    if args.hf_repo:
        return SUPPORTED_HF_MODELS[args.hf_repo]["model_name"]
    return os.path.basename(os.path.normpath(args.model_dir))

def run_command(command, shell=False, log_step=None):
    """Run a system command and ensure it succeeds."""
    if log_step:
        log_file = os.path.join(args.log_dir, log_step + ".log")
        with open(log_file, "w") as f:
            try:
                subprocess.run(command, shell=shell, check=True, stdout=f, stderr=f)
            except subprocess.CalledProcessError as e:
                logging.error(f"Error occurred while running command: {e}, check details in {log_file}")
                sys.exit(1)
    else:
        try:
            subprocess.run(command, shell=shell, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error occurred while running command: {e}")
        sys.exit(1)

def prepare_model():
    _, arch = system_info()
    hf_url = args.hf_repo
    model_dir = args.model_dir
    quant_type = args.quant_type
    quant_embd = args.quant_embd
    if hf_url is not None:
        # download the model
        model_dir = os.path.join(model_dir, SUPPORTED_HF_MODELS[hf_url]["model_name"])
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        logging.info(f"Downloading model {hf_url} from HuggingFace to {model_dir}...")
        run_command(["huggingface-cli", "download", hf_url, "--local-dir", model_dir], log_step="download_model")
    elif not os.path.exists(model_dir):
        logging.error(f"Model directory {model_dir} does not exist.")
        sys.exit(1)
    else:
        logging.info(f"Loading model from directory {model_dir}.")
    gguf_path = os.path.join(model_dir, "ggml-model-" + quant_type + ".gguf")
    if not os.path.exists(gguf_path) or os.path.getsize(gguf_path) == 0:
        logging.info(f"Converting HF model to GGUF format...")
        if quant_type.startswith("tl"):
            run_command([sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir, "--outtype", quant_type, "--quant-embd"], log_step="convert_to_tl")
        else: # i2s
            # convert to f32
            run_command([sys.executable, "utils/convert-hf-to-gguf-bitnet.py", model_dir, "--outtype", "f32"], log_step="convert_to_f32_gguf")
            f32_model = os.path.join(model_dir, "ggml-model-f32.gguf")
            i2s_model = os.path.join(model_dir, "ggml-model-i2_s.gguf")
            # quantize to i2s
            if platform.system() != "Windows":
                if quant_embd:
                    run_command(["./build/bin/llama-quantize", "--token-embedding-type", "f16", f32_model, i2s_model, "I2_S", "1", "1"], log_step="quantize_to_i2s")
                else:
                    run_command(["./build/bin/llama-quantize", f32_model, i2s_model, "I2_S", "1"], log_step="quantize_to_i2s")
            else:
                if quant_embd:
                    run_command(["./build/bin/Release/llama-quantize", "--token-embedding-type", "f16", f32_model, i2s_model, "I2_S", "1", "1"], log_step="quantize_to_i2s")
                else:
                    run_command(["./build/bin/Release/llama-quantize", f32_model, i2s_model, "I2_S", "1"], log_step="quantize_to_i2s")

        logging.info(f"GGUF model saved at {gguf_path}")
    else:
        logging.info(f"GGUF model already exists at {gguf_path}")

def setup_gguf():
    # Install the pip package
    run_command([sys.executable, "-m", "pip", "install", "3rdparty/llama.cpp/gguf-py"], log_step="install_gguf")

def gen_code():
    _, arch = system_info()
    
    llama3_f3_models = set([model['model_name'] for model in SUPPORTED_HF_MODELS.values() if model['model_name'].startswith("Falcon3") or model['model_name'].startswith("Llama")])

    if arch == "arm64":
        if args.use_pretuned:
            pretuned_kernels = os.path.join("preset_kernels", get_model_name())
            if not os.path.exists(pretuned_kernels):
                logging.error(f"Pretuned kernels not found for model {args.hf_repo}")
                sys.exit(1)
            if args.quant_type == "tl1":
                shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl1.h"), "include/bitnet-lut-kernels.h")
                shutil.copyfile(os.path.join(pretuned_kernels, "kernel_config_tl1.ini"), "include/kernel_config.ini")
            elif args.quant_type == "tl2":
                shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl2.h"), "include/bitnet-lut-kernels.h")
                shutil.copyfile(os.path.join(pretuned_kernels, "kernel_config_tl2.ini"), "include/kernel_config.ini")
        if get_model_name() == "bitnet_b1_58-large":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-large", "--BM", "256,128,256", "--BK", "128,64,128", "--bm", "32,64,32"], log_step="codegen")
        elif get_model_name() in llama3_f3_models:
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "Llama3-8B-1.58-100B-tokens", "--BM", "256,128,256,128", "--BK", "128,64,128,64", "--bm", "32,64,32,64"], log_step="codegen")
        elif get_model_name() == "bitnet_b1_58-3B":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "64,128,64", "--bm", "32,64,32"], log_step="codegen")
        elif get_model_name() == "BitNet-b1.58-2B-4T":
            run_command([sys.executable, "utils/codegen_tl1.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "64,128,64", "--bm", "32,64,32"], log_step="codegen")
        else:
            raise NotImplementedError()
    else:
        if args.use_pretuned:
            # cp preset_kernels/model_name/bitnet-lut-kernels_tl1.h to include/bitnet-lut-kernels.h
            pretuned_kernels = os.path.join("preset_kernels", get_model_name())
            if not os.path.exists(pretuned_kernels):
                logging.error(f"Pretuned kernels not found for model {args.hf_repo}")
                sys.exit(1)
            shutil.copyfile(os.path.join(pretuned_kernels, "bitnet-lut-kernels-tl2.h"), "include/bitnet-lut-kernels.h")
        if get_model_name() == "bitnet_b1_58-large":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-large", "--BM", "256,128,256", "--BK", "96,192,96", "--bm", "32,32,32"], log_step="codegen")
        elif get_model_name() in llama3_f3_models:
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "Llama3-8B-1.58-100B-tokens", "--BM", "256,128,256,128", "--BK", "96,96,96,96", "--bm", "32,32,32,32"], log_step="codegen")
        elif get_model_name() == "bitnet_b1_58-3B":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "96,96,96", "--bm", "32,32,32"], log_step="codegen")
        elif get_model_name() == "BitNet-b1.58-2B-4T":
            run_command([sys.executable, "utils/codegen_tl2.py", "--model", "bitnet_b1_58-3B", "--BM", "160,320,320", "--BK", "96,96,96", "--bm", "32,32,32"], log_step="codegen")    
        else:
            raise NotImplementedError()


def compile():
    # Check if cmake is installed
    cmake_exists = subprocess.run(["cmake", "--version"], capture_output=True)
    if cmake_exists.returncode != 0:
        logging.error("Cmake is not available. Please install CMake and try again.")
        sys.exit(1)
    _, arch = system_info()
    if arch not in COMPILER_EXTRA_ARGS.keys():
        logging.error(f"Arch {arch} is not supported yet")
        exit(0)
    logging.info("Compiling the code using CMake.")
    run_command(["cmake", "-B", "build", *COMPILER_EXTRA_ARGS[arch], *OS_EXTRA_ARGS.get(platform.system(), [])], log_step="generate_build_files")
    # run_command(["cmake", "--build", "build", "--target", "llama-cli", "--config", "Release"])
    run_command(["cmake", "--build", "build", "--config", "Release"], log_step="compile")

def main():
    setup_gguf()
    gen_code()
    compile()
    prepare_model()
    
def parse_args():
    _, arch = system_info()
    parser = argparse.ArgumentParser(description='Setup the environment for running the inference')
    parser.add_argument("--hf-repo", "-hr", type=str, help="Model used for inference", choices=SUPPORTED_HF_MODELS.keys())
    parser.add_argument("--model-dir", "-md", type=str, help="Directory to save/load the model", default="models")
    parser.add_argument("--log-dir", "-ld", type=str, help="Directory to save the logging info", default="logs")
    parser.add_argument("--quant-type", "-q", type=str, help="Quantization type", choices=SUPPORTED_QUANT_TYPES[arch], default="i2_s")
    parser.add_argument("--quant-embd", action="store_true", help="Quantize the embeddings to f16")
    parser.add_argument("--use-pretuned", "-p", action="store_true", help="Use the pretuned kernel parameters")
    return parser.parse_args()

def signal_handler(sig, frame):
    logging.info("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    args = parse_args()
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    main()