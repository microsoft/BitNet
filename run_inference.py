import os
import sys
import signal
import platform
import argparse
import subprocess
import logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def check_cuda():
    """Check if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logging.warning("CUDA is not available. Falling back to CPU.")
    except ImportError:
        logging.warning("PyTorch not installed. CUDA check skipped.")
    return False

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        logging.info(f"Executing command: {' '.join(command)}")
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_inference():
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")

    if not os.path.exists(main_path):
        logging.error(f"The executable {main_path} does not exist. Please ensure the build directory is correct.")
        sys.exit(1)

    command = [
        f'{main_path}',
        '-m', args.model,
        '-n', str(args.n_predict),
        '-t', str(args.threads),
        '-p', args.prompt,
        '-ngl', '1' if check_cuda() else '0',
        '-c', str(args.ctx_size),
        '--temp', str(args.temperature),
        "-b", "1",
    ]

    if args.conversation:
        command.append("-cnv")

    logging.info("Starting inference process...")
    run_command(command)

def signal_handler(sig, frame):
    logging.info("Ctrl+C pressed, exiting...")
    sys.exit(0)

if __name__ == "__main__":
    setup_logging()
    logging.info("Initializing inference script.")
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description='Run inference')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict when generating text", required=False, default=128)
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to generate text from", required=True)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the prompt context", required=False, default=2048)
    parser.add_argument("-temp", "--temperature", type=float, help="Temperature, a hyperparameter that controls the randomness of the generated text", required=False, default=0.8)
    parser.add_argument("-cnv", "--conversation", action='store_true', help="Whether to enable chat mode or not (for instruct models.)")

    args = parser.parse_args()
    logging.info("Parsed arguments successfully.")
    run_inference()