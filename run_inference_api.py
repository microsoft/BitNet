# run_inference.py

import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """Run a system command, return stdout or raise on error."""
    result = subprocess.run(command, shell=shell, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed: {result.stderr.strip()}")
    return result.stdout

def generate(
    prompt: str,
    model: str = "models/bitnet_b1_58-3B/ggml-model-i2_s.gguf",
    n_predict: int = 128,
    threads: int = 2,
    ctx_size: int = 2048,
    temperature: float = 0.8,
    conversation: bool = False,
) -> str:
    # locate the llama-cli binary
    build_dir = "build"
    if platform.system() == "Windows":
        main_path = os.path.join(build_dir, "bin", "Release", "llama-cli.exe")
        if not os.path.exists(main_path):
            main_path = os.path.join(build_dir, "bin", "llama-cli")
    else:
        main_path = os.path.join(build_dir, "bin", "llama-cli")

    cmd = [
        main_path,
        "-m", model,
        "-n", str(n_predict),
        "-t", str(threads),
        "-p", prompt,
        "-ngl", "0",
        "-c", str(ctx_size),
        "--temp", str(temperature),
        "-b", "1",
    ]
    if conversation:
        cmd.append("-cnv")

    return run_command(cmd)

def main():
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))

    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("-m", "--model",     type=str,   default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-n", "--n-predict", type=int,   default=128)
    parser.add_argument("-p", "--prompt",    type=str,   required=True)
    parser.add_argument("-t", "--threads",   type=int,   default=2)
    parser.add_argument("-c", "--ctx-size",  type=int,   default=2048)
    parser.add_argument("-temp", "--temperature", type=float, default=0.8)
    parser.add_argument("-cnv", "--conversation", action="store_true")
    args = parser.parse_args()

    output = generate(
        prompt=args.prompt,
        model=args.model,
        n_predict=args.n_predict,
        threads=args.threads,
        ctx_size=args.ctx_size,
        temperature=args.temperature,
        conversation=args.conversation,
    )
    print(output)

if __name__ == "__main__":
    main()

