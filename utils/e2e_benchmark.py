import os
import sys
import logging
import argparse
import platform
import subprocess

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

def run_benchmark():
    build_dir =  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build")
    if platform.system() == "Windows":
        bench_path = os.path.join(build_dir, "bin", "Release", "llama-bench.exe")
        if not os.path.exists(bench_path):
            bench_path = os.path.join(build_dir, "bin", "llama-bench")
    else:
        bench_path = os.path.join(build_dir, "bin", "llama-bench")
    if not os.path.exists(bench_path):
        logging.error(f"Benchmark binary not found, please build first.")
        sys.exit(1)
    command = [
        f'{bench_path}',
        '-m', args.model,
        '-n', str(args.n_token),
        '-ngl', '0',
        '-b', '1',
        '-t', str(args.threads),
        '-p', str(args.n_prompt),
        '-r', '5'
    ]
    run_command(command)

def parse_args():
    parser = argparse.ArgumentParser(description='Setup the environment for running the inference')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=True)
    parser.add_argument("-n", "--n-token", type=int, help="Number of generated tokens", required=False, default=128)
    parser.add_argument("-p", "--n-prompt", type=int, help="Prompt to generate text from", required=False, default=512)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    return parser.parse_args()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    run_benchmark()