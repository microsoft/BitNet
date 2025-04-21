import os
import sys
import signal
import platform
import argparse
import subprocess

def run_command(command, shell=False):
    """Run a system command and ensure it succeeds."""
    try:
        subprocess.run(command, shell=shell, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_server():
    build_dir = "build"
    if platform.system() == "Windows":
        server_path = os.path.join(build_dir, "bin", "Release", "llama-server.exe")
        if not os.path.exists(server_path):
            server_path = os.path.join(build_dir, "bin", "llama-server")
    else:
        server_path = os.path.join(build_dir, "bin", "llama-server")
    
    command = [
        f'{server_path}',
        '-m', args.model,
        '-c', str(args.ctx_size),
        '-t', str(args.threads),
        '-n', str(args.n_predict),
        '-ngl', '0',
        '--temp', str(args.temperature),
        '--host', args.host,
        '--port', str(args.port),
        '-cb'  # Enable continuous batching
    ]
    
    if args.prompt:
        command.extend(['-p', args.prompt])
    
    # Note: -cnv flag is removed as it's not supported by the server
    
    print(f"Starting server on {args.host}:{args.port}")
    run_command(command)

def signal_handler(sig, frame):
    print("Ctrl+C pressed, shutting down server...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description='Run llama.cpp server')
    parser.add_argument("-m", "--model", type=str, help="Path to model file", required=False, default="models/bitnet_b1_58-3B/ggml-model-i2_s.gguf")
    parser.add_argument("-p", "--prompt", type=str, help="System prompt for the model", required=False)
    parser.add_argument("-n", "--n-predict", type=int, help="Number of tokens to predict", required=False, default=4096)
    parser.add_argument("-t", "--threads", type=int, help="Number of threads to use", required=False, default=2)
    parser.add_argument("-c", "--ctx-size", type=int, help="Size of the context window", required=False, default=2048)
    parser.add_argument("--temperature", type=float, help="Temperature for sampling", required=False, default=0.8)
    parser.add_argument("--host", type=str, help="IP address to listen on", required=False, default="127.0.0.1")
    parser.add_argument("--port", type=int, help="Port to listen on", required=False, default=8080)
    
    args = parser.parse_args()
    run_server()
