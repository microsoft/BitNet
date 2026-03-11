import os
import sys
import signal
import platform
import argparse
import subprocess
import re
import ipaddress

def validate_path(path):
    """Validate that a path is safe and doesn't contain command injection attempts."""
    # Normalize the path to prevent directory traversal
    normalized = os.path.normpath(path)
    # Check for suspicious patterns that might indicate command injection
    suspicious_patterns = [';', '&', '|', '$', '`', '\n', '\r', '>', '<', '(', ')']
    if any(char in normalized for char in suspicious_patterns):
        raise ValueError(f"Invalid characters detected in path: {path}")
    return normalized

def validate_ip_address(ip):
    """Validate that the IP address is valid."""
    try:
        ipaddress.ip_address(ip)
        return ip
    except ValueError:
        raise ValueError(f"Invalid IP address: {ip}")

def validate_prompt(prompt):
    """Validate prompt to prevent command injection."""
    # Check for suspicious patterns in prompt
    suspicious_patterns = ['$(', '`', '|', ';', '&', '\n', '\r']
    if any(pattern in prompt for pattern in suspicious_patterns):
        raise ValueError(f"Invalid characters detected in prompt")
    return prompt

def run_command(command):
    """Run a system command safely without shell=True."""
    try:
        # Force shell=False to prevent command injection
        subprocess.run(command, shell=False, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running command: {e}")
        sys.exit(1)

def run_server():
    # Validate all user inputs before using them
    try:
        validated_model = validate_path(args.model)
        validated_host = validate_ip_address(args.host)
        if args.prompt:
            validated_prompt = validate_prompt(args.prompt)
    except ValueError as e:
        print(f"Validation error: {e}")
        sys.exit(1)
    
    build_dir = "build"
    if platform.system() == "Windows":
        server_path = os.path.join(build_dir, "bin", "Release", "llama-server.exe")
        if not os.path.exists(server_path):
            server_path = os.path.join(build_dir, "bin", "llama-server")
    else:
        server_path = os.path.join(build_dir, "bin", "llama-server")
    
    command = [
        f'{server_path}',
        '-m', validated_model,
        '-c', str(args.ctx_size),
        '-t', str(args.threads),
        '-n', str(args.n_predict),
        '-ngl', '0',
        '--temp', str(args.temperature),
        '--host', validated_host,
        '--port', str(args.port),
        '-cb'  # Enable continuous batching
    ]
    
    if args.prompt:
        command.extend(['-p', validated_prompt])
    
    # Note: -cnv flag is removed as it's not supported by the server
    
    print(f"Starting server on {validated_host}:{args.port}")
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
