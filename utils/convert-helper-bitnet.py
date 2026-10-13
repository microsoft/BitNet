#!/usr/bin/env python3

import sys
import os
import shutil
import subprocess
from pathlib import Path

def run_command(command_list, cwd=None, check=True):
    print(f"Executing: {' '.join(map(str, command_list))}")
    try:
        process = subprocess.run(command_list, cwd=cwd, check=check, capture_output=False, text=True)
        return process
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(map(str, e.cmd))}")
        print(f"Return code: {e.returncode}")
        raise

def main():
    if len(sys.argv) < 2:
        script_name = Path(sys.argv[0]).name
        print(f"Usage: python {script_name} <model-directory>")
        sys.exit(1)

    model_dir_arg = sys.argv[1]
    model_dir = Path(model_dir_arg).resolve()

    if not model_dir.is_dir():
        print(f"Error: Model directory '{model_dir}' not found or is not a directory.")
        sys.exit(1)

    utils_dir = Path(__file__).parent.resolve()
    project_root_dir = utils_dir.parent

    preprocess_script = utils_dir / "preprocess-huggingface-bitnet.py"
    convert_script = utils_dir / "convert-ms-to-gguf-bitnet.py"
    
    llama_quantize_binary = project_root_dir / "build" / "bin" / "llama-quantize"

    input_file = model_dir / "model.safetensors"
    input_backup_file = model_dir / "model.safetensors.backup"
    preprocessed_output_file = model_dir / "model.safetensors"

    gguf_f32_output = model_dir / "ggml-model-f32-bitnet.gguf"
    gguf_i2s_output = model_dir / "ggml-model-i2s-bitnet.gguf"

    if not preprocess_script.is_file():
        print(f"Error: Preprocess script not found at '{preprocess_script}'")
        sys.exit(1)
    if not convert_script.is_file():
        print(f"Error: Convert script not found at '{convert_script}'")
        sys.exit(1)
    if not llama_quantize_binary.is_file():
        print(f"Error: llama-quantize binary not found at '{llama_quantize_binary}'")
        sys.exit(1)

    if not input_file.is_file():
        print(f"Error: Input safetensors file not found at '{input_file}'")
        sys.exit(1)

    # A previous interrupted conversion may have left the only original here.
    # Never overwrite it or clean up the input unless this run moved it first.
    if input_backup_file.exists():
        print(f"Error: Backup already exists: '{input_backup_file}'. Restore or move it before retrying.")
        return 1

    backup_created = False
    try:
        print(f"Backing up '{input_file}' to '{input_backup_file}'")
        shutil.move(input_file, input_backup_file)
        backup_created = True

        print("Preprocessing huggingface checkpoint...")
        cmd_preprocess = [
            sys.executable,
            str(preprocess_script),
            "--input", str(input_backup_file),
            "--output", str(preprocessed_output_file)
        ]
        run_command(cmd_preprocess)

        print("Converting to GGUF (f32)...")
        cmd_convert = [
            sys.executable,
            str(convert_script),
            str(model_dir),
            "--vocab-type", "bpe",
            "--outtype", "f32",
            "--concurrency", "1",
            "--outfile", str(gguf_f32_output)
        ]
        run_command(cmd_convert)

        print("Quantizing model to I2_S...")
        cmd_quantize = [
            str(llama_quantize_binary),
            str(gguf_f32_output),
            str(gguf_i2s_output),
            "I2_S",
            "1"
        ]
        run_command(cmd_quantize)

        print("Convert successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    finally:
        print("Cleaning up intermediate files...")
        if backup_created and preprocessed_output_file.exists():
            print(f"Removing preprocessed file: {preprocessed_output_file}")
            try:
                preprocessed_output_file.unlink()
            except OSError as e:
                print(f"Warning: Could not remove {preprocessed_output_file}: {e}")
        
        # if gguf_f32_output.exists():
        #     print(f"Removing f32 GGUF: {gguf_f32_output}")
        #     try:
        #         gguf_f32_output.unlink()
        #     except OSError as e:
        #         print(f"Warning: Could not remove {gguf_f32_output}: {e}")
        
        if backup_created:
            # Do not discard the backup if cleanup failed and the transformed
            # checkpoint still occupies the original path.
            if input_file.exists():
                raise RuntimeError(f"Cannot restore checkpoint; original retained at {input_backup_file}")
            shutil.move(input_backup_file, input_file)

if __name__ == "__main__":
    sys.exit(main() or 0)
