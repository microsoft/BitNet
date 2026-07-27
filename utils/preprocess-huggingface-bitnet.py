from safetensors import safe_open
from safetensors.torch import save_file
import torch


def quant_weight_fp16(weight):
    """Quantize weight tensor to ternary values (-1, 0, 1) using FP16 precision.
    
    Args:
        weight: Input weight tensor to quantize
        
    Returns:
        Quantized weight tensor with values in range [-1, 1]
    """
    weight = weight.to(torch.float)
    # Calculate scaling factor based on mean absolute value, with minimum clamp for stability
    scale = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    # Scale, round to nearest integer, clamp to ternary values, then unscale
    quantized_weight = (weight * scale).round().clamp(-1, 1) / scale
    return quantized_weight


def quant_model(input_path, output_path):
    """Quantize specific weight tensors in a safetensors model file.
    
    Quantizes projection and gate weights commonly found in transformer models
    while preserving other tensors unchanged.
    
    Args:
        input_path: Path to input safetensors file
        output_path: Path to output safetensors file
    """
    tensors = {}

    # Define weight tensor names to quantize (transformer projection layers)
    quantizable_keywords = [
        'q_proj.weight',     # Query projection
        'k_proj.weight',     # Key projection  
        'v_proj.weight',     # Value projection
        'o_proj.weight',     # Output projection
        'gate_proj.weight',  # Gate projection
        'up_proj.weight',    # Up projection
        'down_proj.weight'   # Down projection
    ]

    print(f'[INFO] Loading model from {input_path}')
    with safe_open(input_path, framework='pt') as f:
        for tensor_name in f.keys():
            tensors[tensor_name] = f.get_tensor(tensor_name)

            # Check if this tensor should be quantized
            if any(keyword in tensor_name for keyword in quantizable_keywords):
                print(f'[INFO] Quantizing {tensor_name}')
                tensors[tensor_name] = quant_weight_fp16(tensors[tensor_name])
    
    print(f'[INFO] Saving quantized model to {output_path}')
    print('[INFO] This may take a while...')
    save_file(tensors, output_path)
    print('[INFO] Quantization complete!')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Quantize transformer model weights in safetensors format to ternary values"
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True,
        help="Path to input safetensors file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Path to output safetensors file"
    )
    args = parser.parse_args()

    quant_model(
        input_path=args.input,
        output_path=args.output,
    )