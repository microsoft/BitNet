from safetensors import safe_open
from safetensors.torch import save_file
import torch

def quant_weight_fp16(weight):
    weight = weight.to(torch.float)
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    new_weight = (weight * s).round().clamp(-1, 1) / s
    return new_weight

def quant_model(input, output):
    tensors = {}

    with safe_open(input, framework='pt') as f:
        for name in f.keys():
            tensors[name] = f.get_tensor(name)

            keyword_list = [
                'q_proj.weight', 
                'k_proj.weight', 
                'v_proj.weight',
                'o_proj.weight',
                'gate_proj.weight',
                'up_proj.weight',
                'down_proj.weight'
            ]

            if any(keyword in name for keyword in keyword_list):
                print(f'[INFO] Quantizing {name}')
                tensors[name] = quant_weight_fp16(tensors[name])
    
    print(f'[INFO] Saving to {output}\nThis may take a while.')
    save_file(tensors, output)
                

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Safetensors back to Torch .pth checkpoint")
    parser.add_argument(
        "--input", type=str, required=True,
    )
    parser.add_argument(
        "--output", type=str, required=True,
    )
    args = parser.parse_args()

    quant_model(
        input=args.input,
        output=args.output,
    )