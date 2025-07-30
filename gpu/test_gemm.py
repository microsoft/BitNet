import torch
from model import BitLinear, BitLinearKernel
from pack_weight import convert_weight_int8_to_int2

from torch import nn


def quant_weight_int8(weight):
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    new_weight = (weight * s).round().clamp(-1, 1).to(torch.int8)
    new_scale = (1.0 / s).to(torch.bfloat16)
    return new_weight, new_scale.reshape(1).repeat(6)

def quant_weight(weight):
    s = 1.0 / weight.abs().mean().clamp_(min=1e-5)
    new_weight = (weight * s).round().clamp(-1, 1) / s

    return new_weight

def convert_int8_to_int2(weight):
    return convert_weight_int8_to_int2(weight)


def test_BitLinear():
    in_dim = 2560   # 64
    out_dim = 3840  # 32
    default_dtype = torch.bfloat16
    x = torch.randn(128, in_dim, dtype=default_dtype).cuda()  # (batch, in_features)
    
    layer0 = BitLinear(in_dim, out_dim).cuda()
    layer0 = layer0.to(default_dtype) 
    nn.init.kaiming_uniform_(layer0.weight, nonlinearity='relu')
    with torch.no_grad():
        layer0.weight.copy_( quant_weight(layer0.weight))
    nn.init.zeros_(layer0.bias)
    out0 = layer0(x)

    assert not torch.isnan(out0).any()
    assert layer0.weight.dtype == default_dtype
    # print(layer0.weight.dtype, layer0.weight.shape)
    
    layer1 = BitLinearKernel(in_dim, out_dim).cuda()
    weight_int8, scale = quant_weight_int8(layer0.weight)
    weight = convert_int8_to_int2(weight_int8)

    
    with torch.no_grad():
        layer1.weight.copy_(weight)
        layer1.weight_scale.copy_(scale)
    print(layer1.weight, layer1.weight_scale)
    out1 = layer1(x, weight_int8)
    assert out1.dtype == default_dtype

    print(f"Non-kernel output: {out0}, Kernel output: {out1}")
    assert torch.equal(out0, out1), "Outputs from non-kernel and kernel paths should match"


if __name__ == "__main__":
    test_BitLinear()
