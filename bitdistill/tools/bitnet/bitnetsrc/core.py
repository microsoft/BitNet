from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from mmfreelm.modules import RMSNorm
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm as RMSNorm


def activation_quant(x):
    """
    Per-token quantization to 8 bits. No grouping is needed for quantization.

    Args:
        x: An activation tensor with shape [n, d].

    Returns:
        A quantized activation tensor with shape [n, d].
    """
    # Compute the scale factor
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    # Compute the scale factorw = w.to(torch.float32)
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

def weight_quant_minmax(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    # Compute the scale factorw = w.to(torch.float32)
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def weight_quant_gptq(w):
    """
    GPTQ-inspired quantization using second-order information
    Approximates Hessian with weight covariance for single-function implementation
    """
    # Flatten to 2D for matrix operations
    original_shape = w.shape
    if w.dim() > 2:
        w_flat = w.view(w.shape[0], -1)
    else:
        w_flat = w
    
    # Approximate Hessian diagonal using weight variance (simplified)
    if w_flat.shape[1] > 1:
        # Compute approximate Hessian diagonal
        hess_diag = w_flat.var(dim=0, keepdim=True) + 1e-8
        
        # GPTQ-style optimal scaling with second-order info
        inv_hess = 1.0 / hess_diag.sqrt()
        scale = inv_hess.mean() / w_flat.abs().mean().clamp_(min=1e-5)
    else:
        scale = 1.0 / w_flat.abs().mean().clamp_(min=1e-5)
    
    # Quantize with second-order aware scaling
    u = (w_flat * scale).round().clamp_(-1, 1) / scale
    return u.view(original_shape)


def weight_quant_block(w, block_size=256):
    """Alternative implementation: block-wise across all dimensions"""

    if w.dim() < 2:
        # Fallback to regular quantization for 1D tensors
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
        return u
    
    original_shape = w.shape
    total_elements = w.numel()
    
    # Flatten completely
    w_flat = w.view(-1)
    
    # Calculate number of blocks
    num_blocks = (total_elements + block_size - 1) // block_size
    
    # Pad if necessary
    if total_elements % block_size != 0:
        pad_size = num_blocks * block_size - total_elements
        w_flat = torch.cat([w_flat, torch.zeros(pad_size, device=w.device, dtype=w.dtype)])
    
    # Reshape into blocks: [num_blocks, block_size]
    w_blocks = w_flat.view(num_blocks, block_size)
    
    # Compute scale for each block: [num_blocks, 1]
    scales = 1.0 / w_blocks.abs().mean(dim=1, keepdim=True).clamp_(min=1e-5)
    
    # Quantize each block
    u_blocks = (w_blocks * scales).round().clamp_(-1, 1) / scales
    
    # Reshape back to flat format
    u_flat = u_blocks.view(-1)
    
    # Remove padding if it was added
    if total_elements % block_size != 0:
        u_flat = u_flat[:total_elements]
    
    # Reshape back to original shape
    u = u_flat.view(original_shape)
    
    return u

def weight_quant_perchannel(w):
    # Different scale for each output channel (row)
    if w.dim() >= 2:
        # Compute scale for each row
        scales = 1.0 / w.abs().mean(dim=tuple(range(1, w.dim())), keepdim=True).clamp_(min=1e-5)
        u = (w * scales).round().clamp_(-1, 1) / scales
    else:
        # Fallback to regular quantization for 1D tensors
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def weight_quant_awq(w):
    """
    AWQ-inspired activation-aware quantization
    Uses weight importance based on channel-wise activation patterns
    """
    # Simulate activation importance using weight magnitude patterns
    if w.dim() >= 2:
        # Channel importance approximation (normally would use real activations)
        channel_importance = w.abs().mean(dim=tuple(range(1, w.dim())), keepdim=True)
        
        # Protect important channels with different scaling
        importance_threshold = channel_importance.float().quantile(0.8)
        important_mask = channel_importance > importance_threshold
        
        # Different scales for important vs regular channels
        scale_base = 1.0 / w.abs().mean().clamp_(min=1e-5)
        
        # Reduce quantization aggressiveness for important channels
        scale = torch.where(important_mask, scale_base * 0.7, scale_base * 1.2)
        
        u = (w * scale).round().clamp_(-1, 1) / scale
    else:
        # Fallback for 1D tensors
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        u = (w * scale).round().clamp_(-1, 1) / scale
    
    return u

def weight_log_quant(w, num_levels=3):  # 1.58 bit可对应3 levels（-1, 0, 1）或自定义
    """
    Per-tensor log-domain quantization.

    Args:
        w: A weight tensor with shape [d, k].
        num_levels: Number of quantization levels (default 3: -1, 0, 1)

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    w = w.to(torch.float32)
    sign = w.sign()
    w_abs = w.abs() + 1e-8  # Avoid log(0)

    # Optional: clip极大值，防止极端异常值影响量化区间
    # clip_val = torch.quantile(w_abs, 0.999)  # 只clip极端大于99.9分位的值
    # w_abs = torch.clamp(w_abs, max=clip_val)

    # 转为log域，保证数值不为0
    w_log = torch.log(w_abs)

    # 均匀量化log域数据
    min_log = w_log.min()
    max_log = w_log.max()
    q_step = (max_log - min_log) / (num_levels-1)
    q_levels = torch.linspace(min_log, max_log, num_levels, device=w.device)

    # 找到最近的level索引，并反量化
    idx = ((w_log - min_log) / q_step).round().clamp(0, num_levels - 1).long()
    w_log_q = q_levels[idx]

    # 逆log还原
    w_q = sign * torch.exp(w_log_q)
    w_q = w_q.to(torch.bfloat16)
    return w_q


def fft_weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.

    Args:
        w: A weight tensor with shape [d, k].

    Returns:
        A quantized weight tensor with shape [d, k].
    """
    # Compute the scale factor
    w = w.to(torch.float32)
    w = torch.fft.fft2(w)
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    w = w.to(torch.bfloat16)
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1) / scale
    w = w.to(torch.float32)
    w = torch.fft.ifft2(w).real
    w = w.to(torch.bfloat16)
    return u

import os
def is_main_process():
    return int(os.environ.get("RANK", "0")) == 0


class BitLinear(nn.Linear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features, out_features, should_rms=False, bias=False, weight_quant_method='minmax'):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        # Initialize the superclass nn.Linear with the given parameters
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)
        self.should_rms = should_rms
        if self.should_rms:
            self.norm = RMSNorm(in_features, eps=1e-8)
        self.weight_quant_method = weight_quant_method

        quant_methods = {
            'minmax': weight_quant_minmax,
            'gptq': weight_quant_gptq,
            'block': weight_quant_block,
            'perchannel': weight_quant_perchannel,
            'awq': weight_quant_awq,
            'log': weight_log_quant,
            'fft': fft_weight_quant
        }
        if is_main_process():
            print("=====> Using Quant Method", self.weight_quant_method)
        if self.weight_quant_method in quant_methods:
            self.weight_quant_func = quant_methods[self.weight_quant_method]
        else:
            raise ValueError(f"Unsupported weight quant method: {self.weight_quant_method}")

    def forward(self, x):
        """
        Overrides the forward pass to include quantization.

        Args:
            x: An input tensor with shape [n, d].

        Returns:
            An output tensor with shape [n, d].
        """
        # Weight tensor
        w = self.weight

        # Apply RMS normalization to the input
        x_norm = x
        if self.should_rms:
            x_norm = self.norm(x)
        
        # Apply quantization to both activations and weights
        # Uses Straight-Through Estimator (STE) trick with .detach() for gradient flow
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (self.weight_quant_func(w) - w).detach()
        # Perform linear operation with quantized values
        y = F.linear(x_quant, w_quant, self.bias)

        return y