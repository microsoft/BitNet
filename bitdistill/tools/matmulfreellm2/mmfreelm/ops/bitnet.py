import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from mmfreelm.modules import RMSNorm
from mmfreelm.modules.layernorm import RMSNormLinear


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
    # Compute the scale factor
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    # Quantize and then de-quantize the tensor
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u



class BitLinear(nn.Linear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        # Initialize the superclass nn.Linear with the given parameters
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features, eps=1e-8)

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
        x_norm = self.norm(x)

        # Apply quantization to both activations and weights
        # Uses Straight-Through Estimator (STE) trick with .detach() for gradient flow
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()

        # Perform linear operation with quantized values
        y = F.linear(x_quant, w_quant)
        return y


class BitLinear_wonorm_bmm(nn.Linear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features, out_features, bias=True):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        # Initialize the superclass nn.Linear with the given parameters
        super(BitLinear_wonorm_bmm, self).__init__(in_features, out_features, bias=bias)

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

        # Apply quantization to both activations and weights
        # Uses Straight-Through Estimator (STE) trick with .detach() for gradient flow
        x_quant = x + (activation_quant(x) - x).detach()
        w_quant = w + (weight_quant(w) - w).detach()

        # Perform linear operation with quantized values
        y = torch.bmm(x_quant, w_quant)
        return y
