# Copyright (c) Microsoft. All rights reserved.
# PyTorch model wrapper for Metal backend

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F

# Try to import Metal extension
try:
    import bitnet_metal

    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("Warning: Metal extension not available. Falling back to MPS or CPU.")


def bitnet_int8xint2_linear_metal(input0, input1, s, ws):
    """
    Metal-accelerated int8 x int2 linear layer.

    Args:
        input0: int8 tensor [M, K] - quantized input activations
        input1: int8 tensor [N, K/4] - packed 2-bit weights
        s: bfloat16 tensor [1] - input scale
        ws: bfloat16 tensor [4] - weight scales

    Returns:
        bfloat16 tensor [M, N] - output
    """
    if not METAL_AVAILABLE:
        raise RuntimeError("Metal extension not available")

    out_shape = list(input0.shape)
    out_shape[-1] = input1.shape[0]

    M = input0.shape[0]
    if len(out_shape) == 3:
        M *= input0.shape[1]
    N = input1.shape[0]
    K = input1.shape[1] * 4

    ret = torch.zeros(*out_shape, dtype=torch.bfloat16, device=input0.device)

    # Call Metal kernel
    bitnet_metal.bitlinear_metal(
        M,
        N,
        K,
        input0.data_ptr(),
        input1.data_ptr(),
        ret.data_ptr(),
        s.data_ptr(),
        ws.data_ptr(),
    )

    return ret


def bitnet_int8xint2_linear_mps(input0, input1, s, ws):
    """
    MPS fallback using PyTorch operations.
    This is slower but works without custom Metal kernels.
    """
    # Decode 2-bit weights to int8
    N, K_packed = input1.shape
    K = K_packed * 4

    # Unpack weights: each byte has 4 2-bit values
    weights = torch.zeros((N, K), dtype=torch.int8, device=input0.device)
    for i in range(4):
        shift = i * 2
        mask = 0x03
        # Extract 2-bit values and map 0->-1, 1->0, 2->1
        unpacked = ((input1 >> shift) & mask).to(torch.int8) - 1
        weights[:, i::4] = unpacked

    # Matrix multiplication: int8 x int8 -> int32
    # PyTorch MPS doesn't support int8 matmul directly, so convert to int16
    input_int16 = input0.to(torch.int16)
    weights_int16 = weights.to(torch.int16)
    result = torch.matmul(input_int16, weights_int16.t())

    # Apply scales and convert to bfloat16
    # result = acc / s * ws
    result_float = result.to(torch.float32)
    result_float = result_float / s.to(torch.float32)

    # Apply weight scales (per-channel)
    ws_idx = torch.arange(N, device=input0.device) % 4
    result_float = result_float * ws[ws_idx].to(torch.float32).unsqueeze(0)

    return result_float.to(torch.bfloat16)


def pack_weight_int8_to_int2(weight_int8):
    """
    Pack int8 weights (values -1, 0, +1) into 2-bit format.

    Args:
        weight_int8: [N, K] int8 tensor with values in {-1, 0, 1}

    Returns:
        [N, K/4] uint8 packed tensor
    """
    N, K = weight_int8.shape
    assert K % 4 == 0, "K must be divisible by 4"

    # Map -1->0, 0->1, 1->2
    mapped = (weight_int8 + 1).to(torch.uint8)

    # Pack 4 values per byte
    packed = torch.zeros((N, K // 4), dtype=torch.uint8, device=weight_int8.device)
    for i in range(4):
        packed |= (mapped[:, i::4] & 0x03) << (i * 2)

    return packed


@dataclass
class ModelArgs:
    dim: int = 2560
    n_layers: int = 30
    n_heads: int = 20
    n_kv_heads: int = 5
    vocab_size: int = 128256
    ffn_dim: int = 6912
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    use_kernel: bool = True  # Use Metal kernels if available
    use_mps_fallback: bool = True  # Use MPS if Metal kernels unavailable


LayerCache = Tuple[torch.Tensor, torch.Tensor]


class BitLinearMetal(nn.Module):
    """Metal-accelerated BitLinear layer."""

    in_features: int
    out_features: int
    weight: torch.Tensor
    weight_scale: torch.Tensor
    use_mps_fallback: bool

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_mps_fallback: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_mps_fallback = use_mps_fallback

        # Weight stored as packed int2 (4 values per byte)
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features // 4, dtype=torch.int8),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.zeros(4, dtype=torch.bfloat16), requires_grad=False
        )

    # Note: torch.compile disabled for compatibility
    # @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127).to(torch.int8), s

    def forward(self, input):
        input, s = self.quant_input(input)

        if METAL_AVAILABLE and input.device.type == "mps":
            return bitnet_int8xint2_linear_metal(
                input, self.weight, s, self.weight_scale
            )
        elif self.use_mps_fallback and input.device.type == "mps":
            return bitnet_int8xint2_linear_mps(input, self.weight, s, self.weight_scale)
        else:
            # CPU fallback
            return bitnet_int8xint2_linear_mps(input, self.weight, s, self.weight_scale)


class BitLinear(nn.Linear):
    """Standard BitLinear without kernel acceleration."""

    # @torch.compile
    def quant_input(self, input):
        s = 127 / input.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        return (input * s).round().clamp(-128, 127) / s

    def forward(self, input):
        input = self.quant_input(input)
        return F.linear(input, self.weight)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.rope_theta = rope_theta

        self.n_local_heads = n_heads
        self.n_local_kv_heads = n_kv_heads

        Linear = BitLinearMetal if use_kernel else BitLinear

        self.wqkv = Linear(
            dim,
            (self.n_local_heads + 2 * self.n_local_kv_heads) * head_dim,
            bias=False,
        )
        self.wo = Linear(
            self.n_local_heads * head_dim,
            dim,
            bias=False,
        )

        self.attn_sub_norm = nn.RMSNorm(dim, norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cache: LayerCache,
    ) -> torch.Tensor:
        # x shape: [batch * seq_len, dim]
        # For simplicity, treat each token independently (no cross-attention)

        xqkv = self.wqkv(x)
        xq = xqkv[:, : (self.n_local_heads * self.head_dim)]
        xkv = xqkv[:, (self.n_local_heads * self.head_dim) :]
        xk, xv = xkv.chunk(2, 1)

        # Reshape: [batch*seq, n_heads * head_dim] -> [batch*seq, n_heads, head_dim]
        xq = xq.view(-1, self.n_local_heads, self.head_dim)
        xk = xk.view(-1, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(-1, self.n_local_kv_heads, self.head_dim)

        # Group query attention
        heads_per_group = self.n_local_heads // self.n_local_kv_heads
        xq_grouped = xq.view(-1, self.n_local_kv_heads, heads_per_group, self.head_dim)

        # Expand keys and values
        xk_expanded = xk.unsqueeze(2).expand(-1, -1, heads_per_group, -1)
        xv_expanded = xv.unsqueeze(2).expand(-1, -1, heads_per_group, -1)

        # Scaled dot-product attention: [..., n_kv_heads, heads_per_group, head_dim] x [..., n_kv_heads, head_dim, 1]
        scores = torch.matmul(xq_grouped, xk_expanded.transpose(-2, -1)) / (
            self.head_dim**0.5
        )
        attn = F.softmax(scores, dim=-1)

        output = torch.matmul(attn, xv_expanded)
        output = output.reshape(-1, self.n_local_heads * self.head_dim)  # Flatten back
        output = self.attn_sub_norm(output)
        output = self.wo(output)

        return output


# @torch.compile
def squared_relu(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x) ** 2


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        norm_eps: float,
        use_kernel: bool,
    ):
        super().__init__()

        Linear = BitLinearMetal if use_kernel else BitLinear

        self.w13 = Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )
        self.w2 = Linear(
            hidden_dim,
            dim,
            bias=False,
        )
        self.ffn_sub_norm = nn.RMSNorm(hidden_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x13 = self.w13(x)
        x1, x3 = x13.chunk(2, -1)
        inner = self.ffn_sub_norm(squared_relu(x1) * x3)
        output = self.w2(inner)
        return output


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.dim % args.n_heads == 0
        head_dim = args.dim // args.n_heads
        if args.n_kv_heads is not None:
            n_kv_heads = args.n_kv_heads
        else:
            n_kv_heads = args.n_heads

        assert args.n_heads % n_kv_heads == 0

        # Create attention layer
        self.attention = Attention(
            dim=args.dim,
            head_dim=head_dim,
            n_heads=args.n_heads,
            n_kv_heads=n_kv_heads,
            rope_theta=args.rope_theta,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.ffn_dim,
            norm_eps=args.norm_eps,
            use_kernel=args.use_kernel,
        )
        self.attention_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, cache: LayerCache) -> torch.Tensor:
        h = x + self.attention.forward(self.attention_norm(x), cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size > 0

        self.tok_embeddings = nn.Embedding(
            num_embeddings=args.vocab_size,
            embedding_dim=args.dim,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

        self.norm = nn.RMSNorm(args.dim, eps=args.norm_eps)

        self.output = nn.Linear(
            args.dim,
            args.vocab_size,
            bias=False,
        )

    @torch.no_grad()
    def forward(
        self,
        token_values: torch.Tensor,
        cache: list[LayerCache],
    ) -> torch.Tensor:
        h = self.tok_embeddings(token_values)

        # Flatten batch and sequence dimensions for processing
        batch_size, seq_len, dim = h.shape
        h = h.reshape(-1, dim)  # [batch*seq, dim]

        for i, layer in enumerate(self.layers):
            h = layer(h, cache[i])

        # Reshape back to [batch, seq, vocab_size]
        h = h.reshape(batch_size, seq_len, -1)
        logits = self.output(self.norm(h))
        return logits.float()


def make_cache(
    args: ModelArgs,
    length: int,
    device: Optional[Union[str, torch.device]] = None,
    n_layers: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> list[LayerCache]:
    """
    Allocate a cache to be used with the Transformer module.
    """
    head_dim = args.dim // args.n_heads
    n_kv_heads = args.n_kv_heads
    if n_kv_heads is None:
        n_kv_heads = args.n_heads
    n_local_kv_heads = n_kv_heads

    if n_layers is None:
        n_layers = args.n_layers

    shape = (1, length, n_local_kv_heads, 1, head_dim)
    heads_per_group = args.n_heads // n_kv_heads
    expansion = (-1, -1, -1, heads_per_group, -1)
    return [
        (
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
            torch.zeros(shape, device=device, dtype=dtype).expand(expansion),
        )
        for _ in range(n_layers)
    ]
