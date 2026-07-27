from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from mmfreelm.modules import RMSNorm
from mmfreelm.utils import contiguous


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


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
@triton.jit
def _layer_norm_fwd_quant_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    RESIDUAL,  # pointer to the residual
    RESIDUAL_OUT,  # pointer to the residual
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_res_row,
    stride_res_out_row,
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL_OUT: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_y_row
    if HAS_RESIDUAL:
        RESIDUAL += row * stride_res_row
    if STORE_RESIDUAL_OUT:
        RESIDUAL_OUT += row * stride_res_out_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        residual = tl.load(RESIDUAL + cols, mask=cols <
                           N, other=0.0).to(tl.float32)
        x += residual
    if STORE_RESIDUAL_OUT:
        tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
    if not IS_RMS_NORM:
        mean = tl.sum(x, axis=0) / N
        tl.store(Mean + row, mean)
        xbar = tl.where(cols < N, x - mean, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    else:
        xbar = tl.where(cols < N, x, 0.0)
        var = tl.sum(xbar * xbar, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
    if HAS_BIAS:
        b = tl.load(B + cols, mask=mask).to(tl.float32)
    x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd

    y = x_hat * w if HAS_WEIGHT else x_hat
    if HAS_BIAS:
        y = y + b

    # Aply quantization to the output
    scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
    # Quantize and then de-quantize the tensor
    y = tl.math.round(y * scale)
    y = tl.maximum(tl.minimum(y, 127), -128) / scale

    # Write output
    tl.store(Y + cols, y, mask=mask)


def _layer_norm_fwd_quant(
    x, weight, bias, eps, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False
):
    if residual is not None:
        residual_dtype = residual.dtype
    M, N = x.shape
    assert x.stride(-1) == 1
    if residual is not None:
        assert residual.stride(-1) == 1
        assert residual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
    assert y.stride(-1) == 1
    if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
        residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
        assert residual_out.stride(-1) == 1
    else:
        residual_out = None
    mean = torch.empty((M,), dtype=torch.float32, device="cuda") if not is_rms_norm else None
    rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    with torch.cuda.device(x.device.index):
        _layer_norm_fwd_quant_kernel[(M,)](
            x,
            y,
            weight,
            bias,
            residual,
            residual_out,
            mean,
            rstd,
            x.stride(0),
            y.stride(0),
            residual.stride(0) if residual is not None else 0,
            residual_out.stride(0) if residual_out is not None else 0,
            N,
            eps,
            is_rms_norm,
            BLOCK_N,
            residual is not None,
            residual_out is not None,
            weight is not None,
            bias is not None,
        )
    # residual_out is None if residual is None and residual_dtype == input_dtype
    return y, mean, rstd, residual_out if residual_out is not None else x


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N", "HAS_DRESIDUAL", "STORE_DRESIDUAL", "IS_RMS_NORM", "HAS_BIAS"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
@triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _layer_norm_bwd_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    DW,  # pointer to the partial sum of weights gradient
    DB,  # pointer to the partial sum of biases gradient
    DRESIDUAL,
    DRESIDUAL_IN,
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_y_row,
    stride_dy_row,
    stride_dx_row,
    stride_dres_row,
    stride_dres_in_row,
    M,  # number of rows in X
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    rows_per_program,
    IS_RMS_NORM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_DRESIDUAL: tl.constexpr,
    STORE_DRESIDUAL: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    X += row_start * stride_x_row
    if HAS_DRESIDUAL:
        DRESIDUAL += row_start * stride_dres_row
    if STORE_DRESIDUAL:
        DRESIDUAL_IN += row_start * stride_dres_in_row
    DY += row_start * stride_dy_row
    DX += row_start * stride_dx_row
    if RECOMPUTE_OUTPUT:
        Y += row_start * stride_y_row
    if HAS_WEIGHT:
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        dw = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if RECOMPUTE_OUTPUT and HAS_BIAS:
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_BIAS:
        db = tl.zeros((BLOCK_N,), dtype=tl.float32)
    row_end = min((row_block_id + 1) * rows_per_program, M)
    for row in range(row_start, row_end):
        # Load data to SRAM
        x = tl.load(X + cols, mask=mask, other=0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0).to(tl.float32)
        if not IS_RMS_NORM:
            mean = tl.load(Mean + row)
        rstd = tl.load(Rstd + row)
        # Compute dx
        xhat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        xhat = tl.where(mask, xhat, 0.0)
        if RECOMPUTE_OUTPUT:
            y = xhat * w if HAS_WEIGHT else xhat
            if HAS_BIAS:
                y = y + b

            # Aply quantization to the output
            scale = 127.0 / tl.maximum(tl.max(tl.abs(y), 0), 1e-5)
            # Quantize and then de-quantize the tensor
            y = tl.math.round(y * scale)
            y = tl.maximum(tl.minimum(y, 127), -128) / scale

            tl.store(Y + cols, y, mask=mask)
        wdy = dy
        if HAS_WEIGHT:
            wdy = dy * w
            dw += dy * xhat
        if HAS_BIAS:
            db += dy
        if not IS_RMS_NORM:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            c2 = tl.sum(wdy, axis=0) / N
            dx = (wdy - (xhat * c1 + c2)) * rstd
        else:
            c1 = tl.sum(xhat * wdy, axis=0) / N
            dx = (wdy - xhat * c1) * rstd
        if HAS_DRESIDUAL:
            dres = tl.load(DRESIDUAL + cols, mask=mask, other=0).to(tl.float32)
            dx += dres
        # Write dx
        if STORE_DRESIDUAL:
            tl.store(DRESIDUAL_IN + cols, dx, mask=mask)
        tl.store(DX + cols, dx, mask=mask)

        X += stride_x_row
        if HAS_DRESIDUAL:
            DRESIDUAL += stride_dres_row
        if STORE_DRESIDUAL:
            DRESIDUAL_IN += stride_dres_in_row
        if RECOMPUTE_OUTPUT:
            Y += stride_y_row
        DY += stride_dy_row
        DX += stride_dx_row
    if HAS_WEIGHT:
        tl.store(DW + row_block_id * N + cols, dw, mask=mask)
    if HAS_BIAS:
        tl.store(DB + row_block_id * N + cols, db, mask=mask)


def _layer_norm_bwd(
    dy,
    x,
    weight,
    bias,
    eps,
    mean,
    rstd,
    dresidual=None,
    has_residual=False,
    is_rms_norm=False,
    x_dtype=None,
    recompute_output=False,
):
    M, N = x.shape
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    assert dy.shape == (M, N)
    if dresidual is not None:
        assert dresidual.stride(-1) == 1
        assert dresidual.shape == (M, N)
    if weight is not None:
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)
    # allocate output
    dx = (
        torch.empty_like(x)
        if x_dtype is None
        else torch.empty(M, N, dtype=x_dtype, device=x.device)
    )
    dresidual_in = torch.empty_like(
        x) if has_residual and dx.dtype != x.dtype else None
    y = torch.empty(M, N, dtype=dy.dtype,
                    device=dy.device) if recompute_output else None

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    _dw = (
        torch.empty((sm_count, N), dtype=torch.float32, device=weight.device)
        if weight is not None
        else None
    )
    _db = (
        torch.empty((sm_count, N), dtype=torch.float32, device=bias.device)
        if bias is not None
        else None
    )
    rows_per_program = math.ceil(M / sm_count)
    grid = (sm_count,)
    with torch.cuda.device(x.device.index):
        _layer_norm_bwd_kernel[grid](
            x,
            weight,
            bias,
            y,
            dy,
            dx,
            _dw,
            _db,
            dresidual,
            dresidual_in,
            mean,
            rstd,
            x.stride(0),
            0 if not recompute_output else y.stride(0),
            dy.stride(0),
            dx.stride(0),
            dresidual.stride(0) if dresidual is not None else 0,
            dresidual_in.stride(0) if dresidual_in is not None else 0,
            M,
            N,
            eps,
            rows_per_program,
            is_rms_norm,
            BLOCK_N,
            dresidual is not None,
            dresidual_in is not None,
            weight is not None,
            bias is not None,
        )
    dw = _dw.sum(0).to(weight.dtype) if weight is not None else None
    db = _db.sum(0).to(bias.dtype) if bias is not None else None
    # Don't need to compute dresidual_in separately in this case
    if has_residual and dx.dtype == x.dtype:
        dresidual_in = dx
    return (dx, dw, db, dresidual_in) if not recompute_output else (dx, dw, db, dresidual_in, y)


class LayerNormLinearQuantFn(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(
        ctx,
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual=None,
        eps=1e-6,
        prenorm=False,
        residual_in_fp32=False,
        is_rms_norm=False,
    ):
        x_shape_og = x.shape
        # reshape input data into 2D tensor
        x = x.reshape(-1, x.shape[-1])
        if residual is not None:
            assert residual.shape == x_shape_og
            residual = residual.reshape(-1, residual.shape[-1])
        residual_dtype = (
            residual.dtype
            if residual is not None
            else (torch.float32 if residual_in_fp32 else None)
        )
        y, mean, rstd, residual_out = _layer_norm_fwd_quant(
            x,
            norm_weight,
            norm_bias,
            eps,
            residual,
            out_dtype=None if not torch.is_autocast_enabled() else torch.get_autocast_gpu_dtype(),
            residual_dtype=residual_dtype,
            is_rms_norm=is_rms_norm,
        )
        y = y.reshape(x_shape_og)
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else y.dtype
        linear_weight = weight_quant(linear_weight).to(dtype)
        linear_bias = linear_bias.to(dtype) if linear_bias is not None else None
        out = F.linear(y.to(linear_weight.dtype), linear_weight, linear_bias)
        # We don't store y, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(residual_out, norm_weight, norm_bias, linear_weight, mean, rstd)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.is_rms_norm = is_rms_norm
        ctx.has_residual = residual is not None
        ctx.prenorm = prenorm
        ctx.x_dtype = x.dtype
        ctx.linear_bias_is_none = linear_bias is None
        return out if not prenorm else (out, residual_out.reshape(x_shape_og))

    @staticmethod
    @contiguous
    def backward(ctx, dout, *args):
        x, norm_weight, norm_bias, linear_weight, mean, rstd = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dy = F.linear(dout, linear_weight.t())
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        assert dy.shape == x.shape
        if ctx.prenorm:
            dresidual = args[0]
            dresidual = dresidual.reshape(-1, dresidual.shape[-1])
            assert dresidual.shape == x.shape
        else:
            dresidual = None
        dx, dnorm_weight, dnorm_bias, dresidual_in, y = _layer_norm_bwd(
            dy,
            x,
            norm_weight,
            norm_bias,
            ctx.eps,
            mean,
            rstd,
            dresidual,
            ctx.has_residual,
            ctx.is_rms_norm,
            x_dtype=ctx.x_dtype,
            recompute_output=True
        )
        dlinear_weight = torch.einsum("bo,bi->oi", dout, y)
        return (
            dx.reshape(ctx.x_shape_og),
            dnorm_weight,
            dnorm_bias,
            dlinear_weight,
            dlinear_bias,
            dresidual_in.reshape(ctx.x_shape_og) if ctx.has_residual else None,
            None,
            None,
            None,
            None,
        )


def layer_norm_linear_quant_fn(
    x,
    norm_weight,
    norm_bias,
    linear_weight,
    linear_bias,
    residual=None,
    eps=1e-6,
    prenorm=False,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    return LayerNormLinearQuantFn.apply(
        x,
        norm_weight,
        norm_bias,
        linear_weight,
        linear_bias,
        residual,
        eps,
        prenorm,
        residual_in_fp32,
        is_rms_norm,
    )


class BitLinear(nn.Linear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features, out_features, lambda_, should_rms=False, bias=False):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        # Initialize the superclass nn.Linear with the given parameters
        super(BitLinear, self).__init__(in_features, out_features, bias=bias)
        self.lambda_ = lambda_
        self.should_rms = should_rms
        if self.should_rms:
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
        x_norm = x
        if self.should_rms:
            x_norm = self.norm(x)
        
        # Apply quantization to both activations and weights
        # Uses Straight-Through Estimator (STE) trick with .detach() for gradient flow
        x_quant = x_norm + self.lambda_() * (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + self.lambda_() * (weight_quant(w) - w).detach()
        # Perform linear operation with quantized values
        y = F.linear(x_quant, w_quant)

        return y


class FusedBitLinear(BitLinear):
    """
    A custom linear layer that applies quantization on both activations and weights.
    This is primarily for training; kernel optimization is needed for efficiency in deployment.
    """

    def __init__(self, in_features, out_features, bias=False):
        """
        Initializes the BitLinear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias. Default: True.
        """
        # Initialize the superclass nn.Linear with the given parameters
        super(FusedBitLinear, self).__init__(in_features, out_features, bias=bias)

    def forward(self, x):
        return layer_norm_linear_quant_fn(
            x,
            self.norm.weight,
            self.norm.bias,
            self.weight,
            self.bias,
            is_rms_norm=True
        )
