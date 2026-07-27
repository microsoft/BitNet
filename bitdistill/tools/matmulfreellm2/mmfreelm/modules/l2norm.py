# -*- coding: utf-8 -*-
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_RESIDUAL": lambda args: args["RESIDUAL"] is not None})
@triton.jit
def _l2_norm_fwd_1pass_kernel(
    X,  # pointer to the input
    Y,  # pointer to the output
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    Y += row * stride_x_row
    # Compute mean and variance
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    xbar = tl.where(cols < N, x, 0.0)
    var = tl.sum(xbar * xbar, axis=0) 
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    y = x * rstd
    # Write output
    tl.store(Y + cols, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
        triton.Config({}, num_warps=32),
    ],
    key=["N"],
)
# @triton.heuristics({"HAS_BIAS": lambda args: args["B"] is not None})
# @triton.heuristics({"HAS_DRESIDUAL": lambda args: args["DRESIDUAL"] is not None})
# @triton.heuristics({"STORE_DRESIDUAL": lambda args: args["DRESIDUAL_IN"] is not None})
# @triton.heuristics({"RECOMPUTE_OUTPUT": lambda args: args["Y"] is not None})
@triton.jit
def _l2_norm_bwd_kernel(
    X,  # pointer to the input
    # Y,  # pointer to the output to be recomputed
    DY,  # pointer to the output gradient
    DX,  # pointer to the input gradient
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    X += row * stride_x_row
    DX += row * stride_x_row
    DY += row * stride_x_row

    # Y += row * stride_y_row
    cols = tl.arange(0, BLOCK_N)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    x = tl.where(cols < N, x, 0.0)
    var = tl.sum(x * x) 
    rstd = 1 / tl.sqrt(var + eps)
    # tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    mask = cols < N
    # y = x * rstd
    dy = tl.load(DY + cols, mask=cols < N, other=0.0).to(tl.float32)
    dy = tl.where(cols < N, dy, 0.0)
    # dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x 
    dx = dy * rstd - tl.sum(dy * x) * (1 / (var+eps)) * rstd * x
    tl.store(DX + cols, dx, mask=mask)

def _l2_norm_fwd(
    x, eps=1e-6
):
    x_shape_og = x.shape
    x = x.reshape(-1, x.shape[-1])
    if x.stride(-1) != 1:
        x = x.contiguous()
        M, N = x.shape
    assert x.stride(-1) == 1 
    # allocate output
    y = torch.empty_like(x)
    assert y.stride(-1) == 1
    N = x.shape[-1]
    M = x.shape[0]
    # rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    with torch.cuda.device(x.device.index):
        _l2_norm_fwd_1pass_kernel[(M,)](
            x,
            y,
            x.stride(0),
            N,
            eps,
            # is_rms_norm,
            BLOCK_N,
            # residual is not None,
            # residual_out is not None,
            # bias is not None,
        )
    return y.reshape(x_shape_og)

def _l2_norm_bwd(
    x, dy, eps=1e-5,
):
    x_shape_og = x.shape
    x = x.reshape(-1, dy.shape[-1])
    dy = dy.reshape(-1, dy.shape[-1])
    if dy.stride(-1) != 1:
        dy = dy.contiguous()
    assert dy.shape == x.shape
    # allocate output
    dx = torch.empty_like(x)
    N = x.shape[-1]
    M = x.shape[0]
    assert x.stride(-1) == 1
    assert dy.stride(-1) == 1
    # rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    if N > BLOCK_N:
        raise RuntimeError(
            "This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of warps
    with torch.cuda.device(x.device.index):
        _l2_norm_bwd_kernel[(M,)](
            x,
            dy,
            dx,
            x.stride(0),
            N,
            eps,
            BLOCK_N,
        )
    return dx.reshape(x_shape_og)


class L2NormFN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        eps=1e-6,
    ):
        # reshape input data into 2D tensor
        y = _l2_norm_fwd(x, eps)
        ctx.x_shape_og = x_shape_og
        ctx.eps = eps
        ctx.x_dtype = x.dtype
        ctx.save_for_backward(x)
        return y 

    @staticmethod
    def backward(ctx, dy, *args):
        x, = ctx.saved_tensors
        dx = _l2_norm_bwd(
            x,
            dy,
            ctx.eps,
        )
        return (
            dx,
            None
        )

l2_norm_fn = L2NormFN.apply

if __name__ == '__main__':
    x = torch.rand(10, 10, 100).cuda().requires_grad_(True)
    y = torch.nn.functional.normalize(x, dim=-1, p=2)
    dy = torch.rand_like(y)
    y.backward(dy, retain_graph=True)
    x_grad, x.grad = x.grad, None
    y2 = l2_norm_fn(x, 1e-6)
    print((y-y2).abs().max())
    y2.backward(dy, retain_graph=True)
    x_grad2, x.grad = x.grad, None
    print((x_grad2-x_grad).abs().max())
    breakpoint()    
    

    
    
