# -*- coding: utf-8 -*-

# Copyright (c) 2023, Songlin Yang

from typing import Tuple

import torch
import triton
import triton.language as tl

from mmfreelm.utils import contiguous


@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def fused_recurrent_hgrn_fwd_kernel(
    x,
    g,
    o,
    h0,
    ht,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_x = x + i_bh * T * D + o_d
    p_g = g + i_bh * T * D + o_d
    p_o = o + i_bh * T * D + o_d

    b_h = tl.zeros([BD], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_bh * D + o_d
        b_h += tl.load(p_h0, mask=mask, other=0).to(tl.float32)
    for _ in range(0, T):
        b_x = tl.load(p_x, mask=mask, other=0).to(tl.float32)
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_h = b_g * b_h + b_x
        tl.store(p_o, b_h.to(p_o.dtype.element_ty), mask=mask)

        p_x += D
        p_g += D
        p_o += D

    if STORE_FINAL_STATE:
        p_ht = ht + i_bh * D + o_d
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BD': 32}, num_warps=1),
        triton.Config({'BD': 32}, num_warps=2),
        triton.Config({'BD': 32}, num_warps=4),
        triton.Config({'BD': 32}, num_warps=8),
        triton.Config({'BD': 64}, num_warps=1),
        triton.Config({'BD': 64}, num_warps=2),
        triton.Config({'BD': 64}, num_warps=4),
        triton.Config({'BD': 64}, num_warps=8),
        triton.Config({'BD': 128}, num_warps=1),
        triton.Config({'BD': 128}, num_warps=2),
        triton.Config({'BD': 128}, num_warps=4),
        triton.Config({'BD': 128}, num_warps=8),
    ],
    key=['D']
)
@triton.jit
def fused_recurrent_hgrn_bwd_kernel(
    g,
    o,
    dx,
    dg,
    do,
    h0,
    T: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr
):
    i_d, i_bh = tl.program_id(0), tl.program_id(1)
    o_d = i_d * BD + tl.arange(0, BD)
    mask = o_d < D

    p_g = g + (i_bh * T + T - 1) * D + o_d
    p_o = o + (i_bh * T + T - 2) * D + o_d
    p_dx = dx + (i_bh * T + T - 1) * D + o_d
    p_dg = dg + (i_bh * T + T - 1) * D + o_d
    p_do = do + (i_bh * T + T - 1) * D + o_d

    b_dh = tl.zeros([BD], dtype=tl.float32)
    for i in range(T - 1, -1, -1):
        b_g = tl.load(p_g, mask=mask, other=0).to(tl.float32)
        b_do = tl.load(p_do, mask=mask, other=0).to(tl.float32)
        if i > 0:
            b_o = tl.load(p_o, mask=mask, other=0).to(tl.float32)
        elif USE_INITIAL_STATE:
            b_o = tl.load(h0 + i_bh * D + o_d, mask=mask, other=0).to(tl.float32)
        else:
            b_o = tl.zeros([BD], dtype=tl.float32)

        b_dh = b_dh + b_do
        b_dx = b_dh
        b_dg = b_dh * b_o
        b_dh = b_dh * b_g
        tl.store(p_dx, b_dx.to(p_dx.dtype.element_ty), mask=mask)
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=mask)

        p_g -= D
        p_o -= D
        p_dx -= D
        p_dg -= D
        p_do -= D


class FusedRecurrentHGRNFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    def forward(ctx, x, g, initial_state=None, output_final_state=False):
        B, H, T, D = x.shape

        final_state = None
        if output_final_state:
            final_state = x.new_empty(B, H, D)

        o = torch.empty_like(x)
        def grid(meta): return (triton.cdiv(D, meta['BD']), B * H)
        fused_recurrent_hgrn_fwd_kernel[grid](
            x, g, o, initial_state, final_state,
            T, D,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None
        )
        ctx.save_for_backward(g, o, initial_state)
        return o, final_state

    @staticmethod
    @contiguous
    def backward(ctx, do, dht=None):
        g, o, initial_state = ctx.saved_tensors
        B, H, T, D = do.shape

        dx = torch.empty_like(o)
        dg = torch.empty_like(g)
        def grid(meta): return (triton.cdiv(D, meta['BD']), B * H)
        fused_recurrent_hgrn_bwd_kernel[grid](
            g, o, dx, dg, do, initial_state,
            T, D,
            USE_INITIAL_STATE=initial_state is not None,
        )

        return dx, dg, None, None


def fused_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    if initial_state is not None:
        initial_state = initial_state.detach()
    o, final_state = FusedRecurrentHGRNFunction.apply(x, g, initial_state, output_final_state)
    return o, final_state


if __name__ == '__main__':
    from mmfreelm.ops.hgrn.naive import naive_recurrent_hgrn
    B, H, T, D = 8, 4, 512, 128
    dtype = torch.bfloat16
    torch.manual_seed(42)
    # [batch_size, n_heads, seq_len, d_head]
    x = torch.randn((B, H, T, D), dtype=dtype, device='cuda')
    g = torch.randn((B, H, T, D), dtype=dtype, device='cuda').sigmoid()
    x = (1 - g) * x
    print(f'x:\t{float(x.min()):>10.6f}\t{float(x.max()):>10.6f}')
    print(f'g:\t{float(g.min()):>10.6f}\t{float(g.max()):>10.6f}')
    x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))
    print(f"DTYPE:\t{x.dtype}")
    do = torch.randn_like(x)
    h0 = torch.randn_like(x[:, :, 0])
    ref, ref_ht = naive_recurrent_hgrn(x, g, h0, output_final_state=True)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dg, g.grad = g.grad.clone(), None

    tri, tri_ht = fused_recurrent_hgrn(x, g, h0, output_final_state=True)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dg, g.grad = g.grad.clone(), None
    print("  \t    DIFF\t    MAX")
    print(' o\t', f"{float((ref - tri).abs().max()):>10.6f}\t{float(ref.max()):>10.6f}")
    print('ht\t', f"{float((ref_ht[0] - tri_ht[0]).abs().max()):>10.6f}\t{float(ref.max()):>10.6f}")
    print('dx\t', f"{float((ref_dx - tri_dx).abs().max()):>10.6f}\t{float(ref_dx.max()):>10.6f}")
    print('dg\t', f"{float((ref_dg - tri_dg).abs().max()):>10.6f}\t{float(ref_dg.max()):>10.6f}")
    print('Done!')

    @triton.testing.perf_report(
        triton.testing.Benchmark(
            # argument names to use as an x-axis for the plot
            x_names=['seq_len'],
            # different possible values for `x_name`
            x_vals=[128 * 2 ** i for i in range(0, 8)],
            # argument name whose value corresponds to a different line in the plot
            line_arg='provider',
            # possible values for `line_arg``
            line_vals=['recurrent', 'recurrent_bwd'],
            # label name for the lines
            line_names=['recurrent', 'recurrent_bwd'],
            # line styles
            styles=[('green', '-'), ('blue', '--'), ('red', '-.'), ('cyan', ':'), ('yellow', 'dotted'), ('black', 'dashed')],
            ylabel="Execution Time (ms)",  # label name for the y-axis
            # name for the plot. Used also as a file name for saving the plot.
            plot_name="Performance",
            args={},
        )
    )
    def benchmark(seq_len, provider):
        dtype = torch.bfloat16
        B, H, D = 16, 4, 128

        x = torch.randn((B, H, seq_len, D), dtype=dtype, device='cuda')
        g = torch.randn((B, H, seq_len, D), dtype=dtype, device='cuda').sigmoid()
        x = (1 - g) * x
        x, g = (i.detach().clone().to(dtype).requires_grad_() for i in (x, g))
        do = torch.randn_like(x, dtype=dtype)
        quantiles = [0.5, 0.2, 0.8]
        results = 0, 0, 0
        if provider == 'recurrent':
            results = triton.testing.do_bench(lambda: fused_recurrent_hgrn(x, g), quantiles=quantiles)
        if provider == 'recurrent_bwd':
            results = triton.testing.do_bench(lambda: fused_recurrent_hgrn(x, g)[0].backward(do), quantiles=quantiles)
        return results
    benchmark.run(print_data=True)
