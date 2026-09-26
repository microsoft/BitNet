"""Torch fallback for int2 prefill (no fp16 model in VRAM).

Upstream's GPU flow keeps TWO models resident: fp16 (prefill, 5.2GB) and
int2 (decode, 1.75GB). The fp16 weights are already ternary-rounded, so an
int2 prefill is mathematically equivalent. This module inverts the W2A8
packing (see pack_weight.py: permute 16x32 -> compress 4xint2 -> interleave
bits) to recover ternary weights in torch, then runs the same
int8-rounded-activation GEMM as BitLinear.

Bit layout recap (all verified against pack_weight.py):
- packed tensor: (N, K//4) int8. Flat C-order -> int32 words, each word
  holds 16 2-bit slots. interleave moved slot `offset` (0..15) to
  `shift = (offset % 4) * 8 + (offset // 4) * 2` within its word.
- after un-interleave: int8 bytes, byte j = v0 | v1<<2 | v2<<4 | v3<<6.
- permuted (N, K) 2-bit values: flat position
  p = ((bi*NBj + bj) * 16 + ii) * 32 + jj  (bi: N//16 block, etc.)
  holds logical weight [bi*16 + map[ii,jj,0], bj*32 + map[ii,jj,1]],
  with map from B_global_16x32_to_shared_load_16x32_layout.
- logical values are +2 biased: {-1,0,+1} stored as {1,2,3}.
- scales: weight_scale[:G] with G row-groups (wqkv: [n_h*hd, n_kv*hd,
  n_kv*hd]; w13: [ffn, ffn]; w2/wo: single group).
"""

import numpy as np
import torch

_PERM_CACHE = {}
_INV_DEV_CACHE = {}


def _inverse_perm_on_device(N, K, device):
    """Flat gather indices as a CUDA tensor.

    Built once per (N, K, device) OUTSIDE graph capture: creating it via
    .to(device) inside a CUDA-graph replay is illegal (CPU->CUDA copy).
    """
    key = (N, K, str(device))
    t = _INV_DEV_CACHE.get(key)
    if t is None:
        t = torch.from_numpy(_build_inverse_perm(N, K).reshape(-1)).to(device)
        _INV_DEV_CACHE[key] = t
    return t


def _build_inverse_perm(N, K):
    """Flat gather indices: logical_flat = permuted_flat[inv]."""
    key = (N, K)
    if key in _PERM_CACHE:
        return _PERM_CACHE[key]
    NBj = K // 32
    fwd = np.zeros((N, K), dtype=np.int64)  # permuted pos -> logical pos
    for bi in range(N // 16):
        for bj in range(K // 32):
            for ii in range(16):
                for jj in range(32):
                    thread_id = ii * 2 + jj // 16
                    row = (thread_id // 16) * 8 + (thread_id % 8)
                    col = (jj % 16) + 16 * ((thread_id % 16) // 8)
                    p = ((bi * NBj + bj) * 16 + ii) * 32 + jj
                    fwd.flat[p] = (bi * 16 + row) * K + (bj * 32 + col)
    inv = np.zeros((N, K), dtype=np.int64)
    inv.flat[fwd.flat[:]] = np.arange(N * K)
    _PERM_CACHE[key] = inv
    return inv


def uninterleave_int2(packed_i8):
    """(..., M) int8 -> (..., M) int8 with the interleave bit-shuffle undone.

    Operates on int32 words: out_word = sum_o(((w >> shift(o)) & 3) << 2o).
    """
    flat = packed_i8.reshape(-1)
    assert flat.numel() % 4 == 0
    w = flat.view(torch.int32)
    shifts = [(o % 4) * 8 + (o // 4) * 2 for o in range(16)]
    out = torch.zeros_like(w)  # stays int32: all ops below are int32
    for o in range(16):
        out = out | (((w >> shifts[o]) & 3) << (2 * o))
    return out.view(torch.int8).reshape(packed_i8.shape)


def decompress_int2_to_vals(comp):
    """(..., M) int8 -> (..., 4M) int8 2-bit values (still +2 biased)."""
    c = comp.to(torch.int64)
    v0 = c & 3
    v1 = (c >> 2) & 3
    v2 = (c >> 4) & 3
    v3 = (c >> 6) & 3
    return torch.stack([v0, v1, v2, v3], dim=-1).reshape(
        comp.shape[:-1] + (comp.shape[-1] * 4,)).to(torch.int8)


@torch.compile
def dequantize_int2_weight(packed, scales, row_splits=None, out_dtype=torch.bfloat16):
    """packed: (N, K//4) int8 tensor. scales: (4,) bf16. Returns (N, K) fp.

    row_splits: list of row counts per scale group, e.g. wqkv ->
    [n_heads*hd, n_kv*hd, n_kv*hd]. None = single group (scales[0]).
    """
    N, K4 = packed.shape
    K = K4 * 4
    dev = packed.device
    flat = uninterleave_int2(packed.reshape(-1)).reshape(N, K4)
    vals = decompress_int2_to_vals(flat).reshape(N, K)  # +2 biased
    inv_t = _inverse_perm_on_device(N, K, dev)
    logical = vals.reshape(-1)[inv_t].reshape(N, K).to(torch.float32)
    ternary = logical - 2.0
    if row_splits is None:
        s = scales[0].to(torch.float32)
        return (ternary * s).to(out_dtype)
    assert sum(row_splits) == N, (row_splits, N)
    out = torch.empty((N, K), dtype=torch.float32, device=dev)
    r0 = 0
    for g, nr in enumerate(row_splits):
        out[r0:r0 + nr] = ternary[r0:r0 + nr] * scales[g].to(torch.float32)
        r0 += nr
    return out.to(out_dtype)
