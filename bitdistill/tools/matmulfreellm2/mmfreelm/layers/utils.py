# -*- coding: utf-8 -*-

from typing import Optional

import torch
from einops import rearrange

from mmfreelm.modules.utils import checkpoint

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None


@checkpoint
def proj_then_conv1d(
    x: torch.Tensor,
    proj_weight: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: Optional[torch.Tensor] = None,
    cache: Optional[torch.Tensor] = None
) -> torch.Tensor:
    # We do matmul and transpose BLH -> HBL at the same time
    x = rearrange(proj_weight @ rearrange(x, "b l d -> d (b l)"), "d (b l) -> b d l", l=x.shape[-2])

    if causal_conv1d_fn is None:
        raise ImportError("`causal_conv1d_fn` is not available. Please install `causal-conv1d` first.")
    if cache is None:
        x = causal_conv1d_fn(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            activation="silu",
        ).transpose(1, 2)
    else:
        assert x.shape[-1] == 1, "Only support decoding with 1 token at a time for now"
        x = x.squeeze(-1)
        x = causal_conv1d_update(
            x=x,
            weight=rearrange(conv1d_weight, "d 1 w -> d w"),
            bias=conv1d_bias,
            cache=cache,
            activation="silu",
        )
    return x
