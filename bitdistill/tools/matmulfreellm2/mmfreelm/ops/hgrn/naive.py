# -*- coding: utf-8 -*-

from typing import Optional

import torch


def naive_recurrent_hgrn(
    x: torch.Tensor,
    g: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: Optional[bool] = False
) -> torch.Tensor:
    dtype = x.dtype
    x, g = map(lambda i: i.float(), (x, g))
    B, H, T, D = x.shape

    h = torch.zeros(B, H, D, dtype=torch.float, device=x.device)
    o = torch.zeros_like(x)

    final_state = None
    if initial_state is not None:
        h += initial_state.detach()

    for i in range(T):
        h = g[:, :, i] * h + x[:, :, i]
        o[:, :, i] = h

    if output_final_state:
        final_state = h
    return o.to(dtype), final_state
