from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np

if TYPE_CHECKING:
    from typing_extensions import TypeAlias
    NDArray: TypeAlias = 'np.ndarray[Any, Any]'

def permute(weights: NDArray, n_head: int, n_head_kv: int) -> NDArray:
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))
