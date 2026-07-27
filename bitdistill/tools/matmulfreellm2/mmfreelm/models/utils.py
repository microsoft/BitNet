# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


class RecurrentCache(Cache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    It stores the states of each layer as the tensor of shape `[batch_size, key_dim, value_dim]`.
    """

    def __init__(
        self,
        seen_tokens: int = 0
    ) -> RecurrentCache:

        self.states: List[torch.Tensor] = []
        self._seen_tokens = seen_tokens  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> torch.Tensor:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states:
            yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        state: Tuple[torch.Tensor],
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor]:
        """
        Updates the cache with the new `state` for the layer `layer_idx`.

        Parameters:
            state (`Tuple[torch.Tensor]`):
                The new state to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            The updated state.
        """

        if isinstance(state, torch.Tensor):
            state = (state,)
        if len(self.states) <= layer_idx:
            self.states.append(state)
        else:
            for i, s in enumerate(state):
                self.states[layer_idx][i].copy_(s)
            # update the number of seen tokens once we achieve the last layer
            if layer_idx == len(self) - 1:
                self._seen_tokens += 1

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._seen_tokens

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. RecurrentCache does not have a maximum length."""
        return None

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.states)):
            device = self.states[layer_idx].device
            self.states[layer_idx] = self.states[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[torch.Tensor]:
        return tuple(self.states)

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        seen_tokens: int = 0
    ) -> RecurrentCache:
        """Converts a cache in the legacy cache format into an equivalent `RecurrentCache`."""

        cache = cls(seen_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                cache.update(past_key_values[layer_idx], layer_idx)
        return cache
