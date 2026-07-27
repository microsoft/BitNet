# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from mmfreelm.modules.layernorm import layer_norm_fn
from mmfreelm.modules.utils import checkpoint


@checkpoint
def flatten_diag_outer_product(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N)
    return z[..., indicies[0], indicies[1]]


@checkpoint
def flatten_diag_outer_product_off1(x, y):
    z = torch.einsum("...i,...j->...ij", x, y)
    N = z.size(-1)
    indicies = torch.triu_indices(N, N, 1)
    indices2 = torch.arange(0, N)
    return z[..., indicies[0], indicies[1]], z[..., indices2, indices2]


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


class HedgehogFeatureMap(nn.Module):

    r"""
    Hedgehog feature map as introduced in
    `The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry <https://arxiv.org/abs/2402.04347>`_
    """

    def __init__(
        self,
        head_dim: int
    ) -> HedgehogFeatureMap:
        super().__init__()
        # Trainable map
        self.layer = nn.Linear(head_dim, head_dim)
        self.init_weights_()

    def init_weights_(self):
        """Initialize trainable map as identity"""
        with torch.no_grad():
            identity = torch.eye(*self.layer.weight.shape[-2:], dtype=torch.float)
            self.layer.weight.copy_(identity.to(self.layer.weight))
        nn.init.zeros_(self.layer.bias)

    def forward(self, x: torch.Tensor):
        x = self.layer(x)  # shape b, h, l, d
        return torch.cat([2*x, -2*x], dim=-1).softmax(-1)


class T2RFeatureMap(nn.Module):

    r"""
    Simple linear mapping feature map as in
    `Finetuning Pretrained Transformers into RNNs <https://arxiv.org/abs/2103.13076>`_
    """

    def __init__(
        self,
        head_dim: int,
        dot_dim: int = None
    ) -> T2RFeatureMap:
        super().__init__()
        # Trainable map
        if dot_dim is None:
            dot_dim = head_dim
        self.layer = nn.Linear(head_dim, dot_dim)

    def forward(self, x: torch.Tensor):
        return self.layer(x).relu()


class DPFPFeatureMap(nn.Module):

    r"""
    Deterministic Parameter-Free Projection (DPFP) feature map in
    `Linear Transformers Are Secretly Fast Weight Programmers <https://arxiv.org/abs/2102.11174>`_
    """

    def __init__(
        self,
        head_dim: int,
        nu: int = 4
    ) -> DPFPFeatureMap:
        super().__init__()
        self.nu = nu

    def forward(self, x: torch.Tensor):
        x = torch.cat([x.relu(), -x.relu()], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1, self.nu+1)], dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled


class HadamardFeatureMap(nn.Module):
    def __init__(
        self,
        head_dim: int
    ) -> HadamardFeatureMap:
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, head_dim)
        self.layer2 = nn.Linear(head_dim, head_dim)

    def forward(self, x: torch.Tensor):
        return self.layer1(x) * self.layer2(x)


class LearnableOuterProductFeatureMap(nn.Module):
    def __init__(
        self,
        head_dim: int,
        feature_dim: int
    ) -> LearnableOuterProductFeatureMap:
        super().__init__()
        # Trainable map
        self.layer1 = nn.Linear(head_dim, feature_dim, bias=False)
        self.layer2 = nn.Linear(head_dim, feature_dim, bias=False)
        self.normalizer = feature_dim ** -0.5

    def forward(self, x: torch.Tensor):
        return flatten_diag_outer_product(self.layer1(x), self.layer2(x))


class LearnablePolySketchNonNegativeFeatureMap(nn.Module):

    def __init__(
        self,
        head_dim: int,
        sketch_size: Optional[int] = None,
        degree: Optional[int] = 2
    ) -> LearnablePolySketchNonNegativeFeatureMap:
        super().__init__()

        assert is_power_of_2(degree) and degree >= 2, f"The degree {degree} must be a power of 2"

        self.head_dim = head_dim
        self.sketch_size = sketch_size if sketch_size is not None else head_dim
        self.degree = degree

        self.gamma = nn.Parameter(torch.ones(head_dim))
        self.beta = nn.Parameter(torch.zeros(head_dim))
        # NOTE: the sketch layers defined here are quite different from the original paper
        # currently we simply use linear layers without any non-linear activations
        self.sketches1 = nn.ModuleList([
            nn.Linear(head_dim, sketch_size, bias=False),
            *[nn.Linear(sketch_size, sketch_size, bias=False) for _ in range(int(math.log2(self.degree)) - 2)]
        ])
        self.sketches2 = nn.ModuleList([
            nn.Linear(head_dim, sketch_size, bias=False),
            *[nn.Linear(sketch_size, sketch_size, bias=False) for _ in range(int(math.log2(self.degree)) - 2)]
        ])

    def forward(self, x: torch.Tensor):
        # Section 2.1
        x = layer_norm_fn(x, self.gamma, self.beta)
        # first map the input to sketch size with learnable parameters
        x = self.sketches1[0](x) * self.sketches2[0](x) * self.head_dim ** -0.5
        for i in range(1, int(math.log2(self.degree)) - 1):
            x = self.sketches1[i](x) * self.sketches2[i](x) * self.head_dim ** -0.5
        # do sketch mapping for log2(p) - 1 times in total
        # do p=2 mapping to ensure non-negativity
        return flatten_diag_outer_product(x, x)


class TaylorFeatureMap(nn.Module):
    def __init__(
        self,
        head_dim: int
    ) -> TaylorFeatureMap:
        super().__init__()
        self.head_dim = head_dim
        self.r2 = math.sqrt(2)
        self.rd = math.sqrt(self.head_dim)
        self.rrd = math.sqrt(self.rd)

    def forward(self, x: torch.Tensor):
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        return torch.cat([torch.ones_like(x[..., 0:1]), x / self.rrd, x2_2 / (self.rd * self.r2), x2_1 / self.rd], dim=-1)


class RebasedFeatureMap(nn.Module):

    def __init__(
        self,
        head_dim: int,
        use_gamma: Optional[bool] = True,
        use_beta: Optional[bool] = True,
        normalize: Optional[bool] = True
    ) -> RebasedFeatureMap:
        super().__init__()

        self.head_dim = head_dim
        self.use_gamma = use_gamma
        self.use_beta = use_beta
        self.normalize = normalize

        self.gamma = None
        self.beta = None
        if use_gamma:
            self.gamma = nn.Parameter(torch.ones(head_dim))
        if use_beta:
            self.beta = nn.Parameter(torch.zeros(head_dim))

    def forward(self, x: torch.Tensor, flatten: Optional[bool] = True):
        if self.use_beta and self.use_gamma and self.normalize:
            x = layer_norm_fn(x, self.gamma, self.beta)
        elif self.normalize:
            x = F.layer_norm(x, (self.head_dim,), self.gamma, self.beta)
        elif self.use_gamma and self.use_beta:
            x = torch.addcmul(self.beta, x, self.gamma)
        elif self.use_gamma:
            x = x.mul(self.gamma)
        else:
            raise RuntimeError(f"Not supported combination of `use_gamma`, `use_beta` and `normalize`, "
                               f"which is currentlt set as (`{self.use_gamma}`, `{self.use_beta}`, `{self.normalize}`)")
        if not flatten:
            return x
        x2_1, x2_2 = flatten_diag_outer_product_off1(x, x)
        # rebased use learnable parameters to approximate any quadratic function
        return torch.cat([x2_2 * self.head_dim ** -0.5, x2_1 * (2 / self.head_dim) ** 0.5], dim=-1)
