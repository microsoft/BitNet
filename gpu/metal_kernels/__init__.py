# Metal Backend Package
"""
BitNet Metal Backend for Apple Silicon

Provides optimized inference on Apple GPUs (M1, M2, M3 series).
"""

from .model import (
    Transformer,
    ModelArgs,
    BitLinearMetal,
    BitLinear,
    pack_weight_int8_to_int2,
    make_cache,
)

__version__ = "0.1.0"
__all__ = [
    "Transformer",
    "ModelArgs",
    "BitLinearMetal",
    "BitLinear",
    "pack_weight_int8_to_int2",
    "make_cache",
]
