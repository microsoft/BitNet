"""Regression coverage for Llama-family I2_S checkpoint conversion."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils"))
spec = importlib.util.spec_from_file_location(
    "bitnet_converter", ROOT / "utils/convert-hf-to-gguf-bitnet.py"
)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class RecordingWriter:
    def __init__(self):
        self.tensors = {}

    def add_tensor(self, name, data, **kwargs):
        self.tensors[name] = (data.copy(), kwargs)


def run_conversion(ftype, data, scale, name="model.layers.0.mlp.down_proj.weight"):
    model = object.__new__(converter.LlamaModel)
    model.hparams = {"num_attention_heads": 2, "num_hidden_layers": 1}
    model.block_count = 1
    model.tensor_map = converter.gguf.get_tensor_name_map(model.model_arch, 1)
    model.gguf_writer = RecordingWriter()
    model.ftype = ftype
    tensors = [(name, data)]
    if scale is not None:
        tensors.append((name.replace(".weight", ".weight_scale"), scale))
    model.get_tensors = lambda: iter(tensors)
    converter.args = SimpleNamespace(quant_embd=False)
    model.write_tensors()
    return next(iter(model.gguf_writer.tensors.values()))


@pytest.mark.parametrize(
    "name",
    [
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
    ],
)
def test_llama_i2_s_preserves_ternary_weights_and_reciprocal_scale(name):
    # Falcon's packed columns hold four groups of 2-bit values per input row.
    packed = torch.tensor(
        np.resize(np.array([0x24, 0x99, 0x66], dtype=np.uint8), (32, 128))
    )
    scale = torch.tensor([4.0])
    f16, _ = run_conversion(
        converter.gguf.GGMLQuantizationType.F16, packed, scale, name
    )
    data, info = run_conversion(
        converter.gguf.GGMLQuantizationType.I2_S, packed, scale, name
    )

    assert info["raw_dtype"] == converter.gguf.GGMLQuantizationType.I2_S
    blocks = data[:4096].reshape(-1, 32)
    values = np.stack(
        [(blocks >> shift) & 3 for shift in (6, 4, 2, 0)], axis=1
    ).reshape(128, 128)
    recovered_scale = np.frombuffer(data[4096:4100].tobytes(), dtype=np.float32)[0]
    np.testing.assert_array_equal((values.astype(np.int16) - 1) * recovered_scale, f16)


@pytest.mark.parametrize("multiplier", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_offline_scale_is_rejected(multiplier):
    with pytest.raises(ValueError, match="Invalid weight scale"):
        run_conversion(
            converter.gguf.GGMLQuantizationType.I2_S,
            torch.zeros((32, 128), dtype=torch.uint8),
            torch.tensor([multiplier]),
        )


def test_float_weights_use_absmean_quantization():
    weights = torch.tensor(
        np.resize(np.array([-2.0, -0.1, 0.0, 2.0], dtype=np.float32), (128, 128))
    )
    data, info = run_conversion(converter.gguf.GGMLQuantizationType.I2_S, weights, None)
    assert info["raw_dtype"] == converter.gguf.GGMLQuantizationType.I2_S
    blocks = data[:4096].reshape(-1, 32)
    values = np.stack(
        [(blocks >> shift) & 3 for shift in (6, 4, 2, 0)], axis=1
    ).reshape(128, 128)
    recovered_scale = np.frombuffer(data[4096:4100].tobytes(), dtype=np.float32)[0]
    expected_scale = weights.abs().mean().item()
    assert recovered_scale == pytest.approx(expected_scale)
    np.testing.assert_array_equal(
        values.astype(np.int16) - 1,
        np.clip(np.round(weights.numpy() / expected_scale), -1, 1),
    )


def test_i2_s_rejects_non_block_aligned_tensors():
    with pytest.raises(ValueError, match="row width divisible by 128"):
        run_conversion(
            converter.gguf.GGMLQuantizationType.I2_S, torch.ones((4, 32)), None
        )


def test_i2_s_rejects_invalid_packed_ternary_values():
    with pytest.raises(ValueError, match="Invalid ternary weight values"):
        run_conversion(
            converter.gguf.GGMLQuantizationType.I2_S,
            torch.full((32, 128), 0xFF, dtype=torch.uint8),
            torch.tensor([4.0]),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_i2_s_rejects_nonfinite_float_weights(value):
    with pytest.raises(ValueError, match="Non-finite weights"):
        run_conversion(
            converter.gguf.GGMLQuantizationType.I2_S,
            torch.full((128, 128), value),
            None,
        )
