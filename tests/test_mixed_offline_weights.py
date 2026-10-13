"""Check per-tensor quantization dispatch without a model allocation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

source = Path(__file__).resolve().parents[1] / "utils/convert-hf-to-gguf-bitnet.py"
tree = ast.parse(source.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BitnetModel")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "modify_tensors")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
ast.fix_missing_locations(module)
ns = {}
exec(compile(module, str(source), "exec"), ns)

class MixedOfflineWeightTests(unittest.TestCase):
    def test_only_packed_tensor_skips_weight_quantization(self):
        packed = "model.layers.0.self_attn.q_proj.weight"
        floating = "model.layers.1.self_attn.q_proj.weight"
        quant = Mock(return_value="quantized")
        model = SimpleNamespace(_has_offline_quant=True, _offline_quantized_names={packed.removesuffix(".weight")}, weight_quant=quant, map_tensor_name=lambda name: name)
        self.assertEqual(ns["modify_tensors"](model, "raw", packed, 0), [(packed, "raw")])
        quant.assert_not_called()
        self.assertEqual(ns["modify_tensors"](model, "raw", floating, 1), [(floating, "quantized")])
        quant.assert_called_once_with("raw")

    def test_norms_are_not_quantized(self):
        quant = Mock()
        model = SimpleNamespace(weight_quant=quant, map_tensor_name=lambda name: name)
        name = "model.norm.weight"
        self.assertEqual(ns["modify_tensors"](model, "raw", name, None), [(name, "raw")])
        quant.assert_not_called()

if __name__ == "__main__": unittest.main()
