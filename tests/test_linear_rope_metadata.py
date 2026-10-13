import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

source = Path(__file__).resolve().parents[1] / "utils/convert-hf-to-gguf-bitnet.py"
tree = ast.parse(source.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LlamaModel")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "set_gguf_parameters")
wrapper = ast.ClassDef(name="LlamaModel", bases=[ast.Name(id="Base", ctx=ast.Load())], keywords=[], body=[method], decorator_list=[])
module = ast.Module(body=[wrapper], type_ignores=[])
ast.fix_missing_locations(module)
class Base:
    def set_gguf_parameters(self): pass
ns = {"Base": Base, "json": json, "gguf": SimpleNamespace(RopeScalingType=SimpleNamespace(LINEAR=1))}
exec(compile(module, str(source), "exec"), ns)

class LinearRopeMetadataTests(unittest.TestCase):
    def test_modern_and_legacy_linear_rope_configs(self):
        for key in ("rope_type", "type"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                obj = ns["LlamaModel"]()
                obj.dir_model = Path(directory)
                obj.gguf_writer = Mock()
                obj.hparams = {"vocab_size": 32, "hidden_size": 8, "num_attention_heads": 2, "rope_scaling": {key: "linear", "factor": 4.0}}
                obj.set_gguf_parameters()
                obj.gguf_writer.add_rope_scaling_type.assert_called_once_with(1)
                obj.gguf_writer.add_rope_scaling_factor.assert_called_once_with(4.0)

    def test_other_rope_types_are_not_labeled_linear(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = ns["LlamaModel"]()
            obj.dir_model = Path(directory)
            obj.gguf_writer = Mock()
            obj.hparams = {"vocab_size": 32, "hidden_size": 8, "num_attention_heads": 2, "rope_scaling": {"rope_type": "dynamic", "factor": 4.0}}
            obj.set_gguf_parameters()
            obj.gguf_writer.add_rope_scaling_type.assert_not_called()

if __name__ == "__main__": unittest.main()
