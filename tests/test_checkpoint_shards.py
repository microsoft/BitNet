"""Exercise shard discovery without requiring torch or model downloads."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

source = Path(__file__).resolve().parents[1] / "utils/convert-hf-to-gguf-bitnet.py"
tree = ast.parse(source.read_text())
model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_get_part_names")
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
get_part_names = namespace["_get_part_names"]

class CheckpointShardTests(unittest.TestCase):
    def test_two_pass_safetensors(self):
        parts = get_part_names(SimpleNamespace(is_safetensors=True, num_parts=2))
        expected = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        self.assertEqual(list(parts), expected)
        self.assertEqual(list(parts), expected)

    def test_two_pass_pytorch(self):
        parts = get_part_names(SimpleNamespace(is_safetensors=False, num_parts=2))
        expected = ["pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin"]
        self.assertEqual(list(parts), expected)
        self.assertEqual(list(parts), expected)

    def test_single_part_stays_reusable(self):
        for safetensors, name in [(True, "model.safetensors"), (False, "pytorch_model.bin")]:
            parts = get_part_names(SimpleNamespace(is_safetensors=safetensors, num_parts=1))
            self.assertEqual(list(parts), [name])
            self.assertEqual(list(parts), [name])

if __name__ == "__main__":
    unittest.main()
