import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

source = Path(__file__).resolve().parents[1] / "utils/convert-hf-to-gguf-bitnet.py"
tree = ast.parse(source.read_text())
model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_get_part_names")
format_method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_is_model_safetensors")
ns = {"json": json, "Path": Path, "Model": SimpleNamespace(count_model_parts=lambda root, suffix: len(list(root.glob("*" + suffix))))}
exec(compile(ast.Module(body=[method, format_method], type_ignores=[]), str(source), "exec"), ns)

class CheckpointIndexTests(unittest.TestCase):
    def test_index_controls_shard_names_and_deduplicates(self):
        for safetensors in (True, False):
            with self.subTest(safetensors=safetensors), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                index = "model.safetensors.index.json" if safetensors else "pytorch_model.bin.index.json"
                extension = ".safetensors" if safetensors else ".bin"
                names = ["custom-a" + extension, "custom-b" + extension]
                for name in names + ["optimizer" + extension]: (root / name).touch()
                (root / index).write_text(json.dumps({"weight_map": {"a": names[0], "b": names[1], "c": names[0]}}))
                parts = ns["_get_part_names"](SimpleNamespace(dir_model=root, is_safetensors=safetensors, num_parts=3))
                self.assertEqual(list(parts), names)
                self.assertEqual(list(parts), names)

    def test_pytorch_index_selects_format_over_unrelated_safetensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "pytorch_model.bin.index.json").write_text("{}")
            (root / "optimizer.safetensors").touch()
            self.assertFalse(ns["_is_model_safetensors"](SimpleNamespace(dir_model=root)))

    def test_missing_indexed_shard_fails_before_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "missing.safetensors"}}))
            with self.assertRaises(FileNotFoundError):
                ns["_get_part_names"](SimpleNamespace(dir_model=root, is_safetensors=True, num_parts=2))

if __name__ == "__main__": unittest.main()
