"""Check the dummy generator without allocating full model tensors."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

source = Path(__file__).resolve().parents[1] / "utils/generate-dummy-bitnet-model.py"
tree = ast.parse(source.read_text())
model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BitnetModel")
method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "generate_tensors")
namespace = {"torch": SimpleNamespace(randn=lambda shape, **kwargs: shape, float32="f32"), "model_config": {"tiny": {"hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 0, "num_attention_heads": 2}}}
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module, str(source), "exec"), namespace)

class DummyVocabularyTests(unittest.TestCase):
    def test_embedding_matches_configured_vocabulary(self):
        for vocab_size in (32000, 32002, 128256):
            with self.subTest(vocab_size=vocab_size):
                model = SimpleNamespace(params="tiny", hparams={"vocab_size": vocab_size})
                name, shape = next(namespace["generate_tensors"](model))
                self.assertEqual(name, "model.embed_tokens.weight")
                self.assertEqual(shape, (vocab_size, 8))

if __name__ == "__main__":
    unittest.main()
