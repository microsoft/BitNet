import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

source = Path(__file__).resolve().parents[1] / "utils/convert-bitnet-embedding-to-gguf.py"
tree = ast.parse(source.read_text())
method = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "set_gguf_parameters")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
ast.fix_missing_locations(module)
ns = {"json": json, "logger": logging.getLogger("test"), "gguf": SimpleNamespace(PoolingType=SimpleNamespace(MEAN=1, CLS=2, LAST=3))}
exec(compile(module, str(source), "exec"), ns)

class EmbeddingPoolingTests(unittest.TestCase):
    def test_default_follows_documented_last_token_pooling(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = Mock()
            ns["set_gguf_parameters"](writer, {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 32}, Path(directory), 1)
            writer.add_pooling_type.assert_called_once_with(3)

    def test_explicit_pooling_config_is_preserved(self):
        for flag, value in [("pooling_mode_mean_tokens", 1), ("pooling_mode_cls_token", 2), ("pooling_mode_lasttoken", 3)]:
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "modules.json").write_text(json.dumps([{"type": "sentence_transformers.models.Pooling", "path": "1_Pooling"}]))
                (root / "1_Pooling").mkdir()
                (root / "1_Pooling/config.json").write_text(json.dumps({flag: True}))
                writer = Mock()
                ns["set_gguf_parameters"](writer, {"num_hidden_layers": 1, "hidden_size": 8, "num_attention_heads": 2, "intermediate_size": 16, "vocab_size": 32}, root, 1)
                writer.add_pooling_type.assert_called_once_with(value)

if __name__ == "__main__": unittest.main()
