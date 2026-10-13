import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock

source = Path(__file__).resolve().parents[1] / "utils/convert-bitnet-embedding-to-gguf.py"
tree = ast.parse(source.read_text())
methods = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_set_vocab_gemma3", "_does_token_look_special"}]
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + methods, type_ignores=[])
ast.fix_missing_locations(module)
ns = {"json": json, "gguf": SimpleNamespace(TokenType=SimpleNamespace(UNUSED=5, CONTROL=3, USER_DEFINED=4, BYTE=6, UNKNOWN=2, NORMAL=1), SpecialVocab=lambda *a, **k: SimpleNamespace(add_to_gguf=lambda writer: None))}
exec(compile(module, str(source), "exec"), ns)

class GemmaAddedTokensTests(unittest.TestCase):
    def test_added_tokens_outside_base_vocab_keep_ids_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"<unk>": 0, "base": 1}, "merges": []}}))
            tokenizer = SimpleNamespace(added_tokens_decoder={2: SimpleNamespace(content="<eos>", special=True), 3: SimpleNamespace(content="custom", special=False)})
            writer = Mock()
            ns["_set_vocab_gemma3"](writer, root, tokenizer, 5)
            writer.add_token_list.assert_called_once_with([b"<unk>", b"base", b"<eos>", b"custom", b"[PAD4]"])
            writer.add_token_types.assert_called_once_with([2, 1, 3, 4, 5])
            writer.add_token_scores.assert_called_once_with([0.0, 0.0, -1000.0, -1000.0, -10000.0])

if __name__ == "__main__": unittest.main()
