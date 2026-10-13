"""Run tokenizer export methods with a tiny SentencePiece stand-in."""
import ast
import json
import logging
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
class Tokenizer:
    def __init__(self, path): pass
    def vocab_size(self): return 2
    def id_to_piece(self, id): return ["<unk>", "base"][id]
    def get_score(self, id): return 0.0
    def is_unknown(self, id): return id == 0
    def is_control(self, id): return False
    def is_unused(self, id): return False
    def is_byte(self, id): return False
class Writer:
    def __getattr__(self, name):
        def save(value): setattr(self, name, value)
        return save

def export(filename, directory):
    source = ROOT / "utils" / filename
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_set_vocab_sentencepiece")
    special = SimpleNamespace(add_to_gguf=lambda writer: None)
    ns = {"json": json, "logger": logging.getLogger("test"), "SentencePieceTokenTypes": SimpleNamespace(NORMAL=1, UNKNOWN=2, CONTROL=3, UNUSED=5, BYTE=6, USER_DEFINED=4), "gguf": SimpleNamespace(SpecialVocab=lambda *args, **kwargs: special)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), ns)
    writer = Writer()
    with patch.dict(sys.modules, {"sentencepiece": SimpleNamespace(SentencePieceProcessor=Tokenizer)}):
        ns["_set_vocab_sentencepiece"](SimpleNamespace(dir_model=directory, hparams={"vocab_size": 4}, gguf_writer=writer))
    return writer

class SentencePieceIdsTests(unittest.TestCase):
    def check(self, tokens, expected=None):
        for filename in ("convert-hf-to-gguf-bitnet.py", "generate-dummy-bitnet-model.py"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "tokenizer.model").touch()
                (root / "added_tokens.json").write_text(json.dumps(tokens))
                if expected is None:
                    with self.assertRaises(ValueError): export(filename, root)
                else:
                    self.assertEqual(export(filename, root).add_token_list, expected)

    def test_json_order_does_not_change_token_ids(self):
        self.check({"last": 3, "first": 2}, [b"<unk>", b"base", b"first", b"last"])

    def test_existing_token_is_not_appended(self):
        self.check({"base": 1, "last": 3, "first": 2}, [b"<unk>", b"base", b"first", b"last"])

    def test_conflicting_base_id_is_rejected(self):
        self.check({"different": 1, "last": 3, "first": 2})

if __name__ == "__main__": unittest.main()
