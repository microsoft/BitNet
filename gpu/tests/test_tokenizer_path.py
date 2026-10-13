import ast
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import torch


def load_method(name):
    tree = ast.parse(Path(__file__).parents[1].joinpath("generate.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastGen")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    method.decorator_list = []
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ns = {"torch": torch, "Path": Path, "__file__": str(Path(__file__).parents[1] / "generate.py")}
    exec(compile(ast.fix_missing_locations(module), "generate.py", "exec"), ns)
    return ns[method.name], ns



class TokenizerPathTests(unittest.TestCase):
    def check_path(self, custom):
        fn, ns = load_method("build")
        paths = []
        ns.update(time=types.SimpleNamespace(time=lambda: 0), fast=types.SimpleNamespace(ModelArgs=lambda **kw: None), Tokenizer=lambda p: paths.append(p))
        with patch.object(torch, "set_default_device", side_effect=RuntimeError("stop after tokenizer")):
            with self.assertRaisesRegex(RuntimeError, "stop after tokenizer"):
                fn("unused", None, "cpu", tokenizer_path=custom)
        return paths[0]

    def test_explicit_tokenizer_is_used(self):
        self.assertEqual(self.check_path("custom/tokenizer.model"), "custom/tokenizer.model")

    def test_default_tokenizer_is_relative_to_script(self):
        self.assertEqual(self.check_path(None), str(Path(__file__).parents[1]/"tokenizer.model"))

if __name__ == "__main__": unittest.main()
