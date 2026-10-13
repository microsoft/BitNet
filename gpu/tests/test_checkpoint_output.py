import ast
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import torch

class CheckpointOutputTests(unittest.TestCase):
    def test_bare_input_filename_writes_beside_input(self):
        tree = ast.parse(Path(__file__).parents[1].joinpath("convert_checkpoint.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "convert_ts_checkpoint")
        fn.decorator_list = []
        ns = dict(torch=torch, Path=Path, os=os, model=types.SimpleNamespace(ModelArgs=lambda: types.SimpleNamespace()))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "convert_checkpoint.py", "exec"), ns)
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            try:
                os.chdir(tmp)
                with patch.object(torch, "load", return_value={}), patch.object(torch, "save") as save:
                    ns["convert_ts_checkpoint"](input_path="model_state.pt")
                self.assertEqual([Path(call.args[1]) for call in save.call_args_list], [Path("model_state_int2.pt"),Path("model_state_fp16.pt")])
            finally: os.chdir(previous)

if __name__ == "__main__": unittest.main()
