import ast
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import numpy as np
import torch

class BenchmarkTests(unittest.TestCase):
    def run_benchmark(self, correct):
        tree = ast.parse(Path(__file__).parents[1].joinpath("test.py").read_text())
        main = next(n for n in tree.body if isinstance(n, ast.If))
        main.body[0].value = ast.parse("[(16,32)]",mode="eval").body
        calls = []
        def timer(**kw):
            calls.append(kw)
            return types.SimpleNamespace(timeit=lambda n:types.SimpleNamespace(mean=1e-6))
        def linear(input0, weight, s, ws, ret):
            if correct: return (input0.to(torch.int32) @ weight.to(torch.int32).T).to(torch.bfloat16)
            return torch.zeros_like(ret)
        ns = dict(torch=torch,np=np,benchmark=types.SimpleNamespace(Timer=timer),convert_weight_int8_to_int2=lambda w:w,bitnet_int8xint2_linear=linear)
        real_ones, real_to = torch.ones, torch.Tensor.to
        def ones(*args,**kw):
            kw.pop("device",None)
            return real_ones(*args,**kw)
        def to(tensor,*args,**kw):
            if args and args[0] == "cuda": args = ("cpu",)+args[1:]
            return real_to(tensor,*args,**kw)
        def randint(low,high,shape,**kw): return real_ones(shape,dtype=kw["dtype"])
        with patch.object(torch,"randint",randint), patch.object(torch,"ones",ones), patch.object(torch.Tensor,"to",to), patch.object(torch.Tensor,"cuda",lambda s:s):
            exec(compile(ast.fix_missing_locations(ast.Module(body=main.body,type_ignores=[])),"test.py","exec"),ns)
        return calls

    def test_mismatch_stops_before_performance_measurement(self):
        with self.assertRaisesRegex(AssertionError,"N=16, K=32"):
            self.run_benchmark(False)

    def test_correct_result_reaches_benchmarks(self):
        self.assertEqual(len(self.run_benchmark(True)),2)

if __name__ == "__main__": unittest.main()
