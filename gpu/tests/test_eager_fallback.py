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


class Bias:
    @staticmethod
    def from_seqlens(**kwargs):
        Bias.padding = kwargs["kv_padding"]
        return types.SimpleNamespace(q_seqinfo=types.SimpleNamespace(to=lambda x: None), k_seqinfo=types.SimpleNamespace(to=lambda x: None, seqlen=torch.tensor(kwargs["kv_seqlen"])))


class Stats:
    def phase(self, *args): pass
    def end_phase(self, tokens): self.tokens = tokens


def generator(length=4, batch=1):
    return types.SimpleNamespace(gen_args=types.SimpleNamespace(prompt_length=4, gen_length=length), max_seq_length=4+length, model_args=types.SimpleNamespace(vocab_size=8), tokenizer=types.SimpleNamespace(eot_id=7))


import os

class EagerTests(unittest.TestCase):
    def test_disabled_graphs_run_models_with_unpadded_prompt_lengths(self):
        fn, ns = load_method("generate_all")
        ns.update(AttnBias=Bias, Stats=Stats)
        g = generator(2,2)
        g._cache = object()
        seen = []
        def prefill(**kw):
            seen.append(("prefill", kw["token_values"].tolist(), kw["attn_bias"].k_seqinfo.seqlen.tolist(), Bias.padding))
            return torch.eye(8)[:5]
        def decode(**kw):
            seen.append(("decode", kw["attn_bias"].k_seqinfo.seqlen.tolist(), Bias.padding))
            return torch.zeros(2,8)
        g.prefill_model = types.SimpleNamespace(forward_with_attn_bias=prefill)
        g.decode_model = types.SimpleNamespace(forward_with_attn_bias=decode)
        def forbidden(*args): raise AssertionError("CUDA graph called in eager mode")
        g._prefill_compile_model = g._generate_compile_model = forbidden
        with patch.object(torch.Tensor, "cuda", lambda s:s), patch.object(torch.cuda,"synchronize",lambda:None):
            _, answers = fn(g, [[1,2],[3,4,5]], False, False)
        self.assertEqual(seen, [("prefill",[1,2,3,4,5],[2,3],6),("decode",[3,4],6)])
        self.assertEqual([x[0] for x in answers], [1,4])

    def test_environment_skips_graph_capture_during_construction(self):
        fn, ns = load_method("__init__")
        cache = object()
        ns.update(os=os, time=types.SimpleNamespace(time=lambda:0), fast=types.SimpleNamespace(make_cache=lambda **kw:cache))
        def forbidden(): raise AssertionError("capture attempted")
        g = types.SimpleNamespace(compile_prefill=forbidden,compile_generate=forbidden)
        args = types.SimpleNamespace(prompt_length=4,gen_length=2,gen_bsz=1)
        with patch.dict(os.environ,{"NO_CUDA_GRAPHS":"1"}): fn(g,args,None,None,None,None)
        self.assertIs(g._cache,cache)
        self.assertIsNone(g._prefill_compile_model)

if __name__ == "__main__": unittest.main()
