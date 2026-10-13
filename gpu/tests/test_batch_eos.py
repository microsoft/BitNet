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


class EndOfTurnTests(unittest.TestCase):
    def setup_generator(self, first_tokens):
        fn,ns = load_method("generate_all")
        ns.update(AttnBias=Bias,Stats=Stats)
        g = generator(4,2)
        logits = torch.zeros(8,8)
        logits[[0,4],first_tokens[0]] = 1
        logits[[1,5],first_tokens[1]] = 1
        g._prefill_compile_model = lambda *args:logits
        return fn,g

    def test_one_finished_row_does_not_stop_other_rows(self):
        fn,g = self.setup_generator([7,1])
        calls = []
        def decode(tokens,lengths):
            calls.append(tokens.tolist())
            logits = torch.zeros(2,8)
            logits[0,7] = 1
            logits[1,2 if len(calls)==1 else 7] = 1
            return logits
        g._generate_compile_model = decode
        with patch.object(torch.Tensor,"cuda",lambda s:s),patch.object(torch.cuda,"synchronize",lambda:None):
            _,answers = fn(g,[[1],[2,3]],True,False)
        self.assertEqual(calls,[[7,1],[7,2]])
        self.assertEqual(answers,[[7],[1,2,7]])

    def test_all_finished_prefill_skips_decode(self):
        fn,g = self.setup_generator([7,7])
        def forbidden(*args): raise AssertionError("decode after all rows finished")
        g._generate_compile_model = forbidden
        with patch.object(torch.Tensor,"cuda",lambda s:s),patch.object(torch.cuda,"synchronize",lambda:None):
            _,answers = fn(g,[[1],[2,3]],True,False)
        self.assertEqual(answers,[[7],[7]])

if __name__ == "__main__":unittest.main()
