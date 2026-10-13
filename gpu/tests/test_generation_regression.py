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

class OutputTests(unittest.TestCase):
    def run_generation(self, length, eos):
        fn, ns = load_method("generate_all")
        ns.update(AttnBias=Bias, Stats=Stats)
        g = generator(length)
        g._prefill_compile_model = lambda *args: torch.eye(8)[:4]
        g._generate_compile_model = lambda *args: torch.eye(8)[7 if eos else 3].reshape(1,8)
        with patch.object(torch.Tensor, "cuda", lambda s: s), patch.object(torch.cuda, "synchronize", lambda: None):
            return fn(g, [[1,2]], True, False)

    def test_single_token_budget(self):
        stats, answers = self.run_generation(1, False)
        self.assertEqual(answers, [[1]])
        self.assertEqual(stats.tokens, 1)

    def test_early_eos_has_no_unwritten_tail(self):
        stats, answers = self.run_generation(4, True)
        self.assertEqual(answers, [[1,7]])
        self.assertEqual(stats.tokens, 2)

    def test_full_budget_has_no_unwritten_tail(self):
        stats, answers = self.run_generation(4, False)
        self.assertEqual(answers, [[1,3,3,3]])
        self.assertEqual(stats.tokens, 4)

if __name__ == "__main__": unittest.main()
