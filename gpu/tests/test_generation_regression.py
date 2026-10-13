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
    return types.SimpleNamespace(gen_args=types.SimpleNamespace(prompt_length=4, gen_length=length, gen_bsz=batch), max_seq_length=4+length, model_args=types.SimpleNamespace(vocab_size=8), tokenizer=types.SimpleNamespace(eot_id=7))

class CacheTests(unittest.TestCase):
    def test_rejects_inputs_outside_capture(self):
        fn, ns = load_method("generate_all")
        g = generator(2, 2)
        for prompts in [[], [[1]], [[],[1]], [[1]*5,[2]]]:
            with self.assertRaises(ValueError): fn(g, prompts, True, False)

    def test_short_batched_prompts_keep_capture_layout(self):
        fn, ns = load_method("generate_all")
        ns.update(AttnBias=Bias, Stats=Stats)
        g = generator(2,2)
        seen = []
        def prefill(tokens, lengths):
            seen.append(None if lengths is None else lengths.tolist())
            # rope_padded places each query at kv_length - query_length + query_pos.
            # The captured query length is four; supplying shorter KV lengths
            # would write the first query before its cache region.
            kv_lengths = [4,4] if lengths is None else lengths.tolist()
            positions = [[kv_length - 4 + i for i in range(4)] for kv_length in kv_lengths]
            self.assertEqual(positions, [[0,1,2,3],[0,1,2,3]])
            return torch.eye(8)
        g._prefill_compile_model = prefill
        g._generate_compile_model = lambda *args: torch.zeros(2,8)
        with patch.object(torch.Tensor, "cuda", lambda s: s), patch.object(torch.cuda, "synchronize", lambda: None):
            _, answers = fn(g, [[1,2],[3,4,5]], True, False)
        self.assertEqual(Bias.padding, 6)
        self.assertEqual(seen, [None])
        self.assertEqual([answer[0] for answer in answers], [1,6])

if __name__ == "__main__": unittest.main()
