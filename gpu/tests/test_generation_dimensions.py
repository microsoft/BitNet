import ast
from dataclasses import dataclass
from pathlib import Path
import unittest


def arguments():
    tree = ast.parse(Path(__file__).parents[1].joinpath("generate.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "GenArgs")
    ns = {"dataclass": dataclass}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])), "generate.py", "exec"),ns)
    return ns["GenArgs"]

class DimensionTests(unittest.TestCase):
    def test_invalid_dimensions_fail_before_cache_allocation(self):
        cls = arguments()
        for name in ["gen_length", "gen_bsz", "prompt_length"]:
            for value in [0, -1, 1.5, True]:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError,name):
                    cls(**{name:value})

    def test_single_token_boundaries_and_defaults(self):
        cls = arguments()
        self.assertEqual(cls().gen_length,32)
        self.assertEqual(cls(gen_length=1, gen_bsz=1, prompt_length=1).prompt_length,1)

if __name__ == "__main__": unittest.main()
