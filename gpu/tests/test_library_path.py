import ast
from pathlib import Path
import types
import unittest

class LibraryPathTests(unittest.TestCase):
    def test_runtime_and_benchmark_load_library_beside_source(self):
        for name in ["model.py", "test.py"]:
            with self.subTest(module=name):
                source = Path(__file__).parents[1]/name
                tree = ast.parse(source.read_text())
                assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "bitnet_lib" for t in n.targets))
                paths = []
                ns = dict(Path=Path,__file__=str(source),ctypes=types.SimpleNamespace(CDLL=lambda p:paths.append(p)))
                exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment],type_ignores=[])),name,"exec"),ns)
                self.assertEqual(paths,[str(source.resolve().parent/"bitnet_kernels/libbitnet.so")])

if __name__ == "__main__": unittest.main()
