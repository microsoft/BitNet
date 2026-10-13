import ast
import os
from pathlib import Path
import tempfile
import unittest
import numpy as np

source = Path(__file__).resolve().parents[1] / "utils/convert-hf-to-gguf-bitnet.py"
tree = ast.parse(source.read_text())
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"process_tl1", "preprocess_weights_tl1", "preprocess_two_weights_tl2", "preprocess_three_weights_tl2", "preprocess_weights_tl2"}]
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + functions, type_ignores=[])
ast.fix_missing_locations(module)

class KernelConfigLocationTests(unittest.TestCase):
    def test_conversion_uses_repository_config_from_other_directory(self):
        for mode, columns in (("tl1", 128), ("tl2", 384)):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "utils").mkdir()
                (root / "include").mkdir()
                (root / "elsewhere").mkdir()
                (root / "include/kernel_config.ini").write_text(f"[Kernels_0]\nm=32\nk={columns}\nbm=32\nbk={columns}\nbmm=32\n")
                ns = {"np": np, "Path": Path, "__file__": str(root / "utils/converter.py")}
                exec(compile(module, str(source), "exec"), ns)
                weights = np.ones((32, columns), dtype=np.float32)
                previous = Path.cwd()
                try:
                    os.chdir(root)
                    expected = ns["preprocess_weights_" + mode](weights)
                    os.chdir(root / "elsewhere")
                    actual = ns["preprocess_weights_" + mode](weights)
                finally:
                    os.chdir(previous)
                np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__": unittest.main()
