"""Benchmark tables use backend-dependent columns in pinned llama-bench."""
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "utils/quantize_embeddings.py"
spec = importlib.util.spec_from_file_location("quantize_embeddings", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class BenchmarkColumnsTests(unittest.TestCase):
    def parse(self, headers, values):
        table = "| " + " | ".join(headers) + " |\n"
        table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        table += "| " + " | ".join(values) + " |\n"
        return module.EmbeddingQuantizer.parse_benchmark_output(None, table, "q6_k")

    def test_cpu_table(self):
        result = self.parse(["model", "size", "params", "backend", "threads", "test", "t/s"],
            ["bitnet 2B", "100 MiB", "2 B", "CPU", "2", "pp128", "405.73 ± 3.69"])
        self.assertEqual(result["threads_2"], "405.73±3.69")

    def test_metal_table_includes_ngl(self):
        result = self.parse(["model", "size", "params", "backend", "ngl", "threads", "test", "t/s"],
            ["bitnet 2B", "100 MiB", "2 B", "Metal", "0", "4", "pp128", "800.12 ± 2.34"])
        self.assertEqual(result["threads_4"], "800.12±2.34")

    def test_optional_columns_do_not_shift_metrics(self):
        result = self.parse(["model", "size", "params", "backend", "ngl", "threads", "cpu_mask", "test", "t/s"],
            ["bitnet 2B", "100 MiB", "2 B", "CUDA", "0", "8", "0xff", "pp128", "1200.00 ± 5.00"])
        self.assertEqual(result["threads_8"], "1200.00±5.00")

if __name__ == "__main__":
    unittest.main()
