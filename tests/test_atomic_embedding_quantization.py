import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "utils" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class AtomicEmbeddingTests(unittest.TestCase):
    def check(self, name, failed):
        module = load(name)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "input.gguf"
            binary = root / "quantizer"
            model.write_bytes(b"input")
            binary.touch()
            if name == "quantize_embeddings":
                obj = module.EmbeddingQuantizer(model, root, binary, binary, root)
                call = obj.quantize
            else:
                obj = module.PerplexityTester(model, binary, root, root, quantize_bin=binary)
                call = obj.quantize_embedding
            output = root / "ggml-model-i2_s-embed-q6_k.gguf"
            def run(command, **kwargs):
                Path(command[4]).write_bytes(b"" if failed == "empty" else b"partial" if failed else b"complete")
                if failed == "timeout":
                    raise subprocess.TimeoutExpired(command, 600)
                return subprocess.CompletedProcess(command, 1 if failed == "nonzero" else 0, "", "")
            with patch.object(module.subprocess, "run", side_effect=run):
                result = call("Q6_K", "q6_k")
            if failed:
                self.assertFalse(result)
                self.assertFalse(output.exists(), "partial GGUF must not be reusable")
            else:
                self.assertTrue(result)
                self.assertEqual(output.read_bytes(), b"complete")
            self.assertEqual(list(root.glob("*.tmp.gguf")), [])

    def test_failure_and_timeout_never_publish_partial_output(self):
        for name in ("quantize_embeddings", "test_perplexity"):
            for failure in ("nonzero", "timeout", "empty"):
                with self.subTest(name=name, failure=failure):
                    self.check(name, failure)

    def test_success_publishes_output(self):
        for name in ("quantize_embeddings", "test_perplexity"):
            with self.subTest(name=name):
                self.check(name, False)

if __name__ == "__main__":
    unittest.main()
