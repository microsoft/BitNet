import importlib.util
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "test_perplexity.py"


def load_test_perplexity_module():
    spec = importlib.util.spec_from_file_location(
        "bitnet_test_perplexity_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPerplexityPathTests(unittest.TestCase):
    def test_resolve_default_path_uses_script_dir_for_built_in_defaults(self):
        module = load_test_perplexity_module()

        resolved = module.resolve_default_path(
            "../build/bin/llama-perplexity", "../build/bin/llama-perplexity"
        )

        self.assertEqual(
            resolved, (MODULE_PATH.parent / "../build/bin/llama-perplexity").resolve()
        )

    def test_resolve_default_path_keeps_custom_paths(self):
        module = load_test_perplexity_module()

        self.assertEqual(
            module.resolve_default_path(
                "/tmp/custom-bin", "../build/bin/llama-perplexity"
            ),
            Path("/tmp/custom-bin"),
        )
        self.assertEqual(
            module.resolve_default_path(
                "models/custom.gguf", "../build/bin/llama-perplexity"
            ),
            Path("models/custom.gguf"),
        )

    def test_perplexity_tester_uses_resolved_default_paths(self):
        module = load_test_perplexity_module()

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.gguf"
            model_path.write_text("stub")

            expected_bin = (
                MODULE_PATH.parent / "../build/bin/llama-perplexity"
            ).resolve()
            expected_quant = (
                MODULE_PATH.parent / "../build/bin/llama-quantize"
            ).resolve()
            expected_data = (MODULE_PATH.parent / "../data").resolve()

            original_exists = module.Path.exists

            def fake_exists(self):
                if self in {expected_bin, expected_quant, expected_data, model_path}:
                    return True
                return original_exists(self)

            module.Path.exists = fake_exists
            try:
                tester = module.PerplexityTester(str(model_path))
            finally:
                module.Path.exists = original_exists

            self.assertEqual(tester.llama_perplexity_bin, expected_bin)
            self.assertEqual(tester.quantize_bin, expected_quant)
            self.assertEqual(tester.data_dir, expected_data)

    def test_main_uses_constructor_default_paths(self):
        module = load_test_perplexity_module()
        captured = {}

        class FakeTester:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def run_all_tests(self, **kwargs):
                captured["run_all_tests"] = kwargs

        with (
            mock.patch.object(module, "PerplexityTester", FakeTester),
            mock.patch.object(
                sys, "argv", ["test_perplexity.py", "--model", "model.gguf"]
            ),
        ):
            exit_code = module.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            captured["llama_perplexity_bin"], module.DEFAULT_LLAMA_PERPLEXITY
        )
        self.assertEqual(captured["quantize_bin"], module.DEFAULT_QUANTIZE_BIN)
        self.assertEqual(captured["data_dir"], module.DEFAULT_DATA_DIR)


if __name__ == "__main__":
    unittest.main()
