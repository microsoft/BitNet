import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get("BITNET_TEST_ROOT", Path(__file__).resolve().parents[1]))


class EmbeddingSetup(unittest.TestCase):
    def load(self, directory, quant_type="i2_s", quant_embd=False):
        spec = importlib.util.spec_from_file_location("setup_env", ROOT / "setup_env.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.args = SimpleNamespace(hf_repo=None, model_dir=str(directory),
                                      quant_type=quant_type, quant_embd=quant_embd,
                                      use_pretuned=False)
        return module

    def test_embedding_setup_uses_dedicated_converter_without_lut_codegen(self):
        for name in ["bitnet-embedding-0.6b", "bitnet-embedding-270m"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                model = Path(directory) / name
                model.mkdir()
                module = self.load(model)
                with patch.object(module, "system_info", return_value=("Linux", "x86_64")), \
                        patch.object(module, "run_command") as run:
                    module.gen_code()
                    run.assert_not_called()
                    module.prepare_model()
                run.assert_called_once()
                command = run.call_args.args[0]
                self.assertEqual(command[1], "utils/convert-bitnet-embedding-to-gguf.py")
                self.assertEqual(command[command.index("--outfile") + 1],
                                 str(model / "ggml-model-i2_s.gguf"))
                self.assertEqual(command[command.index("--outtype") + 1], "i2_s")

    def test_unsupported_embedding_modes_fail_before_codegen(self):
        for arch, quant, quant_embd in [("arm64", "i2_s", False),
                                        ("x86_64", "tl2", False),
                                        ("x86_64", "i2_s", True)]:
            module = self.load(Path("bitnet-embedding-270m"), quant, quant_embd)
            with self.subTest(arch=arch, quant=quant, quant_embd=quant_embd), \
                    patch.object(module, "system_info", return_value=("Linux", arch)), \
                    patch.object(module, "run_command") as run:
                with self.assertRaisesRegex(ValueError, "Embedding models require"):
                    module.gen_code()
                run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
