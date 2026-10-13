import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(os.environ.get("BITNET_TEST_ROOT", Path(__file__).resolve().parents[1]))

class I2SModelDirectory(unittest.TestCase):
    def load(self, quant_type, model_dir="/models/my-local-checkpoint"):
        spec = importlib.util.spec_from_file_location("setup_env", ROOT / "setup_env.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.args = SimpleNamespace(quant_type=quant_type, hf_repo=None, model_dir=model_dir, use_pretuned=False)
        return module

    def test_i2s_setup_does_not_require_a_lut_model_directory_name(self):
        for arch in ("arm64", "x86_64"):
            with self.subTest(arch=arch):
                module = self.load("i2_s")
                with patch.object(module, "system_info", return_value=("Linux", arch)), patch.object(module, "run_command") as run:
                    module.gen_code()
                run.assert_not_called()

    def test_tl_setup_still_generates_matching_model_kernel(self):
        for arch, quant in (("arm64", "tl1"), ("x86_64", "tl2")):
            with self.subTest(arch=arch):
                module = self.load(quant, "/models/bitnet_b1_58-3B")
                with patch.object(module, "system_info", return_value=("Linux", arch)), patch.object(module, "run_command") as run:
                    module.gen_code()
                self.assertIn("utils/codegen_" + quant + ".py", run.call_args.args[0])

if __name__ == "__main__": unittest.main()
