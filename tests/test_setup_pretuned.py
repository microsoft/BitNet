import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(os.environ.get("BITNET_TEST_ROOT", Path(__file__).resolve().parents[1]))
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class PretunedKernels(unittest.TestCase):
    def test_presets_are_preserved_and_matching_config_is_copied(self):
        for arch, kind in [("arm64", "tl1"), ("x86_64", "tl2")]:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as d:
                m = load("setup_env")
                m.args = SimpleNamespace(use_pretuned=True, quant_type=kind, hf_repo="1bitLLM/bitnet_b1_58-3B")
                root = Path(d)
                preset = root / "preset_kernels" / "bitnet_b1_58-3B"
                preset.mkdir(parents=True)
                (root / "include").mkdir()
                (preset / f"bitnet-lut-kernels-{kind}.h").write_text("custom tuned header")
                (preset / f"kernel_config_{kind}.ini").write_text("custom tuned config")
                old = os.getcwd()
                try:
                    os.chdir(d)
                    with patch.object(m, "system_info", return_value=("Linux", arch)), patch.object(m, "run_command") as run:
                        m.gen_code()
                    self.assertEqual((root / "include/bitnet-lut-kernels.h").read_text(), "custom tuned header")
                    self.assertEqual((root / "include/kernel_config.ini").read_text(), "custom tuned config")
                    run.assert_not_called()
                finally: os.chdir(old)

    def test_missing_config_leaves_existing_header_untouched(self):
        m = load("setup_env")
        m.args = SimpleNamespace(use_pretuned=True, quant_type="tl2", hf_repo="1bitLLM/bitnet_b1_58-3B")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            preset = root / "preset_kernels/bitnet_b1_58-3B"
            preset.mkdir(parents=True)
            (preset / "bitnet-lut-kernels-tl2.h").write_text("new header")
            (root / "include").mkdir()
            (root / "include/bitnet-lut-kernels.h").write_text("original header")
            old = os.getcwd()
            try:
                os.chdir(d)
                with patch.object(m, "system_info", return_value=("Linux", "x86_64")), patch.object(m, "run_command"), self.assertRaises(SystemExit):
                    m.gen_code()
                self.assertEqual((root / "include/bitnet-lut-kernels.h").read_text(), "original header")
            finally: os.chdir(old)

    def test_without_pretuned_codegen_still_runs(self):
        m = load("setup_env")
        m.args = SimpleNamespace(use_pretuned=False, quant_type="tl1", hf_repo="1bitLLM/bitnet_b1_58-3B")
        with patch.object(m, "system_info", return_value=("Linux", "arm64")), patch.object(m, "run_command") as run:
            m.gen_code()
        self.assertEqual(run.call_count, 1)
        self.assertIn("utils/codegen_tl1.py", run.call_args.args[0])

if __name__ == "__main__": unittest.main()
