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

class TLEmbeddingOption(unittest.TestCase):
    def test_embedding_quantization_is_opt_in_for_both_tl_formats(self):
        for quant_type in ["tl1", "tl2"]:
            for enabled in [False, True]:
                with self.subTest(quant_type=quant_type, enabled=enabled), tempfile.TemporaryDirectory() as d:
                    m = load("setup_env")
                    m.args = SimpleNamespace(hf_repo=None, model_dir=d, quant_type=quant_type, quant_embd=enabled)
                    with patch.object(m, "system_info", return_value=("Linux", "arm64")), patch.object(m, "run_command") as run:
                        m.prepare_model()
                    cmd = run.call_args.args[0]
                    self.assertEqual("--quant-embd" in cmd, enabled)
                    self.assertEqual(cmd[cmd.index("--outtype") + 1], quant_type)

if __name__ == "__main__": unittest.main()
