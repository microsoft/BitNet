import importlib.util
import os
from pathlib import Path
import tempfile
import runpy
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(os.environ.get("BITNET_TEST_ROOT", Path(__file__).resolve().parents[1]))
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class RuntimePaths(unittest.TestCase):
    def test_binary_resolves_from_checkout_and_explicit_model_stays_relative(self):
        for name, entry, binary in [("run_inference", "run_inference", "llama-cli"), ("run_inference_server", "run_server", "llama-server")]:
            for system in ["Linux", "Windows"]:
                with self.subTest(name=name, system=system), tempfile.TemporaryDirectory() as d:
                    m = load(name)
                    m.args = SimpleNamespace(model="user-model.gguf", n_predict=1, threads=2, prompt="hello", ctx_size=128, temperature=0.0, conversation=False, host="127.0.0.1", port=8080)
                    old = os.getcwd()
                    try:
                        os.chdir(d)
                        with patch.object(m.platform, "system", return_value=system), patch.object(m.os.path, "exists", return_value=True), patch.object(m, "run_command") as run:
                            getattr(m, entry)()
                        cmd = run.call_args.args[0]
                        suffix = ["build", "bin", "Release", binary + ".exe"] if system == "Windows" else ["build", "bin", binary]
                        self.assertEqual(cmd[0], str(ROOT.joinpath(*suffix)))
                        self.assertEqual(cmd[cmd.index("-m") + 1], "user-model.gguf")
                    finally:
                        os.chdir(old)

    def test_default_model_resolves_from_checkout_when_cli_is_launched_elsewhere(self):
        for name in ["run_inference", "run_inference_server"]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as d:
                old = os.getcwd()
                try:
                    os.chdir(d)
                    with patch.object(sys, "argv", [str(ROOT / (name + ".py")), "-p", "hello"]), patch("subprocess.run") as run:
                        runpy.run_path(str(ROOT / (name + ".py")), run_name="__main__")
                    cmd = run.call_args.args[0]
                    self.assertEqual(cmd[cmd.index("-m") + 1], str(ROOT / "models/bitnet_b1_58-3B/ggml-model-i2_s.gguf"))
                finally: os.chdir(old)

if __name__ == "__main__": unittest.main()
