import importlib.util
import subprocess
import tempfile
import types
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "setup_env.py"


def load_setup_env_module():
    spec = importlib.util.spec_from_file_location(
        "bitnet_setup_env_under_test", MODULE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunCommandTests(unittest.TestCase):
    def test_run_command_success_without_log_step_does_not_exit(self):
        module = load_setup_env_module()

        called = {}

        def fake_run(command, shell=False, check=False, **kwargs):
            called["command"] = command
            called["shell"] = shell
            called["check"] = check
            return types.SimpleNamespace(returncode=0)

        module.subprocess.run = fake_run

        module.run_command(["echo", "ok"])

        self.assertEqual(called["command"], ["echo", "ok"])
        self.assertFalse(called["shell"])
        self.assertTrue(called["check"])

    def test_run_command_failure_without_log_step_exits(self):
        module = load_setup_env_module()

        def fake_run(command, shell=False, check=False, **kwargs):
            raise subprocess.CalledProcessError(returncode=7, cmd=command)

        module.subprocess.run = fake_run

        with self.assertRaises(SystemExit) as exc:
            module.run_command(["bad"])

        self.assertEqual(exc.exception.code, 1)

    def test_run_command_failure_with_log_step_exits(self):
        module = load_setup_env_module()

        with tempfile.TemporaryDirectory() as tmp:
            module.args = types.SimpleNamespace(log_dir=tmp)

            def fake_run(command, shell=False, check=False, **kwargs):
                raise subprocess.CalledProcessError(returncode=9, cmd=command)

            module.subprocess.run = fake_run

            with self.assertRaises(SystemExit) as exc:
                module.run_command(["bad"], log_step="compile")

            self.assertEqual(exc.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
