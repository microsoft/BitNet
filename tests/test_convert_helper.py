import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("helper", Path(__file__).resolve().parents[1] / "utils/convert-helper-bitnet.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)

class CheckpointPreservationTests(unittest.TestCase):
    def run_case(self, backup=False, fail_move=False, successful=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            utils = root / "utils"
            utils.mkdir()
            for name in ("preprocess-huggingface-bitnet.py", "convert-ms-to-gguf-bitnet.py"):
                (utils / name).touch()
            binary = root / "build/bin/llama-quantize"
            binary.parent.mkdir(parents=True)
            binary.touch()
            model = root / "model"
            model.mkdir()
            original = model / "model.safetensors"
            original.write_bytes(b"original")
            saved = model / "model.safetensors.backup"
            if backup:
                saved.write_bytes(b"previous-original")
            def preprocess(*args, **kwargs):
                original.write_bytes(b"processed")
            effect = preprocess if successful else RuntimeError("preprocess failed")
            with patch.object(helper, "__file__", str(utils / "helper.py")), patch.object(helper.sys, "argv", ["helper", str(model)]), patch.object(helper, "run_command", side_effect=effect) as command:
                if fail_move:
                    with patch.object(helper.shutil, "move", side_effect=OSError("move failed")):
                        status = helper.main()
                else:
                    status = helper.main()
            self.assertEqual(status, None if successful else 1)
            self.assertEqual(original.read_bytes(), b"original")
            if backup:
                self.assertEqual(saved.read_bytes(), b"previous-original")
                command.assert_not_called()
            else:
                self.assertFalse(saved.exists())

    def test_success_restores_original(self):
        self.run_case(successful=True)

    def test_existing_backup_is_preserved(self):
        self.run_case(backup=True)

    def test_failed_move_does_not_delete_original(self):
        self.run_case(fail_move=True)

    def test_preprocess_failure_restores_original(self):
        self.run_case()

if __name__ == "__main__":
    unittest.main()
