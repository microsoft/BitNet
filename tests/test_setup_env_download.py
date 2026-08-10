"""Regression test for issue #560.

`setup_env.py` must download models with the current Hugging Face CLI entry
point (`hf`). The legacy `huggingface-cli` command was removed from recent
`huggingface_hub` releases, so the documented `-hr/--hf-repo` download flow
fails with "command not found" when the script still shells out to it.
"""

import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import setup_env  # noqa: E402

DOWNLOAD_LOG_STEP = "download_model"
LEGACY_CLI = "huggingface-cli"
CURRENT_CLI = "hf"


class _StopAfterDownload(Exception):
    """Sentinel to halt prepare_model() right after the download command."""


class DownloadCommandTest(unittest.TestCase):
    def test_download_uses_current_hf_cli(self):
        captured = []

        def fake_run_command(command, shell=False, log_step=None):
            captured.append(command)
            if log_step == DOWNLOAD_LOG_STEP:
                raise _StopAfterDownload

        hf_repo = next(iter(setup_env.SUPPORTED_HF_MODELS))
        fake_args = types.SimpleNamespace(
            hf_repo=hf_repo,
            model_dir="models",
            quant_type="i2_s",
            quant_embd=False,
        )

        with mock.patch.object(setup_env, "run_command", fake_run_command), \
                mock.patch.object(setup_env, "args", fake_args, create=True), \
                mock.patch.object(Path, "mkdir", lambda *a, **k: None):
            with self.assertRaises(_StopAfterDownload):
                setup_env.prepare_model()

        download_cmd = captured[-1]
        self.assertEqual(download_cmd[0], CURRENT_CLI)
        self.assertEqual(download_cmd[1], "download")
        self.assertNotIn(LEGACY_CLI, download_cmd)


if __name__ == "__main__":
    unittest.main()
