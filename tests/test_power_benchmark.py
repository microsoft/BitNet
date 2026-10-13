import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(os.environ.get("BITNET_TEST_ROOT", Path(__file__).resolve().parents[1]))


class PowerBenchmarkTests(unittest.TestCase):
    def test_failed_benchmark_returns_failure_without_a_csv_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "build" / "bin" / "llama-bench"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\necho 'model loading failed' >&2\nexit 7\n")
            binary.chmod(0o755)
            model = root / "model.gguf"
            model.touch()
            mocks = root / "mocks"
            mocks.mkdir()
            top = mocks / "top"
            top.write_text("#!/bin/sh\necho 'Cpu(s) 0 0 0 0 0 0 99'\n")
            top.chmod(0o755)
            grep = mocks / "grep"
            grep.write_text("#!/bin/sh\ncase \"$*\" in\n*/proc/cpuinfo*) echo 'cpu MHz : 1000';;\n*) exec "
                            + shutil.which("grep") + ' "$@";;\nesac\n')
            grep.chmod(0o755)
            environment = dict(os.environ, PATH=str(mocks) + os.pathsep + os.environ["PATH"])
            csv = root / "results.csv"
            process = subprocess.run(["bash", str(ROOT / "utils" / "test_power.sh"),
                                      str(model), str(csv), "1", "1"], cwd=root,
                                     env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(process.returncode, 7, process.stdout + process.stderr)
            self.assertIn("model loading failed", process.stderr)
            self.assertEqual(len(csv.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
