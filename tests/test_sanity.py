import subprocess
import sys


def test_setup_env_help():
    # ensure setup_env.py prints help without error
    res = subprocess.run([sys.executable, "setup_env.py", "--help"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert res.returncode == 0
    assert b"Setup the environment for running the inference" in res.stdout or b"Setup the environment" in res.stdout
