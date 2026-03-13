from types import SimpleNamespace
from unittest.mock import patch

import setup_env
from utils import e2e_benchmark


def test_setup_env_run_command_does_not_exit_on_success_without_log_step():
    with patch("setup_env.subprocess.run") as mock_run, patch("setup_env.sys.exit") as mock_exit:
        setup_env.run_command(["echo", "ok"])

    mock_run.assert_called_once_with(["echo", "ok"], shell=False, check=True)
    mock_exit.assert_not_called()


def test_setup_env_run_command_exits_on_failure_without_log_step():
    error = setup_env.subprocess.CalledProcessError(1, ["echo", "fail"])
    with patch("setup_env.subprocess.run", side_effect=error), patch(
        "setup_env.sys.exit"
    ) as mock_exit:
        setup_env.run_command(["echo", "fail"])

    mock_exit.assert_called_once_with(1)


def test_e2e_benchmark_run_command_does_not_exit_on_success_without_log_step():
    with patch("utils.e2e_benchmark.subprocess.run") as mock_run, patch(
        "utils.e2e_benchmark.sys.exit"
    ) as mock_exit:
        e2e_benchmark.run_command(["echo", "ok"])

    mock_run.assert_called_once_with(["echo", "ok"], shell=False, check=True)
    mock_exit.assert_not_called()


def test_e2e_benchmark_run_benchmark_uses_run_command_without_exiting_early():
    e2e_benchmark.args = SimpleNamespace(
        model="model.gguf",
        n_token=32,
        threads=4,
        n_prompt=64,
    )

    with patch("utils.e2e_benchmark.platform.system", return_value="Linux"), patch(
        "utils.e2e_benchmark.os.path.exists", return_value=True
    ), patch("utils.e2e_benchmark.run_command") as mock_run:
        e2e_benchmark.run_benchmark()

    mock_run.assert_called_once()
