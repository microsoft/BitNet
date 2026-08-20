from unittest.mock import patch

from utils.tune_gemm_config import GemmTuner


ORIGINAL_CONFIG = "#define ORIGINAL 1\n"
CANDIDATE = {
    "act_parallel": True,
    "row_block_size": 8,
    "col_block_size": 64,
    "parallel_size": 4,
}


def make_tuner(tmp_path):
    config_path = tmp_path / "gemm-config.h"
    config_path.write_text(ORIGINAL_CONFIG)
    return GemmTuner(config_path, "model.gguf"), config_path


def record_candidate(tuner, *, with_result):
    def test_configuration(**configuration):
        tuner.generate_config(**configuration)
        if not with_result:
            return None

        result = {
            **configuration,
            "config_name": "candidate",
            "pp_throughput": 1.0,
            "pp_std_dev": 0.0,
        }
        tuner.results.append(result)
        return result

    return test_configuration


def assert_backup_removed(tuner):
    assert not tuner.backup_path.exists()


def test_restores_config_when_no_candidate_succeeds(tmp_path):
    tuner, config_path = make_tuner(tmp_path)

    with patch.object(
        tuner,
        "test_configuration",
        side_effect=record_candidate(tuner, with_result=False),
    ):
        tuner.run_tuning([CANDIDATE], output_csv=tmp_path / "results.csv")

    assert config_path.read_text() == ORIGINAL_CONFIG
    assert_backup_removed(tuner)


def test_restores_config_when_final_rebuild_fails(tmp_path):
    tuner, config_path = make_tuner(tmp_path)

    with (
        patch.object(
            tuner,
            "test_configuration",
            side_effect=record_candidate(tuner, with_result=True),
        ),
        patch.object(tuner, "rebuild_project", return_value=False),
        patch("builtins.input", return_value="y"),
    ):
        tuner.run_tuning([CANDIDATE], output_csv=tmp_path / "results.csv")

    assert config_path.read_text() == ORIGINAL_CONFIG
    assert_backup_removed(tuner)


def test_keeps_config_when_final_rebuild_succeeds(tmp_path):
    tuner, config_path = make_tuner(tmp_path)

    with (
        patch.object(
            tuner,
            "test_configuration",
            side_effect=record_candidate(tuner, with_result=True),
        ),
        patch.object(tuner, "rebuild_project", return_value=True),
        patch("builtins.input", return_value="y"),
    ):
        tuner.run_tuning([CANDIDATE], output_csv=tmp_path / "results.csv")

    assert config_path.read_text() == (
        "#define ACT_PARALLEL\n"
        "#define ROW_BLOCK_SIZE 8\n"
        "#define COL_BLOCK_SIZE 64\n"
        "#define PARALLEL_SIZE 4\n"
    )
    assert_backup_removed(tuner)
