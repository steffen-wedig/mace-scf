"""Energy-only training: compute_forces False skips the force pass entirely."""

import copy

import pytest

from .harness import (
    DEFAULT_REFERENCE_DATA,
    RunTrainCase,
    make_default_metrics_case,
    run_train_case,
)


def _energy_only_case(name, **config_updates):
    base_case = make_default_metrics_case("LocalSplitCharges")
    config = copy.deepcopy(dict(base_case.config))
    config["name"] = name
    config["compute_forces"] = False
    config["error_table"] = "PerAtomEnergyRMSE"
    for stage in config["train_schedule"].values():
        stage["loss"] = {"energy_per_atom": 1.0}
    config.update(config_updates)
    return RunTrainCase(name=name, config=config, timeout_s=base_case.timeout_s)


@pytest.fixture
def reference_data():
    if not DEFAULT_REFERENCE_DATA.exists():
        pytest.skip(f"run_train reference data not found: {DEFAULT_REFERENCE_DATA}")
    return DEFAULT_REFERENCE_DATA


def test_energy_only_training_logs_energy_errors_without_forces(tmp_path, reference_data):
    case = _energy_only_case("energy_only_local_split_charges")
    result = run_train_case(
        tmp_path / case.name, case, reference_data=reference_data, check=True
    )

    log_text = result.log_text()
    assert "RMSE_E_per_atom=" in log_text
    assert "RMSE_F=" not in log_text
    assert "RMSE F / meV / A" not in log_text
    evaluations = [record for record in result.metrics() if record.get("mode") == "eval"]
    assert evaluations
    assert all("rmse_f" not in record for record in evaluations)
    assert result.models()


def test_a_force_loss_without_forces_is_refused(tmp_path, reference_data):
    case = _energy_only_case("energy_only_with_force_loss")
    case.config["train_schedule"][0]["loss"]["forces"] = 10.0
    result = run_train_case(tmp_path / case.name, case, reference_data=reference_data)
    assert result.returncode != 0
    assert "have a force loss term" in result.stderr + result.stdout


def test_a_force_error_table_without_forces_is_refused(tmp_path, reference_data):
    case = _energy_only_case("energy_only_with_force_table", error_table="PerAtomRMSE")
    result = run_train_case(tmp_path / case.name, case, reference_data=reference_data)
    assert result.returncode != 0
    assert "use error_table PerAtomEnergyRMSE" in result.stderr + result.stdout
