"""End-to-end synthetic fixtures never imply historical or production evidence."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor import resources
from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import swing_return_training as owner
from market_predictor.research.swing_return_inputs import ReturnInputs
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.training.return_artifacts import load_return_model
from market_predictor.swing.training.return_estimators import predict_return
from market_predictor.swing.training.return_validation import security_transfer_mask

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def training(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "data/runtime"))
    package = tmp_path / "src/market_predictor"
    for name in owner.IMPLEMENTATION:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, target)
    monkeypatch.setattr(owner, "__file__", str(package / "research/swing_return_training.py"))
    config = tmp_path / "config.json"
    shutil.copyfile(REPO / "configs/swing_return_training.json", config)
    dates = tuple(xcals.get_calendar("XNYS").sessions_in_range("2019-07-09", "2024-05-28").date)
    names = pd.Series([f"issuer-{i}" for i in range(20)])
    holdout = security_transfer_mask(names)
    identities = [names[holdout].iloc[0], *names[~holdout].iloc[:2]]
    metadata = pd.DataFrame([(day, name) for day in dates for name in identities], columns=["session_date_et", "security_id"])
    metadata["decision_id"] = [f"row-{i}" for i in range(len(metadata))]
    metadata["ticker"] = metadata.security_id
    metadata["sector"] = "test_sector"
    metadata["decision_time_utc"] = swing_prediction_cutoffs(metadata.session_date_et)
    ends = {day: xcals.get_calendar("XNYS").session_close(
        xcals.get_calendar("XNYS").sessions_window(pd.Timestamp(day), 11)[-1]) for day in dates}
    metadata["research_label_mature_at"] = pd.to_datetime(metadata.session_date_et.map(ends), utc=True)
    metadata["feature_eligible"] = True
    metadata.loc[metadata.index[::17], "feature_eligible"] = False
    metadata["fixed_horizon_supervision_available"] = metadata.research_label_mature_at.lt(metadata.decision_time_utc.max())
    metadata["security_holdout"] = security_transfer_mask(metadata.security_id)
    signal = np.sin(np.arange(len(metadata)) / 7)
    metadata["spy_fixed_horizon_excess_return"] = signal * .03
    metadata.loc[~metadata.fixed_horizon_supervision_available, "spy_fixed_horizon_excess_return"] = np.nan
    metadata.loc[~metadata.fixed_horizon_supervision_available, "research_label_mature_at"] = pd.NaT
    features = pd.DataFrame({"signal": signal, "empty": np.nan}, dtype=np.float32)
    memory_policy = TrainingReadinessPolicy.model_validate_json((REPO / "configs/swing_training_readiness.json").read_text())
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"synthetic_test": true}')
    data = ReturnInputs(features, metadata, dates, {"evidence.json": file_sha256(evidence)}, memory_policy,
        metadata.decision_time_utc.max())
    monkeypatch.setattr(owner, "load_return_inputs", lambda root, policy: data)
    monkeypatch.setattr(owner, "_guard", lambda policy: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    return dict(root=tmp_path, config=config, config_sha256=file_sha256(config), output=tmp_path / "data/research/run", data=data)


def run(state: dict[str, Any], resume: str | None = None) -> dict[str, Any]:
    return owner.train_swing_returns(**{name: state[name] for name in ("root", "config", "config_sha256", "output")},
        resume_checkpoint_sha256=resume)


def test_two_models_two_scopes_end_to_end_and_resume(training: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    result = run(training)
    assert result["final_models"] == 2 and result["fold_scope_fits"] == 16
    assert len(result["units"]) == 18 and result["serving_eligible"] is False
    assert result["portfolio_evaluated"] is False and result["historical_test_opened"] is False
    request = json.loads((training["output"] / "_request.json").read_text())
    holdout = set(request["holdout_security_ids"])
    assert [len(fold["score"]) for fold in request["folds"]] == [179, 179, 179, 181]
    for key, record in result["units"].items():
        directory = training["root"] / record["path"]
        unit = json.loads((directory / "_manifest.json").read_text())["unit"]
        if "security_transfer" in key:
            assert not holdout.intersection(unit["training_security_ids"])
        else:
            assert holdout.intersection(unit["training_security_ids"])
        model = load_return_model(directory, manifest_sha256=record["manifest_sha256"])
        assert np.isfinite(predict_return(model, training["data"].features.iloc[:4])).all()
        with pytest.raises(DataReadinessError, match="cannot authorize"):
            load_return_model(directory, manifest_sha256=record["manifest_sha256"], purpose="serving")
        if "final_refit" not in key:
            predictions = pd.read_parquet(directory / "predictions.parquet")
            assert len(predictions) == unit["scoring_rows"]
            assert predictions.predicted_excess_return.notna().equals(predictions.scope_eligible)
            if "fold-4" in key:
                assert (predictions.predicted_excess_return.notna() & ~predictions.fixed_horizon_supervision_available).any()
    checkpoint_hash = file_sha256(training["output"] / "_checkpoint.json")
    fit = Mock(side_effect=AssertionError("completed fitting must not repeat"))
    monkeypatch.setattr(owner, "fit_return_regressor", fit)
    assert run(training, checkpoint_hash) == result
    fit.assert_not_called()
    with pytest.raises(DataReadinessError, match="independently pinned"):
        run(training)
    target = training["root"] / next(iter(result["units"].values()))["path"] / "_manifest.json"
    target.write_text(target.read_text() + " ")
    with pytest.raises(DataReadinessError, match="pinned checkpoint"):
        run(training, checkpoint_hash)


def test_initial_failure_preserves_empty_resume_checkpoint(training: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, "fit_return_regressor", Mock(side_effect=MemoryBudgetError("test limit")))
    with pytest.raises(MemoryBudgetError):
        run(training)
    checkpoint = json.loads((training["output"] / "_checkpoint.json").read_text())
    assert checkpoint["units"] == {}
    assert not (training["output"] / "_manifest.json").exists()


def test_scipy_only_runtime_change_rejects_resume(training: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, "fit_return_regressor", Mock(side_effect=MemoryBudgetError("test limit")))
    with pytest.raises(MemoryBudgetError):
        run(training)
    version = owner.importlib.metadata.version
    monkeypatch.setattr(owner.importlib.metadata, "version", lambda name: "changed-scipy" if name == "scipy" else version(name))
    with pytest.raises(DataReadinessError, match="request differs"):
        run(training, file_sha256(training["output"] / "_checkpoint.json"))


def test_transient_peak_cannot_publish(training: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner.fit_return_regressor
    peak = [1 * 1024**3]
    monkeypatch.setattr(resources, "process_memory_snapshot", lambda: (1024**3, peak[0]))
    def spike(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        peak[0] = 6 * 1024**3
        return result
    monkeypatch.setattr(owner, "fit_return_regressor", spike)
    with pytest.raises(MemoryBudgetError, match="peak"):
        run(training)
    assert not list(training["output"].glob("**/model.joblib"))
    assert not (training["output"] / "_manifest.json").exists()


def test_busy_lease_blocks_before_config_read(training: dict[str, Any]) -> None:
    training["config"].unlink()
    with heavy_job_lease("test", runtime_dir=training["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            run(training)


def test_output_cannot_use_serving_registry(training: dict[str, Any]) -> None:
    training["output"] = training["root"] / "data/models/forbidden"
    with pytest.raises(DataReadinessError, match="never the serving registry"):
        run(training)


@pytest.mark.parametrize("field,value", [("folds", 3), ("holdout_fraction", .1), ("missingness", "drop_missing"),
    ("target", "future_return"), ("scope", "production"), ("unexpected", True)])
def test_experiment_changes_rejected(field: str, value: object) -> None:
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    payload[field] = value
    with pytest.raises(ValidationError):
        ReturnTrainingPolicy.model_validate(payload)
