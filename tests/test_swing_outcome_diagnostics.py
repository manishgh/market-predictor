"""Synthetic UNIT evidence for retained-outcome diagnostics, not market performance."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.swing.training import outcome_diagnostics as owner
from market_predictor.swing.training.return_validation import regression_diagnostics


def predictions() -> pd.DataFrame:
    frame = pd.DataFrame({"decision_id": ["a", "b", "c", "d", "e", "f"],
        "security_id": ["s", "t", "u", "s", "t", "u"],
        "session_date_et": [date(2024, 1, 2)] * 3 + [date(2024, 1, 3)] * 3,
        "feature_eligible": True, "security_holdout": [True, False, False] * 2,
        "scope_eligible": [True] * 5 + [False],
        "fixed_horizon_supervision_available": [True, False, False, False, False, False],
        owner.TARGET: [0.01, 0.02, np.nan, np.nan, 0.03, np.nan],
        "predicted_excess_return": [0.01] * 5 + [np.nan],
        "prediction_status": ["research_prediction"] * 5 + ["outside_eligible_scope"]})
    for column in owner.REASON_COLUMNS:
        frame[column] = "[]"
    frame.loc[1, "qqq_missing_reasons"] = '["missing_quote"]'
    frame.loc[2, "stock_missing_reasons"] = '["initial_fit_terminal_immature"]'
    frame.loc[2, "spy_missing_reasons"] = '["initial_fit_terminal_immature"]'
    frame.loc[3, "stock_missing_reasons"] = '["missing_quote", "missing_quote", "corporate_terms"]'
    return frame


def test_existing_mask_disjoint_counts_overlapping_reasons_and_no_mutation() -> None:
    frame = predictions()
    original = frame.copy(deep=True)
    metrics = regression_diagnostics(frame, owner.TARGET)
    result = owner.summarize_outcome_availability(frame)
    summary = result["summary"]
    assert summary["rows"] == 6 and summary["scored_rows"] == 5 and summary["unscored_rows"] == 1
    assert summary["known_scored_outcomes"] == 1 and summary["unknown_scored_outcomes"] == 4
    assert summary["scored_target_missing"] == 2
    assert summary["scored_target_present_supervision_unavailable"] == 2
    assert summary["scored_target_missing"] + summary["scored_target_present_supervision_unavailable"] == summary["unknown_scored_outcomes"]
    assert summary["known_scored_outcomes"] + summary["unknown_scored_outcomes"] == summary["scored_rows"]
    assert summary["scored_rows"] + summary["unscored_rows"] == summary["rows"]
    assert summary["unknown_rows_without_recorded_reasons"] == 1
    assert summary["unknown_rows_by_reason_role"] == {"stock": 2, "spy": 1, "qqq": 1, "sector": 0}
    assert summary["recorded_reason_counts"]["stock"] == {
        "corporate_terms": 1, "initial_fit_terminal_immature": 1, "missing_quote": 1}
    assert sum(row["scored_rows"] for row in result["sessions"]) == 5
    assert sum(row["unknown_scored_outcomes"] for row in result["securities"]) == 4
    for group in ("sessions", "securities"):
        for name in owner.COUNT_NAMES:
            assert sum(row[name] for row in result[group]) == summary[name]
    pd.testing.assert_frame_equal(frame, original)
    assert regression_diagnostics(frame, owner.TARGET) == metrics
    for name in ("rows", "scored_rows", "known_scored_outcomes", "unknown_scored_outcomes"):
        assert summary[name] == metrics[name]


def test_no_scored_rows_and_reordered_input_keep_denominators() -> None:
    frame = predictions()
    frame["scope_eligible"] = False
    frame["predicted_excess_return"] = np.nan
    frame["prediction_status"] = "outside_eligible_scope"
    result = owner.summarize_outcome_availability(frame)
    assert result["summary"]["rows"] == result["summary"]["unscored_rows"] == 6
    assert result["summary"]["unknown_scored_outcomes"] == 0
    assert owner.summarize_outcome_availability(frame.iloc[::-1]) == result


@pytest.mark.parametrize("reason", [None, "null", "[", "{}", '[""]', '[1]', '[" "]'])
def test_malformed_reasons_refused_even_for_unscored_rows(reason: object) -> None:
    frame = predictions()
    frame.loc[5, "sector_missing_reasons"] = reason
    with pytest.raises(DataReadinessError, match="reason arrays"):
        owner.summarize_outcome_availability(frame)


@pytest.mark.parametrize("change", ["duplicate", "date", "boolean", "target", "prediction", "scope", "status", "missing_column"])
def test_invalid_saved_metadata_refused(change: str) -> None:
    frame = predictions()
    if change == "duplicate":
        frame.loc[1, "decision_id"] = "a"
    elif change == "date":
        frame.loc[0, "session_date_et"] = "2024-01-02"
    elif change == "boolean":
        frame["scope_eligible"] = frame.scope_eligible.astype(object)
        frame.loc[0, "scope_eligible"] = 1
    elif change == "target":
        frame.loc[0, owner.TARGET] = np.inf
    elif change == "prediction":
        frame.loc[0, "predicted_excess_return"] = np.inf
    elif change == "scope":
        frame.loc[0, "scope_eligible"] = False
    elif change == "status":
        frame.loc[0, "prediction_status"] = "wrong"
    else:
        frame = frame.drop(columns="stock_missing_reasons")
    with pytest.raises(DataReadinessError):
        owner.summarize_outcome_availability(frame)


def save(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_sha256(path)


@pytest.fixture
def saved_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    # Only UNIT fixtures bypass memory telemetry; the real-data publisher never does.
    monkeypatch.setattr(owner, "guard", lambda *args: None)
    root = tmp_path / "source"
    directory = root / "data/research/retained"
    folds = [{"number": number, "score": [(date(2024, 1, 2) + timedelta(days=2 * (number - 1) + offset)).isoformat()
        for offset in range(2)]} for number in range(1, 5)]
    request = {"schema": "market_predictor.swing_return_training_request", "policy": {
        "target": owner.TARGET, "scope": "initial_fit_research_only"}, "feature_names": ["feature"],
        "folds": folds, "holdout_security_ids": ["s"], **dict.fromkeys(owner.FLAGS, False)}
    request_sha = save(directory / "_request.json", request)
    records = {}
    for family in owner.FAMILIES:
        records[f"{family}/final_refit"] = {}
        for scope in owner.SCOPES:
            for fold in folds:
                key = f"{family}/{scope}/fold-{fold['number']}"
                unit_directory = directory / key
                unit_directory.mkdir(parents=True)
                frame = predictions()
                frame["decision_id"] = frame.decision_id + str(fold["number"])
                frame["session_date_et"] = [date.fromisoformat(day) for day in fold["score"] for _ in range(3)]
                frame.loc[5, "feature_eligible"] = False
                if scope == "security_transfer":
                    frame["scope_eligible"] = frame.feature_eligible & frame.security_holdout
                    frame.loc[~frame.scope_eligible, "predicted_excess_return"] = np.nan
                    frame.loc[~frame.scope_eligible, "prediction_status"] = "outside_eligible_scope"
                frame.to_parquet(unit_directory / "predictions.parquet", index=False)
                (unit_directory / "model.joblib").write_bytes(b"UNIT fixture: never deserialize")
                unit = {"family": family, "scope": scope, "target": owner.TARGET, "feature_names": ["feature"],
                    "scoring_rows": len(frame), "scoring_decision_ids_sha256": json_sha256(sorted(frame.decision_id)),
                    "scope_eligible_rows": int(frame.scope_eligible.sum())}
                manifest = {"schema": "market_predictor.swing_return_research_model", "unit": unit,
                    "request_sha256": request_sha, "serving_eligible": False, "promotion_eligible": False,
                    "metrics": regression_diagnostics(frame, owner.TARGET),
                    "files": {name: file_sha256(unit_directory / name) for name in ("predictions.parquet", "model.joblib")}}
                sha = save(unit_directory / "_manifest.json", manifest)
                records[key] = {"path": unit_directory.relative_to(root).as_posix(), "manifest_sha256": sha}
    manifest = {"schema": "market_predictor.swing_return_training", "status": "complete_research_only",
        "request_sha256": request_sha, "units": records, **dict.fromkeys(owner.FLAGS, False)}
    return {"root": root, "directory": directory.relative_to(root),
        "manifest_sha256": save(directory / "_manifest.json", manifest), "output": tmp_path / "reports/audit.json"}


def test_publisher_reconciles_folds_keeps_sources_separate_and_refuses_overwrite(saved_run: dict[str, Any]) -> None:
    result = owner.audit_saved_outcomes(**saved_run)
    assert result["summary"]["scored_rows"] == 20
    assert result["summary"]["unknown_scored_outcomes"] == 16
    assert len(result["sessions"]) == 8 and len(result["securities"]) == 3
    assert result["reason_counts_overlap"] is True and result["new_fit_authorized"] is False
    assert all(result[flag] is False for flag in owner.FLAGS)
    assert all(Path(name).is_absolute() for name in result["implementation_files"])
    assert all(not Path(name).is_absolute() for name in result["source_files"])
    assert {"python", "numpy", "pandas", "pyarrow"} == set(result["runtime"])
    assert any(name.endswith("heavy_jobs.py") for name in result["implementation_files"])
    assert file_sha256(saved_run["output"]) == result["report_sha256"]
    with pytest.raises(DataReadinessError, match="new JSON"):
        owner.audit_saved_outcomes(**saved_run)


def test_security_transfer_audits_only_saved_holdout_scope(saved_run: dict[str, Any]) -> None:
    result = owner.audit_saved_outcomes(**saved_run, scope="security_transfer")
    assert result["summary"]["scored_rows"] == 8 and result["summary"]["known_scored_outcomes"] == 4


def test_tampered_model_is_detected_without_deserialization(saved_run: dict[str, Any]) -> None:
    path = saved_run["root"] / saved_run["directory"] / "regularized_linear_return/temporal/fold-1/model.joblib"
    path.write_bytes(b"tampered")
    with pytest.raises(DataReadinessError, match="artifact changed"):
        owner.audit_saved_outcomes(**saved_run)
    assert not saved_run["output"].exists()


def test_expanded_parquet_limit_refuses_publication(saved_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, "MAX_EXPANDED_BYTES", 1)
    with pytest.raises(DataReadinessError, match="expanded row or byte"):
        owner.audit_saved_outcomes(**saved_run)
    assert not saved_run["output"].exists()


@pytest.mark.parametrize("field,value", [("scope_eligible_rows", 1), ("scoring_rows", 1), ("scope", "security_transfer")])
def test_pinned_but_conflicting_unit_refuses_publication(saved_run: dict[str, Any], field: str, value: object) -> None:
    directory = saved_run["root"] / saved_run["directory"]
    path = directory / "regularized_linear_return/temporal/fold-1/_manifest.json"
    unit = json.loads(path.read_text())
    unit["unit"][field] = value
    run = json.loads((directory / "_manifest.json").read_text())
    run["units"]["regularized_linear_return/temporal/fold-1"]["manifest_sha256"] = save(path, unit)
    saved_run["manifest_sha256"] = save(directory / "_manifest.json", run)
    with pytest.raises(DataReadinessError):
        owner.audit_saved_outcomes(**saved_run)
    assert not saved_run["output"].exists()


def test_busy_lease_prevents_any_artifact_reads(saved_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = saved_run["root"] / "runtime"
    monkeypatch.setattr(owner, "heavy_job_runtime_dir", lambda: runtime)

    def unexpected_read(*args: Any, **kwargs: Any) -> None:
        pytest.fail("an artifact was read before the shared lease")

    monkeypatch.setattr(owner, "verify_completed_return_run", unexpected_read)
    with owner.heavy_job_lease("UNIT competing owner", runtime_dir=runtime):
        with pytest.raises(HeavyJobBusyError):
            owner.audit_saved_outcomes(**saved_run)
    assert not saved_run["output"].exists()


def test_source_mutation_during_final_guard_prevents_publication(saved_run: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    path = saved_run["root"] / saved_run["directory"] / "_request.json"

    def mutate_during_pressure_pause(stage: str) -> None:
        if stage == "saved outcome audit publication":
            path.write_text("changed after consumption", encoding="utf-8")

    monkeypatch.setattr(owner, "guard", mutate_during_pressure_pause)
    with pytest.raises(DataReadinessError, match="source changed before publication"):
        owner.audit_saved_outcomes(**saved_run)
    assert not saved_run["output"].exists()


def test_pinned_holdout_mismatch_prevents_publication(saved_run: dict[str, Any]) -> None:
    directory = saved_run["root"] / saved_run["directory"]
    key = "regularized_linear_return/temporal/fold-1"
    prediction_path = directory / key / "predictions.parquet"
    frame = pd.read_parquet(prediction_path)
    frame.loc[1, "security_holdout"] = True
    frame.to_parquet(prediction_path, index=False)
    unit_path = directory / key / "_manifest.json"
    unit = json.loads(unit_path.read_text())
    unit["files"]["predictions.parquet"] = file_sha256(prediction_path)
    manifest = json.loads((directory / "_manifest.json").read_text())
    manifest["units"][key]["manifest_sha256"] = save(unit_path, unit)
    saved_run["manifest_sha256"] = save(directory / "_manifest.json", manifest)
    with pytest.raises(DataReadinessError, match="holdout assignment"):
        owner.audit_saved_outcomes(**saved_run)
    assert not saved_run["output"].exists()


def test_overlapping_frozen_fold_sessions_prevent_double_counting(saved_run: dict[str, Any]) -> None:
    directory = saved_run["root"] / saved_run["directory"]
    request = json.loads((directory / "_request.json").read_text())
    request["folds"][1]["score"] = request["folds"][0]["score"]
    request_sha = save(directory / "_request.json", request)
    manifest = json.loads((directory / "_manifest.json").read_text())
    manifest["request_sha256"] = request_sha
    for key, record in manifest["units"].items():
        if key.endswith("final_refit"):
            continue
        unit_path = directory / key / "_manifest.json"
        unit = json.loads(unit_path.read_text())
        unit["request_sha256"] = request_sha
        if key.endswith("fold-2"):
            prediction_path = directory / key / "predictions.parquet"
            frame = pd.read_parquet(prediction_path)
            frame["session_date_et"] = [date.fromisoformat(day) for day in request["folds"][1]["score"] for _ in range(3)]
            frame.to_parquet(prediction_path, index=False)
            unit["files"]["predictions.parquet"] = file_sha256(prediction_path)
        record["manifest_sha256"] = save(unit_path, unit)
    saved_run["manifest_sha256"] = save(directory / "_manifest.json", manifest)
    with pytest.raises(DataReadinessError, match="folds overlap"):
        owner.audit_saved_outcomes(**saved_run)
    assert not saved_run["output"].exists()
