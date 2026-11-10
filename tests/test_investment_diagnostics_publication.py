"""Synthetic saved-run proof checks; no estimator or original raw source is loaded."""
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.dataset import InvestmentDataset
from market_predictor.investment.training import (
    FAMILIES,
    HORIZONS,
    PROFILES,
    SCOPES,
    SCORE_YEARS,
    TARGET,
    investment_diagnostics,
    investment_fold,
)
from market_predictor.research import investment_diagnostics as module
from market_predictor.research import investment_features
from market_predictor.research.investment_training import FLAGS, _fold_record
from market_predictor.swing.training.return_validation import security_transfer_mask


def write(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str), encoding="utf-8")
    return {"path": str(path), "sha256": file_sha256(path)}


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_, **__: nullcontext())
    monkeypatch.setattr(module, "guard", lambda *_: None)
    monkeypatch.setattr(module, "_implementation", lambda: {})
    monkeypatch.setattr(investment_features, "guard", lambda *_: None)
    candidates = pd.Series([f"issuer:{i}" for i in range(200)])
    held = security_transfer_mask(candidates)
    securities = candidates.loc[~held].head(24).tolist() + candidates.loc[held].head(12).tolist()
    calendar = xcals.get_calendar("XNYS")
    dates = [calendar.date_to_session(str(stamp.date()), direction="previous")
        for stamp in pd.date_range("2019-07-31", "2024-04-30", freq="ME")]
    dates.append(pd.Timestamp("2024-05-28"))
    records = []
    for day in dates:
        for horizon in HORIZONS:
            end = calendar.session_close(calendar.sessions_window(day, horizon + 1)[-1])
            for index, security in enumerate(securities):
                known = index != 0
                actual = (index - 16) / 1600 if known else np.nan
                records.append({"decision_id": json_sha256({"security_id": security, "decision_session": day.date().isoformat()}),
                    "security_id": security, "decision_session": day.date(), "decision_time_utc": calendar.session_close(day),
                    "horizon_sessions": horizon, "sector": "A" if index % 2 else "B", "feature_eligible": True,
                    "research_eligible": known, "economic_maturity_at_utc": end,
                    "label_available_at_utc": pd.Timestamp("2025-12-31T00:00:00Z") if known else pd.NaT,
                    TARGET: actual, "stock_net_return": actual + 0.05, "spy_gross_return": 0.05 if known else np.nan,
                    "qqq_gross_return": 0.07 if known else np.nan, "sector_gross_return": 0.06 if known else np.nan})
    metadata = pd.DataFrame(records)
    names = tuple(f"feature-{i}" for i in range(215))
    features = pd.DataFrame(0.0, index=metadata.index, columns=names)
    source = write(tmp_path / "source.json", {"fixture": "pinned admitted saved data"})
    dataset = InvestmentDataset(features, metadata, {PROFILES[0]: names[:124], PROFILES[1]: names},
        {source["path"]: source["sha256"]}, datetime(2026, 1, 1, tzinfo=UTC), "f" * 64)
    monkeypatch.setattr(module, "load_investment_dataset", lambda **_: dataset)
    dataset_pin = write(tmp_path / "dataset.json", {"fixture": "dataset"})
    admission_pin = write(tmp_path / "admission.json", {"dataset": dataset_pin})
    settings_pin = write(tmp_path / "settings.json", {"fixture": "closed fixed settings"})
    train_request = write(tmp_path / "training-request.json", {"schema_version": "market_predictor.investment_training_request",
        "scope": "retrospective_research", "dataset_manifest": dataset_pin, "dataset_admission": admission_pin,
        "estimator_settings": settings_pin})
    folds = [investment_fold(metadata, h, year, scope) for h in HORIZONS for scope in SCOPES for year in SCORE_YEARS]
    assert all(not fold.missing_reasons for fold in folds)
    fold_records = [_fold_record(fold, metadata) for fold in folds]
    experiment = {"request_sha256": train_request["sha256"], "model_columns": dataset.model_columns,
        "feature_contract_sha256": dataset.feature_contract_sha256, "parameters": {family: {"fixed": 42} for family in FAMILIES},
        "folds": fold_records, **FLAGS}
    directory = tmp_path / "run"
    experiment_pin = write(directory / "experiment.json", experiment)
    units: dict[str, Any] = {}
    aggregates = {}
    number = 0
    for horizon in HORIZONS:
        for profile in PROFILES:
            for family in FAMILIES:
                for scope in SCOPES:
                    collected = []
                    for fold in folds:
                        if fold.horizon_sessions != horizon or fold.scope != scope:
                            continue
                        key = f"{horizon}/{profile}/{family}/{scope}/{fold.score_year}"
                        train = metadata.iloc[fold.train_indices]
                        saved = metadata.iloc[fold.score_indices].loc[:, list(module.CONTEXT)].copy()
                        # A saved arbitrary forecast, never a fitted or score-tuned model.
                        saved[module.PREDICTION] = 0.002 if profile == PROFILES[0] else 0.003
                        saved["actual_fit_started_at_utc"] = pd.Timestamp("2026-01-02T00:00:00Z")
                        saved["actual_fitted_at_utc"] = pd.Timestamp("2026-01-02T00:00:01Z")
                        saved["prediction_generated_at_utc"] = pd.Timestamp("2026-01-02T00:00:02Z")
                        saved["evidence_scope"] = "retrospective_research"
                        saved["prediction_missing_reason"] = None
                        unit_path = directory / f"unit-{number}"
                        unit_path.mkdir(parents=True)
                        number += 1
                        saved.to_parquet(unit_path / "predictions.parquet", index=False)
                        (unit_path / "model.joblib").write_bytes(b"not a model; must never be deserialized")
                        identity = {**_fold_record(fold, metadata), "profile": profile, "experiment_sha256": experiment_pin["sha256"]}
                        record = {"schema": "market_predictor.investment_research_model", "scope": "retrospective_research",
                            "request_sha256": train_request["sha256"], "unit": identity, "family": family,
                            "parameters": experiment["parameters"][family], "feature_names": dataset.model_columns[profile],
                            "actual_fit_started_at_utc": "2026-01-02T00:00:00Z", "actual_fitted_at_utc": "2026-01-02T00:00:01Z",
                            "dataset_admitted_at_utc": dataset.admitted_at_utc.isoformat(),
                            "latest_training_label_available_at_utc": train.label_available_at_utc.max().isoformat(),
                            "latest_training_economic_maturity_at_utc": train.economic_maturity_at_utc.max().isoformat(),
                            "training_rows": len(train), "training_decision_ids_sha256": identity["training_decision_ids_sha256"],
                            "training_security_ids": identity["training_security_ids"], "metrics": investment_diagnostics(saved),
                            "files": {name: file_sha256(unit_path / name) for name in ("model.joblib", "predictions.parquet")}, **FLAGS}
                        units[key] = write(unit_path / "_manifest.json", record)
                        collected.append(saved)
                    aggregates[f"{horizon}/{profile}/{family}/{scope}"] = investment_diagnostics(pd.concat(collected, ignore_index=True))
                units[f"{horizon}/{profile}/{family}/final_refit"] = write(directory / f"final-{number}.json", {"not_loaded": True})
                number += 1
    manifest = {"schema": "market_predictor.investment_training", "status": "complete_research_only",
        "scope": "retrospective_research", "request_sha256": train_request["sha256"], "experiment_sha256": experiment_pin["sha256"],
        "fold_scope_fits": 48, "final_models": 8, "units": units, "aggregate_metrics": aggregates, "folds": fold_records,
        "source_files": {"old/raw/provenance/not/reopened": "b" * 64}, "completed_at_utc": "2026-01-04T00:00:00Z", **FLAGS}
    manifest_pin = write(directory / "_manifest.json", manifest)
    request = {"schema_version": "market_predictor.investment_diagnostics_request", "scope": "saved_retrospective_predictions",
        "training_manifest": manifest_pin, "training_request": train_request}
    request_pin = write(tmp_path / "request.json", request)
    return {"request": request, "request_pin": request_pin, "manifest": manifest, "dataset": dataset,
        "manifest_path": directory / "_manifest.json", "output": tmp_path / "diagnostic"}


def run(root: Path, value: dict[str, Any]) -> dict[str, Any]:
    return module.diagnose_investment_predictions(root=root, request=Path(value["request_pin"]["path"]),
        request_sha256=value["request_pin"]["sha256"], output=value["output"])


def test_all_saved_units_are_explained_with_exact_metrics_and_no_model_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = fixture(tmp_path, monkeypatch)
    before = {str(path): file_sha256(path) for path in (tmp_path / "run").rglob("*") if path.is_file()}
    run(tmp_path, value)
    report = json.loads((value["output"] / "_manifest.json").read_text())
    assert len(report["unit_diagnostics"]) == 48 and len(report["aggregate_diagnostics"]) == 16
    assert len(report["paired_diagnostics"]["units"]) == 24 and len(report["paired_diagnostics"]["aggregates"]) == 8
    assert not report["models_loaded"] and not report["models_refitted"] and not report["calibrator_fitted"]
    assert all(file_sha256(Path(path)) == digest for path, digest in before.items())
    assert not (tmp_path / "old/raw/provenance/not/reopened").exists()


@pytest.mark.parametrize("poison", ["extra_unit", "missing_aggregate", "prediction_hash", "identity", "saved_metric", "fit_clock",
    "self_consistent_backdate", "generation_after_completion"])
def test_saved_artifact_or_identity_conflict_refuses_diagnostics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str) -> None:
    value = fixture(tmp_path, monkeypatch)
    manifest = value["manifest"]
    key = next(key for key in manifest["units"] if not key.endswith("final_refit"))
    pin = manifest["units"][key]
    record_path = Path(pin["path"])
    record = json.loads(record_path.read_text())
    predictions_path = record_path.parent / "predictions.parquet"
    if poison == "extra_unit":
        manifest["units"]["unexpected"] = pin
    elif poison == "missing_aggregate":
        manifest["aggregate_metrics"].pop(next(iter(manifest["aggregate_metrics"])))
    elif poison == "prediction_hash":
        predictions_path.write_bytes(b"tampered")
    elif poison == "identity":
        frame = pd.read_parquet(predictions_path)
        frame.loc[0, "security_id"] = "foreign"
        frame.to_parquet(predictions_path, index=False)
        record["files"]["predictions.parquet"] = file_sha256(predictions_path)
    elif poison == "saved_metric":
        record["metrics"]["mse"] += 0.01
    elif poison == "fit_clock":
        record["actual_fitted_at_utc"] = "2026-01-02T00:00:03Z"
    elif poison in ("self_consistent_backdate", "generation_after_completion"):
        frame = pd.read_parquet(predictions_path)
        if poison == "self_consistent_backdate":
            record["actual_fit_started_at_utc"] = "2025-12-30T00:00:00Z"
            record["actual_fitted_at_utc"] = "2025-12-30T00:00:01Z"
            frame["actual_fit_started_at_utc"] = pd.Timestamp(record["actual_fit_started_at_utc"])
            frame["actual_fitted_at_utc"] = pd.Timestamp(record["actual_fitted_at_utc"])
            frame["prediction_generated_at_utc"] = pd.Timestamp("2025-12-30T00:00:02Z")
        else:
            frame["prediction_generated_at_utc"] = pd.Timestamp("2026-01-05T00:00:00Z")
        frame.to_parquet(predictions_path, index=False)
        record["files"]["predictions.parquet"] = file_sha256(predictions_path)
    if poison in ("identity", "saved_metric", "fit_clock", "self_consistent_backdate", "generation_after_completion"):
        manifest["units"][key] = write(record_path, record)
    value["request"]["training_manifest"] = write(value["manifest_path"], manifest)
    value["request_pin"] = write(Path(value["request_pin"]["path"]), value["request"])
    with pytest.raises(DataReadinessError):
        run(tmp_path, value)
    assert not value["output"].exists()


def test_guard_refuses_before_any_input_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_, **__: nullcontext())
    monkeypatch.setattr(module, "_implementation", lambda: pytest.fail("input preparation occurred before guard"))

    def refuse(*_: Any) -> None:
        raise MemoryBudgetError("synthetic pressure")

    monkeypatch.setattr(module, "guard", refuse)
    with pytest.raises(MemoryBudgetError, match="pressure"):
        module.diagnose_investment_predictions(root=tmp_path, request=tmp_path / "unread.json",
            request_sha256="a" * 64, output=tmp_path / "output")


def test_cli_forwards_exact_diagnostic_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import typer
    from typer.testing import CliRunner

    from market_predictor.commands import investment_research as commands

    calls = []
    monkeypatch.setattr(commands, "diagnose_investment_predictions", lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    app = typer.Typer()
    commands.register_investment_research_commands(app)
    result = CliRunner().invoke(app, ["diagnose-investment-predictions", "--root", str(tmp_path), "--request",
        str(tmp_path / "request.json"), "--request-sha256", "a" * 64, "--output", str(tmp_path / "output")])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "request": tmp_path / "request.json", "request_sha256": "a" * 64,
        "output": tmp_path / "output"}]
