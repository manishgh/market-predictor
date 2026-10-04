"""Paired numerical fixtures and immutable historical-score integration checks."""
from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.training import return_comparison as owner


def predictions() -> pd.DataFrame:
    return pd.DataFrame({"decision_id": ["a", "b", "c", "d"], "security_id": ["s", "t", "s", "t"],
        "session_date_et": [date(2024, 1, 2)] * 2 + [date(2024, 1, 3)] * 2,
        "feature_eligible": True, "security_holdout": [True, False, True, False], "scope_eligible": True,
        "fixed_horizon_supervision_available": [True, True, True, False],
        owner.TARGET: [1.0, 3.0, 2.0, np.nan], "predicted_excess_return": [0.0, 0.0, 0.0, 0.0],
        "prediction_status": "research_prediction"})


def test_equal_date_errors_unknown_rows_and_row_permutation() -> None:
    baseline = predictions()
    candidate = baseline.copy()
    candidate.predicted_excess_return = [1.0, 3.0, 2.0, 0.0]
    result = owner.summarize_session_errors(owner.paired_session_errors(baseline, candidate.iloc[::-1]))
    assert result["baseline_mse"] == 4.5  # (mean(1, 9) + 4) / 2, not mean(1, 9, 4).
    assert result["candidate_mse"] == 0 and result["paired_mse_difference"] == -4.5
    assert result["baseline_mae"] == 2 and result["candidate_mae"] == 0
    assert result["known_scored_outcomes"] == 3 and result["unknown_scored_outcomes"] == 1
    assert result["baseline_rank_sessions"] == 0 and result["candidate_rank_sessions"] == 1
    assert result["paired_rank_sessions"] == 0 and result["paired_rank_difference"] is None


def test_identical_tied_ranks_and_unequal_fold_lengths() -> None:
    frame = predictions()
    frame.predicted_excess_return = [1.0, 2.0, 1.0, 0.0]
    one = owner.paired_session_errors(frame, frame)
    two = one.iloc[:1].copy()
    two["session"] = "2024-01-04"
    result = owner.summarize_session_errors(pd.concat([one, two]))
    assert result["baseline_mse"] == pytest.approx(2 / 3)
    assert result["paired_mse_difference"] == result["paired_mae_difference"] == 0
    assert result["paired_rank_difference"] == 0 and result["paired_rank_sessions"] == 2


def test_no_known_outcomes_keeps_null_metrics_and_counts() -> None:
    frame = predictions()
    frame["fixed_horizon_supervision_available"] = False
    result = owner.summarize_session_errors(owner.paired_session_errors(frame, frame))
    assert result["baseline_mse"] is result["paired_mse_difference"] is None
    assert result["error_sessions"] == 0 and result["unknown_scored_outcomes"] == 4


@pytest.mark.parametrize("change", ["duplicate", "missing", "target", "null", "eligibility", "security", "session",
    "infinite_prediction", "infinite_target", "status", "null_flag"])
def test_mismatched_population_and_metadata_rejected(change: str) -> None:
    base, proposed = predictions(), predictions()
    if change == "duplicate":
        proposed.loc[1, "decision_id"] = "a"
    elif change == "missing":
        proposed = proposed.iloc[:-1]
    else:
        column, value = {"target": (owner.TARGET, 7.0), "null": (owner.TARGET, np.nan),
            "eligibility": ("feature_eligible", False), "security": ("security_id", "other"),
            "session": ("session_date_et", date(2024, 1, 4)), "infinite_prediction": ("predicted_excess_return", np.inf),
            "infinite_target": (owner.TARGET, np.inf), "status": ("prediction_status", "invalid"),
            "null_flag": ("scope_eligible", None)}[change]
        if change == "null_flag":
            proposed["scope_eligible"] = proposed.scope_eligible.astype(object)
        proposed.loc[0, column] = value
    with pytest.raises(DataReadinessError):
        owner.paired_session_errors(base, proposed)


def test_overlapping_fold_sessions_rejected() -> None:
    one = owner.paired_session_errors(predictions(), predictions())
    with pytest.raises(DataReadinessError, match="disjoint"):
        owner.summarize_session_errors(pd.concat([one, one]))


def test_average_tie_ranks() -> None:
    frame = predictions().iloc[:3].copy()
    frame["session_date_et"] = date(2024, 1, 2)
    frame[owner.TARGET] = [1.0, 1.0, 3.0]
    frame["predicted_excess_return"] = [2.0, 2.0, 7.0]
    result = owner.summarize_session_errors(owner.paired_session_errors(frame, frame))
    assert result["baseline_rank"] == pytest.approx(1.0)
    assert result["paired_rank_difference"] == 0


def save(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return file_sha256(path)


@pytest.fixture
def runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(owner, "_guard", lambda: None)
    root = tmp_path.resolve()
    source = Path(owner.__file__)
    bound = root / "src/market_predictor/swing/training"
    bound.mkdir(parents=True)
    for name in ("return_comparison.py", "return_artifacts.py"):
        shutil.copyfile(source.with_name(name), bound / name)
    monkeypatch.setattr(owner, "__file__", str(bound / source.name))
    kwargs: dict[str, Any] = {"root": root, "output": Path("data/reports/comparison.json")}
    for arm in ("baseline", "candidate"):
        directory = root / "data/research" / arm
        folds = [{"number": i, "train": ["2023-01-03"], "embargo": [],
            "score": [(date(2024, 1, 2) + timedelta(days=2 * (i - 1) + j)).isoformat() for j in range(2)]}
            for i in range(1, 5)]
        request = {"schema": "market_predictor.swing_return_training_request", "policy": {
            "target": owner.TARGET, "scope": "initial_fit_research_only", "weighting": "equal_date_total_mean_row_one",
            "feature_profile": arm}, "feature_names": [arm], "folds": folds, "holdout_security_ids": ["s"],
            "input_decision_ids_sha256": "population", "input_rows": 16, "versions": {}, "memory_policy": {},
            **dict.fromkeys(owner.FLAGS, False)}
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
                    frame["session_date_et"] = [date.fromisoformat(day) for day in fold["score"] for _ in range(2)]
                    if scope == "security_transfer":
                        frame["scope_eligible"] = frame.security_holdout
                        frame.loc[~frame.scope_eligible, "predicted_excess_return"] = np.nan
                        frame.loc[~frame.scope_eligible, "prediction_status"] = "outside_eligible_scope"
                    frame.to_parquet(unit_directory / "predictions.parquet", index=False)
                    (unit_directory / "model.joblib").write_bytes(b"never deserialize")
                    unit = {"family": family, "scope": scope, "target": owner.TARGET, "feature_names": [arm],
                        "scoring_rows": len(frame), "scoring_decision_ids_sha256": json_sha256(sorted(frame.decision_id)),
                        "scope_eligible_rows": int(frame.scope_eligible.sum()),
                        "training_weights_sha256": "unchanged"}
                    manifest = {"schema": "market_predictor.swing_return_research_model", "unit": unit,
                        "request_sha256": request_sha, "serving_eligible": False, "promotion_eligible": False,
                        "files": {name: file_sha256(unit_directory / name) for name in ("predictions.parquet", "model.joblib")}}
                    digest = save(unit_directory / "_manifest.json", manifest)
                    records[key] = {"path": unit_directory.relative_to(root).as_posix(), "manifest_sha256": digest}
        manifest = {"schema": "market_predictor.swing_return_training", "status": "complete_research_only",
            "request_sha256": request_sha, "units": records, **dict.fromkeys(owner.FLAGS, False)}
        kwargs[arm] = directory.relative_to(root)
        kwargs[f"{arm}_sha256"] = save(directory / "_manifest.json", manifest)
    return kwargs


def test_saved_scores_integration_and_immutable_report(runs: dict[str, Any]) -> None:
    result = owner.compare_saved_returns(**runs)
    assert len(result["groups"]) == 4 and result["new_fit_authorized"] is False
    assert result["groups"]["regularized_linear_return/temporal"]["summary"]["paired_mse_difference"] == 0
    assert file_sha256(runs["root"] / runs["output"]) == result["report_sha256"]
    assert result["arms"]["baseline"]["directory"] == runs["baseline"].as_posix()
    with pytest.raises(DataReadinessError, match="new report"):
        owner.compare_saved_returns(**runs)


def test_swapped_arms_are_explicit(runs: dict[str, Any]) -> None:
    swapped = {**runs, "baseline": runs["candidate"], "baseline_sha256": runs["candidate_sha256"],
        "candidate": runs["baseline"], "candidate_sha256": runs["baseline_sha256"]}
    result = owner.compare_saved_returns(**swapped)
    assert result["arms"]["baseline"]["directory"] == runs["candidate"].as_posix()
    assert result["arms"]["candidate"]["manifest_sha256"] == runs["baseline_sha256"]


def test_both_arms_share_contradictory_holdout_rejected(runs: dict[str, Any]) -> None:
    for arm in ("baseline", "candidate"):
        directory = runs["root"] / runs[arm]
        request = json.loads((directory / "_request.json").read_text())
        request["holdout_security_ids"] = ["t"]
        request_sha = save(directory / "_request.json", request)
        manifest = json.loads((directory / "_manifest.json").read_text())
        manifest["request_sha256"] = request_sha
        for key, record in manifest["units"].items():
            if key.endswith("final_refit"):
                continue
            path = directory / key / "_manifest.json"
            unit = json.loads(path.read_text())
            unit["request_sha256"] = request_sha
            record["manifest_sha256"] = save(path, unit)
        runs[f"{arm}_sha256"] = save(directory / "_manifest.json", manifest)
    with pytest.raises(DataReadinessError, match="transfer scope differs"):
        owner.compare_saved_returns(**runs)
    assert not (runs["root"] / runs["output"]).exists()


def test_artifact_tamper_rejected_before_report(runs: dict[str, Any]) -> None:
    path = runs["root"] / runs["candidate"] / "regularized_linear_return/temporal/fold-1/model.joblib"
    path.write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="artifact changed"):
        owner.compare_saved_returns(**runs)
    assert not (runs["root"] / runs["output"]).exists()


@pytest.mark.parametrize("change", ["holdout", "weighting", "unit_weights", "transfer_scope"])
def test_pinned_but_unpaired_run_rejected(runs: dict[str, Any], change: str) -> None:
    directory = runs["root"] / runs["candidate"]
    manifest = json.loads((directory / "_manifest.json").read_text())
    if change in {"holdout", "weighting"}:
        request = json.loads((directory / "_request.json").read_text())
        if change == "holdout":
            request["holdout_security_ids"] = ["t"]
        else:
            request["policy"]["weighting"] = "per_row"
        manifest["request_sha256"] = save(directory / "_request.json", request)
    else:
        key = "regularized_linear_return/security_transfer/fold-1"
        unit_directory = directory / key
        unit = json.loads((unit_directory / "_manifest.json").read_text())
        if change == "unit_weights":
            unit["unit"]["training_weights_sha256"] = "other_weights"
        else:
            frame = pd.read_parquet(unit_directory / "predictions.parquet")
            frame["scope_eligible"] = True
            frame["predicted_excess_return"] = 0.0
            frame["prediction_status"] = "research_prediction"
            frame.to_parquet(unit_directory / "predictions.parquet", index=False)
            unit["files"]["predictions.parquet"] = file_sha256(unit_directory / "predictions.parquet")
        manifest["units"][key]["manifest_sha256"] = save(unit_directory / "_manifest.json", unit)
    runs["candidate_sha256"] = save(directory / "_manifest.json", manifest)
    with pytest.raises(DataReadinessError):
        owner.compare_saved_returns(**runs)
    assert not (runs["root"] / runs["output"]).exists()


def test_source_mutation_after_read_rejected(runs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    summarize = owner.summarize_session_errors
    def mutate(frame: pd.DataFrame) -> dict[str, Any]:
        result = summarize(frame)
        if len(frame) == 8:
            (runs["root"] / runs["baseline"] / "_request.json").write_text("{}", encoding="utf-8")
        return result
    monkeypatch.setattr(owner, "summarize_session_errors", mutate)
    with pytest.raises(DataReadinessError, match="before report publication"):
        owner.compare_saved_returns(**runs)
    assert not (runs["root"] / runs["output"]).exists()


def test_comparison_cli_registered() -> None:
    from market_predictor.cli_surface import command_names
    from market_predictor.research_cli import app
    assert "compare-saved-swing-returns" in command_names(app)
