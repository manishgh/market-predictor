"""Exact paired diagnostics from immutable historical predictions; never model admission."""
from __future__ import annotations

import hashlib
import io
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_peak_memory_budget, release_process_memory
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.training.return_artifacts import verify_unit

TARGET = "spy_fixed_horizon_excess_return"
FAMILIES = ("regularized_linear_return", "shallow_boosted_return")
SCOPES = ("temporal", "security_transfer")
FLAGS = ("serving_eligible", "promotion_eligible", "portfolio_evaluated", "outer_validation_opened",
    "historical_test_opened")
PROFILE_BINDINGS = {"feature_profile", "published_profile", "readiness", "readiness_config"}


def paired_session_errors(baseline: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    """Align every decision before evaluating; unknown outcomes remain counted."""
    columns = {"decision_id", "security_id", "session_date_et", "feature_eligible", "security_holdout",
        "scope_eligible", "fixed_horizon_supervision_available", TARGET, "predicted_excess_return", "prediction_status"}
    for frame in (baseline, candidate):
        if not columns.issubset(frame.columns) or frame.empty:
            raise DataReadinessError("paired predictions lack required rows or columns")
        if (not frame.decision_id.map(lambda value: isinstance(value, str) and bool(value)).all()
                or frame.decision_id.duplicated().any()):
            raise DataReadinessError("paired predictions require unique nonempty decision IDs")
        for name in ("feature_eligible", "security_holdout", "scope_eligible", "fixed_horizon_supervision_available"):
            if frame[name].isna().any() or not frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise DataReadinessError("paired eligibility flags must be non-null booleans")
        eligible = frame.scope_eligible
        predicted = frame.predicted_excess_return
        if (not predicted.notna().equals(eligible) or not np.isfinite(predicted[eligible]).all()
                or not np.isfinite(frame.loc[frame[TARGET].notna(), TARGET]).all()
                or not frame.prediction_status.equals(pd.Series(np.where(eligible, "research_prediction",
                    "outside_eligible_scope"), index=frame.index, name="prediction_status"))):
            raise DataReadinessError("paired predictions have invalid values, status or scoring eligibility")
    left = baseline.sort_values("decision_id").reset_index(drop=True)
    right = candidate.sort_values("decision_id").reset_index(drop=True)
    try:
        assert_frame_equal(left.drop(columns="predicted_excess_return"), right.drop(columns="predicted_excess_return"),
            check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise DataReadinessError("paired scoring population, targets, nulls or metadata differ") from exc
    known = left.scope_eligible & left.fixed_horizon_supervision_available & left[TARGET].notna()
    records = []
    for day, population in left.groupby("session_date_et", sort=True, dropna=False):
        if pd.isna(day):
            raise DataReadinessError("paired scoring session is missing")
        positions = population.index[known.loc[population.index]]
        actual = left.loc[positions, TARGET].to_numpy(dtype=np.float64)
        base = left.loc[positions, "predicted_excess_return"].to_numpy(dtype=np.float64)
        proposed = right.loc[positions, "predicted_excess_return"].to_numpy(dtype=np.float64)
        ranks: list[float | None] = []
        for forecast in (base, proposed):
            rank = None
            if len(actual) >= 2 and np.unique(actual).size > 1 and np.unique(forecast).size > 1:
                rank = float(pd.Series(actual).rank().corr(pd.Series(forecast).rank()))
            ranks.append(rank)
        record: dict[str, Any] = {"session": day.isoformat(), "rows": len(population),
            "scored_rows": int(population.scope_eligible.sum()), "known_rows": len(positions),
            "baseline_rank": ranks[0], "candidate_rank": ranks[1],
            "paired_rank_difference": ranks[1] - ranks[0] if ranks[0] is not None and ranks[1] is not None else None}
        for name, forecast in (("baseline", base), ("candidate", proposed), ("zero", np.zeros(len(actual)))):
            record[f"{name}_mse"] = float(np.mean((forecast - actual)**2)) if len(actual) else None
            record[f"{name}_mae"] = float(np.mean(np.abs(forecast - actual))) if len(actual) else None
        records.append(record)
    return pd.DataFrame(records)


def summarize_session_errors(sessions: pd.DataFrame) -> dict[str, Any]:
    """Each evaluable date has equal weight, regardless of fold size or stock count."""
    if sessions.empty or sessions.session.duplicated().any():
        raise DataReadinessError("paired folds must contain disjoint nonempty session inventories")
    means = {}
    for name in ("baseline_mse", "candidate_mse", "zero_mse", "baseline_mae", "candidate_mae", "zero_mae",
            "baseline_rank", "candidate_rank", "paired_rank_difference"):
        values = sessions[name].dropna()
        means[name] = float(values.mean()) if len(values) else None
    for metric in ("mse", "mae"):
        base, candidate = means[f"baseline_{metric}"], means[f"candidate_{metric}"]
        means[f"paired_{metric}_difference"] = candidate - base if candidate is not None and base is not None else None
    return {**means, "rows": int(sessions.rows.sum()), "scored_rows": int(sessions.scored_rows.sum()),
        "known_scored_outcomes": int(sessions.known_rows.sum()),
        "unknown_scored_outcomes": int((sessions.scored_rows - sessions.known_rows).sum()),
        "scoring_sessions": len(sessions), "error_sessions": int(sessions.baseline_mse.notna().sum()),
        "baseline_rank_sessions": int(sessions.baseline_rank.notna().sum()),
        "candidate_rank_sessions": int(sessions.candidate_rank.notna().sum()),
        "paired_rank_sessions": int(sessions.paired_rank_difference.notna().sum())}


def _guard() -> None:
    assert_peak_memory_budget(stage="saved prediction comparison", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available()


def _run(root: Path, directory: Path, digest: str, pins: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = inside(root, directory)
    manifest_path = path / "_manifest.json"
    manifest = pinned_object(manifest_path, digest)
    pins[manifest_path.relative_to(root).as_posix()] = digest
    request_path = path / "_request.json"
    request = pinned_object(request_path, manifest["request_sha256"])
    pins[request_path.relative_to(root).as_posix()] = manifest["request_sha256"]
    if (manifest.get("schema") != "market_predictor.swing_return_training"
            or manifest.get("status") != "complete_research_only"
            or request.get("schema") != "market_predictor.swing_return_training_request"
            or any(record.get(flag) is not False for record in (manifest, request) for flag in FLAGS)):
        raise DataReadinessError("comparison requires complete initial-fit development runs without promotion")
    expected = {f"{family}/{scope}/fold-{fold}" for family in FAMILIES for scope in SCOPES for fold in range(1, 5)}
    expected |= {f"{family}/final_refit" for family in FAMILIES}
    if (set(manifest["units"]) != expected or request["policy"]["target"] != TARGET
            or request["policy"]["scope"] != "initial_fit_research_only"
            or [fold["number"] for fold in request["folds"]] != [1, 2, 3, 4]):
        raise DataReadinessError("comparison run target, scope or fold inventory differs")
    return manifest, request


def compare_saved_returns(*, root: Path, baseline: Path, baseline_sha256: str, candidate: Path,
    candidate_sha256: str, output: Path) -> dict[str, Any]:
    """Read historical scores without certifying old training inputs for a new fit."""
    root = root.resolve()
    output = inside(root, output)
    if not output.is_relative_to(root / "data/reports") or output.exists():
        raise DataReadinessError("comparison output must be a new report below data/reports")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("compare-saved-swing-returns", runtime_dir=runtime):
        _guard()
        pins: dict[str, str] = {}
        for module in (__file__, Path(__file__).with_name("return_artifacts.py")):
            path = Path(module)
            pins[path.relative_to(root).as_posix()] = file_sha256(path)
        runs = [_run(root, path, digest, pins) for path, digest in
            ((baseline, baseline_sha256), (candidate, candidate_sha256))]
        requests = [run[1] for run in runs]
        arms = {name: {"directory": inside(root, path).relative_to(root).as_posix(),
            "manifest_sha256": digest, "request_sha256": run[0]["request_sha256"],
            "feature_names": run[1]["feature_names"]} for name, path, digest, run in
            zip(("baseline", "candidate"), (baseline, candidate), (baseline_sha256, candidate_sha256), runs, strict=True)}
        shared = [{name: value for name, value in request.items() if name not in
            {"policy", "feature_names", "source_files"}} for request in requests]
        policies = [{name: value for name, value in request["policy"].items() if name not in PROFILE_BINDINGS}
            for request in requests]
        if policies[0] != policies[1] or shared[0] != shared[1]:
            raise DataReadinessError("paired runs use different calendars, populations, holdouts or fitting settings")
        groups = {}
        for family in FAMILIES:
            for scope in SCOPES:
                pieces = []
                for fold in requests[0]["folds"]:
                    _guard()
                    key = f"{family}/{scope}/fold-{fold['number']}"
                    frames, units = [], []
                    for manifest, request in runs:
                        record = manifest["units"][key]
                        directory = inside(root, record["path"])
                        unit_path = directory / "_manifest.json"
                        unit_manifest = pinned_object(unit_path, record["manifest_sha256"])
                        unit = unit_manifest["unit"]
                        if (unit["family"] != family or unit["scope"] != scope or unit["target"] != TARGET
                                or unit["feature_names"] != request["feature_names"]):
                            raise DataReadinessError("comparison unit family, scope, features or target differs")
                        verify_unit(directory, manifest_sha256=record["manifest_sha256"],
                            request_sha256=manifest["request_sha256"], unit=unit)
                        pins[unit_path.relative_to(root).as_posix()] = record["manifest_sha256"]
                        for name, digest in unit_manifest["files"].items():
                            pins[(directory / name).relative_to(root).as_posix()] = digest
                        prediction_path = directory / "predictions.parquet"
                        if prediction_path.stat().st_size > 64 * 1024**2:
                            raise DataReadinessError("saved predictions exceed bounded reader size")
                        content = prediction_path.read_bytes()
                        if hashlib.sha256(content).hexdigest() != unit_manifest["files"]["predictions.parquet"]:
                            raise DataReadinessError("saved prediction bytes changed before decoding")
                        frame = pd.read_parquet(io.BytesIO(content))
                        expected_scope = frame.feature_eligible & (frame.security_holdout if scope == "security_transfer" else True)
                        if (set(day.isoformat() for day in frame.session_date_et) != set(fold["score"])
                                or len(frame) != unit["scoring_rows"]
                                or json_sha256(sorted(frame.decision_id)) != unit["scoring_decision_ids_sha256"]
                                or not frame.security_id.isin(request["holdout_security_ids"]).equals(frame.security_holdout)
                                or int(frame.scope_eligible.sum()) != unit["scope_eligible_rows"]
                                or not expected_scope.equals(frame.scope_eligible)):
                            raise DataReadinessError("saved prediction calendar, population or transfer scope differs")
                        frames.append(frame)
                        units.append({name: value for name, value in unit.items() if name != "feature_names"})
                    if units[0] != units[1]:
                        raise DataReadinessError("paired units have different training identities, weights or maturity")
                    pieces.append(paired_session_errors(*frames))
                    del frames, frame, content
                    release_process_memory()
                sessions = pd.concat(pieces, ignore_index=True)
                groups[f"{family}/{scope}"] = {"summary": summarize_session_errors(sessions),
                    "sessions": sessions.astype(object).where(sessions.notna(), None).to_dict(orient="records")}
        result = {"schema": "market_predictor.swing_return_comparison", "status": "historical_diagnostics_complete",
            "aggregation": "equal_weight_per_session_over_disjoint_scoring_folds",
            "difference_direction": "candidate_minus_baseline_negative_error_difference_is_improvement",
            "target": TARGET, "source_files": pins, "arms": arms, "groups": groups,
            "comparison_runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
            **dict.fromkeys(FLAGS, False), "new_fit_authorized": False,
            "independent_observations_assumed": False}
        for name, digest in pins.items():
            if file_sha256(inside(root, name)) != digest:
                raise DataReadinessError("comparison source changed before report publication")
        _guard()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".comparison-") as temporary:
            staged = Path(temporary) / output.name
            write_json_object(staged, result)
            # Atomic exposure with no overwrite, including an external concurrent writer.
            if os.name == "nt":
                os.rename(staged, output)  # Windows rename refuses an existing destination.
            else:
                os.link(staged, output)
        return {**result, "report_sha256": file_sha256(output)}
