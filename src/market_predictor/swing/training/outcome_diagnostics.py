"""Explain retained outcome availability without changing supervision or model fits."""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import release_process_memory
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.labels.fixed_horizon_readiness import ROLES
from market_predictor.swing.training.return_artifacts import FAMILIES, FLAGS, SCOPES, TARGET, verify_completed_return_run, verify_unit

REASON_ROLES = ("stock", *ROLES)
REASON_COLUMNS = tuple(f"{role}_missing_reasons" for role in REASON_ROLES)
COLUMNS = ("decision_id", "security_id", "session_date_et", "feature_eligible", "security_holdout",
    "scope_eligible", "fixed_horizon_supervision_available", TARGET, "predicted_excess_return",
    "prediction_status", *REASON_COLUMNS)
COUNT_NAMES = ("rows", "scored_rows", "unscored_rows", "known_scored_outcomes", "unknown_scored_outcomes",
    "scored_target_missing", "scored_target_present_supervision_unavailable", "unknown_rows_without_recorded_reasons")
MAX_PAYLOAD_BYTES = 64 * 1024**2
MAX_EXPANDED_BYTES = 256 * 1024**2
MAX_ROWS = 1_000_000


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _validated(frame: pd.DataFrame) -> pd.DataFrame:
    _require(not frame.empty and frame.columns.is_unique and set(COLUMNS).issubset(frame),
        "saved outcome diagnostics require a nonempty frame with unique required columns")
    data = frame.loc[:, list(COLUMNS)].copy().reset_index(drop=True)
    for column in ("decision_id", "security_id"):
        _require(bool(data[column].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all()),
            f"invalid saved {column}")
    _require(not data.decision_id.duplicated().any() and not data.duplicated(["security_id", "session_date_et"]).any(),
        "saved outcome diagnostics require unique decisions and security/session pairs")
    _require(bool(data.session_date_et.map(lambda value: type(value) is date).all()), "saved session dates must be dates")
    for column in ("feature_eligible", "security_holdout", "scope_eligible", "fixed_horizon_supervision_available"):
        _require(bool(data[column].map(lambda value: isinstance(value, (bool, np.bool_))).all()),
            f"saved {column} requires explicit booleans")
    for column in (TARGET, "predicted_excess_return"):
        _require(getattr(data[column].dtype, "kind", None) in ("i", "u", "f"), f"saved {column} must be numeric")
        _require(bool(np.isfinite(data.loc[data[column].notna(), column].to_numpy(dtype=float)).all()),
            f"saved {column} must be finite or unavailable")
    scored = data.predicted_excess_return.notna()
    _require(scored.equals(data.scope_eligible), "saved predictions disagree with scoring scope")
    _require(not (scored & ~data.feature_eligible).any(), "saved scoring scope includes ineligible features")
    expected_status = pd.Series(np.where(scored, "research_prediction", "outside_eligible_scope"),
        name="prediction_status")
    _require(data.prediction_status.equals(expected_status), "saved prediction status disagrees with scoring scope")
    for column in REASON_COLUMNS:
        parsed = []
        for value in data[column]:
            try:
                reasons = json.loads(value) if isinstance(value, str) else None
            except ValueError as exc:
                raise DataReadinessError(f"{column} must contain JSON reason arrays") from exc
            _require(isinstance(reasons, list) and all(isinstance(reason, str) and bool(reason.strip()) for reason in reasons),
                f"{column} must contain JSON reason arrays")
            parsed.append(tuple(sorted(set(cast(list[str], reasons)))))
        data[column] = parsed
    return data


def _counts(frame: pd.DataFrame) -> dict[str, Any]:
    scored = frame.predicted_excess_return.notna()
    target_present = frame[TARGET].notna()
    # Preserve the exact existing regression_diagnostics mask, including all benchmarks.
    known = scored & frame.fixed_horizon_supervision_available & target_present
    unknown = scored & ~known
    any_reason = pd.Series(False, index=frame.index)
    reasons: dict[str, dict[str, int]] = {}
    role_rows: dict[str, int] = {}
    for role, column in zip(REASON_ROLES, REASON_COLUMNS, strict=True):
        with_reason = frame[column].map(bool)
        any_reason |= with_reason
        role_rows[role] = int((unknown & with_reason).sum())
        counter: Counter[str] = Counter()
        for values in frame.loc[unknown, column]:
            counter.update(values)
        reasons[role] = dict(sorted(counter.items()))
    return {"rows": len(frame), "scored_rows": int(scored.sum()), "unscored_rows": int((~scored).sum()),
        "known_scored_outcomes": int(known.sum()), "unknown_scored_outcomes": int(unknown.sum()),
        "scored_target_missing": int((scored & ~target_present).sum()),
        "scored_target_present_supervision_unavailable": int((scored & target_present & ~frame.fixed_horizon_supervision_available).sum()),
        "unknown_rows_without_recorded_reasons": int((unknown & ~any_reason).sum()),
        "unknown_rows_by_reason_role": role_rows, "recorded_reason_counts": reasons}


def summarize_outcome_availability(frame: pd.DataFrame) -> dict[str, Any]:
    """Count saved facts; reason memberships overlap and do not establish provider causes."""
    data = _validated(frame)
    return {"summary": _counts(data),
        "sessions": [{"session": day.isoformat(), **_counts(group)} for day, group in data.groupby("session_date_et", sort=True)],
        "securities": [{"security_id": identity, **_counts(group)} for identity, group in data.groupby("security_id", sort=True)]}


def _combine(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {key: sum(record[key] for record in records) for key in COUNT_NAMES}
    result["unknown_rows_by_reason_role"] = {
        role: sum(record["unknown_rows_by_reason_role"][role] for record in records) for role in REASON_ROLES}
    reasons = {}
    for role in REASON_ROLES:
        counter: Counter[str] = Counter()
        for record in records:
            counter.update(record["recorded_reason_counts"][role])
        reasons[role] = dict(sorted(counter.items()))
    result["recorded_reason_counts"] = reasons
    return result


def audit_saved_outcomes(*, root: Path, directory: Path, manifest_sha256: str, output: Path,
        family: str = "regularized_linear_return", scope: str = "temporal") -> dict[str, Any]:
    """Publish one immutable selected-family/scope audit from four saved scoring folds."""
    _require(family in FAMILIES and scope in SCOPES, "unsupported saved model family or evaluation scope")
    root = root.resolve()
    directory = inside(root, directory)
    output = output.resolve()
    _require(output.suffix == ".json" and not output.exists(), "outcome audit output must be a new JSON file")
    runtime = heavy_job_runtime_dir()
    runtime = runtime if runtime.is_absolute() else root / runtime
    with heavy_job_lease("audit-saved-swing-outcomes", runtime_dir=runtime):
        guard("saved outcome audit")
        manifest, request = verify_completed_return_run(directory, manifest_sha256=manifest_sha256)
        sources = {(directory / "_manifest.json").relative_to(root).as_posix(): manifest_sha256,
            (directory / "_request.json").relative_to(root).as_posix(): manifest["request_sha256"]}
        package = Path(__file__).resolve().parents[2]
        implementation = {str(package / name): file_sha256(package / name) for name in (
            "swing/training/outcome_diagnostics.py", "swing/training/return_artifacts.py",
            "swing/labels/fixed_horizon_readiness.py", "swing/datasets/symbol_corrections.py",
            "canonical/store.py", "core/errors.py", "core/json_integrity.py", "core/runtime_memory.py",
            "core/system_memory.py", "evidence/io.py", "evidence/hashing.py", "heavy_jobs.py",
            "locking.py", "resources.py", "process_memory.py")}
        folds: list[dict[str, Any]] = []
        seen_decisions: set[str] = set()
        seen_sessions: set[str] = set()
        for fold in request["folds"]:
            guard("saved outcome fold audit")
            key = f"{family}/{scope}/fold-{fold['number']}"
            record = manifest["units"][key]
            unit_directory = inside(root, record["path"])
            unit_path = unit_directory / "_manifest.json"
            saved = pinned_object(unit_path, record["manifest_sha256"])
            unit = saved["unit"]
            _require(unit["family"] == family and unit["scope"] == scope and unit["target"] == TARGET
                and unit["feature_names"] == request["feature_names"], "outcome audit unit differs from frozen request")
            _require(set(saved["files"]) == {"model.joblib", "predictions.parquet"}, "unexpected saved unit artifact inventory")
            _require(all((unit_directory / name).stat().st_size <= MAX_PAYLOAD_BYTES for name in saved["files"]),
                "saved outcome artifact exceeds byte budget")
            verify_unit(unit_directory, manifest_sha256=record["manifest_sha256"],
                request_sha256=manifest["request_sha256"], unit=unit)
            sources[unit_path.relative_to(root).as_posix()] = record["manifest_sha256"]
            sources.update({(unit_directory / name).relative_to(root).as_posix(): digest for name, digest in saved["files"].items()})
            with (unit_directory / "predictions.parquet").open("rb") as stream:
                content = stream.read(MAX_PAYLOAD_BYTES + 1)
            _require(len(content) <= MAX_PAYLOAD_BYTES and hashlib.sha256(content).hexdigest() == saved["files"]["predictions.parquet"],
                "saved outcome predictions changed or exceed byte budget")
            parquet = cast(Any, pq).ParquetFile(io.BytesIO(content))
            expanded = sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.metadata.num_row_groups))
            _require(parquet.metadata.num_rows <= MAX_ROWS and expanded <= MAX_EXPANDED_BYTES,
                "saved outcome predictions exceed expanded row or byte budget")
            frame = parquet.read(columns=list(COLUMNS)).to_pandas()
            _require(len(frame) == unit["scoring_rows"] and json_sha256(sorted(frame.decision_id)) == unit["scoring_decision_ids_sha256"],
                "saved outcome prediction inventory differs from frozen unit")
            _require(frame.security_id.isin(request["holdout_security_ids"]).equals(frame.security_holdout),
                "saved outcome holdout assignment differs")
            expected_scope = frame.feature_eligible & (frame.security_holdout if scope == "security_transfer" else True)
            _require(expected_scope.equals(frame.scope_eligible) and int(frame.scope_eligible.sum()) == unit["scope_eligible_rows"],
                "saved outcome scoring scope differs")
            result = summarize_outcome_availability(frame)
            sessions = {row["session"] for row in result["sessions"]}
            identities = set(frame.decision_id)
            _require(sessions == set(fold["score"]) and not seen_sessions.intersection(sessions)
                and not seen_decisions.intersection(identities), "saved outcome folds overlap or differ from frozen calendar")
            seen_sessions.update(sessions)
            seen_decisions.update(identities)
            _require(len(seen_decisions) <= MAX_ROWS, "saved outcome campaign exceeds row budget")
            metrics = saved["metrics"]
            _require(all(result["summary"][name] == metrics[name] for name in
                ("rows", "scored_rows", "known_scored_outcomes", "unknown_scored_outcomes")),
                "saved outcome counts differ from original metrics")
            folds.append({"fold": fold["number"], **result})
            del frame, content, parquet
            release_process_memory()
        security_records: dict[str, list[dict[str, Any]]] = {}
        for fold_result in folds:
            for row in fold_result["securities"]:
                security_records.setdefault(row["security_id"], []).append(row)
        report = {"schema": "market_predictor.saved_outcome_diagnostics", "status": "historical_diagnostics_complete",
            "family": family, "scope": scope, "target": TARGET, "source_root": str(root),
            "source_files": sources, "implementation_files": implementation,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                "pandas": pd.__version__, "pyarrow": pa.__version__},
            "count_unit": "stock_date_rows_in_saved_scoring_folds",
            "known_outcome_rule": "scored AND fixed_horizon_supervision_available AND SPY_target_present",
            "reason_count_unit": "unknown_scored_rows_with_each_recorded_reason_counted_once_per_role",
            "reason_denominator": "unknown_scored_outcomes_in_the_same_summary_session_or_security",
            "reason_counts_overlap": True, "recorded_reasons_are_provider_cause_proof": False,
            "unknown_partition": ["scored_target_missing", "scored_target_present_supervision_unavailable"],
            "summary": _combine([item["summary"] for item in folds]), "folds": folds,
            "sessions": sorted([row for item in folds for row in item["sessions"]], key=lambda row: row["session"]),
            "securities": [{"security_id": identity, **_combine(rows)} for identity, rows in sorted(security_records.items())],
            **dict.fromkeys(FLAGS, False), "new_fit_authorized": False}
        guard("saved outcome audit publication")
        for name, digest in sources.items():
            _require(file_sha256(inside(root, name)) == digest, "saved outcome source changed before publication")
        for name, digest in implementation.items():
            _require(file_sha256(Path(name)) == digest, "outcome audit implementation changed before publication")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix=".outcome-audit-") as temporary:
            staged = Path(temporary) / output.name
            write_json_object(staged, report)
            if os.name == "nt":
                os.rename(staged, output)
            else:
                os.link(staged, output)
        return {**report, "report_sha256": file_sha256(output)}
