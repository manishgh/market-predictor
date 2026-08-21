"""Exact requirement-level coverage audit for KS4 one-minute paths."""
from __future__ import annotations



import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday.specialist_collection import (
    SPECIALIST_ONE_MINUTE_COLLECTION_SCHEMA,
)
from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
    intraday_specialist_policy_identity,
    load_intraday_specialist_research_config,
)
from market_predictor.intraday.specialist_dataset import (
    verify_specialist_collection_plan,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.core.errors import DataReadinessError

SPECIALIST_COVERAGE_AUDIT_SCHEMA = "intraday.specialist_coverage_audit.v1"
SPECIALIST_REQUIREMENT_COVERAGE_SCHEMA = (
    "intraday.specialist_requirement_coverage.v1"
)
SPECIALIST_SETUP_COVERAGE_SCHEMA = "intraday.specialist_setup_coverage.v1"
_MINUTE_NS = 60_000_000_000


def build_intraday_specialist_coverage_audit(
    *,
    collection_plan_directory: Path,
    collection_directory: Path,
    policy_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Audit every planned interval against observed one-minute bars."""

    if output_directory.exists():
        raise DataReadinessError(
            f"KS4 coverage output must be new: {output_directory}"
        )
    plan = verify_specialist_collection_plan(collection_plan_directory)
    config = load_intraday_specialist_research_config(policy_path)
    if (
        _nested(plan, "policy", "policy_sha256")
        != config.policy_sha256()
    ):
        raise DataReadinessError("KS4 plan and coverage policy differ")
    collection_manifest_path = collection_directory / "_manifest.json"
    collection = _load_json(collection_manifest_path)
    _verify_collection_terminal(collection)
    artifact_index = _build_artifact_index(collection)
    requirement_paths = sorted(
        (collection_plan_directory / "requirements").glob("*.parquet")
    )
    if not requirement_paths:
        raise DataReadinessError("KS4 coverage plan has no requirements")

    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    files: list[dict[str, Any]] = []
    total_requirements = 0
    exact_requirements = 0
    total_setups = 0
    grid_complete_setups = 0
    reason_counts: dict[str, int] = {}
    role_failures: dict[str, int] = {
        "stock": 0,
        "spy": 0,
        "qqq": 0,
        "sector_benchmark": 0,
    }
    try:
        for requirement_path in requirement_paths:
            month = requirement_path.stem
            requirements = pd.read_parquet(requirement_path)
            audited = audit_requirement_coverage(
                requirements,
                artifact_index=artifact_index,
            )
            setups = aggregate_setup_coverage(audited)
            requirement_output = (
                temporary / "requirements" / f"{month}.parquet"
            )
            setup_output = temporary / "setups" / f"{month}.parquet"
            requirement_output.parent.mkdir(parents=True, exist_ok=True)
            setup_output.parent.mkdir(parents=True, exist_ok=True)
            audited.to_parquet(requirement_output, index=False)
            setups.to_parquet(setup_output, index=False)
            files.extend(
                [
                    _file_record(
                        requirement_output,
                        temporary,
                        rows=len(audited),
                    ),
                    _file_record(
                        setup_output,
                        temporary,
                        rows=len(setups),
                    ),
                ]
            )
            total_requirements += len(audited)
            exact_requirements += int(audited["coverage_exact"].sum())
            total_setups += len(setups)
            grid_complete_setups += int(setups["grid_complete"].sum())
            for reason, count in audited["coverage_reason"].value_counts().items():
                reason_counts[str(reason)] = (
                    reason_counts.get(str(reason), 0) + int(count)
                )
            failed = audited.loc[~audited["coverage_exact"]]
            for role in role_failures:
                role_failures[role] += int(
                    failed["roles_json"].str.contains(
                        f'"{role}"',
                        regex=False,
                    ).sum()
                )
            release_process_memory()
            _guard_memory(config, f"KS4 coverage {month}")
        if total_requirements != int(
            str(_nested(plan, "summary", "one_minute_requirements"))
        ):
            raise DataReadinessError(
                "KS4 coverage did not classify every requirement"
            )
        coverage_fingerprint = _coverage_fingerprint(
            files=files,
            plan_fingerprint=str(plan["plan_fingerprint"]),
            collection_manifest_sha256=file_sha256(
                collection_manifest_path
            ),
            policy_sha256=config.policy_sha256(),
        )
        report: dict[str, Any] = {
            "schema": SPECIALIST_COVERAGE_AUDIT_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "coverage_fingerprint": coverage_fingerprint,
            "collection_plan": {
                "path": str(collection_plan_directory),
                "plan_fingerprint": str(plan["plan_fingerprint"]),
                "manifest_sha256": file_sha256(
                    collection_plan_directory / "_manifest.json"
                ),
            },
            "collection": {
                "path": str(collection_directory),
                "manifest_sha256": file_sha256(
                    collection_manifest_path
                ),
                "request_sha256": str(collection["request_sha256"]),
                "unit_bundle_fingerprint": str(
                    collection["unit_bundle_fingerprint"]
                ),
                "transport_status": str(collection["status"]),
                "rows": int(collection["total_rows"]),
            },
            "policy": intraday_specialist_policy_identity(policy_path),
            "summary": {
                "requirements": total_requirements,
                "exact_requirements": exact_requirements,
                "inexact_requirements": (
                    total_requirements - exact_requirements
                ),
                "requirement_exact_rate": (
                    exact_requirements / total_requirements
                    if total_requirements
                    else 0.0
                ),
                "setups": total_setups,
                "grid_complete_setups": grid_complete_setups,
                "grid_incomplete_setups": (
                    total_setups - grid_complete_setups
                ),
                "setup_grid_complete_rate": (
                    grid_complete_setups / total_setups
                    if total_setups
                    else 0.0
                ),
                "coverage_reason_counts": dict(sorted(reason_counts.items())),
                "failed_requirement_roles": role_failures,
                "model_data_ready": False,
            },
            "memory": memory_audit(
                hard_budget_gib=config.maximum_process_memory_gib,
                headroom_gib=config.memory_guard_headroom_gib,
            ).to_record(),
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        (temporary / "_manifest.json").write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_directory)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def audit_requirement_coverage(
    requirements: pd.DataFrame,
    *,
    artifact_index: Mapping[tuple[str, str], Path],
) -> pd.DataFrame:
    """Classify exact minute coverage for one requirement shard."""

    required_columns = {
        "requirement_id",
        "setup_id",
        "strategy_id",
        "ticker",
        "roles_json",
        "segment_kind",
        "session_date_et",
        "requested_start_utc",
        "requested_end_utc",
        "decision_time_utc",
        "price_feed",
        "adjustment",
        "timeframe",
    }
    missing = sorted(required_columns.difference(requirements.columns))
    if missing:
        raise DataReadinessError(
            "KS4 coverage requirement columns missing: "
            + ", ".join(missing)
        )
    output = requirements.copy()
    if bool(output["requirement_id"].duplicated().any()):
        raise DataReadinessError(
            "KS4 coverage shard has duplicate requirement IDs"
        )
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["session_key"] = pd.to_datetime(
        output["session_date_et"]
    ).dt.date.astype(str)
    starts = pd.to_datetime(
        output["requested_start_utc"],
        utc=True,
        errors="raise",
    )
    ends = pd.to_datetime(
        output["requested_end_utc"],
        utc=True,
        errors="raise",
    )
    start_ns = pd.DatetimeIndex(starts).as_unit("ns").asi8
    end_ns = pd.DatetimeIndex(ends).as_unit("ns").asi8
    duration_ns = end_ns - start_ns
    aligned = (
        (duration_ns > 0)
        & (duration_ns % _MINUTE_NS == 0)
        & (start_ns % _MINUTE_NS == 0)
        & (end_ns % _MINUTE_NS == 0)
    )
    expected = np.where(aligned, duration_ns // _MINUTE_NS, 0).astype(
        "int32"
    )
    observed = np.zeros(len(output), dtype="int32")
    path_available = np.zeros(len(output), dtype=bool)
    keys = list(zip(output["ticker"], output["session_key"], strict=True))
    positions_by_artifact: dict[Path, list[int]] = {}
    for position, key in enumerate(keys):
        path = artifact_index.get(key)
        if path is not None:
            positions_by_artifact.setdefault(path, []).append(position)
    for path, positions in positions_by_artifact.items():
        bars = pd.read_parquet(
            path,
            columns=["ticker", "bar_start_utc"],
        )
        bars["ticker"] = bars["ticker"].astype(str).str.upper().str.strip()
        for ticker, ticker_positions in _positions_by_ticker(
            output,
            positions,
        ).items():
            times = (
                pd.DatetimeIndex(
                    pd.to_datetime(
                        bars.loc[
                            bars["ticker"].eq(ticker),
                            "bar_start_utc",
                        ],
                        utc=True,
                        errors="raise",
                    )
                )
                .as_unit("ns")
                .asi8.copy()
            )
            if len(times) == 0:
                continue
            times.sort()
            if bool(np.any(np.diff(times) <= 0)):
                raise DataReadinessError(
                    f"KS4 collected bars are not unique and ordered: {path}"
                )
            if bool(np.any(times % _MINUTE_NS != 0)):
                raise DataReadinessError(
                    f"KS4 collected bars are not minute aligned: {path}"
                )
            index = np.asarray(ticker_positions, dtype=int)
            left = np.searchsorted(times, start_ns[index], side="left")
            right = np.searchsorted(times, end_ns[index], side="left")
            observed[index] = (right - left).astype("int32")
            path_available[index] = True
    if bool(np.any(observed > expected)):
        raise DataReadinessError(
            "KS4 observed coverage exceeds planned interval length"
        )
    exact = aligned & path_available & (observed == expected)
    missing_bars = np.maximum(expected - observed, 0).astype("int32")
    reason = np.full(len(output), "exact", dtype=object)
    reason[~aligned] = "invalid_interval"
    reason[aligned & ~path_available] = "missing_ticker_session"
    reason[aligned & path_available & ~exact] = "missing_minutes"
    output["coverage_schema_version"] = (
        SPECIALIST_REQUIREMENT_COVERAGE_SCHEMA
    )
    output["expected_bars"] = expected
    output["observed_bars"] = observed
    output["missing_bars"] = missing_bars
    output["coverage_exact"] = exact
    output["coverage_reason"] = pd.Series(reason, dtype="string")
    return output.drop(columns=["session_key"])


def aggregate_setup_coverage(requirements: pd.DataFrame) -> pd.DataFrame:
    """Aggregate requirement outcomes into one fail-closed setup decision."""

    if requirements.empty:
        return pd.DataFrame(
            columns=[
                "coverage_schema_version",
                "setup_id",
                "strategy_id",
        "decision_time_utc",
                "requirement_count",
                "exact_requirement_count",
                "missing_bars",
                "stock_complete",
                "spy_complete",
                "qqq_complete",
                "sector_benchmark_complete",
                "grid_complete",
                "coverage_reasons_json",
            ]
        )
    grouped = requirements.groupby("setup_id", sort=False)
    output = grouped.agg(
        strategy_id=("strategy_id", "first"),
        decision_time_utc=("decision_time_utc", "first"),
        requirement_count=("requirement_id", "size"),
        exact_requirement_count=("coverage_exact", "sum"),
        missing_bars=("missing_bars", "sum"),
    ).reset_index()
    for role in ("stock", "spy", "qqq", "sector_benchmark"):
        role_rows = requirements[
            requirements["roles_json"].str.contains(
                f'"{role}"',
                regex=False,
            )
        ]
        complete = role_rows.groupby("setup_id", sort=False)[
            "coverage_exact"
        ].all()
        output[f"{role}_complete"] = (
            output["setup_id"].map(complete).fillna(False).astype(bool)
        )
    output["grid_complete"] = (
        output["exact_requirement_count"].eq(output["requirement_count"])
        & output["stock_complete"]
        & output["spy_complete"]
        & output["qqq_complete"]
        & output["sector_benchmark_complete"]
    )
    failures = requirements.loc[~requirements["coverage_exact"]]
    reasons = failures.groupby("setup_id", sort=False)[
        "coverage_reason"
    ].agg(lambda values: json.dumps(sorted(set(map(str, values)))))
    output["coverage_reasons_json"] = output["setup_id"].map(reasons)
    output["coverage_reasons_json"] = output[
        "coverage_reasons_json"
    ].fillna("[]")
    output.insert(
        0,
        "coverage_schema_version",
        SPECIALIST_SETUP_COVERAGE_SCHEMA,
    )
    return output


def _positions_by_ticker(
    requirements: pd.DataFrame,
    positions: Sequence[int],
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for position in positions:
        ticker = str(requirements.iloc[position]["ticker"])
        result.setdefault(ticker, []).append(position)
    return result


def _verify_collection_terminal(collection: Mapping[str, Any]) -> None:
    if collection.get("schema") != SPECIALIST_ONE_MINUTE_COLLECTION_SCHEMA:
        raise DataReadinessError(
            "unsupported KS4 one-minute collection schema"
        )
    if collection.get("status") != "transport_complete":
        raise DataReadinessError(
            "KS4 one-minute collection is not transport complete"
        )
    if collection.get("failed_units") != {}:
        raise DataReadinessError(
            "KS4 one-minute collection contains failed units"
        )
    requested = int(collection.get("requested_units", -1))
    completed = int(collection.get("completed_units", -1))
    artifacts = collection.get("artifacts")
    if (
        requested < 1
        or completed != requested
        or not isinstance(artifacts, list)
        or len(artifacts) != requested
    ):
        raise DataReadinessError(
            "KS4 one-minute collection terminal counts differ"
        )


def _build_artifact_index(
    collection: Mapping[str, Any],
) -> dict[tuple[str, str], Path]:
    artifacts = cast(list[object], collection["artifacts"])
    result: dict[tuple[str, str], Path] = {}
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise DataReadinessError(
                "KS4 collection artifact record is malformed"
            )
        path = Path(str(raw["path"]))
        if not path.is_file():
            raise DataReadinessError(
                f"KS4 collected unit is missing: {path}"
            )
        session = str(raw["asof_date"])
        symbol_rows = raw.get("symbol_rows")
        if not isinstance(symbol_rows, Mapping):
            raise DataReadinessError(
                f"KS4 collected unit has no symbol rows: {path}"
            )
        for symbol in symbol_rows:
            key = (str(symbol).upper().strip(), session)
            if key in result:
                raise DataReadinessError(
                    f"KS4 ticker-session appears in two units: {key}"
                )
            result[key] = path
    return result


def _coverage_fingerprint(
    *,
    files: Sequence[Mapping[str, Any]],
    plan_fingerprint: str,
    collection_manifest_sha256: str,
    policy_sha256: str,
) -> str:
    payload = {
        "schema": SPECIALIST_COVERAGE_AUDIT_SCHEMA,
        "plan_fingerprint": plan_fingerprint,
        "collection_manifest_sha256": collection_manifest_sha256,
        "policy_sha256": policy_sha256,
        "files": [
            {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
                "rows": int(record["rows"]),
            }
            for record in sorted(
                files,
                key=lambda item: str(item["path"]),
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _file_record(
    path: Path,
    root: Path,
    *,
    rows: int,
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DataReadinessError(f"missing KS4 JSON artifact: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"unreadable KS4 JSON artifact: {path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(
            f"KS4 JSON artifact is not an object: {path}"
        )
    return {str(key): value for key, value in loaded.items()}


def _nested(
    payload: Mapping[str, Any],
    outer: str,
    inner: str,
) -> object:
    value = payload.get(outer)
    if not isinstance(value, Mapping):
        raise DataReadinessError(
            f"KS4 manifest field is not an object: {outer}"
        )
    return value.get(inner)


def _guard_memory(
    config: IntradaySpecialistResearchConfig,
    stage: str,
) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    assert_peak_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
