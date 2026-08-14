"""Reproducible row-level audit for the immutable A4.3 intraday dataset."""

from __future__ import annotations

import json
import math
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_bar_dataset import (
    MEMORY_HARD_BUDGET_GIB,
    MEMORY_HEADROOM_GIB,
    load_complete_intraday_bar_dataset,
)
from market_predictor.edge_rebuild.intraday_bar_only_five_minute import (
    load_complete_selected_session_five_minute_projection,
)
from market_predictor.resources import assert_memory_budget
from market_predictor.v3.errors import DataReadinessError

INTRADAY_BAR_DATASET_AUDIT_SCHEMA: Final = (
    "edge_rebuild.intraday_bar_dataset_audit.v1"
)
_BANNED_FEATURE_TOKENS: Final = (
    "macd",
    "catalyst",
    "sentiment",
    "news",
    "trade_",
    "quote_",
)
_AUDIT_COLUMNS: Final = (
    "ticker",
    "session_date_et",
    "decision_time_utc",
    "feature_available_at_utc",
    "label_available_at_utc",
    "exit_bar_end_utc",
    "feature_eligible",
    "label_eligible",
    "dataset_eligible",
    "atr_14_5m",
    "five_minute_bar_observed",
    "ordered_feature_sha256",
    "feature_schema_version",
    "label_schema_version",
)
_ZERO_CHECKS: Final = (
    "duplicate_ticker_decision_rows",
    "feature_cutoff_violations",
    "label_availability_violations",
    "eligibility_implication_violations",
    "eligible_atr_violations",
    "ordered_feature_hash_violations",
    "schema_identity_violations",
    "missing_five_minute_feature_eligible_violations",
)


def publish_intraday_bar_dataset_audit(
    *,
    dataset_directory: Path,
    five_minute_projection_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Replay authorities, audit rows by session, and publish bound evidence."""

    _require_output_isolation(
        output_path,
        (dataset_directory, five_minute_projection_directory),
    )
    manifest = load_complete_intraday_bar_dataset(dataset_directory)
    projection = load_complete_selected_session_five_minute_projection(
        five_minute_projection_directory
    )
    incomplete_pairs = _incomplete_projection_pairs(
        five_minute_projection_directory,
        projection,
    )
    training_contract = manifest.get("training_contract")
    if not isinstance(training_contract, Mapping):
        raise DataReadinessError("intraday bar dataset omits its training contract")
    expected_feature_hash = str(training_contract.get("ordered_feature_sha256", ""))
    raw_features = training_contract.get("ordered_feature_names")
    if not isinstance(raw_features, list) or not expected_feature_hash:
        raise DataReadinessError("intraday bar dataset feature identity is invalid")
    features = [str(value) for value in raw_features]
    prohibited = sorted(
        name
        for name in features
        if any(token in name.lower() for token in _BANNED_FEATURE_TOKENS)
    )
    counters = _empty_counters()
    tickers: set[str] = set()
    incomplete_stats = {
        pair: {"rows": 0, "feature_eligible_rows": 0, "observed_rows": 0}
        for pair in incomplete_pairs
    }
    sessions = cast(list[str], manifest["planned_sessions"])
    for index, session_date in enumerate(sessions, start=1):
        rows = pd.read_parquet(
            dataset_directory
            / "sessions"
            / f"session_date_et={session_date}"
            / "rows.parquet",
            columns=list(_AUDIT_COLUMNS),
        )
        _audit_session(
            rows,
            session_date=session_date,
            expected_feature_hash=expected_feature_hash,
            counters=counters,
            tickers=tickers,
            incomplete_stats=incomplete_stats,
        )
        if index % 50 == 0:
            _guard_memory("intraday bar dataset audit")
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise DataReadinessError("intraday bar dataset summary is invalid")
    try:
        expected_rows = int(str(summary["rows"]))
        expected_eligible_rows = int(str(summary["dataset_eligible_rows"]))
    except (KeyError, ValueError) as exc:
        raise DataReadinessError(
            "intraday bar dataset summary counts are invalid"
        ) from exc
    report = {
        "schema": INTRADAY_BAR_DATASET_AUDIT_SCHEMA,
        "dataset_directory": str(dataset_directory.resolve()),
        "dataset_manifest_sha256": file_sha256(dataset_directory / "_manifest.json"),
        "dataset_authority_sha256": file_sha256(dataset_directory / "_authority.json"),
        "dataset_request_sha256": str(manifest["request_sha256"]),
        "dataset_transformation_sha256": str(manifest["transformation_sha256"]),
        "session_unit_inventory_sha256": str(
            manifest["session_unit_inventory_sha256"]
        ),
        "five_minute_projection_directory": str(
            five_minute_projection_directory.resolve()
        ),
        "five_minute_projection_manifest_sha256": file_sha256(
            five_minute_projection_directory / "_manifest.json"
        ),
        "five_minute_projection_authority_sha256": file_sha256(
            five_minute_projection_directory / "_authority.json"
        ),
        "five_minute_projection_inventory_sha256": str(
            projection["file_inventory_sha256"]
        ),
        "sessions": len(sessions),
        "tickers": len(tickers),
        **counters,
        "prohibited_model_features": prohibited,
        "projection_incomplete_pairs": len(incomplete_pairs),
        "projection_incomplete_pairs_represented": sum(
            value["rows"] > 0 for value in incomplete_stats.values()
        ),
        "projection_incomplete_pairs_with_observed_rows": sum(
            value["observed_rows"] > 0 for value in incomplete_stats.values()
        ),
        "projection_incomplete_pairs_with_eligible_earlier_rows": sum(
            value["feature_eligible_rows"] > 0
            for value in incomplete_stats.values()
        ),
    }
    report["status"] = (
        "pass"
        if all(counters[key] == 0 for key in _ZERO_CHECKS)
        and not prohibited
        and counters["rows"] == expected_rows
        and counters["dataset_eligible_rows"] == expected_eligible_rows
        else "fail"
    )
    _publish_report(output_path, report)
    if report["status"] != "pass":
        raise DataReadinessError(
            f"intraday bar dataset audit failed; evidence={output_path}"
        )
    return report


def _audit_session(
    frame: pd.DataFrame,
    *,
    session_date: str,
    expected_feature_hash: str,
    counters: dict[str, int],
    tickers: set[str],
    incomplete_stats: dict[tuple[str, str], dict[str, int]],
) -> None:
    counters["rows"] += len(frame)
    if frame.empty:
        return
    tickers.update(frame["ticker"].astype(str))
    counters["duplicate_ticker_decision_rows"] += int(
        frame.duplicated(["ticker", "decision_time_utc"]).sum()
    )
    feature_ok = frame["feature_eligible"].fillna(False).astype(bool)
    label_ok = frame["label_eligible"].fillna(False).astype(bool)
    dataset_ok = frame["dataset_eligible"].fillna(False).astype(bool)
    counters["feature_eligible_rows"] += int(feature_ok.sum())
    counters["label_eligible_rows"] += int(label_ok.sum())
    counters["dataset_eligible_rows"] += int(dataset_ok.sum())
    decision = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
    feature_at = pd.to_datetime(
        frame["feature_available_at_utc"], utc=True, errors="coerce"
    )
    label_at = pd.to_datetime(
        frame["label_available_at_utc"], utc=True, errors="coerce"
    )
    exit_end = pd.to_datetime(frame["exit_bar_end_utc"], utc=True, errors="coerce")
    counters["feature_cutoff_violations"] += int(
        (
            feature_ok
            & (feature_at.isna() | decision.isna() | feature_at.gt(decision))
        ).sum()
    )
    counters["label_availability_violations"] += int(
        (
            label_ok
            & (
                label_at.isna()
                | exit_end.isna()
                | label_at.lt(exit_end)
                | label_at.le(decision)
            )
        ).sum()
    )
    counters["eligibility_implication_violations"] += int(
        (dataset_ok & ~(feature_ok & label_ok)).sum()
    )
    atr = pd.to_numeric(frame["atr_14_5m"], errors="coerce")
    counters["eligible_atr_violations"] += int(
        (dataset_ok & (~atr.map(math.isfinite) | atr.le(0))).sum()
    )
    counters["ordered_feature_hash_violations"] += int(
        frame["ordered_feature_sha256"]
        .astype(str)
        .ne(expected_feature_hash)
        .sum()
    )
    counters["schema_identity_violations"] += int(
        frame["feature_schema_version"]
        .astype(str)
        .ne("edge_rebuild.intraday_bar_features.v1")
        .sum()
    )
    counters["schema_identity_violations"] += int(
        frame["label_schema_version"]
        .astype(str)
        .ne("edge_rebuild.intraday_bar_labels.v1")
        .sum()
    )
    observed = frame["five_minute_bar_observed"].fillna(False).astype(bool)
    counters["missing_five_minute_feature_eligible_violations"] += int(
        (~observed & feature_ok).sum()
    )
    for ticker, indices in frame.groupby("ticker", observed=True).groups.items():
        pair = (session_date, str(ticker))
        stats = incomplete_stats.get(pair)
        if stats is None:
            continue
        part = frame.loc[indices]
        stats["rows"] += len(part)
        stats["feature_eligible_rows"] += int(
            part["feature_eligible"].fillna(False).astype(bool).sum()
        )
        stats["observed_rows"] += int(
            part["five_minute_bar_observed"].fillna(False).astype(bool).sum()
        )


def _incomplete_projection_pairs(
    root: Path,
    manifest: Mapping[str, Any],
) -> set[tuple[str, str]]:
    parts = []
    for raw in cast(list[Mapping[str, Any]], manifest["files"]):
        if raw.get("role") != "coverage":
            continue
        path = _resolve_inside(root, str(raw.get("path", "")))
        if file_sha256(path) != str(raw.get("sha256", "")):
            raise DataReadinessError("five-minute coverage hash differs during audit")
        parts.append(
            pd.read_parquet(
                path,
                columns=["ticker", "session_date_et", "coverage_status"],
            )
        )
    if not parts:
        raise DataReadinessError("five-minute projection has no coverage records")
    coverage = pd.concat(parts, ignore_index=True)
    return {
        (str(row.session_date_et), str(row.ticker))
        for row in coverage.loc[
            coverage["coverage_status"].ne("complete")
        ].itertuples(index=False)
    }


def _empty_counters() -> dict[str, int]:
    return {
        "rows": 0,
        "feature_eligible_rows": 0,
        "label_eligible_rows": 0,
        "dataset_eligible_rows": 0,
        **{key: 0 for key in _ZERO_CHECKS},
    }


def _publish_report(path: Path, report: Mapping[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataReadinessError(
                f"intraday bar audit report is unreadable: {path}"
            ) from exc
        if existing != report:
            raise DataReadinessError(
                f"intraday bar audit report is immutable and differs: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)


def _require_output_isolation(output: Path, inputs: tuple[Path, ...]) -> None:
    target = output.resolve()
    for source in inputs:
        resolved = source.resolve()
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise DataReadinessError("intraday bar audit output overlaps an authority")


def _resolve_inside(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DataReadinessError("intraday bar audit source path is invalid")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise DataReadinessError("intraday bar audit source escapes its authority")
    return path


def _guard_memory(stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )
