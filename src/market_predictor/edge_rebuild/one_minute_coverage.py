"""Replay selected-session one-minute coverage and publish exclusions."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.history_collection import (
    load_complete_intraday_history_collection,
)
from market_predictor.edge_rebuild.history_contracts import (
    SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA,
)
from market_predictor.edge_rebuild.intraday_history import (
    load_complete_intraday_history_plan,
    load_plan_json,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.v3.errors import DataReadinessError

COVERAGE_SCHEMA = "edge_rebuild.selected_session_one_minute_coverage.v1"
COVERAGE_AUTHORITY_SCHEMA = (
    "edge_rebuild.selected_session_one_minute_coverage_authority.v1"
)


def publish_selected_session_one_minute_coverage(
    *,
    plan_directory: Path,
    collection_directory: Path,
    five_minute_collection_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish pair-level coverage and whole-security exclusions."""

    if output_directory.exists():
        raise DataReadinessError(
            f"one-minute coverage output must be new: {output_directory}"
        )
    plan = load_complete_intraday_history_plan(plan_directory)
    plan_request = load_plan_json(plan_directory / "_request.json")
    collection = load_complete_intraday_history_collection(collection_directory)
    collection_request = _read_json(collection_directory / "_request.json")
    five_minute_collection = load_complete_intraday_history_collection(
        five_minute_collection_directory
    )
    five_minute_request = _read_json(
        five_minute_collection_directory / "_request.json"
    )
    _require_collection_identity(collection_request, timeframe="1Min")
    _require_collection_identity(five_minute_request, timeframe="5Min")
    selection = plan_request.get("selection")
    if (
        plan.get("schema") != SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA
        or plan.get("acquisition", {}).get("timeframe") != "1Min"
        or not isinstance(selection, Mapping)
        or selection.get("strategy_id") != strategy_contract.intraday.strategy_id
        or selection.get("strategy_contract_sha256") != strategy_contract.sha256()
        or collection.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or collection.get("status") != "transport_complete"
    ):
        raise DataReadinessError(
            "one-minute coverage inputs do not share the active strategy and plan"
        )
    required = _required_pairs(plan_directory)
    observed = _observed_pair_rows(collection)
    five_minute_observed = _observed_pair_rows(five_minute_collection)
    required_keys = set(
        zip(
            required["session_date_et"].astype(str),
            required["ticker"].astype(str),
            strict=True,
        )
    )
    unexpected = sorted(set(observed).difference(required_keys))
    if unexpected:
        raise DataReadinessError(
            f"one-minute collection contains {len(unexpected)} unplanned symbol-sessions"
        )
    coverage = required.copy()
    keys = list(
        zip(
            coverage["session_date_et"].astype(str),
            coverage["ticker"].astype(str),
            strict=True,
        )
    )
    coverage["observed_rows"] = [observed.get(key, 0) for key in keys]
    coverage["one_minute_observation_density"] = coverage["observed_rows"].div(
        coverage["expected_rows"]
    )
    coverage["expected_five_minute_rows"] = (
        coverage["expected_rows"].add(4).floordiv(5).astype("int32")
    )
    coverage["observed_five_minute_rows"] = [
        five_minute_observed.get(key, 0) for key in keys
    ]
    coverage["five_minute_bar_continuity"] = coverage[
        "observed_five_minute_rows"
    ].div(coverage["expected_five_minute_rows"])
    continuity_floor = strategy_contract.intraday_universe.minimum_bar_continuity
    coverage["coverage_status"] = (
        coverage["observed_rows"].gt(0)
        & coverage["five_minute_bar_continuity"].ge(continuity_floor)
    ).map(
        {True: "complete", False: "incomplete"}
    )
    exclusions = _whole_security_exclusions(
        coverage,
        continuity_floor=continuity_floor,
    )
    security_count = int(coverage["ticker"].nunique())
    excluded_count = len(exclusions)
    excluded_share = excluded_count / max(1, security_count)
    ceiling = strategy_contract.data_quality.maximum_security_exclusion_fraction
    ready = excluded_share <= ceiling
    request = {
        "schema": COVERAGE_SCHEMA,
        "plan_path": str(plan_directory),
        "plan_manifest_sha256": file_sha256(plan_directory / "_manifest.json"),
        "collection_path": str(collection_directory),
        "collection_manifest_sha256": file_sha256(
            collection_directory / "_manifest.json"
        ),
        "five_minute_collection_path": str(five_minute_collection_directory),
        "five_minute_collection_manifest_sha256": file_sha256(
            five_minute_collection_directory / "_manifest.json"
        ),
        "strategy_contract_path": str(strategy_contract_path),
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "strategy_contract_sha256": strategy_contract.sha256(),
    }
    staging = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.staging"
    )
    staging.mkdir(parents=True)
    try:
        coverage_path = staging / "stock_session_coverage.parquet"
        exclusions_path = staging / "excluded_securities.parquet"
        coverage.to_parquet(coverage_path, index=False)
        exclusions.to_parquet(exclusions_path, index=False)
        _write_json(staging / "_request.json", request)
        manifest = {
            **request,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "ready" if ready else "blocked_exclusion_ceiling",
            "ready_for_feature_build": ready,
            "summary": {
                "required_stock_sessions": int(len(coverage)),
                "observed_stock_sessions": int(coverage["observed_rows"].gt(0).sum()),
                "empty_stock_sessions": int(coverage["observed_rows"].eq(0).sum()),
                "complete_stock_sessions": int(
                    coverage["coverage_status"].eq("complete").sum()
                ),
                "incomplete_stock_sessions": int(
                    coverage["coverage_status"].eq("incomplete").sum()
                ),
                "minimum_bar_continuity": continuity_floor,
                "securities": security_count,
                "excluded_securities": excluded_count,
                "excluded_security_share": excluded_share,
                "maximum_excluded_security_share": ceiling,
            },
            "files": [
                _file_record(coverage_path, staging, len(coverage)),
                _file_record(exclusions_path, staging, len(exclusions)),
                _file_record(staging / "_request.json", staging, 1),
            ],
        }
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": COVERAGE_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "ready_for_feature_build": ready,
            },
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_directory)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_complete_one_minute_coverage(directory: Path) -> dict[str, Any]:
    manifest = _read_json(directory / "_manifest.json")
    authority = _read_json(directory / "_authority.json")
    if (
        manifest.get("schema") != COVERAGE_SCHEMA
        or authority.get("schema") != COVERAGE_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(directory / "_manifest.json")
        or authority.get("ready_for_feature_build")
        != manifest.get("ready_for_feature_build")
    ):
        raise DataReadinessError(
            f"one-minute coverage lacks matching authority: {directory}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise DataReadinessError("one-minute coverage manifest has no files")
    for raw in files:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("one-minute coverage file record is malformed")
        path = _resolve_inside(directory, str(raw.get("path", "")))
        if not path.is_file() or file_sha256(path) != raw.get("sha256"):
            raise DataReadinessError(f"one-minute coverage file failed hash: {path}")
    return manifest


def _required_pairs(plan_directory: Path) -> pd.DataFrame:
    parts = [
        pd.read_parquet(
            path,
            columns=[
                "session_date_et",
                "session_open_utc",
                "session_close_utc",
                "ticker",
            ],
        )
        for path in sorted((plan_directory / "stock_sessions").glob("*.parquet"))
    ]
    if not parts:
        raise DataReadinessError("one-minute plan contains no stock-session table")
    frame = pd.concat(parts, ignore_index=True)
    frame["session_date_et"] = pd.to_datetime(
        frame["session_date_et"], errors="raise"
    ).dt.date.astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    opens = pd.to_datetime(frame["session_open_utc"], utc=True, errors="raise")
    closes = pd.to_datetime(frame["session_close_utc"], utc=True, errors="raise")
    frame["expected_rows"] = (
        (closes - opens).dt.total_seconds().div(60).astype("int32")
    )
    frame = frame.drop_duplicates().sort_values(
        ["session_date_et", "ticker"], kind="stable"
    )
    if (
        frame.empty
        or bool(frame["ticker"].eq("").any())
        or bool(frame["expected_rows"].le(0).any())
    ):
        raise DataReadinessError("one-minute plan stock-session table is empty")
    return frame.loc[
        :, ["session_date_et", "ticker", "expected_rows"]
    ].reset_index(drop=True)


def _observed_pair_rows(collection: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    observed: dict[tuple[str, str], int] = {}
    artifacts = collection.get("artifacts")
    if not isinstance(artifacts, list):
        raise DataReadinessError("one-minute collection has no artifacts")
    for raw in artifacts:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("one-minute collection artifact is malformed")
        session = str(raw.get("asof_date", ""))
        symbol_rows = raw.get("symbol_rows")
        if not isinstance(symbol_rows, Mapping):
            raise DataReadinessError("one-minute artifact has no symbol row counts")
        for ticker, count in symbol_rows.items():
            key = (session, str(ticker).upper().strip())
            if key in observed:
                raise DataReadinessError(
                    f"one-minute collection repeats symbol-session {key}"
                )
            observed[key] = int(count)
    return observed


def _whole_security_exclusions(
    coverage: pd.DataFrame,
    *,
    continuity_floor: float,
) -> pd.DataFrame:
    grouped = coverage.groupby("ticker", sort=True)
    summary = grouped.agg(
        required_sessions=("observed_rows", "size"),
        observed_sessions=("observed_rows", lambda values: int(values.gt(0).sum())),
        empty_sessions=("observed_rows", lambda values: int(values.eq(0).sum())),
        expected_five_minute_rows=("expected_five_minute_rows", "sum"),
        observed_five_minute_rows=("observed_five_minute_rows", "sum"),
    ).reset_index()
    summary["five_minute_bar_continuity"] = summary[
        "observed_five_minute_rows"
    ].div(summary["expected_five_minute_rows"])
    excluded = summary.loc[
        summary["empty_sessions"].gt(0)
        | summary["five_minute_bar_continuity"].lt(continuity_floor)
    ].copy()
    excluded["reason"] = "five_minute_continuity_below_floor"
    excluded.loc[excluded["empty_sessions"].gt(0), "reason"] = (
        "one_or_more_selected_sessions_have_no_one_minute_trade_path"
    )
    return excluded.reset_index(drop=True)


def _require_collection_identity(
    request: Mapping[str, Any],
    *,
    timeframe: str,
) -> None:
    if (
        request.get("provider") != "alpaca"
        or request.get("timeframe") != timeframe
        or request.get("price_feed") != "sip"
        or request.get("adjustment") != "all"
    ):
        raise DataReadinessError(
            f"coverage requires Alpaca SIP/all {timeframe} collection identity"
        )


def _file_record(path: Path, root: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": int(rows),
    }


def _resolve_inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise DataReadinessError("one-minute coverage path escapes its root")
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"one-minute coverage JSON is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise DataReadinessError(f"one-minute coverage JSON is not an object: {path}")
    return payload
