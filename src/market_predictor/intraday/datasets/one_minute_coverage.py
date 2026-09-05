"""Replay selected-session one-minute coverage and publish exclusions."""
from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.contracts.history_collection import (
    SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA,
)
from market_predictor.intraday.datasets.history import (
    json_sha256,
    load_complete_intraday_history_plan,
    load_plan_json,
)
from market_predictor.intraday.datasets.history_collection import (
    load_complete_intraday_history_collection,
)
from market_predictor.intraday.datasets.selection import (
    load_complete_intraday_selection,
)
from market_predictor.modeling.strategy_contract import StrategyContract

COVERAGE_SCHEMA = "edge_rebuild.selected_session_one_minute_coverage.v2"
COVERAGE_AUTHORITY_SCHEMA = "edge_rebuild.selected_session_one_minute_coverage_authority.v2"
_CANONICAL_SCHEMA: Final = "edge_rebuild.intraday_materialization.v1"
_CANONICAL_AUTHORITY_SCHEMA: Final = "edge_rebuild.intraday_materialization_authority.v1"
_CANONICAL_COLUMNS: Final = (
    "ticker",
    "session_date_et",
    "session_segment",
    "timeframe",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "source",
    "price_feed",
    "adjustment",
)


def publish_selected_session_one_minute_coverage(
    *,
    plan_directory: Path,
    collection_directory: Path,
    five_minute_canonical_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish exact pair-level coverage and whole-security exclusions."""

    if output_directory.exists():
        raise DataReadinessError(f"one-minute coverage output must be new: {output_directory}")
    plan = load_complete_intraday_history_plan(plan_directory)
    plan_request = load_plan_json(plan_directory / "_request.json")
    collection = load_complete_intraday_history_collection(collection_directory)
    collection_request = _read_json(collection_directory / "_request.json")
    _require_collection_identity(collection_request, timeframe="1Min")
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
        raise DataReadinessError("one-minute coverage inputs do not share the active strategy and plan")
    _verify_selection_lineage(selection)
    required = _required_pairs(plan_directory)
    observed = _observed_pair_rows(collection)
    five_minute_observed, canonical_identity = _canonical_pair_rows(
        five_minute_canonical_directory,
        required=required,
    )
    required_keys = set(
        zip(
            required["session_date_et"].astype(str),
            required["ticker"].astype(str),
            strict=True,
        )
    )
    unexpected = sorted(set(observed).difference(required_keys))
    if unexpected:
        raise DataReadinessError(f"one-minute collection contains {len(unexpected)} unplanned symbol-sessions")
    coverage = required.copy()
    keys = list(
        zip(
            coverage["session_date_et"].astype(str),
            coverage["ticker"].astype(str),
            strict=True,
        )
    )
    coverage["observed_rows"] = [observed.get(key, 0) for key in keys]
    coverage["one_minute_observation_density"] = coverage["observed_rows"].div(coverage["expected_rows"])
    coverage["observed_five_minute_rows"] = [five_minute_observed.get(key, 0) for key in keys]
    coverage["five_minute_bar_continuity"] = coverage["observed_five_minute_rows"].div(coverage["expected_five_minute_rows"])
    continuity_floor = strategy_contract.intraday_universe.minimum_bar_continuity
    coverage["coverage_status"] = (coverage["observed_rows"].gt(0) & coverage["five_minute_bar_continuity"].ge(continuity_floor)).map(
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
        "collection_manifest_sha256": file_sha256(collection_directory / "_manifest.json"),
        "five_minute_canonical_path": str(five_minute_canonical_directory),
        **canonical_identity,
        "strategy_contract_path": str(strategy_contract_path),
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "strategy_contract_sha256": strategy_contract.sha256(),
    }
    staging = output_directory.with_name(f".{output_directory.name}.{uuid.uuid4().hex}.staging")
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
                "complete_stock_sessions": int(coverage["coverage_status"].eq("complete").sum()),
                "incomplete_stock_sessions": int(coverage["coverage_status"].eq("incomplete").sum()),
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
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("ready_for_feature_build") != manifest.get("ready_for_feature_build")
    ):
        raise DataReadinessError(f"one-minute coverage lacks matching authority: {directory}")
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


def _verify_selection_lineage(selection: Mapping[str, Any]) -> None:
    selection_path = Path(str(selection.get("path", "")))
    try:
        manifest = load_complete_intraday_selection(selection_path)
    except (OSError, ValueError, DataReadinessError) as exc:
        raise DataReadinessError("one-minute plan does not reference a trusted current selection") from exc
    selected = next(
        (raw for raw in manifest.get("tables", []) if isinstance(raw, Mapping) and raw.get("path") == "selected_stock_sessions.parquet"),
        None,
    )
    if (
        not isinstance(selected, Mapping)
        or selection.get("manifest_sha256") != file_sha256(selection_path / "_manifest.json")
        or selection.get("request_sha256") != manifest.get("request_sha256")
        or selection.get("table_sha256") != selected.get("sha256")
        or selection.get("strategy_id") != manifest.get("strategy_id")
        or selection.get("strategy_contract_sha256") != manifest.get("strategy_contract_sha256")
        or int(selection.get("stock_sessions", -1)) != int(selected.get("rows", -2))
    ):
        raise DataReadinessError("one-minute plan selection lineage does not match its current authority")


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
    frame["session_date_et"] = pd.to_datetime(frame["session_date_et"], errors="raise").dt.date.astype(str)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["session_open_utc"] = pd.to_datetime(frame["session_open_utc"], utc=True, errors="raise")
    frame["session_close_utc"] = pd.to_datetime(frame["session_close_utc"], utc=True, errors="raise")
    duration = frame["session_close_utc"] - frame["session_open_utc"]
    one_minute = duration.dt.total_seconds().div(60)
    five_minute = duration.dt.total_seconds().div(300)
    if bool(one_minute.mod(1).ne(0).any()) or bool(five_minute.mod(1).ne(0).any()):
        raise DataReadinessError("selected XNYS session is not divisible into exact one/five-minute bars")
    frame["expected_rows"] = one_minute.astype("int32")
    frame["expected_five_minute_rows"] = five_minute.astype("int32")
    frame = frame.drop_duplicates().sort_values(["session_date_et", "ticker"], kind="stable")
    if (
        frame.empty
        or bool(frame["ticker"].eq("").any())
        or bool(frame["expected_rows"].le(0).any())
        or bool(frame["expected_five_minute_rows"].le(0).any())
        or bool(frame.duplicated(["session_date_et", "ticker"]).any())
    ):
        raise DataReadinessError("one-minute plan stock-session table is invalid")
    return frame.reset_index(drop=True)


def _canonical_pair_rows(
    directory: Path,
    *,
    required: pd.DataFrame,
) -> tuple[dict[tuple[str, str], int], dict[str, str]]:
    records, identity = verify_canonical_five_minute_store(directory)
    required_by_ticker = {str(ticker): group.copy() for ticker, group in required.groupby("ticker", sort=True)}
    regular = {str(record["ticker"]): record for record in records if record["store"] == "regular"}
    observed: dict[tuple[str, str], int] = {}
    for ticker, sessions in required_by_ticker.items():
        record = regular.get(ticker)
        if record is None:
            continue
        path = _resolve_inside(directory, str(record["path"]))
        try:
            bars = pd.read_parquet(path, columns=list(_CANONICAL_COLUMNS))
        except (OSError, ValueError, KeyError) as exc:
            raise DataReadinessError(f"canonical regular 5m identity failed for {ticker}") from exc
        _validate_canonical_identity(bars, ticker=ticker)
        bars["session_date_et"] = pd.to_datetime(bars["session_date_et"], errors="raise").dt.date.astype(str)
        bars["bar_start_utc"] = pd.to_datetime(bars["bar_start_utc"], utc=True, errors="raise")
        bars["bar_end_utc"] = pd.to_datetime(bars["bar_end_utc"], utc=True, errors="raise")
        bars["available_at_utc"] = pd.to_datetime(bars["available_at_utc"], utc=True, errors="raise")
        for row in sessions.itertuples(index=False):
            session = str(row.session_date_et)
            selected = bars.loc[bars["session_date_et"].eq(session)]
            if selected.empty:
                continue
            starts = pd.DatetimeIndex(selected["bar_start_utc"])
            expected = pd.date_range(
                start=pd.Timestamp(row.session_open_utc),
                end=pd.Timestamp(row.session_close_utc),
                freq="5min",
                inclusive="left",
            )
            if (
                starts.has_duplicates
                or not starts.isin(expected).all()
                or not selected["bar_end_utc"].eq(selected["bar_start_utc"] + pd.Timedelta(minutes=5)).all()
                or not selected["available_at_utc"].ge(selected["bar_end_utc"]).all()
            ):
                raise DataReadinessError(f"canonical regular 5m path is not causal and exact for {ticker} {session}")
            observed[(session, ticker)] = len(selected)
    return observed, identity


def verify_canonical_five_minute_store(
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _read_json(manifest_path)
    authority = _read_json(authority_path)
    integrity = manifest.get("integrity")
    if (
        manifest.get("schema") != _CANONICAL_SCHEMA
        or authority.get("schema") != _CANONICAL_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or not isinstance(integrity, Mapping)
        or int(integrity.get("blocking_defect_count", -1)) != 0
        or integrity.get("identity_breaks") not in ([], None)
        or integrity.get("fabricated_bars") not in ([], None)
    ):
        raise DataReadinessError(f"canonical regular 5m store lacks complete authority: {directory}")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("canonical regular 5m manifest has no files")
    records: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    total_rows = 0
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("canonical file record is malformed")
        record = {str(key): value for key, value in raw.items()}
        store = str(record.get("store", ""))
        ticker = str(record.get("ticker", "")).upper().strip()
        relative = str(record.get("path", "")).replace("\\", "/")
        identity = (store, ticker)
        if store not in {"regular", "extended"} or not ticker or identity in identities or not relative.startswith(f"{store}/5m/"):
            raise DataReadinessError("canonical file inventory identity is invalid")
        path = _resolve_inside(directory, relative)
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise DataReadinessError(f"canonical 5m file failed hash: {path}")
        try:
            parquet = pq.ParquetFile(path, memory_map=True)  # type: ignore[no-untyped-call]
        except Exception as exc:
            raise DataReadinessError(f"canonical 5m Parquet is unreadable: {path}") from exc
        rows = 0 if parquet.metadata is None else parquet.metadata.num_rows
        if rows <= 0 or rows != int(record.get("rows", -1)):
            raise DataReadinessError(f"canonical 5m file row count differs: {path}")
        record["ticker"] = ticker
        record["store"] = store
        records.append(record)
        identities.add(identity)
        total_rows += rows
    if "total_rows" in manifest and int(manifest["total_rows"]) != total_rows:
        raise DataReadinessError("canonical 5m manifest total row count differs")
    file_inventory_sha256 = json_sha256(
        [
            {
                "path": record["path"],
                "rows": int(record["rows"]),
                "sha256": record["sha256"],
                "store": record["store"],
                "ticker": record["ticker"],
            }
            for record in sorted(records, key=lambda item: str(item["path"]))
        ]
    )
    return records, {
        "five_minute_canonical_manifest_sha256": file_sha256(manifest_path),
        "five_minute_canonical_authority_sha256": file_sha256(authority_path),
        "five_minute_canonical_file_inventory_sha256": file_inventory_sha256,
    }


def _validate_canonical_identity(frame: pd.DataFrame, *, ticker: str) -> None:
    if (
        frame.empty
        or not frame["ticker"].astype(str).str.upper().eq(ticker).all()
        or not frame["session_segment"].astype(str).str.lower().eq("regular").all()
        or not frame["timeframe"].astype(str).str.lower().eq("5m").all()
        or not frame["source"].astype(str).str.lower().eq("alpaca").all()
        or not frame["price_feed"].astype(str).str.lower().eq("sip").all()
        or not frame["adjustment"].astype(str).str.lower().eq("all").all()
    ):
        raise DataReadinessError(f"canonical regular 5m identity failed for {ticker}")


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
                raise DataReadinessError(f"one-minute collection repeats symbol-session {key}")
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
    summary["five_minute_bar_continuity"] = summary["observed_five_minute_rows"].div(summary["expected_five_minute_rows"])
    excluded = summary.loc[summary["empty_sessions"].gt(0) | summary["five_minute_bar_continuity"].lt(continuity_floor)].copy()
    excluded["reason"] = "five_minute_continuity_below_floor"
    excluded.loc[excluded["empty_sessions"].gt(0), "reason"] = "one_or_more_selected_sessions_have_no_one_minute_trade_path"
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
        raise DataReadinessError(f"coverage requires Alpaca SIP/all {timeframe} collection identity")


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
