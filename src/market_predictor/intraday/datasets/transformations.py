"""Atomic, lineage-bound publisher for the causal intraday training dataset."""
from __future__ import annotations



from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa

from market_predictor.canonical.store import (
    file_sha256,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.intraday_history import (
    json_sha256,
)
from market_predictor.intraday.contracts.dataset_schemas import (
    _REQUIRED_BENCHMARKS,
    _SAFE_TICKER,
    _Artifact,
)


def _resolve_inside(parent: Path, subpath: str) -> Path:
    resolved = (parent / subpath).resolve()
    if parent.resolve() not in resolved.parents:
        raise ValueError(f"Path escape attempt: {subpath}")
    return resolved


def _normalize_arrow_records(
    records: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
) -> list[dict[str, Any]]:
    return [
        {
            field.name: _normalize_arrow_value(record.get(field.name), field.type)
            for field in schema
        }
        for record in records
    ]

def _normalize_arrow_value(value: object, data_type: pa.DataType) -> object:
    if _is_missing(value):
        return None
    if pa.types.is_string(data_type):
        return str(value)
    if pa.types.is_integer(data_type):
        return int(cast(int, value))
    if pa.types.is_timestamp(data_type):
        return pd.Timestamp(value)
    return value

def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    try:
        return bool(missing)
    except ValueError:
        return False

def _membership_sector_exclusions(
    memberships: pd.DataFrame,
    *,
    selected_tickers: set[str],
) -> frozenset[str]:
    required = {"ticker", "primary_benchmark"}
    if not required.issubset(memberships.columns):
        raise DataReadinessError(
            "membership authority omits sector benchmark identity"
        )
    data = memberships.loc[:, sorted(required)].copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["primary_benchmark"] = (
        data["primary_benchmark"].astype(str).str.upper().str.strip()
    )
    sector_benchmarks = _REQUIRED_BENCHMARKS.difference({"SPY", "QQQ"})
    selected = data["ticker"].isin(selected_tickers)
    invalid = selected & ~data["primary_benchmark"].isin(sector_benchmarks)
    return frozenset(data.loc[invalid, "ticker"].astype(str))

def _normalize_selection(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["session_date_et"] = pd.to_datetime(data["session_date_et"], errors="raise").dt.date.astype(str)
    data["activation_time_utc"] = pd.to_datetime(data["activation_time_utc"], utc=True, errors="raise")
    data["median_volume_prior_sessions"] = pd.to_numeric(data["median_volume_prior_sessions"], errors="coerce")
    local_dates = data["activation_time_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
    if (
        bool(data["ticker"].map(lambda value: _SAFE_TICKER.fullmatch(value) is None).any())
        or bool(local_dates.ne(data["session_date_et"]).any())
        or bool(data["activation_time_utc"].dt.second.ne(0).any())
        or bool(data["activation_time_utc"].dt.microsecond.ne(0).any())
        or bool(data["median_volume_prior_sessions"].le(0).any())
    ):
        raise DataReadinessError("selection contains invalid causal activation rows")
    return data.sort_values(["session_date_et", "activation_time_utc", "ticker"], kind="stable").reset_index(drop=True)

def _collection_artifacts(root: Path, manifest: Mapping[str, Any]) -> tuple[_Artifact, ...]:
    output: list[_Artifact] = []
    for raw in cast(list[Mapping[str, Any]], manifest["artifacts"]):
        symbol_rows_raw = raw.get("symbol_rows")
        if not isinstance(symbol_rows_raw, Mapping):
            raise DataReadinessError("collection artifact lacks symbol row counts")
        symbol_rows = {str(key).upper().strip(): int(value) for key, value in symbol_rows_raw.items()}
        output.append(
            _Artifact(
                path=_resolve_inside(root, str(raw.get("path", ""))),
                session_date_et=str(raw.get("asof_date", "")),
                symbol_rows=symbol_rows,
                sha256=str(raw.get("sha256", "")),
            )
        )
        if len(output[-1].sha256) != 64:
            raise DataReadinessError("collection artifact lacks a valid SHA-256")
    return tuple(output)

def _stock_artifact_index(
    artifacts: tuple[_Artifact, ...],
) -> dict[tuple[str, str], _Artifact]:
    index: dict[tuple[str, str], _Artifact] = {}
    for artifact in artifacts:
        for ticker, rows in artifact.symbol_rows.items():
            if rows <= 0:
                continue
            key = (artifact.session_date_et, ticker)
            if key in index:
                raise DataReadinessError(f"stock collection repeats {key}")
            index[key] = artifact
    return index

def _benchmark_artifact_index(
    artifacts: tuple[_Artifact, ...],
) -> dict[str, tuple[_Artifact, ...]]:
    by_session: dict[str, list[_Artifact]] = {}
    for artifact in artifacts:
        by_session.setdefault(artifact.session_date_et, []).append(artifact)
    return {key: tuple(value) for key, value in by_session.items()}

def _load_stock_session_batch(
    session_date: str,
    tickers: list[str],
    *,
    artifacts: Mapping[tuple[str, str], _Artifact],
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    requested = set(tickers)
    if not requested or len(requested) != len(tickers):
        raise DataReadinessError(
            f"stock session batch has empty or duplicate identities for {session_date}"
        )
    by_path: dict[Path, tuple[_Artifact, set[str]]] = {}
    for ticker in sorted(requested):
        artifact = artifacts.get((session_date, ticker))
        if artifact is None:
            raise DataReadinessError(
                f"stock one-minute path is missing for {(session_date, ticker)}"
            )
        existing = by_path.get(artifact.path)
        if existing is None:
            by_path[artifact.path] = (artifact, {ticker})
        else:
            existing[1].add(ticker)

    coverage_session = coverage.loc[
        coverage["session_date_et"].eq(session_date)
        & coverage["ticker"].isin(requested)
    ].copy()
    if len(coverage_session) != len(requested):
        raise DataReadinessError(
            f"stock one-minute coverage is incomplete for {session_date}"
        )
    expected_coverage = {
        str(row.ticker): int(row.observed_rows)
        for row in coverage_session.itertuples(index=False)
    }
    frames: list[pd.DataFrame] = []
    for path, (artifact, path_tickers) in sorted(
        by_path.items(), key=lambda item: str(item[0])
    ):
        if file_sha256(path) != artifact.sha256:
            raise DataReadinessError(
                f"stock one-minute artifact hash differs for {session_date}: {path}"
            )
        frame = pd.read_parquet(path)
        normalized = frame["ticker"].astype(str).str.upper().str.strip()
        selected = frame.loc[normalized.isin(path_tickers)].copy()
        selected["ticker"] = normalized.loc[selected.index]
        observed = {
            str(ticker): int(rows)
            for ticker, rows in selected.groupby("ticker", observed=True).size().items()
        }
        expected = {ticker: int(artifact.symbol_rows[ticker]) for ticker in path_tickers}
        if observed != expected:
            raise DataReadinessError(
                f"stock one-minute artifact rows differ for {session_date}: {path}"
            )
        for ticker, rows in expected.items():
            if expected_coverage.get(ticker) != rows:
                raise DataReadinessError(
                    f"stock one-minute coverage row count differs for {(session_date, ticker)}"
                )
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    if set(combined["ticker"].astype(str)) != requested:
        raise DataReadinessError(
            f"stock one-minute batch identity differs for {session_date}"
        )
    return combined

def _load_benchmark_session(
    session_date: str,
    *,
    artifacts: Mapping[str, tuple[_Artifact, ...]],
    required_tickers: frozenset[str],
) -> pd.DataFrame:
    paths = artifacts.get(session_date)
    if not paths:
        raise DataReadinessError(f"benchmark one-minute path is missing for {session_date}")
    missing_paths = [artifact.path for artifact in paths if not artifact.path.is_file()]
    if missing_paths:
        raise DataReadinessError(f"benchmark one-minute path is missing: {missing_paths[0]}")
    frames = []
    for artifact in paths:
        if file_sha256(artifact.path) != artifact.sha256:
            raise DataReadinessError(
                f"benchmark one-minute artifact hash differs for {session_date}: "
                f"{artifact.path}"
            )
        frames.append(pd.read_parquet(artifact.path))
    frame = pd.concat(frames, ignore_index=True)
    expected_rows: dict[str, int] = {}
    for artifact in paths:
        for ticker, rows in artifact.symbol_rows.items():
            expected_rows[ticker] = expected_rows.get(ticker, 0) + int(rows)
    observed_rows = {
        str(ticker): int(rows)
        for ticker, rows in frame.groupby("ticker", observed=True).size().items()
    }
    if observed_rows != expected_rows:
        raise DataReadinessError(
            f"benchmark session {session_date} row counts differ from collection authority"
        )
    observed = set(frame["ticker"].astype(str).str.upper().str.strip())
    if observed != set(required_tickers):
        missing = sorted(set(required_tickers).difference(observed))
        extra = sorted(observed.difference(required_tickers))
        raise DataReadinessError(f"benchmark session {session_date} identity differs; missing={missing}, extra={extra}")
    if bool(frame.duplicated(["ticker", "bar_start_utc"]).any()):
        raise DataReadinessError(f"benchmark session {session_date} repeats minute rows")
    starts = pd.to_datetime(frame["bar_start_utc"], utc=True, errors="raise")
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(session_date)
    open_at = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
    close_at = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
    expected = set(pd.date_range(open_at, close_at, freq="1min", inclusive="left"))
    for ticker, indices in frame.groupby("ticker", sort=False, observed=True).groups.items():
        if not set(starts.loc[indices]).issubset(expected):
            raise DataReadinessError(
                f"benchmark minute path exceeds the exchange session for {(session_date, str(ticker))}"
            )
    return frame

def _load_coverage_tables(root: Path, manifest: Mapping[str, Any]) -> tuple[pd.DataFrame, set[str]]:
    records = {str(raw["path"]): raw for raw in cast(list[Mapping[str, Any]], manifest["files"])}
    coverage_path = _resolve_inside(root, "stock_session_coverage.parquet")
    exclusions_path = _resolve_inside(root, "excluded_securities.parquet")
    if "stock_session_coverage.parquet" not in records or "excluded_securities.parquet" not in records:
        raise DataReadinessError("coverage authority omits required tables")
    coverage = pd.read_parquet(coverage_path)
    exclusions = pd.read_parquet(exclusions_path)
    required = {"ticker", "session_date_et", "observed_rows", "coverage_status"}
    if not required.issubset(coverage.columns) or "ticker" not in exclusions.columns:
        raise DataReadinessError("coverage tables have invalid schemas")
    coverage["ticker"] = coverage["ticker"].astype(str).str.upper().str.strip()
    coverage["session_date_et"] = pd.to_datetime(coverage["session_date_et"], errors="raise").dt.date.astype(str)
    return coverage, set(exclusions["ticker"].astype(str).str.upper().str.strip())

def _membership_for_pair(
    memberships: pd.DataFrame,
    *,
    ticker: str,
    session_date: str,
) -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(session_date)
    session_open = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
    session_close = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
    rows = memberships[memberships["ticker"].astype(str).str.upper().str.strip().eq(ticker)].copy()
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="coerce")
    active = (
        rows["effective_from_utc"].le(session_open)
        & rows["available_at_utc"].le(session_open)
        & (rows["effective_to_utc"].isna() | rows["effective_to_utc"].gt(session_open))
    )
    rows = rows.loc[active].copy()
    if len(rows) != 1:
        raise DataReadinessError(f"PIT membership is not unique for {(session_date, ticker)}")
    rows["session_date_et"] = date.fromisoformat(session_date)
    rows["session_open_utc"] = session_open
    rows["session_close_utc"] = session_close
    return rows

def _finalize_dataset_rows(
    frame: pd.DataFrame,
    *,
    request_sha256: str,
    parent_lineage_sha256: str,
) -> pd.DataFrame:
    data = frame.copy()
    data["dataset_eligible"] = data["feature_eligible"].astype(bool) & data["label_eligible"].astype(bool) & data["rank_label"].notna()
    data["dataset_ineligible_reason"] = pd.Series(pd.NA, index=data.index, dtype="string")
    feature_bad = ~data["feature_eligible"].astype(bool)
    label_bad = ~feature_bad & ~data["label_eligible"].astype(bool)
    rank_bad = ~feature_bad & ~label_bad & data["rank_label"].isna()
    data.loc[feature_bad, "dataset_ineligible_reason"] = "feature:" + data.loc[feature_bad, "feature_ineligible_reason"].astype(str)
    data.loc[label_bad, "dataset_ineligible_reason"] = "label:" + data.loc[label_bad, "label_ineligible_reason"].astype(str)
    data.loc[rank_bad, "dataset_ineligible_reason"] = "rank:insufficient_contemporaneous_group"
    data["dataset_row_id"] = [
        json_sha256(
            {
                "ticker": str(row.ticker),
                "session_date_et": str(row.session_date_et),
                "volume_bar_number": int(row.volume_bar_number),
                "feature_available_at_utc": pd.Timestamp(row.feature_available_at_utc).isoformat(),
                "request_sha256": request_sha256,
            }
        )
        for row in data.itertuples(index=False)
    ]
    data["dataset_request_sha256"] = request_sha256
    data["parent_lineage_sha256"] = parent_lineage_sha256
    return data

def _split_decision_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn bars unavailable before the close into explicit abstentions.

    The feature builder intentionally preserves every completed volume bar,
    including a final bar whose provider availability can be after the close.
    Such a row is valid feature evidence but cannot start a next-minute trade.
    """

    feature_at = pd.to_datetime(frame["feature_available_at_utc"], utc=True, errors="raise")
    session_close = pd.to_datetime(frame["session_close_utc"], utc=True, errors="raise")
    closed = feature_at.ge(session_close)
    decisions = frame.loc[~closed].copy()
    abstained = frame.loc[closed].copy()
    if not abstained.empty:
        abstained["feature_eligible"] = False
        abstained["feature_ineligible_reason"] = "feature_available_at_or_after_session_close"
    return decisions, abstained