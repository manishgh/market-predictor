from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow.dataset as ds
from pydantic import Field

from market_predictor.v3.partitions import DEFAULT_DEVELOPMENT_CUTOFF_UTC
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract

DEFAULT_REQUIRED_BENCHMARKS = ("SPY", "QQQ", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY")
POINT_IN_TIME_UNIVERSE_COLUMNS = {
    "ticker",
    "effective_from_utc",
    "effective_to_utc",
    "sector",
    "industry",
    "market_cap_bucket",
    "liquidity_bucket",
    "primary_benchmark",
    "universe_snapshot_id",
}


class DevelopmentReadinessConfig(FrozenContract):
    minimum_tickers: int = Field(default=300, ge=2)
    minimum_sessions: int = Field(default=252, ge=2)
    required_benchmarks: tuple[str, ...] = DEFAULT_REQUIRED_BENCHMARKS
    development_cutoff_utc: datetime = DEFAULT_DEVELOPMENT_CUTOFF_UTC
    require_sip: bool = True
    schema_version: str = ML_V3_SCHEMA_VERSION


class ReadinessCheck(FrozenContract):
    name: str
    status: Literal["pass", "fail"]
    observed: Any = None
    required: Any = None


def audit_development_readiness(
    *,
    bars_path: Path,
    universe_path: Path,
    benchmark_dir: Path,
    config: DevelopmentReadinessConfig = DevelopmentReadinessConfig(),
) -> dict[str, Any]:
    checks: list[ReadinessCheck] = []
    bar_summary = _scan_bars(bars_path)
    checks.extend(_bar_checks(bar_summary, config))
    universe_summary = _scan_universe(universe_path)
    checks.extend(_universe_checks(universe_summary, config))
    benchmark_summary = _scan_benchmarks(benchmark_dir, config.required_benchmarks)
    checks.extend(_benchmark_checks(benchmark_summary, bar_summary, config))
    failures = [check.name for check in checks if check.status == "fail"]
    return {
        "schema": "ml_v3.development_readiness.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "development_cutoff_utc": config.development_cutoff_utc.isoformat(),
        "ready": not failures,
        "failures": failures,
        "checks": [check.model_dump(mode="json") for check in checks],
        "bars": bar_summary,
        "universe": universe_summary,
        "benchmarks": benchmark_summary,
    }


def _scan_bars(path: Path) -> dict[str, Any]:
    dataset = _parquet_dataset(path)
    names = set(dataset.schema.names)
    ticker_column = "ticker" if "ticker" in names else "symbol" if "symbol" in names else None
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(names))
    if ticker_column is None:
        missing.append("ticker|symbol")
    if missing:
        return {"path": str(path), "columns": sorted(names), "missing_columns": missing, "rows": 0}
    projection = [ticker_column, "timestamp"]
    if "price_feed" in names:
        projection.append("price_feed")
    tickers: set[str] = set()
    sessions: set[object] = set()
    feeds: set[str] = set()
    rows = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    regular_first: pd.Timestamp | None = None
    regular_last: pd.Timestamp | None = None
    regular_rows = 0
    for batch in dataset.scanner(columns=projection, batch_size=250_000).to_batches():
        frame = batch.to_pandas()
        rows += len(frame)
        tickers.update(frame[ticker_column].dropna().astype(str).str.upper().str.strip())
        timestamp = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not timestamp.empty:
            batch_first = timestamp.min()
            batch_last = timestamp.max()
            first = batch_first if first is None or batch_first < first else first
            last = batch_last if last is None or batch_last > last else last
            sessions.update(timestamp.dt.tz_convert("America/New_York").dt.date)
            eastern = timestamp.dt.tz_convert("America/New_York")
            minute = eastern.dt.hour * 60 + eastern.dt.minute
            regular = timestamp[minute.between(9 * 60 + 30, 16 * 60 - 1)]
            regular_rows += len(regular)
            if not regular.empty:
                batch_regular_first = regular.min()
                batch_regular_last = regular.max()
                regular_first = batch_regular_first if regular_first is None or batch_regular_first < regular_first else regular_first
                regular_last = batch_regular_last if regular_last is None or batch_regular_last > regular_last else regular_last
        if "price_feed" in frame:
            feeds.update(frame["price_feed"].dropna().astype(str).str.lower().str.strip())
    return {
        "path": str(path),
        "columns": sorted(names),
        "missing_columns": [],
        "rows": rows,
        "tickers": len(tickers),
        "ticker_sample": sorted(tickers)[:20],
        "sessions": len(sessions),
        "first_timestamp_utc": first.isoformat() if first is not None else None,
        "last_timestamp_utc": last.isoformat() if last is not None else None,
        "regular_rows": regular_rows,
        "regular_first_timestamp_utc": regular_first.isoformat() if regular_first is not None else None,
        "regular_last_timestamp_utc": regular_last.isoformat() if regular_last is not None else None,
        "price_feeds": sorted(feeds),
        "price_feed_declared": "price_feed" in names,
    }


def _scan_universe(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "rows": 0, "columns": [], "missing_columns": sorted(POINT_IN_TIME_UNIVERSE_COLUMNS)}
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    columns = set(frame.columns)
    ticker_column = "ticker" if "ticker" in columns else "Ticker" if "Ticker" in columns else None
    tickers = int(frame[ticker_column].nunique()) if ticker_column else 0
    return {
        "path": str(path),
        "exists": True,
        "rows": len(frame),
        "tickers": tickers,
        "columns": sorted(columns),
        "missing_columns": sorted(POINT_IN_TIME_UNIVERSE_COLUMNS.difference(columns)),
    }


def _scan_benchmarks(directory: Path, symbols: tuple[str, ...]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for symbol in symbols:
        path = directory / f"{symbol}.parquet"
        if not path.exists():
            records[symbol] = {"exists": False, "path": str(path)}
            continue
        summary = _scan_bars(path)
        summary["exists"] = True
        records[symbol] = summary
    return records


def _bar_checks(summary: dict[str, Any], config: DevelopmentReadinessConfig) -> list[ReadinessCheck]:
    missing = summary.get("missing_columns", [])
    tickers = int(summary.get("tickers", 0))
    sessions = int(summary.get("sessions", 0))
    feeds = set(summary.get("price_feeds", []))
    last = pd.Timestamp(summary["last_timestamp_utc"]) if summary.get("last_timestamp_utc") else None
    return [
        _check("bars_schema", not missing, missing, "no missing OHLCV identity columns"),
        _check("minimum_tickers", tickers >= config.minimum_tickers, tickers, config.minimum_tickers),
        _check("minimum_sessions", sessions >= config.minimum_sessions, sessions, config.minimum_sessions),
        _check(
            "development_cutoff",
            last is not None and last <= pd.Timestamp(config.development_cutoff_utc),
            last.isoformat() if last is not None else None,
            config.development_cutoff_utc.isoformat(),
        ),
        _check(
            "sip_feed_provenance",
            (not config.require_sip) or (bool(summary.get("price_feed_declared")) and feeds == {"sip"}),
            sorted(feeds) if feeds else "not declared",
            "sip" if config.require_sip else "any declared feed",
        ),
    ]


def _universe_checks(summary: dict[str, Any], config: DevelopmentReadinessConfig) -> list[ReadinessCheck]:
    missing = summary.get("missing_columns", [])
    tickers = int(summary.get("tickers", 0))
    return [
        _check("point_in_time_universe_schema", not missing, missing, "all effective membership columns"),
        _check("universe_minimum_tickers", tickers >= config.minimum_tickers, tickers, config.minimum_tickers),
    ]


def _benchmark_checks(
    summaries: dict[str, Any],
    bars: dict[str, Any],
    config: DevelopmentReadinessConfig,
) -> list[ReadinessCheck]:
    missing = sorted(symbol for symbol, summary in summaries.items() if not summary.get("exists"))
    coverage_failures: list[str] = []
    feed_failures: list[str] = []
    bar_start = pd.Timestamp(bars["regular_first_timestamp_utc"]) if bars.get("regular_first_timestamp_utc") else None
    bar_end = pd.Timestamp(bars["regular_last_timestamp_utc"]) if bars.get("regular_last_timestamp_utc") else None
    for symbol, summary in summaries.items():
        if not summary.get("exists"):
            continue
        start = pd.Timestamp(summary["regular_first_timestamp_utc"]) if summary.get("regular_first_timestamp_utc") else None
        end = pd.Timestamp(summary["regular_last_timestamp_utc"]) if summary.get("regular_last_timestamp_utc") else None
        if bar_start is None or bar_end is None or start is None or end is None or start > bar_start or end < bar_end:
            coverage_failures.append(symbol)
        if config.require_sip and (not summary.get("price_feed_declared") or set(summary.get("price_feeds", [])) != {"sip"}):
            feed_failures.append(symbol)
    return [
        _check("required_benchmarks", not missing, missing, list(config.required_benchmarks)),
        _check("benchmark_time_coverage", not coverage_failures, coverage_failures, "full ticker-bar interval"),
        _check("benchmark_sip_provenance", not feed_failures, feed_failures, "sip"),
    ]


def _parquet_dataset(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet dataset: {path}")
    return ds.dataset(path, format="parquet")  # type: ignore[no-untyped-call]


def _check(name: str, passed: bool, observed: Any, required: Any) -> ReadinessCheck:
    return ReadinessCheck(name=name, status="pass" if passed else "fail", observed=observed, required=required)
