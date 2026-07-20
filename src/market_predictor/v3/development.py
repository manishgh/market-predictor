from __future__ import annotations

import gc
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd
import pyarrow.parquet as pq
from pydantic import Field

from market_predictor.registry import file_sha256
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.features import build_v3_ticker_features, finalize_v3_cross_sectional_features
from market_predictor.v3.labels import V3LabelConfig, build_v3_labels
from market_predictor.v3.schema import FrozenContract

DEVELOPMENT_BUILDER_SCHEMA = "ml_v3.development_dataset_build.v2"


class DevelopmentDatasetConfig(FrozenContract):
    minimum_cross_section: int = Field(default=300, ge=2)
    workers: int = Field(default=4, ge=1, le=16)
    decision_stride_bars: int = Field(default=12, ge=1)
    rotate_decision_offset_by_session: bool = True
    horizons_bars: tuple[int, ...] = (6, 12, 24)
    primary_horizon_bars: int = 12
    bar_minutes: int = 5
    round_trip_cost_bps: float = 10.0
    decision_start_date: date


def build_monthly_development_dataset(
    *,
    bars_directory: Path,
    benchmark_directory: Path,
    memberships_path: Path,
    technical_directory: Path,
    output_directory: Path,
    config: DevelopmentDatasetConfig,
    source_availability_path: Path | None = None,
    reuse_technical: bool = False,
    resume_output: bool = False,
) -> dict[str, Any]:
    """Build memory-bounded V3 labels through per-ticker and monthly stages."""
    if resume_output and not reuse_technical:
        raise ValueError("resume_output requires reuse_technical")
    if resume_output:
        if not output_directory.is_dir():
            raise DataReadinessError(f"Missing development output directory to resume: {output_directory}")
    else:
        _require_new_directory(output_directory)
    memberships = _read_frame(memberships_path)
    availability = _read_frame(source_availability_path) if source_availability_path is not None else None
    bar_files = sorted(bars_directory.glob("*.parquet"))
    if not bar_files:
        raise DataReadinessError(f"No per-symbol parquet files found in {bars_directory}")
    output_directory.mkdir(parents=True, exist_ok=resume_output)
    memberships_sha256 = file_sha256(memberships_path)
    technical_manifest_path = technical_directory / "_technical_manifest.json"
    ticker_results: list[dict[str, Any]]
    failures: list[dict[str, str]]
    if reuse_technical:
        ticker_results, failures = _load_technical_manifest(
            technical_manifest_path,
            bars_directory=bars_directory,
            memberships_sha256=memberships_sha256,
            expected_tickers=len(bar_files),
        )
    else:
        _require_new_directory(technical_directory)
        technical_directory.mkdir(parents=True)
        ticker_results, failures = _build_all_ticker_shards(
            bar_files=bar_files,
            memberships=memberships,
            availability=availability,
            technical_directory=technical_directory,
            workers=config.workers,
        )
    report: dict[str, Any] = {
        "schema": DEVELOPMENT_BUILDER_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "inputs": {
            "bars_directory": str(bars_directory),
            "benchmark_directory": str(benchmark_directory),
            "memberships_path": str(memberships_path),
            "memberships_sha256": memberships_sha256,
            "source_availability_path": str(source_availability_path) if source_availability_path else None,
            "market_calendar": "XNYS",
            "market_calendar_package_version": version("exchange-calendars"),
        },
        "ticker_shards": sorted(ticker_results, key=lambda item: str(item["ticker"])),
        "failures": sorted(failures, key=lambda item: item["path"]),
        "technical_reused": reuse_technical,
        "months": [],
    }
    if resume_output:
        report = _load_output_manifest(output_directory / "_build_manifest.json", expected=report)
        report["technical_reused"] = True
    else:
        _write_report(output_directory, report)
    if not reuse_technical:
        technical_manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise DataReadinessError(f"{len(failures)} ticker feature shards failed; see {output_directory / '_build_manifest.json'}")

    label_config = V3LabelConfig(
        horizons_bars=config.horizons_bars,
        primary_horizon_bars=config.primary_horizon_bars,
        bar_minutes=config.bar_minutes,
        round_trip_cost_bps=config.round_trip_cost_bps,
        minimum_ranking_group=config.minimum_cross_section,
        decision_stride_bars=config.decision_stride_bars,
        rotate_decision_offset_by_session=config.rotate_decision_offset_by_session,
    )
    month_results = list(report.get("months", []))
    completed_months = {str(item["month"]) for item in month_results}
    for month_directory in sorted(technical_directory.glob("month_*")):
        month = month_directory.name.removeprefix("month_")
        if pd.Period(month, freq="M").end_time.date() < config.decision_start_date:
            continue
        if month in completed_months:
            continue
        technical = pd.read_parquet(month_directory)
        if technical.empty:
            continue
        benchmarks = _read_benchmark_window(
            benchmark_directory,
            start=pd.Timestamp(technical["timestamp"].min()),
            end=pd.Timestamp(technical["timestamp"].max()),
        )
        technical_rows_before_grid = len(technical)
        technical = _restrict_to_market_session_grid(technical, benchmarks, interval_minutes=config.bar_minutes)
        off_grid_rows_removed = technical_rows_before_grid - len(technical)
        labels = build_v3_labels(technical, benchmarks, config=label_config, partition="development")
        labels = labels[pd.to_datetime(labels["session_date_et"]).dt.date >= config.decision_start_date].reset_index(drop=True)
        labels["_session_date_et"] = pd.to_datetime(labels["session_date_et"]).dt.date
        features = finalize_v3_cross_sectional_features(
            labels,
            benchmarks,
            minimum_cross_section=config.minimum_cross_section,
        ).drop(columns="_session_date_et")
        output_path = output_directory / f"{month}.parquet"
        features.to_parquet(output_path, index=False)
        month_results.append(
            {
                "month": month,
                "technical_rows": technical_rows_before_grid,
                "market_grid_rows": len(technical),
                "off_grid_rows_removed": off_grid_rows_removed,
                "feature_rows": len(features),
                "label_rows": len(features),
                "tickers": int(features["ticker"].nunique()) if not features.empty else 0,
                "sessions": int(features["session_date_et"].nunique()) if not features.empty else 0,
                "path": str(output_path),
                "sha256": file_sha256(output_path),
            }
        )
        month_results.sort(key=lambda item: str(item["month"]))
        report["months"] = month_results
        _write_report(output_directory, report)
        del technical, benchmarks, features, labels
        gc.collect()
    if not month_results or sum(int(item["label_rows"]) for item in month_results) == 0:
        raise DataReadinessError("No monthly V3 development labels were produced")
    report["months"] = month_results
    report["summary"] = {
        "tickers": len(ticker_results),
        "technical_rows": sum(int(item["regular_membership_rows"]) for item in ticker_results),
        "label_rows": sum(int(item["label_rows"]) for item in month_results),
        "months": len(month_results),
        "first_month": month_results[0]["month"],
        "last_month": month_results[-1]["month"],
    }
    report["dataset_fingerprint"] = _dataset_fingerprint(month_results, config)
    _write_report(output_directory, report)
    return report


def load_verified_development_dataset(
    directory: Path,
    *,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a completed monthly dataset only after validating its frozen manifest."""
    manifest_path = directory / "_build_manifest.json"
    if not directory.is_dir() or not manifest_path.exists():
        raise DataReadinessError(f"Missing completed development dataset manifest: {manifest_path}")
    try:
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"Development dataset manifest is unreadable: {manifest_path}") from exc
    if manifest.get("schema") != DEVELOPMENT_BUILDER_SCHEMA:
        raise DataReadinessError(f"Unsupported development dataset schema: {manifest.get('schema')}")
    months = manifest.get("months")
    summary = manifest.get("summary")
    if not isinstance(months, list) or not months or not isinstance(summary, dict):
        raise DataReadinessError("Development dataset manifest is incomplete")
    expected_paths: list[Path] = []
    for item in months:
        month = str(item.get("month", ""))
        artifact = directory / f"{month}.parquet"
        if not artifact.exists() or file_sha256(artifact) != item.get("sha256"):
            raise DataReadinessError(f"Development dataset shard is missing or hash-invalid: {artifact}")
        expected_paths.append(artifact)
    actual_paths = sorted(directory.glob("*.parquet"))
    if {path.resolve() for path in actual_paths} != {path.resolve() for path in expected_paths}:
        raise DataReadinessError("Development dataset contains unregistered parquet shards")
    fingerprint_payload = {
        "builder_schema": DEVELOPMENT_BUILDER_SCHEMA,
        "config": manifest.get("config"),
        "months": [{"month": item["month"], "sha256": item["sha256"]} for item in months],
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")).hexdigest()
    if fingerprint != manifest.get("dataset_fingerprint"):
        raise DataReadinessError("Development dataset fingerprint does not match its manifest")
    projected_columns = columns
    if columns is not None:
        available = set(cast(Any, pq.read_schema)(expected_paths[0]).names)
        projected_columns = [column for column in columns if column in available]
    try:
        dataset = pd.read_parquet(directory, columns=projected_columns)
    except (KeyError, ValueError) as exc:
        raise DataReadinessError("Development dataset does not contain the requested training projection") from exc
    if len(dataset) != int(summary.get("label_rows", -1)):
        raise DataReadinessError("Development dataset physical row count does not match its manifest")
    return dataset, manifest


def _build_all_ticker_shards(
    *,
    bar_files: list[Path],
    memberships: pd.DataFrame,
    availability: pd.DataFrame | None,
    technical_directory: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    ticker_results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_build_ticker_shards, path, memberships, availability, technical_directory): path
            for path in bar_files
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                ticker_results.append(future.result())
            except Exception as exc:
                failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return ticker_results, failures


def _load_technical_manifest(
    path: Path,
    *,
    bars_directory: Path,
    memberships_sha256: str,
    expected_tickers: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not path.exists():
        raise DataReadinessError(f"Missing technical resume manifest: {path}")
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    inputs = payload.get("inputs", {})
    ticker_results = payload.get("ticker_shards", [])
    failures = payload.get("failures", [])
    if inputs.get("bars_directory") != str(bars_directory):
        raise DataReadinessError("Technical resume manifest bars directory does not match")
    if inputs.get("memberships_sha256") != memberships_sha256:
        raise DataReadinessError("Technical resume manifest membership hash does not match")
    if failures or len(ticker_results) != expected_tickers:
        raise DataReadinessError(
            f"Technical resume manifest is incomplete: tickers={len(ticker_results)}/{expected_tickers}, failures={len(failures)}"
        )
    return list(ticker_results), list(failures)


def _load_output_manifest(path: Path, *, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        raise DataReadinessError(f"Missing development resume manifest: {path}")
    payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
    if payload.get("schema") != expected["schema"]:
        raise DataReadinessError("Development resume manifest schema does not match")
    if payload.get("config") != expected["config"]:
        raise DataReadinessError("Development resume manifest config does not match")
    if payload.get("inputs") != expected["inputs"]:
        raise DataReadinessError("Development resume manifest inputs do not match")
    if payload.get("failures"):
        raise DataReadinessError("Development resume manifest contains ticker failures")
    for month in payload.get("months", []):
        artifact = Path(str(month.get("path", "")))
        if not artifact.exists() or file_sha256(artifact) != month.get("sha256"):
            raise DataReadinessError(f"Development resume shard is missing or hash-invalid: {artifact}")
    return payload


def _build_ticker_shards(
    path: Path,
    memberships: pd.DataFrame,
    availability: pd.DataFrame | None,
    technical_directory: Path,
) -> dict[str, Any]:
    bars = pd.read_parquet(path)
    if "ticker" not in bars.columns and "symbol" in bars.columns:
        bars = bars.rename(columns={"symbol": "ticker"})
    if "ticker" not in bars.columns or bars.empty:
        raise DataReadinessError(f"Ticker bars are empty or missing identity: {path}")
    tickers = bars["ticker"].dropna().astype(str).str.upper().str.strip().unique()
    if len(tickers) != 1:
        raise DataReadinessError(f"Per-symbol parquet must contain exactly one ticker: {path}")
    ticker = str(tickers[0])
    ticker_memberships = memberships[memberships["ticker"].astype(str).str.upper().str.strip() == ticker].copy()
    if ticker_memberships.empty:
        raise DataReadinessError(f"No point-in-time membership interval for {ticker}")
    ticker_memberships["effective_from_utc"] = pd.to_datetime(ticker_memberships["effective_from_utc"], utc=True)
    ticker_memberships["effective_to_utc"] = pd.to_datetime(ticker_memberships["effective_to_utc"], utc=True)
    first_membership = ticker_memberships.sort_values("effective_from_utc").iloc[0]
    bars["ticker"] = ticker
    bars["primary_benchmark"] = first_membership["primary_benchmark"]
    bars["universe_snapshot_id"] = first_membership["universe_snapshot_id"]
    ticker_availability = None
    if availability is not None and not availability.empty:
        ticker_availability = availability[availability["ticker"].astype(str).str.upper().str.strip() == ticker]
    technical = build_v3_ticker_features(bars, source_availability=ticker_availability)
    eligible_parts: list[pd.DataFrame] = []
    for membership in ticker_memberships.to_dict(orient="records"):
        start = pd.Timestamp(membership["effective_from_utc"])
        end = pd.Timestamp(membership["effective_to_utc"]) if pd.notna(membership["effective_to_utc"]) else None
        mask = technical["timestamp"].ge(start)
        if end is not None:
            mask &= technical["timestamp"].lt(end)
        part = technical.loc[mask].copy()
        if part.empty:
            continue
        for column in (
            "primary_benchmark",
            "universe_snapshot_id",
            "sector",
            "industry",
            "market_cap_bucket",
            "liquidity_bucket",
        ):
            if column in membership:
                part[column] = membership[column]
        eligible_parts.append(part)
    if not eligible_parts:
        raise DataReadinessError(f"No bars fall inside point-in-time membership for {ticker}")
    eligible = pd.concat(eligible_parts, ignore_index=True).drop_duplicates(["ticker", "timestamp"])
    eastern = eligible["timestamp"].dt.tz_convert("America/New_York")
    minute = eastern.dt.hour * 60 + eastern.dt.minute
    eligible = eligible[minute.between(9 * 60 + 30, 16 * 60 - 1)].copy()
    eligible["_month"] = eligible["timestamp"].dt.strftime("%Y-%m")
    month_rows: dict[str, int] = {}
    for month, group in eligible.groupby("_month", sort=True):
        output = group.drop(columns="_month")
        month_directory = technical_directory / f"month_{month}"
        month_directory.mkdir(parents=True, exist_ok=True)
        output.to_parquet(month_directory / f"{ticker}.parquet", index=False)
        month_rows[str(month)] = len(output)
    return {
        "ticker": ticker,
        "raw_rows": len(bars),
        "regular_membership_rows": len(eligible),
        "months": month_rows,
    }


def _read_benchmark_window(directory: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(
        directory,
        filters=[("timestamp", ">=", start.to_pydatetime()), ("timestamp", "<=", end.to_pydatetime())],
    )
    if "ticker" not in frame.columns and "symbol" in frame.columns:
        frame = frame.rename(columns={"symbol": "ticker"})
    if frame.empty:
        raise DataReadinessError(f"No benchmark rows cover {start.isoformat()} through {end.isoformat()}")
    return frame


def _restrict_to_market_session_grid(
    technical: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    interval_minutes: int,
) -> pd.DataFrame:
    benchmark_identity = benchmarks[["ticker", "timestamp"]].copy()
    benchmark_identity["ticker"] = benchmark_identity["ticker"].astype(str).str.upper().str.strip()
    required = {"QQQ", "SPY", *technical["primary_benchmark"].astype(str).str.upper().str.strip().unique()}
    technical_start = pd.Timestamp(technical["timestamp"].min())
    technical_end = pd.Timestamp(technical["timestamp"].max())
    eastern = technical["timestamp"].dt.tz_convert("America/New_York")
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule.loc[str(eastern.dt.date.min()) : str(eastern.dt.date.max())]  # type: ignore[misc]
    expected_parts = [
        pd.date_range(row.open, row.close, freq=f"{interval_minutes}min", inclusive="left")
        for row in schedule.itertuples()
    ]
    if not expected_parts:
        raise DataReadinessError("XNYS calendar has no sessions for the monthly technical window")
    expected_grid = expected_parts[0]
    for part in expected_parts[1:]:
        expected_grid = expected_grid.append(part)
    expected_grid = expected_grid[(expected_grid >= technical_start) & (expected_grid <= technical_end)]
    if len(expected_grid) == 0:
        raise DataReadinessError("XNYS calendar has no bar timestamps inside the monthly technical window")
    market = benchmark_identity[
        benchmark_identity["ticker"].isin(required) & benchmark_identity["timestamp"].isin(expected_grid)
    ]
    market_counts = market.groupby("timestamp", sort=False)["ticker"].nunique().reindex(expected_grid, fill_value=0)
    missing_grid = market_counts[market_counts.ne(len(required))]
    if not missing_grid.empty:
        raise DataReadinessError(
            f"Required benchmark market grid is incomplete at {missing_grid.index[0]}: "
            f"symbols={int(missing_grid.iloc[0])}/{len(required)}"
        )
    market_grid = expected_grid
    filtered = technical[technical["timestamp"].isin(market_grid)].copy()
    if filtered.empty:
        raise DataReadinessError("No ticker bars align with the shared benchmark monthly market session grid")
    return filtered


def _dataset_fingerprint(months: list[dict[str, Any]], config: DevelopmentDatasetConfig) -> str:
    payload = {
        "builder_schema": DEVELOPMENT_BUILDER_SCHEMA,
        "config": config.model_dump(mode="json"),
        "months": [{"month": item["month"], "sha256": item["sha256"]} for item in months],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _write_report(directory: Path, report: dict[str, Any]) -> None:
    (directory / "_build_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def _require_new_directory(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite development artifact directory: {path}")


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
