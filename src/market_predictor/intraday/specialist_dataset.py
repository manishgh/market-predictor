"""Causal setup extraction and selective one-minute requirements for KS4."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday.specialist_contracts import (
    INTRADAY_SPECIALIST_IDS,
    IntradaySpecialistResearchConfig,
    IntradaySpecialistStrategyConfig,
    intraday_specialist_policy_identity,
    load_intraday_specialist_research_config,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError
from market_predictor.v3.features import (
    finalize_v3_cross_sectional_features,
)

SPECIALIST_SETUP_BUNDLE_SCHEMA = "intraday.specialist_setup_bundle.v1"
SPECIALIST_SETUP_SCHEMA = "intraday.specialist_setup.v1"
SPECIALIST_REQUIREMENT_SCHEMA = "intraday.specialist_one_minute_requirement.v1"
SPECIALIST_COLLECTION_PLAN_SCHEMA = "intraday.specialist_collection_plan.v1"

_BASE_SOURCE_COLUMNS = (
    "ticker",
    "timestamp",
    "decision_time_utc",
    "feature_available_at_utc",
    "session_date_et",
    "primary_benchmark",
    "universe_snapshot_id",
    "sector",
    "industry",
    "market_cap_bucket",
    "liquidity_bucket",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "atr_14",
    "session_vwap",
    "cross_section_eligible",
    "price_feed",
    "adjustment",
)
_DERIVED_SOURCE_FEATURES = frozenset(
    {
        "close_location_5m",
        "return_3bar_atr_units",
        "dist_session_vwap_atr_units",
    }
)
_FUTURE_COLUMN_MARKERS = (
    "entry_",
    "exit_",
    "future",
    "label",
    "target",
    "stop",
    "outcome",
    "mfe",
    "mae",
    "net_return",
    "path_",
    "bars_to_",
    "ranking_",
)
_CROSS_SECTIONAL_FEATURE_PREFIXES = (
    "qqq_",
    "spy_",
    "sector_return_",
    "rel_return_",
    "eligible_breadth_",
    "regime_",
    "xs_",
)
_CROSS_SECTIONAL_FEATURES = frozenset({"cross_section_eligible"})


def specialist_source_projection(
    config: IntradaySpecialistResearchConfig,
) -> tuple[str, ...]:
    """Return the only V3 columns KS4 may read before exact 1m enrichment."""

    setup_features = {
        rule.feature
        for strategy in config.strategies.values()
        for rule in strategy.setup_rules
    }
    five_minute_features = {
        feature
        for feature in config.technical_features
        if not feature.endswith("_1m")
        and feature not in _DERIVED_SOURCE_FEATURES
    }
    projected = list(_BASE_SOURCE_COLUMNS)
    for column in sorted(setup_features | five_minute_features):
        if column not in projected and column not in _DERIVED_SOURCE_FEATURES:
            projected.append(column)
    prohibited = [
        column
        for column in projected
        if any(marker in column.lower() for marker in _FUTURE_COLUMN_MARKERS)
    ]
    if prohibited:
        raise DataReadinessError(
            "KS4 source projection includes future-defined columns: "
            + ", ".join(sorted(prohibited))
        )
    return tuple(projected)


def specialist_technical_projection(
    config: IntradaySpecialistResearchConfig,
) -> tuple[str, ...]:
    """Return strict pre-cross-section columns read from full 5m shards."""

    projected = [
        column
        for column in specialist_source_projection(config)
        if column not in _CROSS_SECTIONAL_FEATURES
        and not column.startswith(_CROSS_SECTIONAL_FEATURE_PREFIXES)
    ]
    for column in ("_session_date_et", "decision_group_id"):
        if column not in projected:
            projected.append(column)
    return tuple(projected)


def extract_specialist_setups(
    source: pd.DataFrame,
    *,
    config: IntradaySpecialistResearchConfig,
    source_dataset_fingerprint: str,
) -> dict[str, pd.DataFrame]:
    """Extract the seven frozen setup populations from one causal source shard."""

    projection = specialist_source_projection(config)
    missing = sorted(set(projection).difference(source.columns))
    if missing:
        raise SchemaMismatchError(
            "KS4 source shard is missing projected columns: "
            + ", ".join(missing)
        )
    data = source.loc[:, list(projection)].copy()
    _validate_source_rows(data, config)
    data = _normalize_completed_five_minute_rows(data, config)
    data = _add_derived_five_minute_features(data)
    data = _add_actual_session_close(data)
    results: dict[str, pd.DataFrame] = {}
    for strategy_id in INTRADAY_SPECIALIST_IDS:
        strategy = config.strategies[strategy_id]
        selected = _apply_setup_rules(data, strategy)
        selected = _apply_setup_spacing(selected, strategy)
        results[strategy_id] = _stamp_setup_identity(
            selected,
            strategy_id=strategy_id,
            strategy=strategy,
            source_dataset_fingerprint=source_dataset_fingerprint,
        )
    return results


def build_intraday_specialist_setup_bundle(
    *,
    technical_directory: Path,
    benchmark_directory: Path,
    policy_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Build an immutable setup bundle and exact selective 1m request plan."""

    if output_directory.exists():
        raise DataReadinessError(
            f"KS4 setup output must be a new path: {output_directory}"
        )
    config = load_intraday_specialist_research_config(policy_path)
    manifest = _load_verified_technical_manifest(
        technical_directory,
        benchmark_directory=benchmark_directory,
        expected_schema=config.required_technical_manifest_schema,
    )
    projection = specialist_source_projection(config)
    technical_projection = specialist_technical_projection(config)
    source_months = sorted(
        path.name.removeprefix("month_")
        for path in technical_directory.glob("month_*")
        if path.is_dir()
    )
    if not source_months:
        raise DataReadinessError("KS4 technical source has no monthly shards")
    month_input_hashes = {
        month: _directory_content_sha256(
            technical_directory / f"month_{month}"
        )
        for month in source_months
    }
    dataset_fingerprint = _technical_source_fingerprint(
        technical_directory,
        benchmark_directory=benchmark_directory,
        manifest=manifest,
        month_input_hashes=month_input_hashes,
    )
    decision_start = pd.Timestamp(
        str(manifest["config"]["decision_start_date"])
    ).date()
    source_grid_rows_removed: dict[str, int] = {}
    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        strategy_rows = {
            strategy_id: 0 for strategy_id in INTRADAY_SPECIALIST_IDS
        }
        setup_count = 0
        setup_tickers: set[str] = set()
        setup_sessions: set[object] = set()
        for month in source_months:
            month_directory = technical_directory / f"month_{month}"
            first_file = next(
                iter(sorted(month_directory.glob("*.parquet"))),
                None,
            )
            if first_file is None:
                raise DataReadinessError(
                    "KS4 technical month has no ticker shards: "
                    f"{month_directory}"
                )
            available = set(cast(Any, pq.read_schema)(first_file).names)
            missing = sorted(
                set(technical_projection).difference(available)
            )
            if missing:
                raise SchemaMismatchError(
                    f"KS4 technical month {month} is missing columns: "
                    + ", ".join(missing)
                )
            technical = pd.read_parquet(
                month_directory,
                columns=list(technical_projection),
            )
            technical = technical[
                pd.to_datetime(technical["timestamp"], utc=True)
                .dt.tz_convert("America/New_York")
                .dt.date
                >= decision_start
            ].copy()
            if technical.empty:
                continue
            technical = _repair_fixed_opening_range(technical)
            benchmarks = _read_benchmark_window(
                benchmark_directory,
                start=pd.Timestamp(technical["timestamp"].min()),
                end=pd.Timestamp(technical["timestamp"].max()),
            )
            _guard_memory(config, f"KS4 loaded technical month {month}")
            session_dates = sorted(
                set(
                    pd.to_datetime(technical["timestamp"], utc=True)
                    .dt.tz_convert("America/New_York")
                    .dt.date
                )
            )
            technical_session = (
                pd.to_datetime(technical["timestamp"], utc=True)
                .dt.tz_convert("America/New_York")
                .dt.date
            )
            benchmark_session = (
                pd.to_datetime(benchmarks["timestamp"], utc=True)
                .dt.tz_convert("America/New_York")
                .dt.date
            )
            month_parts: dict[str, list[pd.DataFrame]] = {
                strategy_id: [] for strategy_id in INTRADAY_SPECIALIST_IDS
            }
            month_rows_removed = 0
            for batch_start in range(
                0,
                len(session_dates),
                config.cross_section_batch_sessions,
            ):
                batch_dates = session_dates[
                    batch_start : batch_start
                    + config.cross_section_batch_sessions
                ]
                batch_label = (
                    f"{batch_dates[0]}..{batch_dates[-1]}"
                )
                _guard_memory(
                    config,
                    f"KS4 pre-finalization {batch_label}",
                )
                daily_technical = technical[
                    technical_session.isin(batch_dates)
                ].copy()
                daily_benchmarks = benchmarks[
                    benchmark_session.isin(batch_dates)
                ].copy()
                daily_technical, removed = (
                    restrict_to_complete_benchmark_grid(
                        daily_technical,
                        daily_benchmarks,
                    )
                )
                month_rows_removed += removed
                if daily_technical.empty:
                    continue
                frame = finalize_v3_cross_sectional_features(
                    daily_technical,
                    daily_benchmarks,
                    minimum_cross_section=int(
                        manifest["config"]["minimum_cross_section"]
                    ),
                )
                missing_final = sorted(
                    set(projection).difference(frame.columns)
                )
                if missing_final:
                    raise SchemaMismatchError(
                        "KS4 finalized session "
                        f"{batch_label} is missing columns: "
                        + ", ".join(missing_final)
                    )
                extracted = extract_specialist_setups(
                    frame.loc[:, list(projection)],
                    config=config,
                    source_dataset_fingerprint=dataset_fingerprint,
                )
                for strategy_id, setups in extracted.items():
                    if not setups.empty:
                        month_parts[strategy_id].append(setups)
                del daily_technical, daily_benchmarks, frame, extracted
                _guard_memory(
                    config,
                    f"KS4 post-finalization {batch_label}",
                )
            source_grid_rows_removed[month] = month_rows_removed
            monthly_setups = {
                strategy_id: (
                    pd.concat(parts, ignore_index=True)
                    .sort_values(
                        ["decision_time_utc", "ticker"],
                        kind="stable",
                    )
                    .reset_index(drop=True)
                    if parts
                    else _empty_setup_frame(projection)
                )
                for strategy_id, parts in month_parts.items()
            }
            month_nonempty = [
                frame
                for frame in monthly_setups.values()
                if not frame.empty
            ]
            if month_nonempty:
                all_month_setups = pd.concat(
                    month_nonempty,
                    ignore_index=True,
                )
                if bool(all_month_setups["setup_id"].duplicated().any()):
                    raise DataReadinessError(
                        f"KS4 setup identities collide in month {month}"
                    )
                setup_count += len(all_month_setups)
                setup_tickers.update(
                    all_month_setups["ticker"].astype(str)
                )
                setup_sessions.update(all_month_setups["session_date_et"])
            else:
                all_month_setups = _empty_setup_frame(projection)
            for strategy_id, frame in monthly_setups.items():
                strategy_rows[strategy_id] += len(frame)
                path = (
                    temporary
                    / "setups"
                    / _strategy_slug(strategy_id)
                    / f"{month}.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                files.append(
                    _file_record(path, temporary, rows=len(frame))
                )
            del (
                technical,
                benchmarks,
                technical_session,
                benchmark_session,
                month_parts,
                monthly_setups,
                month_nonempty,
                all_month_setups,
            )
            _guard_memory(config, f"KS4 setup extraction for {month}")
        if setup_count == 0:
            raise DataReadinessError(
                "KS4 setup extraction produced no eligible rows"
            )
        bundle_fingerprint = _bundle_fingerprint(
            files=files,
            dataset_fingerprint=dataset_fingerprint,
            policy_sha256=config.policy_sha256(),
        )
        report: dict[str, Any] = {
            "schema": SPECIALIST_SETUP_BUNDLE_SCHEMA,
            "setup_schema": SPECIALIST_SETUP_SCHEMA,
            "requirement_schema": SPECIALIST_REQUIREMENT_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "bundle_fingerprint": bundle_fingerprint,
            "source": {
                "technical_directory": str(technical_directory),
                "benchmark_directory": str(benchmark_directory),
                "manifest_sha256": file_sha256(
                    technical_directory / "_technical_manifest.json"
                ),
                "dataset_fingerprint": dataset_fingerprint,
                "builder_schema": manifest["schema"],
                "first_month": source_months[0],
                "last_month": source_months[-1],
                "month_input_sha256": month_input_hashes,
                "rows_removed_for_incomplete_benchmark_grid": (
                    source_grid_rows_removed
                ),
            },
            "policy": intraday_specialist_policy_identity(policy_path),
            "source_projection": list(projection),
            "timing_correction": {
                "source_timestamp_semantics": "five_minute_bar_start",
                "signal_time_semantics": "completed_bar_end",
                "feature_finalization_delay_seconds": (
                    config.intraday_finalization_delay_seconds
                ),
                "decision_time_semantics": "first_whole_minute_after_availability",
                "entry_latency_minutes": config.entry_latency_minutes,
            },
            "source_sampling": {
                "decision_stride_bars": 1,
                "rotate_decision_offset_by_session": False,
                "exhaustive_five_minute_population": True,
                "legacy_labeled_bundle_used": False,
            },
            "strategy_rows": strategy_rows,
            "summary": {
                "setups": setup_count,
                "tickers": len(setup_tickers),
                "sessions": len(setup_sessions),
            },
            "memory": memory_audit(
                hard_budget_gib=config.maximum_process_memory_gib,
                headroom_gib=config.memory_guard_headroom_gib,
            ).to_record(),
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        manifest_path = temporary / "_manifest.json"
        manifest_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_directory)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_specialist_setup_bundle(
    directory: Path,
) -> dict[str, Any]:
    """Verify a complete KS4 setup bundle without loading data shards."""

    manifest_path = directory / "_manifest.json"
    if not manifest_path.is_file():
        raise DataReadinessError(
            f"missing KS4 setup bundle manifest: {manifest_path}"
        )
    try:
        manifest = cast(
            dict[str, Any],
            json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"unreadable KS4 setup bundle manifest: {manifest_path}"
        ) from exc
    if manifest.get("schema") != SPECIALIST_SETUP_BUNDLE_SCHEMA:
        raise DataReadinessError("unsupported KS4 setup bundle schema")
    files = cast(list[dict[str, Any]], manifest.get("files"))
    if not files:
        raise DataReadinessError("KS4 setup bundle has no registered files")
    expected = {directory / str(record["path"]) for record in files}
    actual = {
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "_manifest.json"
    }
    if expected != actual:
        raise DataReadinessError("KS4 setup bundle contains an unexpected file set")
    for record in files:
        path = directory / str(record["path"])
        if file_sha256(path) != str(record["sha256"]):
            raise DataReadinessError(
                f"KS4 setup bundle file is hash-invalid: {path}"
            )
    expected_fingerprint = _bundle_fingerprint(
        files=files,
        dataset_fingerprint=str(manifest["source"]["dataset_fingerprint"]),
        policy_sha256=str(manifest["policy"]["policy_sha256"]),
    )
    if expected_fingerprint != manifest.get("bundle_fingerprint"):
        raise DataReadinessError("KS4 setup bundle fingerprint is invalid")
    return manifest


def build_intraday_specialist_collection_plan(
    *,
    setup_directory: Path,
    policy_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Expand a verified setup bundle into monthly selective 1m windows."""

    if output_directory.exists():
        raise DataReadinessError(
            f"KS4 collection plan output must be new: {output_directory}"
        )
    setup_manifest = verify_specialist_setup_bundle(setup_directory)
    config = load_intraday_specialist_research_config(policy_path)
    if (
        setup_manifest.get("policy", {}).get("policy_sha256")
        != config.policy_sha256()
    ):
        raise DataReadinessError(
            "KS4 setup bundle policy does not match collection policy"
        )
    months = sorted(
        {
            path.stem
            for path in (setup_directory / "setups").glob("*/*.parquet")
        }
    )
    if not months:
        raise DataReadinessError("KS4 setup bundle has no monthly setup shards")
    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        requirement_count = 0
        window_count = 0
        bridge_count = 0
        required_tickers: set[str] = set()
        for month in months:
            parts = [
                pd.read_parquet(
                    setup_directory
                    / "setups"
                    / _strategy_slug(strategy_id)
                    / f"{month}.parquet"
                )
                for strategy_id in INTRADAY_SPECIALIST_IDS
            ]
            nonempty = [part for part in parts if not part.empty]
            if not nonempty:
                continue
            setups = pd.concat(nonempty, ignore_index=True)
            minute_grid = _regular_minute_grid(setups)
            requirements = build_one_minute_requirements(
                setups,
                minimum_warmup_bars=(
                    config.minimum_one_minute_warmup_bars
                ),
                regular_minute_grid=minute_grid,
                intraday_finalization_delay_seconds=(
                    config.intraday_finalization_delay_seconds
                ),
            )
            windows = merge_one_minute_requirements(requirements)
            bridge = build_requirement_window_bridge(windows)
            requirement_count += len(requirements)
            window_count += len(windows)
            bridge_count += len(bridge)
            required_tickers.update(windows["ticker"].astype(str))
            for name, frame in (
                ("requirements", requirements),
                ("collection_windows", windows),
                ("requirement_window_bridge", bridge),
            ):
                path = temporary / name / f"{month}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_parquet(path, index=False)
                files.append(
                    _file_record(path, temporary, rows=len(frame))
                )
            del parts, nonempty, setups, requirements, windows, bridge
            _guard_memory(config, f"KS4 collection planning for {month}")
        if window_count == 0:
            raise DataReadinessError(
                "KS4 collection planning produced no windows"
            )
        plan_fingerprint = _collection_plan_fingerprint(
            files=files,
            setup_bundle_fingerprint=str(
                setup_manifest["bundle_fingerprint"]
            ),
            policy_sha256=config.policy_sha256(),
        )
        report: dict[str, Any] = {
            "schema": SPECIALIST_COLLECTION_PLAN_SCHEMA,
            "requirement_schema": SPECIALIST_REQUIREMENT_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan_fingerprint": plan_fingerprint,
            "setup_bundle": {
                "path": str(setup_directory),
                "bundle_fingerprint": setup_manifest[
                    "bundle_fingerprint"
                ],
                "manifest_sha256": file_sha256(
                    setup_directory / "_manifest.json"
                ),
            },
            "policy": intraday_specialist_policy_identity(policy_path),
            "summary": {
                "one_minute_requirements": requirement_count,
                "merged_collection_windows": window_count,
                "requirement_window_bridge_rows": bridge_count,
                "required_tickers": len(required_tickers),
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


def verify_specialist_collection_plan(directory: Path) -> dict[str, Any]:
    """Verify a complete KS4 collection plan without loading its shards."""

    path = directory / "_manifest.json"
    if not path.is_file():
        raise DataReadinessError(
            f"missing KS4 collection plan manifest: {path}"
        )
    try:
        manifest = cast(
            dict[str, Any],
            json.loads(path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"unreadable KS4 collection plan manifest: {path}"
        ) from exc
    if manifest.get("schema") != SPECIALIST_COLLECTION_PLAN_SCHEMA:
        raise DataReadinessError("unsupported KS4 collection plan schema")
    files = cast(list[dict[str, Any]], manifest.get("files"))
    if not files:
        raise DataReadinessError("KS4 collection plan has no files")
    expected = {directory / str(record["path"]) for record in files}
    actual = {
        artifact
        for artifact in directory.rglob("*")
        if artifact.is_file() and artifact.name != "_manifest.json"
    }
    if expected != actual:
        raise DataReadinessError(
            "KS4 collection plan contains an unexpected file set"
        )
    for record in files:
        artifact = directory / str(record["path"])
        if file_sha256(artifact) != str(record["sha256"]):
            raise DataReadinessError(
                f"KS4 collection plan file is hash-invalid: {artifact}"
            )
    expected_fingerprint = _collection_plan_fingerprint(
        files=files,
        setup_bundle_fingerprint=str(
            manifest["setup_bundle"]["bundle_fingerprint"]
        ),
        policy_sha256=str(manifest["policy"]["policy_sha256"]),
    )
    if expected_fingerprint != manifest.get("plan_fingerprint"):
        raise DataReadinessError(
            "KS4 collection plan fingerprint is invalid"
        )
    return manifest


def iter_specialist_setup_shards(
    directory: Path,
    *,
    strategy_id: str,
) -> Iterator[pd.DataFrame]:
    """Yield one verified monthly setup shard at a time."""

    if strategy_id not in INTRADAY_SPECIALIST_IDS:
        raise ValueError(f"unsupported KS4 strategy: {strategy_id}")
    verify_specialist_setup_bundle(directory)
    shard_directory = directory / "setups" / _strategy_slug(strategy_id)
    for path in sorted(shard_directory.glob("*.parquet")):
        yield pd.read_parquet(path)


def iter_one_minute_window_shards(
    directory: Path,
) -> Iterator[pd.DataFrame]:
    """Yield one verified monthly selective collection plan at a time."""

    verify_specialist_collection_plan(directory)
    for path in sorted((directory / "collection_windows").glob("*.parquet")):
        yield pd.read_parquet(path)


def build_one_minute_requirements(
    setups: pd.DataFrame,
    *,
    minimum_warmup_bars: int,
    regular_minute_grid: pd.DatetimeIndex,
    intraday_finalization_delay_seconds: int = 30,
) -> pd.DataFrame:
    """Expand setups to stock and benchmark exact-path requirements."""

    required = {
        "setup_id",
        "strategy_id",
        "ticker",
        "primary_benchmark",
        "decision_time_utc",
        "feature_available_at_utc",
        "horizon_minutes",
    }
    missing = sorted(required.difference(setups.columns))
    if missing:
        raise SchemaMismatchError(
            "KS4 setups are missing request fields: " + ", ".join(missing)
        )
    grid = regular_minute_grid.sort_values().unique()
    grid_ns = grid.as_unit("ns").asi8
    availability_ns = (
        grid_ns
        + 60_000_000_000
        + intraday_finalization_delay_seconds * 1_000_000_000
    )
    grid_sessions = grid.tz_convert("America/New_York").date
    session_start_index: dict[object, int] = {}
    session_end_index: dict[object, int] = {}
    for index, session_date in enumerate(grid_sessions):
        session_start_index.setdefault(session_date, index)
        session_end_index[session_date] = index + 1
    rows: list[dict[str, Any]] = []
    for setup in setups.loc[:, sorted(required)].itertuples(index=False):
        decision = pd.Timestamp(setup.decision_time_utc)
        feature_available = pd.Timestamp(setup.feature_available_at_utc)
        if decision.tzinfo is None:
            raise DataReadinessError("KS4 setup decision time is timezone-naive")
        if feature_available.tzinfo is None:
            raise DataReadinessError(
                "KS4 setup feature availability is timezone-naive"
            )
        decision = decision.tz_convert("UTC")
        feature_available = feature_available.tz_convert("UTC")
        if feature_available >= decision:
            raise DataReadinessError(
                "KS4 setup features must be available before entry"
            )
        decision_ns = decision.as_unit("ns").value
        entry_index = int(np.searchsorted(grid_ns, decision_ns, side="left"))
        if entry_index >= len(grid_ns) or grid_ns[entry_index] != decision_ns:
            raise DataReadinessError(
                f"KS4 decision is not an exact regular-session minute: {decision}"
            )
        eligible_end_index = int(
            np.searchsorted(
                availability_ns,
                feature_available.as_unit("ns").value,
                side="right",
            )
        )
        eligible_end_index = min(eligible_end_index, entry_index)
        if eligible_end_index < minimum_warmup_bars:
            raise DataReadinessError(
                f"KS4 decision lacks {minimum_warmup_bars} prior market minutes"
            )
        horizon = int(setup.horizon_minutes)
        end_index = entry_index + horizon
        if end_index > len(grid_ns):
            raise DataReadinessError("KS4 label window exceeds the minute grid")
        expected_end = decision + pd.Timedelta(minutes=horizon)
        if end_index < len(grid_ns) and pd.Timestamp(grid[end_index]) != expected_end:
            raise DataReadinessError(
                f"KS4 label window crosses a session boundary: {setup.setup_id}"
            )
        decision_session = decision.tz_convert("America/New_York").date()
        current_session_start = session_start_index.get(decision_session)
        if current_session_start is None:
            raise DataReadinessError(
                "KS4 decision session is absent from the minute grid"
            )
        warmup_start_index = min(
            eligible_end_index - minimum_warmup_bars,
            current_session_start,
        )
        planned_warmup_bars = eligible_end_index - warmup_start_index
        if planned_warmup_bars < minimum_warmup_bars:
            raise DataReadinessError("KS4 causal warm-up grid is incomplete")
        warmup_segments = _grid_index_segments(
            grid,
            grid_sessions=grid_sessions,
            session_end_index=session_end_index,
            start_index=warmup_start_index,
            end_index=eligible_end_index,
        )
        label_segments = _grid_index_segments(
            grid,
            grid_sessions=grid_sessions,
            session_end_index=session_end_index,
            start_index=entry_index,
            end_index=end_index,
        )
        if len(label_segments) != 1:
            raise DataReadinessError(
                f"KS4 label window is not one exact session: {setup.setup_id}"
            )
        roles: dict[str, set[str]] = {}
        for ticker, role in (
            (str(setup.ticker), "stock"),
            ("SPY", "broad_benchmark"),
            ("QQQ", "growth_benchmark"),
            (str(setup.primary_benchmark), "sector_benchmark"),
        ):
            roles.setdefault(ticker.upper().strip(), set()).add(role)
        for ticker, ticker_roles in sorted(roles.items()):
            for segment_kind, segments in (
                ("warmup", warmup_segments),
                ("label", label_segments),
            ):
                for segment_start, segment_end, session_date in segments:
                    rows.append(
                        {
                            "requirement_id": _stable_hash(
                                str(setup.setup_id),
                                ticker,
                                segment_kind,
                                segment_start.isoformat(),
                                segment_end.isoformat(),
                            ),
                            "setup_id": str(setup.setup_id),
                            "strategy_id": str(setup.strategy_id),
                            "ticker": ticker,
                            "roles_json": json.dumps(sorted(ticker_roles)),
                            "segment_kind": segment_kind,
                            "session_date_et": session_date,
                            "requested_start_utc": segment_start,
                            "requested_end_utc": segment_end,
                            "decision_time_utc": decision,
                            "feature_available_at_utc": feature_available,
                            "horizon_minutes": horizon,
                            "minimum_warmup_bars": minimum_warmup_bars,
                            "planned_warmup_bars": planned_warmup_bars,
                            "intraday_finalization_delay_seconds": (
                                intraday_finalization_delay_seconds
                            ),
                            "price_feed": "sip",
                            "adjustment": "all",
                            "timeframe": "1m",
                        }
                    )
    output = pd.DataFrame(rows)
    if output.empty:
        raise DataReadinessError("KS4 one-minute requirements are empty")
    if bool(output["requirement_id"].duplicated().any()):
        raise DataReadinessError("KS4 one-minute requirement identities collide")
    return output.sort_values(
        ["ticker", "requested_start_utc", "setup_id"],
        kind="stable",
    ).reset_index(drop=True)


def merge_one_minute_requirements(requirements: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping ticker windows without weakening setup traceability."""

    required = {
        "requirement_id",
        "ticker",
        "roles_json",
        "session_date_et",
        "requested_start_utc",
        "requested_end_utc",
    }
    missing = sorted(required.difference(requirements.columns))
    if missing:
        raise SchemaMismatchError(
            "KS4 one-minute requirements cannot be merged: "
            + ", ".join(missing)
        )
    rows: list[dict[str, Any]] = []
    for (ticker, session_date), part in requirements.groupby(
        ["ticker", "session_date_et"],
        sort=True,
    ):
        ordered = part.sort_values(
            ["requested_start_utc", "requested_end_utc"],
            kind="stable",
        )
        current: dict[str, Any] | None = None
        for requirement in ordered.itertuples(index=False):
            start = pd.Timestamp(requirement.requested_start_utc)
            end = pd.Timestamp(requirement.requested_end_utc)
            roles = set(json.loads(str(requirement.roles_json)))
            if current is None or start > cast(pd.Timestamp, current["end"]):
                if current is not None:
                    rows.append(
                        _finalize_window(
                            str(ticker),
                            session_date,
                            current,
                        )
                    )
                current = {
                    "start": start,
                    "end": end,
                    "requirement_ids": [str(requirement.requirement_id)],
                    "roles": roles,
                }
                continue
            current["end"] = max(cast(pd.Timestamp, current["end"]), end)
            cast(list[str], current["requirement_ids"]).append(
                str(requirement.requirement_id)
            )
            cast(set[str], current["roles"]).update(roles)
        if current is not None:
            rows.append(
                _finalize_window(str(ticker), session_date, current)
            )
    output = pd.DataFrame(rows)
    if output.empty:
        raise DataReadinessError("KS4 merged one-minute collection plan is empty")
    return output.sort_values(
        ["ticker", "requested_start_utc"],
        kind="stable",
    ).reset_index(drop=True)


def build_requirement_window_bridge(
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Create the explicit requirement-to-merged-window lineage bridge."""

    required = {"request_id", "requirement_ids_json"}
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise SchemaMismatchError(
            "KS4 collection windows lack bridge fields: "
            + ", ".join(missing)
        )
    rows = [
        {
            "requirement_id": str(requirement_id),
            "request_id": str(window.request_id),
        }
        for window in windows.itertuples(index=False)
        for requirement_id in json.loads(
            str(window.requirement_ids_json)
        )
    ]
    bridge = pd.DataFrame(
        rows,
        columns=["requirement_id", "request_id"],
    )
    if bridge.empty or bool(bridge["requirement_id"].duplicated().any()):
        raise DataReadinessError(
            "KS4 requirement-to-window bridge is empty or ambiguous"
        )
    return bridge.sort_values(
        ["request_id", "requirement_id"],
        kind="stable",
    ).reset_index(drop=True)


def _load_verified_technical_manifest(
    directory: Path,
    *,
    benchmark_directory: Path,
    expected_schema: str,
) -> dict[str, Any]:
    path = directory / "_technical_manifest.json"
    if not path.is_file():
        raise DataReadinessError(
            f"missing V3 technical manifest for KS4: {path}"
        )
    try:
        payload = cast(
            dict[str, Any],
            json.loads(path.read_text(encoding="utf-8")),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"unreadable V3 technical manifest for KS4: {path}"
        ) from exc
    if payload.get("schema") != expected_schema:
        raise DataReadinessError(
            f"KS4 source builder schema mismatch: {payload.get('schema')}"
        )
    ticker_shards = payload.get("ticker_shards")
    inputs = payload.get("inputs")
    if (
        not isinstance(ticker_shards, list)
        or not ticker_shards
        or not isinstance(inputs, dict)
        or payload.get("failures")
    ):
        raise DataReadinessError("KS4 source technical manifest is incomplete")
    declared_benchmarks = Path(str(inputs.get("benchmark_directory", "")))
    if declared_benchmarks != benchmark_directory:
        raise DataReadinessError(
            "KS4 benchmark directory does not match technical lineage"
        )
    expected_tickers = {
        str(record.get("ticker", "")).upper()
        for record in ticker_shards
        if isinstance(record, dict)
    }
    if "" in expected_tickers:
        raise DataReadinessError("KS4 technical manifest has an empty ticker")
    for month_directory in sorted(directory.glob("month_*")):
        actual_tickers = {
            path.stem.upper()
            for path in month_directory.glob("*.parquet")
        }
        if not actual_tickers.issubset(expected_tickers):
            raise DataReadinessError(
                f"KS4 technical month contains unknown tickers: {month_directory}"
            )
    benchmark_files = sorted(benchmark_directory.glob("*.parquet"))
    benchmark_tickers = {path.stem.upper() for path in benchmark_files}
    required_benchmarks = {
        "SPY",
        "QQQ",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    }
    if benchmark_tickers != required_benchmarks:
        raise DataReadinessError(
            "KS4 benchmark directory does not contain the exact frozen ETF set"
        )
    return payload


def _technical_source_fingerprint(
    technical_directory: Path,
    *,
    benchmark_directory: Path,
    manifest: Mapping[str, Any],
    month_input_hashes: Mapping[str, str],
) -> str:
    payload = {
        "technical_manifest_sha256": file_sha256(
            technical_directory / "_technical_manifest.json"
        ),
        "technical_months": dict(sorted(month_input_hashes.items())),
        "benchmarks": {
            path.name: file_sha256(path)
            for path in sorted(benchmark_directory.glob("*.parquet"))
        },
        "memberships_sha256": str(
            cast(Mapping[str, Any], manifest["inputs"])[
                "memberships_sha256"
            ]
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _directory_content_sha256(directory: Path) -> str:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        raise DataReadinessError(
            f"KS4 source directory contains no parquet files: {directory}"
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(file_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _read_benchmark_window(
    directory: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    columns = [
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "price_feed",
        "adjustment",
    ]
    parts: list[pd.DataFrame] = []
    for path in sorted(directory.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        timestamps = _strict_utc(
            frame["timestamp"],
            f"KS4 benchmark timestamps {path.name}",
        )
        selected = frame[
            timestamps.between(start, end, inclusive="both")
        ].copy()
        if selected.empty:
            raise DataReadinessError(
                f"KS4 benchmark has no rows in month window: {path}"
            )
        selected = selected.rename(columns={"symbol": "ticker"})
        selected["timestamp"] = timestamps.loc[selected.index]
        if bool(
            selected["price_feed"]
            .astype(str)
            .str.lower()
            .ne("sip")
            .any()
        ):
            raise DataReadinessError(
                f"KS4 benchmark is not SIP: {path}"
            )
        if bool(
            selected["adjustment"]
            .astype(str)
            .str.lower()
            .ne("all")
            .any()
        ):
            raise DataReadinessError(
                f"KS4 benchmark adjustment is not all: {path}"
            )
        parts.append(selected)
    output = pd.concat(parts, ignore_index=True)
    if bool(output.duplicated(["ticker", "timestamp"]).any()):
        raise DataReadinessError("KS4 benchmark window contains duplicates")
    return output


def _repair_fixed_opening_range(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    timestamps = _strict_utc(
        output["timestamp"],
        "KS4 technical opening-range timestamps",
    )
    eastern = timestamps.dt.tz_convert("America/New_York")
    minute = eastern.dt.hour * 60 + eastern.dt.minute
    session_date = eastern.dt.date
    keys = [output["ticker"], session_date]
    opening = minute.between(9 * 60 + 30, 9 * 60 + 59)
    opening_count = (
        opening.astype("int8").groupby(keys, sort=False).transform("sum")
    )
    fixed_high = (
        pd.to_numeric(output["high"], errors="coerce")
        .where(opening)
        .groupby(keys, sort=False)
        .transform("max")
    )
    fixed_low = (
        pd.to_numeric(output["low"], errors="coerce")
        .where(opening)
        .groupby(keys, sort=False)
        .transform("min")
    )
    exact = opening_count.eq(6)
    output["opening_range_high"] = fixed_high.where(exact)
    output["opening_range_low"] = fixed_low.where(exact)
    close = pd.to_numeric(output["close"], errors="coerce")
    output["opening_range_width_pct"] = (
        output["opening_range_high"] - output["opening_range_low"]
    ) / close
    output["dist_opening_range_high"] = (
        close / output["opening_range_high"] - 1.0
    )
    output["dist_opening_range_low"] = (
        close / output["opening_range_low"] - 1.0
    )
    return output


def restrict_to_complete_benchmark_grid(
    technical: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    broad = benchmarks[
        benchmarks["ticker"].astype(str).str.upper().isin({"SPY", "QQQ"})
    ]
    broad_counts = broad.groupby("timestamp", sort=False)["ticker"].nunique()
    complete_broad = set(broad_counts[broad_counts.eq(2)].index)
    sector_pairs = pd.MultiIndex.from_frame(
        benchmarks[["ticker", "timestamp"]].assign(
            ticker=lambda frame: frame["ticker"]
            .astype(str)
            .str.upper()
            .str.strip()
        )
    )
    primary = (
        technical["primary_benchmark"]
        .astype(str)
        .str.upper()
        .str.strip()
    )
    technical_pairs = pd.MultiIndex.from_arrays(
        [primary, technical["timestamp"]]
    )
    keep = (
        technical["timestamp"].isin(complete_broad)
        & technical_pairs.isin(sector_pairs)
    )
    removed = int((~keep).sum())
    return technical.loc[keep].copy(), removed


def _validate_source_rows(
    data: pd.DataFrame,
    config: IntradaySpecialistResearchConfig,
) -> None:
    if data.empty:
        return
    ticker = data["ticker"].astype(str).str.upper().str.strip()
    if bool(ticker.eq("").any()):
        raise DataReadinessError("KS4 source contains an empty ticker")
    if bool(data.assign(ticker=ticker).duplicated(["ticker", "timestamp"]).any()):
        raise DataReadinessError("KS4 source contains duplicate ticker bars")
    timestamps = _strict_utc(data["timestamp"], "KS4 source timestamp")
    source_decisions = _strict_utc(
        data["decision_time_utc"],
        "KS4 source decision_time_utc",
    )
    source_availability = _strict_utc(
        data["feature_available_at_utc"],
        "KS4 source feature_available_at_utc",
    )
    if bool(source_decisions.ne(timestamps).any()) or bool(
        source_availability.ne(timestamps).any()
    ):
        raise DataReadinessError(
            "KS4 source timing does not match the audited V3 bar-start convention"
        )
    feed = data["price_feed"].astype(str).str.lower().str.strip()
    adjustment = data["adjustment"].astype(str).str.lower().str.strip()
    if bool(feed.ne(config.required_price_feed).any()):
        raise DataReadinessError("KS4 setup extraction requires SIP source bars")
    if bool(adjustment.ne(config.required_adjustment).any()):
        raise DataReadinessError(
            "KS4 setup extraction requires adjustment=all"
        )
    numeric = ["open", "high", "low", "close", "volume", "atr_14"]
    converted = data[numeric].apply(pd.to_numeric, errors="coerce")
    invalid = (
        converted.isna().any(axis=1)
        | converted[["open", "high", "low", "close", "atr_14"]]
        .le(0)
        .any(axis=1)
        | converted["volume"].lt(0)
    )
    if bool(invalid.any()):
        raise DataReadinessError("KS4 source contains invalid OHLCV or ATR")


def _normalize_completed_five_minute_rows(
    data: pd.DataFrame,
    config: IntradaySpecialistResearchConfig,
) -> pd.DataFrame:
    output = data.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["primary_benchmark"] = (
        output["primary_benchmark"].astype(str).str.upper().str.strip()
    )
    output["source_bar_start_utc"] = _strict_utc(
        output.pop("timestamp"),
        "KS4 source timestamp",
    )
    output["source_decision_time_utc"] = _strict_utc(
        output.pop("decision_time_utc"),
        "KS4 source decision time",
    )
    output["source_feature_available_at_utc"] = _strict_utc(
        output.pop("feature_available_at_utc"),
        "KS4 source feature availability",
    )
    output["signal_time_utc"] = output[
        "source_bar_start_utc"
    ] + pd.Timedelta(minutes=5)
    output["feature_available_at_utc"] = output[
        "signal_time_utc"
    ] + pd.Timedelta(
        seconds=config.intraday_finalization_delay_seconds
    )
    output["decision_time_utc"] = output[
        "signal_time_utc"
    ] + pd.Timedelta(minutes=config.entry_latency_minutes)
    output["bar_end_utc"] = output["signal_time_utc"]
    output["session_date_et"] = (
        output["signal_time_utc"]
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    eastern = output["signal_time_utc"].dt.tz_convert(
        "America/New_York"
    )
    output["session_minute_et"] = (
        eastern.dt.hour * 60 + eastern.dt.minute
    ).astype("int16")
    return output


def _add_derived_five_minute_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    high = pd.to_numeric(output["high"], errors="coerce")
    low = pd.to_numeric(output["low"], errors="coerce")
    close = pd.to_numeric(output["close"], errors="coerce")
    spread = (high - low).replace(0, np.nan)
    output["close_location_5m"] = (close - low) / spread
    return_3bar = pd.to_numeric(output["return_3bar"], errors="coerce")
    prior_close_3bar = close / (1.0 + return_3bar).replace(0, np.nan)
    atr = pd.to_numeric(output["atr_14"], errors="coerce").replace(0, np.nan)
    output["return_3bar_atr_units"] = (
        close - prior_close_3bar
    ) / atr
    session_vwap = pd.to_numeric(
        output["session_vwap"],
        errors="coerce",
    )
    output["dist_session_vwap_atr_units"] = (
        close - session_vwap
    ) / atr
    output["atr_14_price_5m"] = pd.to_numeric(
        output["atr_14"],
        errors="coerce",
    )
    return output


def _add_actual_session_close(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    calendar = xcals.get_calendar("XNYS")
    session_dates = sorted(set(output["session_date_et"]))
    close_by_date = {
        pd.Timestamp(session).date(): pd.Timestamp(
            calendar.session_close(session)
        ).tz_convert("UTC")
        for session in calendar.sessions_in_range(
            session_dates[0],
            session_dates[-1],
        )
    }
    output["actual_session_close_utc"] = pd.to_datetime(
        output["session_date_et"].map(close_by_date),
        utc=True,
    )
    if bool(output["actual_session_close_utc"].isna().any()):
        raise DataReadinessError(
            "KS4 setup rows include a non-XNYS session date"
        )
    return output


def _apply_setup_rules(
    data: pd.DataFrame,
    strategy: IntradaySpecialistStrategyConfig,
) -> pd.DataFrame:
    selected = data["session_minute_et"].between(
        strategy.first_decision_minute_et,
        strategy.last_decision_minute_et_exclusive - 1,
    )
    selected &= data["cross_section_eligible"].eq(1)
    selected &= (
        data["decision_time_utc"]
        + pd.to_timedelta(strategy.horizon_minutes, unit="m")
    ).le(data["actual_session_close_utc"])
    for rule in strategy.setup_rules:
        values = pd.to_numeric(data[rule.feature], errors="coerce")
        selected &= values.notna()
        if rule.minimum is not None:
            selected &= values.ge(rule.minimum)
        if rule.maximum is not None:
            selected &= values.le(rule.maximum)
    return data.loc[selected].copy()


def _apply_setup_spacing(
    data: pd.DataFrame,
    strategy: IntradaySpecialistStrategyConfig,
) -> pd.DataFrame:
    if data.empty:
        return data
    keep: list[int] = []
    spacing = pd.Timedelta(minutes=strategy.minimum_setup_spacing_minutes)
    for (_, _), part in data.groupby(
        ["ticker", "session_date_et"],
        sort=False,
    ):
        last: pd.Timestamp | None = None
        for index, value in part.sort_values(
            "decision_time_utc",
            kind="stable",
        )["decision_time_utc"].items():
            decision = pd.Timestamp(value)
            if last is None or decision - last >= spacing:
                keep.append(int(index))
                last = decision
    return data.loc[keep].copy()


def _stamp_setup_identity(
    data: pd.DataFrame,
    *,
    strategy_id: str,
    strategy: IntradaySpecialistStrategyConfig,
    source_dataset_fingerprint: str,
) -> pd.DataFrame:
    output = data.copy()
    output.insert(0, "setup_schema_version", SPECIALIST_SETUP_SCHEMA)
    output.insert(1, "strategy_id", strategy_id)
    output.insert(2, "horizon_minutes", strategy.horizon_minutes)
    output.insert(3, "direction", strategy.direction)
    output.insert(4, "session_segment", strategy.session_segment)
    output.insert(
        5,
        "setup_id",
        [
            _stable_hash(
                strategy_id,
                str(ticker),
                pd.Timestamp(decision).isoformat(),
                source_dataset_fingerprint,
            )
            for ticker, decision in zip(
                output["ticker"],
                output["decision_time_utc"],
                strict=True,
            )
        ],
    )
    output["source_dataset_fingerprint"] = source_dataset_fingerprint
    return output.sort_values(
        ["decision_time_utc", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def _regular_minute_grid(setups: pd.DataFrame) -> pd.DatetimeIndex:
    decisions = _strict_utc(
        setups["decision_time_utc"],
        "KS4 setup decisions",
    )
    first = decisions.min() - pd.Timedelta(days=10)
    last = decisions.max() + pd.Timedelta(days=2)
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(
        first.date(),
        last.date(),
    )
    minutes: list[pd.DatetimeIndex] = []
    for session in sessions:
        open_at = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
        close_at = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
        minutes.append(
            pd.date_range(
                open_at,
                close_at - pd.Timedelta(minutes=1),
                freq="1min",
            )
        )
    if not minutes:
        raise DataReadinessError("KS4 could not construct the XNYS minute grid")
    return minutes[0].append(minutes[1:])


def _finalize_window(
    ticker: str,
    session_date: object,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    start = cast(pd.Timestamp, current["start"])
    end = cast(pd.Timestamp, current["end"])
    requirement_ids = sorted(cast(Sequence[str], current["requirement_ids"]))
    roles = sorted(cast(Iterable[str], current["roles"]))
    return {
        "request_id": _stable_hash(
            ticker.upper(),
            start.isoformat(),
            end.isoformat(),
            *requirement_ids,
        ),
        "ticker": ticker.upper(),
        "session_date_et": pd.Timestamp(session_date).date(),
        "requested_start_utc": start,
        "requested_end_utc": end,
        "requirement_count": len(requirement_ids),
        "requirement_ids_json": json.dumps(requirement_ids),
        "roles_json": json.dumps(roles),
        "price_feed": "sip",
        "adjustment": "all",
        "timeframe": "1m",
    }


def _file_record(path: Path, root: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _bundle_fingerprint(
    *,
    files: Sequence[Mapping[str, Any]],
    dataset_fingerprint: str,
    policy_sha256: str,
) -> str:
    payload = {
        "schema": SPECIALIST_SETUP_BUNDLE_SCHEMA,
        "dataset_fingerprint": dataset_fingerprint,
        "policy_sha256": policy_sha256,
        "files": [
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "rows": int(item["rows"]),
            }
            for item in sorted(files, key=lambda record: str(record["path"]))
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _collection_plan_fingerprint(
    *,
    files: Sequence[Mapping[str, Any]],
    setup_bundle_fingerprint: str,
    policy_sha256: str,
) -> str:
    payload = {
        "schema": SPECIALIST_COLLECTION_PLAN_SCHEMA,
        "setup_bundle_fingerprint": setup_bundle_fingerprint,
        "policy_sha256": policy_sha256,
        "files": [
            {
                "path": str(item["path"]),
                "sha256": str(item["sha256"]),
                "rows": int(item["rows"]),
            }
            for item in sorted(files, key=lambda record: str(record["path"]))
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _empty_setup_frame(projection: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "setup_schema_version",
            "strategy_id",
            "horizon_minutes",
            "direction",
            "session_segment",
            "setup_id",
            *projection,
            "source_bar_start_utc",
            "source_decision_time_utc",
            "source_feature_available_at_utc",
            "bar_end_utc",
            "signal_time_utc",
            "session_minute_et",
            "close_location_5m",
            "return_3bar_atr_units",
            "dist_session_vwap_atr_units",
            "atr_14_price_5m",
            "source_dataset_fingerprint",
        ]
    )


def _strategy_slug(strategy_id: str) -> str:
    return strategy_id.lower().replace(".", "_")


def _grid_index_segments(
    grid: pd.DatetimeIndex,
    *,
    grid_sessions: np.ndarray,
    session_end_index: Mapping[object, int],
    start_index: int,
    end_index: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, object]]:
    if start_index >= end_index:
        return []
    segments: list[tuple[pd.Timestamp, pd.Timestamp, object]] = []
    segment_start = start_index
    while segment_start < end_index:
        session_date = grid_sessions[segment_start]
        segment_end = min(
            int(session_end_index[session_date]),
            end_index,
        )
        segments.append(
            (
                pd.Timestamp(grid[segment_start]),
                pd.Timestamp(grid[segment_end - 1])
                + pd.Timedelta(minutes=1),
                session_date,
            )
        )
        segment_start = segment_end
    return segments


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    if not isinstance(values.dtype, pd.DatetimeTZDtype):
        raise DataReadinessError(f"{name} must be timezone-aware")
    parsed = values.dt.tz_convert("UTC")
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{name} contains invalid timestamps")
    return parsed


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
