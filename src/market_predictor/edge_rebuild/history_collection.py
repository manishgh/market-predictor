"""Resumable Alpaca transport for ER1A five-minute history units."""
from __future__ import annotations



import hashlib
import json
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

import pandas as pd

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.history_contracts import (
    IntradayTransportConfig,
)
from market_predictor.intraday.datasets.history import (
    load_complete_intraday_history_plan,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.sources.alpaca import AlpacaBarsPage, AlpacaSource
from market_predictor.core.errors import DataReadinessError, SchemaMismatchError

HISTORY_COLLECTION_SCHEMA = "edge_rebuild.intraday_history_collection.v1"
HISTORY_UNIT_SCHEMA = "edge_rebuild.intraday_history_unit.v1"
HISTORY_AUTHORITY_SCHEMA = "edge_rebuild.intraday_history_authority.v1"
EXACT_HISTORY_COLLECTION_SCHEMA = "edge_rebuild.intraday_history_collection.v2"
EXACT_HISTORY_UNIT_SCHEMA = "edge_rebuild.intraday_history_unit.v2"
EXACT_HISTORY_AUTHORITY_SCHEMA = "edge_rebuild.intraday_history_authority.v2"
_SAFE_RATE_HEADERS = {
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
}
SourceFactory = Callable[[], AlpacaSource]


def collect_intraday_history(
    *,
    plan_directory: Path,
    policy_path: Path,
    output_directory: Path,
    config: IntradayTransportConfig,
    source_factory: SourceFactory,
    maximum_units_this_run: int | None = None,
) -> dict[str, Any]:
    """Collect every immutable ER1A unit with hash-verified resume."""

    plan = load_complete_intraday_history_plan(plan_directory)
    timeframe = _transport_timeframe(config)
    normalized_timeframe = _canonical_timeframe(timeframe)
    if plan.get("policy_sha256") != config.sha256():
        raise DataReadinessError("plan and collection policy differ")
    if (output_directory / "_authority.json").exists():
        raise DataReadinessError(
            "completed history collection is immutable"
        )
    if maximum_units_this_run is not None and maximum_units_this_run < 1:
        raise ValueError("maximum_units_this_run must be positive")
    unit_parts = [
        pd.read_parquet(path)
        for path in sorted(
            (plan_directory / "units" / timeframe).glob("*.parquet")
        )
    ]
    if not unit_parts:
        raise DataReadinessError(
            f"intraday plan contains no {timeframe} units"
        )
    units = pd.concat(unit_parts, ignore_index=True)
    if bool(units["unit_id"].duplicated().any()):
        raise DataReadinessError("ER1A plan contains duplicate units")
    plan_fingerprint = str(plan["plan_fingerprint"])
    request_payload: dict[str, Any] = {
        "schema": EXACT_HISTORY_COLLECTION_SCHEMA,
        "plan_schema": str(plan.get("schema", "")),
        "plan_path": str(plan_directory),
        "plan_fingerprint": plan_fingerprint,
        "plan_manifest_sha256": file_sha256(
            plan_directory / "_manifest.json"
        ),
        "policy_path": str(policy_path),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": config.sha256(),
        "provider": "alpaca",
        "timeframe": timeframe,
        "price_feed": "sip",
        "adjustment": "all",
        "workers": config.collection_workers,
        "retries": config.collection_retries,
        "request_timeout_seconds": config.request_timeout_seconds,
        "maximum_pages_per_unit": config.maximum_pages_per_unit,
        "maximum_failures_before_stop": (
            config.maximum_failures_before_stop
        ),
    }
    request_sha256 = _json_sha256(request_payload)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(
        output_directory / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )
    bars_directory = output_directory / "bars"
    attempts_directory = output_directory / "attempts"
    raw_pages_directory = output_directory / "raw_pages"
    bars_directory.mkdir(parents=True, exist_ok=True)
    attempts_directory.mkdir(parents=True, exist_ok=True)
    raw_pages_directory.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for raw in units.to_dict(orient="records"):
        row = {str(key): value for key, value in raw.items()}
        unit_id = str(row["unit_id"])
        month = pd.Timestamp(row["session_date_et"]).strftime("%Y-%m")
        existing = _load_existing_unit(
            bars_directory / month / f"{unit_id}.parquet",
            root=output_directory,
            expected_unit=row,
            plan_fingerprint=plan_fingerprint,
            request_sha256=request_sha256,
            timeframe=normalized_timeframe,
        )
        if existing is None:
            pending.append(row)
        else:
            completed[unit_id] = existing
    _guard_memory(config, "ER1A history collection start")
    local = threading.local()

    def get_source() -> AlpacaSource:
        source = getattr(local, "alpaca_source", None)
        if source is None:
            source = source_factory()
            source.client.timeout = int(config.request_timeout_seconds)
            if source.settings.alpaca_stock_feed.strip().lower() != "sip":
                raise DataReadinessError(
                    "ER1A history collection requires Alpaca SIP"
                )
            local.alpaca_source = source
        return cast(AlpacaSource, source)

    def collect_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return _collect_unit(
            row,
            source=get_source(),
            root=output_directory,
            bars_directory=bars_directory,
            attempts_directory=attempts_directory,
            raw_pages_directory=raw_pages_directory,
            plan_fingerprint=plan_fingerprint,
            request_sha256=request_sha256,
            config=config,
            timeframe=timeframe,
        )

    scheduled = (
        pending
        if maximum_units_this_run is None
        else pending[:maximum_units_this_run]
    )
    failures = _run_bounded_collection(
        pending=scheduled,
        collect_row=collect_row,
        completed=completed,
        config=config,
    )
    unattempted = len(units) - len(completed) - len(failures)
    transport_complete = not failures and unattempted == 0
    status: dict[str, Any] = {
        "schema": EXACT_HISTORY_COLLECTION_SCHEMA,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha256,
        "plan_fingerprint": plan_fingerprint,
        "status": (
            "transport_complete"
            if transport_complete
            else "transport_incomplete"
        ),
        "coverage_status": "not_evaluated",
        "model_data_ready": False,
        "requested_units": len(units),
        "completed_units": len(completed),
        "failed_units": failures,
        "unattempted_units": unattempted,
        "stop_reason": (
            "complete"
            if transport_complete
            else (
                "failure_circuit"
                if failures
                else "operational_batch_limit"
            )
        ),
        "resumed_units": len(units) - len(pending),
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
    }
    _atomic_json(output_directory / "_status.json", status)
    if not transport_complete:
        return status
    if len(completed) != len(units):
        raise DataReadinessError(
            "ER1A collection lacks a terminal result for every unit"
        )
    records = [completed[unit_id] for unit_id in sorted(completed)]
    _archive_unreferenced_raw_pages(output_directory, records)
    manifest = {
        **status,
        "artifacts": records,
        "raw_page_inventory": _referenced_raw_page_inventory(
            output_directory,
            records,
        ),
        "total_rows": sum(int(record["rows"]) for record in records),
        "observed_symbols": sorted(
            {
                symbol
                for record in records
                for symbol, count in cast(
                    Mapping[str, int],
                    record["symbol_rows"],
                ).items()
                if int(count) > 0
            }
        ),
    }
    _atomic_json(output_directory / "_manifest.json", manifest)
    _atomic_json(
        output_directory / "_authority.json",
        {
            "schema": EXACT_HISTORY_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(
                output_directory / "_manifest.json"
            ),
            "request_sha256": request_sha256,
            "plan_fingerprint": plan_fingerprint,
        },
    )
    return manifest


def _run_bounded_collection(
    *,
    pending: list[dict[str, Any]],
    collect_row: Callable[[Mapping[str, Any]], dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    config: IntradayTransportConfig,
) -> dict[str, str]:
    failures: dict[str, str] = {}
    rows = iter(pending)
    futures: dict[Future[dict[str, Any]], str] = {}
    executor = ThreadPoolExecutor(max_workers=config.collection_workers)
    try:
        for _ in range(config.collection_workers):
            row = next(rows, None)
            if row is not None:
                futures[executor.submit(collect_row, row)] = str(
                    row["unit_id"]
                )
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                unit_id = futures.pop(future)
                try:
                    completed[unit_id] = future.result()
                except Exception as exc:
                    failures[unit_id] = (
                        f"{type(exc).__name__}: {str(exc)[:500]}"
                    )
                release_process_memory()
                _guard_memory(
                    config,
                    f"ER1A history persist {unit_id}",
                )
                if (
                    len(failures)
                    < config.maximum_failures_before_stop
                ):
                    row = next(rows, None)
                    if row is not None:
                        futures[executor.submit(collect_row, row)] = str(
                            row["unit_id"]
                        )
            if (
                len(failures)
                >= config.maximum_failures_before_stop
            ):
                for future in futures:
                    future.cancel()
                break
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return failures


def load_complete_intraday_history_collection(
    directory: Path,
) -> dict[str, Any]:
    request = _load_json(directory / "_request.json")
    manifest = _load_json(directory / "_manifest.json")
    authority = _load_json(directory / "_authority.json")
    request_sha256 = str(request.get("request_sha256", ""))
    payload = {
        key: value
        for key, value in request.items()
        if key != "request_sha256"
    }
    expected_authority_schema = (
        EXACT_HISTORY_AUTHORITY_SCHEMA
        if manifest.get("schema") == EXACT_HISTORY_COLLECTION_SCHEMA
        else HISTORY_AUTHORITY_SCHEMA
    )
    expected_request_schema = (
        EXACT_HISTORY_COLLECTION_SCHEMA
        if manifest.get("schema") == EXACT_HISTORY_COLLECTION_SCHEMA
        else HISTORY_COLLECTION_SCHEMA
    )
    if (
        request.get("schema") != expected_request_schema
        or _json_sha256(payload) != request_sha256
        or manifest.get("schema") not in {
            EXACT_HISTORY_COLLECTION_SCHEMA,
            HISTORY_COLLECTION_SCHEMA,
        }
        or manifest.get("status") != "transport_complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != expected_authority_schema
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError(
            "ER1A history collection lacks complete authority"
        )
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise DataReadinessError(
            "ER1A history manifest has no unit artifacts"
        )
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise DataReadinessError(
                "ER1A history unit record is malformed"
            )
        expected_unit_schema = (
            EXACT_HISTORY_UNIT_SCHEMA
            if manifest.get("schema") == EXACT_HISTORY_COLLECTION_SCHEMA
            else HISTORY_UNIT_SCHEMA
        )
        if raw.get("schema") != expected_unit_schema:
            raise DataReadinessError(
                "history collection mixes authority schema generations"
            )
        path = _resolve_inside(directory, str(raw.get("path", "")))
        sidecar = path.with_suffix(".manifest.json")
        if (
            not path.is_file()
            or not sidecar.is_file()
            or file_sha256(path) != raw.get("sha256")
            or _load_json(sidecar) != dict(raw)
        ):
            raise DataReadinessError(
                f"ER1A history unit does not verify: {path}"
            )
        _verify_raw_pages(directory, raw)
        if raw.get("schema") == EXACT_HISTORY_UNIT_SCHEMA:
            _verify_exact_unit_replay(directory, raw, path)
    if manifest.get("schema") == EXACT_HISTORY_COLLECTION_SCHEMA:
        expected_inventory = manifest.get("raw_page_inventory")
        referenced_inventory = _referenced_raw_page_inventory(
            directory,
            cast(list[Mapping[str, Any]], raw_artifacts),
        )
        if (
            not isinstance(expected_inventory, list)
            or expected_inventory != referenced_inventory
            or referenced_inventory != _raw_page_inventory(directory)
        ):
            raise DataReadinessError(
                "exact raw provider page inventory changed"
            )
    return manifest


def _collect_unit(
    row: Mapping[str, Any],
    *,
    source: AlpacaSource,
    root: Path,
    bars_directory: Path,
    attempts_directory: Path,
    raw_pages_directory: Path,
    plan_fingerprint: str,
    request_sha256: str,
    config: IntradayTransportConfig,
    timeframe: str,
) -> dict[str, Any]:
    unit_id = str(row["unit_id"])
    started_at = datetime.now(UTC)
    attempt_id = uuid.uuid4().hex
    attempt_path = attempts_directory / unit_id / f"{attempt_id}.json"
    canonical_symbols = _json_string_list(row["canonical_symbols_json"])
    provider_symbols = _json_string_list(row["provider_symbols_json"])
    mapping = _json_string_mapping(row["provider_to_canonical_json"])
    if (
        set(mapping) != set(provider_symbols)
        or set(mapping.values()) != set(canonical_symbols)
    ):
        raise DataReadinessError(
            f"ER1A unit symbol mapping is inconsistent: {unit_id}"
        )
    start = _aware_datetime(row["requested_start_utc"])
    end = _aware_datetime(row["requested_end_utc"])
    asof = pd.Timestamp(row["session_date_et"]).date()
    page_token: str | None = None
    seen_tokens: set[str] = set()
    raw_rows: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    try:
        while True:
            page = source.fetch_bars_page(
                provider_symbols,
                start,
                end - timedelta(microseconds=1),
                timeframe=timeframe,
                page_token=page_token,
                asof=asof,
                limit=10_000,
                retries=config.collection_retries,
            )
            page_number = len(pages) + 1
            raw_page_path = (
                raw_pages_directory
                / unit_id
                / attempt_id
                / f"page-{page_number:05d}.body"
            )
            raw_body = _require_exact_page(page)
            _atomic_bytes(raw_page_path, raw_body)
            raw_sidecar_path = raw_page_path.with_suffix(".json")
            page_record = _page_record(
                page,
                page_number=page_number,
                raw_page_path=raw_page_path,
                raw_sidecar_path=raw_sidecar_path,
                root=root,
            )
            _atomic_json(raw_sidecar_path, page_record)
            page_rows = 0
            for provider, values in page.bars.items():
                canonical = mapping.get(provider)
                if canonical is None:
                    raise DataReadinessError(
                        "Alpaca returned an unmapped provider symbol"
                    )
                for value in values:
                    raw_rows.append(
                        {
                            "ticker": canonical,
                            "timestamp": value.get("t"),
                            "open": value.get("o"),
                            "high": value.get("h"),
                            "low": value.get("l"),
                            "close": value.get("c"),
                            "volume": value.get("v"),
                        }
                    )
                    page_rows += 1
            pages.append({**page_record, "rows": page_rows})
            if len(raw_rows) > int(row["maximum_expected_rows"]):
                raise DataReadinessError(
                    f"Alpaca unit exceeded its row budget: {unit_id}"
                )
            next_token = page.next_page_token
            if next_token is None:
                break
            if len(pages) >= config.maximum_pages_per_unit:
                raise DataReadinessError(
                    f"Alpaca unit exceeded its page budget: {unit_id}"
                )
            if next_token in seen_tokens:
                raise DataReadinessError(
                    f"Alpaca repeated a page token: {unit_id}"
                )
            seen_tokens.add(next_token)
            page_token = next_token
        canonical_ingested_at = datetime.now(UTC)
        bars = _canonical_bars(
            raw_rows,
            finalization_delay_seconds=(
                config.intraday_finalization_delay_seconds
            ),
            ingested_at=canonical_ingested_at,
            timeframe=_canonical_timeframe(timeframe),
        )
        if not bars.empty:
            outside = bars["bar_start_utc"].lt(start) | bars[
                "bar_start_utc"
            ].ge(end)
            starts = pd.to_datetime(bars["bar_start_utc"], utc=True)
            if (
                bool(outside.any())
                or bool(
                    bars.duplicated(["ticker", "bar_start_utc"]).any()
                )
                or not _timestamps_aligned(starts, timeframe)
                or not set(bars["ticker"].astype(str)).issubset(
                    canonical_symbols
                )
            ):
                raise DataReadinessError(
                    f"ER1A canonical unit content is invalid: {unit_id}"
                )
        symbol_rows = {
            symbol: (
                int(bars["ticker"].eq(symbol).sum())
                if not bars.empty
                else 0
            )
            for symbol in canonical_symbols
        }
        expected_rows = int(row["expected_bars_per_symbol"])
        symbol_coverage = {
            symbol: {
                "expected_rows": expected_rows,
                "observed_rows": count,
                "status": (
                    "complete"
                    if count == expected_rows
                    else "unavailable"
                    if count == 0
                    else "sparse"
                ),
            }
            for symbol, count in symbol_rows.items()
        }
        month = asof.strftime("%Y-%m")
        path = bars_directory / month / f"{unit_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(bars, path)
        record: dict[str, Any] = {
            "schema": EXACT_HISTORY_UNIT_SCHEMA,
            "unit_id": unit_id,
            "plan_fingerprint": plan_fingerprint,
            "request_sha256": request_sha256,
            "path": str(path.relative_to(root)),
            "sha256": file_sha256(path),
            "rows": len(bars),
            "symbol_rows": symbol_rows,
            "symbol_coverage": symbol_coverage,
            "provider_to_canonical": mapping,
            "requested_start_utc": start.isoformat(),
            "requested_end_utc": end.isoformat(),
            "provider_end_inclusive_utc": (
                end - timedelta(microseconds=1)
            ).isoformat(),
            "asof_date": asof.isoformat(),
            "timeframe": _canonical_timeframe(timeframe),
            "price_feed": "sip",
            "adjustment": "all",
            "intraday_finalization_delay_seconds": (
                config.intraday_finalization_delay_seconds
            ),
            "canonical_ingested_at_utc": canonical_ingested_at.isoformat(),
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "pages": pages,
        }
        _atomic_json(path.with_suffix(".manifest.json"), record)
        _atomic_json(
            attempt_path,
            {
                "attempt_id": attempt_id,
                "unit_id": unit_id,
                "status": "observed",
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "rows": len(bars),
            },
        )
        return record
    except Exception as exc:
        _atomic_json(
            attempt_path,
            {
                "attempt_id": attempt_id,
                "unit_id": unit_id,
                "status": "failed",
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "pages": pages,
            },
        )
        raise


def _canonical_bars(
    rows: list[dict[str, Any]],
    *,
    finalization_delay_seconds: int,
    ingested_at: datetime,
    timeframe: str,
) -> pd.DataFrame:
    if rows:
        return canonicalize_bars(
            pd.DataFrame(rows),
            timeframe=timeframe,
            source="alpaca",
            price_feed="sip",
            adjustment="all",
            ingested_at_utc=ingested_at,
            availability_policy="market_interval_close",
            intraday_finalization_delay=pd.Timedelta(
                seconds=finalization_delay_seconds
            ),
        )
    return canonicalize_bars(
        pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        ),
        timeframe=timeframe,
        source="alpaca",
        price_feed="sip",
        adjustment="all",
    )


def _load_existing_unit(
    path: Path,
    *,
    root: Path,
    expected_unit: Mapping[str, Any],
    plan_fingerprint: str,
    request_sha256: str,
    timeframe: str,
) -> dict[str, Any] | None:
    sidecar = path.with_suffix(".manifest.json")
    if not path.exists() and not sidecar.exists():
        return None
    if not path.exists() or not sidecar.exists():
        quarantine = path.parent / "_quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        for orphan in (path, sidecar):
            if orphan.exists():
                orphan.replace(
                    quarantine / f"{orphan.name}.{suffix}.orphan"
                )
        return None
    manifest = _load_json(sidecar)
    symbols = set(_json_string_list(
        expected_unit["canonical_symbols_json"]
    ))
    start = _aware_datetime(expected_unit["requested_start_utc"])
    end = _aware_datetime(expected_unit["requested_end_utc"])
    if (
        manifest.get("schema") not in {
            EXACT_HISTORY_UNIT_SCHEMA,
            HISTORY_UNIT_SCHEMA,
        }
        or manifest.get("unit_id") != expected_unit["unit_id"]
        or manifest.get("plan_fingerprint") != plan_fingerprint
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("path") != str(path.relative_to(root))
        or manifest.get("requested_start_utc") != start.isoformat()
        or manifest.get("requested_end_utc") != end.isoformat()
        or manifest.get("timeframe") != timeframe
        or manifest.get("price_feed") != "sip"
        or manifest.get("adjustment") != "all"
        or not isinstance(manifest.get("symbol_rows"), Mapping)
        or set(cast(Mapping[str, object], manifest["symbol_rows"]))
        != symbols
        or manifest.get("sha256") != file_sha256(path)
    ):
        raise DataReadinessError(
            f"ER1A collected unit integrity failed: {path}"
        )
    _verify_raw_pages(root, manifest)
    if manifest.get("schema") == EXACT_HISTORY_UNIT_SCHEMA:
        _verify_exact_unit_replay(root, manifest, path)
    frame = pd.read_parquet(
        path,
        columns=[
            "ticker",
            "bar_start_utc",
            "source",
            "timeframe",
            "price_feed",
            "adjustment",
        ],
    )
    starts = pd.to_datetime(frame["bar_start_utc"], utc=True)
    if (
        len(frame) != int(manifest.get("rows", -1))
        or (
            not frame.empty
            and (
                bool(frame["price_feed"].ne("sip").any())
                or bool(frame["adjustment"].ne("all").any())
                or bool(frame["source"].ne("alpaca").any())
                or bool(frame["timeframe"].ne(timeframe).any())
                or not set(frame["ticker"].astype(str)).issubset(symbols)
                or bool(starts.lt(start).any())
                or bool(starts.ge(end).any())
                or not _timestamps_aligned(starts, timeframe)
                or bool(
                    frame.duplicated(["ticker", "bar_start_utc"]).any()
                )
            )
        )
    ):
        raise DataReadinessError(
            f"ER1A collected unit content failed: {path}"
        )
    return manifest


def _verify_raw_pages(root: Path, unit: Mapping[str, Any]) -> None:
    pages = unit.get("pages")
    if not isinstance(pages, list) or not pages:
        raise DataReadinessError("collected unit has no raw provider pages")
    for raw in pages:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("collected unit has a malformed raw page")
        path = _resolve_inside(root, str(raw.get("raw_page_path", "")))
        expected_bytes = int(raw.get("raw_page_bytes", -1))
        if (
            not path.is_file()
            or path.stat().st_size != expected_bytes
            or file_sha256(path) != raw.get("raw_page_sha256")
        ):
            raise DataReadinessError(
                f"collected raw provider page failed integrity: {path}"
            )
        if unit.get("schema") == EXACT_HISTORY_UNIT_SCHEMA:
            sidecar = _resolve_inside(
                root,
                str(raw.get("raw_sidecar_path", "")),
            )
            if not sidecar.is_file():
                raise DataReadinessError(
                    f"collected raw provider sidecar is missing: {sidecar}"
                )
            expected_sidecar = {
                str(key): value
                for key, value in raw.items()
                if key != "rows"
            }
            if _load_json(sidecar) != expected_sidecar:
                raise DataReadinessError(
                    f"collected raw provider sidecar changed: {sidecar}"
                )
            _verify_exact_page_body(path, raw)


def _transport_timeframe(config: IntradayTransportConfig) -> str:
    for field in ("history_timeframe", "feature_timeframe", "context_timeframe"):
        value = getattr(config, field, None)
        if value in {"1Min", "5Min"}:
            return str(value)
    raise DataReadinessError("collection policy has no supported intraday timeframe")


def _timeframe_minutes(timeframe: str) -> int:
    normalized = timeframe.lower()
    if normalized == "1m" or normalized == "1min":
        return 1
    if normalized == "5m" or normalized == "5min":
        return 5
    raise DataReadinessError(f"unsupported intraday timeframe: {timeframe}")


def _canonical_timeframe(timeframe: str) -> str:
    minutes = _timeframe_minutes(timeframe)
    return f"{minutes}m"


def _timestamps_aligned(starts: pd.Series, timeframe: str) -> bool:
    minutes = _timeframe_minutes(timeframe)
    return not bool(
        (
            starts.dt.minute.mod(minutes).ne(0)
            | starts.dt.second.ne(0)
            | starts.dt.microsecond.ne(0)
        ).any()
    )


def _json_string_list(value: object) -> tuple[str, ...]:
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SchemaMismatchError("invalid ER1A symbol list") from exc
    if (
        not isinstance(loaded, list)
        or not loaded
        or any(not isinstance(item, str) or not item for item in loaded)
    ):
        raise SchemaMismatchError("invalid ER1A symbol list")
    return tuple(loaded)


def _json_string_mapping(value: object) -> dict[str, str]:
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SchemaMismatchError("invalid ER1A symbol mapping") from exc
    if (
        not isinstance(loaded, dict)
        or not loaded
        or any(
            not isinstance(key, str)
            or not key
            or not isinstance(item, str)
            or not item
            for key, item in loaded.items()
        )
    ):
        raise SchemaMismatchError("invalid ER1A symbol mapping")
    return {str(key): str(item) for key, item in loaded.items()}


def _aware_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError("ER1A unit bound is timezone-naive")
    return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())


def _require_exact_page(page: AlpacaBarsPage) -> bytes:
    if (
        page.raw_body is None
        or page.requested_url is None
        or page.status_code != 200
        or page.retrieved_at_utc is None
        or page.retrieved_at_utc.tzinfo is None
        or page.final_url != page.requested_url
        or page.redirect_chain
        or _media_type(_response_header(page, "content-type"))
        != "application/json"
    ):
        raise DataReadinessError(
            "Alpaca page lacks exact direct-HTTP transport evidence"
        )
    try:
        payload = json.loads(page.raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            "Alpaca exact page body is not UTF-8 JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise DataReadinessError("Alpaca exact page body is not an object")
    expected_bars = {
        symbol: list(values) for symbol, values in page.bars.items()
    }
    if (
        payload.get("bars", {}) != expected_bars
        or payload.get("next_page_token") != page.next_page_token
    ):
        raise DataReadinessError(
            "Alpaca parsed page differs from its exact HTTP body"
        )
    return page.raw_body


def _page_record(
    page: AlpacaBarsPage,
    *,
    page_number: int,
    raw_page_path: Path,
    raw_sidecar_path: Path,
    root: Path,
) -> dict[str, Any]:
    raw_body = _require_exact_page(page)
    return {
        "page_number": page_number,
        "request_page_token": page.request_page_token,
        "next_page_token": page.next_page_token,
        "requested_url": page.requested_url,
        "final_url": page.final_url,
        "status_code": page.status_code,
        "retrieved_at_utc": page.retrieved_at_utc.isoformat()
        if page.retrieved_at_utc is not None
        else None,
        "redirect_chain": list(page.redirect_chain),
        "body_representation": "http_entity_encoded",
        "content_type": _response_header(page, "content-type"),
        "content_encoding": _response_header(page, "content-encoding"),
        "raw_page_path": raw_page_path.relative_to(root).as_posix(),
        "raw_sidecar_path": raw_sidecar_path.relative_to(root).as_posix(),
        "raw_page_sha256": hashlib.sha256(raw_body).hexdigest(),
        "raw_page_bytes": len(raw_body),
        "rate_headers": {
            key.lower(): value
            for key, value in page.response_headers.items()
            if key.lower() in _SAFE_RATE_HEADERS
        },
    }


def _response_header(page: AlpacaBarsPage, name: str) -> str | None:
    return next(
        (
            str(value)
            for key, value in page.response_headers.items()
            if key.lower() == name
        ),
        None,
    )


def _verify_exact_page_body(path: Path, record: Mapping[str, Any]) -> None:
    if (
        record.get("status_code") != 200
        or record.get("final_url") != record.get("requested_url")
        or record.get("redirect_chain") != []
        or record.get("body_representation") != "http_entity_encoded"
        or _media_type(cast(str | None, record.get("content_type")))
        != "application/json"
        or record.get("content_encoding") not in (None, "", "identity")
    ):
        raise DataReadinessError("exact provider transport metadata is invalid")
    retrieved = pd.Timestamp(record.get("retrieved_at_utc"))
    if retrieved.tzinfo is None:
        raise DataReadinessError("exact provider retrieval time is naive")
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError("exact provider body is not UTF-8 JSON") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("next_page_token") != record.get("next_page_token")
        or not isinstance(payload.get("bars", {}), dict)
    ):
        raise DataReadinessError("exact provider body and sidecar differ")


def _verify_exact_unit_replay(
    root: Path,
    unit: Mapping[str, Any],
    canonical_path: Path,
) -> None:
    pages = cast(list[Mapping[str, Any]], unit["pages"])
    mapping_value = unit.get("provider_to_canonical")
    coverage_value = unit.get("symbol_coverage")
    if (
        not isinstance(mapping_value, Mapping)
        or not isinstance(coverage_value, Mapping)
    ):
        raise DataReadinessError("exact unit lacks mapping or coverage evidence")
    mapping = {str(key): str(value) for key, value in mapping_value.items()}
    if int(unit.get("intraday_finalization_delay_seconds", -1)) != 60:
        raise DataReadinessError(
            "exact unit finalization delay differs from the frozen policy"
        )
    raw_rows: list[dict[str, Any]] = []
    expected_request_token: str | None = None
    terminal_seen = False
    for expected_number, page in enumerate(pages, start=1):
        if (
            int(page.get("page_number", -1)) != expected_number
            or page.get("request_page_token") != expected_request_token
            or terminal_seen
        ):
            raise DataReadinessError("exact provider page chain is invalid")
        _verify_unit_request_url(str(page.get("requested_url", "")), unit, page)
        path = _resolve_inside(root, str(page["raw_page_path"]))
        payload = json.loads(path.read_bytes().decode("utf-8"))
        raw_bars = cast(Mapping[str, object], payload.get("bars", {}))
        unexpected = set(raw_bars).difference(mapping)
        if unexpected:
            raise DataReadinessError("exact provider body has unexpected symbols")
        page_rows = 0
        for provider, values in raw_bars.items():
            if not isinstance(values, list):
                raise DataReadinessError("exact provider bars are malformed")
            for value in values:
                if not isinstance(value, Mapping):
                    raise DataReadinessError("exact provider bar is malformed")
                raw_rows.append(
                    {
                        "ticker": mapping[provider],
                        "timestamp": value.get("t"),
                        "open": value.get("o"),
                        "high": value.get("h"),
                        "low": value.get("l"),
                        "close": value.get("c"),
                        "volume": value.get("v"),
                    }
                )
                page_rows += 1
        if page_rows != int(page.get("rows", -1)):
            raise DataReadinessError("exact provider page row count changed")
        expected_request_token = cast(str | None, page.get("next_page_token"))
        terminal_seen = expected_request_token is None
    if not terminal_seen:
        raise DataReadinessError("exact provider page chain is not terminal")
    ingested_at = _aware_datetime(unit.get("canonical_ingested_at_utc"))
    expected = _canonical_bars(
        raw_rows,
        finalization_delay_seconds=int(
            unit.get("intraday_finalization_delay_seconds", -1)
        ),
        ingested_at=ingested_at,
        timeframe=str(unit["timeframe"]),
    )
    actual = pd.read_parquet(canonical_path)
    if list(actual.columns) != list(expected.columns):
        raise DataReadinessError("canonical bar schema does not replay")
    for field in (
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "ingested_at_utc",
    ):
        expected[field] = pd.to_datetime(expected[field], utc=True, errors="raise")
        actual[field] = pd.to_datetime(actual[field], utc=True, errors="raise")
    order = ["ticker", "bar_start_utc"]
    expected = expected.sort_values(order, kind="stable").reset_index(drop=True)
    actual = actual.sort_values(order, kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            actual,
            expected,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as exc:
        raise DataReadinessError(
            "canonical bars do not replay from exact provider bytes"
        ) from exc
    observed_counts = expected.groupby("ticker", observed=True).size().to_dict()
    for ticker, value in coverage_value.items():
        observed_rows = int(observed_counts.get(str(ticker), 0))
        expected_rows = int(value.get("expected_rows", -1)) if isinstance(value, Mapping) else -1
        expected_status = (
            "complete"
            if observed_rows == expected_rows
            else "unavailable"
            if observed_rows == 0
            else "sparse"
        )
        if (
            not isinstance(value, Mapping)
            or int(value.get("observed_rows", -1))
            != observed_rows
            or value.get("status") != expected_status
        ):
            raise DataReadinessError("exact symbol coverage changed")


def _verify_unit_request_url(
    requested_url: str,
    unit: Mapping[str, Any],
    page: Mapping[str, Any],
) -> None:
    parsed = urlsplit(requested_url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise DataReadinessError("exact provider request port is invalid") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "data.alpaca.markets"
        or port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v2/stocks/bars"
        or parsed.fragment
    ):
        raise DataReadinessError("exact provider request endpoint changed")
    query = parse_qs(parsed.query, keep_blank_values=True)
    mapping = cast(Mapping[str, str], unit["provider_to_canonical"])
    expected_timeframe = "1Min" if unit.get("timeframe") == "1m" else "5Min"
    expected = {
        "symbols": ",".join(mapping),
        "timeframe": expected_timeframe,
        "start": str(unit["requested_start_utc"]),
        "end": str(unit["provider_end_inclusive_utc"]),
        "feed": "sip",
        "limit": "10000",
        "adjustment": "all",
        "sort": "asc",
        "asof": str(unit["asof_date"]),
    }
    request_token = page.get("request_page_token")
    if request_token is not None:
        expected["page_token"] = str(request_token)
    if any(len(values) != 1 for values in query.values()):
        raise DataReadinessError("exact provider request query is ambiguous")
    actual = {key: values[0] for key, values in query.items()}
    if actual != expected:
        raise DataReadinessError("exact provider request URL changed")


def _raw_page_inventory(root: Path) -> list[dict[str, object]]:
    raw_root = root / "raw_pages"
    if not raw_root.exists():
        return []
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(raw_root.rglob("*"))
        if path.is_file()
    ]


def _referenced_raw_page_inventory(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, object]]:
    relative_paths = sorted(
        {
            str(page[field]).replace("\\", "/")
            for record in records
            for page in cast(list[Mapping[str, Any]], record["pages"])
            for field in ("raw_page_path", "raw_sidecar_path")
        }
    )
    inventory: list[dict[str, object]] = []
    for relative in relative_paths:
        path = _resolve_inside(root, relative)
        if not path.is_file():
            raise DataReadinessError(
                f"referenced raw provider page is missing: {path}"
            )
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return inventory


def _media_type(value: str | None) -> str:
    return (value or "").split(";", maxsplit=1)[0].strip().lower()


def _archive_unreferenced_raw_pages(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    referenced = {
        str(page[field]).replace("\\", "/")
        for record in records
        for page in cast(list[Mapping[str, Any]], record["pages"])
        for field in ("raw_page_path", "raw_sidecar_path")
    }
    raw_root = root / "raw_pages"
    if not raw_root.exists():
        return
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in referenced:
            continue
        destination = root / "failed_attempt_pages" / path.relative_to(raw_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise DataReadinessError(
            f"ER1A artifact escapes collection root: {relative}"
        )
    return candidate


def _write_or_validate_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    if path.exists():
        if _load_json(path) != dict(payload):
            raise DataReadinessError(
                f"ER1A resume identity differs: {path}"
            )
        return
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                dict(payload),
                indent=2,
                sort_keys=True,
                default=str,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"ER1A JSON artifact is unreadable: {path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(
            f"ER1A JSON artifact is not an object: {path}"
        )
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _guard_memory(config: IntradayTransportConfig, stage: str) -> None:
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


def discard_incomplete_collection(directory: Path) -> None:
    """Remove only a caller-verified incomplete ER1A collection."""

    if (directory / "_authority.json").exists():
        raise DataReadinessError(
            "refusing to discard an authoritative ER1A collection"
        )
    shutil.rmtree(directory)
