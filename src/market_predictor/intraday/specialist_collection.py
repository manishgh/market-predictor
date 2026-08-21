"""Batched and resumable Alpaca SIP collection for KS4 one-minute paths."""
from __future__ import annotations



import gzip
import hashlib
import json
import shutil
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import file_sha256
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
from market_predictor.sources.alpaca import AlpacaBarsPage, AlpacaSource
from market_predictor.symbols import provider_symbol
from market_predictor.core.errors import DataReadinessError, SchemaMismatchError

SPECIALIST_ACQUISITION_UNIT_BUNDLE_SCHEMA = (
    "intraday.specialist_acquisition_units.v1"
)
SPECIALIST_ONE_MINUTE_COLLECTION_SCHEMA = (
    "intraday.specialist_one_minute_collection.v1"
)
SPECIALIST_ONE_MINUTE_UNIT_SCHEMA = (
    "intraday.specialist_one_minute_unit.v1"
)
_SAFE_RATE_HEADERS = {
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
}
SourceFactory = Callable[[], AlpacaSource]


def build_intraday_specialist_acquisition_units(
    *,
    collection_plan_directory: Path,
    policy_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Consolidate exact windows into bounded full-session Alpaca units."""

    if output_directory.exists():
        raise DataReadinessError(
            f"KS4 acquisition unit output must be new: {output_directory}"
        )
    plan = verify_specialist_collection_plan(collection_plan_directory)
    config = load_intraday_specialist_research_config(policy_path)
    pair_parts = [
        pd.read_parquet(
            path,
            columns=[
                "ticker",
                "session_date_et",
                "price_feed",
                "adjustment",
                "timeframe",
            ],
        )
        for path in sorted(
            (collection_plan_directory / "collection_windows").glob(
                "*.parquet"
            )
        )
    ]
    if not pair_parts:
        raise DataReadinessError(
            "KS4 collection plan has no window shards"
        )
    raw_pairs = pd.concat(pair_parts, ignore_index=True)
    if (
        bool(raw_pairs["price_feed"].astype(str).str.lower().ne("sip").any())
        or bool(
            raw_pairs["adjustment"]
            .astype(str)
            .str.lower()
            .ne("all")
            .any()
        )
        or bool(
            raw_pairs["timeframe"].astype(str).str.lower().ne("1m").any()
        )
    ):
        raise DataReadinessError(
            "KS4 collection plan contains non-SIP/all/1m windows"
        )
    pairs = (
        raw_pairs
        .assign(
            ticker=lambda frame: frame["ticker"]
            .astype(str)
            .str.upper()
            .str.strip(),
            session_date_et=lambda frame: pd.to_datetime(
                frame["session_date_et"]
            ).dt.date,
        )
        .drop_duplicates(["ticker", "session_date_et"])
        .sort_values(["session_date_et", "ticker"], kind="stable")
        .reset_index(drop=True)
    )
    if bool(pairs["ticker"].eq("").any()):
        raise DataReadinessError("KS4 acquisition plan has an empty ticker")
    calendar = xcals.get_calendar("XNYS")
    plan_fingerprint = str(plan["plan_fingerprint"])
    by_month: dict[str, list[dict[str, Any]]] = {}
    unit_ids: set[str] = set()
    expected_rows = 0
    provider_symbols_seen: set[str] = set()
    for session_date, session_pairs in pairs.groupby(
        "session_date_et",
        sort=True,
    ):
        session = calendar.date_to_session(
            pd.Timestamp(session_date),
            direction="none",
        )
        open_at = pd.Timestamp(calendar.session_open(session)).tz_convert(
            "UTC"
        )
        close_at = pd.Timestamp(calendar.session_close(session)).tz_convert(
            "UTC"
        )
        session_minutes = int(
            (close_at - open_at).total_seconds() // 60
        )
        symbols_per_unit = min(
            config.alpaca_unit_max_symbols,
            config.alpaca_unit_max_expected_rows // session_minutes,
        )
        if symbols_per_unit < 1:
            raise DataReadinessError(
                f"KS4 row cap cannot fit one session: {session_date}"
            )
        symbol_pairs = sorted(
            (
                str(ticker),
                provider_symbol(str(ticker), "alpaca"),
            )
            for ticker in session_pairs["ticker"]
        )
        provider_to_canonical = {
            provider: canonical for canonical, provider in symbol_pairs
        }
        if len(provider_to_canonical) != len(symbol_pairs):
            raise DataReadinessError(
                f"KS4 provider-symbol mapping collides on {session_date}"
            )
        provider_symbols_seen.update(provider_to_canonical)
        for offset in range(0, len(symbol_pairs), symbols_per_unit):
            chunk = symbol_pairs[offset : offset + symbols_per_unit]
            canonical_symbols = [canonical for canonical, _ in chunk]
            provider_symbols = [provider for _, provider in chunk]
            mapping = {
                provider: canonical for canonical, provider in chunk
            }
            unit_id = _stable_hash(
                plan_fingerprint,
                pd.Timestamp(session_date).date().isoformat(),
                open_at.isoformat(),
                close_at.isoformat(),
                *provider_symbols,
                "1Min",
                "sip",
                "all",
            )
            if unit_id in unit_ids:
                raise DataReadinessError(
                    "KS4 acquisition unit identity collision"
                )
            unit_ids.add(unit_id)
            row_count = session_minutes * len(chunk)
            expected_rows += row_count
            month = pd.Timestamp(session_date).strftime("%Y-%m")
            by_month.setdefault(month, []).append(
                {
                    "unit_id": unit_id,
                    "session_date_et": pd.Timestamp(session_date).date(),
                    "requested_start_utc": open_at,
                    "requested_end_utc": close_at,
                    "asof_date": pd.Timestamp(session_date).date(),
                    "canonical_symbols_json": json.dumps(
                        canonical_symbols,
                        separators=(",", ":"),
                    ),
                    "provider_symbols_json": json.dumps(
                        provider_symbols,
                        separators=(",", ":"),
                    ),
                    "provider_to_canonical_json": json.dumps(
                        mapping,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "symbol_count": len(chunk),
                    "session_minutes": session_minutes,
                    "maximum_expected_rows": row_count,
                    "timeframe": "1Min",
                    "price_feed": "sip",
                    "adjustment": "all",
                    "sort": "asc",
                    "limit": 10_000,
                    "collection_plan_fingerprint": plan_fingerprint,
                }
            )
    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        for month, rows in sorted(by_month.items()):
            frame = pd.DataFrame(rows).sort_values(
                ["requested_start_utc", "unit_id"],
                kind="stable",
            )
            path = temporary / "units" / f"{month}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            files.append(_file_record(path, temporary, rows=len(frame)))
        unit_fingerprint = _unit_bundle_fingerprint(
            files=files,
            collection_plan_fingerprint=plan_fingerprint,
            policy_sha256=config.policy_sha256(),
        )
        report: dict[str, Any] = {
            "schema": SPECIALIST_ACQUISITION_UNIT_BUNDLE_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "unit_bundle_fingerprint": unit_fingerprint,
            "collection_plan": {
                "path": str(collection_plan_directory),
                "plan_fingerprint": plan_fingerprint,
                "manifest_sha256": file_sha256(
                    collection_plan_directory / "_manifest.json"
                ),
            },
            "policy": intraday_specialist_policy_identity(policy_path),
            "collection_plan_policy": plan["policy"],
            "provider": {
                "name": "alpaca",
                "timeframe": "1Min",
                "price_feed": "sip",
                "adjustment": "all",
                "sort": "asc",
                "limit": 10_000,
                "calendar": "XNYS",
                "calendar_version": version("exchange-calendars"),
                "asof_policy": "session_date_et",
                "full_regular_session_superset": True,
            },
            "summary": {
                "units": len(unit_ids),
                "ticker_sessions": len(pairs),
                "sessions": int(pairs["session_date_et"].nunique()),
                "canonical_tickers": int(pairs["ticker"].nunique()),
                "provider_symbols": len(provider_symbols_seen),
                "maximum_expected_rows": expected_rows,
            },
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


def verify_intraday_specialist_acquisition_units(
    directory: Path,
) -> dict[str, Any]:
    """Verify a complete acquisition-unit bundle."""

    manifest = _load_json(directory / "_manifest.json")
    if manifest.get("schema") != SPECIALIST_ACQUISITION_UNIT_BUNDLE_SCHEMA:
        raise DataReadinessError(
            "unsupported KS4 acquisition-unit bundle schema"
        )
    files = _verify_registered_files(directory, manifest)
    expected = _unit_bundle_fingerprint(
        files=files,
        collection_plan_fingerprint=str(
            _nested(manifest, "collection_plan", "plan_fingerprint")
        ),
        policy_sha256=str(
            _nested(manifest, "policy", "policy_sha256")
        ),
    )
    if expected != manifest.get("unit_bundle_fingerprint"):
        raise DataReadinessError(
            "KS4 acquisition-unit fingerprint is invalid"
        )
    return manifest


def iter_acquisition_unit_shards(
    directory: Path,
) -> Iterator[pd.DataFrame]:
    verify_intraday_specialist_acquisition_units(directory)
    for path in sorted((directory / "units").glob("*.parquet")):
        yield pd.read_parquet(path)


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


def _verify_registered_files(
    directory: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("KS4 manifest has no registered files")
    files = [
        cast(dict[str, Any], record)
        for record in raw_files
        if isinstance(record, dict)
    ]
    if len(files) != len(raw_files):
        raise DataReadinessError("KS4 manifest file record is malformed")
    expected = {directory / str(record["path"]) for record in files}
    actual = {
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name != "_manifest.json"
    }
    if expected != actual:
        raise DataReadinessError("KS4 bundle has an unexpected file set")
    for record in files:
        path = directory / str(record["path"])
        if file_sha256(path) != str(record["sha256"]):
            raise DataReadinessError(
                f"KS4 bundle file is hash-invalid: {path}"
            )
    return files


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
        raise DataReadinessError(f"KS4 JSON artifact is not an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _file_record(path: Path, root: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _unit_bundle_fingerprint(
    *,
    files: Sequence[Mapping[str, Any]],
    collection_plan_fingerprint: str,
    policy_sha256: str,
) -> str:
    payload = {
        "schema": SPECIALIST_ACQUISITION_UNIT_BUNDLE_SCHEMA,
        "collection_plan_fingerprint": collection_plan_fingerprint,
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


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def collect_intraday_specialist_one_minute(
    *,
    acquisition_units_directory: Path,
    policy_path: Path,
    output_directory: Path,
    source_factory: SourceFactory,
) -> dict[str, Any]:
    """Collect all acquisition units with integrity-checked unit resume."""

    units_manifest = verify_intraday_specialist_acquisition_units(
        acquisition_units_directory
    )
    config = load_intraday_specialist_research_config(policy_path)
    if (
        _nested(units_manifest, "policy", "policy_sha256")
        != config.policy_sha256()
    ):
        raise DataReadinessError(
            "KS4 acquisition units and collection policy differ"
        )
    if (output_directory / "_manifest.json").exists():
        raise DataReadinessError(
            "completed KS4 one-minute collection is immutable"
        )
    unit_parts = [
        pd.read_parquet(path)
        for path in sorted(
            (acquisition_units_directory / "units").glob("*.parquet")
        )
    ]
    if not unit_parts:
        raise DataReadinessError("KS4 acquisition unit bundle is empty")
    units = pd.concat(unit_parts, ignore_index=True)
    if bool(units["unit_id"].duplicated().any()):
        raise DataReadinessError("KS4 acquisition units contain duplicates")
    unit_bundle_fingerprint = str(
        units_manifest["unit_bundle_fingerprint"]
    )
    request_payload = {
        "schema": SPECIALIST_ONE_MINUTE_COLLECTION_SCHEMA,
        "unit_bundle_path": str(acquisition_units_directory),
        "unit_bundle_fingerprint": unit_bundle_fingerprint,
        "unit_manifest_sha256": file_sha256(
            acquisition_units_directory / "_manifest.json"
        ),
        "policy": intraday_specialist_policy_identity(policy_path),
        "provider": "alpaca",
        "timeframe": "1Min",
        "price_feed": "sip",
        "adjustment": "all",
        "workers": config.alpaca_collection_workers,
        "retries": config.alpaca_collection_retries,
        "request_timeout_seconds": config.alpaca_request_timeout_seconds,
    }
    request_sha256 = hashlib.sha256(
        json.dumps(
            request_payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
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
    for row in units.to_dict(orient="records"):
        unit_id = str(row["unit_id"])
        month = pd.Timestamp(row["session_date_et"]).strftime("%Y-%m")
        existing = _load_existing_collected_unit(
            bars_directory / month / f"{unit_id}.parquet",
            expected_unit=row,
            expected_unit_id=unit_id,
            expected_unit_bundle_fingerprint=unit_bundle_fingerprint,
            expected_request_sha256=request_sha256,
        )
        if existing is None:
            pending.append({str(key): value for key, value in row.items()})
        else:
            completed[unit_id] = existing
    _guard_memory(config, "KS4 one-minute collection start")
    local = threading.local()

    def get_source() -> AlpacaSource:
        source = getattr(local, "alpaca_source", None)
        if source is None:
            source = source_factory()
            source.client.timeout = config.alpaca_request_timeout_seconds
            if source.settings.alpaca_stock_feed.strip().lower() != "sip":
                raise DataReadinessError(
                    "KS4 one-minute collection requires Alpaca SIP"
                )
            local.alpaca_source = source
        return cast(AlpacaSource, source)

    failures: dict[str, str] = {}

    def collect_row(row: Mapping[str, Any]) -> dict[str, Any]:
        return _collect_one_unit(
            row,
            source=get_source(),
            bars_directory=bars_directory,
            attempts_directory=attempts_directory,
            raw_pages_directory=raw_pages_directory,
            unit_bundle_fingerprint=unit_bundle_fingerprint,
            request_sha256=request_sha256,
            config=config,
        )

    with ThreadPoolExecutor(
        max_workers=config.alpaca_collection_workers
    ) as executor:
        futures = {
            executor.submit(collect_row, row): str(row["unit_id"])
            for row in pending
        }
        for future in as_completed(futures):
            unit_id = futures[future]
            try:
                completed[unit_id] = future.result()
            except Exception as exc:
                failures[unit_id] = (
                    f"{type(exc).__name__}: {str(exc)[:500]}"
                )
            release_process_memory()
            _guard_memory(config, f"KS4 one-minute persist {unit_id}")
    status = {
        "schema": SPECIALIST_ONE_MINUTE_COLLECTION_SCHEMA,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha256,
        "unit_bundle_fingerprint": unit_bundle_fingerprint,
        "status": (
            "transport_incomplete"
            if failures
            else "transport_complete"
        ),
        "coverage_status": "not_evaluated",
        "model_data_ready": False,
        "requested_units": len(units),
        "completed_units": len(completed),
        "failed_units": failures,
        "resumed_units": len(units) - len(pending),
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
    }
    _atomic_json(output_directory / "_status.json", status)
    if failures:
        return status
    if len(completed) != len(units):
        raise DataReadinessError(
            "KS4 collection ended without a terminal result for every unit"
        )
    records = [completed[unit_id] for unit_id in sorted(completed)]
    final = {
        **status,
        "artifacts": records,
        "total_rows": sum(int(record["rows"]) for record in records),
        "observed_symbols": sorted(
            {
                symbol
                for record in records
                for symbol in cast(
                    Mapping[str, int],
                    record["symbol_rows"],
                )
                if int(
                    cast(Mapping[str, int], record["symbol_rows"])[symbol]
                )
                > 0
            }
        ),
    }
    _atomic_json(output_directory / "_manifest.json", final)
    return final


def _collect_one_unit(
    row: Mapping[str, Any],
    *,
    source: AlpacaSource,
    bars_directory: Path,
    attempts_directory: Path,
    raw_pages_directory: Path,
    unit_bundle_fingerprint: str,
    request_sha256: str,
    config: IntradaySpecialistResearchConfig,
) -> dict[str, Any]:
    unit_id = str(row["unit_id"])
    started_at = datetime.now(UTC)
    attempt_id = uuid.uuid4().hex
    attempt_path = attempts_directory / unit_id / f"{attempt_id}.json"
    canonical_symbols = _json_string_list(
        row["canonical_symbols_json"],
        name="canonical symbols",
    )
    provider_symbols = _json_string_list(
        row["provider_symbols_json"],
        name="provider symbols",
    )
    mapping = _json_string_mapping(
        row["provider_to_canonical_json"],
        name="provider symbol mapping",
    )
    if set(mapping) != set(provider_symbols) or set(mapping.values()) != set(
        canonical_symbols
    ):
        raise DataReadinessError(
            f"KS4 acquisition mapping is inconsistent: {unit_id}"
        )
    start = _aware_datetime(row["requested_start_utc"])
    end = _aware_datetime(row["requested_end_utc"])
    asof = pd.Timestamp(row["asof_date"]).date()
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
                timeframe="1Min",
                page_token=page_token,
                asof=asof,
                limit=10_000,
                retries=config.alpaca_collection_retries,
            )
            page_rows = 0
            page_number = len(pages) + 1
            raw_page_path = (
                raw_pages_directory
                / unit_id
                / attempt_id
                / f"page-{page_number:05d}.json.gz"
            )
            raw_page = page.raw_payload or {
                "bars": {
                    symbol: list(rows)
                    for symbol, rows in page.bars.items()
                },
                "next_page_token": page.next_page_token,
            }
            _atomic_gzip_json(raw_page_path, raw_page)
            for provider, rows in page.bars.items():
                canonical = mapping.get(provider)
                if canonical is None:
                    raise DataReadinessError(
                        f"Alpaca returned an unmapped provider symbol: {provider}"
                    )
                for provider_row in rows:
                    raw_rows.append(
                        {
                            "ticker": canonical,
                            "timestamp": provider_row.get("t"),
                            "open": provider_row.get("o"),
                            "high": provider_row.get("h"),
                            "low": provider_row.get("l"),
                            "close": provider_row.get("c"),
                            "volume": provider_row.get("v"),
                        }
                    )
                    page_rows += 1
            pages.append(
                {
                    "page_number": page_number,
                    "request_page_token": page.request_page_token,
                    "next_page_token": page.next_page_token,
                    "rows": page_rows,
                    "response_sha256": _page_response_sha256(page),
                    "raw_page_path": str(raw_page_path),
                    "raw_page_sha256": file_sha256(raw_page_path),
                    "raw_page_bytes": raw_page_path.stat().st_size,
                    "rate_headers": {
                        key.lower(): value
                        for key, value in page.response_headers.items()
                        if key.lower() in _SAFE_RATE_HEADERS
                    },
                }
            )
            if len(raw_rows) > int(row["maximum_expected_rows"]) * 2:
                raise DataReadinessError(
                    f"Alpaca unit exceeded its bounded row budget: {unit_id}"
                )
            next_token = page.next_page_token
            if next_token is None:
                break
            if len(pages) >= config.alpaca_max_pages_per_unit:
                raise DataReadinessError(
                    f"Alpaca exceeded the page budget for unit {unit_id}"
                )
            if next_token in seen_tokens:
                raise DataReadinessError(
                    f"Alpaca repeated a page token for unit {unit_id}"
                )
            seen_tokens.add(next_token)
            page_token = next_token
        ingested_at = datetime.now(UTC)
        if raw_rows:
            bars = canonicalize_bars(
                pd.DataFrame(raw_rows),
                timeframe="1m",
                source="alpaca",
                price_feed="sip",
                adjustment="all",
                ingested_at_utc=ingested_at,
                availability_policy="market_interval_close",
                intraday_finalization_delay=pd.Timedelta(
                    seconds=config.intraday_finalization_delay_seconds
                ),
            )
        else:
            bars = canonicalize_bars(
                pd.DataFrame(
                    columns=[
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                    ]
                ),
                timeframe="1m",
                source="alpaca",
                price_feed="sip",
                adjustment="all",
            )
        if not bars.empty:
            outside = bars["bar_start_utc"].lt(start) | bars[
                "bar_start_utc"
            ].ge(end)
            if bool(outside.any()):
                raise DataReadinessError(
                    f"Alpaca returned rows outside unit bounds: {unit_id}"
                )
            if bool(
                bars.duplicated(
                    ["ticker", "bar_start_utc"]
                ).any()
            ):
                raise DataReadinessError(
                    f"Alpaca returned duplicate one-minute bars: {unit_id}"
                )
        symbol_rows = {
            symbol: int(bars["ticker"].eq(symbol).sum())
            if not bars.empty
            else 0
            for symbol in canonical_symbols
        }
        month = asof.strftime("%Y-%m")
        path = bars_directory / month / f"{unit_id}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_parquet(bars, path)
        record: dict[str, Any] = {
            "schema": SPECIALIST_ONE_MINUTE_UNIT_SCHEMA,
            "unit_id": unit_id,
            "unit_bundle_fingerprint": unit_bundle_fingerprint,
            "request_sha256": request_sha256,
            "path": str(path),
            "sha256": file_sha256(path),
            "rows": len(bars),
            "symbol_rows": symbol_rows,
            "requested_start_utc": start.isoformat(),
            "requested_end_utc": end.isoformat(),
            "provider_end_inclusive_utc": (
                end - timedelta(microseconds=1)
            ).isoformat(),
            "asof_date": asof.isoformat(),
            "timeframe": "1m",
            "price_feed": "sip",
            "adjustment": "all",
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


def _load_existing_collected_unit(
    path: Path,
    *,
    expected_unit: Mapping[str, Any],
    expected_unit_id: str,
    expected_unit_bundle_fingerprint: str,
    expected_request_sha256: str,
) -> dict[str, Any] | None:
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists() and not manifest_path.exists():
        return None
    if not path.exists() or not manifest_path.exists():
        quarantine = path.parent / "_quarantine"
        quarantine.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        for orphan in (path, manifest_path):
            if orphan.exists():
                orphan.replace(
                    quarantine / f"{orphan.name}.{suffix}.orphan"
                )
        return None
    manifest = _load_json(manifest_path)
    expected_symbols = set(
        _json_string_list(
            expected_unit["canonical_symbols_json"],
            name="canonical symbols",
        )
    )
    expected_start = _aware_datetime(
        expected_unit["requested_start_utc"]
    )
    expected_end = _aware_datetime(expected_unit["requested_end_utc"])
    expected_asof = pd.Timestamp(expected_unit["asof_date"]).date()
    symbol_rows = manifest.get("symbol_rows")
    if (
        manifest.get("schema") != SPECIALIST_ONE_MINUTE_UNIT_SCHEMA
        or manifest.get("unit_id") != expected_unit_id
        or manifest.get("unit_bundle_fingerprint")
        != expected_unit_bundle_fingerprint
        or manifest.get("request_sha256") != expected_request_sha256
        or manifest.get("requested_start_utc") != expected_start.isoformat()
        or manifest.get("requested_end_utc") != expected_end.isoformat()
        or manifest.get("asof_date") != expected_asof.isoformat()
        or manifest.get("timeframe") != "1m"
        or manifest.get("price_feed") != "sip"
        or manifest.get("adjustment") != "all"
        or not isinstance(symbol_rows, Mapping)
        or set(str(symbol) for symbol in symbol_rows) != expected_symbols
        or manifest.get("sha256") != file_sha256(path)
    ):
        raise DataReadinessError(
            f"KS4 collected unit integrity failed: {path}"
        )
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
    if len(frame) != int(manifest.get("rows", -1)):
        raise DataReadinessError(
            f"KS4 collected unit row count failed: {path}"
        )
    if not frame.empty and (
        bool(frame["price_feed"].ne("sip").any())
        or bool(frame["adjustment"].ne("all").any())
        or bool(frame["source"].ne("alpaca").any())
        or bool(frame["timeframe"].ne("1m").any())
        or not set(frame["ticker"].astype(str)).issubset(expected_symbols)
        or bool(frame["bar_start_utc"].lt(expected_start).any())
        or bool(frame["bar_start_utc"].ge(expected_end).any())
        or bool(frame.duplicated(["ticker", "bar_start_utc"]).any())
    ):
        raise DataReadinessError(
            f"KS4 collected unit content failed: {path}"
        )
    return manifest


def _json_string_list(value: object, *, name: str) -> tuple[str, ...]:
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SchemaMismatchError(f"invalid KS4 {name}") from exc
    if (
        not isinstance(loaded, list)
        or not loaded
        or any(not isinstance(item, str) or not item for item in loaded)
    ):
        raise SchemaMismatchError(f"invalid KS4 {name}")
    return tuple(loaded)


def _json_string_mapping(value: object, *, name: str) -> dict[str, str]:
    try:
        loaded = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise SchemaMismatchError(f"invalid KS4 {name}") from exc
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
        raise SchemaMismatchError(f"invalid KS4 {name}")
    return {str(key): str(item) for key, item in loaded.items()}


def _aware_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError("KS4 unit bound is timezone-naive")
    return cast(
        datetime,
        timestamp.tz_convert("UTC").to_pydatetime(),
    )


def _page_response_sha256(page: AlpacaBarsPage) -> str:
    payload = {
        "request_page_token": page.request_page_token,
        "next_page_token": page.next_page_token,
        "bars": page.bars,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != dict(payload):
            raise DataReadinessError(
                f"KS4 resume identity differs: {path}"
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


def _atomic_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        temporary.write_bytes(
            gzip.compress(serialized, compresslevel=6, mtime=0)
        )
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
