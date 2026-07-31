"""Exact, resumable Alpaca daily collection for swing acquisition-plan v2."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final, Protocol, cast
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.swing_history_acquisition import (
    AUTHORITY_SCHEMA as PLAN_AUTHORITY_SCHEMA,
)
from market_predictor.edge_rebuild.swing_history_acquisition import (
    DAILY_BAR_UNITS_FILE,
    PLAN_SCHEMA,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.v3.errors import DataReadinessError

COLLECTION_SCHEMA: Final = "edge_rebuild.swing_history_collection.v1"
COLLECTION_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_history_collection_authority.v1"
UNIT_SCHEMA: Final = "edge_rebuild.swing_history_collection_unit.v1"
TIMEFRAME: Final = "1Day"
PRICE_FEED: Final = "sip"
ADJUSTMENT: Final = "all"
MAXIMUM_WORKERS: Final = 2
MAXIMUM_MEMORY_GIB: Final = 4.0
MEMORY_HEADROOM_GIB: Final = 0.75
MAXIMUM_PAGES_PER_UNIT: Final = 100
MAXIMUM_UNAVAILABLE_SECURITY_FRACTION: Final = 0.05
EASTERN: Final = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class SwingDailyPage:
    request_page_token: str | None
    next_page_token: str | None
    response_symbol: str
    response_timeframe: str
    response_feed: str
    response_adjustment: str
    bars: tuple[dict[str, Any], ...]
    response_headers: dict[str, str]
    raw_payload: dict[str, Any] | None = None


class SwingDailyPageSource(Protocol):
    def fetch_daily_page(
        self,
        symbol: str,
        start: datetime,
        end_exclusive: datetime,
        *,
        page_token: str | None,
        asof: date,
    ) -> SwingDailyPage: ...


SourceFactory = Callable[[], SwingDailyPageSource]
ProviderSymbol = Callable[[str], str]


class AlpacaSwingDailyPageSource:
    """Adapt Alpaca's paged response to the collector's explicit contract."""

    def __init__(self, source: AlpacaSource) -> None:
        if source.settings.alpaca_stock_feed.strip().lower() != PRICE_FEED:
            raise DataReadinessError("swing history collection requires Alpaca SIP")
        self._source = source

    def fetch_daily_page(
        self,
        symbol: str,
        start: datetime,
        end_exclusive: datetime,
        *,
        page_token: str | None,
        asof: date,
    ) -> SwingDailyPage:
        page = self._source.fetch_bars_page(
            (symbol,),
            start,
            end_exclusive - timedelta(microseconds=1),
            timeframe=TIMEFRAME,
            page_token=page_token,
            asof=asof,
            limit=10_000,
            retries=5,
        )
        returned = tuple(page.bars)
        response_symbol = returned[0] if returned else symbol
        return SwingDailyPage(
            request_page_token=page.request_page_token,
            next_page_token=page.next_page_token,
            response_symbol=response_symbol,
            response_timeframe=TIMEFRAME,
            response_feed=PRICE_FEED,
            response_adjustment=ADJUSTMENT,
            bars=page.bars.get(symbol, ()),
            response_headers=page.response_headers,
            raw_payload=page.raw_payload,
        )


@dataclass(frozen=True, slots=True)
class _VerifiedPlan:
    units: pd.DataFrame
    hashes: dict[str, str]
    universe_sha256: str


def collect_swing_history_plan(
    *,
    plan_directory: Path,
    output_directory: Path,
    source_factory: SourceFactory,
    provider_symbol_for: ProviderSymbol,
    maximum_units_this_run: int | None = None,
) -> dict[str, Any]:
    """Collect every exact v2 plan unit with immutable per-unit resume."""

    if maximum_units_this_run is not None and maximum_units_this_run < 1:
        raise ValueError("maximum_units_this_run must be positive")
    plan = _load_verified_plan(plan_directory)
    units = _bind_provider_symbols(plan.units, provider_symbol_for)
    if (output_directory / "_authority.json").exists():
        raise DataReadinessError("completed swing history collection is immutable")
    if (output_directory / "_manifest.json").exists():
        raise DataReadinessError("swing history collection has an orphan final manifest")
    _guard("swing history collection start")

    request_payload: dict[str, Any] = {
        "schema": COLLECTION_SCHEMA,
        "plan_directory": str(plan_directory.resolve()),
        "plan_schema": PLAN_SCHEMA,
        "plan_hashes": plan.hashes,
        "universe_sha256": plan.universe_sha256,
        "provider_unit_set_sha256": _provider_unit_set_sha256(units),
        "provider_symbols": {str(ticker): str(group.iloc[0]["provider_symbol"]) for ticker, group in units.groupby("ticker", sort=True)},
        "provider": "alpaca",
        "timeframe": TIMEFRAME,
        "price_feed": PRICE_FEED,
        "adjustment": ADJUSTMENT,
        "workers": MAXIMUM_WORKERS,
        "maximum_memory_gib": MAXIMUM_MEMORY_GIB,
    }
    request_sha256 = _json_sha256(request_payload)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(
        output_directory / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )

    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for raw in units.to_dict(orient="records"):
        unit = {str(key): value for key, value in raw.items()}
        existing = _load_existing_unit(
            output_directory,
            unit,
            request_sha256=request_sha256,
        )
        if existing is None:
            pending.append(unit)
        else:
            completed[str(unit["unit_id"])] = existing

    scheduled = pending if maximum_units_this_run is None else pending[:maximum_units_this_run]
    local = threading.local()

    def get_source() -> SwingDailyPageSource:
        source = getattr(local, "swing_daily_source", None)
        if source is None:
            source = source_factory()
            local.swing_daily_source = source
        return cast(SwingDailyPageSource, source)

    def collect_unit(unit: Mapping[str, Any]) -> dict[str, Any]:
        return _collect_unit(
            output_directory,
            unit,
            source=get_source(),
            request_sha256=request_sha256,
        )

    failures = _run_bounded(
        units=scheduled,
        collect_unit=collect_unit,
        completed=completed,
    )
    terminal_ids = set(completed)
    failed_ids = set(failures)
    unattempted = [
        _unit_identity_record(unit) for unit in units.to_dict(orient="records") if str(unit["unit_id"]) not in terminal_ids | failed_ids
    ]
    unavailable_records = [record for record in completed.values() if record["status"] == "unavailable"]
    stock_security_ids = set(units.loc[units["role"].eq("stock"), "security_id"].astype(str))
    unavailable_security_ids = {str(record["security_id"]) for record in unavailable_records if record["role"] == "stock"}
    unavailable_fraction = len(unavailable_security_ids) / len(stock_security_ids) if stock_security_ids else 0.0
    unavailable_within_policy = unavailable_fraction <= MAXIMUM_UNAVAILABLE_SECURITY_FRACTION
    unavailable = sorted(
        (
            _unit_status_record(
                record,
                allowed=bool(record["unavailable_allowed"]) and unavailable_within_policy,
            )
            for record in unavailable_records
        ),
        key=lambda item: item["unit_id"],
    )
    non_allowed_unavailable = [record for record in unavailable if not bool(record["allowed"])]
    complete = not failures and not unattempted and not non_allowed_unavailable
    collection_status = "complete_with_unavailable" if complete and unavailable else ("complete" if complete else "incomplete")
    status: dict[str, Any] = {
        "schema": COLLECTION_SCHEMA,
        "status": collection_status,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha256,
        "plan_hashes": plan.hashes,
        "universe_sha256": plan.universe_sha256,
        "requested_units": len(units),
        "terminal_units": len(completed),
        "observed_units": sum(record["status"] == "observed" for record in completed.values()),
        "unavailable_units": unavailable,
        "unavailable_security_count": len(unavailable_security_ids),
        "unavailable_security_fraction": unavailable_fraction,
        "maximum_unavailable_security_fraction": MAXIMUM_UNAVAILABLE_SECURITY_FRACTION,
        "failed_units": [failures[key] for key in sorted(failures)],
        "unattempted_units": unattempted,
        "resumed_units": len(units) - len(pending),
        "maximum_units_this_run": maximum_units_this_run,
        "stop_reason": (
            "complete" if complete else ("non_allowed_failure" if failures or non_allowed_unavailable else "operational_batch_limit")
        ),
        "memory": memory_audit(
            hard_budget_gib=MAXIMUM_MEMORY_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
        ).to_record(),
    }
    _atomic_json(output_directory / "_status.json", status)
    if not complete:
        return status

    artifacts = [completed[unit_id] for unit_id in sorted(completed)]
    unit_set_sha256 = _unit_artifact_set_sha256(artifacts)
    manifest: dict[str, Any] = {
        **status,
        "unit_artifacts": artifacts,
        "unit_set_sha256": unit_set_sha256,
        "total_rows": sum(int(record["rows"]) for record in artifacts),
    }
    _guard_peak("swing history collection publication")
    _atomic_json(output_directory / "_manifest.json", manifest)
    _atomic_json(
        output_directory / "_authority.json",
        {
            "schema": COLLECTION_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(output_directory / "_manifest.json"),
            "request_sha256": request_sha256,
            "plan_authority_sha256": plan.hashes["authority_sha256"],
            "plan_units_sha256": plan.hashes["units_sha256"],
            "universe_sha256": plan.universe_sha256,
            "unit_set_sha256": unit_set_sha256,
        },
    )
    try:
        return load_complete_swing_history_collection(
            output_directory,
            plan_directory=plan_directory,
        )
    except Exception:
        (output_directory / "_authority.json").unlink(missing_ok=True)
        (output_directory / "_manifest.json").unlink(missing_ok=True)
        _atomic_json(
            output_directory / "_status.json",
            {
                **status,
                "status": "incomplete",
                "stop_reason": "publication_verification_failure",
            },
        )
        raise


def load_complete_swing_history_collection(
    directory: Path,
    *,
    plan_directory: Path,
) -> dict[str, Any]:
    """Verify final authority, current plan identity, and every unit artifact."""

    plan = _load_verified_plan(plan_directory)
    request = _load_json(directory / "_request.json")
    manifest = _load_json(directory / "_manifest.json")
    authority = _load_json(directory / "_authority.json")
    request_sha256 = str(request.get("request_sha256", ""))
    request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
    if (
        _json_sha256(request_payload) != request_sha256
        or request.get("schema") != COLLECTION_SCHEMA
        or request.get("plan_hashes") != plan.hashes
        or request.get("universe_sha256") != plan.universe_sha256
        or request.get("provider") != "alpaca"
        or request.get("timeframe") != TIMEFRAME
        or request.get("price_feed") != PRICE_FEED
        or request.get("adjustment") != ADJUSTMENT
        or int(request.get("workers", -1)) != MAXIMUM_WORKERS
        or float(request.get("maximum_memory_gib", -1.0)) != MAXIMUM_MEMORY_GIB
        or manifest.get("schema") != COLLECTION_SCHEMA
        or manifest.get("status") not in {"complete", "complete_with_unavailable"}
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("plan_hashes") != plan.hashes
        or manifest.get("universe_sha256") != plan.universe_sha256
        or manifest.get("failed_units") != []
        or manifest.get("unattempted_units") != []
        or float(manifest.get("unavailable_security_fraction", 1.0)) > MAXIMUM_UNAVAILABLE_SECURITY_FRACTION
        or authority.get("schema") != COLLECTION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or authority.get("plan_authority_sha256") != plan.hashes["authority_sha256"]
        or authority.get("plan_units_sha256") != plan.hashes["units_sha256"]
        or authority.get("universe_sha256") != plan.universe_sha256
    ):
        raise DataReadinessError("swing history collection lacks valid complete authority")
    raw_artifacts = manifest.get("unit_artifacts")
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(plan.units):
        raise DataReadinessError("swing history collection unit inventory is incomplete")
    expected_units = {
        str(row["unit_id"]): {str(key): value for key, value in row.items()}
        for row in _bind_provider_symbols(plan.units, lambda ticker: _provider_from_request(request, ticker)).to_dict(orient="records")
    }
    expected_frame = pd.DataFrame(expected_units.values())
    if request.get("provider_unit_set_sha256") != _provider_unit_set_sha256(expected_frame):
        raise DataReadinessError("swing history collection provider-unit identity is invalid")
    verified: list[dict[str, Any]] = []
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("swing history collection unit record is malformed")
        unit_id = str(raw.get("unit_id", ""))
        expected = expected_units.get(unit_id)
        if expected is None:
            raise DataReadinessError(f"unexpected swing history unit: {unit_id}")
        actual = _load_existing_unit(directory, expected, request_sha256=request_sha256)
        if actual is None or actual != dict(raw):
            raise DataReadinessError(f"swing history unit does not verify: {unit_id}")
        if actual["status"] == "unavailable" and not bool(actual["unavailable_allowed"]):
            raise DataReadinessError(f"non-allowed unavailable unit was authorized: {unit_id}")
        verified.append(actual)
    unit_set_sha256 = _unit_artifact_set_sha256(verified)
    unavailable_records = [record for record in verified if record["status"] == "unavailable"]
    stock_security_ids = {str(unit["security_id"]) for unit in expected_units.values() if unit["role"] == "stock"}
    unavailable_security_ids = {str(record["security_id"]) for record in unavailable_records if record["role"] == "stock"}
    unavailable_fraction = len(unavailable_security_ids) / len(stock_security_ids) if stock_security_ids else 0.0
    expected_unavailable = sorted(
        (_unit_status_record(record, allowed=True) for record in unavailable_records),
        key=lambda item: item["unit_id"],
    )
    expected_status = "complete_with_unavailable" if expected_unavailable else "complete"
    if (
        manifest.get("unit_set_sha256") != unit_set_sha256
        or authority.get("unit_set_sha256") != unit_set_sha256
        or manifest.get("status") != expected_status
        or int(manifest.get("requested_units", -1)) != len(expected_units)
        or int(manifest.get("terminal_units", -1)) != len(verified)
        or int(manifest.get("observed_units", -1)) != len(verified) - len(unavailable_records)
        or manifest.get("unavailable_units") != expected_unavailable
        or int(manifest.get("unavailable_security_count", -1)) != len(unavailable_security_ids)
        or float(manifest.get("unavailable_security_fraction", -1.0)) != unavailable_fraction
        or int(manifest.get("total_rows", -1)) != sum(int(record["rows"]) for record in verified)
    ):
        raise DataReadinessError("swing history collection semantic replay failed")
    return manifest


def _load_verified_plan(directory: Path) -> _VerifiedPlan:
    request_path = directory / "_request.json"
    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    for path in (request_path, manifest_path, authority_path):
        if not path.is_file():
            raise DataReadinessError(f"swing history plan file is missing: {path}")
    request = _load_json(request_path)
    manifest = _load_json(manifest_path)
    authority = _load_json(authority_path)
    units_record = cast(object, manifest.get("daily_bars"))
    if not isinstance(units_record, Mapping):
        raise DataReadinessError("swing history plan daily-bar inventory is invalid")
    artifact = units_record.get("units_artifact")
    membership = manifest.get("membership")
    request_membership = request.get("membership_authority")
    if not isinstance(artifact, Mapping) or not isinstance(membership, Mapping) or not isinstance(request_membership, Mapping):
        raise DataReadinessError("swing history plan authority inventory is invalid")
    units_path = _resolve_inside(directory, str(artifact.get("path", "")))
    if not units_path.is_file():
        raise DataReadinessError(f"swing history plan units are missing: {units_path}")
    request_sha256 = file_sha256(request_path)
    units_sha256 = file_sha256(units_path)
    universe_sha256 = str(membership.get("universe_sha256", ""))
    if (
        request.get("schema") != PLAN_SCHEMA
        or manifest.get("schema") != PLAN_SCHEMA
        or manifest.get("status") != "ready_for_daily_history_collection"
        or manifest.get("outcomes_read") is not False
        or manifest.get("request_sha256") != request_sha256
        or units_record.get("status") != "ready"
        or units_record.get("source") != "alpaca"
        or units_record.get("timeframe") != TIMEFRAME
        or units_record.get("price_feed") != PRICE_FEED
        or units_record.get("adjustment") != ADJUSTMENT
        or artifact.get("path") != DAILY_BAR_UNITS_FILE
        or not units_path.is_file()
        or int(artifact.get("bytes", -1)) != units_path.stat().st_size
        or artifact.get("sha256") != units_sha256
        or authority.get("schema") != PLAN_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or authority.get("units_sha256") != units_sha256
        or authority.get("universe_sha256") != universe_sha256
        or request_membership.get("universe_sha256") != universe_sha256
        or request_membership.get("parent_lineage") != membership.get("parent_lineage")
        or len(universe_sha256) != 64
    ):
        raise DataReadinessError("swing history acquisition plan authority is invalid")
    units = _load_plan_units(units_path)
    _validate_plan_unit_coverage(units, manifest=manifest, daily_bars=units_record)
    return _VerifiedPlan(
        units=units,
        hashes={
            "request_sha256": request_sha256,
            "manifest_sha256": file_sha256(manifest_path),
            "authority_sha256": file_sha256(authority_path),
            "units_sha256": units_sha256,
        },
        universe_sha256=universe_sha256,
    )


def _load_plan_units(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    columns = ["security_id", "ticker", "start_date", "end_date", "role"]
    if list(frame.columns) != columns or frame.empty:
        raise DataReadinessError("swing history plan unit schema is invalid")
    if bool(frame.duplicated(columns).any()):
        raise DataReadinessError("swing history plan has duplicate units")
    records: list[dict[str, str]] = []
    for raw in frame.to_dict(orient="records"):
        record = {str(key): str(value).strip() for key, value in raw.items()}
        if any(not record[column] for column in columns):
            raise DataReadinessError("swing history plan unit contains empty identity fields")
        if record["ticker"] != record["ticker"].upper() or record["role"] not in {"stock", "benchmark"}:
            raise DataReadinessError("swing history plan unit ticker or role is invalid")
        try:
            start = date.fromisoformat(record["start_date"])
            end = date.fromisoformat(record["end_date"])
        except ValueError as exc:
            raise DataReadinessError("swing history plan unit date is invalid") from exc
        if start > end:
            raise DataReadinessError("swing history plan unit date range is reversed")
        payload = {column: record[column] for column in columns}
        unit_sha256 = _json_sha256(payload)
        records.append(
            {
                **payload,
                "unit_id": f"swing-daily-{unit_sha256[:24]}",
                "plan_unit_sha256": unit_sha256,
            }
        )
    units = pd.DataFrame(records)
    if bool(units["unit_id"].duplicated().any()):
        raise DataReadinessError("swing history plan unit identities collide")
    return units.sort_values(["role", "ticker", "start_date", "security_id"], kind="stable").reset_index(drop=True)


def _validate_plan_unit_coverage(
    units: pd.DataFrame,
    *,
    manifest: Mapping[str, Any],
    daily_bars: Mapping[str, Any],
) -> None:
    if (
        len(units) != int(daily_bars.get("planned_units", -1))
        or int(units["role"].eq("stock").sum()) != int(daily_bars.get("stock_units", -1))
        or int(units["role"].eq("benchmark").sum()) != int(daily_bars.get("benchmark_units", -1))
    ):
        raise DataReadinessError("swing history plan unit counts are invalid")
    raw_ranges = manifest.get("missing_session_ranges")
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise DataReadinessError("swing history plan has no missing-session ranges")
    ranges: list[tuple[date, date]] = []
    for raw in raw_ranges:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("swing history plan missing-session range is invalid")
        try:
            ranges.append(
                (
                    date.fromisoformat(str(raw["first_session"])),
                    date.fromisoformat(str(raw["last_session"])),
                )
            )
        except (KeyError, ValueError) as exc:
            raise DataReadinessError("swing history plan missing-session range is invalid") from exc
    for row in units.to_dict(orient="records"):
        start = date.fromisoformat(str(row["start_date"]))
        end = date.fromisoformat(str(row["end_date"]))
        if not any(range_start <= start <= end <= range_end for range_start, range_end in ranges):
            raise DataReadinessError("swing history unit escapes every missing-session range")
    benchmark_units = units[units["role"].eq("benchmark")]
    benchmark_tickers = set(benchmark_units["ticker"].astype(str))
    if not {"SPY", "QQQ"}.issubset(benchmark_tickers):
        raise DataReadinessError("swing history plan lacks SPY or QQQ benchmark coverage")
    expected = {
        (ticker, range_start.isoformat(), range_end.isoformat()) for ticker in benchmark_tickers for range_start, range_end in ranges
    }
    actual = {(str(row["ticker"]), str(row["start_date"]), str(row["end_date"])) for row in benchmark_units.to_dict(orient="records")}
    if actual != expected:
        raise DataReadinessError("swing history benchmark units do not cover every missing range exactly")


def _bind_provider_symbols(units: pd.DataFrame, mapper: ProviderSymbol) -> pd.DataFrame:
    bound = units.copy()
    bound["provider_symbol"] = bound["ticker"].map(lambda ticker: mapper(str(ticker)).strip().upper())
    if bool(bound["provider_symbol"].eq("").any()):
        raise DataReadinessError("swing history provider symbol mapping is empty")
    if bool(bound.groupby("ticker")["provider_symbol"].nunique().gt(1).any()):
        raise DataReadinessError("swing history provider symbol mapping is unstable")
    return bound


def _provider_from_request(request: Mapping[str, Any], ticker: str) -> str:
    mappings = request.get("provider_symbols")
    if not isinstance(mappings, Mapping):
        raise DataReadinessError("swing history request lacks provider-symbol mappings")
    value = str(mappings.get(ticker, "")).strip().upper()
    if not value:
        raise DataReadinessError(f"swing history request lacks provider symbol for {ticker}")
    return value


def _provider_unit_set_sha256(units: pd.DataFrame) -> str:
    return _json_sha256(
        [
            {
                "unit_id": str(row["unit_id"]),
                "plan_unit_sha256": str(row["plan_unit_sha256"]),
                "provider_symbol": str(row["provider_symbol"]),
            }
            for row in units.to_dict(orient="records")
        ]
    )


def _collect_unit(
    root: Path,
    unit: Mapping[str, Any],
    *,
    source: SwingDailyPageSource,
    request_sha256: str,
) -> dict[str, Any]:
    unit_id = str(unit["unit_id"])
    unit_directory = root / "units" / unit_id
    started_at = datetime.now(UTC)
    attempt_id = uuid.uuid4().hex
    attempt_path = unit_directory / "attempts" / f"{attempt_id}.json"
    raw_directory = unit_directory / "raw" / attempt_id
    start_date = date.fromisoformat(str(unit["start_date"]))
    end_date = date.fromisoformat(str(unit["end_date"]))
    start = _session_midnight_utc(start_date)
    end_exclusive = _session_midnight_utc(end_date + timedelta(days=1))
    maximum_expected_rows = (end_date - start_date).days + 1
    provider_symbol = str(unit["provider_symbol"])
    page_token: str | None = None
    seen_tokens: set[str] = set()
    pages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    try:
        while True:
            page = source.fetch_daily_page(
                provider_symbol,
                start,
                end_exclusive,
                page_token=page_token,
                asof=end_date,
            )
            page_number = len(pages) + 1
            raw_path = raw_directory / f"page-{page_number:05d}.json.gz"
            _atomic_gzip_json(
                raw_path,
                page.raw_payload
                or {
                    "bars": {page.response_symbol: list(page.bars)},
                    "next_page_token": page.next_page_token,
                },
            )
            pages.append(
                {
                    "page_number": page_number,
                    "request_page_token": page.request_page_token,
                    "next_page_token": page.next_page_token,
                    "response_symbol": page.response_symbol,
                    "response_timeframe": page.response_timeframe,
                    "response_feed": page.response_feed,
                    "response_adjustment": page.response_adjustment,
                    "rows": len(page.bars),
                    "response_sha256": _page_sha256(page),
                    "raw_path": str(raw_path.relative_to(root)),
                    "raw_sha256": file_sha256(raw_path),
                    "raw_bytes": raw_path.stat().st_size,
                }
            )
            _validate_page_contract(page, provider_symbol=provider_symbol, page_token=page_token)
            rows.extend(page.bars)
            if len(rows) > maximum_expected_rows:
                raise DataReadinessError(f"Alpaca daily unit exceeds one row per calendar date: {unit_id}")
            next_token = page.next_page_token
            if next_token is None:
                break
            if next_token in seen_tokens or len(pages) >= MAXIMUM_PAGES_PER_UNIT:
                raise DataReadinessError(f"invalid Alpaca pagination for {unit_id}")
            seen_tokens.add(next_token)
            page_token = next_token
        bars = _validated_bars(rows, unit=unit, ingested_at=datetime.now(UTC))
        status = "unavailable" if bars.empty else "observed"
        unavailable_allowed = status != "unavailable" or str(unit["role"]) == "stock"
        if status == "unavailable" and not unavailable_allowed:
            raise DataReadinessError(f"benchmark unit is unavailable: {unit_id}")
        bars_path = unit_directory / "bars.parquet"
        _atomic_parquet(bars, bars_path)
        unit_manifest: dict[str, Any] = {
            "schema": UNIT_SCHEMA,
            "unit_id": unit_id,
            "plan_unit_sha256": str(unit["plan_unit_sha256"]),
            "request_sha256": request_sha256,
            "security_id": str(unit["security_id"]),
            "ticker": str(unit["ticker"]),
            "provider_symbol": provider_symbol,
            "role": str(unit["role"]),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "status": status,
            "unavailable_allowed": unavailable_allowed,
            "timeframe": TIMEFRAME,
            "price_feed": PRICE_FEED,
            "adjustment": ADJUSTMENT,
            "bars_path": str(bars_path.relative_to(root)),
            "bars_sha256": file_sha256(bars_path),
            "bars_bytes": bars_path.stat().st_size,
            "rows": len(bars),
            "pages": pages,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = unit_directory / "_manifest.json"
        _atomic_json(manifest_path, unit_manifest)
        _atomic_json(
            attempt_path,
            {
                "attempt_id": attempt_id,
                "unit_id": unit_id,
                "status": status,
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "rows": len(bars),
            },
        )
        result = _load_existing_unit(root, unit, request_sha256=request_sha256)
        if result is None:
            raise DataReadinessError(f"completed swing unit did not verify: {unit_id}")
        return result
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


def _validate_page_contract(
    page: SwingDailyPage,
    *,
    provider_symbol: str,
    page_token: str | None,
) -> None:
    if (
        page.request_page_token != page_token
        or page.response_symbol != provider_symbol
        or page.response_timeframe != TIMEFRAME
        or page.response_feed.lower() != PRICE_FEED
        or page.response_adjustment.lower() != ADJUSTMENT
    ):
        raise DataReadinessError("Alpaca daily response contract differs from the exact unit request")


def _validated_bars(
    rows: list[dict[str, Any]],
    *,
    unit: Mapping[str, Any],
    ingested_at: datetime,
) -> pd.DataFrame:
    columns = [
        "security_id",
        "ticker",
        "role",
        "bar_start_utc",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "timeframe",
        "price_feed",
        "adjustment",
        "ingested_at_utc",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    start = date.fromisoformat(str(unit["start_date"]))
    end = date.fromisoformat(str(unit["end_date"]))
    records: list[dict[str, Any]] = []
    required = {"t", "o", "h", "l", "c", "v"}
    for row in rows:
        if not required.issubset(row):
            raise DataReadinessError("Alpaca daily bar is missing OHLCV or timestamp fields")
        try:
            timestamp = pd.Timestamp(row["t"])
        except (TypeError, ValueError) as exc:
            raise DataReadinessError("Alpaca daily bar timestamp is invalid") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise DataReadinessError("Alpaca daily bar timestamp must be timezone-aware")
        timestamp = timestamp.tz_convert("UTC")
        eastern = timestamp.tz_convert(EASTERN)
        session = date(eastern.year, eastern.month, eastern.day)
        if eastern.hour != 0 or eastern.minute != 0 or eastern.second != 0 or eastern.microsecond != 0 or session < start or session > end:
            raise DataReadinessError("Alpaca daily bar timestamp or session date is outside the exact unit")
        try:
            open_price = float(row["o"])
            high = float(row["h"])
            low = float(row["l"])
            close = float(row["c"])
            volume = float(row["v"])
        except (TypeError, ValueError) as exc:
            raise DataReadinessError("Alpaca daily bar OHLCV values are invalid") from exc
        values = (open_price, high, low, close, volume)
        if (
            not all(math.isfinite(value) for value in values)
            or min(open_price, high, low, close) <= 0
            or volume < 0
            or not volume.is_integer()
            or high < max(open_price, close, low)
            or low > min(open_price, close, high)
        ):
            raise DataReadinessError("Alpaca daily bar violates OHLCV invariants")
        records.append(
            {
                "security_id": str(unit["security_id"]),
                "ticker": str(unit["ticker"]),
                "role": str(unit["role"]),
                "bar_start_utc": timestamp,
                "session_date": session.isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": int(volume),
                "source": "alpaca",
                "timeframe": TIMEFRAME,
                "price_feed": PRICE_FEED,
                "adjustment": ADJUSTMENT,
                "ingested_at_utc": ingested_at,
            }
        )
    frame = pd.DataFrame(records, columns=columns).sort_values("bar_start_utc", kind="stable")
    if bool(frame["bar_start_utc"].duplicated().any()) or bool(frame["session_date"].duplicated().any()):
        raise DataReadinessError("Alpaca daily response contains duplicate sessions")
    return frame.reset_index(drop=True)


def _load_existing_unit(
    root: Path,
    unit: Mapping[str, Any],
    *,
    request_sha256: str,
) -> dict[str, Any] | None:
    unit_id = str(unit["unit_id"])
    unit_directory = root / "units" / unit_id
    manifest_path = unit_directory / "_manifest.json"
    bars_path = unit_directory / "bars.parquet"
    if not manifest_path.exists() and not bars_path.exists():
        return None
    if not manifest_path.is_file() or not bars_path.is_file():
        raise DataReadinessError(f"swing history unit has orphan final artifacts: {unit_id}")
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != UNIT_SCHEMA
        or manifest.get("unit_id") != unit_id
        or manifest.get("plan_unit_sha256") != unit["plan_unit_sha256"]
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("security_id") != unit["security_id"]
        or manifest.get("ticker") != unit["ticker"]
        or manifest.get("provider_symbol") != unit["provider_symbol"]
        or manifest.get("role") != unit["role"]
        or manifest.get("start_date") != unit["start_date"]
        or manifest.get("end_date") != unit["end_date"]
        or manifest.get("status") not in {"observed", "unavailable"}
        or manifest.get("timeframe") != TIMEFRAME
        or manifest.get("price_feed") != PRICE_FEED
        or manifest.get("adjustment") != ADJUSTMENT
        or manifest.get("bars_path") != str(bars_path.relative_to(root))
        or manifest.get("bars_sha256") != file_sha256(bars_path)
        or int(manifest.get("bars_bytes", -1)) != bars_path.stat().st_size
    ):
        raise DataReadinessError(f"swing history unit resume identity differs: {unit_id}")
    _verify_pages(root, manifest)
    frame = pd.read_parquet(bars_path)
    if len(frame) != int(manifest.get("rows", -1)):
        raise DataReadinessError(f"swing history unit row count differs: {unit_id}")
    if frame.empty:
        if manifest.get("status") != "unavailable":
            raise DataReadinessError(f"empty swing history unit is not unavailable: {unit_id}")
    else:
        expected = _validated_bars(
            [
                {
                    "t": row["bar_start_utc"],
                    "o": row["open"],
                    "h": row["high"],
                    "l": row["low"],
                    "c": row["close"],
                    "v": row["volume"],
                }
                for row in frame.to_dict(orient="records")
            ],
            unit=unit,
            ingested_at=datetime.now(UTC),
        )
        if len(expected) != len(frame) or manifest.get("status") != "observed":
            raise DataReadinessError(f"swing history unit content differs: {unit_id}")
        for column, expected_value in (
            ("security_id", str(unit["security_id"])),
            ("ticker", str(unit["ticker"])),
            ("role", str(unit["role"])),
            ("source", "alpaca"),
            ("timeframe", TIMEFRAME),
            ("price_feed", PRICE_FEED),
            ("adjustment", ADJUSTMENT),
        ):
            if bool(frame[column].astype(str).ne(expected_value).any()):
                raise DataReadinessError(f"swing history unit {column} differs: {unit_id}")
    unavailable_allowed = manifest.get("status") != "unavailable" or unit["role"] == "stock"
    if bool(manifest.get("unavailable_allowed")) != unavailable_allowed:
        raise DataReadinessError(f"swing history unavailable policy differs: {unit_id}")
    return {
        "unit_id": unit_id,
        "plan_unit_sha256": str(unit["plan_unit_sha256"]),
        "security_id": str(unit["security_id"]),
        "ticker": str(unit["ticker"]),
        "provider_symbol": str(unit["provider_symbol"]),
        "role": str(unit["role"]),
        "start_date": str(unit["start_date"]),
        "end_date": str(unit["end_date"]),
        "status": str(manifest["status"]),
        "unavailable_allowed": unavailable_allowed,
        "rows": int(manifest["rows"]),
        "bars_path": str(manifest["bars_path"]),
        "bars_sha256": str(manifest["bars_sha256"]),
        "unit_manifest_path": str(manifest_path.relative_to(root)),
        "unit_manifest_sha256": file_sha256(manifest_path),
    }


def _verify_pages(root: Path, manifest: Mapping[str, Any]) -> None:
    pages = manifest.get("pages")
    if not isinstance(pages, list) or not pages:
        raise DataReadinessError("swing history unit has no raw response pages")
    expected_page_token: str | None = None
    for page_number, raw in enumerate(pages, start=1):
        if not isinstance(raw, Mapping):
            raise DataReadinessError("swing history raw-page record is invalid")
        if (
            int(raw.get("page_number", -1)) != page_number
            or raw.get("request_page_token") != expected_page_token
            or raw.get("response_symbol") != manifest.get("provider_symbol")
            or raw.get("response_timeframe") != TIMEFRAME
            or raw.get("response_feed") != PRICE_FEED
            or raw.get("response_adjustment") != ADJUSTMENT
        ):
            raise DataReadinessError("swing history raw-page contract is invalid")
        path = _resolve_inside(root, str(raw.get("raw_path", "")))
        if not path.is_file() or int(raw.get("raw_bytes", -1)) != path.stat().st_size or raw.get("raw_sha256") != file_sha256(path):
            raise DataReadinessError(f"swing history raw page does not verify: {path}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as exc:
            raise DataReadinessError(f"swing history raw page is unreadable: {path}") from exc
        if not isinstance(payload, Mapping):
            raise DataReadinessError(f"swing history raw page is malformed: {path}")
        raw_bars = payload.get("bars")
        response_symbol = str(raw.get("response_symbol", ""))
        if not isinstance(raw_bars, Mapping) or set(str(symbol) for symbol in raw_bars) - {response_symbol}:
            raise DataReadinessError(f"swing history raw page symbol payload is invalid: {path}")
        response_rows = raw_bars.get(response_symbol, [])
        if (
            not isinstance(response_rows, list)
            or any(not isinstance(row, Mapping) for row in response_rows)
            or len(response_rows) != int(raw.get("rows", -1))
        ):
            raise DataReadinessError(f"swing history raw page row inventory is invalid: {path}")
        replay_sha256 = _json_sha256(
            {
                "request_page_token": raw.get("request_page_token"),
                "next_page_token": payload.get("next_page_token"),
                "response_symbol": response_symbol,
                "response_timeframe": raw.get("response_timeframe"),
                "response_feed": raw.get("response_feed"),
                "response_adjustment": raw.get("response_adjustment"),
                "bars": tuple({str(key): value for key, value in cast(Mapping[str, Any], row).items()} for row in response_rows),
            }
        )
        if replay_sha256 != raw.get("response_sha256"):
            raise DataReadinessError(f"swing history raw page response identity is invalid: {path}")
        if raw.get("next_page_token") != payload.get("next_page_token"):
            raise DataReadinessError(f"swing history raw page pagination identity is invalid: {path}")
        next_token = raw.get("next_page_token")
        expected_page_token = str(next_token) if next_token is not None else None
    if expected_page_token is not None:
        raise DataReadinessError("swing history raw-page sequence is incomplete")


def _run_bounded(
    *,
    units: list[dict[str, Any]],
    collect_unit: Callable[[Mapping[str, Any]], dict[str, Any]],
    completed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    failures: dict[str, dict[str, Any]] = {}
    rows = iter(units)
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    executor = ThreadPoolExecutor(max_workers=MAXIMUM_WORKERS)
    try:
        for _ in range(MAXIMUM_WORKERS):
            unit = next(rows, None)
            if unit is not None:
                futures[executor.submit(collect_unit, unit)] = unit
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                unit = futures.pop(future)
                unit_id = str(unit["unit_id"])
                try:
                    completed[unit_id] = future.result()
                except Exception as exc:
                    failures[unit_id] = {
                        **_unit_identity_record(unit),
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                    }
                release_process_memory()
                _guard(f"swing history unit {unit_id}")
                next_unit = next(rows, None)
                if next_unit is not None:
                    futures[executor.submit(collect_unit, next_unit)] = next_unit
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return failures


def _unit_identity_record(unit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": str(unit["unit_id"]),
        "security_id": str(unit["security_id"]),
        "ticker": str(unit["ticker"]),
        "role": str(unit["role"]),
        "start_date": str(unit["start_date"]),
        "end_date": str(unit["end_date"]),
    }


def _unit_status_record(
    record: Mapping[str, Any],
    *,
    allowed: bool,
) -> dict[str, Any]:
    return {
        **_unit_identity_record(record),
        "allowed": allowed,
        "reason": "provider_observed_empty",
    }


def _unit_artifact_set_sha256(records: list[dict[str, Any]]) -> str:
    return _json_sha256(
        [
            {
                "unit_id": str(record["unit_id"]),
                "unit_manifest_sha256": str(record["unit_manifest_sha256"]),
                "bars_sha256": str(record["bars_sha256"]),
                "status": str(record["status"]),
            }
            for record in sorted(records, key=lambda item: str(item["unit_id"]))
        ]
    )


def _page_sha256(page: SwingDailyPage) -> str:
    return _json_sha256(
        {
            "request_page_token": page.request_page_token,
            "next_page_token": page.next_page_token,
            "response_symbol": page.response_symbol,
            "response_timeframe": page.response_timeframe,
            "response_feed": page.response_feed,
            "response_adjustment": page.response_adjustment,
            "bars": page.bars,
        }
    )


def _session_midnight_utc(session: date) -> datetime:
    return datetime.combine(session, time.min, tzinfo=EASTERN).astimezone(UTC)


def _resolve_inside(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise DataReadinessError(f"swing history artifact escapes authority root: {relative}")
    return candidate


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != dict(payload):
            raise DataReadinessError(f"swing history collection request drifted: {path}")
        return
    _atomic_json(path, payload)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"swing history JSON is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise DataReadinessError(f"swing history JSON must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_gzip_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as stream:
            json.dump(dict(payload), stream, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _guard(stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )


def _guard_peak(stage: str) -> None:
    assert_peak_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )
