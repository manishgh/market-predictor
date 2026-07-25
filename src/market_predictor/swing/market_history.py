from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport, audit_canonical_bars
from market_predictor.canonical.contracts import SourceCollection
from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.resources import assert_memory_budget, memory_audit, release_process_memory
from market_predictor.v3.errors import DataReadinessError

SWING_DAILY_HISTORY_SCHEMA = "swing.daily_history_collection.v1"
SWING_DAILY_HISTORY_MANIFEST_SCHEMA = "swing.daily_history_manifest.v1"
DEFAULT_BENCHMARKS = (
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
)
DailyBarFetcher = Callable[[str, datetime, datetime], pd.DataFrame]


@dataclass(frozen=True, slots=True)
class DailyHistoryCollectionResult:
    status: str
    requested_symbols: int
    observed_symbols: int
    unavailable_symbols: tuple[str, ...]
    failed_symbols: tuple[str, ...]
    skipped_symbols: int
    manifest_path: Path | None
    status_path: Path


def collect_swing_daily_history(
    *,
    memberships_path: Path,
    start_date: date,
    end_date: date,
    out_dir: Path,
    fetcher: DailyBarFetcher,
    price_feed: str,
    workers: int = 4,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.5,
) -> DailyHistoryCollectionResult:
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    normalized_feed = price_feed.strip().lower()
    if normalized_feed != "sip":
        raise DataReadinessError("swing daily history collection requires explicit Alpaca SIP provenance")
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")
    if not memberships_path.exists():
        raise FileNotFoundError(memberships_path)

    memberships = _read_frame(memberships_path)
    required = {"ticker", "security_id", "effective_from_utc", "effective_to_utc", "primary_benchmark"}
    missing = sorted(required.difference(memberships.columns))
    if missing:
        raise DataReadinessError(f"point-in-time memberships are missing columns: {missing}")
    symbols = sorted(
        {
            *memberships["ticker"].dropna().astype(str).str.upper().str.strip(),
            *memberships["primary_benchmark"].dropna().astype(str).str.upper().str.strip(),
            *(symbol.strip().upper() for symbol in benchmarks),
        }
        - {""}
    )
    membership_hash = file_sha256(memberships_path)
    request = {
        "schema": SWING_DAILY_HISTORY_SCHEMA,
        "memberships_path": str(memberships_path.resolve()),
        "memberships_sha256": membership_hash,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "source": "alpaca",
        "price_feed": normalized_feed,
        "adjustment": "all",
        "timeframe": "1d",
        "symbols": symbols,
    }
    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = out_dir / "_manifest.json"
    if final_manifest_path.exists():
        raise DataReadinessError(f"completed daily history collection is immutable: {final_manifest_path}")
    _write_or_validate_request(out_dir / "_request.json", request, request_hash)

    assert_memory_budget(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="swing daily history collection start",
    )
    fetch_start = datetime.combine(start_date, time.min, tzinfo=UTC)
    fetch_end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=UTC)
    requested_end = datetime.combine(end_date, time.max, tzinfo=UTC)
    artifact_dir = out_dir / "bars" / "1d"
    attempts_dir = out_dir / "attempts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)

    observed: dict[str, dict[str, Any]] = {}
    unavailable: set[str] = set()
    failures: dict[str, str] = {}
    skipped = 0
    pending: list[str] = []
    latest_attempts = _latest_attempt_collections(attempts_dir)
    for symbol in symbols:
        previous = latest_attempts.get(symbol)
        if previous is not None and previous.status == "observed_empty":
            unavailable.add(symbol)
            skipped += 1
            continue
        existing = _load_existing_symbol(
            symbol=symbol,
            path=artifact_dir / f"{symbol}.parquet",
            request_hash=request_hash,
            start_date=start_date,
            end_date=end_date,
        )
        if existing is None:
            pending.append(symbol)
        else:
            observed[symbol] = existing
            skipped += 1

    def collect_symbol(symbol: str) -> tuple[str, dict[str, Any] | None, SourceCollection]:
        started_at = datetime.now(UTC)
        collection_id = f"alpaca-{symbol}-{uuid4().hex}"
        try:
            assert_memory_budget(
                hard_budget_gib=memory_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage=f"daily history fetch {symbol}",
            )
            raw = fetcher(symbol, fetch_start, fetch_end)
            completed_at = datetime.now(UTC)
            if raw.empty:
                collection = SourceCollection(
                    collection_id=collection_id,
                    ticker=symbol,
                    source_family="alpaca_daily_bars",
                    requested_start_utc=fetch_start,
                    requested_end_utc=requested_end,
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    status="observed_empty",
                    row_count=0,
                )
                return symbol, None, collection
            bars = canonicalize_bars(
                raw,
                timeframe="1d",
                ticker=symbol,
                source="alpaca",
                price_feed=normalized_feed,
                adjustment="all",
                ingested_at_utc=completed_at,
                availability_policy="market_interval_close",
            )
            session_dates = bars["bar_start_utc"].dt.tz_convert(ZoneInfo("America/New_York")).dt.date
            bars = bars.loc[session_dates.between(start_date, end_date)].reset_index(drop=True)
            audit = CanonicalAuditReport(checks=audit_canonical_bars(bars, require_sip=True))
            path = artifact_dir / f"{symbol}.parquet"
            manifest = write_canonical_artifact(
                bars,
                path,
                artifact_type="bars",
                audit=audit,
                inputs={
                    "collection_request_sha256": request_hash,
                    "memberships_sha256": membership_hash,
                },
                production_ready=True,
            )
            completed_at = datetime.now(UTC)
            collection = SourceCollection(
                collection_id=collection_id,
                ticker=symbol,
                source_family="alpaca_daily_bars",
                requested_start_utc=fetch_start,
                requested_end_utc=requested_end,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                status="observed",
                row_count=len(bars),
            )
            return symbol, _artifact_record(symbol, path, bars, manifest), collection
        except Exception as exc:
            completed_at = datetime.now(UTC)
            collection = SourceCollection(
                collection_id=collection_id,
                ticker=symbol,
                source_family="alpaca_daily_bars",
                requested_start_utc=fetch_start,
                requested_end_utc=requested_end,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                status="failed",
                row_count=0,
                error_type=type(exc).__name__,
            )
            return symbol, {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}, collection
        finally:
            release_process_memory()

    with ThreadPoolExecutor(max_workers=min(workers, len(pending) or 1)) as executor:
        futures = {executor.submit(collect_symbol, symbol): symbol for symbol in pending}
        for future in as_completed(futures):
            symbol, artifact, collection = future.result()
            _write_attempt(attempts_dir, collection, artifact)
            if artifact is not None and "error" not in artifact:
                observed[symbol] = artifact
            elif collection.status == "observed_empty":
                unavailable.add(symbol)
            else:
                failures[symbol] = (
                    str(artifact["error"])
                    if artifact is not None
                    else "DataReadinessError: Alpaca returned no daily bars"
                )
            assert_memory_budget(
                hard_budget_gib=memory_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage=f"daily history persist {symbol}",
            )

    ledger = _latest_collection_ledger(attempts_dir, symbols, observed)
    ledger_path = out_dir / "_source_collections.parquet"
    _atomic_parquet(ledger, ledger_path)
    memory = memory_audit(hard_budget_gib=memory_budget_gib, headroom_gib=memory_headroom_gib).to_record()
    terminal_symbols = len(observed) + len(unavailable)
    final_status = (
        "complete_with_gaps"
        if not failures and terminal_symbols == len(symbols) and unavailable
        else "complete"
        if not failures and terminal_symbols == len(symbols)
        else "incomplete"
    )
    status_payload: dict[str, Any] = {
        "schema": SWING_DAILY_HISTORY_MANIFEST_SCHEMA,
        "request_sha256": request_hash,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "status": final_status,
        "requested_symbols": len(symbols),
        "observed_symbols": len(observed),
        "unavailable_symbols": sorted(unavailable),
        "failed_symbols": failures,
        "skipped_symbols": skipped,
        "source_collections_path": str(ledger_path),
        "source_collections_sha256": file_sha256(ledger_path),
        "memory": memory,
    }
    status_path = out_dir / "_status.json"
    _atomic_json(status_path, status_payload)
    if final_status == "incomplete":
        return DailyHistoryCollectionResult(
            status="incomplete",
            requested_symbols=len(symbols),
            observed_symbols=len(observed),
            unavailable_symbols=tuple(sorted(unavailable)),
            failed_symbols=tuple(sorted(failures)),
            skipped_symbols=skipped,
            manifest_path=None,
            status_path=status_path,
        )

    artifacts = [observed[symbol] for symbol in symbols if symbol in observed]
    final_payload = {
        **status_payload,
        "status": final_status,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_rows": sum(int(record["rows"]) for record in artifacts),
    }
    _atomic_json(final_manifest_path, final_payload)
    return DailyHistoryCollectionResult(
        status=final_status,
        requested_symbols=len(symbols),
        observed_symbols=len(observed),
        unavailable_symbols=tuple(sorted(unavailable)),
        failed_symbols=(),
        skipped_symbols=skipped,
        manifest_path=final_manifest_path,
        status_path=status_path,
    )


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported membership format: {path.suffix}")


def _write_or_validate_request(path: Path, request: dict[str, Any], request_hash: str) -> None:
    payload = {**request, "request_sha256": request_hash}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded != payload:
            raise DataReadinessError(f"daily history resume request does not match {path}")
        return
    _atomic_json(path, payload)


def _load_existing_symbol(
    *,
    symbol: str,
    path: Path,
    request_hash: str,
    start_date: date,
    end_date: date,
) -> dict[str, Any] | None:
    if not path.exists() and not manifest_path_for(path).exists():
        return None
    bars, manifest = load_canonical_artifact(path, expected_type="bars")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or inputs.get("collection_request_sha256") != request_hash:
        raise DataReadinessError(f"existing daily history artifact has a different request identity: {path}")
    if set(bars["ticker"].astype(str)) != {symbol} or set(bars["price_feed"].astype(str)) != {"sip"}:
        raise DataReadinessError(f"existing daily history artifact has invalid symbol/feed identity: {path}")
    session_dates = bars["bar_start_utc"].dt.tz_convert(ZoneInfo("America/New_York")).dt.date
    if not session_dates.between(start_date, end_date).all():
        raise DataReadinessError(f"existing daily history artifact exceeds the frozen window: {path}")
    audit = CanonicalAuditReport(checks=audit_canonical_bars(bars, require_sip=True))
    audit.raise_for_failure()
    return _artifact_record(symbol, path, bars, manifest)


def _artifact_record(
    symbol: str,
    path: Path,
    bars: pd.DataFrame,
    manifest: dict[str, object],
) -> dict[str, Any]:
    starts = pd.to_datetime(bars["bar_start_utc"], utc=True)
    return {
        "ticker": symbol,
        "path": str(path),
        "manifest_path": str(manifest_path_for(path)),
        "sha256": str(manifest["artifact_sha256"]),
        "rows": len(bars),
        "first_bar_start_utc": starts.min().isoformat(),
        "last_bar_start_utc": starts.max().isoformat(),
        "price_feed": "sip",
        "adjustment": "all",
    }


def _write_attempt(
    attempts_dir: Path,
    collection: SourceCollection,
    artifact: dict[str, Any] | None,
) -> None:
    payload = {
        "collection": collection.model_dump(mode="json"),
        "artifact": artifact,
    }
    path = attempts_dir / f"{collection.ticker}_{collection.collection_id}.json"
    _atomic_json(path, payload)


def _latest_collection_ledger(
    attempts_dir: Path,
    symbols: list[str],
    observed: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    latest = _latest_attempt_collections(attempts_dir)
    records: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    for symbol in symbols:
        latest_collection = latest.get(symbol)
        if latest_collection is None and symbol in observed:
            latest_collection = SourceCollection(
                collection_id=f"resumed-{symbol}-{uuid4().hex}",
                ticker=symbol,
                source_family="alpaca_daily_bars",
                requested_start_utc=now,
                requested_end_utc=now,
                started_at_utc=now,
                completed_at_utc=now,
                status="observed",
                row_count=int(observed[symbol]["rows"]),
            )
        if latest_collection is not None:
            records.append(latest_collection.model_dump())
    return pd.DataFrame(records).sort_values("ticker", kind="stable").reset_index(drop=True)


def _latest_attempt_collections(attempts_dir: Path) -> dict[str, SourceCollection]:
    latest: dict[str, SourceCollection] = {}
    for path in attempts_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        attempt_collection = SourceCollection.model_validate(payload["collection"])
        previous = latest.get(attempt_collection.ticker)
        if previous is None or attempt_collection.completed_at_utc > previous.completed_at_utc:
            latest[attempt_collection.ticker] = attempt_collection
    return latest


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
