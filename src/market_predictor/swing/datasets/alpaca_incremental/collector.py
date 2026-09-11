"""Sequential, resumable source-only collection with bounded page replay."""
from __future__ import annotations

import gzip
import json
import os
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests

from market_predictor.canonical.store import file_sha256
from market_predictor.config import Settings
from market_predictor.core.errors import MemoryBudgetError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.swing.datasets.alpaca_incremental.config import FAMILIES, Config, load_config
from market_predictor.swing.datasets.alpaca_incremental.pages import MAX_BODY, MAX_LINE, Unit, page_record, validate_record
from market_predictor.swing.datasets.alpaca_incremental.storage import IntegrityError, publish, verified


def guard_memory() -> None:
    assert_system_memory_available(minimum_available_gib=2.0, maximum_used_percent=85.0)
    assert_memory_budget(hard_budget_gib=5.0, headroom_gib=0.25, stage="alpaca incremental page")


def _http_status(error: Exception) -> int | None:
    current: BaseException | None = error
    for _ in range(10):
        if isinstance(current, requests.HTTPError) and current.response is not None:
            return current.response.status_code
        current = current.__cause__ if current else None
    return None


def _replay(directory: Path, receipt: dict[str, Any], unit: Unit, config: Config, request_hash: str) -> dict[str, Any]:
    try:
        if receipt.get("unit") != unit.record() or receipt.get("request_sha256") != request_hash:
            raise IntegrityError("successful unit request differs")
        name = receipt.get("archive")
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("archive-") or not name.endswith(".jsonl.gz"):
            raise IntegrityError("invalid archive reference")
        archive = directory / name
        if archive.resolve().parent != directory.resolve() or file_sha256(archive) != receipt.get("archive_sha256"):
            raise IntegrityError("successful archive SHA mismatch")
        counts = dict.fromkeys(unit.symbols, 0)
        pages = rows = 0
        token: str | None = None
        seen: set[str] = set()
        with gzip.open(archive, "rb") as handle:
            while True:
                guard_memory()
                line = handle.readline(MAX_LINE + 1)
                if not line:
                    break
                if len(line) > MAX_LINE or not line.endswith(b"\n") or pages >= config.max_pages_per_unit or (pages and token is None):
                    raise IntegrityError("archive line or pagination exceeds contract")
                record = parse_strict_json_object(line, label="incremental archive page")
                page_counts, row_count, token = validate_record(record, unit, config, token)
                if token is not None:
                    if token in seen:
                        raise IntegrityError("archive repeated continuation token")
                    seen.add(token)
                for symbol, count in page_counts.items():
                    counts[symbol] += count
                rows += row_count
                pages += 1
                del record, line
        if (not pages or token is not None or receipt.get("pages") != pages
                or receipt.get("rows") != rows or receipt.get("counts") != counts):
            raise IntegrityError("archive is incomplete or counters differ")
        return receipt
    except MemoryBudgetError:
        raise
    except (OSError, ValueError, TypeError, KeyError, EOFError) as exc:
        raise IntegrityError("successful unit failed exact replay") from exc


class _Run:
    def __init__(self, config: Config, output: Path, request: dict[str, Any], source: AlpacaSource | None,
                 max_units: int | None, offline: bool) -> None:
        self.config, self.output, self.request = config, output, request
        self.source, self.max_units, self.offline = source, max_units, offline
        self.attempted = self.fetched = self.reused = self.split_attempts = 0
        self.paused = False
        self.integrity_failures = 0
        self.accounted = 0
        self.family_summary: dict[str, Any] = {}
        self.scope_summary: dict[str, Any] = {}
        for scope in ("daily", "revision", "partial"):
            self.scope_summary[scope] = {family: {"rows": 0, "observed_units": 0, "no_data_units": 0,
                "failed_units": 0, "pending_units": 0,
                "symbols": {symbol: {"rows": 0, "observed_days": 0, "no_data_days": 0, "failed_days": 0, "pending_days": 0}
                    for symbol in request["symbols"]}} for family in FAMILIES}
        self.family_summary = self.scope_summary["daily"]

    def _account(self, unit: Unit, status: str, receipt: dict[str, Any] | None = None) -> bool:
        scope = "partial" if unit.cutoff else "daily" if unit.capture == "daily" else "revision"
        summary = self.scope_summary[scope][unit.family]
        counts = receipt["counts"] if receipt else {}
        summary[f"{status}_units"] += 1
        summary["rows"] += receipt["rows"] if receipt else 0
        for symbol in unit.symbols:
            symbol_status = ("observed" if counts[symbol] else "no_data") if receipt else status
            summary["symbols"][symbol][f"{symbol_status}_days"] += 1
            summary["symbols"][symbol]["rows"] += counts.get(symbol, 0)
        self.accounted += 1
        if not self.offline and self.accounted % 50 == 0:
            publish(self.output / "progress.json", {"status": "in_progress", "source_only": True,
                "accounted_units": self.accounted, "attempted_units": self.attempted, "fetched_units": self.fetched,
                "reused_units": self.reused, "last_unit": unit.key, "updated_at_utc": datetime.now(UTC).isoformat()}, replace=True)
        return receipt is not None

    def _download(self, unit: Unit, directory: Path) -> dict[str, Any]:
        if self.source is None:
            raise RuntimeError("online acquisition requires a source")
        if self.source.settings.alpaca_stock_feed.lower().strip() != "sip":
            raise ValueError("incremental bars require SIP")
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / f"archive-{uuid4().hex}.jsonl.gz"
        counts = dict.fromkeys(unit.symbols, 0)
        rows = pages = 0
        token: str | None = None
        seen: set[str] = set()
        previous_guard = self.source.client.before_request

        def before_request() -> None:
            guard_memory()
            if previous_guard is not None:
                previous_guard()

        self.source.client.before_request = before_request
        try:
            with archive.open("xb") as file, gzip.GzipFile(fileobj=file, mode="wb", mtime=0) as handle:
                for _ in range(self.config.max_pages_per_unit):
                    guard_memory()
                    page = unit.fetch(self.source, self.config, token)
                    guard_memory()
                    try:
                        record = page_record(page, unit, self.config, token)
                    except (ValueError, TypeError, KeyError):
                        # Retain rejected bytes, but never publish them as successful evidence.
                        if page.raw_body is not None and len(page.raw_body) <= MAX_BODY:
                            with gzip.open(directory / f"rejected-{uuid4().hex}.body.gz", "xb") as rejected:
                                rejected.write(page.raw_body)
                        raise
                    page_counts, row_count, token = validate_record(record, unit, self.config, token)
                    if token is not None:
                        if token in seen:
                            raise ValueError("provider repeated continuation token")
                        seen.add(token)
                    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
                    if len(encoded) > MAX_LINE:
                        raise ValueError("serialized page exceeds bounded size")
                    handle.write(encoded)
                    for symbol, count in page_counts.items():
                        counts[symbol] += count
                    rows += row_count
                    pages += 1
                    del page, record, encoded
                    if token is None:
                        break
                if token is not None:
                    raise ValueError("provider exceeded page limit; unit remains incomplete")
            with archive.open("r+b") as reader:
                os.fsync(reader.fileno())
            receipt = {"unit": unit.record(), "request_sha256": self.request["request_sha256"],
                "archive": archive.name, "archive_sha256": file_sha256(archive),
                "pages": pages, "rows": rows, "counts": counts}
            publish(directory / "success.json", receipt)
            return receipt
        finally:
            self.source.client.before_request = previous_guard

    def visit(self, unit: Unit) -> bool:
        directory = self.output / "units" / unit.key
        success, split = directory / "success.json", directory / "split.json"
        if self.paused:
            return self._account(unit, "pending")
        try:
            if success.exists():
                if split.exists():
                    raise IntegrityError("unit has conflicting success and split receipts")
                receipt = _replay(directory, verified(success), unit, self.config, self.request["request_sha256"])
                self.reused += 1
                return self._account(unit, "observed" if receipt["rows"] else "no_data", receipt)
            if split.exists():
                expected = {"unit": unit.record(), "request_sha256": self.request["request_sha256"], "reason": "provider_symbol_rejection"}
                if len(unit.symbols) < 2 or verified(split) != expected:
                    raise IntegrityError("invalid retained split decision")
                midpoint = len(unit.symbols) // 2
                left = self.visit(replace(unit, symbols=unit.symbols[:midpoint]))
                right = self.visit(replace(unit, symbols=unit.symbols[midpoint:]))
                return left and right
        except MemoryBudgetError:
            self.paused = True
            return self._account(unit, "pending")
        except IntegrityError:
            self.integrity_failures += 1
            return self._account(unit, "failed")
        if self.offline or (self.max_units is not None and self.attempted >= self.max_units):
            return self._account(unit, "failed" if (directory / "failure.json").exists() else "pending")
        self.attempted += 1
        try:
            receipt = self._download(unit, directory)
        except MemoryBudgetError:
            self.paused = True
            return self._account(unit, "pending")
        except Exception as exc:
            status = _http_status(exc)
            # Never persist exception strings, URLs, request headers or credential-bearing messages.
            publish(directory / "failure.json", {"unit": unit.record(), "request_sha256": self.request["request_sha256"],
                "status": "failed", "http_status": status, "reason": "provider_or_page_validation_failed",
                "observed_at_utc": datetime.now(UTC).isoformat()}, replace=True)
            if status in (400, 404, 422) and len(unit.symbols) > 1:
                publish(split, {"unit": unit.record(), "request_sha256": self.request["request_sha256"],
                    "reason": "provider_symbol_rejection"})
                self.split_attempts += 1
                return self.visit(unit)
            return self._account(unit, "failed")
        self.fetched += 1
        return self._account(unit, "observed" if receipt["rows"] else "no_data", receipt)


def _run(config_path: Path, through: date, cutoff: datetime, max_units: int | None, offline: bool,
         source: AlpacaSource | None) -> dict[str, Any]:
    guard_memory()
    config, output, request = load_config(config_path)
    if through < config.start or through > cutoff.date():
        raise ValueError("through must be between start and today UTC")
    frozen = output / "request.json"
    if frozen.exists():
        if verified(frozen) != request:
            raise IntegrityError("output is bound to a different frozen request")
    elif offline:
        raise IntegrityError("offline collection requires an existing frozen request")
    else:
        if output.exists() and any(output.iterdir()):
            raise IntegrityError("new output must be empty; old archives cannot be reused as destinations")
        publish(frozen, request)
    run = _Run(config, output, request, source, max_units, offline)
    if not offline:
        publish(output / "progress.json", {"status": "in_progress", "source_only": True,
            "attempted_units": 0, "updated_at_utc": cutoff.isoformat()}, replace=True)
    symbols = tuple(request["symbols"])
    completed_through = min(through, cutoff.date() - timedelta(days=1))
    watermark: date | None = None
    prefix_complete = True
    day = config.start
    while day <= completed_through:
        day_complete = True
        for offset in range(0, len(symbols), config.batch_size):
            batch = symbols[offset:offset + config.batch_size]
            for family in FAMILIES:
                if not run.visit(Unit(day, family, batch)):
                    day_complete = False
        prefix_complete = prefix_complete and day_complete
        if prefix_complete:
            watermark = day
        day += timedelta(days=1)
    # Reobservations never replace a daily receipt or advance its freshness watermark.
    revisions = output / "revision_captures"
    if offline:
        for path in revisions.iterdir() if revisions.exists() else ():
            if path.suffix != ".json":
                continue
            record = verified(path)
            day = date.fromisoformat(record["day"])
            capture_date = date.fromisoformat(record["capture_date"])
            if (record.get("request_sha256") != request["request_sha256"] or not config.start <= day < capture_date <= cutoff.date()
                    or not 1 <= (capture_date - day).days <= config.revision_overlap_days):
                raise IntegrityError("invalid saved revision capture")
            if day <= completed_through:
                _revision(run, symbols, day, capture_date)
    else:
        revision_start = max(config.start, cutoff.date() - timedelta(days=config.revision_overlap_days))
        day = revision_start
        while day <= completed_through:
            record = {"day": day.isoformat(), "capture_date": cutoff.date().isoformat(), "request_sha256": request["request_sha256"]}
            path = revisions / f"{cutoff.date().isoformat()}-{day.isoformat()}.json"
            if path.exists():
                if verified(path) != record:
                    raise IntegrityError("revision capture binding differs")
            else:
                publish(path, record)
            _revision(run, symbols, day, cutoff.date())
            day += timedelta(days=1)
    partial_capture_count = 0
    if through == cutoff.date():
        if offline:
            # Replay saved partial captures, never invent a new capture during an offline audit.
            capture_files = (output / "captures").glob("*.json")
            for path in capture_files:
                capture = verified(path)
                observed = datetime.fromisoformat(capture["cutoff"])
                if capture.get("request_sha256") != request["request_sha256"] or observed.utcoffset() != timedelta(0) or observed > cutoff:
                    raise IntegrityError("invalid partial capture binding")
                if observed.date() == through:
                    _partial(run, symbols, observed)
                    partial_capture_count += 1
        else:
            publish(output / "captures" / f"{cutoff.strftime('%Y%m%dT%H%M%S%fZ')}.json",
                {"cutoff": cutoff.isoformat(), "request_sha256": request["request_sha256"]})
            _partial(run, symbols, cutoff)
            partial_capture_count += 1
    failed = sum(f["failed_units"] for scope in run.scope_summary.values() for f in scope.values())
    pending = sum(f["pending_units"] for scope in run.scope_summary.values() for f in scope.values())
    status = ("paused_memory" if run.paused else "integrity_failed" if run.integrity_failures else
        "partial_failures" if failed else "bounded_incomplete" if pending or (through == cutoff.date() and not partial_capture_count)
        else "verified" if offline else "complete")
    summary: dict[str, Any] = {"schema": "market_predictor.alpaca_incremental_status.v1", "status": status,
        "request_sha256": request["request_sha256"], "through": through.isoformat(), "cutoff_utc": cutoff.isoformat(),
        "complete_through_utc_date": watermark.isoformat() if watermark else None,
        "today_complete": False, "attempted_units": run.attempted, "fetched_units": run.fetched,
        "reused_units": run.reused, "split_attempts": run.split_attempts, "integrity_failed_units": run.integrity_failures,
        "failed_units": failed, "pending_units": pending, "families": run.family_summary,
        "revisions": run.scope_summary["revision"], "partial_today": run.scope_summary["partial"],
        "partial_capture_count": partial_capture_count, "count_basis": "unit observations; captures may repeat a calendar day",
        "revision_overlap_days": config.revision_overlap_days, "older_revisions_verified": False,
        "source_only": True, "training_ready": False, "issuer_attribution": False, "active_membership_authority": False,
        "news_history_basis": "retrieved_now_revision_snapshot_research_only",
        "news_query_timestamp": "updated_at",
        "no_data_meaning": "no returned bars or provider-tagged news for this symbol after full pagination; not coverage proof",
        "status_path": str(output / "status.json"),
        "lag_policy": "default yesterday UTC; local-midnight scheduling can lag the local calendar by one day"}
    if not offline:
        publish(output / "status.json", summary, replace=True)
        publish(output / "progress.json", {"status": status, "source_only": True, "attempted_units": run.attempted,
            "fetched_units": run.fetched, "reused_units": run.reused, "status_path": str(output / "status.json")}, replace=True)
    return summary


def _partial(run: _Run, symbols: tuple[str, ...], cutoff: datetime) -> None:
    for offset in range(0, len(symbols), run.config.batch_size):
        for family in FAMILIES:
            run.visit(Unit(cutoff.date(), family, symbols[offset:offset + run.config.batch_size],
                capture=f"partial-{cutoff.isoformat()}", cutoff=cutoff))


def _revision(run: _Run, symbols: tuple[str, ...], day: date, capture_date: date) -> None:
    for offset in range(0, len(symbols), run.config.batch_size):
        for family in FAMILIES:
            run.visit(Unit(day, family, symbols[offset:offset + run.config.batch_size], capture=f"revision-{capture_date.isoformat()}"))


def collect(config_path: Path, *, through: date | None = None, max_units: int | None = None,
            offline: bool = False, source: AlpacaSource | None = None, root: Path | None = None) -> dict[str, Any]:
    """Own the heavy lease. Inject an AlpacaSource with a fixture client for tests.

    Through today produces a separate partial snapshot, never a completed daily unit.
    Max units counts new acquisition attempts, including rejected parents before splitting.
    Injected sources remain caller-owned; internally created HTTP sessions always close.
    """
    if max_units is not None and (type(max_units) is not int or not 1 <= max_units <= 1000000):
        raise ValueError("max_units must be an integer in 1..1000000")
    cutoff = datetime.now(UTC)
    requested_through = through if through is not None else cutoff.date() - timedelta(days=1)
    if type(requested_through) is not date or requested_through > cutoff.date():
        raise ValueError("through must be a UTC calendar date no later than today")
    root = (root or Path.cwd()).resolve()
    config_path = (root / config_path).resolve()
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("alpaca-incremental-source-only", runtime_dir=root / runtime, config_path=config_path):
        owned_source: AlpacaSource | None = None
        try:
            guard_memory()
            if not offline and source is None:
                owned_source = AlpacaSource(Settings())
                source = owned_source
            return _run(config_path.resolve(), requested_through, cutoff, max_units, offline, source)
        except MemoryBudgetError:
            return {"status": "paused_memory", "source_only": True, "training_ready": False,
                "complete_through_utc_date": None, "today_complete": False}
        finally:
            if owned_source is not None:
                owned_source.client.session.close()
