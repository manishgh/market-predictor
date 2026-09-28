"""Receipted bar and corporate-action collection for maturing swing outcomes.

Every provider page is stored byte for byte under its SHA-256, and each request unit gets one
immutable, self-hashed receipt, written after its pages. A receipt is `complete` only when its
page chain ended; anything else is `failed` and records why. Loading verifies every hash,
anchors and links the page chain, and decodes the stored pages again with the shared provider
decoders. A receipt so proves that what it stores is consistent and complete; it is self-hashed,
not signed, so it does not prove who wrote it.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

from market_predictor.collection.http_records import http_response_from_record, http_response_record
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.alpaca import AlpacaSource, decode_bars_page_response
from market_predictor.sources.alpaca_corporate_actions import (
    corporate_action_parameters,
    decode_corporate_actions_page,
    fetch_corporate_actions_page,
)
from market_predictor.sources.http import HttpByteResponse

BAR_RECEIPT_SCHEMA = "market_predictor.outcome_bar_receipt"
ACTION_RECEIPT_SCHEMA = "market_predictor.outcome_corporate_action_receipt"
BarTimeframe = Literal["1Day", "1Min"]
# The maturation bar schema `daily_path_bars` produces.
DAILY_BAR_COLUMNS = (
    "ticker", "session_date_et", "timeframe", "bar_start_utc", "bar_end_utc", "available_at_utc",
    "open", "high", "low", "close", "volume", "price_feed", "adjustment", "source_artifact_sha256",
)
_PAGE_SYMBOLS = 50
_PAGE_LIMIT = 10_000
_MAXIMUM_PAGES = 20
_DAILY_FINALIZATION = timedelta(minutes=15)
_NEW_YORK = "America/New_York"
# Provider and protocol failures become failed receipts; a memory-guard stop ends the run.
_RECORDED_FAILURES = (RuntimeError, OSError, ValueError, DataReadinessError)


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class OutcomeBarUnit:
    """One request: the symbols' bars over consecutive XNYS sessions, as named on the decision session.

    Alpaca's `asof` maps each symbol to the company it named on the decision session and
    follows that company's later renames.
    """

    decision_session: date
    first_session: date
    last_session: date
    symbols: tuple[str, ...]
    timeframe: BarTimeframe = "1Day"

    def __post_init__(self) -> None:
        if (
            not self.symbols
            or len(self.symbols) > _PAGE_SYMBOLS
            or tuple(sorted(set(self.symbols))) != self.symbols
            or any(not symbol or symbol != symbol.strip().upper() or "," in symbol for symbol in self.symbols)
        ):
            raise ValueError(f"an outcome bar unit needs 1 to {_PAGE_SYMBOLS} sorted, unique, upper-case symbols")
        if not self.decision_session <= self.first_session <= self.last_session:
            raise ValueError("an outcome bar unit needs its sessions in order after the decision")
        if self.timeframe == "1Min" and self.first_session != self.last_session:
            raise ValueError("a one-minute outcome bar unit covers one session")
        # The decision bar comes from the same receipt as the path, since its close scales the ATR.
        if self.timeframe == "1Day" and self.first_session != self.decision_session:
            raise ValueError("a daily outcome bar unit starts on its decision session")
        for session in (self.decision_session, self.first_session, self.last_session):
            if not _is_session(session):
                raise ValueError(f"not an XNYS session: {session}")

    @property
    def start(self) -> datetime:
        if self.timeframe == "1Min":
            return _session_open(self.first_session)
        return datetime.combine(self.first_session, time(), UTC)

    @property
    def end(self) -> datetime:
        if self.timeframe == "1Min":
            return _session_close(self.last_session) - timedelta(microseconds=1)
        return datetime.combine(self.last_session, time(), UTC) + timedelta(days=1, microseconds=-1)

    def request_params(self, page_token: str | None) -> dict[str, Any]:
        """The exact query `AlpacaSource.fetch_bars_page` sends for this unit and page."""
        params: dict[str, Any] = {
            "symbols": ",".join(self.symbols),
            "timeframe": self.timeframe,
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "feed": "sip",
            "limit": _PAGE_LIMIT,
            "adjustment": "all",
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        params["asof"] = self.decision_session.isoformat()
        return params

    def record(self) -> dict[str, object]:
        return {
            "decision_session_et": self.decision_session.isoformat(),
            "first_session_et": self.first_session.isoformat(),
            "last_session_et": self.last_session.isoformat(),
            "symbols": list(self.symbols),
            "timeframe": self.timeframe,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> OutcomeBarUnit:
        timeframe = record.get("timeframe")
        if timeframe not in ("1Day", "1Min"):
            raise DataReadinessError("an outcome bar receipt names an unknown timeframe")
        return cls(
            decision_session=date.fromisoformat(str(record["decision_session_et"])),
            first_session=date.fromisoformat(str(record["first_session_et"])),
            last_session=date.fromisoformat(str(record["last_session_et"])),
            symbols=tuple(str(symbol) for symbol in _sequence(record["symbols"])),
            timeframe=timeframe,
        )


@dataclass(frozen=True)
class CorporateActionUnit:
    """One symbol's corporate actions whose provider process date falls in the window."""

    decision_session: date
    symbol: str
    process_start: date
    process_end: date

    def __post_init__(self) -> None:
        if self.process_start > self.process_end:
            raise ValueError("a corporate-action unit needs an ordered process window")
        corporate_action_parameters(ticker=self.symbol, start=self.process_start, end=self.process_end)

    def request_params(self, page_token: str | None) -> dict[str, Any]:
        return corporate_action_parameters(
            ticker=self.symbol, start=self.process_start, end=self.process_end, page_token=page_token
        )

    def record(self) -> dict[str, object]:
        return {
            "decision_session_et": self.decision_session.isoformat(),
            "symbol": self.symbol,
            "process_start": self.process_start.isoformat(),
            "process_end": self.process_end.isoformat(),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> CorporateActionUnit:
        return cls(
            decision_session=date.fromisoformat(str(record["decision_session_et"])),
            symbol=str(record["symbol"]),
            process_start=date.fromisoformat(str(record["process_start"])),
            process_end=date.fromisoformat(str(record["process_end"])),
        )


@dataclass(frozen=True)
class BarReceipt:
    """A verified bar receipt; `rows` holds each symbol's decoded provider rows when complete.

    A bar retrieved before it was final (a daily bar before its session's close plus the
    finalization delay, a minute bar before the session close) counts as not returned.
    """

    receipt_id: str
    unit: OutcomeBarUnit
    complete: bool
    started_at_utc: datetime
    finished_at_utc: datetime
    rows: Mapping[str, tuple[tuple[dict[str, Any], datetime], ...]]

    def sessions_with_rows(self, symbol: str) -> frozenset[date]:
        """Sessions with any final row for the symbol, usable or not; usability is the validator's call."""
        return frozenset(_bar_session(row["t"], self.unit) for row, _ in self.rows.get(symbol, ()))


@dataclass(frozen=True)
class ActionReceipt:
    """A verified corporate-action receipt; `actions` holds every page's records by family."""

    receipt_id: str
    unit: CorporateActionUnit
    complete: bool
    started_at_utc: datetime
    finished_at_utc: datetime
    actions: Mapping[str, tuple[dict[str, Any], ...]]


def collect_bars(
    source: AlpacaSource,
    units: Sequence[OutcomeBarUnit],
    *,
    root: Path,
    clock: Callable[[], datetime] = _now,
) -> list[dict[str, object]]:
    """Request each unit's bars and publish one receipt per unit after its pages."""
    receipts: list[dict[str, object]] = []
    for unit in units:
        assert_system_memory_available(maximum_used_percent=90.0)
        started = clock()
        pages: list[dict[str, object]] = []
        failure: dict[str, str] | None = None
        token: str | None = None
        try:
            for _ in range(_MAXIMUM_PAGES):
                page = source.fetch_bars_page(
                    unit.symbols,
                    unit.start,
                    unit.end,
                    timeframe=unit.timeframe,
                    page_token=token,
                    asof=unit.decision_session,
                    limit=_PAGE_LIMIT,
                    adjustment="all",
                )
                if page.transport_response is None:
                    raise DataReadinessError("a bar page arrived without its response record")
                _store_body(root, page.transport_response)
                pages.append({"request_page_token": token, "response": http_response_record(page.transport_response)})
                for symbol_rows in page.bars.values():
                    for row in symbol_rows:
                        _bar_session(row.get("t"), unit)
                token = page.next_page_token
                if token is None:
                    break
            else:
                raise DataReadinessError(f"the bar page chain exceeds {_MAXIMUM_PAGES} pages")
        except MemoryBudgetError:
            raise
        except _RECORDED_FAILURES as exc:
            failure = _failure(exc)
        receipts.append(_publish(root, "bar_receipts", BAR_RECEIPT_SCHEMA, unit.decision_session, unit.record(),
                                 pages, failure, started, clock()))
    return receipts


def collect_corporate_actions(
    source: AlpacaSource,
    units: Sequence[CorporateActionUnit],
    *,
    root: Path,
    clock: Callable[[], datetime] = _now,
) -> list[dict[str, object]]:
    """Request each symbol's corporate actions and publish one receipt per unit after its pages."""
    receipts: list[dict[str, object]] = []
    for unit in units:
        assert_system_memory_available(maximum_used_percent=90.0)
        started = clock()
        pages: list[dict[str, object]] = []
        failure: dict[str, str] | None = None
        token: str | None = None
        identities: list[str] = []
        try:
            for _ in range(_MAXIMUM_PAGES):
                response = fetch_corporate_actions_page(
                    source, ticker=unit.symbol, start=unit.process_start, end=unit.process_end, page_token=token
                )
                _store_body(root, response)
                pages.append({"request_page_token": token, "response": http_response_record(response)})
                families, token = decode_corporate_actions_page(response, expected_params=unit.request_params(token))
                identities.extend(str(item["id"]) for records in families.values() for item in records if item.get("id"))
                if len(identities) != len(set(identities)):
                    # A paging fault on the provider's side; such a receipt never counts as complete.
                    raise DataReadinessError("the provider repeated an action across pages")
                if token is None:
                    break
            else:
                raise DataReadinessError(f"the corporate-action page chain exceeds {_MAXIMUM_PAGES} pages")
        except MemoryBudgetError:
            raise
        except _RECORDED_FAILURES as exc:
            failure = _failure(exc)
        receipts.append(_publish(root, "action_receipts", ACTION_RECEIPT_SCHEMA, unit.decision_session, unit.record(),
                                 pages, failure, started, clock()))
    return receipts


def load_bar_receipts(root: Path, decision_sessions: Iterable[date]) -> tuple[BarReceipt, ...]:
    """Verify the bar receipts of the given decision sessions and decode their pages again."""
    receipts: list[BarReceipt] = []
    for record in _receipt_records(root, "bar_receipts", BAR_RECEIPT_SCHEMA, decision_sessions):
        unit = OutcomeBarUnit.from_record(_mapping(record["unit"]))
        started, finished = _attempt_window(record)
        rows: dict[str, list[tuple[dict[str, Any], datetime]]] = {}
        for response, token, next_token in _verified_pages(root, record, started, finished):
            page = decode_bars_page_response(response, expected_params=unit.request_params(token))
            if page.next_page_token != next_token:
                raise DataReadinessError("a bar receipt's page chain does not link")
            for symbol, symbol_rows in page.bars.items():
                for row in symbol_rows:
                    if response.retrieved_at_utc >= _final_at(_bar_session(row.get("t"), unit), unit.timeframe):
                        rows.setdefault(symbol, []).append((row, response.retrieved_at_utc))
        receipts.append(
            BarReceipt(
                receipt_id=str(record["receipt_id"]),
                unit=unit,
                complete=record["status"] == "complete",
                started_at_utc=started,
                finished_at_utc=finished,
                rows={symbol: tuple(values) for symbol, values in rows.items()},
            )
        )
    return tuple(receipts)


def load_action_receipts(root: Path, decision_sessions: Iterable[date]) -> tuple[ActionReceipt, ...]:
    """Verify the corporate-action receipts of the given decision sessions and decode their pages again."""
    receipts: list[ActionReceipt] = []
    for record in _receipt_records(root, "action_receipts", ACTION_RECEIPT_SCHEMA, decision_sessions):
        unit = CorporateActionUnit.from_record(_mapping(record["unit"]))
        started, finished = _attempt_window(record)
        actions: dict[str, list[dict[str, Any]]] = {}
        for response, token, next_token in _verified_pages(root, record, started, finished):
            families, decoded_next = decode_corporate_actions_page(response, expected_params=unit.request_params(token))
            if decoded_next != next_token:
                raise DataReadinessError("a corporate-action receipt's page chain does not link")
            for family, records in families.items():
                actions.setdefault(family, []).extend(records)
        identities = [str(item.get("id")) for records in actions.values() for item in records if item.get("id")]
        if len(identities) != len(set(identities)):
            raise DataReadinessError("a corporate-action receipt repeats an action across its pages")
        receipts.append(
            ActionReceipt(
                receipt_id=str(record["receipt_id"]),
                unit=unit,
                complete=record["status"] == "complete",
                started_at_utc=started,
                finished_at_utc=finished,
                actions={family: tuple(values) for family, values in actions.items()},
            )
        )
    return tuple(receipts)


def complete_bar_receipts(
    receipts: Iterable[BarReceipt],
    *,
    decision_session: date,
    symbol: str,
    first_session: date,
    last_session: date,
    timeframe: BarTimeframe = "1Day",
) -> tuple[BarReceipt, ...]:
    """The complete receipts that asked for the symbol over every session in the range, oldest first.

    Each reflects the price adjustments as of its own retrieval, so a path takes all of its
    sessions from one of them.
    """
    return tuple(
        sorted(
            (
                receipt
                for receipt in receipts
                if receipt.complete
                and receipt.unit.timeframe == timeframe
                and receipt.unit.decision_session == decision_session
                and symbol in receipt.unit.symbols
                and receipt.unit.first_session <= first_session
                and last_session <= receipt.unit.last_session
            ),
            key=lambda receipt: (receipt.finished_at_utc, receipt.receipt_id),
        )
    )


def daily_path_bars(receipt: BarReceipt, symbol: str) -> pd.DataFrame:
    """The symbol's daily bars from one complete receipt, in the maturation bar schema.

    A bar is available at the later of its session close plus the finalization delay and the
    time its page was retrieved. Unusable bars are kept for the observation validator to mark.
    """
    if not receipt.complete or receipt.unit.timeframe != "1Day" or symbol not in receipt.unit.symbols:
        raise DataReadinessError("daily path bars need a complete daily receipt that requested the symbol")
    sessions = [_bar_session(row.get("t"), receipt.unit) for row, _ in receipt.rows.get(symbol, ())]
    if len(sessions) != len(set(sessions)):
        # Two bars for one session in one response is a provider defect, never a gap.
        raise DataReadinessError(f"a receipt holds duplicated sessions for {symbol}: {receipt.receipt_id}")
    records: list[dict[str, object]] = []
    for row, retrieved_at in receipt.rows.get(symbol, ()):
        session = _bar_session(row.get("t"), receipt.unit)
        close = _session_close(session)
        records.append(
            {
                "ticker": symbol,
                "session_date_et": session,
                "timeframe": "1d",
                "bar_start_utc": _session_open(session),
                "bar_end_utc": close,
                "available_at_utc": max(close + _DAILY_FINALIZATION, retrieved_at.astimezone(UTC)),
                "open": _number(row, "o"),
                "high": _number(row, "h"),
                "low": _number(row, "l"),
                "close": _number(row, "c"),
                "volume": _number(row, "v"),
                "price_feed": "sip",
                "adjustment": "all",
                "source_artifact_sha256": receipt.receipt_id,
            }
        )
    return pd.DataFrame(records, columns=list(DAILY_BAR_COLUMNS))


def _publish(
    root: Path,
    collection: str,
    schema: str,
    decision_session: date,
    unit: dict[str, object],
    pages: list[dict[str, object]],
    failure: dict[str, str] | None,
    started: datetime,
    finished: datetime,
) -> dict[str, object]:
    # The attempt spans the process clock and every page's retrieval clock, so a clock step
    # between the two can never make a stored receipt fail its own verification.
    retrieved = [_aware(_mapping(page["response"])["retrieved_at_utc"]) for page in pages]
    content: dict[str, object] = {
        "schema": schema,
        "unit": unit,
        "status": "failed" if failure else "complete",
        "failure": failure,
        "attempt_started_at_utc": min([started.astimezone(UTC), *retrieved]).isoformat(),
        "attempt_finished_at_utc": max([finished.astimezone(UTC), *retrieved]).isoformat(),
        "pages": pages,
    }
    receipt = {**content, "receipt_id": json_sha256(content)}
    _write_once(root / collection / decision_session.isoformat() / f"{receipt['receipt_id']}.json", _encode(receipt))
    return receipt


def _receipt_records(
    root: Path,
    collection: str,
    schema: str,
    decision_sessions: Iterable[date],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for session in sorted(set(decision_sessions)):
        for path in sorted((root / collection / session.isoformat()).glob("*.json")):
            try:
                record = parse_strict_json_object(path.read_bytes(), label="outcome receipt")
            except (OSError, ValueError) as exc:
                raise DataReadinessError(f"an outcome receipt is unreadable: {path}") from exc
            content = {key: value for key, value in record.items() if key != "receipt_id"}
            if (
                record.get("schema") != schema
                or record.get("receipt_id") != path.stem
                or json_sha256(content) != path.stem
                or record.get("status") not in ("complete", "failed")
                or (record["status"] == "failed") != (record.get("failure") is not None)
                or _mapping(record.get("unit")).get("decision_session_et") != session.isoformat()
            ):
                raise DataReadinessError(f"an outcome receipt does not verify: {path}")
            records.append(dict(record))
    return records


def _verified_pages(
    root: Path,
    record: Mapping[str, Any],
    started: datetime,
    finished: datetime,
) -> list[tuple[HttpByteResponse, str | None, str | None]]:
    """Each complete receipt's pages rebuilt from their stored bodies, with their request and next tokens."""
    if record["status"] != "complete":
        return []
    pages = [_mapping(page) for page in _sequence(record["pages"])]
    if not pages or pages[0].get("request_page_token") is not None:
        raise DataReadinessError("a complete outcome receipt must start with the chain's first page")
    verified: list[tuple[HttpByteResponse, str | None, str | None]] = []
    for index, page in enumerate(pages):
        response_record = _mapping(page["response"])
        body_sha256 = str(response_record.get("body_sha256"))
        if len(body_sha256) != 64 or any(character not in "0123456789abcdef" for character in body_sha256):
            raise DataReadinessError("an outcome receipt names an invalid page body")
        body_path = root / "bodies" / f"{body_sha256}.json"
        try:
            body = body_path.read_bytes()
        except OSError as exc:
            raise DataReadinessError(f"an outcome receipt's page body is missing: {body_path}") from exc
        response = http_response_from_record(response_record, body)
        if not started <= response.retrieved_at_utc <= finished:
            raise DataReadinessError("an outcome receipt's page was retrieved outside its attempt")
        # Page i+1 was requested with the token page i returned; the last page returned none.
        next_token = _optional_token(pages[index + 1].get("request_page_token")) if index + 1 < len(pages) else None
        verified.append((response, _optional_token(page.get("request_page_token")), next_token))
    return verified


def _attempt_window(record: Mapping[str, Any]) -> tuple[datetime, datetime]:
    started = _aware(record.get("attempt_started_at_utc"))
    finished = _aware(record.get("attempt_finished_at_utc"))
    if finished < started:
        raise DataReadinessError("an outcome receipt's attempt ends before it starts")
    return started, finished


def _bar_session(value: object, unit: OutcomeBarUnit) -> date:
    """The XNYS session a provider bar belongs to; a bar outside the unit's sessions is refused."""
    try:
        stamp = pd.Timestamp(str(value))
    except ValueError as exc:
        raise DataReadinessError(f"a provider bar has an invalid timestamp: {value}") from exc
    if stamp.tzinfo is None:
        raise DataReadinessError(f"a provider bar timestamp is not timezone-aware: {value}")
    local = stamp.tz_convert(_NEW_YORK)
    session: date = local.date()
    if unit.timeframe == "1Day" and (local.hour, local.minute, local.second, local.microsecond) != (0, 0, 0, 0):
        raise DataReadinessError(f"a daily provider bar is not stamped at midnight New York time: {value}")
    if not unit.first_session <= session <= unit.last_session or not _is_session(session):
        raise DataReadinessError(f"a provider bar lies outside the requested sessions: {value}")
    return session


def _final_at(session: date, timeframe: BarTimeframe) -> datetime:
    """When a bar of the session is final: the close plus the delay for daily bars, the close for minutes."""
    close = _session_close(session)
    return close + _DAILY_FINALIZATION if timeframe == "1Day" else close


def _store_body(root: Path, response: HttpByteResponse) -> None:
    _write_once(root / "bodies" / f"{response.sha256}.json", response.body)


def _write_once(path: Path, payload: bytes) -> None:
    """Publish immutable bytes durably; an existing file must hold exactly the same bytes."""
    if path.exists():
        if path.read_bytes() != payload:
            raise DataReadinessError(f"an immutable outcome file already holds different bytes: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    """Make a rename durable; Windows has no directory handles to sync."""
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encode(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _failure(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def _is_session(session: date) -> bool:
    return bool(xcals.get_calendar("XNYS").is_session(pd.Timestamp(session)))


def _session_open(session: date) -> datetime:
    opened: datetime = xcals.get_calendar("XNYS").session_open(pd.Timestamp(session)).to_pydatetime()
    return opened


def _session_close(session: date) -> datetime:
    closed: datetime = xcals.get_calendar("XNYS").session_close(pd.Timestamp(session)).to_pydatetime()
    return closed


def _number(row: Mapping[str, Any], key: str) -> float:
    """A bar field as a number; a missing or non-numeric one is NaN, so the validator marks the bar unusable."""
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def _aware(value: object) -> datetime:
    try:
        moment = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise DataReadinessError("an outcome receipt time is invalid") from exc
    if moment.utcoffset() is None:
        raise DataReadinessError("an outcome receipt time is not timezone-aware")
    return moment.astimezone(UTC)


def _optional_token(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise DataReadinessError("an outcome receipt page token is invalid")
    return value


def _mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataReadinessError("an outcome receipt field is not an object")
    return {str(key): item for key, item in value.items()}


def _sequence(value: object) -> list[Any]:
    if not isinstance(value, list):
        raise DataReadinessError("an outcome receipt field is not a list")
    return value
