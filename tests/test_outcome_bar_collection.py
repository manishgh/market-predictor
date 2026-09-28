from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest import mock
from urllib.parse import urlencode

import pytest

import market_predictor.collection.outcome_bars as outcome_bars
from market_predictor.collection.outcome_bars import (
    BarReceipt,
    CorporateActionUnit,
    OutcomeBarUnit,
    collect_bars,
    collect_corporate_actions,
    complete_bar_receipts,
    daily_path_bars,
    load_action_receipts,
    load_bar_receipts,
)
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.monitoring_lease import MonitoringBusyError, monitoring_lease
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.http import HttpByteResponse

DECISION = date(2026, 7, 24)
SESSIONS = [date(2026, 7, 27), date(2026, 7, 28), date(2026, 7, 29)]
RETRIEVED = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
Responder = Callable[[str, dict[str, Any]], tuple[int, object]]


class _Client:
    """Answers like Alpaca's HTTP client: the requested URL carries exactly the sent query."""

    def __init__(self, respond: Responder, retrieved: datetime = RETRIEVED) -> None:
        self.respond = respond
        self.retrieved = retrieved
        self.requests: list[dict[str, Any]] = []

    def get_bytes_with_metadata(self, url: str, *, params: dict[str, Any], **_: object) -> HttpByteResponse:
        self.requests.append(dict(params))
        status, payload = self.respond(url, params)
        body = json.dumps(payload, sort_keys=True).encode()
        requested = f"{url}?{urlencode(params)}"
        return HttpByteResponse(
            body=body, requested_url=requested, final_url=requested, redirect_chain=(), status_code=status,
            retrieved_at_utc=self.retrieved, content_type="application/json; charset=utf-8", content_encoding=None,
            etag=None, last_modified=None, body_length=len(body), sha256=sha256(body).hexdigest(),
            body_representation="http_entity_encoded",
            safe_headers=(("content-type", "application/json; charset=utf-8"),),
        )


def _source(respond: Responder, retrieved: datetime = RETRIEVED) -> AlpacaSource:
    source = AlpacaSource(Settings(ALPACA_API_KEY_ID="synthetic-key", ALPACA_API_SECRET_KEY="synthetic-secret"))
    source.client = _Client(respond, retrieved)  # type: ignore[assignment]
    return source


def _clock(retrieved: datetime = RETRIEVED) -> Callable[[], datetime]:
    times: Iterator[datetime] = iter([retrieved - timedelta(seconds=1), retrieved + timedelta(seconds=1)] * 50)
    return lambda: next(times)


def _bar(session: date, close: float = 100.0, volume: float = 1_000.0) -> dict[str, object]:
    return {"t": f"{session.isoformat()}T04:00:00Z", "o": close, "h": close + 1, "l": close - 1, "c": close, "v": volume}


def _minute(session: date) -> dict[str, object]:
    return {"t": f"{session.isoformat()}T13:30:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 10}


def _daily(bars: dict[str, list[dict[str, object]]], token: str | None = None) -> Responder:
    return lambda _url, _params: (200, {"bars": bars, "next_page_token": token})


def _chain(first: dict[str, list[dict[str, object]]], second: dict[str, list[dict[str, object]]]) -> Responder:
    def respond(_url: str, params: dict[str, Any]) -> tuple[int, object]:
        if params.get("page_token") is None:
            return 200, {"bars": first, "next_page_token": "second"}
        return 200, {"bars": second, "next_page_token": None}

    return respond


def _unit(*symbols: str, timeframe: str = "1Day", last: date = SESSIONS[-1]) -> OutcomeBarUnit:
    if timeframe == "1Min":
        return OutcomeBarUnit(DECISION, last, last, tuple(sorted(symbols)), "1Min")
    return OutcomeBarUnit(DECISION, DECISION, last, tuple(sorted(symbols)))


def _covering(root: Path, *, last: date = SESSIONS[-1]) -> tuple[BarReceipt, ...]:
    return complete_bar_receipts(
        load_bar_receipts(root, [DECISION]), decision_session=DECISION, symbol="MSFT",
        first_session=DECISION, last_session=last,
    )


def test_a_daily_unit_is_collected_and_reloaded_with_absent_symbols_as_no_bars(tmp_path: Path) -> None:
    source = _source(_daily({"MSFT": [_bar(session) for session in [DECISION, *SESSIONS]]}))

    [receipt] = collect_bars(source, [_unit("MSFT", "TWTR")], root=tmp_path, clock=_clock())
    [loaded] = load_bar_receipts(tmp_path, [DECISION])

    assert receipt["status"] == "complete" and loaded.complete
    assert loaded.sessions_returned("MSFT") == frozenset([DECISION, *SESSIONS])
    assert loaded.sessions_returned("TWTR") == frozenset()
    request = source.client.requests[0]  # type: ignore[attr-defined]
    assert (request["asof"], request["adjustment"], request["timeframe"]) == ("2026-07-24", "all", "1Day")
    bars = daily_path_bars(loaded, "MSFT")
    assert list(bars["session_date_et"]) == [DECISION, *SESSIONS]
    # A bar is available once it both closed and was retrieved.
    assert (bars["available_at_utc"] == RETRIEVED).all()
    assert set(bars["source_artifact_sha256"]) == {loaded.receipt_id}


def test_an_empty_response_is_a_complete_receipt_without_bars(tmp_path: Path) -> None:
    collect_bars(_source(_daily({})), [_unit("TWTR")], root=tmp_path, clock=_clock())

    [loaded] = load_bar_receipts(tmp_path, [DECISION])

    assert loaded.complete and loaded.sessions_returned("TWTR") == frozenset()


def test_a_page_chain_is_followed_and_verified_on_load(tmp_path: Path) -> None:
    source = _source(_chain({"MSFT": [_bar(SESSIONS[0])]}, {"MSFT": [_bar(session) for session in SESSIONS[1:]]}))

    [receipt] = collect_bars(source, [_unit("MSFT")], root=tmp_path, clock=_clock())
    [loaded] = load_bar_receipts(tmp_path, [DECISION])

    assert len(receipt["pages"]) == 2  # type: ignore[arg-type]
    assert loaded.sessions_returned("MSFT") == frozenset(SESSIONS)


def test_a_receipt_without_its_first_page_is_refused(tmp_path: Path) -> None:
    collect_bars(_source(_chain({"MSFT": [_bar(SESSIONS[0])]}, {"MSFT": [_bar(SESSIONS[1])]})), [_unit("MSFT")],
                 root=tmp_path, clock=_clock())
    path = next((tmp_path / "bar_receipts").rglob("*.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    content = {key: value for key, value in record.items() if key != "receipt_id"}
    content["pages"] = content["pages"][1:]
    receipt_id = outcome_bars.json_sha256(content)
    path.unlink()
    (path.parent / f"{receipt_id}.json").write_text(json.dumps({**content, "receipt_id": receipt_id}), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="first page"):
        load_bar_receipts(tmp_path, [DECISION])


def test_a_failed_response_is_a_failed_receipt_that_never_supplies_a_path(tmp_path: Path) -> None:
    [receipt] = collect_bars(
        _source(lambda _url, _params: (429, {"message": "too many requests"})),
        [_unit("MSFT")], root=tmp_path, clock=_clock(),
    )

    assert receipt["status"] == "failed" and receipt["failure"]["type"] == "RuntimeError"  # type: ignore[index]
    assert len(load_bar_receipts(tmp_path, [DECISION])) == 1 and _covering(tmp_path) == ()


def test_a_bar_outside_the_requested_sessions_fails_the_receipt(tmp_path: Path) -> None:
    [receipt] = collect_bars(
        _source(_daily({"MSFT": [_bar(date(2026, 7, 30))]})), [_unit("MSFT")], root=tmp_path, clock=_clock()
    )

    assert receipt["status"] == "failed"
    assert "outside the requested sessions" in receipt["failure"]["message"]  # type: ignore[index]


def test_daily_bars_must_be_stamped_at_midnight_new_york(tmp_path: Path) -> None:
    [receipt] = collect_bars(
        _source(_daily({"MSFT": [{**_bar(SESSIONS[0]), "t": "2026-07-28T00:00:00Z"}]})), [_unit("MSFT")],
        root=tmp_path, clock=_clock(),
    )

    assert receipt["status"] == "failed" and "midnight" in receipt["failure"]["message"]  # type: ignore[index]


def test_each_receipt_supplies_a_whole_path_on_its_own_price_basis(tmp_path: Path) -> None:
    later = RETRIEVED + timedelta(days=1)
    collect_bars(_source(_daily({"MSFT": [_bar(s, 100.0) for s in SESSIONS]})), [_unit("MSFT")],
                 root=tmp_path, clock=_clock())
    # A 2-for-1 split between the two retrievals halves every adjusted price in the later one.
    collect_bars(_source(_daily({"MSFT": [_bar(s, 50.0) for s in SESSIONS[:2]]}), later), [_unit("MSFT")],
                 root=tmp_path, clock=_clock(later))
    # A narrower later request does not cover the path, so it is no candidate for it.
    narrower = later + timedelta(hours=1)
    collect_bars(_source(_daily({"MSFT": [_bar(SESSIONS[0], 50.0)]}), narrower), [_unit("MSFT", last=SESSIONS[0])],
                 root=tmp_path, clock=_clock(narrower))

    older, newer = _covering(tmp_path)

    assert list(daily_path_bars(older, "MSFT")["close"]) == [100.0, 100.0, 100.0]
    assert list(daily_path_bars(newer, "MSFT")["close"]) == [50.0, 50.0]
    assert len(_covering(tmp_path, last=SESSIONS[0])) == 3


def test_bars_retrieved_before_they_were_final_count_as_not_returned(tmp_path: Path) -> None:
    # 20:05 UTC on July 29 is after its close but before the 15-minute finalization.
    early = datetime(2026, 7, 29, 20, 5, tzinfo=UTC)
    collect_bars(_source(_daily({"MSFT": [_bar(s) for s in SESSIONS]}), early), [_unit("MSFT")],
                 root=tmp_path, clock=_clock(early))

    [loaded] = load_bar_receipts(tmp_path, [DECISION])

    assert loaded.sessions_returned("MSFT") == frozenset(SESSIONS[:2])


def test_a_duplicated_session_is_a_defect_not_a_gap(tmp_path: Path) -> None:
    collect_bars(_source(_daily({"MSFT": [_bar(SESSIONS[0]), _bar(SESSIONS[0], 101.0)]})), [_unit("MSFT")],
                 root=tmp_path, clock=_clock())
    [loaded] = load_bar_receipts(tmp_path, [DECISION])

    with pytest.raises(DataReadinessError, match="duplicated sessions"):
        daily_path_bars(loaded, "MSFT")


def test_a_missing_price_keeps_the_bar_as_unusable(tmp_path: Path) -> None:
    collect_bars(_source(_daily({"MSFT": [{**_bar(SESSIONS[0]), "c": None}]})), [_unit("MSFT")],
                 root=tmp_path, clock=_clock())
    [loaded] = load_bar_receipts(tmp_path, [DECISION])

    bars = daily_path_bars(loaded, "MSFT")

    assert len(bars) == 1 and bars["close"].isna().all()


def test_tampered_receipts_and_bodies_are_refused(tmp_path: Path) -> None:
    collect_bars(_source(_daily({"MSFT": [_bar(s) for s in SESSIONS]})), [_unit("MSFT")], root=tmp_path, clock=_clock())
    receipt_path = next((tmp_path / "bar_receipts").rglob("*.json"))
    body_path = next((tmp_path / "bodies").glob("*.json"))
    original_receipt, original_body = receipt_path.read_bytes(), body_path.read_bytes()

    receipt_path.write_bytes(original_receipt.replace(b'"complete"', b'"failed"'))
    with pytest.raises(DataReadinessError, match="does not verify"):
        load_bar_receipts(tmp_path, [DECISION])
    receipt_path.write_bytes(original_receipt)

    body_path.write_bytes(original_body.replace(b"100.0", b"101.0"))
    with pytest.raises(DataReadinessError, match="does not match its record"):
        load_bar_receipts(tmp_path, [DECISION])

    body_path.unlink()
    with pytest.raises(DataReadinessError, match="body is missing"):
        load_bar_receipts(tmp_path, [DECISION])


def test_minute_receipts_are_matched_to_their_session(tmp_path: Path) -> None:
    for session in SESSIONS[:2]:
        collect_bars(_source(_daily({"MSFT": [_minute(session)]})), [_unit("MSFT", timeframe="1Min", last=session)],
                     root=tmp_path, clock=_clock())

    receipts = load_bar_receipts(tmp_path, [DECISION])
    [first] = complete_bar_receipts(receipts, decision_session=DECISION, symbol="MSFT",
                                    first_session=SESSIONS[0], last_session=SESSIONS[0], timeframe="1Min")

    assert first.unit.start == datetime(2026, 7, 27, 13, 30, tzinfo=UTC)
    assert first.sessions_returned("MSFT") == frozenset({SESSIONS[0]})


def test_units_refuse_malformed_requests() -> None:
    with pytest.raises(ValueError, match="sorted"):
        OutcomeBarUnit(DECISION, DECISION, SESSIONS[-1], ("MSFT", "AAPL"))
    with pytest.raises(ValueError, match="XNYS session"):
        OutcomeBarUnit(date(2026, 7, 25), date(2026, 7, 25), SESSIONS[-1], ("MSFT",))
    with pytest.raises(ValueError, match="starts on its decision session"):
        OutcomeBarUnit(DECISION, SESSIONS[0], SESSIONS[-1], ("MSFT",))
    with pytest.raises(ValueError, match="one session"):
        OutcomeBarUnit(DECISION, SESSIONS[0], SESSIONS[1], ("MSFT",), "1Min")


def test_a_memory_guard_stop_ends_the_run_without_a_receipt(tmp_path: Path) -> None:
    with mock.patch.object(outcome_bars, "assert_system_memory_available", side_effect=MemoryBudgetError("full")), \
            pytest.raises(MemoryBudgetError):
        collect_bars(_source(_daily({})), [_unit("MSFT")], root=tmp_path, clock=_clock())

    assert not (tmp_path / "bar_receipts").exists()


def _action_unit() -> CorporateActionUnit:
    return CorporateActionUnit(DECISION, "TWTR", date(2026, 7, 24), date(2026, 10, 1))


def _actions(first: dict[str, object], second: dict[str, object]) -> Responder:
    def respond(_url: str, params: dict[str, Any]) -> tuple[int, object]:
        if params.get("page_token") is None:
            return 200, {"corporate_actions": first, "next_page_token": "second"}
        return 200, {"corporate_actions": second, "next_page_token": None}

    return respond


def test_corporate_actions_are_collected_and_reloaded(tmp_path: Path) -> None:
    merger = {"id": "merger-1", "acquiree_symbol": "TWTR", "acquiree_cusip": "90184L102", "rate": "54.20",
              "effective_date": "2022-10-28", "process_date": "2022-10-28"}

    [receipt] = collect_corporate_actions(_source(_actions({"cash_mergers": [merger]}, {})), [_action_unit()],
                                          root=tmp_path, clock=_clock())
    [loaded] = load_action_receipts(tmp_path, [DECISION])

    assert receipt["status"] == "complete" and len(receipt["pages"]) == 2  # type: ignore[arg-type]
    assert loaded.actions == {"cash_mergers": (merger,)}


def test_an_action_repeated_across_pages_is_refused(tmp_path: Path) -> None:
    merger = {"id": "merger-1", "acquiree_symbol": "TWTR", "effective_date": "2022-10-28"}
    collect_corporate_actions(_source(_actions({"cash_mergers": [merger]}, {"cash_mergers": [merger]})),
                              [_action_unit()], root=tmp_path, clock=_clock())

    with pytest.raises(DataReadinessError, match="repeats an action"):
        load_action_receipts(tmp_path, [DECISION])


def test_a_refused_corporate_action_request_is_a_failed_receipt(tmp_path: Path) -> None:
    [receipt] = collect_corporate_actions(
        _source(lambda _url, _params: (500, {"message": "error"})), [_action_unit()], root=tmp_path, clock=_clock()
    )

    assert receipt["status"] == "failed"
    [loaded] = load_action_receipts(tmp_path, [DECISION])
    assert not loaded.complete and loaded.actions == {}


def test_monitoring_steps_wait_for_each_other_and_report_a_held_lease(tmp_path: Path) -> None:
    held = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with monitoring_lease("register", runtime_dir=tmp_path):
            held.set()
            release.wait(5)

    holder = threading.Thread(target=hold)
    holder.start()
    held.wait(5)
    with pytest.raises(MonitoringBusyError, match="register"):
        with monitoring_lease("collect", runtime_dir=tmp_path, wait_seconds=0.2):
            pass
    release.set()
    holder.join()

    # A lock that times out inside a step is its own failure, not a busy lease.
    with pytest.raises(LockTimeout), monitoring_lease("collect", runtime_dir=tmp_path, wait_seconds=0.2):
        with file_lock(tmp_path / "inner"), file_lock(tmp_path / "inner", timeout=0.0):
            pass
