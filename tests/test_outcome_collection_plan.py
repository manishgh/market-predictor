from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from market_predictor.governance.outcomes.collection_plan import FREEZE_AFTER, plan_outcome_collection
from market_predictor.governance.outcomes.repository import OutcomeRepository
from tests.test_outcome_maturation import (
    DECISION,
    LAST,
    RETRIEVED,
    TERMS,
    _collected,
    _collected_actions,
    _mature,
    _swing_bars,
)
from tests.test_outcome_repository import _intent
from tests.test_performance_monitoring import _intent_variant

AFTER_CLOSE = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
BENCHMARKS = {"QQQ", "SPY", "XLK"}


def _repository(root: Path) -> OutcomeRepository:
    repository = OutcomeRepository(root / "outcomes")
    repository.record_intent(_intent())
    return repository


def test_bars_are_requested_every_night_until_settled_then_weekly(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    receipts = tmp_path / "receipts"

    [unit] = plan_outcome_collection(repository, receipts_root=receipts, now=AFTER_CLOSE, terms=TERMS).bar_units
    assert (unit.decision_session, unit.first_session, unit.last_session) == (DECISION, DECISION, LAST)
    assert set(unit.symbols) == {"MSFT", *BENCHMARKS}

    # An unsettled receipt still leaves the bars due the next night.
    _collected(tmp_path, retrieved=datetime(2026, 8, 9, 12, tzinfo=UTC))
    assert plan_outcome_collection(repository, receipts_root=receipts, now=AFTER_CLOSE + timedelta(days=1), terms=TERMS).bar_units

    _collected(tmp_path)
    settled = plan_outcome_collection(repository, receipts_root=receipts, now=RETRIEVED + timedelta(days=1), terms=TERMS)
    weekly = plan_outcome_collection(repository, receipts_root=receipts, now=RETRIEVED + timedelta(days=8), terms=TERMS)
    assert settled.bar_units == () and len(weekly.bar_units) == 1


def test_nothing_is_collected_before_the_close_or_after_the_freeze(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    receipts = tmp_path / "receipts"
    deadline = datetime(2026, 8, 7, 20, 0, tzinfo=UTC) + timedelta(days=TERMS.grace_days)

    before = plan_outcome_collection(repository, receipts_root=receipts, now=datetime(2026, 8, 7, 19, tzinfo=UTC), terms=TERMS)
    frozen = plan_outcome_collection(repository, receipts_root=receipts, now=deadline + FREEZE_AFTER + timedelta(hours=1), terms=TERMS)

    assert before.bar_units == () and frozen.bar_units == ()


def test_an_operator_can_collect_one_frozen_intent(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    intent = _intent()
    frozen = datetime(2026, 8, 14, 20, 0, tzinfo=UTC) + FREEZE_AFTER + timedelta(days=1)

    plan = plan_outcome_collection(repository, receipts_root=tmp_path / "receipts", now=frozen, terms=TERMS,
                                   only=(intent.maturation_key, intent.decision_session_et))

    [unit] = plan.bar_units
    assert "MSFT" in unit.symbols


def test_an_interior_gap_asks_for_that_session_in_minutes(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bars = _swing_bars()
    receipts = _collected(tmp_path, bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].eq(date(2026, 7, 27)))])
    _mature(repository, receipts)

    plan = plan_outcome_collection(repository, receipts_root=receipts, now=RETRIEVED + timedelta(days=1), terms=TERMS)

    [minute] = [unit for unit in plan.bar_units if unit.timeframe == "1Min"]
    assert (minute.first_session, minute.symbols) == (date(2026, 7, 27), ("MSFT",))


def test_a_tail_gap_asks_for_the_corporate_actions_of_every_name_it_took(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    bars = _swing_bars()
    receipts = _collected(tmp_path, bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].ge(date(2026, 7, 29)))])
    _mature(repository, receipts)
    now = RETRIEVED + timedelta(days=1)

    first = plan_outcome_collection(repository, receipts_root=receipts, now=now, terms=TERMS)
    assert [(unit.symbol, unit.process_start, unit.process_end) for unit in first.action_units] == [
        ("MSFT", DECISION, now.date())
    ]

    _collected_actions(receipts, {"name_changes": [{"id": "rename-1", "old_symbol": "MSFT", "new_symbol": "MSFX",
                                                    "process_date": "2026-07-28"}]})
    # Before the deadline every name's actions are asked for again each night.
    renamed = plan_outcome_collection(repository, receipts_root=receipts, now=now, terms=TERMS)
    assert [unit.symbol for unit in renamed.action_units] == ["MSFT", "MSFX"]
    # After it, only weekly: the ticker was asked three days ago, the new name never.
    after_deadline = datetime(2026, 8, 15, 20, 0, tzinfo=UTC)
    weekly = plan_outcome_collection(repository, receipts_root=receipts, now=after_deadline, terms=TERMS)
    assert [unit.symbol for unit in weekly.action_units] == ["MSFX"]


def test_units_hold_at_most_fifty_symbols_with_their_benchmarks(tmp_path: Path) -> None:
    repository = OutcomeRepository(tmp_path / "outcomes")
    decision = datetime(2026, 7, 24, 22, 0, tzinfo=UTC)
    for index in range(60):
        repository.record_intent(_intent_variant(f"T{index:03d}", "1", probability=0.8, decision_time=decision))

    units = plan_outcome_collection(repository, receipts_root=tmp_path / "receipts", now=AFTER_CLOSE, terms=TERMS).bar_units

    assert len(units) == 2
    assert all(len(unit.symbols) <= 50 and BENCHMARKS <= set(unit.symbols) for unit in units)
    assert sum(len(set(unit.symbols) - BENCHMARKS) for unit in units) == 60
