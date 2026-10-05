"""Synthetic holdings distinguish stock trading costs from buy-and-hold benchmarks."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import exchange_calendars as xcals
import pytest
from pydantic import ValidationError

from market_predictor.swing.contracts import SECTOR_BENCHMARKS
from market_predictor.swing.contracts.holding_accounting import (
    EvidenceReference,
    ExecutionEvent,
    HoldingSpecification,
    KnownMark,
)
from market_predictor.swing.evaluation.holding_accounting import replay_holding

ENTRY = datetime(2020, 1, 2, 14, 30, tzinfo=UTC)
CALENDAR = xcals.get_calendar("XNYS")
ENDS = tuple(CALENDAR.session_close(day).to_pydatetime()
             for day in CALENDAR.sessions_in_range("2020-01-02", "2020-01-15"))
EVIDENCE = (EvidenceReference(
    reference="synthetic-price-authority", artifact_sha256="a" * 64,
    interpretation_policy_sha256="b" * 64, record_locator="synthetic constant raw-price path",
    retrieved_at=ENDS[-1] + timedelta(days=1), available_at=ENTRY,
),)


def _spec(**changes: Any) -> HoldingSpecification:
    return HoldingSpecification(**{
        "research_contract_sha256": "c" * 64, "decision_id": "synthetic-benchmark",
        "security_id": "SPY", "sector": "benchmark", "initial_position_id": "shares",
        "initial_entry_price": 100.0, "initial_entry_timestamp": ENTRY, "price_basis": "raw_with_no_adjustment",
        "currency": "USD", "entry_evidence": EVIDENCE, "session_end_timestamps": ENDS,
        "cost_prepaid_fraction": 0.0, "policy": "fixed_horizon",
        "marks": tuple(KnownMark(position_id="shares", mark_at=at, value_per_unit=100.0,
                                 currency="USD", evidence=EVIDENCE) for at in ENDS),
        **changes,
    })


@pytest.mark.parametrize("ticker", ["SPY", "QQQ", *SECTOR_BENCHMARKS])
def test_frictionless_explicit_benchmark_keeps_its_whole_initial_equity(ticker: str) -> None:
    outcome = replay_holding(_spec(security_id=ticker))
    assert all(snapshot.total_value == pytest.approx(1.0) for snapshot in outcome.snapshots)


@pytest.mark.parametrize("changes", [
    {"security_id": "stock:MSFT"},
    {"sector": "Information Technology"},
    {"policy": "managed"},
    {"cost_prepaid_fraction": 0.001},
])
def test_zero_cost_requires_the_whole_explicit_benchmark_role(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError, match="20 bps"):
        _spec(**changes)


def test_selected_stock_keeps_exactly_one_prepaid_round_trip_charge() -> None:
    outcome = replay_holding(_spec(security_id="stock:MSFT", sector="Information Technology", cost_prepaid_fraction=0.002))
    assert all(snapshot.total_value == pytest.approx(1.0) for snapshot in outcome.snapshots)
    assert all(snapshot.net_return == pytest.approx(-0.002) for snapshot in outcome.snapshots)


def test_frictionless_benchmark_cannot_contain_a_sale() -> None:
    sale = ExecutionEvent(event_id="sale", effective_at=ENDS[-1], order=0, evidence=EVIDENCE,
                          position_id="shares", security_id="SPY", fraction_of_owned=1.0,
                          price_per_unit=100.0, currency="USD", proceeds_id="sale-proceeds", reason="fixed_horizon")
    with pytest.raises(ValidationError, match="20 bps"):
        _spec(events=(sale,))
