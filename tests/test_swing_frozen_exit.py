"""Synthetic raw-share paths for the two frozen research exit policies."""
from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.holding_accounting import (
    CorporateActionEvent,
    ExecutionEvent,
    HoldingSpecification,
    KnownMark,
    PositionLeg,
)
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.labels.frozen_exit import DecisionAtr, compile_frozen_exit
from tests.test_swing_ordinary_holding import CALENDAR, RETRIEVED, inputs


@pytest.fixture
def case() -> dict[str, Any]:
    base = inputs.__wrapped__()
    research = load_swing_research_contract(Path("configs/swing_research.toml"))
    strategy = load_strategy_contract(Path("configs/edge_rebuild_strategy_contract.toml"))
    decision, days = base["decision"], base["sessions"]
    observations = base["observations"].assign(open=100.0, high=101.0, low=99.0, close=100.0)
    closes = tuple(CALENDAR.session_close(day.isoformat()).to_pydatetime() for day in days)
    spec = HoldingSpecification(
        research_contract_sha256=research.sha256(), decision_id=decision["decision_id"],
        security_id=decision["security_id"], sector=decision["sector"], initial_position_id="ordinary-A",
        initial_entry_price=100.0, initial_entry_timestamp=CALENDAR.session_open(days[0].isoformat()).to_pydatetime(),
        price_basis="raw_with_no_adjustment", currency="USD", entry_evidence=(base["evidence"][days[0]],),
        session_end_timestamps=closes, cost_prepaid_fraction=0.002, policy="fixed_horizon",
        marks=tuple(KnownMark(position_id="ordinary-A", mark_at=close, value_per_unit=100.0,
            currency="USD", evidence=(base["evidence"][day],)) for day, close in zip(days, closes, strict=True)),
    )
    atr = DecisionAtr(decision_id=decision["decision_id"], security_id=decision["security_id"], value=2.0,
        observation_end_at=CALENDAR.session_close(decision["session_date_et"].isoformat()).to_pydatetime(),
        available_at=decision["decision_time_utc"], evidence=base["evidence"][days[0]].model_copy(update={
            "reference": "test-only/decision-atr.json", "record_locator": "raw ATR14 at decision"}))
    return {"specification": spec, "decision_session": decision["session_date_et"],
        "decision_time": decision["decision_time_utc"], "decision_atr": atr,
        "observations": observations, "evidence": base["evidence"], "strategy": strategy,
        "research": research, "exit_policy": "target_stop_ten_session_timeout", "simulation": base["simulation"]}


def _sale(result: Any) -> ExecutionEvent:
    assert result.simulation is not None
    return next(event for event in result.simulation.specification.events if isinstance(event, ExecutionEvent))


@pytest.mark.parametrize("policy", ["target_stop_ten_session_timeout", "stop_ten_session_timeout"])
def test_timeout_preserves_sources_cost_once_and_next_open_cash(case: dict[str, Any], policy: str) -> None:
    case["exit_policy"] = policy
    original = case["specification"]
    observations = case["observations"].copy(deep=True)
    result = compile_frozen_exit(**case)
    assert result.exit_reason == "timeout" and result.exit_clock_basis == "tenth_session_close"
    assert _sale(result).price_per_unit == 100.0
    assert result.simulation.specification.marks[:10] == original.marks
    assert result.simulation.outcome.snapshots[-1].net_return == pytest.approx(-0.002)
    assert result.simulation.outcome.snapshots[-1].available_cash == 0.0
    assert result.simulation.next_open_settlement.available_cash == pytest.approx(1.0)
    assert not result.source_admitted and not result.economic_eligible
    assert result.missing_reasons == ()
    pd.testing.assert_frame_equal(observations, case["observations"])


def test_target_and_uncapped_policy_are_distinct(case: dict[str, Any]) -> None:
    day = case["observations"].index[2]
    case["observations"].loc[day, "high"] = 108.0
    target = compile_frozen_exit(**case)
    stop_only = compile_frozen_exit(**{**case, "exit_policy": "stop_ten_session_timeout"})
    assert target.exit_reason == "target" and _sale(target).price_per_unit == 106.0
    assert target.exit_clock_basis == "session_close_research_convention"
    assert target.exit_timestamp == CALENDAR.session_close(day.isoformat())
    assert stop_only.exit_reason == "timeout" and stop_only.target_price is None
    assert target.policy_sha256 != stop_only.policy_sha256


@pytest.mark.parametrize(("open_price", "high", "low", "reason", "fill", "clock"), [
    (94.0, 101.0, 93.0, "stop", 94.0, "session_open_gap"),
    (110.0, 111.0, 99.0, "target", 106.0, "session_open_gap"),
    (100.0, 107.0, 96.0, "stop", 97.0, "session_close_research_convention"),
    (110.0, 111.0, 96.0, "stop", 97.0, "session_close_research_convention"),
])
def test_gap_and_collision_follow_canonical_fill_policy(
    case: dict[str, Any], open_price: float, high: float, low: float, reason: str, fill: float, clock: str,
) -> None:
    day = case["observations"].index[1]
    case["observations"].loc[day, ["open", "high", "low"]] = [open_price, high, low]
    result = compile_frozen_exit(**case)
    assert result.exit_reason == reason and _sale(result).price_per_unit == fill
    assert result.exit_clock_basis == clock
    expected = CALENDAR.session_open(day.isoformat()) if clock == "session_open_gap" else CALENDAR.session_close(day.isoformat())
    assert result.exit_timestamp == expected


@pytest.mark.parametrize("problem", ["missing", "invalid", "evidence", "independent_invalid"])
def test_unavailable_before_exit_is_never_a_timeout(case: dict[str, Any], problem: str) -> None:
    day = case["observations"].index[1]
    if problem == "missing":
        case["observations"] = case["observations"].drop(day)
        del case["evidence"][day]
    elif problem == "invalid":
        case["observations"].loc[day, "volume"] = 0.0
    elif problem == "evidence":
        del case["evidence"][day]
    else:
        case["observations"].loc[day, "outcome_observation_valid"] = False
    result = compile_frozen_exit(**case)
    assert result.exit_reason is None and result.simulation is None
    assert result.missing_reasons[0].reference_id.endswith(str(day))


def test_missing_after_proven_sale_does_not_erase_exit(case: dict[str, Any]) -> None:
    days = tuple(case["observations"].index)
    case["observations"].loc[days[1], "low"] = 96.0
    case["observations"] = case["observations"].iloc[:2].copy()
    case["evidence"] = {day: case["evidence"][day] for day in days[:2]}
    result = compile_frozen_exit(**case)
    assert result.exit_reason == "stop" and result.missing_reasons == ()
    assert result.simulation.outcome.snapshots[-1].net_return == pytest.approx(-0.032)


@pytest.mark.parametrize("field,value", [("security_id", "wrong"), ("adjustment", "all"),
    ("price_feed", "iex"), ("timeframe", "15Min"), ("source", "other")])
def test_raw_source_identity_cannot_change(case: dict[str, Any], field: str, value: str) -> None:
    case["observations"].loc[case["observations"].index[0], field] = value
    with pytest.raises(DataReadinessError, match="raw SIP"):
        compile_frozen_exit(**case)


@pytest.mark.parametrize("field,value", [("decision_id", "wrong"), ("security_id", "wrong")])
def test_atr_must_belong_to_decision_and_security(case: dict[str, Any], field: str, value: str) -> None:
    case["decision_atr"] = case["decision_atr"].model_copy(update={field: value})
    with pytest.raises(DataReadinessError, match="ATR identity"):
        compile_frozen_exit(**case)


def test_future_atr_and_changed_raw_basis_reject(case: dict[str, Any]) -> None:
    atr = case["decision_atr"]
    case["decision_atr"] = atr.model_copy(update={"available_at": case["decision_time"] + timedelta(seconds=1)})
    with pytest.raises(DataReadinessError, match="ATR identity"):
        compile_frozen_exit(**case)
    case["decision_atr"] = atr.model_copy(update={"price_basis": "all_adjusted"})
    with pytest.raises(ValueError):
        compile_frozen_exit(**case)


@pytest.mark.parametrize("change", ["entry", "horizon", "policy", "mark", "retrieval", "research"])
def test_mismatched_lot_or_evidence_rejects(case: dict[str, Any], change: str) -> None:
    spec = case["specification"]
    if change == "entry":
        case["specification"] = spec.model_copy(update={"initial_entry_price": 101.0})
    elif change == "horizon":
        case["specification"] = spec.model_copy(update={"session_end_timestamps": (*spec.session_end_timestamps[:-1],
            spec.session_end_timestamps[-1] + timedelta(days=1))})
    elif change == "policy":
        case["exit_policy"] = "fixed_horizon"
    elif change == "mark":
        marks = (spec.marks[0].model_copy(update={"value_per_unit": 101.0}), *spec.marks[1:])
        case["specification"] = spec.model_copy(update={"marks": marks})
    elif change == "retrieval":
        day = case["observations"].index[0]
        case["evidence"][day] = case["evidence"][day].model_copy(update={"retrieved_at": RETRIEVED + timedelta(seconds=1)})
    else:
        case["specification"] = spec.model_copy(update={"research_contract_sha256": "0" * 64})
    with pytest.raises(DataReadinessError):
        compile_frozen_exit(**case)


def _replacement(case: dict[str, Any], index: int) -> CorporateActionEvent:
    day = case["observations"].index[index]
    return CorporateActionEvent(event_id="replace-A", effective_at=CALENDAR.session_open(day.isoformat()).to_pydatetime(),
        order=0, evidence=(case["evidence"][day],), owned_position_id="ordinary-A", owned_security_id="common:A",
        treatment="replace", fractional_treatment="proportional", legs=(PositionLeg(position_id="successor",
            kind="tradable_shares", security_id="common:B", units_per_owned_unit=1.0, currency="USD"),))


def test_ownership_transition_before_sale_is_explicitly_unavailable(case: dict[str, Any]) -> None:
    case["specification"] = case["specification"].model_copy(update={"events": (_replacement(case, 1),)})
    result = compile_frozen_exit(**case)
    assert result.simulation is None
    assert result.missing_reasons[0].code == "exit_ownership_transition_unsupported"


def test_corporate_fact_after_sale_is_retained_not_invented(case: dict[str, Any]) -> None:
    replacement = _replacement(case, 4)
    case["specification"] = case["specification"].model_copy(update={"events": (replacement,)})
    case["observations"].loc[case["observations"].index[1], "low"] = 96.0
    result = compile_frozen_exit(**case)
    assert replacement in result.simulation.specification.events
    assert result.missing_reasons == ()


def test_preexisting_execution_cannot_be_reinterpreted(case: dict[str, Any]) -> None:
    result = compile_frozen_exit(**case)
    case["specification"] = result.simulation.specification
    with pytest.raises(DataReadinessError, match="without an execution"):
        compile_frozen_exit(**case)
