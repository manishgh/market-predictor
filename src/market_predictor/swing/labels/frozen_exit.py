"""Compile the two frozen research exits without admitting their source evidence."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any, Literal

import exchange_calendars as xcals
import pandas as pd
from pydantic import Field

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.execution_policy import executable_fill_price
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.contracts.holding_accounting import (
    AccountingGap,
    CorporateActionEvent,
    EvidenceReference,
    ExecutionEvent,
    HoldingContract,
    HoldingSpecification,
    KnownMark,
    Timestamp,
)
from market_predictor.swing.contracts.research import SwingResearchContract
from market_predictor.swing.contracts.trade_simulation import SimulatedHolding, TradeSimulationContext
from market_predictor.swing.evaluation.trade_simulation import simulate_ordinary_sales
from market_predictor.swing.labels.holding_paths import validate_outcome_observations

ExitPolicy = Literal["target_stop_ten_session_timeout", "stop_ten_session_timeout"]


class DecisionAtr(HoldingContract):
    """An independently bound decision-time ATR, expressed in this lot's raw dollars."""

    decision_id: str = Field(min_length=1)
    security_id: str = Field(min_length=1)
    value: float = Field(gt=0)
    observation_end_at: Timestamp
    available_at: Timestamp
    timeframe: Literal["1Day"] = "1Day"
    lookback_bars: Literal[14] = 14
    price_basis: Literal["raw_with_no_adjustment"] = "raw_with_no_adjustment"
    availability_basis: Literal["completed_session_historical_proxy"] = "completed_session_historical_proxy"
    evidence: EvidenceReference


class FrozenExitResult(HoldingContract):
    exit_policy: ExitPolicy
    policy_sha256: str
    input_specification_sha256: str
    decision_atr_sha256: str
    exit_reason: Literal["stop", "target", "timeout"] | None = None
    exit_timestamp: Timestamp | None = None
    exit_clock_basis: Literal["session_open_gap", "session_close_research_convention", "tenth_session_close"] | None = None
    target_price: float | None = None
    stop_price: float
    simulation: SimulatedHolding | None = None
    missing_reasons: tuple[AccountingGap, ...] = ()
    source_admitted: Literal[False] = False
    economic_eligible: Literal[False] = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _observations(
    observations: pd.DataFrame, evidence: Mapping[date, EvidenceReference],
    specification: HoldingSpecification, sessions: tuple[date, ...],
) -> tuple[pd.DataFrame, pd.Series, dict[date, EvidenceReference]]:
    _require(observations.columns.is_unique, "exit observations require unique columns")
    if observations.empty:
        _require(not evidence, "exit evidence has no observation")
        return observations, pd.Series(dtype=bool), {}
    _require(observations.index.name == "session_date_et" and observations.index.is_unique
        and all(type(day) is date for day in observations.index)
        and set(observations.index).issubset(sessions), "exit observation index differs from the exact horizon")
    for field, expected in {"security_id": specification.security_id, "source": "alpaca",
            "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw"}.items():
        _require(field in observations and bool(observations[field].eq(expected).fillna(False).all()),
            f"exit observation {field} differs from the raw SIP lot")
    _require("ticker" in observations and observations.ticker.map(
        lambda value: isinstance(value, str) and bool(value.strip())).all()
        and observations.ticker.nunique() == 1, "exit observations require one source ticker")
    _require({"outcome_observation_valid", "ingested_at_utc"}.issubset(observations),
        "exit observations require independent validation and retrieval clocks")
    admitted = observations.outcome_observation_valid.eq(True).fillna(False)
    validated = validate_outcome_observations(observations.reset_index()).set_index("session_date_et")
    _require(all(type(day) is date for day in evidence) and set(evidence).issubset(observations.index),
        "exit evidence lies outside supplied observations")
    references: dict[date, EvidenceReference] = {}
    for day, reference in evidence.items():
        _require(type(reference) is EvidenceReference, "exit price evidence cannot be a simulation assumption")
        reference = EvidenceReference.model_validate_json(reference.model_dump_json())
        _require(reference.record_locator == f"security_id={specification.security_id};session_date={day}"
            and reference.available_at is None
            and reference.retrieved_at == pd.Timestamp(observations.loc[day, "ingested_at_utc"]),
            "exit evidence identity, session or retrieval clock differs")
        references[day] = reference
    _require(len({(ref.reference, ref.artifact_sha256, ref.interpretation_policy_sha256)
        for ref in references.values()}) <= 1, "exit observations splice raw source authorities")
    return validated, admitted, references


def compile_frozen_exit(
    *, specification: HoldingSpecification, decision_session: date, decision_time: datetime,
    decision_atr: DecisionAtr, observations: pd.DataFrame, evidence: Mapping[date, EvidenceReference],
    strategy: StrategyContract, research: SwingResearchContract, exit_policy: ExitPolicy,
    simulation: TradeSimulationContext,
) -> FrozenExitResult:
    """Add a supported ordinary-share exit and replay existing sale settlement.

    The caller owns source admission and ATR construction. The input lot must be
    unsimulated and contain no execution. Existing corporate facts and marks are
    retained. An ownership replacement before a resolved exit is unavailable;
    neither successor prices nor replacement-share barrier geometry are inferred.
    Daily intrabar fills use the session close as an explicit research clock, not
    an observed execution timestamp. Missing evidence after a proven sale does not
    erase that sale; remaining corporate claims still require their own marks.
    """
    specification = HoldingSpecification.model_validate_json(specification.model_dump_json())
    decision_atr = DecisionAtr.model_validate_json(decision_atr.model_dump_json())
    strategy = StrategyContract.model_validate_json(strategy.model_dump_json())
    research = SwingResearchContract.model_validate_json(research.model_dump_json())
    research.assert_strategy_matches(strategy)
    policy = strategy.swing
    _require(exit_policy in research.exit_policies, "exit policy is outside the frozen research contract")
    _require((policy.target_atr_multiple, policy.stop_atr_multiple, policy.atr_lookback_bars,
        policy.atr_timeframe, policy.same_bar_barrier_resolution) == (3.0, 1.5, 14, "1Day", "stop_first"),
        "exit geometry differs from the frozen daily ATR policy")
    _require(specification.research_contract_sha256 == research.sha256()
        and specification.cost_prepaid_fraction == research.base_round_trip_cost_bps / 10_000
        and specification.currency == "USD", "exit lot research identity, currency or cost differs")
    _require(not any(isinstance(event, ExecutionEvent) for event in specification.events),
        "exit compiler requires an unsimulated lot without an execution")
    calendar = xcals.get_calendar("XNYS")
    _require(type(decision_session) is date and calendar.is_session(decision_session.isoformat()),
        "exit decision must be an XNYS session")
    _require(decision_time == swing_prediction_cutoffs(pd.Series([decision_session])).iloc[0],
        "exit decision cutoff differs from the canonical cutoff")
    sessions = tuple(day.date() for day in calendar.sessions_window(pd.Timestamp(decision_session), 11)[1:])
    closes = tuple(calendar.session_close(day.isoformat()).to_pydatetime() for day in sessions)
    _require(specification.session_end_timestamps == closes
        and specification.initial_entry_timestamp == calendar.session_open(sessions[0].isoformat()).to_pydatetime(),
        "exit lot must use the exact next-open and ten XNYS closes")
    _require(decision_atr.decision_id == specification.decision_id
        and decision_atr.security_id == specification.security_id
        and decision_atr.price_basis == specification.price_basis
        and decision_atr.observation_end_at == calendar.session_close(decision_session.isoformat()).to_pydatetime()
        and decision_atr.observation_end_at <= decision_atr.available_at <= decision_time,
        "exit ATR identity, raw price basis or causal clock differs")
    _require(type(decision_atr.evidence) is EvidenceReference
        and (decision_atr.evidence.available_at is None
            or decision_atr.evidence.available_at <= decision_atr.available_at),
        "exit ATR cannot claim unsupported source availability")
    stop = specification.initial_entry_price - policy.stop_atr_multiple * decision_atr.value
    target = specification.initial_entry_price + policy.target_atr_multiple * decision_atr.value
    _require(stop > 0, "exit stop must be positive in the raw price basis")
    identity: dict[str, Any] = dict(exit_policy=exit_policy,
        policy_sha256=json_sha256({"research": research.sha256(), "strategy": strategy.sha256(), "exit": exit_policy}),
        input_specification_sha256=json_sha256(specification.model_dump(mode="json")),
        decision_atr_sha256=json_sha256(decision_atr.model_dump(mode="json")),
        target_price=target if exit_policy == "target_stop_ten_session_timeout" else None, stop_price=stop)
    bars, admitted, references = _observations(observations, evidence, specification, sessions)
    marks = {(mark.position_id, mark.mark_at): mark for mark in specification.marks}
    for index, day in enumerate(sessions):
        gap = None
        if day not in bars.index:
            gap = "exit_observation_missing"
        elif not bool(admitted.loc[day]) or not bool(bars.loc[day, "outcome_observation_valid"]):
            gap = "exit_observation_invalid"
        elif day not in references:
            gap = "exit_evidence_missing"
        if gap is not None:
            return FrozenExitResult(**identity, missing_reasons=(AccountingGap(code=gap,
                reference_id=f"{specification.security_id}:{day}"),))
        row = bars.loc[day]
        if index == 0:
            _require(float(row["open"]) == specification.initial_entry_price
                and references[day] in specification.entry_evidence, "exit entry differs from bound raw open/evidence")
        reason: Literal["stop", "target", "timeout"] | None = None
        if float(row["low"]) <= stop:
            reason = "stop"
        elif exit_policy == "target_stop_ten_session_timeout" and float(row["high"]) >= target:
            reason = "target"
        elif index == 9:
            reason = "timeout"
        at = closes[index]
        clock: Literal["session_open_gap", "session_close_research_convention", "tenth_session_close"] = (
            "tenth_session_close" if reason == "timeout" else "session_close_research_convention")
        if (reason == "stop" and float(row["open"]) <= stop
                or reason == "target" and float(row["open"]) >= target):
            at = calendar.session_open(day.isoformat()).to_pydatetime()
            clock = "session_open_gap"
        unsupported = [event for event in specification.events if isinstance(event, CorporateActionEvent)
            and (event.effective_at is None or event.treatment == "replace" and event.effective_at <= at)]
        if unsupported:
            return FrozenExitResult(**identity, missing_reasons=tuple(AccountingGap(
                code="exit_ownership_transition_unsupported", reference_id=event.event_id) for event in unsupported))
        if reason is None:
            mark = marks.get((specification.initial_position_id, closes[index]))
            _require(isinstance(mark, KnownMark) and mark.currency == "USD"
                and mark.value_per_unit == float(row["close"]) and references[day] in mark.evidence,
                "exit pre-sale mark differs from bound raw close/evidence")
            continue
        fill = executable_fill_price(outcome={"stop": "stop_first", "target": "target_first", "timeout": "timeout"}[reason],
            target_price=target, stop_price=stop, trigger_open=float(row["open"]), final_price=float(row["close"]))
        sale = ExecutionEvent(event_id=f"{specification.decision_id}:{exit_policy}:exit", effective_at=at,
            order=max((event.order for event in specification.events if event.effective_at == at and event.order is not None),
                default=-1) + 1, evidence=(references[day], decision_atr.evidence),
            position_id=specification.initial_position_id, security_id=specification.security_id,
            fraction_of_owned=1.0, price_per_unit=fill, currency="USD",
            proceeds_id=f"{specification.decision_id}:{exit_policy}:proceeds", reason="managed_exit")
        compiled = HoldingSpecification.model_validate({**specification.model_dump(), "policy": "managed",
            "events": (*specification.events, sale)})
        replay = simulate_ordinary_sales(compiled, simulation)
        gaps = tuple(AccountingGap(code=code, reference_id=reference) for code, reference in sorted({
            (gap.code, gap.reference_id) for snapshot in replay.outcome.snapshots for gap in snapshot.gaps}))
        return FrozenExitResult(**identity, exit_reason=reason, exit_timestamp=at, exit_clock_basis=clock,
            simulation=replay, missing_reasons=gaps)
    raise AssertionError("the tenth verified session must resolve timeout")
