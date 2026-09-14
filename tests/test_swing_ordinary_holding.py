"""Bounded ordinary-path adapter tests; all numerical data are synthetic fixtures."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, ExecutionEvent, SimulationReference
from market_predictor.swing.contracts.trade_simulation import TradeSimulationContext, TradeSimulationPolicy
from market_predictor.swing.evaluation.holding_accounting import project_holding_targets, replay_holding
from market_predictor.swing.labels.holding_paths import validate_outcome_observations
from market_predictor.swing.labels.ordinary_holding import build_ordinary_holding

CALENDAR = xcals.get_calendar("XNYS")
RETRIEVED = datetime(2026, 9, 9, tzinfo=UTC)


@pytest.fixture
def inputs() -> dict[str, Any]:
    day = date(2024, 1, 2)
    sessions = tuple(session.date() for session in CALENDAR.sessions_window(pd.Timestamp(day), 11)[1:])
    rows = [{"security_id": "common:A", "ticker": "AAA", "session_date_et": session,
        "bar_start_utc": CALENDAR.session_open(session.isoformat()),
        "bar_end_utc": CALENDAR.session_close(session.isoformat()),
        "available_at_utc": RETRIEVED, "ingested_at_utc": RETRIEVED,
        "open": 100.0 + index, "high": 112.0, "low": 99.0, "close": 101.0 + index,
        "volume": 1000.0, "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw"}
        for index, session in enumerate(sessions)]
    evidence = {session: EvidenceReference(reference="test-only/raw.parquet", artifact_sha256="a" * 64,
        record_locator=f"security_id=common:A;session_date={session}", interpretation_policy_sha256="b" * 64,
        retrieved_at=RETRIEVED, available_at=None) for session in sessions}
    policy = TradeSimulationPolicy(
        schema_version="market_predictor.ordinary_trade_simulation", scope="retrospective_research_only",
        fill_source="canonical_execution_events_only", sale_proceeds_valuation="one_usd_per_executed_dollar",
        sale_proceeds_reuse="next_xnys_session_open", corporate_claims="no_inferred_payment_or_valuation",
        cost_policy="preserve_existing_prepaid_cost", label_clock="separate_retrospective_maturity_not_observed_availability",
        production_eligible=False,
    )
    context = TradeSimulationContext(policy=policy, policy_reference=EvidenceReference(
        reference="test-only/research-policy", artifact_sha256="d" * 64, record_locator="research assumption",
        interpretation_policy_sha256=policy.sha256(), retrieved_at=RETRIEVED, available_at=None))
    return {"decision": {"decision_id": "canonical-decision-A", "security_id": "common:A", "ticker": "AAA",
        "sector": "healthcare", "session_date_et": day,
        "decision_time_utc": swing_prediction_cutoffs(pd.Series([day])).iloc[0]},
        "sessions": sessions, "observations": validate_outcome_observations(pd.DataFrame(rows)).set_index("session_date_et"),
        "evidence": evidence, "research_contract_sha256": "c" * 64, "cost_prepaid_fraction": 0.002, "simulation": context}


def test_fixed_component_uses_canonical_replay_cost_once_and_no_managed(inputs: dict[str, Any]) -> None:
    before = inputs["observations"].copy(deep=True)
    result = build_ordinary_holding(**inputs)
    assert result.specification is not None and result.outcome is not None and result.targets is not None
    assert result.simulation is not None
    assert result.outcome == replay_holding(result.specification,
        simulation_policy_sha256=inputs["simulation"].policy.sha256())
    assert result.targets == project_holding_targets(result.outcome, None)
    assert result.targets.fixed_horizon_gross_return == pytest.approx(0.10)
    assert result.targets.fixed_horizon_net_return == pytest.approx(0.098)
    assert result.targets.managed_horizon_net_return is None
    assert result.targets.managed_horizon_gross_return is None
    assert result.targets.managed_exit_timestamp is None
    assert result.targets.label_available_at is None
    assert result.missing_reasons == () and result.label_eligible is False
    assert result.outcome.production_eligible is False
    assert len(result.outcome.snapshots) == 10
    assert result.outcome.snapshots[-1].tradable_value == 0.0
    assert result.outcome.snapshots[-1].unpaid_proceeds_value == pytest.approx(1.10)
    assert not result.outcome.snapshots[-1].fully_settled
    assert result.simulation.next_open_settlement.available_cash == pytest.approx(1.10)
    sale = next(event for event in result.specification.events if isinstance(event, ExecutionEvent))
    assert sale.reason == "fixed_horizon" and sale.price_per_unit == 110.0
    assert sale.effective_at == result.specification.session_end_timestamps[-1]
    assert sale.evidence == (inputs["evidence"][inputs["sessions"][-1]],)
    assert any(isinstance(ref, SimulationReference) for event in result.specification.events for ref in event.evidence)
    assert result.specification.entry_evidence[0] == inputs["evidence"][inputs["sessions"][0]]
    pd.testing.assert_frame_equal(before, inputs["observations"])


@pytest.mark.parametrize("security", ["SPY", "QQQ", "XLV"])
def test_benchmark_uses_same_function_and_canonical_identity(inputs: dict[str, Any], security: str) -> None:
    stock = build_ordinary_holding(**inputs)
    inputs["decision"] = {**inputs["decision"], "security_id": security, "ticker": security,
        "decision_id": "canonical-decision-" + security}
    inputs["observations"] = inputs["observations"].assign(security_id=security, ticker=security)
    inputs["evidence"] = {day: ref.model_copy(update={"record_locator": f"security_id={security};session_date={day}"})
        for day, ref in inputs["evidence"].items()}
    benchmark = build_ordinary_holding(**inputs)
    assert benchmark.outcome is not None and stock.outcome is not None
    assert benchmark.outcome.security_id == security
    projected = project_holding_targets(stock.outcome, None, benchmarks=(benchmark.outcome,))
    assert projected.benchmarks[0].fixed_horizon_excess_return == pytest.approx(-0.002)


@pytest.mark.parametrize(("field", "value"), [
    ("security_id", "wrong-owner"), ("ticker", "wrong-symbol"), ("price_feed", "iex"),
    ("adjustment", "all"), ("source", "other"), ("timeframe", "1Min"),
])
def test_rejects_wrong_identity_non_sip_not_raw(inputs: dict[str, Any], field: str, value: str) -> None:
    inputs["observations"].loc[inputs["sessions"][3], field] = value
    with pytest.raises(DataReadinessError, match=field):
        build_ordinary_holding(**inputs)


@pytest.mark.parametrize("index", [0, 4, 9])
@pytest.mark.parametrize("problem", ["missing", "invalid", "evidence"])
def test_entry_or_midpath_or_exit_unavailable_is_null(inputs: dict[str, Any], index: int, problem: str) -> None:
    session = inputs["sessions"][index]
    if problem == "missing":
        inputs["observations"] = inputs["observations"].drop(session)
        del inputs["evidence"][session]
    elif problem == "invalid":
        inputs["observations"].loc[session, "volume"] = 0.0
    else:
        del inputs["evidence"][session]
    result = build_ordinary_holding(**inputs)
    assert result.outcome is None and result.targets is None and result.specification is None
    prefix = "entry" if index == 0 else "path"
    suffix = "evidence_missing" if problem == "evidence" else "observation_" + problem
    assert result.missing_reasons[0].code == f"{prefix}_{suffix}"
    assert result.missing_reasons[0].reference_id == f"common:A:{session}"
    assert not result.label_eligible


def test_empty_reader_frame_reports_all_missing_sessions(inputs: dict[str, Any]) -> None:
    inputs["observations"] = pd.DataFrame()
    inputs["evidence"] = {}
    result = build_ordinary_holding(**inputs)
    assert result.outcome is None and len(result.missing_reasons) == 10


@pytest.mark.parametrize("value", [False, None, pd.NA])
def test_independent_invalid_flag_is_not_overridden(inputs: dict[str, Any], value: Any) -> None:
    inputs["observations"]["outcome_observation_valid"] = inputs["observations"]["outcome_observation_valid"].astype(object)
    inputs["observations"].loc[inputs["sessions"][4], "outcome_observation_valid"] = value
    assert build_ordinary_holding(**inputs).missing_reasons[0].code == "path_observation_invalid"


@pytest.mark.parametrize("cost", [0.0, 0.004, 0.001, float("nan")])
def test_cost_policy_cannot_change(inputs: dict[str, Any], cost: float) -> None:
    inputs["cost_prepaid_fraction"] = cost
    with pytest.raises(DataReadinessError, match="20 bps"):
        build_ordinary_holding(**inputs)


def test_exact_calendar_cannot_be_derived_from_surviving_rows(inputs: dict[str, Any]) -> None:
    sessions = inputs["sessions"]
    inputs["sessions"] = (*sessions[:4], *sessions[5:], CALENDAR.next_session(sessions[-1].isoformat()).date())
    with pytest.raises(DataReadinessError, match="exact ten"):
        build_ordinary_holding(**inputs)


@pytest.mark.parametrize("field", ["decision_time_utc", "security_id"])
def test_wrong_decision_identity_fails(inputs: dict[str, Any], field: str) -> None:
    inputs["decision"][field] = "wrong"
    with pytest.raises(DataReadinessError, match="cutoff|security_id"):
        build_ordinary_holding(**inputs)


@pytest.mark.parametrize("change", ["wrong_identity", "wrong_session", "clock", "mixed_source"])
def test_per_session_evidence_binding(inputs: dict[str, Any], change: str) -> None:
    session = inputs["sessions"][4]
    ref = inputs["evidence"][session]
    updates = {"wrong_identity": {"record_locator": f"security_id=other;session_date={session}"},
        "wrong_session": {"record_locator": "security_id=common:A;session_date=2024-01-02"},
        "clock": {"available_at": RETRIEVED}, "mixed_source": {"artifact_sha256": "e" * 64}}[change]
    inputs["evidence"][session] = ref.model_copy(update=updates)
    with pytest.raises(DataReadinessError, match="evidence|one bound"):
        build_ordinary_holding(**inputs)


class _PoisonNumber:
    def __float__(self) -> float:
        raise AssertionError("held-out numerical value accessed")


def test_future_poison_is_rejected_before_numeric_access(inputs: dict[str, Any]) -> None:
    extra = inputs["observations"].iloc[[-1]].copy()
    extra.index = pd.Index([date(2024, 5, 29)], name="session_date_et")
    for column in ("open", "high", "low", "close", "volume"):
        extra[column] = _PoisonNumber()
    inputs["observations"] = pd.concat([inputs["observations"], extra])
    with pytest.raises(DataReadinessError, match="outside exact sessions"):
        build_ordinary_holding(**inputs)


@pytest.mark.parametrize("day", [date(2019, 7, 8), date(2024, 5, 14), date(2025, 1, 2)])
def test_no_warmup_or_heldout_decisions(inputs: dict[str, Any], day: date) -> None:
    inputs["decision"]["session_date_et"] = day
    with pytest.raises(DataReadinessError, match="initial-fit"):
        build_ordinary_holding(**inputs)


def test_simulation_policy_tamper_fails(inputs: dict[str, Any]) -> None:
    context = inputs["simulation"]
    inputs["simulation"] = context.model_copy(update={"policy_reference": context.policy_reference.model_copy(
        update={"interpretation_policy_sha256": "f" * 64})})
    with pytest.raises(DataReadinessError, match="bind policy"):
        build_ordinary_holding(**inputs)


@pytest.mark.parametrize("field", ["security_id", "ticker", "source", "price_feed", "adjustment"])
def test_nullable_metadata_cannot_pass_identity_checks(inputs: dict[str, Any], field: str) -> None:
    inputs["observations"][field] = inputs["observations"][field].astype("string")
    inputs["observations"].loc[inputs["sessions"][3], field] = pd.NA
    with pytest.raises(DataReadinessError, match=field):
        build_ordinary_holding(**inputs)


def test_duplicate_session_cannot_pass_exact_path_gate(inputs: dict[str, Any]) -> None:
    inputs["observations"] = pd.concat([inputs["observations"], inputs["observations"].iloc[[4]]])
    with pytest.raises(DataReadinessError, match="duplicate"):
        build_ordinary_holding(**inputs)


def test_adapter_performs_no_source_or_network_io(inputs: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure adapter attempted source/network IO")

    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr("socket.socket.connect", forbidden)
    assert build_ordinary_holding(**inputs).outcome is not None
