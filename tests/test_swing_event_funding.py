from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.training.swing_types import SwingTrainingConfig
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    CorporateActionEvent,
    EvidenceReference,
    ExecutionEvent,
    HoldingSpecification,
    KnownMark,
    PaymentEvent,
    PositionLeg,
)
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.evaluation.accounting import evaluate_event_aware_swing_accounting
from market_predictor.swing.evaluation.holding_accounting import replay_holding
from market_predictor.swing.evaluation.ledger import build_event_aware_funded_swing_ledger
from market_predictor.swing.evaluation.trade_simulation import load_trade_simulation_context
from market_predictor.swing.labels.holding_accounting import build_event_aware_swing_target_row

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = load_swing_research_contract(ROOT / "configs/swing_research.toml")
CALENDAR = xcals.get_calendar("XNYS")
DAYS = tuple(CALENDAR.sessions_in_range("2024-01-02", "2024-03-01"))


def _simulation():
    path = ROOT / "configs/swing_trade_simulation.toml"
    return load_trade_simulation_context(path, expected_sha256=file_sha256(path))


def _unpaid_sale_spec(**kwargs):
    spec = _spec(**kwargs)
    return spec.model_copy(update={"events": tuple(e for e in spec.events if isinstance(e, ExecutionEvent))})


def test_simulated_labels_and_ledger_share_returns_without_broker_receipts():
    simulation = _simulation()
    fixed, managed = _unpaid_sale_spec(policy="fixed_horizon"), _unpaid_sale_spec()
    benchmarks = {role: _spec(case="benchmark", security=ticker, policy="fixed_horizon")
        for role, ticker in (("spy", "SPY"), ("qqq", "QQQ"), ("sector", "XLK"))}
    row = build_event_aware_swing_target_row(fixed, managed, benchmarks=benchmarks,
        research_contract_sha256=RESEARCH.sha256(), simulation=simulation)
    report = build_event_aware_funded_swing_ledger(_selected(managed), [managed], SwingTrainingConfig(),
        session_calendar=(DAYS[0].date().isoformat(),), research_contract_sha256=RESEARCH.sha256(),
        execution_policy="managed", simulation=simulation)
    assert row["future_net_return_10d"] == pytest.approx(0.098)
    assert report["compounded_return"] == pytest.approx(0.1 * row["future_net_return_10d"])
    assert report["final_unpaid_proceeds"] == pytest.approx(0.11)
    assert report["final_cash"] == pytest.approx(0.8998)
    assert report["fully_settled"] is False
    assert report["total_cost"] == pytest.approx(0.0002)
    assert row["trade_simulation"] == report["trade_simulation"]
    assert row["label_available_at_utc"] is None
    assert row["research_label_mature_at_utc"] == fixed.session_end_timestamps[9]
    assert row["label_eligible"] is False
    assert report["accounting_eligible"] is False


def test_simulation_consumers_retain_unit_settlement_without_extending_performance():
    simulation = _simulation()
    fixed, managed = _unpaid_sale_spec(policy="fixed_horizon"), _unpaid_sale_spec()
    benchmarks = {role: _spec(case="benchmark", security=ticker, policy="fixed_horizon")
        for role, ticker in (("spy", "SPY"), ("qqq", "QQQ"), ("sector", "XLK"))}
    row = build_event_aware_swing_target_row(fixed, managed, benchmarks=benchmarks,
        research_contract_sha256=RESEARCH.sha256(), simulation=simulation)
    report = build_event_aware_funded_swing_ledger(_selected(managed), [managed], SwingTrainingConfig(),
        session_calendar=(DAYS[0].date().isoformat(),), research_contract_sha256=RESEARCH.sha256(),
        execution_policy="managed", simulation=simulation)
    metadata = report["simulation_replays"][managed.decision_id]
    assert metadata == row["simulation_replays"]["managed"]
    assert metadata["input_specification_sha256"]
    assert len(metadata["generated_event_ids"]) == 2
    assert len(metadata["generated_mark_keys"]) == 1
    assert metadata["next_open_settlement"]["available_cash"] == pytest.approx(1.1)
    assert metadata["next_open_settlement"]["fully_settled"] is True
    assert metadata["settlement_amount_basis"] == "per_unit_initial_entry_notional_not_account_cash"
    assert len(report["daily_records"]) == 10
    assert report["final_cash"] == pytest.approx(0.8998)
    assert report["compounded_return"] == pytest.approx(0.0098)


def test_simulated_prior_sale_funds_exact_next_open_not_current_session_open():
    simulation = _simulation()
    reports = []
    for same_session in (False, True):
        first = _unpaid_sale_spec(length=19)
        at = CALENDAR.session_open(DAYS[10]).to_pydatetime() if same_session else first.session_end_timestamps[8]
        sale = first.events[0].model_copy(update={"effective_at": at})
        first = first.model_copy(update={"events": (sale,)})
        specs = [first, *(_unpaid_sale_spec(decision_index=index, length=19-index) for index in range(1, 10))]
        reports.append(build_event_aware_funded_swing_ledger(_selected(*specs), specs, SwingTrainingConfig(),
            session_calendar=tuple(day.date().isoformat() for day in DAYS[:10]),
            research_contract_sha256=RESEARCH.sha256(), execution_policy="managed", simulation=simulation))
    assert reports[0]["daily_records"][9]["funding_scale"] == 1.0
    assert reports[1]["daily_records"][9]["funding_scale"] < 1.0
    for report in reports:
        assert report["status"] == "computed"
        for row in report["daily_records"]:
            assert row["equity"] == pytest.approx(row["cash"] + row["holdings"])


def _spec(*, case="ordinary", security="issuer", decision_index=0, length=10, policy="managed"):
    entry = CALENDAR.session_open(DAYS[decision_index + 1]).to_pydatetime()
    ends = tuple(CALENDAR.session_close(day).to_pydatetime() for day in DAYS[decision_index + 1:decision_index + 1 + length])
    evidence = (EvidenceReference(reference="synthetic test only", artifact_sha256="a" * 64,
        record_locator="fixture", interpretation_policy_sha256="b" * 64,
        retrieved_at=entry + timedelta(days=100), available_at=entry),)
    events = []
    marks = []
    if case == "ordinary":
        moment = ends[9]
        events = [
            ExecutionEvent(event_id="sale", effective_at=moment, order=0, evidence=evidence,
                position_id="shares", security_id=security, fraction_of_owned=1.0,
                price_per_unit=110.0, currency="USD", proceeds_id="sale-claim",
                reason="managed_exit" if policy == "managed" else "fixed_horizon"),
            PaymentEvent(event_id="sale-payment", effective_at=moment, order=1, evidence=evidence,
                claim_id="sale-claim", amount_per_claim_unit=1.0, currency="USD", pending_proceeds_id="sale-pending"),
            CashAvailabilityEvent(event_id="sale-cash", effective_at=moment, order=2,
                evidence=evidence, payment_event_id="sale-payment", currency="USD"),
        ]
        marks = [KnownMark(position_id="shares", mark_at=end, value_per_unit=100.0,
            currency="USD", evidence=evidence) for end in ends[:9]]
    elif case in {"unpaid", "cvr"}:
        legs = [PositionLeg(position_id="claim", kind="unpaid_proceeds", security_id=security,
            units_per_owned_unit=1.0, currency="USD", face_value_per_unit=100.0)]
        if case == "cvr":
            legs.append(PositionLeg(position_id="right", kind="contingent_right", security_id=security,
                units_per_owned_unit=1.0, currency="USD", payout_cap_per_unit=35.0))
        events = [CorporateActionEvent(event_id="merger", effective_at=ends[1], order=0,
            evidence=evidence, owned_position_id="shares", owned_security_id=security,
            treatment="replace", fractional_treatment="proportional", legs=tuple(legs))]
        marks = [KnownMark(position_id="shares", mark_at=ends[0], value_per_unit=100.0,
            currency="USD", evidence=evidence)]
        if case == "cvr":
            events.extend([
                PaymentEvent(event_id="merger-payment", effective_at=ends[1], order=1,
                    evidence=evidence, claim_id="claim", amount_per_claim_unit=100.0,
                    currency="USD", pending_proceeds_id="merger-pending"),
                CashAvailabilityEvent(event_id="merger-cash", effective_at=ends[1], order=2,
                    evidence=evidence, payment_event_id="merger-payment", currency="USD"),
            ])
        else:
            marks.extend(KnownMark(position_id="claim", mark_at=end, value_per_unit=98.0,
                currency="USD", evidence=evidence) for end in ends[1:])
    elif case == "benchmark":
        marks = [KnownMark(position_id="shares", mark_at=end, value_per_unit=100.0,
            currency="USD", evidence=evidence) for end in ends]
    return HoldingSpecification(research_contract_sha256=RESEARCH.sha256(),
        decision_id=f"{decision_index}|{security}", security_id=security, sector="technology",
        initial_position_id="shares", initial_entry_price=100.0, initial_entry_timestamp=entry,
        price_basis="raw_with_no_adjustment", currency="USD", entry_evidence=evidence,
        session_end_timestamps=ends, cost_prepaid_fraction=0.002, policy=policy,
        events=tuple(events), marks=tuple(marks))


def _selected(*specs):
    return pd.DataFrame([{
        "decision_id": spec.decision_id, "security_id": spec.security_id, "sector": spec.sector,
        "session_date_et": CALENDAR.previous_session(spec.initial_entry_timestamp.date()).date().isoformat(),
        "decision_group_id": CALENDAR.previous_session(spec.initial_entry_timestamp.date()).date().isoformat(),
        "primary_benchmark": "XLK",
    } for spec in specs])


def _ledger(specs, *, count=1, cost=0.0):
    return build_event_aware_funded_swing_ledger(
        _selected(*specs), specs, SwingTrainingConfig(),
        session_calendar=tuple(day.date().isoformat() for day in DAYS[:count]),
        research_contract_sha256=RESEARCH.sha256(), execution_policy="managed", additional_round_trip_cost=cost,
    )


def test_ordinary_event_lot_reconciles_with_funded_cash_and_once_only_cost():
    spec = _spec()
    lot = replay_holding(spec)
    ledger = _ledger([spec])
    assert ledger["status"] == "computed"
    assert ledger["fully_settled"] is True
    assert ledger["final_cash"] == pytest.approx(1.0 + 0.1 * lot.snapshots[9].net_return)
    assert ledger["total_cost"] == pytest.approx(0.0002)
    assert ledger["average_daily_turnover"] == pytest.approx((0.1 + 0.11 / 0.9998) / 10)
    assert ledger["maximum_gross_exposure"] == pytest.approx(0.11 / 1.0098)
    assert _ledger([spec], cost=0.002)["final_cash"] == pytest.approx(1.0096)
    assert ledger["accounting_eligible"] is False


def test_cash_plus_unknown_right_does_not_turn_cap_into_equity_or_fund_later_entries():
    spec = _spec(case="cvr", length=12)
    later = _spec(decision_index=2)
    ledger = _ledger([spec, later], count=3)
    assert ledger["status"] == "valuation_unavailable"
    assert ledger["known_cash_at_block"] == pytest.approx(0.9998)
    assert ledger["funded_trades"] == 1
    assert ledger["selected_trades"] == 2
    assert ledger["compounded_return"] is None
    assert ledger["daily_returns"] is None
    assert ledger["gaps"][0]["component_marks_per_entry_unit"]["contingent"] is None


def test_known_receivable_remains_nonspendable_at_horizon_and_reconciles_nav():
    ledger = _ledger([_spec(case="unpaid")])
    assert ledger["status"] == "computed"
    assert ledger["fully_settled"] is False
    assert ledger["final_cash"] == pytest.approx(0.8998)
    assert ledger["final_unpaid_proceeds"] == pytest.approx(0.098)
    assert ledger["final_holdings"] == pytest.approx(0.098)
    assert ledger["compounded_return"] == pytest.approx(-0.0022)
    assert ledger["residual_decision_ids"] == ["0|issuer"]
    for row in ledger["daily_records"]:
        assert row["equity"] == pytest.approx(row["cash"] + row["holdings"])


def test_residuals_require_observations_beyond_lot_horizon_through_portfolio_end():
    with pytest.raises(DataReadinessError, match="residual claims"):
        _ledger([_spec(case="unpaid"), _spec(decision_index=1)], count=2)


def test_actual_preopen_cash_can_fund_entry_but_intraday_release_cannot():
    original = _spec(case="unpaid", length=19)
    original = original.model_copy(update={"marks": tuple(
        mark.model_copy(update={"value_per_unit": 100.0}) for mark in original.marks)})
    evidence = original.entry_evidence
    tenth_open = CALENDAR.session_open(DAYS[10]).to_pydatetime()
    results = []
    for seconds in (-1, 1):
        moment = tenth_open + timedelta(seconds=seconds)
        payment = PaymentEvent(event_id="payment", effective_at=moment, order=0, evidence=evidence,
            claim_id="claim", amount_per_claim_unit=100.0, currency="USD", pending_proceeds_id="pending")
        release = CashAvailabilityEvent(event_id="available", effective_at=moment, order=1,
            evidence=evidence, payment_event_id="payment", currency="USD")
        first = original.model_copy(update={"events": (*original.events, payment, release)})
        specs = [first, *(_spec(decision_index=index) for index in range(1, 10))]
        results.append(_ledger(specs, count=10))
    assert results[0]["daily_records"][9]["funding_scale"] == 1.0
    assert results[1]["daily_records"][9]["funding_scale"] < 1.0
    for report in results:
        assert report["status"] == "computed"
        assert report["final_unpaid_proceeds"] == 0.0
        for row in report["daily_records"]:
            assert row["equity"] == pytest.approx(row["cash"] + row["holdings"])


def test_prior_session_sale_proceeds_fund_next_open_only_after_actual_availability():
    original = _spec(length=19)
    tenth_open = CALENDAR.session_open(DAYS[10]).to_pydatetime()
    results = {}
    for case in ("prior_preopen", "prior_intraday", "same_session"):
        sale_time = original.session_end_timestamps[8] if case != "same_session" else tenth_open + timedelta(seconds=1)
        release_time = tenth_open + timedelta(seconds=-1 if case == "prior_preopen" else 1)
        sale, payment, release = original.events
        events = (sale.model_copy(update={"effective_at": sale_time}),
            payment.model_copy(update={"effective_at": release_time}),
            release.model_copy(update={"effective_at": release_time}))
        marks = original.marks + (KnownMark(position_id="sale-claim",
            mark_at=original.session_end_timestamps[8], value_per_unit=1.0,
            currency="USD", evidence=original.entry_evidence),)
        first = original.model_copy(update={"events": events, "marks": marks})
        results[case] = _ledger([first, *(_spec(decision_index=index) for index in range(1, 10))], count=10)
    assert results["prior_preopen"]["daily_records"][9]["funding_scale"] == 1.0
    assert results["prior_intraday"]["daily_records"][9]["funding_scale"] < 1.0
    assert results["same_session"]["daily_records"][9]["funding_scale"] < 1.0
    for report in results.values():
        assert report["status"] == "computed"
        assert report["fully_settled"] is True
        for row in report["daily_records"]:
            assert row["equity"] == pytest.approx(row["cash"] + row["holdings"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "sector", "policy", "hash"])
def test_selection_cannot_drop_or_substitute_event_outcomes(mutation):
    spec = _spec()
    selected = _selected(spec)
    specs = [spec]
    if mutation == "missing":
        specs = []
    elif mutation == "duplicate":
        specs.append(spec)
    elif mutation == "sector":
        selected["sector"] = "healthcare"
    elif mutation == "policy":
        specs = [_spec(policy="fixed_horizon")]
    else:
        specs = [spec.model_copy(update={"research_contract_sha256": "d" * 64})]
    with pytest.raises(DataReadinessError):
        build_event_aware_funded_swing_ledger(selected, specs, SwingTrainingConfig(),
            session_calendar=(DAYS[0].date().isoformat(),), research_contract_sha256=RESEARCH.sha256(),
            execution_policy="managed")


def test_label_projection_uses_same_economics_and_keeps_unknown_target_nullable():
    benchmarks = {name: _spec(case="benchmark", security=ticker, policy="fixed_horizon")
        for name, ticker in (("spy", "SPY"), ("qqq", "QQQ"), ("sector", "XLK"))}
    row = build_event_aware_swing_target_row(_spec(policy="fixed_horizon"), _spec(),
        benchmarks=benchmarks, research_contract_sha256=RESEARCH.sha256())
    assert row["future_net_return_10d"] == pytest.approx(0.098)
    assert row["future_excess_return_10d_vs_spy"] == pytest.approx(0.098)
    assert row["label_values_available"] is True
    assert row["label_eligible"] is False
    blocked = build_event_aware_swing_target_row(_spec(case="cvr", policy="fixed_horizon"), _spec(case="cvr"),
        benchmarks=benchmarks, research_contract_sha256=RESEARCH.sha256())
    assert blocked["future_net_return_10d"] is None
    assert blocked["label_values_available"] is False


@pytest.mark.parametrize("case,expected", [("ordinary", 0.0098), ("unpaid", -0.0022)])
def test_event_accounting_compares_total_component_nav_not_cash_alone(case, expected):
    spec = _spec(case=case)
    report = evaluate_event_aware_swing_accounting(_selected(spec), (spec,),
        {ticker: _spec(case="benchmark", security=ticker, policy="fixed_horizon") for ticker in ("SPY", "QQQ", "XLK")},
        config=SwingTrainingConfig(),
        strategy_contract=load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml"),
        research_contract=RESEARCH, session_calendar=(DAYS[0].date().isoformat(),))
    assert report["base_ledger"]["compounded_return"] == pytest.approx(expected)
    assert report["price_basis_status"] == "event_source_admission_pending"
    assert report["eligible"] is False


def test_event_accounting_does_not_score_unknown_right():
    spec = _spec(case="cvr")
    report = evaluate_event_aware_swing_accounting(_selected(spec), (spec,),
        {ticker: _spec(case="benchmark", security=ticker, policy="fixed_horizon") for ticker in ("SPY", "QQQ", "XLK")},
        config=SwingTrainingConfig(),
        strategy_contract=load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml"),
        research_contract=RESEARCH, session_calendar=(DAYS[0].date().isoformat(),))
    assert report["status"] == "valuation_unavailable"
    assert report["comparisons"] is None
    assert report["summary"]["economic_conditions_passed"] is False


@pytest.mark.parametrize("unknown_right", [False, True])
def test_simulation_context_reaches_base_stress_and_unavailable_accounting(unknown_right):
    simulation = _simulation()
    spec = _spec(case="cvr") if unknown_right else _unpaid_sale_spec()
    report = evaluate_event_aware_swing_accounting(_selected(spec), (spec,),
        {ticker: _spec(case="benchmark", security=ticker, policy="fixed_horizon") for ticker in ("SPY", "QQQ", "XLK")},
        config=SwingTrainingConfig(), simulation=simulation,
        strategy_contract=load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml"),
        research_contract=RESEARCH, session_calendar=(DAYS[0].date().isoformat(),))
    assert report["trade_simulation"]["policy_sha256"] == simulation.policy.sha256()
    assert report["eligible"] is False
    if unknown_right:
        assert report["status"] == "valuation_unavailable"
        assert report["comparisons"] is None
    else:
        assert report["base_ledger"]["compounded_return"] == pytest.approx(0.0098)
        assert report["stress_ledger"]["compounded_return"] == pytest.approx(0.0096)
        assert report["base_ledger"]["trade_simulation"] == report["stress_ledger"]["trade_simulation"]
