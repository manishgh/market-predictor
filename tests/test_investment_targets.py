from __future__ import annotations

import hashlib
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Literal

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.contracts import (
    HOLDING_ROLES,
    HoldingRole,
    InvestmentHoldingBinding,
    InvestmentSourceIdentity,
    InvestmentTargetPolicy,
    InvestmentTargetRequest,
)
from market_predictor.investment.targets import project_investment_targets
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
from market_predictor.swing.evaluation.holding_accounting import replay_holding


def _fixture(horizon: Literal[63, 252] = 63) -> tuple[InvestmentTargetRequest, dict[HoldingRole, HoldingSpecification]]:
    policy = InvestmentTargetPolicy(horizon_sessions=horizon)
    calendar = xcals.get_calendar("XNYS")
    decision = date(2025, 11, 25)
    sessions = calendar.sessions_window(pd.Timestamp(decision), horizon + 1)[1:]
    ends = tuple(calendar.session_close(day).to_pydatetime() for day in sessions)
    entry = calendar.session_open(sessions[0]).to_pydatetime()
    holdings: dict[HoldingRole, HoldingSpecification] = {}
    bindings = []
    for role, symbol, terminal in zip(HOLDING_ROLES, ("MSFT", "SPY", "QQQ", "XLK"), (110., 104., 106., 108.), strict=True):
        reference = EvidenceReference(reference=f"raw/{symbol}", artifact_sha256="a" * 64,
            record_locator=f"security={symbol}", interpretation_policy_sha256="b" * 64,
            retrieved_at=entry, available_at=entry)
        spec = HoldingSpecification(research_contract_sha256=policy.sha256(), decision_id="decision",
            security_id=f"security:{symbol}", sector="technology", initial_position_id=f"shares:{symbol}",
            initial_entry_price=100., initial_entry_timestamp=entry, price_basis="raw_with_no_adjustment",
            currency="USD", entry_evidence=(reference,), session_end_timestamps=ends,
            cost_prepaid_fraction=.002, policy="fixed_horizon", marks=tuple(KnownMark(
                position_id=f"shares:{symbol}", mark_at=end, value_per_unit=terminal if end == ends[-1] else 100.,
                currency="USD", evidence=(reference.model_copy(update={"available_at": end + timedelta(minutes=5),
                    "retrieved_at": end + timedelta(minutes=6)}),)) for end in ends))
        holdings[role] = spec
        bindings.append(InvestmentHoldingBinding(role=role, security_id=spec.security_id, symbol=symbol,
            specification_sha256=json_sha256(spec.model_dump(mode="json")), sources=(InvestmentSourceIdentity(
                reference=reference.reference, artifact_sha256=reference.artifact_sha256,
                interpretation_policy_sha256=reference.interpretation_policy_sha256),)))
    request = InvestmentTargetRequest(policy=policy, decision_id="decision", decision_session=decision,
        decision_time_utc=calendar.session_close(pd.Timestamp(decision)).to_pydatetime() + timedelta(hours=1),
        sector="technology", sector_membership_source=InvestmentSourceIdentity(reference="membership",
            artifact_sha256="c" * 64, interpretation_policy_sha256="d" * 64), holdings=tuple(bindings))
    return request, holdings


def _rebind(request: InvestmentTargetRequest, holdings: dict[HoldingRole, HoldingSpecification]) -> InvestmentTargetRequest:
    return request.model_copy(update={"holdings": tuple(binding.model_copy(update={
        "specification_sha256": json_sha256(holdings[binding.role].model_dump(mode="json"))
    }) for binding in request.holdings)})


@pytest.mark.parametrize("horizon", [63, 252])
def test_exact_forecast_horizons_costs_and_benchmarks(horizon: Literal[63, 252]) -> None:
    request, holdings = _fixture(horizon)
    result = project_investment_targets(request, holdings)
    assert result.status == "available" and result.label_eligible
    assert result.entry_time_utc == holdings["stock"].initial_entry_timestamp
    assert result.horizon_end_utc == holdings["stock"].session_end_timestamps[-1]
    assert result.label_available_at_utc == result.horizon_end_utc + timedelta(minutes=5)
    assert result.holdings[0].gross_return == pytest.approx(.1)
    assert result.holdings[0].net_return == pytest.approx(.098)
    assert result.net_excess_vs_spy == pytest.approx(.058)
    assert result.net_excess_vs_qqq == pytest.approx(.038)
    assert result.net_excess_vs_sector == pytest.approx(.018)
    assert not result.training_ready and not result.production_eligible and not result.promotion_eligible
    assert result.request_sha256 == request.sha256()
    # Thanksgiving is absent; Black Friday keeps its early close.
    ends = holdings["stock"].session_end_timestamps
    assert all(end.date() != date(2025, 11, 27) for end in ends)
    assert next(end for end in ends if end.date() == date(2025, 11, 28)).hour == 18
    assert result.label_available_at_utc > replay_holding(holdings["stock"]).label_available_at


def test_final_source_clock_can_arrive_later_than_horizon() -> None:
    request, holdings = _fixture()
    spec = holdings["qqq"]
    mark = spec.marks[-1]
    assert isinstance(mark, KnownMark)
    later = mark.mark_at + timedelta(days=3)
    reference = mark.evidence[0].model_copy(update={"available_at": later, "retrieved_at": later})
    holdings["qqq"] = spec.model_copy(update={"marks": (*spec.marks[:-1], mark.model_copy(update={"evidence": (reference,)}))})
    result = project_investment_targets(_rebind(request, holdings), holdings)
    assert result.label_available_at_utc == later


def test_unknown_source_clock_is_not_a_training_label() -> None:
    request, holdings = _fixture()
    spec = holdings["stock"]
    holdings["stock"] = spec.model_copy(update={"entry_evidence": (
        spec.entry_evidence[0].model_copy(update={"available_at": None}),)})
    result = project_investment_targets(_rebind(request, holdings), holdings)
    assert result.status == "unavailable" and not result.label_eligible
    assert result.label_available_at_utc is None
    assert result.holdings[0].gross_return == pytest.approx(.1)  # disclosed diagnostic only
    assert "stock:source_availability_unknown" in result.missing_reasons


def _distribution(spec: HoldingSpecification, *, known: bool) -> HoldingSpecification:
    reference = spec.entry_evidence[0]
    event_at = spec.session_end_timestamps[20]
    event = CorporateActionEvent(event_id="distribution", effective_at=event_at, order=0,
        evidence=(reference,), owned_position_id=spec.initial_position_id, owned_security_id=spec.security_id,
        treatment="distribution", fractional_treatment="proportional", legs=(PositionLeg(position_id="claim",
            kind="contingent_right", security_id=spec.security_id, units_per_owned_unit=1., currency="USD"),))
    marks = tuple(KnownMark(position_id="claim", mark_at=end, value_per_unit=10., currency="USD",
        evidence=(reference.model_copy(update={"available_at": end, "retrieved_at": end}),))
        for end in spec.session_end_timestamps[20:]) if known else ()
    return spec.model_copy(update={"events": (event,), "marks": (*spec.marks, *marks)})


def test_distribution_is_retained_and_future_payment_cannot_change_value() -> None:
    request, holdings = _fixture()
    spec = _distribution(holdings["stock"], known=True)
    holdings["stock"] = spec
    first = project_investment_targets(_rebind(request, holdings), holdings)
    assert first.holdings[0].gross_return == pytest.approx(.2)
    future = spec.session_end_timestamps[-1] + timedelta(days=5)
    reference = spec.entry_evidence[0].model_copy(update={"available_at": future, "retrieved_at": future})
    payment = PaymentEvent(event_id="payment", effective_at=future, order=0, evidence=(reference,),
        claim_id="claim", amount_per_claim_unit=1000., currency="USD", pending_proceeds_id="cash")
    release = CashAvailabilityEvent(event_id="release", effective_at=future, order=1,
        evidence=(reference,), payment_event_id="payment", currency="USD")
    holdings["stock"] = spec.model_copy(update={"events": (*spec.events, payment, release)})
    later = project_investment_targets(_rebind(request, holdings), holdings)
    assert later.holdings[0].gross_return == first.holdings[0].gross_return
    assert later.label_available_at_utc == first.label_available_at_utc


def test_unknown_residual_claim_never_becomes_zero() -> None:
    request, holdings = _fixture()
    holdings["stock"] = _distribution(holdings["stock"], known=False)
    result = project_investment_targets(_rebind(request, holdings), holdings)
    assert not result.label_eligible and result.status == "unavailable"
    assert result.holdings[0].gross_return is None and result.net_excess_vs_spy is None
    assert any("mark_unavailable:claim" in reason for reason in result.missing_reasons)


def test_ordinary_sale_is_refused_even_when_valid_for_swing_kernel() -> None:
    request, holdings = _fixture()
    spec = holdings["stock"]
    sale = ExecutionEvent(event_id="sale", effective_at=spec.session_end_timestamps[9], order=0,
        evidence=spec.entry_evidence, position_id=spec.initial_position_id, security_id=spec.security_id,
        fraction_of_owned=1., price_per_unit=100., currency="USD", proceeds_id="proceeds", reason="fixed_horizon")
    holdings["stock"] = spec.model_copy(update={"events": (sale,)})
    with pytest.raises(ValueError, match="forbid ordinary"):
        project_investment_targets(_rebind(request, holdings), holdings)


@pytest.mark.parametrize("change", ["hash", "source", "decision", "interval", "identity", "policy"])
def test_bound_specification_and_metadata_tampering_is_refused(change: str) -> None:
    request, holdings = _fixture()
    spec = holdings["spy"]
    if change == "source":
        holdings["spy"] = spec.model_copy(update={"entry_evidence": (
            spec.entry_evidence[0].model_copy(update={"artifact_sha256": "e" * 64}),)})
    elif change == "decision":
        holdings["spy"] = spec.model_copy(update={"decision_id": "another"})
    elif change == "interval":
        holdings["spy"] = spec.model_copy(update={"session_end_timestamps": spec.session_end_timestamps[:-1]})
    elif change == "identity":
        holdings["spy"] = spec.model_copy(update={"security_id": "another"})
    else:
        holdings["spy"] = spec.model_copy(update={"research_contract_sha256": "e" * 64})
    if change != "hash":
        request = _rebind(request, holdings)
    with pytest.raises(ValueError):
        project_investment_targets(request, holdings)


def test_required_benchmarks_and_policy_reject_unsupported_inputs() -> None:
    request, holdings = _fixture()
    with pytest.raises(ValueError, match="exactly"):
        project_investment_targets(request, {role: spec for role, spec in holdings.items() if role != "qqq"})
    with pytest.raises(ValueError):
        InvestmentTargetPolicy(horizon_sessions=10)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        InvestmentTargetPolicy(horizon_sessions=63, round_trip_cost_bps=0.)
    duplicate = request.model_copy(update={"holdings": (request.holdings[0], request.holdings[0], *request.holdings[2:])})
    with pytest.raises(ValueError):
        project_investment_targets(duplicate, holdings)


def test_pinned_swing_implementation_bytes_remain_unchanged() -> None:
    root = Path(__file__).resolve().parents[1] / "src/market_predictor"
    expected = {
        "swing/contracts/holding_accounting.py": "bb02f2a88b67f2f12820a3cd0223314ab1e8944aa2110d93fc662e374c766132",
        "swing/evaluation/holding_accounting.py": "e01a99a6263e77452072b0498b71e7c80acca34f2520aa9ac21a7a51ee02f249",
        "swing/labels/ordinary_holding.py": "caee56dfc5d679f2495124d2faa304cfdc21817c26cfd8280bf84415a4adb605",
        "swing/evaluation/trade_simulation.py": "da75f7c8fe667b1d642cbe3cdaa02feee99aa83ae85e563d9cc697fda121acaf",
    }
    for name, digest in expected.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest


def test_result_timezone_is_aware() -> None:
    request, holdings = _fixture()
    result = project_investment_targets(request, holdings)
    assert result.entry_time_utc.tzinfo == UTC
