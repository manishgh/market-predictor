from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import exchange_calendars as xcals
import pytest
from pydantic import ValidationError

from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    CorporateActionEvent,
    EvidenceReference,
    ExecutionEvent,
    HoldingSpecification,
    KnownMark,
    PaymentEvent,
    PositionLeg,
    UnavailableMark,
)
from market_predictor.swing.evaluation.holding_accounting import project_holding_targets, replay_holding

CALENDAR = xcals.get_calendar("XNYS")
SESSIONS = CALENDAR.sessions_in_range("2024-01-02", "2024-01-19")
ENDS = tuple(CALENDAR.session_close(s).to_pydatetime() for s in SESSIONS)
ENTRY = CALENDAR.session_open(SESSIONS[0]).to_pydatetime()


def _evidence(at: datetime | None = ENTRY) -> tuple[EvidenceReference, ...]:
    return (EvidenceReference(
        reference="synthetic-fixture", artifact_sha256="a" * 64, record_locator="row:1",
        interpretation_policy_sha256="b" * 64, retrieved_at=ENDS[-1] + timedelta(days=30), available_at=at,
    ),)


def _mark(position: str, at: datetime, value: float, **changes: Any) -> KnownMark:
    return KnownMark(**{
        "position_id": position, "mark_at": at, "value_per_unit": value,
        "currency": "USD", "evidence": _evidence(at), **changes,
    })


def _spec(**changes: Any) -> HoldingSpecification:
    return HoldingSpecification(**{
        "research_contract_sha256": "c" * 64,
        "decision_id": "d1", "security_id": "common:A", "sector": "healthcare",
        "initial_position_id": "shares", "initial_entry_price": 100.0,
        "initial_entry_timestamp": ENTRY, "currency": "USD", "entry_evidence": _evidence(),
        "price_basis": "raw_with_no_adjustment", "session_end_timestamps": ENDS[:10],
        "cost_prepaid_fraction": 0.002, "policy": "managed",
        "marks": tuple(_mark("shares", at, 100.0 + i) for i, at in enumerate(ENDS)), **changes,
    })


def _sale(at: datetime = ENDS[1], **changes: Any) -> ExecutionEvent:
    return ExecutionEvent(**{
        "event_id": "sale", "effective_at": at, "order": 0, "evidence": _evidence(at),
        "position_id": "shares", "security_id": "common:A", "fraction_of_owned": 1.0,
        "price_per_unit": 95.0, "currency": "USD", "proceeds_id": "sale-claim", "reason": "managed_exit", **changes,
    })


def _payment(claim: str, at: datetime, amount: float = 1.0, **changes: Any) -> PaymentEvent:
    return PaymentEvent(**{
        "event_id": "pay:" + claim, "effective_at": at, "order": 1, "evidence": _evidence(at),
        "claim_id": claim, "amount_per_claim_unit": amount, "currency": "USD",
        "pending_proceeds_id": "pending:" + claim, **changes,
    })


def _cash(claim: str, at: datetime, **changes: Any) -> CashAvailabilityEvent:
    return CashAvailabilityEvent(**{
        "event_id": "cash:" + claim, "effective_at": at, "order": 2, "evidence": _evidence(at),
        "payment_event_id": "pay:" + claim, "currency": "USD", **changes,
    })


def _action(at: datetime = ENDS[0], **changes: Any) -> CorporateActionEvent:
    return CorporateActionEvent(**{
        "event_id": "dividend", "effective_at": at, "order": 0, "evidence": _evidence(at),
        "owned_position_id": "shares", "owned_security_id": "common:A", "treatment": "distribution",
        "fractional_treatment": "proportional", "legs": (PositionLeg(
            position_id="div-claim", kind="unpaid_proceeds", security_id="common:A",
            units_per_owned_unit=1.0, currency="USD", face_value_per_unit=2.0,
        ),), **changes,
    })


def test_ordinary_share_parity_normalization_cost_once_and_no_implicit_liquidation() -> None:
    outcome = replay_holding(_spec())
    assert len(outcome.snapshots) == 10
    for i, snapshot in enumerate(outcome.snapshots):
        assert snapshot.tradable_value == pytest.approx((100 + i) / 100)
        assert snapshot.total_value == snapshot.tradable_value
        assert snapshot.gross_return == pytest.approx(i / 100)
        assert snapshot.net_return == pytest.approx(i / 100 - 0.002)
        assert snapshot.cumulative_cash_released == snapshot.available_cash == 0.0
        assert snapshot.unpaid_proceeds_value == snapshot.contingent_right_value == 0.0
        assert not snapshot.gaps
    assert outcome.residual_positions[0].units == 0.01
    assert not outcome.fully_settled and not outcome.production_eligible
    assert not outcome.cash_releases and not outcome.executed_sales


def test_sale_requires_payment_and_separate_evidenced_availability() -> None:
    sale, pay = _sale(), _payment("sale-claim", ENDS[2])
    marks = _spec().marks + tuple(_mark("sale-claim", at, 1.0) for at in ENDS[1:2]) + tuple(
        _mark("pending:sale-claim", at, 1.0) for at in ENDS[2:]
    )
    unpaid = replay_holding(_spec(events=(sale, pay), marks=marks))
    assert unpaid.snapshots[1].unpaid_proceeds_value == pytest.approx(0.95)
    assert unpaid.snapshots[2].unpaid_proceeds_value == pytest.approx(0.95)
    assert unpaid.snapshots[-1].cumulative_cash_released == 0.0
    assert unpaid.residual_positions[0].position_id == "pending:sale-claim"
    assert not unpaid.fully_settled
    cash_at = ENDS[3] - timedelta(hours=2)
    settled = replay_holding(_spec(events=(sale, pay, _cash("sale-claim", cash_at)), marks=marks))
    assert settled.snapshots[2].cumulative_cash_released == 0.0
    assert settled.snapshots[3].cumulative_cash_released == pytest.approx(0.95)
    assert settled.snapshots[3].unpaid_proceeds_value == 0.0
    assert settled.snapshots[-1].net_return == pytest.approx(-0.052)
    assert settled.managed_exit_timestamp == sale.effective_at
    assert settled.fully_settled
    assert settled.cash_releases[0].available_at == cash_at
    assert settled.cash_releases[0].source_kind == "sale_proceeds"
    assert settled.executed_sales[0].normalized_proceeds == pytest.approx(0.95)


def test_cash_plus_unknown_cvr_never_uses_cap_or_face_as_mark() -> None:
    legs = (
        PositionLeg(position_id="cash-claim", kind="unpaid_proceeds", security_id="common:A",
                    units_per_owned_unit=1.0, currency="USD", face_value_per_unit=380.0),
        PositionLeg(position_id="cvr", kind="contingent_right", security_id="right:A",
                    units_per_owned_unit=1.0, currency="USD", payout_cap_per_unit=35.0),
    )
    events = (
        _action(treatment="replace", legs=legs),
        _payment("cash-claim", ENDS[1], 380.0), _cash("cash-claim", ENDS[1]),
    )
    out = replay_holding(_spec(initial_entry_price=370.0, events=events))
    end = out.snapshots[9]
    assert end.cumulative_cash_released == pytest.approx(380 / 370)
    assert end.tradable_value == end.unpaid_proceeds_value == 0.0
    assert end.contingent_right_value is None and end.total_value is None
    assert end.gross_return is None and end.net_return is None
    assert [(p.position_id, p.security_id) for p in end.residual_positions] == [("cvr", "right:A")]
    assert not end.fully_settled


def test_post_horizon_payment_retains_residual_then_releases_without_backfill() -> None:
    events = (_action(treatment="replace"), _payment("div-claim", ENDS[10], 2.0), _cash("div-claim", ENDS[11]))
    marks = tuple(_mark("div-claim", at, 2.0) for at in ENDS[:10]) + (_mark("pending:div-claim", ENDS[10], 1.0),)
    fixed = replay_holding(_spec(policy="fixed_horizon", events=events, marks=marks, session_end_timestamps=ENDS))
    managed = replay_holding(_spec(events=events, marks=marks, session_end_timestamps=ENDS))
    assert fixed.snapshots[9].total_value == pytest.approx(0.02)
    assert fixed.snapshots[9].cumulative_cash_released == 0.0
    assert fixed.snapshots[9].residual_positions[0].position_id == "div-claim"
    assert fixed.snapshots[10].residual_positions[0].position_id == "pending:div-claim"
    assert fixed.snapshots[11].cumulative_cash_released == pytest.approx(0.02)
    assert fixed.fully_settled
    targets = project_holding_targets(fixed, managed)
    assert targets.horizon_end_timestamp == ENDS[9]
    assert targets.fixed_horizon_net_return == fixed.snapshots[9].net_return


def test_earned_claim_survives_stop_and_corporate_cash_can_be_available_preopen() -> None:
    preopen = CALENDAR.session_open(SESSIONS[3]).to_pydatetime() - timedelta(minutes=1)
    events = (_action(), _sale(), _payment("sale-claim", ENDS[1]), _cash("sale-claim", ENDS[1]),
              _payment("div-claim", preopen, 2.0), _cash("div-claim", preopen))
    marks = _spec().marks + tuple(_mark("div-claim", at, 2.0) for at in ENDS[:3])
    out = replay_holding(_spec(events=events, marks=marks))
    assert out.snapshots[1].total_value == pytest.approx(0.97)
    assert out.snapshots[1].cumulative_cash_released == pytest.approx(0.95)
    assert out.snapshots[3].cumulative_cash_released == pytest.approx(0.97)
    assert out.cash_releases[1].source_kind == "corporate_payment"
    assert out.cash_releases[1].available_at == preopen
    assert out.snapshots[9].net_return == pytest.approx(-0.032)


def test_event_after_sale_never_earns_a_claim_or_payment() -> None:
    events = (_sale(), _payment("sale-claim", ENDS[1]), _cash("sale-claim", ENDS[1]),
              _action(ENDS[2]), _payment("div-claim", ENDS[3], 2.0), _cash("div-claim", ENDS[3]))
    out = replay_holding(_spec(events=events))
    assert out.fully_settled
    assert len(out.cash_releases) == 1
    assert out.snapshots[9].total_value == pytest.approx(0.95)


def test_entry_time_action_requires_ordering_against_purchase() -> None:
    at_entry = replay_holding(_spec(events=(_action(ENTRY),)))
    assert at_entry.snapshots[0].total_value is None
    assert {gap.code for gap in at_entry.snapshots[0].gaps} >= {"entry_event_order_unavailable"}
    assert [position.position_id for position in at_entry.snapshots[0].residual_positions] == ["shares"]
    with pytest.raises(ValidationError, match="before entry"):
        _spec(events=(_action(ENTRY - timedelta(microseconds=1)),))
    later = _action(ENTRY + timedelta(microseconds=1))
    marks = _spec().marks + tuple(_mark("div-claim", at, 2.0) for at in ENDS)
    supported = replay_holding(_spec(events=(later,), marks=marks))
    assert supported.snapshots[0].unpaid_proceeds_value == pytest.approx(0.02)


@pytest.mark.parametrize("poison", ["order", "missing_availability", "late_availability", "undated_payment"])
def test_unearned_post_disposal_metadata_cannot_poison_labels(poison: str) -> None:
    realized = (_sale(), _payment("sale-claim", ENDS[1]), _cash("sale-claim", ENDS[1]))
    action = _action(ENDS[2])
    pay = _payment("div-claim", ENDS[3], 2.0)
    if poison == "order":
        action = action.model_copy(update={"order": None})
    elif poison == "undated_payment":
        pay = pay.model_copy(update={"effective_at": None})
    else:
        action = action.model_copy(update={"evidence": _evidence(
            None if poison == "missing_availability" else ENDS[-1] + timedelta(days=10))})
    baseline = replay_holding(_spec(events=realized))
    extra = (action, pay, _cash("div-claim", ENDS[3]))
    observed = replay_holding(_spec(events=(*realized, *extra)))
    assert observed.snapshots[9] == baseline.snapshots[9]
    assert observed.label_available_at == baseline.label_available_at


def test_declared_future_position_is_not_equivalent_to_disposed_position() -> None:
    conversion = _action(ENDS[2], event_id="convert", treatment="replace", legs=(PositionLeg(
        position_id="successor", kind="tradable_shares", security_id="common:B",
        units_per_owned_unit=1.0, currency="USD"),))
    premature = _action(ENDS[1], owned_position_id="successor", owned_security_id="common:B")
    with pytest.raises(ValueError, match="not yet created"):
        replay_holding(_spec(events=(conversion, premature)))


def test_managed_exit_cannot_extend_forecast_while_residual_marks_can() -> None:
    with pytest.raises(ValidationError, match="cannot extend"):
        _spec(session_end_timestamps=ENDS, events=(_sale(ENDS[10]),))


@pytest.mark.parametrize("defect", ["duplicate_payment", "duplicate_cash", "duplicate_event", "duplicate_claim"])
def test_duplicate_claim_or_payment_fails(defect: str) -> None:
    event = _action()
    pay, cash = _payment("div-claim", ENDS[1], 2.0), _cash("div-claim", ENDS[1])
    extra = {
        "duplicate_payment": _payment("div-claim", ENDS[2], event_id="pay-again", pending_proceeds_id="other"),
        "duplicate_cash": _cash("div-claim", ENDS[2], event_id="cash-again"),
        "duplicate_event": event,
        "duplicate_claim": _action(ENDS[2], event_id="second-action"),
    }[defect]
    with pytest.raises(ValidationError, match="duplicate"):
        _spec(events=(event, pay, cash, extra))


@pytest.mark.parametrize("field,value", [
    ("initial_entry_price", True), ("initial_entry_price", "100"), ("initial_entry_price", float("nan")),
    ("initial_entry_price", 0.0), ("cost_prepaid_fraction", True), ("cost_prepaid_fraction", 0.0),
    ("currency", "EUR"), ("price_basis", "all"), ("session_end_timestamps", list(ENDS[:10])),
    ("initial_entry_timestamp", ENTRY.replace(tzinfo=None)), ("decision_id", "  "),
])
def test_invalid_types_and_frozen_policy_rejected(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        _spec(**{field: value})


@pytest.mark.parametrize("field,value", [
    ("artifact_sha256", "z" * 64), ("artifact_sha256", "a" * 63),
    ("interpretation_policy_sha256", "bad"), ("record_locator", ""), ("reference", ""),
])
def test_evidence_requires_structural_provenance(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        EvidenceReference(**{**_evidence()[0].model_dump(), field: value})


def test_models_are_frozen_extra_fields_and_empty_evidence_fail() -> None:
    spec = _spec()
    with pytest.raises(ValidationError):
        spec.currency = None
    with pytest.raises(ValidationError):
        _spec(entry_evidence=())
    with pytest.raises(ValidationError):
        _spec(adjustment="all")
    with pytest.raises(ValidationError):
        _mark("shares", ENDS[0], True)
    with pytest.raises(ValidationError):
        UnavailableMark(position_id="shares", mark_at=ENDS[0], reason="missing", value_per_unit=0.0)


def test_future_marks_not_carried_back_and_missing_later_mark_does_not_poison_label() -> None:
    spec = _spec(session_end_timestamps=ENDS, marks=tuple(_mark("shares", at, 100.0) for at in ENDS[:10]))
    outcome = replay_holding(spec)
    assert outcome.snapshots[9].total_value == 1.0
    assert outcome.snapshots[10].total_value is None
    fixed = replay_holding(_spec(policy="fixed_horizon", marks=spec.marks, session_end_timestamps=ENDS))
    targets = project_holding_targets(fixed, outcome)
    assert targets.fixed_horizon_net_return == targets.managed_horizon_net_return == -0.002
    poisoned = replay_holding(_spec(marks=(_mark("shares", ENDS[10], 99999.0),)))
    assert all(s.total_value is None for s in poisoned.snapshots)


def test_delayed_publication_changes_label_clock_not_economic_mark_or_cash_clock() -> None:
    published = ENDS[9] + timedelta(minutes=1)
    outcome = replay_holding(_spec(marks=(_mark("shares", ENDS[9], 110.0, evidence=_evidence(published)),)))
    assert outcome.snapshots[9].total_value == pytest.approx(1.1)
    assert outcome.label_available_at == published
    events = (_sale(), _payment("sale-claim", ENDS[1]), _cash("sale-claim", ENDS[1], evidence=_evidence(published)))
    paid = replay_holding(_spec(events=events))
    assert paid.snapshots[1].cumulative_cash_released == pytest.approx(0.95)
    assert paid.cash_releases[0].available_at == ENDS[1]
    assert paid.cash_releases[0].evidence_available_at == published
    assert paid.label_available_at == published


def test_unknown_historical_evidence_availability_stays_research_only() -> None:
    outcome = replay_holding(_spec(entry_evidence=_evidence(None)))
    assert outcome.snapshots[9].total_value == pytest.approx(1.09)
    assert outcome.label_available_at is None
    assert outcome.snapshots[9].source_available_at is None
    assert not outcome.production_eligible


@pytest.mark.parametrize("changes,code", [
    ({"effective_at": None}, "effective_date_unavailable"),
    ({"order": None}, "event_order_unavailable"),
    ({"fractional_treatment": None}, "fractional_treatment_unavailable"),
])
def test_unknown_dates_order_or_fractional_treatment_are_gaps(changes: dict[str, Any], code: str) -> None:
    outcome = replay_holding(_spec(events=(_action(**changes),)))
    assert outcome.snapshots[9].total_value is None
    assert code in {gap.code for gap in outcome.snapshots[9].gaps}
    assert outcome.snapshots[9].cumulative_cash_released == 0.0


def test_unknown_currency_never_inferred_from_marks() -> None:
    out = replay_holding(_spec(currency=None))
    assert out.snapshots[9].total_value is None
    assert "currency_unavailable" in {gap.code for gap in out.snapshots[9].gaps}


def test_payment_before_claim_or_cash_before_payment_fails() -> None:
    with pytest.raises(ValueError, match="existing"):
        replay_holding(_spec(events=(_payment("not-earned", ENDS[0]),)))
    with pytest.raises(ValueError, match="matching payment"):
        replay_holding(_spec(events=(_cash("not-paid", ENDS[0]),)))


def test_wrong_actual_security_class_fails() -> None:
    with pytest.raises(ValueError, match="actually owned"):
        replay_holding(_spec(events=(_action(owned_security_id="preferred:A"),)))


@pytest.mark.parametrize("defect", ["wrong_close", "missing_session", "wrong_entry"])
def test_exact_xnys_clock_required(defect: str) -> None:
    changes: dict[str, Any] = {
        "wrong_close": {"session_end_timestamps": tuple(t + timedelta(minutes=1) for t in ENDS[:10])},
        "missing_session": {"session_end_timestamps": ENDS[:5] + ENDS[6:11]},
        "wrong_entry": {"initial_entry_timestamp": ENTRY + timedelta(minutes=1)},
    }[defect]
    with pytest.raises(ValueError, match="XNYS"):
        replay_holding(_spec(**changes))


def test_fixed_horizon_execution_and_benchmark_projection_use_day_ten() -> None:
    fixed = replay_holding(_spec(policy="fixed_horizon", session_end_timestamps=ENDS))
    managed = replay_holding(_spec())
    benchmark = replay_holding(_spec(policy="fixed_horizon", security_id="SPY"))
    targets = project_holding_targets(fixed, managed, benchmarks=(benchmark,))
    assert targets.benchmarks[0].fixed_horizon_excess_return == pytest.approx(-0.002)
    assert targets.horizon_end_timestamp == ENDS[9]
    with pytest.raises(ValidationError, match="exact horizon"):
        _spec(policy="fixed_horizon", session_end_timestamps=ENDS, events=(_sale(ENDS[-1], reason="fixed_horizon"),))
    with pytest.raises(ValueError, match="explicitly fixed"):
        project_holding_targets(managed, fixed)


def test_zero_mark_is_explicit_and_not_unavailable() -> None:
    outcome = replay_holding(_spec(marks=(_mark("shares", ENDS[9], 0.0),)))
    assert outcome.snapshots[9].total_value == 0.0
    assert outcome.snapshots[9].net_return == -1.002


def test_projection_requires_research_contract_identity_not_interpretation_identity() -> None:
    fixed = replay_holding(_spec(policy="fixed_horizon"))
    managed = replay_holding(_spec())
    assert project_holding_targets(fixed, managed).research_contract_sha256 == "c" * 64
    different = replay_holding(_spec(policy="fixed_horizon", security_id="SPY", research_contract_sha256="d" * 64))
    with pytest.raises(ValueError, match="research contract"):
        project_holding_targets(fixed, managed, benchmarks=(different,))


def test_explicit_fixed_horizon_sale_releases_cash_only_with_settlement_events() -> None:
    at = ENDS[9]
    sale = _sale(at, reason="fixed_horizon", price_per_unit=109.0)
    pending = replay_holding(_spec(policy="fixed_horizon", events=(sale,)))
    assert pending.snapshots[9].total_value is None
    assert pending.snapshots[9].cumulative_cash_released == 0.0
    assert pending.residual_positions[0].position_id == "sale-claim"
    settled = replay_holding(_spec(policy="fixed_horizon", events=(
        sale, _payment("sale-claim", at), _cash("sale-claim", at),
    )))
    assert settled.snapshots[9].total_value == pytest.approx(1.09)
    assert settled.snapshots[9].net_return == pytest.approx(0.088)
    assert settled.fully_settled
    assert settled.managed_exit_timestamp is None


def test_unknown_availability_date_does_not_release_cash() -> None:
    events = (_action(), _payment("div-claim", ENDS[1], 2.0), _cash("div-claim", ENDS[1], effective_at=None))
    out = replay_holding(_spec(events=events))
    assert not out.cash_releases
    assert out.snapshots[9].total_value is None
    assert "effective_date_unavailable" in {g.code for g in out.snapshots[9].gaps}


def test_tied_event_order_is_gap_not_input_tuple_precedence() -> None:
    out = replay_holding(_spec(events=(_sale(), _payment("sale-claim", ENDS[1], order=0))))
    assert out.snapshots[1].total_value is None
    assert "event_order_unavailable" in {g.code for g in out.snapshots[1].gaps}
    assert not out.cash_releases


def test_cash_availability_before_payment_and_wrong_sale_amount_fail() -> None:
    with pytest.raises(ValueError, match="matching payment"):
        replay_holding(_spec(events=(_sale(), _payment("sale-claim", ENDS[3]), _cash("sale-claim", ENDS[2]))))
    with pytest.raises(ValueError, match="reconcile"):
        replay_holding(_spec(events=(_sale(), _payment("sale-claim", ENDS[1], 2.0))))


def test_delivery_transforms_owned_class_and_new_class_can_be_executed() -> None:
    new = PositionLeg(position_id="new-shares", kind="tradable_shares", security_id="common:B",
                      units_per_owned_unit=2.0, currency="USD")
    events = (_action(treatment="replace", legs=(new,)),
              _sale(position_id="new-shares", security_id="common:B", price_per_unit=50.0),
              _payment("sale-claim", ENDS[1]), _cash("sale-claim", ENDS[1]))
    out = replay_holding(_spec(events=events, marks=(_mark("new-shares", ENDS[0], 50.0),)))
    assert out.snapshots[0].residual_positions[0].security_id == "common:B"
    assert out.snapshots[0].residual_positions[0].units == 0.02
    assert out.snapshots[0].total_value == 1.0
    assert out.snapshots[9].total_value == 1.0
    assert out.executed_sales[0].security_id == "common:B"


def test_undeclared_class_cannot_silently_be_treated_as_unowned() -> None:
    with pytest.raises(ValidationError, match="declared tradable"):
        _spec(events=(_action(owned_position_id="undeclared"),))


def test_skipped_payment_event_id_cannot_mask_an_earned_claim_id() -> None:
    earned = _action()
    unearned_leg = PositionLeg(position_id="unearned", kind="unpaid_proceeds", security_id="common:A",
                              units_per_owned_unit=1.0, currency="USD")
    events = (
        earned, _sale(), _payment("sale-claim", ENDS[1]), _cash("sale-claim", ENDS[1]),
        _action(ENDS[2], event_id="unearned-action", legs=(unearned_leg,)),
        _payment("unearned", ENDS[3], event_id="div-claim"),
        _payment("div-claim", ENDS[4], 2.0), _cash("div-claim", ENDS[4]),
    )
    marks = _spec().marks + tuple(_mark("div-claim", at, 2.0) for at in ENDS[:4])
    out = replay_holding(_spec(events=events, marks=marks))
    assert out.snapshots[9].cumulative_cash_released == pytest.approx(0.97)
    assert out.fully_settled
