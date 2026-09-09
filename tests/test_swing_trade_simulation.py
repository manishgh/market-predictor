"""Focused research-sale assumptions and canonical holding replay regressions."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    CorporateActionEvent,
    EvidenceReference,
    ExecutionEvent,
    HoldingSpecification,
    KnownMark,
    PaymentEvent,
    PositionLeg,
    SimulationReference,
    UnavailableMark,
)
from market_predictor.swing.contracts.trade_simulation import TradeSimulationContext
from market_predictor.swing.evaluation.holding_accounting import replay_holding, replay_next_open_settlement
from market_predictor.swing.evaluation.trade_simulation import load_trade_simulation_context, simulate_ordinary_sales

CALENDAR = xcals.get_calendar("XNYS")
POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "swing_trade_simulation.toml"
RETRIEVED = datetime(2026, 9, 9, tzinfo=UTC)


def _evidence(at: datetime | None) -> tuple[EvidenceReference, ...]:
    return (EvidenceReference(
        reference="test-only-source", artifact_sha256="a" * 64, record_locator="row:1",
        interpretation_policy_sha256="b" * 64, retrieved_at=RETRIEVED, available_at=at,
    ),)


def _mark(position: str, at: datetime, value: float = 100.0, **changes: Any) -> KnownMark:
    return KnownMark(**{
        "position_id": position, "mark_at": at, "value_per_unit": value,
        "currency": "USD", "evidence": _evidence(at), **changes,
    })


def _spec(start: str = "2024-01-02", **changes: Any) -> HoldingSpecification:
    first = datetime.fromisoformat(start)
    sessions = CALENDAR.sessions_in_range(start, (first + timedelta(days=25)).date().isoformat())[:10]
    entry = CALENDAR.session_open(sessions[0]).to_pydatetime()
    closes = tuple(CALENDAR.session_close(session).to_pydatetime() for session in sessions)
    return HoldingSpecification(**{
        "research_contract_sha256": "c" * 64, "decision_id": "simulation-test",
        "security_id": "common:A", "sector": "healthcare", "initial_position_id": "shares",
        "initial_entry_price": 100.0, "initial_entry_timestamp": entry, "currency": "USD",
        "entry_evidence": _evidence(entry), "price_basis": "raw_with_no_adjustment",
        "session_end_timestamps": closes, "cost_prepaid_fraction": 0.002, "policy": "managed",
        "marks": tuple(_mark("shares", at) for at in closes), **changes,
    })


def _sale(at: datetime, **changes: Any) -> ExecutionEvent:
    return ExecutionEvent(**{
        "event_id": "sale", "effective_at": at, "order": 0, "evidence": _evidence(at),
        "position_id": "shares", "security_id": "common:A", "fraction_of_owned": 1.0,
        "price_per_unit": 95.0, "currency": "USD", "proceeds_id": "sale-claim",
        "reason": "managed_exit", **changes,
    })


def test_simulated_payment_cannot_become_observed_by_replacing_release(context: TradeSimulationContext) -> None:
    spec = _spec()
    simulated = simulate_ordinary_sales(spec.model_copy(update={"events": (_sale(spec.session_end_timestamps[1]),)}), context)
    events = tuple(event.model_copy(update={"event_id": "source-only-release", "evidence": _evidence(event.effective_at)})
        if isinstance(event, CashAvailabilityEvent) else event for event in simulated.specification.events)
    out = replay_holding(simulated.specification.model_copy(update={"events": events}),
        simulation_policy_sha256=context.policy.sha256())
    assert out.cash_releases[0].basis == "research_assumption"
    assert out.production_eligible is False


@pytest.mark.parametrize("unused", ["mark", "corporate_event"])
def test_unconsumed_late_evidence_does_not_change_research_maturity(context: TradeSimulationContext, unused: str) -> None:
    spec = _spec()
    spec = spec.model_copy(update={"events": (_sale(spec.session_end_timestamps[1]),)})
    baseline = simulate_ordinary_sales(spec, context)
    late = spec.session_end_timestamps[9] + timedelta(days=365)
    if unused == "mark":
        marks = tuple(mark.model_copy(update={"evidence": _evidence(late)}) if index == 8 else mark
            for index, mark in enumerate(spec.marks))
        spec = spec.model_copy(update={"marks": marks})
    else:
        action = CorporateActionEvent(event_id="unearned-dividend", effective_at=spec.session_end_timestamps[5],
            order=0, evidence=_evidence(late), owned_position_id="shares", owned_security_id="common:A",
            treatment="distribution", fractional_treatment="proportional", legs=(PositionLeg(position_id="unearned",
                kind="unpaid_proceeds", security_id="common:A", units_per_owned_unit=1.0, currency="USD"),))
        spec = spec.model_copy(update={"events": (*spec.events, action)})
    changed = simulate_ordinary_sales(spec, context)
    assert changed.outcome == baseline.outcome
    assert changed.research_label_mature_at == baseline.research_label_mature_at


@pytest.fixture
def context() -> TradeSimulationContext:
    return load_trade_simulation_context(POLICY_PATH, expected_sha256=file_sha256(POLICY_PATH))


def test_loader_pins_file_and_semantic_policy_without_historical_clock(context: TradeSimulationContext) -> None:
    ref = context.policy_reference
    assert ref.artifact_sha256 == file_sha256(POLICY_PATH)
    assert ref.interpretation_policy_sha256 == context.policy.sha256()
    assert ref.reference == str(POLICY_PATH.resolve())
    assert ref.available_at is None
    assert ref.retrieved_at.tzinfo is not None
    assert context.policy.scope == "retrospective_research_only"
    assert context.policy.corporate_claims == "no_inferred_payment_or_valuation"
    assert context.policy.production_eligible is False


def test_loader_rejects_file_tamper_even_when_toml_meaning_is_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "policy.toml"
    path.write_bytes(POLICY_PATH.read_bytes())
    pinned = file_sha256(path)
    path.write_bytes(path.read_bytes() + b"\n# changed bytes\n")
    with pytest.raises(DataReadinessError, match="hash"):
        load_trade_simulation_context(path, expected_sha256=pinned)


def test_loader_rechecks_hash_after_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "policy.toml"
    path.write_bytes(POLICY_PATH.read_bytes())
    pinned = file_sha256(path)
    original = Path.read_text

    def tampered_read(self: Path, *args: Any, **kwargs: Any) -> str:
        result = original(self, *args, **kwargs)
        if self == path:
            self.write_bytes(self.read_bytes() + b"\n# concurrent mutation\n")
        return result

    monkeypatch.setattr(Path, "read_text", tampered_read)
    with pytest.raises(DataReadinessError, match="changed during loading"):
        load_trade_simulation_context(path, expected_sha256=pinned)


@pytest.mark.parametrize("poison", ["historical_clock", "wrong_policy"])
def test_simulation_rejects_false_context_provenance(context: TradeSimulationContext, poison: str) -> None:
    spec = _spec()
    change = {"available_at": spec.initial_entry_timestamp} if poison == "historical_clock" else {
        "interpretation_policy_sha256": "f" * 64,
    }
    ref = context.policy_reference.model_copy(update=change)
    with pytest.raises(DataReadinessError, match="bind policy without claimed historical availability"):
        simulate_ordinary_sales(spec, context.model_copy(update={"policy_reference": ref}))


def test_simulation_reference_cannot_claim_observed_availability(context: TradeSimulationContext) -> None:
    with pytest.raises(ValidationError, match="historical availability"):
        SimulationReference(**{
            **context.policy_reference.model_dump(), "basis": "research_assumption", "available_at": RETRIEVED,
        })


@pytest.mark.parametrize("start,close_utc,reuse_utc", [
    ("2024-01-09", "2024-01-09T21:00:00+00:00", "2024-01-10T14:30:00+00:00"),
    ("2024-01-05", "2024-01-05T21:00:00+00:00", "2024-01-08T14:30:00+00:00"),
    ("2024-01-12", "2024-01-12T21:00:00+00:00", "2024-01-16T14:30:00+00:00"),
    ("2024-03-08", "2024-03-08T21:00:00+00:00", "2024-03-11T13:30:00+00:00"),
    ("2024-11-01", "2024-11-01T20:00:00+00:00", "2024-11-04T14:30:00+00:00"),
    ("2024-07-03", "2024-07-03T17:00:00+00:00", "2024-07-05T13:30:00+00:00"),
    ("2024-11-29", "2024-11-29T18:00:00+00:00", "2024-12-02T14:30:00+00:00"),
], ids=["weekday", "weekend", "mlk-holiday", "spring-dst", "fall-dst", "early-close-holiday", "early-close-weekend"])
def test_close_sale_cash_waits_for_exact_next_xnys_open(
    context: TradeSimulationContext, start: str, close_utc: str, reuse_utc: str,
) -> None:
    spec = _spec(start)
    close, reuse = datetime.fromisoformat(close_utc), datetime.fromisoformat(reuse_utc)
    assert spec.session_end_timestamps[0] == close
    sale = _sale(close)
    spec = spec.model_copy(update={"events": (sale,)})
    before = spec.model_dump_json()
    result = simulate_ordinary_sales(spec, context)
    assert spec.model_dump_json() == before
    assert result.input_specification_sha256 == json_sha256(spec.model_dump(mode="json"))
    generated = [event for event in result.specification.events if event.event_id in result.generated_event_ids]
    assert len(generated) == 2
    pay, cash = generated
    assert isinstance(pay, PaymentEvent) and isinstance(cash, CashAvailabilityEvent)
    assert pay.effective_at == cash.effective_at == reuse
    assert pay.order is not None and cash.order is not None and pay.order < cash.order
    assert pay.claim_id == sale.proceeds_id and pay.amount_per_claim_unit == 1.0
    assert cash.payment_event_id == pay.event_id
    for event in generated:
        assert event.event_id.startswith("simulated-sale:")
        assert event.evidence[:-1] == sale.evidence
        assert isinstance(event.evidence[-1], SimulationReference)
        assert event.evidence[-1].available_at is None
    at_sale, after = result.outcome.snapshots[:2]
    assert at_sale.available_cash == at_sale.cumulative_cash_released == 0.0
    assert at_sale.unpaid_proceeds_value == at_sale.total_value == pytest.approx(0.95)
    assert after.available_cash == after.total_value == pytest.approx(0.95)
    assert after.unpaid_proceeds_value == 0.0 and after.fully_settled
    assert all(snapshot.net_return == pytest.approx(-0.052) for snapshot in result.outcome.snapshots)
    assert result.specification.cost_prepaid_fraction == spec.cost_prepaid_fraction == 0.002
    release, = result.outcome.cash_releases
    assert release.available_at == reuse and release.amount == pytest.approx(0.95)
    assert release.basis == "research_assumption" and release.evidence_available_at is None
    assert result.outcome.label_available_at is None
    assert result.research_label_mature_at == spec.session_end_timestamps[9]
    assert result.production_eligible is result.outcome.production_eligible is False


def test_partial_sales_use_remaining_owned_quantity_and_do_not_duplicate_cost(context: TradeSimulationContext) -> None:
    spec = _spec()
    closes = spec.session_end_timestamps
    first = _sale(closes[0], fraction_of_owned=0.5, price_per_unit=120.0)
    second = _sale(closes[1], event_id="second-sale", proceeds_id="second-claim", fraction_of_owned=0.5, price_per_unit=80.0)
    result = simulate_ordinary_sales(spec.model_copy(update={"events": (first, second)}), context)
    day_one, day_two, day_three = result.outcome.snapshots[:3]
    assert {p.position_id: p.units for p in day_one.residual_positions} == pytest.approx({"shares": 0.005, "sale-claim": 0.6})
    assert day_one.available_cash == 0.0 and day_one.total_value == pytest.approx(1.1)
    assert {p.position_id: p.units for p in day_two.residual_positions} == pytest.approx({"shares": 0.0025, "second-claim": 0.2})
    assert day_two.available_cash == pytest.approx(0.6) and day_two.total_value == pytest.approx(1.05)
    assert day_three.available_cash == pytest.approx(0.8)
    assert day_three.tradable_value == pytest.approx(0.25) and day_three.total_value == pytest.approx(1.05)
    assert day_three.net_return == pytest.approx(0.048)
    assert [sale.normalized_proceeds for sale in result.outcome.executed_sales] == pytest.approx([0.6, 0.2])
    assert len(result.generated_event_ids) == 4 and len(result.generated_mark_keys) == 2
    assert result.outcome.managed_exit_timestamp is None
    assert result.next_open_settlement.residual_positions[0].units == pytest.approx(0.0025)
    assert not result.next_open_settlement.fully_settled


def test_tenth_close_has_one_unit_unpaid_claim_then_same_nav_cash_without_eleventh_bar(context: TradeSimulationContext) -> None:
    spec = _spec(policy="fixed_horizon")
    horizon = spec.session_end_timestamps[9]
    sale = _sale(horizon, reason="fixed_horizon", price_per_unit=100.0)
    result = simulate_ordinary_sales(spec.model_copy(update={"events": (sale,)}), context)
    assert len(result.outcome.snapshots) == len(result.specification.session_end_timestamps) == 10
    assert result.specification.session_end_timestamps == spec.session_end_timestamps
    assert all(mark.mark_at <= horizon for mark in result.specification.marks)
    final = result.outcome.snapshots[-1]
    claim, = final.residual_positions
    assert claim.position_id == sale.proceeds_id and claim.kind == "unpaid_proceeds"
    assert claim.units == claim.value == final.total_value == 1.0
    assert final.available_cash == 0.0 and not final.fully_settled
    assert final.net_return == -0.002 and not result.outcome.cash_releases
    assert result.generated_mark_keys == ((sale.proceeds_id, horizon),)
    next_open = result.next_open_settlement
    assert next_open.cutoff == datetime(2024, 1, 17, 14, 30, tzinfo=UTC)
    assert next_open.available_cash == final.total_value
    assert next_open.fully_settled and not next_open.residual_positions and not next_open.gaps
    assert not {"net_return", "gross_return", "total_value"}.intersection(next_open.model_dump())
    assert replay_holding(result.specification, simulation_policy_sha256=context.policy.sha256()) == result.outcome


@pytest.mark.parametrize("missing", ["absent", "explicit_unavailable"])
def test_missing_retained_share_marks_are_not_filled(context: TradeSimulationContext, missing: str) -> None:
    spec = _spec()
    close = spec.session_end_timestamps[0]
    marks = spec.marks[1:]
    if missing == "explicit_unavailable":
        marks = (*marks, UnavailableMark(position_id="shares", mark_at=close, reason="source missing"))
    result = simulate_ordinary_sales(spec.model_copy(update={
        "events": (_sale(close, fraction_of_owned=0.5),), "marks": marks,
    }), context)
    first = result.outcome.snapshots[0]
    assert first.tradable_value is first.total_value is first.net_return is None
    assert first.unpaid_proceeds_value == pytest.approx(0.475)
    assert ("mark_unavailable", "shares") in {(gap.code, gap.reference_id) for gap in first.gaps}
    assert result.research_label_mature_at is None
    assert tuple(mark for mark in result.specification.marks if mark.position_id == "shares") == marks


@pytest.mark.parametrize("with_sale", [False, True])
def test_unknown_corporate_cash_and_cvr_are_neither_paid_nor_marked(context: TradeSimulationContext, with_sale: bool) -> None:
    spec = _spec()
    close = spec.session_end_timestamps[0]
    action = CorporateActionEvent(
        event_id="corporate-action", effective_at=close, order=0, evidence=_evidence(close),
        owned_position_id="shares", owned_security_id="common:A", treatment="distribution" if with_sale else "replace",
        fractional_treatment="proportional", legs=(
            PositionLeg(position_id="corporate-cash", kind="unpaid_proceeds", security_id="common:A",
                        units_per_owned_unit=1.0, currency="USD", face_value_per_unit=380.0),
            PositionLeg(position_id="cvr", kind="contingent_right", security_id="right:A",
                        units_per_owned_unit=1.0, currency="USD", payout_cap_per_unit=35.0),
        ),
    )
    events = (action, _sale(spec.session_end_timestamps[1])) if with_sale else (action,)
    result = simulate_ordinary_sales(spec.model_copy(update={"events": events}), context)
    final = result.outcome.snapshots[-1]
    assert final.unpaid_proceeds_value is final.contingent_right_value is final.total_value is final.net_return is None
    assert final.available_cash == pytest.approx(0.95 if with_sale else 0.0)
    assert {p.position_id for p in final.residual_positions} == {"corporate-cash", "cvr"}
    assert all(p.value is None for p in final.residual_positions)
    assert not final.fully_settled and result.research_label_mature_at is None
    assert {mark.position_id for mark in result.specification.marks} <= {"shares", "sale-claim"}
    assert all(event.claim_id == "sale-claim" for event in result.specification.events if isinstance(event, PaymentEvent))
    assert len(result.generated_event_ids) == (2 if with_sale else 0)
    assert not result.next_open_settlement.fully_settled


@pytest.mark.parametrize("conflict", ["payment", "known_mark", "unavailable_mark"])
def test_existing_sale_payment_or_mark_is_rejected(context: TradeSimulationContext, conflict: str) -> None:
    spec = _spec()
    close, later = spec.session_end_timestamps[:2]
    events = (_sale(close),)
    marks = spec.marks
    if conflict == "payment":
        events = (*events, PaymentEvent(
            event_id="existing-payment", effective_at=later, order=0, evidence=_evidence(later),
            claim_id="sale-claim", amount_per_claim_unit=1.0, currency="USD", pending_proceeds_id="pending",
        ))
    elif conflict == "known_mark":
        marks = (*marks, _mark("sale-claim", close, 0.9))
    else:
        marks = (*marks, UnavailableMark(position_id="sale-claim", mark_at=close, reason="unknown receipt"))
    with pytest.raises(DataReadinessError, match="supplied sale"):
        simulate_ordinary_sales(spec.model_copy(update={"events": events, "marks": marks}), context)


@pytest.mark.parametrize("policy_hash", [None, "f" * 64])
def test_replay_requires_matching_explicit_simulation_context(context: TradeSimulationContext, policy_hash: str | None) -> None:
    spec = _spec()
    result = simulate_ordinary_sales(spec.model_copy(update={"events": (_sale(spec.session_end_timestamps[0]),)}), context)
    with pytest.raises(ValueError, match="context|policy"):
        replay_holding(result.specification, simulation_policy_sha256=policy_hash)


@pytest.mark.parametrize("replay", ["holding", "settlement"])
@pytest.mark.parametrize("supply_context", [False, True])
def test_reserved_ids_cannot_launder_stripped_simulation_basis(
    context: TradeSimulationContext, replay: str, supply_context: bool,
) -> None:
    spec = _spec()
    result = simulate_ordinary_sales(spec.model_copy(update={"events": (_sale(spec.session_end_timestamps[0]),)}), context)
    payload = result.specification.model_dump()
    for item in (*payload["events"], *payload["marks"]):
        for reference in item.get("evidence", ()):
            reference.pop("basis", None)
    stripped = HoldingSpecification.model_validate(payload)
    assert any(event.event_id.startswith("simulated-sale:") for event in stripped.events)
    policy_hash = context.policy.sha256() if supply_context else None
    with pytest.raises(ValueError, match="simulation|simulated|assumption|research|context"):
        if replay == "holding":
            replay_holding(stripped, simulation_policy_sha256=policy_hash)
        else:
            replay_next_open_settlement(stripped, simulation_policy_sha256=policy_hash)


@pytest.mark.parametrize("source", ["entry", "sale", "mark"])
def test_known_late_source_timestamp_survives_other_unknown_metadata(context: TradeSimulationContext, source: str) -> None:
    spec = _spec()
    late = spec.session_end_timestamps[9] + timedelta(days=4)
    mixed = (*_evidence(None), *_evidence(late))
    sale = _sale(spec.session_end_timestamps[1])
    changes: dict[str, Any] = {"entry_evidence": _evidence(None)}
    if source == "entry":
        changes["entry_evidence"] = mixed
    elif source == "sale":
        sale = sale.model_copy(update={"evidence": mixed})
    else:
        changes["marks"] = (spec.marks[0].model_copy(update={"evidence": mixed}), *spec.marks[1:])
    changes["events"] = (sale,)
    result = simulate_ordinary_sales(spec.model_copy(update=changes), context)
    assert all(snapshot.total_value is not None and not snapshot.gaps for snapshot in result.outcome.snapshots)
    assert result.outcome.label_available_at is None
    assert result.research_label_mature_at == late
    assert result.research_label_mature_at < RETRIEVED


def test_repeated_application_rejected_and_fresh_replay_deterministic(context: TradeSimulationContext) -> None:
    spec = _spec()
    spec = spec.model_copy(update={"events": (_sale(spec.session_end_timestamps[0]),)})
    result = simulate_ordinary_sales(spec, context)
    assert simulate_ordinary_sales(spec, context) == result
    with pytest.raises((ValueError, DataReadinessError), match="context|simulation|simulated|duplicate|supplied"):
        simulate_ordinary_sales(result.specification, context)
