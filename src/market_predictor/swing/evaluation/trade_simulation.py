"""Model ordinary-sale cash flows without inventing corporate or broker evidence."""
from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path

import exchange_calendars as xcals

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    Event,
    EvidenceReference,
    ExecutionEvent,
    HoldingSpecification,
    KnownMark,
    Mark,
    PaymentEvent,
    SimulationReference,
)
from market_predictor.swing.contracts.trade_simulation import (
    SimulatedHolding,
    TradeSimulationContext,
    TradeSimulationPolicy,
)
from market_predictor.swing.evaluation.holding_accounting import replay_holding, replay_next_open_settlement


def load_trade_simulation_context(path: Path, *, expected_sha256: str) -> TradeSimulationContext:
    """Pin a local assumption document; its read time is not historical availability."""
    if path.stat().st_size > 65536 or file_sha256(path) != expected_sha256:
        raise DataReadinessError("trade simulation policy size or file hash differs")
    policy = TradeSimulationPolicy.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))
    if file_sha256(path) != expected_sha256:
        raise DataReadinessError("trade simulation policy changed during loading")
    return TradeSimulationContext(policy=policy, policy_reference=EvidenceReference(
        reference=str(path.resolve()), artifact_sha256=expected_sha256,
        record_locator="research assumption document; not a historical broker receipt",
        interpretation_policy_sha256=policy.sha256(), retrieved_at=datetime.now(UTC), available_at=None,
    ))


def simulation_metadata(context: TradeSimulationContext) -> dict[str, object]:
    return {"scope": context.policy.scope, "policy_sha256": context.policy.sha256(),
        "policy_reference": context.policy_reference.model_dump(mode="json"),
        "proceeds_reuse": context.policy.sale_proceeds_reuse, "production_eligible": False}


def simulation_replay_metadata(result: SimulatedHolding) -> dict[str, object]:
    """Compact replay identity and unit-lot settlement, never account-funded cash."""
    return {"input_specification_sha256": result.input_specification_sha256,
        "simulation_policy_sha256": result.simulation_policy_sha256,
        "generated_event_ids": list(result.generated_event_ids),
        "generated_mark_keys": [[position, timestamp.isoformat()] for position, timestamp in result.generated_mark_keys],
        "next_open_settlement": result.next_open_settlement.model_dump(mode="json"),
        "settlement_amount_basis": "per_unit_initial_entry_notional_not_account_cash",
        "production_eligible": False}


def simulate_ordinary_sales(spec: HoldingSpecification, context: TradeSimulationContext) -> SimulatedHolding:
    """Generate only sale-claim mechanics, then invoke the one canonical lot replay.

    Existing executions must already come from the canonical fill evaluator. This
    function neither chooses a fill nor certifies its source. Corporate claims and
    observed clocks are never repaired by an ordinary-sale assumption.
    """
    spec = HoldingSpecification.model_validate_json(spec.model_dump_json())
    context = TradeSimulationContext.model_validate_json(context.model_dump_json())
    if (context.policy_reference.interpretation_policy_sha256 != context.policy.sha256()
            or context.policy_reference.available_at is not None):
        raise DataReadinessError("simulation reference must bind policy without claimed historical availability")
    # Validate exact session closes and input execution ownership through canonical replay.
    replay_holding(spec)
    calendar = xcals.get_calendar("XNYS")
    sales = [event for event in spec.events if isinstance(event, ExecutionEvent)]
    claims = {sale.proceeds_id for sale in sales}
    if any(isinstance(event, PaymentEvent) and event.claim_id in claims for event in spec.events):
        raise DataReadinessError("simulation cannot replace or duplicate supplied sale payments")
    if any(mark.position_id in claims for mark in spec.marks):
        raise DataReadinessError("simulation cannot replace supplied sale-proceeds marks")
    events: list[Event] = list(spec.events)
    marks: list[Mark] = list(spec.marks)
    generated_events: list[str] = []
    generated_marks: list[tuple[str, datetime]] = []
    for sale in sorted(sales, key=lambda item: item.event_id):
        at = sale.effective_at
        if at is None or sale.order is None or sale.currency != "USD":
            raise DataReadinessError("simulated proceeds require a dated, ordered USD execution")
        session = at.astimezone(UTC).date().isoformat()
        if not calendar.is_session(session) or not (
            calendar.session_open(session).to_pydatetime() <= at <= calendar.session_close(session).to_pydatetime()
        ):
            raise DataReadinessError("simulated execution must lie within an actual XNYS session")
        reuse = calendar.session_open(calendar.next_session(session)).to_pydatetime()
        prefix = "simulated-sale:" + json_sha256({"decision": spec.decision_id, "sale": sale.event_id,
            "policy": context.policy.sha256()})
        payment_id, cash_id, pending_id = prefix + ":payment", prefix + ":cash", prefix + ":pending"
        order = max((event.order for event in events if event.effective_at == reuse and event.order is not None),
            default=-1) + 1
        evidence = (*sale.evidence, SimulationReference(**context.policy_reference.model_dump(), basis="research_assumption"))
        events.extend((
            PaymentEvent(event_id=payment_id, effective_at=reuse, order=order, evidence=evidence,
                claim_id=sale.proceeds_id, amount_per_claim_unit=1.0, currency="USD", pending_proceeds_id=pending_id),
            CashAvailabilityEvent(event_id=cash_id, effective_at=reuse, order=order + 1, evidence=evidence,
                payment_event_id=payment_id, currency="USD"),
        ))
        generated_events.extend((payment_id, cash_id))
        for close in spec.session_end_timestamps:
            if at <= close < reuse:
                marks.append(KnownMark(position_id=sale.proceeds_id, mark_at=close,
                    value_per_unit=1.0, currency="USD", evidence=evidence))
                generated_marks.append((sale.proceeds_id, close))
    derived = HoldingSpecification.model_validate({**spec.model_dump(), "events": tuple(events), "marks": tuple(marks)})
    outcome = replay_holding(derived, simulation_policy_sha256=context.policy.sha256())
    first_ten = outcome.snapshots[:10]
    # This proxy is separate from label_available_at and never certifies an as-of retraining history.
    maturity = None
    if all(snapshot.total_value is not None and not snapshot.gaps for snapshot in first_ten):
        known = [snapshot.latest_known_source_available_at for snapshot in first_ten
            if snapshot.latest_known_source_available_at is not None]
        maturity = max((first_ten[-1].session_end_timestamp, *known))
    return SimulatedHolding(input_specification_sha256=json_sha256(spec.model_dump(mode="json")),
        simulation_policy_sha256=context.policy.sha256(), policy_reference=context.policy_reference,
        generated_event_ids=tuple(generated_events), generated_mark_keys=tuple(generated_marks),
        specification=derived, outcome=outcome, research_label_mature_at=maturity,
        next_open_settlement=replay_next_open_settlement(derived, simulation_policy_sha256=context.policy.sha256()))
