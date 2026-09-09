"""USD lot transitions shared by labels and funded accounting, never source admission.

Amounts and quantities are normalized to one unit of initial entry notional. No
OHLC execution, reinvestment, forced liquidation, or price-ratio adapter lives here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

import exchange_calendars as xcals

from market_predictor.swing.contracts.holding_accounting import (
    AccountingGap,
    BenchmarkTarget,
    CashAvailabilityEvent,
    CashRelease,
    CorporateActionEvent,
    Currency,
    Event,
    Evidence,
    ExecutedSale,
    ExecutionEvent,
    HoldingSpecification,
    HoldingTargets,
    KnownMark,
    LotOutcome,
    PaymentEvent,
    PositionKind,
    ResidualPosition,
    SessionSnapshot,
    UnavailableMark,
)


@dataclass(frozen=True)
class _Position:
    position_id: str
    security_id: str
    kind: PositionKind
    units: float
    currency: Currency
    evidence_at: datetime | None
    sale_proceeds: bool = False


@dataclass
class _State:
    positions: dict[str, _Position]
    pending: dict[str, tuple[str, float, datetime | None, bool]] = field(default_factory=dict)
    skipped: set[str] = field(default_factory=set)
    disposed: set[str] = field(default_factory=set)
    skipped_payments: set[str] = field(default_factory=set)
    released: list[CashRelease] = field(default_factory=list)
    sales: list[ExecutedSale] = field(default_factory=list)
    gaps: list[AccountingGap] = field(default_factory=list)
    managed_exit_at: datetime | None = None
    source_times: list[datetime | None] = field(default_factory=list)

    def gap(self, code: str, reference: str) -> None:
        self.gaps.append(AccountingGap(code=code, reference_id=reference))


def _evidence_at(evidence: Evidence) -> datetime | None:
    if any(item.available_at is None for item in evidence):
        return None
    return max(item.available_at for item in evidence if item.available_at is not None)


def _latest(times: list[datetime | None]) -> datetime | None:
    return None if not times or any(t is None for t in times) else max(t for t in times if t is not None)


def _causal_time(position: _Position, evidence: Evidence) -> datetime | None:
    return _latest([_evidence_at(evidence), position.evidence_at])


def _number(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("non-finite or negative normalized lot arithmetic")
    return value


def _validate_calendar(spec: HoldingSpecification) -> None:
    calendar = xcals.get_calendar("XNYS")
    ends = spec.session_end_timestamps
    sessions = calendar.sessions_in_range(ends[0].astimezone(UTC).date(), ends[-1].astimezone(UTC).date())
    expected = tuple(calendar.session_close(session).to_pydatetime() for session in sessions)
    if ends != expected or spec.initial_entry_timestamp != calendar.session_open(sessions[0]).to_pydatetime():
        raise ValueError("holding requires consecutive exact XNYS closes and first-session open entry")


def _corporate(state: _State, event: CorporateActionEvent) -> bool:
    owned = state.positions.get(event.owned_position_id)
    if owned is None:
        if event.owned_position_id not in state.disposed | state.skipped:
            raise ValueError("corporate action references a position not yet created or never owned")
        state.skipped.update(leg.position_id for leg in event.legs)
        return True
    if owned.security_id != event.owned_security_id or owned.kind != "tradable_shares":
        raise ValueError("corporate transformation must match the actually owned security class")
    if event.fractional_treatment is None:
        state.gap("fractional_treatment_unavailable", event.event_id)
        return False
    if any(leg.currency is None for leg in event.legs):
        state.gap("currency_unavailable", event.event_id)
        return False
    if event.treatment == "replace":
        del state.positions[owned.position_id]
        state.disposed.add(owned.position_id)
    for leg in event.legs:
        state.positions[leg.position_id] = _Position(
            leg.position_id, leg.security_id, leg.kind,
            _number(owned.units * leg.units_per_owned_unit), leg.currency,
            _causal_time(owned, event.evidence),
        )
    return True


def _execute(state: _State, event: ExecutionEvent) -> bool:
    owned = state.positions.get(event.position_id)
    if owned is None or owned.kind != "tradable_shares" or owned.security_id != event.security_id:
        raise ValueError("execution requires actually owned tradable shares of the named class")
    if event.currency is None:
        state.gap("currency_unavailable", event.event_id)
        return False
    sold = _number(owned.units * event.fraction_of_owned)
    if event.fraction_of_owned == 1.0:
        del state.positions[owned.position_id]
        state.disposed.add(owned.position_id)
    else:
        state.positions[owned.position_id] = _Position(
            owned.position_id, owned.security_id, owned.kind,
            _number(owned.units - sold), owned.currency, owned.evidence_at,
        )
    # These are dollar claim units, not an implicit dollar claim mark.
    proceeds = _number(sold * event.price_per_unit)
    state.positions[event.proceeds_id] = _Position(
        event.proceeds_id, owned.security_id, "unpaid_proceeds", proceeds,
        event.currency, _causal_time(owned, event.evidence), True,
    )
    assert event.effective_at is not None
    state.sales.append(ExecutedSale(
        source_event_id=event.event_id, executed_at=event.effective_at,
        security_id=owned.security_id, position_id=owned.position_id, normalized_proceeds=proceeds,
    ))
    if event.reason == "managed_exit" and not any(p.kind == "tradable_shares" for p in state.positions.values()):
        state.managed_exit_at = event.effective_at
    return True


def _pay(state: _State, event: PaymentEvent) -> bool:
    if event.claim_id in state.skipped:
        state.skipped.add(event.pending_proceeds_id)
        state.skipped_payments.add(event.event_id)
        return True
    claim = state.positions.get(event.claim_id)
    if claim is None or claim.kind == "tradable_shares" or event.claim_id in {p[0] for p in state.pending.values()}:
        raise ValueError("payment requires an existing, unpaid, non-share claim")
    if claim.sale_proceeds and event.amount_per_claim_unit != 1.0:
        raise ValueError("sale payment must reconcile the executed dollar proceeds")
    if event.currency is None:
        state.gap("currency_unavailable", event.event_id)
        return False
    amount = _number(claim.units * event.amount_per_claim_unit)
    evidence_at = _causal_time(claim, event.evidence)
    del state.positions[claim.position_id]
    if amount > 0:
        state.positions[event.pending_proceeds_id] = _Position(
            event.pending_proceeds_id, claim.security_id, "unpaid_proceeds", amount,
            event.currency, evidence_at, claim.sale_proceeds,
        )
    state.pending[event.event_id] = (event.pending_proceeds_id, amount, evidence_at, claim.sale_proceeds)
    return True


def _release(state: _State, event: CashAvailabilityEvent) -> bool:
    if event.payment_event_id in state.skipped_payments:
        return True
    pending = state.pending.get(event.payment_event_id)
    if pending is None:
        raise ValueError("cash availability must follow its matching payment exactly once")
    if event.currency is None:
        state.gap("currency_unavailable", event.event_id)
        return False
    position_id, amount, evidence_at, sale = state.pending.pop(event.payment_event_id)
    state.positions.pop(position_id, None)
    known = _evidence_at(event.evidence)
    assert event.effective_at is not None
    state.released.append(CashRelease(
        availability_event_id=event.event_id, payment_event_id=event.payment_event_id,
        available_at=event.effective_at, evidence_available_at=_latest([evidence_at, known]),
        amount=amount, source_kind="sale_proceeds" if sale else "corporate_payment",
    ))
    return True


def _provably_unearned(state: _State, event: Event) -> bool:
    if isinstance(event, CorporateActionEvent):
        return event.owned_position_id in state.disposed | state.skipped
    if isinstance(event, PaymentEvent):
        return event.claim_id in state.skipped
    if isinstance(event, CashAvailabilityEvent):
        return event.payment_event_id in state.skipped_payments
    return False


def _record_unearned(state: _State, event: Event) -> None:
    if isinstance(event, CorporateActionEvent):
        state.skipped.update(leg.position_id for leg in event.legs)
    elif isinstance(event, PaymentEvent):
        state.skipped.add(event.pending_proceeds_id)
        state.skipped_payments.add(event.event_id)


def _propagate_unearned_claims(state: _State, events: tuple[Event, ...]) -> None:
    """Descendants of an unearned claim cannot become payable, even if undated."""
    previous = -1
    while previous != len(state.skipped) + len(state.skipped_payments):
        previous = len(state.skipped) + len(state.skipped_payments)
        for event in events:
            if (isinstance(event, PaymentEvent) and event.claim_id in state.skipped
                    or isinstance(event, CorporateActionEvent) and event.owned_position_id in state.skipped):
                _record_unearned(state, event)


def _state_at(spec: HoldingSpecification, cutoff: datetime) -> _State:
    entry_known = _evidence_at(spec.entry_evidence)
    state = _State({spec.initial_position_id: _Position(
        spec.initial_position_id, spec.security_id, "tradable_shares",
        _number(1.0 / spec.initial_entry_price), spec.currency, entry_known,
    )})
    state.source_times.append(entry_known)
    if spec.currency is None:
        state.gap("currency_unavailable", spec.initial_position_id)
    if state.gaps:
        return state
    due = [e for e in spec.events if e.effective_at is not None and e.effective_at <= cutoff]
    due.sort(key=lambda e: (e.effective_at or cutoff, e.order if e.order is not None else -1))
    for event in due:
        if _provably_unearned(state, event):
            _record_unearned(state, event)
            _propagate_unearned_claims(state, spec.events)
            continue
        if event.effective_at == spec.initial_entry_timestamp:
            state.gap("entry_event_order_unavailable", event.event_id)
            break
        same_time = [other for other in due if other.effective_at == event.effective_at
            and not _provably_unearned(state, other)]
        if event.order is None or any(other.order == event.order and other.event_id != event.event_id for other in same_time):
            state.gap("event_order_unavailable", event.event_id)
            break
        state.source_times.append(_evidence_at(event.evidence))
        if isinstance(event, CorporateActionEvent):
            applied = _corporate(state, event)
        elif isinstance(event, ExecutionEvent):
            applied = _execute(state, event)
        elif isinstance(event, PaymentEvent):
            applied = _pay(state, event)
        else:
            applied = _release(state, event)
        if not applied:
            break
    for event in spec.events:
        if event.effective_at is None:
            # A dated disposal cannot order an undated corporate entitlement.
            if not isinstance(event, CorporateActionEvent) and _provably_unearned(state, event):
                _record_unearned(state, event)
            else:
                state.gap("effective_date_unavailable", event.event_id)
    return state


def _snapshot(spec: HoldingSpecification, cutoff: datetime, state: _State) -> SessionSnapshot:
    marks = {m.position_id: m for m in spec.marks if m.mark_at == cutoff}
    structural_gap = bool(state.gaps)
    components: dict[PositionKind, list[float | None]] = {
        "tradable_shares": [], "unpaid_proceeds": [], "contingent_right": [],
    }
    positions: list[ResidualPosition] = []
    for position in sorted(state.positions.values(), key=lambda p: p.position_id):
        mark = marks.get(position.position_id)
        value: float | None = None
        if mark is None or isinstance(mark, UnavailableMark):
            state.gap("mark_unavailable", position.position_id)
        elif mark.currency is None or position.currency is None:
            state.gap("currency_unavailable", position.position_id)
        elif isinstance(mark, KnownMark) and not structural_gap:
            state.source_times.append(_evidence_at(mark.evidence))
            value = _number(position.units * mark.value_per_unit)
        components[position.kind].append(value)
        positions.append(ResidualPosition(
            position_id=position.position_id, security_id=position.security_id, kind=position.kind,
            units=position.units, currency=position.currency, value=value,
        ))
    values = {
        kind: None if structural_gap or any(v is None for v in entries)
        else _number(math.fsum(v for v in entries if v is not None))
        for kind, entries in components.items()
    }
    cash = _number(math.fsum(release.amount for release in state.released))
    gaps = tuple(sorted(set((g.code, g.reference_id) for g in state.gaps)))
    total = None if gaps else _number(cash + math.fsum(v for v in values.values() if v is not None))
    gross = None if total is None else total - 1.0
    source_available = _latest(state.source_times)
    return SessionSnapshot(
        session_end_timestamp=cutoff, cumulative_cash_released=cash, available_cash=cash,
        tradable_value=values["tradable_shares"], unpaid_proceeds_value=values["unpaid_proceeds"],
        contingent_right_value=values["contingent_right"], total_value=total, gross_return=gross,
        net_return=None if gross is None else gross - spec.cost_prepaid_fraction,
        residual_positions=tuple(positions), fully_settled=not positions and not state.pending and not gaps,
        gaps=tuple(AccountingGap(code=code, reference_id=reference) for code, reference in gaps),
        source_available_at=source_available,
        label_available_at=None if total is None or source_available is None else max(cutoff, source_available),
    )


def replay_holding(spec: HoldingSpecification) -> LotOutcome:
    """Replay supplied closes, including residuals beyond the fixed day-ten target.

    Each close uses economic timestamps, never later marks or payments. Delayed
    publication changes label availability, not historical cash or valuation time.
    Cash funding consumers deduplicate by availability_event_id within the lot and
    preserve their ordinary-sale rule. Evidence clocks are metadata, not cash dates.
    """
    if not isinstance(spec, HoldingSpecification):
        raise TypeError("replay_holding requires a HoldingSpecification")
    spec = HoldingSpecification.model_validate(spec.model_dump())
    _validate_calendar(spec)
    snapshots: list[SessionSnapshot] = []
    for cutoff in spec.session_end_timestamps:
        state = _state_at(spec, cutoff)
        snapshots.append(_snapshot(spec, cutoff, state))
    final = snapshots[-1]
    return LotOutcome(
        research_contract_sha256=spec.research_contract_sha256,
        decision_id=spec.decision_id, security_id=spec.security_id, sector=spec.sector, policy=spec.policy,
        initial_entry_price=spec.initial_entry_price, initial_entry_timestamp=spec.initial_entry_timestamp,
        cost_prepaid_fraction=spec.cost_prepaid_fraction, snapshots=tuple(snapshots),
        managed_exit_timestamp=state.managed_exit_at, cash_releases=tuple(state.released), executed_sales=tuple(state.sales),
        residual_positions=final.residual_positions, fully_settled=final.fully_settled,
        label_available_at=snapshots[9].label_available_at,
    )


def project_holding_targets(
    fixed: LotOutcome, managed: LotOutcome, *, benchmarks: tuple[LotOutcome, ...] = (),
) -> HoldingTargets:
    """Compare the first ten sessions only, including claims earned before exits."""
    if fixed.policy != "fixed_horizon" or managed.policy != "managed":
        raise ValueError("target projection needs explicitly fixed and managed outcomes")
    if (
        (fixed.decision_id, fixed.security_id, fixed.sector, fixed.initial_entry_price, fixed.cost_prepaid_fraction)
        != (managed.decision_id, managed.security_id, managed.sector, managed.initial_entry_price, managed.cost_prepaid_fraction)
    ):
        raise ValueError("fixed and managed outcomes must describe the same entry lot")
    horizon = tuple(s.session_end_timestamp for s in fixed.snapshots[:10])
    for outcome in (managed, *benchmarks):
        if outcome.research_contract_sha256 != fixed.research_contract_sha256:
            raise ValueError("target outcomes must share the same research contract")
        if outcome.initial_entry_timestamp != fixed.initial_entry_timestamp or tuple(
            s.session_end_timestamp for s in outcome.snapshots[:10]
        ) != horizon:
            raise ValueError("stock and benchmark executable entry/horizon timestamps must match")
    if any(b.policy != "fixed_horizon" for b in benchmarks) or len({b.security_id for b in benchmarks}) != len(benchmarks):
        raise ValueError("benchmarks must be unique fixed-horizon outcomes")
    fixed_end, managed_end = fixed.snapshots[9], managed.snapshots[9]
    comparisons = []
    for benchmark in benchmarks:
        gross = benchmark.snapshots[9].gross_return
        comparisons.append(BenchmarkTarget(
            benchmark_security_id=benchmark.security_id, horizon_gross_return=gross,
            fixed_horizon_excess_return=None if gross is None or fixed_end.net_return is None else fixed_end.net_return - gross,
            managed_horizon_excess_return=None if gross is None or managed_end.net_return is None else managed_end.net_return - gross,
        ))
    return HoldingTargets(
        research_contract_sha256=fixed.research_contract_sha256,
        decision_id=fixed.decision_id, initial_entry_timestamp=fixed.initial_entry_timestamp,
        horizon_end_timestamp=horizon[-1], managed_exit_timestamp=managed.managed_exit_timestamp,
        fixed_horizon_gross_return=fixed_end.gross_return, fixed_horizon_net_return=fixed_end.net_return,
        managed_horizon_gross_return=managed_end.gross_return, managed_horizon_net_return=managed_end.net_return,
        benchmarks=tuple(comparisons),
        label_available_at=_latest([o.snapshots[9].label_available_at for o in (fixed, managed, *benchmarks)]),
    )
