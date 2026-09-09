"""Research lot accounting contracts. Provenance references do not admit source facts."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1, pattern=r"^\S(?:.*\S)?$")]
Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Timestamp = Annotated[datetime, AwareDatetime]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Currency = Literal["USD"] | None
PositionKind = Literal["tradable_shares", "unpaid_proceeds", "contingent_right"]


class HoldingContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class EvidenceReference(HoldingContract):
    reference: Identifier
    artifact_sha256: Sha256
    record_locator: Identifier
    interpretation_policy_sha256: Sha256
    retrieved_at: Timestamp
    available_at: Timestamp | None


class SimulationReference(EvidenceReference):
    basis: Literal["research_assumption"]

    @model_validator(mode="after")
    def no_observed_clock(self) -> Self:
        if self.available_at is not None:
            raise ValueError("research assumptions cannot claim observed historical availability")
        return self


Evidence = Annotated[tuple[SimulationReference | EvidenceReference, ...], Field(min_length=1)]


class KnownMark(HoldingContract):
    status: Literal["known"] = "known"
    position_id: Identifier
    mark_at: Timestamp
    value_per_unit: Nonnegative
    currency: Currency
    evidence: Evidence


class UnavailableMark(HoldingContract):
    status: Literal["unavailable"] = "unavailable"
    position_id: Identifier
    mark_at: Timestamp
    reason: Identifier


Mark = Annotated[KnownMark | UnavailableMark, Field(discriminator="status")]


class PositionLeg(HoldingContract):
    """Units delivered per actually owned input unit; face/cap is never a mark."""

    position_id: Identifier
    kind: PositionKind
    security_id: Identifier
    units_per_owned_unit: Positive
    currency: Currency
    face_value_per_unit: Nonnegative | None = None
    payout_cap_per_unit: Nonnegative | None = None


class HoldingEvent(HoldingContract):
    event_id: Identifier
    effective_at: Timestamp | None
    order: Annotated[int, Field(ge=0)] | None
    evidence: Evidence


class CorporateActionEvent(HoldingEvent):
    kind: Literal["corporate_action"] = "corporate_action"
    owned_position_id: Identifier
    owned_security_id: Identifier
    treatment: Literal["replace", "distribution"]
    fractional_treatment: Literal["proportional"] | None
    legs: Annotated[tuple[PositionLeg, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_legs(self) -> Self:
        ids = [leg.position_id for leg in self.legs]
        if len(ids) != len(set(ids)) or self.owned_position_id in ids:
            raise ValueError("corporate legs require distinct new position IDs")
        return self


class ExecutionEvent(HoldingEvent):
    """A fill supplied by the canonical label evaluator, never inferred from marks."""

    kind: Literal["execution"] = "execution"
    position_id: Identifier
    security_id: Identifier
    fraction_of_owned: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    price_per_unit: Positive
    currency: Currency
    proceeds_id: Identifier
    reason: Literal["managed_exit", "fixed_horizon"]


class PaymentEvent(HoldingEvent):
    """Extinguish the entire named claim and create a pending cash delivery."""

    kind: Literal["payment"] = "payment"
    claim_id: Identifier
    amount_per_claim_unit: Nonnegative
    currency: Currency
    pending_proceeds_id: Identifier


class CashAvailabilityEvent(HoldingEvent):
    """Evidence that the complete payment is spendable at effective_at."""

    kind: Literal["cash_available"] = "cash_available"
    payment_event_id: Identifier
    currency: Currency


Event = Annotated[
    CorporateActionEvent | ExecutionEvent | PaymentEvent | CashAvailabilityEvent,
    Field(discriminator="kind"),
]


class HoldingSpecification(HoldingContract):
    research_contract_sha256: Sha256
    decision_id: Identifier
    security_id: Identifier
    sector: Identifier
    initial_position_id: Identifier
    initial_entry_price: Positive
    initial_entry_timestamp: Timestamp
    price_basis: Literal["raw_with_no_adjustment"]
    currency: Currency
    entry_evidence: Evidence
    session_end_timestamps: Annotated[tuple[Timestamp, ...], Field(min_length=10)]
    cost_prepaid_fraction: Nonnegative
    policy: Literal["fixed_horizon", "managed"]
    events: tuple[Event, ...] = ()
    marks: tuple[Mark, ...] = ()

    @model_validator(mode="after")
    def coherent_scope(self) -> Self:
        ends = self.session_end_timestamps
        if tuple(sorted(set(ends))) != ends or self.initial_entry_timestamp >= ends[0]:
            raise ValueError("ten exact increasing session ends must follow entry")
        if self.cost_prepaid_fraction != 0.002:
            raise ValueError("the approved prepaid round-trip cost is fixed at 20 bps")
        ids = [event.event_id for event in self.events]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate event ID")
        position_ids = {self.initial_position_id}
        paid: set[str] = set()
        released: set[str] = set()
        for event in self.events:
            if event.effective_at is not None and event.effective_at < self.initial_entry_timestamp:
                raise ValueError("events before entry belong outside this lot specification")
            new_ids: list[str] = []
            if isinstance(event, CorporateActionEvent):
                new_ids = [leg.position_id for leg in event.legs]
            elif isinstance(event, ExecutionEvent):
                new_ids = [event.proceeds_id]
                if self.policy == "fixed_horizon" and event.reason != "fixed_horizon":
                    raise ValueError("fixed-horizon lots cannot contain managed exits")
                if event.reason == "fixed_horizon" and event.effective_at != ends[9]:
                    raise ValueError("fixed-horizon fills must occur at the exact horizon end")
                if event.reason == "managed_exit" and event.effective_at is not None and event.effective_at > ends[9]:
                    raise ValueError("managed exits cannot extend the ten-session forecast horizon")
                if event.effective_at is not None and event.effective_at > ends[9]:
                    raise ValueError("execution after the forecast horizon is outside the approved policy")
            elif isinstance(event, PaymentEvent):
                if event.claim_id in paid:
                    raise ValueError("duplicate payment for claim")
                paid.add(event.claim_id)
                new_ids = [event.pending_proceeds_id]
            else:
                if event.payment_event_id in released:
                    raise ValueError("duplicate cash availability for payment")
                released.add(event.payment_event_id)
            if position_ids.intersection(new_ids):
                raise ValueError("duplicate position/claim ID")
            position_ids.update(new_ids)
        mark_ids = [(mark.position_id, mark.mark_at) for mark in self.marks]
        if len(set(mark_ids)) != len(mark_ids):
            raise ValueError("duplicate mark for position/time")
        declared_classes = {self.initial_position_id: (self.security_id, "tradable_shares")}
        for event in self.events:
            if isinstance(event, CorporateActionEvent):
                declared_classes.update((leg.position_id, (leg.security_id, leg.kind)) for leg in event.legs)
        for event in self.events:
            if isinstance(event, CorporateActionEvent) and declared_classes.get(event.owned_position_id) != (
                event.owned_security_id, "tradable_shares",
            ):
                raise ValueError("corporate action must reference the actually owned or declared tradable class")
        return self


class AccountingGap(HoldingContract):
    code: Identifier
    reference_id: Identifier


class ResidualPosition(HoldingContract):
    position_id: Identifier
    kind: PositionKind
    security_id: Identifier
    units: Positive
    currency: Currency
    value: Nonnegative | None


class SessionSnapshot(HoldingContract):
    session_end_timestamp: Timestamp
    cumulative_cash_released: Nonnegative
    available_cash: Nonnegative
    tradable_value: Nonnegative | None
    unpaid_proceeds_value: Nonnegative | None
    contingent_right_value: Nonnegative | None
    total_value: Nonnegative | None
    gross_return: float | None
    net_return: float | None
    residual_positions: tuple[ResidualPosition, ...]
    fully_settled: bool
    gaps: tuple[AccountingGap, ...]
    source_available_at: Timestamp | None
    label_available_at: Timestamp | None
    latest_known_source_available_at: Timestamp | None = None


class CashRelease(HoldingContract):
    availability_event_id: Identifier
    payment_event_id: Identifier
    available_at: Timestamp
    evidence_available_at: Timestamp | None
    amount: Nonnegative
    source_kind: Literal["sale_proceeds", "corporate_payment"]
    basis: Literal["source_interpretation", "research_assumption"] = "source_interpretation"


class ExecutedSale(HoldingContract):
    source_event_id: Identifier
    executed_at: Timestamp
    security_id: Identifier
    position_id: Identifier
    normalized_proceeds: Nonnegative


class LotOutcome(HoldingContract):
    research_contract_sha256: Sha256
    decision_id: Identifier
    security_id: Identifier
    sector: Identifier
    policy: Literal["fixed_horizon", "managed"]
    initial_entry_price: Positive
    initial_entry_timestamp: Timestamp
    cost_prepaid_fraction: Nonnegative
    snapshots: Annotated[tuple[SessionSnapshot, ...], Field(min_length=10)]
    managed_exit_timestamp: Timestamp | None
    cash_releases: tuple[CashRelease, ...]
    executed_sales: tuple[ExecutedSale, ...]
    residual_positions: tuple[ResidualPosition, ...]
    fully_settled: bool
    label_available_at: Timestamp | None
    simulation_policy_sha256: Sha256 | None = None
    production_eligible: Literal[False] = False


class SettlementSnapshot(HoldingContract):
    """Cash/units only; no new price observation or extended investment return."""

    cutoff: Timestamp
    available_cash: Nonnegative
    residual_positions: tuple[ResidualPosition, ...]
    gaps: tuple[AccountingGap, ...]
    fully_settled: bool


class BenchmarkTarget(HoldingContract):
    benchmark_security_id: Identifier
    horizon_gross_return: float | None
    fixed_horizon_excess_return: float | None
    managed_horizon_excess_return: float | None


class HoldingTargets(HoldingContract):
    research_contract_sha256: Sha256
    decision_id: Identifier
    initial_entry_timestamp: Timestamp
    horizon_end_timestamp: Timestamp
    managed_exit_timestamp: Timestamp | None
    fixed_horizon_gross_return: float | None
    fixed_horizon_net_return: float | None
    managed_horizon_gross_return: float | None
    managed_horizon_net_return: float | None
    benchmarks: tuple[BenchmarkTarget, ...]
    label_available_at: Timestamp | None
    production_eligible: Literal[False] = False
