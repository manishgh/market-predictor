"""Reviewed source instructions for fixed-horizon valuation, never source admission."""
from __future__ import annotations

from datetime import date
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from market_predictor.swing.contracts.holding_accounting import (
    AccountingGap,
    CashAvailabilityEvent,
    CorporateActionEvent,
    Evidence,
    HoldingContract,
    Identifier,
    Mark,
    PaymentEvent,
    Sha256,
)


class SourcePin(HoldingContract):
    path: Identifier
    sha256: Sha256


class PositionSourceBinding(HoldingContract):
    position_id: Identifier
    security_id: Identifier
    unit_id: Identifier
    provider_symbol: Identifier
    bars_sha256: Sha256
    first_session: date
    last_session: date
    ownership_evidence: Evidence

    @model_validator(mode="after")
    def ordered_interval(self) -> Self:
        if self.first_session > self.last_session:
            raise ValueError("position binding interval is reversed")
        return self


class ActionCoverage(HoldingContract):
    security_id: Identifier
    first_session: date
    last_session: date
    evidence: Evidence


CorporateFact = Annotated[
    CorporateActionEvent | PaymentEvent | CashAvailabilityEvent, Field(discriminator="kind")
]


class FixedHoldingRequest(HoldingContract):
    decision_id: Identifier
    decision_session: date
    security_id: Identifier
    sector: Identifier
    initial_position_id: Identifier
    bindings: tuple[PositionSourceBinding, ...]
    events: tuple[CorporateFact, ...] = ()
    non_share_marks: tuple[Mark, ...] = ()
    action_coverage: tuple[ActionCoverage, ...] = ()
    unresolved_source_gaps: tuple[AccountingGap, ...] = ()


class FixedHoldingBatch(HoldingContract):
    schema_version: Literal["market_predictor.fixed_holding_materialization"] = Field(alias="schema")
    source_selection: SourcePin
    interpretation_policy: SourcePin
    research_contract: SourcePin
    requests: Annotated[tuple[FixedHoldingRequest, ...], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def unique_requests(self) -> Self:
        ids = [request.decision_id for request in self.requests]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate holding request decision ID")
        return self
