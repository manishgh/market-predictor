"""Frozen, partial research admission for corrected initial-fit fixed targets."""
from __future__ import annotations

from datetime import date
from typing import Literal, Self

from pydantic import Field, model_validator

from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin


class DecisionCorrection(HoldingContract):
    security_id: Identifier
    parent_ticker: Identifier
    ticker: Identifier
    first_session: date
    last_session: date
    document_ids: tuple[Identifier, ...]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.first_session > self.last_session or not self.document_ids:
            raise ValueError("correction needs an ordered interval and reviewed documents")
        return self


class OfficialWindow(HoldingContract):
    security_id: Identifier
    first_session: date
    last_session: date
    document: SourcePin
    record_locator: Identifier
    reason: Identifier

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.first_session > self.last_session:
            raise ValueError("official contradiction interval is reversed")
        return self


class ReviewedCashDistributionScope(HoldingContract):
    """Reviewed temporal scope only, never valuation or payment authorization."""

    action_id: Identifier
    symbol: Identifier
    family: Literal["cash_dividends"]
    entitlement_basis: Literal["reviewed_cash_ex_distribution_cutoff"]
    record_sha256: Sha256
    ex_date: date
    record_date: date
    official_payable_date: date
    inventory: SourcePin
    archive: Identifier
    report_sha256: Sha256
    document_id: Identifier
    document: SourcePin
    record_locator: Identifier

    @model_validator(mode="after")
    def ex_distribution_scope(self) -> Self:
        if self.record_date < self.ex_date or self.official_payable_date < self.ex_date:
            raise ValueError("reviewed cash cutoff cannot replace a record-before-ex or payment-before-ex due-bill scope")
        return self


class CorrectedOutcomePolicy(HoldingContract):
    schema_version: Literal["market_predictor.corrected_outcomes"]
    scope: Literal["partial_source_bounded_initial_fit_research"]
    source_selection: SourcePin
    parent_config: SourcePin
    action_config: SourcePin
    action_archive: Identifier
    action_audit_sha256: Sha256
    symbol_corrections: SourcePin
    research_contract: SourcePin
    simulation_policy: SourcePin
    decision_corrections: tuple[DecisionCorrection, ...]
    official_windows: tuple[OfficialWindow, ...]
    reviewed_cash_distribution_scopes: tuple[ReviewedCashDistributionScope, ...] = ()
    decision_start: date
    numerical_end: date
    cost_prepaid_fraction: float
    dividend_policy: Literal["evidenced_marks_only_no_inferred_receivables"]
    managed_policy: Literal["unavailable"]
    batch_rows: int = Field(ge=1, le=256)
    promotion_eligible: Literal[False] = False

    @model_validator(mode="after")
    def frozen_scope(self) -> Self:
        scopes = self.reviewed_cash_distribution_scopes
        if len({scope.action_id for scope in scopes}) != len(scopes):
            raise ValueError("duplicate reviewed action scope")
        if self.cost_prepaid_fraction != 0.002:
            raise ValueError("corrected outcomes require the approved 20 bps prepaid cost")
        if self.decision_start != date(2019, 7, 9) or self.numerical_end != date(2024, 5, 28):
            raise ValueError("corrected outcomes cannot expand the frozen initial-fit numerical scope")
        for index, rule in enumerate(self.decision_corrections):
            for other in self.decision_corrections[index + 1:]:
                if (rule.security_id == other.security_id and rule.parent_ticker == other.parent_ticker
                        and rule.first_session <= other.last_session and other.first_session <= rule.last_session):
                    raise ValueError("overlapping decision corrections")
        return self
