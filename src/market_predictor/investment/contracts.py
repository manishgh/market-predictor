"""Explicit marked-value forecasts over the unchanged shared holding kernel."""
from __future__ import annotations

import math
from datetime import date
from typing import Literal, Self

from pydantic import Field, model_validator

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256, Timestamp

HoldingRole = Literal["stock", "spy", "qqq", "sector"]
HOLDING_ROLES: tuple[HoldingRole, ...] = ("stock", "spy", "qqq", "sector")


class InvestmentTargetPolicy(HoldingContract):
    contract: Literal["market_predictor.investment_target_policy"] = "market_predictor.investment_target_policy"
    horizon_sessions: Literal[63, 252]
    target: Literal["marked_holding_net_excess_vs_spy"] = "marked_holding_net_excess_vs_spy"
    entry: Literal["next_xnys_session_open"] = "next_xnys_session_open"
    terminal: Literal["marked_value_no_liquidation"] = "marked_value_no_liquidation"
    price_basis: Literal["raw_with_no_adjustment"] = "raw_with_no_adjustment"
    cost_assumption: Literal["research_round_trip_prepaid_at_entry"] = "research_round_trip_prepaid_at_entry"
    round_trip_cost_bps: float = Field(default=20.0, ge=20.0, le=20.0)
    benchmark_basis: Literal["net_stock_minus_gross_benchmark"] = "net_stock_minus_gross_benchmark"

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json"))


class InvestmentSourceIdentity(HoldingContract):
    """Declared source bytes and interpretation; identity does not prove admission."""
    reference: Identifier
    artifact_sha256: Sha256
    interpretation_policy_sha256: Sha256


class InvestmentHoldingBinding(HoldingContract):
    role: HoldingRole
    security_id: Identifier
    symbol: Identifier
    specification_sha256: Sha256
    sources: tuple[InvestmentSourceIdentity, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        if self.symbol != self.symbol.upper() or any(c.isspace() for c in self.symbol):
            raise ValueError("holding symbol must be canonical uppercase")
        references = [source.reference for source in self.sources]
        if len(set(references)) != len(references):
            raise ValueError("holding source references must be unique")
        if tuple(references) != tuple(sorted(references)):
            raise ValueError("holding sources must be sorted by reference")
        return self


class InvestmentTargetRequest(HoldingContract):
    contract: Literal["market_predictor.investment_target_request"] = "market_predictor.investment_target_request"
    policy: InvestmentTargetPolicy
    decision_id: Identifier
    decision_session: date
    decision_time_utc: Timestamp
    sector: Identifier
    sector_membership_source: InvestmentSourceIdentity
    holdings: tuple[InvestmentHoldingBinding, ...]

    @model_validator(mode="after")
    def complete_bindings(self) -> Self:
        if tuple(binding.role for binding in self.holdings) != HOLDING_ROLES:
            raise ValueError("holdings must bind stock, SPY, QQQ and sector in canonical order")
        if len({binding.security_id for binding in self.holdings}) != 4:
            raise ValueError("stock and benchmark security identities must be distinct")
        if len({binding.symbol for binding in self.holdings}) != 4:
            raise ValueError("stock and benchmark symbols must be distinct")
        if self.holdings[1].symbol != "SPY" or self.holdings[2].symbol != "QQQ":
            raise ValueError("broad and growth benchmarks must be SPY and QQQ")
        return self

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json"))


class InvestmentHoldingValue(HoldingContract):
    role: HoldingRole
    security_id: Identifier
    specification_sha256: Sha256
    gross_return: float | None
    net_return: float | None
    label_available_at_utc: Timestamp | None
    missing_reasons: tuple[Identifier, ...]


class InvestmentTargetResult(HoldingContract):
    contract: Literal["market_predictor.investment_target_result"] = "market_predictor.investment_target_result"
    request_sha256: Sha256
    policy_sha256: Sha256
    decision_id: Identifier
    horizon_sessions: Literal[63, 252]
    entry_time_utc: Timestamp
    horizon_end_utc: Timestamp
    holdings: tuple[InvestmentHoldingValue, ...]
    net_excess_vs_spy: float | None
    net_excess_vs_qqq: float | None
    net_excess_vs_sector: float | None
    label_available_at_utc: Timestamp | None
    label_eligible: bool
    status: Literal["available", "unavailable"]
    missing_reasons: tuple[Identifier, ...]
    scope: Literal["research_only"] = "research_only"
    training_ready: Literal[False] = False
    promotion_eligible: Literal[False] = False
    production_eligible: Literal[False] = False

    @model_validator(mode="after")
    def consistent_values(self) -> Self:
        if tuple(value.role for value in self.holdings) != HOLDING_ROLES:
            raise ValueError("result needs all four holding values")
        if self.entry_time_utc >= self.horizon_end_utc:
            raise ValueError("horizon must follow entry")
        complete = not self.missing_reasons and all(not value.missing_reasons for value in self.holdings)
        if self.label_eligible != complete or (self.status == "available") != complete:
            raise ValueError("target availability and reasons disagree")
        clocks = [value.label_available_at_utc for value in self.holdings]
        if complete:
            if any(clock is None for clock in clocks) or self.label_available_at_utc != max(
                self.horizon_end_utc, *(clock for clock in clocks if clock is not None)
            ):
                raise ValueError("target availability must include every holding and horizon")
        elif self.label_available_at_utc is not None:
            raise ValueError("unavailable target cannot claim label availability")
        stock = self.holdings[0]
        for value in self.holdings:
            if (value.gross_return is None) != (value.net_return is None):
                raise ValueError("gross and net values must be available together")
            if value.gross_return is not None and value.net_return is not None and not math.isclose(
                value.gross_return - 0.002, value.net_return, abs_tol=1e-12
            ):
                raise ValueError("holding cost must be applied exactly once")
            if complete and value.gross_return is None:
                raise ValueError("available target needs every valuation")
        for benchmark, excess in zip(self.holdings[1:],
            (self.net_excess_vs_spy, self.net_excess_vs_qqq, self.net_excess_vs_sector), strict=True):
            expected = None if stock.net_return is None or benchmark.gross_return is None else stock.net_return - benchmark.gross_return
            if (expected is None) != (excess is None) or (
                expected is not None and excess is not None and not math.isclose(expected, excess, abs_tol=1e-12)
            ):
                raise ValueError("benchmark excess does not reconcile")
        return self
