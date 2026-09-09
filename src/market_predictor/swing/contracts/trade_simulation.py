"""Explicit research assumptions, separate from historical market observations."""
from __future__ import annotations

from typing import Literal

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import (
    EvidenceReference,
    HoldingContract,
    HoldingSpecification,
    LotOutcome,
    SettlementSnapshot,
    Sha256,
    Timestamp,
)


class TradeSimulationPolicy(HoldingContract):
    schema_version: Literal["market_predictor.ordinary_trade_simulation"]
    scope: Literal["retrospective_research_only"]
    fill_source: Literal["canonical_execution_events_only"]
    sale_proceeds_valuation: Literal["one_usd_per_executed_dollar"]
    sale_proceeds_reuse: Literal["next_xnys_session_open"]
    corporate_claims: Literal["no_inferred_payment_or_valuation"]
    cost_policy: Literal["preserve_existing_prepaid_cost"]
    label_clock: Literal["separate_retrospective_maturity_not_observed_availability"]
    production_eligible: Literal[False]

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json"))


class TradeSimulationContext(HoldingContract):
    policy: TradeSimulationPolicy
    policy_reference: EvidenceReference


class SimulatedHolding(HoldingContract):
    scope: Literal["retrospective_research_only"] = "retrospective_research_only"
    input_specification_sha256: Sha256
    simulation_policy_sha256: Sha256
    policy_reference: EvidenceReference
    generated_event_ids: tuple[str, ...]
    generated_mark_keys: tuple[tuple[str, Timestamp], ...]
    specification: HoldingSpecification
    outcome: LotOutcome
    next_open_settlement: SettlementSnapshot
    research_label_mature_at: Timestamp | None
    production_eligible: Literal[False] = False
