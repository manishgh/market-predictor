"""Explicit historical-backtest assumptions; never observed account cash."""
from __future__ import annotations

from typing import Literal

import exchange_calendars as xcals

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    CorporateActionEvent,
    EvidenceReference,
    HoldingContract,
    HoldingSpecification,
    PaymentEvent,
    SimulationReference,
)
from market_predictor.swing.contracts.holding_materialization import SourcePin


class BacktestCorporateActionPolicy(HoldingContract):
    schema_version: Literal["market_predictor.backtest_corporate_actions"]
    scope: Literal["retrospective_research_only"]
    ordinary_dividend_cash: Literal["declared_rate_at_ex_date_close"]
    dividend_reuse: Literal["subsequent_sessions_only"]
    split_units: Literal["recorded_ratio_proportional_normalized_shares"]
    benchmark_distributions: Literal["retain_cash_without_reinvestment"]
    unknown_actions: Literal["valuation_unavailable_without_filtering_selected_lots"]
    production_eligible: Literal[False]
    reviewed_action_dates: SourcePin | None = None
    reviewed_cash_payouts: SourcePin | None = None
    reviewed_parent_units: SourcePin | None = None
    documented_cash_payouts: Literal["issuer_declared_usd_at_ex_date_close"] | None = None

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_none=True))


class BacktestCorporateActionContext(HoldingContract):
    policy: BacktestCorporateActionPolicy
    policy_reference: EvidenceReference

    def assumed_event_ids(self, spec: HoldingSpecification) -> set[str]:
        """Authorize only matched approved-dividend entitlement/payment/cash triples."""
        digest = self.policy.sha256()
        if (self.policy_reference.interpretation_policy_sha256 != digest
                or self.policy_reference.available_at is not None):
            raise ValueError("corporate backtest assumption reference differs from its policy")
        calendar = xcals.get_calendar("XNYS")
        allowed: set[str] = set()
        for event in spec.events:
            if not isinstance(event, CorporateActionEvent) or not any(
                isinstance(ref, SimulationReference) for ref in event.evidence
            ):
                continue
            if (not event.event_id.startswith("simulated-dividend:") or event.treatment != "distribution"
                    or event.fractional_treatment != "proportional" or len(event.legs) != 1
                    or event.effective_at is None or event.effective_at <= spec.initial_entry_timestamp):
                raise ValueError("corporate assumption is restricted to owned approved cash dividends")
            leg = event.legs[0]
            if (leg.kind != "unpaid_proceeds" or leg.currency != "USD" or leg.face_value_per_unit != 1.0
                    or not any(type(ref) is EvidenceReference for ref in event.evidence)):
                raise ValueError("dividend assumption requires separately referenced source terms")
            session = event.effective_at.date().isoformat()
            if not calendar.is_session(session) or event.effective_at != calendar.session_open(session).to_pydatetime():
                raise ValueError("dividend entitlement must occur at its ex-date session open")
            payments = [item for item in spec.events if isinstance(item, PaymentEvent) and item.claim_id == leg.position_id]
            if len(payments) != 1:
                raise ValueError("dividend assumption requires one complete payment")
            payment = payments[0]
            releases = [item for item in spec.events if isinstance(item, CashAvailabilityEvent)
                        and item.payment_event_id == payment.event_id]
            close = calendar.session_close(session).to_pydatetime()
            if (len(releases) != 1 or payment.effective_at != close or releases[0].effective_at != close
                    or payment.currency != "USD" or releases[0].currency != "USD"
                    or payment.amount_per_claim_unit != 1.0 or payment.order is None
                    or releases[0].order is None or releases[0].order <= payment.order):
                raise ValueError("assumed dividend cash must follow payment at the ex-date close")
            for item in (event, payment, releases[0]):
                assumed = [ref for ref in item.evidence if isinstance(ref, SimulationReference)]
                if not assumed or any(ref.interpretation_policy_sha256 != digest for ref in assumed):
                    raise ValueError("dividend events require the exact corporate assumption policy")
                allowed.add(item.event_id)
        return allowed
