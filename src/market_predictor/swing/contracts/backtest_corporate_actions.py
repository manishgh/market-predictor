"""Explicit historical-backtest assumptions; never observed account cash."""
from __future__ import annotations

from typing import Literal

import exchange_calendars as xcals

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    CorporateActionEvent,
    EvidenceReference,
    ExecutionEvent,
    HoldingContract,
    HoldingSpecification,
    KnownMark,
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
    reviewed_record_dispositions: SourcePin | None = None
    reviewed_spin_offs: SourcePin | None = None
    documented_spin_offs: Literal["proportional_daughter_shares_sold_with_parent"] | None = None
    documented_cash_payouts: Literal["issuer_declared_usd_at_ex_date_close"] | None = None

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json", exclude_none=True))


class BacktestCorporateActionContext(HoldingContract):
    policy: BacktestCorporateActionPolicy
    policy_reference: EvidenceReference

    def _spin_ids(self, spec: HoldingSpecification, event: CorporateActionEvent) -> set[str]:
        digest, pin = self.policy.sha256(), self.policy.reviewed_spin_offs
        if (self.policy.documented_spin_offs is None or pin is None or event.treatment != "distribution"
                or event.fractional_treatment != "proportional" or len(event.legs) != 1
                or event.owned_position_id != spec.initial_position_id or event.owned_security_id != spec.security_id
                or event.effective_at is None or event.effective_at <= spec.initial_entry_timestamp):
            raise ValueError("documented spin assumption requires the exact approved parent distribution")
        leg = event.legs[0]
        if (leg.kind != "tradable_shares" or leg.currency != "USD" or leg.security_id == spec.security_id
                or not any(type(ref) is EvidenceReference and ref.reference == pin.path
                    and ref.artifact_sha256 == pin.sha256 and ref.interpretation_policy_sha256 == digest for ref in event.evidence)):
            raise ValueError("documented spin requires exact source-bound daughter shares")
        calendar = xcals.get_calendar("XNYS")
        session = event.effective_at.date().isoformat()
        if not calendar.is_session(session) or event.effective_at != calendar.session_open(session).to_pydatetime():
            raise ValueError("documented spin entitlement requires the ex-session open")
        assumptions = [ref for ref in event.evidence if isinstance(ref, SimulationReference)]
        if not assumptions or any(ref.interpretation_policy_sha256 != digest for ref in assumptions):
            raise ValueError("documented spin requires its exact assumption policy")
        allowed = {event.event_id}
        parents = [sale for sale in spec.events if isinstance(sale, ExecutionEvent)
            and sale.position_id == event.owned_position_id and sale.security_id == spec.security_id]
        daughters = [sale for sale in spec.events if isinstance(sale, ExecutionEvent) and sale.position_id == leg.position_id]
        if not parents:
            if daughters:
                raise ValueError("daughter sale cannot precede a supplied parent exit")
            return allowed
        if len(parents) != 1 or parents[0].fraction_of_owned != 1.0 or parents[0].effective_at is None:
            raise ValueError("documented daughter liquidation requires one complete parent exit")
        parent = parents[0]
        assert parent.effective_at is not None
        if parent.effective_at < event.effective_at:
            if daughters:
                raise ValueError("a pre-distribution parent sale cannot retain daughter shares")
            return allowed
        if len(daughters) != 1:
            raise ValueError("entitled daughter shares must be sold exactly once with parent")
        sale = daughters[0]
        at = parent.effective_at
        day = at.date().isoformat()
        if (sale.security_id != leg.security_id or sale.fraction_of_owned != 1.0 or sale.currency != "USD"
                or sale.reason != parent.reason or sale.effective_at != at or sale.order is None
                or parent.order is None or sale.order <= parent.order
                or at not in (calendar.session_open(day).to_pydatetime(), calendar.session_close(day).to_pydatetime())):
            raise ValueError("documented daughter execution differs from the parent exit clock or owned units")
        prices = [mark for mark in spec.marks if isinstance(mark, KnownMark) and mark.position_id == leg.position_id
            and mark.mark_at == at and mark.currency == "USD" and mark.value_per_unit == sale.price_per_unit]
        if (len(prices) != 1 or not prices[0].evidence or not all(type(ref) is EvidenceReference for ref in prices[0].evidence)
                or not any(ref in sale.evidence and ref.record_locator == f"security_id={leg.security_id};session_date={day}"
                    for ref in prices[0].evidence)):
            raise ValueError("documented daughter execution lacks its corresponding bound raw open or close")
        assumed = [ref for ref in sale.evidence if isinstance(ref, SimulationReference)]
        if not assumed or any(ref.interpretation_policy_sha256 != digest for ref in assumed):
            raise ValueError("documented daughter execution requires its approved clock policy")
        allowed.add(sale.event_id)
        return allowed

    def assumed_event_ids(self, spec: HoldingSpecification) -> set[str]:
        """Authorize only matched approved-dividend entitlement/payment/cash triples."""
        digest = self.policy.sha256()
        if (self.policy_reference.interpretation_policy_sha256 != digest
                or self.policy_reference.available_at is not None):
            raise ValueError("corporate backtest assumption reference differs from its policy")
        calendar = xcals.get_calendar("XNYS")
        allowed: set[str] = set()
        spin_entitlements: set[tuple[str, object, str]] = set()
        for event in spec.events:
            if not isinstance(event, CorporateActionEvent) or not any(
                isinstance(ref, SimulationReference) for ref in event.evidence
            ):
                continue
            if event.event_id.startswith("simulated-spin-off:"):
                allowed.update(self._spin_ids(spec, event))
                key = event.owned_position_id, event.effective_at, event.legs[0].security_id
                if key in spin_entitlements:
                    raise ValueError("documented daughter entitlement duplicated")
                spin_entitlements.add(key)
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
