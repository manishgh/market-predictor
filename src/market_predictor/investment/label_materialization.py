"""Build 63/252-session future-return targets using the shared holding kernel."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import exchange_calendars as xcals
import pandas as pd

from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.contracts import (
    HOLDING_ROLES,
    SPY_PAIR_ROLES,
    HoldingRole,
    InvestmentHoldingBinding,
    InvestmentSourceIdentity,
    InvestmentSpyPairRequest,
    InvestmentTargetPolicy,
    InvestmentTargetRequest,
)
from market_predictor.investment.label_sources import InvestmentSourceReader
from market_predictor.investment.targets import project_investment_spy_pair, project_investment_targets
from market_predictor.swing.contracts.holding_accounting import (
    CashAvailabilityEvent,
    CorporateActionEvent,
    EvidenceReference,
    HoldingSpecification,
    KnownMark,
    Mark,
    PaymentEvent,
    SimulationReference,
)


@dataclass(frozen=True)
class InvestmentSourceValidation:
    """Membership/action source-check results and clocks; never training approval."""
    evidence: tuple[EvidenceReference, ...]
    missing_reasons: tuple[str, ...] = ()
    events: tuple[CorporateActionEvent | PaymentEvent | CashAvailabilityEvent, ...] = ()
    non_share_marks: tuple[Mark, ...] = ()
    share_position_ids: tuple[str, ...] = ()
    valuation_basis: str = "observed_raw_prices_and_action_absence"


ActionSourceValidation = Callable[[str, str, tuple[date, ...], str], InvestmentSourceValidation]


def _identities(evidence: tuple[EvidenceReference, ...]) -> tuple[InvestmentSourceIdentity, ...]:
    sources: dict[str, InvestmentSourceIdentity] = {}
    for item in evidence:
        if isinstance(item, SimulationReference):
            raise ValueError("investment source validation cannot use simulation provenance")
        if item.available_at is not None and item.available_at > item.retrieved_at:
            raise ValueError("investment source availability cannot follow capture")
        source = InvestmentSourceIdentity(reference=item.reference, artifact_sha256=item.artifact_sha256,
            interpretation_policy_sha256=item.interpretation_policy_sha256)
        if item.reference in sources and sources[item.reference] != source:
            raise ValueError("investment source has conflicting identities")
        sources[item.reference] = source
    return tuple(sources[name] for name in sorted(sources))


def _compile_investment_holdings(*, decision: Mapping[str, Any], horizon: Literal[63, 252],
    reader: InvestmentSourceReader, membership: InvestmentSourceValidation,
    action_source_validation: ActionSourceValidation, roles: tuple[HoldingRole, ...]) -> dict[str, Any]:
    """Build only the requested holdings, with exact session and corporate-action evidence."""
    calendar = xcals.get_calendar("XNYS")
    raw_day = decision.get("decision_session", decision.get("session_date_et"))
    day = date.fromisoformat(raw_day) if isinstance(raw_day, str) else raw_day
    policy = InvestmentTargetPolicy(horizon_sessions=horizon)
    sessions = tuple(stamp.date() for stamp in calendar.sessions_window(pd.Timestamp(day), horizon + 1)[1:])
    ends = tuple(calendar.session_close(stamp.isoformat()).to_pydatetime() for stamp in sessions)
    entry = calendar.session_open(sessions[0].isoformat()).to_pydatetime()
    reasons = list(membership.missing_reasons)
    if not membership.evidence:
        reasons.append("membership_source_evidence_missing")
    if any(isinstance(item, SimulationReference) for item in membership.evidence):
        raise ValueError("investment membership cannot use simulation provenance")
    if any(item.available_at is None for item in membership.evidence):
        reasons.append("membership_source_availability_unknown")
    membership_sources = _identities(membership.evidence)
    sector_symbol = decision.get("sector_benchmark", decision.get("primary_benchmark"))
    components: dict[HoldingRole, tuple[str, str]] = {"stock": (str(decision["security_id"]), str(decision["ticker"])),
        "spy": ("benchmark:SPY", "SPY"), "qqq": ("benchmark:QQQ", "QQQ"),
        "sector": (f"benchmark:{sector_symbol}", str(sector_symbol))}
    specs: dict[HoldingRole, HoldingSpecification] = {}
    bindings = []
    component_reports = []
    for role in roles:
        identity, symbol = components[role]
        observations = reader.read(identity, symbol, sessions, benchmark=role != "stock")
        gaps = list(observations.missing_reasons)
        position = f"{decision['decision_id']}:{role}:shares"
        source_validation = action_source_validation(identity, observations.provider_symbol or symbol, sessions, position)
        gaps.extend(source_validation.missing_reasons)
        if not source_validation.evidence:
            gaps.append("corporate_action_coverage_evidence_missing")
        if any(item.available_at is None for item in source_validation.evidence):
            gaps.append("corporate_action_source_availability_unknown")
        _identities(source_validation.evidence)
        if source_validation.share_position_ids and len(source_validation.share_position_ids) != len(sessions):
            raise ValueError("investment share-position marks must cover exact horizon")
        action_refs = tuple(item for event in source_validation.events for item in event.evidence)
        mark_refs = tuple(item for mark in source_validation.non_share_marks if isinstance(mark, KnownMark) for item in mark.evidence)
        if any(item.available_at is None for item in (*action_refs, *mark_refs)):
            gaps.append("corporate_action_economics_availability_unknown")
        all_evidence = (*membership.evidence, *source_validation.evidence, *observations.evidence.values(), *action_refs, *mark_refs)
        sources = _identities(all_evidence)
        component_reports.append({"role": role, "security_id": identity, "symbol": symbol,
            "missing_reasons": sorted(set(gaps)), "valuation_basis": source_validation.valuation_basis,
            "sources": [source.model_dump(mode="json") for source in sources]})
        reasons.extend(f"{role}:{gap}" for gap in gaps)
        if gaps or not membership_sources:
            continue
        context = (*membership.evidence, *source_validation.evidence)
        spec = HoldingSpecification(research_contract_sha256=policy.sha256(), decision_id=str(decision["decision_id"]),
            security_id=identity, sector=str(decision["sector"]), initial_position_id=position,
            initial_entry_price=float(observations.frame.loc[sessions[0], "open"]), initial_entry_timestamp=entry,
            price_basis="raw_with_no_adjustment", currency="USD",
            entry_evidence=(*context, observations.evidence[sessions[0]]), session_end_timestamps=ends,
            cost_prepaid_fraction=0.002, policy="fixed_horizon", events=source_validation.events,
            marks=(*tuple(KnownMark(position_id=share_position,
                mark_at=end, value_per_unit=float(observations.frame.loc[stamp, "close"]), currency="USD",
                evidence=(*context, observations.evidence[stamp])) for stamp, end, share_position in zip(sessions, ends,
                    source_validation.share_position_ids or (position,) * len(sessions), strict=True)), *source_validation.non_share_marks))
        specs[role] = spec
        bindings.append(InvestmentHoldingBinding(role=role, security_id=identity, symbol=symbol,
            specification_sha256=json_sha256(spec.model_dump(mode="json")), sources=sources))
    result: dict[str, Any] | None = None
    request: dict[str, Any] | None = None
    if not reasons:
        arguments = {"policy": policy, "decision_id": str(decision["decision_id"]), "decision_session": day,
            "decision_time_utc": pd.Timestamp(decision["decision_time_utc"]).to_pydatetime(), "sector": str(decision["sector"]),
            "sector_membership_source": membership_sources[0], "holdings": tuple(bindings)}
        if roles == HOLDING_ROLES:
            target_request = InvestmentTargetRequest.model_validate(arguments)
            request = target_request.model_dump(mode="json")
            result = project_investment_targets(target_request, specs).model_dump(mode="json")
        else:
            pair_request = InvestmentSpyPairRequest.model_validate(arguments)
            request = pair_request.model_dump(mode="json")
            result = project_investment_spy_pair(pair_request, specs).model_dump(mode="json")
        reasons.extend(result["missing_reasons"])
    return {"decision_id": str(decision["decision_id"]), "security_id": str(decision["security_id"]),
        "decision_session": day.isoformat(), "horizon_sessions": horizon, "policy_sha256": policy.sha256(),
        "entry_time_utc": entry.isoformat(), "economic_maturity_utc": ends[-1].isoformat(),
        "status": "source_validated" if not reasons else "unresolved",
        "missing_reasons": sorted(set(reasons)), "components": component_reports,
        "request": request, "target": result,
        "specifications": {role: spec.model_dump(mode="json") for role, spec in specs.items()},
        "training_ready": False, "promotion_eligible": False, "production_eligible": False}


def compile_investment_label(*, decision: Mapping[str, Any], horizon: Literal[63, 252],
    reader: InvestmentSourceReader, membership: InvestmentSourceValidation,
    action_source_validation: ActionSourceValidation) -> dict[str, Any]:
    """Compile the unchanged four-role investment training label."""
    row = _compile_investment_holdings(decision=decision, horizon=horizon, reader=reader, membership=membership,
        action_source_validation=action_source_validation, roles=HOLDING_ROLES)
    row["label_available_at_utc"] = None if row["target"] is None else row["target"]["label_available_at_utc"]
    row["label_eligible"] = not row["missing_reasons"]
    return row


def compile_investment_spy_pair(*, decision: Mapping[str, Any], horizon: Literal[63, 252],
    reader: InvestmentSourceReader, membership: InvestmentSourceValidation,
    action_source_validation: ActionSourceValidation) -> dict[str, Any]:
    """Compile stock and SPY without requiring auxiliary benchmark evidence."""
    row = _compile_investment_holdings(decision=decision, horizon=horizon, reader=reader, membership=membership,
        action_source_validation=action_source_validation, roles=SPY_PAIR_ROLES)
    pair = row.pop("target")
    row["outcome_available_at_utc"] = None if pair is None else pair["outcome_available_at_utc"]
    row["pair"] = pair
    return row
