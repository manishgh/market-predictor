"""Investment horizon selection over canonical lot snapshots, without sale simulation."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import exchange_calendars as xcals
import pandas as pd

from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.contracts import (
    HOLDING_ROLES,
    HoldingRole,
    InvestmentHoldingBinding,
    InvestmentHoldingValue,
    InvestmentTargetRequest,
    InvestmentTargetResult,
)
from market_predictor.swing.contracts.holding_accounting import (
    EvidenceReference,
    ExecutionEvent,
    HoldingSpecification,
    KnownMark,
    SimulationReference,
)
from market_predictor.swing.evaluation.holding_accounting import replay_holding


def _verify_sources(spec: HoldingSpecification, binding: InvestmentHoldingBinding) -> None:
    references: list[EvidenceReference] = list(spec.entry_evidence)
    references.extend(reference for event in spec.events for reference in event.evidence)
    references.extend(reference for mark in spec.marks if isinstance(mark, KnownMark) for reference in mark.evidence)
    declared = {source.reference: (source.artifact_sha256, source.interpretation_policy_sha256) for source in binding.sources}
    observed: set[str] = set()
    for reference in references:
        if isinstance(reference, SimulationReference):
            raise ValueError("investment marks and corporate events cannot use ordinary-sale assumptions")
        if declared.get(reference.reference) != (reference.artifact_sha256, reference.interpretation_policy_sha256):
            raise ValueError("holding evidence differs from its bound source identity")
        observed.add(reference.reference)
    if observed != set(declared):
        raise ValueError("holding source inventory differs from consumed evidence")


def project_investment_targets(
    request: InvestmentTargetRequest, holdings: Mapping[HoldingRole, HoldingSpecification],
) -> InvestmentTargetResult:
    """Validate exact request bindings and select Nth marked value, never liquidating.

    Source identities are checked, not source authority or coverage. Those independent
    publication gates remain necessary before these research targets enter training.
    """
    request = InvestmentTargetRequest.model_validate_json(request.model_dump_json())
    if set(holdings) != set(HOLDING_ROLES):
        raise ValueError("target needs exactly stock, SPY, QQQ and sector holdings")
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(request.decision_session)
    if not calendar.is_session(session):
        raise ValueError("decision must be an XNYS session")
    dates = calendar.sessions_window(session, request.policy.horizon_sessions + 1)[1:]
    entry: datetime = calendar.session_open(dates[0]).to_pydatetime()
    ends = tuple(calendar.session_close(day).to_pydatetime() for day in dates)
    if not calendar.session_close(session).to_pydatetime() <= request.decision_time_utc < entry:
        raise ValueError("decision time must follow its session close and precede next open")
    values: list[InvestmentHoldingValue] = []
    for binding in request.holdings:
        spec = HoldingSpecification.model_validate_json(holdings[binding.role].model_dump_json())
        if json_sha256(spec.model_dump(mode="json")) != binding.specification_sha256:
            raise ValueError("holding specification hash differs from request")
        if (spec.research_contract_sha256 != request.policy.sha256() or spec.decision_id != request.decision_id
                or spec.security_id != binding.security_id or spec.sector != request.sector
                or spec.policy != "fixed_horizon" or spec.initial_entry_timestamp != entry
                or spec.session_end_timestamps != ends or spec.cost_prepaid_fraction != 0.002):
            raise ValueError("holding policy, decision, security or exact horizon interval differs")
        if any(isinstance(event, ExecutionEvent) for event in spec.events):
            raise ValueError("investment marked-value targets forbid ordinary execution events")
        _verify_sources(spec, binding)
        outcome = replay_holding(spec)
        snapshots = outcome.snapshots
        final = snapshots[request.policy.horizon_sessions - 1]
        reasons = {f"{snapshot.session_end_timestamp.date()}:{gap.code}:{gap.reference_id}"
            for snapshot in snapshots for gap in snapshot.gaps}
        if any(snapshot.total_value is None for snapshot in snapshots):
            reasons.add("holding_valuation_unavailable")
        if any(snapshot.label_available_at is None for snapshot in snapshots):
            reasons.add("source_availability_unknown")
        available = None if reasons else max(
            snapshot.label_available_at for snapshot in snapshots if snapshot.label_available_at is not None)
        values.append(InvestmentHoldingValue(role=binding.role, security_id=spec.security_id,
            specification_sha256=binding.specification_sha256, gross_return=final.gross_return,
            net_return=final.net_return, label_available_at_utc=available, missing_reasons=tuple(sorted(reasons))))
    missing = tuple(f"{value.role}:{reason}" for value in values for reason in value.missing_reasons)
    stock = values[0]
    excess = [None if stock.net_return is None or benchmark.gross_return is None
        else stock.net_return - benchmark.gross_return for benchmark in values[1:]]
    available = None if missing else max(ends[-1], *(
        value.label_available_at_utc for value in values if value.label_available_at_utc is not None))
    return InvestmentTargetResult(request_sha256=request.sha256(), policy_sha256=request.policy.sha256(),
        decision_id=request.decision_id, horizon_sessions=request.policy.horizon_sessions,
        entry_time_utc=entry.astimezone(UTC), horizon_end_utc=ends[-1].astimezone(UTC), holdings=tuple(values),
        net_excess_vs_spy=excess[0], net_excess_vs_qqq=excess[1], net_excess_vs_sector=excess[2],
        label_available_at_utc=available, label_eligible=not missing,
        status="unavailable" if missing else "available", missing_reasons=missing)
