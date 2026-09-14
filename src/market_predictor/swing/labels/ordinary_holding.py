"""Pure fixed-horizon adapter for one independently admitted ordinary raw path."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Literal

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_accounting import (
    AccountingGap,
    EvidenceReference,
    ExecutionEvent,
    HoldingContract,
    HoldingSpecification,
    HoldingTargets,
    KnownMark,
    LotOutcome,
)
from market_predictor.swing.contracts.trade_simulation import SimulatedHolding, TradeSimulationContext
from market_predictor.swing.evaluation.holding_accounting import project_holding_targets
from market_predictor.swing.evaluation.trade_simulation import simulate_ordinary_sales
from market_predictor.swing.labels.holding_paths import validate_outcome_observations


class OrdinaryHoldingResult(HoldingContract):
    specification: HoldingSpecification | None = None
    outcome: LotOutcome | None = None
    targets: HoldingTargets | None = None
    simulation: SimulatedHolding | None = None
    missing_reasons: tuple[AccountingGap, ...] = ()
    label_eligible: Literal[False] = False


def build_ordinary_holding(
    *, decision: Mapping[str, Any], sessions: tuple[date, ...], observations: pd.DataFrame,
    evidence: Mapping[date, EvidenceReference], research_contract_sha256: str,
    cost_prepaid_fraction: float, simulation: TradeSimulationContext,
) -> OrdinaryHoldingResult:
    """Compile next-open/tenth-close research fills and project canonical replay.

    The caller owns independent ownership/action/source admission and exact joins.
    Only use this path when no corporate component is owed. ``decision`` carries
    decision_id, security_id, ticker, sector, session_date_et and decision_time_utc;
    benchmark decisions use their canonical ETF security identities unchanged.
    Observations/evidence are the bounded output of read_bound_observations, not
    an unfiltered history. Metadata violations raise; absent/unusable path rows
    return null components with reasons. No managed policy or ATR is inferred.
    """
    for key in ("decision_id", "security_id", "ticker", "sector"):
        value = decision.get(key)
        if not isinstance(value, str) or not value or value.strip() != value:
            raise DataReadinessError(f"ordinary holding requires canonical decision {key}")
    day = decision.get("session_date_et")
    if type(day) is not date or not date(2019, 7, 9) <= day <= date(2024, 5, 13):
        raise DataReadinessError("ordinary holding decision outside mature initial-fit sessions")
    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(day.isoformat()):
        raise DataReadinessError("ordinary holding decision is not an XNYS session")
    cutoff = swing_prediction_cutoffs(pd.Series([day])).iloc[0]
    if decision.get("decision_time_utc") != cutoff:
        raise DataReadinessError("ordinary holding decision cutoff differs from canonical swing cutoff")
    expected = tuple(session.date() for session in calendar.sessions_window(pd.Timestamp(day), 11)[1:])
    if (len(sessions) != 10 or any(type(session) is not date for session in sessions)
            or sessions != expected or sessions[-1] > date(2024, 5, 28)):
        raise DataReadinessError("ordinary holding requires exact ten next-open/tenth-close initial-fit XNYS sessions")
    if cost_prepaid_fraction != 0.002:
        raise DataReadinessError("ordinary holding requires the approved explicit 20 bps prepaid cost")
    if (not isinstance(research_contract_sha256, str) or len(research_contract_sha256) != 64
            or any(char not in "0123456789abcdef" for char in research_contract_sha256)):
        raise DataReadinessError("ordinary holding research contract hash is invalid")
    # Reject wider/held-out frames using only their index before accessing numerics.
    if not observations.empty and (observations.index.name != "session_date_et" or not observations.index.is_unique
            or any(type(value) is not date for value in observations.index)
            or not set(observations.index).issubset(sessions)):
        raise DataReadinessError("ordinary holding observation index is duplicate, invalid or outside exact sessions")
    if not observations.columns.is_unique:
        raise DataReadinessError("ordinary holding observation columns must be unique")
    metadata = {"security_id": decision["security_id"], "ticker": decision["ticker"],
        "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw"}
    if not observations.empty:
        for key, value in metadata.items():
            if key not in observations or not observations[key].eq(value).fillna(False).all():
                raise DataReadinessError(f"ordinary holding observation {key} differs from required identity/raw SIP source")
        if "outcome_observation_valid" not in observations or "ingested_at_utc" not in observations:
            raise DataReadinessError("ordinary holding requires independently validated raw observations")
    if any(type(value) is not date for value in evidence) or not set(evidence).issubset(observations.index):
        raise DataReadinessError("ordinary holding evidence lies outside supplied observations")
    references: dict[date, EvidenceReference] = {}
    for session, reference in evidence.items():
        if type(reference) is not EvidenceReference:
            raise DataReadinessError("ordinary holding raw observation evidence cannot be a simulation assumption")
        reference = EvidenceReference.model_validate_json(reference.model_dump_json())
        if (reference.record_locator != f"security_id={decision['security_id']};session_date={session}"
                or reference.available_at is not None
                or reference.retrieved_at != pd.Timestamp(observations.loc[session, "ingested_at_utc"])):
            raise DataReadinessError("ordinary holding evidence identity/session/retrieval clock differs from bound observation")
        references[session] = reference
    if len({(ref.reference, ref.artifact_sha256, ref.interpretation_policy_sha256) for ref in references.values()}) > 1:
        raise DataReadinessError("ordinary holding requires one bound raw source path and interpretation")
    validated = observations
    admitted_valid = pd.Series(False, index=observations.index)
    if not observations.empty:
        admitted_valid = observations["outcome_observation_valid"].eq(True).fillna(False)
        validated = validate_outcome_observations(observations.reset_index()).set_index("session_date_et")
    gaps = []
    for index, session in enumerate(sessions):
        kind = "entry" if index == 0 else "path"
        if session not in observations.index:
            code = f"{kind}_observation_missing"
        elif (not bool(admitted_valid.loc[session])
                or not bool(validated.loc[session, "outcome_observation_valid"])):
            code = f"{kind}_observation_invalid"
        elif session not in references:
            code = f"{kind}_evidence_missing"
        else:
            continue
        gaps.append(AccountingGap(code=code, reference_id=f"{decision['security_id']}:{session}"))
    if gaps:
        return OrdinaryHoldingResult(missing_reasons=tuple(gaps))
    ends = tuple(calendar.session_close(session.isoformat()).to_pydatetime() for session in sessions)
    entry_at = calendar.session_open(sessions[0].isoformat()).to_pydatetime()
    position = f"{decision['decision_id']}:{decision['security_id']}:ordinary-shares"
    spec = HoldingSpecification(
        research_contract_sha256=research_contract_sha256, decision_id=decision["decision_id"],
        security_id=decision["security_id"], sector=decision["sector"], initial_position_id=position,
        initial_entry_price=float(validated.loc[sessions[0], "open"]), initial_entry_timestamp=entry_at,
        price_basis="raw_with_no_adjustment", currency="USD", entry_evidence=(references[sessions[0]],),
        session_end_timestamps=ends, cost_prepaid_fraction=cost_prepaid_fraction, policy="fixed_horizon",
        marks=tuple(KnownMark(position_id=position, mark_at=end, value_per_unit=float(validated.loc[session, "close"]),
            currency="USD", evidence=(references[session],)) for session, end in zip(sessions, ends, strict=True)),
        events=(ExecutionEvent(event_id=position + ":fixed-exit", effective_at=ends[-1], order=0,
            evidence=(references[sessions[-1]],), position_id=position, security_id=decision["security_id"],
            fraction_of_owned=1.0, price_per_unit=float(validated.loc[sessions[-1], "close"]), currency="USD",
            proceeds_id=position + ":sale-proceeds", reason="fixed_horizon"),),
    )
    simulated = simulate_ordinary_sales(spec, simulation)
    outcome = simulated.outcome
    replay_gaps = tuple(gap for snapshot in outcome.snapshots for gap in snapshot.gaps)
    return OrdinaryHoldingResult(specification=simulated.specification, outcome=outcome,
        targets=project_holding_targets(outcome, None) if not replay_gaps else None,
        simulation=simulated, missing_reasons=replay_gaps)
