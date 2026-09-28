from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from market_predictor.collection.outcome_bars import (
    ActionReceipt,
    BarReceipt,
    complete_bar_receipts,
    load_action_receipts,
    load_bar_receipts,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import (
    OPERATOR_VERIFIED,
    MaturationAttempt,
    MaturedOutcome,
    PredictionMaturationIntent,
    swing_horizon_sessions,
)
from market_predictor.governance.outcomes.evidence import (
    EvidenceTerms,
    cessation_evidence,
    path_evidence,
)
from market_predictor.governance.outcomes.maturation import (
    AttemptContext,
    maturation_attempt,
    mature_prediction,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.sessions import horizon_last_close


def mature_pending_intents(
    repository: OutcomeRepository,
    *,
    receipts_root: Path,
    observed_as_of: datetime,
    terms: EvidenceTerms,
    memberships: pd.DataFrame | None = None,
) -> dict[str, int]:
    """Mature the pending index's canonical intents whose horizon has closed; history is never scanned.

    The summary counts index entries by what happened to them. An entry is left in place while
    its horizon is open or while its registration has not yet written the semantic record (the
    registration rerun completes it); an entry whose semantic record names another intent can
    never be canonical and is dropped. An `unresolvable` intent stays indexed and is re-evaluated
    every run, so newly collected evidence can still mature it.
    """
    summary = {
        "index_entries": 0,
        "horizon_open": 0,
        "matured": 0,
        "pending": 0,
        "blocked": 0,
        "unresolvable": 0,
        "registration_incomplete": 0,
        "not_canonical_dropped": 0,
        "already_matured": 0,
        "observed_before_latest_attempt": 0,
    }
    receipts: dict[date, tuple[tuple[BarReceipt, ...], tuple[ActionReceipt, ...]]] = {}
    # Evidence is what had been collected by the observation time; a backdated run never reads ahead.
    observed = observed_as_of.astimezone(UTC)
    for maturation_key, session in repository.pending():
        summary["index_entries"] += 1
        if repository.has_outcome(maturation_key, session):
            repository.drop_pending(maturation_key, session)
            summary["already_matured"] += 1
            continue
        intent = repository.load_intent(maturation_key, session)
        canonical_key = repository.semantic_canonical_key(intent.semantic_prediction_id, session)
        if canonical_key is None:
            summary["registration_incomplete"] += 1
            continue
        if canonical_key != maturation_key:
            repository.drop_pending(maturation_key, session)
            summary["not_canonical_dropped"] += 1
            continue
        latest = repository.latest_attempt(maturation_key, session)
        # An attempt log runs forward in time, so a run observing an earlier time adds nothing.
        if latest is not None and observed < latest.observed_as_of_utc:
            summary["observed_before_latest_attempt"] += 1
            continue
        try:
            # An outcome is taken only once the horizon has closed, so a target or stop reached
            # earlier still records the fixed-horizon return over the whole path.
            if horizon_last_close(session, swing_horizon_sessions(intent.horizon), through=observed_as_of) is None:
                summary["horizon_open"] += 1
                continue
            if session not in receipts:
                receipts[session] = (
                    tuple(r for r in load_bar_receipts(receipts_root, [session]) if r.finished_at_utc <= observed),
                    tuple(r for r in load_action_receipts(receipts_root, [session]) if r.finished_at_utc <= observed),
                )
            bar_receipts, action_receipts = receipts[session]
            result, evidence = resolve_intent(
                intent,
                bar_receipts=bar_receipts,
                action_receipts=action_receipts,
                memberships=memberships,
                observed_as_of=observed_as_of,
                terms=terms,
            )
        except (DataReadinessError, KeyError, TypeError, ValueError) as exc:
            attempt = maturation_attempt(
                intent,
                observed_as_of=observed_as_of,
                status="blocked",
                reasons=(f"invalid_maturation_input:{type(exc).__name__}",),
                context=_context(terms),
            )
            repository.record_attempt(attempt, decision_session=session)
            summary["blocked"] += 1
            continue
        if isinstance(result, MaturedOutcome):
            repository.record_outcome(intent, result, evidence_rows=evidence)
            summary["matured"] += 1
        elif latest is not None and latest.reasons == (OPERATOR_VERIFIED,):
            # An operator's resolution stands until the evidence matures the outcome.
            summary["unresolvable"] += 1
        else:
            repository.record_attempt(result, decision_session=session)
            summary[result.status] += 1
    return summary


def record_operator_resolution(
    repository: OutcomeRepository,
    *,
    maturation_key: str,
    decision_session: date,
    operator_id: str,
    reference: str,
    observed_as_of: datetime,
    terms: EvidenceTerms,
) -> MaturationAttempt:
    """Record an operator's verified finding that a pending outcome can never mature.

    For a stock that stopped trading without provider evidence of why; `reference` names the
    evidence the operator checked. Only a later maturation supersedes it.
    """
    intent = repository.load_intent(maturation_key, decision_session)
    if repository.has_outcome(maturation_key, decision_session):
        raise DataReadinessError("an outcome that matured cannot be resolved by an operator")
    if horizon_last_close(decision_session, swing_horizon_sessions(intent.horizon), through=observed_as_of) is None:
        raise DataReadinessError("an operator resolves an outcome only after its horizon has closed")
    attempt = maturation_attempt(
        intent,
        observed_as_of=observed_as_of,
        status="unresolvable",
        reasons=(OPERATOR_VERIFIED,),
        context=_context(terms),
        operator_id=operator_id,
        operator_reference=reference,
    )
    return repository.record_attempt(attempt, decision_session=decision_session)


def resolve_intent(
    intent: PredictionMaturationIntent,
    *,
    bar_receipts: Sequence[BarReceipt],
    action_receipts: Sequence[ActionReceipt],
    memberships: pd.DataFrame | None,
    observed_as_of: datetime,
    terms: EvidenceTerms,
) -> tuple[MaturedOutcome | MaturationAttempt, list[dict[str, object]]]:
    """Mature an intent on its receipts, or explain why it cannot mature yet or ever.

    When the barrier was not reached before a proven gap, the path can never be completed:
    - an interior gap (usable bars after it) is `unresolvable` once a one-minute receipt shows
      the stock did not trade that session, and a collection defect if it did;
    - a tail gap is `unresolvable` with cessation evidence, and otherwise stays pending until
      it turns overdue for operator action.
    """
    evidence = path_evidence(intent, bar_receipts, terms=terms)
    result, rows = mature_prediction(
        intent, evidence.bars, observed_as_of=observed_as_of, proven_stock_gaps=evidence.proven_stock_gaps
    )
    if isinstance(result, MaturedOutcome):
        return result, rows
    status = "pending"
    reasons = result.reasons
    receipt_ids = list(evidence.receipt_ids)
    gap = evidence.first_gap
    if gap is not None and gap in evidence.proven_stock_gaps and f"{intent.ticker}:{gap}" in result.missing_intervals:
        if evidence.trades_after(gap):
            minute = complete_bar_receipts(
                bar_receipts,
                decision_session=intent.decision_session_et,
                symbol=intent.ticker,
                first_session=gap,
                last_session=gap,
                timeframe="1Min",
            )
            # A minute receipt proves the halt only once the session had settled.
            minute = tuple(receipt for receipt in minute if terms.settles_session(gap, receipt))
            receipt_ids.extend(receipt.receipt_id for receipt in minute)
            if not minute:
                reasons = ("interior_gap_needs_minute_bars",)
            elif any(receipt.sessions_with_rows(intent.ticker) for receipt in minute):
                reasons = ("interior_gap_contradicted",)
            else:
                status, reasons = "unresolvable", ("interior_gap",)
        else:
            cessation = cessation_evidence(
                intent, action_receipts=action_receipts, memberships=memberships, terms=terms
            )
            if cessation is None:
                reasons = ("stock_gap_without_cessation",)
            else:
                status, reasons = "unresolvable", (cessation[0],)
                receipt_ids.extend(cessation[1])
    attempt = maturation_attempt(
        intent,
        observed_as_of=observed_as_of,
        status=status,
        reasons=reasons,
        missing_intervals=result.missing_intervals,
        context=_context(terms, receipt_ids),
    )
    return attempt, []


def _context(terms: EvidenceTerms, receipt_ids: Sequence[str] = ()) -> AttemptContext:
    return AttemptContext(
        grace_days=terms.grace_days,
        settlement_days=terms.settlement_days,
        drift_policy_sha256=terms.drift_policy_sha256,
        receipt_ids=tuple(receipt_ids),
    )
