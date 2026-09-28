from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from market_predictor.collection.outcome_bars import (
    ActionReceipt,
    BarReceipt,
    complete_bar_receipts,
    daily_path_bars,
    load_action_receipts,
    load_bar_receipts,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import (
    OPERATOR_VERIFIED,
    MaturationAttempt,
    MaturedOutcome,
    PredictionMaturationIntent,
    SensitivityFill,
    swing_horizon_sessions,
)
from market_predictor.governance.outcomes.evidence import (
    Cessation,
    EvidenceTerms,
    cessation_evidence,
    path_evidence,
    usable_rows,
)
from market_predictor.governance.outcomes.maturation import (
    AttemptContext,
    maturation_attempt,
    mature_prediction,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.sensitivity import acquirer_terms, sensitivity_fills
from market_predictor.governance.outcomes.sessions import horizon_last_close

# The latest reasons of an intent whose stock gap is proven but not yet explained.
_PROVEN_GAP_REASONS = (
    ("stock_gap_without_cessation",),
    ("interior_gap_needs_minute_bars",),
    ("interior_gap_contradicted",),
)


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
            if latest is not None and latest.reasons == (OPERATOR_VERIFIED,):
                # Unreadable evidence never undoes an operator's resolution.
                summary["unresolvable"] += 1
                continue
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
        elif latest is not None and latest.reasons == (OPERATOR_VERIFIED,) and result.status != "unresolvable":
            # An operator's resolution stands until the evidence matures the outcome or names
            # a more specific reason.
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
    receipts_root: Path,
    observed_as_of: datetime,
    terms: EvidenceTerms,
) -> MaturationAttempt:
    """Record an operator's verified finding that a pending outcome can never mature.

    Only for a proven stock gap without provider evidence of why the stock stopped trading;
    `reference` names the evidence the operator checked. A later maturation, or provider
    evidence naming a more specific reason, supersedes it. Its stress fill is a delisting.
    """
    intent = repository.load_intent(maturation_key, decision_session)
    if repository.has_outcome(maturation_key, decision_session):
        raise DataReadinessError("an outcome that matured cannot be resolved by an operator")
    latest = repository.latest_attempt(maturation_key, decision_session)
    stuck = latest is not None and (latest.reasons in _PROVEN_GAP_REASONS or latest.status == "blocked")
    if not stuck or observed_as_of < terms.deadline(intent):
        # Collection lag or a benchmark gap would otherwise leave the metrics by an operator's choice.
        raise DataReadinessError("an operator resolves only an overdue outcome with a proven stock gap or blocked evidence")
    observed = observed_as_of.astimezone(UTC)
    bar_receipts = tuple(r for r in load_bar_receipts(receipts_root, [decision_session]) if r.finished_at_utc <= observed)
    evidence = path_evidence(intent, bar_receipts, terms=terms)
    attempt = maturation_attempt(
        intent,
        observed_as_of=observed_as_of,
        status="unresolvable",
        reasons=(OPERATOR_VERIFIED,),
        context=_context(terms, evidence.receipt_ids),
        operator_id=operator_id,
        operator_reference=reference,
        sensitivity=sensitivity_fills(
            intent, evidence=evidence, reason=OPERATOR_VERIFIED, cessation_record=None,
            acquirer_close=None, acquirer_collected=False,
        ),
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
    cessation: Cessation | None = None
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
                status, reasons = "unresolvable", (cessation.reason,)
                receipt_ids.extend(cessation.receipt_ids)
    sensitivity: tuple[SensitivityFill, ...] = ()
    if status == "unresolvable":
        record = cessation.record if cessation is not None else None
        acquirer = acquirer_terms(reasons[0], record)
        acquirer_close, collected, acquirer_receipts = _acquirer_close(intent, bar_receipts, acquirer)
        receipt_ids.extend(acquirer_receipts)
        sensitivity = sensitivity_fills(
            intent, evidence=evidence, reason=reasons[0], cessation_record=record,
            acquirer_close=acquirer_close, acquirer_collected=collected,
        )
    attempt = maturation_attempt(
        intent,
        observed_as_of=observed_as_of,
        status=status,
        reasons=reasons,
        missing_intervals=result.missing_intervals,
        context=_context(terms, receipt_ids),
        sensitivity=sensitivity,
    )
    return attempt, []


def _acquirer_close(
    intent: PredictionMaturationIntent,
    bar_receipts: Sequence[BarReceipt],
    acquirer: tuple[str, date] | None,
) -> tuple[float | None, bool, tuple[str, ...]]:
    """The acquirer's usable close on the effective session, whether its bars were collected, and the receipt."""
    if acquirer is None:
        return None, False, ()
    symbol, effective = acquirer
    decision = intent.decision_session_et
    covering = complete_bar_receipts(
        bar_receipts, decision_session=decision, symbol=symbol, first_session=decision, last_session=effective
    )
    for receipt in reversed(covering):
        try:
            rows = usable_rows(daily_path_bars(receipt, symbol), symbol)
        except DataReadinessError:
            continue
        if effective in rows:
            return float(rows[effective]["close"]), True, (receipt.receipt_id,)
    return None, bool(covering), tuple(receipt.receipt_id for receipt in covering[-1:])


def _context(terms: EvidenceTerms, receipt_ids: Sequence[str] = ()) -> AttemptContext:
    return AttemptContext(
        grace_days=terms.grace_days,
        settlement_days=terms.settlement_days,
        drift_policy_sha256=terms.drift_policy_sha256,
        receipt_ids=tuple(receipt_ids),
    )
