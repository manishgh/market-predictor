"""Which outcome evidence to collect, from the pending index, the latest attempts and the receipts held.

Maturation leaves each intent's latest attempt, so the plan follows its conclusion:
- bars are requested every night until a settled receipt exists, then weekly;
- a gap inside the path needs a one-minute request for that session;
- a gap at the end of the path without cessation evidence needs the ticker's corporate
  actions, and those of every later name, nightly until the deadline, then weekly;
- an intent 90 days past its deadline is frozen: nothing is collected for it again unless an
  operator names it (`only`), which also sets its cadence aside.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from market_predictor.collection.outcome_bars import (
    ActionReceipt,
    BarReceipt,
    CorporateActionUnit,
    OutcomeBarUnit,
    complete_bar_receipts,
    load_action_receipts,
    load_bar_receipts,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import (
    OPERATOR_VERIFIED,
    PredictionMaturationIntent,
    swing_horizon_sessions,
)
from market_predictor.governance.outcomes.evidence import (
    EvidenceTerms,
    cessation_evidence,
    horizon_close,
    path_evidence,
    symbols_in_use,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.sensitivity import acquirer_terms
from market_predictor.governance.outcomes.sessions import session_after

RECOLLECT_AFTER = timedelta(days=7)
NIGHTLY = timedelta(hours=20)
FREEZE_AFTER = timedelta(days=90)
_PAGE_SYMBOLS = 50


@dataclass(frozen=True)
class CollectionPlan:
    bar_units: tuple[OutcomeBarUnit, ...]
    action_units: tuple[CorporateActionUnit, ...]


def plan_outcome_collection(
    repository: OutcomeRepository,
    *,
    receipts_root: Path,
    now: datetime,
    terms: EvidenceTerms,
    only: tuple[str, date] | None = None,
) -> CollectionPlan:
    """Plan tonight's collection; `only` names one intent (maturation key, decision session) to collect now."""
    daily: dict[tuple[date, date], set[str]] = defaultdict(set)
    benchmarks: dict[tuple[date, date], set[str]] = defaultdict(set)
    minute: set[tuple[date, date, str]] = set()
    acquirers: set[tuple[date, date, str]] = set()
    actions: dict[tuple[date, str], date] = {}
    held: dict[date, tuple[tuple[BarReceipt, ...], tuple[ActionReceipt, ...]]] = {}
    entries = repository.pending() if only is None else [only]
    for key, session in entries:
        if repository.has_outcome(key, session):
            continue
        intent = repository.load_intent(key, session)
        if repository.semantic_canonical_key(intent.semantic_prediction_id, session) != key:
            continue
        forced = only is not None
        if horizon_close(intent) > now or (not forced and now > terms.deadline(intent) + FREEZE_AFTER):
            continue
        if session not in held:
            held[session] = (load_bar_receipts(receipts_root, [session]), load_action_receipts(receipts_root, [session]))
        bar_receipts, action_receipts = held[session]
        last = session_after(session, swing_horizon_sessions(intent.horizon))
        latest = repository.latest_attempt(key, session)
        reasons = latest.reasons if latest is not None else ()
        if forced or _bars_due(intent, bar_receipts, last=last, now=now, terms=terms):
            daily[(session, last)].add(intent.ticker)
            policy = intent.label_policy
            benchmarks[(session, last)].update(
                {str(policy["broad_benchmark"]).upper(), str(policy["growth_benchmark"]).upper(), intent.primary_benchmark}
            )
        if reasons == ("interior_gap_needs_minute_bars",):
            gap = _first_gap(intent, bar_receipts, terms)
            if gap is not None:
                minute.add((session, gap, intent.ticker))
        if reasons in (("stock_gap_without_cessation",), (OPERATOR_VERIFIED,)):
            # Provider evidence may still name a more specific reason than an operator's.
            gap = _first_gap(intent, bar_receipts, terms)
            end = min(now.date(), (gap or last) + FREEZE_AFTER)
            interval = NIGHTLY if now <= terms.deadline(intent) else RECOLLECT_AFTER
            for symbol in symbols_in_use(intent, action_receipts, terms=terms):
                if forced or _actions_due(action_receipts, session=session, symbol=symbol, now=now, interval=interval):
                    actions[(session, symbol)] = max(end, actions.get((session, symbol), end))
        if reasons in (("stock_merger",), ("stock_and_cash_merger",)):
            # A stock merger's stress fill values its shares at the acquirer's close.
            cessation = cessation_evidence(intent, action_receipts=action_receipts, memberships=None, terms=terms)
            acquirer = acquirer_terms(reasons[0], cessation.record if cessation is not None else None)
            if acquirer is not None and not complete_bar_receipts(
                bar_receipts, decision_session=session, symbol=acquirer[0], first_session=session, last_session=acquirer[1]
            ):
                acquirers.add((session, acquirer[1], acquirer[0]))
    bar_units: list[OutcomeBarUnit] = []
    for (session, last), stocks in sorted(daily.items()):
        shared = sorted(benchmarks[(session, last)])
        size = _PAGE_SYMBOLS - len(shared)
        ordered = sorted(stocks - set(shared))
        for start in range(0, max(len(ordered), 1), size):
            bar_units.append(OutcomeBarUnit(session, session, last, tuple(sorted({*ordered[start : start + size], *shared}))))
    bar_units.extend(
        OutcomeBarUnit(session, gap, gap, (symbol,), "1Min") for session, gap, symbol in sorted(minute)
    )
    for session, effective, symbol in sorted(acquirers):
        try:
            bar_units.append(OutcomeBarUnit(session, session, max(session, effective), (symbol,)))
        except ValueError:
            continue  # A symbol the bars endpoint cannot take; the stress fill waits without it.
    action_units = tuple(
        CorporateActionUnit(session, symbol, session, end) for (session, symbol), end in sorted(actions.items())
    )
    return CollectionPlan(bar_units=tuple(bar_units), action_units=action_units)


def _bars_due(
    intent: PredictionMaturationIntent,
    receipts: tuple[BarReceipt, ...],
    *,
    last: date,
    now: datetime,
    terms: EvidenceTerms,
) -> bool:
    """Every night until a settled receipt covers the path, then weekly."""
    decision = intent.decision_session_et
    covering = complete_bar_receipts(receipts, decision_session=decision, symbol=intent.ticker,
                                     first_session=decision, last_session=last)
    if not any(terms.settles(intent, receipt) for receipt in covering):
        return True
    requested = [
        receipt.finished_at_utc
        for receipt in receipts
        if receipt.unit.timeframe == "1Day" and receipt.unit.decision_session == decision and intent.ticker in receipt.unit.symbols
    ]
    return now - max(requested) >= RECOLLECT_AFTER


def _actions_due(
    receipts: tuple[ActionReceipt, ...], *, session: date, symbol: str, now: datetime, interval: timedelta
) -> bool:
    requested = [
        receipt.finished_at_utc
        for receipt in receipts
        if receipt.unit.decision_session == session and receipt.unit.symbol == symbol
    ]
    return not requested or now - max(requested) >= interval


def _first_gap(intent: PredictionMaturationIntent, receipts: tuple[BarReceipt, ...], terms: EvidenceTerms) -> date | None:
    try:
        return path_evidence(intent, receipts, terms=terms).first_gap
    except DataReadinessError:
        return None
