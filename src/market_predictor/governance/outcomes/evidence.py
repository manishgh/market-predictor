"""Outcome evidence from collection receipts: each path's bars, the gaps they prove, and cessation.

A path takes all of its bars from one receipt, the complete one with the most usable sessions,
since each receipt carries the price adjustments of its own retrieval. A session is a proven
gap only when a settled receipt asked for it and no complete receipt ever returned a usable
bar for it. A receipt is settled when it was retrieved at least the settlement period after
the horizon's last close.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from market_predictor.canonical.cutoffs import NEW_YORK
from market_predictor.collection.outcome_bars import (
    DAILY_BAR_COLUMNS,
    ActionReceipt,
    BarReceipt,
    complete_bar_receipts,
    daily_path_bars,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import PredictionMaturationIntent, swing_horizon_sessions
from market_predictor.governance.outcomes.sessions import session_after, session_close
from market_predictor.swing.labels.holding_paths import validate_outcome_observations

_MERGER_FAMILIES = {
    "cash_mergers": "cash_merger",
    "stock_mergers": "stock_merger",
    "stock_and_cash_mergers": "stock_and_cash_merger",
    "reorganizations": "reorganization",
}


@dataclass(frozen=True)
class EvidenceTerms:
    """The drift policy's terms for outcome evidence."""

    grace_days: int
    settlement_days: int
    drift_policy_sha256: str

    def deadline(self, intent: PredictionMaturationIntent) -> datetime:
        """The horizon's last close plus the grace: an outcome still missing then is overdue."""
        return horizon_close(intent) + timedelta(days=self.grace_days)

    def settles(self, intent: PredictionMaturationIntent, receipt: BarReceipt) -> bool:
        """Whether every page of the receipt was retrieved the settlement period after the last close."""
        return receipt.started_at_utc >= horizon_close(intent) + timedelta(days=self.settlement_days)

    def settles_session(self, session: date, receipt: BarReceipt) -> bool:
        """Whether every page of the receipt was retrieved the settlement period after one session's close."""
        return receipt.started_at_utc >= session_close(session) + timedelta(days=self.settlement_days)


@dataclass(frozen=True)
class Cessation:
    """Why a stock stopped trading: the reason, the receipts that prove it and the provider's record."""

    reason: str
    receipt_ids: tuple[str, ...]
    record: Mapping[str, Any] | None


@dataclass(frozen=True)
class PathEvidence:
    """One intent's bars from its chosen receipts, and what the stock's receipts prove."""

    bars: pd.DataFrame
    path_sessions: tuple[date, ...]
    stock_usable: frozenset[date]
    proven_stock_gaps: frozenset[date]
    receipt_ids: tuple[str, ...]

    @property
    def first_gap(self) -> date | None:
        """The first path session without a usable stock bar in the chosen receipt."""
        return next((session for session in self.path_sessions if session not in self.stock_usable), None)

    def trades_after(self, gap: date) -> bool:
        """Whether the chosen receipt holds a usable stock bar after the gap: an interior gap."""
        return any(session > gap for session in self.stock_usable)


def horizon_close(intent: PredictionMaturationIntent) -> datetime:
    return session_close(session_after(intent.decision_session_et, swing_horizon_sessions(intent.horizon)))


def path_evidence(
    intent: PredictionMaturationIntent,
    receipts: Sequence[BarReceipt],
    *,
    terms: EvidenceTerms,
) -> PathEvidence:
    decision = intent.decision_session_et
    horizon = swing_horizon_sessions(intent.horizon)
    path_sessions = tuple(session_after(decision, offset) for offset in range(1, horizon + 1))
    last = path_sessions[-1]
    policy = intent.label_policy
    benchmarks = (str(policy["broad_benchmark"]).upper(), str(policy["growth_benchmark"]).upper(), intent.primary_benchmark)
    frames: list[pd.DataFrame] = []
    receipt_ids: set[str] = set()
    stock = _candidates(receipts, decision=decision, symbol=intent.ticker, last=last)
    stock_usable: frozenset[date] = frozenset()
    if stock:
        chosen = _chosen(stock)
        frames.append(chosen.frame)
        receipt_ids.add(chosen.receipt.receipt_id)
        stock_usable = chosen.usable
    for symbol in dict.fromkeys(benchmarks):
        candidates = _candidates(receipts, decision=decision, symbol=symbol, last=last)
        if candidates:
            chosen = _chosen(candidates)
            frames.append(chosen.frame)
            receipt_ids.add(chosen.receipt.receipt_id)
    settled = [candidate for candidate in stock if terms.settles(intent, candidate.receipt)]
    ever_usable = frozenset().union(*(candidate.usable for candidate in stock))
    proven = frozenset(session for session in path_sessions if settled and session not in ever_usable)
    receipt_ids.update(candidate.receipt.receipt_id for candidate in settled)
    return PathEvidence(
        bars=pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(DAILY_BAR_COLUMNS)),
        path_sessions=path_sessions,
        stock_usable=frozenset(session for session in stock_usable if session in path_sessions),
        proven_stock_gaps=proven,
        receipt_ids=tuple(sorted(receipt_ids)),
    )


def cessation_evidence(
    intent: PredictionMaturationIntent,
    *,
    action_receipts: Sequence[ActionReceipt],
    memberships: pd.DataFrame | None,
    terms: EvidenceTerms,
) -> Cessation | None:
    """Why the stock stopped trading, from actions effective between the decision and the deadline.

    Returns None without such evidence. Renames are followed, so a
    merger naming a later symbol of the same company counts. Worthless removals and name
    changes carry only a process date in the provider's schema. The provider processes a
    worthless removal long after the delisting, so one processed on or after the decision
    session counts; mergers and reorganizations need their effective date inside the window.
    """
    window = (intent.decision_session_et, terms.deadline(intent).astimezone(NEW_YORK).date())
    symbols = symbols_in_use(intent, action_receipts, terms=terms)
    for receipt in _complete_actions(action_receipts, intent, symbols):
        for family, reason in _MERGER_FAMILIES.items():
            for record in receipt.actions.get(family, ()):
                if record.get("acquiree_symbol") in symbols and _within(record.get("effective_date"), window):
                    return Cessation(reason, (receipt.receipt_id,), record)
        for record in receipt.actions.get("worthless_removals", ()):
            if record.get("symbol") in symbols and _within(record.get("process_date"), (window[0], date.max)):
                return Cessation("worthless_removal", (receipt.receipt_id,), record)
    if memberships is not None and _membership_removed(memberships, intent, window):
        return Cessation("membership_removal", (), None)
    return None


def symbols_in_use(
    intent: PredictionMaturationIntent,
    action_receipts: Sequence[ActionReceipt],
    *,
    terms: EvidenceTerms,
) -> frozenset[str]:
    """The decision-time ticker and every later name the company took within the window."""
    window = (intent.decision_session_et, terms.deadline(intent).astimezone(NEW_YORK).date())
    symbols = {intent.ticker}
    while True:
        renamed = {
            str(record["new_symbol"]).upper()
            for receipt in _complete_actions(action_receipts, intent, frozenset(symbols))
            for record in receipt.actions.get("name_changes", ())
            if record.get("old_symbol") in symbols and record.get("new_symbol") and _within(record.get("process_date"), window)
        }
        if renamed <= symbols:
            return frozenset(symbols)
        symbols |= renamed


@dataclass(frozen=True)
class _Candidate:
    receipt: BarReceipt
    frame: pd.DataFrame
    usable: frozenset[date]


def _candidates(receipts: Sequence[BarReceipt], *, decision: date, symbol: str, last: date) -> list[_Candidate]:
    """Each covering receipt's path for the symbol; a defective receipt is skipped, never evidence."""
    covering = complete_bar_receipts(
        receipts, decision_session=decision, symbol=symbol, first_session=decision, last_session=last
    )
    candidates: list[_Candidate] = []
    for receipt in covering:
        try:
            frame = daily_path_bars(receipt, symbol)
        except DataReadinessError:
            continue
        frame = frame.loc[frame["session_date_et"].le(last)].reset_index(drop=True)
        candidates.append(_Candidate(receipt, frame, _usable_sessions(frame)))
    if covering and not candidates:
        raise DataReadinessError(f"every receipt covering {symbol} for {decision} is defective")
    return candidates


def _chosen(candidates: Sequence[_Candidate]) -> _Candidate:
    """The receipt with the most usable sessions, the latest among equals."""
    return max(candidates, key=lambda item: (len(item.usable), item.receipt.finished_at_utc, item.receipt.receipt_id))


def _usable_sessions(frame: pd.DataFrame) -> frozenset[date]:
    if frame.empty:
        return frozenset()
    validated = validate_outcome_observations(frame)
    return frozenset(validated.loc[validated["outcome_observation_valid"], "session_date_et"])


def usable_rows(bars: pd.DataFrame, ticker: str) -> dict[date, pd.Series]:
    """A ticker's usable bars by session: exactly one row, valid for the observation validator."""
    rows = bars.loc[bars["ticker"].eq(ticker)].reset_index(drop=True)
    if rows.empty:
        return {}
    valid = validate_outcome_observations(rows)["outcome_observation_valid"]
    single = rows.groupby("session_date_et")["ticker"].transform("size").eq(1)
    return {row["session_date_et"]: row for _, row in rows.loc[valid & single].iterrows()}


def _complete_actions(
    receipts: Sequence[ActionReceipt],
    intent: PredictionMaturationIntent,
    symbols: frozenset[str],
) -> list[ActionReceipt]:
    return [
        receipt
        for receipt in receipts
        if receipt.complete
        and receipt.unit.decision_session == intent.decision_session_et
        and receipt.unit.symbol in symbols
    ]


def _within(value: object, window: tuple[date, date]) -> bool:
    try:
        moment = date.fromisoformat(str(value))
    except ValueError:
        return False
    return window[0] <= moment <= window[1]


def _membership_removed(memberships: pd.DataFrame, intent: PredictionMaturationIntent, window: tuple[date, date]) -> bool:
    """Whether the security's point-in-time membership ended inside the window."""
    rows = memberships.loc[memberships["security_id"].astype(str).eq(intent.canonical_security_id)]
    ends: Mapping[int, object] = rows["effective_to_utc"].to_dict()
    for value in ends.values():
        if value is None or pd.isna(value):
            continue
        ended = pd.Timestamp(str(value))
        if ended.tzinfo is None:
            raise DataReadinessError("a membership end is not timezone-aware")
        if window[0] <= ended.tz_convert(NEW_YORK).date() <= window[1]:
            return True
    return False
