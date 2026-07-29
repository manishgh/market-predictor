"""Verify that a membership interval's symbol claim is supported by evidence.

The point-in-time universe is built by taking each security's current ticker and
back-filling it, breaking the back-fill only where the provider's
corporate-actions feed reports a change. Where that feed is incomplete the
back-fill silently becomes a hindsight assertion: the artifact claims a security
traded under a symbol in 2021 on no evidence beyond it trading under that symbol
today.

Such a row is structurally valid and factually wrong, so hashes, authorities, and
non-overlap checks all pass. It also defeats the two existing defences, because
membership filtering applies a wrong interval faithfully and identity grouping
cannot help when the contaminated rows carry the correct security id.

This module checks the claim against observed bars. A symbol held by one
continuous security produces one continuous price series; a symbol that changed
hands produces a gap, a level break, or both. An interval whose claim cannot be
supported is excluded rather than corrected, because correcting it would mean
substituting an inferred rename date for evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from market_predictor.edge_rebuild.corpus_integrity import IntegrityThresholds
from market_predictor.v3.errors import DataReadinessError

MEMBERSHIP_IDENTITY_SCHEMA = "edge_rebuild.membership_identity.v1"
REQUIRED_MEMBERSHIP_COLUMNS = (
    "ticker",
    "security_id",
    "effective_from_utc",
    "effective_to_utc",
)
REQUIRED_EVIDENCE_COLUMNS = ("session", "ticker", "bars", "last_close")


@dataclass
class MembershipVerification:
    """Which membership intervals are supported by bar evidence, and which are not."""

    schema: str = MEMBERSHIP_IDENTITY_SCHEMA
    verified: list[dict[str, Any]] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    unevaluated: list[dict[str, Any]] = field(default_factory=list)

    @property
    def excluded_securities(self) -> set[str]:
        return {str(item["security_id"]) for item in self.excluded}

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "verified_intervals": len(self.verified),
            "excluded_intervals": len(self.excluded),
            "unevaluated_intervals": len(self.unevaluated),
            "excluded_securities": sorted(self.excluded_securities),
            "exclusions": self.excluded,
        }


def verify_membership_identity(
    memberships: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    thresholds: IntegrityThresholds | None = None,
) -> MembershipVerification:
    """Check each membership interval's symbol claim against observed bars.

    `evidence` carries one row per observed (ticker, session) with `bars` and
    `last_close`. Intervals with no overlapping evidence are reported as
    unevaluated rather than excluded, because absence of bars in a window the
    corpus never covered proves nothing either way.
    """

    limits = thresholds or IntegrityThresholds()
    for frame, columns, name in (
        (memberships, REQUIRED_MEMBERSHIP_COLUMNS, "membership"),
        (evidence, REQUIRED_EVIDENCE_COLUMNS, "evidence"),
    ):
        missing = sorted(set(columns).difference(frame.columns))
        if missing:
            raise DataReadinessError(
                f"{name} input lacks required columns: {missing}"
            )

    observed = evidence.loc[evidence["bars"] > 0].copy()
    observed["session"] = observed["session"].astype(str)
    observed = observed.sort_values(["ticker", "session"])
    by_ticker = {
        str(ticker): group for ticker, group in observed.groupby("ticker", sort=False)
    }
    covered = (
        (observed["session"].min(), observed["session"].max())
        if not observed.empty
        else (None, None)
    )

    result = MembershipVerification()
    for row in memberships.itertuples():
        ticker = str(row.ticker)
        interval: dict[str, Any] = {
            "ticker": ticker,
            "security_id": str(row.security_id),
            "effective_from": str(pd.Timestamp(row.effective_from_utc).date()),
            "effective_to": (
                None
                if pd.isna(row.effective_to_utc)
                else str(pd.Timestamp(row.effective_to_utc).date())
            ),
        }
        window = _interval_evidence(by_ticker.get(ticker), interval)
        if window is None or len(window) < 2:
            interval["reason"] = _unevaluated_reason(window, covered, interval)
            result.unevaluated.append(interval)
            continue
        breach = _identity_breach(window, limits)
        if breach is None:
            result.verified.append({**interval, "evidence_sessions": len(window)})
        else:
            result.excluded.append({**interval, **breach, "evidence_sessions": len(window)})
    return result


def _interval_evidence(
    group: pd.DataFrame | None,
    interval: dict[str, Any],
) -> pd.DataFrame | None:
    if group is None or group.empty:
        return None
    inside = group[group["session"] >= interval["effective_from"]]
    if interval["effective_to"] is not None:
        inside = inside[inside["session"] < interval["effective_to"]]
    return inside


def _unevaluated_reason(
    window: pd.DataFrame | None,
    covered: tuple[str | None, str | None],
    interval: dict[str, Any],
) -> str:
    start, end = covered
    if start is None:
        return "no_bar_corpus"
    if interval["effective_to"] is not None and interval["effective_to"] <= start:
        return "interval_precedes_corpus"
    if interval["effective_from"] >= end:
        return "interval_follows_corpus"
    if window is None or window.empty:
        return "no_bars_in_covered_interval"
    return "insufficient_evidence"


def _identity_breach(
    window: pd.DataFrame,
    limits: IntegrityThresholds,
) -> dict[str, Any] | None:
    """One continuous security produces one continuous series."""

    closes = pd.to_numeric(window["last_close"], errors="coerce")
    previous = closes.shift()
    pair = pd.concat([closes, previous], axis=1)
    ratio = pair.max(axis=1) / pair.min(axis=1)
    worst = ratio.max()
    if pd.notna(worst) and float(worst) > limits.maximum_session_close_ratio:
        position = ratio.idxmax()
        return {
            "reason": "symbol_changed_hands",
            "detail": "close_ratio",
            "ratio": round(float(worst), 6),
            "at_session": str(window.loc[position, "session"]),
        }
    sessions = list(window["session"])
    gap = _longest_calendar_gap(sessions)
    if gap > limits.maximum_interior_gap_sessions:
        return {
            "reason": "symbol_unproven_for_interval",
            "detail": "interior_gap",
            "gap_sessions": gap,
            "at_session": sessions[min(gap, len(sessions) - 1)],
        }
    return None


def _longest_calendar_gap(sessions: list[str]) -> int:
    """Longest run of absent trading days, approximated from calendar dates.

    Weekends make this an approximation, so it is deliberately compared against
    a threshold far above normal weekend and holiday spacing.
    """

    if len(sessions) < 2:
        return 0
    stamps = pd.to_datetime(pd.Series(sessions))
    deltas = stamps.diff().dt.days.dropna()
    if deltas.empty:
        return 0
    return int(max(0, deltas.max() - 1))
