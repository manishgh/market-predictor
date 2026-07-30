"""Fail-closed corpus integrity gates.

Every defect these gates catch passed the existing hash, authority, and
non-overlap checks. Content identity proves a file was not altered; it says
nothing about whether the file contains what it claims. These gates check the
claim.

Three independent failures motivated each gate, all observed in real collected
data:

- A collection unit returning one bar was recorded as success alongside one
  returning seventy-eight, so three sessions reached 37% completeness unnoticed.
- A ticker whose symbol was reused by a second company produced a price series
  that jumped 36x overnight and passed every structural check.
- A provider emitted flat zero-volume placeholder bars for a dead symbol and
  they reached the model input layer as if observed.

Absence is not automatically a defect. An index member that does not trade for
five minutes of the regular session is rare and suspicious; a security that does
not trade at 04:15 is ordinary. The gates are therefore segment-aware, and
extended-hours absence is never treated as missing data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from market_predictor.v3.errors import DataReadinessError

CORPUS_INTEGRITY_SCHEMA = "edge_rebuild.corpus_integrity.v1"
REGULAR_SEGMENT = "regular"
EXTENDED_SEGMENTS = frozenset({"premarket", "postmarket"})


@dataclass(frozen=True, slots=True)
class IntegrityThresholds:
    """Frozen limits. Widening any of these requires new evidence, not preference."""

    minimum_regular_completeness: float = 0.50
    minimum_relative_completeness: float = 0.35
    minimum_sessions_for_baseline: int = 20
    minimum_session_median_completeness: float = 0.90
    maximum_session_close_ratio: float = 3.0
    maximum_interior_gap_sessions: int = 5
    reject_zero_volume: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_regular_completeness <= 1.0:
            raise ValueError("regular-session completeness floor must be in (0, 1]")
        if not 0.0 < self.minimum_relative_completeness <= 1.0:
            raise ValueError("relative completeness floor must be in (0, 1]")
        if self.minimum_sessions_for_baseline < 2:
            raise ValueError("a symbol baseline needs at least two sessions")
        if not 0.0 < self.minimum_session_median_completeness <= 1.0:
            raise ValueError("session median completeness floor must be in (0, 1]")
        if self.maximum_session_close_ratio <= 1.0:
            raise ValueError("close-ratio ceiling must exceed 1.0")
        if self.maximum_interior_gap_sessions < 1:
            raise ValueError("interior gap ceiling must be positive")


@dataclass
class IntegrityReport:
    """Findings, separated by whether they are defects or ordinary absence."""

    schema: str = CORPUS_INTEGRITY_SCHEMA
    truncated_ticker_sessions: list[dict[str, Any]] = field(default_factory=list)
    truncated_sessions: list[dict[str, Any]] = field(default_factory=list)
    identity_breaks: list[dict[str, Any]] = field(default_factory=list)
    fabricated_bars: list[dict[str, Any]] = field(default_factory=list)
    benign_thin_trading: int = 0
    checked_ticker_sessions: int = 0
    checked_bars: int = 0

    @property
    def defect_count(self) -> int:
        return (
            len(self.truncated_ticker_sessions)
            + len(self.truncated_sessions)
            + len(self.identity_breaks)
            + len(self.fabricated_bars)
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "checked_bars": self.checked_bars,
            "checked_ticker_sessions": self.checked_ticker_sessions,
            "defect_count": self.defect_count,
            "truncated_ticker_sessions": self.truncated_ticker_sessions,
            "truncated_sessions": self.truncated_sessions,
            "identity_breaks": self.identity_breaks,
            "fabricated_bars": self.fabricated_bars,
            "benign_thin_trading": self.benign_thin_trading,
        }

    def raise_if_defective(self, label: str) -> None:
        if self.defect_count:
            raise DataReadinessError(
                f"{label} failed corpus integrity: "
                f"{len(self.truncated_sessions)} truncated sessions, "
                f"{len(self.truncated_ticker_sessions)} truncated ticker-sessions, "
                f"{len(self.identity_breaks)} identity breaks, "
                f"{len(self.fabricated_bars)} fabricated-bar groups"
            )


def verify_corpus_integrity(
    ticker_sessions: pd.DataFrame,
    *,
    label: str,
    thresholds: IntegrityThresholds | None = None,
    member_sessions: Mapping[str, Iterable[str]] | None = None,
) -> IntegrityReport:
    """Verify one corpus and return every finding without stopping at the first.

    `ticker_sessions` carries one row per observed (ticker, session) with
    columns `session`, `ticker`, `segment`, `bars`, `expected_bars`,
    `zero_volume_bars`, `distinct_close`, and `last_close`.

    `member_sessions` maps ticker to the sessions on which it was an index
    member, and is required to distinguish an interior gap from a security that
    had simply not joined the universe yet.
    """

    limits = thresholds or IntegrityThresholds()
    required = {
        "session",
        "ticker",
        "segment",
        "bars",
        "expected_bars",
        "zero_volume_bars",
        "distinct_close",
        "last_close",
    }
    missing = sorted(required.difference(ticker_sessions.columns))
    if missing:
        raise DataReadinessError(f"{label} integrity input lacks columns: {missing}")
    report = IntegrityReport(
        checked_ticker_sessions=len(ticker_sessions),
        checked_bars=int(pd.to_numeric(ticker_sessions["bars"]).sum()),
    )
    if ticker_sessions.empty:
        return report

    frame = ticker_sessions.copy()
    frame["completeness"] = frame["bars"] / frame["expected_bars"].replace(0, pd.NA)
    regular = frame[frame["segment"] == REGULAR_SEGMENT]
    extended = frame[frame["segment"].isin(EXTENDED_SEGMENTS)]
    # Extended-hours absence is a genuine no-trade observation, never a defect.
    report.benign_thin_trading = int(len(extended))

    _check_regular_completeness(regular, limits, report)
    _check_session_completeness(regular, limits, report)
    _check_fabricated_bars(frame, limits, report)
    _check_identity_continuity(regular, limits, report, member_sessions)
    return report


def _check_regular_completeness(
    regular: pd.DataFrame,
    limits: IntegrityThresholds,
    report: IntegrityReport,
) -> None:
    """A session far below what this symbol normally prints is a transport gap.

    Completeness is judged against the symbol's own typical session, never a
    global floor. A high-priced, low-share-volume symbol genuinely does not
    trade in every five-minute bucket; AutoZone prints in roughly half of them
    and that data is correct. A global floor would call every such session a
    defect while missing a liquid symbol that dropped from full coverage to a
    tenth of it, which is the failure actually worth catching.

    Whether such a symbol is tradable is a separate question, answered by the
    eligibility filter at decision time, not here.
    """

    if regular.empty:
        return
    baseline = regular.groupby("ticker")["completeness"].transform("median")
    # A symbol needs enough history for its own baseline to mean anything.
    observed = regular.groupby("ticker")["completeness"].transform("size")
    relative = regular["completeness"] / baseline.where(baseline > 0)
    truncated = regular[
        (relative < limits.minimum_relative_completeness)
        & (observed >= limits.minimum_sessions_for_baseline)
    ]
    for row, own, share in zip(
        truncated.itertuples(),
        baseline[truncated.index],
        relative[truncated.index],
        strict=True,
    ):
        report.truncated_ticker_sessions.append(
            {
                "session": str(row.session),
                "ticker": str(row.ticker),
                "bars": int(row.bars),
                "expected_bars": int(row.expected_bars),
                "completeness": round(float(row.completeness), 6),
                "symbol_typical_completeness": round(float(own), 6),
                "share_of_typical": round(float(share), 6),
            }
        )


def _check_session_completeness(
    regular: pd.DataFrame,
    limits: IntegrityThresholds,
    report: IntegrityReport,
) -> None:
    """A whole session can be truncated while every ticker still reports bars."""

    if regular.empty:
        return
    grouped = regular.groupby("session")["completeness"]
    for session, median in grouped.median().items():
        if float(median) < limits.minimum_session_median_completeness:
            report.truncated_sessions.append(
                {
                    "session": str(session),
                    "median_completeness": round(float(median), 6),
                    "tickers": int(grouped.size().loc[session]),
                }
            )


def _check_fabricated_bars(
    frame: pd.DataFrame,
    limits: IntegrityThresholds,
    report: IntegrityReport,
) -> None:
    """Zero-volume and frozen-price bars are provider placeholders, not trades."""

    if not limits.reject_zero_volume:
        return
    zero_volume = frame[pd.to_numeric(frame["zero_volume_bars"], errors="coerce").fillna(0) > 0]
    for row in zero_volume.itertuples():
        report.fabricated_bars.append(
            {
                "session": str(row.session),
                "ticker": str(row.ticker),
                "reason": "zero_volume",
                "bars": int(row.zero_volume_bars),
            }
        )
    # A symbol resting at one price across a handful of pre-market prints is
    # ordinary, so a frozen price only means anything during the regular session.
    regular = frame[frame["segment"] == REGULAR_SEGMENT]
    frozen = regular[(regular["bars"] >= 10) & (regular["distinct_close"] == 1)]
    for row in frozen.itertuples():
        report.fabricated_bars.append(
            {
                "session": str(row.session),
                "ticker": str(row.ticker),
                "reason": "frozen_price",
                "bars": int(row.bars),
            }
        )


def _check_identity_continuity(
    regular: pd.DataFrame,
    limits: IntegrityThresholds,
    report: IntegrityReport,
    member_sessions: Mapping[str, Iterable[str]] | None,
) -> None:
    """A reused or renamed symbol shows a long gap, a level jump, or both."""

    if regular.empty:
        return
    ordered = regular.sort_values(["ticker", "session"])
    closes = pd.to_numeric(ordered["last_close"], errors="coerce")
    ordered = ordered.assign(close_value=closes)
    previous = ordered.groupby("ticker")["close_value"].shift()
    ratio = pd.concat([ordered["close_value"], previous], axis=1).max(axis=1) / pd.concat(
        [ordered["close_value"], previous], axis=1
    ).min(axis=1)
    jumped = ordered[(ratio > limits.maximum_session_close_ratio) & previous.notna()]
    for row, value in zip(jumped.itertuples(), ratio[jumped.index], strict=True):
        report.identity_breaks.append(
            {
                "ticker": str(row.ticker),
                "session": str(row.session),
                "reason": "close_ratio",
                "ratio": round(float(value), 6),
                "close": float(row.close_value),
            }
        )
    if member_sessions is None:
        return
    observed: dict[str, set[str]] = {}
    for row in ordered.itertuples():
        observed.setdefault(str(row.ticker), set()).add(str(row.session))
    for ticker, expected in member_sessions.items():
        have = observed.get(str(ticker), set())
        if not have:
            continue
        ordered_sessions = sorted(str(s) for s in expected)
        inside = [s for s in ordered_sessions if s >= min(have) and s <= max(have)]
        run = 0
        for session in inside:
            if session in have:
                if run > limits.maximum_interior_gap_sessions:
                    report.identity_breaks.append(
                        {
                            "ticker": str(ticker),
                            "session": session,
                            "reason": "interior_gap",
                            "gap_sessions": run,
                        }
                    )
                run = 0
            else:
                run += 1
