"""Event-to-feature reconciliation (R3 P0-3).

Every accepted event id resolves to exactly one status: ``matched`` or an
explicit rejection reason. Zero unexplained events is the invariant that lets
promotion trust the alignment audit instead of hardcoded zeros. The reconciliation
is content-addressed so the same events and decisions always produce the same hash.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from market_predictor.canonical.joins import DEFAULT_EVENT_WINDOWS

RECONCILE_STATUSES: tuple[str, ...] = (
    "matched",
    "duplicate",
    "wrong_ticker",
    "unavailable_future",
    "unknown_relevance",
    "irrelevant",
    "outside_window",
)

_REJECTION_STATUSES: frozenset[str] = frozenset(RECONCILE_STATUSES) - {"matched"}
# Statuses that indicate a lineage error rather than an expected exclusion.
_ERROR_STATUSES: frozenset[str] = frozenset({"duplicate", "wrong_ticker", "unavailable_future", "outside_window"})


def reconcile_events(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
    relevance_floor: float = 0.5,
) -> pd.DataFrame:
    """Assign every accepted event id exactly one reconciliation status.

    Precedence (first match wins): duplicate id, wrong ticker (not in the decision
    universe), unknown relevance (NaN), irrelevant (below floor), unavailable/future
    (no decision at or after the event's availability), matched (a decision falls
    within the event's largest lookback window), else outside_window.
    """

    max_window_ns = int(max(windows.values()).value)
    decision_tickers = set(decisions["ticker"].astype(str).str.upper().str.strip())
    prepared_decisions = decisions.assign(
        _ticker=decisions["ticker"].astype(str).str.upper().str.strip(),
        _decision=pd.to_datetime(decisions["decision_time_utc"], utc=True, errors="coerce"),
    )
    decision_times_ns: dict[str, np.ndarray] = {}
    for ticker, part in prepared_decisions.groupby("_ticker", sort=False):
        valid = part["_decision"].dropna()
        if valid.empty:
            decision_times_ns[str(ticker)] = np.empty(0, dtype=np.int64)
        else:
            decision_times_ns[str(ticker)] = np.sort(pd.DatetimeIndex(valid).as_unit("ns").asi8)

    tickers = events["ticker"].astype(str).str.upper().str.strip().to_numpy()
    availability = pd.to_datetime(events["feature_available_at_utc"], utc=True, errors="coerce")
    availability_ns = pd.DatetimeIndex(availability).as_unit("ns").asi8
    availability_missing = availability.isna().to_numpy()
    relevance = pd.to_numeric(events.get("relevance"), errors="coerce").to_numpy(dtype=float)
    event_ids = events["event_id"].astype(str).to_numpy()

    seen: set[str] = set()
    statuses: list[str] = []
    for index in range(len(event_ids)):
        event_id = event_ids[index]
        if event_id in seen:
            statuses.append("duplicate")
            continue
        seen.add(event_id)
        statuses.append(
            _classify(
                ticker=tickers[index],
                availability_ns=availability_ns[index],
                availability_missing=bool(availability_missing[index]),
                relevance=relevance[index],
                decision_tickers=decision_tickers,
                decision_times_ns=decision_times_ns,
                max_window_ns=max_window_ns,
                relevance_floor=relevance_floor,
            )
        )
    return pd.DataFrame({"event_id": event_ids, "ticker": tickers, "status": statuses})


def _classify(
    *,
    ticker: str,
    availability_ns: int,
    availability_missing: bool,
    relevance: float,
    decision_tickers: set[str],
    decision_times_ns: Mapping[str, np.ndarray],
    max_window_ns: int,
    relevance_floor: float,
) -> str:
    if ticker not in decision_tickers:
        return "wrong_ticker"
    if not np.isfinite(relevance):
        return "unknown_relevance"
    if relevance < relevance_floor:
        return "irrelevant"
    if availability_missing:
        return "unavailable_future"
    times = decision_times_ns.get(ticker, np.empty(0, dtype=np.int64))
    position = int(np.searchsorted(times, availability_ns, side="left"))
    if position >= times.size:
        return "unavailable_future"
    if int(times[position]) - availability_ns <= max_window_ns:
        return "matched"
    return "outside_window"


def reconciliation_summary(artifact: pd.DataFrame) -> dict[str, int]:
    counts = artifact["status"].astype(str).value_counts().to_dict()
    summary = {status: int(counts.get(status, 0)) for status in RECONCILE_STATUSES}
    summary["total_events"] = int(len(artifact))
    summary["unexplained_events"] = int((~artifact["status"].astype(str).isin(RECONCILE_STATUSES)).sum())
    summary["lineage_error_events"] = int(artifact["status"].astype(str).isin(_ERROR_STATUSES).sum())
    return summary


def stamped_scalar(frame: pd.DataFrame, column: str, *, default: int = 0) -> int:
    """Read a per-dataset integer stamped as a constant column (or the default)."""

    if column in frame.columns and len(frame):
        return int(pd.to_numeric(frame[column], errors="coerce").fillna(default).iloc[0])
    return default


def stamped_hash(frame: pd.DataFrame, column: str) -> str:
    """Read a per-dataset hash stamped as a constant column (or empty string)."""

    if column in frame.columns and len(frame):
        value = str(frame[column].iloc[0])
        return value if value and value.lower() != "nan" else ""
    return ""


def reconciliation_sha256(artifact: pd.DataFrame) -> str:
    payload = (
        artifact.loc[:, ["event_id", "status"]]
        .astype(str)
        .sort_values(["event_id", "status"])
        .to_dict("records")
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
