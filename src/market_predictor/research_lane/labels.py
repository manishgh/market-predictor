"""Research label: next-session adjusted open to adjusted close h sessions after the decision.

Decision after the close of session t; entry at the adjusted open of t+1; exit at
the adjusted close of t+h. Label = close[t+h] / open[t+1] - 1 (a fraction, not a
percent). Built on provider-adjusted prices, so every result is a "research label".
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
import numpy.typing as npt

from market_predictor.research_lane.panel import ResearchPanel

HORIZONS: Final = (5, 10, 21, 63)
FIRST_DECISION_DATE: Final = date(2019, 7, 9)
KNOWN: Final = 0
WINDOW_PAST_DEVELOPMENT_END: Final = 1
ENTRY_BAR_MISSING: Final = 2
EXIT_BAR_MISSING: Final = 3
MISSING_REASONS: Final = {
    WINDOW_PAST_DEVELOPMENT_END: "window_runs_past_development_end",
    ENTRY_BAR_MISSING: "entry_bar_missing",
    EXIT_BAR_MISSING: "exit_bar_missing",
}

FloatArray = npt.NDArray[np.float64]
ReasonArray = npt.NDArray[np.int8]


@dataclass(frozen=True)
class HorizonLabels:
    horizon: int
    stock: FloatArray
    reason: ReasonArray
    spy: FloatArray
    qqq: FloatArray


def interval_returns(open_: npt.NDArray[np.floating], close: npt.NDArray[np.floating],
                     horizon: int) -> tuple[FloatArray, ReasonArray]:
    """Return close[t+h]/open[t+1]-1 per decision row t with a reason code for unknown labels.

    Rows whose exit session is beyond the last loaded session (the development end)
    are unknown. Non-finite or non-positive prices count as missing bars.
    """
    if horizon < 1:
        raise ValueError("label horizon must be at least one session")
    opens = np.asarray(open_, dtype=np.float64)
    closes = np.asarray(close, dtype=np.float64)
    if opens.shape != closes.shape:
        raise ValueError("open and close arrays differ in shape")
    sessions = opens.shape[0]
    values = np.full(opens.shape, np.nan, dtype=np.float64)
    reason = np.full(opens.shape, WINDOW_PAST_DEVELOPMENT_END, dtype=np.int8)
    decisions = sessions - horizon
    if decisions > 0:
        entry = opens[1:decisions + 1]
        exit_ = closes[horizon:sessions]
        entry_ok = np.isfinite(entry) & (entry > 0)
        exit_ok = np.isfinite(exit_) & (exit_ > 0)
        known = entry_ok & exit_ok
        with np.errstate(invalid="ignore", divide="ignore"):
            values[:decisions] = np.where(known, exit_ / np.where(entry_ok, entry, 1.0) - 1.0, np.nan)
        reason[:decisions] = np.where(~entry_ok, ENTRY_BAR_MISSING, np.where(~exit_ok, EXIT_BAR_MISSING, KNOWN))
    return values, reason


def research_labels(panel: ResearchPanel, horizon: int) -> HorizonLabels:
    stock, reason = interval_returns(panel.open, panel.close, horizon)
    spy, _ = interval_returns(panel.spy_open, panel.spy_close, horizon)
    qqq, _ = interval_returns(panel.qqq_open, panel.qqq_close, horizon)
    return HorizonLabels(horizon=horizon, stock=stock, reason=reason, spy=spy, qqq=qqq)


def first_decision_row(panel: ResearchPanel, first_decision_date: date = FIRST_DECISION_DATE) -> int:
    """Index of the first session on or after the first permitted decision date."""
    return int(np.searchsorted(panel.sessions, np.datetime64(first_decision_date, "D"), side="left"))
