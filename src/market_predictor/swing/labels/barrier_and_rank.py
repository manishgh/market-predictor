"""Two labels per row: what the trade did, and how it ranked against peers.

The barrier label answers a question about one stock. Starting from the entry
price, did the position reach its profit target, hit its stop, or survive to
expiry? It encodes the exit rule, so it reflects that a real position is closed
early rather than carried to a fixed horizon. A label that ignores the stop
credits the strategy with drawdowns it would never have sat through.

The rank label answers a question about the cross-section. Among every stock
tradable on that date, was this one near the top of forward return, near the
bottom, or in the middle? It is what makes selection possible: a barrier label
says which trades worked but gives no basis for choosing among the dozens that
qualify on the same day.

Neither substitutes for the other. Using only the barrier label reproduces a
failure already on record, where selection among same-day qualifiers fell back
to trading volume, which has no relationship to expected return. Using only the
rank label produces a stock picker that assumes every position is held to expiry
regardless of how far it moved against you.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

import market_predictor.modeling.label_outcomes as label_outcomes
from market_predictor.core.errors import DataReadinessError
from market_predictor.execution_policy import executable_fill_price

BARRIER_COLUMNS: Final = (
    "barrier_label",
    "exit_session",
    "exit_price",
    "holding_sessions",
    "target_price",
    "stop_price",
)
RANK_COLUMNS: Final = ("rank_label", "forward_return", "rank_percentile")


@dataclass(frozen=True, slots=True)
class BarrierSpec:
    """Barrier geometry, expressed in multiples of average true range."""

    target_atr_multiple: float
    stop_atr_multiple: float
    horizon_sessions: int
    same_bar_resolution: str = "stop_first"

    def __post_init__(self) -> None:
        if self.target_atr_multiple <= self.stop_atr_multiple:
            raise ValueError("target multiple must exceed stop multiple")
        if self.stop_atr_multiple <= 0:
            raise ValueError("stop multiple must be positive")
        if self.horizon_sessions < 1:
            raise ValueError("horizon must span at least one session")
        if self.same_bar_resolution != "stop_first":
            raise ValueError(
                "a bar shows both barriers were touched but not which came "
                "first; only the stop-first resolution is conservative"
            )


def apply_triple_barrier(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    *,
    spec: BarrierSpec,
) -> pd.DataFrame:
    """Resolve each entry to the barrier it reached first.

    `bars` holds one ordered row per session for a single security, with
    `session`, `open`, `high`, `low`, `close`. `entries` holds the decision
    sessions and the `atr` observed at each decision.

    Entry is the open of the session after the decision, so no part of the
    decision bar is used to price the fill. Barriers are then checked against
    each subsequent session's high and low. A session whose range spans both
    barriers is resolved to the stop, because the bar records that both prices
    traded without recording their order.
    """

    _require(bars, ("session", "open", "high", "low", "close"), "bars")
    _require(entries, ("session", "atr"), "entries")
    if bars.empty or entries.empty:
        return _empty_barrier_frame(entries)

    ordered = bars.sort_values("session", kind="stable").reset_index(drop=True)
    sessions = ordered["session"].to_numpy()
    opens = ordered["open"].to_numpy(dtype=float)
    highs = ordered["high"].to_numpy(dtype=float)
    lows = ordered["low"].to_numpy(dtype=float)
    closes = ordered["close"].to_numpy(dtype=float)
    position_of = {session: index for index, session in enumerate(sessions)}

    records: list[dict[str, object]] = []
    for decision_session, atr in zip(
        entries["session"], entries["atr"], strict=True
    ):
        decision_index = position_of.get(decision_session)
        if decision_index is None or not np.isfinite(atr) or atr <= 0:
            records.append(_unresolved_record(decision_session))
            continue
        entry_index = decision_index + 1
        last_index = entry_index + spec.horizon_sessions - 1
        if entry_index >= len(sessions) or last_index >= len(sessions):
            # The horizon runs past the data, so the outcome is unknown rather
            # than a timeout. Labelling it zero would invent an observation.
            records.append(_unresolved_record(decision_session))
            continue

        entry_price = float(opens[entry_index])
        target = entry_price + spec.target_atr_multiple * float(atr)
        stop = entry_price - spec.stop_atr_multiple * float(atr)
        window = slice(entry_index, last_index + 1)
        touched_stop = lows[window] <= stop
        touched_target = highs[window] >= target

        label = label_outcomes.TIMEOUT
        offset = spec.horizon_sessions - 1
        exit_price = float(closes[last_index])
        first_stop = int(np.argmax(touched_stop)) if touched_stop.any() else -1
        first_target = int(np.argmax(touched_target)) if touched_target.any() else -1
        if first_stop >= 0 or first_target >= 0:
            if first_stop < 0:
                label, offset, exit_price = (
                    label_outcomes.TARGET_HIT,
                    first_target,
                    target,
                )
            elif first_target < 0:
                label, offset, exit_price = (
                    label_outcomes.STOP_HIT,
                    first_stop,
                    stop,
                )
            elif first_stop <= first_target:
                # Ties included: the same bar touching both resolves to the stop.
                label, offset, exit_price = (
                    label_outcomes.STOP_HIT,
                    first_stop,
                    stop,
                )
            else:
                label, offset, exit_price = (
                    label_outcomes.TARGET_HIT,
                    first_target,
                    target,
                )
        trigger_open = float(opens[entry_index + offset])
        exit_price = executable_fill_price(
            outcome=(
                "stop_first"
                if label == label_outcomes.STOP_HIT
                else "target_first"
                if label == label_outcomes.TARGET_HIT
                else "timeout"
            ),
            target_price=target,
            stop_price=stop,
            trigger_open=trigger_open,
            final_price=exit_price,
        )

        records.append(
            {
                "session": decision_session,
                "barrier_label": label,
                "exit_session": sessions[entry_index + offset],
                "exit_price": exit_price,
                "holding_sessions": offset + 1,
                "target_price": target,
                "stop_price": stop,
            }
        )
    return pd.DataFrame.from_records(records)


def apply_cross_sectional_rank(
    panel: pd.DataFrame,
    *,
    top_quantile: float,
    bottom_quantile: float,
    within_sector: bool,
    minimum_cross_section: int,
) -> pd.DataFrame:
    """Label each row by where its forward return sat among that date's peers.

    Ranking is computed inside one session, never across time, so a stock is
    compared with what else was available to buy at that moment. Ranking within
    a sector compares a stock with its peers rather than with whichever sector
    happened to be in favour, which keeps a sector rotation from being scored as
    stock selection.

    A cross-section smaller than the minimum yields no labels for that group.
    Quantiles of a handful of rows describe the handful, not the market.
    """

    required = ["session", "forward_return"]
    if within_sector:
        required.append("sector")
    _require(panel, tuple(required), "panel")
    if not 0.0 < top_quantile < 0.5 or not 0.0 < bottom_quantile < 0.5:
        raise ValueError("quantiles must each be inside (0, 0.5)")
    if top_quantile + bottom_quantile >= 1.0:
        raise ValueError("quantiles must leave a middle band")
    if panel.empty:
        return panel.assign(
            rank_label=pd.Series(dtype=int),
            rank_percentile=np.nan,
            ranking_group_size=pd.Series(dtype="int32"),
        )

    keys = ["session", "sector"] if within_sector else ["session"]
    frame = panel.copy()
    returns = pd.to_numeric(frame["forward_return"], errors="coerce")
    frame["forward_return"] = returns
    grouped = frame.groupby(keys, sort=False)["forward_return"]
    # `pct=True` maps each row to its position in its own group, so the cut is
    # relative to that session rather than to a threshold carried across time.
    percentile = grouped.rank(pct=True, method="average")
    group_size = grouped.transform("size").astype("int32")
    eligible = group_size >= minimum_cross_section

    # Strict above, inclusive below, so the two tails hold equal counts: a
    # percentile of exactly the cut belongs to the middle on the top side and to
    # the tail on the bottom side.
    label = pd.Series(label_outcomes.RANK_MIDDLE, index=frame.index, dtype=int)
    label[percentile > 1.0 - top_quantile] = label_outcomes.RANK_TOP
    label[percentile <= bottom_quantile] = label_outcomes.RANK_BOTTOM
    frame["rank_label"] = label.where(eligible & returns.notna())
    frame["rank_percentile"] = percentile.where(eligible & returns.notna())
    frame["ranking_group_size"] = group_size
    return frame


def forward_return_from_barrier(
    barriers: pd.DataFrame,
    entry_prices: pd.Series,
) -> pd.Series:
    """Realised return of the managed position, not of a fixed-horizon hold.

    Using the barrier exit rather than the horizon close keeps the ranking
    consistent with how the position is actually closed.
    """

    _require(barriers, ("exit_price",), "barriers")
    exits = pd.to_numeric(barriers["exit_price"], errors="coerce")
    entries = pd.to_numeric(entry_prices, errors="coerce")
    return (exits / entries.where(entries > 0)) - 1.0


def _require(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} is missing columns: {missing}")


def _unresolved_record(session: object) -> dict[str, object]:
    return {
        "session": session,
        "barrier_label": pd.NA,
        "exit_session": pd.NA,
        "exit_price": np.nan,
        "holding_sessions": pd.NA,
        "target_price": np.nan,
        "stop_price": np.nan,
    }


def _empty_barrier_frame(entries: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {name: pd.Series(dtype="object") for name in ("session", *BARRIER_COLUMNS)},
        index=pd.RangeIndex(0),
    ).astype({"session": entries["session"].dtype} if len(entries) else {})
