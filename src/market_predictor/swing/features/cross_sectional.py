"""Rescale features against the other stocks trading at the same moment.

A raw indicator carries no information about choice. "RSI is 68" says nothing
about whether this stock is worth buying instead of the four hundred others
available that morning; if every stock reads 68, none of them is strong. What
matters is the distance from what the rest of the market is doing right now.

Rescaling each feature against its own cross-section converts a market-timing
signal into a stock-selection signal. Without it a model learns "buy when the
indicator is high", which fires on everything at once and reproduces the index.
With it the model learns "buy whichever stock stands furthest above its peers
today", which is the only thing that can beat the index.

This was the missing piece behind the swing rejection already on record: that
strategy earned a real return and still lost to holding the benchmark, because
nothing in its features expressed relative strength.

Two rescalings are provided. The z-score keeps the shape of the distribution and
is sensitive to outliers. The rank keeps only the ordering, is unaffected by
outliers, and is what ranking models consume. Both are computed within a single
timestamp, never across time, so no value can be informed by a later session.
"""
from __future__ import annotations



from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError

Z_SUFFIX: Final = "_xs_z"
RANK_SUFFIX: Final = "_xs_rank"
SECTOR_Z_SUFFIX: Final = "_sector_z"


@dataclass(frozen=True, slots=True)
class CrossSectionSpec:
    """How a feature panel is rescaled against its own cross-section."""

    minimum_cross_section: int = 50
    winsorize_quantile: float = 0.01
    emit_zscore: bool = True
    emit_rank: bool = True
    emit_sector_relative: bool = True

    def __post_init__(self) -> None:
        if self.minimum_cross_section < 20:
            raise ValueError(
                "a cross-section under twenty describes the handful of stocks "
                "in it, not the market"
            )
        if not 0.0 <= self.winsorize_quantile < 0.25:
            raise ValueError("winsorize quantile must be inside [0, 0.25)")
        if not (self.emit_zscore or self.emit_rank):
            raise ValueError("at least one rescaling must be emitted")


def add_cross_sectional_features(
    panel: pd.DataFrame,
    features: Sequence[str],
    *,
    spec: CrossSectionSpec,
    timestamp_column: str = "session",
    sector_column: str = "sector",
) -> pd.DataFrame:
    """Attach cross-sectional rescalings of `features` to `panel`.

    Grouping is by `timestamp_column` alone, so every comparison is between
    stocks observable at the same instant. A group smaller than the minimum is
    left unscaled rather than scaled against too few peers, because a z-score
    over a dozen stocks measures that dozen.

    Sector-relative values compare a stock with its own sector. Without them a
    sector-wide rally is indistinguishable from stock selection, and the model
    learns to chase whichever sector is in favour.
    """

    missing = sorted(set(features).difference(panel.columns))
    if missing:
        raise DataReadinessError(f"panel is missing feature columns: {missing}")
    if timestamp_column not in panel.columns:
        raise DataReadinessError(f"panel is missing {timestamp_column!r}")
    if spec.emit_sector_relative and sector_column not in panel.columns:
        raise DataReadinessError(f"panel is missing {sector_column!r}")
    if panel.empty:
        return panel.copy()

    frame = panel.copy()
    by_time = frame.groupby(timestamp_column, sort=False)
    eligible = by_time[timestamp_column].transform("size") >= spec.minimum_cross_section
    derived: dict[str, pd.Series] = {}

    for name in features:
        values = pd.to_numeric(frame[name], errors="coerce")
        clipped = _winsorize(values, by_time, spec.winsorize_quantile)
        if spec.emit_zscore:
            derived[f"{name}{Z_SUFFIX}"] = _zscore(clipped, by_time).where(
                eligible
            )
        if spec.emit_rank:
            # Centred on zero so the middle of the cross-section is neutral and
            # the sign carries meaning on its own.
            ranked = clipped.groupby(frame[timestamp_column]).rank(pct=True)
            derived[f"{name}{RANK_SUFFIX}"] = (ranked * 2.0 - 1.0).where(
                eligible
            )

    if spec.emit_sector_relative:
        keys = [timestamp_column, sector_column]
        by_sector = frame.groupby(keys, sort=False)
        sector_eligible = (
            by_sector[timestamp_column].transform("size") >= spec.minimum_cross_section
        )
        for name in features:
            values = pd.to_numeric(frame[name], errors="coerce")
            clipped = _winsorize(values, by_sector, spec.winsorize_quantile)
            derived[f"{name}{SECTOR_Z_SUFFIX}"] = _zscore(
                clipped,
                by_sector,
            ).where(
                sector_eligible
            )
    collisions = sorted(set(derived).intersection(frame.columns))
    if collisions:
        raise DataReadinessError(
            f"cross-sectional output columns already exist: {collisions}"
        )
    return pd.concat(
        [frame, pd.DataFrame(derived, index=frame.index)],
        axis=1,
    )


def cross_sectional_feature_names(
    features: Sequence[str],
    *,
    spec: CrossSectionSpec,
) -> list[str]:
    """Columns `add_cross_sectional_features` will produce, in a stable order."""

    names: list[str] = []
    for name in features:
        if spec.emit_zscore:
            names.append(f"{name}{Z_SUFFIX}")
        if spec.emit_rank:
            names.append(f"{name}{RANK_SUFFIX}")
    if spec.emit_sector_relative:
        names.extend(f"{name}{SECTOR_Z_SUFFIX}" for name in features)
    return names


def _winsorize(
    values: pd.Series,
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
    quantile: float,
) -> pd.Series:
    """Clip each group's tails before scaling.

    One stock halving on a single session would otherwise dominate the mean and
    standard deviation, pushing every other stock's score toward zero and
    erasing the day's information.
    """

    if quantile <= 0.0:
        return values
    lower = values.groupby(grouped.ngroup()).transform(
        lambda part: part.quantile(quantile)
    )
    upper = values.groupby(grouped.ngroup()).transform(
        lambda part: part.quantile(1.0 - quantile)
    )
    return values.clip(lower=lower, upper=upper)


def _zscore(
    values: pd.Series,
    grouped: pd.core.groupby.generic.DataFrameGroupBy,
) -> pd.Series:
    group_id = grouped.ngroup()
    mean = values.groupby(group_id).transform("mean")
    # Sample standard deviation, so a group of one yields NaN rather than a
    # division by zero dressed up as a finite score.
    deviation = values.groupby(group_id).transform("std")
    scaled = (values - mean) / deviation.where(deviation > 0)
    return scaled.replace([np.inf, -np.inf], np.nan)
