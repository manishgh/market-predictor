"""Deterministic, auditable portfolio selection for swing candidates."""

from __future__ import annotations

import math

import pandas as pd

from market_predictor.v3.errors import DataReadinessError

EFFECTIVE_SECTOR_WEIGHT_COLUMN = "__effective_sector_weight_limit"
AVAILABLE_SECTOR_COUNT_COLUMN = "__available_sector_count"


def select_constrained_swing_portfolio(
    candidates: pd.DataFrame,
    *,
    maximum_trades: int,
    target_maximum_sector_weight: float,
    hard_maximum_sector_weight: float,
    minimum_distinct_sectors: int,
) -> pd.DataFrame:
    """Select highest probabilities under the approved adaptive sector policy.

    The target limit remains 20% when five or more sectors are available. A
    four- or three-sector candidate set uses its mathematically necessary 25%
    or 33.3% limit, never exceeding the governed hard maximum.
    """

    required = {
        "decision_group_id",
        "decision_time_utc",
        "security_id",
        "sector",
        "__probability",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise DataReadinessError(
            f"swing selection candidates are missing columns: {missing}"
        )
    if maximum_trades < 1 or minimum_distinct_sectors < 3:
        raise ValueError("swing selection limits are invalid")
    if not 0.0 < target_maximum_sector_weight <= hard_maximum_sector_weight <= 1.0:
        raise ValueError("swing sector-weight limits are invalid")
    if hard_maximum_sector_weight + 1e-12 < 1.0 / minimum_distinct_sectors:
        raise ValueError("hard sector limit cannot support the required sectors")
    if candidates.empty:
        return _empty_selection(candidates)
    if candidates["sector"].isna().any() or candidates["security_id"].isna().any():
        raise DataReadinessError("swing selection identity contains null values")

    selected: list[pd.DataFrame] = []
    for _, group in candidates.groupby(
        "decision_group_id",
        sort=False,
        observed=True,
    ):
        ordered = group.sort_values(
            ["__probability", "security_id"],
            ascending=[False, True],
            kind="stable",
        )
        available_sectors = int(ordered["sector"].astype(str).nunique())
        if available_sectors < minimum_distinct_sectors:
            continue
        effective_limit = max(
            target_maximum_sector_weight,
            1.0 / available_sectors,
        )
        if effective_limit > hard_maximum_sector_weight + 1e-12:
            continue
        target_max = min(maximum_trades, len(ordered))
        chosen: pd.DataFrame | None = None
        for target in range(target_max, minimum_distinct_sectors - 1, -1):
            sector_cap = math.floor(target * effective_limit + 1e-12)
            if sector_cap < 1:
                continue
            counts: dict[str, int] = {}
            indices: list[object] = []
            for index, row in ordered.iterrows():
                sector = str(row["sector"])
                if counts.get(sector, 0) >= sector_cap:
                    continue
                counts[sector] = counts.get(sector, 0) + 1
                indices.append(index)
                if len(indices) == target:
                    break
            if len(indices) != target or len(counts) < minimum_distinct_sectors:
                continue
            candidate = ordered.loc[indices].copy()
            weights = candidate.groupby("sector", observed=True).size() / len(candidate)
            if float(weights.max()) <= effective_limit + 1e-12:
                candidate[EFFECTIVE_SECTOR_WEIGHT_COLUMN] = effective_limit
                candidate[AVAILABLE_SECTOR_COUNT_COLUMN] = available_sectors
                chosen = candidate
                break
        if chosen is not None:
            selected.append(chosen)
    if not selected:
        return _empty_selection(candidates)
    return pd.concat(selected, ignore_index=True).sort_values(
        ["decision_time_utc", "decision_group_id", "security_id"],
        kind="stable",
    )


def _empty_selection(candidates: pd.DataFrame) -> pd.DataFrame:
    empty = candidates.iloc[0:0].copy()
    empty[EFFECTIVE_SECTOR_WEIGHT_COLUMN] = pd.Series(dtype="float64")
    empty[AVAILABLE_SECTOR_COUNT_COLUMN] = pd.Series(dtype="int64")
    return empty
