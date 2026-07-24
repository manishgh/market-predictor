"""Versioned, conservative execution policy.

Turns idealized label outcomes into executable economics: gap-through fills,
a liquidity/price/volatility-aware round-trip cost, participation caps, a cost
stress grid, and capacity curves. The policy is immutable and content-addressed
so its identity can be bound into promotion evidence alongside the prediction
policy.

Calibration honesty: the coefficient defaults are deliberately conservative
placeholders. Real spread/quote/impact calibration requires market microstructure
data that is not yet available, so downstream promotion treats capacity/liquidity
evidence as conservative — never as a validated fill model. The *structure*
(bucketing, stress, participation, capacity) is what R2 delivers; the calibrated
numbers are environment_pending.

Units are unambiguous by construction: ``price`` is USD, ``atr_pct`` is a return
fraction (0.02 == 2%), ``participation`` is order-notional / bar-dollar-volume,
and ``dollar_volume`` is USD.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

EXECUTION_POLICY_ID = "market_predictor.execution_policy.v1"


class ExecutionCostPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    commission_bps: float = Field(default=1.0, ge=0)
    base_half_spread_bps: float = Field(default=3.0, ge=0)
    low_price_half_spread_coef_bps: float = Field(default=2.0, ge=0)
    low_price_reference_usd: float = Field(default=10.0, gt=0)
    slippage_fraction_of_atr: float = Field(default=0.05, ge=0)
    impact_bps_at_participation_cap: float = Field(default=40.0, ge=0)
    participation_cap: float = Field(default=0.05, gt=0, le=1)
    min_fillable_dollar_volume: float = Field(default=50_000.0, ge=0)
    stress_multipliers: tuple[float, ...] = (1.0, 1.5, 2.0)
    capacity_capital_usd: tuple[float, ...] = (
        100_000.0,
        1_000_000.0,
        5_000_000.0,
        20_000_000.0,
    )

    def spec(self) -> dict[str, Any]:
        return {"policy_id": EXECUTION_POLICY_ID, **self.model_dump()}

    def sha256(self) -> str:
        return hashlib.sha256(
            json.dumps(self.spec(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


DEFAULT_EXECUTION_POLICY = ExecutionCostPolicy()
EXECUTION_POLICY_SHA256 = DEFAULT_EXECUTION_POLICY.sha256()


def execution_policy_identity(policy: ExecutionCostPolicy | None = None) -> dict[str, str]:
    policy = policy or DEFAULT_EXECUTION_POLICY
    return {
        "execution_policy_id": EXECUTION_POLICY_ID,
        "execution_policy_sha256": policy.sha256(),
    }


# --------------------------------------------------------------------------- #
# Executable fills (gap-through)
# --------------------------------------------------------------------------- #
def executable_fill_prices(
    *,
    outcome: np.ndarray,
    target_price: np.ndarray,
    stop_price: np.ndarray,
    trigger_open: np.ndarray,
    final_price: np.ndarray,
) -> np.ndarray:
    """Conservative barrier fills for a long position.

    - ``stop_first``: fill at the worse of the barrier and the triggering bar's
      open. A bar that opens through the stop cannot fill at the stop; it fills
      at the (lower) open.
    - ``target_first``: fill at the target only. A favorable gap above the target
      is not credited (a resting limit is assumed, never a better market fill).
    - otherwise (timeout): the final path close.
    """

    fill = np.asarray(final_price, dtype=float).copy()
    is_stop = outcome == "stop_first"
    is_target = outcome == "target_first"
    fill[is_stop] = np.minimum(np.asarray(trigger_open, dtype=float)[is_stop], np.asarray(stop_price, dtype=float)[is_stop])
    fill[is_target] = np.asarray(target_price, dtype=float)[is_target]
    return fill


def executable_fill_price(
    *,
    outcome: str,
    target_price: float,
    stop_price: float,
    trigger_open: float,
    final_price: float,
) -> float:
    """Scalar form of :func:`executable_fill_prices` (one trade)."""

    if outcome == "stop_first":
        return float(min(trigger_open, stop_price))
    if outcome == "target_first":
        return float(target_price)
    return float(final_price)


# --------------------------------------------------------------------------- #
# Round-trip cost model (bucketed by liquidity / price / volatility)
# --------------------------------------------------------------------------- #
def round_trip_cost_bps(
    *,
    price: float,
    atr_pct: float,
    participation: float = 0.0,
    policy: ExecutionCostPolicy | None = None,
    stress: float = 1.0,
) -> float:
    policy = policy or DEFAULT_EXECUTION_POLICY
    safe_price = max(float(price), 1e-6)
    half_spread = policy.base_half_spread_bps + policy.low_price_half_spread_coef_bps * max(
        policy.low_price_reference_usd / safe_price - 1.0, 0.0
    )
    slippage = policy.slippage_fraction_of_atr * max(float(atr_pct), 0.0) * 10_000.0
    capped = min(max(float(participation), 0.0), policy.participation_cap)
    impact = policy.impact_bps_at_participation_cap * (capped / policy.participation_cap)
    return float(stress) * (policy.commission_bps + 2.0 * half_spread + slippage + impact)


def round_trip_cost_fraction(
    price: pd.Series,
    atr_pct: pd.Series,
    *,
    policy: ExecutionCostPolicy | None = None,
    participation: pd.Series | None = None,
    stress: float = 1.0,
) -> pd.Series:
    """Vectorized round-trip cost as a return fraction, aligned to ``price.index``."""

    policy = policy or DEFAULT_EXECUTION_POLICY
    safe_price = pd.to_numeric(price, errors="coerce").clip(lower=1e-6)
    atr = pd.to_numeric(atr_pct, errors="coerce").clip(lower=0.0).fillna(0.0)
    half_spread = policy.base_half_spread_bps + policy.low_price_half_spread_coef_bps * (
        policy.low_price_reference_usd / safe_price - 1.0
    ).clip(lower=0.0)
    slippage = policy.slippage_fraction_of_atr * atr * 10_000.0
    if participation is None:
        impact: pd.Series | float = 0.0
    else:
        capped = pd.to_numeric(participation, errors="coerce").clip(lower=0.0, upper=policy.participation_cap).fillna(0.0)
        impact = policy.impact_bps_at_participation_cap * (capped / policy.participation_cap)
    bps = stress * (policy.commission_bps + 2.0 * half_spread + slippage + impact)
    return (bps / 10_000.0).astype(float)


def participation_fraction(notional: pd.Series | float, dollar_volume: pd.Series) -> pd.Series:
    volume = pd.to_numeric(dollar_volume, errors="coerce")
    fraction = notional / volume
    return fraction.where(volume > 0, other=np.inf)


def execution_cost_fraction(
    selected: pd.DataFrame,
    *,
    price_column: str,
    atr_pct_column: str,
    policy: ExecutionCostPolicy | None = None,
    stress: float = 1.0,
) -> pd.Series | None:
    """Per-trade round-trip cost fraction, or ``None`` when inputs are absent.

    Returns ``None`` if the price/volatility columns are missing so callers can
    fall back to the frozen flat label cost for lean fixtures.
    """

    if price_column not in selected.columns or atr_pct_column not in selected.columns:
        return None
    return round_trip_cost_fraction(
        selected[price_column],
        selected[atr_pct_column],
        policy=policy,
        stress=stress,
    )


def flat_stress_surcharge(stress: float, policy: ExecutionCostPolicy | None = None) -> float:
    """Extra cost fraction applied to a flat-cost fallback under a stress multiplier."""

    policy = policy or DEFAULT_EXECUTION_POLICY
    if stress <= 1.0:
        return 0.0
    return (stress - 1.0) * (policy.commission_bps + 2.0 * policy.base_half_spread_bps) / 10_000.0


STRESS_ECONOMIC_FIELDS = {
    "avg_trade_return": "stress_avg_trade_return",
    "avg_excess_return_vs_spy": "stress_avg_excess_return_vs_spy",
    "cumulative_return": "stress_cumulative_return",
    "profit_factor": "stress_profit_factor",
    "max_drawdown": "stress_max_drawdown",
}


def merge_stress_summary(
    base_conservative: pd.DataFrame,
    stress_conservative: pd.DataFrame,
    *,
    multiplier: float,
    fields: dict[str, str],
) -> pd.DataFrame:
    """Attach stressed economics onto the base conservative row as ``stress_*`` columns."""

    out = base_conservative.copy()
    stress_row = stress_conservative.iloc[0]
    anchor = out.index[0]
    out.loc[anchor, "cost_stress_multiplier"] = float(multiplier)
    for source, target in fields.items():
        out.loc[anchor, target] = stress_row.get(source)
    return out


# --------------------------------------------------------------------------- #
# Capacity curve
# --------------------------------------------------------------------------- #
def capacity_curve(
    selected: pd.DataFrame,
    *,
    gross_return_column: str,
    dollar_volume_column: str,
    price_column: str,
    atr_pct_column: str,
    capital_weight: float,
    policy: ExecutionCostPolicy | None = None,
) -> pd.DataFrame:
    """Net economics per capital level under participation-scaled costs.

    ``capital_weight`` is the fraction of deployed capital allocated to each
    selected trade (e.g. ``1 / top_k``). For every capital level the order
    notional is ``capital * capital_weight``, participation is that notional over
    the entry bar dollar volume, and trades whose participation would exceed the
    cap or whose bar is below the minimum fillable dollar volume are dropped as
    no-fill. The curve exposes how realized net return decays with size.
    """

    policy = policy or DEFAULT_EXECUTION_POLICY

    def _series(column: str) -> pd.Series:
        if column in selected.columns:
            return pd.to_numeric(selected[column], errors="coerce")
        return pd.Series(np.nan, index=selected.index, dtype=float)

    gross = _series(gross_return_column)
    dollar_volume = _series(dollar_volume_column)
    price = _series(price_column)
    atr = _series(atr_pct_column)
    complete_liquidity = (
        gross.notna()
        & dollar_volume.notna()
        & dollar_volume.gt(0)
        & price.notna()
        & atr.notna()
    )
    liquidity_evidence_complete = bool(total := len(selected)) and bool(
        complete_liquidity.all()
    )
    records: list[dict[str, object]] = []
    for capital in policy.capacity_capital_usd:
        notional = float(capital) * capital_weight
        participation = participation_fraction(notional, dollar_volume)
        fillable = (
            dollar_volume.ge(policy.min_fillable_dollar_volume)
            & participation.le(policy.participation_cap)
            & complete_liquidity
        )
        filled = int(fillable.sum())
        if filled == 0:
            records.append(
                {
                    "capital_usd": float(capital),
                    "capital_per_trade_usd": notional,
                    "selected_trades": total,
                    "filled_trades": 0,
                    "no_fill_rate": 1.0 if total else float("nan"),
                    "avg_participation": float("nan"),
                    "avg_net_return": float("nan"),
                    "liquidity_evidence_complete": liquidity_evidence_complete,
                }
            )
            continue
        cost = round_trip_cost_fraction(
            price[fillable],
            atr[fillable],
            policy=policy,
            participation=participation[fillable],
        )
        net = gross[fillable] - cost
        records.append(
            {
                "capital_usd": float(capital),
                "capital_per_trade_usd": notional,
                "selected_trades": total,
                "filled_trades": filled,
                "no_fill_rate": float(1.0 - filled / total) if total else float("nan"),
                "avg_participation": float(participation[fillable].mean()),
                "avg_net_return": float(net.mean()),
                "liquidity_evidence_complete": liquidity_evidence_complete,
            }
        )
    return pd.DataFrame(records)
