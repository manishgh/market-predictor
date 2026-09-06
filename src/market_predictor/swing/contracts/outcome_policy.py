from __future__ import annotations

from market_predictor.label_policy import policy_sha256
from market_predictor.modeling.strategy_contract import SwingContract


def swing_outcome_policy(contract: SwingContract) -> dict[str, object]:
    """Return the complete policy needed to reproduce a served swing outcome."""

    return {
        "policy": "market_predictor.swing_outcome_policy.v1",
        "horizon_sessions": contract.horizon_sessions,
        "entry_reference": contract.entry_reference,
        "exit_rule": contract.exit_rule,
        "decision_cutoff": contract.decision_cutoff,
        "same_bar_barrier_resolution": contract.same_bar_barrier_resolution,
        "target_atr_multiple": contract.target_atr_multiple,
        "stop_atr_multiple": contract.stop_atr_multiple,
        "atr_timeframe": contract.atr_timeframe,
        "atr_lookback_bars": contract.atr_lookback_bars,
        "round_trip_cost_bps": contract.round_trip_cost_bps,
        "broad_benchmark": "SPY",
        "growth_benchmark": "QQQ",
        "sector_benchmark": "point_in_time_sector_etf",
    }


def swing_outcome_policy_sha256(contract: SwingContract) -> str:
    return policy_sha256(swing_outcome_policy(contract))
