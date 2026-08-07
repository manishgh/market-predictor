"""Trading policy functions for determining actionable signals from raw predictions."""

from typing import Any

from market_predictor.prediction_contracts import IntradayPrediction, SwingPrediction
from market_predictor.prediction_policy import (
    INTRADAY_WATCH,
    INTRADAY_WATCH_MAX_DOWNSIDE,
    SWING_LOW,
    SWING_STRONG,
    SWING_WATCH,
    intraday_action,
)
from market_predictor.readiness import INVALID, VALID, WARN


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        val = float(value)
        if val != val:
            return None
        return val
    except (ValueError, TypeError):
        return None

def combined_readiness(
    swing: SwingPrediction | None,
    intraday: IntradayPrediction | None,
) -> str:
    statuses = [row.readiness.status for row in [swing, intraday] if row is not None]
    if not statuses:
        return INVALID
    if INVALID in statuses:
        return INVALID
    if WARN in statuses:
        return WARN
    return VALID

def determine_final_signal(swing: SwingPrediction | None, intraday: IntradayPrediction | None) -> str:
    if swing is None and intraday is None:
        return "not_ready"
    if swing is not None and swing.readiness.status != VALID:
        return "not_ready"
    if intraday is not None and intraday.readiness.status != VALID:
        return "not_ready"
    
    swing_prob = swing.probability if swing else None
    intra_prob = intraday.opportunity_probability if intraday else None
    intra_downside = intraday.downside_probability if intraday else None
    
    intraday_supports_entry = intra_prob is None or (
        intra_prob >= INTRADAY_WATCH
        and (intra_downside is None or intra_downside <= INTRADAY_WATCH_MAX_DOWNSIDE)
    )
    
    if swing_prob is not None and swing_prob >= SWING_STRONG and intraday_supports_entry:
        return "high_conviction_watch"
    if swing_prob is not None and swing_prob >= SWING_WATCH and intraday_supports_entry:
        return "watch_for_entry"
    if (
        intra_prob is not None
        and intra_prob >= SWING_STRONG
        and (intra_downside is None or intra_downside <= SWING_LOW)
        and swing_prob is None
    ):
        return "intraday_watch"
    if swing_prob is not None and swing_prob >= SWING_WATCH and intra_prob is not None and intra_prob < 0.50:
        return "swing_positive_wait_for_intraday"
    
    return "neutral"

def determine_intraday_signal(
    opportunity_probability: Any,
    downside_probability: Any,
) -> str:
    return intraday_action(
        _float_or_none(opportunity_probability),
        _float_or_none(downside_probability),
    )
