from __future__ import annotations

import numpy as np
import pandas as pd


MARKET_REGIME_FEATURES = [
    "benchmark_context_return",
    "benchmark_context_abs_return",
    "market_regime_score",
    "market_regime_risk_on",
    "market_regime_neutral",
    "market_regime_risk_off",
    "market_regime_high_volatility",
]


def add_market_regime_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """Add market-regime labels from benchmark context already present in a feature frame.

    The function intentionally uses only same-row benchmark columns. For intraday rows it
    favors SPY/QQQ bar returns; for swing rows it favors SPY daily/weekly context.
    """
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    context_return = _context_return(out)
    abs_return = context_return.abs()
    intraday = _has_any(out, ["qqq_return_6bar", "spy_return_6bar", "qqq_return_3bar", "spy_return_3bar"])
    positive_threshold = 0.0015 if intraday else 0.02
    negative_threshold = -positive_threshold
    high_vol_threshold = 0.004 if intraday else 0.035
    volume_pressure = _numeric(out, "spy_volume_z20").combine_first(_numeric(out, "sector_volume_z20")).fillna(0.0)

    out["benchmark_context_return"] = context_return
    out["benchmark_context_abs_return"] = abs_return
    out["market_regime_score"] = (context_return / positive_threshold).replace([np.inf, -np.inf], np.nan).clip(-3.0, 3.0)
    out["market_regime"] = "neutral"
    out.loc[context_return.ge(positive_threshold), "market_regime"] = "risk_on"
    out.loc[context_return.le(negative_threshold), "market_regime"] = "risk_off"
    out["market_regime_high_volatility"] = (abs_return.ge(high_vol_threshold) | volume_pressure.ge(1.5)).astype(int)
    out["market_regime_risk_on"] = out["market_regime"].eq("risk_on").astype(int)
    out["market_regime_neutral"] = out["market_regime"].eq("neutral").astype(int)
    out["market_regime_risk_off"] = out["market_regime"].eq("risk_off").astype(int)
    return out


def _context_return(frame: pd.DataFrame) -> pd.Series:
    columns = [
        "qqq_return_6bar",
        "spy_return_6bar",
        "qqq_return_3bar",
        "spy_return_3bar",
        "spy_return_5d_past",
        "spy_return_20d_past",
        "sector_return_5d_past",
        "sector_return_20d_past",
        "spy_return_1d",
        "sector_return_1d",
    ]
    pieces = [_numeric(frame, col) for col in columns if col in frame.columns]
    if not pieces:
        return pd.Series(0.0, index=frame.index, dtype="float")
    return pd.concat(pieces, axis=1).mean(axis=1).fillna(0.0)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float")
    return pd.to_numeric(frame[column], errors="coerce")


def _has_any(frame: pd.DataFrame, columns: list[str]) -> bool:
    return any(column in frame.columns for column in columns)
