from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REGULAR_OPEN = "09:30:00"
REGULAR_CLOSE = "16:00:00"


def build_intraday_decision_report(
    *,
    scores: pd.DataFrame,
    one_minute_dir: Path,
    candidates: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge 5m model scores with latest 1m confirmation features."""
    if scores.empty:
        return scores.copy()
    frame = scores.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    probability_col = _probability_col(frame)
    if probability_col:
        frame["entry_model_probability"] = pd.to_numeric(frame[probability_col], errors="coerce")
        frame["entry_model_rank"] = frame["entry_model_probability"].rank(ascending=False, method="first")
    confirmations = []
    for ticker in frame["ticker"].dropna().astype(str).unique():
        path = one_minute_dir / f"{ticker}.parquet"
        confirmations.append(_confirmation_for_path(ticker, path))
    confirmation_frame = pd.DataFrame(confirmations)
    output = frame.merge(confirmation_frame, on="ticker", how="left")
    if candidates is not None and not candidates.empty:
        candidate_cols = [
            col
            for col in [
                "ticker",
                "company",
                "sector",
                "industry",
                "intraday_theme",
                "intraday_candidate_score",
                "abs_change_pct",
                "volume",
                "dollar_volume_m",
            ]
            if col in candidates.columns
        ]
        metadata = candidates[candidate_cols].copy()
        metadata["ticker"] = metadata["ticker"].astype(str).str.upper().str.strip()
        output = output.merge(metadata.drop_duplicates("ticker"), on="ticker", how="left", suffixes=("", "_finviz"))
    output["intraday_decision"] = output.apply(_decision_label, axis=1)
    output["decision_rank_score"] = output.apply(_decision_rank_score, axis=1)
    return output.sort_values(["decision_rank_score", "entry_model_probability"], ascending=[False, False], na_position="last")


def latest_one_minute_confirmation(ticker: str, bars: pd.DataFrame) -> dict[str, Any]:
    if bars.empty:
        return _empty_confirmation(ticker, "missing_1m_bars")
    frame = bars.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    eastern = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["_session_date"] = eastern.dt.date
    frame["_session_time"] = eastern.dt.strftime("%H:%M:%S")
    regular = frame[(frame["_session_time"] >= REGULAR_OPEN) & (frame["_session_time"] <= REGULAR_CLOSE)].copy()
    if regular.empty:
        return _empty_confirmation(ticker, "no_regular_session_1m_bars")
    latest_session = regular["_session_date"].max()
    session = regular[regular["_session_date"].eq(latest_session)].copy()
    if session.empty:
        return _empty_confirmation(ticker, "no_latest_session_1m_bars")
    close = float(session["close"].iloc[-1])
    volume = session["volume"].fillna(0.0)
    typical_price = session[["high", "low", "close"]].mean(axis=1)
    vwap_denominator = float(volume.sum())
    vwap = float((typical_price * volume).sum() / vwap_denominator) if vwap_denominator > 0 else np.nan
    rolling_15_volume = volume.rolling(15, min_periods=1).sum()
    latest_15_volume = float(rolling_15_volume.iloc[-1])
    baseline_15_volume = float(rolling_15_volume.iloc[:-1].median()) if len(rolling_15_volume) > 1 else np.nan
    volume_burst = latest_15_volume / baseline_15_volume if baseline_15_volume and np.isfinite(baseline_15_volume) else np.nan
    opening_range = session[(session["_session_time"] >= REGULAR_OPEN) & (session["_session_time"] < "10:00:00")]
    opening_high = float(opening_range["high"].max()) if not opening_range.empty else np.nan
    opening_low = float(opening_range["low"].min()) if not opening_range.empty else np.nan
    record = {
        "ticker": ticker.upper(),
        "one_minute_status": "ok",
        "one_minute_latest_timestamp": session["timestamp"].iloc[-1].isoformat(),
        "one_minute_session_date": str(latest_session),
        "one_minute_close": close,
        "one_minute_vwap": vwap,
        "one_minute_dist_vwap": close / vwap - 1.0 if np.isfinite(vwap) and vwap else np.nan,
        "one_minute_return_5m": _window_return(session["close"], 5),
        "one_minute_return_15m": _window_return(session["close"], 15),
        "one_minute_return_30m": _window_return(session["close"], 30),
        "one_minute_volume_15m": latest_15_volume,
        "one_minute_volume_burst_15m": volume_burst,
        "opening_range_high": opening_high,
        "opening_range_low": opening_low,
        "above_opening_range": bool(np.isfinite(opening_high) and close > opening_high),
        "below_opening_range": bool(np.isfinite(opening_low) and close < opening_low),
    }
    record["one_minute_confirmation_signal"] = _confirmation_signal(record)
    return record


def _confirmation_for_path(ticker: str, path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_confirmation(ticker, "missing_1m_file")
    return latest_one_minute_confirmation(ticker, pd.read_parquet(path))


def _confirmation_signal(record: dict[str, Any]) -> str:
    dist_vwap = _float(record.get("one_minute_dist_vwap"))
    ret15 = _float(record.get("one_minute_return_15m"))
    burst = _float(record.get("one_minute_volume_burst_15m"))
    if dist_vwap > 0 and ret15 > 0 and burst >= 1.2 and bool(record.get("above_opening_range")):
        return "bullish_breakout_confirmation"
    if dist_vwap > 0 and ret15 > 0 and burst >= 1.0:
        return "bullish_vwap_confirmation"
    if dist_vwap < 0 and ret15 < 0 and burst >= 1.0:
        return "bearish_pressure"
    if dist_vwap > 0 and ret15 >= 0:
        return "constructive_above_vwap"
    return "neutral"


def _decision_label(row: pd.Series) -> str:
    rank = _float(row.get("entry_model_rank"))
    probability = _float(row.get("entry_model_probability"))
    confirmation = str(row.get("one_minute_confirmation_signal", "neutral"))
    if confirmation == "bearish_pressure":
        return "avoid_entry_1m_bearish"
    if rank <= 20 and confirmation in {"bullish_breakout_confirmation", "bullish_vwap_confirmation"}:
        return "entry_watch_confirmed"
    if rank <= 50 and confirmation in {"bullish_breakout_confirmation", "bullish_vwap_confirmation", "constructive_above_vwap"}:
        return "watch_for_entry"
    if rank <= 20 and probability >= 0:
        return "model_positive_wait_for_1m_confirmation"
    return "neutral"


def _decision_rank_score(row: pd.Series) -> float:
    probability = _float(row.get("entry_model_probability"))
    if not np.isfinite(probability):
        probability = 0.0
    confirmation = str(row.get("one_minute_confirmation_signal", "neutral"))
    bonus = {
        "bullish_breakout_confirmation": 0.08,
        "bullish_vwap_confirmation": 0.05,
        "constructive_above_vwap": 0.02,
        "bearish_pressure": -0.10,
    }.get(confirmation, 0.0)
    burst = _float(row.get("one_minute_volume_burst_15m"))
    burst_bonus = min(max(burst - 1.0, 0.0), 3.0) * 0.01 if np.isfinite(burst) else 0.0
    return float(probability + bonus + burst_bonus)


def _probability_col(frame: pd.DataFrame) -> str | None:
    cols = [str(col) for col in frame.columns if str(col).endswith("_probability")]
    preferred = [col for col in cols if "entry_success" in col]
    if preferred:
        return preferred[-1]
    return cols[-1] if cols else None


def _window_return(values: pd.Series, minutes: int) -> float:
    if len(values) <= minutes:
        return np.nan
    current = _float(values.iloc[-1])
    previous = _float(values.iloc[-minutes - 1])
    return current / previous - 1.0 if np.isfinite(current) and np.isfinite(previous) and previous else np.nan


def _empty_confirmation(ticker: str, status: str) -> dict[str, Any]:
    return {
        "ticker": ticker.upper(),
        "one_minute_status": status,
        "one_minute_confirmation_signal": "unavailable",
    }


def _float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return np.nan
    return converted if np.isfinite(converted) else np.nan
