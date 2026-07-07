from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlertConfig:
    min_score: float = 2.0
    volume_confirm_z: float = 0.75
    strong_volume_z: float = 1.5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    breakout_window: int = 20


def prepare_alert_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Add alert-only indicators without assuming the training schema has them."""
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    if "ticker" not in data.columns:
        data["ticker"] = ""
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    data = data.sort_values(["ticker", "date"], na_position="last").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume", "rsi_14", "macd_signal_diff", "volume_z20"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    if "close" not in data.columns:
        return data
    grouped = data.groupby("ticker", sort=False)
    for span in [10, 20, 50]:
        column = f"ema_{span}"
        if column not in data.columns:
            data[column] = grouped["close"].transform(lambda series, span=span: series.ewm(span=span, adjust=False).mean())
        data[f"dist_ema_{span}"] = data["close"] / data[column] - 1.0
    if "macd_signal_diff" not in data.columns:
        ema12 = grouped["close"].transform(lambda series: series.ewm(span=12, adjust=False).mean())
        ema26 = grouped["close"].transform(lambda series: series.ewm(span=26, adjust=False).mean())
        macd = ema12 - ema26
        signal = macd.groupby(data["ticker"], sort=False).transform(lambda series: series.ewm(span=9, adjust=False).mean())
        data["macd_signal_diff"] = macd - signal
    prior_close = grouped["close"].shift(1)
    data["prior_close"] = prior_close
    data["prior_ema_20"] = grouped["ema_20"].shift(1)
    data["prior_ema_50"] = grouped["ema_50"].shift(1)
    data["prior_macd_signal_diff"] = grouped["macd_signal_diff"].shift(1)
    if "rsi_14" in data.columns:
        data["prior_rsi_14"] = grouped["rsi_14"].shift(1)
    data["prior_20d_high"] = grouped["close"].transform(lambda series: series.shift(1).rolling(20, min_periods=10).max())
    data["prior_20d_low"] = grouped["close"].transform(lambda series: series.shift(1).rolling(20, min_periods=10).min())
    return data


def generate_indicator_alerts(frame: pd.DataFrame, config: AlertConfig | None = None, *, latest_only: bool = False) -> pd.DataFrame:
    config = config or AlertConfig()
    data = prepare_alert_indicators(frame)
    if data.empty:
        return pd.DataFrame(columns=_alert_columns())
    rows: list[dict[str, object]] = []
    source = data.groupby("ticker", sort=False).tail(1) if latest_only else data
    for _, row in source.iterrows():
        rows.extend(_alerts_for_row(row, config))
    alerts = pd.DataFrame(rows, columns=_alert_columns())
    if alerts.empty:
        return alerts
    alerts = alerts[alerts["score"] >= config.min_score].copy()
    return alerts.sort_values(["date", "ticker", "score"], ascending=[True, True, False]).reset_index(drop=True)


def backtest_indicator_alerts(
    dataset: pd.DataFrame,
    *,
    horizon_days: int,
    config: AlertConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or AlertConfig()
    future_col = f"future_return_{horizon_days}d"
    target_col = f"target_up_{horizon_days}d"
    alerts = generate_indicator_alerts(dataset, config=config, latest_only=False)
    if alerts.empty:
        return alerts, pd.DataFrame(columns=_summary_columns())
    labels = dataset[["ticker", "date", future_col, target_col]].copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce").dt.date
    labels[future_col] = pd.to_numeric(labels[future_col], errors="coerce")
    labels[target_col] = pd.to_numeric(labels[target_col], errors="coerce")
    scored = alerts.merge(labels, on=["ticker", "date"], how="left")
    scored["is_up"] = scored[target_col] == 1
    scored["direction_win"] = np.where(
        scored["direction"].eq("up"),
        scored[future_col] > 0,
        scored[future_col] < 0,
    )
    scored = scored.dropna(subset=[future_col])
    summary_rows = []
    for keys, group in scored.groupby(["alert_type", "direction"], dropna=False):
        alert_type, direction = keys
        summary_rows.append(_summary_row(str(alert_type), str(direction), group, future_col))
    summary_rows.append(_summary_row("ALL", "all", scored, future_col))
    summary = pd.DataFrame(summary_rows, columns=_summary_columns()).sort_values(
        ["direction_win_rate", "count"], ascending=[False, False]
    )
    return scored.reset_index(drop=True), summary.reset_index(drop=True)


def write_alert_outputs(alerts: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(out, index=False)
    return out


def _alerts_for_row(row: pd.Series, config: AlertConfig) -> list[dict[str, object]]:
    alerts: list[dict[str, object]] = []
    ticker = str(row.get("ticker", "") or "").upper()
    date = row.get("date")
    close = _num(row.get("close"))
    volume_z = _num(row.get("volume_z20"))
    rsi = _num(row.get("rsi_14"))
    prior_rsi = _num(row.get("prior_rsi_14"))
    macd = _num(row.get("macd_signal_diff"))
    prior_macd = _num(row.get("prior_macd_signal_diff"))
    ema20 = _num(row.get("ema_20"))
    ema50 = _num(row.get("ema_50"))
    prior_close = _num(row.get("prior_close"))
    prior_ema20 = _num(row.get("prior_ema_20"))
    prior_ema50 = _num(row.get("prior_ema_50"))
    prior_high = _num(row.get("prior_20d_high"))
    prior_low = _num(row.get("prior_20d_low"))

    def add(alert_type: str, direction: str, base: float, detail: str) -> None:
        score = base + _volume_bonus(volume_z, config)
        alerts.append(
            {
                "ticker": ticker,
                "date": date,
                "alert_type": alert_type,
                "direction": direction,
                "severity": _severity(score),
                "score": round(float(score), 4),
                "close": close,
                "volume_z20": volume_z,
                "rsi_14": rsi,
                "macd_signal_diff": macd,
                "details": detail,
            }
        )

    if np.isfinite(prior_macd) and np.isfinite(macd):
        if prior_macd <= 0 < macd:
            add("macd_bullish_cross", "up", 2.0, "MACD histogram crossed above signal baseline.")
        if prior_macd >= 0 > macd:
            add("macd_bearish_cross", "down", 2.0, "MACD histogram crossed below signal baseline.")
    if all(np.isfinite(value) for value in [prior_close, prior_ema20, close, ema20]):
        if prior_close <= prior_ema20 and close > ema20:
            add("ema20_bullish_reclaim", "up", 2.2, "Close reclaimed EMA20.")
        if prior_close >= prior_ema20 and close < ema20:
            add("ema20_bearish_loss", "down", 2.2, "Close lost EMA20.")
    if all(np.isfinite(value) for value in [prior_ema20, prior_ema50, ema20, ema50]):
        if prior_ema20 <= prior_ema50 and ema20 > ema50:
            add("ema20_ema50_bullish_cross", "up", 2.6, "EMA20 crossed above EMA50.")
        if prior_ema20 >= prior_ema50 and ema20 < ema50:
            add("ema20_ema50_bearish_cross", "down", 2.6, "EMA20 crossed below EMA50.")
    if np.isfinite(prior_rsi) and np.isfinite(rsi):
        if prior_rsi < config.rsi_oversold <= rsi:
            add("rsi_oversold_rebound", "up", 1.8, "RSI rebounded out of oversold.")
        if prior_rsi > config.rsi_overbought >= rsi:
            add("rsi_overbought_rollover", "down", 1.8, "RSI rolled over from overbought.")
    if all(np.isfinite(value) for value in [close, prior_high, volume_z]) and close > prior_high and volume_z >= config.volume_confirm_z:
        add("volume_confirmed_breakout", "up", 2.5, "Close broke above prior 20-day high with volume confirmation.")
    if all(np.isfinite(value) for value in [close, prior_low, volume_z]) and close < prior_low and volume_z >= config.volume_confirm_z:
        add("volume_confirmed_breakdown", "down", 2.5, "Close broke below prior 20-day low with volume confirmation.")
    return alerts


def _summary_row(alert_type: str, direction: str, group: pd.DataFrame, future_col: str) -> dict[str, object]:
    return {
        "alert_type": alert_type,
        "direction": direction,
        "count": int(len(group)),
        "direction_win_rate": float(group["direction_win"].mean()) if len(group) else np.nan,
        "avg_future_return": float(group[future_col].mean()) if len(group) else np.nan,
        "median_future_return": float(group[future_col].median()) if len(group) else np.nan,
        "avg_score": float(group["score"].mean()) if len(group) else np.nan,
        "high_severity_count": int(group["severity"].eq("high").sum()) if len(group) else 0,
    }


def _volume_bonus(volume_z: float, config: AlertConfig) -> float:
    if not np.isfinite(volume_z):
        return 0.0
    if volume_z >= config.strong_volume_z:
        return 1.0
    if volume_z >= config.volume_confirm_z:
        return 0.5
    if volume_z <= -1.0:
        return -0.2
    return 0.0


def _severity(score: float) -> str:
    if score >= 3.5:
        return "high"
    if score >= 2.5:
        return "medium"
    return "low"


def _num(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _alert_columns() -> list[str]:
    return [
        "ticker",
        "date",
        "alert_type",
        "direction",
        "severity",
        "score",
        "close",
        "volume_z20",
        "rsi_14",
        "macd_signal_diff",
        "details",
    ]


def _summary_columns() -> list[str]:
    return [
        "alert_type",
        "direction",
        "count",
        "direction_win_rate",
        "avg_future_return",
        "median_future_return",
        "avg_score",
        "high_severity_count",
    ]
