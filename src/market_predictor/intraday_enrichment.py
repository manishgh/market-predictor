from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from market_predictor.intraday_catalysts import INTRADAY_CATALYST_FEATURES, add_intraday_catalyst_features
from market_predictor.market_regime import MARKET_REGIME_FEATURES, add_market_regime_labels


INTRADAY_ENRICHED_FEATURES = [
    "session_minutes_from_open",
    "session_progress",
    "is_opening_30m",
    "is_midday",
    "is_power_hour",
    "session_minute_sin",
    "session_minute_cos",
    "session_vwap",
    "dist_session_vwap",
    "session_vwap_slope_3bar",
    "session_vwap_slope_6bar",
    "opening_range_high",
    "opening_range_low",
    "opening_range_width_pct",
    "dist_opening_range_high",
    "dist_opening_range_low",
    "above_opening_range",
    "below_opening_range",
    "return_1bar",
    "return_3bar",
    "return_6bar",
    "return_12bar",
    "return_acceleration_3v6",
    "volume_burst_20bar",
    "relative_volume_same_minute_20d",
    "ema10_gt_ema20",
    "ema20_gt_ema50",
    "close_gt_ema20",
    "macd_improving",
    "setup_candidate_score",
    "intraday_candidate_score",
    "finviz_abs_change_pct",
    "finviz_dollar_volume_m",
    "theme_semis_ai_hardware",
    "theme_software_ai_data",
    "theme_biotech_healthcare",
    "theme_space_aerospace_mobility",
    "theme_crypto_fintech_high_beta",
    "theme_consumer_high_beta",
    "qqq_return_1bar",
    "qqq_return_3bar",
    "qqq_return_6bar",
    "spy_return_1bar",
    "spy_return_3bar",
    "spy_return_6bar",
    "rel_return_1bar_vs_qqq",
    "rel_return_3bar_vs_qqq",
    "rel_return_6bar_vs_qqq",
    "rel_return_1bar_vs_spy",
    "rel_return_3bar_vs_spy",
    "rel_return_6bar_vs_spy",
    "one_minute_dist_vwap",
    "one_minute_return_5m",
    "one_minute_return_15m",
    "one_minute_return_30m",
    "one_minute_volume_burst_15m",
    *MARKET_REGIME_FEATURES,
    *INTRADAY_CATALYST_FEATURES,
]


def build_enriched_intraday_dataset(
    frame: pd.DataFrame,
    *,
    candidates: pd.DataFrame | None = None,
    one_minute_dir: Path | None = None,
    benchmark_dir: Path | None = None,
    event_dirs: list[Path] | None = None,
    market_context_path: Path | None = None,
    setup_only: bool = True,
    min_setup_score: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["timestamp"] = pd.to_datetime(data.get("timestamp", data.get("date")), errors="coerce", utc=True)
    data = data.dropna(subset=["ticker", "timestamp"]).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    data = add_intraday_technical_features(data)
    data = _merge_candidate_metadata(data, candidates)
    data = _merge_benchmark_features(data, benchmark_dir)
    data = _merge_one_minute_features(data, one_minute_dir)
    data, catalyst_audit = add_intraday_catalyst_features(
        data,
        event_dirs=event_dirs,
        market_context_path=market_context_path,
    )
    data = add_market_regime_labels(data)
    data["is_intraday_setup_candidate"] = data["setup_candidate_score"].ge(min_setup_score)
    before_rows = len(data)
    before_tickers = int(data["ticker"].nunique())
    if setup_only:
        data = data[data["is_intraday_setup_candidate"]].copy()
    audit = _audit(data, before_rows=before_rows, before_tickers=before_tickers, setup_only=setup_only, min_setup_score=min_setup_score)
    if not catalyst_audit.empty:
        audit = audit.merge(catalyst_audit, on="ticker", how="left", suffixes=("", "_catalyst"))
    return data.reset_index(drop=True), audit


def add_intraday_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build point-in-time intraday technical features for training and live parity."""
    if frame.empty:
        return frame.copy()
    data = frame.copy()
    if "timestamp" not in data.columns:
        if "date" not in data.columns:
            raise ValueError("Intraday technical features require timestamp or date.")
        data["timestamp"] = data["date"]
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce", utc=True)
    if data["timestamp"].isna().any():
        raise ValueError("Intraday technical features contain invalid timestamps.")
    if "volume" not in data.columns:
        data["volume"] = np.nan
    data = data.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    return _add_setup_features(_add_session_features(data))


def _add_session_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    eastern = frame["timestamp"].dt.tz_convert("America/New_York")
    frame["_session_date"] = eastern.dt.date
    minute_of_day = eastern.dt.hour * 60 + eastern.dt.minute
    market_open = 9 * 60 + 30
    market_close = 16 * 60
    frame["session_minutes_from_open"] = (minute_of_day - market_open).astype(float)
    frame["session_progress"] = (frame["session_minutes_from_open"] / (market_close - market_open)).clip(0.0, 1.0)
    frame["is_opening_30m"] = frame["session_minutes_from_open"].between(0, 30).astype(int)
    frame["is_midday"] = frame["session_minutes_from_open"].between(120, 270).astype(int)
    frame["is_power_hour"] = frame["session_minutes_from_open"].between(330, 390).astype(int)
    radians = 2.0 * np.pi * frame["session_progress"].fillna(0.0)
    frame["session_minute_sin"] = np.sin(radians)
    frame["session_minute_cos"] = np.cos(radians)
    typical = frame[["high", "low", "close"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    grouped = frame.groupby(["ticker", "_session_date"], sort=False)
    dollar_flow = typical * volume
    cum_dollar = dollar_flow.groupby([frame["ticker"], frame["_session_date"]], sort=False).cumsum()
    cum_volume = volume.groupby([frame["ticker"], frame["_session_date"]], sort=False).cumsum()
    frame["session_vwap"] = (cum_dollar / cum_volume.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    close = pd.to_numeric(frame["close"], errors="coerce")
    frame["dist_session_vwap"] = close / frame["session_vwap"] - 1.0
    frame["session_vwap_slope_3bar"] = grouped["session_vwap"].pct_change(3)
    frame["session_vwap_slope_6bar"] = grouped["session_vwap"].pct_change(6)

    # The opening range must grow bar by bar. A full-window aggregate would leak
    # the 09:35-10:00 highs and lows into the 09:30 feature row.
    opening_mask = frame["session_minutes_from_open"].between(0, 30)
    opening_high = pd.to_numeric(frame["high"], errors="coerce").where(opening_mask)
    opening_low = pd.to_numeric(frame["low"], errors="coerce").where(opening_mask)
    frame["opening_range_high"] = opening_high.groupby(
        [frame["ticker"], frame["_session_date"]], sort=False
    ).cummax()
    frame["opening_range_low"] = opening_low.groupby(
        [frame["ticker"], frame["_session_date"]], sort=False
    ).cummin()
    frame[["opening_range_high", "opening_range_low"]] = grouped[
        ["opening_range_high", "opening_range_low"]
    ].ffill()
    frame["opening_range_width_pct"] = (
        frame["opening_range_high"] - frame["opening_range_low"]
    ) / close.replace(0, np.nan)
    frame["dist_opening_range_high"] = close / frame["opening_range_high"] - 1.0
    frame["dist_opening_range_low"] = close / frame["opening_range_low"] - 1.0
    frame["above_opening_range"] = close.gt(frame["opening_range_high"]).astype(int)
    frame["below_opening_range"] = close.lt(frame["opening_range_low"]).astype(int)
    return frame


def _add_setup_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    grouped = frame.groupby("ticker", sort=False)
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    for bars in [1, 3, 6, 12]:
        frame[f"return_{bars}bar"] = grouped["close"].pct_change(bars)
    frame["return_acceleration_3v6"] = frame["return_3bar"] - (frame["return_6bar"] / 2.0)
    rolling_volume = grouped["volume"].transform(lambda series: pd.to_numeric(series, errors="coerce").rolling(20, min_periods=5).median())
    frame["volume_burst_20bar"] = volume / rolling_volume.replace(0, np.nan)
    eastern = frame["timestamp"].dt.tz_convert("America/New_York")
    minute_slot = eastern.dt.hour * 60 + eastern.dt.minute
    same_minute_baseline = volume.groupby([frame["ticker"], minute_slot], sort=False).transform(
        lambda series: series.shift(1).rolling(20, min_periods=5).median()
    )
    frame["relative_volume_same_minute_20d"] = volume / same_minute_baseline.replace(0, np.nan)
    frame["ema10_gt_ema20"] = (pd.to_numeric(frame.get("ema_10"), errors="coerce") > pd.to_numeric(frame.get("ema_20"), errors="coerce")).astype(int)
    frame["ema20_gt_ema50"] = (pd.to_numeric(frame.get("ema_20"), errors="coerce") > pd.to_numeric(frame.get("ema_50"), errors="coerce")).astype(int)
    frame["close_gt_ema20"] = (close > pd.to_numeric(frame.get("ema_20"), errors="coerce")).astype(int)
    frame["macd_improving"] = (
        pd.to_numeric(frame.get("macd_signal_diff"), errors="coerce") > pd.to_numeric(frame.get("prior_macd_signal_diff"), errors="coerce")
    ).astype(int)
    score = pd.Series(0.0, index=frame.index)
    score += pd.to_numeric(frame["volume_burst_20bar"], errors="coerce").ge(1.3).astype(float)
    volume_z20 = pd.to_numeric(
        frame.get("volume_z20", pd.Series(np.nan, index=frame.index)), errors="coerce"
    )
    score += volume_z20.ge(0.5).astype(float)
    score += frame["close_gt_ema20"].astype(float)
    score += frame["ema10_gt_ema20"].astype(float)
    score += frame["macd_improving"].astype(float)
    score += frame["dist_session_vwap"].gt(0.0).astype(float)
    score += frame["above_opening_range"].astype(float)
    score += pd.to_numeric(frame["return_3bar"], errors="coerce").gt(0.0).astype(float)
    frame["setup_candidate_score"] = score
    return frame


def _merge_candidate_metadata(data: pd.DataFrame, candidates: pd.DataFrame | None) -> pd.DataFrame:
    frame = data.copy()
    if candidates is None or candidates.empty:
        return _add_empty_theme_columns(frame)
    meta = candidates.copy()
    meta["ticker"] = meta["ticker"].astype(str).str.upper().str.strip()
    rename = {
        "abs_change_pct": "finviz_abs_change_pct",
        "dollar_volume_m": "finviz_dollar_volume_m",
    }
    meta = meta.rename(columns=rename)
    keep = [
        col
        for col in [
            "ticker",
            "intraday_theme",
            "intraday_candidate_score",
            "finviz_abs_change_pct",
            "finviz_dollar_volume_m",
        ]
        if col in meta.columns
    ]
    frame = frame.merge(meta[keep].drop_duplicates("ticker"), on="ticker", how="left")
    return _add_theme_columns(frame)


def _add_theme_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    themes = out.get("intraday_theme", pd.Series("", index=out.index)).fillna("").astype(str)
    for theme in [
        "semis_ai_hardware",
        "software_ai_data",
        "biotech_healthcare",
        "space_aerospace_mobility",
        "crypto_fintech_high_beta",
        "consumer_high_beta",
    ]:
        out[f"theme_{theme}"] = themes.eq(theme).astype(int)
    return out


def _add_empty_theme_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["intraday_candidate_score"] = np.nan
    out["finviz_abs_change_pct"] = np.nan
    out["finviz_dollar_volume_m"] = np.nan
    for theme in [
        "semis_ai_hardware",
        "software_ai_data",
        "biotech_healthcare",
        "space_aerospace_mobility",
        "crypto_fintech_high_beta",
        "consumer_high_beta",
    ]:
        out[f"theme_{theme}"] = 0
    return out


def _merge_benchmark_features(data: pd.DataFrame, benchmark_dir: Path | None) -> pd.DataFrame:
    frame = data.copy()
    for symbol in ["QQQ", "SPY"]:
        bench = _benchmark_features(benchmark_dir, symbol)
        prefix = symbol.lower()
        if bench.empty:
            for bars in [1, 3, 6]:
                frame[f"{prefix}_return_{bars}bar"] = np.nan
                frame[f"rel_return_{bars}bar_vs_{prefix}"] = np.nan
            continue
        frame = frame.merge(bench, on="timestamp", how="left")
        for bars in [1, 3, 6]:
            frame[f"rel_return_{bars}bar_vs_{prefix}"] = pd.to_numeric(frame[f"return_{bars}bar"], errors="coerce") - pd.to_numeric(
                frame[f"{prefix}_return_{bars}bar"], errors="coerce"
            )
    return frame


def _benchmark_features(benchmark_dir: Path | None, symbol: str) -> pd.DataFrame:
    if benchmark_dir is None:
        return pd.DataFrame()
    path = benchmark_dir / f"{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prefix = symbol.lower()
    out = pd.DataFrame({"timestamp": frame["timestamp"]})
    for bars in [1, 3, 6]:
        out[f"{prefix}_return_{bars}bar"] = close.pct_change(bars)
    return out


def _merge_one_minute_features(data: pd.DataFrame, one_minute_dir: Path | None) -> pd.DataFrame:
    frame = data.copy()
    if one_minute_dir is None or not one_minute_dir.exists():
        for col in [
            "one_minute_dist_vwap",
            "one_minute_return_5m",
            "one_minute_return_15m",
            "one_minute_return_30m",
            "one_minute_volume_burst_15m",
        ]:
            frame[col] = np.nan
        return frame
    pieces = []
    for ticker, group in frame.groupby("ticker", sort=False):
        path = one_minute_dir / f"{ticker}.parquet"
        if not path.exists():
            pieces.append(group)
            continue
        features = _one_minute_asof_features(path)
        if features.empty:
            pieces.append(group)
            continue
        merged = pd.merge_asof(
            group.sort_values("timestamp"),
            features.sort_values("timestamp"),
            on="timestamp",
            direction="backward",
            tolerance=pd.Timedelta(minutes=4),
        )
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True) if pieces else frame


def _one_minute_asof_features(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if frame.empty:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp")
    for col in ["high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    eastern = frame["timestamp"].dt.tz_convert("America/New_York")
    session = eastern.dt.date
    typical = frame[["high", "low", "close"]].mean(axis=1)
    volume = frame["volume"].fillna(0.0)
    cum_dollar = (typical * volume).groupby(session, sort=False).cumsum()
    cum_volume = volume.groupby(session, sort=False).cumsum()
    vwap = cum_dollar / cum_volume.replace(0, np.nan)
    out = pd.DataFrame({"timestamp": frame["timestamp"]})
    out["one_minute_dist_vwap"] = frame["close"] / vwap - 1.0
    for minutes in [5, 15, 30]:
        out[f"one_minute_return_{minutes}m"] = frame["close"].pct_change(minutes)
    rolling_15 = volume.rolling(15, min_periods=5).sum()
    baseline = rolling_15.rolling(120, min_periods=15).median()
    out["one_minute_volume_burst_15m"] = rolling_15 / baseline.replace(0, np.nan)
    return out


def _audit(data: pd.DataFrame, *, before_rows: int, before_tickers: int, setup_only: bool, min_setup_score: float) -> pd.DataFrame:
    grouped = data.groupby("ticker").agg(
        rows=("ticker", "count"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        setup_score_mean=("setup_candidate_score", "mean"),
        target_entry_success_rate=("target_entry_success_12b", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["before_rows"] = before_rows
    grouped["before_tickers"] = before_tickers
    grouped["setup_only"] = setup_only
    grouped["min_setup_score"] = min_setup_score
    return grouped
