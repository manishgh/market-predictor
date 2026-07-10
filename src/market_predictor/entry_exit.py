from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.features import add_price_features
from market_predictor.intraday_catalysts import INTRADAY_CATALYST_FEATURES
from market_predictor.market_regime import MARKET_REGIME_FEATURES, add_market_regime_labels
from market_predictor.model import DEFAULT_FEATURES, DateGroupedPurgedWalkForwardSplit
from market_predictor.registry import write_model_manifest
from market_predictor.volatile import VOLATILE_EXTRA_FEATURES


ENTRY_EXIT_SCHEMA_VERSION = "entry_exit.v1"

ENTRY_EXIT_EXTRA_FEATURES = [
    "ema_10",
    "ema_20",
    "ema_50",
    "dist_ema_10",
    "dist_ema_20",
    "dist_ema_50",
    "prior_close",
    "prior_ema_20",
    "prior_ema_50",
    "prior_macd_signal_diff",
    "prior_rsi_14",
    "prior_20d_high",
    "prior_20d_low",
    "setup_risk_reward",
    "setup_stop_pct",
    "setup_target_pct",
    "entry_horizon_bars",
    "session_minutes_from_open",
    "session_progress",
    "is_opening_30m",
    "is_midday",
    "is_power_hour",
    "session_minute_sin",
    "session_minute_cos",
    "dist_session_vwap",
    "dist_opening_range_high",
    "dist_opening_range_low",
    "above_opening_range",
    "below_opening_range",
    "return_1bar",
    "return_3bar",
    "return_6bar",
    "return_12bar",
    "volume_burst_20bar",
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

ENTRY_EXIT_FEATURES = list(dict.fromkeys([*DEFAULT_FEATURES, *VOLATILE_EXTRA_FEATURES, *ENTRY_EXIT_EXTRA_FEATURES]))

ENTRY_EXIT_FEATURE_SETS = {"all", "technical", "catalyst"}

CATALYST_FEATURE_NAMES = set(INTRADAY_CATALYST_FEATURES)
CATALYST_FEATURE_NAMES.update(
    {
        "news_count",
        "news_count_z30",
        "has_news",
        "sentiment_mean",
        "sentiment_min",
        "sentiment_max",
        "sentiment_pos_frac",
        "sentiment_neg_frac",
        "sentiment_momentum_5d",
        "market_context_news_count",
        "market_context_sentiment_mean",
        "market_context_sentiment_min",
        "market_context_sentiment_max",
        "market_context_sentiment_neg_frac",
        "market_context_sentiment_pos_frac",
        "market_context_news_count_z30",
        "market_context_sentiment_momentum_5d",
        "source_count_alpaca",
        "source_count_reddit",
        "source_count_seeking_alpha",
        "source_count_sec",
        "source_count_finviz",
        "reddit_mentions",
        "reddit_velocity_7d",
        "reddit_newly_trending",
        "reddit_sentiment_mean",
        "reddit_score_sum",
        "reddit_comments_sum",
        "reddit_upvote_ratio_mean",
        "event_count",
        "event_reaction_2h_mean",
        "event_reaction_2h_abs_max",
        "event_reaction_volume_sum",
        "premarket_gap_mean",
        "premarket_day_return_mean",
        "intraday_reaction_2h_mean",
        "intraday_to_close_mean",
        "afterhours_next_open_gap_mean",
        "afterhours_next_day_return_mean",
        "buzz_spike_x_volume_z",
        "sentiment_x_news_attention",
        "earnings_x_eps_surprise",
        "catalyst_x_volume_z",
        "reaction_x_sentiment",
        "premarket_gap_x_sentiment",
    }
)
CATALYST_FEATURE_PREFIXES = (
    "event_",
    "source_count_",
    "reddit_",
    "market_context_",
    "news_count_",
    "sentiment_",
)


@dataclass(frozen=True)
class EntryExitLabelConfig:
    horizon_bars: int = 5
    take_profit_atr: float = 1.5
    stop_loss_atr: float = 1.0
    min_rows_per_ticker: int = 120
    min_labeled_rows_per_ticker: int = 40
    ambiguous_policy: str = "stop"
    bar_kind: str = "swing"
    allow_overnight: bool = False


def build_entry_exit_dataset(
    frame: pd.DataFrame,
    *,
    config: EntryExitLabelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or EntryExitLabelConfig()
    if config.horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1.")
    if config.take_profit_atr <= 0 or config.stop_loss_atr <= 0:
        raise ValueError("ATR target/stop multipliers must be positive.")
    if config.ambiguous_policy not in {"stop", "target", "ignore"}:
        raise ValueError("ambiguous_policy must be one of: stop, target, ignore.")
    data = frame.copy()
    if "ticker" not in data.columns:
        if "symbol" in data.columns:
            data = data.rename(columns={"symbol": "ticker"})
        else:
            raise ValueError("Entry/exit dataset requires a ticker or symbol column.")
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    if "date" not in data.columns:
        if "timestamp" in data.columns:
            data["date"] = data["timestamp"]
        else:
            raise ValueError("Entry/exit dataset requires a date or timestamp column.")
    data["_mp_timestamp"] = pd.to_datetime(data["date"], errors="coerce", utc=True)
    if data["_mp_timestamp"].isna().any():
        raise ValueError("Entry/exit dataset contains invalid date/timestamp values.")
    data["date"] = data["_mp_timestamp"].dt.tz_convert(None)
    data["_mp_session_date"] = data["_mp_timestamp"].dt.tz_convert("America/New_York").dt.date
    data = data.dropna(subset=["ticker"]).sort_values(["ticker", "_mp_timestamp"]).reset_index(drop=True)
    data = _ensure_price_state(data)
    data = _prepare_entry_exit_indicators(data)
    data = _add_entry_exit_features(data, config)
    data = _add_path_labels(data, config)
    data, audit = _apply_entry_exit_readiness(data, config)
    return data.reset_index(drop=True), audit


def train_entry_exit_model(
    dataset: pd.DataFrame,
    *,
    target_col: str,
    model_out: Path,
    predictions_out: Path | None = None,
    metrics_out: Path | None = None,
    max_iter: int = 350,
    learning_rate: float = 0.04,
    embargo_rows: int | None = None,
    feature_set: str = "all",
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    if target_col not in dataset.columns:
        raise ValueError(f"Dataset missing target column: {target_col}")
    feature_set = feature_set.strip().lower()
    if feature_set not in ENTRY_EXIT_FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {sorted(ENTRY_EXIT_FEATURE_SETS)}")
    data = add_market_regime_labels(dataset).sort_values(["date", "ticker"]).dropna(subset=[target_col]).copy()
    if len(data) < 200:
        raise ValueError("Need at least 200 labeled rows to train an entry/exit model.")
    y = data[target_col].astype(int)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=learning_rate, random_state=42),
            ),
        ]
    )
    embargo = int(embargo_rows if embargo_rows is not None else max(1, min(20, _infer_horizon_from_target(target_col))))
    validation_groups = _validation_time_groups(data)
    splits = min(5, max(2, pd.Series(validation_groups).nunique() // 25))
    cv = DateGroupedPurgedWalkForwardSplit(
        n_splits=splits,
        embargo_groups=embargo,
        min_train_size=min(5000, max(300, len(data) // 3)),
    )
    candidate_features = [
        col
        for col in ENTRY_EXIT_FEATURES
        if col in data.columns and data[col].notna().any() and _feature_allowed_for_set(col, feature_set)
    ]
    if len(candidate_features) < 5:
        raise ValueError(f"Too few candidate entry/exit features for feature_set={feature_set}: {len(candidate_features)}")
    candidate_x = data[candidate_features]
    split_indices = list(cv.split(candidate_x, y, groups=validation_groups))
    if len(split_indices) < 2:
        raise ValueError("Not enough rows for purged walk-forward validation.")
    features, excluded_fold_sparse = _drop_features_without_fold_training_coverage(data, candidate_features, split_indices)
    min_features = 5 if feature_set == "catalyst" else 15
    if len(features) < min_features:
        raise ValueError(f"Too few usable entry/exit features after temporal coverage checks: {len(features)}")
    x = data[features]
    probabilities = pd.Series(index=y.index, dtype="float")
    predictions = pd.Series(index=y.index, dtype="float")
    for train_idx, test_idx in split_indices:
        fold_model = clone(model)
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        probabilities.iloc[test_idx] = fold_model.predict_proba(x.iloc[test_idx])[:, 1]
        predictions.iloc[test_idx] = (probabilities.iloc[test_idx] >= 0.5).astype(int)
    scored = probabilities.notna()
    y_scored = y[scored]
    pred_scored = predictions[scored].astype(int)
    prob_scored = probabilities[scored].astype(float)
    precision, recall, f1, _ = precision_recall_fscore_support(y_scored, pred_scored, average="binary", zero_division=0)
    try:
        roc_auc = float(roc_auc_score(y_scored, prob_scored))
    except ValueError:
        roc_auc = float("nan")
    lift = _top_decile_lift(y_scored, prob_scored)
    report = classification_report(y_scored, pred_scored, digits=3, zero_division=0)
    model.fit(x, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "features": features,
        "target_col": target_col,
        "schema_version": ENTRY_EXIT_SCHEMA_VERSION,
        "model_type": "entry_path",
        "feature_set": feature_set,
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "embargo_groups": embargo,
        "validation_split": "date_grouped_purged_walk_forward",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, model_out)
    metrics = {
        "schema_version": ENTRY_EXIT_SCHEMA_VERSION,
        "model_type": "entry_path",
        "target_col": target_col,
        "feature_set": feature_set,
        "rows": int(len(data)),
        "tickers": int(data["ticker"].nunique()),
        "features": int(len(features)),
        "candidate_features": int(len(candidate_features)),
        "excluded_fold_sparse_features": int(len(excluded_fold_sparse)),
        "validated_rows": int(scored.sum()),
        "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_scored, pred_scored)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "top_decile_positive_rate": lift["top_decile_positive_rate"],
        "base_positive_rate": lift["base_positive_rate"],
        "top_decile_lift": lift["top_decile_lift"],
        "validation_split": payload["validation_split"],
        "embargo_groups": int(embargo),
        "model_out": str(model_out),
        "trained_at_utc": payload["trained_at_utc"],
    }
    manifest = write_model_manifest(
        model_path=model_out,
        model_type="entry_path",
        schema_version=ENTRY_EXIT_SCHEMA_VERSION,
        target_col=target_col,
        features=features,
        training_data=data,
        metrics=metrics,
        validation_split=payload["validation_split"],
        extra={
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "embargo_groups": embargo,
            "feature_set": feature_set,
            "excluded_fold_sparse_features": excluded_fold_sparse,
        },
    )
    metrics["manifest_path"] = str(model_out.with_suffix(model_out.suffix + ".manifest.json"))
    metrics["artifact_sha256"] = manifest["artifact_sha256"]
    audit_cols = _oos_audit_columns(data, target_col)
    oos = data.loc[scored, audit_cols].copy()
    oos["oos_probability"] = prob_scored.to_numpy()
    oos["oos_prediction"] = pred_scored.to_numpy()
    if predictions_out:
        predictions_out.parent.mkdir(parents=True, exist_ok=True)
        oos.to_csv(predictions_out, index=False)
    if metrics_out:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(metrics_out, index=False)
        metrics_out.with_suffix(".txt").write_text(report, encoding="utf-8")
    summary = (
        f"target={target_col}\n"
        f"schema={ENTRY_EXIT_SCHEMA_VERSION}\n"
        f"feature_set={feature_set}\n"
        f"rows={len(data)} tickers={data['ticker'].nunique()} features={len(features)}\n"
        f"candidate_features={len(candidate_features)} excluded_fold_sparse_features={len(excluded_fold_sparse)}\n"
        f"validation_split={payload['validation_split']} embargo_groups={embargo}\n"
        f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} roc_auc={metrics['roc_auc']:.4f} "
        f"top_decile_lift={metrics['top_decile_lift']:.4f}\n"
        f"{report}"
    )
    return summary, metrics, oos


def score_entry_exit_frame(dataset: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
    features = payload["features"]
    dataset = add_market_regime_labels(dataset)
    missing = [col for col in features if col not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset missing model features: {missing[:10]}")
    scored = dataset.copy()
    probability_col = _probability_column(str(payload.get("target_col", "entry_exit")))
    prediction_col = probability_col.replace("probability", "prediction")
    scored[probability_col] = payload["model"].predict_proba(scored[features])[:, 1]
    scored[prediction_col] = (scored[probability_col] >= 0.5).astype(int)
    scored["entry_exit_model_target"] = payload.get("target_col")
    scored["entry_exit_model_schema"] = payload.get("schema_version")
    return scored


def _feature_allowed_for_set(feature: str, feature_set: str) -> bool:
    is_catalyst = _is_catalyst_feature(feature)
    if feature_set == "all":
        return True
    if feature_set == "catalyst":
        return is_catalyst
    if feature_set == "technical":
        return not is_catalyst
    raise ValueError(f"Unknown feature_set: {feature_set}")


def _is_catalyst_feature(feature: str) -> bool:
    if feature in CATALYST_FEATURE_NAMES:
        return True
    return any(feature.startswith(prefix) for prefix in CATALYST_FEATURE_PREFIXES)


def _drop_features_without_fold_training_coverage(
    data: pd.DataFrame,
    candidate_features: list[str],
    split_indices: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    excluded: list[str] = []
    for feature in candidate_features:
        series = data[feature]
        has_values_in_every_train_fold = all(series.iloc[train_idx].notna().any() for train_idx, _ in split_indices)
        if has_values_in_every_train_fold:
            kept.append(feature)
        else:
            excluded.append(feature)
    return kept, excluded


def _oos_audit_columns(data: pd.DataFrame, target_col: str) -> list[str]:
    base = ["ticker", "date", target_col]
    suffix = _infer_horizon_from_target(target_col)
    path_cols = [
        f"entry_exit_outcome_{suffix}b",
        f"bars_to_exit_{suffix}b",
        f"max_favorable_excursion_{suffix}b",
        f"max_adverse_excursion_{suffix}b",
        f"horizon_return_from_entry_{suffix}b",
        f"target_exit_risk_{suffix}b",
        f"target_timeout_positive_{suffix}b",
    ]
    context_cols = [
        "market_regime",
        *MARKET_REGIME_FEATURES,
        "news_count",
        "event_count",
        "market_context_news_count",
        "source_count_alpaca",
        "source_count_reddit",
        "source_count_seeking_alpha",
        "source_count_sec",
        "source_count_finviz",
        "volume_z20",
        "setup_candidate_score",
        "dist_session_vwap",
        "one_minute_dist_vwap",
        "one_minute_volume_burst_15m",
        *INTRADAY_CATALYST_FEATURES,
    ]
    return [col for col in [*base, *path_cols, *context_cols] if col in data.columns]


def _ensure_price_state(data: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Entry/exit dataset missing OHLC columns: {missing}")
    out = data.copy()
    for column in ["open", "high", "low", "close", "volume"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    if "atr_14" in out.columns and "atr_pct_14" in out.columns:
        return out
    pieces = []
    for _, group in out.groupby("ticker", sort=False):
        enriched = add_price_features(group.sort_values("date"))
        pieces.append(enriched)
    return pd.concat(pieces, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


def _prepare_entry_exit_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data = data.sort_values(["ticker", "date"], na_position="last").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume", "rsi_14", "macd_signal_diff", "volume_z20"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
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
    if "rsi_14" not in data.columns:
        data["rsi_14"] = grouped["close"].transform(_rsi_14)
    data["prior_close"] = grouped["close"].shift(1)
    data["prior_ema_20"] = grouped["ema_20"].shift(1)
    data["prior_ema_50"] = grouped["ema_50"].shift(1)
    data["prior_macd_signal_diff"] = grouped["macd_signal_diff"].shift(1)
    data["prior_rsi_14"] = grouped["rsi_14"].shift(1)
    data["prior_20d_high"] = grouped["close"].transform(lambda series: series.shift(1).rolling(20, min_periods=10).max())
    data["prior_20d_low"] = grouped["close"].transform(lambda series: series.shift(1).rolling(20, min_periods=10).min())
    return data


def _rsi_14(series: pd.Series) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _add_path_labels(data: pd.DataFrame, config: EntryExitLabelConfig) -> pd.DataFrame:
    pieces = []
    for _, group in data.groupby("ticker", sort=False):
        pieces.append(_add_path_labels_for_ticker(group.sort_values("date").reset_index(drop=True), config))
    return pd.concat(pieces, ignore_index=True)


def _add_path_labels_for_ticker(group: pd.DataFrame, config: EntryExitLabelConfig) -> pd.DataFrame:
    frame = group.copy()
    suffix = f"{config.horizon_bars}b"
    n_rows = len(frame)
    open_values = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype="float")
    high_values = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype="float")
    low_values = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype="float")
    close_values = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype="float")
    atr_values = pd.to_numeric(frame["atr_14"], errors="coerce").to_numpy(dtype="float")

    entry_prices = np.full(n_rows, np.nan)
    target_prices = np.full(n_rows, np.nan)
    stop_prices = np.full(n_rows, np.nan)
    bars_to_exit = np.full(n_rows, np.nan)
    mfe = np.full(n_rows, np.nan)
    mae = np.full(n_rows, np.nan)
    success = np.full(n_rows, np.nan)
    stop_first = np.full(n_rows, np.nan)
    timeout_positive = np.full(n_rows, np.nan)
    horizon_return = np.full(n_rows, np.nan)
    outcomes = np.full(n_rows, None, dtype=object)

    row_index = np.arange(n_rows)
    entry_index = row_index + 1
    has_entry_row = entry_index < n_rows
    entry = np.full(n_rows, np.nan)
    entry[has_entry_row] = open_values[entry_index[has_entry_row]]
    atr = atr_values
    valid_setup = has_entry_row & np.isfinite(entry) & np.isfinite(atr) & (atr > 0)
    target = entry + config.take_profit_atr * atr
    stop = entry - config.stop_loss_atr * atr

    window_count = np.zeros(n_rows, dtype="int")
    max_high = np.full(n_rows, -np.inf)
    min_low = np.full(n_rows, np.inf)
    final_close = np.full(n_rows, np.nan)
    resolved = np.zeros(n_rows, dtype=bool)

    use_session_boundary = _is_intraday_config(config) and not config.allow_overnight and "_mp_session_date" in frame.columns
    sessions = frame["_mp_session_date"].to_numpy(dtype=object) if use_session_boundary else None
    entry_sessions = np.full(n_rows, None, dtype=object)
    if sessions is not None:
        entry_sessions[has_entry_row] = sessions[entry_index[has_entry_row]]

    for offset in range(1, config.horizon_bars + 1):
        future_index = row_index + offset
        in_range = future_index < n_rows
        valid_window = valid_setup & in_range
        if sessions is not None:
            valid_window &= sessions[future_index.clip(max=n_rows - 1)] == entry_sessions
        if not valid_window.any():
            continue
        future_high = np.full(n_rows, np.nan)
        future_low = np.full(n_rows, np.nan)
        future_close = np.full(n_rows, np.nan)
        future_high[valid_window] = high_values[future_index[valid_window]]
        future_low[valid_window] = low_values[future_index[valid_window]]
        future_close[valid_window] = close_values[future_index[valid_window]]

        window_count[valid_window] += 1
        max_high[valid_window] = np.fmax(max_high[valid_window], future_high[valid_window])
        min_low[valid_window] = np.fmin(min_low[valid_window], future_low[valid_window])
        final_close[valid_window] = future_close[valid_window]

        unresolved = valid_window & ~resolved
        hit_target = unresolved & np.isfinite(future_high) & (future_high >= target)
        hit_stop = unresolved & np.isfinite(future_low) & (future_low <= stop)
        ambiguous = hit_target & hit_stop
        target_only = hit_target & ~hit_stop
        stop_only = hit_stop & ~hit_target

        for mask, outcome in [
            (ambiguous, "ambiguous"),
            (target_only, "target_first"),
            (stop_only, "stop_first"),
        ]:
            if mask.any():
                outcomes[mask] = outcome
                bars_to_exit[mask] = float(offset)
                resolved[mask] = True

    has_window = valid_setup & (window_count > 0)
    timeout = has_window & ~resolved
    outcomes[timeout] = "timeout"
    bars_to_exit[timeout] = window_count[timeout].astype(float)

    entry_prices[has_window] = entry[has_window]
    target_prices[has_window] = target[has_window]
    stop_prices[has_window] = stop[has_window]
    finite_entry = has_window & np.isfinite(entry) & (entry != 0)
    mfe[finite_entry] = max_high[finite_entry] / entry[finite_entry] - 1.0
    mae[finite_entry] = min_low[finite_entry] / entry[finite_entry] - 1.0
    horizon_return[finite_entry] = final_close[finite_entry] / entry[finite_entry] - 1.0

    ambiguous = outcomes == "ambiguous"
    success[has_window] = 0.0
    stop_first[has_window] = 0.0
    success[(outcomes == "target_first") | (ambiguous & (config.ambiguous_policy == "target"))] = 1.0
    stop_first[(outcomes == "stop_first") | (ambiguous & (config.ambiguous_policy == "stop"))] = 1.0
    if config.ambiguous_policy == "ignore":
        success[ambiguous] = np.nan
        stop_first[ambiguous] = np.nan
    timeout_positive[has_window] = 0.0
    timeout_positive[timeout & np.isfinite(horizon_return) & (horizon_return > 0)] = 1.0

    frame[f"entry_price_next_open_{suffix}"] = entry_prices
    frame[f"entry_target_price_{suffix}"] = target_prices
    frame[f"entry_stop_price_{suffix}"] = stop_prices
    frame[f"entry_exit_outcome_{suffix}"] = outcomes
    frame[f"bars_to_exit_{suffix}"] = bars_to_exit
    frame[f"max_favorable_excursion_{suffix}"] = mfe
    frame[f"max_adverse_excursion_{suffix}"] = mae
    frame[f"horizon_return_from_entry_{suffix}"] = horizon_return
    frame[f"target_entry_success_{suffix}"] = pd.Series(success, index=frame.index, dtype="float")
    frame[f"target_exit_risk_{suffix}"] = pd.Series(stop_first, index=frame.index, dtype="float")
    frame[f"target_timeout_positive_{suffix}"] = pd.Series(timeout_positive, index=frame.index, dtype="float")
    tail = frame.index[-config.horizon_bars :]
    label_cols = [col for col in frame.columns if col.endswith(suffix) and col.startswith(("target_", "horizon_", "max_", "bars_to_", "entry_"))]
    frame.loc[tail, label_cols] = np.nan
    return frame


def _add_entry_exit_features(data: pd.DataFrame, config: EntryExitLabelConfig) -> pd.DataFrame:
    frame = data.copy()
    close = pd.to_numeric(frame.get("close"), errors="coerce")
    atr = pd.to_numeric(frame.get("atr_14"), errors="coerce")
    frame["setup_stop_pct"] = (config.stop_loss_atr * atr).div(close).replace([np.inf, -np.inf], np.nan)
    frame["setup_target_pct"] = (config.take_profit_atr * atr).div(close).replace([np.inf, -np.inf], np.nan)
    frame["setup_risk_reward"] = frame["setup_target_pct"].div(frame["setup_stop_pct"]).replace([np.inf, -np.inf], np.nan)
    frame["entry_horizon_bars"] = float(config.horizon_bars)
    return frame


def _apply_entry_exit_readiness(data: pd.DataFrame, config: EntryExitLabelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_col = f"target_entry_success_{config.horizon_bars}b"
    counts = data.groupby("ticker").agg(
        rows=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        labeled_rows=(target_col, lambda series: int(pd.to_numeric(series, errors="coerce").notna().sum())),
    )
    counts["eligible_rows"] = counts["rows"] >= config.min_rows_per_ticker
    counts["eligible_labels"] = counts["labeled_rows"] >= config.min_labeled_rows_per_ticker
    counts["model_eligible"] = counts["eligible_rows"] & counts["eligible_labels"]
    eligible = set(counts[counts["model_eligible"]].index)
    filtered = data[data["ticker"].isin(eligible)].copy()
    audit = counts.reset_index()
    audit["schema_version"] = ENTRY_EXIT_SCHEMA_VERSION
    audit["config"] = str(asdict(config))
    return filtered, audit


def _validation_time_groups(data: pd.DataFrame) -> pd.Series:
    if "_mp_session_date" in data.columns:
        return pd.Series(data["_mp_session_date"], index=data.index)
    if "timestamp" in data.columns:
        return pd.to_datetime(data["timestamp"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.date
    return pd.to_datetime(data["date"], errors="coerce", utc=True).dt.tz_convert("America/New_York").dt.date


def _is_intraday_config(config: EntryExitLabelConfig) -> bool:
    kind = config.bar_kind.strip().lower()
    return kind not in {"swing", "daily", "day", "1d", "d"}


def _probability_column(target_col: str) -> str:
    safe = target_col.replace("target_", "").replace("-", "_")
    return f"{safe}_probability"


def _infer_horizon_from_target(target_col: str) -> int:
    match = pd.Series([target_col]).str.extract(r"_(\d+)b$").iloc[0, 0]
    if pd.isna(match):
        return 5
    return int(match)


def _top_decile_lift(y_true: pd.Series, probability: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"target": y_true.astype(int), "probability": probability.astype(float)}).dropna()
    if frame.empty:
        return {"top_decile_positive_rate": float("nan"), "base_positive_rate": float("nan"), "top_decile_lift": float("nan")}
    cutoff = frame["probability"].quantile(0.9)
    top = frame[frame["probability"] >= cutoff]
    base_rate = float(frame["target"].mean())
    top_rate = float(top["target"].mean()) if not top.empty else float("nan")
    lift = top_rate / base_rate if base_rate else float("nan")
    return {"top_decile_positive_rate": top_rate, "base_positive_rate": base_rate, "top_decile_lift": float(lift)}


def _num(value: object) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
