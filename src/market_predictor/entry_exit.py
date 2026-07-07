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
from market_predictor.model import DEFAULT_FEATURES, PurgedWalkForwardSplit
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
    "entry_risk_reward",
    "entry_stop_pct",
    "entry_target_pct",
    "entry_horizon_bars",
]

ENTRY_EXIT_FEATURES = list(dict.fromkeys([*DEFAULT_FEATURES, *VOLATILE_EXTRA_FEATURES, *ENTRY_EXIT_EXTRA_FEATURES]))


@dataclass(frozen=True)
class EntryExitLabelConfig:
    horizon_bars: int = 5
    take_profit_atr: float = 1.5
    stop_loss_atr: float = 1.0
    min_rows_per_ticker: int = 120
    min_labeled_rows_per_ticker: int = 40
    ambiguous_policy: str = "stop"
    bar_kind: str = "swing"


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
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["ticker", "date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    data = _ensure_price_state(data)
    data = _prepare_entry_exit_indicators(data)
    data = _add_path_labels(data, config)
    data = _add_entry_exit_features(data, config)
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
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    if target_col not in dataset.columns:
        raise ValueError(f"Dataset missing target column: {target_col}")
    data = dataset.sort_values(["date", "ticker"]).dropna(subset=[target_col]).copy()
    if len(data) < 200:
        raise ValueError("Need at least 200 labeled rows to train an entry/exit model.")
    features = [col for col in ENTRY_EXIT_FEATURES if col in data.columns and data[col].notna().any()]
    if len(features) < 15:
        raise ValueError(f"Too few usable entry/exit features: {len(features)}")
    x = data[features]
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
    splits = min(5, max(2, len(data) // 500))
    cv = PurgedWalkForwardSplit(n_splits=splits, embargo=embargo, min_train_size=min(5000, max(300, len(data) // 3)))
    if cv.get_n_splits(x, y) < 2:
        raise ValueError("Not enough rows for purged walk-forward validation.")
    probabilities = pd.Series(index=y.index, dtype="float")
    predictions = pd.Series(index=y.index, dtype="float")
    for train_idx, test_idx in cv.split(x, y):
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
        "model_type": "entry_exit",
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "embargo_rows": embargo,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(payload, model_out)
    metrics = {
        "schema_version": ENTRY_EXIT_SCHEMA_VERSION,
        "model_type": "entry_exit",
        "target_col": target_col,
        "rows": int(len(data)),
        "tickers": int(data["ticker"].nunique()),
        "features": int(len(features)),
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
        "model_out": str(model_out),
        "trained_at_utc": payload["trained_at_utc"],
    }
    oos = data.loc[scored, ["ticker", "date", target_col]].copy()
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
        f"rows={len(data)} tickers={data['ticker'].nunique()} features={len(features)}\n"
        f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} roc_auc={metrics['roc_auc']:.4f} "
        f"top_decile_lift={metrics['top_decile_lift']:.4f}\n"
        f"{report}"
    )
    return summary, metrics, oos


def score_entry_exit_frame(dataset: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
    features = payload["features"]
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
    entry_prices: list[float] = []
    target_prices: list[float] = []
    stop_prices: list[float] = []
    outcomes: list[str | None] = []
    bars_to_exit: list[float] = []
    mfe: list[float] = []
    mae: list[float] = []
    success: list[float] = []
    stop_first: list[float] = []
    timeout_positive: list[float] = []
    horizon_return: list[float] = []
    for idx in range(len(frame)):
        entry_idx = idx + 1
        end_idx = min(idx + config.horizon_bars, len(frame) - 1)
        if entry_idx > end_idx:
            entry_prices.append(np.nan)
            target_prices.append(np.nan)
            stop_prices.append(np.nan)
            outcomes.append(None)
            bars_to_exit.append(np.nan)
            mfe.append(np.nan)
            mae.append(np.nan)
            success.append(np.nan)
            stop_first.append(np.nan)
            timeout_positive.append(np.nan)
            horizon_return.append(np.nan)
            continue
        entry = _num(frame.iloc[entry_idx].get("open"))
        atr = _num(frame.iloc[idx].get("atr_14"))
        if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
            entry_prices.append(np.nan)
            target_prices.append(np.nan)
            stop_prices.append(np.nan)
            outcomes.append(None)
            bars_to_exit.append(np.nan)
            mfe.append(np.nan)
            mae.append(np.nan)
            success.append(np.nan)
            stop_first.append(np.nan)
            timeout_positive.append(np.nan)
            horizon_return.append(np.nan)
            continue
        target = entry + config.take_profit_atr * atr
        stop = entry - config.stop_loss_atr * atr
        window = frame.iloc[entry_idx : end_idx + 1]
        max_high = pd.to_numeric(window["high"], errors="coerce").max()
        min_low = pd.to_numeric(window["low"], errors="coerce").min()
        final_close = _num(frame.iloc[end_idx].get("close"))
        result = "timeout"
        exit_bars = float(config.horizon_bars)
        for row_offset, (_, row) in enumerate(window.iterrows(), start=1):
            high = _num(row.get("high"))
            low = _num(row.get("low"))
            hit_target = np.isfinite(high) and high >= target
            hit_stop = np.isfinite(low) and low <= stop
            if hit_target and hit_stop:
                result = "ambiguous"
                exit_bars = float(row_offset)
                break
            if hit_target:
                result = "target_first"
                exit_bars = float(row_offset)
                break
            if hit_stop:
                result = "stop_first"
                exit_bars = float(row_offset)
                break
        ret = final_close / entry - 1.0 if np.isfinite(final_close) and entry else np.nan
        entry_prices.append(entry)
        target_prices.append(target)
        stop_prices.append(stop)
        outcomes.append(result)
        bars_to_exit.append(exit_bars)
        mfe.append(max_high / entry - 1.0 if np.isfinite(max_high) and entry else np.nan)
        mae.append(min_low / entry - 1.0 if np.isfinite(min_low) and entry else np.nan)
        horizon_return.append(ret)
        success.append(1.0 if result == "target_first" else 0.0)
        stop_first.append(1.0 if result == "stop_first" or (result == "ambiguous" and config.ambiguous_policy == "stop") else 0.0)
        timeout_positive.append(1.0 if result == "timeout" and np.isfinite(ret) and ret > 0 else 0.0)
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
    suffix = f"{config.horizon_bars}b"
    entry = pd.to_numeric(frame.get(f"entry_price_next_open_{suffix}"), errors="coerce")
    stop = pd.to_numeric(frame.get(f"entry_stop_price_{suffix}"), errors="coerce")
    target = pd.to_numeric(frame.get(f"entry_target_price_{suffix}"), errors="coerce")
    frame["entry_stop_pct"] = entry.sub(stop).div(entry).replace([np.inf, -np.inf], np.nan)
    frame["entry_target_pct"] = target.sub(entry).div(entry).replace([np.inf, -np.inf], np.nan)
    frame["entry_risk_reward"] = frame["entry_target_pct"].div(frame["entry_stop_pct"]).replace([np.inf, -np.inf], np.nan)
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
