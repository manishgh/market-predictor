from __future__ import annotations

from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.registry import write_model_manifest


DEFAULT_FEATURES = [
    "return_1d",
    "return_5d_past",
    "return_10d_past",
    "return_20d_past",
    "realized_vol_10d",
    "realized_vol_20d",
    "realized_vol_60d",
    "atr_pct_14",
    "rsi_14",
    "macd_signal_diff",
    "dist_sma_20",
    "dist_sma_50",
    "sma20_gt_sma50",
    "volume_z20",
    "gap_pct",
    "pct_from_52w_high",
    "pct_from_52w_low",
    "spy_return_1d",
    "spy_return_5d_past",
    "spy_return_10d_past",
    "spy_return_20d_past",
    "spy_realized_vol_20d",
    "spy_volume_z20",
    "spy_gap_pct",
    "sector_return_1d",
    "sector_return_5d_past",
    "sector_return_10d_past",
    "sector_return_20d_past",
    "sector_realized_vol_20d",
    "sector_volume_z20",
    "sector_gap_pct",
    "rel_return_1d_vs_spy",
    "rel_return_5d_vs_spy",
    "rel_return_10d_vs_spy",
    "rel_return_20d_vs_spy",
    "rel_return_1d_vs_sector",
    "rel_return_5d_vs_sector",
    "rel_return_10d_vs_sector",
    "rel_return_20d_vs_sector",
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
    "event_earnings_count",
    "event_analyst_count",
    "event_guidance_count",
    "event_ma_count",
    "event_fda_count",
    "event_contract_count",
    "event_sec_count",
    "event_offering_count",
    "event_insider_count",
    "event_reaction_2h_mean",
    "event_reaction_2h_abs_max",
    "event_reaction_volume_sum",
    "premarket_gap_mean",
    "premarket_day_return_mean",
    "intraday_reaction_2h_mean",
    "intraday_to_close_mean",
    "afterhours_next_open_gap_mean",
    "afterhours_next_day_return_mean",
    "sec_eps_diluted_recent",
    "sec_eps_basic_recent",
    "sec_revenue_recent",
    "sec_net_income_recent",
    "quant_rating_score",
    "valuation_score",
    "growth_score",
    "profitability_score",
    "momentum_score",
    "eps_revision_score",
    "eps_surprise",
    "revenue_surprise",
    "days_to_earnings",
    "buzz_spike_x_volume_z",
    "sentiment_x_news_attention",
    "earnings_x_eps_surprise",
    "catalyst_x_volume_z",
    "reaction_x_sentiment",
    "premarket_gap_x_sentiment",
    "revision_x_days_to_earnings",
]


WATCH_SCORE_FEATURES = [
    "return_1d",
    "return_5d_past",
    "rel_return_1d_vs_spy",
    "rel_return_5d_vs_spy",
    "rel_return_1d_vs_sector",
    "rel_return_5d_vs_sector",
    "spy_return_1d",
    "sector_return_1d",
    "volume_z20",
    "news_count",
    "sentiment_mean",
    "market_context_news_count",
    "market_context_sentiment_mean",
    "market_context_sentiment_neg_frac",
    "source_count_alpaca",
    "source_count_reddit",
    "source_count_seeking_alpha",
    "source_count_sec",
    "source_count_finviz",
    "event_count",
    "event_sec_count",
    "event_offering_count",
    "event_insider_count",
    "event_reaction_2h_mean",
    "premarket_gap_mean",
    "intraday_reaction_2h_mean",
    "afterhours_next_open_gap_mean",
    "reddit_sentiment_mean",
    "reddit_score_sum",
    "reddit_comments_sum",
    "quant_rating_score",
    "momentum_score",
    "eps_revision_score",
]


EVENT_SWING_FEATURES = [
    "sentiment_numeric",
    "sentiment_abs",
    "title_has_ticker",
    "text_has_ticker",
    "generic_movers_headline",
    "event_relevance_score",
    "engagement_score",
    "engagement_comments",
    "engagement_upvote_ratio",
    "source_is_alpaca",
    "source_is_reddit",
    "source_is_seeking_alpha",
    "source_is_sec",
    "time_is_pre_market",
    "time_is_intraday",
    "time_is_after_hours",
    "event_earnings",
    "event_analyst",
    "event_guidance",
    "event_ma",
    "event_fda",
    "event_contract",
    "event_sec",
    "event_offering",
    "event_insider",
    "pre_return_1d",
    "pre_return_5d_past",
    "pre_return_10d_past",
    "pre_return_20d_past",
    "pre_realized_vol_10d",
    "pre_realized_vol_20d",
    "pre_realized_vol_60d",
    "pre_atr_pct_14",
    "pre_rsi_14",
    "pre_macd_signal_diff",
    "pre_dist_sma_20",
    "pre_dist_sma_50",
    "pre_sma20_gt_sma50",
    "pre_volume_z20",
    "pre_gap_pct",
    "pre_pct_from_52w_high",
    "pre_pct_from_52w_low",
    "pre_spy_return_1d",
    "pre_spy_return_5d_past",
    "pre_spy_volume_z20",
    "pre_sector_return_1d",
    "pre_sector_return_5d_past",
    "pre_sector_volume_z20",
    "pre_rel_return_1d_vs_spy",
    "pre_rel_return_5d_vs_spy",
    "pre_rel_return_1d_vs_sector",
    "pre_rel_return_5d_vs_sector",
    "pre_market_context_news_count",
    "pre_market_context_sentiment_mean",
    "pre_market_context_sentiment_min",
    "pre_market_context_sentiment_max",
    "pre_market_context_sentiment_neg_frac",
    "pre_market_context_sentiment_pos_frac",
    "pre_market_context_news_count_z30",
    "pre_market_context_sentiment_momentum_5d",
    "event_day_gap_pct",
    "event_day_open_to_close_return",
    "event_day_close_vs_prev_close",
    "pre_1h_return",
    "pre_4h_return",
    "reaction_1h",
    "reaction_2h",
    "reaction_4h",
    "reaction_abs_2h",
    "reaction_volume_2h",
    "first_bar_delay_minutes",
    "reaction_2h_x_sentiment",
    "reaction_abs_2h_x_sentiment_abs",
    "event_gap_x_sentiment",
]


class PurgedWalkForwardSplit:
    def __init__(self, n_splits: int = 5, embargo: int = 5, min_train_size: int = 40) -> None:
        self.n_splits = n_splits
        self.embargo = embargo
        self.min_train_size = min_train_size

    def split(self, x: pd.DataFrame, y: pd.Series | None = None, groups: object | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n_rows = len(x)
        fold_size = max(1, n_rows // (self.n_splits + 1))
        for split_num in range(self.n_splits):
            test_start = self.min_train_size + split_num * fold_size
            test_end = min(test_start + fold_size, n_rows)
            train_end = max(0, test_start - self.embargo)
            if train_end < self.min_train_size or test_start >= n_rows or test_end <= test_start:
                continue
            yield np.arange(0, train_end), np.arange(test_start, test_end)

    def get_n_splits(self, x: pd.DataFrame | None = None, y: pd.Series | None = None, groups: object | None = None) -> int:
        if x is None:
            return self.n_splits
        return sum(1 for _ in self.split(x, y, groups))


class DateGroupedPurgedWalkForwardSplit:
    """Walk-forward splitter that keeps each timestamp/date group in one fold.

    Row-count embargoes are not sufficient for multi-symbol trading data because
    a single market day can contain hundreds of adjacent rows. This splitter
    treats the supplied ``groups`` values as ordered time buckets and embargoes
    whole buckets.
    """

    def __init__(self, n_splits: int = 5, embargo_groups: int = 1, min_train_size: int = 40) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1.")
        if embargo_groups < 0:
            raise ValueError("embargo_groups must be >= 0.")
        self.n_splits = n_splits
        self.embargo_groups = embargo_groups
        self.min_train_size = min_train_size

    def split(
        self,
        x: pd.DataFrame,
        y: pd.Series | None = None,
        groups: object | None = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("DateGroupedPurgedWalkForwardSplit requires groups.")
        n_rows = len(x)
        group_series = pd.Series(groups).reset_index(drop=True)
        if len(group_series) != n_rows:
            raise ValueError("groups length must match x length.")
        normalized = self._normalize_groups(group_series)
        ordered_groups = list(pd.Index(normalized.dropna().unique()).sort_values())
        if len(ordered_groups) < self.n_splits + 1:
            return
        fold_size = max(1, len(ordered_groups) // (self.n_splits + 1))
        for split_num in range(self.n_splits):
            test_start_group = min(len(ordered_groups), self.min_train_groups(ordered_groups, fold_size) + split_num * fold_size)
            test_end_group = min(test_start_group + fold_size, len(ordered_groups))
            train_end_group = max(0, test_start_group - self.embargo_groups)
            if test_start_group >= len(ordered_groups) or test_end_group <= test_start_group:
                continue
            train_groups = set(ordered_groups[:train_end_group])
            test_groups = set(ordered_groups[test_start_group:test_end_group])
            train_idx = np.flatnonzero(normalized.isin(train_groups).to_numpy())
            test_idx = np.flatnonzero(normalized.isin(test_groups).to_numpy())
            if len(train_idx) < self.min_train_size or len(test_idx) == 0:
                continue
            yield train_idx, test_idx

    def get_n_splits(self, x: pd.DataFrame | None = None, y: pd.Series | None = None, groups: object | None = None) -> int:
        if x is None:
            return self.n_splits
        return sum(1 for _ in self.split(x, y, groups))

    def min_train_groups(self, ordered_groups: list[object], fold_size: int) -> int:
        return min(max(1, fold_size), max(1, len(ordered_groups) // 3))

    @staticmethod
    def _normalize_groups(groups: pd.Series) -> pd.Series:
        converted = pd.to_datetime(groups, errors="coerce", utc=True)
        if converted.notna().any():
            return converted.dt.tz_convert(None)
        return groups.astype("string")


def train_direction_model(
    dataset: pd.DataFrame,
    model_out: Path,
    horizon_days: int | None = None,
    *,
    max_iter: int = 200,
    learning_rate: float = 0.05,
) -> str:
    features = [col for col in DEFAULT_FEATURES if col in dataset.columns]
    target_col = f"target_up_{horizon_days}d" if horizon_days else "target_up_5d"
    if target_col not in dataset.columns:
        target_candidates = sorted(col for col in dataset.columns if col.startswith("target_up_"))
        if not target_candidates:
            raise ValueError("Dataset has no target_up_* label column.")
        target_col = target_candidates[-1]
    data = dataset.sort_values("date").dropna(subset=[target_col]).copy()
    features = [col for col in features if data[col].notna().any()]
    x = data[features]
    y = data[target_col].astype(int)
    if len(data) < 40:
        raise ValueError("Need at least 40 daily rows for a basic time-series validation.")

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

    splits = min(5, max(2, len(data) // 30))
    embargo = int(horizon_days or 5)
    cv = DateGroupedPurgedWalkForwardSplit(
        n_splits=splits,
        embargo_groups=embargo,
        min_train_size=min(60, max(30, len(data) // 3)),
    )
    validation_groups = data["date"]
    if cv.get_n_splits(x, y, groups=validation_groups) < 2:
        raise ValueError("Not enough rows for purged walk-forward validation after applying embargo.")
    predictions = pd.Series(index=y.index, dtype="float")
    for train_idx, test_idx in cv.split(x, y, groups=validation_groups):
        fold_model = clone(model)
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions.iloc[test_idx] = fold_model.predict(x.iloc[test_idx])
    scored = predictions.notna()
    report = classification_report(y[scored], predictions[scored].astype(int), digits=3)
    model.fit(x, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "features": features,
        "horizon_days": horizon_days,
        "target_col": target_col,
        "model_type": "direction",
        "schema_version": "direction.v1",
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "validation_split": "date_grouped_purged_walk_forward",
        "embargo_groups": embargo,
    }
    joblib.dump(payload, model_out)
    write_model_manifest(
        model_path=model_out,
        model_type="direction",
        schema_version="direction.v1",
        target_col=target_col,
        features=features,
        training_data=data,
        metrics={
            "train_rows": int(len(data)),
            "validated_rows": int(scored.sum()),
            "features": int(len(features)),
            "validation_split": payload["validation_split"],
            "embargo_groups": embargo,
        },
        validation_split=payload["validation_split"],
        extra={"horizon_days": horizon_days, "max_iter": max_iter, "learning_rate": learning_rate},
    )
    return (
        f"target={target_col}\n"
        f"validation_split=date_grouped_purged_walk_forward\n"
        f"embargo_groups={embargo}\n"
        f"max_iter={max_iter}\n"
        f"learning_rate={learning_rate}\n"
        f"features={len(features)}\n{report}"
    )


def train_event_swing_model(
    dataset: pd.DataFrame,
    model_out: Path,
    *,
    target_col: str = "target_next_1d_up",
    include_reaction_features: bool = True,
    max_iter: int = 250,
    learning_rate: float = 0.04,
) -> str:
    report, _ = train_event_swing_model_with_metrics(
        dataset,
        model_out,
        target_col=target_col,
        include_reaction_features=include_reaction_features,
        max_iter=max_iter,
        learning_rate=learning_rate,
    )
    return report


def train_event_swing_model_with_metrics(
    dataset: pd.DataFrame,
    model_out: Path,
    *,
    target_col: str = "target_next_1d_up",
    include_reaction_features: bool = True,
    max_iter: int = 250,
    learning_rate: float = 0.04,
) -> tuple[str, dict[str, object]]:
    if target_col not in dataset.columns:
        raise ValueError(f"Dataset has no target column named {target_col}.")
    features = [col for col in EVENT_SWING_FEATURES if col in dataset.columns]
    if not include_reaction_features:
        blocked_prefixes = ("reaction_",)
        blocked_names = {
            "event_day_open_to_close_return",
            "event_day_close_vs_prev_close",
            "reaction_volume_2h",
            "first_bar_delay_minutes",
        }
        features = [
            col
            for col in features
            if not col.startswith(blocked_prefixes) and col not in blocked_names
        ]
    data = dataset.sort_values("event_timestamp").dropna(subset=[target_col]).copy()
    features = [col for col in features if data[col].notna().any()]
    if len(data) < 80:
        raise ValueError("Need at least 80 event rows for event-swing validation.")
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
    splits = min(5, max(2, len(data) // 200))
    cv = DateGroupedPurgedWalkForwardSplit(
        n_splits=splits,
        embargo_groups=10,
        min_train_size=min(500, max(80, len(data) // 3)),
    )
    validation_groups = pd.to_datetime(data["event_timestamp"], errors="coerce", utc=True).dt.date
    if cv.get_n_splits(x, y, groups=validation_groups) < 2:
        raise ValueError("Not enough event rows for purged walk-forward validation after applying embargo.")
    predictions = pd.Series(index=y.index, dtype="float")
    for train_idx, test_idx in cv.split(x, y, groups=validation_groups):
        fold_model = clone(model)
        fold_model.fit(x.iloc[train_idx], y.iloc[train_idx])
        predictions.iloc[test_idx] = fold_model.predict(x.iloc[test_idx])
    scored = predictions.notna()
    report = classification_report(y[scored], predictions[scored].astype(int), digits=3)
    accuracy = float(accuracy_score(y[scored], predictions[scored].astype(int)))
    model.fit(x, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "features": features,
        "target_col": target_col,
        "include_reaction_features": include_reaction_features,
        "model_type": "event_swing",
        "schema_version": "event_swing.v1",
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "validation_split": "date_grouped_purged_walk_forward",
        "embargo_groups": 10,
    }
    joblib.dump(payload, model_out)
    metrics = {
        "target_col": target_col,
        "include_reaction_features": include_reaction_features,
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "features": len(features),
        "train_rows": int(len(data)),
        "validated_rows": int(scored.sum()),
        "accuracy": accuracy,
        "validation_split": "date_grouped_purged_walk_forward",
        "embargo_groups": 10,
        "model_out": str(model_out),
    }
    manifest = write_model_manifest(
        model_path=model_out,
        model_type="event_swing",
        schema_version="event_swing.v1",
        target_col=target_col,
        features=features,
        training_data=data,
        metrics=metrics,
        validation_split=payload["validation_split"],
        extra={
            "include_reaction_features": include_reaction_features,
            "max_iter": max_iter,
            "learning_rate": learning_rate,
        },
    )
    metrics["manifest_path"] = str(model_out.with_suffix(model_out.suffix + ".manifest.json"))
    metrics["artifact_sha256"] = manifest["artifact_sha256"]
    return (
        f"target={target_col}\n"
        f"include_reaction_features={include_reaction_features}\n"
        f"validation_split=date_grouped_purged_walk_forward\n"
        f"embargo_groups=10\n"
        f"max_iter={max_iter}\n"
        f"learning_rate={learning_rate}\n"
        f"features={len(features)}\n{report}"
    ), metrics


def predict_latest(dataset: pd.DataFrame, model_path: Path) -> dict:
    payload = joblib.load(model_path)
    model = payload["model"]
    features = payload["features"]
    horizon_days = payload.get("horizon_days")
    target_col = payload.get("target_col")
    latest = dataset.sort_values("date").iloc[-1:]
    probability = model.predict_proba(latest[features])[0][1]
    return {
        "date": str(latest.iloc[0]["date"]),
        "horizon_days": horizon_days,
        "target": target_col,
        "probability_up": float(probability),
        "prediction": int(probability >= 0.5),
    }


def predict_event_swing_frame(dataset: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
    model = payload["model"]
    features = payload["features"]
    target_col = payload.get("target_col")
    frame = dataset.copy()
    missing = [col for col in features if col not in frame.columns]
    if missing:
        raise ValueError(f"Event dataset missing model features: {missing[:10]}")
    probability = model.predict_proba(frame[features])[:, 1]
    frame["model_probability_up"] = probability
    frame["model_prediction"] = (probability >= 0.5).astype(int)
    frame["model_target"] = target_col
    frame["model_path"] = str(model_path)
    return frame


def heuristic_watch_score(dataset: pd.DataFrame, weights: dict | None = None) -> dict:
    w = {
        "sentiment_mean": 1.2,
        "reddit_sentiment_mean": 1.0,
        "news_count": 0.25,
        "reddit_count": 0.25,
        "reddit_score_sum": 0.002,
        "reddit_comments_sum": 0.003,
        "positive_volume_z20": 0.35,
        "return_1d": 0.75,
        "return_5d_past": 0.5,
        "quant_rating_above_neutral": 0.6,
        "momentum_above_neutral": 0.4,
        "eps_revision_above_neutral": 0.35,
        "positive_eps_surprise": 0.2,
    }
    w.update(weights or {})
    latest = dataset.sort_values("date").iloc[-1]
    z = 0.0
    z += float(w["sentiment_mean"]) * float(latest.get("sentiment_mean", 0) or 0)
    z += float(w["reddit_sentiment_mean"]) * float(latest.get("reddit_sentiment_mean", 0) or 0)
    z += float(w["news_count"]) * min(float(latest.get("news_count", 0) or 0), 20)
    z += float(w["reddit_count"]) * min(float(latest.get("source_count_reddit", 0) or 0), 20)
    z += float(w["reddit_score_sum"]) * min(float(latest.get("reddit_score_sum", 0) or 0), 1000)
    z += float(w["reddit_comments_sum"]) * min(float(latest.get("reddit_comments_sum", 0) or 0), 1000)
    z += float(w["positive_volume_z20"]) * max(float(latest.get("volume_z20", 0) or 0), 0)
    z += float(w["return_1d"]) * float(latest.get("return_1d", 0) or 0)
    z += float(w["return_5d_past"]) * float(latest.get("return_5d_past", 0) or 0)
    z += float(w["quant_rating_above_neutral"]) * max(float(latest.get("quant_rating_score", 3) or 3) - 3, 0)
    z += float(w["momentum_above_neutral"]) * max(float(latest.get("momentum_score", 3) or 3) - 3, 0)
    z += float(w["eps_revision_above_neutral"]) * max(float(latest.get("eps_revision_score", 3) or 3) - 3, 0)
    z += float(w["positive_eps_surprise"]) * max(float(latest.get("eps_surprise", 0) or 0), 0)
    z += 1.2 * float(latest.get("event_reaction_2h_mean", 0) or 0)
    z += 0.8 * float(latest.get("premarket_gap_mean", 0) or 0)
    z += 0.8 * float(latest.get("intraday_reaction_2h_mean", 0) or 0)
    z += 0.6 * float(latest.get("afterhours_next_open_gap_mean", 0) or 0)
    direction = "bullish_watch" if z >= 3 else "neutral_watch" if z >= 1 else "avoid_or_wait"
    return {
        "date": str(latest["date"]),
        "watch_score": round(z, 3),
        "signal": direction,
        "latest_close": float(latest["close"]),
        "news_count": int(latest.get("news_count", 0) or 0),
        "reddit_count": int(latest.get("source_count_reddit", 0) or 0),
        "event_count": int(latest.get("event_count", 0) or 0),
        "event_reaction_2h_mean": float(latest.get("event_reaction_2h_mean", 0) or 0),
        "premarket_gap_mean": float(latest.get("premarket_gap_mean", 0) or 0),
        "intraday_reaction_2h_mean": float(latest.get("intraday_reaction_2h_mean", 0) or 0),
        "afterhours_next_open_gap_mean": float(latest.get("afterhours_next_open_gap_mean", 0) or 0),
        "reddit_score_sum": float(latest.get("reddit_score_sum", 0) or 0),
        "sentiment_mean": float(latest.get("sentiment_mean", 0) or 0),
        "reddit_sentiment_mean": float(latest.get("reddit_sentiment_mean", 0) or 0),
        "quant_rating_score": float(latest.get("quant_rating_score", 0) or 0),
        "momentum_score": float(latest.get("momentum_score", 0) or 0),
        "eps_revision_score": float(latest.get("eps_revision_score", 0) or 0),
    }
