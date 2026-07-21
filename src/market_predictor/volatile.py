from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.model import DEFAULT_FEATURES, DateGroupedPurgedWalkForwardSplit
from market_predictor.registry import verify_model_artifact, write_model_manifest

VOLATILE_SCHEMA_VERSION = "volatile_mover.v1"

VOLATILE_EXTRA_FEATURES = [
    "abs_return_1d",
    "abs_return_5d",
    "news_volume_attention",
    "catalyst_pressure",
    "sentiment_abs_mean",
    "price_volume_pressure",
    "volatile_setup_score",
    "theme_ai_data_software_infra",
    "theme_ai_semis_photonics_hardware",
    "theme_healthcare_biotech_catalyst",
    "theme_healthcare_tools_devices",
    "theme_industrial_space_mobility",
    "theme_seed_high_beta_mover",
]

VOLATILE_FEATURES = list(dict.fromkeys([*DEFAULT_FEATURES, *VOLATILE_EXTRA_FEATURES]))


@dataclass(frozen=True)
class VolatileLabelConfig:
    next_day_big_move_threshold: float = 0.03
    next_week_big_move_threshold: float = 0.08
    min_rows_per_ticker: int = 120
    min_news_rows_per_ticker: int = 3
    min_training_rows: int = 500
    min_training_tickers: int = 20


def load_volatile_universe(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame(columns=["ticker", "theme_bucket"])
    universe = pd.read_csv(path)
    ticker_col = "Ticker" if "Ticker" in universe.columns else "ticker"
    if ticker_col not in universe.columns:
        raise ValueError(f"Universe file {path} must contain Ticker or ticker.")
    universe = universe.copy()
    universe["ticker"] = universe[ticker_col].astype(str).str.upper().str.strip()
    if "theme_bucket" not in universe.columns:
        universe["theme_bucket"] = "unknown"
    return universe.drop_duplicates("ticker")


def build_volatile_dataset(
    daily_1d: pd.DataFrame,
    *,
    daily_5d: pd.DataFrame | None = None,
    universe: pd.DataFrame | None = None,
    config: VolatileLabelConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or VolatileLabelConfig()
    data = daily_1d.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.date
    if daily_5d is not None and not daily_5d.empty:
        five = daily_5d.copy()
        five["ticker"] = five["ticker"].astype(str).str.upper().str.strip()
        five["date"] = pd.to_datetime(five["date"], errors="coerce").dt.date
        merge_cols = [
            col
            for col in [
                "ticker",
                "date",
                "entry_next_open_5d",
                "future_return_5d",
                "target_up_5d",
                "target_bucket_5d",
            ]
            if col in five.columns
        ]
        data = data.drop(columns=[col for col in merge_cols if col not in {"ticker", "date"} and col in data.columns], errors="ignore")
        data = data.merge(five[merge_cols].drop_duplicates(["ticker", "date"]), on=["ticker", "date"], how="left")

    if universe is not None and not universe.empty:
        allowed = set(universe["ticker"])
        data = data[data["ticker"].isin(allowed)].copy()
        data = data.merge(universe[["ticker", "theme_bucket"]].drop_duplicates("ticker"), on="ticker", how="left")
    else:
        data["theme_bucket"] = "unknown"

    data = _add_volatile_features(data)
    data = _add_volatile_labels(data, config)
    data, audit = _apply_readiness_gates(data, config)
    return data.reset_index(drop=True), audit


def train_volatile_model(
    dataset: pd.DataFrame,
    *,
    target_col: str,
    model_out: Path,
    predictions_out: Path | None = None,
    metrics_out: Path | None = None,
    max_iter: int = 400,
    learning_rate: float = 0.035,
    embargo_rows: int = 5,
) -> tuple[str, dict[str, Any], pd.DataFrame]:
    if target_col not in dataset.columns:
        raise ValueError(f"Dataset missing target column: {target_col}")
    data = dataset.sort_values(["date", "ticker"]).dropna(subset=[target_col]).copy()
    if len(data) < 500:
        raise ValueError("Need at least 500 rows to train a volatile mover model.")
    features = [col for col in VOLATILE_FEATURES if col in data.columns and data[col].notna().any()]
    if len(features) < 20:
        raise ValueError(f"Too few usable volatile features: {len(features)}")
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
    splits = min(5, max(2, data["date"].nunique() // 25))
    cv = DateGroupedPurgedWalkForwardSplit(
        n_splits=splits,
        embargo_groups=embargo_rows,
        min_train_size=min(5000, max(500, len(data) // 3)),
    )
    validation_groups = data["date"]
    if cv.get_n_splits(x, y, groups=validation_groups) < 2:
        raise ValueError("Not enough rows for purged walk-forward validation.")
    probabilities = pd.Series(index=y.index, dtype="float")
    predictions = pd.Series(index=y.index, dtype="float")
    for train_idx, test_idx in cv.split(x, y, groups=validation_groups):
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
        "schema_version": VOLATILE_SCHEMA_VERSION,
        "model_type": "volatile_mover",
        "max_iter": max_iter,
        "learning_rate": learning_rate,
        "embargo_groups": embargo_rows,
        "validation_split": "date_grouped_purged_walk_forward",
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }
    joblib.dump(payload, model_out)
    metrics = {
        "schema_version": VOLATILE_SCHEMA_VERSION,
        "model_type": "volatile_mover",
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
        "validation_split": payload["validation_split"],
        "embargo_groups": int(embargo_rows),
        "model_out": str(model_out),
        "trained_at_utc": payload["trained_at_utc"],
    }
    manifest = write_model_manifest(
        model_path=model_out,
        model_type="volatile_mover",
        schema_version=VOLATILE_SCHEMA_VERSION,
        target_col=target_col,
        features=features,
        training_data=data,
        metrics=metrics,
        validation_split=payload["validation_split"],
        extra={
            "max_iter": max_iter,
            "learning_rate": learning_rate,
            "embargo_groups": embargo_rows,
        },
    )
    metrics["manifest_path"] = str(model_out.with_suffix(model_out.suffix + ".manifest.json"))
    metrics["artifact_sha256"] = manifest["artifact_sha256"]
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
        f"schema={VOLATILE_SCHEMA_VERSION}\n"
        f"rows={len(data)} tickers={data['ticker'].nunique()} features={len(features)}\n"
        f"validation_split={payload['validation_split']} embargo_groups={embargo_rows}\n"
        f"accuracy={metrics['accuracy']:.4f} precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} roc_auc={metrics['roc_auc']:.4f} "
        f"top_decile_lift={metrics['top_decile_lift']:.4f}\n"
        f"{report}"
    )
    return summary, metrics, oos


def score_volatile_frame(dataset: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    verify_model_artifact(model_path)
    payload = joblib.load(model_path)
    features = payload["features"]
    missing = [col for col in features if col not in dataset.columns]
    if missing:
        raise ValueError(f"Dataset missing model features: {missing[:10]}")
    model = payload["model"]
    scored = dataset.copy()
    scored["volatile_model_probability"] = model.predict_proba(scored[features])[:, 1]
    scored["volatile_model_prediction"] = (scored["volatile_model_probability"] >= 0.5).astype(int)
    scored["volatile_model_target"] = payload.get("target_col")
    scored["volatile_model_schema"] = payload.get("schema_version")
    return scored


def _add_volatile_features(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for col in ["return_1d", "return_5d_past", "volume_z20", "news_count", "news_count_z30", "event_count", "sentiment_mean"]:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["abs_return_1d"] = frame["return_1d"].abs()
    frame["abs_return_5d"] = frame["return_5d_past"].abs()
    sentiment_abs = frame["sentiment_mean"].abs()
    frame["sentiment_abs_mean"] = sentiment_abs
    frame["news_volume_attention"] = frame["news_count"].fillna(0) * (1.0 + frame["volume_z20"].fillna(0).clip(lower=0))
    source_cols = [
        col
        for col in [
            "source_count_alpaca",
            "source_count_reddit",
            "source_count_seeking_alpha",
            "source_count_sec",
            "source_count_finviz",
        ]
        if col in frame.columns
    ]
    frame["catalyst_pressure"] = frame["event_count"].fillna(0)
    if source_cols:
        frame["catalyst_pressure"] += frame[source_cols].apply(lambda series: pd.to_numeric(series, errors="coerce")).fillna(0).sum(axis=1)
    frame["price_volume_pressure"] = frame["abs_return_1d"].fillna(0) * (1.0 + frame["volume_z20"].fillna(0).clip(lower=0))
    frame["volatile_setup_score"] = (
        frame["abs_return_1d"].fillna(0) * 100.0
        + frame["volume_z20"].fillna(0).clip(lower=0)
        + frame["news_count_z30"].fillna(0).clip(lower=0)
        + frame["event_count"].fillna(0).clip(lower=0) * 0.25
        + sentiment_abs.fillna(0)
    )
    theme = frame.get("theme_bucket", pd.Series("unknown", index=frame.index)).fillna("unknown").astype(str)
    for bucket in [
        "ai_data_software_infra",
        "ai_semis_photonics_hardware",
        "healthcare_biotech_catalyst",
        "healthcare_tools_devices",
        "industrial_space_mobility",
        "seed_high_beta_mover",
    ]:
        frame[f"theme_{bucket}"] = theme.eq(bucket).astype(int)
    return frame


def _add_volatile_labels(data: pd.DataFrame, config: VolatileLabelConfig) -> pd.DataFrame:
    frame = data.copy()
    if "future_return_1d" in frame.columns:
        ret1 = pd.to_numeric(frame["future_return_1d"], errors="coerce")
        frame["target_next_day_up"] = _nullable_label(ret1 > 0, ret1)
        frame["target_next_day_big_up"] = _nullable_label(ret1 >= config.next_day_big_move_threshold, ret1)
        frame["target_next_day_big_down"] = _nullable_label(ret1 <= -config.next_day_big_move_threshold, ret1)
        frame["target_next_day_abs_move"] = _nullable_label(ret1.abs() >= config.next_day_big_move_threshold, ret1)
    if "future_return_5d" in frame.columns:
        ret5 = pd.to_numeric(frame["future_return_5d"], errors="coerce")
        frame["target_next_week_up"] = _nullable_label(ret5 > 0, ret5)
        frame["target_next_week_big_up"] = _nullable_label(ret5 >= config.next_week_big_move_threshold, ret5)
        frame["target_next_week_big_down"] = _nullable_label(ret5 <= -config.next_week_big_move_threshold, ret5)
        frame["target_next_week_abs_move"] = _nullable_label(ret5.abs() >= config.next_week_big_move_threshold, ret5)
    return frame


def _nullable_label(mask: pd.Series, source: pd.Series) -> pd.Series:
    label = mask.astype("Int64")
    label[source.isna()] = pd.NA
    return label


def _apply_readiness_gates(data: pd.DataFrame, config: VolatileLabelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = data.groupby("ticker").agg(
        rows=("date", "count"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        news_rows=("news_count", lambda series: int((pd.to_numeric(series, errors="coerce").fillna(0) > 0).sum())),
        total_news=("news_count", lambda series: float(pd.to_numeric(series, errors="coerce").fillna(0).sum())),
    )
    counts["eligible_rows"] = counts["rows"] >= config.min_rows_per_ticker
    counts["eligible_news"] = counts["news_rows"] >= config.min_news_rows_per_ticker
    counts["model_eligible"] = counts["eligible_rows"] & counts["eligible_news"]
    eligible = set(counts[counts["model_eligible"]].index)
    filtered = data[data["ticker"].isin(eligible)].copy()
    audit = counts.reset_index()
    audit["schema_version"] = VOLATILE_SCHEMA_VERSION
    audit["config"] = str(asdict(config))
    return filtered, audit


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
