from __future__ import annotations

"""Development-only, cost-aware intraday model training and evaluation."""

import math
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from market_predictor.core.errors import DataReadinessError

MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_candidate.v1"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_evaluation.v1"
AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_authority.v1"
FUTURE_EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_evaluation.v1"
FUTURE_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_authority.v1"
_AUTHORITY_NAME: Final = "_authority.json"
_MANIFEST_NAME: Final = "_manifest.json"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_FUTURE_EVALUATION_NAME: Final = "future_evaluation.json"
_POSITION_LEDGER_NAME: Final = "position_ledger.parquet"
_DAILY_LEDGER_NAME: Final = "daily_ledger.parquet"
_VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.parquet"


def _predictive_metrics(scored: pd.DataFrame) -> dict[str, Any]:
    positive = scored["net_return"].gt(0.0).astype("int8").to_numpy()
    stop = scored["stop_hit"].astype("int8").to_numpy()
    opportunity = scored["predicted_net_return"].to_numpy(dtype="float64")
    downside = scored["predicted_stop_probability"].to_numpy(dtype="float64")
    opportunity_auc = _binary_auc(positive, opportunity)
    stop_auc = _binary_auc(stop, downside)
    base_rate = float(positive.mean())
    depth = max(1, math.ceil(len(scored) * 0.10))
    top = positive[np.argsort(-opportunity, kind="stable")[:depth]]
    stop_prevalence = float(stop.mean())
    brier = float(brier_score_loss(stop, downside))
    baseline_brier = stop_prevalence * (1.0 - stop_prevalence)
    return {
        "positive_net_return_roc_auc": opportunity_auc,
        "positive_net_return_pr_auc": _binary_pr_auc(positive, opportunity),
        "positive_net_return_prevalence": base_rate,
        "top_decile_positive_net_return_rate": float(top.mean()),
        "top_decile_positive_net_return_lift": float(top.mean() / base_rate) if base_rate > 0.0 else None,
        "stop_hit_roc_auc": stop_auc,
        "stop_hit_pr_auc": _binary_pr_auc(stop, downside),
        "stop_hit_prevalence": stop_prevalence,
        "stop_hit_brier": brier,
        "stop_hit_baseline_brier": baseline_brier,
        "stop_hit_brier_skill": 1.0 - brier / baseline_brier if baseline_brier > 0.0 else None,
        "stop_hit_ece": _expected_calibration_error(stop, downside),
    }


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else None


def _expected_calibration_error(target: np.ndarray, probability: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, 9)
    error = 0.0
    for index in range(10):
        selected = bins == index
        if selected.any():
            error += float(selected.mean()) * abs(float(probability[selected].mean()) - float(target[selected].mean()))
    return error


def _binary_pr_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    return float(average_precision_score(target, score)) if len(np.unique(target)) == 2 else None


def _moving_block_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    block_sessions: int,
    seed: int,
) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if len(finite) < block_sessions * 2:
        return {
            "estimate": float(finite.mean()) if len(finite) else None,
            "low": None,
            "high": None,
            "sessions": len(finite),
        }
    starts = np.arange(0, len(finite) - block_sessions + 1)
    blocks_needed = math.ceil(len(finite) / block_sessions)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype="float64")
    for index in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([finite[start : start + block_sessions] for start in chosen])[: len(finite)]
        means[index] = float(sample.mean())
    return {
        "estimate": float(finite.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "sessions": len(finite),
    }


def _moving_block_bootstrap(
    daily_returns: np.ndarray,
    *,
    samples: int,
    block_sessions: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(daily_returns, dtype="float64")
    if len(values) < block_sessions * 2:
        raise DataReadinessError("moving-block bootstrap has insufficient daily portfolio returns")
    starts = np.arange(0, len(values) - block_sessions + 1)
    blocks_needed = math.ceil(len(values) / block_sessions)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype="float64")
    compounded = np.empty(samples, dtype="float64")
    for index in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block_sessions] for start in chosen])[: len(values)]
        means[index] = float(sample.mean())
        compounded[index] = float(np.prod(1.0 + sample) - 1.0)
    return {
        "estimand": "daily_capital_weighted_portfolio_return",
        "sessions": len(values),
        "block_sessions": block_sessions,
        "bootstrap_samples": samples,
        "average_daily_net_return": _interval(values.mean(), means),
        "compounded_net_return": _interval(np.prod(1.0 + values) - 1.0, compounded),
    }


def _interval(estimate: float, samples: np.ndarray) -> dict[str, float]:
    return {
        "estimate": float(estimate),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
    }
