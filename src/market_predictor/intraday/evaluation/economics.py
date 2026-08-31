"""Economic ranking diagnostics for intraday model evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from market_predictor.intraday.evaluation.metrics import _moving_block_mean_interval
from market_predictor.intraday.training.config import IntradayDevelopmentConfig


def _economic_ranking_metrics(
    scored: pd.DataFrame,
    config: IntradayDevelopmentConfig,
) -> dict[str, Any]:
    gains: list[tuple[str, float]] = []
    captures: list[float] = []
    for _, group in scored.groupby("decision_group_id", sort=False, observed=True):
        if len(group) < 2:
            continue
        depth = min(config.maximum_candidates_per_decision, len(group))
        actual = group["net_return"].to_numpy(dtype="float64")
        score = group["predicted_net_return"].to_numpy(dtype="float64")
        model_mean = float(actual[np.argsort(-score, kind="stable")[:depth]].mean())
        random_expected = float(actual.mean())
        ideal_mean = float(np.sort(actual)[-depth:].mean())
        gains.append((str(group["session_date_et"].iloc[0]), model_mean - random_expected))
        denominator = ideal_mean - random_expected
        captures.append(
            (model_mean - random_expected) / denominator if denominator > 0.0 else 0.0
        )
    gain_values = np.asarray([value for _, value in gains], dtype="float64")
    session_gains = (
        pd.DataFrame(gains, columns=["session", "gain"])
        .groupby("session", sort=True, observed=True)["gain"]
        .mean()
        .to_numpy(dtype="float64")
        if gains
        else np.asarray([], dtype="float64")
    )
    gain_interval = _moving_block_mean_interval(
        session_gains,
        samples=config.bootstrap_samples,
        block_sessions=config.bootstrap_block_sessions,
        seed=config.random_seed + 211,
    )
    return {
        "ranking_groups": len(gains),
        "economic_rank_gain_over_exact_random_baseline": (
            float(gain_values.mean()) if gains else 0.0
        ),
        "economic_rank_gain_bootstrap_95_ci": gain_interval,
        "economic_rank_capture_ratio": (
            float(np.mean(captures)) if captures else 0.0
        ),
        "random_baseline_method": "exact_cross_sectional_mean_return",
        "raw_ndcg_reported": False,
    }
