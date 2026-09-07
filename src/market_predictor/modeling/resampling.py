"""Time-block uncertainty estimates for complete, calendar-aligned return series."""
from __future__ import annotations

import math

import numpy as np

from market_predictor.core.errors import DataReadinessError


def moving_block_mean_interval(
    values: np.ndarray,
    samples: int,
    block_sessions: int,
    seed: int,
    *,
    lower_tail_probability: float = 0.025,
) -> dict[str, float | int]:
    finite = np.asarray(values, dtype="float64")
    if finite.ndim != 1 or not np.isfinite(finite).all():
        raise DataReadinessError("bootstrap requires a complete finite daily series; missing dates cannot be dropped")
    if not 0 < lower_tail_probability < 0.5 or samples < 1 or block_sessions < 1:
        raise DataReadinessError("invalid moving-block bootstrap policy")
    if len(finite) < block_sessions:
        raise DataReadinessError("moving-block bootstrap has fewer sessions than one block")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype="float64")
    maximum_start = len(finite) - block_sessions
    block_count = math.ceil(len(finite) / block_sessions)
    for index in range(samples):
        starts = rng.integers(0, maximum_start + 1, size=block_count)
        sampled = np.concatenate(
            [finite[start : start + block_sessions] for start in starts]
        )[: len(finite)]
        means[index] = float(sampled.mean())
    return {
        "estimate": float(finite.mean()),
        "low": float(np.quantile(means, lower_tail_probability)),
        "high": float(np.quantile(means, 1.0 - lower_tail_probability)),
        "lower_tail_probability": lower_tail_probability,
        "sessions": len(finite),
        "bootstrap_samples": samples,
        "block_sessions": block_sessions,
    }
