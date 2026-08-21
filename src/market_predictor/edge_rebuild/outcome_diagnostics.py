"""Shared binary-outcome and negative-control diagnostics."""
from __future__ import annotations



from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from market_predictor.core.errors import DataReadinessError

OUTCOME_DIAGNOSTIC_SCHEMA: Final = "edge_rebuild.outcome_diagnostics.v1"


def binary_outcome_diagnostic(
    outcome: Sequence[object] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
    *,
    definition: str,
) -> dict[str, Any]:
    """Measure one named binary outcome against a finite ordering score."""

    labels, scores = _validated_vectors(outcome, score)
    has_two_classes = np.unique(labels).size == 2
    return {
        "schema": OUTCOME_DIAGNOSTIC_SCHEMA,
        "definition": definition,
        "rows": int(len(labels)),
        "positive_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)) if has_two_classes else None,
        "pr_auc": (
            float(average_precision_score(labels, scores))
            if has_two_classes
            else None
        ),
    }


def label_permutation_control(
    outcome: Sequence[object] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
    *,
    random_seed: int,
    repetitions: int = 64,
) -> dict[str, Any]:
    """Verify that globally shuffled labels produce chance discrimination.

    Score ranks are calculated once, making the control linear in rows per
    repetition. The four-standard-error limit detects systematic non-chance
    behavior without treating ordinary Monte Carlo variation as a failure.
    """

    labels, scores = _validated_vectors(outcome, score)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives < 1 or negatives < 1:
        return {
            "schema": OUTCOME_DIAGNOSTIC_SCHEMA,
            "name": "global_label_permutation_auc",
            "repetitions": 0,
            "random_seed": random_seed,
            "status": "not_applicable_single_class",
            "expected_auc": 0.5,
            "mean_auc": None,
            "standard_error": None,
            "acceptance_tolerance": None,
            "minimum_auc": None,
            "maximum_auc": None,
            "passed": None,
        }
    if repetitions < 32:
        raise ValueError("label-permutation control requires at least 32 repetitions")

    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype="float64")
    baseline = positives * (positives + 1) / 2.0
    denominator = float(positives * negatives)
    generator = np.random.default_rng(random_seed)
    aucs = np.empty(repetitions, dtype="float64")
    for index in range(repetitions):
        shuffled = generator.permutation(labels)
        aucs[index] = (float(ranks[shuffled == 1].sum()) - baseline) / denominator

    mean_auc = float(aucs.mean())
    standard_error = float(aucs.std(ddof=1) / np.sqrt(repetitions))
    tolerance = max(0.02, 4.0 * standard_error)
    passed = abs(mean_auc - 0.5) <= tolerance
    if not passed:
        raise DataReadinessError(
            "label-permutation negative control retained abnormal discrimination: "
            f"mean_auc={mean_auc:.6f}, tolerance={tolerance:.6f}"
        )
    return {
        "schema": OUTCOME_DIAGNOSTIC_SCHEMA,
        "name": "global_label_permutation_auc",
        "repetitions": repetitions,
        "random_seed": random_seed,
        "status": "passed",
        "expected_auc": 0.5,
        "mean_auc": mean_auc,
        "standard_error": standard_error,
        "acceptance_tolerance": tolerance,
        "minimum_auc": float(aucs.min()),
        "maximum_auc": float(aucs.max()),
        "passed": True,
    }


def _validated_vectors(
    outcome: Sequence[object] | np.ndarray | pd.Series,
    score: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    labels = pd.to_numeric(pd.Series(outcome), errors="coerce").to_numpy(
        dtype="float64"
    )
    scores = pd.to_numeric(pd.Series(score), errors="coerce").to_numpy(
        dtype="float64"
    )
    if len(labels) < 1 or len(labels) != len(scores):
        raise DataReadinessError("outcome and score vectors must be non-empty and aligned")
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise DataReadinessError("outcome and score vectors must be finite")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise DataReadinessError("binary outcome must contain only zero and one")
    return labels.astype("int8"), scores
