from __future__ import annotations

import numpy as np
import pytest

from market_predictor.edge_rebuild.outcome_diagnostics import (
    binary_outcome_diagnostic,
    label_permutation_control,
)
from market_predictor.core.errors import DataReadinessError


def test_binary_outcome_diagnostic_names_the_measured_event() -> None:
    result = binary_outcome_diagnostic(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        definition="managed net return after costs is positive",
    )

    assert result["definition"] == "managed net return after costs is positive"
    assert result["positive_rate"] == pytest.approx(0.5)
    assert result["roc_auc"] == pytest.approx(1.0)


def test_label_permutation_control_is_deterministic_and_near_chance() -> None:
    labels = np.tile([0, 1], 250)
    scores = np.linspace(-1.0, 1.0, len(labels))

    first = label_permutation_control(labels, scores, random_seed=17)
    second = label_permutation_control(labels, scores, random_seed=17)

    assert first == second
    assert first["passed"] is True
    assert abs(first["mean_auc"] - 0.5) <= first["acceptance_tolerance"]


def test_outcome_diagnostics_reject_invalid_vectors() -> None:
    with pytest.raises(DataReadinessError, match="zero and one"):
        binary_outcome_diagnostic([0, 2], [0.1, 0.2], definition="invalid")
    unavailable = label_permutation_control([1, 1], [0.1, 0.2], random_seed=1)
    assert unavailable["status"] == "not_applicable_single_class"
    assert unavailable["passed"] is None
