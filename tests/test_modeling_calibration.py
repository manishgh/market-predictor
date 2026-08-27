import numpy as np
import pandas as pd

from market_predictor.modeling.calibration import CAUSAL_ISOTONIC_METHOD, CausalCalibrationFit, fit_prior_isotonic


def test_prior_isotonic_returns_constructible_causal_fit() -> None:
    probabilities = np.linspace(0.01, 0.99, 120)
    targets = (probabilities >= 0.50).astype(int)
    availability = pd.Series(pd.date_range("2026-01-01", periods=120, freq="h", tz="UTC"))
    scoring_cutoff = pd.Timestamp("2026-01-07T00:00:00Z")

    result = fit_prior_isotonic(
        probabilities,
        targets,
        availability,
        before_utc=scoring_cutoff,
        min_rows=100,
    )

    assert isinstance(result, CausalCalibrationFit)
    assert result.method == CAUSAL_ISOTONIC_METHOD
    assert result.training_rows == 120
    assert result.train_cutoff_utc < scoring_cutoff
