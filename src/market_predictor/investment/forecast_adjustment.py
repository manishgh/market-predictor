"""Fixed, outcome-independent batch centering and shrinkage of investment forecasts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.diagnostics import PREDICTION
from market_predictor.investment.training import HORIZONS


def centered_shrunk_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Center each complete eligible date roster and blend equally with zero.

    The caller supplies one verified horizon/fold/scope roster, before selecting
    stocks or looking up outcomes. Unknown outcomes participate in the mean.
    This is a fixed batch rule, not a learned calibrator or a single-stock API.
    Fit and derived-generation clocks are owned by the publication caller.
    """
    required = {"decision_id", "security_id", "decision_session", "horizon_sessions", "feature_eligible", PREDICTION}
    if frame.empty or not required.issubset(frame.columns) or not frame.index.is_unique:
        raise DataReadinessError("forecast adjustment requires a nonempty uniquely indexed prediction roster")
    identities = ["decision_id", "security_id", "decision_session", "horizon_sessions"]
    if (frame[identities].isna().any().any() or frame.decision_id.duplicated().any()
            or frame.duplicated(["security_id", "decision_session"]).any()):
        raise DataReadinessError("forecast adjustment identities must be complete and unique")
    if frame.horizon_sessions.nunique() != 1 or frame.horizon_sessions.iloc[0] not in HORIZONS:
        raise DataReadinessError("forecast adjustment must separate supported horizons")
    if "scope" in frame and (frame.scope.isna().any() or frame.scope.nunique() != 1):
        raise DataReadinessError("forecast adjustment must separate scoring scopes")
    eligible = frame.feature_eligible
    if eligible.isna().any() or getattr(eligible.dtype, "kind", None) != "b":
        raise DataReadinessError("forecast eligibility must be explicit boolean values")
    if getattr(frame[PREDICTION].dtype, "kind", None) not in ("i", "u", "f"):
        raise DataReadinessError("forecasts must be numeric, not boolean or text")
    values = frame[PREDICTION].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isinf(values).any() or not np.array_equal(np.isfinite(values), eligible.to_numpy(dtype=bool)):
        raise DataReadinessError("forecast adjustment scoring mask differs from feature eligibility")
    result = frame.copy(deep=True)
    result[PREDICTION] = values.copy()
    if eligible.any():
        scored = result.loc[eligible]
        means = scored.groupby("decision_session", sort=False)[PREDICTION].transform("mean")
        adjusted = 0.5 * (scored[PREDICTION].to_numpy(dtype=np.float64) - means.to_numpy(dtype=np.float64))
        if not np.isfinite(adjusted).all():
            raise DataReadinessError("centered and shrunk forecasts must remain finite")
        result.loc[eligible, PREDICTION] = adjusted
    return result
