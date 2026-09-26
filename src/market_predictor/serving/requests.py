"""Public swing admission; historical research contracts are not API permissions."""

from __future__ import annotations

from typing import Literal

from market_predictor.core.prediction_contracts import PredictionRequest


class SwingPredictionRequest(PredictionRequest):
    """The public API serves only the ten-session swing route."""

    horizon: Literal["auto", "10b"] = "auto"
