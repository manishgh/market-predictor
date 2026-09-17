"""Public swing admission; historical research contracts are not API permissions."""

from __future__ import annotations

from typing import Literal

from market_predictor.core.prediction_contracts import InvestmentReplayRequest, PredictionRequest


class SwingPredictionRequest(PredictionRequest):
    mode: Literal["swing"] = "swing"
    horizon: Literal["auto", "10b"] = "auto"


class SwingInvestmentReplayRequest(InvestmentReplayRequest):
    model_view: Literal["swing"] = "swing"
