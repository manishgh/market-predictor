"""Canonical intraday feature, exact-path label, validation, and model pipeline."""

from market_predictor.intraday.contracts import (
    INTRADAY_FEATURE_SCHEMA_VERSION,
    INTRADAY_MODEL_SCHEMA_VERSION,
    INTRADAY_MODEL_TYPE,
    IntradayDatasetConfig,
    IntradayPromotionConfig,
    IntradayTrainingConfig,
)
from market_predictor.intraday.dataset import build_intraday_dataset
from market_predictor.intraday.model import score_intraday_frame, train_intraday_model
from market_predictor.intraday.promotion import (
    IntradayPromotionEvidence,
    load_intraday_training_evidence,
    promote_intraday_model,
    promotion_evidence_from_result,
)

__all__ = [
    "INTRADAY_FEATURE_SCHEMA_VERSION",
    "INTRADAY_MODEL_SCHEMA_VERSION",
    "INTRADAY_MODEL_TYPE",
    "IntradayDatasetConfig",
    "IntradayPromotionEvidence",
    "IntradayPromotionConfig",
    "IntradayTrainingConfig",
    "build_intraday_dataset",
    "load_intraday_training_evidence",
    "promote_intraday_model",
    "promotion_evidence_from_result",
    "score_intraday_frame",
    "train_intraday_model",
]
