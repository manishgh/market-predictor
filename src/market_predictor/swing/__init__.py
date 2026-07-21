"""Canonical daily swing feature, label, validation, and model pipeline."""

from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
    SwingDatasetConfig,
    SwingPromotionConfig,
    SwingTrainingConfig,
)
from market_predictor.swing.dataset import build_swing_dataset
from market_predictor.swing.model import score_swing_frame, train_swing_model
from market_predictor.swing.promotion import (
    SwingPromotionEvidence,
    load_swing_training_evidence,
    promote_swing_model,
    promotion_evidence_from_result,
)

__all__ = [
    "SWING_FEATURE_SCHEMA_VERSION",
    "SWING_MODEL_SCHEMA_VERSION",
    "SWING_MODEL_TYPE",
    "SwingDatasetConfig",
    "SwingPromotionConfig",
    "SwingPromotionEvidence",
    "SwingTrainingConfig",
    "build_swing_dataset",
    "load_swing_training_evidence",
    "promote_swing_model",
    "promotion_evidence_from_result",
    "score_swing_frame",
    "train_swing_model",
]
