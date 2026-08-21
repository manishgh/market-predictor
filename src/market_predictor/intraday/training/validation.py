from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_predictor.intraday.training.config import IntradayDevelopmentConfig

"""Development-only, cost-aware intraday model training and evaluation."""

import hashlib
from dataclasses import dataclass
from typing import Final

import pandas as pd

from market_predictor.core.errors import DataReadinessError

MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_candidate.v1"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_evaluation.v1"
AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_authority.v1"
FUTURE_EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_evaluation.v1"
FUTURE_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_authority.v1"
_AUTHORITY_NAME: Final = "_authority.json"
_MANIFEST_NAME: Final = "_manifest.json"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_FUTURE_EVALUATION_NAME: Final = "future_evaluation.json"
_POSITION_LEDGER_NAME: Final = "position_ledger.parquet"
_DAILY_LEDGER_NAME: Final = "daily_ledger.parquet"
_VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.parquet"


@dataclass(frozen=True, slots=True)
class _Fold:
    fold: int
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    embargo_sessions: tuple[str, ...]


def _walk_forward_folds(
    data: pd.DataFrame,
    sessions: tuple[str, ...],
    config: IntradayDevelopmentConfig,
) -> tuple[_Fold, ...]:
    remaining = len(sessions) - config.minimum_train_sessions
    fold_size = remaining // config.validation_folds
    if fold_size < config.minimum_validation_sessions:
        raise DataReadinessError("development history is too short for walk-forward validation")
    folds: list[_Fold] = []
    for index in range(config.validation_folds):
        validation_start = config.minimum_train_sessions + index * fold_size
        validation_end = len(sessions) if index == config.validation_folds - 1 else validation_start + fold_size
        embargo_start = validation_start - config.embargo_sessions
        validation = sessions[validation_start:validation_end]
        train_candidates = sessions[:embargo_start]
        first_validation = data.loc[data["session_date_et"].eq(validation[0]), "decision_time_utc"].min()
        safe_train = tuple(
            session
            for session in train_candidates
            if data.loc[data["session_date_et"].eq(session), "label_available_at_utc"].max() < first_validation
        )
        if len(safe_train) < 2:
            raise DataReadinessError(f"fold {index} has insufficient purged training history")
        folds.append(
            _Fold(
                fold=index,
                train_sessions=safe_train,
                validation_sessions=validation,
                embargo_sessions=sessions[embargo_start:validation_start],
            )
        )
    return tuple(folds)


def _stable_security_holdout(
    data: pd.DataFrame,
    fraction: float,
) -> frozenset[str]:
    securities = sorted(set(data["security_id"].astype(str)))
    threshold = int(fraction * 2**64)
    selected = frozenset(
        security for security in securities if int(hashlib.sha256(security.encode("utf-8")).hexdigest()[:16], 16) < threshold
    )
    if not selected or len(selected) == len(securities):
        raise DataReadinessError("stable security holdout produced an empty partition")
    return selected


def _security_set_sha256(securities: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(securities)).encode("utf-8")).hexdigest()
