from __future__ import annotations

import math
from typing import Any, cast

import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_training import (
    HORIZON_SESSIONS,
    SwingTrainingConfig,
    WalkForwardFold,
    _ordered_sessions,
)
from market_predictor.edge_rebuild.temporal_manifest import (
    TemporalManifestConfig,
    TemporalSchedule,
)
from market_predictor.v3.errors import DataReadinessError


def _governed_folds(schedule: TemporalSchedule) -> tuple[WalkForwardFold, ...]:
    return tuple(
        WalkForwardFold(
            fold=fold.fold,
            train_sessions=tuple(value.isoformat() for value in fold.train_sessions),
            purge_sessions=(),
            embargo_sessions=tuple(value.isoformat() for value in fold.embargo_sessions),
            validation_sessions=tuple(
                value.isoformat() for value in fold.validation_sessions
            ),
        )
        for fold in schedule.folds
    )



def _governed_model_sessions(schedule: TemporalSchedule) -> tuple[str, ...]:
    sessions = {
        value.isoformat()
        for fold in schedule.folds
        for value in (*fold.train_sessions, *fold.validation_sessions)
    }
    return tuple(sorted(sessions))



def _split_record(
    *,
    folds: tuple[WalkForwardFold, ...],
    schedule: TemporalSchedule,
    temporal_config: TemporalManifestConfig,
    temporal_policy_sha256: str,
    strategy_contract: StrategyContract,
) -> dict[str, Any]:
    final_refit_sessions = tuple(
        value.isoformat() for value in schedule.final_refit_sessions
    )
    final_test_sessions = tuple(
        value.isoformat() for value in schedule.locked_test_sessions
    )
    final_embargo_sessions = tuple(
        value.isoformat() for value in schedule.final_embargo_sessions
    )
    return {
        "method": "governed_exact_xnys_temporal_manifest",
        "boundary_policy": "explicit_dates_with_verified_xnys_counts",
        "window_rationale": (
            "approximately 4.9 years initial fit plus one validation year plus "
            "one locked test year; causal-news cutoff is authoritative"
        ),
        "split_unit": "whole_session_date_et",
        "random_or_row_split_used": False,
        "temporal_manifest_schema": temporal_config.schema_version,
        "temporal_manifest_config_sha256": temporal_config.sha256(),
        "temporal_manifest_policy_sha256": temporal_policy_sha256,
        "security_holdout_fraction": (
            strategy_contract.validation.unseen_ticker_holdout_fraction
        ),
        "security_holdout_assignment": (
            strategy_contract.validation.unseen_ticker_assignment
        ),
        "security_holdout_identity": "security_id",
        "purge_sessions": 0,
        "modeled_decision_start": temporal_config.modeled_decision_start.isoformat(),
        "embargo_sessions": len(final_embargo_sessions),
        "final_refit_sessions": len(final_refit_sessions),
        "first_final_refit_session": final_refit_sessions[0],
        "last_final_refit_session": final_refit_sessions[-1],
        "locked_final_test_sessions": len(final_test_sessions),
        "final_embargo_session_dates": list(final_embargo_sessions),
        "first_final_test_session": final_test_sessions[0],
        "last_final_test_session": final_test_sessions[-1],
        "folds": [
            {
                "fold": fold.fold,
                "train_sessions": len(fold.train_sessions),
                "first_train_session": fold.train_sessions[0],
                "last_train_session": fold.train_sessions[-1],
                "purge_session_dates": list(fold.purge_sessions),
                "embargo_session_dates": list(fold.embargo_sessions),
                "validation_sessions": len(fold.validation_sessions),
                "first_validation_session": fold.validation_sessions[0],
                "last_validation_session": fold.validation_sessions[-1],
            }
            for fold in folds
        ],
    }



def _split_fit_calibration(
    train: pd.DataFrame, config: SwingTrainingConfig
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sessions = _ordered_sessions(train)
    calibration_count = max(
        config.minimum_calibration_sessions,
        math.ceil(len(sessions) * config.calibration_fraction),
    )
    calibration_start = len(sessions) - calibration_count
    fit_end = calibration_start - HORIZON_SESSIONS
    if fit_end < 20:
        raise DataReadinessError("training fold is too short for calibration embargo")
    fit_sessions = sessions[:fit_end]
    calibration_sessions = sessions[calibration_start:]
    fit = train.loc[
        train["session_date_et"].isin(fit_sessions),
        ["decision_id", "session_date_et", "decision_time_utc", "label_available_at_utc"],
    ]
    calibration = train.loc[
        train["session_date_et"].isin(calibration_sessions),
        ["decision_id", "session_date_et", "decision_time_utc", "label_available_at_utc"],
    ]
    _assert_label_purge(fit, calibration, "fit/calibration")
    return fit_sessions, calibration_sessions



def _assert_label_purge(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    if left.empty or right.empty:
        raise DataReadinessError(f"{label} partition is empty")
    if cast(pd.Timestamp, left["label_available_at_utc"].max()) >= cast(
        pd.Timestamp, right["decision_time_utc"].min()
    ):
        raise DataReadinessError(f"{label} is not causally purged")
    if set(left["session_date_et"]).intersection(right["session_date_et"]):
        raise DataReadinessError(f"{label} has exchange-session overlap")
    if set(left["decision_id"]).intersection(right["decision_id"]):
        raise DataReadinessError(f"{label} has decision-row overlap")



