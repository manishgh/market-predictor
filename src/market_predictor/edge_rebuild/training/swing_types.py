"""Production-grade candidate training for the ten-session edge-rebuild swing strategy."""
from __future__ import annotations



import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from market_predictor.edge_rebuild.swing_features import (
    SWING_BASELINE_ABLATION_ORDER,
    SWING_FEATURE_PROFILE,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
)
from market_predictor.core.errors import DataReadinessError

TRAINING_SCHEMA: Final = "edge_rebuild.swing_training.v5"
MODEL_SCHEMA: Final = "edge_rebuild.swing_candidate.v5"
EVALUATION_SCHEMA: Final = "edge_rebuild.swing_evaluation.v7"
MODEL_CARD_SCHEMA: Final = "edge_rebuild.swing_model_card.v7"
OUTPUT_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_candidate_authority.v5"
SWING_BASELINE_BUNDLE_PREFIX: Final = "swing_baseline_bundle."
DECISION_START_DATE: Final = date(2019, 7, 9)
HORIZON_SESSIONS: Final = 10
ALLOWED_PROFILES: Final = (
    SWING_FEATURE_PROFILE,
)
# The learned families, per profile and per (rate, depth) point. `dual_hurdle`
# was dropped: it scored 0.452-0.462 AUC on the v12 run -- below chance -- had no
# test covering it, and its four slots pushed the grid past the contract's
# six-candidate experiment budget.
_XGB_GRID: Final = (
    ("xgbranker", "xgboost_ranker"),
    ("xgbregressor", "xgboost_regressor"),
)
_XGB_FAMILIES: Final = len(_XGB_GRID)
_MANIFEST_NAME: Final = "_manifest.json"
_AUTHORITY_NAME: Final = "_authority.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_TEXT_COLUMNS: Final = (
    "decision_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "sector",
    "primary_benchmark",
    "market_regime",
)


@dataclass(frozen=True, slots=True)
class SwingTrainingConfig:
    """Frozen controls for sequential candidate fitting and temporal evaluation."""

    decision_start_date: str = "2019-07-09"
    horizon_sessions: int = 10
    calibration_fraction: float = 0.20
    minimum_calibration_sessions: int = 63
    minimum_rows: int = 100_000
    minimum_securities: int = 100
    maximum_trades_per_decision: int = 25
    probability_thresholds: tuple[float, ...] = (
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
    )
    logistic_c_values: tuple[float, ...] = (1.0,)
    xgb_learning_rates: tuple[float, ...] = (0.05,)
    xgb_n_estimators: int = 150
    # Tree depth is a policy field rather than a literal in the grid builder. It
    # was hard-coded to (3, 5), which put half the candidate count outside the
    # budget arithmetic below and let the real grid grow to fourteen while the
    # constructor still reported six.
    xgb_max_depths: tuple[int, ...] = (3,)
    maximum_learned_candidates: int = 6
    bootstrap_samples: int = 2_000
    bootstrap_block_sessions: int = 20
    random_seed: int = 42
    expected_round_trip_cost_bps: float = 20.0
    maximum_process_memory_gib: float = 5.0
    memory_guard_headroom_gib: float = 0.75

    def __post_init__(self) -> None:
        if self.decision_start_date != DECISION_START_DATE.isoformat():
            raise ValueError("the frozen swing decision start is 2019-07-09")
        if self.horizon_sessions != HORIZON_SESSIONS:
            raise ValueError("the active swing strategy has an exact ten-session horizon")
        if not 0.10 <= self.calibration_fraction <= 0.35:
            raise ValueError("calibration_fraction must be between 0.10 and 0.35")
        if self.minimum_calibration_sessions < 20:
            raise ValueError("calibration requires at least twenty sessions")
        if self.minimum_rows < 1 or self.minimum_securities < 20:
            raise ValueError("training population minimums are invalid")
        if not 1 <= self.maximum_trades_per_decision <= 50:
            raise ValueError("maximum_trades_per_decision must be in [1, 50]")
        if not self.probability_thresholds or any(
            value <= 0.0 or value >= 1.0 for value in self.probability_thresholds
        ):
            raise ValueError("probability thresholds must be in (0, 1)")
        if tuple(sorted(set(self.probability_thresholds))) != self.probability_thresholds:
            raise ValueError("probability thresholds must be unique and ascending")
        if not self.logistic_c_values or any(value <= 0 for value in self.logistic_c_values):
            raise ValueError("logistic C values must be positive")
        if not self.xgb_learning_rates or any(
            value <= 0 for value in self.xgb_learning_rates
        ):
            raise ValueError("xgboost learning rates must be positive")
        if self.xgb_n_estimators < 1:
            raise ValueError("xgboost estimator count must be positive")
        if not self.xgb_max_depths or any(value < 1 for value in self.xgb_max_depths):
            raise ValueError("xgb max depths must be at least 1")
        # Four nested logistic feature ablations plus two full-feature tree
        # families are the preregistered swing-baseline experiment budget.
        candidate_count = (
            len(SWING_BASELINE_ABLATION_ORDER) * len(self.logistic_c_values)
            + len(self.xgb_learning_rates)
            * len(self.xgb_max_depths)
            * _XGB_FAMILIES
        )
        if candidate_count > self.maximum_learned_candidates:
            raise ValueError("candidate grid exceeds the frozen sequential budget")
        if not 2_000 <= self.bootstrap_samples <= 10_000:
            raise ValueError("bootstrap_samples must be in [2000, 10000]")
        if not HORIZON_SESSIONS <= self.bootstrap_block_sessions <= 126:
            raise ValueError("bootstrap blocks must span 10 to 126 sessions")
        if self.expected_round_trip_cost_bps <= 0:
            raise ValueError("expected round-trip cost must be positive")
        if not 0 < self.maximum_process_memory_gib <= 5.0:
            raise ValueError("process memory hard limit must be in (0, 5] GiB")
        if not 0 < self.memory_guard_headroom_gib < self.maximum_process_memory_gib:
            raise ValueError("memory headroom must be below the hard limit")




@dataclass(frozen=True, slots=True)
class SwingPanelBinding:
    root: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    authority_sha256: str
    request_sha256: str
    strategy_contract_sha256: str


@dataclass(frozen=True, slots=True)
class SwingProfileData:
    frame: pd.DataFrame
    profile: str
    feature_columns: tuple[str, ...]
    decision_ids_sha256: str
    panel: SwingPanelBinding


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    profile: str
    feature_group: str
    feature_columns: tuple[str, ...]
    estimator_family: str
    hyperparameters: Mapping[str, float | int | str]


@dataclass(frozen=True, slots=True)
class FittedCandidate:
    estimator: Any
    calibrator: LogisticRegression
    feature_columns: tuple[str, ...]
    fit_sessions: int
    calibration_sessions: int
    calibration_cutoff_utc: str


@dataclass(frozen=True, slots=True)
class SwingTrainingResult:
    output_directory: Path
    selected_candidate_id: str | None
    evaluation: Mapping[str, Any]
    model_card: Mapping[str, Any]




























































def _guard(config: SwingTrainingConfig, stage: str, *, peak: bool) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    if peak:
        assert_peak_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage=stage,
        )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )



def _resolve_inside(root: Path, raw: object) -> Path:
    path = (root / str(raw)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DataReadinessError(f"artifact escapes authority root: {raw}") from exc
    if not path.is_file():
        raise DataReadinessError(f"authority artifact is missing: {path}")
    return path


def _strict_bool(value: object) -> bool:
    return value is True or isinstance(value, np.bool_) and bool(value)


def _is_unapproved_source_feature(value: str) -> bool:
    normalized = value.lower()
    return any(
        token in normalized
        for token in (
            "source_count_sec_",
            "sec_filing",
            "source_count_finviz_",
            "finviz_news",
            "global_context",
            "gdelt",
            "reddit",
            "seeking_alpha",
        )
    )



def _sequence_sha256(values: Sequence[str] | pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _iso(value: object) -> str:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return cast(str, parsed.tz_convert("UTC").isoformat())
