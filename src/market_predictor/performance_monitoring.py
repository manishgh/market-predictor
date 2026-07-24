from __future__ import annotations

import json
import math
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Self, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.outcome_contracts import (
    MaturedOutcomeV1,
    PredictionMaturationIntentV2,
    content_sha256,
)
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.v3.errors import DataReadinessError

PERFORMANCE_REPORT_VERSION = "market_predictor.selected_policy_performance.v2"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTITY_COLUMNS = [
    "model_release_id",
    "model_artifact_sha256",
    "prediction_policy_sha256",
    "label_policy_sha256",
    "execution_policy_sha256",
    "view",
    "horizon",
]


class SelectedPolicyCohortV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    cohort_id: str = Field(pattern=SHA256_PATTERN)
    model_release_id: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    prediction_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    label_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_artifact_set_sha256: str = Field(pattern=SHA256_PATTERN)
    source_intent_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    source_outcome_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    view: Literal["swing", "intraday"]
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d)$")
    cohort_type: Literal[
        "all",
        "market_regime",
        "sector",
        "market_cap_bucket",
        "liquidity_bucket",
        "calibration_bin",
    ]
    cohort_value: str = Field(min_length=1, max_length=128)
    window_start_utc: datetime
    window_end_utc: datetime
    total_predictions: int = Field(ge=1)
    eligible_predictions: int = Field(ge=0)
    selected_predictions: int = Field(ge=0)
    actionable_predictions: int = Field(ge=0)
    matured_selected_samples: int = Field(ge=0)
    pending_selected_samples: int = Field(ge=0)
    independent_decision_groups: int = Field(ge=0)
    evidence_status: Literal["sufficient", "insufficient_evidence"]
    selection_rate: float = Field(ge=0, le=1)
    actionable_rate: float = Field(ge=0, le=1)
    mean_probability: float | None = Field(default=None, ge=0, le=1)
    probability_p10: float | None = Field(default=None, ge=0, le=1)
    probability_p50: float | None = Field(default=None, ge=0, le=1)
    probability_p90: float | None = Field(default=None, ge=0, le=1)
    mean_decision_score: float | None = None
    decision_score_p10: float | None = None
    decision_score_p50: float | None = None
    decision_score_p90: float | None = None
    mean_selected_rank: float | None = Field(default=None, ge=1)
    selected_rank_p90: float | None = Field(default=None, ge=1)
    opportunity_observed_rate: float | None = Field(default=None, ge=0, le=1)
    opportunity_brier_score: float | None = Field(default=None, ge=0, le=1)
    opportunity_calibration_error: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    mean_downside_probability: float | None = Field(default=None, ge=0, le=1)
    downside_observed_rate: float | None = Field(default=None, ge=0, le=1)
    downside_brier_score: float | None = Field(default=None, ge=0, le=1)
    downside_calibration_error: float | None = Field(default=None, ge=0, le=1)
    average_net_return: float | None = None
    average_excess_return_vs_spy: float | None = None
    cumulative_net_return: float | None = None
    win_rate: float | None = Field(default=None, ge=0, le=1)
    max_drawdown: float | None = Field(default=None, ge=0)
    first_decision_time_utc: datetime
    last_decision_time_utc: datetime
    last_matured_outcome_utc: datetime | None = None

    @field_validator(
        "window_start_utc",
        "window_end_utc",
        "first_decision_time_utc",
        "last_decision_time_utc",
        "last_matured_outcome_utc",
    )
    @classmethod
    def aware_times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() is None:
            raise ValueError(
                "selected-policy performance timestamps must be timezone-aware"
            )
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if not (
            self.actionable_predictions
            <= self.selected_predictions
            <= self.eligible_predictions
            <= self.total_predictions
        ):
            raise ValueError("selected-policy cohort counts are inconsistent")
        if (
            self.matured_selected_samples + self.pending_selected_samples
            != self.actionable_predictions
        ):
            raise ValueError("selected-policy maturation counts are inconsistent")
        if self.window_start_utc >= self.window_end_utc:
            raise ValueError("selected-policy report window is invalid")
        if not (
            self.window_start_utc
            <= self.first_decision_time_utc
            <= self.last_decision_time_utc
            <= self.window_end_utc
        ):
            raise ValueError("selected-policy decision times are outside the window")
        if not math.isclose(
            self.selection_rate,
            self.selected_predictions / self.total_predictions,
            abs_tol=1e-12,
        ):
            raise ValueError("selected-policy selection rate is inconsistent")
        if not math.isclose(
            self.actionable_rate,
            self.actionable_predictions / self.total_predictions,
            abs_tol=1e-12,
        ):
            raise ValueError("selected-policy actionable rate is inconsistent")
        probability_fields = (
            self.mean_probability,
            self.probability_p10,
            self.probability_p50,
            self.probability_p90,
            self.mean_decision_score,
            self.decision_score_p10,
            self.decision_score_p50,
            self.decision_score_p90,
            self.mean_selected_rank,
            self.selected_rank_p90,
        )
        if self.actionable_predictions == 0 and any(
            value is not None for value in probability_fields
        ):
            raise ValueError("empty selected-policy cohort has score evidence")
        outcome_fields = (
            self.opportunity_observed_rate,
            self.opportunity_brier_score,
            self.opportunity_calibration_error,
            self.average_net_return,
            self.average_excess_return_vs_spy,
            self.cumulative_net_return,
            self.win_rate,
            self.max_drawdown,
            self.last_matured_outcome_utc,
        )
        if self.matured_selected_samples == 0:
            if any(value is not None for value in outcome_fields):
                raise ValueError("unmatured selected-policy cohort has outcome evidence")
        elif any(value is None for value in outcome_fields):
            raise ValueError("matured selected-policy cohort lacks outcome evidence")
        downside_fields = (
            self.mean_downside_probability,
            self.downside_observed_rate,
            self.downside_brier_score,
            self.downside_calibration_error,
        )
        if self.view == "intraday" and self.matured_selected_samples > 0:
            if any(value is None for value in downside_fields):
                raise ValueError("intraday cohort lacks downside calibration evidence")
        elif any(value is not None for value in downside_fields):
            raise ValueError("swing or unmatured cohort has downside evidence")
        content = self.model_dump(mode="json", exclude={"cohort_id"})
        if content_sha256(content) != self.cohort_id:
            raise ValueError("selected-policy cohort identity is invalid")
        return self


class SelectedPolicyPerformanceReportV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal[
        "market_predictor.selected_policy_performance.v2"
    ] = "market_predictor.selected_policy_performance.v2"
    report_id: str = Field(pattern=SHA256_PATTERN)
    generated_at_utc: datetime
    lookback_days: int = Field(ge=1)
    minimum_matured_samples: int = Field(ge=1)
    window_start_utc: datetime
    window_end_utc: datetime
    source_intent_ids: tuple[str, ...]
    source_outcome_ids: tuple[str, ...]
    rows: tuple[SelectedPolicyCohortV2, ...]

    @field_validator(
        "generated_at_utc",
        "window_start_utc",
        "window_end_utc",
    )
    @classmethod
    def aware_report_times(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError(
                "selected-policy report timestamp must be timezone-aware"
            )
        return value.astimezone(UTC)

    @field_validator("source_intent_ids", "source_outcome_ids")
    @classmethod
    def canonical_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value
        ):
            raise ValueError("selected-policy source identity is invalid")
        if tuple(sorted(set(value))) != value:
            raise ValueError(
                "selected-policy source identities must be unique and sorted"
            )
        return value

    @model_validator(mode="after")
    def validate_report_identity(self) -> Self:
        if (
            self.window_end_utc != self.generated_at_utc
            or self.window_start_utc
            != self.window_end_utc - timedelta(days=self.lookback_days)
        ):
            raise ValueError("selected-policy report window is inconsistent")
        content = self.model_dump(mode="json", exclude={"report_id"})
        if content_sha256(content) != self.report_id:
            raise ValueError("selected-policy report identity is invalid")
        return self


def build_performance_cohorts(
    repository: OutcomeRepository,
    *,
    generated_at: datetime | None = None,
    minimum_samples: int = 30,
    lookback_days: int = 60,
) -> dict[str, object]:
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    generated = _utc(generated_at or datetime.now(UTC))
    window_start = generated - timedelta(days=lookback_days)
    records: list[dict[str, object]] = []
    source_intent_ids: set[str] = set()
    source_outcome_ids: set[str] = set()
    for intent in repository.intents():
        if (
            repository.semantic_canonical_key(intent.semantic_prediction_id)
            != intent.maturation_key
            or not window_start <= intent.decision_time_utc <= generated
        ):
            continue
        outcome = _matured_selected_outcome(
            repository,
            intent,
            generated_at=generated,
        )
        source_intent_ids.add(intent.maturation_key)
        if outcome is not None:
            source_outcome_ids.add(outcome.outcome_id)
        records.append(_monitoring_record(intent, outcome))
    frame = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    if not frame.empty:
        for column in (
            "decision_time_utc",
            "matured_at_utc",
            "exit_time_utc",
        ):
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        cohort_specs = [
            ("all", None),
            ("market_regime", "market_regime"),
            ("sector", "sector"),
            ("market_cap_bucket", "market_cap_bucket"),
            ("liquidity_bucket", "liquidity_bucket"),
            ("calibration_bin", "calibration_bin"),
        ]
        for cohort_type, cohort_column in cohort_specs:
            group_columns = [
                *_IDENTITY_COLUMNS,
                *([cohort_column] if cohort_column else []),
            ]
            for group_values, group in frame.groupby(
                group_columns,
                dropna=False,
                sort=True,
            ):
                values = (
                    group_values
                    if isinstance(group_values, tuple)
                    else (group_values,)
                )
                identity = dict(zip(group_columns, values, strict=True))
                row = _cohort_row(
                    group,
                    identity=identity,
                    cohort_type=cohort_type,
                    cohort_value=(
                        "all"
                        if cohort_column is None
                        else str(identity[cohort_column])
                    ),
                    minimum_samples=minimum_samples,
                    window_start=window_start,
                    window_end=generated,
                )
                rows.append(
                    SelectedPolicyCohortV2.model_validate(row).model_dump(
                        mode="json"
                    )
                )
    rows.sort(
        key=lambda row: (
            str(row["model_release_id"]),
            str(row["view"]),
            str(row["horizon"]),
            str(row["cohort_type"]),
            str(row["cohort_value"]),
        )
    )
    report_identity: dict[str, object] = {
        "contract_version": PERFORMANCE_REPORT_VERSION,
        "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "lookback_days": lookback_days,
        "minimum_matured_samples": minimum_samples,
        "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
        "window_end_utc": generated.isoformat().replace("+00:00", "Z"),
        "source_intent_ids": sorted(source_intent_ids),
        "source_outcome_ids": sorted(source_outcome_ids),
        "rows": rows,
    }
    report = SelectedPolicyPerformanceReportV2.model_validate(
        {
            **report_identity,
            "report_id": content_sha256(report_identity),
        }
    )
    return report.model_dump(mode="json")


def validate_performance_report(value: object) -> dict[str, object]:
    report = SelectedPolicyPerformanceReportV2.model_validate(value)
    return report.model_dump(mode="json")


def load_performance_report(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return validate_performance_report(loaded)


def write_performance_report(
    path: Path,
    report: dict[str, object],
) -> dict[str, object]:
    validated = validate_performance_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(validated, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return validated


def _matured_selected_outcome(
    repository: OutcomeRepository,
    intent: PredictionMaturationIntentV2,
    *,
    generated_at: datetime,
) -> MaturedOutcomeV1 | None:
    if not intent.actionable:
        return None
    if not repository.has_outcome(intent.maturation_key):
        return None
    outcome = repository.load_outcome(intent.maturation_key)
    if outcome.matured_at_utc > generated_at:
        return None
    if (
        outcome.maturation_key != intent.maturation_key
        or outcome.semantic_prediction_id != intent.semantic_prediction_id
        or outcome.snapshot_id != intent.snapshot_id
        or outcome.ticker != intent.ticker
        or outcome.view != intent.view
        or outcome.horizon != intent.horizon
    ):
        raise DataReadinessError(
            "selected-policy outcome identity does not match its intent"
        )
    return outcome


def _monitoring_record(
    intent: PredictionMaturationIntentV2,
    outcome: MaturedOutcomeV1 | None,
) -> dict[str, object]:
    if intent.view == "intraday":
        downside = cast(float, intent.downside_probability)
        decision_score = intent.probability * (1.0 - downside)
    else:
        decision_score = intent.probability
    return {
        "maturation_key": intent.maturation_key,
        "outcome_id": outcome.outcome_id if outcome is not None else None,
        "model_release_id": intent.model_release_id,
        "model_artifact_sha256": intent.model_artifact_sha256,
        "feature_artifact_sha256": intent.feature_artifact_sha256,
        "prediction_policy_sha256": intent.prediction_policy_sha256,
        "label_policy_sha256": intent.label_policy_sha256,
        "execution_policy_sha256": intent.execution_policy_sha256,
        "view": intent.view,
        "horizon": intent.horizon,
        "market_regime": intent.market_regime,
        "sector": intent.sector,
        "market_cap_bucket": intent.market_cap_bucket,
        "liquidity_bucket": intent.liquidity_bucket,
        "calibration_bin": intent.calibration_bin,
        "decision_group_id": intent.decision_group_id,
        "decision_time_utc": intent.decision_time_utc,
        "probability": intent.probability,
        "downside_probability": intent.downside_probability,
        "decision_score": decision_score,
        "rank": intent.rank,
        "selection_eligible": intent.selection_eligible,
        "selected_for_policy": intent.selected_for_policy,
        "actionable": intent.actionable,
        "opportunity_target": (
            outcome.opportunity_target if outcome is not None else None
        ),
        "downside_target": (
            outcome.downside_target if outcome is not None else None
        ),
        "net_return": outcome.net_return if outcome is not None else None,
        "excess_return_vs_spy": (
            outcome.excess_return_vs_spy if outcome is not None else None
        ),
        "exit_time_utc": outcome.exit_time_utc if outcome is not None else None,
        "matured_at_utc": (
            outcome.matured_at_utc if outcome is not None else None
        ),
    }


def _cohort_row(
    group: pd.DataFrame,
    *,
    identity: dict[str, object],
    cohort_type: str,
    cohort_value: str,
    minimum_samples: int,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, object]:
    ordered = group.sort_values(
        ["decision_time_utc", "decision_group_id", "maturation_key"],
        kind="stable",
    )
    selected = ordered[ordered["actionable"].astype(bool)]
    matured = selected[selected["outcome_id"].notna()]
    total = len(ordered)
    eligible_count = int(ordered["selection_eligible"].astype(bool).sum())
    selected_count = int(ordered["selected_for_policy"].astype(bool).sum())
    actionable_count = len(selected)
    matured_count = len(matured)
    feature_ids = sorted(
        set(ordered["feature_artifact_sha256"].astype(str))
    )
    intent_ids = sorted(ordered["maturation_key"].astype(str))
    outcome_ids = sorted(matured["outcome_id"].astype(str))
    score_metrics = _score_metrics(selected)
    outcome_metrics = _outcome_metrics(
        matured,
        view=str(identity["view"]),
    )
    content: dict[str, object] = {
        **{
            column: str(identity[column])
            for column in _IDENTITY_COLUMNS
        },
        "feature_artifact_set_sha256": content_sha256(feature_ids),
        "source_intent_ids_sha256": content_sha256(intent_ids),
        "source_outcome_ids_sha256": content_sha256(outcome_ids),
        "cohort_type": cohort_type,
        "cohort_value": cohort_value,
        "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
        "window_end_utc": window_end.isoformat().replace("+00:00", "Z"),
        "total_predictions": total,
        "eligible_predictions": eligible_count,
        "selected_predictions": selected_count,
        "actionable_predictions": actionable_count,
        "matured_selected_samples": matured_count,
        "pending_selected_samples": actionable_count - matured_count,
        "independent_decision_groups": int(
            matured["decision_group_id"].nunique()
        ),
        "evidence_status": (
            "sufficient"
            if matured_count >= minimum_samples
            else "insufficient_evidence"
        ),
        "selection_rate": selected_count / total,
        "actionable_rate": actionable_count / total,
        **score_metrics,
        **outcome_metrics,
        "first_decision_time_utc": _timestamp_text(
            ordered["decision_time_utc"].min()
        ),
        "last_decision_time_utc": _timestamp_text(
            ordered["decision_time_utc"].max()
        ),
    }
    return {**content, "cohort_id": content_sha256(content)}


def _score_metrics(selected: pd.DataFrame) -> dict[str, float | None]:
    if selected.empty:
        return {
            "mean_probability": None,
            "probability_p10": None,
            "probability_p50": None,
            "probability_p90": None,
            "mean_decision_score": None,
            "decision_score_p10": None,
            "decision_score_p50": None,
            "decision_score_p90": None,
            "mean_selected_rank": None,
            "selected_rank_p90": None,
        }
    probability = selected["probability"].to_numpy(float)
    scores = selected["decision_score"].to_numpy(float)
    ranks = pd.to_numeric(selected["rank"], errors="coerce").dropna().to_numpy(
        float
    )
    return {
        "mean_probability": float(np.mean(probability)),
        "probability_p10": _quantile(probability, 0.10),
        "probability_p50": _quantile(probability, 0.50),
        "probability_p90": _quantile(probability, 0.90),
        "mean_decision_score": float(np.mean(scores)),
        "decision_score_p10": _quantile(scores, 0.10),
        "decision_score_p50": _quantile(scores, 0.50),
        "decision_score_p90": _quantile(scores, 0.90),
        "mean_selected_rank": (
            float(np.mean(ranks)) if len(ranks) else None
        ),
        "selected_rank_p90": (
            _quantile(ranks, 0.90) if len(ranks) else None
        ),
    }


def _outcome_metrics(
    matured: pd.DataFrame,
    *,
    view: str,
) -> dict[str, object]:
    empty: dict[str, object] = {
        "opportunity_observed_rate": None,
        "opportunity_brier_score": None,
        "opportunity_calibration_error": None,
        "mean_downside_probability": None,
        "downside_observed_rate": None,
        "downside_brier_score": None,
        "downside_calibration_error": None,
        "average_net_return": None,
        "average_excess_return_vs_spy": None,
        "cumulative_net_return": None,
        "win_rate": None,
        "max_drawdown": None,
        "last_matured_outcome_utc": None,
    }
    if matured.empty:
        return empty
    probability = matured["probability"].to_numpy(float)
    opportunity = matured["opportunity_target"].to_numpy(float)
    returns = matured["net_return"].to_numpy(float)
    period_returns = (
        matured.groupby(
            ["decision_time_utc", "decision_group_id"],
            sort=True,
        )["net_return"]
        .mean()
        .to_numpy(float)
    )
    equity = np.cumprod(1.0 + period_returns)
    equity_with_origin = np.concatenate(([1.0], equity))
    peak = np.maximum.accumulate(equity_with_origin)
    drawdown = 1.0 - np.divide(
        equity_with_origin,
        peak,
        out=np.ones_like(equity_with_origin),
        where=peak != 0,
    )
    result = {
        **empty,
        "opportunity_observed_rate": float(np.mean(opportunity)),
        "opportunity_brier_score": float(
            np.mean(np.square(probability - opportunity))
        ),
        "opportunity_calibration_error": _expected_calibration_error(
            probability,
            opportunity,
        ),
        "average_net_return": float(np.mean(returns)),
        "average_excess_return_vs_spy": float(
            matured["excess_return_vs_spy"].mean()
        ),
        "cumulative_net_return": float(
            np.prod(1.0 + period_returns) - 1.0
        ),
        "win_rate": float(np.mean(returns > 0)),
        "max_drawdown": float(np.max(drawdown, initial=0.0)),
        "last_matured_outcome_utc": _timestamp_text(
            matured["matured_at_utc"].max()
        ),
    }
    if view == "intraday":
        downside_probability = matured["downside_probability"].to_numpy(float)
        downside = matured["downside_target"].to_numpy(float)
        result.update(
            {
                "mean_downside_probability": float(
                    np.mean(downside_probability)
                ),
                "downside_observed_rate": float(np.mean(downside)),
                "downside_brier_score": float(
                    np.mean(np.square(downside_probability - downside))
                ),
                "downside_calibration_error": _expected_calibration_error(
                    downside_probability,
                    downside,
                ),
            }
        )
    return result


def _expected_calibration_error(
    probability: np.ndarray,
    target: np.ndarray,
) -> float:
    bins = np.minimum((probability * 10).astype(int), 9)
    total = len(probability)
    error = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        count = int(mask.sum())
        if count:
            error += (count / total) * abs(
                float(np.mean(probability[mask]))
                - float(np.mean(target[mask]))
            )
    return float(error)


def _quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile))


def _timestamp_text(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("performance evidence timestamp must be timezone-aware")
    text = cast(str, timestamp.tz_convert("UTC").isoformat())
    return text.replace("+00:00", "Z")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("performance report timestamp must be timezone-aware")
    return value.astimezone(UTC)
