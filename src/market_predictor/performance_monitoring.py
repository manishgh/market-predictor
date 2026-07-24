from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Self
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.outcome_contracts import content_sha256
from market_predictor.outcome_repository import OutcomeRepository

PERFORMANCE_REPORT_VERSION = "market_predictor.performance_cohorts.v1"


class PerformanceCohortV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_release_id: str = Field(pattern=r"^[0-9a-f]{64}$")
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
    samples: int = Field(ge=1)
    evidence_status: Literal["sufficient", "insufficient_evidence"]
    mean_probability: float = Field(ge=0, le=1)
    observed_rate: float = Field(ge=0, le=1)
    brier_score: float = Field(ge=0, le=1)
    calibration_error: float = Field(ge=0, le=1)
    average_net_return: float
    average_excess_return_vs_spy: float
    win_rate: float = Field(ge=0, le=1)
    max_drawdown: float = Field(ge=0)
    first_exit_time_utc: datetime
    last_exit_time_utc: datetime

    @field_validator("first_exit_time_utc", "last_exit_time_utc")
    @classmethod
    def aware_times(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("performance cohort timestamps must be timezone-aware")
        return value.astimezone(UTC)


class PerformanceReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal["market_predictor.performance_cohorts.v1"] = (
        "market_predictor.performance_cohorts.v1"
    )
    report_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at_utc: datetime
    minimum_samples: int = Field(ge=1)
    source_outcome_ids: tuple[str, ...]
    rows: tuple[PerformanceCohortV1, ...]

    @field_validator("generated_at_utc")
    @classmethod
    def aware_generation_time(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("performance report timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("source_outcome_ids")
    @classmethod
    def canonical_outcome_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value
        ):
            raise ValueError("performance source outcome identity is invalid")
        if tuple(sorted(set(value))) != value:
            raise ValueError("performance source outcomes must be unique and sorted")
        return value

    @model_validator(mode="after")
    def validate_report_identity(self) -> Self:
        content = self.model_dump(mode="json", exclude={"report_id"})
        if content_sha256(content) != self.report_id:
            raise ValueError("performance report identity is invalid")
        return self


def build_performance_cohorts(
    repository: OutcomeRepository,
    *,
    generated_at: datetime | None = None,
    minimum_samples: int = 30,
) -> dict[str, object]:
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    records: list[dict[str, object]] = []
    for outcome in repository.outcomes():
        if (
            repository.semantic_canonical_key(outcome.semantic_prediction_id)
            != outcome.maturation_key
        ):
            continue
        intent = repository.load_intent(outcome.maturation_key)
        records.append(
            {
                "outcome_id": outcome.outcome_id,
                "model_release_id": intent.model_release_id,
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
                "target": outcome.opportunity_target,
                "net_return": outcome.net_return,
                "excess_return_vs_spy": outcome.excess_return_vs_spy,
                "exit_time_utc": outcome.exit_time_utc,
            }
        )
    frame = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    if not frame.empty:
        frame["exit_time_utc"] = pd.to_datetime(frame["exit_time_utc"], utc=True)
        frame["decision_time_utc"] = pd.to_datetime(
            frame["decision_time_utc"],
            utc=True,
        )
        base = ["model_release_id", "view", "horizon"]
        cohort_specs = [
            ("all", None),
            ("market_regime", "market_regime"),
            ("sector", "sector"),
            ("market_cap_bucket", "market_cap_bucket"),
            ("liquidity_bucket", "liquidity_bucket"),
            ("calibration_bin", "calibration_bin"),
        ]
        for cohort_type, cohort_column in cohort_specs:
            group_columns = base + ([cohort_column] if cohort_column else [])
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
                group_identity = dict(zip(group_columns, values, strict=True))
                row = _cohort_row(
                        group,
                        identity=group_identity,
                        cohort_type=cohort_type,
                        cohort_value=(
                            "all"
                            if cohort_column is None
                            else str(group_identity[cohort_column])
                        ),
                        minimum_samples=minimum_samples,
                    )
                rows.append(
                    PerformanceCohortV1.model_validate(row).model_dump(mode="json")
                )
    generated = (generated_at or datetime.now(UTC)).astimezone(UTC)
    source_outcome_ids = sorted(str(record["outcome_id"]) for record in records)
    report_identity: dict[str, object] = {
        "contract_version": PERFORMANCE_REPORT_VERSION,
        "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "minimum_samples": minimum_samples,
        "source_outcome_ids": source_outcome_ids,
        "rows": rows,
    }
    report = PerformanceReportV1.model_validate(
        {
            **report_identity,
            "report_id": content_sha256(report_identity),
        }
    )
    return report.model_dump(mode="json")


def validate_performance_report(value: object) -> dict[str, object]:
    report = PerformanceReportV1.model_validate(value)
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


def _cohort_row(
    group: pd.DataFrame,
    *,
    identity: dict[str, object],
    cohort_type: str,
    cohort_value: str,
    minimum_samples: int,
) -> dict[str, object]:
    ordered = group.sort_values(
        ["decision_time_utc", "decision_group_id", "exit_time_utc"],
        kind="stable",
    )
    probability = ordered["probability"].to_numpy(float)
    target = ordered["target"].to_numpy(float)
    returns = ordered["net_return"].to_numpy(float)
    period_returns = (
        ordered.groupby(
            ["decision_time_utc", "decision_group_id"],
            sort=False,
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
    count = len(ordered)
    return {
        "model_release_id": str(identity["model_release_id"]),
        "view": str(identity["view"]),
        "horizon": str(identity["horizon"]),
        "cohort_type": cohort_type,
        "cohort_value": cohort_value,
        "samples": count,
        "evidence_status": (
            "sufficient" if count >= minimum_samples else "insufficient_evidence"
        ),
        "mean_probability": float(np.mean(probability)),
        "observed_rate": float(np.mean(target)),
        "brier_score": float(np.mean(np.square(probability - target))),
        "calibration_error": float(abs(np.mean(probability) - np.mean(target))),
        "average_net_return": float(np.mean(returns)),
        "average_excess_return_vs_spy": float(
            ordered["excess_return_vs_spy"].mean()
        ),
        "win_rate": float(np.mean(returns > 0)),
        "max_drawdown": float(np.max(drawdown, initial=0.0)),
        "first_exit_time_utc": ordered["exit_time_utc"].min().isoformat(),
        "last_exit_time_utc": ordered["exit_time_utc"].max().isoformat(),
    }
