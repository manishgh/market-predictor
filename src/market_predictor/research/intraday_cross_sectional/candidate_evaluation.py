from __future__ import annotations

from typing import Any

import numpy as np
from pydantic import Field

from market_predictor.core.schema import FrozenContract


class CrossSectionalPromotionGateConfig(FrozenContract):
    minimum_sessions: int = Field(default=20, ge=2)
    minimum_selected_trades: int = Field(default=100, ge=1)
    minimum_mean_ndcg_at_k: float = Field(default=0.55, ge=0, le=1)
    minimum_holdout_ndcg_at_k: float = Field(default=0.50, ge=0, le=1)
    minimum_average_trade_return: float = 0.0
    minimum_average_return_ci_low: float = 0.0
    minimum_profit_factor: float = Field(default=1.05, ge=0)
    maximum_drawdown: float = Field(default=0.25, ge=0, le=1)
    maximum_calibration_ece: float = Field(default=0.10, ge=0, le=1)
    required_calibration_families: tuple[str, ...] = ("downside_classifier",)


def evaluate_candidate_acceptance(
    *,
    ranking_audit: dict[str, Any] | None,
    holdout_metrics: dict[str, Any] | None,
    calibration_audits: dict[str, dict[str, Any]] | None,
    config: CrossSectionalPromotionGateConfig = CrossSectionalPromotionGateConfig(),
) -> dict[str, Any]:
    failures: list[str] = []
    if ranking_audit is None:
        failures.append("ranking economics audit is required")
    else:
        failures.extend(str(item) for item in ranking_audit.get("readiness_failures", []))
        _minimum_gate(failures, ranking_audit, "selected_sessions", config.minimum_sessions)
        _minimum_gate(
            failures,
            ranking_audit,
            "selected_trades",
            config.minimum_selected_trades,
        )
        _minimum_gate(
            failures,
            ranking_audit,
            "mean_ndcg_at_k",
            config.minimum_mean_ndcg_at_k,
        )
        _minimum_gate(
            failures,
            ranking_audit,
            "average_trade_return",
            config.minimum_average_trade_return,
        )
        interval = ranking_audit.get("average_trade_return_interval", {})
        _minimum_gate(
            failures,
            interval,
            "low",
            config.minimum_average_return_ci_low,
            prefix="average return CI",
        )
        _minimum_gate(
            failures,
            ranking_audit,
            "profit_factor",
            config.minimum_profit_factor,
        )
        _maximum_gate(failures, ranking_audit, "max_drawdown", config.maximum_drawdown)
    if holdout_metrics is None:
        failures.append("ticker-holdout metrics are required")
    else:
        _minimum_gate(
            failures,
            holdout_metrics,
            "mean_ndcg_at_k",
            config.minimum_holdout_ndcg_at_k,
            prefix="holdout",
        )
    for family in config.required_calibration_families:
        audit = calibration_audits.get(family) if calibration_audits else None
        if audit is None:
            failures.append(f"calibration audit is required for {family}")
            continue
        after = audit.get("after", {})
        _maximum_gate(
            failures,
            after,
            "expected_calibration_error",
            config.maximum_calibration_ece,
            prefix=family,
        )
    return {
        "schema": "ml_v3.promotion_evidence.v1",
        "passed": not failures,
        "failures": failures,
        "thresholds": config.model_dump(mode="json"),
    }


def _minimum_gate(
    failures: list[str],
    record: dict[str, Any],
    key: str,
    threshold: float,
    *,
    prefix: str = "ranking",
) -> None:
    value = _finite_float(record.get(key))
    if value is None or value < threshold:
        failures.append(f"{prefix} {key} {value} < {threshold}")


def _maximum_gate(
    failures: list[str],
    record: dict[str, Any],
    key: str,
    threshold: float,
    *,
    prefix: str = "ranking",
) -> None:
    value = _finite_float(record.get(key))
    if value is None or value > threshold:
        failures.append(f"{prefix} {key} {value} > {threshold}")


def _finite_float(value: object) -> float | None:
    try:
        converted = float(str(value))
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None
