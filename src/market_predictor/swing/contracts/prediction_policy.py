from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SwingPredictionPolicy(BaseModel):
    """Complete ranking and portfolio policy for a served swing model."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract_version: Literal["market_predictor.swing_prediction_policy.v1"] = (
        "market_predictor.swing_prediction_policy.v1"
    )
    horizon_sessions: int = Field(ge=1, le=100)
    minimum_probability: float = Field(gt=0.0, lt=1.0)
    maximum_predictions_per_decision: int = Field(ge=1, le=100)
    target_maximum_sector_weight: float = Field(gt=0.0, le=1.0)
    hard_maximum_sector_weight: float = Field(gt=0.0, le=1.0)
    minimum_distinct_sectors: int = Field(ge=1, le=100)

    @model_validator(mode="after")
    def validate_constraints(self) -> Self:
        if self.target_maximum_sector_weight > self.hard_maximum_sector_weight:
            raise ValueError("target sector weight exceeds hard sector weight")
        if self.maximum_predictions_per_decision < self.minimum_distinct_sectors:
            raise ValueError("prediction limit cannot satisfy sector minimum")
        return self

    def specification(self) -> dict[str, object]:
        return {
            "contract_version": self.contract_version,
            "policy_id": self.contract_version,
            "role": "prediction_intelligence_only_no_alerts_or_execution",
            "horizon_sessions": self.horizon_sessions,
            "decision_score": "model_probability",
            "eligibility": (
                "readiness=valid_and_model_probability_is_finite_and_"
                "model_probability>=minimum_probability"
            ),
            "selection": "probability_rank_with_adaptive_sector_constraints",
            "minimum_probability": self.minimum_probability,
            "maximum_predictions_per_decision": self.maximum_predictions_per_decision,
            "target_maximum_sector_weight": self.target_maximum_sector_weight,
            "hard_maximum_sector_weight": self.hard_maximum_sector_weight,
            "minimum_distinct_sectors": self.minimum_distinct_sectors,
            "tie_breakers": ["model_probability:desc", "security_id:asc"],
        }

    def sha256(self) -> str:
        return _sha256(self.specification())


def parse_swing_prediction_policy(
    payload: Mapping[str, object],
    *,
    expected_sha256: str | None = None,
) -> SwingPredictionPolicy:
    policy = SwingPredictionPolicy.model_validate(
        {
            "contract_version": payload.get("contract_version"),
            "horizon_sessions": payload.get("horizon_sessions"),
            "minimum_probability": payload.get("minimum_probability"),
            "maximum_predictions_per_decision": payload.get(
                "maximum_predictions_per_decision"
            ),
            "target_maximum_sector_weight": payload.get(
                "target_maximum_sector_weight"
            ),
            "hard_maximum_sector_weight": payload.get("hard_maximum_sector_weight"),
            "minimum_distinct_sectors": payload.get("minimum_distinct_sectors"),
        }
    )
    if dict(payload) != policy.specification():
        raise ValueError("swing prediction policy semantics are not canonical")
    if expected_sha256 is not None and policy.sha256() != expected_sha256:
        raise ValueError("swing prediction policy does not match its bound hash")
    return policy


def _sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
