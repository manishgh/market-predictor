"""Persisted swing model artifact identities."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator

from market_predictor.core.errors import DataReadinessError

SWING_CANDIDATE_MODEL_SCHEMA: Final = "edge_rebuild.swing_candidate"
TRAINING_SCHEMA: Final = "edge_rebuild.swing_training"
EVALUATION_SCHEMA: Final = "edge_rebuild.swing_evaluation"
MODEL_CARD_SCHEMA: Final = "edge_rebuild.swing_model_card"
OUTPUT_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_candidate_authority"


class TrainingInformationBoundary(BaseModel):
    """Final-fit decision end and latest label used anywhere in model selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_decisions_end_session: date
    training_labels_available_through_utc: datetime

    @field_validator("training_decisions_end_session", mode="before")
    @classmethod
    def require_session_date(cls, value: object) -> date:
        if type(value) is date:
            return value
        if isinstance(value, str):
            parsed = date.fromisoformat(value)
            if parsed.isoformat() == value:
                return parsed
        raise ValueError("training_decisions_end_session must be an ISO session date")

    @field_validator("training_labels_available_through_utc", mode="before")
    @classmethod
    def require_aware_timestamp(cls, value: object) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value)
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("training_labels_available_through_utc must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def require_labels_after_decisions(self) -> TrainingInformationBoundary:
        if self.training_labels_available_through_utc.date() < self.training_decisions_end_session:
            raise ValueError("label availability cannot precede the final training decision session")
        return self


def candidate_training_information_boundary(payload: Mapping[str, object]) -> TrainingInformationBoundary:
    """Refuse old/missing boundaries; never infer label availability from a date."""

    try:
        return TrainingInformationBoundary.model_validate({
            name: payload.get(name) for name in TrainingInformationBoundary.model_fields
        })
    except (ValidationError, ValueError) as exc:
        raise DataReadinessError(f"candidate training information boundary is invalid: {exc}") from exc
