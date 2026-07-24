from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_predictor.outcome_contracts import MaturedOutcomeV1
from market_predictor.outcome_maturation import (
    maturation_attempt,
    mature_prediction,
)
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.v3.errors import DataReadinessError


def mature_pending_intents(
    repository: OutcomeRepository,
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    source_artifact_sha256: str,
) -> dict[str, int]:
    summary = {
        "intents": 0,
        "matured": 0,
        "pending": 0,
        "blocked": 0,
        "duplicate_semantic": 0,
        "already_matured": 0,
    }
    for intent in repository.intents():
        summary["intents"] += 1
        if repository.has_outcome(intent.maturation_key):
            summary["already_matured"] += 1
            continue
        canonical_key = repository.semantic_canonical_key(
            intent.semantic_prediction_id
        )
        if canonical_key != intent.maturation_key:
            attempt = maturation_attempt(
                intent,
                observed_as_of=observed_as_of,
                status="blocked",
                reasons=("duplicate_semantic_prediction",),
            )
            repository.record_attempt(attempt)
            summary["duplicate_semantic"] += 1
            continue
        try:
            result, evidence = mature_prediction(
                intent,
                bars,
                observed_as_of=observed_as_of,
                source_artifact_sha256=source_artifact_sha256,
            )
        except (DataReadinessError, KeyError, TypeError, ValueError) as exc:
            attempt = maturation_attempt(
                intent,
                observed_as_of=observed_as_of,
                status="blocked",
                reasons=(f"invalid_maturation_input:{type(exc).__name__}",),
            )
            repository.record_attempt(attempt)
            summary["blocked"] += 1
            continue
        if isinstance(result, MaturedOutcomeV1):
            repository.record_outcome(result, evidence_rows=evidence)
            summary["matured"] += 1
        else:
            repository.record_attempt(result)
            summary[result.status] += 1
    return summary
