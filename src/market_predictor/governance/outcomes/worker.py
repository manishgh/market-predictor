from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import MaturedOutcome
from market_predictor.governance.outcomes.maturation import (
    maturation_attempt,
    mature_prediction,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository


def mature_pending_intents(
    repository: OutcomeRepository,
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    source_artifact_sha256: str,
) -> dict[str, int]:
    """Mature every canonical intent in the pending index; history is never scanned."""
    summary = {
        "intents": 0,
        "matured": 0,
        "pending": 0,
        "blocked": 0,
        "duplicate_semantic": 0,
        "already_matured": 0,
    }
    for maturation_key, session in repository.pending():
        summary["intents"] += 1
        if repository.has_outcome(maturation_key, session):
            repository.drop_pending(maturation_key, session)
            summary["already_matured"] += 1
            continue
        intent = repository.load_intent(maturation_key, session)
        canonical_key = repository.semantic_canonical_key(intent.semantic_prediction_id, session)
        if canonical_key != intent.maturation_key:
            attempt = maturation_attempt(
                intent,
                observed_as_of=observed_as_of,
                status="blocked",
                reasons=("duplicate_semantic_prediction",),
            )
            repository.record_attempt(attempt, decision_session=session)
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
            repository.record_attempt(attempt, decision_session=session)
            summary["blocked"] += 1
            continue
        if isinstance(result, MaturedOutcome):
            repository.record_outcome(intent, result, evidence_rows=evidence)
            summary["matured"] += 1
        else:
            repository.record_attempt(result, decision_session=session)
            summary[result.status] += 1
    return summary
