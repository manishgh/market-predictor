from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import MaturedOutcome, swing_horizon_sessions
from market_predictor.governance.outcomes.maturation import (
    maturation_attempt,
    mature_prediction,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.sessions import horizon_last_close


def mature_pending_intents(
    repository: OutcomeRepository,
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    source_artifact_sha256: str,
) -> dict[str, int]:
    """Mature the pending index's canonical intents whose horizon has closed; history is never scanned.

    The summary counts index entries by what happened to them. An entry is left in place while
    its horizon is open or while its registration has not yet written the semantic record (the
    registration rerun completes it); an entry whose semantic record names another intent can
    never be canonical and is dropped.
    """
    summary = {
        "index_entries": 0,
        "horizon_open": 0,
        "matured": 0,
        "pending": 0,
        "blocked": 0,
        "registration_incomplete": 0,
        "not_canonical_dropped": 0,
        "already_matured": 0,
    }
    for maturation_key, session in repository.pending():
        summary["index_entries"] += 1
        if repository.has_outcome(maturation_key, session):
            repository.drop_pending(maturation_key, session)
            summary["already_matured"] += 1
            continue
        intent = repository.load_intent(maturation_key, session)
        canonical_key = repository.semantic_canonical_key(intent.semantic_prediction_id, session)
        if canonical_key is None:
            summary["registration_incomplete"] += 1
            continue
        if canonical_key != maturation_key:
            repository.drop_pending(maturation_key, session)
            summary["not_canonical_dropped"] += 1
            continue
        try:
            # An outcome is taken only once the horizon has closed, so a target or stop reached
            # earlier still records the fixed-horizon return over the whole path.
            if horizon_last_close(session, swing_horizon_sessions(intent.horizon), through=observed_as_of) is None:
                summary["horizon_open"] += 1
                continue
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
