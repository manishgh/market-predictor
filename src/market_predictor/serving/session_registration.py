"""Nightly registration: one monitored cross-section per release, route and session."""
from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_predictor.core.cross_section_contracts import CrossSectionMember
from market_predictor.core.prediction_contracts import PredictionConflictError, PredictionModelUnavailableError
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.session_records import (
    MonitoringRoute,
    SessionRecord,
    SessionRecordStore,
    decision_cutoff,
    make_session_record,
)
from market_predictor.monitoring_lease import monitoring_lease
from market_predictor.resources import assert_memory_budget
from market_predictor.serving.outcome_intents import (
    SnapshotRegistration,
    maturation_intents_from_response,
    monitoring_observations_from_response,
)
from market_predictor.serving.prediction_service import PredictionService
from market_predictor.serving.snapshot_store import PredictionSnapshotStore
from market_predictor.serving.swing_features import _expected_swing_decision_time


def register_session_predictions(
    service: PredictionService, repository: OutcomeRepository, *, as_of: datetime,
    runtime_dir: Path | None = None, wait_seconds: float = 900.0,
) -> SessionRecord:
    if as_of.utcoffset() is None:
        raise ValueError("registration as_of must be timezone-aware")
    decision = _expected_swing_decision_time(pd.Timestamp(as_of)).to_pydatetime()
    if decision > as_of:
        raise ValueError("registration requires the completed nightly decision cutoff")
    with monitoring_lease("register-session-predictions", runtime_dir=runtime_dir, wait_seconds=wait_seconds):
        # A verified anchor is required; an unknown release cannot be named as a failure.
        route = service.monitoring_route(as_of)
        if route.promoted_at_utc > decision:
            raise ValueError("this release was not active at the session's decision")
        store = SessionRecordStore(repository.root)
        store.record_route(route)
        session = pd.Timestamp(decision).tz_convert("America/New_York").date()
        try:
            assert_memory_budget(hard_budget_gib=service.memory_budget_gib, headroom_gib=service.memory_headroom_gib,
                                 stage="before monitoring session registration")
            result = service.predict_swing_cross_section(as_of)
            if result.response.models["swing"].release_id != route.model_release_id:
                raise PredictionConflictError
            snapshot = service.snapshot_store.record_cross_section(
                result.response, result.members, as_of=as_of, promoted_at=result.promoted_at_utc,
            )
            assert snapshot.snapshot_id is not None
            registered = register_cross_section_snapshot(service.snapshot_store, repository, snapshot.snapshot_id)
            assert registered.session_record is not None
            return registered.session_record
        except Exception as exc:
            previous = store.load(route, session)
            if previous is None or previous.status != "registered":
                cause = str(exc.__cause__ or exc)
                reason = "model_unavailable" if isinstance(exc, PredictionModelUnavailableError) else (
                    "exclusion_ceiling_exceeded" if "ceiling" in cause else
                    "inputs_unavailable" if getattr(exc, "code", "") == "prediction_not_ready" else "registration_error"
                )
                store.record(make_session_record(route=route, decision_session=session, status="failed",
                                                 recorded_at_utc=datetime.now(UTC), failure_reason=reason))
            raise


def register_cross_section_snapshot(
    snapshots: PredictionSnapshotStore, repository: OutcomeRepository, snapshot_id: str,
) -> SnapshotRegistration:
    """Caller holds the monitoring lease; the final marker commits every source id."""
    _, response, envelope = snapshots.load(snapshot_id)
    content = envelope["content"]
    if content["scope"] != "decision_cross_section":
        raise PredictionConflictError
    members = tuple(CrossSectionMember.model_validate(value) for value in content["members"])
    model = response.models["swing"]
    route = MonitoringRoute.model_validate({
        "view": "swing", "horizon": model.resolved_horizon,
        "model_release_id": model.release_id, "model_artifact_sha256": model.artifact_sha256,
        "prediction_policy_sha256": model.prediction_policy_sha256, "label_policy_sha256": model.label_policy_sha256,
        "execution_policy_sha256": model.execution_policy_sha256, "promoted_at_utc": content["promoted_at_utc"],
    })
    assert response.evidence is not None
    session = pd.Timestamp(response.evidence.prediction_cutoff_utc).tz_convert("America/New_York").date()
    if response.evidence.prediction_cutoff_utc != decision_cutoff(session):
        raise PredictionConflictError
    store = SessionRecordStore(repository.root)
    store.record_route(route)
    existing = store.load(route, session)
    if existing is not None and existing.status == "registered" and existing.snapshot_id != snapshot_id:
        raise PredictionConflictError
    intents = maturation_intents_from_response(response, snapshot_id=snapshot_id)
    observations = monitoring_observations_from_response(
        response, snapshot_id=snapshot_id, intents={(intent.ticker, intent.view): intent for intent in intents}, members=members,
    )
    if any(repository.semantic_canonical_key(intent.semantic_prediction_id, session) not in (None, intent.maturation_key)
           for intent in intents):
        raise PredictionConflictError
    for intent in intents:
        repository.record_intent(intent)
    for observation in observations:
        repository.record_observation(observation)
    # Collection and maturation consume the semantic pending index. A changed snapshot
    # after a partial write must not commit intents that those consumers cannot reach.
    # Resume the original immutable snapshot; never silently rebind its semantic rows.
    if any(repository.semantic_canonical_key(intent.semantic_prediction_id, session) != intent.maturation_key for intent in intents):
        raise PredictionConflictError
    record = make_session_record(
        route=route, decision_session=session, status="registered", failure_reason=None, recorded_at_utc=datetime.now(UTC),
        snapshot_id=snapshot_id, snapshot_sha256=envelope["content_sha256"],
        member_set_sha256=content_sha256([member.model_dump(mode="json") for member in sorted(members, key=lambda m: m.security_id)]),
        members=len(members), scored=len(intents),
        abstentions=dict(Counter(member.abstention_reason for member in members if member.abstention_reason)),
        observation_ids=tuple(sorted(observation.observation_id for observation in observations)),
        intent_ids=tuple(sorted(intent.maturation_key for intent in intents)),
    )
    committed = store.record(record)
    return SnapshotRegistration(intents, {}, committed)
