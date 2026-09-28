"""Explicit synthetic session commits for accounting unit tests only."""
from collections import defaultdict
from datetime import timedelta

from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.governance.outcomes.session_records import (
    MonitoringRoute,
    SessionRecordStore,
    decision_cutoff,
    make_session_record,
)


def commit_test_population(repository):
    store = SessionRecordStore(repository.root)
    groups = defaultdict(list)
    for session in repository.sessions():
        intents = {row.maturation_key: row for row in repository.session_intents(session)}
        keys = repository.session_canonical_keys(session, intents)
        for observation in repository.session_observations(session, intents):
            if observation.maturation_key is not None and keys.get(observation.semantic_prediction_id) != observation.maturation_key:
                continue
            identity = {key: getattr(observation, key) for key in (
                "view", "horizon", "model_release_id", "model_artifact_sha256", "prediction_policy_sha256",
                "label_policy_sha256", "execution_policy_sha256",
            )}
            groups[tuple(identity.items())].append(observation)
    for identity, observations in groups.items():
        route = MonitoringRoute(**dict(identity), promoted_at_utc=min(decision_cutoff(row.decision_session_et) for row in observations))
        by_session = defaultdict(list)
        for row in observations:
            by_session[row.decision_session_et].append(row)
        for session, rows in by_session.items():
            if store.load(route, session) is not None:
                continue
            scored = sum(row.probability is not None for row in rows)
            store.record(make_session_record(
                route=route, decision_session=session, status="registered", failure_reason=None,
                recorded_at_utc=decision_cutoff(session) + timedelta(seconds=1),
                snapshot_id=rows[0].snapshot_id, snapshot_sha256="e" * 64,
                member_set_sha256=content_sha256(sorted(row.ticker for row in rows)),
                members=len(rows), scored=scored,
                abstentions={"live_inputs_incomplete": len(rows) - scored} if len(rows) != scored else {},
                observation_ids=tuple(sorted(row.observation_id for row in rows)),
                intent_ids=tuple(sorted(row.maturation_key for row in rows if row.maturation_key is not None)),
            ))
