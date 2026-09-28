from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

import market_predictor.serving.prediction_service as prediction_module
from market_predictor.core.prediction_contracts import PredictionConflictError, PredictionReadinessError, PredictionRequest
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.session_records import SessionRecordStore, expected_sessions
from market_predictor.serving.outcome_intents import register_snapshot_intents
from market_predictor.serving.session_registration import register_session_predictions
from tests.support.swing_serving import NOW, UnavailableInputs, swing_serving


@pytest.fixture(autouse=True)
def private_monitoring_lease(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))


def _snapshot(service, result):
    return service.snapshot_store.record_cross_section(result.response, result.members, as_of=NOW, promoted_at=result.promoted_at_utc)


def test_identical_and_reordered_decisions_reuse_original_audit_snapshot(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch, member_count=121).service
    first_result = service.predict_swing_cross_section(NOW)
    first = _snapshot(service, first_result)
    second_result = service.predict_swing_cross_section(NOW)
    assert first_result.response.request_id != second_result.response.request_id
    second_result = replace(second_result, members=tuple(reversed(second_result.members)), response=second_result.response.model_copy(
        update={"predictions": list(reversed(second_result.response.predictions))},
    ))
    second = _snapshot(service, second_result)
    assert second == first
    context, loaded, envelope = service.snapshot_store.load(first.snapshot_id)
    assert len(context.tickers) == len(loaded.predictions) == 121
    assert envelope["content"]["scope"] == "decision_cross_section"
    assert loaded.snapshot_sha256 != loaded.snapshot_id


def test_registration_while_drift_blocked_commits_every_member_once(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch, member_count=120, excluded_tickers=("INPUT",), peer_floor_tickers=("THIN",)).service
    repository = OutcomeRepository(tmp_path / "outcomes")
    record = register_session_predictions(service, repository, as_of=NOW)
    assert record.status == "registered"
    assert record.members == 122 and record.scored == 120
    assert record.abstentions == {"live_inputs_incomplete": 1, "sector_peer_floor": 1}
    intents = {intent.maturation_key: intent for intent in repository.session_intents(record.decision_session)}
    observations = repository.session_observations(record.decision_session, intents)
    assert len(observations) == 122
    unscored = [item for item in observations if item.probability is None]
    assert len(unscored) == 2 and all(item.market_regime is None and item.market_cap_bucket is None for item in unscored)
    assert register_session_predictions(service, repository, as_of=NOW) == record
    assert len(repository.session_observations(record.decision_session, intents)) == 122


def test_republished_source_cannot_double_count_committed_session(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch).service
    repository = OutcomeRepository(tmp_path / "outcomes")
    record = register_session_predictions(service, repository, as_of=NOW)
    original = service.swing_live_input_provider.load
    monkeypatch.setattr(service.swing_live_input_provider, "load", lambda **kwargs: replace(original(**kwargs), manifest_sha256="1" * 64))
    with pytest.raises(PredictionConflictError):
        register_session_predictions(service, repository, as_of=NOW)
    assert SessionRecordStore(repository.root).load(record.route, record.decision_session) == record
    assert len(repository.session_intents(record.decision_session)) == 60


def test_failed_inputs_can_retry_to_registered(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch).service
    provider = service.swing_live_input_provider
    route = service.monitoring_route(NOW)
    repository = OutcomeRepository(tmp_path / "outcomes")
    service.swing_live_input_provider = UnavailableInputs()
    with pytest.raises(PredictionReadinessError):
        register_session_predictions(service, repository, as_of=NOW)
    store = SessionRecordStore(repository.root)
    failed = store.load(route, NOW.date())
    assert failed.status == "failed" and failed.failure_reason == "inputs_unavailable"
    service.swing_live_input_provider = provider
    completed = register_session_predictions(service, repository, as_of=NOW)
    assert completed.status == "registered"
    assert store.load(route, NOW.date(), as_of=failed.recorded_at_utc) == failed


def test_partial_write_is_failed_then_retry_restores_complete_population(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch).service
    repository = OutcomeRepository(tmp_path / "outcomes")
    original = repository.record_intent
    count = 0
    def interrupt(intent):
        nonlocal count
        count += 1
        if count == 6:
            raise OSError("simulated interruption")
        return original(intent)
    monkeypatch.setattr(repository, "record_intent", interrupt)
    with pytest.raises(OSError, match="interruption"):
        register_session_predictions(service, repository, as_of=NOW)
    store = SessionRecordStore(repository.root)
    [failed] = store.records(as_of=datetime.now(UTC))
    assert failed.status == "failed" and not failed.observation_ids
    monkeypatch.setattr(repository, "record_intent", original)
    record = register_session_predictions(service, repository, as_of=NOW)
    assert len(record.observation_ids) == len(record.intent_ids) == 60


def test_request_snapshots_cannot_register_intents(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch, enforce_drift=False, persist_snapshots=True).service
    response = service.predict(PredictionRequest(tickers=["T000"], as_of=NOW))
    repository = OutcomeRepository(tmp_path / "outcomes")
    with pytest.raises(PredictionConflictError):
        register_snapshot_intents(service.snapshot_store, repository, response.snapshot_id)
    assert repository.sessions() == ()


def test_all_abstaining_members_register_without_intents(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch).service
    live = prediction_module.build_live_swing_features()
    live = replace(live, technical_market=live.technical_market.iloc[:0], catalyst_full=live.catalyst_full.iloc[:0],
                   context=live.context.iloc[:0],
                   members=tuple(m.model_copy(update={"abstention_reason": "sector_peer_floor"}) for m in live.members))
    monkeypatch.setattr(prediction_module, "build_live_swing_features", lambda *args, **kwargs: live)
    repository = OutcomeRepository(tmp_path / "outcomes")
    record = register_session_predictions(service, repository, as_of=NOW)
    assert record.members == 60 and record.scored == 0
    assert record.intent_ids == () and len(record.observation_ids) == 60
    assert repository.pending() == []


def test_cross_section_snapshot_tampering_and_member_loss_fail(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch).service
    result = service.predict_swing_cross_section(NOW)
    with pytest.raises(PredictionConflictError):
        _snapshot(service, replace(result, members=result.members[:-1]))
    response = _snapshot(service, result)
    path = service.snapshot_store.path_for(response.snapshot_id)
    value = json.loads(path.read_text())
    value["content"]["response"]["predictions"][0]["swing"]["probability"] = 0.01
    path.write_text(json.dumps(value))
    with pytest.raises(PredictionConflictError):
        service.snapshot_store.load(response.snapshot_id)


def test_route_activation_is_first_cutoff_at_or_after_promotion(tmp_path, monkeypatch):
    route = swing_serving(tmp_path, monkeypatch).service.monitoring_route(NOW)
    route = route.model_copy(update={"promoted_at_utc": NOW})
    sessions = expected_sessions(route, start=NOW-timedelta(days=7), end=NOW+timedelta(days=2))
    assert sessions == (NOW.date()+timedelta(days=1), NOW.date()+timedelta(days=2))



def test_cross_section_replay_does_not_reconstruct_limited_http_request(tmp_path, monkeypatch):
    from market_predictor.core.prediction_contracts import InvestmentReplayRequest
    from market_predictor.serving.investment_replay import InvestmentReplayService
    service = swing_serving(tmp_path, monkeypatch, member_count=121).service
    saved = _snapshot(service, service.predict_swing_cross_section(NOW))
    replay = InvestmentReplayService(snapshot_store=service.snapshot_store, price_provider=None)
    response = replay.replay(InvestmentReplayRequest(snapshot_id=saved.snapshot_id, ticker="T120",
                                                     evaluation_as_of=NOW+timedelta(days=1)))
    assert not any("snapshot has no" in reason for reason in response.reasons)


def test_registration_command_runs_only_on_production_surface(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from market_predictor.cli_surface import command_names
    from market_predictor.collection_cli import app as collection
    from market_predictor.production_cli import app
    from market_predictor.research_cli import app as research
    service = swing_serving(tmp_path, monkeypatch).service
    monkeypatch.setattr("market_predictor.commands.session_monitoring._service", lambda: service)
    assert "register-session-predictions" not in command_names(collection) | command_names(research)
    result = CliRunner().invoke(app, ["register-session-predictions", "--as-of", NOW.isoformat(),
                                    "--outcome-dir", str(tmp_path / "outcomes")])
    assert result.exit_code == 0, result.output
    value = json.loads(result.output)
    assert value["status"] == "registered" and value["members"] == 60


def test_all_abstaining_snapshot_cannot_misstate_model_identity(tmp_path, monkeypatch):
    service = swing_serving(tmp_path, monkeypatch).service
    result = service.predict_swing_cross_section(NOW)
    evidence = result.response.evidence.model_copy(update={"model_release_ids": {"swing": "0" * 64}})
    result = replace(result, response=result.response.model_copy(update={"evidence": evidence}))
    with pytest.raises(PredictionConflictError):
        _snapshot(service, result)



def test_changed_snapshot_after_partial_write_cannot_commit_unreachable_intents(tmp_path, monkeypatch):
    from market_predictor.serving.outcome_intents import maturation_intents_from_response
    service = swing_serving(tmp_path, monkeypatch).service
    result = service.predict_swing_cross_section(NOW)
    original = _snapshot(service, result)
    repository = OutcomeRepository(tmp_path / "outcomes")
    intents = maturation_intents_from_response(original, snapshot_id=original.snapshot_id)
    repository.record_intent(intents[0])
    # Membership provenance changes the decision snapshot, but not this scored intent's semantic id.
    member = result.members[0]
    changed_member = member.model_copy(update={"membership_available_at_utc": member.membership_available_at_utc - timedelta(seconds=1)})
    changed = _snapshot(service, replace(result, members=(changed_member, *result.members[1:])))
    assert original.snapshot_id != changed.snapshot_id
    with pytest.raises(PredictionConflictError):
        register_snapshot_intents(service.snapshot_store, repository, changed.snapshot_id)
    assert SessionRecordStore(repository.root).records(as_of=datetime.now(UTC)) == []
    # The first immutable decision is recoverable without rebinding or deleting evidence.
    registration = register_snapshot_intents(service.snapshot_store, repository, original.snapshot_id)
    assert registration.session_record.status == "registered"
    assert all(repository.semantic_canonical_key(intent.semantic_prediction_id, intent.decision_session_et) == intent.maturation_key
               for intent in registration.intents)
