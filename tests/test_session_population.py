"""Poison and integration checks for committed monitoring population."""
from datetime import UTC, date, datetime, timedelta

import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.governance.outcomes.performance import build_performance_cohorts, validate_performance_report
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.session_coverage import build_session_coverage
from market_predictor.governance.outcomes.session_records import (
    MonitoringRoute,
    SessionRecordStore,
    decision_cutoff,
    make_session_record,
)
from market_predictor.serving.session_registration import register_session_predictions
from tests import test_drift_policy as drift_tests
from tests.support.monitoring import commit_test_population
from tests.support.swing_serving import NOW, swing_serving
from tests.test_outcome_repository import _attempt
from tests.test_performance_monitoring import _intent_variant


def _report(repository):
    return build_performance_cohorts(repository, generated_at=datetime.now(UTC), minimum_samples=1)


def test_report_uses_complete_committed_inventory_and_preserves_unknowns(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    service = swing_serving(tmp_path, monkeypatch, member_count=120,
                            excluded_tickers=("INPUT",), peer_floor_tickers=("THIN",)).service
    repository = OutcomeRepository(tmp_path / "outcomes")
    record = register_session_predictions(service, repository, as_of=NOW)
    # A stray request/partial-write row must never enter the denominator or the overdue gate.
    raw = _intent_variant("STRAY", "1", probability=0.8, decision_time=datetime(2026, 3, 2, 22, tzinfo=UTC))
    repository.record_intent(raw)
    report = _report(repository)
    row = next(row for row in report["rows"] if row["cohort_type"] == "all")
    assert (row["total_predictions"], row["eligible_predictions"]) == (122, 120)
    assert row["route_oldest_pending_decision_session_et"] != "2026-03-02"
    assert set(report["source_observation_ids"]) == set(record.observation_ids)
    assert set(report["source_intent_ids"]) == set(record.intent_ids)
    assert report["source_session_record_ids"] == [record.record_id]
    unknown = next(row for row in report["rows"] if row["cohort_type"] == "market_regime" and row["cohort_value"] == "unavailable")
    assert unknown["total_predictions"] == 2 and unknown["mean_probability"] is None
    # Committed inventory cannot be silently shrunk after a file disappears.
    path = repository.root / "sessions" / record.decision_session.isoformat() / "observations" / f"{record.observation_ids[0]}.json"
    path.unlink()
    with pytest.raises(DataReadinessError, match="source evidence"):
        _report(repository)


def test_partial_session_counts_only_after_last_commit_and_asof_time(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    service = swing_serving(tmp_path, monkeypatch).service
    repository = OutcomeRepository(tmp_path / "outcomes")
    original = repository.record_intent
    calls = 0
    def crash(intent):
        nonlocal calls
        calls += 1
        if calls == 6:
            raise OSError("interrupted")
        return original(intent)
    monkeypatch.setattr(repository, "record_intent", crash)
    with pytest.raises(OSError, match="interrupted"):
        register_session_predictions(service, repository, as_of=NOW)
    failed_report = _report(repository)
    assert failed_report["rows"] == [] and failed_report["source_intent_ids"] == []
    assert failed_report["session_coverage"][0]["failed_sessions"] == [NOW.date().isoformat()]
    as_of = datetime.now(UTC)
    monkeypatch.setattr(repository, "record_intent", original)
    registered = register_session_predictions(service, repository, as_of=NOW)
    assert build_performance_cohorts(repository, generated_at=as_of)["rows"] == []
    report = _report(repository)
    assert next(row for row in report["rows"] if row["cohort_type"] == "all")["total_predictions"] == 60
    assert report["source_session_record_ids"] == [registered.record_id]


def test_coverage_isolated_by_release_and_promotion_cutoff(tmp_path):
    first = MonitoringRoute(model_release_id="a" * 64, model_artifact_sha256="b" * 64,
                            prediction_policy_sha256="c" * 64, label_policy_sha256="d" * 64,
                            execution_policy_sha256="e" * 64, promoted_at_utc=decision_cutoff(date(2026, 7, 23)))
    second = first.model_copy(update={"model_release_id": "f" * 64,
                                     "promoted_at_utc": decision_cutoff(date(2026, 7, 24)) + timedelta(seconds=1)})
    failed = make_session_record(route=first, decision_session=date(2026, 7, 24), status="failed",
                                 failure_reason="not_run", operator_id="test-operator", operator_reason="test outage",
                                 recorded_at_utc=decision_cutoff(date(2026, 7, 24)) + timedelta(seconds=1))
    store = SessionRecordStore(tmp_path)
    store.record(failed)
    store.record_route(second)
    end = decision_cutoff(date(2026, 7, 27))
    coverage = build_session_coverage(store.routes(), store.records(as_of=end), start=first.promoted_at_utc, end=end)
    by_release = {item.route.model_release_id: item for item in coverage}
    assert by_release[first.model_release_id].failed_sessions == (date(2026, 7, 24),)
    assert by_release[second.model_release_id].expected_sessions == (date(2026, 7, 27),)
    assert by_release[second.model_release_id].missing_sessions == (date(2026, 7, 27),)


def _coverage_changed(report, *, failed=(), missing=()):
    coverage = report["session_coverage"][0]
    removed = set(failed) | set(missing)
    coverage["registered_sessions"] = [day for day in coverage["expected_sessions"] if day not in removed]
    coverage["failed_sessions"] = sorted(failed)
    coverage["missing_sessions"] = sorted(missing)
    coverage["source_record_ids"] = sorted(content_sha256(day) for day in coverage["expected_sessions"] if day not in missing)
    report["source_session_record_ids"] = coverage["source_record_ids"]
    report["report_id"] = content_sha256({key: value for key, value in report.items() if key != "report_id"})
    return report


def _policy_case():
    case = drift_tests.DriftPolicyTests()
    case.setUp()
    return case


def test_missing_session_grace_is_exact_and_failed_days_only_warn():
    case = _policy_case()
    cutoff = decision_cutoff(date(2026, 7, 16))
    case.now = cutoff + timedelta(days=case.policy.pending_grace_days)
    report = _coverage_changed(case._report(samples=20), missing=("2026-07-16",))
    assert "monitoring_sessions_missing" not in case._evaluate(report).reasons
    case.now += timedelta(microseconds=1)
    blocked = case._evaluate(_coverage_changed(case._report(samples=20), missing=("2026-07-16",)))
    assert blocked.actionability == "not_ready" and "monitoring_sessions_missing" in blocked.reasons
    report = case._report(samples=20)
    failed = report["session_coverage"][0]["expected_sessions"][:10]
    warning = case._evaluate(_coverage_changed(report, failed=failed))
    assert (warning.state, warning.actionability) == ("unavailable", "not_ready")
    assert "monitoring_registered_session_share_low" in warning.reasons
    warming = case._evaluate(_coverage_changed(case._report(samples=5), failed=failed))
    assert (warming.state, warming.actionability) == ("warming", "rank_only")


def test_coverage_cannot_erase_expected_session_even_when_report_rehashed():
    case = _policy_case()
    report = case._report(samples=20)
    coverage = report["session_coverage"][0]
    coverage["expected_sessions"] = coverage["expected_sessions"][1:]
    coverage["registered_sessions"] = coverage["registered_sessions"][1:]
    coverage["source_record_ids"] = sorted(content_sha256(day) for day in coverage["registered_sessions"])
    report["report_id"] = content_sha256({key: value for key, value in report.items() if key != "report_id"})
    with pytest.raises(ValidationError, match="activation and window"):
        validate_performance_report(report)


def test_attempt_sources_are_asof_and_change_cohort_identity(tmp_path):
    repository = OutcomeRepository(tmp_path)
    intent = _intent_variant("MSFT", "1", probability=0.8)
    repository.record_intent(intent)
    commit_test_population(repository)
    as_of = datetime(2026, 8, 10, tzinfo=UTC)
    first = build_performance_cohorts(repository, generated_at=as_of)
    deciding = _attempt(intent, status="unresolvable", reasons=("cash_merger",), observed=as_of)
    repository.record_attempt(deciding, decision_session=intent.decision_session_et)
    second = build_performance_cohorts(repository, generated_at=as_of)
    assert second["source_attempt_ids"] == [deciding.attempt_id]
    first_row = next(row for row in first["rows"] if row["cohort_type"] == "all")
    second_row = next(row for row in second["rows"] if row["cohort_type"] == "all")
    assert second_row["source_attempt_ids_sha256"] == content_sha256([deciding.attempt_id])
    assert first_row["cohort_id"] != second_row["cohort_id"]
    assert second["report_id"] != first["report_id"]

    future = _attempt(intent, status="unresolvable", reasons=("stock_merger",), observed=as_of + timedelta(days=1))
    repository.record_attempt(future, decision_session=intent.decision_session_et)
    assert build_performance_cohorts(repository, generated_at=as_of) == second


def test_committed_intent_remains_pending_when_partial_write_owns_semantic_index(tmp_path):
    from market_predictor.governance.outcomes.contracts import maturation_key_sha256
    from tests.test_performance_monitoring import _record
    repository = OutcomeRepository(tmp_path)
    committed = _intent_variant("OLD", "1", probability=0.8, decision_time=datetime(2026, 3, 2, 22, tzinfo=UTC))
    partial = committed.model_copy(update={"snapshot_id": "2" * 64,
        "maturation_key": maturation_key_sha256("2" * 64, committed.semantic_prediction_id)})
    # Persist the successful inventory first in the synthetic fixture, then remove only the
    # semantic index, emulating an earlier partial snapshot having won that shared index.
    repository.record_intent(committed)
    recent = _intent_variant("RECENT", "1", probability=0.8)
    _record(repository, recent, target=1, net_return=0.02, excess_return=0.01)
    commit_test_population(repository)
    semantic_path = (repository.root / "sessions" / committed.decision_session_et.isoformat()
                     / "semantic" / f"{committed.semantic_prediction_id}.json")
    semantic_path.unlink()
    repository.record_intent(partial)
    repository.drop_pending(committed.maturation_key, committed.decision_session_et)
    report = build_performance_cohorts(repository, generated_at=datetime(2026, 8, 10, tzinfo=UTC), lookback_days=30)
    row = next(row for row in report["rows"] if row["cohort_type"] == "all")
    assert row["total_predictions"] == 1
    assert row["route_oldest_pending_decision_session_et"] == "2026-03-02"
    assert committed.maturation_key in report["source_intent_ids"]
    assert partial.maturation_key not in report["source_intent_ids"]



def test_future_outcome_cannot_clear_old_committed_pending_at_earlier_report_time(tmp_path):
    from market_predictor.governance.outcomes.contracts import MaturedOutcome
    from tests.test_outcome_repository import _evidence, _outcome
    repository = OutcomeRepository(tmp_path)
    old = _intent_variant("OLD", "1", probability=0.8, decision_time=datetime(2026, 3, 2, 22, tzinfo=UTC))
    recent = _intent_variant("RECENT", "1", probability=0.8)
    for intent in (old, recent):
        repository.record_intent(intent)
    commit_test_population(repository)
    as_of = datetime(2026, 8, 10, tzinfo=UTC)
    first = build_performance_cohorts(repository, generated_at=as_of, lookback_days=30)
    evidence = _evidence(old)
    content = _outcome(old, evidence).model_dump(mode="python", exclude={"outcome_id"})
    content["matured_at_utc"] = as_of + timedelta(days=1)
    content["label_available_at_utc"] = content["matured_at_utc"]
    outcome = MaturedOutcome.model_validate({**content, "outcome_id": content_sha256(content)})
    repository.record_outcome(old, outcome, evidence_rows=evidence)
    assert build_performance_cohorts(repository, generated_at=as_of, lookback_days=30) == first


def test_report_requires_new_session_and_attempt_evidence_fields():
    case = _policy_case()
    report = case._report(samples=20)
    report.pop("source_attempt_ids")
    report["report_id"] = content_sha256({key: value for key, value in report.items() if key != "report_id"})
    with pytest.raises(ValidationError, match="source_attempt_ids"):
        validate_performance_report(report)
