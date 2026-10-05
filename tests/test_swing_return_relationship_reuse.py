"""Synthetic comparison/inspection boundaries; no claims about real saved sources."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy
from market_predictor.swing.contracts.return_relationship_reuse import RelationshipReusePolicy
from market_predictor.swing.datasets import relationship_historical_evidence as historical
from market_predictor.swing.datasets import return_relationship_reuse as owner
from market_predictor.swing.datasets.adjusted_history_bindings import AdjustedHistoryBindings
from market_predictor.swing.datasets.adjusted_history_source import AdjustedHistorySource
from market_predictor.swing.datasets.predictor_abstention_derivation import ReviewedPredictorFailure
from market_predictor.swing.datasets.preserved_relationship_abstentions import (
    preserved_abstention_evidence,
    preserved_physical_prefix,
    read_preserved_stock,
)


def _bars() -> pd.DataFrame:
    starts = pd.to_datetime(["2024-05-24T13:30:00Z", "2024-05-28T13:30:00Z"]).as_unit("ns")
    return pd.DataFrame({"security_id": "test-security", "ticker": "AAA", "session_date_et": starts.date,
        "bar_start_utc": starts, "bar_end_utc": starts + pd.Timedelta(hours=6, minutes=30),
        "available_at_utc": starts + pd.Timedelta(hours=6, minutes=45),
        "open": [100.0, 101.0], "close": [101.0, 102.0], "volume": [1000.0, 1100.0],
        "price_feed": "sip", "adjustment": "all", "source": "alpaca", "timeframe": "1d",
        "availability_policy": "market_interval_close", "ingested_at_utc": pd.Timestamp("2026-09-01T00:00:00Z")})


def _compare(old: pd.DataFrame, current: pd.DataFrame) -> dict[str, Any]:
    return owner.compare_source_inputs(old, current, tuple(_bars().session_date_et), scope="synthetic-stock")


def test_absolute_prices_must_match_even_when_all_returns_match() -> None:
    old = _bars()
    current = old.copy()
    current[["open", "close"]] *= 2
    assert (old.close / old.open).equals(current.close / current.open)
    result = _compare(old, current)
    assert result["equal"] is False
    assert result["difference_counts"] == {"open": 2, "close": 2}


@pytest.mark.parametrize("column,value", [
    ("volume", 0.0), ("ticker", "BBB"), ("security_id", "another-issuer"),
    ("price_feed", "iex"), ("adjustment", "split"), ("source", "other"),
    ("availability_policy", "observed"), ("timeframe", "1h"),
])
def test_consumed_source_identity_and_physical_values_are_exact(column: str, value: Any) -> None:
    old, current = _bars(), _bars()
    current.loc[0, column] = value
    result = _compare(old, current)
    assert result["equal"] is False
    assert result["difference_counts"][column] == 1


@pytest.mark.parametrize("column", ["bar_start_utc", "bar_end_utc", "available_at_utc"])
def test_all_consumed_clocks_are_compared(column: str) -> None:
    old, current = _bars(), _bars()
    current.loc[0, column] += pd.Timedelta(nanoseconds=1)
    assert _compare(old, current)["difference_counts"][column] == 1


def test_capture_ingestion_clock_is_separate_from_consumed_proxy_clock() -> None:
    old, current = _bars(), _bars()
    current["ingested_at_utc"] += pd.Timedelta(days=1)
    result = _compare(old, current)
    assert result["equal"] is True
    assert result["ingestion_clocks_compared"] is False
    assert "capture_provenance" in result["ingestion_clock_role"]


def test_missing_observation_never_shifts_the_surviving_history() -> None:
    result = _compare(_bars(), _bars().iloc[1:])
    assert result["equal"] is False
    assert result["rows"] == 2
    assert result["difference_counts"]["present"] == 1


def test_duplicate_sessions_cannot_be_collapsed() -> None:
    current = pd.concat([_bars(), _bars().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataReadinessError, match="duplicate"):
        _compare(_bars(), current)


def test_every_trailing_position_is_compared_not_just_formula_endpoints() -> None:
    sessions = tuple(date(2020, 1, 1) + timedelta(days=index) for index in range(300))
    decisions = pd.DataFrame({"session_date_et": [sessions[252], sessions[260]]})
    stock = owner.consumed_sessions(decisions, sessions, benchmark=False)
    spy = owner.consumed_sessions(decisions, sessions, benchmark=True)
    assert stock == sessions[:261]
    assert spy == sessions[192:261]


def test_session_outside_independent_calendar_rejects() -> None:
    with pytest.raises(DataReadinessError, match="calendar"):
        owner.consumed_sessions(pd.DataFrame({"session_date_et": [date(2024, 1, 2)]}), (date(2024, 1, 3),), benchmark=False)


@pytest.mark.parametrize("column,values", [
    ("feature", [1.0, 0.0]), ("reason", ["", "new_reason"]),
    ("clock", pd.to_datetime(["2024-05-28T20:15Z", "2024-05-28T20:15Z"])),
])
def test_saved_feature_values_nulls_reasons_and_clocks_are_exact(column: str, values: Any) -> None:
    old = pd.DataFrame({"feature": [1.0, float("nan")], "reason": ["", "missing_history"],
        "clock": pd.to_datetime(["2024-05-28T20:15Z", None])})
    current = old.copy()
    current[column] = values
    assert owner.compare_frames(old, current, list(old), scope="additions")["equal"] is False


def test_boolean_is_not_a_numeric_feature_value() -> None:
    assert owner.compare_frames(pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [True]}), ["x"], scope="test")["equal"] is False


def test_inherited_dtype_and_row_order_are_preserved() -> None:
    old = pd.DataFrame({"decision_id": ["one", "two"], "value": pd.Series([1, 2], dtype="float32")})
    current = old.astype({"value": "float64"})
    assert owner.compare_frames(old, current, list(old), scope="parent", check_dtype=True)["equal"] is False
    assert owner.compare_frames(old, old.iloc[::-1], list(old), scope="parent")["equal"] is False


def _historical_bar_fixture(root: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    frame = _bars().drop(columns=["security_id", "session_date_et"])
    frame["high"], frame["low"] = frame.close + 1, frame.open - 1
    frame["schema_version"] = historical.HISTORICAL_CANONICAL_SCHEMA
    later = frame.iloc[[0]].copy()
    later["bar_start_utc"] = pd.Timestamp("2024-05-29T13:30Z")
    later["bar_end_utc"] = pd.Timestamp("2024-05-29T20:00Z")
    later["available_at_utc"] = pd.Timestamp("2024-05-29T20:15Z")
    later["close"] = 999999.0
    frame = pd.concat([frame, later], ignore_index=True)
    path = root / "old/bars/AAA.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    sidecar = {"schema": historical.HISTORICAL_MANIFEST_SCHEMA,
        "canonical_schema_version": historical.HISTORICAL_CANONICAL_SCHEMA,
        "artifact_type": "bars", "artifact_sha256": file_sha256(path), "rows": len(frame),
        "columns": list(frame.columns), "production_ready": True, "inputs": {}}
    write_json_object(manifest_path_for(path), sidecar)
    record = {"path": "bars/AAA.parquet", "ticker": "AAA", "rows": len(frame), "sha256": file_sha256(path),
        "canonical_manifest_sha256": file_sha256(manifest_path_for(path))}
    request = {"source_files": {path.relative_to(root).as_posix(): record["sha256"],
        manifest_path_for(path).relative_to(root).as_posix(): record["canonical_manifest_sha256"]}}
    return request, {"artifact": record, "kind": "combined", "security_id": "test-security"}, path


def test_historical_bar_inspection_projects_only_initial_fit(tmp_path: Path) -> None:
    request, item, path = _historical_bar_fixture(tmp_path)
    files: dict[str, str] = {}
    result = historical.historical_bars(tmp_path, request, item, files)
    assert len(result) == 2
    assert max(result.session_date_et) == date(2024, 5, 28)
    assert result.close.tolist() == [101.0, 102.0]
    assert files[path.relative_to(tmp_path).as_posix()] == item["artifact"]["sha256"]
    with pytest.raises(DataReadinessError, match="schema"):
        load_canonical_artifact(path, expected_type="bars", allow_research=True)


def test_historical_inspection_rejects_changed_raw_bytes(tmp_path: Path) -> None:
    request, item, path = _historical_bar_fixture(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(DataReadinessError, match="changed"):
        historical.historical_bars(tmp_path, request, item, {})


def test_historical_inspection_rejects_changed_sidecar(tmp_path: Path) -> None:
    request, item, path = _historical_bar_fixture(tmp_path)
    manifest_path_for(path).write_text("{}", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="hash"):
        historical.historical_bars(tmp_path, request, item, {})


def test_historical_source_requires_unique_path_and_digest(tmp_path: Path) -> None:
    request, item, _ = _historical_bar_fixture(tmp_path)
    request["source_files"]["another/bars/AAA.parquet"] = item["artifact"]["sha256"]
    with pytest.raises(DataReadinessError, match="one exact"):
        historical.historical_bars(tmp_path, request, item, {})


def test_report_publication_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "proof.json"
    owner._publish_report(path, {"status": "failed_differences"})
    original = path.read_bytes()
    with pytest.raises(DataReadinessError, match="overwrite"):
        owner._publish_report(path, {"status": "passed_exact_reuse"})
    assert path.read_bytes() == original


def _policy() -> RelationshipReusePolicy:
    pin = SourcePin(path="fixture.json", sha256="1" * 64)
    return RelationshipReusePolicy(schema_version="market_predictor.relationship_reuse_config",
        historical_relationship_publication=pin, historical_relationship_receipt=pin,
        historical_parent_publication=pin, historical_parent_receipt=pin, peer_reuse_report=pin,
        configuration_difference_report=pin, feature_config=pin, strategy_contract=pin)


def test_unreviewed_peer_or_configuration_report_cannot_be_substituted(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="exact separately reviewed"):
        owner.inspect_reuse_evidence(tmp_path, _policy())


def test_current_policy_has_explicit_authority_and_no_obsolete_alias() -> None:
    pin = SourcePin(path="fixture.json", sha256="1" * 64)
    fields = dict(schema_version="market_predictor.return_relationship_publication_config",
        parent_publication=pin, parent_saved_row_verification=pin, feature_config=pin,
        strategy_contract=pin, reuse_equivalence_authority=pin)
    assert ReturnRelationshipPublicationPolicy(**fields).reuse_equivalence_authority == pin
    with pytest.raises(ValidationError, match="feature_plan_snapshot"):
        ReturnRelationshipPublicationPolicy(**fields, feature_plan_snapshot=pin)


def test_failed_comparison_cannot_authorize_a_current_parent(tmp_path: Path) -> None:
    reuse = _policy()
    config = tmp_path / "reuse.json"
    write_json_object(config, reuse.model_dump(mode="json"))
    report = tmp_path / "failed.json"
    write_json_object(report, {"schema": owner.SCHEMA, "status": "failed_differences",
        "policy": reuse.model_dump(mode="json"),
        "config": {"path": "reuse.json", "sha256": file_sha256(config)}})
    policy = ReturnRelationshipPublicationPolicy(schema_version="market_predictor.return_relationship_publication_config",
        parent_publication=reuse.historical_parent_publication, parent_saved_row_verification=reuse.historical_parent_receipt,
        feature_config=reuse.feature_config, strategy_contract=reuse.strategy_contract,
        reuse_equivalence_authority=SourcePin(path="failed.json", sha256=file_sha256(report)))
    with pytest.raises(DataReadinessError, match="failed, stale"):
        owner.verified_reuse_parent(tmp_path, policy)


def test_missing_explicit_equivalence_pin_never_uses_historical_inspection(tmp_path: Path) -> None:
    reuse = _policy()
    policy = ReturnRelationshipPublicationPolicy(schema_version="market_predictor.return_relationship_publication_config",
        parent_publication=reuse.historical_parent_publication, parent_saved_row_verification=reuse.historical_parent_receipt,
        feature_config=reuse.feature_config, strategy_contract=reuse.strategy_contract)
    with pytest.raises(DataReadinessError, match="explicit equivalence"):
        owner.verified_reuse_parent(tmp_path, policy)


def _historical_parent_fixture(root: Path, *, admitted_request: bool = False) -> tuple[SourcePin, SourcePin]:
    """Synthetic unit fixture matching the frozen baseline metadata ownership."""
    directory = root / "historical-parent"
    directory.mkdir()
    # Exact frozen request convention: admission flags were absent, not false.
    request = {"schema": "market_predictor.research_join_request", "rows": 59, "historical_first_seen_proven": False}
    if admitted_request:
        request["training_eligible"] = True
    write_json_object(directory / "_request.json", request)
    request_pin = file_sha256(directory / "_request.json")
    months: dict[str, Any] = {}
    for month in pd.period_range("2019-07", "2024-05", freq="M").astype(str):
        path = directory / month / "technical_market.parquet"
        path.parent.mkdir()
        frame = pd.DataFrame({"decision_id": [f"decision-{month}"]})
        frame.to_parquet(path, index=False)
        sidecar = {"schema": historical.HISTORICAL_MANIFEST_SCHEMA,
            "canonical_schema_version": historical.HISTORICAL_CANONICAL_SCHEMA,
            "artifact_type": "swing_research_join", "artifact_sha256": file_sha256(path), "rows": 1,
            "columns": list(frame), "production_ready": False, "inputs": {"request_sha256": request_pin}}
        write_json_object(manifest_path_for(path), sidecar)
        identities = json_sha256(list(frame.decision_id))
        child = {"path": f"{month}/technical_market.parquet", "sha256": file_sha256(path),
            "manifest_sha256": file_sha256(manifest_path_for(path)), "audit": {"rows": 1},
            "model_columns": [], "availability_columns": []}
        months[month] = {"rows": 1, "decision_ids_sha256": identities, "profiles": {"technical_market": child}}
    manifest = {"schema": "market_predictor.research_join", "status": "complete_research_only", "exclusions_added": [],
        "request_sha256": request_pin, "months": months, "rows": 59, "training_eligible": False, "promotion_eligible": False}
    write_json_object(directory / "_manifest.json", manifest)
    publication = SourcePin(path="historical-parent/_manifest.json", sha256=file_sha256(directory / "_manifest.json"))
    receipt = {"status": "passed", "manifest_sha256": publication.sha256, "scope": "published_join_population_clocks_original_targets",
        "months": 59, "unique_decisions": 59, "outcome_filtered_rows": 0, "matched_profile_population": True,
        "original_outcome_values_exact": True, "training_eligible": False, "promotion_eligible": False}
    write_json_object(root / "original-receipt.json", receipt)
    return publication, SourcePin(path="original-receipt.json", sha256=file_sha256(root / "original-receipt.json"))


def test_actual_historical_parent_request_shape_does_not_invent_admission_flags(tmp_path: Path) -> None:
    publication, receipt = _historical_parent_fixture(tmp_path)
    result = historical.inspect_historical_publication(tmp_path, publication, receipt, profile="technical_market")
    assert result.manifest["rows"] == 59
    assert "training_eligible" not in result.request and "promotion_eligible" not in result.request
    assert result.receipt["training_eligible"] is False


def test_historical_parent_request_positive_admission_flag_rejects(tmp_path: Path) -> None:
    publication, receipt = _historical_parent_fixture(tmp_path, admitted_request=True)
    with pytest.raises(DataReadinessError, match="original receipt"):
        historical.inspect_historical_publication(tmp_path, publication, receipt, profile="technical_market")


def _preserved_prefix_inputs() -> tuple[pd.DataFrame, ReviewedPredictorFailure, dict[str, Any]]:
    frame = _bars()
    frame["high"], frame["low"] = frame.close + 1, frame.open - 1
    frame["schema_version"] = "market_data"
    frame.loc[1, "volume"] = 0.0
    boundary = frame.iloc[1]
    observation = {"session_date_et": str(boundary.session_date_et), "invalid_fields": ["volume"],
        "ohlcv": {name: float(boundary[name]) for name in ("open", "high", "low", "close", "volume")},
        "clocks": {name: pd.Timestamp(boundary[name]).isoformat() for name in ("bar_start_utc", "bar_end_utc", "available_at_utc")}}
    pin = SourcePin(path="original.json", sha256="1" * 64)
    fact = ReviewedPredictorFailure(group_key="2" * 64, security_id="test-security", symbol="AAA", rows=2,
        parent_failure_sha256="3" * 64, reason_code="invalid_observation_stream", quarantine="suffix_from_first_invalid",
        first_invalid_session=date(2024, 5, 28), boundary_observation_sha256=json_sha256(observation),
        source_artifacts=(pin,), reviewed_evidence=(pin,), detail="Synthetic immutable zero-volume boundary.")
    return frame, fact, {"bounded_rows": 2, "invalid_rows": [observation]}


def test_preserved_prefix_uses_existing_validator_and_keeps_physical_metadata() -> None:
    frame, fact, observation = _preserved_prefix_inputs()
    result = preserved_physical_prefix(frame, fact, observation, tuple(frame.session_date_et))
    pd.testing.assert_frame_equal(result, frame.iloc[:1].reset_index(drop=True))
    assert observation["bounded_rows"] == 2
    assert fact.group_key == "2" * 64


@pytest.mark.parametrize("poison", ["earlier_invalid", "missing_boundary", "valid_boundary", "repriced_boundary", "retimed_boundary"])
def test_preserved_prefix_rejects_changes_to_original_boundary_or_clean_prefix(poison: str) -> None:
    frame, fact, observation = _preserved_prefix_inputs()
    if poison == "earlier_invalid":
        frame.loc[0, "volume"] = 0.0
    elif poison == "missing_boundary":
        frame = frame.iloc[:1]
    elif poison == "valid_boundary":
        frame.loc[1, "volume"] = 1.0
    elif poison == "repriced_boundary":
        frame.loc[1, "close"] += 0.25
    else:
        frame.loc[1, "available_at_utc"] += pd.Timedelta(nanoseconds=1)
    with pytest.raises(DataReadinessError):
        preserved_physical_prefix(frame, fact, observation, tuple(frame.session_date_et))


def test_preserved_prefix_never_rewrites_original_observation_row_count() -> None:
    frame, fact, observation = _preserved_prefix_inputs()
    observation["bounded_rows"] = 3
    with pytest.raises(DataReadinessError, match="complete bounded"):
        preserved_physical_prefix(frame, fact, observation, tuple(frame.session_date_et))
    assert observation["bounded_rows"] == 3


def test_whole_history_unknown_stays_empty_with_no_identity_resolution() -> None:
    frame, fact, observation = _preserved_prefix_inputs()
    unknown = fact.model_copy(update={"reason_code": "unverified_issuer_history", "quarantine": "entire_failed_group",
        "first_invalid_session": None, "boundary_observation_sha256": None})
    result = preserved_physical_prefix(frame, unknown, observation, tuple(frame.session_date_et))
    assert result.empty and list(result) == list(frame)


def _scope_evidence_fixture(root: Path) -> tuple[dict[str, Any], AdjustedHistoryBindings, tuple[date, ...]]:
    frame, fact, observation = _preserved_prefix_inputs()
    old_directory = root / "old"
    old_directory.mkdir()
    old_source = old_directory / "raw.bin"
    old_source.write_bytes(b"synthetic historical source, not an admitted current archive")
    old_source_pin = SourcePin(path="old/raw.bin", sha256=file_sha256(old_source))
    report_path = old_directory / "observations.json"
    write_json_object(report_path, {"schema": "market_predictor.predictor_source_failure_observations",
        "numeric_first": "2018-05-29", "numeric_last": "2024-05-28", "observations": [
            {**observation, "security_id": fact.security_id, "ticker": fact.symbol,
                "source_path": old_source_pin.path, "source_sha256": old_source_pin.sha256}]})
    report_pin = SourcePin(path="old/observations.json", sha256=file_sha256(report_path))
    fact = fact.model_copy(update={"source_artifacts": (old_source_pin,), "reviewed_evidence": (report_pin,)})
    identifiers = json_sha256(["decision-one", "decision-two"])
    old_request = {"stock_inventory": {fact.group_key: {"security_id": fact.security_id, "rows": 2,
        "decision_ids_sha256": identifiers, "quarantine": fact.model_dump(mode="json")}},
        "source_files": {old_source_pin.path: old_source_pin.sha256, report_pin.path: report_pin.sha256}}
    request_path = old_directory / "_request.json"
    write_json_object(request_path, old_request)
    manifest_path = old_directory / "_manifest.json"
    write_json_object(manifest_path, {"request_sha256": file_sha256(request_path)})
    receipt_path = old_directory / "receipt.json"
    write_json_object(receipt_path, {"manifest_sha256": file_sha256(manifest_path), "status": "passed", "additions_source_replayed": True})
    current_directory = root / "current"
    current_directory.mkdir()
    source_path = current_directory / "bars.parquet"
    source_path.write_bytes(b"not read: negative current-binding tests stop before strict archive reader")
    current_pin = SourcePin(path="current/bars.parquet", sha256=file_sha256(source_path))
    window = {"security_id": fact.security_id, "ticker": fact.symbol, "role": "stock",
        "start_date": "2018-05-29", "end_date": "2024-05-28"}
    record = {**window, "bars_path": "bars.parquet", "bars_sha256": current_pin.sha256}
    sessions = tuple(frame.session_date_et)
    mapping = {"historical_group_key": fact.group_key, "historical_fact": fact.model_dump(mode="json"),
        "historical_request": {"path": "old/_request.json", "sha256": file_sha256(request_path)},
        "historical_receipt": {"path": "old/receipt.json", "sha256": file_sha256(receipt_path)},
        "historical_publication": {"path": "old/_manifest.json", "sha256": file_sha256(manifest_path)},
        "observation": report_pin.model_dump(mode="json"), "current_unit_id": "current-query",
        "current_source": current_pin.model_dump(mode="json"), "security_id": fact.security_id, "rows": 2,
        "decision_ids_sha256": identifiers, "parent_window": dict(window), "query_window": dict(window),
        "expected_sessions_sha256": json_sha256([str(day) for day in sessions]),
        "scope": "unchanged_valid_prefix_only", "source_issuer_verified": False}
    source = AdjustedHistorySource(root, current_directory, MappingProxyType({"current-query": record}),
        MappingProxyType({current_pin.path: current_pin.sha256}))
    bindings = AdjustedHistoryBindings(
        source, MappingProxyType({"current-query": {"parent": window, "query": window}}), source.source_files,
    )
    return mapping, bindings, sessions


def test_preserved_scope_references_keep_original_group_and_sources(tmp_path: Path) -> None:
    mapping, _, _ = _scope_evidence_fixture(tmp_path)
    fact, observation, files = preserved_abstention_evidence(tmp_path, mapping)
    assert fact.group_key == mapping["historical_group_key"]
    assert fact.source_artifacts[0].path == "old/raw.bin"
    assert observation["bounded_rows"] == 2
    assert {"old/raw.bin", "old/observations.json", "old/receipt.json"}.issubset(files)
    assert mapping["source_issuer_verified"] is False


@pytest.mark.parametrize("path", ["old/raw.bin", "old/observations.json", "old/_request.json", "old/receipt.json"])
def test_preserved_scope_rejects_historical_evidence_tamper(tmp_path: Path, path: str) -> None:
    mapping, _, _ = _scope_evidence_fixture(tmp_path)
    source = tmp_path / path
    source.write_bytes(source.read_bytes() + b"changed")
    with pytest.raises(DataReadinessError):
        preserved_abstention_evidence(tmp_path, mapping)


@pytest.mark.parametrize("poison", ["security", "query", "membership", "population", "boundary", "extra_field"])
def test_preserved_scope_rejects_changed_current_ownership_or_historical_scope(tmp_path: Path, poison: str) -> None:
    mapping, bindings, sessions = _scope_evidence_fixture(tmp_path)
    if poison == "security":
        mapping["security_id"] = "different-issuer"
    elif poison == "query":
        mapping["current_unit_id"] = "different-query"
    elif poison == "membership":
        sessions = sessions[1:]
    elif poison == "population":
        mapping["decision_ids_sha256"] = "f" * 64
    elif poison == "boundary":
        mapping["historical_fact"]["first_invalid_session"] = "2024-05-24"
    else:
        mapping["current_failure_checkpoint"] = "invented"
    with pytest.raises(DataReadinessError):
        read_preserved_stock(tmp_path, bindings, "current-query", mapping, sessions)
