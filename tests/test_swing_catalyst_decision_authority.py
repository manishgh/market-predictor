from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.reconciliation import ASSIGNMENT_COLUMNS, reconciliation_sha256
from market_predictor.canonical.store import file_sha256, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.features.catalyst_decision_authority import (
    COVERAGE_ARTIFACT_TYPE,
    DECISION_ARTIFACT_TYPE,
    DECISION_AUTHORITY_SCHEMA,
    DECISION_MANIFEST_SCHEMA,
    DECISION_REQUEST_SCHEMA,
    LINEAGE_MANIFEST_SCHEMA,
    RANKING_SOURCE_FAMILIES,
    _deduplicate_verified_events,
    attach_catalyst_decision_features,
    load_catalyst_decision_authority,
    publish_catalyst_decision_authority,
)

DECISION_TIME = pd.Timestamp("2025-01-10T21:00:00Z")


def test_persisted_catalyst_decision_authority_identities_are_frozen() -> None:
    assert LINEAGE_MANIFEST_SCHEMA == "swing.catalyst_lineage_manifest.v2"
    assert DECISION_REQUEST_SCHEMA == "edge_rebuild.catalyst_decision_request.v3"
    assert DECISION_AUTHORITY_SCHEMA == "edge_rebuild.catalyst_decision_authority.v7"
    assert DECISION_MANIFEST_SCHEMA == "edge_rebuild.catalyst_decision_manifest.v7"
    assert DECISION_ARTIFACT_TYPE == "edge_rebuild_catalyst_decisions"
    assert COVERAGE_ARTIFACT_TYPE == "edge_rebuild_catalyst_coverage"


def test_unrelated_issuer_events_within_thirty_minutes_are_both_retained() -> None:
    first = _deduplication_row(
        event_id="event-a",
        source_event_id="source-event-a",
        decision_id="decision-a",
        security_id="security-a",
        ticker="AAA",
        source_family="sec",
        available_at=DECISION_TIME - pd.Timedelta(minutes=25),
        content="issuer-a-content",
    )
    second = _deduplication_row(
        event_id="event-b",
        source_event_id="source-event-b",
        decision_id="decision-b",
        security_id="security-b",
        ticker="BBB",
        source_family="finviz",
        available_at=DECISION_TIME - pd.Timedelta(minutes=5),
        content="issuer-b-content",
    )

    retained = _deduplicate_verified_events(pd.DataFrame.from_records([first, second]))

    assert set(retained["event_id"]) == {"event-a", "event-b"}


def test_same_issuer_and_content_identity_is_deduplicated() -> None:
    first = _deduplication_row(
        event_id="event-a",
        source_event_id="source-event-a",
        decision_id="decision-a",
        security_id="security-a",
        ticker="AAA",
        source_family="alpaca",
        available_at=DECISION_TIME - pd.Timedelta(minutes=25),
        content="same-content",
    )
    second = _deduplication_row(
        event_id="event-b",
        source_event_id="source-event-b",
        decision_id="decision-a",
        security_id="security-a",
        ticker="AAA",
        source_family="finviz",
        available_at=DECISION_TIME - pd.Timedelta(minutes=5),
        content="same-content",
    )

    retained = _deduplicate_verified_events(pd.DataFrame.from_records([first, second]))

    assert retained["event_id"].tolist() == ["event-a"]


def test_content_scores_from_separate_decisions_are_not_conflicting_duplicates() -> None:
    first = _deduplication_row(event_id="event-a", source_event_id="source-a", decision_id="decision-a",
        security_id="security-a", ticker="AAA", source_family="alpaca",
        available_at=DECISION_TIME - pd.Timedelta(days=12), content="same-content")
    second = {**first, "event_id": "event-b", "source_event_id": "source-b", "decision_id": "decision-b",
        "feature_available_at_utc": DECISION_TIME - pd.Timedelta(days=2),
        "sentiment_numeric": first["sentiment_numeric"] + 5.960464477539063e-08}
    retained = _deduplicate_verified_events(pd.DataFrame([first, second]))
    assert set(retained.event_id) == {"event-a", "event-b"}
    assert retained.set_index("event_id").loc["event-b", "sentiment_numeric"] == second["sentiment_numeric"]
    same_decision = _deduplicate_verified_events(pd.DataFrame([first, {**second, "decision_id": "decision-a"}]))
    assert same_decision.event_id.tolist() == ["event-a"]
    assert same_decision.sentiment_numeric.iloc[0] == first["sentiment_numeric"]


def test_separate_same_text_publications_keep_earliest_original_relevance_and_score() -> None:
    first = _deduplication_row(event_id="event-a", source_event_id="source-a", decision_id="decision-a",
        security_id="security-a", ticker="AAA", source_family="alpaca",
        available_at=DECISION_TIME - pd.Timedelta(hours=3), content="same-content")
    second = {**first, "event_id": "event-b", "source_event_id": "source-b",
        "feature_available_at_utc": DECISION_TIME - pd.Timedelta(hours=1),
        "sentiment_numeric": -0.5, "relevance": 0.96}
    for rows in ([first, second], [second, first]):
        retained = _deduplicate_verified_events(pd.DataFrame(rows))
        assert retained.event_id.tolist() == ["event-a"]
        assert retained.sentiment_numeric.iloc[0] == first["sentiment_numeric"]
        assert retained.relevance.iloc[0] == first["relevance"]
        assert retained.feature_available_at_utc.iloc[0] == first["feature_available_at_utc"]


@pytest.mark.parametrize("column", ["sentiment_numeric", "relevance"])
def test_same_provider_event_still_rejects_conflicting_scores(column: str) -> None:
    first = _deduplication_row(event_id="event-a", source_event_id="source-a", decision_id="decision-a",
        security_id="security-a", ticker="AAA", source_family="alpaca",
        available_at=DECISION_TIME - pd.Timedelta(hours=3), content="same-content")
    for event in ("event-a", "another-relation"):
        second = {**first, "event_id": event, column: first[column] + 0.01}
        with pytest.raises(DataReadinessError, match="durable catalyst"):
            _deduplicate_verified_events(pd.DataFrame([first, second]))


def test_same_text_ties_keep_source_priority_then_event_identity() -> None:
    first = _deduplication_row(event_id="event-a", source_event_id="source-a", decision_id="decision-a",
        security_id="security-a", ticker="AAA", source_family="alpaca",
        available_at=DECISION_TIME - pd.Timedelta(hours=1), content="same-content")
    second = {**first, "event_id": "event-b", "source_event_id": "source-b", "sentiment_numeric": -0.5}
    external = {**second, "event_id": "event-0", "source_event_id": "source-0", "source_family": "finviz"}
    for rows in ([first, second, external], [external, second, first]):
        retained = _deduplicate_verified_events(pd.DataFrame(rows))
        assert retained.event_id.tolist() == ["event-a"]
        assert retained.sentiment_numeric.iloc[0] == first["sentiment_numeric"]


@pytest.mark.parametrize("mutation", ["missing_policy", "wrong_policy", "old_request", "old_authority"])
def test_loader_rejects_rehashed_wrong_duplicate_policy(tmp_path: Path, mutation: str) -> None:
    lineage = _lineage(tmp_path / "lineage", generation="1")
    output = tmp_path / "published"
    publish_catalyst_decision_authority([lineage], output)
    manifest_path = output / "_manifest.json"
    authority_path = output / "_authority.json"
    manifest = json.loads(manifest_path.read_text())
    authority = json.loads(authority_path.read_text())
    if mutation == "missing_policy":
        manifest["request"].pop("text_duplicate_policy")
    elif mutation == "wrong_policy":
        manifest["request"]["text_duplicate_policy"] = "average_scores"
    elif mutation == "old_request":
        manifest["request"]["schema"] = "edge_rebuild.catalyst_decision_request.v2"
    else:
        authority["schema"] = "edge_rebuild.catalyst_decision_authority.v6"
    manifest["request_sha256"] = _json_sha256(manifest["request"])
    authority["request_sha256"] = manifest["request_sha256"]
    _write_json(manifest_path, manifest)
    authority["artifact_sha256"] = file_sha256(manifest_path)
    _write_json(authority_path, authority)
    with pytest.raises(DataReadinessError, match="policy differs|authority does not verify"):
        load_catalyst_decision_authority(output)


def test_conflicting_durable_event_identity_fails_closed() -> None:
    first = _deduplication_row(
        event_id="event-a",
        source_event_id="source-event-a",
        decision_id="decision-a",
        security_id="security-a",
        ticker="AAA",
        source_family="alpaca",
        available_at=DECISION_TIME - pd.Timedelta(minutes=25),
        content="same-content",
    )
    conflicting = {
        **first,
        "decision_id": "decision-b",
        "security_id": "security-b",
        "ticker": "BBB",
    }

    with pytest.raises(DataReadinessError, match="conflicting durable catalyst event identity"):
        _deduplicate_verified_events(pd.DataFrame.from_records([first, conflicting]))


def test_publishes_deduplicated_direct_decision_authority_and_attaches_known_zeros(
    tmp_path: Path,
) -> None:
    first = _lineage(tmp_path / "lineage-1", generation="1")
    second = _lineage(tmp_path / "lineage-2", generation="2")

    authority = publish_catalyst_decision_authority(
        [second, first],
        tmp_path / "authority",
    )

    assert authority.manifest["eligible_assignment_rows_read"] == 4
    assert authority.manifest["unique_assignment_rows"] == 2
    assert authority.manifest["duplicate_assignment_rows_merged"] == 2
    assert authority.decisions["event_count_1d"].tolist() == [1]
    assert authority.decisions["event_count_3d"].tolist() == [1]
    assert authority.decisions["source_count_alpaca_3d"].tolist() == [1.0]
    assert authority.decisions["source_count_sec_3d"].tolist() == [0.0]
    assert authority.decisions["evidence_lineage_count"].tolist() == [2]
    assert authority.manifest["sentiment_scorer_identity"] == {
        "model": "ProsusAI/finbert",
        "revision": "revision-1",
    }
    decisions = pd.DataFrame(
        {
            "decision_id": ["decision-1", "decision-2"],
            "security_id": ["security-1", "security-1"],
            "ticker": ["ABC", "ABC"],
            "decision_time_utc": [
                DECISION_TIME,
                DECISION_TIME + pd.Timedelta(hours=1),
            ],
        }
    )
    attached = attach_catalyst_decision_features(decisions, authority)

    assert attached["catalyst_source_complete_3d"].tolist() == [True, True]
    assert attached["event_count_3d"].tolist() == [1.0, 0.0]
    assert attached["source_count_alpaca_3d"].tolist() == [1.0, 0.0]
    for family in RANKING_SOURCE_FAMILIES[1:]:
        assert attached[f"source_count_{family}_3d"].tolist() == [0.0, 0.0]


def test_replay_rejects_stale_catalyst_authority_version(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path / "lineage", generation="1")
    output = tmp_path / "authority"
    publish_catalyst_decision_authority([lineage], output)
    authority_path = output / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["schema"] = "edge_rebuild.catalyst_decision_authority.v3"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="does not verify"):
        load_catalyst_decision_authority(output)


def test_attachment_preserves_unknown_coverage_as_missing(tmp_path: Path) -> None:
    lineage = _lineage(
        tmp_path / "lineage",
        generation="1",
        unknown_source="alpaca",
    )
    authority = publish_catalyst_decision_authority([lineage], tmp_path / "authority")
    decisions = pd.DataFrame(
        {
            "decision_id": ["decision-1", "decision-2"],
            "security_id": ["security-1", "security-1"],
            "ticker": ["ABC", "ABC"],
            "decision_time_utc": [DECISION_TIME, DECISION_TIME + pd.Timedelta(hours=1)],
        }
    )

    attached = attach_catalyst_decision_features(decisions, authority)

    assert attached["catalyst_source_complete_3d"].tolist() == [False, False]
    assert attached["event_count_3d"].isna().all()
    assert attached["source_count_alpaca_3d"].isna().all()


def test_optional_source_gap_does_not_erase_required_alpaca_evidence(
    tmp_path: Path,
) -> None:
    lineage = _lineage(
        tmp_path / "lineage",
        generation="1",
        unknown_source="sec",
    )
    authority = publish_catalyst_decision_authority([lineage], tmp_path / "authority")
    decisions = pd.DataFrame(
        {
            "decision_id": ["decision-1", "decision-2"],
            "security_id": ["security-1", "security-1"],
            "ticker": ["ABC", "ABC"],
            "decision_time_utc": [
                DECISION_TIME,
                DECISION_TIME + pd.Timedelta(hours=1),
            ],
        }
    )

    attached = attach_catalyst_decision_features(decisions, authority)

    assert attached["catalyst_source_complete_3d"].tolist() == [True, True]
    assert attached["event_count_3d"].tolist() == [1.0, 0.0]
    assert attached["source_count_sec_3d"].isna().all()
    assert attached["source_count_alpaca_3d"].tolist() == [1.0, 0.0]


def test_optional_direct_source_cannot_change_alpaca_model_aggregates(
    tmp_path: Path,
) -> None:
    lineage = _lineage(
        tmp_path / "lineage",
        generation="1",
        optional_direct_source="sec",
    )

    authority = publish_catalyst_decision_authority([lineage], tmp_path / "authority")

    assert authority.decisions["event_count_3d"].tolist() == [1]
    assert authority.decisions["source_count_alpaca_3d"].tolist() == [1.0]
    assert authority.decisions["source_count_sec_3d"].tolist() == [0.0]


def test_rejects_unknown_coverage_claimed_as_zero(tmp_path: Path) -> None:
    lineage = _lineage(
        tmp_path / "lineage",
        generation="1",
        unknown_source="sec",
        poison_unknown_zero=True,
    )

    with pytest.raises(DataReadinessError, match="known-zero/unknown"):
        publish_catalyst_decision_authority([lineage], tmp_path / "authority")


def test_strict_loader_rejects_tampered_decision_artifact(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path / "lineage", generation="1")
    authority = publish_catalyst_decision_authority([lineage], tmp_path / "authority")
    path = authority.directory / "decision_catalysts.parquet"
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(DataReadinessError, match="integrity"):
        load_catalyst_decision_authority(authority.directory)


def test_attachment_rejects_post_decision_feature_availability(tmp_path: Path) -> None:
    lineage = _lineage(tmp_path / "lineage", generation="1")
    authority = publish_catalyst_decision_authority([lineage], tmp_path / "authority")
    authority.decisions.loc[0, "latest_event_feature_available_at_utc"] = DECISION_TIME + pd.Timedelta(seconds=1)
    decisions = pd.DataFrame(
        {
            "decision_id": ["decision-1"],
            "security_id": ["security-1"],
            "ticker": ["ABC"],
            "decision_time_utc": [DECISION_TIME],
        }
    )

    with pytest.raises(DataReadinessError, match="after decision_time_utc"):
        attach_catalyst_decision_features(decisions, authority)


def test_collection_completed_after_decision_cannot_prove_known_zero(
    tmp_path: Path,
) -> None:
    lineage = _lineage(
        tmp_path / "lineage",
        generation="1",
        coverage_completion_offset=pd.Timedelta(minutes=1),
        production_ready=True,
    )
    authority = publish_catalyst_decision_authority(
        [lineage],
        tmp_path / "authority",
        production_ready=True,
    )
    decisions = pd.DataFrame(
        {
            "decision_id": ["decision-2"],
            "security_id": ["security-1"],
            "ticker": ["ABC"],
            "decision_time_utc": [DECISION_TIME],
        }
    )

    attached = attach_catalyst_decision_features(decisions, authority)

    assert attached["catalyst_source_complete_3d"].tolist() == [False]
    assert attached["event_count_3d"].isna().all()


def test_rejects_mixed_sentiment_scorer_identity_across_generations(
    tmp_path: Path,
) -> None:
    first = _lineage(
        tmp_path / "lineage-1",
        generation="1",
        scorer_revision="revision-1",
    )
    second = _lineage(
        tmp_path / "lineage-2",
        generation="2",
        scorer_revision="revision-2",
    )

    with pytest.raises(DataReadinessError, match="mixed sentiment scorer identity"):
        publish_catalyst_decision_authority(
            [first, second],
            tmp_path / "authority",
        )


def test_production_authority_requires_production_lineage_and_load_mode(
    tmp_path: Path,
) -> None:
    lineage = _lineage(
        tmp_path / "lineage",
        generation="1",
        production_ready=True,
    )
    authority = publish_catalyst_decision_authority(
        [lineage],
        tmp_path / "authority",
        production_ready=True,
    )

    loaded = load_catalyst_decision_authority(
        authority.directory,
        require_production_ready=True,
    )

    assert loaded.manifest["production_ready"] is True
    with pytest.raises(DataReadinessError, match="required authority mode"):
        load_catalyst_decision_authority(
            authority.directory,
            require_production_ready=False,
        )


@pytest.mark.parametrize("field,value", [
    ("scope_policy", "legacy"), ("source_coverage_admitted", True),
    ("excluded_security_ids", ["security-a"]),
])
def test_feature_publication_rejects_obsolete_or_certified_scope(tmp_path: Path, field: str, value: object) -> None:
    lineage = _lineage(tmp_path / "lineage", generation="observed-policy")
    path = lineage / "_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(path, manifest)
    with pytest.raises(DataReadinessError, match="scope policy"):
        publish_catalyst_decision_authority([lineage], tmp_path / "authority")


def _lineage(
    root: Path,
    *,
    generation: str,
    unknown_source: str | None = None,
    poison_unknown_zero: bool = False,
    optional_direct_source: str | None = None,
    coverage_completion_offset: pd.Timedelta = pd.Timedelta(0),
    scorer_model: str = "ProsusAI/finbert",
    scorer_revision: str = "revision-1",
    production_ready: bool = False,
) -> Path:
    root.mkdir(parents=True)
    (root / "events").mkdir()
    (root / "assignments").mkdir()
    request = {
        "schema": "swing.catalyst_lineage_request.v2",
        "generation": generation,
        "scope_policy": "observed_articles_not_coverage_admission",
        "coverage_blindspot_security_ids": [],
        "source_coverage_admitted": False,
        "excluded_security_ids": [],
        "decisions_sha256": "1" * 64,
        "production_ready": production_ready,
    }
    request_sha256 = _json_sha256(request)
    _write_json(root / "_request.json", {**request, "request_sha256": request_sha256})
    coverage = _coverage(
        unknown_source,
        poison_unknown_zero,
        completion_offset=coverage_completion_offset,
    )
    coverage_manifest = write_canonical_artifact(
        coverage,
        root / "source_coverage.parquet",
        artifact_type="catalyst_source_coverage",
        audit=_passing_audit("coverage", len(coverage)),
        inputs={"catalyst_lineage_request_sha256": request_sha256},
        production_ready=production_ready,
    )
    event_records = [
        {
            "event_id": "event-direct",
            "source_event_id": _content_sha256("source-event-direct"),
            "source_security_id": "security-1",
            "security_id": "security-1",
            "ticker": "ABC",
            "source_family": "alpaca",
            "feature_available_at_utc": DECISION_TIME - pd.Timedelta(hours=2),
            "sentiment_input_sha256": _content_sha256("event-direct-content"),
            "relation_channel": "direct_issuer",
            "training_eligible": True,
            "sentiment_model": scorer_model,
            "sentiment_model_revision": scorer_revision,
        },
        {
            "event_id": "event-sector",
            "source_event_id": _content_sha256("source-event-sector"),
            "source_security_id": "security-1",
            "security_id": "security-1",
            "ticker": "ABC",
            "source_family": "finviz",
            "feature_available_at_utc": DECISION_TIME - pd.Timedelta(hours=2),
            "sentiment_input_sha256": _content_sha256("event-sector-content"),
            "relation_channel": "sector_context",
            "training_eligible": False,
            "sentiment_model": scorer_model,
            "sentiment_model_revision": scorer_revision,
        },
    ]
    if optional_direct_source is not None:
        event_records.append(
            {
                "event_id": "event-optional-direct",
                "source_event_id": _content_sha256("source-event-optional-direct"),
                "source_security_id": "security-1",
                "security_id": "security-1",
                "ticker": "ABC",
                "source_family": optional_direct_source,
                "feature_available_at_utc": DECISION_TIME - pd.Timedelta(hours=2),
                "sentiment_input_sha256": _content_sha256("event-optional-direct-content"),
                "relation_channel": "direct_issuer",
                "training_eligible": True,
                "sentiment_model": scorer_model,
                "sentiment_model_revision": scorer_revision,
            }
        )
    events = pd.DataFrame.from_records(event_records)
    event_path = root / "events" / "chunk.parquet"
    event_manifest = write_canonical_artifact(
        events,
        event_path,
        artifact_type="catalyst_events",
        audit=_passing_audit("events", len(events)),
        inputs={"catalyst_lineage_request_sha256": request_sha256},
        production_ready=production_ready,
    )
    assignments = _assignments(optional_direct_source=optional_direct_source)
    assignment_path = root / "assignments" / "chunk.parquet"
    assignment_material_sha256 = reconciliation_sha256(assignments)
    assignment_manifest = write_canonical_artifact(
        assignments,
        assignment_path,
        artifact_type="catalyst_event_assignments",
        audit=_passing_audit("assignments", len(assignments)),
        inputs={
            "catalyst_lineage_request_sha256": request_sha256,
            "catalyst_events_sha256": str(event_manifest["artifact_sha256"]),
            "assignment_sha256": assignment_material_sha256,
        },
        production_ready=production_ready,
    )
    inventory = {
        "schema": "swing.catalyst_feature_inventory.v1",
        "request_sha256": request_sha256,
        "training_eligible_channels": ["direct_issuer"],
        "production_ready": production_ready,
    }
    _write_json(root / "feature_inventory.json", inventory)
    artifact_record = {
        "chunk_id": "chunk",
        "source_event_sha256": "2" * 64,
        "relation_sha256": "3" * 64,
        "sentiment_sha256": "4" * 64,
        "event_path": str(event_path.resolve()),
        "event_sha256": event_manifest["artifact_sha256"],
        "event_rows": len(events),
        "training_eligible_rows": int(events["training_eligible"].sum()),
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": assignment_manifest["artifact_sha256"],
        "assignment_material_sha256": assignment_material_sha256,
        "assignment_rows": len(assignments),
    }
    lineage_material = {
        "request": request,
        "coverage_sha256": coverage_manifest["artifact_sha256"],
        "artifacts": [artifact_record],
        "feature_inventory": inventory,
    }
    manifest = {
        "schema": "swing.catalyst_lineage_manifest.v2",
        "scope_policy": "observed_articles_not_coverage_admission",
        "coverage_blindspot_security_ids": [],
        "source_coverage_admitted": False,
        "excluded_security_ids": [],
        "request_sha256": request_sha256,
        "status": "complete",
        "requested_chunks": 1,
        "observed_chunks": 1,
        "failed_chunks": {},
        "relation_rows": len(events),
        "training_eligible_rows": int(events["training_eligible"].sum()),
        "assignment_rows": len(assignments),
        "assignment_status_counts": {"assigned": len(assignments)},
        "coverage": {
            "path": str((root / "source_coverage.parquet").resolve()),
            "sha256": coverage_manifest["artifact_sha256"],
            "rows": len(coverage),
        },
        "feature_inventory": {
            "path": str((root / "feature_inventory.json").resolve()),
            "sha256": file_sha256(root / "feature_inventory.json"),
        },
        "artifacts": [artifact_record],
        "lineage_sha256": _json_sha256(lineage_material),
        "production_ready": production_ready,
    }
    _write_json(root / "_manifest.json", manifest)
    return root


def _assignments(*, optional_direct_source: str | None = None) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    sources = [
        ("event-direct", "alpaca"),
        ("event-sector", "finviz"),
    ]
    if optional_direct_source is not None:
        sources.append(("event-optional-direct", optional_direct_source))
    for event_id, source_family in sources:
        for window, seconds in (("1d", 86_400), ("3d", 259_200)):
            records.append(
                {
                    "assignment_id": hashlib.sha256(f"{event_id}|decision-1|{window}".encode()).hexdigest(),
                    "event_id": event_id,
                    "ticker": "ABC",
                    "security_id": "security-1",
                    "source_family": source_family,
                    "feature_available_at_utc": DECISION_TIME - pd.Timedelta(hours=2),
                    "decision_id": "decision-1",
                    "decision_time_utc": DECISION_TIME,
                    "window_name": window,
                    "window_seconds": seconds,
                    "status": "assigned",
                    "sentiment_numeric": 0.5,
                    "relevance": 0.8,
                    "schema_version": "event_assignment.v3",
                }
            )
    return pd.DataFrame.from_records(records, columns=ASSIGNMENT_COLUMNS)


def _deduplication_row(
    *,
    event_id: str,
    source_event_id: str,
    decision_id: str,
    security_id: str,
    ticker: str,
    source_family: str,
    available_at: pd.Timestamp,
    content: str,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "ticker": ticker,
        "event_ticker": ticker,
        "security_id": security_id,
        "source_family": source_family,
        "feature_available_at_utc": available_at,
        "decision_id": decision_id,
        "decision_time_utc": DECISION_TIME,
        "window_name": "1d",
        "window_seconds": 86_400,
        "status": "assigned",
        "sentiment_numeric": 0.5,
        "relevance": 0.8,
        "source_event_id": source_event_id,
        "source_security_id": security_id,
        "content_identity_sha256": _content_sha256(content),
        "source_lineages": json.dumps(["lineage"]),
    }


def _content_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _coverage(
    unknown_source: str | None,
    poison_unknown_zero: bool,
    *,
    completion_offset: pd.Timedelta,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in RANKING_SOURCE_FAMILIES:
        unknown = family == unknown_source
        state = "failed_or_unobserved" if unknown else ("observed_complete" if family == "alpaca" else "observed_empty")
        for sequence, decision_offset in enumerate(
            (pd.Timedelta(0), pd.Timedelta(hours=1)),
            start=1,
        ):
            requested_end = DECISION_TIME + decision_offset
            rows.append(
                {
                    "collection_id": f"collection-{family}-{sequence}",
                    "chunk_id": f"chunk-{family}-{sequence}",
                    "security_id": "security-1",
                    "ticker": "ABC",
                    "source_family": family,
                    "requested_start_utc": requested_end - pd.Timedelta(days=4),
                    "requested_end_utc": requested_end,
                    "started_at_utc": requested_end - pd.Timedelta(minutes=1),
                    "completed_at_utc": requested_end + completion_offset,
                    "status": "failed" if unknown else ("observed" if family == "alpaca" else "observed_empty"),
                    "row_count": 1 if family == "alpaca" else 0,
                    "coverage_state": state,
                    "missingness_known": False if unknown else True,
                    "zero_event_semantics": (
                        "known_zero_events"
                        if unknown and poison_unknown_zero
                        else {
                            "observed_complete": "observed_history",
                            "observed_empty": "known_zero_events",
                            "failed_or_unobserved": "unknown_failed",
                        }[state]
                    ),
                    "training_eligible": False if unknown else True,
                    "schema_version": "swing.catalyst_source_coverage.v1",
                }
            )
    return pd.DataFrame(rows)


def _passing_audit(name: str, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="synthetic verified artifact",
            ),
        )
    )


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
