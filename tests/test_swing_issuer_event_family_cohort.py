from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events import family_evidence
from market_predictor.catalysts.issuer_events.attribution_history import (
    attribute_alpaca_news_history,
    load_event_attribution_history,
)
from market_predictor.catalysts.issuer_events.family_evidence import load_issuer_family_evidence
from market_predictor.catalysts.issuer_events.news_history_contracts import (
    NEWS_HISTORY_MANIFEST_SCHEMA,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.issuer_family_combined_envelope import (
    AUTHORITY_SCHEMA,
    COHORT_AUDIT_ARTIFACT_TYPE,
    FAMILY_ASSIGNMENTS_ARTIFACT_TYPE,
    FAMILY_COVERAGE_ARTIFACT_TYPE,
    FAMILY_EVENTS_ARTIFACT_TYPE,
    MANIFEST_SCHEMA,
    NEUTRAL_PROJECTION_SCHEMA,
    UNCLASSIFIED_EVENTS_ARTIFACT_TYPE,
)
from market_predictor.swing.datasets import issuer_event_family_cohort as authority_module
from market_predictor.swing.datasets.issuer_event_family_cohort import (
    SwingIssuerFamilyCohort,
    load_swing_issuer_family_cohort,
    publish_swing_issuer_family_cohort,
)

_POLICY_PATH = Path(__file__).parents[1] / "configs" / "swing_event_family_policy.toml"
_EVENT_AVAILABLE = pd.Timestamp("2025-01-02T14:00:00Z")
_DECISION_TIME = pd.Timestamp("2025-01-03T14:00:00Z")


@dataclass(frozen=True)
class _Inputs:
    collection_dir: Path
    collection_audit_path: Path
    attribution_dir: Path
    decisions_path: Path
    output_directory: Path


def test_retained_issuer_family_envelope_identities_are_frozen() -> None:
    assert AUTHORITY_SCHEMA == "edge_rebuild.issuer_event_family_authority.v2"
    assert MANIFEST_SCHEMA == "edge_rebuild.issuer_event_family_manifest.v2"
    assert FAMILY_EVENTS_ARTIFACT_TYPE == "issuer_event_family_events"
    assert FAMILY_ASSIGNMENTS_ARTIFACT_TYPE == "issuer_event_family_assignments"
    assert FAMILY_COVERAGE_ARTIFACT_TYPE == "issuer_event_family_coverage"
    assert COHORT_AUDIT_ARTIFACT_TYPE == "issuer_event_family_cohort_audit"
    assert UNCLASSIFIED_EVENTS_ARTIFACT_TYPE == "issuer_event_family_unclassified_events"
    assert NEUTRAL_PROJECTION_SCHEMA == "market_predictor.issuer_family_neutral_projection.v1"


def test_publishes_immutable_multilabel_authority(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)

    authority = _publish(inputs)

    assert authority.manifest["state"] == "complete"
    assert authority.manifest["production_ready"] is False
    assert set(authority.events["event_family"]) == {"earnings", "guidance"}
    assert authority.events["research_eligible"].astype(bool).all()
    assert not authority.events["production_eligible"].astype(bool).any()
    assert authority.manifest["family_status"] == {
        "earnings": "admitted",
        "guidance": "admitted",
        "sec_material_event": "blocked_missing_source",
        "analyst_revision": "absent",
        "offering": "absent",
        "merger_acquisition": "absent",
        "regulatory_decision": "absent",
        "product_event": "absent",
    }
    assigned = authority.assignments.loc[authority.assignments["status"].eq("assigned")]
    assert set(assigned["event_family"]) == {"earnings", "guidance"}
    assert set(assigned["window_name"]) == {"1d", "3d"}

    expected_identity = file_sha256(inputs.output_directory / "_authority.json")
    loaded = load_swing_issuer_family_cohort(
        inputs.output_directory,
        expected_authority_sha256=expected_identity,
    )
    assert_frame_equal(loaded.events, authority.events)
    with pytest.raises(DataReadinessError, match="immutable"):
        _publish(inputs)


def test_neutral_projection_exposes_only_events_and_coverage(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    cohort = _publish(inputs)

    evidence = load_issuer_family_evidence(
        inputs.output_directory,
        expected_authority_sha256=file_sha256(inputs.output_directory / "_authority.json"),
    )

    assert_frame_equal(evidence.events, cohort.events)
    assert_frame_equal(evidence.coverage, cohort.coverage)
    assert not hasattr(evidence, "assignments")
    assert not hasattr(evidence, "cohort_audit")
    assert evidence.combined_envelope_sha256 == file_sha256(
        inputs.output_directory / "_authority.json"
    )
    assert len(evidence.full_inventory_sha256) == 64
    assert len(evidence.neutral_projection_sha256) == 64


def test_neutral_projection_identity_excludes_swing_decisions(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    _publish(inputs)
    first = load_issuer_family_evidence(inputs.output_directory)

    decisions, decision_manifest = load_canonical_artifact(
        inputs.decisions_path,
        expected_type="decisions",
        allow_research=True,
    )
    decisions.loc[:, "sector"] = "Healthcare"
    rewritten = write_canonical_artifact(
        decisions,
        inputs.decisions_path,
        artifact_type="decisions",
        audit=_audit(len(decisions)),
        inputs={
            str(key): str(value)
            for key, value in dict(decision_manifest.get("inputs", {})).items()
        },
        production_ready=False,
    )
    assert rewritten["artifact_sha256"] != decision_manifest["artifact_sha256"]
    inputs.decisions_path.with_name(
        f"{inputs.decisions_path.name}.lock"
    ).unlink(missing_ok=True)
    second_inputs = _Inputs(
        collection_dir=inputs.collection_dir,
        collection_audit_path=inputs.collection_audit_path,
        attribution_dir=inputs.attribution_dir,
        decisions_path=inputs.decisions_path,
        output_directory=tmp_path / "second-authority",
    )
    _publish(second_inputs)
    second = load_issuer_family_evidence(second_inputs.output_directory)

    assert_frame_equal(first.events, second.events)
    assert_frame_equal(first.coverage, second.coverage)
    assert first.neutral_projection_sha256 == second.neutral_projection_sha256
    assert first.full_inventory_sha256 != second.full_inventory_sha256


def test_unclassified_event_is_retained_but_ineligible(tmp_path: Path) -> None:
    events = _events(
        _event(
            event_id="unclassified",
            title="Acme schedules its annual shareholder meeting",
        )
    )

    authority = _publish(_write_inputs(tmp_path, events=events))

    assert authority.events.empty
    assert authority.manifest["unclassified_event_rows"] == 1
    assert len(authority.unclassified_artifact_records) == 1
    record = authority.unclassified_artifact_records[0]
    unclassified, _ = load_canonical_artifact(
        authority.directory / str(record["path"]),
        expected_type="issuer_event_family_unclassified_events",
        allow_research=True,
    )
    row = unclassified.iloc[0]
    assert row["classification_state"] == "unclassified"
    assert row["event_family"] == ""
    assert not bool(row["research_eligible"])
    assert row["exclusion_reason"] == "unclassified_event_family"
    assert authority.assignments.empty


def test_indirect_relation_is_audited_but_not_materialized(tmp_path: Path) -> None:
    events = _events(
        _event(
            security_id="security:source",
            ticker="SRC",
            title=(
                "Storage supplier reports Q2 earnings and raises full-year guidance"
            ),
        )
    )

    authority = _publish(_write_inputs(tmp_path, events=events))

    assert authority.events.empty
    assert authority.assignments.empty
    assert authority.manifest["excluded_relation_channel_counts"] == {
        "business_exposure": 1,
        "sector_context": 0,
    }


def test_future_availability_poison_is_rejected(tmp_path: Path) -> None:
    events = _events(
        _event(
            event_id="future-poison",
            title="Acme reports Q2 earnings",
            published_at="2025-01-02T15:00:00Z",
            available_at="2025-01-02T14:00:00Z",
        )
    )
    relations = _relations(
        _relation(event_id="future-poison", relation_id="future-poison-relation")
    )

    policy = authority_module.load_swing_issuer_family_cohort_policy(_POLICY_PATH)
    with pytest.raises(
        DataReadinessError,
        match="published|availability|future|backdated",
    ):
        authority_module._build_family_events(
            events,
            relations,
            policy=policy,
            coverage_known=True,
        )


def test_source_issuer_identity_poison_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="identity conflicts"):
        _publish(_write_inputs(tmp_path, poison_source_identity=True))


def test_known_zero_and_unknown_coverage_remain_distinct(tmp_path: Path) -> None:
    coverage = _coverage(
        _coverage_row(
            chunk_id="chunk-1",
            security_id="security:acme",
            ticker="ACME",
            status="observed",
        ),
        _coverage_row(
            chunk_id="chunk-empty",
            security_id="security:empty",
            ticker="EMPTY",
            status="observed_empty",
        ),
        _coverage_row(
            chunk_id="chunk-failed",
            security_id="security:failed",
            ticker="FAILED",
            status="failed",
        ),
    )

    authority = _publish(_write_inputs(tmp_path, coverage=coverage))
    by_ticker = authority.coverage.groupby("ticker", sort=True).first()

    assert by_ticker.loc["EMPTY", "coverage_state"] == "observed_empty"
    assert bool(by_ticker.loc["EMPTY", "missingness_known"])
    assert by_ticker.loc["EMPTY", "zero_event_semantics"] == "known_zero_events"
    assert by_ticker.loc["FAILED", "coverage_state"] == "failed_or_unobserved"
    assert not bool(by_ticker.loc["FAILED", "missingness_known"])
    assert by_ticker.loc["FAILED", "zero_event_semantics"] == "unknown_failed"


def test_known_coverage_uses_consistent_timestamp_units() -> None:
    coverage = pd.DataFrame(
        {
            "security_id": ["security:acme"],
            "source_family": ["alpaca"],
            "event_family": ["earnings"],
            "requested_start_utc": pd.Series(
                ["2025-01-01T00:00:00Z"], dtype="datetime64[us, UTC]"
            ),
            "requested_end_utc": pd.Series(
                ["2025-01-10T00:00:00Z"], dtype="datetime64[us, UTC]"
            ),
            "missingness_known": [True],
        }
    )
    decisions = pd.DataFrame(
        {
            "security_id": ["security:acme", "security:acme"],
            "decision_id": ["under-warmed", "covered"],
            "decision_time_utc": pd.Series(
                ["2025-01-02T00:00:00Z", "2025-01-04T00:00:00Z"],
                dtype="datetime64[us, UTC]",
            ),
        }
    )

    known = authority_module._known_coverage_decision_ids(
        coverage,
        decisions,
        family="earnings",
        max_window=pd.Timedelta(days=3),
        source_family="alpaca",
    )

    assert known == {"covered"}


def test_causal_issuer_company_is_attached_from_identity_authority() -> None:
    events = _events(_event()).drop(
        columns=["issuer_company", "issuer_company_available_at_utc"]
    )
    intervals = authority_module._identity_intervals_by_security(
        _identities(events)
    )

    attached = authority_module._attach_causal_issuer_companies(events, intervals)

    assert attached["issuer_company"].tolist() == ["Acme"]
    assert attached["issuer_company_available_at_utc"].tolist() == [
        pd.Timestamp("2020-01-01T00:00:00Z")
    ]


def test_alpaca_coverage_does_not_cover_sec_material_events() -> None:
    policy = authority_module.load_swing_issuer_family_cohort_policy(_POLICY_PATH)
    coverage = authority_module._build_family_coverage(
        _coverage(
            _coverage_row(
                chunk_id="chunk-1",
                security_id="security:acme",
                ticker="ACME",
                status="observed",
            )
        ),
        relation_chunk_ids={"chunk-1"},
        blind_security_ids=set(),
        policy=policy,
        collection_completed_at=pd.Timestamp("2025-01-04T00:00:00Z"),
    )

    assert "sec_material_event" not in set(coverage["event_family"])


def test_replicated_family_coverage_must_remain_identical() -> None:
    policy = authority_module.load_swing_issuer_family_cohort_policy(_POLICY_PATH)
    coverage = authority_module._build_family_coverage(
        _coverage(
            _coverage_row(
                chunk_id="chunk-1",
                security_id="security:acme",
                ticker="ACME",
                status="observed",
            )
        ),
        relation_chunk_ids={"chunk-1"},
        blind_security_ids=set(),
        policy=policy,
        collection_completed_at=pd.Timestamp("2025-01-04T00:00:00Z"),
    )
    target = coverage.index[coverage["event_family"].eq("guidance")][0]
    coverage.loc[target, "requested_end_utc"] = pd.Timestamp(
        "2025-01-03T00:00:00Z"
    )

    with pytest.raises(DataReadinessError, match="differs across replicated"):
        family_evidence.validate_replicated_family_coverage(coverage)


@pytest.mark.parametrize("target", ["events", "coverage"])
def test_source_family_outside_policy_is_rejected(target: str) -> None:
    policy = authority_module.load_swing_issuer_family_cohort_policy(_POLICY_PATH)
    if target == "events":
        events = _events(_event(source_family="unsupported"))
        with pytest.raises(DataReadinessError, match="outside policy"):
            authority_module._build_family_events(
                events,
                _relations(_relation(event_id="event-1", relation_id="relation-1")),
                policy=policy,
                coverage_known=True,
            )
    else:
        coverage_row = _coverage_row(
            chunk_id="chunk-1",
            security_id="security:acme",
            ticker="ACME",
            status="observed",
        )
        coverage_row["source_family"] = "unsupported"
        with pytest.raises(DataReadinessError, match="outside policy"):
            authority_module._build_family_coverage(
                _coverage(coverage_row),
                relation_chunk_ids={"chunk-1"},
                blind_security_ids=set(),
                policy=policy,
                collection_completed_at=pd.Timestamp("2025-01-04T00:00:00Z"),
            )


def test_cohort_assignments_require_corresponding_source_coverage() -> None:
    policy = authority_module.load_swing_issuer_family_cohort_policy(_POLICY_PATH)
    coverage = authority_module._build_family_coverage(
        _coverage(
            _coverage_row(
                chunk_id="chunk-1",
                security_id="security:acme",
                ticker="ACME",
                status="observed",
            )
        ),
        relation_chunk_ids={"chunk-1"},
        blind_security_ids=set(),
        policy=policy,
        collection_completed_at=pd.Timestamp("2025-01-04T00:00:00Z"),
    )
    events = pd.DataFrame.from_records(
        [
            {
                "family_event_id": "alpaca-event",
                "event_family": "earnings",
                "source_family": "alpaca",
                "security_id": "security:acme",
                "feature_available_at_utc": pd.Timestamp("2025-01-01T12:00:00Z"),
                "research_eligible": True,
            },
            {
                "family_event_id": "sec-event",
                "event_family": "earnings",
                "source_family": "sec",
                "security_id": "security:acme",
                "feature_available_at_utc": pd.Timestamp("2025-01-01T13:00:00Z"),
                "research_eligible": True,
            },
        ]
    )
    decisions = pd.DataFrame.from_records(
        [
            {
                "decision_id": "alpaca-covered",
                "security_id": "security:acme",
                "decision_time_utc": pd.Timestamp("2025-01-02T14:00:00Z"),
                "sector": "Technology",
            },
            {
                "decision_id": "sec-not-covered",
                "security_id": "security:acme",
                "decision_time_utc": pd.Timestamp("2025-01-03T14:00:00Z"),
                "sector": "Technology",
            },
        ]
    )
    assignments = pd.DataFrame.from_records(
        [
            {
                "status": "assigned",
                "event_family": "earnings",
                "original_source_family": "alpaca",
                "decision_id": "alpaca-covered",
                "event_id": "alpaca-event",
            },
            {
                "status": "assigned",
                "event_family": "earnings",
                "original_source_family": "sec",
                "decision_id": "sec-not-covered",
                "event_id": "sec-event",
            },
        ]
    )

    audit = authority_module._build_cohort_audit(
        events,
        assignments,
        coverage,
        decisions,
        policy=policy,
    )

    overall = audit.loc[
        audit["event_family"].eq("earnings")
        & audit["dimension_type"].eq("overall")
    ].iloc[0]
    assert overall["known_coverage_decision_count"] == 2
    assert overall["assigned_decision_count"] == 1
    assert overall["abstention_count"] == 1
    assert overall["abstention_rate"] == pytest.approx(0.5)
    for dimension in ("calendar_month", "sector"):
        row = audit.loc[
            audit["event_family"].eq("earnings")
            & audit["dimension_type"].eq(dimension)
        ].iloc[0]
        assert row["known_coverage_decision_count"] == 2
        assert row["assigned_decision_count"] == 1
    source_rows = audit.loc[
        audit["event_family"].eq("earnings")
        & audit["dimension_type"].eq("source_family")
    ].set_index("dimension_value")
    assert source_rows.loc["alpaca", "assigned_decision_count"] == 1
    assert source_rows.loc["sec", "assigned_decision_count"] == 0
    assert source_rows.loc["sec", "known_coverage_decision_count"] == 0


def test_cohort_record_rejects_more_assignments_than_known_coverage() -> None:
    with pytest.raises(DataReadinessError, match="exceed known coverage"):
        authority_module._cohort_record(
            "earnings",
            "overall",
            "all",
            pd.DataFrame(columns=["security_id", "feature_available_at_utc"]),
            pd.DataFrame({"decision_id": ["not-covered"]}),
            0,
        )


@pytest.mark.parametrize(
    ("abstention_count", "abstention_rate"),
    [(0, 0.5), (1, 0.25)],
)
def test_cohort_audit_rejects_inconsistent_abstention(
    abstention_count: int,
    abstention_rate: float,
) -> None:
    frame = pd.DataFrame.from_records(
        [
            {
                "event_family": "earnings",
                "dimension_type": "overall",
                "dimension_value": "all",
                "event_count": 1,
                "security_count": 1,
                "assigned_decision_count": 1,
                "known_coverage_decision_count": 2,
                "abstention_count": abstention_count,
                "abstention_rate": abstention_rate,
                "first_event_available_at_utc": pd.Timestamp(
                    "2025-01-01T12:00:00Z"
                ),
                "last_event_available_at_utc": pd.Timestamp(
                    "2025-01-01T12:00:00Z"
                ),
                "schema_version": authority_module.AUTHORITY_SCHEMA,
            }
        ]
    )

    with pytest.raises(DataReadinessError):
        authority_module._cohort_audit(frame).raise_for_failure()


@pytest.mark.parametrize("target", ["child", "authority"])
def test_tampered_child_or_authority_is_rejected(
    tmp_path: Path,
    target: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    _publish(inputs)

    if target == "child":
        child = inputs.output_directory / "family_events.parquet"
        child.write_bytes(child.read_bytes() + b"tampered")
    else:
        authority_path = inputs.output_directory / "_authority.json"
        payload = _read_json(authority_path)
        payload["request_sha256"] = "0" * 64
        _write_json(authority_path, payload)

    with pytest.raises(DataReadinessError):
        load_swing_issuer_family_cohort(inputs.output_directory)
    with pytest.raises(DataReadinessError):
        load_issuer_family_evidence(inputs.output_directory)


@pytest.mark.parametrize(
    ("target", "value"),
    [
        ("authority_artifact", "wrong.json"),
        ("request_schema", "wrong.request.v1"),
        ("classifier_policy", "0" * 64),
        ("child_artifact_path", "C:\\outside\\family_events.parquet"),
        ("child_schema", "wrong.canonical.manifest.v1"),
        ("child_canonical_version", "wrong.canonical.v1"),
    ],
)
def test_malformed_envelope_contract_is_rejected(
    tmp_path: Path,
    target: str,
    value: str,
) -> None:
    inputs = _write_inputs(tmp_path)
    _publish(inputs)
    manifest_path = inputs.output_directory / "_manifest.json"
    authority_path = inputs.output_directory / "_authority.json"

    if target == "authority_artifact":
        authority = _read_json(authority_path)
        authority["artifact"] = value
        _write_json(authority_path, authority)
    elif target in {"request_schema", "classifier_policy"}:
        manifest = _read_json(manifest_path)
        request = manifest["request"]
        assert isinstance(request, dict)
        request[
            "schema" if target == "request_schema" else "classifier_policy_sha256"
        ] = value
        _write_json(manifest_path, manifest)
        authority = _read_json(authority_path)
        authority["artifact_sha256"] = file_sha256(manifest_path)
        _write_json(authority_path, authority)
    else:
        child_path = manifest_path_for(
            inputs.output_directory / "family_events.parquet"
        )
        child = _read_json(child_path)
        child[
            {
                "child_artifact_path": "artifact_path",
                "child_schema": "schema",
                "child_canonical_version": "canonical_schema_version",
            }[target]
        ] = value
        _write_json(child_path, child)

    with pytest.raises(DataReadinessError):
        load_issuer_family_evidence(inputs.output_directory)
    with pytest.raises(DataReadinessError):
        load_swing_issuer_family_cohort(inputs.output_directory)


def test_symlinked_envelope_artifact_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path)
    _publish(inputs)
    original = Path.is_symlink

    def _is_symlink(path: Path) -> bool:
        return path.name == "family_events.parquet" or original(path)

    monkeypatch.setattr(Path, "is_symlink", _is_symlink)

    with pytest.raises(DataReadinessError, match="symlink"):
        load_issuer_family_evidence(inputs.output_directory)


def test_manifest_family_status_is_recomputed_by_loader(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    _publish(inputs)
    manifest_path = inputs.output_directory / "_manifest.json"
    authority_path = inputs.output_directory / "_authority.json"
    manifest = _read_json(manifest_path)
    family_status = manifest["family_status"]
    assert isinstance(family_status, dict)
    family_status["earnings"] = "absent"
    _write_json(manifest_path, manifest)
    authority = _read_json(authority_path)
    authority["artifact_sha256"] = file_sha256(manifest_path)
    _write_json(authority_path, authority)

    with pytest.raises(DataReadinessError, match="family status"):
        load_swing_issuer_family_cohort(inputs.output_directory)


def test_coherently_resigned_cohort_tamper_fails_semantic_replay(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    _publish(inputs)
    original_projection = load_issuer_family_evidence(
        inputs.output_directory
    ).neutral_projection_sha256
    cohort_path = inputs.output_directory / "cohort_audit.parquet"
    cohort, child = load_canonical_artifact(
        cohort_path,
        expected_type="issuer_event_family_cohort_audit",
        allow_research=True,
    )
    target = cohort.index[
        cohort["dimension_type"].eq("overall")
        & cohort["event_family"].eq("earnings")
    ][0]
    assigned = int(cohort.loc[target, "assigned_decision_count"])
    known = int(cohort.loc[target, "known_coverage_decision_count"])
    assert assigned > 0 and known >= assigned
    cohort.loc[target, "assigned_decision_count"] = assigned - 1
    cohort.loc[target, "abstention_count"] = known - assigned + 1
    cohort.loc[target, "abstention_rate"] = (known - assigned + 1) / known
    child_inputs = child["inputs"]
    assert isinstance(child_inputs, dict)
    rewritten = write_canonical_artifact(
        cohort,
        cohort_path,
        artifact_type="issuer_event_family_cohort_audit",
        audit=authority_module._cohort_audit(cohort),
        inputs={str(key): value for key, value in child_inputs.items()},
        production_ready=False,
    )
    cohort_path.with_name(f"{cohort_path.name}.lock").unlink(missing_ok=True)

    manifest_path = inputs.output_directory / "_manifest.json"
    authority_path = inputs.output_directory / "_authority.json"
    manifest = _read_json(manifest_path)
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    cohort_record = artifacts["cohort_audit"]
    assert isinstance(cohort_record, dict)
    cohort_record["sha256"] = rewritten["artifact_sha256"]
    _write_json(manifest_path, manifest)
    authority = _read_json(authority_path)
    authority["artifact_sha256"] = file_sha256(manifest_path)
    _write_json(authority_path, authority)

    neutral = load_issuer_family_evidence(inputs.output_directory)
    assert neutral.neutral_projection_sha256 == original_projection

    with pytest.raises(DataReadinessError, match="semantic replay"):
        load_swing_issuer_family_cohort(inputs.output_directory)


def test_publication_is_deterministic_under_input_order_shuffle(tmp_path: Path) -> None:
    event_rows = (
        _event(event_id="event-z", title="FDA approves Acme therapy"),
        _event(event_id="event-a", title="Acme reports Q2 earnings"),
    )
    first = _publish(
        _write_inputs(
            tmp_path / "first",
            events=_events(*event_rows),
        )
    )
    second = _publish(
        _write_inputs(
            tmp_path / "second",
            events=_events(*reversed(event_rows)),
        )
    )

    assert_frame_equal(first.events, second.events)
    assert_frame_equal(first.assignments, second.assignments)
    assert_frame_equal(first.coverage, second.coverage)
    assert_frame_equal(first.cohort_audit, second.cohort_audit)


def test_relative_output_directory_publishes_canonical_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _write_inputs(tmp_path / "inputs")
    monkeypatch.chdir(tmp_path)
    relative_inputs = _Inputs(
        collection_dir=inputs.collection_dir,
        collection_audit_path=inputs.collection_audit_path,
        attribution_dir=inputs.attribution_dir,
        decisions_path=inputs.decisions_path,
        output_directory=Path("relative-authority"),
    )

    published = _publish(relative_inputs)
    loaded = load_swing_issuer_family_cohort(tmp_path / "relative-authority")

    assert published.directory == (tmp_path / "relative-authority").resolve()
    assert_frame_equal(loaded.events, published.events)


def test_proxy_evidence_is_never_production_eligible(tmp_path: Path) -> None:
    authority = _publish(_write_inputs(tmp_path))

    assert authority.events["availability_policy"].eq(
        "provider_publication_proxy"
    ).all()
    assert not authority.events["production_eligible"].astype(bool).any()
    assert not authority.coverage["production_eligible"].astype(bool).any()
    assert authority.manifest["production_eligible_event_rows"] == 0
    assert authority.manifest["production_ready"] is False
    assert authority.authority["production_ready"] is False
    assert "proxy" in str(authority.manifest["promotion_blocker"])


def _publish(inputs: _Inputs) -> SwingIssuerFamilyCohort:
    return publish_swing_issuer_family_cohort(
        collection_dir=inputs.collection_dir,
        collection_audit_path=inputs.collection_audit_path,
        attribution_dir=inputs.attribution_dir,
        decisions_path=inputs.decisions_path,
        policy_path=_POLICY_PATH,
        output_directory=inputs.output_directory,
    )


def _write_inputs(
    root: Path,
    *,
    events: pd.DataFrame | None = None,
    coverage: pd.DataFrame | None = None,
    poison_source_identity: bool = False,
) -> _Inputs:
    root.mkdir(parents=True, exist_ok=True)
    collection_dir = root / "collection"
    attribution_dir = root / "attribution"
    events_path = collection_dir / "events" / "chunk-1.parquet"
    coverage_path = collection_dir / "source_collections.parquet"
    labels_path = root / "business_labels.parquet"
    identities_path = root / "security_identities.parquet"
    decisions_path = root / "decisions.parquet"
    request_sha256 = "c" * 64

    source_events = events if events is not None else _events(_event())
    source_security_id = str(source_events.iloc[0]["security_id"])
    source_ticker = str(source_events.iloc[0]["ticker"])
    source_coverage = (
        coverage
        if coverage is not None
        else _coverage(
            _coverage_row(
                chunk_id="chunk-1",
                security_id=source_security_id,
                ticker=source_ticker,
                status="observed",
            )
        )
    )

    event_manifest = write_canonical_artifact(
        source_events,
        events_path,
        artifact_type="events",
        audit=_audit(len(source_events)),
        inputs={"collection_request_sha256": request_sha256, "chunk_id": "chunk-1"},
        production_ready=False,
    )
    labels = _labels()
    write_canonical_artifact(
        labels,
        labels_path,
        artifact_type="security_business_labels",
        audit=_audit(len(labels)),
        production_ready=False,
    )
    identities = _identities(source_events)
    write_canonical_artifact(
        identities,
        identities_path,
        artifact_type="security_business_label_coverage",
        audit=_audit(len(identities)),
        production_ready=False,
    )
    coverage_manifest = write_canonical_artifact(
        source_coverage,
        coverage_path,
        artifact_type="source_collections",
        audit=_audit(len(source_coverage)),
        production_ready=False,
    )
    decisions = pd.DataFrame(
        {
            "security_id": ["security:acme"],
            "ticker": ["ACME"],
            "decision_time_utc": [_DECISION_TIME],
            "sector": ["Technology"],
        }
    )
    write_canonical_artifact(
        decisions,
        decisions_path,
        artifact_type="decisions",
        audit=_audit(len(decisions)),
        production_ready=False,
    )

    _write_json(
        collection_dir / "_manifest.json",
        {
            "schema": NEWS_HISTORY_MANIFEST_SCHEMA,
            "status": "complete",
            "production_ready": False,
            "availability_policy": "provider_publication_proxy",
            "request_sha256": request_sha256,
            "completed_at_utc": "2025-01-04T00:00:00Z",
            "requested_chunks": 1
            + int(source_coverage["status"].astype(str).eq("observed_empty").sum()),
            "observed_chunks": 1,
            "empty_chunks": int(
                source_coverage["status"].astype(str).eq("observed_empty").sum()
            ),
            "failed_chunks": {},
            "source_collections_path": str(coverage_path.resolve()),
            "source_collections_sha256": coverage_manifest["artifact_sha256"],
            "artifacts": [
                {
                    "chunk_id": "chunk-1",
                    "security_id": source_security_id,
                    "ticker": source_ticker,
                    "path": str(events_path.resolve()),
                    "manifest_path": str(manifest_path_for(events_path).resolve()),
                    "sha256": event_manifest["artifact_sha256"],
                    "rows": len(source_events),
                }
            ],
            "artifact_count": 1,
            "total_rows": len(source_events),
        },
    )
    collection_audit_path = root / "collection_audit.json"
    _write_json(
        collection_audit_path,
        {
            "passed": True,
            "request_sha256": request_sha256,
            "coverage_blindspot_security_ids": [],
        },
    )
    attribution_result = attribute_alpaca_news_history(
        collection_dir=collection_dir,
        collection_audit_path=collection_audit_path,
        business_labels_path=labels_path,
        security_identities_path=identities_path,
        out_dir=attribution_dir,
    )
    assert attribution_result["status"] == "complete"
    load_event_attribution_history(attribution_dir)
    if poison_source_identity:
        _poison_relation_source_identity(attribution_dir)
    return _Inputs(
        collection_dir=collection_dir,
        collection_audit_path=collection_audit_path,
        attribution_dir=attribution_dir,
        decisions_path=decisions_path,
        output_directory=root / "authority",
    )


def _event(
    *,
    event_id: str = "event-1",
    security_id: str = "security:acme",
    ticker: str = "ACME",
    title: str = "Acme reports Q2 earnings and raises full-year guidance",
    published_at: str = "2025-01-02T13:50:00Z",
    available_at: str = "2025-01-02T14:00:00Z",
    source_family: str = "alpaca",
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "security_id": security_id,
        "ticker": ticker,
        "source_family": source_family,
        "issuer_company": "Acme Corporation",
        "issuer_company_available_at_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
        "title": title,
        "summary": "",
        "text": "",
        "published_at_utc": pd.Timestamp(published_at),
        "feature_available_at_utc": pd.Timestamp(available_at),
        "availability_policy": "provider_publication_proxy",
    }


def _events(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _relation(
    *,
    event_id: str,
    relation_id: str,
    channel: str = "direct_issuer",
) -> dict[str, object]:
    return {
        "relation_id": relation_id,
        "event_id": event_id,
        "source_security_id": "security:acme",
        "source_ticker": "ACME",
        "target_security_id": "security:acme",
        "target_ticker": "ACME",
        "relation_channel": channel,
        "relation_score": 1.0 if channel == "direct_issuer" else 0.7,
        "feature_available_at_utc": _EVENT_AVAILABLE,
    }


def _relations(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _coverage_row(
    *,
    chunk_id: str,
    security_id: str,
    ticker: str,
    status: str,
) -> dict[str, object]:
    return {
        "collection_id": "collection-1",
        "chunk_id": chunk_id,
        "security_id": security_id,
        "ticker": ticker,
        "source_family": "alpaca",
        "requested_start_utc": pd.Timestamp("2024-12-30T00:00:00Z"),
        "requested_end_utc": pd.Timestamp("2025-01-04T00:00:00Z"),
        "completed_at_utc": pd.Timestamp("2025-01-04T00:00:00Z"),
        "status": status,
    }


def _coverage(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": ["security:acme"],
            "ticker": ["ACME"],
            "company": ["Acme"],
            "business_tag": ["offering.infrastructure.storage"],
            "label_type": ["offering"],
            "match_terms": ['["storage supplier"]'],
            "tag_rank": [1],
            "confidence": [0.9],
            "relation_use": ["exposure"],
            "effective_from_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
        }
    )


def _identities(events: pd.DataFrame) -> pd.DataFrame:
    source_pairs = {
        (str(row["security_id"]), str(row["ticker"]).upper())
        for row in events.to_dict(orient="records")
    }
    source_pairs.add(("security:acme", "ACME"))
    rows = [
        {
            "security_id": security_id,
            "ticker": ticker,
            "company": "Acme" if ticker == "ACME" else "Source Corp",
            "effective_from_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
            "effective_to_utc": pd.NaT,
            "available_at_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
        }
        for security_id, ticker in sorted(source_pairs)
    ]
    return pd.DataFrame.from_records(rows)


def _poison_relation_source_identity(attribution_dir: Path) -> None:
    root = _read_json(attribution_dir / "_manifest.json")
    artifacts = root["artifacts"]
    assert isinstance(artifacts, list) and len(artifacts) == 1
    record = artifacts[0]
    assert isinstance(record, dict)
    relation_path = Path(str(record["path"]))
    relations, relation_manifest = load_canonical_artifact(
        relation_path,
        expected_type="event_security_relations",
        allow_research=True,
    )
    relations.loc[:, "source_security_id"] = "security:wrong"
    inputs = relation_manifest["inputs"]
    assert isinstance(inputs, dict)
    rewritten = write_canonical_artifact(
        relations,
        relation_path,
        artifact_type="event_security_relations",
        audit=_audit(len(relations)),
        inputs={str(key): str(value) for key, value in inputs.items()},
        production_ready=False,
    )
    relation_path.with_name(f"{relation_path.name}.lock").unlink(missing_ok=True)
    record["sha256"] = rewritten["artifact_sha256"]
    _write_json(attribution_dir / "_manifest.json", root)
    _write_json(attribution_dir / "_status.json", root)
    load_event_attribution_history(attribution_dir)


def _audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="fixture",
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="minimal canonical fixture",
            ),
        )
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
