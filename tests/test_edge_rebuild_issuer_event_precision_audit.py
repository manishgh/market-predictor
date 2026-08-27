from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.news_history_contracts import (
    NEWS_HISTORY_MANIFEST_SCHEMA,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild import issuer_event_precision_audit as audit_module
from market_predictor.edge_rebuild.issuer_event_family_authority import (
    publish_issuer_event_family_authority,
)
from market_predictor.edge_rebuild.issuer_event_precision_audit import (
    IssuerEventPrecisionSample,
    finalize_issuer_event_precision_audit,
    load_issuer_event_precision_audit,
    load_issuer_event_precision_sample,
    publish_issuer_event_precision_sample,
    wilson_lower_bound,
)
from market_predictor.swing.event_attribution_history import (
    attribute_alpaca_news_history,
)
from market_predictor.swing.event_families import EVENT_FAMILIES

_ROOT = Path(__file__).parents[1]
_FAMILY_POLICY = _ROOT / "configs" / "swing_event_family_policy.toml"
_EVENT_AVAILABLE = pd.Timestamp("2025-01-02T14:00:00Z")


@dataclass(frozen=True)
class _PublishedInputs:
    authority_dir: Path
    precision_policy: Path


def test_sample_is_deterministic_causal_blind_and_hash_bound(tmp_path: Path) -> None:
    inputs = _publish_inputs(tmp_path / "source")

    first = publish_issuer_event_precision_sample(
        authority_directory=inputs.authority_dir,
        policy_path=inputs.precision_policy,
        output_directory=tmp_path / "sample-one",
    )
    second = publish_issuer_event_precision_sample(
        authority_directory=inputs.authority_dir,
        policy_path=inputs.precision_policy,
        output_directory=tmp_path / "sample-two",
    )

    assert_frame_equal(first.sample, second.sample)
    assert first.sample["title"].str.strip().ne("").all()
    assert first.sample["source"].eq("alpaca").all()
    assert first.sample["issuer_company"].eq("Acme").all()
    assert first.sample["identity_status"].eq("resolved").all()
    assert (
        pd.to_datetime(first.sample["identity_available_at_utc"], utc=True)
        <= pd.to_datetime(first.sample["feature_available_at_utc"], utc=True)
    ).all()
    forbidden = ("return", "price", "outcome", "probability", "target_return")
    assert not any(term in column.lower() for column in first.sample for term in forbidden)
    assert first.sample["cluster_selection_sha256"].str.len().eq(64).all()
    assert first.sample["row_selection_sha256"].str.len().eq(64).all()
    assert first.sample["sample_role"].eq("inferential").all()
    assert not first.sample["inference_cluster_id"].duplicated().any()
    assert first.manifest["training_eligible"] is False
    assert first.manifest["alerts_eligible"] is False

    one = pd.read_csv(first.directory / "reviewer_one_template.csv", dtype=str)
    two = pd.read_csv(first.directory / "reviewer_two_template.csv", dtype=str)
    assert one["reviewer_slot"].eq("1").all()
    assert two["reviewer_slot"].eq("2").all()
    assert one["reviewer_id"].isna().all()
    assert two["reviewer_id"].isna().all()
    assert "reviewer_two_id" not in one.columns
    assert "reviewer_one_id" not in two.columns


def test_all_yes_admits_populated_families_and_blocks_missing_families(
    tmp_path: Path,
) -> None:
    sample = _publish_sample(tmp_path)
    one, two, adjudication = _review_ledgers(tmp_path, sample.directory)

    result = finalize_issuer_event_precision_audit(
        sample_directory=sample.directory,
        reviewer_one_path=one,
        reviewer_two_path=two,
        adjudication_path=adjudication,
        output_directory=tmp_path / "final",
    )

    metrics = result.family_metrics.set_index("event_family")
    populated = set(sample.sample["proposed_event_family"])
    for family in populated:
        assert metrics.loc[family, "status"] == "admitted"
        assert metrics.loc[family, "joint_successes"] == metrics.loc[family, "inferential_cluster_count"]
    for family in set(EVENT_FAMILIES).difference(populated):
        assert metrics.loc[family, "status"] == "blocked"
        assert "missing_family_population" in metrics.loc[family, "blocker_reasons"]
    assert result.manifest["audit_status"] == "partial"
    assert result.manifest["production_ready"] is False
    one.unlink()
    two.unlink()
    adjudication.unlink()
    loaded = load_issuer_event_precision_audit(
        result.directory,
        expected_authority_sha256=file_sha256(result.directory / "_authority.json"),
    )
    assert_frame_equal(loaded.family_metrics, result.family_metrics)


def test_same_reviewer_is_rejected(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    one, two, adjudication = _review_ledgers(tmp_path, sample.directory, reviewer_two_id="reviewer-a")

    with pytest.raises(DataReadinessError, match="same reviewer"):
        finalize_issuer_event_precision_audit(
            sample_directory=sample.directory,
            reviewer_one_path=one,
            reviewer_two_path=two,
            adjudication_path=adjudication,
            output_directory=tmp_path / "final",
        )


def test_unresolved_disagreement_counts_as_failure(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    first_id = str(sample.sample.iloc[0]["sample_id"])
    one, two, adjudication = _review_ledgers(
        tmp_path,
        sample.directory,
        reviewer_two_overrides={first_id: {"family_correct": "uncertain"}},
    )

    result = finalize_issuer_event_precision_audit(
        sample_directory=sample.directory,
        reviewer_one_path=one,
        reviewer_two_path=two,
        adjudication_path=adjudication,
        output_directory=tmp_path / "final",
    )

    review = result.reviews.set_index("sample_id").loc[first_id]
    assert review["resolution_state"] == "unresolved_failure"
    assert not bool(review["family_correct"])
    assert not bool(review["issuer_target_correct"])
    assert not bool(review["event_announced_or_completed"])
    family = str(review["event_family"])
    metric = result.family_metrics.set_index("event_family").loc[family]
    assert metric["unresolved_count"] == 1
    assert metric["status"] == "blocked"


def test_wrong_issuer_is_a_hard_blocker(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    first_id = str(sample.sample.iloc[0]["sample_id"])
    override = {
        first_id: {
            "issuer_target_correct": "no",
            "action_subject_text": "Another issuer",
        }
    }
    one, two, adjudication = _review_ledgers(
        tmp_path,
        sample.directory,
        reviewer_one_overrides=override,
        reviewer_two_overrides=override,
    )

    result = finalize_issuer_event_precision_audit(
        sample_directory=sample.directory,
        reviewer_one_path=one,
        reviewer_two_path=two,
        adjudication_path=adjudication,
        output_directory=tmp_path / "final",
    )

    family = str(sample.sample.iloc[0]["proposed_event_family"])
    metric = result.family_metrics.set_index("event_family").loc[family]
    assert metric["wrong_issuer_count"] == 1
    assert "wrong_issuer_found" in metric["blocker_reasons"]
    assert metric["status"] == "blocked"


def test_adjudication_resolves_disagreement_with_third_reviewer(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    first_id = str(sample.sample.iloc[0]["sample_id"])
    one, two, adjudication = _review_ledgers(
        tmp_path,
        sample.directory,
        reviewer_two_overrides={first_id: {"family_correct": "no", "correct_family": "none"}},
        adjudication_overrides={
            first_id: {
                "adjudicator_id": "reviewer-c",
                "family_correct": "yes",
                "issuer_target_correct": "yes",
                "event_announced_or_completed": "yes",
            }
        },
    )

    result = finalize_issuer_event_precision_audit(
        sample_directory=sample.directory,
        reviewer_one_path=one,
        reviewer_two_path=two,
        adjudication_path=adjudication,
        output_directory=tmp_path / "final",
    )

    review = result.reviews.set_index("sample_id").loc[first_id]
    assert review["resolution_state"] == "adjudicated"
    assert bool(review["joint_correct"])


def test_source_authority_filter_requires_research_eligibility_and_allowed_source() -> None:
    frame = pd.DataFrame.from_records(
        [
            _authority_event_row("eligible", "earnings", "alpaca", True),
            _authority_event_row("not-research", "earnings", "alpaca", False),
            _authority_event_row("wrong-source", "earnings", "sec", True),
            _authority_event_row("sec", "sec_material_event", "sec", True),
        ]
    )

    eligible = audit_module._source_authorized_events(frame)

    assert eligible["family_event_id"].tolist() == ["eligible", "sec"]


def test_cluster_selection_hash_is_independent_of_cluster_row_count() -> None:
    source = _candidate_source("Shared issuer earnings headline")
    first = audit_module._candidate_index_row(
        _candidate_family("event-a", "security:a"),
        source,
        chunk_id="chunk-1",
        policy_sha256="a" * 64,
    )
    second = audit_module._candidate_index_row(
        _candidate_family("event-b", "security:b"),
        source,
        chunk_id="chunk-1",
        policy_sha256="a" * 64,
    )

    assert first[7] == second[7]
    assert first[8] == second[8]
    assert first[9] != second[9]


def test_paired_wrong_issuer_row_is_excluded_from_inference() -> None:
    policy = audit_module.load_issuer_event_precision_policy(_ROOT / "configs" / "issuer_event_precision_audit.toml")
    connection = sqlite3.connect(":memory:")
    audit_module._create_candidate_index(connection)
    source = _candidate_source("Shared issuer earnings headline")
    connection.executemany(
        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            audit_module._candidate_index_row(
                _candidate_family("event-a", "security:a"),
                source,
                chunk_id="chunk-1",
                policy_sha256="a" * 64,
            ),
            audit_module._candidate_index_row(
                _candidate_family("event-b", "security:b"),
                source,
                chunk_id="chunk-1",
                policy_sha256="a" * 64,
            ),
        ],
    )
    selected = audit_module._select_uniform_cluster_rows(
        connection,
        policy=policy,
        policy_sha256="a" * 64,
        authority_sha256="b" * 64,
    )
    connection.close()
    selected_frame = pd.DataFrame.from_records(selected)

    assert selected_frame["sample_role"].value_counts().to_dict() == {
        "inferential": 1,
        "paired_wrong_issuer_diagnostic": 1,
    }
    diagnostic = selected_frame.loc[selected_frame["sample_role"].eq("paired_wrong_issuer_diagnostic")].iloc[0]
    inferential = selected_frame.loc[selected_frame["sample_role"].eq("inferential")].iloc[0]
    assert diagnostic["paired_inferential_sample_id"] == inferential["sample_id"]


def test_reviewer_ids_are_stripped_before_distinct_validation(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    one, two, adjudication = _review_ledgers(
        tmp_path,
        sample.directory,
        reviewer_one_id="  reviewer-a  ",
        reviewer_two_id="  reviewer-b  ",
    )

    result = finalize_issuer_event_precision_audit(
        sample_directory=sample.directory,
        reviewer_one_path=one,
        reviewer_two_path=two,
        adjudication_path=adjudication,
        output_directory=tmp_path / "final",
    )

    assert result.reviews["reviewer_one_id"].eq("reviewer-a").all()
    assert result.reviews["reviewer_two_id"].eq("reviewer-b").all()


def test_reviewer_slot_rejects_multiple_normalized_identities(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    one, two, adjudication = _review_ledgers(tmp_path, sample.directory)
    ledger = pd.read_csv(one, dtype=str, keep_default_na=False)
    ledger.loc[ledger.index[-1], "reviewer_id"] = "reviewer-c"
    ledger.to_csv(one, index=False, lineterminator="\n")

    with pytest.raises(DataReadinessError, match="exactly one identity"):
        finalize_issuer_event_precision_audit(
            sample_directory=sample.directory,
            reviewer_one_path=one,
            reviewer_two_path=two,
            adjudication_path=adjudication,
            output_directory=tmp_path / "final",
        )


def test_correction_field_disagreement_requires_adjudication(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    first_id = str(sample.sample.iloc[0]["sample_id"])
    one, two, adjudication = _review_ledgers(
        tmp_path,
        sample.directory,
        reviewer_one_overrides={
            first_id: {
                "event_announced_or_completed": "no",
                "false_positive_reason": "reason one",
            }
        },
        reviewer_two_overrides={
            first_id: {
                "event_announced_or_completed": "no",
                "false_positive_reason": "reason two",
            }
        },
        adjudication_overrides={
            first_id: {
                "adjudicator_id": "reviewer-c",
                "family_correct": "yes",
                "issuer_target_correct": "yes",
                "event_announced_or_completed": "yes",
            }
        },
    )

    result = finalize_issuer_event_precision_audit(
        sample_directory=sample.directory,
        reviewer_one_path=one,
        reviewer_two_path=two,
        adjudication_path=adjudication,
        output_directory=tmp_path / "final",
    )

    review = result.reviews.set_index("sample_id").loc[first_id]
    assert bool(review["adjudication_required"])
    assert review["resolution_state"] == "adjudicated"


def test_malformed_review_fails_before_authority_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = _publish_sample(tmp_path)
    one, two, adjudication = _review_ledgers(tmp_path, sample.directory)
    reviewer_one = pd.read_csv(one, dtype=str, keep_default_na=False)
    reviewer_one.loc[0, "false_positive_reason"] = "invalid annotation"
    reviewer_one.to_csv(one, index=False, lineterminator="\n")

    replay_started = False

    def fail_if_replay_starts(_directory: Path) -> IssuerEventPrecisionSample:
        nonlocal replay_started
        replay_started = True
        raise AssertionError("authority replay started before ledger preflight")

    monkeypatch.setattr(
        audit_module,
        "load_issuer_event_precision_sample",
        fail_if_replay_starts,
    )
    with pytest.raises(DataReadinessError, match="false-positive reason"):
        finalize_issuer_event_precision_audit(
            sample_directory=sample.directory,
            reviewer_one_path=one,
            reviewer_two_path=two,
            adjudication_path=adjudication,
            output_directory=tmp_path / "final",
        )
    assert replay_started is False


def test_reviewer_agreement_and_kappa_are_per_field_diagnostics() -> None:
    reviews = pd.DataFrame(
        {
            "reviewer_one_family_correct": ["yes", "no"] * 5,
            "reviewer_two_family_correct": ["yes", "no"] * 5,
            "reviewer_one_issuer_target_correct": ["yes", "no"] * 5,
            "reviewer_two_issuer_target_correct": ["yes", "no"] * 5,
            "reviewer_one_event_announced_or_completed": ["yes", "no"] * 5,
            "reviewer_two_event_announced_or_completed": ["yes", "no"] * 5,
        }
    )

    metrics = audit_module._reviewer_agreement_by_field(reviews, minimum_decisions=10)

    assert set(metrics) == {
        "family_correct",
        "issuer_target_correct",
        "event_announced_or_completed",
    }
    for agreement, kappa, estimable in metrics.values():
        assert agreement == pytest.approx(1.0)
        assert kappa == pytest.approx(1.0)
        assert estimable


def test_each_estimable_reviewer_field_has_its_own_admission_gate() -> None:
    policy = replace(
        audit_module.load_issuer_event_precision_policy(_ROOT / "configs" / "issuer_event_precision_audit.toml"),
        minimum_reviewer_agreement=0.9,
        minimum_reviewer_kappa=0.8,
        minimum_kappa_decisions=10,
    )
    sample = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(10)],
            "sample_role": ["inferential"] * 10,
            "inference_cluster_id": [f"cluster-{index}" for index in range(10)],
            "proposed_event_family": ["earnings"] * 10,
            "identity_status": ["resolved"] * 10,
        }
    )
    alternating = ["yes", "no"] * 5
    opposite = ["no", "yes"] * 5
    reviews = pd.DataFrame(
        {
            "sample_id": sample["sample_id"],
            "sample_role": ["inferential"] * 10,
            "event_family": ["earnings"] * 10,
            "resolution_state": ["adjudicated"] * 10,
            "family_correct": [True] * 10,
            "issuer_target_correct": [True] * 10,
            "event_announced_or_completed": [True] * 10,
            "joint_correct": [True] * 10,
            "wrong_issuer": [False] * 10,
            "reviewer_one_family_correct": alternating,
            "reviewer_two_family_correct": alternating,
            "reviewer_one_issuer_target_correct": alternating,
            "reviewer_two_issuer_target_correct": alternating,
            "reviewer_one_event_announced_or_completed": alternating,
            "reviewer_two_event_announced_or_completed": opposite,
        }
    )
    metrics = audit_module._build_family_metrics(
        sample,
        reviews,
        population={
            "earnings": {"eligible_events": 10, "clusters": 10, "issuers": 10},
            **{family: {"eligible_events": 0, "clusters": 0, "issuers": 0} for family in EVENT_FAMILIES if family != "earnings"},
        },
        rule_variant_metrics=pd.DataFrame(columns=audit_module.RULE_VARIANT_METRIC_COLUMNS),
        policy=policy,
    )

    earnings = metrics.set_index("event_family").loc["earnings"]
    blockers = json.loads(str(earnings["blocker_reasons"]))
    assert "event_announced_or_completed_reviewer_agreement_below_threshold" in blockers
    assert "event_announced_or_completed_reviewer_kappa_below_threshold" in blockers
    assert not any(reason.startswith("family_correct_reviewer") for reason in blockers)
    assert not any(reason.startswith("issuer_target_correct_reviewer") for reason in blockers)


def test_rule_variant_gate_uses_inferential_clusters_only() -> None:
    policy = audit_module.load_issuer_event_precision_policy(_ROOT / "configs" / "issuer_event_precision_audit.toml")
    sample = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(10)],
            "sample_role": ["inferential"] * 10,
            "proposed_event_family": ["earnings"] * 10,
            "rule_variant": ["reported_results"] * 10,
        }
    )
    diagnostic = pd.DataFrame(
        {
            "sample_id": ["diagnostic"],
            "sample_role": ["paired_wrong_issuer_diagnostic"],
            "proposed_event_family": ["earnings"],
            "rule_variant": ["reported_results"],
        }
    )
    reviews = pd.DataFrame(
        {
            "sample_id": [f"sample-{index}" for index in range(10)] + ["diagnostic"],
            "joint_correct": [False] * 11,
        }
    )

    metrics = audit_module._build_rule_variant_metrics(
        pd.concat([sample, diagnostic], ignore_index=True),
        reviews,
        population={
            "earnings": {"reported_results": 50},
            **{family: {} for family in EVENT_FAMILIES if family != "earnings"},
        },
        policy=policy,
    )

    row = metrics.iloc[0]
    assert row["inferential_cluster_count"] == 10
    assert row["joint_successes"] == 0
    assert row["status"] == "blocked"


@pytest.mark.parametrize("target", ["sample", "authority"])
def test_tampered_sample_or_authority_is_rejected(tmp_path: Path, target: str) -> None:
    sample = _publish_sample(tmp_path)
    if target == "sample":
        path = sample.directory / "sample.parquet"
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        path = sample.directory / "_authority.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["request_sha256"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataReadinessError):
        load_issuer_event_precision_sample(sample.directory)


def test_tampered_local_reviewer_ledger_is_rejected(tmp_path: Path) -> None:
    sample = _publish_sample(tmp_path)
    one, two, adjudication = _review_ledgers(tmp_path, sample.directory)
    result = finalize_issuer_event_precision_audit(
        sample_directory=sample.directory,
        reviewer_one_path=one,
        reviewer_two_path=two,
        adjudication_path=adjudication,
        output_directory=tmp_path / "final",
    )
    local = result.directory / "reviewer_one.csv"
    local.write_bytes(local.read_bytes() + b"\n")

    with pytest.raises(DataReadinessError):
        load_issuer_event_precision_audit(result.directory)


def test_wilson_is_one_sided_and_rejects_invalid_counts() -> None:
    assert wilson_lower_bound(100, 100) == pytest.approx(0.973657, abs=1e-6)
    assert wilson_lower_bound(95, 100) < 0.95
    with pytest.raises(ValueError):
        wilson_lower_bound(1, 0)


def _publish_sample(tmp_path: Path) -> IssuerEventPrecisionSample:
    inputs = _publish_inputs(tmp_path / "source")
    return publish_issuer_event_precision_sample(
        authority_directory=inputs.authority_dir,
        policy_path=inputs.precision_policy,
        output_directory=tmp_path / "sample",
    )


def _review_ledgers(
    root: Path,
    sample_dir: Path,
    *,
    reviewer_one_id: str = "reviewer-a",
    reviewer_two_id: str = "reviewer-b",
    reviewer_one_overrides: dict[str, dict[str, str]] | None = None,
    reviewer_two_overrides: dict[str, dict[str, str]] | None = None,
    adjudication_overrides: dict[str, dict[str, str]] | None = None,
) -> tuple[Path, Path, Path]:
    one = pd.read_csv(sample_dir / "reviewer_one_template.csv", dtype=str).fillna("")
    two = pd.read_csv(sample_dir / "reviewer_two_template.csv", dtype=str).fillna("")
    adjudication = pd.read_csv(sample_dir / "adjudication_template.csv", dtype=str).fillna("")
    for frame, reviewer_id in ((one, reviewer_one_id), (two, reviewer_two_id)):
        frame["reviewer_id"] = reviewer_id
        frame["family_correct"] = "yes"
        frame["issuer_target_correct"] = "yes"
        frame["event_announced_or_completed"] = "yes"
    _apply_overrides(one, reviewer_one_overrides or {})
    _apply_overrides(two, reviewer_two_overrides or {})
    _apply_overrides(adjudication, adjudication_overrides or {})
    one_path = root / "reviewer-one.csv"
    two_path = root / "reviewer-two.csv"
    adjudication_path = root / "adjudication.csv"
    one.to_csv(one_path, index=False, lineterminator="\n")
    two.to_csv(two_path, index=False, lineterminator="\n")
    adjudication.to_csv(adjudication_path, index=False, lineterminator="\n")
    return one_path, two_path, adjudication_path


def _apply_overrides(frame: pd.DataFrame, overrides: dict[str, dict[str, str]]) -> None:
    for sample_id, values in overrides.items():
        mask = frame["sample_id"].eq(sample_id)
        assert mask.sum() == 1
        for column, value in values.items():
            frame.loc[mask, column] = value


def _publish_inputs(root: Path) -> _PublishedInputs:
    root.mkdir(parents=True, exist_ok=True)
    collection_dir = root / "collection"
    attribution_dir = root / "attribution"
    authority_dir = root / "authority"
    events_path = collection_dir / "events" / "chunk-1.parquet"
    coverage_path = collection_dir / "source_collections.parquet"
    labels_path = root / "business_labels.parquet"
    identities_path = root / "security_identities.parquet"
    decisions_path = root / "decisions.parquet"
    collection_request_sha256 = "c" * 64
    events = pd.DataFrame.from_records(
        [
            _event(
                "event-earnings",
                "Acme reports Q2 earnings and raises full-year guidance",
            ),
            _event("event-analyst", "Morgan Stanley upgrades Acme"),
            _event("event-offering", "Acme launches public stock offering"),
            _event("event-regulatory", "FDA approves Acme drug candidate"),
        ]
    )
    event_manifest = write_canonical_artifact(
        events,
        events_path,
        artifact_type="events",
        audit=_audit(len(events)),
        inputs={
            "collection_request_sha256": collection_request_sha256,
            "chunk_id": "chunk-1",
        },
        production_ready=False,
    )
    labels = pd.DataFrame(
        {
            "security_id": ["security:acme"],
            "ticker": ["ACME"],
            "company": ["Acme"],
            "business_tag": ["offering.infrastructure.storage"],
            "label_type": ["offering"],
            "match_terms": ['["acme"]'],
            "tag_rank": [1],
            "confidence": [0.99],
            "relation_use": ["exposure"],
            "effective_from_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
        }
    )
    write_canonical_artifact(
        labels,
        labels_path,
        artifact_type="security_business_labels",
        audit=_audit(len(labels)),
        production_ready=False,
    )
    identities = pd.DataFrame(
        {
            "security_id": ["security:acme"],
            "ticker": ["ACME"],
            "company": ["Acme"],
            "effective_from_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
        }
    )
    write_canonical_artifact(
        identities,
        identities_path,
        artifact_type="security_business_label_coverage",
        audit=_audit(len(identities)),
        production_ready=False,
    )
    coverage = pd.DataFrame.from_records(
        [
            {
                "collection_id": "collection-1",
                "chunk_id": "chunk-1",
                "security_id": "security:acme",
                "ticker": "ACME",
                "source_family": "alpaca",
                "requested_start_utc": pd.Timestamp("2024-12-30T00:00:00Z"),
                "requested_end_utc": pd.Timestamp("2025-01-04T00:00:00Z"),
                "completed_at_utc": pd.Timestamp("2025-01-04T00:00:00Z"),
                "status": "observed",
            }
        ]
    )
    coverage_manifest = write_canonical_artifact(
        coverage,
        coverage_path,
        artifact_type="source_collections",
        audit=_audit(len(coverage)),
        production_ready=False,
    )
    decisions = pd.DataFrame(
        {
            "security_id": ["security:acme"],
            "ticker": ["ACME"],
            "decision_time_utc": [pd.Timestamp("2025-01-03T14:00:00Z")],
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
    collection_manifest = {
        "schema": NEWS_HISTORY_MANIFEST_SCHEMA,
        "status": "complete",
        "production_ready": False,
        "availability_policy": "provider_publication_proxy",
        "request_sha256": collection_request_sha256,
        "completed_at_utc": "2025-01-04T00:00:00Z",
        "requested_chunks": 1,
        "observed_chunks": 1,
        "empty_chunks": 0,
        "failed_chunks": {},
        "source_collections_path": str(coverage_path.resolve()),
        "source_collections_sha256": coverage_manifest["artifact_sha256"],
        "artifacts": [
            {
                "chunk_id": "chunk-1",
                "security_id": "security:acme",
                "ticker": "ACME",
                "path": str(events_path.resolve()),
                "manifest_path": str(manifest_path_for(events_path).resolve()),
                "sha256": event_manifest["artifact_sha256"],
                "rows": len(events),
            }
        ],
        "artifact_count": 1,
        "total_rows": len(events),
    }
    _write_json(collection_dir / "_manifest.json", collection_manifest)
    collection_audit = root / "collection-audit.json"
    _write_json(
        collection_audit,
        {
            "passed": True,
            "request_sha256": collection_request_sha256,
            "coverage_blindspot_security_ids": [],
        },
    )
    attribution = attribute_alpaca_news_history(
        collection_dir=collection_dir,
        collection_audit_path=collection_audit,
        business_labels_path=labels_path,
        security_identities_path=identities_path,
        out_dir=attribution_dir,
    )
    assert attribution["status"] == "complete", json.dumps(attribution, default=str)
    publish_issuer_event_family_authority(
        collection_dir=collection_dir,
        collection_audit_path=collection_audit,
        attribution_dir=attribution_dir,
        decisions_path=decisions_path,
        policy_path=_FAMILY_POLICY,
        output_directory=authority_dir,
    )
    precision_policy = root / "precision-policy.toml"
    precision_policy.write_text(_precision_policy_text(), encoding="utf-8")
    return _PublishedInputs(authority_dir=authority_dir, precision_policy=precision_policy)


def _event(event_id: str, title: str) -> dict[str, object]:
    return {
        "event_id": event_id,
        "security_id": "security:acme",
        "ticker": "ACME",
        "source_family": "alpaca",
        "source": "alpaca",
        "issuer_company": "Acme",
        "issuer_company_available_at_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
        "title": title,
        "summary": f"Summary for {event_id}",
        "text": "",
        "published_at_utc": pd.Timestamp("2025-01-02T13:50:00Z"),
        "feature_available_at_utc": _EVENT_AVAILABLE,
        "availability_policy": "provider_publication_proxy",
    }


def _authority_event_row(
    event_id: str,
    family: str,
    source_family: str,
    research_eligible: bool,
) -> dict[str, object]:
    return {
        "family_event_id": event_id,
        "source_event_id": f"source-{event_id}",
        "relation_id": f"relation-{event_id}",
        "security_id": "security:acme",
        "event_family": family,
        "source_family": source_family,
        "research_eligible": research_eligible,
    }


def _candidate_source(title: str) -> dict[str, object]:
    return {
        "event_id": "source-shared",
        "source_family": "alpaca",
        "title": title,
        "published_at_utc": _EVENT_AVAILABLE,
    }


def _candidate_family(
    event_id: str,
    security_id: str,
) -> dict[str, object]:
    return {
        "family_event_id": event_id,
        "source_event_id": "source-shared",
        "relation_id": f"relation-{event_id}",
        "security_id": security_id,
        "source_family": "alpaca",
        "event_family": "earnings",
        "published_at_utc": _EVENT_AVAILABLE,
        "feature_available_at_utc": _EVENT_AVAILABLE,
        "classification_rule_id": "earnings_reported",
        "matched_text": "reports earnings",
    }


def _precision_policy_text() -> str:
    sections = []
    for family in EVENT_FAMILIES:
        sections.append(
            f"""
[family.{family}]
sample_clusters = 10
minimum_population_clusters = 1
minimum_population_issuers = 1
minimum_family_lcb = 0.0
minimum_issuer_lcb = 0.0
minimum_event_lcb = 0.0
minimum_joint_lcb = 0.0
""".strip()
        )
    return (
        """schema_version = "market_predictor.issuer_event_precision_audit.v2"
confidence_level = 0.95
reviewers_per_item = 2
unresolved_policy = "failure"
require_distinct_reviewers = true
no_wrong_issuer = true
maximum_process_memory_gib = 5.0
memory_guard_headroom_gib = 0.5
paired_wrong_issuer_diagnostics_per_cluster = 1
minimum_reviewer_agreement = 0.0
minimum_reviewer_kappa = 0.0
minimum_kappa_decisions = 1
rule_variant_gate_minimum_population_clusters = 100
minimum_rule_variant_sample_clusters = 1
minimum_rule_variant_joint_lcb = 0.0

"""
        + "\n\n".join(sections)
        + "\n"
    )


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


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
