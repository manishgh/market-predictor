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
from market_predictor.edge_rebuild import issuer_event_family_authority as authority_module
from market_predictor.edge_rebuild.issuer_event_family_authority import (
    IssuerEventFamilyAuthority,
    load_issuer_event_family_authority,
    publish_issuer_event_family_authority,
)
from market_predictor.swing.event_attribution_history import (
    attribute_alpaca_news_history,
    load_event_attribution_history,
)
from market_predictor.swing.news_history import NEWS_HISTORY_MANIFEST_SCHEMA
from market_predictor.v3.errors import DataReadinessError

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


def test_publishes_immutable_multilabel_authority(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)

    authority = _publish(inputs)

    assert authority.manifest["state"] == "complete"
    assert authority.manifest["production_ready"] is False
    assert set(authority.events["event_family"]) == {"earnings", "guidance"}
    assert authority.events["research_eligible"].astype(bool).all()
    assert not authority.events["production_eligible"].astype(bool).any()
    assigned = authority.assignments.loc[authority.assignments["status"].eq("assigned")]
    assert set(assigned["event_family"]) == {"earnings", "guidance"}
    assert set(assigned["window_name"]) == {"1d", "3d"}

    expected_identity = file_sha256(inputs.output_directory / "_authority.json")
    loaded = load_issuer_event_family_authority(
        inputs.output_directory,
        expected_authority_sha256=expected_identity,
    )
    assert_frame_equal(loaded.events, authority.events)
    with pytest.raises(DataReadinessError, match="immutable"):
        _publish(inputs)


def test_unclassified_event_is_retained_but_ineligible(tmp_path: Path) -> None:
    events = _events(
        _event(
            event_id="unclassified",
            title="Acme schedules its annual shareholder meeting",
        )
    )

    authority = _publish(_write_inputs(tmp_path, events=events))

    assert len(authority.events) == 1
    row = authority.events.iloc[0]
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

    policy = authority_module.load_swing_event_family_policy(_POLICY_PATH)
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
        load_issuer_event_family_authority(inputs.output_directory)


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


def _publish(inputs: _Inputs) -> IssuerEventFamilyAuthority:
    return publish_issuer_event_family_authority(
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
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "security_id": security_id,
        "ticker": ticker,
        "source_family": "alpaca",
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
        "requested_start_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
        "requested_end_utc": pd.Timestamp("2025-01-03T00:00:00Z"),
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
