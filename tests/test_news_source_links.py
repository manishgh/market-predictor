"""Tiny synthetic UNIT sources only; no operational attribution/admission claims."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from contextlib import closing, nullcontext
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.core.system_memory import SystemMemory
from market_predictor.research import news_source_links as n
from market_predictor.research.news_runtime_memory import guard as _runtime_guard

CUT = datetime(2024, 1, 10, 22, tzinfo=UTC)
STAMP = CUT - timedelta(hours=2)


@pytest.fixture(autouse=True)
def synthetic_resource_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resource boundaries have their own UNIT suite; these test source bindings.
    monkeypatch.setattr(n.runtime, "guard", lambda stage="news source links": None)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(root: Path, name: str, value: object) -> n.SourcePin:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(n._json(value), encoding="utf-8")
    return n.SourcePin(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _metadata(identity: str = "A", *, source: str = "alpaca", story: str = "story") -> dict[str, Any]:
    return {
        "source_family": source,
        "source_id": story,
        "source_version_sha256": _sha("raw"),
        "security_id": identity,
        "published_at_utc": STAMP.isoformat(),
        "version_available_at_utc": STAMP.isoformat(),
        "event_available_at_utc": STAMP.isoformat(),
        "identity_available_at_utc": STAMP.isoformat(),
        "first_seen_at_utc": (CUT + timedelta(days=1)).isoformat(),
        "text_sha256": _sha("synthetic"),
        "availability_semantics": "historical_proxy",
        "source_metadata": {
            "archive": "corrected",
            "event_id": "e",
            "query_security_id": "query",
            "query_ticker": "AAA",
            "raw_sha256": _sha("raw"),
        },
    }


def _record(identity: str = "v") -> dict[str, Any]:
    return {
        "version_id": identity,
        "disposition": "enriched",
        "enrichment": {
            "published_at_utc": STAMP.isoformat(),
            "available_at_utc": STAMP.isoformat(),
            "categories": ["other_unresolved"],
            "truncation_reasons": [],
            "cues": [{"category": "other_unresolved", "business_direction": "unclear", "status": "unclear"}],
        },
        "sentiment": {"status": "missing_exact_source_version", "score": None, "available_at_utc": None},
    }


@pytest.fixture
def db(tmp_path: Path) -> Any:
    with closing(n._connect(tmp_path / "unit.sqlite")) as connection:
        n._schema(connection)
        yield connection


def _seed_relation(db: sqlite3.Connection, *, target: str = "A") -> None:
    metadata = _metadata()
    event = {key: metadata[key] for key in ("published_at_utc", "version_available_at_utc")}
    key = ("corrected", "e", "query", "AAA", _sha("raw"))
    db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (*key, n._json(event), "{}"))
    relation = {"target_security_id": target, "target_ticker": "AAA", "identity_available_at_utc": STAMP, "feature_available_at_utc": STAMP}
    db.execute("INSERT INTO relations VALUES (?,?,?,?,?,?,?)", (*key, n._json(relation), n._json({"artifact": _sha("relation")})))


def test_direct_relation_does_not_require_a_saved_sentiment_score(db: sqlite3.Connection) -> None:
    _seed_relation(db)
    metadata = _metadata()
    links = n._alpaca_links(metadata, db)
    version = n._version("v", _record(), metadata, _sha("proof"))
    assert links[0][:3] == ("A", STAMP, STAMP)
    assert version.categories == ("other",)
    assert version.sentiment_score is None
    assert version.cue_available_at_utc == STAMP
    assert version.known_at_utc == STAMP  # Historical proxy does not pretend first-seen observation.


@pytest.mark.parametrize(
    "key,value",
    [("archive", "wrong"), ("event_id", "other"), ("query_security_id", "other"), ("query_ticker", "BBB"), ("raw_sha256", "0" * 64)],
)
def test_original_query_and_raw_binding_cannot_be_replaced(db: sqlite3.Connection, key: str, value: str) -> None:
    _seed_relation(db)
    metadata = _metadata()
    metadata["source_metadata"][key] = value
    assert n._alpaca_links(metadata, db) == []


def test_clock_poison_rejects_exact_identity_match(db: sqlite3.Connection) -> None:
    _seed_relation(db)
    metadata = _metadata()
    metadata["published_at_utc"] = CUT.isoformat()
    with pytest.raises(n.DataReadinessError, match="clocks differ"):
        n._alpaca_links(metadata, db)


def test_query_identity_without_direct_relation_does_not_create_link(db: sqlite3.Connection) -> None:
    assert n._alpaca_links(_metadata(), db) == []


def test_independently_proven_target_is_not_limited_to_query_cohort(db: sqlite3.Connection) -> None:
    _seed_relation(db, target="B")
    assert n._alpaca_links(_metadata("A"), db)[0][0] == "B"


def test_missing_cue_clock_retains_unusable_version() -> None:
    metadata = _metadata()
    del metadata["identity_available_at_utc"]
    result = n._version("v", _record(), metadata, _sha("proof"))
    assert result.usable is False
    assert result.known_at_utc == STAMP
    assert result.categories == ()


def test_sentiment_has_separate_later_availability() -> None:
    record = _record()
    record["sentiment"] = {"status": "matched", "score": -0.7, "available_at_utc": CUT.isoformat()}
    version = n._version("v", record, _metadata(), _sha("proof"))
    assert version.cue_available_at_utc == STAMP
    assert version.sentiment_available_at_utc == CUT
    assert version.sentiment_score == -0.7


def _insert_version(db: sqlite3.Connection, identity: str, metadata: dict[str, Any], record: dict[str, Any], *, link: bool) -> None:
    version = n._version(identity, record, metadata, _sha(identity))
    db.execute(
        "INSERT INTO versions VALUES (?,?,?,?,?,?,?)",
        (
            identity,
            identity,
            version.source_family,
            version.source_document_id,
            version.known_at_utc.isoformat(),
            n._json(asdict(version)),
            None,
        ),
    )
    if link:
        company = n.features.VerifiedCompanyLink(identity, "A", STAMP, STAMP, _sha("link"), "historical_proxy")
        db.execute("INSERT INTO links VALUES (?,?,?,?)", (identity, "A", STAMP.isoformat(), n._json(asdict(company))))
        db.execute(
            "INSERT OR IGNORE INTO candidates VALUES (?,?,?,?)", ("A", version.source_family, version.source_document_id, STAMP.isoformat())
        )


def test_period_keeps_old_link_and_newer_unusable_unlinked_revision(db: sqlite3.Connection) -> None:
    _insert_version(db, "old", _metadata(), _record(), link=True)
    metadata = _metadata()
    metadata["version_available_at_utc"] = CUT.isoformat()
    metadata["source_version_sha256"] = _sha("new")
    _insert_version(db, "new", metadata, {"disposition": "unavailable_text"}, link=False)
    reader = n.NewsSourceLinkReader({}, {}, db)
    result = reader.for_period("A", CUT, CUT)
    assert {row.version_id for row in result.versions} == {"old", "new"}
    assert [row.version_id for row in result.links] == ["old"]
    assert result.coverage == () and not result.source_coverage_admitted
    technical = n.features.TechnicalValue(None, None, "historical_proxy")
    decision = n.features.NewsDecision("d", "A", CUT, technical, technical, technical)
    (features,) = n.features.aggregate_news_decisions(
        decisions=(decision,), versions=result.versions, links=result.links, purpose="research_proxy"
    )
    assert features.suppressed_story_count == 1
    assert features.as_values()["news_direct_other_present_3d"] is None


def test_future_revision_is_masked_and_unknown_clock_is_explicit(db: sqlite3.Connection) -> None:
    _insert_version(db, "old", _metadata(), _record(), link=True)
    metadata = _metadata()
    metadata["version_available_at_utc"] = (CUT + timedelta(hours=1)).isoformat()
    metadata["source_version_sha256"] = _sha("future")
    _insert_version(db, "future", metadata, {"disposition": "unavailable_text"}, link=False)
    db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?)", ("unknown", "unknown", "alpaca", "story", None, None, "missing_clock"))
    result = n.NewsSourceLinkReader({}, {}, db).for_decision("A", CUT)
    assert [row.version_id for row in result.versions] == ["old"]
    assert len(result.uncertainties) == 1
    assert result.uncertainties[0].reason == "missing_clock"
    assert result.uncertainties[0].known_at_utc is None
    assert result.uncertainties[0].applies_at(CUT)
    assert not result.uncertainties[0].applies_at(STAMP - timedelta(seconds=1))


def test_slice_bound_rejects_instead_of_silently_truncating(db: sqlite3.Connection) -> None:
    _insert_version(db, "one", _metadata(), _record(), link=True)
    _insert_version(db, "two", _metadata(), _record(), link=False)
    with pytest.raises(n.DataReadinessError, match="never truncate"):
        n.NewsSourceLinkReader({}, {}, db).for_decision("A", CUT, max_versions=1)


def test_sec_document_identity_does_not_collapse_distinct_exhibits(db: sqlite3.Connection) -> None:
    for unit in ("accession/1", "accession/2"):
        metadata = _metadata(source="sec", story=unit)
        metadata["source_metadata"] = {
            "receipt_sha256": _sha(unit),
            "body_sha256": _sha("raw"),
            "sec_cik": "123",
            "retrieval_sec_cik": "123",
            "identity_authority_sha256": _sha("identity"),
        }
        filing = {
            "security_id": "A",
            "ticker": "AAA",
            "sec_cik": "123",
            "accepted_at_utc": STAMP,
            "available_at_utc": STAMP,
            "identity_available_at_utc": STAMP,
            "identity_authority_sha256": _sha("identity"),
        }
        db.execute(
            "INSERT INTO sec_receipts VALUES (?,?,?,?,?)",
            (
                unit,
                _sha("raw"),
                "A",
                n._json({"receipt": {"receipt_sha256": _sha(unit), "sec_cik": "123"}, "filing": filing}),
                n._json({"unit": unit}),
            ),
        )
        assert n._sec_links(metadata, db)[0][0] == "A"
        assert n._version(unit, _record(), metadata, _sha("proof")).source_document_id == unit
    metadata["source_metadata"]["sec_cik"] = "456"
    with pytest.raises(n.DataReadinessError, match="identity or clocks"):
        n._sec_links(metadata, db)


def test_strict_policy_rejects_status_instead_of_source_pins() -> None:
    with pytest.raises(ValidationError):
        n.NewsSourceLinkPolicy.model_validate_json(json.dumps({"schema": n.SCHEMA + "_config", "verified": True}))


def test_open_reader_rechecks_real_input_on_context_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(n.saved, "_guard", lambda: None)
    monkeypatch.setattr(n, "_implementation", lambda root: {})
    source = _write(tmp_path, "input.json", {"synthetic_unit": True})
    directory = tmp_path / "index"
    directory.mkdir()
    with closing(n._connect(directory / "index.sqlite")) as db:
        n._schema(db)
    policy = {
        "schema": n.SCHEMA + "_config",
        "corpus_manifest": source.model_dump(),
        "population_manifest": source.model_dump(),
        "population_database": source.model_dump(),
        "identity_manifest": source.model_dump(),
        "archives": [
            {"name": name, "collection_manifest": source.model_dump(), "attribution_manifest": source.model_dump()}
            for name in ("early", "later", "corrected")
        ],
    }
    config = _write(tmp_path, "unit_config.json", policy)
    request = _write(
        tmp_path, "index/_request.json", {"implementation_files": {}, "policy": policy, "config": config.model_dump(), **n.CLOSED}
    )
    manifest = _write(
        tmp_path,
        "index/_manifest.json",
        {
            "schema": n.SCHEMA,
            "status": "complete_source_links_only",
            "coverage": "unknown",
            "request_sha256": request.sha256,
            "implementation_files": {},
            "source_files": {source.path: source.sha256},
            "artifacts": {"_request.json": request.sha256, "index.sqlite": n.saved._hash(directory / "index.sqlite")},
            **n.CLOSED,
        },
    )
    with pytest.raises(n.DataReadinessError, match="changed"):
        with n.open_news_source_link_index(root=tmp_path, index=manifest):
            (tmp_path / source.path).write_text("mutated", encoding="utf-8")


def test_public_builder_owns_lease_before_input_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state: list[str] = []
    monkeypatch.setattr(n, "heavy_job_lease", lambda *args, **kwargs: state.append("lease") or nullcontext())
    monkeypatch.setattr(n.saved, "_guard", lambda: None)
    config = _write(tmp_path, "invalid.json", {"synthetic_unit": True})
    with pytest.raises(ValidationError):
        n.build_news_source_link_index(root=tmp_path, config=config, output=tmp_path / "output")
    assert state == ["lease"]
    assert not (tmp_path / "output" / "_manifest.json").exists()


def _original_archive(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[n.NewsLinkArchive, dict[str, Any]]:
    # Reuse the existing explicitly synthetic original-event fixture, not its scores.
    from tests.test_news_sentiment_reuse import _child, _fixture, _hash, _request
    from tests.test_news_sentiment_reuse import _write as write_json

    source, version = _fixture(root, monkeypatch)
    collection = json.loads((root / source.collection_manifest.path).read_bytes())
    original = collection["artifacts"][0]
    evidence = _write(root, "unit_identity.json", {"synthetic_unit": True})
    request = {
        "collection_manifest_path": source.collection_manifest.path,
        "collection_manifest_sha256": source.collection_manifest.sha256,
        "collection_request_sha256": collection["request_sha256"],
        "attribution_policy_sha256": _sha("policy"),
    }
    for stem in ("security_identities", "business_labels", "collection_audit"):
        request[stem + "_path"] = evidence.path
        request[stem + "_sha256"] = evidence.sha256
    identities = pd.DataFrame(
        [
            {
                "security_id": "cohort-company-A",
                "ticker": "A",
                "company": "Synthetic A",
                "effective_from_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
                "effective_to_utc": None,
                "available_at_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
            }
        ]
    )
    identities["effective_to_utc"] = pd.to_datetime(identities["effective_to_utc"], utc=True)
    identities.to_parquet(root / "unit_identities.parquet", index=False)
    request["security_identities_path"] = "unit_identities.parquet"
    request["security_identities_sha256"] = _hash(root / "unit_identities.parquet")
    registry_sha = n.original_attribution._identity_registry_sha256(n.original_attribution._prepare_identities(identities))
    request_sha = _request(root / "relations/_request.json", request)
    clock = version.metadata["version_available_at_utc"]
    relation = {
        "relation_id": "r",
        "event_id": "event-query-A",
        "source_security_id": "legacy-query-A",
        "source_ticker": "A",
        "target_security_id": "cohort-company-A",
        "target_ticker": "A",
        "relation_channel": "direct_issuer",
        "event_feature_available_at_utc": clock,
        "identity_available_at_utc": clock,
        "feature_available_at_utc": clock,
        "attribution_policy_sha256": _sha("policy"),
        "security_identity_registry_sha256": registry_sha,
    }
    child = _child(
        root,
        "relations/rows.parquet",
        pd.DataFrame([relation]),
        {
            "chunk_id": "chunk",
            "event_attribution_request_sha256": request_sha,
            "source_event_artifact_sha256": original["sha256"],
            "security_identities_sha256": request["security_identities_sha256"],
            "business_labels_sha256": request["business_labels_sha256"],
            "attribution_policy_sha256": request["attribution_policy_sha256"],
        },
    )
    sidecar_path = root / child["manifest_path"]
    sidecar = json.loads(sidecar_path.read_bytes())
    sidecar["columns"] = list(relation)
    write_json(sidecar_path, sidecar)
    child["source_event_sha256"] = original["sha256"]
    manifest = _write(
        root,
        "relations/_manifest.json",
        {"schema": "swing.event_attribution_manifest.v1", "status": "complete", "request_sha256": request_sha, "artifacts": [child]},
    )
    metadata = version.metadata
    metadata["source_metadata"]["archive"] = "corrected"
    metadata["security_id"] = "cohort-company-A"
    assert _hash(root / manifest.path) == manifest.sha256
    return n.NewsLinkArchive(name="corrected", collection_manifest=source.collection_manifest, attribution_manifest=manifest), metadata


def test_original_parquet_sidecars_bind_links_without_opening_sentiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db: sqlite3.Connection
) -> None:
    archive, metadata = _original_archive(tmp_path, monkeypatch)
    # A malformed saved score has no role in source-link admission.
    (tmp_path / "sentiment/scores.parquet").write_bytes(b"not a sentiment artifact")
    files: dict[str, str] = {}
    n._load_alpaca(tmp_path, archive, db, pd.DataFrame(), pd.DataFrame(), files)
    assert n._alpaca_links(metadata, db)[0][0] == "cohort-company-A"
    assert "collection/events.parquet" in files
    assert "relations/rows.parquet.manifest.json" in files
    assert not any(name.startswith("sentiment/") for name in files)


def test_original_relation_source_artifact_poison_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db: sqlite3.Connection) -> None:
    archive, _ = _original_archive(tmp_path, monkeypatch)
    manifest = json.loads((tmp_path / archive.attribution_manifest.path).read_bytes())
    manifest["artifacts"][0]["source_event_sha256"] = _sha("wrong-original")
    pin = _write(tmp_path, archive.attribution_manifest.path, manifest)
    archive = archive.model_copy(update={"attribution_manifest": pin})
    with pytest.raises(n.DataReadinessError, match="raw event artifact differs"):
        n._load_alpaca(tmp_path, archive, db, pd.DataFrame(), pd.DataFrame(), {})


def test_target_without_its_own_active_identity_is_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, db: sqlite3.Connection
) -> None:
    archive, metadata = _original_archive(tmp_path, monkeypatch)
    target = tmp_path / "relations/rows.parquet"
    frame = pd.read_parquet(target)
    frame.loc[0, "target_security_id"] = "unproven-other-company"
    frame.to_parquet(target, index=False)
    digest = n.saved._hash(target)
    sidecar = json.loads(Path(str(target) + ".manifest.json").read_bytes())
    sidecar["artifact_sha256"] = digest
    _write(tmp_path, "relations/rows.parquet.manifest.json", sidecar)
    manifest = json.loads((tmp_path / archive.attribution_manifest.path).read_bytes())
    manifest["artifacts"][0]["sha256"] = digest
    pin = _write(tmp_path, archive.attribution_manifest.path, manifest)
    n._load_alpaca(tmp_path, archive.model_copy(update={"attribution_manifest": pin}), db, pd.DataFrame(), pd.DataFrame(), {})
    assert n._alpaca_links(metadata, db) == []


def test_public_build_and_read_preserve_all_versions_and_closed_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """UNIT orchestration: synthetic bookends/SEC inventory, actual artifact link and SQLite writer."""
    from market_predictor.universe.legacy_query_identity import PROOF_COLUMNS

    archive, metadata = _original_archive(tmp_path, monkeypatch)
    metadata.update(
        source_family="alpaca",
        source_id="story",
        text_sha256=_sha("synthetic"),
        event_available_at_utc=metadata["version_available_at_utc"],
        identity_available_at_utc=metadata["version_available_at_utc"],
    )
    source_path = tmp_path / "population.sqlite"
    with sqlite3.connect(source_path) as source:
        source.executescript(
            "CREATE TABLE versions(version_id TEXT PRIMARY KEY,cluster_id TEXT,security_id TEXT,source_family TEXT,content_json TEXT);"
            "CREATE TABLE records(phase TEXT,ordinal INTEGER,version_id TEXT,record_json TEXT);"
        )
        source.execute("INSERT INTO versions VALUES (?,?,?,?,?)", ("v", "cluster", "cohort-company-A", "alpaca", n._json(metadata)))
        source.execute("INSERT INTO records VALUES (?,?,?,?)", ("alpaca", 1, "v", n._json(metadata)))
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    record = _record()
    record.update(
        cluster_id="cluster", retained_security_id="cohort-company-A", source_family="alpaca", content_json_sha256=_sha(n._json(metadata))
    )
    record["enrichment"].update(version_ref="v", source="alpaca", document_ref="alpaca:story", method="rule_cues")
    (corpus_dir / "part-000000.jsonl").write_text(n._json(record) + "\n", encoding="utf-8")
    placeholder = _write(tmp_path, "unit_parent.json", {"synthetic_unit": True})
    policy = {
        "schema": n.SCHEMA + "_config",
        "corpus_manifest": placeholder.model_dump(),
        "population_manifest": placeholder.model_dump(),
        "population_database": {"path": "population.sqlite", "sha256": n.saved._hash(source_path)},
        "identity_manifest": placeholder.model_dump(),
        "archives": [archive.model_copy(update={"name": name}).model_dump() for name in ("early", "later", "corrected")],
    }
    config = _write(tmp_path, "unit_config.json", policy)
    monkeypatch.setattr(n, "_implementation", lambda root: {})
    monkeypatch.setattr(n, "heavy_job_lease", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        n,
        "_bookends",
        lambda root, policy, files: (source_path, {}, {"artifacts": {"part-000000.jsonl": "unit"}, "source_versions": 1}, corpus_dir),
    )
    monkeypatch.setattr(n, "_identities", lambda *args: (pd.DataFrame(columns=n.BRIDGE_COLUMNS), pd.DataFrame(columns=PROOF_COLUMNS)))
    monkeypatch.setattr(n, "_load_sec", lambda *args: None)
    output = tmp_path / "output"
    result = n.build_news_source_link_index(root=tmp_path, config=config, output=output)
    assert result["counts"]["query_copy_records"] == result["counts"]["verified_company_links"] == 1
    assert all(result[key] is False for key in n.CLOSED)
    pin = n.SourcePin(path="output/_manifest.json", sha256=result["manifest_sha256"])
    with n.open_news_source_link_index(root=tmp_path, index=pin) as reader:
        selected = reader.for_period("cohort-company-A", STAMP, CUT)
        assert len(selected.versions) == len(selected.links) == 1
        assert selected.uncertainties == () and selected.coverage == ()
        assert reader.proof(selected.versions[0].source_proof_sha256)["ordinal"] == 1
        assert reader.proof(selected.links[0].proof_sha256)["relation"]["source_event_sha256"]
    with pytest.raises(n.DataReadinessError, match="already exists"):
        n.build_news_source_link_index(root=tmp_path, config=config, output=output)


@pytest.mark.parametrize("retrieval_cik", ["123", "456", "789"])
def test_sec_receipt_retrieval_cik_belongs_to_cofiler_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    db: sqlite3.Connection,
    retrieval_cik: str,
) -> None:
    """Synthetic original receipt Parquet/ZIP; filing-map owner is a two-filer UNIT double."""
    monkeypatch.setattr(n.saved, "_guard", lambda: None)
    directory = tmp_path / "sec"
    shards = directory / "shards"
    shards.mkdir(parents=True)
    body = b"Synthetic co-filed document only."
    with zipfile.ZipFile(shards / "shard-00000.zip", "w") as archive:
        archive.writestr("document.html", body)
    receipt = dict.fromkeys(n.RECEIPT_COLUMNS)
    receipt.update(
        unit_id="accession/1",
        phase="document",
        sec_cik=retrieval_cik,
        accession_number="accession",
        attempt=1,
        state="archived",
        status_code=200,
        member="document.html",
        body_length=len(body),
        body_sha256=hashlib.sha256(body).hexdigest(),
    )
    receipt["receipt_sha256"] = n.json_sha256({key: value for key, value in receipt.items() if key != "receipt_sha256"})
    pd.DataFrame([{**receipt, "shard": "shard-00000"}]).to_parquet(shards / "shard-00000.parquet", index=False)
    manifest = {
        "shards": {
            "shard-00000": {
                "parquet_sha256": n.saved._hash(shards / "shard-00000.parquet"),
                "zip_sha256": n.saved._hash(shards / "shard-00000.zip"),
                "attempts": 1,
            }
        },
        "totals": {"units_by_phase_and_state": {"document/archived": 1}},
    }
    filings = [
        {
            "security_id": identity,
            "ticker": identity,
            "sec_cik": cik,
            "accepted_at_utc": STAMP,
            "available_at_utc": STAMP,
            "identity_available_at_utc": STAMP,
            "identity_authority_sha256": _sha("identities"),
        }
        for identity, cik in (("A", "123"), ("B", "456"))
    ]
    pin = _write(tmp_path, "synthetic_sec_parent.json", {"synthetic_unit": True})
    settings = {"sec_archive": pin.model_dump(), "sec_inventory": pin.model_dump()}
    monkeypatch.setattr(n.sec_sources, "_sec_metadata", lambda *args: (directory, manifest, {}, {}))
    monkeypatch.setattr(n.sec_sources, "_filing_map", lambda *args: ({"accession": filings}, _sha("acceptance")))
    if retrieval_cik == "789":
        with pytest.raises(n.DataReadinessError, match="outside the proven filer set"):
            n._load_sec(tmp_path, settings, db, {})
        return
    n._load_sec(tmp_path, settings, db, {})
    rows = db.execute("SELECT security_id,payload,proof FROM sec_receipts ORDER BY security_id").fetchall()
    assert [row["security_id"] for row in rows] == ["A", "B"]
    for row, filing in zip(rows, filings, strict=True):
        saved = json.loads(row["payload"])
        proof = json.loads(row["proof"])
        assert saved["receipt"]["sec_cik"] == retrieval_cik
        assert saved["filing"]["sec_cik"] == filing["sec_cik"]
        assert proof["retrieval_sec_cik"] == retrieval_cik.zfill(10)
        metadata = _metadata(filing["security_id"], source="sec", story="accession/1")
        metadata["source_version_sha256"] = receipt["body_sha256"]
        metadata["source_metadata"] = {
            "receipt_sha256": receipt["receipt_sha256"],
            "body_sha256": receipt["body_sha256"],
            "sec_cik": filing["sec_cik"],
            "retrieval_sec_cik": retrieval_cik,
            "identity_authority_sha256": filing["identity_authority_sha256"],
        }
        assert n._sec_links(metadata, db)[0][0] == filing["security_id"]


def test_period_uncertainty_only_applies_after_its_known_clock(db: sqlite3.Connection) -> None:
    _insert_version(db, "old", _metadata(), _record(), link=True)
    db.execute(
        "INSERT INTO versions VALUES (?,?,?,?,?,?,?)",
        ("later", "later", "alpaca", "story", CUT.isoformat(), None, "missing_original_version_hash"),
    )
    reader = n.NewsSourceLinkReader({}, {}, db)
    period = reader.for_period("A", STAMP, CUT)
    assert len(period.uncertainties) == 1
    assert not period.uncertainties[0].applies_at(STAMP)
    assert period.uncertainties[0].applies_at(CUT)
    assert reader.for_decision("A", STAMP).uncertainties == ()


def test_original_binding_io_does_not_reenter_old_eightyfive_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    db: sqlite3.Connection,
) -> None:
    archive, metadata = _original_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(n.runtime, "guard", _runtime_guard)
    monkeypatch.setattr(n.runtime, "assert_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(n.runtime, "system_memory_snapshot", lambda: SystemMemory(1000, 110))

    def obsolete_guard() -> None:
        raise AssertionError("new bridge reentered completed producer memory policy")

    monkeypatch.setattr(n.saved, "_guard", obsolete_guard)
    files: dict[str, str] = {}
    n._load_alpaca(tmp_path, archive, db, pd.DataFrame(), pd.DataFrame(), files)
    n.runtime.recheck_files(tmp_path, files)
    assert n._alpaca_links(metadata, db)[0][0] == "cohort-company-A"
