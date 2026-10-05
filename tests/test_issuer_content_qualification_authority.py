"""Synthetic source/review authority fixtures; no retained provider data or labels."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.governance.issuer_content_qualification import POLICY_SHA256, select_content_review_sample
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.research import issuer_content_qualification_authority as qualification
from market_predictor.research import issuer_content_review_population as population
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction_profile import QUALIFIED_EVENT_COLUMNS
from market_predictor.swing.features.issuer_reaction_profile import _select


def _pin(root: Path, path: Path) -> SourcePin:
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))


def _rewrite(path: Path, document: dict[str, Any]) -> None:
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")


@pytest.fixture
def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(population, "_guard", lambda: None)
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    # Copy actual implementation bytes into the synthetic root, so production
    # pin verification remains exercised rather than patched out.
    package = Path(qualification.__file__).resolve().parents[1]
    for name in qualification.IMPLEMENTATION_PATHS:
        destination = tmp_path / "src/market_predictor" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, destination)
    parent = tmp_path / "data/research/population"
    parent.mkdir(parents=True)
    config = tmp_path / "config.json"
    source = tmp_path / "synthetic_source.json"
    source.write_text('{"synthetic":true}', encoding="utf-8")
    settings = {key: _pin(tmp_path, source).model_dump(mode="json")
                for key in population._CONFIG_KEYS if key != "schema"} | {"schema": population.CONFIG_SCHEMA}
    _rewrite(config, settings)
    implementations = {f"src/market_predictor/{name}": file_sha256(package / name) for name in population.IMPLEMENTATION_PATHS}
    request = {"schema": f"{population.SCHEMA}_request", "config": _pin(tmp_path, config).model_dump(mode="json"),
               "settings": settings, "implementation_files": implementations, "review_policy_sha256": POLICY_SHA256,
               "extraction_policy_sha256": population.EXTRACTION_POLICY_SHA256,
               "training_eligible": False, "serving_eligible": False}
    write_json_object(parent / "_request.json", request)
    with population._connect(parent / "population.sqlite") as connection:
        for family in ("earnings", "guidance"):
            for number in range(200):
                source_id = f"{family}-{number}"
                text = f"Issuer {number} reports Q2 2020 earnings." if family == "earnings" else f"Issuer {number} raises FY2020 guidance."
                cluster = json_sha256([source_id, number])
                raw_hash = json_sha256({"synthetic_body": text})
                row = {
                    "source_family": "alpaca", "source_id": source_id, "source_version_sha256": raw_hash,
                    "security_id": f"issuer-{number}", "ticker": f"T{number}",
                    "published_at_utc": "2020-06-01T20:00:00Z", "version_available_at_utc": "2020-06-01T20:00:00Z",
                    "event_available_at_utc": "2020-06-01T20:00:00Z", "identity_available_at_utc": "2020-01-01T00:00:00Z",
                    "identity_authority_sha256": "a" * 64, "first_seen_at_utc": "2026-09-01T00:00:00Z",
                    "source_locator": f"synthetic#{source_id}", "content_kind": "headline", "encoding": None,
                    "alias_proof": {"synthetic": True}, "source_metadata": {"synthetic": True},
                    "availability_semantics": "historical_proxy", "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "unavailable_reasons": [],
                }
                candidate = {"event_family": family, "rule_id": f"{family}_statement", "fiscal_period": "Q2 2020",
                             "unresolved_reasons": []}
                connection.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)",
                    (json_sha256([cluster, raw_hash]), cluster, row["security_id"], "alpaca", 2020,
                     population._json(row), text, population._json([candidate])))
        connection.commit()
    value = {"root": tmp_path, "parent": parent, "implementations": implementations,
             "source_files": {path.relative_to(tmp_path).as_posix(): file_sha256(path) for path in (config, source)},
             "source": source, "output": tmp_path / "data/research/qualified"}
    return _seal_fixture(value)


def _seal_fixture(value: dict[str, Any]) -> dict[str, Any]:
    root, parent = value["root"], value["parent"]
    # Synthetic fixture rebuilding is deliberate; production outputs never overwrite.
    for name in ("sample.json", "_manifest.json"):
        (parent / name).unlink(missing_ok=True)
    if (parent / "blind").exists():
        blind = (parent / "blind").resolve()
        assert blind.is_relative_to(root.resolve()) and blind.name == "blind"
        shutil.rmtree(blind)
    with sqlite3.connect(parent / "population.sqlite") as connection:
        clusters = population._clusters(connection)
        population._export_review(connection, parent, clusters)
        samples = select_content_review_sample(clusters)
        artifacts = {path.relative_to(parent).as_posix(): file_sha256(path) for path in parent.rglob("*") if path.is_file()}
        manifest = {"schema": population.SCHEMA, "status": "complete_review_population_only", "artifacts": artifacts,
                    "request_sha256": file_sha256(parent / "_request.json"), "source_files": value["source_files"],
                    "implementation_files": value["implementations"],
                    "training_eligible": False, "serving_eligible": False, "promotion_eligible": False}
        write_json_object(parent / "_manifest.json", manifest)
        authority = _pin(root, parent / "_manifest.json")
        documents = []
        for reviewer in ("reviewer-a", "reviewer-b"):
            entries = []
            for item in samples:
                encoded, text = connection.execute(
                    "SELECT content_json,text FROM versions WHERE cluster_id=? AND text IS NOT NULL ORDER BY version_id LIMIT 1",
                    (item.cluster_id,),
                ).fetchone()
                row = json.loads(encoded)
                positive = item.role == "candidate"
                entries.append({"sample_id": item.sample_id, "source_id": row["source_id"],
                    "source_version_sha256": row["source_version_sha256"], "supporting_text_sha256": row["text_sha256"],
                    "supporting_spans": [[0, len(text)]] if positive else [],
                    **dict.fromkeys(qualification._FIELDS, positive)})
            document = {"schema": qualification.REVIEW_SCHEMA, "population_authority": authority.model_dump(mode="json"),
                        "sample_sha256": file_sha256(parent / "sample.json"), "reviewer_id": reviewer,
                        "reviewer_kind": "model_assisted", "reviewer_model": "synthetic-test-model",
                        "evidence_scope": "source_only", "independent_review": True, "reviews": entries}
            path = root / f"{reviewer}.json"
            _rewrite(path, document)
            documents.append(path)
    return value | {"authority": authority, "review_paths": documents, "samples": samples}


def _publish(value: dict[str, Any]) -> dict[str, Any]:
    first, second = value["review_paths"]
    return qualification.publish_issuer_content_qualification(
        root=value["root"], population_authority=value["authority"],
        reviewer_files=(_pin(value["root"], first), _pin(value["root"], second)), output=value["output"],
    )


def test_actual_reviews_create_non_circular_immutable_research_authority(fixture: dict[str, Any]) -> None:
    result = _publish(fixture)
    assert result["research_feature_eligible"]
    assert result["counts"]["qualified_versions"] == 400
    assert all(result[name] is False for name in ("training_eligible", "serving_eligible", "promotion_eligible", "coverage_established"))
    events = pd.read_parquet(fixture["output"] / "events.parquet")
    assert tuple(events.columns) == QUALIFIED_EVENT_COLUMNS
    authority = fixture["output"] / "_authority.json"
    assert events.qualification_authority_sha256.eq(file_sha256(authority)).all()
    assert result["artifacts"]["_authority.json"] == result["qualification_authority_sha256"] == file_sha256(authority)
    assert "events.parquet" not in json.loads(authority.read_text())["source_files"]
    with pytest.raises(DataReadinessError, match="immutable"):
        _publish(fixture)


def test_missing_actual_reviews_publish_explicit_nonadmission(fixture: dict[str, Any]) -> None:
    for path in fixture["review_paths"]:
        document = json.loads(path.read_text())
        document["reviews"] = []
        _rewrite(path, document)
    result = _publish(fixture)
    assert not result["research_feature_eligible"]
    assert result["counts"].get("qualified_versions", 0) == 0
    assert result["counts"]["projected_versions"] == 400
    authority = json.loads((fixture["output"] / "_authority.json").read_text())
    assert "noncandidate_review_sample_incomplete" in authority["metrics"]["families"]["earnings"]["reasons"]


@pytest.mark.parametrize("dependency,accepted", [
    ("swing/contracts/holding_accounting.py", True),
    ("swing/contracts/holding_materialization.py", True),
    ("catalysts/issuer_events/content_review.py", False),
    ("governance/issuer_content_qualification.py", False),
])
def test_completed_population_keeps_unrelated_producer_code_as_provenance(
    fixture: dict[str, Any], dependency: str, accepted: bool,
) -> None:
    """Synthetic historical hashes differ; all source/review bytes stay unchanged."""
    name = f"src/market_predictor/{dependency}"
    fixture["implementations"] = fixture["implementations"] | {name: "f" * 64}
    request_path = fixture["parent"] / "_request.json"
    request = json.loads(request_path.read_text())
    request["implementation_files"] = fixture["implementations"]
    _rewrite(request_path, request)
    fixture = _seal_fixture(fixture)
    if not accepted:
        with pytest.raises(DataReadinessError, match="replay implementation"):
            _publish(fixture)
        return
    result = _publish(fixture)
    assert result["research_feature_eligible"]
    authority = json.loads((fixture["output"] / "_authority.json").read_text())
    assert authority["producer_implementation_files"][name] == "f" * 64
    assert authority["implementation_files"][name] == authority["source_files"][name]
    assert authority["source_files"][name] == file_sha256(fixture["root"] / name)
    assert authority["source_files"][name] != "f" * 64


def test_parent_producer_inventory_must_match_its_original_request(fixture: dict[str, Any]) -> None:
    path = fixture["parent"] / "_manifest.json"
    manifest = json.loads(path.read_text())
    name = "src/market_predictor/swing/contracts/holding_accounting.py"
    manifest["implementation_files"][name] = "f" * 64
    _rewrite(path, manifest)
    fixture["authority"] = _pin(fixture["root"], path)
    with pytest.raises(DataReadinessError, match="producer implementation inventory"):
        _publish(fixture)


@pytest.mark.parametrize("poison", ["undeclared_scope", "nonindependent", "unknown_version", "text_hash", "span", "wrong_parent"])
def test_malformed_or_unbound_review_rejected(fixture: dict[str, Any], poison: str) -> None:
    path = fixture["review_paths"][0]
    document = json.loads(path.read_text())
    if poison == "undeclared_scope":
        del document["evidence_scope"]
    elif poison == "nonindependent":
        document["independent_review"] = False
    elif poison == "unknown_version":
        document["reviews"][0]["source_version_sha256"] = "f" * 64
    elif poison == "text_hash":
        document["reviews"][0]["supporting_text_sha256"] = "f" * 64
    elif poison == "span":
        document["reviews"][0]["supporting_spans"] = [[0, 100000]]
    else:
        document["population_authority"]["sha256"] = "f" * 64
    _rewrite(path, document)
    with pytest.raises(DataReadinessError):
        _publish(fixture)
    assert not (fixture["output"] / "_manifest.json").exists()


@pytest.mark.parametrize("artifact", ["population.sqlite", "sample.json", "blind_packet", "source", "implementation"])
def test_parent_source_and_child_bytes_are_rechecked(fixture: dict[str, Any], artifact: str) -> None:
    path = (next((fixture["parent"] / "blind").glob("*.json")) if artifact == "blind_packet" else
            fixture["source"] if artifact == "source" else
            fixture["root"] / next(iter(fixture["implementations"])) if artifact == "implementation" else
            fixture["parent"] / artifact)
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(DataReadinessError):
        _publish(fixture)


def _revision(value: dict[str, Any], fault: str) -> dict[str, Any]:
    with sqlite3.connect(value["parent"] / "population.sqlite") as connection:
        version, cluster, security, source, year, encoded, text, _ = connection.execute(
            "SELECT * FROM versions WHERE json_extract(content_json,'$.source_id')='earnings-0'",
        ).fetchone()
        row = json.loads(encoded)
        row["source_version_sha256"] = "f" * 64
        row["version_available_at_utc"] = row["event_available_at_utc"] = "2020-06-02T20:00:00Z"
        if fault == "future":
            row["version_available_at_utc"] = "2026-01-01T20:00:00Z"
            row["unavailable_reasons"] = ["version_after_initial_fit_cutoff"]
            text = None
        elif fault == "unknown_identity":
            row["security_id"] = security = None
            row.pop("identity_available_at_utc")
        else:
            text = "Issuer 0 discusses products without any earnings announcement."
            row["text_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        connection.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)",
            (json_sha256([version, fault]), cluster, security, source, year, population._json(row), text, "[]"))
        connection.commit()
    value = _seal_fixture(value)
    # The precision review deliberately binds the original candidate version;
    # negative revisions remain in the full projection history independently.
    for path in value["review_paths"]:
        document = json.loads(path.read_text())
        with sqlite3.connect(value["parent"] / "population.sqlite") as connection:
            original, text = connection.execute(
                "SELECT content_json,text FROM versions WHERE json_extract(content_json,'$.source_id')='earnings-0' "
                "AND json_extract(content_json,'$.source_version_sha256')!=?", ("f" * 64,),
            ).fetchone()
        row = json.loads(original)
        for entry in document["reviews"]:
            if entry["source_id"] == "earnings-0":
                entry.update(source_version_sha256=row["source_version_sha256"], supporting_text_sha256=row["text_sha256"])
                if entry["family_present"]:
                    entry["supporting_spans"] = [[0, len(text)]]
        _rewrite(path, document)
    return value


@pytest.mark.parametrize("fault", ["unknown_identity", "future", "rejected"])
def test_full_history_prevents_old_revision_resurrection(fixture: dict[str, Any], fault: str) -> None:
    fixture = _revision(fixture, fault)
    result = _publish(fixture)
    events = pd.read_parquet(fixture["output"] / "events.parquet")
    history = events.loc[events.event_id.eq("earnings-0")]
    dispositions = [json.loads(line) for line in (fixture["output"] / "version_dispositions.jsonl").read_text().splitlines()]
    assert len(dispositions) == 401
    if fault == "unknown_identity":
        assert history.empty
        assert result["counts"]["excluded_projection_versions"] == 2
    elif fault == "future":
        assert len(history) == 1 and history.qualification_status.eq("qualified").all()
        assert result["counts"]["future_metadata_only_versions"] == 1
    else:
        assert set(history.qualification_status) == {"qualified", "unclassified"}
        decisions = pd.DataFrame({"security_id": ["issuer-0"], "ticker": ["T0"], "decision_id": ["decision"],
                                  "decision_time_utc": [pd.Timestamp("2020-06-03T20:00:00Z")]})
        assert _select(history, decisions).empty


def test_source_mutated_during_projection_prevents_complete_manifest(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = qualification._publish_events

    def mutate(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        fixture["source"].write_text("changed", encoding="utf-8")
        return result

    monkeypatch.setattr(qualification, "_publish_events", mutate)
    with pytest.raises(DataReadinessError, match="changed"):
        _publish(fixture)
    assert not (fixture["output"] / "_manifest.json").exists()


@pytest.mark.parametrize("lineage", [
    "known_bad_issuer", "retained_query_copy", "future_query_copy", "early_unknown_same_cluster", "after_cutoff_query_copy",
])
def test_multisymbol_history_exclusion_uses_only_causal_issuer_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lineage: str,
) -> None:
    """Projection-unit test: admission gates supplied separately, no fake reviews."""
    monkeypatch.setattr(population, "_guard", lambda: None)
    output = tmp_path / "output"
    output.mkdir()
    with population._connect(tmp_path / "synthetic.sqlite") as connection:
        for security in ("issuer-a", "issuer-b"):
            row = {"source_family": "alpaca", "source_id": "multisymbol-story", "security_id": security,
                   "ticker": security.upper(), "source_version_sha256": "a" * 64,
                   "published_at_utc": "2020-06-01T20:00:00Z", "version_available_at_utc": "2020-06-01T20:00:00Z",
                   "event_available_at_utc": "2020-06-01T20:00:00Z", "identity_available_at_utc": "2020-01-01T00:00:00Z",
                   "identity_authority_sha256": "b" * 64, "source_metadata": {}, "unavailable_reasons": []}
            candidate = [{"event_family": "earnings", "fiscal_period": "Q2 2020", "unresolved_reasons": []}]
            cluster = json_sha256(["alpaca", "multisymbol-story", security])
            connection.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)",
                (security, cluster, security, "alpaca", 2020, population._json(row), "synthetic text", population._json(candidate)))
            if security == "issuer-b":
                retained = json.loads(json.dumps(row))
                retained["source_metadata"] = {"query_security_id": "legacy-query-b", "cohort_security_id": "issuer-b",
                    "query_identity_resolution": "bridged", "identity_bridge_row_sha256": "c" * 64}
                if lineage == "future_query_copy":
                    retained["identity_available_at_utc"] = retained["event_available_at_utc"] = "2020-06-03T20:00:00Z"
                elif lineage == "after_cutoff_query_copy":
                    retained["version_available_at_utc"] = retained["event_available_at_utc"] = "2026-01-01T20:00:00Z"
                connection.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                    (1, "alpaca", cluster, security, population._json(retained)))
                newer = json.loads(json.dumps(row))
        newer["security_id"] = "issuer-b" if lineage == "known_bad_issuer" else None
        newer["source_version_sha256"] = "d" * 64
        newer["version_available_at_utc"] = "2020-06-02T20:00:00Z"
        newer.pop("identity_available_at_utc")
        newer["source_metadata"] = {"query_security_id": "legacy-query-b"}
        if lineage == "early_unknown_same_cluster":
            newer["source_metadata"]["query_security_id"] = "issuer-b"
            newer["published_at_utc"] = newer["version_available_at_utc"] = "2020-05-01T20:00:00Z"
        unknown_cluster = json_sha256(["alpaca", "multisymbol-story",
                                      newer["security_id"] or newer["source_metadata"]["query_security_id"]])
        connection.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)",
            ("newer-b", unknown_cluster, newer["security_id"], "alpaca", 2020, population._json(newer), None, "[]"))
        connection.commit()
        metrics = {"families": {"earnings": {"qualified_for_historical_feature_use": True}}}
        report = qualification._publish_events(connection, output, "e" * 64, metrics, {})
    events = pd.read_parquet(output / "events.parquet")
    dispositions = [json.loads(line) for line in (output / "version_dispositions.jsonl").read_text().splitlines()]
    unknown = next(row for row in dispositions if row["version_id"] == "newer-b")
    assert not unknown["projected"]
    if lineage in ("future_query_copy", "early_unknown_same_cluster", "after_cutoff_query_copy"):
        assert "ambiguous_source_event_issuer_lineage" in unknown["reasons"]
        assert unknown["source"]["security_id"] is None
    if lineage == "after_cutoff_query_copy":
        assert unknown["possible_history_issuers"] == ["issuer-a", "issuer-b"]
        assert events.empty
    else:
        assert events.security_id.tolist() == ["issuer-a"]
        assert events.qualification_status.tolist() == ["qualified"]
        assert unknown["possible_history_issuers"] == ["issuer-b"]
        assert {item["security_id"] for item in report["excluded_histories"]} == {"issuer-b"}


def test_lease_is_acquired_before_input_reads(fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def busy(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["runtime_dir"].is_absolute()
        raise HeavyJobBusyError("synthetic active source worker")
        yield

    monkeypatch.setattr(qualification, "heavy_job_lease", busy)
    monkeypatch.setattr(qualification, "_parent", lambda *args: pytest.fail("read before lease"))
    with pytest.raises(HeavyJobBusyError):
        _publish(fixture)


def test_memory_pressure_at_projection_batch_prevents_completed_authority(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = qualification._guard_batch

    def pressure(position: int) -> None:
        original(position)
        if position == 256 and (fixture["output"] / "events.parquet").exists():
            raise MemoryBudgetError("synthetic resource pressure during the second projection batch")

    monkeypatch.setattr(qualification, "_guard_batch", pressure)
    with pytest.raises(MemoryBudgetError, match="second projection batch"):
        _publish(fixture)
    assert not (fixture["output"] / "_manifest.json").exists()
    assert len((fixture["output"] / "version_dispositions.jsonl").read_text().splitlines()) == 256
