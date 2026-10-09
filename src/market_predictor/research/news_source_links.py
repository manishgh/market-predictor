"""Pinned retained-source company associations, never world-actor or model proof.

Original direct-issuer relations are independent of sentiment. SEC associations
bind individual document receipts and causal filing identities. Coverage remains
unknown. Slices retain full known revisions; unrepresentable clocks are explicit
uncertainties requiring caller abstention, never invented availability timestamps.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.catalysts.issuer_events import attribution as original_attribution
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.catalysts.sec_filings.document_collection import RECEIPT_COLUMNS, TERMINAL_STATES
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import issuer_content_review_sources as sec_sources
from market_predictor.research import news_corpus_enrichment as corpus
from market_predictor.research import news_decision_features as features
from market_predictor.research import news_runtime_memory as runtime
from market_predictor.research import news_sentiment_reuse as saved
from market_predictor.research.legacy_query_identity_proofs import load_legacy_query_proofs
from market_predictor.swing.contracts.holding_materialization import SourcePin as HoldingSourcePin
from market_predictor.universe.issuer_news_identity import BRIDGE_COLUMNS, map_news_relations
from market_predictor.universe.legacy_query_identity import map_legacy_query_relations

SCHEMA = "market_predictor.news_source_links"
CLOSED = {
    "training_eligible": False,
    "serving_eligible": False,
    "promotion_eligible": False,
    "world_actor_verified": False,
    "source_coverage_admitted": False,
}
IMPLEMENTATION_PATHS = tuple(
    sorted(
        {
            *saved.IMPLEMENTATION_PATHS,
            *corpus._IMPLEMENTATION_PATHS,
            "research/news_source_links.py",
            "research/news_runtime_memory.py",
            "research/news_decision_features.py",
            "research/issuer_content_review_sources.py",
            "research/legacy_query_identity_proofs.py",
            "universe/issuer_news_identity.py",
            "universe/legacy_query_identity.py",
            "catalysts/sec_filings/document_collection.py",
            "catalysts/sec_filings/acceptance_clock.py",
            "canonical/store.py",
            "canonical/normalize.py",
            "evidence/hashing.py",
            "core/symbols.py",
            "swing/datasets/symbol_corrections.py",
            "swing/contracts/holding_materialization.py",
            "catalysts/issuer_events/attribution.py",
        }
    )
)
MAX_METADATA_BYTES = 4 * 1024**2
MAX_SLICE_VERSIONS = 100_000


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class NewsLinkArchive(_Strict):
    name: Literal["early", "later", "corrected"]
    collection_manifest: SourcePin
    attribution_manifest: SourcePin


class NewsSourceLinkPolicy(_Strict):
    schema_name: Literal["market_predictor.news_source_links_config"] = Field(alias="schema")
    corpus_manifest: SourcePin
    population_manifest: SourcePin
    population_database: SourcePin
    identity_manifest: SourcePin
    archives: tuple[NewsLinkArchive, ...]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _unsupported(value: object) -> None:
    raise TypeError(f"unsupported source-link value: {type(value).__name__}")


def _json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: item.isoformat() if isinstance(item, datetime) else _unsupported(item),
    )


def _metadata(raw: str | bytes) -> dict[str, Any]:
    _require(len(raw) <= MAX_METADATA_BYTES, "source-link metadata exceeds bounded size")
    return dict(parse_strict_json_object(raw, label="source-link metadata"))


def _clock(value: Any) -> datetime:
    return cast(datetime, saved._clock(value).to_pydatetime()).astimezone(UTC)


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): runtime.file_sha256(package / name) for name in IMPLEMENTATION_PATHS}


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    db = sqlite3.connect(path.as_uri() + "?mode=ro" if readonly else str(path), uri=readonly)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA temp_store=FILE")
    if readonly:
        db.execute("PRAGMA query_only=ON")
    return db


def _schema(db: sqlite3.Connection) -> None:
    db.executescript("""
      CREATE TABLE events(archive TEXT,event_id TEXT,security_id TEXT,ticker TEXT,raw_sha TEXT,
        payload TEXT,proof TEXT,PRIMARY KEY(archive,event_id,security_id,ticker,raw_sha));
      CREATE TABLE relations(archive TEXT,event_id TEXT,security_id TEXT,ticker TEXT,raw_sha TEXT,payload TEXT,proof TEXT);
      CREATE INDEX relation_lookup ON relations(archive,event_id,security_id,ticker,raw_sha);
      CREATE TABLE sec_receipts(unit TEXT,raw_sha TEXT,security_id TEXT,payload TEXT,proof TEXT);
      CREATE INDEX sec_lookup ON sec_receipts(unit,raw_sha,security_id);
      CREATE TABLE enriched(version_id TEXT PRIMARY KEY,payload TEXT,metadata TEXT);
      CREATE TABLE versions(copy_id TEXT PRIMARY KEY,version_id TEXT,source TEXT,document TEXT,
        known TEXT,payload TEXT,uncertainty TEXT);
      CREATE INDEX version_story ON versions(source,document,known);
      CREATE TABLE links(copy_id TEXT,security_id TEXT,available TEXT,payload TEXT,
        PRIMARY KEY(copy_id,security_id,payload));
      CREATE INDEX link_stock ON links(security_id,available,copy_id);
      CREATE TABLE candidates(security_id TEXT,source TEXT,document TEXT,available TEXT,
        PRIMARY KEY(security_id,source,document));
      CREATE TABLE proofs(sha256 TEXT PRIMARY KEY,payload TEXT);
    """)


def _pin(root: Path, pin: SourcePin, files: dict[str, str]) -> Path:
    corpus._no_links(root / pin.path)
    return runtime.pin_file(root, pin.path, pin.sha256, files)


def _proof(db: sqlite3.Connection, value: dict[str, Any]) -> str:
    digest = json_sha256(value)
    db.execute("INSERT OR IGNORE INTO proofs VALUES (?,?)", (digest, _json(value)))
    return digest


def _bookends(root: Path, policy: NewsSourceLinkPolicy, files: dict[str, str]) -> tuple[Path, dict[str, Any], dict[str, Any], Path]:
    population_path = _pin(root, policy.population_manifest, files)
    parent = saved._object(population_path)
    _require(
        parent.get("schema") == "market_predictor.issuer_content_review_population"
        and parent.get("status") == "complete_review_population_only",
        "population is not complete source evidence",
    )
    artifacts = parent["artifacts"]
    database = _pin(root, policy.population_database, files)
    _require(
        database == population_path.parent / "population.sqlite"
        and artifacts.get("population.sqlite") == policy.population_database.sha256,
        "population database is not bound to its original manifest",
    )
    request_path = runtime.pin_file(root, str(population_path.parent / "_request.json"), parent["request_sha256"], files)
    _require(artifacts.get("_request.json") == parent["request_sha256"], "population request artifact differs")
    settings = saved._object(request_path)["settings"]
    _require(
        settings["identity_manifest"] == policy.identity_manifest.model_dump(mode="json"), "identity authority differs from population"
    )
    # Raw pages, SEC bodies and identities remain original evidence. Historical
    # producer implementation maps are provenance, not current executed code.
    for name, digest in parent["source_files"].items():
        runtime.pin_file(root, name, digest, files)
    corpus_path = _pin(root, policy.corpus_manifest, files)
    manifest = saved._object(corpus_path)
    _require(
        manifest.get("status") == "complete_feature_cues_only"
        and all(manifest.get(key) is False for key in ("training_eligible", "serving_eligible", "promotion_eligible")),
        "corpus is not closed feature-only evidence",
    )
    request = saved._object(runtime.pin_file(root, str(corpus_path.parent / "_request.json"), manifest["request_sha256"], files))
    _require(request["input_database"] == policy.population_database.model_dump(mode="json"), "corpus source database differs")
    for name, digest in request["source_files"].items():
        runtime.pin_file(root, name, digest, files)
    for name, digest in manifest["artifacts"].items():
        runtime.pin_file(root, str(inside(corpus_path.parent, name)), digest, files)
    return database, settings, manifest, corpus_path.parent


def _identities(
    root: Path, policy: NewsSourceLinkPolicy, settings: dict[str, Any], files: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = _pin(root, policy.identity_manifest, files)
    manifest = saved._object(path)
    _require(manifest.get("schema") == "market_predictor.issuer_news_identity_alignment", "identity manifest schema differs")
    bridge_path = path.parent / "identity_bridge.parquet"
    for item in (bridge_path, Path(str(bridge_path) + ".manifest.json")):
        name = item.relative_to(root).as_posix()
        runtime.pin_file(root, name, manifest["source_files"][name], files)
    sidecar = saved._object(Path(str(bridge_path) + ".manifest.json"))
    _require(
        sidecar["artifact_sha256"] == files[bridge_path.relative_to(root).as_posix()]
        and sidecar["artifact_type"] == "issuer_news_identity_bridge",
        "identity bridge child differs",
    )
    bridge = pd.read_parquet(bridge_path, columns=list(BRIDGE_COLUMNS))
    _require(len(bridge) == sidecar["rows"], "identity bridge count differs")
    legacy, _ = load_legacy_query_proofs(root, settings["legacy_identity_proofs"], files)
    return bridge, legacy


def _child(root: Path, item: dict[str, Any], files: dict[str, str], inputs: dict[str, str]) -> Path:
    adapted = {**item, "manifest_path": item.get("manifest_path", str(item["path"]) + ".manifest.json")}
    return _original_artifact(root, adapted, files, inputs)


def _original_request(root: Path, manifest_path: Path, manifest: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    """Original request validation with new runtime I/O, unchanged semantic hash."""
    path = manifest_path.parent / "_request.json"
    runtime.capture_file(root, path, files)
    request = saved._object(path)
    digest = request.pop("request_sha256", None)
    _require(
        digest == manifest["request_sha256"] and hashlib.sha256(saved._json(request).encode()).hexdigest() == digest,
        "original sentiment request hash differs",
    )
    return request


def _original_artifact(root: Path, item: dict[str, Any], files: dict[str, str], inputs: dict[str, str]) -> Path:
    """Retain every original child/sidecar check without its old resource policy."""
    path = runtime.pin_file(root, str(item["path"]), str(item["sha256"]), files)
    sidecar_path = inside(root, str(item["manifest_path"]))
    _require(sidecar_path == Path(str(path) + ".manifest.json"), "unexpected original sidecar location")
    runtime.capture_file(root, sidecar_path, files)
    sidecar = saved._object(sidecar_path)
    _require(
        sidecar.get("schema") == "market_data.artifact_manifest.v1"
        and sidecar.get("artifact_sha256") == item["sha256"]
        and sidecar.get("rows") == item["rows"],
        "original sentiment child sidecar differs",
    )
    _require(
        all(sidecar.get("inputs", {}).get(key) == value for key, value in inputs.items()), "original sentiment child input binding differs"
    )
    runtime.guard()
    _require(cast(Any, pq).ParquetFile(path).metadata.num_rows == item["rows"], "original sentiment child row count differs")
    return path


def _batches(rows: Iterator[dict[str, Any]], size: int = 256) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def _load_alpaca(
    root: Path, archive: NewsLinkArchive, db: sqlite3.Connection, bridge: pd.DataFrame, legacy: pd.DataFrame, files: dict[str, str]
) -> None:
    collection_path = _pin(root, archive.collection_manifest, files)
    relation_path = _pin(root, archive.attribution_manifest, files)
    collection, attribution = saved._object(collection_path), saved._object(relation_path)
    _require(
        collection.get("schema") == "swing.alpaca_news_history_manifest.v1"
        and attribution.get("schema") == "swing.event_attribution_manifest.v1"
        and attribution.get("status") == "complete",
        "original source/attribution manifest differs",
    )
    _original_request(root, collection_path, collection, files)
    request = _original_request(root, relation_path, attribution, files)
    _require(
        request["collection_manifest_sha256"] == archive.collection_manifest.sha256
        and request["collection_request_sha256"] == collection["request_sha256"]
        and inside(root, request["collection_manifest_path"]) == collection_path,
        "attribution is not bound to original collection",
    )
    for stem in ("security_identities", "business_labels", "collection_audit"):
        runtime.pin_file(root, request[stem + "_path"], request[stem + "_sha256"], files)
    identities = original_attribution._prepare_identities(
        pd.read_parquet(inside(root, request["security_identities_path"]), columns=sorted(original_attribution._IDENTITY_REQUIRED))
    )
    registry_sha = original_attribution._identity_registry_sha256(identities)
    identities_by_security = original_attribution._identities_by_security(identities)
    events = saved._artifacts(collection)
    for chunk, item in sorted(saved._artifacts(attribution).items()):
        runtime.guard()
        _require(chunk in events, "relation child lacks original source events")
        original = events[chunk]
        if saved._clock(original["start_utc"]) > saved.END:
            continue
        _require(item["source_event_sha256"] == original["sha256"], "attribution raw event artifact differs")
        event_path = _child(root, original, files, {"chunk_id": chunk, "collection_request_sha256": collection["request_sha256"]})
        rel_path = _child(
            root,
            item,
            files,
            {
                "chunk_id": chunk,
                "event_attribution_request_sha256": attribution["request_sha256"],
                "source_event_artifact_sha256": original["sha256"],
                "security_identities_sha256": request["security_identities_sha256"],
                "business_labels_sha256": request["business_labels_sha256"],
                "attribution_policy_sha256": request["attribution_policy_sha256"],
            },
        )
        event_columns = tuple(name for name in saved.EVENT_COLUMNS if name not in {"title", "summary", "text"})
        for event in runtime.parquet_rows(event_path, event_columns):
            if saved._clock(event["available_at_utc"]) > saved.END or saved._version_clock(event) > saved.END:
                continue
            payload = {
                key: event[key]
                for key in (
                    "event_id",
                    "security_id",
                    "ticker",
                    "source_family",
                    "raw_sha256",
                    "published_at_utc",
                    "available_at_utc",
                    "provider_updated_at_utc",
                )
            }
            payload["version_available_at_utc"] = saved._version_clock(event)
            proof = {"artifact_sha256": original["sha256"], "collection": archive.collection_manifest.model_dump(mode="json")}
            key = (archive.name, event["event_id"], event["security_id"], event["ticker"], event["raw_sha256"])
            previous = db.execute(
                "SELECT payload FROM events WHERE archive=? AND event_id=? AND security_id=? AND ticker=? AND raw_sha=?", key
            ).fetchone()
            _require(previous is None or previous[0] == _json(payload), "conflicting original event copies")
            db.execute("INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?,?)", (*key, _json(payload), _json(proof)))
        sidecar = saved._object(Path(str(rel_path) + ".manifest.json"))
        for batch in _batches(runtime.parquet_rows(rel_path, sidecar["columns"])):
            direct = [row for row in batch if row["relation_channel"] == "direct_issuer"]
            if not direct:
                continue
            frame = pd.DataFrame(direct)
            mapped = map_legacy_query_relations(map_news_relations(frame, bridge), legacy) if archive.name != "corrected" else frame
            for relation in mapped.to_dict("records"):
                if saved._clock(relation["event_feature_available_at_utc"]) > saved.END:
                    continue
                _require(
                    relation["attribution_policy_sha256"] == request["attribution_policy_sha256"]
                    and relation["security_identity_registry_sha256"] == registry_sha,
                    "relation policy/identity artifact differs from original request",
                )
                original_target = relation.get("identity_original_target_security_id", relation["target_security_id"])
                original_ticker = relation.get("identity_original_target_ticker", relation["target_ticker"])
                identity = original_attribution._active_identity(
                    identities_by_security.get(original_target, ()), saved._clock(relation["event_feature_available_at_utc"])
                )
                if identity is None:
                    continue
                _require(
                    identity.ticker == original_ticker
                    and relation["identity_available_at_utc"] is not None
                    and _clock(relation["identity_available_at_utc"]) >= _clock(identity.available_at_utc),
                    "relation target identity or causal availability differs",
                )
                found = db.execute(
                    "SELECT raw_sha,payload FROM events WHERE archive=? AND event_id=? AND security_id=? AND ticker=?",
                    (archive.name, relation["event_id"], relation["source_security_id"], relation["source_ticker"]),
                ).fetchall()
                _require(len(found) == 1, "direct relation lacks a unique original event/query binding")
                event = _metadata(found[0]["payload"])
                _require(
                    _clock(relation["event_feature_available_at_utc"]) == _clock(event["available_at_utc"]),
                    "relation event clock differs from original event",
                )
                proof = {
                    "relation_artifact_sha256": item["sha256"],
                    "source_event_sha256": original["sha256"],
                    "relation_row_sha256": json_sha256(json.loads(_json(relation))),
                    "attribution_manifest": archive.attribution_manifest.model_dump(mode="json"),
                }
                db.execute(
                    "INSERT INTO relations VALUES (?,?,?,?,?,?,?)",
                    (
                        archive.name,
                        relation["event_id"],
                        relation["source_security_id"],
                        relation["source_ticker"],
                        found[0]["raw_sha"],
                        _json(relation),
                        _json(proof),
                    ),
                )


def _load_sec(root: Path, settings: dict[str, Any], db: sqlite3.Connection, files: dict[str, str]) -> None:
    archive = HoldingSourcePin.model_validate(settings["sec_archive"])
    inventory = HoldingSourcePin.model_validate(settings["sec_inventory"])
    directory, manifest, _, parent = sec_sources._sec_metadata(root, archive, inventory, files)
    filings, clock_pin = sec_sources._filing_map(root, inventory, parent, files, runtime.guard)
    terminal: set[str] = set()
    for name, shard in sorted(manifest["shards"].items()):
        path = runtime.pin_file(root, str(directory / "shards" / f"{name}.parquet"), shard["parquet_sha256"], files)
        count, zip_checked = 0, False
        for row in sec_sources._rows(path, runtime.guard):
            count += 1
            clean = {
                key: int(row[key]) if row[key] is not None and key in {"attempt", "status_code", "body_length"} else row[key]
                for key in RECEIPT_COLUMNS
                if key != "receipt_sha256"
            }
            _require(row["shard"] == name and json_sha256(clean) == row["receipt_sha256"], "SEC receipt hash differs")
            if row["phase"] != "document":
                continue
            _require(row["unit_id"] not in terminal, "SEC document has attempts after final receipt")
            if row["state"] not in TERMINAL_STATES:
                continue
            terminal.add(row["unit_id"])
            _require(row["accession_number"] in filings, "SEC receipt escapes selected inventory")
            filer_ciks = {str(filing["sec_cik"]).zfill(10) for filing in filings[row["accession_number"]]}
            _require(str(row["sec_cik"]).zfill(10) in filer_ciks, "SEC receipt retrieval CIK is outside the proven filer set")
            if row["member"] is not None and not zip_checked:
                runtime.pin_file(root, str(directory / "shards" / f"{name}.zip"), shard["zip_sha256"], files)
                zip_checked = True
            for filing in filings[row["accession_number"]]:
                proof = {
                    "archive": settings["sec_archive"],
                    "receipt_sha256": row["receipt_sha256"],
                    "receipt_artifact_sha256": shard["parquet_sha256"],
                    "acceptance_clock_sha256": clock_pin,
                    "identity_authority_sha256": filing["identity_authority_sha256"],
                    "filing_sec_cik": str(filing["sec_cik"]).zfill(10),
                    "retrieval_sec_cik": str(row["sec_cik"]).zfill(10),
                }
                db.execute(
                    "INSERT INTO sec_receipts VALUES (?,?,?,?,?)",
                    (row["unit_id"], row["body_sha256"], filing["security_id"], _json({"receipt": row, "filing": filing}), _json(proof)),
                )
        _require(count == shard["attempts"], "SEC receipt shard row count differs")
    expected = sum(value for key, value in manifest["totals"]["units_by_phase_and_state"].items() if key.startswith("document/"))
    _require(len(terminal) == expected, "SEC final document receipt inventory differs")


def _load_corpus(directory: Path, manifest: dict[str, Any], source: sqlite3.Connection, db: sqlite3.Connection) -> None:
    count = 0
    for name in sorted(manifest["artifacts"]):
        if not name.startswith("part-") or not name.endswith(".jsonl"):
            continue
        for record in corpus._read_lines(directory / name):
            if count % 256 == 0:
                runtime.guard()
            row = source.execute(
                "SELECT cluster_id,security_id,source_family,content_json FROM versions WHERE version_id=?", (record["version_id"],)
            ).fetchone()
            _require(row is not None, "corpus version is absent from pinned population")
            assert row is not None
            _require(
                record["cluster_id"] == row["cluster_id"]
                and record["retained_security_id"] == row["security_id"]
                and record["source_family"] == row["source_family"],
                "corpus/population identity differs",
            )
            if "content_json_sha256" in record:
                _require(
                    hashlib.sha256(row["content_json"].encode()).hexdigest() == record["content_json_sha256"],
                    "corpus metadata hash differs from original population",
                )
            metadata = _metadata(row["content_json"])
            reduced = {key: record[key] for key in ("version_id", "disposition")}
            if record["disposition"] == "enriched":
                enrichment = record["enrichment"]
                _require(
                    enrichment["version_ref"] == record["version_id"]
                    and enrichment["source"] == row["source_family"]
                    and enrichment["document_ref"] == f"{row['source_family']}:{metadata['source_id']}"
                    and enrichment["method"] == "rule_cues",
                    "corpus cue source reference differs",
                )
                reduced["enrichment"] = {
                    key: enrichment[key] for key in ("published_at_utc", "available_at_utc", "categories", "truncation_reasons")
                }
                reduced["enrichment"]["cues"] = [
                    {key: cue[key] for key in ("category", "business_direction", "status")} for cue in enrichment["cues"]
                ]
                sentiment = record["sentiment"]
                reduced["sentiment"] = {key: sentiment.get(key) for key in ("status", "score", "available_at_utc")}
            db.execute("INSERT INTO enriched VALUES (?,?,?)", (record["version_id"], _json(reduced), _json(metadata)))
            count += 1
    _require(
        count == manifest["source_versions"] == source.execute("SELECT COUNT(*) FROM versions").fetchone()[0],
        "corpus/population version inventory differs",
    )


def _alpaca_links(metadata: dict[str, Any], db: sqlite3.Connection) -> list[tuple[str, datetime, datetime, str]]:
    original = metadata["source_metadata"]
    key = (
        original.get("archive"),
        original.get("event_id"),
        original.get("query_security_id"),
        original.get("query_ticker"),
        metadata.get("source_version_sha256"),
    )
    if any(not isinstance(item, str) or not item for item in key) or original.get("raw_sha256") != key[-1]:
        return []
    rows = db.execute(
        "SELECT payload,proof FROM relations WHERE archive=? AND event_id=? AND security_id=? AND ticker=? AND raw_sha=?", key
    )
    result = []
    for row in rows:
        relation = _metadata(row["payload"])
        event = db.execute(
            "SELECT payload FROM events WHERE archive=? AND event_id=? AND security_id=? AND ticker=? AND raw_sha=?", key
        ).fetchone()
        _require(event is not None, "bound relation lost original source")
        event_value = _metadata(event[0])
        _require(
            _clock(metadata["published_at_utc"]) == _clock(event_value["published_at_utc"])
            and _clock(metadata["version_available_at_utc"]) == _clock(event_value["version_available_at_utc"]),
            "population/original event clocks differ",
        )
        # Query identity binds the source event, not a whitelist of targets. Each
        # target needs its own original causal identity and canonical translation.
        translated = relation.get("identity_translation_status") in {"mapped", "legacy_proven"}
        identity_equal = original.get("query_identity_resolution") == "identity_equal" and relation["target_security_id"] == metadata.get(
            "security_id"
        )
        if not (translated or original["archive"] == "corrected" or identity_equal):
            continue
        identity = relation.get("identity_available_at_utc")
        if identity is None:
            continue
        result.append(
            (
                relation["target_security_id"],
                _clock(relation["feature_available_at_utc"]),
                _clock(identity),
                _proof(
                    db,
                    {
                        "relation": json.loads(row["proof"]),
                        "identity_bridge": relation.get("identity_bridge_row_sha256"),
                        "legacy_identity": relation.get("identity_legacy_proof_row_sha256"),
                    },
                ),
            )
        )
    return result


def _sec_links(metadata: dict[str, Any], db: sqlite3.Connection) -> list[tuple[str, datetime, datetime, str]]:
    original = metadata["source_metadata"]
    rows = db.execute(
        "SELECT payload,proof FROM sec_receipts WHERE unit=? AND raw_sha IS ? AND security_id=?",
        (metadata["source_id"], metadata.get("source_version_sha256"), metadata.get("security_id")),
    ).fetchall()
    _require(len(rows) == 1, "SEC population document does not bind one exact original receipt")
    value = _metadata(rows[0]["payload"])
    receipt, filing = value["receipt"], value["filing"]
    _require(
        original["receipt_sha256"] == receipt["receipt_sha256"]
        and original["body_sha256"] == metadata.get("source_version_sha256")
        and str(original["sec_cik"]).zfill(10) == str(filing["sec_cik"]).zfill(10)
        and str(original["retrieval_sec_cik"]).zfill(10) == str(receipt["sec_cik"]).zfill(10)
        and original["identity_authority_sha256"] == filing["identity_authority_sha256"]
        and _clock(metadata["published_at_utc"]) == _clock(filing["accepted_at_utc"])
        and _clock(metadata["version_available_at_utc"]) == _clock(filing["available_at_utc"]),
        "SEC population receipt/filing identity or clocks differ",
    )
    if not filing["ticker"] or filing["identity_available_at_utc"] is None:
        return []
    return [
        (
            filing["security_id"],
            _clock(filing["available_at_utc"]),
            _clock(filing["identity_available_at_utc"]),
            _proof(db, json.loads(rows[0]["proof"])),
        )
    ]


def _version(copy_id: str, record: dict[str, Any], metadata: dict[str, Any], proof: str) -> features.NewsVersion:
    known = _clock(metadata["version_available_at_utc"])
    published = _clock(metadata["published_at_utc"])
    semantics = metadata["availability_semantics"]
    _require(semantics in ("observed", "historical_proxy"), "unknown retained availability semantics")
    if semantics == "observed":
        known = max(known, _clock(metadata["first_seen_at_utc"]))
    usable = record["disposition"] == "enriched"
    categories: tuple[str, ...] = ()
    cues: tuple[features.DecisionCue, ...] = ()
    cue_clock = None
    score = score_clock = None
    truncated = False
    if usable:
        enrichment = record["enrichment"]
        try:
            cue_clock = max(
                known,
                _clock(metadata["event_available_at_utc"]),
                _clock(metadata["identity_available_at_utc"]),
                _clock(enrichment["available_at_utc"]),
            )
        except (KeyError, ValueError, TypeError, DataReadinessError):
            usable = False
        if usable:
            categories = tuple("other" if name == "other_unresolved" else name for name in enrichment["categories"])
            cues = tuple(
                features.DecisionCue(
                    "other" if cue["category"] == "other_unresolved" else cue["category"], cue["business_direction"], cue["status"]
                )
                for cue in enrichment["cues"]
            )
            truncated = bool(enrichment["truncation_reasons"])
            sentiment = record["sentiment"]
            if sentiment["status"] == "matched":
                score = sentiment["score"]
                score_clock = max(known, _clock(sentiment["available_at_utc"]))
    if not usable:
        cue_clock = None
    return features.NewsVersion(
        copy_id,
        metadata["source_family"],
        metadata["source_id"],
        metadata["source_version_sha256"],
        proof,
        published,
        known,
        cue_clock,
        usable,
        categories,
        cues,
        truncated,
        score,
        score_clock,
        semantics,
        proof,
        known,
    )


def _materialize_versions(
    source: sqlite3.Connection, db: sqlite3.Connection, population_sha: str, progress: Callable[[dict[str, Any]], None] | None
) -> Counter[str]:
    counts: Counter[str] = Counter()
    cursor = source.execute(
        "SELECT phase,ordinal,version_id,record_json FROM records WHERE phase IN ('alpaca','document') ORDER BY phase,ordinal"
    )
    for row in cursor:
        if counts["query_copy_records"] % 256 == 0:
            runtime.guard()
        counts["query_copy_records"] += 1
        metadata = _metadata(row["record_json"])
        found = db.execute("SELECT payload,metadata FROM enriched WHERE version_id=?", (row["version_id"],)).fetchone()
        _require(found is not None, "original query record has no corpus version")
        primary = _metadata(found["metadata"])
        _require(
            all(
                metadata.get(key) == primary.get(key)
                for key in ("source_family", "source_id", "source_version_sha256", "security_id", "text_sha256")
            ),
            "query copy differs from its population version",
        )
        copy_id = json_sha256([row["version_id"], row["phase"], row["ordinal"]])
        proof = _proof(
            db,
            {
                "population_database_sha256": population_sha,
                "phase": row["phase"],
                "ordinal": row["ordinal"],
                "record_sha256": hashlib.sha256(row["record_json"].encode()).hexdigest(),
            },
        )
        associated = _alpaca_links(metadata, db) if row["phase"] == "alpaca" else _sec_links(metadata, db)
        if not associated:
            counts["query_copies_without_verified_company_association"] += 1
        payload, known, uncertainty = None, None, None
        try:
            version = _version(copy_id, _metadata(found["payload"]), metadata, proof)
            payload, known = _json(asdict(version)), version.known_at_utc.isoformat()
        except (KeyError, TypeError, ValueError, DataReadinessError) as error:
            uncertainty = f"unrepresentable_source_version:{type(error).__name__}"
            try:
                known = _clock(metadata["version_available_at_utc"]).isoformat()
            except (KeyError, TypeError, ValueError, DataReadinessError):
                pass
        db.execute(
            "INSERT INTO versions VALUES (?,?,?,?,?,?,?)",
            (copy_id, row["version_id"], metadata["source_family"], metadata["source_id"], known, payload, uncertainty),
        )
        for security, relation_clock, identity_clock, relation_proof in associated:
            link_available = max(relation_clock, identity_clock).isoformat()
            db.execute(
                "INSERT INTO candidates VALUES (?,?,?,?) ON CONFLICT(security_id,source,document) "
                "DO UPDATE SET available=MIN(available,excluded.available)",
                (security, metadata["source_family"], metadata["source_id"], link_available),
            )
            if payload is not None:
                link = features.VerifiedCompanyLink(
                    copy_id, security, relation_clock, identity_clock, relation_proof, metadata["availability_semantics"]
                )
                db.execute(
                    "INSERT OR IGNORE INTO links VALUES (?,?,?,?)",
                    (copy_id, security, max(relation_clock, identity_clock).isoformat(), _json(asdict(link))),
                )
                counts["verified_company_links"] += 1
        counts["unrepresentable_versions" if uncertainty else "kernel_version_copies"] += 1
        if progress is not None and counts["query_copy_records"] % 5000 == 0:
            db.commit()
            progress(dict(counts))
    _require(
        db.execute("SELECT COUNT(DISTINCT version_id) FROM versions").fetchone()[0]
        == db.execute("SELECT COUNT(*) FROM enriched").fetchone()[0],
        "retained query copies do not cover every corpus version",
    )
    return counts


def build_news_source_link_index(
    *, root: Path, config: SourcePin, output: Path, progress: Callable[[dict[str, Any]], None] | None = None
) -> dict[str, Any]:
    """Build a new immutable local bridge under one canonical resource lease."""
    root = root.resolve()
    output = inside(root, output)
    stage = output.with_name(output.name + ".partial")
    corpus._no_links(output)
    corpus._no_links(stage)
    runtime_dir = heavy_job_runtime_dir()
    runtime_dir = runtime_dir if runtime_dir.is_absolute() else root / runtime_dir
    with heavy_job_lease("news-source-link-index", runtime_dir=runtime_dir):
        runtime.guard()
        _require(not output.exists() and not stage.exists(), "source-link output or partial stage already exists")
        files: dict[str, str] = {}
        policy = NewsSourceLinkPolicy.model_validate_json(_pin(root, config, files).read_bytes())
        _require(
            len(policy.archives) == 3 and {item.name for item in policy.archives} == {"early", "later", "corrected"},
            "source-link policy requires all three original archives exactly once",
        )
        implementation = _implementation(root)
        database, settings, manifest, directory = _bookends(root, policy, files)
        bridge, legacy = _identities(root, policy, settings, files)
        stage.mkdir(parents=True)
        request = {
            "schema": SCHEMA + "_request",
            "config": config.model_dump(mode="json"),
            "policy": policy.model_dump(mode="json", by_alias=True),
            "implementation_files": implementation,
            **CLOSED,
        }
        write_json_object(stage / "_request.json", request)
        request_sha = runtime.file_sha256(stage / "_request.json")
        with closing(_connect(stage / "index.sqlite")) as db, closing(_connect(database, readonly=True)) as source:
            _schema(db)
            for archive in policy.archives:
                _load_alpaca(root, archive, db, bridge, legacy, files)
            _load_sec(root, settings, db, files)
            _load_corpus(directory, manifest, source, db)
            counts = _materialize_versions(source, db, policy.population_database.sha256, progress)
            for name in ("events", "relations", "sec_receipts", "enriched"):
                db.execute(f"DROP TABLE {name}")
            db.commit()
            db.execute("VACUUM")
        artifacts = {"index.sqlite": runtime.file_sha256(stage / "index.sqlite"), "_request.json": request_sha}
        runtime.recheck_files(root, files)
        runtime.recheck_files(root, implementation)
        _require(
            all(runtime.file_sha256(stage / name) == digest for name, digest in artifacts.items()),
            "source-link output changed before publication",
        )
        result = {
            "schema": SCHEMA,
            "status": "complete_source_links_only",
            "request_sha256": request_sha,
            "source_files": files,
            "implementation_files": implementation,
            "artifacts": artifacts,
            "counts": dict(counts),
            "coverage": "unknown",
            **CLOSED,
        }
        write_json_object(stage / "_manifest.json", result)
        stage.rename(output)
        return {**result, "manifest_sha256": runtime.file_sha256(output / "_manifest.json")}


@dataclass(frozen=True)
class SourceLinkUncertainty:
    """An unrepresentable retained revision, with honest temporal applicability.

    A missing original version clock cannot be assigned a date. It may affect any
    already associated history; the association proof still cannot act backwards.
    """

    copy_id: str
    source_family: str
    source_document_id: str
    known_at_utc: datetime | None
    association_available_at_utc: datetime
    reason: str

    def applies_at(self, cutoff: datetime) -> bool:
        instant = _clock(cutoff)
        return self.association_available_at_utc <= instant and (self.known_at_utc is None or self.known_at_utc <= instant)


@dataclass(frozen=True)
class SourceLinkedNewsSlice:
    versions: tuple[features.NewsVersion, ...]
    links: tuple[features.VerifiedCompanyLink, ...]
    uncertainties: tuple[SourceLinkUncertainty, ...]
    coverage: tuple[features.SourceCoverage, ...] = ()
    source_coverage_admitted: bool = False


def _decode_version(raw: str) -> features.NewsVersion:
    value = _metadata(raw)
    for key in ("published_at_utc", "known_at_utc", "cue_available_at_utc", "sentiment_available_at_utc", "copy_proof_available_at_utc"):
        if value[key] is not None:
            value[key] = _clock(value[key])
    value["categories"] = tuple(value["categories"])
    value["cues"] = tuple(features.DecisionCue(**cue) for cue in value["cues"])
    return features.NewsVersion(**value)


def _decode_link(raw: str) -> features.VerifiedCompanyLink:
    value = _metadata(raw)
    for key in ("relation_available_at_utc", "identity_available_at_utc"):
        value[key] = _clock(value[key])
    return features.VerifiedCompanyLink(**value)


@dataclass
class NewsSourceLinkReader:
    source_files: Mapping[str, str]
    implementation_files: Mapping[str, str]
    _db: sqlite3.Connection

    def proof(self, sha256: str) -> dict[str, Any]:
        row = self._db.execute("SELECT payload FROM proofs WHERE sha256=?", (sha256,)).fetchone()
        _require(row is not None, "source-link proof is not retained by this index")
        value = _metadata(row[0])
        _require(json_sha256(value) == sha256, "source-link proof payload differs")
        return value

    def for_decision(self, security_id: str, cutoff: datetime, *, max_versions: int = MAX_SLICE_VERSIONS) -> SourceLinkedNewsSlice:
        return self.for_period(security_id, cutoff, cutoff, max_versions=max_versions)

    def for_period(
        self, security_id: str, start_utc: datetime, end_utc: datetime, *, max_versions: int = MAX_SLICE_VERSIONS
    ) -> SourceLinkedNewsSlice:
        """Bounded full histories through end, reusable across decisions in a month.

        Start is scheduling metadata, deliberately not a revision-history filter.
        The pure kernel masks each decision's future versions and company links.
        """
        start, end = _clock(start_utc), _clock(end_utc)
        _require(start <= end and end <= saved.END.to_pydatetime(), "source-link period escapes initial-fit evidence")
        _require(type(max_versions) is int and 0 < max_versions <= MAX_SLICE_VERSIONS, "source-link slice limit differs")
        _require(isinstance(security_id, str) and bool(security_id.strip()), "security identity is required")
        rows = self._db.execute(
            """
          SELECT v.copy_id,v.source,v.document,v.known,v.payload,v.uncertainty,c.available FROM versions v JOIN candidates c
          ON c.source=v.source AND c.document=v.document WHERE c.security_id=? AND c.available<=?
          AND (v.known IS NULL OR v.known<=?) ORDER BY v.source,v.document,v.known,v.copy_id LIMIT ?
        """,
            (security_id, end.isoformat(), end.isoformat(), max_versions + 1),
        ).fetchall()
        _require(len(rows) <= max_versions, "source-link slice exceeds bound; never truncate histories")
        versions, links, uncertainties = [], [], []
        for row in rows:
            if row["uncertainty"]:
                uncertainties.append(
                    SourceLinkUncertainty(
                        row["copy_id"],
                        row["source"],
                        row["document"],
                        None if row["known"] is None else _clock(row["known"]),
                        _clock(row["available"]),
                        row["uncertainty"],
                    )
                )
                continue
            versions.append(_decode_version(row["payload"]))
            for link in self._db.execute(
                "SELECT payload FROM links WHERE copy_id=? AND security_id=? AND available<=?",
                (row["copy_id"], security_id, end.isoformat()),
            ):
                links.append(_decode_link(link["payload"]))
        return SourceLinkedNewsSlice(tuple(versions), tuple(links), tuple(uncertainties))


@contextmanager
def open_news_source_link_index(*, root: Path, index: SourcePin) -> Iterator[NewsSourceLinkReader]:
    """Read under the caller's lease; recheck consumed pins at both bookends."""
    root = root.resolve()
    files: dict[str, str] = {}
    path = _pin(root, index, files)
    manifest = saved._object(path)
    _require(
        manifest.get("schema") == SCHEMA
        and manifest.get("status") == "complete_source_links_only"
        and manifest.get("coverage") == "unknown"
        and all(manifest.get(key) is value for key, value in CLOSED.items()),
        "source-link manifest scope differs",
    )
    _require(manifest["implementation_files"] == _implementation(root), "source-link executed implementation differs")
    for name, digest in manifest["source_files"].items():
        runtime.pin_file(root, name, digest, files)
    for name, digest in manifest["artifacts"].items():
        runtime.pin_file(root, str(inside(path.parent, name)), digest, files)
    _require(
        set(manifest["artifacts"]) == {"index.sqlite", "_request.json"}
        and manifest["artifacts"]["_request.json"] == manifest["request_sha256"],
        "source-link output inventory differs",
    )
    request = saved._object(path.parent / "_request.json")
    policy = NewsSourceLinkPolicy.model_validate_json(_json(request["policy"]))
    config = SourcePin.model_validate(request["config"])
    config_path = _pin(root, config, files)
    _require(NewsSourceLinkPolicy.model_validate_json(config_path.read_bytes()) == policy, "source-link request/config policy differs")
    _require(
        request["implementation_files"] == manifest["implementation_files"]
        and all(request.get(key) is value for key, value in CLOSED.items()),
        "source-link request scope differs",
    )
    with closing(_connect(path.parent / "index.sqlite", readonly=True)) as db:
        try:
            yield NewsSourceLinkReader(files, manifest["implementation_files"], db)
        finally:
            runtime.recheck_files(root, files)
            runtime.recheck_files(root, manifest["implementation_files"])
