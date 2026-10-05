"""Leased, resumable source-only content review population; never feature admission.

SQLite bounds retained text memory. Each source announcement has one sampling unit
while all document/revision/query evidence stays linked to that unit. A complete
manifest is published only after source and implementation pins are rechecked.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterator
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.issuer_events.content_review import (
    EXTRACTION_POLICY_SHA256,
    IssuerContentContext,
    alpaca_content_evidence,
    extract_review_candidates,
    extract_source_text,
    sec_content_evidence,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, read_json_object, write_json_object
from market_predictor.governance.issuer_content_qualification import (
    POLICY_SHA256,
    Family,
    ReviewCluster,
    select_content_review_sample,
)
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.issuer_content_review_sources import (
    END,
    AliasProof,
    ReviewSourceRecord,
    extract_sec_alias_proofs,
    iter_alpaca_review_sources,
    iter_sec_review_sources,
    read_early_alias_proofs,
    recheck_source_pins,
)
from market_predictor.research.legacy_query_identity_proofs import load_legacy_query_proofs, pin_file
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.contracts.holding_materialization import SourcePin

SCHEMA = "market_predictor.issuer_content_review_population"
CONFIG_SCHEMA = f"{SCHEMA}_config"
COMMIT_ROWS = 5000
IMPLEMENTATION_PATHS = (
    "research/issuer_content_review_population.py", "research/issuer_content_review_sources.py",
    "catalysts/issuer_events/content_review.py", "catalysts/sec_filings/document_collection.py",
    "catalysts/sec_filings/acceptance_clock.py", "governance/issuer_content_qualification.py",
    "governance/issuer_event_precision/admission_authority.py", "research/legacy_query_identity_proofs.py",
    "canonical/store.py", "canonical/normalize.py", "evidence/hashing.py", "evidence/io.py",
    "core/json_integrity.py", "core/symbols.py", "core/system_memory.py", "heavy_jobs.py", "resources.py", "locking.py",
    "swing/contracts/holding_materialization.py", "swing/contracts/holding_accounting.py",
)
_CONFIG_KEYS = {"schema", "alpaca_inventory", "sec_archive", "sec_inventory", "early_alias_artifact",
                "early_alias_sidecar", "identity_manifest", "legacy_identity_proofs"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _guard() -> None:
    assert_memory_budget(stage="issuer content source review", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=0.75, maximum_used_percent=90.0)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False, default=str)


def _atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        write_json_object(temporary, value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS aliases (
          proof_id TEXT PRIMARY KEY, security_id TEXT NOT NULL, cik TEXT NOT NULL,
          available_ns INTEGER NOT NULL, proof_json TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records (
          ordinal INTEGER NOT NULL, phase TEXT NOT NULL, cluster_id TEXT NOT NULL,
          version_id TEXT NOT NULL, record_json TEXT NOT NULL, PRIMARY KEY(phase, ordinal));
        CREATE TABLE IF NOT EXISTS versions (
          version_id TEXT PRIMARY KEY, cluster_id TEXT NOT NULL, security_id TEXT,
          source_family TEXT NOT NULL, year INTEGER NOT NULL, content_json TEXT NOT NULL,
          text TEXT, candidates_json TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS version_cluster ON versions(cluster_id);
    """)
    return connection


def _store_alias(connection: sqlite3.Connection, security_id: str, proof: AliasProof) -> None:
    encoded = _json(asdict(proof))
    key = json_sha256([security_id, json.loads(encoded)])
    connection.execute("INSERT OR IGNORE INTO aliases VALUES (?,?,?,?,?)",
                       (key, security_id, proof.sec_cik, proof.available_at_utc.value, encoded))


class _Aliases:
    def __init__(self, connection: sqlite3.Connection) -> None:
        grouped: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for security, clock, encoded in connection.execute("SELECT security_id,available_ns,proof_json FROM aliases"):
            grouped[security][clock].append(json.loads(encoded))
        self.rows = {security: (sorted(values), values) for security, values in grouped.items()}

    def at(self, security: str, clock: pd.Timestamp) -> dict[str, Any] | None:
        if security not in self.rows:
            return None
        clocks, values = self.rows[security]
        index = bisect_right(clocks, clock.value) - 1
        if index < 0:
            return None
        proofs = values[clocks[index]]
        # Competing names at one clock are evidence of ambiguity, never a tie-break.
        if len({(proof["sec_cik"], tuple(proof["aliases"])) for proof in proofs}) != 1:
            return None
        return min(proofs, key=lambda proof: (proof["source_sha256"], proof["source_locator"]))


def _identity_clocks(root: Path, settings: dict[str, Any], pins: dict[str, str]) -> dict[str, pd.Timestamp]:
    # Historical evidence inspection only: this does not admit the bridge as a
    # current canonical artifact or reinterpret its saved schema declaration.
    alignment_path = pin_file(root, settings["identity_manifest"], pins)
    alignment = read_json_object(alignment_path)
    sources = alignment.get("source_files")
    if alignment.get("schema") != "market_predictor.issuer_news_identity_alignment" or not isinstance(sources, dict):
        raise DataReadinessError("identity alignment manifest differs")
    bridge_path = alignment_path.parent / "identity_bridge.parquet"
    sidecar_path = bridge_path.with_suffix(".parquet.manifest.json")
    for path in (bridge_path, sidecar_path):
        relative = path.relative_to(root).as_posix()
        _require(relative in sources, "historical identity bridge parent pin missing")
        pin_file(root, {"path": relative, "sha256": sources[relative]}, pins)
    sidecar = read_json_object(sidecar_path)
    _require(sidecar.get("artifact_sha256") == sources[bridge_path.relative_to(root).as_posix()]
             and sidecar.get("artifact_type") == "issuer_news_identity_bridge"
             and sidecar.get("production_ready") is False, "historical identity bridge sidecar differs")
    columns = ["bridge_row_sha256", "available_at_utc"]
    _require(isinstance(sidecar.get("columns"), list) and set(columns).issubset(sidecar["columns"]),
             "historical identity bridge columns differ")
    bridge = pd.read_parquet(bridge_path, columns=columns)
    _require(type(sidecar.get("rows")) is int and sidecar["rows"] == len(bridge),
             "historical identity bridge row count differs")
    legacy, _ = load_legacy_query_proofs(root, settings["legacy_identity_proofs"], pins)
    result: dict[str, pd.Timestamp] = {}
    for frame, proof_column in ((bridge, "bridge_row_sha256"), (legacy, "proof_row_sha256")):
        clocks = frame["available_at_utc"]
        _require(isinstance(clocks.dtype, pd.DatetimeTZDtype) and str(clocks.dtype.tz) == "UTC"
                 and not clocks.isna().any(), "historical identity clock must be typed nonnull UTC")
        for proof, clock in zip(frame[proof_column], clocks, strict=True):
            _require(isinstance(proof, str) and len(proof) == 64 and set(proof) <= set("0123456789abcdef"),
                     "historical identity row proof must be a SHA256")
            _require(proof not in result or result[proof] == clock, "historical identity proof has competing clocks")
            result[proof] = clock
    recheck_source_pins(root, pins)
    return result


def _source_text(record: ReviewSourceRecord) -> tuple[str | None, str | None, tuple[str, ...]]:
    if record.payload is None:
        return None, None, ()
    if isinstance(record.payload, dict):
        chosen = record.payload.get(record.content_kind)
        if not isinstance(chosen, str):
            return None, None, ("missing_source_text",)
        text, _ = extract_source_text(chosen, html=record.content_kind == "content", locator=record.source_locator)
        return text, None, ()
    # Decode from declared Content-Type; absence explicitly means strict UTF-8,
    # never a lossy decoder or a silent alternate charset fallback.
    import re
    match = re.search(r"charset\s*=\s*[\"']?([\w-]+)", str(record.metadata.get("content_type", "")), re.IGNORECASE)
    declared = match[1].lower() if match else "utf-8"
    encoding = {"utf8": "utf-8", "us-ascii": "ascii", "cp1252": "windows-1252"}.get(declared, declared)
    if encoding not in {"utf-8", "ascii", "windows-1252"}:
        return None, None, ("unsupported_declared_text_charset",)
    try:
        decoded = record.payload.decode(encoding, errors="strict")
    except UnicodeDecodeError:
        return None, encoding, ("source_text_decode_failed",)
    text, _ = extract_source_text(decoded, html=True, locator=record.source_locator)
    return text, encoding, ()


def _content_row(record: ReviewSourceRecord, aliases: _Aliases, identity_clocks: dict[str, pd.Timestamp]
                 ) -> tuple[str, str, dict[str, Any], str | None, list[dict[str, Any]]]:
    security = record.security_id
    cluster_source = (str(record.metadata["accession_number"]) if record.source_family == "sec" else record.source_id)
    # Unknown identity keeps the original query unit; it is reported separately
    # and cannot silently become an attributed issuer sampling unit.
    cluster = json_sha256([record.source_family, cluster_source, security or record.metadata.get("query_security_id")])
    version = json_sha256([cluster, record.source_id, record.source_version_sha256])
    reasons = list(record.unavailable_reasons)
    text_reasons: tuple[str, ...]
    if record.version_available_at_utc > END:
        text, encoding, text_reasons = None, None, ("version_after_initial_fit_cutoff",)
    else:
        text, encoding, text_reasons = _source_text(record)
    reasons.extend(text_reasons)
    alias = aliases.at(security, record.version_available_at_utc) if security else None
    identity_clock: pd.Timestamp | None = None
    if record.source_family == "sec":
        raw_clock = record.metadata.get("identity_available_at_utc")
        if raw_clock is not None:
            identity_clock = pd.Timestamp(raw_clock)
        event_clock = pd.Timestamp(record.metadata["filing"]["available_at_utc"])
    else:
        proof = record.metadata.get("identity_bridge_row_sha256") or record.metadata.get("identity_legacy_proof_row_sha256")
        identity_clock = identity_clocks.get(str(proof))
        # An identity-equal query needs no change of security identity. Its issuer
        # name still has the separately checked original filing availability.
        if record.metadata.get("query_identity_resolution") == "identity_equal":
            identity_clock = record.version_available_at_utc
        event_clock = pd.Timestamp(record.metadata["available_at_utc"])
    if alias is None:
        reasons.append("no_unambiguous_issuer_alias_available_by_source_version")
    if identity_clock is None:
        reasons.append("missing_query_or_filing_identity_availability")
    candidates: list[dict[str, Any]] = []
    row: dict[str, Any] = {"source_family": record.source_family, "source_id": record.source_id,
        "source_version_sha256": record.source_version_sha256, "security_id": security, "ticker": record.ticker,
        "published_at_utc": record.published_at_utc.isoformat(),
        "version_available_at_utc": record.version_available_at_utc.isoformat(),
        "first_seen_at_utc": None if record.first_seen_at_utc is None else record.first_seen_at_utc.isoformat(),
        "source_locator": record.source_locator, "content_kind": record.content_kind, "encoding": encoding,
        "alias_proof": alias, "source_metadata": record.metadata, "availability_semantics": "historical_proxy",
        "text_sha256": None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()}
    if not reasons and security and record.ticker and alias and identity_clock is not None:
        _require(record.first_seen_at_utc is not None, "readable content has no original observation clock")
        authority = json_sha256({"alias": alias, "identity_clock": identity_clock.isoformat(),
                                 "identity_bridge_row_sha256": record.metadata.get("identity_bridge_row_sha256"),
                                 "legacy_proof_row_sha256": record.metadata.get("identity_legacy_proof_row_sha256"),
                                 "inventory_sha256": record.metadata.get("inventory_sha256")})
        context = IssuerContentContext(event_id=record.source_id, security_id=security, ticker=record.ticker,
            issuer_aliases=tuple(alias["aliases"]), identity_authority_sha256=authority,
            identity_available_at_utc=max(identity_clock, pd.Timestamp(alias["available_at_utc"])),
            first_seen_at_utc=record.first_seen_at_utc, availability_semantics="historical_proxy",
            availability_policy_sha256=json_sha256(["retained_initial_fit_source_publication_proxy", event_clock.isoformat()]))
        if isinstance(record.payload, dict):
            _require(record.source_version_sha256 is not None, "Alpaca version hash missing")
            assert record.source_version_sha256 is not None
            evidence = alpaca_content_evidence(record=record.payload, expected_record_sha256=record.source_version_sha256,
                                               context=context)
        else:
            _require(isinstance(record.payload, bytes) and record.source_version_sha256 is not None, "SEC body missing")
            assert isinstance(record.payload, bytes) and record.source_version_sha256 is not None
            _require(encoding in ("utf-8", "ascii", "windows-1252"), "SEC text encoding not selected")
            selected_encoding: Literal["utf-8", "ascii", "windows-1252"] = (
                "ascii" if encoding == "ascii" else "windows-1252" if encoding == "windows-1252" else "utf-8")
            evidence = sec_content_evidence(body=record.payload, expected_body_sha256=record.source_version_sha256,
                accepted_at_utc=record.published_at_utc,
                acceptance_clock_authority_sha256=record.metadata["acceptance_clock_authority_sha256"],
                context=context, document_locator=record.source_locator, encoding=selected_encoding)
        _require(text == evidence.text and row["text_sha256"] == evidence.text_sha256, "source review text adapter differs")
        row.update(event_available_at_utc=max(event_clock, evidence.available_at_utc).isoformat(),
                   identity_available_at_utc=context.identity_available_at_utc.isoformat(),
                   identity_authority_sha256=authority)
        candidates = [asdict(candidate) for candidate in extract_review_candidates(evidence).candidates]
    row["unavailable_reasons"] = sorted(set(reasons))
    return cluster, version, row, text, candidates


def _clusters(connection: sqlite3.Connection) -> list[ReviewCluster]:
    result = []
    cursor = connection.execute("SELECT cluster_id,security_id,source_family,MIN(year) FROM versions "
                                "WHERE security_id IS NOT NULL AND text IS NOT NULL AND length(text)>0 GROUP BY cluster_id")
    for cluster, security, source, year in cursor:
        rules: set[tuple[Family, str]] = set()
        for encoded, in connection.execute("SELECT candidates_json FROM versions WHERE cluster_id=?", (cluster,)):
            for candidate in json.loads(encoded):
                family: Family = candidate["event_family"]
                rules.add((family, candidate["rule_id"]))
        result.append(ReviewCluster(cluster_id=cluster, security_id=security, source_family=source, year=year,
                                   candidate_families=tuple(sorted({family for family, _ in rules})), rule_ids=tuple(sorted(rules))))
    return sorted(result, key=lambda cluster: cluster.cluster_id)


def _export_review(connection: sqlite3.Connection, output: Path, clusters: list[ReviewCluster]) -> dict[str, Any]:
    samples = select_content_review_sample(clusters)
    write_json_object(output / "sample.json", {"schema": f"{SCHEMA}_sample", "policy_sha256": POLICY_SHA256,
                                               "samples": [asdict(item) for item in samples]})
    blind = output / "blind"
    blind.mkdir()
    for sample in samples:
        _guard()
        header = {"schema": f"{SCHEMA}_blind_review",
            "sample_id": sample.sample_id, "event_family_to_assess": sample.event_family,
            "instructions": "Independently assess only the original source: issuer, event family, announced/reported event, "
                            "explicit fiscal period and action; bind supporting text hash/spans; declare uncertainty. "
                            "Do not inspect extractor results, other reviewer decisions, model predictions or returns."}
        path = blind / f"{sample.sample_id}.json"
        # Serialize each bounded source version separately; a filing may have
        # many exhibits and must not accumulate all their text in process memory.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(_json(header)[:-1] + ', "versions": [')
            first = True
            for encoded, text in connection.execute(
                    "SELECT content_json,text FROM versions WHERE cluster_id=? ORDER BY version_id", (sample.cluster_id,)):
                _guard()
                if text is None:
                    continue
                row = json.loads(encoded)
                version = {key: row[key] for key in ("source_family", "source_id", "source_version_sha256", "security_id",
                    "ticker", "published_at_utc", "version_available_at_utc", "first_seen_at_utc", "source_locator", "content_kind",
                    "encoding", "alias_proof", "text_sha256")} | {"text": text}
                stream.write(("" if first else ",") + _json(version))
                first = False
            stream.write("]}\n")
    return {"sampling_clusters": len(clusters), "review_samples": len(samples),
            "candidate_clusters": {family: sum(family in row.candidate_families for row in clusters)
                                   for family in ("earnings", "guidance")}}


def publish_content_review_population(*, root: Path, config: Path, config_sha256: str, output: Path,
                                      resume_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Publish candidate/negative review evidence, retaining unknowns; no admission authority."""
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    _require(output.parent == root / "data/research", "content review output must be a direct child of data/research")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("prepare-issuer-content-review", runtime_dir=runtime, config_path=config):
        _guard()
        pins: dict[str, str] = {}
        config_pin = {"path": config.relative_to(root).as_posix(), "sha256": config_sha256}
        settings = read_json_object(pin_file(root, config_pin, pins))
        _require(set(settings) == _CONFIG_KEYS and settings["schema"] == CONFIG_SCHEMA, "content review configuration differs")
        package = Path(__file__).resolve().parents[1]
        implementations = {f"src/market_predictor/{name}": file_sha256(package / name) for name in IMPLEMENTATION_PATHS}
        request = {"schema": f"{SCHEMA}_request", "config": config_pin, "settings": settings,
                   "implementation_files": implementations, "extraction_policy_sha256": EXTRACTION_POLICY_SHA256,
                   "review_policy_sha256": POLICY_SHA256, "training_eligible": False, "serving_eligible": False}
        _require(not (output / "_manifest.json").exists(), "completed review population is immutable")
        output.mkdir(exist_ok=True)
        request_path, checkpoint_path, database = output / "_request.json", output / "_checkpoint.json", output / "population.sqlite"
        checkpoint: dict[str, Any] = {"phase": "index", "rows_completed": 0, "source_files": pins, "database_sha256": None}
        if request_path.exists():
            _require(resume_checkpoint_sha256 is not None and read_json_object(request_path) == request,
                     "resuming needs the pinned checkpoint and exactly the original request/implementation")
            _require(file_sha256(checkpoint_path) == resume_checkpoint_sha256, "resume checkpoint hash differs")
            checkpoint = read_json_object(checkpoint_path)
            snapshot = inside(output, checkpoint["database_snapshot"])
            _require(checkpoint["request_sha256"] == file_sha256(request_path)
                     and checkpoint["database_sha256"] == file_sha256(snapshot), "review database/checkpoint differs")
            pins.update(checkpoint["source_files"])
            recheck_source_pins(root, pins)
            # Working SQLite bytes are never the resume authority: interruption
            # can commit a batch before its checkpoint pointer is published.
            shutil.copyfile(snapshot, database)
        else:
            _require(resume_checkpoint_sha256 is None and not any(output.iterdir()), "review output lacks an immutable request")
            write_json_object(request_path, request)
        identity_clocks = _identity_clocks(root, settings, pins)
        with closing(_connect(database)) as connection:
            early = read_early_alias_proofs(root=root, artifact=SourcePin.model_validate(settings["early_alias_artifact"]),
                sidecar=SourcePin.model_validate(settings["early_alias_sidecar"]), pins=pins, memory_check=_guard)
            for proof in early:
                _store_alias(connection, f"cik:{proof.sec_cik}", proof)
            def save(phase: str, completed: int) -> None:
                connection.commit()
                previous = read_json_object(checkpoint_path).get("database_snapshot") if checkpoint_path.exists() else None
                directory = output / ".resume"
                directory.mkdir(exist_ok=True)
                _require(shutil.disk_usage(directory).free > database.stat().st_size + 512 * 1024**2,
                         "insufficient disk space for the next immutable resume snapshot; previous checkpoint remains recoverable")
                snapshot = directory / f"{uuid4().hex}.sqlite"
                with closing(sqlite3.connect(snapshot)) as backup:
                    connection.backup(backup)
                _atomic(checkpoint_path, {"request_sha256": file_sha256(request_path), "phase": phase,
                    "rows_completed": completed, "source_files": dict(sorted(pins.items())),
                    "database_sha256": file_sha256(snapshot), "database_snapshot": snapshot.relative_to(output).as_posix()})
                # Only the latest pointer is accepted by the resume API. The prior
                # scratch snapshot is no longer referenced; raw sources stay intact.
                if previous is not None:
                    inside(output, previous).unlink()
            phases = ("index", "alpaca", "document")
            _require(checkpoint["phase"] in (*phases, "export"), "unknown review checkpoint phase")
            for phase in phases:
                if (*phases, "export").index(phase) < (*phases, "export").index(checkpoint["phase"]):
                    continue
                skipped = checkpoint["rows_completed"] if phase == checkpoint["phase"] else 0
                source: Iterator[ReviewSourceRecord]
                if phase == "alpaca":
                    source = iter_alpaca_review_sources(root=root, inventory=SourcePin.model_validate(settings["alpaca_inventory"]),
                                                       pins=pins, memory_check=_guard)
                else:
                    source = iter_sec_review_sources(root=root, archive=SourcePin.model_validate(settings["sec_archive"]),
                        inventory=SourcePin.model_validate(settings["sec_inventory"]), pins=pins, memory_check=_guard,
                        phase="index" if phase == "index" else "document")
                aliases = _Aliases(connection) if phase != "index" else None
                count = 0
                for count, record in enumerate(source, 1):
                    if count <= skipped:
                        continue
                    if phase == "index":
                        if isinstance(record.payload, bytes) and record.source_version_sha256 and record.security_id:
                            extracted = extract_sec_alias_proofs(body=record.payload, body_sha256=record.source_version_sha256,
                                accession=record.metadata["accession_number"], expected_cik=record.metadata["sec_cik"],
                                accepted_at_utc=record.published_at_utc,
                                clock_authority_sha256=record.metadata["acceptance_clock_authority_sha256"],
                                locator=record.source_locator)
                            for proof in extracted.proofs:
                                _store_alias(connection, record.security_id, proof)
                        connection.execute("INSERT INTO records VALUES (?,?,?,?,?)",
                            (count, phase, "", "", _json({"source_id": record.source_id, "metadata": record.metadata,
                                                         "unavailable_reasons": record.unavailable_reasons})))
                    else:
                        assert aliases is not None
                        cluster, version, row, text, candidates = _content_row(record, aliases, identity_clocks)
                        prior = connection.execute("SELECT content_json,text,candidates_json FROM versions WHERE version_id=?",
                                                   (version,)).fetchone()
                        # Query copies retain their own clocks/locators. Shared text
                        # and candidate decisions must agree for one issuer/version.
                        if prior is not None:
                            _require(prior[1] == text and json.loads(prior[2]) == candidates,
                                     "repeated source version has conflicting text or candidate decisions")
                        else:
                            connection.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)",
                                (version, cluster, record.security_id, record.source_family,
                                 record.published_at_utc.tz_convert("America/New_York").year, _json(row), text, _json(candidates)))
                        connection.execute("INSERT INTO records VALUES (?,?,?,?,?)", (count, phase, cluster, version, _json(row)))
                    if count % COMMIT_ROWS == 0:
                        _guard()
                        save(phase, count)
                        print(_json({"phase": phase, "rows_completed": count}), flush=True)
                _require(count >= skipped, "resumed source population became smaller")
                next_phase = {"index": "alpaca", "alpaca": "document", "document": "export"}[phase]
                save(next_phase, 0)
            _guard()
            # Export resume never accepts uncheckpointed files as reviewed evidence.
            _require(not (output / "sample.json").exists() and not (output / "blind").exists(),
                     "partial review export requires explicit inspection; it cannot be silently overwritten")
            clusters = _clusters(connection)
            totals = _export_review(connection, output, clusters)
            totals["source_occurrences"] = dict(connection.execute("SELECT phase,count(*) FROM records GROUP BY phase"))
            totals["source_versions"] = connection.execute("SELECT count(*) FROM versions").fetchone()[0]
            totals["readable_identity_unknown_versions"] = connection.execute(
                "SELECT count(*) FROM versions WHERE security_id IS NULL AND text IS NOT NULL").fetchone()[0]
            totals["unavailable_version_reasons"] = dict(connection.execute(
                "SELECT reason.value,count(*) FROM versions,json_each(versions.content_json,'$.unavailable_reasons') AS reason "
                "GROUP BY reason.value ORDER BY reason.value"))
            totals["version_source_years"] = [
                {"source_family": source, "year": year, "versions": count, "readable": readable}
                for source, year, count, readable in connection.execute(
                    "SELECT source_family,year,count(*),sum(text IS NOT NULL AND length(text)>0) "
                    "FROM versions GROUP BY source_family,year ORDER BY source_family,year")]
            connection.commit()
        recheck_source_pins(root, pins | implementations)
        artifacts = {path.relative_to(output).as_posix(): file_sha256(path)
                     for path in sorted(output.rglob("*")) if path.is_file() and path.name != "_checkpoint.json"
                     and ".resume" not in path.relative_to(output).parts}
        result = {"schema": SCHEMA, "status": "complete_review_population_only", "request_sha256": file_sha256(request_path),
                  "source_files": dict(sorted(pins.items())), "implementation_files": implementations, "artifacts": artifacts,
                  "totals": totals, "qualification": "not_established_requires_two_independent_source_only_reviews",
                  "clustering": "SEC_accession_per_issuer_or_Alpaca_story_per_issuer_all_retained_versions_one_sampling_unit",
                  "unknown_identity_scope": "reported_separately_not_silently_attributed_or_claimed_in_issuer_recall",
                  "recall_scope": "retained_readable_attributed_source_population_not_complete_provider_capture",
                  "training_eligible": False, "serving_eligible": False, "promotion_eligible": False}
        write_json_object(output / "_manifest.json", result)
        return result | {"manifest_sha256": file_sha256(output / "_manifest.json")}
