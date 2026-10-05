"""Source-bound historical content qualification; no model or serving admission.

Review records are externally supplied source-only judgments. This publisher never
creates labels. Every source version retains a disposition; a history with an
unproven in-period identity/clock is excluded in full from the reaction projection.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.issuer_events.content_review import EXTRACTION_POLICY_SHA256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.governance.issuer_content_qualification import (
    POLICY,
    POLICY_SHA256,
    ContentReview,
    SampleItem,
    evaluate_content_reviews,
    select_content_review_sample,
)
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import issuer_content_review_population as population
from market_predictor.research.issuer_content_review_sources import END, recheck_source_pins
from market_predictor.research.legacy_query_identity_proofs import pin_file
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction_profile import QUALIFIED_EVENT_COLUMNS

SCHEMA = "market_predictor.issuer_content_qualification_authority"
REVIEW_SCHEMA = "market_predictor.issuer_content_reviews"
MAX_JSON_BYTES = 64 * 1024**2
IMPLEMENTATION_PATHS = (
    *population.IMPLEMENTATION_PATHS,
    "research/issuer_content_qualification_authority.py",
    "swing/contracts/issuer_reaction_profile.py",
)
# The producer imports these only for SourcePin/HoldingContract. Holding-cost
# changes do not change saved source text, candidates, clusters or review sampling.
# Current consumer copies remain pinned independently during qualification.
_PRODUCER_PROVENANCE_ONLY = frozenset({
    "src/market_predictor/swing/contracts/holding_materialization.py",
    "src/market_predictor/swing/contracts/holding_accounting.py",
})
_FIELDS = ("family_present", "issuer_correct", "announced_or_reported", "explicit_fiscal_period", "action_supported")
_ENTRY_KEYS = {"sample_id", "source_id", "source_version_sha256", "supporting_text_sha256", "supporting_spans", *_FIELDS}
_REVIEW_KEYS = {"schema", "population_authority", "sample_sha256", "reviewer_id", "reviewer_kind", "reviewer_model",
                "evidence_scope", "independent_review", "reviews"}
_BLIND_FIELDS = ("source_family", "source_id", "source_version_sha256", "security_id", "ticker", "published_at_utc",
                 "version_available_at_utc", "first_seen_at_utc", "source_locator", "content_kind", "encoding",
                 "alias_proof", "text_sha256")
_PROJECTION_POLICY = {
    "schema": SCHEMA, "review_policy_sha256": POLICY_SHA256,
    "eligibility": "admitted_family_and_unambiguous_resolved_candidate_with_explicit_period",
    "review_override": "reviewed_exact_version_requires_both_joint_positive_judgments",
    "unproven_history": "exclude_evidenced_source_event_issuer_history_using_only_causal_parent_query_links",
    "ambiguous_history": "suppress_possible_causal_histories_without_asserting_unknown_issuer_identity",
    "future": "metadata_disposition_only_never_review_or_projection",
    "versions": "retain_rejected_and_unclassified_versions_no_old_revision_resurrection",
    "duplicates": "parent_source_announcement_cluster_only_no_inferred_cross_source_equivalence",
    "coverage": "not_established_missing_or_excluded_events_never_zero",
    "producer_code": "retain_original_inventory_replay_source_text_and_sampling_dependencies_currently_"
                     "exclude_source_pin_holding_contract_imports_from_producer_replay_pin_current_consumer_independently",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _object(path: Path) -> dict[str, Any]:
    _require(path.stat().st_size <= MAX_JSON_BYTES, "qualification JSON exceeds byte bound")
    with path.open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    _require(len(raw) <= MAX_JSON_BYTES, "qualification JSON grew beyond byte bound")
    try:
        return parse_strict_json_object(raw, label=str(path))
    except ValueError as exc:
        raise DataReadinessError(str(exc)) from exc


def _pin(root: Path, pin: SourcePin, pins: dict[str, str]) -> Path:
    previous = pins.get(inside(root, pin.path).relative_to(root).as_posix())
    _require(previous is None or previous == pin.sha256, "conflicting qualification source pins")
    return pin_file(root, pin.model_dump(mode="json"), pins)


def _guard_batch(position: int) -> None:
    # The canonical guard also collects the entire Python heap. Running it per
    # version makes multiple 516k-row passes quadratic in retained heap size.
    # Bound the interval; every source hash and every row check still executes.
    if position % 256 == 0:
        population._guard()


def _parent(root: Path, authority: SourcePin, pins: dict[str, str]) -> tuple[Path, dict[str, Any]]:
    path = _pin(root, authority, pins)
    manifest = _object(path)
    _require(manifest.get("schema") == population.SCHEMA and manifest.get("status") == "complete_review_population_only",
             "qualification requires a complete review population")
    _require(all(manifest.get(flag) is False for flag in ("training_eligible", "serving_eligible", "promotion_eligible")),
             "parent population makes unsupported admission claims")
    for field in ("source_files", "implementation_files", "artifacts"):
        _require(isinstance(manifest.get(field), dict) and bool(manifest[field]), f"parent {field} missing")
    folder = path.parent
    for position, (name, digest) in enumerate(manifest["artifacts"].items()):
        _guard_batch(position)
        artifact = inside(folder, name)
        _pin(root, SourcePin(path=artifact.relative_to(root).as_posix(), sha256=digest), pins)
    _require({"population.sqlite", "sample.json", "_request.json"}.issubset(manifest["artifacts"]),
             "parent population artifacts missing")
    request = _object(folder / "_request.json")
    _require(manifest["request_sha256"] == file_sha256(folder / "_request.json")
             and request.get("schema") == f"{population.SCHEMA}_request"
             and request.get("review_policy_sha256") == POLICY_SHA256
             and request.get("extraction_policy_sha256") == EXTRACTION_POLICY_SHA256,
             "parent request or policy differs")
    expected_implementations = {
        f"src/market_predictor/{name}": file_sha256(Path(population.__file__).resolve().parents[1] / name)
        for name in population.IMPLEMENTATION_PATHS
    }
    producer = manifest["implementation_files"]
    _require(request.get("implementation_files") == producer and set(producer) == set(expected_implementations),
             "parent producer implementation inventory differs")
    _require(all(isinstance(digest, str) and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
                 for digest in producer.values()), "parent producer implementation hash is malformed")
    _require(all(digest == expected_implementations[name] for name, digest in producer.items()
                 if name not in _PRODUCER_PROVENANCE_ONLY), "parent replay implementation differs")
    _require(not set(manifest["source_files"]).intersection(producer),
             "parent mixes producer implementation provenance into data sources")
    for position, (name, digest) in enumerate(manifest["source_files"].items()):
        _guard_batch(position)
        _pin(root, SourcePin(path=name, sha256=digest), pins)
    config = SourcePin.model_validate(request["config"])
    _require(manifest["source_files"].get(config.path) == config.sha256, "parent config is not a declared source")
    _require(_object(_pin(root, config, pins)) == request["settings"], "parent config/settings differ")
    _require(set(request["settings"]) == population._CONFIG_KEYS, "parent configuration keys differ")
    for key, raw in request["settings"].items():
        if key != "schema":
            setting = SourcePin.model_validate(raw)
            _require(manifest["source_files"].get(setting.path) == setting.sha256, "parent settings pin is not a declared source")
    population._guard()
    return folder, manifest


def _verify_blind(connection: sqlite3.Connection, folder: Path, item: SampleItem, manifest: dict[str, Any]) -> None:
    """Replay the producer's streamed packet bytes without retaining all versions."""
    name = f"blind/{item.sample_id}.json"
    _require(name in manifest["artifacts"], "sample blind packet missing")
    path = inside(folder, name)
    marker = b', "versions": ['
    with path.open("rb") as stream:
        prefix = stream.read(16384)
        position = prefix.find(marker)
        _require(position >= 0, "blind packet header malformed")
        header_bytes = prefix[:position]
        header = parse_strict_json_object(header_bytes + b"}", label="blind header")
        _require(set(header) == {"schema", "sample_id", "event_family_to_assess", "instructions"}
                 and header["schema"] == f"{population.SCHEMA}_blind_review"
                 and header["sample_id"] == item.sample_id and header["event_family_to_assess"] == item.event_family,
                 "blind packet sample identity differs")
        stream.seek(-3, 2)
        tail = stream.read()
    digest = hashlib.sha256(header_bytes + marker)
    first = True
    for encoded, text in connection.execute(
        "SELECT content_json,text FROM versions WHERE cluster_id=? ORDER BY version_id", (item.cluster_id,),
    ):
        population._guard()
        if text is None:
            continue
        row = json.loads(encoded)
        version = {key: row[key] for key in _BLIND_FIELDS} | {"text": text}
        digest.update((("" if first else ",") + population._json(version)).encode("utf-8"))
        first = False
    digest.update(b"]}\r\n" if tail.endswith(b"\r\n") else b"]}\n")
    _require(digest.hexdigest() == manifest["artifacts"][name], "blind packet differs from exact source versions")


def _reviews(
    root: Path, files: tuple[SourcePin, SourcePin], authority: SourcePin, folder: Path,
    samples: tuple[SampleItem, ...], connection: sqlite3.Connection, pins: dict[str, str],
) -> tuple[list[ContentReview], dict[tuple[str, str, str, str], list[ContentReview]]]:
    _require(len(files) == 2 and files[0].path != files[1].path, "two independent reviewer files required")
    sample_index = {item.sample_id: item for item in samples}
    result = []
    by_version: dict[tuple[str, str, str, str], list[ContentReview]] = defaultdict(list)
    identities: set[str] = set()
    sample_versions: dict[str, tuple[str, str, str]] = {}
    for pin in files:
        document = _object(_pin(root, pin, pins))
        _require(set(document) == _REVIEW_KEYS and document["schema"] == REVIEW_SCHEMA,
                 "reviewer document schema/fields differ")
        _require(document["population_authority"] == authority.model_dump(mode="json")
                 and document["sample_sha256"] == file_sha256(folder / "sample.json"), "reviewer parent/sample differs")
        _require(document["evidence_scope"] == "source_only" and document["independent_review"] is True,
                 "reviewer must explicitly declare independent source-only evidence")
        _require((document["reviewer_kind"] == "human" and document["reviewer_model"] is None)
                 or (document["reviewer_kind"] == "model_assisted" and isinstance(document["reviewer_model"], str)
                     and bool(document["reviewer_model"].strip())), "reviewer assistance must be explicit")
        identity = document["reviewer_id"]
        _require(isinstance(identity, str) and bool(identity.strip()) and identity not in identities,
                 "reviewer identities must be distinct")
        _require(isinstance(document["reviews"], list), "reviewer records must be a list")
        identities.add(identity)
        seen: set[str] = set()
        for position, entry in enumerate(document["reviews"]):
            _guard_batch(position)
            _require(isinstance(entry, dict) and set(entry) == _ENTRY_KEYS, "review record fields differ")
            sample_id = entry["sample_id"]
            _require(sample_id in sample_index and sample_id not in seen, "unknown or duplicate reviewer sample")
            seen.add(sample_id)
            item = sample_index[sample_id]
            found = connection.execute(
                "SELECT content_json,text FROM versions WHERE cluster_id=? "
                "AND json_extract(content_json,'$.source_id')=? AND json_extract(content_json,'$.source_version_sha256')=?",
                (item.cluster_id, entry["source_id"], entry["source_version_sha256"]),
            ).fetchall()
            _require(len(found) == 1, "review source id/version is absent or ambiguous")
            source, text = json.loads(found[0][0]), found[0][1]
            _require(isinstance(text, str) and bool(text) and pd.Timestamp(source["version_available_at_utc"]) <= END,
                     "review source is unreadable or beyond initial fit")
            _require(hashlib.sha256(text.encode("utf-8")).hexdigest() == entry["supporting_text_sha256"] == source["text_sha256"],
                     "review source text hash differs")
            spans = entry["supporting_spans"]
            _require(isinstance(spans, list) and all(isinstance(span, list) and len(span) == 2
                     and all(type(value) is int for value in span) and 0 <= span[0] < span[1] <= len(text) for span in spans),
                     "review spans lie outside exact source text")
            version = (entry["source_id"], entry["source_version_sha256"], entry["supporting_text_sha256"])
            _require(sample_versions.setdefault(sample_id, version) == version, "reviewers assessed different source versions")
            review = ContentReview(
                sample_id=sample_id, reviewer_id=identity, reviewer_kind=document["reviewer_kind"],
                reviewer_model=document["reviewer_model"], evidence_scope=document["evidence_scope"],
                independent_review=document["independent_review"], supporting_text_sha256=entry["supporting_text_sha256"],
                supporting_spans=tuple((span[0], span[1]) for span in spans), **{field: entry[field] for field in _FIELDS},
            )
            result.append(review)
            by_version[(item.cluster_id, entry["source_id"], entry["source_version_sha256"], item.event_family)].append(review)
        population._guard()
    return result, by_version


def _clock(value: Any) -> pd.Timestamp | None:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError):
        return None
    return stamp.tz_convert("UTC") if not pd.isna(stamp) and stamp.tzinfo is not None else None


def _identity_errors(row: dict[str, Any]) -> list[str]:
    reasons = []
    if not all(isinstance(row.get(name), str) and bool(row[name].strip()) for name in ("security_id", "ticker", "source_id")):
        reasons.append("unproven_source_identity")
    clocks = [_clock(row.get(name)) for name in
              ("event_available_at_utc", "identity_available_at_utc", "version_available_at_utc", "published_at_utc")]
    known = [clock for clock in clocks if clock is not None]
    if len(known) != len(clocks):
        reasons.append("unproven_source_clock")
    elif known[0] < max(known[1:]):
        reasons.append("event_clock_precedes_source_or_identity")
    for name in ("identity_authority_sha256", "source_version_sha256"):
        value = row.get(name)
        if not isinstance(value, str) or len(value) != 64 or not set(value) <= set("0123456789abcdef"):
            reasons.append(f"unproven_{name}")
    return reasons


def _history_exclusions(
    connection: sqlite3.Connection,
) -> tuple[dict[tuple[str, str, str], set[str]], dict[str, tuple[str, ...]], dict[str, str]]:
    """Retained query copies can link a legacy query to its proven cohort issuer.

    Links are local to one source event and become usable only when both their
    source version and identity proof were available. A later proof never assigns
    an earlier unknown version. Ambiguity suppresses possible histories, not identity.
    """
    links: dict[tuple[str, str, str, str], set[tuple[pd.Timestamp, str]]] = defaultdict(set)
    for query in ("SELECT cluster_id,content_json FROM versions",
                  "SELECT cluster_id,record_json FROM records WHERE version_id!=''"):
        for position, (cluster, encoded) in enumerate(connection.execute(query)):
            _guard_batch(position)
            row = json.loads(encoded)
            clock = _clock(row.get("version_available_at_utc"))
            if clock is None or clock > END or _identity_errors(row):
                continue
            identity_clock = _clock(row["identity_available_at_utc"])
            assert identity_clock is not None
            available = max(clock, identity_clock)
            if available > END:
                continue
            security, source, event = row["security_id"], row["source_family"], row["source_id"]
            for kind, token in (("cluster", cluster), ("query", security), ("event", "*")):
                links[(source, event, kind, token)].add((available, security))
            metadata = row.get("source_metadata", {})
            query_id, resolution = metadata.get("query_security_id"), metadata.get("query_identity_resolution")
            proof = metadata.get("identity_bridge_row_sha256" if resolution == "bridged" else "identity_legacy_proof_row_sha256")
            proven = (resolution == "identity_equal" and query_id == security) or (
                resolution in ("bridged", "proven_legacy_identity") and metadata.get("cohort_security_id") == security
                and isinstance(proof, str) and len(proof) == 64 and set(proof) <= set("0123456789abcdef"))
            if proven and isinstance(query_id, str) and query_id:
                links[(source, event, "query", query_id)].add((available, security))
        population._guard()
    excluded: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    targets: dict[str, tuple[str, ...]] = {}
    ambiguity: dict[str, str] = {}
    for position, (cluster, version, encoded) in enumerate(connection.execute("SELECT cluster_id,version_id,content_json FROM versions")):
        _guard_batch(position)
        row = json.loads(encoded)
        clock = _clock(row.get("version_available_at_utc"))
        if clock is not None and clock > END:
            targets[version] = ()
            continue
        source, event, security = row["source_family"], row["source_id"], row.get("security_id")
        if isinstance(security, str) and security:
            owners = {security}
        else:
            query_id = row.get("source_metadata", {}).get("query_security_id")
            candidates = links.get((source, event, "cluster", cluster), set()) | links.get((source, event, "query", query_id), set())
            owners = {issuer for available, issuer in candidates if clock is not None and available <= clock}
            if len(owners) != 1:
                ambiguity[version] = "ambiguous_source_event_issuer_lineage"
                if not owners:
                    # Later in-cutoff evidence cannot assign identity backward,
                    # but identifies histories that must remain suppressed.
                    # Prefer exact cluster/query evidence over unrelated issuers.
                    owners = {issuer for _, issuer in candidates}
                    if not owners:
                        owners = {issuer for _, issuer in links.get((source, event, "event", "*"), ())}
        targets[version] = tuple(sorted(owners))
        errors = _identity_errors(row)
        if version in ambiguity:
            errors.append(ambiguity[version])
        for owner in owners:
            if errors:
                excluded[(source, event, owner)].update(errors)
    population._guard()
    return excluded, targets, ambiguity


def _publish_events(
    connection: sqlite3.Connection, output: Path, authority_sha256: str, metrics: dict[str, Any],
    reviews: dict[tuple[str, str, str, str], list[ContentReview]],
) -> dict[str, Any]:
    excluded, history_targets, ambiguous_versions = _history_exclusions(connection)
    query = "SELECT cluster_id,version_id,content_json,candidates_json FROM versions ORDER BY version_id"
    schema = pa.schema([
        (name, pa.timestamp("ns", tz="UTC") if name.endswith("_at_utc") else pa.string())
        for name in QUALIFIED_EVENT_COLUMNS
    ])
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    with (output / "version_dispositions.jsonl").open("x", encoding="utf-8", newline="\n") as dispositions:
        with cast(Any, pq.ParquetWriter)(output / "events.parquet", schema) as writer:
            for position, (cluster, version, encoded, candidates_encoded) in enumerate(connection.execute(query)):
                _guard_batch(position)
                row, candidates = json.loads(encoded), json.loads(candidates_encoded)
                clock = _clock(row.get("version_available_at_utc"))
                future = clock is not None and clock > END
                history_errors = sorted({reason for owner in history_targets[version]
                                         for reason in excluded.get((row["source_family"], row["source_id"], owner), ())})
                if not future:
                    history_errors = sorted(set(history_errors + _identity_errors(row)
                                                + ([ambiguous_versions[version]] if version in ambiguous_versions else [])))
                reasons = list(row["unavailable_reasons"])
                status, family = "unclassified", None
                eligible = [candidate for candidate in candidates if candidate.get("fiscal_period")
                            and not candidate.get("unresolved_reasons")]
                families = {candidate["event_family"] for candidate in eligible}
                if future:
                    reasons.append("version_after_initial_fit_cutoff")
                elif history_errors:
                    reasons.extend(history_errors)
                elif reasons:
                    status = "rejected"
                elif len(families) != 1:
                    reasons.append("ambiguous_event_families" if len(families) > 1 else "no_resolved_explicit_period_candidate")
                    status = "rejected" if candidates else "unclassified"
                else:
                    family = next(iter(families))
                    pair = reviews.get((cluster, row["source_id"], row["source_version_sha256"], family))
                    if pair is not None and (len(pair) != 2 or not all(review.joint for review in pair)):
                        reasons.append("exact_version_review_not_joint_positive")
                        status = "rejected"
                    elif not metrics["families"][family]["qualified_for_historical_feature_use"]:
                        reasons.append("family_review_gates_not_met")
                        status = "rejected"
                    else:
                        status = "qualified"
                include = not future and not history_errors
                disposition = {"cluster_id": cluster, "version_id": version, "source": row,
                               "qualification_status": status, "event_family": family, "reasons": sorted(set(reasons)),
                               "possible_history_issuers": list(history_targets[version]),
                               "projected": include, "qualification_authority_sha256": authority_sha256}
                dispositions.write(json.dumps(disposition, sort_keys=True, allow_nan=False) + "\n")
                counts["source_versions"] += 1
                counts[f"{status}_versions"] += 1
                if future:
                    counts["future_metadata_only_versions"] += 1
                if not include:
                    counts["excluded_projection_versions"] += 1
                    continue
                counts["projected_versions"] += 1
                event = {
                    "security_id": row["security_id"], "ticker": row["ticker"], "source_family": row["source_family"],
                    "event_id": row["source_id"], "event_version_sha256": row["source_version_sha256"],
                    "event_available_at_utc": _clock(row["event_available_at_utc"]),
                    "identity_available_at_utc": _clock(row["identity_available_at_utc"]), "event_family": family,
                    "qualification_status": status, "qualification_authority_sha256": authority_sha256,
                    "duplicate_group_id": cluster,
                }
                rows.append(event)
                if len(rows) == 1000:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
            if rows:
                writer.write_table(pa.Table.from_pylist(rows, schema=schema))
    population._guard()
    return {"counts": dict(counts), "excluded_histories": [
        {"source_family": source, "source_id": event, "security_id": security, "reasons": sorted(reasons)}
        for (source, event, security), reasons in sorted(excluded.items())
    ]}


def publish_issuer_content_qualification(
    *, root: Path, population_authority: SourcePin, reviewer_files: tuple[SourcePin, SourcePin], output: Path,
) -> dict[str, Any]:
    """Publish immutable measured source qualification after both external reviews."""
    root = root.resolve()
    output = inside(root, output)
    _require(output.parent == root / "data/research", "qualification output must be a direct child of data/research")
    runtime = heavy_job_runtime_dir()
    runtime = runtime if runtime.is_absolute() else root / runtime
    with heavy_job_lease("qualify-issuer-content", runtime_dir=runtime):
        population._guard()
        _require(not output.exists(), "qualification output is immutable; use a new output path")
        pins: dict[str, str] = {}
        folder, manifest = _parent(root, population_authority, pins)
        package = Path(__file__).resolve().parents[1]
        implementations = {f"src/market_predictor/{name}": file_sha256(package / name) for name in IMPLEMENTATION_PATHS}
        for name, digest in implementations.items():
            _pin(root, SourcePin(path=name, sha256=digest), pins)
        database = (folder / "population.sqlite").as_uri() + "?mode=ro&immutable=1"
        with closing(sqlite3.connect(database, uri=True)) as connection:
            connection.execute("PRAGMA query_only=ON")
            clusters = population._clusters(connection)
            samples = select_content_review_sample(clusters)
            expected = {"schema": f"{population.SCHEMA}_sample", "policy_sha256": POLICY_SHA256,
                        "samples": [asdict(item) for item in samples]}
            _require(_object(folder / "sample.json") == expected, "frozen review sample replay differs")
            for item in samples:
                _verify_blind(connection, folder, item, manifest)
            reviews, by_version = _reviews(root, reviewer_files, population_authority, folder, samples, connection, pins)
            metrics = evaluate_content_reviews(clusters=clusters, samples=samples, reviews=reviews)
            recheck_source_pins(root, pins)
            output.mkdir()
            request = {"schema": f"{SCHEMA}_request", "population_authority": population_authority.model_dump(mode="json"),
                       "reviewer_files": [pin.model_dump(mode="json") for pin in reviewer_files],
                       "implementation_files": implementations, "policy_sha256": json_sha256(_PROJECTION_POLICY)}
            write_json_object(output / "_request.json", request)
            authority = {"schema": SCHEMA, "request_sha256": file_sha256(output / "_request.json"),
                         "qualification_policy": _PROJECTION_POLICY, "review_policy": POLICY, "metrics": metrics,
                         "producer_implementation_files": manifest["implementation_files"],
                         "implementation_files": implementations,
                         "source_files": dict(sorted(pins.items())), "training_eligible": False,
                         "serving_eligible": False, "promotion_eligible": False, "coverage_established": False}
            write_json_object(output / "_authority.json", authority)
            authority_sha256 = file_sha256(output / "_authority.json")
            projection = _publish_events(connection, output, authority_sha256, metrics, by_version)
        recheck_source_pins(root, pins)
        artifacts = {name: file_sha256(output / name) for name in
                     ("_request.json", "_authority.json", "events.parquet", "version_dispositions.jsonl")}
        _require(artifacts["_authority.json"] == authority_sha256
                 and artifacts["_request.json"] == authority["request_sha256"], "qualification authority changed during projection")
        result = {"schema": f"{SCHEMA}_publication", "status": "complete", "artifacts": artifacts,
                  "qualification_authority_sha256": authority_sha256, **projection,
                  "research_feature_eligible": projection["counts"].get("qualified_versions", 0) > 0,
                  "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
                  "coverage_established": False}
        population._guard()
        write_json_object(output / "_manifest.json", result)
        return result | {"manifest_sha256": file_sha256(output / "_manifest.json")}
