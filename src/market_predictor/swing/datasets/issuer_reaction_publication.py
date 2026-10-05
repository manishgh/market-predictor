"""Immutable monthly 124-to-126 publication from exact reviewed source history.

Saved parent values are retained, not claimed freshly recomputed. Qualification
is independently replayed from the pinned source-only review population; source
articles are not extracted again. Coverage remains explicitly unknown.
"""
from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.governance.issuer_content_qualification import (
    POLICY,
    POLICY_SHA256,
    evaluate_content_reviews,
    select_content_review_sample,
)
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import issuer_content_qualification_authority as qualification
from market_predictor.research import issuer_content_review_population as population
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS, IssuerReactionSources
from market_predictor.swing.contracts.issuer_reaction_profile import issuer_reaction_profile_sha256
from market_predictor.swing.contracts.issuer_reaction_publication import (
    ARTIFACT_TYPE,
    PROFILE,
    PUBLICATION_SCHEMA,
    REQUEST_SCHEMA,
    IssuerReactionPublicationPolicy,
    VerifiedIssuerReactionPublication,
)
from market_predictor.swing.contracts.return_relationship_publication import (
    ReturnRelationshipPublicationPolicy,
    VerifiedReturnRelationshipPublication,
)
from market_predictor.swing.datasets.adjusted_history_bindings import bind_adjusted_history_decisions
from market_predictor.swing.datasets.return_relationship_integrity import (
    assert_closed,
    check_files,
    guard,
    pins,
    read_object,
    replace_checkpoint,
)
from market_predictor.swing.datasets.return_relationship_integrity import (
    current_implementation as relationship_implementation,
)
from market_predictor.swing.datasets.return_relationship_parent import verify_parent
from market_predictor.swing.datasets.return_relationship_rows import read_spy, read_stock
from market_predictor.swing.datasets.return_relationship_sources import (
    RelationshipSourceContext,
    relationship_sources,
    verify_source_context,
)
from market_predictor.swing.datasets.return_relationship_verification import validate_return_relationship_receipt
from market_predictor.swing.features.issuer_reaction_profile import build_issuer_reaction_profile
from market_predictor.swing.features.research_partition import ResearchFeaturePartition

CLOSED = {"training_eligible": False, "promotion_eligible": False, "serving_eligible": False}
EXPECTED_ROWS, EXPECTED_MONTHS = 586305, 59


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def current_implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    names = (*qualification.IMPLEMENTATION_PATHS,
        "swing/contracts/issuer_reaction_publication.py", "swing/datasets/issuer_reaction_publication.py",
        "swing/features/issuer_reaction_profile.py", "swing/features/issuer_reaction.py",
        "swing/contracts/issuer_reaction_profile.py", "swing/contracts/issuer_reaction.py")
    return pins(root, relationship_implementation(root),
                {(package / name).relative_to(root).as_posix(): file_sha256(package / name) for name in names})


def load_policy(root: Path, config: Path, digest: str) -> IssuerReactionPublicationPolicy:
    return IssuerReactionPublicationPolicy.model_validate(read_object(inside(root, config), digest))


def _qualification(root: Path, policy: IssuerReactionPublicationPolicy) -> tuple[Path, dict[str, str], dict[str, str]]:
    manifest_path = inside(root, policy.qualification_publication.path)
    authority_path = inside(root, policy.qualification_authority.path)
    _require(manifest_path.name == "_manifest.json" and authority_path == manifest_path.parent / "_authority.json",
             "qualification requires its exact completed manifest and authority")
    manifest = read_object(manifest_path, policy.qualification_publication.sha256)
    assert_closed(manifest)
    _require(manifest.get("schema") == qualification.SCHEMA + "_publication" and manifest.get("status") == "complete"
             and manifest.get("coverage_established") is False
             and manifest.get("qualification_authority_sha256") == policy.qualification_authority.sha256
             and set(manifest.get("artifacts", {})) == {"_request.json", "_authority.json", "events.parquet", "version_dispositions.jsonl"},
             "qualification publication contract differs")
    folder = manifest_path.parent
    files = {inside(folder, name).relative_to(root).as_posix(): digest for name, digest in manifest["artifacts"].items()}
    files = pins(root, files, {policy.qualification_publication.path: policy.qualification_publication.sha256,
                              policy.qualification_authority.path: policy.qualification_authority.sha256})
    check_files(root, files)
    authority = read_object(authority_path, policy.qualification_authority.sha256)
    request = read_object(folder / "_request.json", manifest["artifacts"]["_request.json"])
    assert_closed(authority)
    _require(authority.get("schema") == qualification.SCHEMA and authority.get("coverage_established") is False
             and authority.get("request_sha256") == manifest["artifacts"]["_request.json"]
             and authority.get("qualification_policy") == qualification._PROJECTION_POLICY
             and authority.get("review_policy") == POLICY
             and request.get("schema") == qualification.SCHEMA + "_request"
             and request.get("policy_sha256") == json_sha256(qualification._PROJECTION_POLICY),
             "qualification authority or frozen policy differs")
    package = Path(qualification.__file__).resolve().parents[1]
    expected_implementations = {f"src/market_predictor/{name}": file_sha256(package / name)
                                for name in qualification.IMPLEMENTATION_PATHS}
    _require(request.get("implementation_files") == authority.get("implementation_files") == expected_implementations,
             "qualification replay implementation changed")
    files = pins(root, files, authority["source_files"], expected_implementations)
    check_files(root, files)
    parent_pin = SourcePin.model_validate(request["population_authority"])
    replay_pins: dict[str, str] = {}
    parent_folder, parent = qualification._parent(root, parent_pin, replay_pins)
    _require(authority.get("producer_implementation_files") == parent["implementation_files"], "qualification producer lineage differs")
    database = (parent_folder / "population.sqlite").as_uri() + "?mode=ro&immutable=1"
    with closing(sqlite3.connect(database, uri=True)) as connection:
        connection.execute("PRAGMA query_only=ON")
        clusters = population._clusters(connection)
        samples = select_content_review_sample(clusters)
        _require(qualification._object(parent_folder / "sample.json") == {
            "schema": population.SCHEMA + "_sample", "policy_sha256": POLICY_SHA256,
            "samples": [asdict(item) for item in samples]}, "qualification sample replay differs")
        for item in samples:
            qualification._verify_blind(connection, parent_folder, item, parent)
        reviewers = tuple(SourcePin.model_validate(value) for value in request["reviewer_files"])
        _require(len(reviewers) == 2, "qualification requires both independent reviews")
        reviews, by_version = qualification._reviews(root, (reviewers[0], reviewers[1]), parent_pin,
            parent_folder, samples, connection, replay_pins)
        metrics = evaluate_content_reviews(clusters=clusters, samples=samples, reviews=reviews)
        _require(metrics == authority["metrics"], "qualification metric replay differs")
        # Use the one disposition/exclusion owner, never duplicate its arithmetic.
        with TemporaryDirectory(prefix="issuer-qualification-replay-") as temporary:
            replay = Path(temporary)
            projection = qualification._publish_events(connection, replay, policy.qualification_authority.sha256, metrics, by_version)
            _require(projection["counts"] == manifest["counts"] and projection["excluded_histories"] == manifest["excluded_histories"],
                     "qualification full-history projection differs")
            for name in ("events.parquet", "version_dispositions.jsonl"):
                _require(file_sha256(replay / name) == manifest["artifacts"][name], "qualification version/disposition replay differs")
    _require(all(authority["source_files"].get(name) == digest for name, digest in replay_pins.items()),
             "qualification replay source is undeclared")
    _require(manifest.get("research_feature_eligible") == (manifest["counts"].get("qualified_versions", 0) > 0),
             "qualification feature claim differs from actual qualified versions")
    check_files(root, files)
    return folder / "events.parquet", files, dict(parent["implementation_files"])


@dataclass(frozen=True)
class ReactionPublicationInputs:
    root: Path
    policy: IssuerReactionPublicationPolicy
    parent: VerifiedReturnRelationshipPublication
    parent_path: Path
    context: RelationshipSourceContext
    events_path: Path
    sources: IssuerReactionSources
    source_files: dict[str, str]
    historical_implementation_files: dict[str, str]


def load_reaction_inputs(root: Path, policy: IssuerReactionPublicationPolicy) -> ReactionPublicationInputs:
    """Verify all parents without acquiring a nested lease; caller owns one."""
    root = root.resolve()
    guard(policy.maximum_system_used_percent)
    receipt_path = inside(root, policy.parent_saved_row_verification.path)
    receipt = read_object(receipt_path, policy.parent_saved_row_verification.sha256)
    parent = validate_return_relationship_receipt(root, policy.parent_publication, receipt)
    _require(len(parent.model_columns) == 124 and parent.manifest["rows"] == EXPECTED_ROWS
             and len(parent.months) == EXPECTED_MONTHS, "reaction parent must be the complete frozen 124-column cohort")
    relationship_policy = ReturnRelationshipPublicationPolicy.model_validate(parent.request["policy"])
    base = verify_parent(root, relationship_policy)
    context = verify_source_context(root, relationship_policy, base)
    bars = relationship_sources(policy.parent_publication.sha256, context, parent.request["stock_inventory"])
    events, qualification_pins, historical = _qualification(root, policy)
    sources = IssuerReactionSources(bars=bars, event_authority_sha256=policy.qualification_authority.sha256,
        identity_authority_sha256=policy.qualification_authority.sha256,
        event_availability_semantics="historical_proxy", identity_availability_semantics="historical_proxy",
        event_availability_policy_sha256=json_sha256({"authority": policy.qualification_authority.model_dump(mode="json"),
            "field": "event_available_at_utc", "basis": "source_only_saved_qualification"}),
        identity_availability_policy_sha256=json_sha256({"authority": policy.qualification_authority.model_dump(mode="json"),
            "field": "identity_available_at_utc", "basis": "causal_source_bound_identity"}))
    files = pins(root, parent.source_files, qualification_pins, context.source_files,
        {policy.parent_saved_row_verification.path: policy.parent_saved_row_verification.sha256})
    check_files(root, files)
    provenance = {f"relationship_parent/{name}": digest
                  for name, digest in parent.request.get("historical_implementation_files", {}).items()}
    provenance.update({f"qualification_producer/{name}": digest for name, digest in historical.items()})
    return ReactionPublicationInputs(root, policy, parent, inside(root, policy.parent_publication.path), context,
        events, sources, files, provenance)


def load_parent_month(inputs: ReactionPublicationInputs, month: str) -> pd.DataFrame:
    record = inputs.parent.months[month]["profiles"]["technical_relationships"]
    path = inside(inputs.parent_path.parent, record["path"])
    frame, _ = load_canonical_artifact(path, expected_type="swing_return_relationships", allow_research=True)
    _require(file_sha256(path) == record["sha256"] and len(frame) == record["rows"]
             and json_sha256(sorted(frame.decision_id)) == record["decision_ids_sha256"], "reaction parent month changed")
    return frame


def iter_reaction_inputs(root: Path, inputs: ReactionPublicationInputs, month: str
                        ) -> Iterator[tuple[str, ResearchFeaturePartition, dict[str, Any]]]:
    """Read one exact ownership query at a time, retaining every parent decision."""
    parent = load_parent_month(inputs, month)
    bound = bind_adjusted_history_decisions(parent, inputs.context.bindings)
    spy = read_spy(inputs.context)
    history = tuple(value.date() for value in xcals.get_calendar("XNYS").sessions_in_range("2018-05-29", "2024-05-28"))
    for (identity, unit), group in bound.groupby(["security_id", "source_group"], sort=True):
        guard(inputs.policy.maximum_system_used_percent)
        key = json_sha256([identity, unit])
        item = inputs.parent.request["stock_inventory"].get(key)
        _require(item is not None, "reaction parent decision lacks exact source ownership")
        baseline_rows = parent.loc[parent.decision_id.isin(group.decision_id)].copy().reset_index(drop=True)
        upper = pd.to_datetime(baseline_rows.decision_time_utc, utc=True).max()
        # Retain the full prior history: an old original announcement can defeat
        # a newer duplicate even when the original is outside the lookback.
        events = pd.read_parquet(inputs.events_path, filters=[("security_id", "==", identity),
            ("event_available_at_utc", "<=", upper.to_pydatetime())])
        coverage = pd.DataFrame({"decision_id": baseline_rows.decision_id, "coverage_status": "unknown",
            "available_at_utc": pd.Series(pd.NaT, index=baseline_rows.index, dtype="datetime64[ns, UTC]")})
        baseline = ResearchFeaturePartition(baseline_rows, inputs.parent.model_columns, inputs.parent.availability_columns, {})
        yield key, baseline, {"qualified_events": events, "coverage": coverage,
            "stock_bars": read_stock(root, inputs.context, item), "spy_bars": spy,
            "history_sessions": history, "sources": inputs.sources, "purpose": "historical_research"}
    check_files(root, inputs.source_files)


def assert_parent_parity(parent: pd.DataFrame, child: pd.DataFrame) -> None:
    _require(not parent.decision_id.duplicated().any() and not child.decision_id.duplicated().any()
             and len(parent) == len(child) and list(child.columns[:len(parent.columns)]) == list(parent.columns)
             and parent.feature_profile.eq("technical_relationships").all() and child.feature_profile.eq(PROFILE).all(),
             "reaction parent population, prefix or profile differs")
    columns = [name for name in parent.columns if name != "feature_profile"]
    try:
        pd.testing.assert_frame_equal(parent[columns].reset_index(drop=True), child[columns].reset_index(drop=True), check_exact=True)
    except AssertionError as error:
        raise DataReadinessError("reaction changed inherited values, clocks, targets, eligibility or identities") from error


def assemble_reaction_month(parent: pd.DataFrame, groups: Sequence[ResearchFeaturePartition]) -> ResearchFeaturePartition:
    _require(bool(groups), "reaction month has no groups")
    first = groups[0]
    _require(all(group.model_columns == first.model_columns and group.availability_columns == first.availability_columns
                 and group.audit["profile_sha256"] == first.audit["profile_sha256"] for group in groups), "reaction groups disagree")
    rows = pd.concat([group.rows for group in groups], ignore_index=True)
    _require(not rows.decision_id.duplicated().any() and set(rows.decision_id) == set(parent.decision_id),
             "reaction groups lose or duplicate decisions")
    rows = rows.set_index("decision_id").loc[parent.decision_id].reset_index().loc[:, groups[0].rows.columns]
    assert_parent_parity(parent, rows)
    audit = {**first.audit, "rows": len(rows),
             "selected_event_rows": sum(cast(int, group.audit["selected_event_rows"]) for group in groups)}
    return ResearchFeaturePartition(rows, first.model_columns, first.availability_columns, audit)


def _request(root: Path, config: SourcePin, inputs: ReactionPublicationInputs) -> dict[str, Any]:
    current = current_implementation(root)
    names = (*inputs.parent.model_columns, *REACTION_COLUMNS)
    clocks = {**inputs.parent.availability_columns, **{name: f"available_at_{name}" for name in REACTION_COLUMNS}}
    return {"schema": REQUEST_SCHEMA, "config": config.model_dump(mode="json"), "policy": inputs.policy.model_dump(mode="json"),
        "cohort_sha256": inputs.parent.request["cohort_sha256"],
        "source_files": pins(root, inputs.source_files, current, {config.path: config.sha256}), "current_implementation_files": current,
        "historical_implementation_files": inputs.historical_implementation_files,
        "historical_implementation_disposition": "parent_recorded_producer_hashes_not_reexecuted",
        "baseline_numerical_replayed": False, "qualification_source_replayed": True,
        "qualification_replay_scope": "pinned_review_population_labels_metrics_all_version_dispositions_not_raw_source_extraction",
        "parent_publication": inputs.policy.parent_publication.model_dump(mode="json"),
        "parent_saved_row_verification": inputs.policy.parent_saved_row_verification.model_dump(mode="json"),
        "qualification_publication": inputs.policy.qualification_publication.model_dump(mode="json"),
        "qualification_authority": inputs.policy.qualification_authority.model_dump(mode="json"),
        "rows": inputs.parent.manifest["rows"], "profile": PROFILE, "model_columns": list(names), "availability_columns": clocks,
        "profile_sha256": issuer_reaction_profile_sha256(inputs.parent.model_columns, inputs.sources),
        "sources": inputs.sources.model_dump(mode="json"), "coverage": "unknown", "exclusions_added": [], **CLOSED}


def _child_files(folder: Path, record: dict[str, Any], request_sha256: str) -> dict[str, str]:
    path = inside(folder, record["path"])
    sidecar = manifest_path_for(path)
    files = {record["path"]: record["sha256"], sidecar.relative_to(folder).as_posix(): record["manifest_sha256"]}
    check_files(folder, files)
    metadata = read_object(sidecar, record["manifest_sha256"])
    _require(metadata.get("artifact_type") == ARTIFACT_TYPE and metadata.get("artifact_sha256") == record["sha256"]
             and metadata.get("rows") == record["rows"] and metadata.get("production_ready") is False
             and metadata.get("inputs") == {"request_sha256": request_sha256}, "reaction monthly sidecar differs")
    frame, _ = load_canonical_artifact(path, expected_type=ARTIFACT_TYPE, allow_research=True, columns=[])
    _require(len(frame) == record["rows"], "reaction physical row count differs")
    return files


def _state_files(folder: Path, state: dict[str, Any], request: dict[str, Any], digest: str,
                 inputs: ReactionPublicationInputs, *, complete: bool) -> dict[str, str]:
    assert_closed(state)
    _require(state.get("schema") == PUBLICATION_SCHEMA and state.get("request_sha256") == digest
             and state.get("status") == ("complete_research_only" if complete else "in_progress")
             and state.get("exclusions_added") == [] and isinstance(state.get("months"), dict), "reaction state differs")
    _require(set(state["months"]).issubset(inputs.parent.months), "reaction state has foreign months")
    files: dict[str, str] = {}
    rows = 0
    for month, record in state["months"].items():
        original = inputs.parent.months[month]
        _require(record["rows"] == original["rows"] and record["decision_ids_sha256"] == original["decision_ids_sha256"]
                 and set(record["profiles"]) == {PROFILE}, "reaction monthly population differs")
        child = record["profiles"][PROFILE]
        _require(child["path"] == f"{month}/{PROFILE}.parquet" and child["rows"] == record["rows"]
                 and child["decision_ids_sha256"] == record["decision_ids_sha256"]
                 and child["model_columns"] == request["model_columns"] and child["availability_columns"] == request["availability_columns"]
                 and child["audit"]["profile_sha256"] == request["profile_sha256"], "reaction child contract differs")
        assert_closed(child["audit"])
        files.update(_child_files(folder, child, digest))
        rows += record["rows"]
    _require(state.get("rows") == rows, "reaction state total differs")
    if complete:
        _require(set(state["months"]) == set(inputs.parent.months) and rows == request["rows"], "reaction publication incomplete")
    return files


def verify_issuer_reaction_publication(root: Path, publication: SourcePin) -> VerifiedIssuerReactionPublication:
    root = root.resolve()
    path = inside(root, publication.path)
    _require(path.name == "_manifest.json", "reaction requires a completed publication manifest")
    manifest = read_object(path, publication.sha256)
    request = read_object(path.parent / "_request.json", manifest["request_sha256"])
    config = SourcePin.model_validate(request["config"])
    inputs = load_reaction_inputs(root, load_policy(root, Path(config.path), config.sha256))
    _require(request == _request(root, config, inputs), "reaction request source or implementation differs")
    children = _state_files(path.parent, manifest, request, manifest["request_sha256"], inputs, complete=True)
    files = pins(root, request["source_files"],
        {inside(path.parent, name).relative_to(root).as_posix(): digest for name, digest in children.items()},
        {publication.path: publication.sha256, (path.parent / "_request.json").relative_to(root).as_posix(): manifest["request_sha256"]})
    check_files(root, files)
    return VerifiedIssuerReactionPublication(request, manifest, files, inputs.parent.manifest, inputs.parent_path,
        tuple(request["model_columns"]), dict(request["availability_columns"]), dict(manifest["months"]))


def materialize_issuer_reactions(root: Path, config: Path, expected_config_sha256: str, output: Path,
    expected_checkpoint_sha256: str | None = None, maximum_months_this_run: int | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    destination, config_path = inside(root, output), inside(root, config)
    _require(destination.is_relative_to(root / "data/features"), "reaction output must be beneath data/features")
    _require(maximum_months_this_run is None or type(maximum_months_this_run) is int and maximum_months_this_run > 0,
             "reaction month limit must be a positive integer")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("materialize-issuer-reactions", runtime_dir=runtime, config_path=config_path):
        policy = load_policy(root, config_path, expected_config_sha256)
        inputs = load_reaction_inputs(root, policy)
        config_pin = SourcePin(path=config_path.relative_to(root).as_posix(), sha256=expected_config_sha256)
        request = _request(root, config_pin, inputs)
        for name in request["source_files"]:
            source = inside(root, name)
            _require(not source.is_relative_to(destination) and not destination.is_relative_to(source), "reaction output overlaps source")
        if destination.exists():
            _require(expected_checkpoint_sha256 is not None, "reaction resume requires an external checkpoint pin")
            name = "_manifest.json" if (destination / "_manifest.json").exists() else "_checkpoint.json"
            state = read_object(destination / name, str(expected_checkpoint_sha256))
            digest = state["request_sha256"]
            _require(read_object(destination / "_request.json", digest) == request, "reaction resume request changed")
            _state_files(destination, state, request, digest, inputs, complete=name == "_manifest.json")
            if name == "_manifest.json":
                return {**state, "manifest_sha256": file_sha256(destination / name)}
        else:
            _require(expected_checkpoint_sha256 is None, "reaction missing pinned output cannot be recreated")
            destination.mkdir(parents=True)
            write_json_object(destination / "_request.json", request)
            digest = file_sha256(destination / "_request.json")
            state = {"schema": PUBLICATION_SCHEMA, "status": "in_progress", "request_sha256": digest,
                     "months": {}, "rows": 0, "exclusions_added": [], **CLOSED}
            replace_checkpoint(destination / "_checkpoint.json", state)
        count = 0
        for month in sorted(inputs.parent.months):
            if month in state["months"]:
                continue
            guard(policy.maximum_system_used_percent)
            parent = load_parent_month(inputs, month)
            groups = [build_issuer_reaction_profile(baseline=baseline, **kwargs)
                      for _, baseline, kwargs in iter_reaction_inputs(root, inputs, month)]
            built = assemble_reaction_month(parent, groups)
            relative = f"{month}/{PROFILE}.parquet"
            path = inside(destination, relative)
            _require(not path.exists() and not manifest_path_for(path).exists(), "reaction refuses to overwrite an uncheckpointed month")
            audit = CanonicalAuditReport(checks=[CanonicalAuditCheck(name="unchanged_parent_values", status="pass",
                rows_checked=len(built.rows), failures=0, detail="Exact inherited values, clocks, targets, eligibility and row order")])
            write_canonical_artifact(built.rows, path, artifact_type=ARTIFACT_TYPE, audit=audit,
                                     inputs={"request_sha256": digest}, production_ready=False)
            child = {"path": relative, "sha256": file_sha256(path), "manifest_sha256": file_sha256(manifest_path_for(path)),
                "rows": len(built.rows), "decision_ids_sha256": json_sha256(sorted(built.rows.decision_id)),
                "model_columns": list(built.model_columns), "availability_columns": built.availability_columns, "audit": built.audit}
            _child_files(destination, child, digest)
            state["months"][month] = {"rows": child["rows"], "decision_ids_sha256": child["decision_ids_sha256"],
                                      "profiles": {PROFILE: child}}
            state["rows"] += child["rows"]
            check_files(root, request["source_files"])
            replace_checkpoint(destination / "_checkpoint.json", state)
            count += 1
            del parent, groups, built
            release_process_memory()
            if maximum_months_this_run is not None and count >= maximum_months_this_run:
                break
        complete = set(state["months"]) == set(inputs.parent.months)
        if complete:
            state["status"] = "complete_research_only"
        _state_files(destination, state, request, digest, inputs, complete=complete)
        check_files(root, request["source_files"])
        _require(file_sha256(destination / "_request.json") == digest, "reaction request changed before publication")
        if complete:
            temporary = destination / f".manifest.{uuid4().hex}.pending"
            final = destination / "_manifest.json"
            try:
                write_json_object(temporary, state)
                if os.name == "nt":
                    os.rename(temporary, final)
                else:
                    os.link(temporary, final)
            finally:
                temporary.unlink(missing_ok=True)
        return {**state, "manifest_sha256" if complete else "checkpoint_sha256":
                file_sha256(destination / ("_manifest.json" if complete else "_checkpoint.json"))}
