"""Current-code numerical replay of immutable, explicitly derived predictors."""
from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.implementation_snapshot import verify_implementation_snapshot
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.datasets import predictor_abstention_derivation as derivation
from market_predictor.swing.datasets import research_features as features
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence
from market_predictor.swing.datasets.corrected_decisions import _metadata_implementation
from market_predictor.swing.datasets.corrected_outcomes import _implementation as _outcome_implementation
from market_predictor.swing.datasets.corrected_outcomes import load_corrected_outcome_policy, verified_corrected_research_sources
from market_predictor.swing.datasets.feature_history_plan import verify_feature_plan_replay
from market_predictor.swing.datasets.symbol_corrections import pinned_object

SCHEMA = "market_predictor.predictor_implementation_replay.v1"


@dataclass(frozen=True, slots=True)
class VerifiedPredictorReplay:
    historical_implementation_files: Mapping[str, str]
    live_evidence_files: Mapping[str, str]


def _corrected_child_pins(root: Path, policy: ResearchFeaturePolicy) -> dict[str, str]:
    authority_path = inside(root, policy.adjusted_archive_authority.path)
    authority = pinned_object(authority_path, policy.adjusted_archive_authority.sha256)
    path = authority_path.parent / "_manifest.json"
    manifest = pinned_object(path, authority["artifact_sha256"])
    pins = {path.relative_to(root).as_posix(): authority["artifact_sha256"]}
    for record in manifest["unit_artifacts"]:
        child = inside(authority_path.parent, record["bars_path"])
        pins = derivation._merge(root, pins, {child.relative_to(root).as_posix(): record["bars_sha256"]})
    return pins


def _split_historical_pins(root: Path, files: Mapping[str, str], declarations: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Only exact independently declared implementation identities leave live checks."""
    bound = derivation._merge(root, files)
    historical = derivation._merge(root, declarations)
    if not historical or any(bound.get(name) != digest for name, digest in historical.items()):
        raise DataReadinessError("historical implementation declarations differ from publication pins")
    return {name: digest for name, digest in bound.items() if name not in historical}, historical


def _feature_plan_evidence(root: Path, policy: ResearchFeaturePolicy, snapshot: SourcePin | None) -> dict[str, str]:
    if snapshot is None:
        return {}
    return verify_feature_plan_replay(root=root, authority=policy.adjusted_plan_authority, implementation_snapshot=snapshot)


def _current_implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    extra = ("swing/datasets/predictor_replay.py", "swing/datasets/predictor_abstention_derivation.py",
        "swing/datasets/feature_history_plan.py", "swing/datasets/history_plan_publication.py",
        "canonical/store.py", "evidence/hashing.py", "heavy_jobs.py", "resources.py", "core/system_memory.py")
    return derivation._merge(root, features.predictor_implementation_files(root), _outcome_implementation(root),
        _metadata_implementation(root), {str((package / name).relative_to(root)): file_sha256(package / name) for name in extra})


def _published_files(root: Path, directory: Path, manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    paths: set[Path] = set()
    for kind in ("groups", "months"):
        if not isinstance(manifest.get(kind), dict) or not manifest[kind]:
            raise DataReadinessError("predictor replay requires complete group and month inventories")
        for part in manifest[kind].values():
            path = inside(directory / kind, part["path"])
            if path in paths:
                raise DataReadinessError("predictor publication aliases partition paths")
            paths.add(path)
            result = derivation._merge(root, result, {path.relative_to(root).as_posix(): part["sha256"],
                manifest_path_for(path).relative_to(root).as_posix(): part["manifest_sha256"]})
    return result


def _load_publication(*, root: Path, publication: SourcePin, migration_bindings: SourcePin,
    implementation_snapshot: SourcePin,
) -> tuple[dict[str, Any], dict[str, Any], derivation.PredictorFailureFacts, dict[str, str], dict[str, str]]:
    path = inside(root, publication.path)
    if path.name != "_manifest.json" or not path.is_relative_to(root / "data/features"):
        raise DataReadinessError("predictor replay requires a completed feature publication")
    manifest = pinned_object(path, publication.sha256)
    if (manifest.get("schema") != "market_predictor.research_predictors"
            or manifest.get("status") != "technical_inputs_complete_research_only"
            or manifest.get("failed_groups") != {} or manifest.get("training_eligible") is not False
            or manifest.get("promotion_eligible") is not False or manifest.get("exclusions_added") != []
            or not isinstance(manifest.get("derivation"), dict)):
        raise DataReadinessError("predictor replay publication contract differs")
    request_path = path.parent / "_request.json"
    request = pinned_object(request_path, manifest["request_sha256"])
    inherited = derivation.validate_historical_predictor_derivation(root=root, manifest=manifest, request=request)
    facts_pin = SourcePin.model_validate(manifest["derivation"]["approved_failure_facts"])
    facts = derivation.PredictorFailureFacts.model_validate_json(
        json.dumps(pinned_object(inside(root, facts_pin.path), facts_pin.sha256)))
    binding = pinned_object(inside(root, migration_bindings.path), migration_bindings.sha256)
    if (binding.get("schema") != "market_predictor.archive_owner_migration_bindings"
            or binding.get("purpose") != "historical_implementation_provenance_not_numerical_replay"
            or binding.get("snapshot") != {"manifest_path": inside(root, implementation_snapshot.path).relative_to(root).as_posix(),
                "manifest_sha256": implementation_snapshot.sha256}):
        raise DataReadinessError("predictor migration snapshot binding differs")
    record = binding["publications"].get(path.relative_to(root).as_posix(), {})
    if (record.get("manifest_sha256") != publication.sha256
            or record.get("request") != request_path.relative_to(root).as_posix()
            or record.get("request_sha256") != manifest["request_sha256"]):
        raise DataReadinessError("predictor migration publication binding differs")
    all_files = derivation._merge(root, inherited, *(request[name] for name in derivation.PIN_MAPS))
    live, historical = _split_historical_pins(root, all_files, record["implementation_files"])
    explicit = derivation._merge(root, request["implementation_files"])
    if any(historical.get(name) != digest for name, digest in explicit.items()):
        raise DataReadinessError("snapshot binding omits a declared predictor implementation")
    archived = verify_implementation_snapshot(root, inside(root, implementation_snapshot.path),
        implementation_snapshot.sha256, historical)
    live = derivation._merge(root, live, archived, _published_files(root, path.parent, manifest),
        {publication.path: publication.sha256, migration_bindings.path: migration_bindings.sha256},
        {request_path.relative_to(root).as_posix(): manifest["request_sha256"]})
    return manifest, request, facts, live, historical


def _check_fresh_sources(root: Path, fresh: Mapping[str, str], live: Mapping[str, str], current: Mapping[str, str]) -> None:
    allowed = derivation._merge(root, live, current)
    if any(allowed.get(name) != digest for name, digest in derivation._merge(root, fresh).items()):
        raise DataReadinessError("predictor replay source identity differs from original evidence")


def _compare_partition(directory: Path, part: dict[str, Any], request_pin: str,
    expected: pd.DataFrame, population: pd.DataFrame,
) -> None:
    features._verify_part(directory, part, request_pin)
    features._check_population(population, expected)
    if (part["rows"] != len(expected) or part["decision_ids_sha256"] != json_sha256(sorted(expected.decision_id))
            or part["feature_eligible_rows"] != int(expected.feature_eligible.sum())):
        raise DataReadinessError("predictor replay partition counts differ")
    saved, child = load_canonical_artifact(inside(directory, part["path"]),
        expected_type="swing_research_technical_inputs", allow_research=True)
    if child["production_ready"] is not False:
        raise DataReadinessError("predictor replay child claims production readiness")
    features._check_population(population, saved)
    # Compare the publisher's persisted representation, not transient pandas
    # object/string containers. Numeric widths and clock types remain strict.
    with BytesIO() as serialized:
        expected.to_parquet(serialized, index=False)
        serialized.seek(0)
        persisted = pd.read_parquet(serialized)
    try:
        pd.testing.assert_frame_equal(saved.sort_values("decision_id").reset_index(drop=True),
            persisted.sort_values("decision_id").reset_index(drop=True), check_exact=True, check_dtype=True, check_like=False)
    except AssertionError as error:
        raise DataReadinessError(f"predictor numerical replay differs: {part['path']}") from error
    features._verify_part(directory, part, request_pin)


def _replay_population(context: features.PredictorSourceContext, manifest: dict[str, Any], request: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    decisions = context.decisions
    groups = {json_sha256([identity, symbol]): group for (identity, symbol), group in
        decisions.groupby(["security_id", "source_group"], sort=True)}
    months = set(decisions.session_date_et.map(lambda day: day.strftime("%Y-%m")))
    source_months = [record["partition_month"] for record in context.records]
    if (len(source_months) != len(set(source_months)) or set(source_months) != months
            or set(groups) != set(manifest["groups"]) or set(manifest["months"]) != months
            or decisions.decision_id.duplicated().any() or len(decisions) != request["expected_rows"]
            or len(decisions) != manifest["rows"]
            or request["decision_ids_sha256"] != json_sha256(sorted(decisions.decision_id))
            or request["cohort_sha256"] != context.sources["request"]["cohort_sha256"]
            or request["retained_security_ids"] != context.sources["request"]["retained_security_ids"]):
        raise DataReadinessError("predictor replay differs from the exhaustive frozen population")
    return groups


def _check_progress(output: Path, report: dict[str, Any]) -> None:
    try:
        saved = pinned_object(output / "_progress.json")
    except (ValueError, OSError) as error:
        raise DataReadinessError("predictor replay progress is unreadable") from error
    if saved != report:
        raise DataReadinessError("predictor replay progress changed")


def _stage_group(root: Path, directory: Path, output: Path, key: str, frame: pd.DataFrame,
    population: pd.DataFrame, manifest: dict[str, Any], report: dict[str, Any], scratch: dict[str, str],
) -> None:
    features._guard()
    _check_progress(output, report)
    part = manifest["groups"][key]
    _compare_partition(directory / "groups", part, part.get("origin_request_sha256", manifest["request_sha256"]),
        frame, population)
    path = output / "reconstructed_groups" / f"{key}.parquet"
    path.parent.mkdir(exist_ok=True)
    frame.reset_index(drop=True).to_parquet(path, index=False)
    scratch[path.relative_to(root).as_posix()] = file_sha256(path)
    report["compared_groups"][key] = part
    features._write(output / "_progress.json", report)


def _compare_months(root: Path, directory: Path, output: Path, decisions: pd.DataFrame,
    manifest: dict[str, Any], report: dict[str, Any], scratch: dict[str, str],
) -> None:
    features._pins(root, scratch)
    paths = [str(inside(root, name)) for name in scratch]
    parquet: Any = pq
    arrow: Any = pds
    schema = pa.unify_schemas([parquet.read_schema(path) for path in paths], promote_options="permissive")
    dataset = arrow.dataset(paths, format="parquet", schema=schema)
    for month, population in decisions.groupby(decisions.session_date_et.map(lambda day: day.strftime("%Y-%m")), sort=True):
        features._guard()
        _check_progress(output, report)
        frame = dataset.to_table(filter=arrow.field("session_date_et").isin(list(population.session_date_et.unique())),
            use_threads=False).to_pandas()
        _compare_partition(directory / "months", manifest["months"][month], manifest["request_sha256"], frame, population)
        report["compared_months"][month] = manifest["months"][month]
        features._write(output / "_progress.json", report)
        del frame
        release_process_memory()
    features._pins(root, scratch)


def replay_predictor_publication(*, root: Path, publication: SourcePin, migration_bindings: SourcePin,
    implementation_snapshot: SourcePin, output: Path, maximum_groups: int | None = None,
    feature_plan_snapshot: SourcePin | None = None,
) -> dict[str, Any]:
    """Rebuild every group and month under current code; partial runs never complete.

    Original artifacts and source pins remain immutable. A new report directory is
    required on every invocation, including after interrupted or partial work.
    """
    root = root.resolve()
    output = inside(root, output)
    if output == root / "data/reports" or not output.is_relative_to(root / "data/reports") or output.exists():
        raise DataReadinessError("predictor replay requires a new directory below data/reports")
    if maximum_groups is not None and (type(maximum_groups) is not int or maximum_groups < 1):
        raise DataReadinessError("predictor replay group limit must be a positive integer")
    runtime = inside(root, heavy_job_runtime_dir())
    with heavy_job_lease("predictor-replay-metadata", runtime_dir=runtime):
        features._guard()
        current = _current_implementation(root)
        manifest, request, facts, live, historical = _load_publication(root=root, publication=publication,
            migration_bindings=migration_bindings, implementation_snapshot=implementation_snapshot)
        files = derivation._merge(root, live, current)
        features._pins(root, files)
        config = inside(root, facts.feature_config.path)
        if config.stat().st_size > 65536:
            raise DataReadinessError("predictor replay feature policy exceeds byte bound")
        policy = ResearchFeaturePolicy.model_validate(tomllib.loads(config.read_text(encoding="utf-8")))
        files = derivation._merge(root, files, _feature_plan_evidence(root, policy, feature_plan_snapshot))
        features._pins(root, files)
        if policy.outcome_source_config != facts.decision_config:
            raise DataReadinessError("predictor replay decision policy differs")
        declared = derivation._merge(root, request["declared_source_files"])
        for name in type(policy).model_fields:
            if name != "schema_version":
                pin = getattr(policy, name)
                if declared.get(inside(root, pin.path).relative_to(root).as_posix()) != pin.sha256:
                    raise DataReadinessError("predictor replay feature policy dependency differs")
        clocks = derivation._clocks(manifest["groups"])
        comparison_request = {"schema": SCHEMA + ".request", "publication": publication.model_dump(mode="json"),
            "original_request_sha256": manifest["request_sha256"],
            "migration_bindings": migration_bindings.model_dump(mode="json"),
            "implementation_snapshot": implementation_snapshot.model_dump(mode="json"),
            "feature_plan_snapshot": None if feature_plan_snapshot is None else feature_plan_snapshot.model_dump(mode="json"),
            "historical_implementation_files": historical, "current_implementation_files": current,
            "source_files": live, "comparison": "exact_all_columns_dtypes_nulls_groups_and_months",
            "maximum_groups": maximum_groups, "resume_policy": "new_directory_full_reconstruction_only"}
        output.mkdir(parents=True)
        write_json_object(output / "_request.json", comparison_request)
        report: dict[str, Any] = {"schema": SCHEMA, "request_sha256": file_sha256(output / "_request.json"),
            "status": "partial", "replay_complete": False, "compared_groups": {}, "compared_months": {},
            "training_eligible": False, "promotion_eligible": False, "exclusions_added": []}
        features._write(output / "_progress.json", report)
    directory = inside(root, publication.path).parent
    source_config = inside(root, facts.decision_config.path)
    source_policy = load_corrected_outcome_policy(root, source_config, facts.decision_config.sha256)
    evidence = load_corporate_action_evidence(root=root, config=inside(root, source_policy.action_config.path),
        archive=inside(root, source_policy.action_archive), expected_audit_sha256=source_policy.action_audit_sha256)
    scratch: dict[str, str] = {}
    failures = {fact.group_key: fact for fact in facts.failures}
    with verified_corrected_research_sources(root, source_config, facts.decision_config.sha256,
        source_policy, evidence) as sources:
        features._pins(root, files)
        _check_fresh_sources(root, sources["source_files"], live, current)
        context = features.prepare_predictor_sources(root=root, policy=policy, source_policy=source_policy, sources=sources,
            feature_plan_snapshot=feature_plan_snapshot)
        if derivation._merge(root, context.adjusted_source_files) != derivation._merge(root, request["adjusted_source_files"]):
            raise DataReadinessError("predictor replay adjusted inventory differs")
        # These child bytes are transitively bound by the verified adjusted authority.
        files = derivation._merge(root, files, _corrected_child_pins(root, policy))
        expected = _replay_population(context, manifest, request)
        decisions = context.decisions
        for key, group in expected.items():
            if key in failures:
                if manifest["groups"][key]["completion_scope"] != derivation._completion_scope(group, failures[key]):
                    raise DataReadinessError("predictor replay failure boundary population differs")
                continue
            if maximum_groups is not None and len(report["compared_groups"]) >= maximum_groups:
                break
            features._guard()
            built = features.build_predictor_group(group, context)
            if dict(built.availability_columns) != clocks:
                raise DataReadinessError("predictor replay availability map differs")
            _stage_group(root, directory, output, key, built.rows, group, manifest, report, scratch)
            features._pins(root, current)
            del built
            release_process_memory()
        features._pins(root, files)
        del context
    remaining = len(failures) if maximum_groups is None else max(0, maximum_groups - len(report["compared_groups"]))
    selected = tuple(fact for fact in facts.failures)[:remaining]
    recovery_facts = facts.model_copy(update={"failures": selected})
    recovered, recovery_files = derivation._recover_prefixes(root=root, facts=recovery_facts,
        expected=expected, files=files, clocks=clocks)
    _check_fresh_sources(root, recovery_files, files, current)
    # Publication follows both verified source contexts, never their unvalidated yield.
    with heavy_job_lease("predictor-replay-finalization", runtime_dir=runtime):
        features._guard()
        features._pins(root, files)
        features._pins(root, scratch)
        for fact in selected:
            group = expected[fact.group_key]
            frame = (recovered.pop(fact.group_key) if fact.quarantine == "suffix_from_first_invalid"
                else derivation._unavailable(group, fact, clocks))
            _stage_group(root, directory, output, fact.group_key, frame, group, manifest, report, scratch)
            del frame
            release_process_memory()
        if set(report["compared_groups"]) == set(expected):
            _compare_months(root, directory, output, decisions, manifest, report, scratch)
            if (sum(part["rows"] for part in report["compared_groups"].values()) != manifest["rows"]
                    or sum(part["rows"] for part in report["compared_months"].values()) != manifest["rows"]
                    or sum(part["feature_eligible_rows"] for part in report["compared_groups"].values())
                        != manifest["feature_eligible_rows"]
                    or sum(part["feature_eligible_rows"] for part in report["compared_months"].values())
                        != manifest["feature_eligible_rows"]):
                raise DataReadinessError("predictor replay aggregate population differs")
        features._pins(root, files)
        features._pins(root, scratch)
        evidence.recheck(root)
        if pinned_object(output / "_request.json", report["request_sha256"]) != comparison_request:
            raise DataReadinessError("predictor replay request changed before finalization")
        _check_progress(output, report)
        if set(report["compared_groups"]) == set(expected) and set(report["compared_months"]) == set(manifest["months"]):
            report.update(status="exact_replay_complete", replay_complete=True, rows=manifest["rows"],
                feature_eligible_rows=manifest["feature_eligible_rows"], rebuilt_prefix_rows=manifest["rebuilt_prefix_rows"],
                unavailable_rows=manifest["unavailable_rows"])
        report["reconstructed_files"] = scratch
        report["verified_source_files"] = files
        write_json_object(output / "_checkpoint.json", report)
        if report["replay_complete"]:
            write_json_object(output / "_manifest.json", report)
    return {**report, "checkpoint_sha256": file_sha256(output / "_checkpoint.json")}


def verify_predictor_replay(*, root: Path, publication: SourcePin, replay: SourcePin) -> VerifiedPredictorReplay:
    """Verify a pinned complete receipt and historical ancestry, without numerical execution.

    Consumers partition only exact name/digest pairs in historical_implementation_files.
    Every live_evidence_files entry remains subject to their ordinary source rechecks.
    """
    root = root.resolve()
    path = inside(root, replay.path)
    if path.name != "_manifest.json" or not path.is_relative_to(root / "data/reports"):
        raise DataReadinessError("predictor replay verification requires a final report manifest")
    report = pinned_object(path, replay.sha256)
    request_path = path.parent / "_request.json"
    comparison = pinned_object(request_path, report["request_sha256"])
    if (comparison.get("schema") != SCHEMA + ".request" or comparison.get("publication") != publication.model_dump(mode="json")
            or comparison.get("comparison") != "exact_all_columns_dtypes_nulls_groups_and_months"
            or comparison.get("resume_policy") != "new_directory_full_reconstruction_only"
            or report.get("schema") != SCHEMA or report.get("status") != "exact_replay_complete"
            or report.get("replay_complete") is not True or report.get("training_eligible") is not False
            or report.get("promotion_eligible") is not False or report.get("exclusions_added") != []):
        raise DataReadinessError("predictor replay is incomplete or belongs to another publication")
    manifest, request, facts, live, historical = _load_publication(root=root, publication=publication,
        migration_bindings=SourcePin.model_validate(comparison["migration_bindings"]),
        implementation_snapshot=SourcePin.model_validate(comparison["implementation_snapshot"]))
    current = _current_implementation(root)
    if (comparison.get("original_request_sha256") != manifest["request_sha256"]
            or comparison.get("historical_implementation_files") != historical or comparison.get("source_files") != live
            or comparison.get("current_implementation_files") != current
            or report.get("compared_groups") != manifest["groups"] or report.get("compared_months") != manifest["months"]
            or any(report.get(name) != manifest[name] for name in
                ("rows", "feature_eligible_rows", "rebuilt_prefix_rows", "unavailable_rows"))):
        raise DataReadinessError("predictor replay coverage, source or implementation identity differs")
    config = inside(root, facts.feature_config.path)
    if config.stat().st_size > 65536:
        raise DataReadinessError("predictor replay feature policy exceeds byte bound")
    policy = ResearchFeaturePolicy.model_validate(tomllib.loads(config.read_text(encoding="utf-8")))
    if "feature_plan_snapshot" not in comparison:
        raise DataReadinessError("predictor replay lacks an explicit feature-plan evidence declaration")
    plan_snapshot = (None if comparison["feature_plan_snapshot"] is None
        else SourcePin.model_validate(comparison["feature_plan_snapshot"]))
    files = derivation._merge(root, live, current, _corrected_child_pins(root, policy),
        _feature_plan_evidence(root, policy, plan_snapshot))
    if report.get("verified_source_files") != files:
        raise DataReadinessError("predictor replay verified evidence inventory differs")
    scratch = derivation._merge(root, report["reconstructed_files"])
    expected_paths = {(path.parent / "reconstructed_groups" / f"{key}.parquet").relative_to(root).as_posix()
        for key in manifest["groups"]}
    if set(scratch) != expected_paths:
        raise DataReadinessError("predictor replay reconstructed group inventory differs")
    checkpoint_path = path.parent / "_checkpoint.json"
    checkpoint_pin = file_sha256(checkpoint_path)
    if pinned_object(checkpoint_path, checkpoint_pin) != report:
        raise DataReadinessError("predictor replay finalized checkpoint differs")
    files = derivation._merge(root, files, scratch, {replay.path: replay.sha256,
        request_path.relative_to(root).as_posix(): report["request_sha256"],
        checkpoint_path.relative_to(root).as_posix(): checkpoint_pin})
    features._pins(root, files)
    return VerifiedPredictorReplay(MappingProxyType(dict(historical)), MappingProxyType(files))
