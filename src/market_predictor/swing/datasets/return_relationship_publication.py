"""Immutable monthly relationship derivatives of the original initial-fit baseline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS, return_relationship_profile_sha256
from market_predictor.swing.contracts.return_relationship_publication import (
    ARTIFACT_TYPE,
    PROFILE,
    PUBLICATION_SCHEMA,
    REQUEST_SCHEMA,
    ReturnRelationshipPublicationPolicy,
    VerifiedReturnRelationshipPublication,
)
from market_predictor.swing.datasets.return_relationship_integrity import (
    assert_closed,
    check_files,
    current_implementation,
    guard,
    load_policy,
    pins,
    read_object,
    replace_checkpoint,
)
from market_predictor.swing.datasets.return_relationship_parent import VerifiedRelationshipParent, load_parent_month, verify_parent
from market_predictor.swing.datasets.return_relationship_rows import (
    IDENTITY_COLUMNS,
    assemble_month,
    assert_parent_parity,
    build_group,
    read_combined,
)
from market_predictor.swing.datasets.return_relationship_sources import (
    RelationshipSourceContext,
    inventory_pins,
    relationship_sources,
    stock_inventory,
    verify_source_context,
)
from market_predictor.swing.datasets.return_relationship_storage import (
    month_additions,
    publish_rows,
    stage_baseline,
    staged_baseline,
    verify_child,
)

CLOSED = {"training_eligible": False, "promotion_eligible": False, "serving_eligible": False}


def _validate_inventory(inventory: dict[str, dict[str, Any]], context: RelationshipSourceContext) -> None:
    if set(inventory) != set(context.predictor_manifest["groups"]):
        raise DataReadinessError("relationship inventory does not cover complete saved ownership")
    failures = {fact.security_id: fact.model_dump(mode="json") for fact in context.facts.failures}
    for key, item in inventory.items():
        identity, symbol = item["security_id"], item["source_group"]
        corrected = context.corrected.get(identity)
        expected = {"security_id": identity, "source_group": symbol,
            "rows": context.predictor_manifest["groups"][key]["rows"],
            "decision_ids_sha256": context.predictor_manifest["groups"][key]["decision_ids_sha256"],
            "kind": "corrected" if corrected is not None else "combined",
            "artifact": corrected if corrected is not None else context.combined.get(symbol),
            "quarantine": failures.get(identity)}
        if key != json_sha256([identity, symbol]) or item != expected or expected["artifact"] is None:
            raise DataReadinessError("relationship inventory source, ownership or quarantine differs")


def _request(root: Path, config: SourcePin, policy: ReturnRelationshipPublicationPolicy,
    parent: VerifiedRelationshipParent, context: RelationshipSourceContext,
    inventory: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    _validate_inventory(inventory, context)
    sources = relationship_sources(policy.parent_publication.sha256, context, inventory)
    current = current_implementation(root)
    files = pins(root, parent.source_files, inventory_pins(root, context, inventory), current,
        {config.path: config.sha256})
    names = (*parent.model_columns, *RETURN_RELATIONSHIP_COLUMNS)
    clocks = {**parent.availability_columns, **{name: f"available_at_{name}" for name in RETURN_RELATIONSHIP_COLUMNS}}
    return {"schema": REQUEST_SCHEMA, "config": config.model_dump(mode="json"), "policy": policy.model_dump(mode="json"),
        "cohort_sha256": parent.request["cohort_sha256"], "source_files": files,
        "current_implementation_files": current, "historical_implementation_files": parent.historical_implementation_files,
        "historical_implementation_disposition": "historical_code_not_reexecuted_or_certified",
        "baseline_numerical_evidence": "original_saved_row_receipt_not_fresh_baseline_replay",
        "parent_publication": policy.parent_publication.model_dump(mode="json"),
        "parent_request_sha256": parent.manifest["request_sha256"],
        "parent_saved_row_verification": policy.parent_saved_row_verification.model_dump(mode="json"),
        "rows": parent.manifest["rows"], "stock_inventory": inventory, "sources": sources.model_dump(mode="json"),
        "source_basis": context.basis, "profile": PROFILE, "model_columns": list(names), "availability_columns": clocks,
        "profile_sha256": return_relationship_profile_sha256(parent.model_columns, sources),
        "source_start": policy.source_start, "source_end": policy.source_end,
        "decision_start": policy.decision_start, "decision_end": policy.decision_end,
        "allowed_parent_column_changes": {"feature_profile": {"from": "technical_market", "to": PROFILE}},
        "exclusions_added": [], "historical_first_seen_proven": False, **CLOSED}


def _verify_state(output: Path, state: dict[str, Any], request: dict[str, Any], request_sha256: str,
    parent: VerifiedRelationshipParent, *, complete: bool,
) -> dict[str, str]:
    assert_closed(state)
    if (state.get("schema") != PUBLICATION_SCHEMA or state.get("request_sha256") != request_sha256
            or state.get("status") != ("complete_research_only" if complete else "in_progress")
            or state.get("exclusions_added") != [] or not isinstance(state.get("groups"), dict)
            or not isinstance(state.get("months"), dict) or not set(state["groups"]).issubset(request["stock_inventory"])
            or not set(state["months"]).issubset(parent.manifest["months"])):
        raise DataReadinessError("relationship publication state or request binding differs")
    files: dict[str, str] = {}
    if state.get("baseline_stage") is not None:
        stage = state["baseline_stage"]
        if stage["path"] != "_baseline_stage/_manifest.json":
            raise DataReadinessError("relationship stage manifest path differs")
        stage_manifest = read_object(inside(output, stage["path"]), stage["sha256"])
        if (stage_manifest.get("request_sha256") != request_sha256
                or stage_manifest.get("inventory_sha256") != json_sha256(request["stock_inventory"])
                or stage_manifest.get("bucket_count") != 8):
            raise DataReadinessError("relationship stage lineage differs")
        files[stage["path"]] = stage["sha256"]
        for name, digest in stage_manifest["files"].items():
            if name not in {f"bucket-{number}.parquet" for number in range(8)}:
                raise DataReadinessError("relationship stage bucket path differs")
            files[f"_baseline_stage/{name}"] = digest
        check_files(output, files)
    elif state["groups"] or complete:
        raise DataReadinessError("relationship groups lack checkpoint-bound baseline staging")
    group_rows = 0
    for key, record in state["groups"].items():
        item = request["stock_inventory"][key]
        if (record["path"] != f"groups/{key}.parquet" or record["rows"] != item["rows"]
                or record["decision_ids_sha256"] != item["decision_ids_sha256"]):
            raise DataReadinessError("relationship staged group inventory differs")
        files.update(verify_child(output, record, request_sha256, group=True))
        group_rows += record["rows"]
    rows = 0
    for month, record in state["months"].items():
        original = parent.manifest["months"][month]
        if (record["rows"] != original["rows"] or record["decision_ids_sha256"] != original["decision_ids_sha256"]
                or set(record["profiles"]) != {PROFILE}):
            raise DataReadinessError("relationship monthly population differs from parent")
        child = record["profiles"][PROFILE]
        if (child["path"] != f"{month}/{PROFILE}.parquet" or child["rows"] != record["rows"]
                or child["decision_ids_sha256"] != record["decision_ids_sha256"]
                or child["model_columns"] != request["model_columns"]
                or child["availability_columns"] != request["availability_columns"]
                or child["audit"].get("profile_sha256") != request["profile_sha256"]):
            raise DataReadinessError("relationship monthly feature contract differs")
        assert_closed(child["audit"])
        files.update(verify_child(output, child, request_sha256))
        rows += record["rows"]
    if state.get("rows") != (rows if complete else group_rows):
        raise DataReadinessError("relationship publication row count differs")
    if complete and (set(state["months"]) != set(parent.manifest["months"])
            or set(state["groups"]) != set(request["stock_inventory"]) or rows != request["rows"] or group_rows != rows):
        raise DataReadinessError("relationship final publication is incomplete")
    return files


def verify_return_relationship_publication(root: Path, publication: SourcePin) -> VerifiedReturnRelationshipPublication:
    """Recheck current sources/authorities and immutable lineage; acquire no lease."""
    root = root.resolve()
    path = inside(root, publication.path)
    if path.name != "_manifest.json":
        raise DataReadinessError("relationship verification requires a completed publication manifest")
    manifest = read_object(path, publication.sha256)
    request_path = path.parent / "_request.json"
    request = read_object(request_path, manifest["request_sha256"])
    if request.get("schema") != REQUEST_SCHEMA:
        raise DataReadinessError("unsupported relationship publication request")
    config = SourcePin.model_validate(request["config"])
    policy = load_policy(root, Path(config.path), config.sha256)
    guard(policy.maximum_system_used_percent)
    parent = verify_parent(root, policy)
    context = verify_source_context(root, policy, parent)
    expected = _request(root, config, policy, parent, context, request["stock_inventory"])
    if expected != request:
        raise DataReadinessError("relationship publication source, implementation or contract changed")
    children = _verify_state(path.parent, manifest, request, manifest["request_sha256"], parent, complete=True)
    files = pins(root, request["source_files"],
        {inside(path.parent, name).relative_to(root).as_posix(): digest for name, digest in children.items()},
        {publication.path: publication.sha256, request_path.relative_to(root).as_posix(): manifest["request_sha256"]})
    check_files(root, files)
    return VerifiedReturnRelationshipPublication(request, manifest, files, parent.manifest, parent.path,
        tuple(request["model_columns"]), dict(request["availability_columns"]), dict(manifest["months"]))


def _summary(output: Path, state: dict[str, Any]) -> dict[str, Any]:
    result = {"status": state["status"], "rows": state["rows"], **CLOSED}
    for basename, key in (("_manifest.json", "manifest_sha256"), ("_checkpoint.json", "checkpoint_sha256")):
        if (output / basename).is_file():
            result[key] = file_sha256(output / basename)
    return result


def materialize_return_relationships(root: Path, config: Path, expected_config_sha256: str, output: Path,
    expected_checkpoint_sha256: str | None = None, maximum_groups_this_run: int | None = None,
) -> dict[str, Any]:
    """One lease, bounded issuer staging, unchanged saved baseline plus four inputs."""
    root = root.resolve()
    config_path, destination = inside(root, config), inside(root, output)
    if maximum_groups_this_run is not None and (type(maximum_groups_this_run) is not int or maximum_groups_this_run < 1):
        raise ValueError("maximum_groups_this_run must be a positive integer")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("materialize-swing-return-relationships", runtime_dir=runtime, config_path=config_path):
        policy = load_policy(root, config_path, expected_config_sha256)
        guard(policy.maximum_system_used_percent)
        parent = verify_parent(root, policy)
        context = verify_source_context(root, policy, parent)
        population = pd.concat([load_parent_month(parent, month, IDENTITY_COLUMNS)
            for month in sorted(parent.manifest["months"])], ignore_index=True)
        if population.decision_id.duplicated().any() or len(population) != parent.manifest["rows"]:
            raise DataReadinessError("relationship parent population is duplicated or incomplete")
        inventory = stock_inventory(population, context)
        del population
        config_pin = SourcePin(path=config_path.relative_to(root).as_posix(), sha256=expected_config_sha256)
        request = _request(root, config_pin, policy, parent, context, inventory)
        for name in request["source_files"]:
            source_path = inside(root, name)
            if source_path.is_relative_to(destination) or destination.is_relative_to(source_path):
                raise DataReadinessError("relationship output overlaps an immutable input")
        return _materialize(root, destination, policy, parent, context, request,
            expected_checkpoint_sha256, maximum_groups_this_run)


def _materialize(root: Path, output: Path, policy: ReturnRelationshipPublicationPolicy,
    parent: VerifiedRelationshipParent, context: RelationshipSourceContext, request: dict[str, Any],
    checkpoint_sha256: str | None, maximum_groups: int | None,
) -> dict[str, Any]:
    request_path, checkpoint_path = output / "_request.json", output / "_checkpoint.json"
    if output.exists():
        if not request_path.is_file() or not checkpoint_path.is_file() or checkpoint_sha256 is None:
            raise DataReadinessError("relationship resume requires an independently pinned checkpoint")
        state = read_object(checkpoint_path, checkpoint_sha256)
        request_pin = state["request_sha256"]
        if read_object(request_path, request_pin) != request:
            raise DataReadinessError("relationship resume sources or implementation changed")
        if (output / "_manifest.json").exists():
            final = read_object(output / "_manifest.json", file_sha256(output / "_manifest.json"))
            if {**state, "status": "complete_research_only"} != final:
                raise DataReadinessError("relationship final manifest differs from checkpoint")
            _verify_state(output, final, request, request_pin, parent, complete=True)
            check_files(root, request["source_files"])
            return _summary(output, final)
        _verify_state(output, state, request, request_pin, parent, complete=False)
    else:
        if checkpoint_sha256 is not None:
            raise DataReadinessError("relationship checkpoint supplied for a new publication")
        output.mkdir(parents=True)
        write_json_object(request_path, request)
        request_pin = file_sha256(request_path)
        state = {"schema": PUBLICATION_SCHEMA, "status": "in_progress", "request_sha256": request_pin,
            "groups": {}, "months": {}, "rows": 0, "exclusions_added": [], **CLOSED}
        replace_checkpoint(checkpoint_path, state)
    source_bindings = relationship_sources(policy.parent_publication.sha256, context, request["stock_inventory"])
    spy = read_combined(context.combined_directory, context.combined["SPY"])
    stages = stage_baseline(output, parent, request["stock_inventory"], request_pin,
        lambda: guard(policy.maximum_system_used_percent), expected=state.get("baseline_stage"))
    state["baseline_stage"] = stages
    replace_checkpoint(checkpoint_path, state)
    count = 0
    for key, item in request["stock_inventory"].items():
        if key in state["groups"]:
            continue
        guard(policy.maximum_system_used_percent)
        frame = build_group(root, parent, context, item, source_bindings, spy, staged_baseline(output, key, stages, item))
        record = publish_rows(output, f"groups/{key}.parquet", frame, request_pin, group=True)
        if record["rows"] != item["rows"] or record["decision_ids_sha256"] != item["decision_ids_sha256"]:
            raise DataReadinessError("relationship new group does not preserve parent decisions")
        state["groups"][key] = record
        state["rows"] += record["rows"]
        replace_checkpoint(checkpoint_path, state)
        del frame
        release_process_memory()
        count += 1
        if maximum_groups is not None and count >= maximum_groups and len(state["groups"]) < len(request["stock_inventory"]):
            check_files(root, request["source_files"])
            return _summary(output, state)
    for month, original in sorted(parent.manifest["months"].items()):
        if month in state["months"]:
            continue
        guard(policy.maximum_system_used_percent)
        baseline = load_parent_month(parent, month)
        result = assemble_month(baseline, month_additions(output, state["groups"], month))
        record = publish_rows(output, f"{month}/{PROFILE}.parquet", result, request_pin)
        saved, _ = load_canonical_artifact(inside(output, record["path"]), expected_type=ARTIFACT_TYPE, allow_research=True)
        assert_parent_parity(baseline, saved)
        audit = {"profile_sha256": request["profile_sha256"], "population_preserved": True,
            "baseline_values_exact_except_profile": True, "outcome_filtered_rows": 0, **CLOSED}
        child = {**record, "model_columns": request["model_columns"], "availability_columns": request["availability_columns"],
            "audit": audit}
        state["months"][month] = {"rows": original["rows"], "decision_ids_sha256": original["decision_ids_sha256"],
            "profiles": {PROFILE: child}}
        replace_checkpoint(checkpoint_path, state)
        del baseline, result, saved
        release_process_memory()
    check_files(root, request["source_files"])
    final = {**state, "status": "complete_research_only"}
    _verify_state(output, final, request, request_pin, parent, complete=True)
    write_json_object(output / "_manifest.json", final)
    return _summary(output, final)
