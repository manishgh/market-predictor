"""Publish monthly research joins from independently pinned completed inputs."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.resources import assert_memory_budget, release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_decisions import verified_corrected_decision_partitions
from market_predictor.swing.datasets.outcome_replay import verify_outcome_replay
from market_predictor.swing.datasets.predictor_abstention_derivation import validate_predictor_derivation
from market_predictor.swing.datasets.predictor_replay import verify_predictor_replay
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.features.catalyst_decision_authority import load_catalyst_decision_authority
from market_predictor.swing.features.panel import CATALYST_RANKING_FEATURES, TECHNICAL_RANKING_FEATURES, swing_model_feature_columns
from market_predictor.swing.features.research_ablation import assemble_research_ablations


def _guard() -> None:
    assert_memory_budget(stage="monthly research join", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0)


def _check(root: Path, files: Mapping[str, str]) -> None:
    for name, digest in files.items():
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError(f"research dataset source changed: {name}")


def _merge_pins(root: Path, *groups: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise DataReadinessError("research source pins require a mapping")
        for name, digest in group.items():
            if (not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise DataReadinessError("research source pin is malformed")
            key = inside(root, name).relative_to(root).as_posix()
            if key in result and result[key] != digest:
                raise DataReadinessError(f"research source pins conflict: {key}")
            result[key] = digest
    return result


def _month_files(root: Path, directories: Mapping[str, Path], technical: Mapping[str, Any],
    targets: Mapping[str, Any], news: Mapping[str, Any], month: str,
) -> dict[str, str]:
    tech = technical["months"][month]
    path = inside(directories["predictors"] / "months", tech["path"])
    pins = {path.relative_to(root).as_posix(): tech["sha256"],
        manifest_path_for(path).relative_to(root).as_posix(): tech["manifest_sha256"]}
    files = targets["months"][month]["files"]
    if "targets.parquet" not in files:
        raise DataReadinessError("research outcomes lack independently pinned targets")
    target_dir = inside(directories["outcomes"], month)
    pins = _merge_pins(root, pins, {inside(target_dir, name).relative_to(root).as_posix(): digest
        for name, digest in files.items()})
    record = news["months"][month]
    if not isinstance(record["directory"], str) or Path(record["directory"]).is_absolute():
        raise DataReadinessError("catalyst month directory must be relative")
    directory = inside(directories["catalysts"], record["directory"])
    authority = pinned_object(directory / "_authority.json", record["authority_sha256"])
    manifest = pinned_object(directory / "_manifest.json", authority["artifact_sha256"])
    pins = _merge_pins(root, pins, {(directory / "_authority.json").relative_to(root).as_posix(): record["authority_sha256"],
        (directory / "_manifest.json").relative_to(root).as_posix(): authority["artifact_sha256"]})
    if set(manifest["artifacts"]) != {"decisions", "coverage"}:
        raise DataReadinessError("catalyst month artifact inventory differs")
    if (type(record.get("catalyst_decision_rows")) is not int
            or record["catalyst_decision_rows"] != manifest["artifacts"]["decisions"]["rows"]):
        raise DataReadinessError("sparse catalyst authority row count differs from monthly index")
    for name, basename in (("decisions", "decision_catalysts.parquet"), ("coverage", "source_coverage.parquet")):
        child = manifest["artifacts"][name]
        path = directory / basename
        if child["path"] != basename:
            raise DataReadinessError("catalyst child path differs")
        pins = _merge_pins(root, pins, {path.relative_to(root).as_posix(): child["sha256"],
            manifest_path_for(path).relative_to(root).as_posix(): child["manifest_sha256"]})
    return pins


def _verified_historical_dependencies(root: Path, declared: Mapping[str, str],
    historical: Mapping[str, str], evidence: Mapping[str, str],
) -> dict[str, str]:
    """Replace only exact replay-verified implementation declarations with evidence."""
    source = _merge_pins(root, declared)
    implementation = _merge_pins(root, historical)
    if any(source.get(name) != digest for name, digest in implementation.items()):
        raise DataReadinessError("replay implementation declarations differ from research source pins")
    return _merge_pins(root, {name: digest for name, digest in source.items() if name not in implementation}, evidence)


def _clocks(manifest: Mapping[str, Any]) -> dict[str, str]:
    groups = manifest["groups"]
    if not groups:
        raise DataReadinessError("research predictors lack feature availability maps")
    result: dict[str, str] | None = None
    for group in groups.values():
        clocks = group["availability_columns"]
        if (not isinstance(clocks, dict) or set(clocks) != set(TECHNICAL_RANKING_FEATURES)
                or any(not isinstance(value, str) or not value for value in clocks.values())
                or (result is not None and clocks != result)):
            raise DataReadinessError("research predictor availability maps differ")
        result = dict(clocks)
    assert result is not None
    return result


def _replace_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        write_json_object(temporary, dict(value))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _verify_month(output: Path, record: Mapping[str, Any], expected: pd.DataFrame, request_pin: str,
    contract: StrategyContract, clocks: Mapping[str, str], month: str,
) -> None:
    if record["decision_ids_sha256"] != json_sha256(sorted(expected.decision_id)) or record["rows"] != len(expected):
        raise DataReadinessError("research join resume population differs")
    if set(record["profiles"]) != {"technical_market", "catalyst_full"}:
        raise DataReadinessError("research join profile inventory differs")
    for name, profile in record["profiles"].items():
        columns = swing_model_feature_columns(contract=contract, catalyst=name == "catalyst_full")
        availability = dict(clocks)
        if name == "catalyst_full":
            availability.update({column: "catalyst_aggregate_asof_utc" for column in CATALYST_RANKING_FEATURES})
        availability.update({column: f"available_at_{column}" for column in columns})
        if (profile["model_columns"] != list(columns) or profile["availability_columns"] != availability
                or profile["path"] != f"{month}/{name}.parquet"
                or profile["audit"].get("training_eligible") is not False
                or profile["audit"].get("promotion_eligible") is not False):
            raise DataReadinessError("research join profile feature contract differs")
        path = inside(output, profile["path"])
        _check(output, {profile["path"]: profile["sha256"],
            manifest_path_for(path).relative_to(output).as_posix(): profile["manifest_sha256"]})
        _, manifest = load_canonical_artifact(path, expected_type="swing_research_join", allow_research=True, columns=[])
        if (manifest["rows"] != len(expected) or manifest["inputs"] != {"request_sha256": request_pin}
                or manifest["production_ready"] is not False):
            raise DataReadinessError("research join child lineage differs")


def materialize_research_dataset(*, root: Path, decision_config: SourcePin, strategy_config: SourcePin,
    predictors: SourcePin, outcomes: SourcePin, catalysts: SourcePin, output: Path,
    expected_checkpoint_sha256: str | None = None,
    predictor_replay: SourcePin | None = None, outcome_replay: SourcePin | None = None,
) -> dict[str, Any]:
    """Join every frozen decision; unresolved economic components remain null.

    Predictor and outcome manifests must be complete in population, not necessarily
    economically usable on every row. This publication cannot grant training,
    promotion, historical reception evidence or managed-exit readiness.
    """
    root = root.resolve()
    output = inside(root, output)
    if output == root / "data/features" or not output.is_relative_to(root / "data/features"):
        raise DataReadinessError("research dataset must publish below data/features")
    pins = _merge_pins(root, *({pin.path: pin.sha256} for pin in
        (decision_config, strategy_config, predictors, outcomes, catalysts)))
    _check(root, pins)
    directories = {name: inside(root, pin.path).parent for name, pin in (
        ("predictors", predictors), ("outcomes", outcomes), ("catalysts", catalysts))}
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in directories.values()):
        raise DataReadinessError("research join output overlaps source evidence")
    if output.exists() != (expected_checkpoint_sha256 is not None):
        raise DataReadinessError("existing research join requires its independent checkpoint hash")
    _guard()
    technical_manifest = pinned_object(inside(root, predictors.path), predictors.sha256)
    target_manifest = pinned_object(inside(root, outcomes.path), outcomes.sha256)
    news_manifest = pinned_object(inside(root, catalysts.path), catalysts.sha256)
    if (technical_manifest.get("schema") != "market_predictor.research_predictors"
            or target_manifest.get("schema") != "market_predictor.corrected_outcomes"
            or technical_manifest.get("status") != "technical_inputs_complete_research_only"
            or technical_manifest.get("failed_groups") != {}
            or target_manifest.get("status") != "partial_research_outcomes"
            or news_manifest.get("schema") != "market_predictor.initial_fit_catalyst_months"
            or news_manifest.get("status") != "complete_research_only"):
        raise DataReadinessError("research join requires completed, source-bound input populations")
    technical_request = pinned_object(directories["predictors"] / "_request.json", technical_manifest["request_sha256"])
    target_request = pinned_object(directories["outcomes"] / "_request.json", target_manifest["request_sha256"])
    technical_dependencies = _merge_pins(root, technical_request["declared_source_files"],
        technical_request["source_files"], technical_request["implementation_files"], technical_request["adjusted_source_files"])
    if predictor_replay is None:
        derivation_files = validate_predictor_derivation(root=root, manifest=technical_manifest, request=technical_request)
        technical_dependencies = _merge_pins(root, technical_dependencies, derivation_files)
    else:
        verified_predictors = verify_predictor_replay(root=root, publication=predictors, replay=predictor_replay)
        technical_dependencies = _verified_historical_dependencies(root, technical_dependencies,
            verified_predictors.historical_implementation_files, verified_predictors.live_evidence_files)
    target_dependencies = _merge_pins(root, target_request["lineage"]["source_files"],
        target_request["lineage"]["implementation_files"])
    if outcome_replay is not None:
        verified_outcomes = verify_outcome_replay(root=root, publication=outcomes, replay=outcome_replay)
        target_dependencies = _verified_historical_dependencies(root, target_dependencies,
            verified_outcomes.historical_implementation_files, verified_outcomes.live_evidence_files)
    if (technical_request.get("schema") != "market_predictor.research_predictor_request"
            or target_request.get("schema") != "market_predictor.corrected_outcome_request"
            or technical_request.get("decision_start") != "2019-07-09"
            or technical_request.get("numeric_end") != "2024-05-28"
            or target_request["lineage"].get("config_sha256") != decision_config.sha256):
        raise DataReadinessError("research source request schema, dates or decision configuration differs")
    declared = _merge_pins(root, technical_request["declared_source_files"])
    if any(declared.get(inside(root, pin.path).relative_to(root).as_posix()) != pin.sha256
            for pin in (decision_config, strategy_config)):
        raise DataReadinessError("research predictors bind another decision or strategy configuration")
    for manifest in (technical_manifest, target_manifest):
        if (manifest.get("training_eligible") is not False or manifest.get("promotion_eligible") is not False
                or manifest.get("exclusions_added") != []):
            raise DataReadinessError("research source claims admission or additional exclusions")
    if target_manifest.get("managed_available") is not False:
        raise DataReadinessError("research outcomes cannot claim managed readiness")
    clocks = _clocks(technical_manifest)
    if not set(technical_manifest["months"]) == set(target_manifest["months"]) == set(news_manifest["months"]):
        raise DataReadinessError("research input month inventories differ")
    dependencies = _merge_pins(root, technical_dependencies, target_dependencies,
        news_manifest["source_files"], pins,
        {(directories[name] / "_request.json").relative_to(root).as_posix(): manifest["request_sha256"]
            for name, manifest in (("predictors", technical_manifest), ("outcomes", target_manifest))})
    monthly_files = {month: _month_files(root, directories, technical_manifest, target_manifest, news_manifest, month)
        for month in technical_manifest["months"]}
    dependencies = _merge_pins(root, dependencies, *monthly_files.values())
    package = Path(__file__).resolve().parents[2]
    for name in ("swing/datasets/research_dataset.py", "swing/datasets/predictor_abstention_derivation.py",
        "swing/features/research_ablation.py",
        "swing/features/research_partition.py", "swing/features/research_join.py", "swing/features/catalyst_aggregates.py",
        "swing/features/catalyst_decision_authority.py", "swing/features/catalyst_decision_identity.py",
        "swing/features/cross_sectional.py", "swing/features/pipeline.py"):
        path = package / name
        dependencies = _merge_pins(root, dependencies, {path.relative_to(root).as_posix(): file_sha256(path)})
    _check(root, dependencies)
    with verified_corrected_decision_partitions(root=root, config=inside(root, decision_config.path),
        expected_config_sha256=decision_config.sha256) as projection:
        if (technical_request["cohort_sha256"] != projection.cohort_sha256
                or target_request["lineage"]["cohort_sha256"] != projection.cohort_sha256
                or news_manifest["cohort_sha256"] != projection.cohort_sha256
                or technical_request["expected_rows"] != projection.expected_rows
                or any(type(m["rows"]) is not int or m["rows"] != projection.expected_rows
                    for m in (technical_manifest, target_manifest, news_manifest))):
            raise DataReadinessError("research inputs differ from the frozen cohort or total population")
        dependencies = _merge_pins(root, dependencies, projection.source_files)
        request = {"schema": "market_predictor.research_join_request", "source_files": dependencies,
            "decision_source_files": dict(projection.source_files), "cohort_sha256": projection.cohort_sha256,
            "rows": projection.expected_rows, "historical_first_seen_proven": False,
            "managed_outcomes_available": False, "profiles": ["technical_market", "catalyst_full"]}
        if output.exists():
            checkpoint = pinned_object(output / "_checkpoint.json", expected_checkpoint_sha256)
            if (checkpoint.get("schema") != "market_predictor.research_join"
                    or checkpoint.get("status") != "partial_in_progress"
                    or checkpoint.get("training_eligible") is not False or checkpoint.get("promotion_eligible") is not False
                    or checkpoint.get("exclusions_added") != []
                    or not isinstance(checkpoint.get("months"), dict)
                    or not set(checkpoint["months"]).issubset(monthly_files)):
                raise DataReadinessError("research checkpoint schema, population or eligibility differs")
            if pinned_object(output / "_request.json", checkpoint["request_sha256"]) != request:
                raise DataReadinessError("research join inputs changed since checkpoint")
        else:
            output.mkdir(parents=True)
            write_json_object(output / "_request.json", request)
            checkpoint = {"schema": "market_predictor.research_join", "status": "partial_in_progress",
                "request_sha256": file_sha256(output / "_request.json"), "months": {},
                "training_eligible": False, "promotion_eligible": False, "exclusions_added": []}
            _replace_json(output / "_checkpoint.json", checkpoint)
        contract = load_strategy_contract(inside(root, strategy_config.path))
        expected_months: set[str] = set()
        for month, expected in projection.partitions:
            _guard()
            if month in expected_months or month not in monthly_files:
                raise DataReadinessError("research join month inventory differs from frozen decisions")
            expected_months.add(month)
            for manifest in (technical_manifest, target_manifest, news_manifest):
                record = manifest["months"][month]
                if (type(record["rows"]) is not int or record["rows"] != len(expected)
                        or record["decision_ids_sha256"] != json_sha256(sorted(expected.decision_id))):
                    raise DataReadinessError("research source month population differs")
            _check(root, monthly_files[month])
            if month in checkpoint["months"]:
                _verify_month(output, checkpoint["months"][month], expected, checkpoint["request_sha256"], contract, clocks, month)
                continue
            tech_record = technical_manifest["months"][month]
            tech_path = inside(directories["predictors"] / "months", tech_record["path"])
            if (file_sha256(tech_path) != tech_record["sha256"]
                    or file_sha256(manifest_path_for(tech_path)) != tech_record["manifest_sha256"]):
                raise DataReadinessError("research technical month changed")
            technical, tech_child = load_canonical_artifact(tech_path, expected_type="swing_research_technical_inputs", allow_research=True)
            _check(root, monthly_files[month])
            _guard()
            if (tech_child["inputs"] != {"research_feature_request_sha256": technical_manifest["request_sha256"]}
                    or tech_child["production_ready"] is not False):
                raise DataReadinessError("research technical child belongs to another request")
            target_record = target_manifest["months"][month]
            target_dir = directories["outcomes"] / month
            _check(target_dir, target_record["files"])
            targets = pd.read_parquet(target_dir / "targets.parquet")
            _check(root, monthly_files[month])
            _guard()
            if any(name in targets and not targets[name].eq(False).all()
                    for name in ("training_eligible", "promotion_eligible")):
                raise DataReadinessError("research outcome rows cannot claim admission")
            news_record = news_manifest["months"][month]
            news = load_catalyst_decision_authority(inside(directories["catalysts"], news_record["directory"]),
                expected_authority_sha256=news_record["authority_sha256"], require_production_ready=False)
            if (type(news_record.get("catalyst_decision_rows")) is not int
                    or news_record["catalyst_decision_rows"] != len(news.decisions)):
                raise DataReadinessError("sparse catalyst authority row count differs from monthly index")
            _check(root, monthly_files[month])
            _guard()
            partitions = assemble_research_ablations(expected_decisions=expected, technical=technical, outcomes=targets,
                catalyst=news, contract=contract, retained_security_ids=frozenset(projection.retained_security_ids),
                technical_availability=clocks)
            _guard()
            profiles: dict[str, Any] = {}
            for profile, partition in partitions.items():
                path = output / month / f"{profile}.parquet"
                if path.exists() or manifest_path_for(path).exists():
                    saved, manifest = load_canonical_artifact(path, expected_type="swing_research_join", allow_research=True)
                    if (manifest["inputs"] != {"request_sha256": checkpoint["request_sha256"]}
                            or manifest["production_ready"] is not False):
                        raise DataReadinessError("uncheckpointed join has foreign lineage")
                    try:
                        pd.testing.assert_frame_equal(saved, partition.rows.reset_index(drop=True))
                    except AssertionError as error:
                        raise DataReadinessError("uncheckpointed join differs from reconstruction") from error
                    del saved
                else:
                    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="exact_research_join", status="pass",
                        failures=0, rows_checked=len(partition.rows),
                        detail="Frozen population and causal clocks verified; not training admission."),))
                    write_canonical_artifact(partition.rows, path, artifact_type="swing_research_join", audit=audit,
                        inputs={"request_sha256": checkpoint["request_sha256"]}, production_ready=False)
                profiles[profile] = {"path": path.relative_to(output).as_posix(), "sha256": file_sha256(path),
                    "manifest_sha256": file_sha256(manifest_path_for(path)), "audit": dict(partition.audit),
                    "model_columns": list(partition.model_columns), "availability_columns": dict(partition.availability_columns)}
            _check(root, monthly_files[month])
            checkpoint["months"][month] = {"rows": len(expected), "decision_ids_sha256": json_sha256(sorted(expected.decision_id)),
                "profiles": profiles}
            _replace_json(output / "_checkpoint.json", checkpoint)
            del technical, targets, news, partitions, partition
            release_process_memory()
        if any(set(manifest["months"]) != expected_months for manifest in (technical_manifest, target_manifest, news_manifest, checkpoint)):
            raise DataReadinessError("research join month inventory differs from frozen decisions")
        _check(root, dependencies)
        result = {**checkpoint, "status": "complete_research_only", "rows": projection.expected_rows}
    # The metadata authority's exit checks must succeed before declaring completion.
    _check(root, dependencies)
    if (pinned_object(output / "_request.json", checkpoint["request_sha256"]) != request
            or pinned_object(output / "_checkpoint.json") != checkpoint):
        raise DataReadinessError("research output metadata changed during publication")
    for record in checkpoint["months"].values():
        for profile in record["profiles"].values():
            path = inside(output, profile["path"])
            _check(output, {profile["path"]: profile["sha256"],
                manifest_path_for(path).relative_to(output).as_posix(): profile["manifest_sha256"]})
    final = output / "_manifest.json"
    if final.exists():
        if pinned_object(final) != result:
            raise DataReadinessError("completed research join manifest changed")
    else:
        _replace_json(final, result)
    return {**result, "manifest_sha256": file_sha256(final), "checkpoint_sha256": file_sha256(output / "_checkpoint.json")}
