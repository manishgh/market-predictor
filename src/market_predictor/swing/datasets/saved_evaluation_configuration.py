"""Prove three naming-only differences without admitting archived config formats."""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.saved_evaluation_configuration import (
    RecoveredEvaluationConfiguration,
    SavedEvaluationConfigurationEvidence,
)
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.datasets.return_relationship_integrity import check_files, read_object

STRATEGY_RENAMES = {"schema_version": ("edge_rebuild.strategy_contract.v2", "edge_rebuild.strategy_contract")}
TEMPORAL_RENAMES = {
    "schema_version": ("edge_rebuild.temporal_manifest.v2", "edge_rebuild.temporal_manifest"),
    "unseen_security_assignment": ("sha256_threshold_security_id_v1", "sha256_threshold_security_id"),
}
FLAGS = ("serving_eligible", "promotion_eligible", "portfolio_evaluated", "outer_validation_opened", "historical_test_opened")


@dataclass(frozen=True)
class VerifiedSavedEvaluationConfiguration:
    strategy_path: Path
    temporal_path: Path
    provenance: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _pin(root: Path, pin: SourcePin, pins: dict[str, str]) -> Path:
    path = inside(root, pin.path)
    name = path.relative_to(root).as_posix()
    _require(name not in pins or pins[name] == pin.sha256, f"saved configuration pin conflict: {name}")
    _require(path.is_file() and file_sha256(path) == pin.sha256, f"saved configuration bytes differ: {name}")
    pins[name] = pin.sha256
    return path


def _object(root: Path, pin: SourcePin, pins: dict[str, str]) -> dict[str, Any]:
    return read_object(_pin(root, pin, pins), pin.sha256)


def _compare(original: Any, current: Any, renames: dict[str, tuple[str, str]], field: str = "") -> list[dict[str, str]]:
    if field in renames:
        old, new = renames[field]
        _require(type(original) is str and type(current) is str and (original, current) == (old, new),
            f"saved configuration naming change differs: {field}")
        return [{"field": field, "historical": old, "current": new}]
    _require(type(original) is type(current), f"saved configuration field type differs: {field}")
    changes: list[dict[str, str]] = []
    if isinstance(original, dict):
        _require(original.keys() == current.keys(), f"saved configuration field inventory differs: {field}")
        for key in sorted(original):
            changes.extend(_compare(original[key], current[key], renames, f"{field}.{key}" if field else key))
    elif isinstance(original, list):
        _require(len(original) == len(current), f"saved configuration list length differs: {field}")
        for index, (old_item, new_item) in enumerate(zip(original, current, strict=True)):
            changes.extend(_compare(old_item, new_item, renames, f"{field}[{index}]"))
    else:
        _require(original == current, f"saved configuration value differs: {field}")
    return changes


def _configuration(root: Path, evidence: RecoveredEvaluationConfiguration, renames: dict[str, tuple[str, str]],
    pins: dict[str, str],
) -> tuple[Path, list[dict[str, str]]]:
    original_path = _pin(root, evidence.original_artifact, pins)
    current_path = _pin(root, evidence.current, pins)
    _require(original_path.stat().st_size <= 1024**2 and current_path.stat().st_size <= 1024**2,
        "saved configuration exceeds bounded TOML size")
    original, current = original_path.read_bytes(), current_path.read_bytes()
    _require(hashlib.sha256(original).hexdigest() == evidence.original_logical.sha256
        and hashlib.sha256(current).hexdigest() == evidence.current.sha256, "saved configuration changed during read")
    blob = hashlib.sha1(f"blob {len(original)}\0".encode() + original, usedforsecurity=False).hexdigest()
    _require(blob == evidence.git_blob, "recovered configuration Git blob differs")
    changes = _compare(tomllib.loads(original.decode("utf-8")), tomllib.loads(current.decode("utf-8")), renames)
    _require({item["field"] for item in changes} == set(renames), "saved configuration naming fields missing")
    return current_path, changes


def verify_saved_evaluation_configuration(*, root: Path, evidence_pin: SourcePin, run_manifest: SourcePin,
    strategy_pin: SourcePin, temporal_pin: SourcePin, original_strategy_pin: SourcePin,
    original_temporal_pin: SourcePin, pins: dict[str, str],
) -> VerifiedSavedEvaluationConfiguration:
    """Caller holds its lease and retains/rechecks all accumulated pins through publication."""
    root = root.resolve()
    evidence = SavedEvaluationConfigurationEvidence.model_validate_json(json.dumps(_object(root, evidence_pin, pins)))
    _require(run_manifest in evidence.runs, "saved evaluation run is outside the explicit evidence scope")
    _require(evidence.strategy.current == strategy_pin and evidence.temporal.current == temporal_pin
        and evidence.strategy.original_logical == original_strategy_pin
        and evidence.temporal.original_logical == original_temporal_pin, "saved evaluation configuration arguments differ")
    run_path = _pin(root, run_manifest, pins)
    _require(run_path.name == "_manifest.json", "saved evaluation requires an exact run manifest")
    manifest = _object(root, run_manifest, pins)
    request_pin = SourcePin(path=(run_path.parent / "_request.json").relative_to(root).as_posix(),
        sha256=manifest["request_sha256"])
    request = _object(root, request_pin, pins)
    _require(manifest.get("schema") == "market_predictor.swing_return_training"
        and manifest.get("status") == "complete_research_only"
        and request.get("schema") == "market_predictor.swing_return_training_request"
        and all(item.get(flag) is False for item in (manifest, request) for flag in FLAGS),
        "saved configuration proof requires a closed research run")
    training = ReturnTrainingPolicy.model_validate(request["policy"])
    readiness = _object(root, training.readiness, pins)
    readiness_policy = TrainingReadinessPolicy.model_validate(_object(root, training.readiness_config, pins))
    _require(readiness_policy.strategy_contract == original_strategy_pin
        and readiness_policy.temporal_contract == original_temporal_pin, "original saved readiness configuration pins differ")
    _require(readiness.get("schema") == "market_predictor.swing_training_readiness"
        and readiness.get("status") == "diagnostic_complete" and readiness.get("scope") == readiness_policy.scope
        and readiness.get("publication_sha256") == readiness_policy.publication.sha256
        and readiness.get("published_profile", "technical_market") == training.published_profile == readiness_policy.published_profile
        and readiness.get("training_eligible") is False and readiness.get("promotion_eligible") is False,
        "saved readiness report identity differs")
    for pin in (training.readiness, training.readiness_config, original_strategy_pin, original_temporal_pin):
        _require(request["source_files"].get(pin.path) == pin.sha256, "configuration lacks original saved request pin")
    for pin in (training.readiness_config, original_strategy_pin, original_temporal_pin):
        _require(readiness["source_files"].get(pin.path) == pin.sha256, "configuration lacks original readiness report pin")
    strategy_path, strategy_changes = _configuration(root, evidence.strategy, STRATEGY_RENAMES, pins)
    temporal_path, temporal_changes = _configuration(root, evidence.temporal, TEMPORAL_RENAMES, pins)
    report = _object(root, evidence.source_configuration_report, pins)
    corroboration = report.get("configurations", {}).get(original_strategy_pin.path, {})
    _require(report.get("schema") == "market_predictor.saved_configuration_differences"
        and report.get("rewritten_historical_pins") is False and report.get("source_equivalence_proven") is False
        and report.get("training_eligible") is False
        and corroboration.get("historical_sha256") == original_strategy_pin.sha256
        and corroboration.get("current_sha256") == strategy_pin.sha256
        and corroboration.get("changed_fields") == strategy_changes
        and isinstance(corroboration.get("historical_git_revision"), str)
        and len(corroboration["historical_git_revision"]) >= 7
        and evidence.strategy.git_commit.startswith(corroboration["historical_git_revision"]),
        "saved strategy configuration report does not corroborate actual bytes")
    check_files(root, pins)
    return VerifiedSavedEvaluationConfiguration(strategy_path, temporal_path, {
        "scope": evidence.scope, "evidence": evidence_pin.model_dump(mode="json"),
        "run_manifest": run_manifest.model_dump(mode="json"), "run_request": request_pin.model_dump(mode="json"),
        "readiness": training.readiness.model_dump(mode="json"),
        "readiness_config": training.readiness_config.model_dump(mode="json"),
        "strategy": evidence.strategy.model_dump(mode="json"), "temporal": evidence.temporal.model_dump(mode="json"),
        "source_configuration_report": evidence.source_configuration_report.model_dump(mode="json"),
        "changed_fields": {"strategy": strategy_changes, "temporal": temporal_changes},
        "git_commit_membership_verified": False, "original_bytes_match_saved_authority": True,
        "all_other_parsed_values_types_and_list_order_equal": True,
        "source_equivalence_proven": False, "new_fit_authorized": False, "promotion_eligible": False,
    })
