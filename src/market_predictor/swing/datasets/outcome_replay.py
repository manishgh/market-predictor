"""Explicit current-code comparison against an immutable outcome publication."""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.implementation_snapshot import verify_implementation_snapshot
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import memory_audit
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.trade_simulation import TradeSimulationContext
from market_predictor.swing.datasets import corrected_outcomes as outcomes
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.evaluation.trade_simulation import load_trade_simulation_context


@dataclass(frozen=True)
class OutcomeReplayVerification:
    historical_implementation_files: dict[str, str]
    live_evidence_files: dict[str, str]


def _current_implementation(root: Path) -> dict[str, str]:
    files = outcomes._implementation(root)
    files[Path(__file__).resolve().relative_to(root).as_posix()] = file_sha256(Path(__file__))
    return files


def verify_outcome_replay(*, root: Path, publication: SourcePin, replay: SourcePin) -> OutcomeReplayVerification:
    """Validate complete current-code replay before a downstream historical join."""
    root = root.resolve()
    original = inside(root, publication.path)
    receipt_path = inside(root, replay.path)
    if (original.name != "_manifest.json" or not original.is_relative_to(root / "data/labels")
            or receipt_path.name != "_manifest.json" or not receipt_path.is_relative_to(root / "data/reports")):
        raise DataReadinessError("outcome replay verification requires publication and completed report manifests")
    manifest = pinned_object(original, publication.sha256)
    request_path = original.parent / "_request.json"
    request = pinned_object(request_path, manifest["request_sha256"])
    receipt = pinned_object(receipt_path, replay.sha256)
    if (receipt.get("schema") != "market_predictor.outcome_implementation_replay"
            or receipt.get("status") != "exact_replay_complete" or receipt.get("replay_complete") is not True
            or receipt.get("source_checks_complete") is not True
            or receipt.get("training_eligible") is not False or receipt.get("promotion_eligible") is not False
            or receipt.get("compared_months") != manifest["months"]
            or manifest.get("schema") != "market_predictor.corrected_outcomes"
            or manifest.get("status") != "partial_research_outcomes"
            or manifest.get("training_eligible") is not False or manifest.get("promotion_eligible") is not False
            or manifest.get("managed_available") is not False or manifest.get("exclusions_added") != []
            or sum(month["rows"] for month in manifest["months"].values()) != manifest["rows"]
            or request.get("schema") != "market_predictor.corrected_outcome_request"):
        raise DataReadinessError("outcome replay is not a complete population comparison")
    replay_request_path = receipt_path.parent / "_request.json"
    comparison = pinned_object(replay_request_path, receipt["request_sha256"])
    snapshot = SourcePin.model_validate(comparison["implementation_snapshot"])
    historical = request["lineage"]["implementation_files"]
    snapshot_files = verify_implementation_snapshot(root=root, manifest=inside(root, snapshot.path),
        expected_sha256=snapshot.sha256, files=historical)
    current = _current_implementation(root)
    lineage = {key: value for key, value in request["lineage"].items() if key != "implementation_files"}
    if comparison != {"schema": "market_predictor.outcome_implementation_replay_request",
            "publication": publication.model_dump(mode="json"), "original_request_sha256": manifest["request_sha256"],
            "implementation_snapshot": snapshot.model_dump(mode="json"),
            "historical_implementation_files": snapshot_files, "current_implementation_files": current,
            "source_lineage": lineage, "expected_months": manifest["months"],
            "comparison": "exact_month_record_target_and_specification_bytes"}:
        raise DataReadinessError("outcome replay publication, source or current implementation identity differs")
    files: dict[str, str] = {}
    groups = [lineage["source_files"], snapshot_files, current, {
        original.relative_to(root).as_posix(): publication.sha256,
        request_path.relative_to(root).as_posix(): manifest["request_sha256"],
        receipt_path.relative_to(root).as_posix(): replay.sha256,
        replay_request_path.relative_to(root).as_posix(): receipt["request_sha256"]}]
    for month, record in manifest["months"].items():
        groups.append({inside(original.parent, f"{month}/{name}").relative_to(root).as_posix(): digest
            for name, digest in record["files"].items()})
    for group in groups:
        for name, digest in group.items():
            key = inside(root, name).relative_to(root).as_posix()
            if key in files and files[key] != digest:
                raise DataReadinessError("outcome replay evidence pins conflict")
            files[key] = digest
    outcomes._check(root, files)
    return OutcomeReplayVerification(dict(historical), files)


def replay_corrected_outcome_publication(
    *, root: Path, config: SourcePin, publication: SourcePin,
    implementation_snapshot: SourcePin, output: Path,
    maximum_months: int | None = None,
    expected_checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Recompute exact target/spec bytes; never update the original publication.

    The snapshot proves historical implementation identity only. Every completed
    comparison is separately bound to the current implementation and original
    source lineage. Partial comparisons never grant replay completion.
    """
    root = root.resolve()
    original = inside(root, publication.path)
    output = inside(root, output)
    if (not output.is_relative_to(root / "data/reports")
            or output == root / "data/reports"):
        raise DataReadinessError("outcome replay requires a directory below data/reports")
    if output.exists() != (expected_checkpoint_sha256 is not None):
        raise DataReadinessError("existing outcome replay requires its external checkpoint pin")
    if (output / "_manifest.json").exists():
        raise DataReadinessError("completed outcome replay cannot be resumed")
    if maximum_months is not None and (type(maximum_months) is not int or not 1 <= maximum_months <= 60):
        raise DataReadinessError("replay month limit must be from 1 to 60")
    if original.name != "_manifest.json" or not original.is_relative_to(root / "data/labels"):
        raise DataReadinessError("outcome replay requires a completed publication manifest")
    manifest = pinned_object(original, publication.sha256)
    if (manifest.get("schema") != "market_predictor.corrected_outcomes"
            or manifest.get("status") != "partial_research_outcomes"
            or manifest.get("training_eligible") is not False
            or manifest.get("promotion_eligible") is not False
            or manifest.get("managed_available") is not False
            or manifest.get("exclusions_added") != []):
        raise DataReadinessError("outcome replay publication contract differs")
    directory = original.parent
    request_path = directory / "_request.json"
    request = pinned_object(request_path, manifest["request_sha256"])
    if request.get("schema") != "market_predictor.corrected_outcome_request":
        raise DataReadinessError("outcome replay request schema differs")
    historical_files = verify_implementation_snapshot(root=root,
        manifest=inside(root, implementation_snapshot.path), expected_sha256=implementation_snapshot.sha256,
        files=request["lineage"]["implementation_files"])
    policy = outcomes.load_corrected_outcome_policy(root, inside(root, config.path), config.sha256)
    outcomes._guard()
    evidence = load_corporate_action_evidence(root=root, config=inside(root, policy.action_config.path),
        archive=inside(root, policy.action_archive), expected_audit_sha256=policy.action_audit_sha256)
    with outcomes.verified_corrected_research_sources(root, inside(root, config.path), config.sha256, policy, evidence) as source:
        current_files = _current_implementation(root)
        expected_lineage = {"config_sha256": config.sha256, "source_files": source["source_files"],
            "cohort_sha256": source["request"]["cohort_sha256"], "action_audit_sha256": evidence.audit_sha256,
            "action_request_sha256": evidence.request_sha256, "action_replay_metadata": evidence.replay_metadata,
            "source_selection_sha256": policy.source_selection.sha256}
        if {key: value for key, value in request["lineage"].items() if key != "implementation_files"} != expected_lineage:
            raise DataReadinessError("outcome replay source, cohort, policy or action identity changed")
        simulation = TradeSimulationContext.model_validate_json(json.dumps(request["simulation"]))
        fresh_simulation = load_trade_simulation_context(inside(root, policy.simulation_policy.path),
            expected_sha256=policy.simulation_policy.sha256)
        if (simulation.policy != fresh_simulation.policy
                or simulation.policy_reference.model_dump(exclude={"retrieved_at"})
                != fresh_simulation.policy_reference.model_dump(exclude={"retrieved_at"})):
            raise DataReadinessError("outcome replay simulation differs from its pinned assumption document")
        outcomes._check_published(directory, manifest["request_sha256"], manifest["months"])
        records = [r for r in source["manifest"]["files"] if r["first_session"] <= str(policy.numerical_end)
            and r["last_session"] >= str(policy.decision_start)]
        if len({r["partition_month"] for r in records}) != len(records):
            raise DataReadinessError("outcome replay has duplicate source months")
        comparison_request = {"schema": "market_predictor.outcome_implementation_replay_request",
            "publication": publication.model_dump(mode="json"), "original_request_sha256": manifest["request_sha256"],
            "implementation_snapshot": implementation_snapshot.model_dump(mode="json"),
            "historical_implementation_files": historical_files, "current_implementation_files": current_files,
            "source_lineage": expected_lineage, "expected_months": manifest["months"],
            "comparison": "exact_month_record_target_and_specification_bytes"}
        prior: dict[str, Any] | None = None
        if expected_checkpoint_sha256 is not None:
            prior = pinned_object(output / "_checkpoint.json", expected_checkpoint_sha256)
            if pinned_object(output / "_request.json", prior["request_sha256"]) != comparison_request:
                raise DataReadinessError("resumed outcome replay request or implementation differs")
            if (prior.get("schema") != "market_predictor.outcome_implementation_replay"
                    or prior.get("status") != "partial" or prior.get("replay_complete") is not False
                    or prior.get("source_checks_complete") is not True
                    or prior.get("training_eligible") is not False or prior.get("promotion_eligible") is not False
                    or not isinstance(prior.get("compared_months"), dict)
                    or any(value != manifest["months"].get(month)
                        for month, value in prior["compared_months"].items())):
                raise DataReadinessError("resumed outcome replay checkpoint is not verified partial evidence")
        else:
            output.mkdir(parents=True)
            outcomes._write(output / "_request.json", comparison_request)
        request_pin = (prior["request_sha256"] if prior is not None
            else hashlib.sha256(outcomes._json(comparison_request)).hexdigest())
        outcomes._check(output, {"_request.json": request_pin})
        report: dict[str, Any] = {"schema": "market_predictor.outcome_implementation_replay",
            "request_sha256": request_pin, "status": "partial",
            "compared_months": {} if prior is None else dict(prior["compared_months"]), "replay_complete": False,
            "source_checks_complete": False,
            "training_eligible": False, "promotion_eligible": False}
        outcomes._write(output / "_progress.json", report)
        expected_months: set[str] = set()
        completed_this_run = 0
        for record in records:
            outcomes._guard()
            frame = outcomes.load_corrected_decision_partition(root, source, record, policy)
            if frame.empty:
                continue
            month = record["partition_month"]
            expected_months.add(month)
            if month not in manifest["months"]:
                raise DataReadinessError("outcome replay publication lost a source month")
            if month in report["compared_months"]:
                continue
            if maximum_months is not None and completed_this_run >= maximum_months:
                continue
            staging = output / f".{month}.{uuid4().hex}.pending"
            started = time.perf_counter()
            result = outcomes._month(root, staging, frame, policy, config.sha256, source, simulation)
            if result != manifest["months"][month]:
                raise DataReadinessError(f"outcome replay differs from published targets/specifications: {month}")
            # Only this invocation's bounded scratch directory is removed.
            if staging.parent != output or not staging.resolve().is_relative_to(output.resolve()):
                raise DataReadinessError("outcome replay scratch directory escaped")
            shutil.rmtree(staging)
            report["compared_months"][month] = result
            completed_this_run += 1
            outcomes._check(root, current_files)
            outcomes._check(root, historical_files)
            outcomes._write(output / "_progress.json", {**report, "last_month": month,
                "last_month_seconds": time.perf_counter() - started,
                "resources": memory_audit(hard_budget_gib=5.0, headroom_gib=0.75).to_record()})
        outcomes._check(root, source["source_files"])
        outcomes._check(root, current_files)
        outcomes._check(root, historical_files)
        evidence.recheck(root)
        outcomes._check_published(directory, manifest["request_sha256"], manifest["months"])
        if file_sha256(original) != publication.sha256:
            raise DataReadinessError("original outcome manifest changed during replay")
        compared = report["compared_months"]
        if set(compared) == set(manifest["months"]) == expected_months:
            if (sum(m["rows"] for m in compared.values()) != manifest["rows"]
                    or manifest["rows"] != source["plan"]["requirements"]["in_window_decisions"]):
                raise DataReadinessError("outcome replay population differs")
            report.update(status="exact_replay_complete", replay_complete=True)
    # Source contexts perform their own exit checks. Publish success only after
    # those checks, then reacquire ownership and recheck the frozen bytes.
    with heavy_job_lease("swing-outcome-replay-finalization", runtime_dir=inside(root, heavy_job_runtime_dir())):
        outcomes._guard()
        outcomes._check(root, source["source_files"])
        outcomes._check(root, current_files)
        outcomes._check(root, historical_files)
        outcomes._check_published(directory, manifest["request_sha256"], manifest["months"])
        if (file_sha256(original) != publication.sha256
                or file_sha256(output / "_request.json") != report["request_sha256"]):
            raise DataReadinessError("outcome replay inputs changed before finalization")
        if (expected_checkpoint_sha256 is not None
                and file_sha256(output / "_checkpoint.json") != expected_checkpoint_sha256):
            raise DataReadinessError("outcome replay checkpoint changed during resumed comparison")
        evidence.recheck(root)
        report["source_checks_complete"] = True
        if report["replay_complete"]:
            outcomes._write(output / "_manifest.json", report)
        else:
            outcomes._write(output / "_checkpoint.json", report)
    return report
