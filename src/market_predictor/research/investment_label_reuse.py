"""Explicit, hash-pinned inheritance of unchanged investment security checkpoints."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.research.investment_features import FLAGS, ArtifactReference, _object
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256, Timestamp
from market_predictor.swing.contracts.holding_materialization import SourcePin

# The frozen November 10 amendment changes only orchestration, this consumer and CLI.
# Every other original exported source file must still have the exact original hash.
REVIEWED_IMPLEMENTATION_CHANGES = frozenset({
    "research/investment_labels.py", "research/investment_label_reuse.py", "commands/investment_research.py"})
RUNTIME_ADDITIONS = frozenset({"research/investment_label_reuse.py", "commands/investment_research.py"})


class ParentSecurityCheckpoint(HoldingContract):
    security_id: Identifier
    checkpoint: SourcePin
    labels: SourcePin
    declared_rows: int = Field(gt=0)
    row_keys_sha256: Sha256


class InvestmentLabelParentReceipt(HoldingContract):
    schema_name: Literal["market_predictor.investment_label_parent_receipt"] = Field(alias="schema")
    request: SourcePin
    parent_job: SourcePin
    parent_common_sources: SourcePin
    parent_implementation_export: SourcePin
    consumer_implementation_export: SourcePin
    parent_job_identity: Sha256
    parent_verified_at_utc: Timestamp
    declared_checkpoint_count: int = Field(gt=0)
    declared_rows: int = Field(gt=0)
    security_checkpoints: tuple[ParentSecurityCheckpoint, ...] = Field(min_length=1)


class InvestmentCheckpointOrigin(HoldingContract):
    origin: Literal["inherited", "new"]
    security_id: Identifier
    checkpoint: SourcePin
    labels: SourcePin
    request: SourcePin
    job: SourcePin
    job_identity: Sha256
    verified_at_utc: Timestamp
    common_sources: SourcePin
    implementation: dict[str, Sha256]
    producer_export: SourcePin | None
    producer_commit: Identifier | None


def comparison_bindings_distinct(row: dict[str, Any]) -> bool:
    """Apply the strict four-role identity requirement before any price/action reads."""
    sector = row.get("sector_benchmark", row.get("primary_benchmark"))
    identities = (str(row["security_id"]), "benchmark:SPY", "benchmark:QQQ", f"benchmark:{sector}")
    symbols = (str(row["ticker"]), "SPY", "QQQ", str(sector))
    return len(set(identities)) == 4 and len(set(symbols)) == 4


def row_keys_sha256(records: list[dict[str, Any]]) -> str:
    return json_sha256(sorted([[row["decision_id"], row["horizon_sessions"]] for row in records]))


def _read(root: Path, pin: SourcePin, pins: dict[str, str]) -> dict[str, Any]:
    return _object(root, ArtifactReference(path=pin.path, sha256=pin.sha256), pins)


def _normalized(root: Path, pins: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, digest in pins.items():
        path = str((root / name).resolve())
        if path in result and result[path] != digest:
            raise DataReadinessError("investment parent source has conflicting canonical pins")
        result[path] = digest
    return result


def _add(pins: dict[str, str], additions: dict[str, str]) -> None:
    for name, digest in additions.items():
        if name in pins and pins[name] != digest:
            raise DataReadinessError("investment parent source pin conflicts with current inputs")
        pins[name] = digest


def producer_origin(implementation: dict[str, str], pins: dict[str, str]) -> tuple[SourcePin | None, str | None]:
    """Pin the actual immutable source export when running from an exported producer."""
    package = Path(__file__).resolve().parents[1]
    export = package.parent.parent / "_export.json"
    if not export.is_file():
        return None, None
    pin = SourcePin(path=str(export), sha256=file_sha256(export))
    value = _read(export.parent, pin, pins)
    commit = value.get("commit")
    if not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise DataReadinessError("investment producer export commit is invalid")
    source_files = {"src/market_predictor/" + path.relative_to(package).as_posix(): file_sha256(path)
        for path in package.rglob("*") if path.is_file() and "__pycache__" not in path.relative_to(package).parts}
    if value.get("files") != source_files:
        raise DataReadinessError("investment producer export file inventory or bytes differ")
    for relative, digest in source_files.items():
        pins[str((export.parent / relative).resolve())] = digest
    for name, digest in implementation.items():
        relative = Path(name).relative_to(export.parent).as_posix()
        if value.get("files", {}).get(relative) != digest:
            raise DataReadinessError("investment producer export differs from running implementation")
    return pin, commit


@dataclass(frozen=True)
class VerifiedParentCheckpoints:
    receipt: InvestmentLabelParentReceipt
    entries: dict[str, ParentSecurityCheckpoint]
    implementation: dict[str, str]
    producer_commit: str
    source_files: dict[str, str]

    def origin(self, entry: ParentSecurityCheckpoint) -> InvestmentCheckpointOrigin:
        return InvestmentCheckpointOrigin(origin="inherited", security_id=entry.security_id,
            checkpoint=entry.checkpoint, labels=entry.labels, request=self.receipt.request,
            job=self.receipt.parent_job, job_identity=self.receipt.parent_job_identity,
            verified_at_utc=self.receipt.parent_verified_at_utc, common_sources=self.receipt.parent_common_sources,
            implementation=self.implementation, producer_export=self.receipt.parent_implementation_export,
            producer_commit=self.producer_commit)


def verify_parent_checkpoints(*, root: Path, receipt_pin: SourcePin, request: SourcePin,
    groups: dict[str, list[dict[str, Any]]], current_base_sources: dict[str, str],
    current_implementation: dict[str, str], current_job: Path, output: Path) -> VerifiedParentCheckpoints:
    """Refuse incompatible/incomplete inheritance; never recompute a rejected parent."""
    pins: dict[str, str] = {}
    value = _read(root, receipt_pin, pins)
    receipt = InvestmentLabelParentReceipt.model_validate_json(json.dumps(value))
    consumer_export, _ = producer_origin(current_implementation, pins)
    if (consumer_export is None or receipt.consumer_implementation_export.sha256 != consumer_export.sha256
        or inside(root, receipt.consumer_implementation_export.path) != inside(root, consumer_export.path)):
        raise DataReadinessError("investment parent receipt is bound to a different consumer producer")
    if (receipt.request.sha256 != request.sha256
        or inside(root, receipt.request.path) != inside(root, request.path)):
        raise DataReadinessError("investment parent request differs from frozen request")
    _read(root, receipt.request, pins)
    job = _read(root, receipt.parent_job, pins)
    if (job.get("job_identity") != receipt.parent_job_identity
        or job.get("verified_at_utc") != receipt.parent_verified_at_utc.isoformat()):
        raise DataReadinessError("investment parent job or verification clock differs")
    parent_root = inside(root, receipt.parent_job.path).parent
    if (inside(root, receipt.parent_job.path).name != "_job.json"
        or inside(root, receipt.parent_common_sources.path) != parent_root / "_verified_sources.json"
        or parent_root == output.resolve() or parent_root.is_relative_to(output.resolve())
        or output.resolve().is_relative_to(parent_root)):
        raise DataReadinessError("investment parent checkpoint paths overlap or differ")
    common = _read(root, receipt.parent_common_sources, pins)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in common.items()):
        raise DataReadinessError("investment parent common inventory is invalid")
    export = _read(root, receipt.parent_implementation_export, pins)
    export_root = inside(root, receipt.parent_implementation_export.path).parent
    commit = export.get("commit")
    if (inside(root, receipt.parent_implementation_export.path).name != "_export.json"
        or not isinstance(commit, str) or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit)
        or export_root.name != commit or not isinstance(export.get("files"), dict)):
        raise DataReadinessError("investment parent immutable producer export is invalid")
    package = Path(__file__).resolve().parents[1]
    exported_names = set()
    for relative, digest in export["files"].items():
        if not isinstance(relative, str) or not relative.startswith("src/market_predictor/"):
            raise DataReadinessError("investment parent export source path is invalid")
        name = relative.removeprefix("src/market_predictor/")
        original = inside(export_root, relative)
        if file_sha256(original) != digest:
            raise DataReadinessError("investment parent producer bytes changed")
        pins[str(original)] = digest
        exported_names.add(name)
        if name not in REVIEWED_IMPLEMENTATION_CHANGES and file_sha256(package / name) != digest:
            raise DataReadinessError("investment parent core implementation differs")
        if name not in REVIEWED_IMPLEMENTATION_CHANGES:
            pins[str((package / name).resolve())] = digest
    current_names = {path.relative_to(package).as_posix() for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.relative_to(package).parts}
    if current_names != exported_names | (set(REVIEWED_IMPLEMENTATION_CHANGES) - exported_names):
        raise DataReadinessError("investment parent/current implementation file populations differ")
    common_normal = _normalized(root, common)
    original_runtime = {name: digest for name, digest in common_normal.items()
        if Path(name).is_relative_to(export_root / "src" / "market_predictor")}
    expected_runtime = {str((export_root / "src" / "market_predictor" / Path(name).relative_to(package)).resolve())
        for name in current_implementation if Path(name).relative_to(package).as_posix() not in RUNTIME_ADDITIONS}
    if set(original_runtime) != expected_runtime:
        raise DataReadinessError("investment parent runtime implementation inventory differs")
    if json_sha256({"request_sha256": request.sha256, "implementation": original_runtime}) != receipt.parent_job_identity:
        raise DataReadinessError("investment parent producer/job identity differs")
    old_base = {name: digest for name, digest in common_normal.items()
        if name not in original_runtime and name != str(inside(root, receipt.parent_job.path))}
    current_runtime = _normalized(root, current_implementation)
    new_base = {name: digest for name, digest in _normalized(root, current_base_sources).items()
        if name not in current_runtime and name != str(current_job.resolve())}
    if old_base != new_base:
        raise DataReadinessError("investment parent nonimplementation inputs differ")
    entries = {entry.security_id: entry for entry in receipt.security_checkpoints}
    paths = {inside(root, entry.checkpoint.path) for entry in receipt.security_checkpoints}
    if (len(entries) != len(receipt.security_checkpoints) or len(paths) != len(entries)
        or receipt.declared_checkpoint_count != len(entries)
        or paths != set((parent_root / "securities").glob("*/_manifest.json"))
        or sum(entry.declared_rows for entry in entries.values()) != receipt.declared_rows):
        raise DataReadinessError("investment parent checkpoint inventory is incomplete or duplicated")
    for security, entry in entries.items():
        guard("investment inherited security verification")
        if security not in groups or inside(root, entry.checkpoint.path) != (
            parent_root / "securities" / json_sha256(security) / "_manifest.json"):
            raise DataReadinessError("investment parent security path or population differs")
        decisions = groups[security]
        if any(row["status"] == "eligible" and not comparison_bindings_distinct(row) for row in decisions):
            raise DataReadinessError("investment parent eligible comparison bindings are not distinct")
        checkpoint = _read(root, entry.checkpoint, pins)
        labels_path = inside(root, entry.checkpoint.path).parent / "labels.json"
        if (checkpoint.get("schema") != "market_predictor.investment_security_labels"
            or checkpoint.get("status") != "security_complete" or checkpoint.get("security_id") != security
            or checkpoint.get("job_identity") != receipt.parent_job_identity
            or checkpoint.get("declared_rows") != entry.declared_rows
            or checkpoint.get("files") != {"labels.json": entry.labels.sha256}
            or inside(root, entry.labels.path) != labels_path
            or any(checkpoint.get(key) is not value for key, value in FLAGS.items())):
            raise DataReadinessError("investment parent security checkpoint identity differs")
        local = _normalized(root, checkpoint["source_files"])
        required = {str(inside(root, pin.path)): pin.sha256
            for pin in (receipt.request, receipt.parent_job, receipt.parent_common_sources)}
        if any(local.get(name) != digest for name, digest in required.items()):
            raise DataReadinessError("investment parent checkpoint source binding differs")
        _add(pins, local)
        records = _read(root, entry.labels, pins)["rows"]
        expected = {(row["decision_id"], horizon) for row in decisions for horizon in (63, 252)}
        if (len(records) != len(expected) or entry.declared_rows != len(expected)
            or {(row["decision_id"], row["horizon_sessions"]) for row in records} != expected
            or row_keys_sha256(records) != entry.row_keys_sha256):
            raise DataReadinessError("investment parent exact horizon row population differs")
        metadata = {row["decision_id"]: row for row in decisions}
        for record in records:
            row = metadata[record["decision_id"]]
            if (record["security_id"] != security or record["decision_session"] != row["decision_session"]
                or record["roster_status"] != row["status"]
                or record["membership_verified_at_utc"] != receipt.parent_verified_at_utc.isoformat()
                or any(record.get(key) is not value for key, value in FLAGS.items())):
                raise DataReadinessError("investment parent row identity or verification clock differs")
            for component in record["components"]:
                for source in component["sources"]:
                    if local.get(str((root / source["reference"]).resolve())) != source["artifact_sha256"]:
                        raise DataReadinessError("investment parent row source is not pinned by its checkpoint")
    _add(pins, common_normal)
    for name, digest in pins.items():
        if file_sha256(Path(name)) != digest:
            raise DataReadinessError(f"investment parent source bytes changed: {name}")
    return VerifiedParentCheckpoints(receipt, entries, original_runtime, commit, pins)
