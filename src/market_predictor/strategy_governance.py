from __future__ import annotations

import hashlib
import re
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.v3.errors import ArtifactIntegrityError, DataReadinessError

CHECKPOINT_IDS = tuple(f"KS{index}" for index in range(10))
SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_SHA_PATTERN = r"^[0-9a-f]{40}$"
CATALOG_ID_PATTERN = re.compile(
    r"\b(?:SWING|INTRADAY|RISK|META)(?:\.[A-Z0-9_]+)+\.V[1-9]\d*\b"
)
CHECKPOINT_ID_PATTERN = re.compile(r"\bKS(?:[0-9])\b")


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PlanBinding(FrozenContract):
    path: str
    sha256: str = Field(pattern=SHA256_PATTERN)
    commit_sha: str = Field(pattern=GIT_SHA_PATTERN)
    remote_ref: str = Field(min_length=1, max_length=256)

    @field_validator("path")
    @classmethod
    def relative_plan_path(cls, value: str) -> str:
        return _validated_relative_path(value)


class EvidenceArtifact(FrozenContract):
    evidence_id: str = Field(pattern=r"^KS[0-9]+-E[1-9]\d*$")
    description: str = Field(min_length=1, max_length=500)
    path: str
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("path")
    @classmethod
    def relative_evidence_path(cls, value: str) -> str:
        return _validated_relative_path(value)


class ExitGate(FrozenContract):
    gate_id: str = Field(pattern=r"^KS[0-9]+-G[1-9]\d*$")
    description: str = Field(min_length=1, max_length=500)
    status: Literal["pending", "passed", "failed", "blocked"] = "pending"
    evidence_ids: tuple[str, ...] = ()


class VerificationItem(FrozenContract):
    verification_id: str = Field(pattern=r"^KS[0-9]+-V[1-9]\d*$")
    description: str = Field(min_length=1, max_length=500)
    status: Literal["pending", "passed", "failed", "blocked"] = "pending"
    command: str | None = Field(default=None, min_length=1, max_length=1_000)
    evidence_ids: tuple[str, ...] = ()


class CheckpointClosure(FrozenContract):
    closed_at_utc: datetime
    commit_sha: str = Field(pattern=GIT_SHA_PATTERN)
    remote_ref: str = Field(min_length=1, max_length=256)
    summary: str = Field(min_length=1, max_length=1_000)

    @field_validator("closed_at_utc")
    @classmethod
    def aware_close_time(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("closed_at_utc must be timezone-aware")
        return value


class StrategyCheckpoint(FrozenContract):
    checkpoint_id: str = Field(pattern=r"^KS[0-9]+$")
    sequence: int = Field(ge=0, le=9)
    title: str = Field(min_length=1, max_length=200)
    status: Literal[
        "pending",
        "in_progress",
        "completed",
        "blocked",
        "environment_pending",
    ] = "pending"
    problem_statement: str = Field(min_length=1, max_length=1_000)
    in_scope: tuple[str, ...] = Field(min_length=1)
    out_of_scope: tuple[str, ...] = Field(min_length=1)
    contracts: tuple[str, ...] = Field(min_length=1)
    exit_gates: tuple[ExitGate, ...] = Field(min_length=1)
    verification: tuple[VerificationItem, ...] = Field(min_length=1)
    evidence: tuple[EvidenceArtifact, ...] = ()
    blocker: str | None = Field(default=None, min_length=1, max_length=1_000)
    closure: CheckpointClosure | None = None

    @model_validator(mode="after")
    def validate_checkpoint_state(self) -> Self:
        prefix = f"{self.checkpoint_id}-"
        if self.sequence != int(self.checkpoint_id.removeprefix("KS")):
            raise ValueError("checkpoint sequence must match checkpoint_id")
        _require_unique(
            [gate.gate_id for gate in self.exit_gates],
            f"{self.checkpoint_id} exit gate",
        )
        _require_unique(
            [item.verification_id for item in self.verification],
            f"{self.checkpoint_id} verification",
        )
        _require_unique(
            [artifact.evidence_id for artifact in self.evidence],
            f"{self.checkpoint_id} evidence",
        )
        if any(not gate.gate_id.startswith(prefix) for gate in self.exit_gates):
            raise ValueError("exit gate ID must belong to its checkpoint")
        if any(
            not item.verification_id.startswith(prefix)
            for item in self.verification
        ):
            raise ValueError("verification ID must belong to its checkpoint")
        if any(
            not artifact.evidence_id.startswith(prefix)
            for artifact in self.evidence
        ):
            raise ValueError("evidence ID must belong to its checkpoint")

        available_evidence = {artifact.evidence_id for artifact in self.evidence}
        gate_evidence = {
            evidence_id
            for gate in self.exit_gates
            for evidence_id in gate.evidence_ids
        }
        verification_evidence = {
            evidence_id
            for item in self.verification
            for evidence_id in item.evidence_ids
        }
        referenced_evidence = gate_evidence.union(verification_evidence)
        missing = referenced_evidence.difference(available_evidence)
        if missing:
            raise ValueError(f"unknown checkpoint evidence IDs: {sorted(missing)}")
        orphaned = available_evidence.difference(referenced_evidence)
        if orphaned:
            raise ValueError(f"unreferenced checkpoint evidence IDs: {sorted(orphaned)}")

        if self.status in {"blocked", "environment_pending"}:
            if self.blocker is None:
                raise ValueError("blocked checkpoint requires an explicit blocker")
        elif self.blocker is not None:
            raise ValueError("only blocked checkpoints may carry a blocker")

        if self.status == "completed":
            if self.closure is None:
                raise ValueError("completed checkpoint requires closure evidence")
            if not self.evidence:
                raise ValueError("completed checkpoint requires evidence artifacts")
            if any(gate.status != "passed" for gate in self.exit_gates):
                raise ValueError("completed checkpoint requires every exit gate to pass")
            if any(not gate.evidence_ids for gate in self.exit_gates):
                raise ValueError("completed checkpoint gates require evidence IDs")
            if any(item.status != "passed" for item in self.verification):
                raise ValueError(
                    "completed checkpoint requires every verification item to pass"
                )
            if any(
                item.command is None or not item.evidence_ids
                for item in self.verification
            ):
                raise ValueError(
                    "completed checkpoint verification requires commands and evidence"
                )
            if any(artifact.sha256 is None for artifact in self.evidence):
                raise ValueError(
                    "completed checkpoint evidence requires content hashes"
                )
        elif self.closure is not None:
            raise ValueError("only completed checkpoints may carry closure evidence")
        return self


class CatalogEntry(FrozenContract):
    item_id: str = Field(min_length=1, max_length=128)
    display_name: str = Field(min_length=1, max_length=200)
    kind: Literal["strategy", "risk", "meta"]
    mode: Literal["swing", "intraday", "shared"]
    checkpoint_id: str = Field(pattern=r"^KS[0-9]+$")
    state: Literal[
        "planned",
        "reference_rejected",
        "data_blocked",
        "deferred",
        "candidate_rejected",
        "candidate_passed",
        "promoted",
    ]
    required_evidence: tuple[str, ...] = Field(min_length=1)
    evidence_paths: tuple[str, ...] = ()
    blocker: str | None = Field(default=None, min_length=1, max_length=1_000)

    @field_validator("item_id")
    @classmethod
    def valid_catalog_id(cls, value: str) -> str:
        if CATALOG_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("invalid strategy or component ID")
        if value.startswith(("SWING.", "INTRADAY.")):
            parts = value.split(".")
            if len(parts) != 4 or re.fullmatch(r"[1-9]\d*(?:M|D)", parts[2]) is None:
                raise ValueError("strategy ID requires an explicit M or D horizon")
        return value

    @field_validator("evidence_paths")
    @classmethod
    def relative_evidence_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_validated_relative_path(value) for value in values)

    @model_validator(mode="after")
    def validate_catalog_state(self) -> Self:
        expected_kind = (
            "strategy"
            if self.item_id.startswith(("SWING.", "INTRADAY."))
            else "risk"
            if self.item_id.startswith("RISK.")
            else "meta"
        )
        if self.kind != expected_kind:
            raise ValueError("catalog kind does not match item ID")
        if self.item_id.startswith("SWING.") and self.mode != "swing":
            raise ValueError("SWING strategy must use swing mode")
        if self.item_id.startswith("INTRADAY.") and self.mode != "intraday":
            raise ValueError("INTRADAY strategy must use intraday mode")
        if self.state in {"data_blocked", "deferred"} and self.blocker is None:
            raise ValueError("blocked or deferred catalog entry requires a blocker")
        if self.state not in {"data_blocked", "deferred"} and self.blocker is not None:
            raise ValueError("only blocked or deferred catalog entries may carry blocker")
        if self.state in {
            "reference_rejected",
            "candidate_rejected",
            "candidate_passed",
            "promoted",
        } and not self.evidence_paths:
            raise ValueError("evaluated catalog state requires evidence paths")
        return self


class StrategyExecutionLedger(FrozenContract):
    schema_version: Literal["market_predictor.strategy_execution_ledger.v1"]
    plan: PlanBinding
    checkpoints: tuple[StrategyCheckpoint, ...] = Field(min_length=1)
    catalog: tuple[CatalogEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ledger_inventory(self) -> Self:
        checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in self.checkpoints]
        _require_unique(checkpoint_ids, "checkpoint")
        if tuple(checkpoint_ids) != CHECKPOINT_IDS:
            raise ValueError(
                f"ledger checkpoints must be ordered exactly as {list(CHECKPOINT_IDS)}"
            )
        catalog_ids = [entry.item_id for entry in self.catalog]
        _require_unique(catalog_ids, "catalog")
        checkpoint_set = set(checkpoint_ids)
        invalid_references = {
            entry.checkpoint_id
            for entry in self.catalog
            if entry.checkpoint_id not in checkpoint_set
        }
        if invalid_references:
            raise ValueError(
                f"catalog references unknown checkpoints: {sorted(invalid_references)}"
            )
        return self


def validate_strategy_execution_ledger(
    ledger_path: Path,
    *,
    repository_root: Path,
    verify_git: bool = True,
) -> dict[str, object]:
    root = repository_root.resolve()
    path = _resolve_repository_path(root, ledger_path)
    try:
        ledger = StrategyExecutionLedger.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot read strategy ledger: {path}") from exc

    plan_path = _resolve_repository_path(root, Path(ledger.plan.path))
    plan_bytes = _read_required(plan_path, "strategy plan")
    actual_plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if actual_plan_sha256 != ledger.plan.sha256:
        raise ArtifactIntegrityError(
            "strategy plan hash does not match the ledger binding"
        )

    plan_text = plan_bytes.decode("utf-8")
    plan_catalog = set(CATALOG_ID_PATTERN.findall(plan_text))
    ledger_catalog = {entry.item_id for entry in ledger.catalog}
    if plan_catalog != ledger_catalog:
        raise DataReadinessError(
            "strategy catalog mismatch: "
            f"missing_from_ledger={sorted(plan_catalog - ledger_catalog)}, "
            f"missing_from_plan={sorted(ledger_catalog - plan_catalog)}"
        )

    plan_checkpoints = set(CHECKPOINT_ID_PATTERN.findall(plan_text))
    ledger_checkpoints = {
        checkpoint.checkpoint_id for checkpoint in ledger.checkpoints
    }
    if plan_checkpoints != ledger_checkpoints:
        raise DataReadinessError(
            "strategy checkpoint mismatch: "
            f"missing_from_ledger={sorted(plan_checkpoints - ledger_checkpoints)}, "
            f"missing_from_plan={sorted(ledger_checkpoints - plan_checkpoints)}"
        )

    for checkpoint in ledger.checkpoints:
        for artifact in checkpoint.evidence:
            evidence_path = _resolve_repository_path(root, Path(artifact.path))
            content = _read_required(
                evidence_path,
                f"{checkpoint.checkpoint_id} evidence {artifact.evidence_id}",
            )
            if artifact.sha256 is not None:
                actual_sha256 = hashlib.sha256(content).hexdigest()
                if actual_sha256 != artifact.sha256:
                    raise ArtifactIntegrityError(
                        f"evidence hash mismatch: {artifact.evidence_id}"
                    )

    for entry in ledger.catalog:
        for evidence_path_value in entry.evidence_paths:
            evidence_path = _resolve_repository_path(root, Path(evidence_path_value))
            _read_required(evidence_path, f"{entry.item_id} catalog evidence")

    if verify_git:
        _verify_git_binding(
            root,
            commit_sha=ledger.plan.commit_sha,
            remote_ref=ledger.plan.remote_ref,
            tracked_path=ledger.plan.path,
            expected_sha256=ledger.plan.sha256,
        )
        for checkpoint in ledger.checkpoints:
            if checkpoint.closure is None:
                continue
            _verify_git_commit_on_remote(
                root,
                checkpoint.closure.commit_sha,
                checkpoint.closure.remote_ref,
            )

    checkpoint_counts = Counter(
        checkpoint.status for checkpoint in ledger.checkpoints
    )
    catalog_counts = Counter(entry.state for entry in ledger.catalog)
    return {
        "schema_version": ledger.schema_version,
        "valid": True,
        "plan_path": ledger.plan.path,
        "plan_sha256": ledger.plan.sha256,
        "plan_commit_sha": ledger.plan.commit_sha,
        "checkpoint_count": len(ledger.checkpoints),
        "checkpoint_status": dict(sorted(checkpoint_counts.items())),
        "catalog_count": len(ledger.catalog),
        "catalog_status": dict(sorted(catalog_counts.items())),
        "completed_checkpoints": [
            checkpoint.checkpoint_id
            for checkpoint in ledger.checkpoints
            if checkpoint.status == "completed"
        ],
        "next_checkpoint": next(
            (
                checkpoint.checkpoint_id
                for checkpoint in ledger.checkpoints
                if checkpoint.status != "completed"
            ),
            None,
        ),
    }


def _validated_relative_path(value: str) -> str:
    if "\\" in value:
        raise ValueError("repository paths must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("repository path must be normalized and relative")
    return value


def _require_unique(values: list[str], name: str) -> None:
    duplicates = sorted(
        value for value, count in Counter(values).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate {name} IDs: {duplicates}")


def _resolve_repository_path(root: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArtifactIntegrityError(
            f"artifact escapes repository root: {path}"
        ) from exc
    return resolved


def _read_required(path: Path, description: str) -> bytes:
    if not path.is_file():
        raise ArtifactIntegrityError(f"missing {description}: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise ArtifactIntegrityError(f"cannot read {description}: {path}") from exc


def _verify_git_binding(
    root: Path,
    *,
    commit_sha: str,
    remote_ref: str,
    tracked_path: str,
    expected_sha256: str,
) -> None:
    _verify_git_commit_on_remote(root, commit_sha, remote_ref)
    completed = _run_git(root, "show", f"{commit_sha}:{tracked_path}")
    historical_sha256 = hashlib.sha256(completed.stdout).hexdigest()
    if historical_sha256 != expected_sha256:
        raise ArtifactIntegrityError(
            "strategy plan content at the bound commit does not match plan_sha256"
        )


def _verify_git_commit_on_remote(
    root: Path,
    commit_sha: str,
    remote_ref: str,
) -> None:
    _run_git(root, "cat-file", "-e", f"{commit_sha}^{{commit}}")
    _run_git(root, "rev-parse", "--verify", remote_ref)
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit_sha, remote_ref],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ArtifactIntegrityError(
            f"commit {commit_sha} is not present on {remote_ref}"
        )


def _run_git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactIntegrityError(
            f"git verification failed: git {' '.join(arguments)}"
        ) from exc
