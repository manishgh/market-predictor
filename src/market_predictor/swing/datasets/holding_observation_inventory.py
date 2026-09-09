"""Immutable initial-fit raw-bar diagnostics, not an admitted outcome authority."""
from __future__ import annotations

import json
import os
import re
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget, release_process_memory
from market_predictor.swing.contracts.research_cohort import Sha256
from market_predictor.swing.datasets.holding_observation_requirements import prepare_holding_observation_requirements
from market_predictor.swing.datasets.holding_observations import read_required_holding_observations


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["market_predictor.swing_holding_observation_inventory_request"] = Field(alias="schema")
    preflight_path: str = Field(min_length=1)
    preflight_sha256: Sha256
    parent_request_path: str = Field(min_length=1)
    expected_decisions: int = Field(ge=1, le=100_000)
    expected_securities: int = Field(ge=1, le=2_000)


def _guard() -> None:
    assert_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage="holding observation inventory")
    assert_peak_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage="holding observation inventory")


def _inside(root: Path, path: Path) -> Path:
    value = (root / path).resolve()
    if not value.is_relative_to(root):
        raise DataReadinessError("holding observation path escapes repository")
    return value


def _verify_files(root: Path, files: dict[str, str]) -> None:
    for relative, digest in files.items():
        if file_sha256(resolve_inside_authority(root, relative)) != digest:
            raise DataReadinessError(f"holding observation source hash differs: {relative}")


def _implementation_files() -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    relatives = (
        "swing/datasets/holding_observation_inventory.py", "swing/datasets/holding_observation_requirements.py",
        "swing/datasets/holding_observations.py", "swing/labels/holding_identity.py",
        "swing/labels/holding_paths.py", "canonical/joins.py", "swing/contracts/research_cohort.py",
    )
    return {name: file_sha256(package / name) for name in relatives}


def _raw_index(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("artifacts")
    if not isinstance(records, list) or len(records) > 10_000:
        raise DataReadinessError("holding observation raw inventory is invalid or unbounded")
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("ticker"), str)
                or not record["ticker"] or record["ticker"] in output
                or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str)
                or record.get("price_feed") != "sip" or record.get("adjustment") != "all"):
            raise DataReadinessError("holding observation raw inventory has duplicate or invalid identity/feed")
        output[record["ticker"]] = record
    return output


def _publish(directory: Path, frames: dict[str, pd.DataFrame], report: dict[str, Any]) -> dict[str, Any]:
    """Publish the entire diagnostic together; never replace an existing inventory."""
    directory.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(directory, timeout=0.0):
        if directory.exists():
            raise DataReadinessError("holding observation inventory already exists; use immutable replay")
        staging = directory.with_name(f".{directory.name}.{uuid4().hex}.tmp")
        staging.mkdir()
        created: list[Path] = []
        try:
            outputs: dict[str, str] = {}
            for name, frame in frames.items():
                target = staging / name
                created.append(target)
                frame.to_parquet(target, index=False)
                outputs[name] = file_sha256(target)
            report["outputs"] = outputs
            report["audit_sha256"] = json_sha256(report)
            target = staging / "_manifest.json"
            created.append(target)
            write_json_object(target, report)
            os.rename(staging, directory)
        finally:
            for target in created:
                target.unlink(missing_ok=True)
            if staging.exists():
                staging.rmdir()
    return report


def _replay(root: Path, directory: Path, request: _Request, config: Path, expected_sha256: str) -> dict[str, Any]:
    path = directory / "_manifest.json"
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("holding observation inventory metadata exceeds size bound")
    report: dict[str, Any] = parse_strict_json_object(path.read_bytes(), label=str(path))
    if (report.get("audit_sha256") != expected_sha256
            or report.get("audit_sha256") != json_sha256({k: v for k, v in report.items() if k != "audit_sha256"})
            or report.get("request") != request.model_dump(mode="json", by_alias=True)
            or report.get("implementation_files") != _implementation_files()
            or not isinstance(report.get("source_files"), dict)
            or report["source_files"].get(config.relative_to(root).as_posix()) != file_sha256(config)
            or not isinstance(report.get("outputs"), dict)
            or any(report.get(key) is not False for key in (
                "heldout_numeric_rows_read", "materialization_eligible", "accounting_eligible", "promotion_eligible",
            ))):
        raise DataReadinessError("holding observation immutable replay identity differs")
    _verify_files(root, report["source_files"])
    _verify_files(directory, report["outputs"])
    return report


def run_holding_observation_inventory(
    root: Path, config_path: Path, output_directory: Path, *, expected_audit_sha256: str | None = None,
) -> dict[str, Any]:
    """Recover exact initial-fit requirements without reading validation/test prices."""
    root = root.resolve()
    config = resolve_inside_authority(root, str(config_path))
    output = _inside(root, output_directory)
    with heavy_job_lease("audit-swing-holding-observations", runtime_dir=_inside(root, heavy_job_runtime_dir())):
        try:
            _guard()
            if config.stat().st_size > 1024**2:
                raise DataReadinessError("holding observation configuration exceeds size bound")
            request = _Request.model_validate_json(json.dumps(tomllib.loads(config.read_text(encoding="utf-8"))))
            if output.exists():
                if expected_audit_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", expected_audit_sha256) is None:
                    raise DataReadinessError("immutable replay requires an independently retained expected audit SHA-256")
                return _replay(root, output, request, config, expected_audit_sha256)
            if expected_audit_sha256 is not None:
                raise DataReadinessError("expected audit SHA-256 is only valid for an existing inventory replay")
            prepared = prepare_holding_observation_requirements(
                root, preflight_path=Path(request.preflight_path), preflight_sha256=request.preflight_sha256,
                parent_request_path=Path(request.parent_request_path), expected_decisions=request.expected_decisions,
                expected_securities=request.expected_securities,
            )
            bound = {**prepared.bound_files, config.relative_to(root).as_posix(): file_sha256(config)}
            implementation = _implementation_files()
            raw = _raw_index(prepared.raw_manifest)
            frames: dict[str, pd.DataFrame] = {"decisions.parquet": prepared.decisions}
            cases: list[dict[str, Any]] = []
            totals: Counter[str] = Counter()
            for requirement in prepared.requirements:
                _guard()
                ticker, identity = requirement["ticker"], requirement["security_id"]
                sessions = tuple(requirement["sessions"])
                record = raw.get(ticker)
                source_path = None
                if record is not None:
                    source_path = resolve_inside_authority(root, record["path"])
                    relative = source_path.relative_to(root).as_posix()
                    if relative in bound and bound[relative] != record["sha256"]:
                        raise DataReadinessError("holding observation source has conflicting hashes")
                    bound[relative] = record["sha256"]
                    _verify_files(root, {relative: record["sha256"]})
                metadata = dict(ticker=ticker, security_id=identity, sessions=sessions,
                    numeric_end=prepared.numeric_end, memberships=prepared.memberships)
                # Corrupt observations remain a separate source error, not missing data.
                error = None
                try:
                    frame = read_required_holding_observations(source_path, **metadata)
                except (DataReadinessError, ValueError, TypeError, KeyError, OSError) as exc:
                    if source_path is None:
                        raise
                    frame = read_required_holding_observations(None, **metadata)
                    frame["source_present"] = pd.Series([None] * len(frame), dtype="boolean")
                    frame["observation_status"] = "source_error"
                    error = f"{type(exc).__name__}: {exc}"
                name = f"observations-{json_sha256([identity, ticker])[:20]}.parquet"
                if name in frames:
                    raise DataReadinessError("duplicate holding observation requirement")
                frames[name] = frame
                counts = Counter(str(value) for value in frame.observation_status)
                counts.update(str(value) for value in frame.ownership_status)
                totals.update(counts)
                cases.append({"security_id": identity, "ticker": ticker, "sessions": [str(day) for day in sessions],
                    "artifact_available": source_path is not None, "output": name, "counts": dict(counts), "error": error})
                release_process_memory()
            _guard()
            _verify_files(root, bound)
            if _implementation_files() != implementation:
                raise DataReadinessError("holding observation implementation changed during inventory")
            report = {
                "schema": "market_predictor.swing_holding_observation_inventory",
                "scope": "initial_fit_flagged_holding_observation_diagnostic",
                "status": "diagnostic_complete_with_source_errors" if totals["source_error"] else "diagnostic_complete",
                "request": request.model_dump(mode="json", by_alias=True), "numeric_end": str(prepared.numeric_end),
                "decisions": len(prepared.decisions), "securities": int(prepared.decisions.security_id.nunique()),
                "required_security_ticker_sessions": sum(len(case["sessions"]) for case in cases),
                "counts": dict(totals), "cases": cases, "source_files": bound,
                "implementation_files": implementation, "heldout_numeric_rows_read": False,
                "materialization_eligible": False, "accounting_eligible": False, "promotion_eligible": False,
            }
            return _publish(output, frames, report)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise DataReadinessError(f"invalid holding observation inventory: {exc}") from exc
