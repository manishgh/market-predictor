"""Bounded record projection after independent corporate-action archive replay."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, resolve_inside_authority
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.datasets.corporate_action_collection import collect_holding_corporate_actions


@dataclass(frozen=True, slots=True)
class CorporateActionEvidence:
    audit_sha256: str
    request_sha256: str
    records_by_symbol: dict[str, dict[str, list[dict[str, Any]]]]
    unavailable_symbols: tuple[str, ...]
    source_files: dict[str, str]
    replay_metadata: dict[str, Any]
    projection_sha256: str

    def recheck(self, root: Path) -> None:
        if json_sha256({"records": self.records_by_symbol, "unavailable": list(self.unavailable_symbols),
                "source_files": self.source_files, "replay": self.replay_metadata}) != self.projection_sha256:
            raise DataReadinessError("corporate-action projection changed in memory after replay")
        for name, expected in self.source_files.items():
            if file_sha256(resolve_inside_authority(root, name)) != expected:
                raise DataReadinessError("corporate-action evidence changed after independent replay")


def load_corporate_action_evidence(*, root: Path, config: Path, archive: Path,
    expected_audit_sha256: str) -> CorporateActionEvidence:
    """Verify pagination/transport/source pins, then retain only bounded action records.

    Call before acquiring a downstream heavy-job lease: archive replay owns its own
    lease. Consumers must recheck returned file pins while publishing their authority.
    This projection does not certify ownership, economic interpretation or absence.
    """
    root = root.resolve()
    directory = inside(root, archive)
    replay: dict[str, Any] = {}
    report = collect_holding_corporate_actions(root, config, archive,
        expected_audit_sha256=expected_audit_sha256, replay_metadata=replay)
    runtime = (root / heavy_job_runtime_dir()).resolve()
    if not runtime.is_relative_to(root):
        raise DataReadinessError("corporate-action projection runtime escapes workspace")
    with heavy_job_lease("project-swing-corporate-action-evidence", runtime_dir=runtime):
        return _project(root, directory, report, replay)


def _project(root: Path, directory: Path, report: dict[str, Any], replay: dict[str, Any]) -> CorporateActionEvidence:
    request_path = directory / "_request.json"
    if request_path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("corporate-action request exceeds metadata bound")
    request_bytes = request_path.read_bytes()
    request: dict[str, Any] = parse_strict_json_object(request_bytes, label="corporate-action request")
    if (request.get("request_sha256") != report["request_sha256"]
            or request["request_sha256"] != json_sha256({k: v for k, v in request.items() if k != "request_sha256"})):
        raise DataReadinessError("corporate-action request changed during projection")
    bound = dict(request["bound_files"])
    bound[request_path.relative_to(root).as_posix()] = hashlib.sha256(request_bytes).hexdigest()
    records: dict[str, dict[str, list[dict[str, Any]]]] = {}
    missing = []
    for ticker in report["tickers"]:
        assert_memory_budget(stage="corporate-action record projection", hard_budget_gib=5.0, headroom_gib=0.75)
        assert_system_memory_available()
        symbol = ticker["ticker"]
        successes = [attempt for attempt in ticker["attempts"] if attempt["state"] == "acquired"]
        if not successes:
            missing.append(symbol)
            continue
        if len(successes) != 1:
            raise DataReadinessError("corporate-action projection has ambiguous successful attempts")
        summary = successes[0]
        attempt = inside(directory, f"tickers/{symbol}/{summary['attempt']}")
        receipt_path = attempt / "receipt.json"
        if file_sha256(receipt_path) != summary["receipt_sha256"]:
            raise DataReadinessError("corporate-action receipt changed during projection")
        receipt: dict[str, Any] = parse_strict_json_object(receipt_path.read_bytes(), label="corporate-action receipt")
        bound[receipt_path.relative_to(root).as_posix()] = summary["receipt_sha256"]
        families: dict[str, list[dict[str, Any]]] = {}
        for page in receipt["pages"]:
            body_path = resolve_inside_authority(attempt, page["body_path"])
            if body_path.stat().st_size > request["policy"]["maximum_response_bytes"]:
                raise DataReadinessError("corporate-action projected body exceeds its bound")
            body = body_path.read_bytes()
            if hashlib.sha256(body).hexdigest() != page["body_sha256"]:
                raise DataReadinessError("corporate-action body changed during projection")
            payload: dict[str, Any] = parse_strict_json_object(body, label="corporate-action body")
            for family, values in payload["corporate_actions"].items():
                families.setdefault(family, []).extend(values)
            bound[body_path.relative_to(root).as_posix()] = page["body_sha256"]
            metadata_path = resolve_inside_authority(attempt, page["metadata_path"])
            bound[metadata_path.relative_to(root).as_posix()] = page["metadata_sha256"]
        if {name: len(values) for name, values in families.items()} != summary["counts"]:
            raise DataReadinessError("corporate-action projected inventory differs from replay")
        records[symbol] = families
    digest = json_sha256({"records": records, "unavailable": sorted(missing), "source_files": bound, "replay": replay})
    result = CorporateActionEvidence(report["audit_sha256"], request["request_sha256"], records,
        tuple(sorted(missing)), bound, replay, digest)
    result.recheck(root)
    return result
