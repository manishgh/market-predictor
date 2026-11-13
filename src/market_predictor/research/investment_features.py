"""Bounded, immutable investment research artifacts; never source admission or fitting."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow.parquet as pq
from pydantic import Field

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.acceptance_clock import CONVENTIONS, corrected_acceptance
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.path_integrity import verify_no_reparse_ancestry
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.investment.contracts import InvestmentTargetRequest, InvestmentTargetResult
from market_predictor.investment.evaluation import InvestmentPredictionRow, evaluate_paired_investment_predictions
from market_predictor.investment.fundamentals import (
    FEATURE_CONTRACT_SHA256,
    FEATURE_NAMES,
    FundamentalSnapshot,
    build_fundamental_batch,
)
from market_predictor.investment.sec_facts import adapt_sec_company_facts
from market_predictor.research.sec_acceptance_clock import load_sec_acceptance_clock
from market_predictor.sources.sec import SecSource
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256, Timestamp

MAX_BYTES = 64 * 1024**2
MAX_ROWS = 10_000
MAX_EXPANDED_BYTES = 128 * 1024**2
FLAGS = {"training_ready": False, "promotion_eligible": False, "production_eligible": False}


class ArtifactReference(HoldingContract):
    path: Identifier
    sha256: Sha256


class FundamentalDecision(HoldingContract):
    security_id: Identifier
    cik: Identifier
    decision_time: Timestamp


class InvestmentFeatureBuildRequest(HoldingContract):
    company_facts_manifest: ArtifactReference
    acceptance_clock_manifest: ArtifactReference
    decisions: tuple[FundamentalDecision, ...] = Field(min_length=1, max_length=MAX_ROWS)
    scope: Literal["nonfinancial"]
    availability_basis: Literal["observed", "reconstructed_research"]


class InvestmentEvaluationRequest(HoldingContract):
    requests: tuple[InvestmentTargetRequest, ...] = Field(min_length=1, max_length=MAX_ROWS)
    targets: tuple[InvestmentTargetResult, ...] = Field(min_length=1, max_length=MAX_ROWS)
    baseline: tuple[InvestmentPredictionRow, ...] = Field(min_length=1, max_length=MAX_ROWS)
    candidate: tuple[InvestmentPredictionRow, ...] = Field(min_length=1, max_length=MAX_ROWS)
    evaluation_as_of_utc: Timestamp


def _read(root: Path, reference: ArtifactReference, pins: dict[str, str], *, paths: EvidencePaths | None = None) -> bytes:
    path = inside(root, reference.path) if paths is None else paths.read(reference.path)
    identity = str(path if paths is None else paths.identity(reference.path))
    if path.stat().st_size > MAX_BYTES:
        raise DataReadinessError("investment input exceeds bounded byte budget")
    data = path.read_bytes()
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != reference.sha256:
        raise DataReadinessError(f"investment input hash differs: {reference.path}")
    if pins.get(identity, reference.sha256) != reference.sha256:
        raise DataReadinessError(f"investment input conflicts with an earlier pin: {reference.path}")
    pins[identity] = reference.sha256
    return data


def _object(
    root: Path, reference: ArtifactReference, pins: dict[str, str], *, paths: EvidencePaths | None = None,
) -> dict[str, Any]:
    result = json.loads(_read(root, reference, pins, paths=paths))
    if not isinstance(result, dict):
        raise DataReadinessError("investment input must be a JSON object")
    return result


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {str(package / name): file_sha256(package / name) for name in (
        "research/investment_features.py", "investment/fundamentals.py", "investment/sec_facts.py",
        "investment/evaluation.py", "investment/contracts.py", "investment/targets.py", "sources/sec.py",
        "catalysts/sec_filings/acceptance_clock.py", "research/sec_acceptance_clock.py")}


def _publish(output: Path, report: dict[str, Any], pins: dict[str, str],
             additional: Callable[[Path], None] | None = None, *, paths: EvidencePaths | None = None,
             retained_external: frozenset[str] = frozenset()) -> dict[str, Any]:
    guard("investment artifact publication")
    for name, digest in pins.items():
        source = Path(name) if paths is None or name in retained_external else paths.read(name)
        if name in retained_external:
            source = verify_no_reparse_ancestry(source, label="retained investment source")
        if file_sha256(source) != digest:
            raise DataReadinessError(f"investment source changed before publication: {name}")
    identity = output.resolve() if paths is None else paths.identity(output)
    output = identity if paths is None else paths.write(identity)
    if output.exists():
        raise FileExistsError(f"immutable investment artifact already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".investment-") as temporary:
        staged = Path(temporary) / "artifact"
        staged.mkdir()
        if additional is not None:
            additional(staged)
        report["files"] = {p.name: file_sha256(p) for p in staged.iterdir() if p.is_file()}
        write_json_object(staged / "_manifest.json", report)
        # Renaming a new directory publishes the complete artifact in one operation.
        if paths is not None and paths.write(identity) != output:
            raise DataReadinessError("investment publication destination changed")
        os.rename(staged, output)
    return {"status": report["status"], "manifest_sha256": file_sha256(output / "_manifest.json"),
            "output_directory": str(identity), **FLAGS}


def collect_investment_company_facts(*, root: Path, cik: str, security_id: str,
                                     output: Path, settings: Settings | None = None) -> dict[str, Any]:
    with heavy_job_lease("collect-investment-company-facts", runtime_dir=root / "data/runtime"):
        return collect_company_facts_unleased(root=root, cik=cik, security_id=security_id, output=output, settings=settings)


def collect_company_facts_unleased(*, root: Path, cik: str, security_id: str,
                                     output: Path, settings: Settings | None = None) -> dict[str, Any]:
    """One issuer per immutable collection; another issuer's failure cannot corrupt it."""
    guard("investment Company Facts collection")
    if output.exists():
        raise FileExistsError(f"immutable investment artifact already exists: {output}")
    if not security_id.strip() or security_id != security_id.strip():
        raise ValueError("a stable security identity is required")
    implementation = _implementation()
    payload, response = SecSource(settings or Settings()).fetch_cik_company_facts(cik)
    report = {"schema": "market_predictor.investment_company_facts", "status": "collected_research_only",
              "security_id": security_id, "cik": cik.zfill(10),
              "retrieved_at": response.retrieved_at_utc.isoformat(), "url": response.final_url,
              "response_sha256": response.body_sha256, "implementation_files": implementation, **FLAGS}

    def save(directory: Path) -> None:
        (directory / "response.bin").write_bytes(response.body)
        write_json_object(directory / "company_facts.json", payload)

    return _publish(output, report, implementation, save)


def _clock_sources(root: Path, reference: ArtifactReference,
                   pins: dict[str, str]) -> tuple[dict[str, str], Path, Path]:
    local_pins: dict[str, str] = {}
    conventions, clock = load_sec_acceptance_clock(root, reference.model_dump(), local_pins)
    for name, digest in local_pins.items():
        pins[str(inside(root, name))] = digest
    authority_ref = ArtifactReference.model_validate_json(json.dumps(clock["sec_collection"]))
    authority = _object(root, authority_ref, pins)
    parent = Path(authority_ref.path).parent
    manifest = _object(root, ArtifactReference(path=str(parent / "_manifest.json"),
                                             sha256=authority["manifest_sha256"]), pins)
    artifacts = manifest["artifacts"]
    archive_ref = ArtifactReference(path=str(parent / artifacts["raw_archive"]["path"]),
                                    sha256=artifacts["raw_archive"]["sha256"])
    inventory_ref = ArtifactReference(path=str(parent / artifacts["raw_inventory"]["path"]),
                                      sha256=artifacts["raw_inventory"]["sha256"])
    archive_path = inside(root, archive_ref.path)
    if file_sha256(archive_path) != archive_ref.sha256:
        raise DataReadinessError("SEC archive hash differs")
    pins[str(archive_path)] = archive_ref.sha256
    _read(root, inventory_ref, pins)
    return conventions, inside(root, archive_ref.path), inside(root, inventory_ref.path)


class RetainedClockIndex:
    """Verify the shared archive once; decode and cache only requested issuers."""

    def __init__(self, root: Path, reference: ArtifactReference, pins: dict[str, str]) -> None:
        self.conventions, self.archive_path, self.inventory_path = _clock_sources(root, reference, pins)
        self.cache: dict[str, dict[str, datetime]] = {}

    def get(self, cik: str) -> dict[str, datetime]:
        if cik not in self.cache:
            convention = self.conventions.get(cik)
            if convention not in CONVENTIONS:
                raise DataReadinessError(f"SEC acceptance convention unavailable for CIK {cik}")
            self.cache[cik] = _issuer_clocks(self.archive_path, self.inventory_path, cik, convention)
        return self.cache[cik]


def _retained_clocks(root: Path, reference: ArtifactReference, cik: str,
                     pins: dict[str, str]) -> dict[str, datetime]:
    return RetainedClockIndex(root, reference, pins).get(cik)


def _issuer_clocks(archive_path: Path, inventory_path: Path, cik: str, convention: str) -> dict[str, datetime]:
    inventory = _bounded_inventory(inventory_path, cik)
    clocks: dict[str, datetime] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for item in inventory:
            guard("investment retained filing clocks")
            info = archive.getinfo(item["archive_member"])
            if item["status_code"] != 200 or info.file_size > MAX_BYTES or info.file_size != item["body_length"]:
                raise DataReadinessError("invalid archived SEC response size or status")
            data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != item["body_sha256"]:
                raise DataReadinessError("archived SEC response hash differs")
            payload = json.loads(data)
            rows = payload.get("filings", {}).get("recent", payload)
            accessions, accepted = rows.get("accessionNumber", []), rows.get("acceptanceDateTime", [])
            if len(accessions) != len(accepted):
                raise DataReadinessError("SEC accession and acceptance arrays differ")
            for accession, raw in zip(accessions, accepted, strict=True):
                value = corrected_acceptance(raw, convention).to_pydatetime()
                if accession in clocks and clocks[accession] != value:
                    raise DataReadinessError("conflicting accession acceptance clocks")
                clocks[accession] = value
    return clocks


def _bounded_inventory(path: Path, cik: str) -> list[dict[str, Any]]:
    parquet = cast(Any, pq).ParquetFile(path)
    expanded = sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.metadata.num_row_groups))
    if parquet.metadata.num_rows > MAX_ROWS or expanded > MAX_EXPANDED_BYTES:
        raise DataReadinessError("SEC inventory exceeds expanded row or byte budget")
    result: list[dict[str, Any]] = cast(Any, pq).read_table(path, filters=[("sec_cik", "=", cik)],
        columns=["archive_member", "body_sha256", "body_length", "status_code"]).to_pylist()
    return result


def build_investment_features(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    root = root.resolve()
    with heavy_job_lease("build-investment-features", runtime_dir=root / "data/runtime"):
        guard("investment feature inputs")
        pins = _implementation()
        parsed = InvestmentFeatureBuildRequest.model_validate_json(_read(
            root, ArtifactReference(path=str(request), sha256=request_sha256), pins))
        source = _object(root, parsed.company_facts_manifest, pins)
        if source.get("schema") != "market_predictor.investment_company_facts" or source.get("status") != "collected_research_only":
            raise DataReadinessError("not a completed investment Company Facts collection")
        cik, security_id = source["cik"], source["security_id"]
        if any(d.cik != cik or d.security_id != security_id for d in parsed.decisions):
            raise DataReadinessError("decision identities differ from the issuer source")
        keys = [(d.security_id, d.decision_time) for d in parsed.decisions]
        if len(set(keys)) != len(keys):
            raise DataReadinessError("duplicate investment decisions")
        parent = Path(parsed.company_facts_manifest.path).parent
        if set(source["files"]) != {"company_facts.json", "response.bin"}:
            raise DataReadinessError("Company Facts artifact inventory differs")
        for name, digest in source["files"].items():
            _read(root, ArtifactReference(path=str(parent / name), sha256=digest), pins)
        payload = _object(root, ArtifactReference(path=str(parent / "company_facts.json"),
                                                 sha256=source["files"]["company_facts.json"]), pins)
        clocks = _retained_clocks(root, parsed.acceptance_clock_manifest, cik, pins)
        decoded = adapt_sec_company_facts(payload, security_id=security_id, cik=cik,
            source_sha256=source["files"]["company_facts.json"], retrieved_at=datetime.fromisoformat(source["retrieved_at"]),
            accepted_at_by_accession=clocks, source_policy=parsed.availability_basis)
        snapshots: list[FundamentalSnapshot] = []
        for decision in parsed.decisions:
            guard("investment feature snapshot")
            snapshots.extend(build_fundamental_batch(decoded.facts,
                [(decision.security_id, decision.cik, decision.decision_time)], rejections=decoded.rejections))
        report = {"schema": "market_predictor.investment_fundamental_features", "status": "research_features_complete",
                  "request_sha256": request_sha256, "source_files": pins, "feature_names": list(FEATURE_NAMES),
                  "feature_contract_sha256": FEATURE_CONTRACT_SHA256, "availability_basis": parsed.availability_basis,
                  "security_id": security_id, "cik": cik, "scope": parsed.scope,
                  "fact_count": len(decoded.facts), "decision_count": len(snapshots),
                  "available_counts": {name: sum(s.metrics[name] is not None for s in snapshots) for name in FEATURE_NAMES},
                  "skipped_source_records": list(decoded.skipped),
                  "rejected_source_records": len(decoded.rejections),
                  "rejection_count_unit": "security_bound_rejected_filing_versions",
                  "rejection_counts": dict(sorted(Counter(rejection.reason for rejection in decoded.rejections).items())),
                  "source_admission": "not_granted_by_research_publication", **FLAGS}

        def save(directory: Path) -> None:
            write_json_object(directory / "request.json", parsed.model_dump(mode="json"))
            write_json_object(directory / "facts.json", decoded.model_dump(mode="json"))
            write_json_object(directory / "rejections.json", {"rejections": [r.model_dump(mode="json") for r in decoded.rejections]})
            write_json_object(directory / "features.json", {"snapshots": [s.model_dump(mode="json") for s in snapshots]})

        return _publish(output, report, pins, save)


def evaluate_investment_predictions(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    root = root.resolve()
    with heavy_job_lease("evaluate-investment-predictions", runtime_dir=root / "data/runtime"):
        guard("investment evaluation inputs")
        pins = _implementation()
        parsed = InvestmentEvaluationRequest.model_validate_json(_read(
            root, ArtifactReference(path=str(request), sha256=request_sha256), pins))
        report = evaluate_paired_investment_predictions(requests=parsed.requests, targets=parsed.targets,
            baseline=parsed.baseline, candidate=parsed.candidate, evaluation_as_of_utc=parsed.evaluation_as_of_utc)
        report.update({"schema": "market_predictor.investment_prediction_evaluation",
                       "status": "research_diagnostics_complete", "request_sha256": request_sha256,
                       "source_files": pins, **FLAGS})
        return _publish(output, report, pins, lambda directory: write_json_object(
            directory / "request.json", parsed.model_dump(mode="json")))
