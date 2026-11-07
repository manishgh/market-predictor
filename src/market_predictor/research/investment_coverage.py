"""Resumable serial issuer coverage over one frozen investment roster."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd
from pydantic import Field

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.collection import normalize_sec_identity_relations
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.investment.cohort import investment_roster
from market_predictor.investment.fundamentals import FEATURE_CONTRACT_SHA256, FEATURE_NAMES, build_fundamental_snapshot
from market_predictor.investment.sec_facts import adapt_sec_company_facts
from market_predictor.research.investment_features import (
    FLAGS,
    ArtifactReference,
    RetainedClockIndex,
    _implementation,
    _object,
    _publish,
    _read,
    collect_company_facts_unleased,
)
from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.research_cohort import load_swing_research_cohort


class InvestmentCoverageRequest(HoldingContract):
    cohort: ArtifactReference
    membership: ArtifactReference
    identity: ArtifactReference
    identity_manifest: ArtifactReference
    acceptance_clock_manifest: ArtifactReference
    reused_company_facts: tuple[ArtifactReference, ...] = Field(default=(), max_length=1000)
    source_policy: Literal["reconstructed_research"] = "reconstructed_research"


def _source_pin_path(root: Path, name: str) -> Path:
    # Source inventories also contain producer implementation files. A worktree
    # may be outside the retained-data root, and its absolute hash pin is valid.
    path = Path(name)
    return path if path.is_absolute() else inside(root, name)


def _implementation_pins() -> dict[str, str]:
    pins = _implementation()
    for name in (Path(__file__), Path(__file__).parents[1] / "investment/cohort.py"):
        pins[str(name.resolve())] = file_sha256(name)
    return pins


def _projection(root: Path, ref: ArtifactReference, columns: list[str], pins: dict[str, str]) -> pd.DataFrame:
    path = inside(root, ref.path)
    if file_sha256(path) != ref.sha256:
        raise DataReadinessError(f"investment canonical source hash differs: {ref.path}")
    pins[str(path)] = ref.sha256
    import pyarrow.parquet as pq

    metadata = cast(Any, pq).ParquetFile(path).metadata
    if metadata.num_rows > 100_000 or sum(metadata.row_group(i).total_byte_size for i in range(metadata.num_row_groups)) > 128 * 1024**2:
        raise DataReadinessError("investment canonical projection exceeds bounded budget")
    return pd.read_parquet(path, columns=columns)


def plan_investment_coverage(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Publish the complete roster before any download; fixed monthly calendar and cohort."""
    root = root.resolve()
    with heavy_job_lease("plan-investment-coverage", runtime_dir=root / "data/runtime"):
        guard("investment coverage roster")
        pins = _implementation_pins()
        parsed = InvestmentCoverageRequest.model_validate_json(
            _read(root, ArtifactReference(path=str(request), sha256=request_sha256), pins)
        )
        _read(root, parsed.cohort, pins)
        cohort = load_swing_research_cohort(inside(root, parsed.cohort.path), source_root=root)
        pins.update({str(inside(root, path)): digest for path, digest in cohort.source_files.items()})
        identity_manifest = _object(root, parsed.identity_manifest, pins)
        if (
            identity_manifest.get("schema") != "market_predictor.issuer_news_identity_alignment"
            or identity_manifest.get("source_files", {}).get(parsed.identity.path) != parsed.identity.sha256
        ):
            raise DataReadinessError("investment identity manifest does not bind canonical issuer relations")
        members = _projection(
            root,
            parsed.membership,
            ["security_id", "ticker", "sector", "primary_benchmark", "effective_from_utc", "effective_to_utc", "available_at_utc"],
            pins,
        )
        issuers = _projection(
            root, parsed.identity, ["security_id", "ticker", "sec_cik", "effective_from_utc", "effective_to_utc", "available_at_utc"], pins
        )
        issuers = normalize_sec_identity_relations(issuers)
        rows = investment_roster(cohort, members, issuers)
        report = {
            "schema": "market_predictor.investment_coverage_roster",
            "status": "roster_complete",
            "request_sha256": request_sha256,
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "cohort_sha256": cohort.sha256(),
            "original_securities": 631,
            "excluded_securities": 45,
            "remaining_exclusion_capacity": 18,
            "declared_rows": len(rows),
            "source_files": pins,
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
            "historical_identity_policy": "retrospective_stable_security_interval_proxy",
            "calendar_policy": "last_XNYS_session_each_month_20190709_through_20240528_including_truncated_May",
            "sector_scope": "historical_sector_excludes_Financials_and_Real_Estate_separate_from_data_exclusions",
            **FLAGS,
        }

        def save(directory: Path) -> None:
            write_json_object(directory / "request.json", parsed.model_dump(mode="json"))
            write_json_object(directory / "roster.json", {"rows": rows})

        return _publish(output, report, pins, save)


def _source(
    root: Path, manifest: ArtifactReference, security_id: str, cik: str, pins: dict[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _object(root, manifest, pins)
    if (
        source.get("schema") != "market_predictor.investment_company_facts"
        or source.get("status") != "collected_research_only"
        or source.get("cik") != cik
    ):
        raise DataReadinessError("investment reused issuer collection identity/status differs")
    # Multiple security classes may share an issuer; stable CIK identity, not ticker, binds fundamentals.
    if set(source.get("files", {})) != {"company_facts.json", "response.bin"}:
        raise DataReadinessError("investment CompanyFacts source inventory differs")
    parent = Path(manifest.path).parent
    for name, digest in source["files"].items():
        _read(root, ArtifactReference(path=str(parent / name), sha256=digest), pins)
    payload = _object(root, ArtifactReference(path=str(parent / "company_facts.json"), sha256=source["files"]["company_facts.json"]), pins)
    return source, payload


def _save_feature_rows(directory: Path, *, rows: list[dict[str, Any]]) -> None:
    write_json_object(directory / "features.json", {"rows": rows})


def run_investment_coverage(
    *, root: Path, roster_manifest: Path, roster_manifest_sha256: str, output: Path, settings: Settings | None = None
) -> dict[str, Any]:
    """One shared lease, issuer-isolated checkpoints; a failure never deletes a roster row."""
    root, output = root.resolve(), output.resolve()
    if (output / "final").exists():
        raise FileExistsError("completed immutable investment coverage already exists")
    with heavy_job_lease("run-investment-coverage", runtime_dir=root / "data/runtime"):
        guard("investment coverage inputs")
        pins = _implementation_pins()
        roster_ref = ArtifactReference(path=str(roster_manifest), sha256=roster_manifest_sha256)
        manifest = _object(root, roster_ref, pins)
        if manifest.get("schema") != "market_predictor.investment_coverage_roster" or manifest.get("status") != "roster_complete":
            raise DataReadinessError("investment coverage requires completed pinned roster")
        if manifest.get("feature_contract_sha256") != FEATURE_CONTRACT_SHA256:
            raise DataReadinessError("investment roster feature contract differs")
        for name, digest in manifest["source_files"].items():
            if file_sha256(_source_pin_path(root, name)) != digest:
                raise DataReadinessError(f"investment roster source changed: {name}")
            pins[str(_source_pin_path(root, name))] = digest
        parent = Path(roster_manifest).parent
        loaded = {
            name: _object(root, ArtifactReference(path=str(parent / name), sha256=digest), pins)
            for name, digest in manifest["files"].items()
        }
        request = InvestmentCoverageRequest.model_validate_json(json.dumps(loaded["request.json"]))
        rows = loaded["roster.json"]["rows"]
        roster_digest = manifest["files"]["roster.json"]
        job_identity = json_sha256(
            {"roster": roster_manifest_sha256, "features": FEATURE_CONTRACT_SHA256, "implementation": _implementation_pins()}
        )
        output.mkdir(parents=True, exist_ok=True)
        identity_path = output / "_job.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != {"job_identity": job_identity}:
                raise DataReadinessError("investment resume request or implementation differs")
        else:
            write_json_object(identity_path, {"job_identity": job_identity})
        index = RetainedClockIndex(root, request.acceptance_clock_manifest, pins)
        sources: dict[str, ArtifactReference] = {}
        for reused_reference in request.reused_company_facts:
            local: dict[str, str] = {}
            source = _object(root, reused_reference, local)
            cik = source.get("cik")
            if not isinstance(cik, str) or cik in sources:
                raise DataReadinessError("duplicate or invalid reused investment issuer source")
            sources[cik] = reused_reference
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if row["status"] == "eligible":
                groups.setdefault(row["cik"], []).append(row)
        summary: Counter[tuple[str, str, str, str]] = Counter()
        failures: list[dict[str, str]] = []
        issuer_artifacts: list[dict[str, str]] = []
        for cik, decisions in sorted(groups.items()):
            guard("investment serial issuer coverage")
            checkpoint = output / "issuers" / cik
            try:
                if not checkpoint.exists():
                    local_pins = _implementation_pins()
                    reference = sources.get(cik)
                    if reference is None:
                        collection = output / "sources" / cik
                        if not collection.exists():
                            collect_company_facts_unleased(
                                root=root, cik=cik, security_id=decisions[0]["security_id"], output=collection, settings=settings
                            )
                        reference = ArtifactReference(
                            path=str(collection / "_manifest.json"), sha256=file_sha256(collection / "_manifest.json")
                        )
                    source, payload = _source(root, reference, decisions[0]["security_id"], cik, local_pins)
                    decoded_by_security = {}
                    feature_rows = []
                    for row in decisions:
                        guard("investment monthly feature snapshot")
                        security = row["security_id"]
                        if security not in decoded_by_security:
                            decoded_by_security[security] = adapt_sec_company_facts(
                                payload,
                                security_id=security,
                                cik=cik,
                                source_sha256=source["files"]["company_facts.json"],
                                retrieved_at=datetime.fromisoformat(source["retrieved_at"]),
                                accepted_at_by_accession=index.get(cik),
                                source_policy="reconstructed_research",
                            )
                        snapshot = build_fundamental_snapshot(
                            decoded_by_security[security].facts,
                            security_id=security,
                            cik=cik,
                            decision_time=datetime.fromisoformat(row["decision_time_utc"]),
                        )
                        feature_rows.append({**row, "snapshot": snapshot.model_dump(mode="json")})
                    issuer_report = {
                        "schema": "market_predictor.investment_issuer_coverage",
                        "status": "issuer_complete",
                        "job_identity": job_identity,
                        "cik": cik,
                        "decision_count": len(decisions),
                        "source_files": local_pins,
                        "source_manifest": reference.model_dump(),
                        **FLAGS,
                    }
                    _publish(
                        checkpoint,
                        issuer_report,
                        local_pins,
                        partial(_save_feature_rows, rows=feature_rows),
                    )
                checkpoint_ref = ArtifactReference(
                    path=str(checkpoint / "_manifest.json"), sha256=file_sha256(checkpoint / "_manifest.json")
                )
                checkpoint_manifest = _object(root, checkpoint_ref, pins)
                if checkpoint_manifest.get("job_identity") != job_identity or checkpoint_manifest.get("cik") != cik:
                    raise DataReadinessError("investment issuer checkpoint identity differs")
                for name, digest in checkpoint_manifest["source_files"].items():
                    if file_sha256(_source_pin_path(root, name)) != digest:
                        raise DataReadinessError("investment issuer checkpoint source changed")
                    pins[str(_source_pin_path(root, name))] = digest
                feature_ref = ArtifactReference(
                    path=str(checkpoint / "features.json"), sha256=checkpoint_manifest["files"]["features.json"]
                )
                feature_rows = _object(root, feature_ref, pins)["rows"]
                if [r["decision_id"] for r in feature_rows] != [r["decision_id"] for r in decisions]:
                    raise DataReadinessError("investment issuer checkpoint roster differs")
                for row in feature_rows:
                    for feature in FEATURE_NAMES:
                        state = "available" if row["snapshot"]["metrics"][feature] is not None else "missing"
                        summary[(feature, row["decision_session"][:4], row["sector"], state)] += 1
                issuer_artifacts.append({"cik": cik, **checkpoint_ref.model_dump()})
            except MemoryBudgetError:
                raise
            except (OSError, ValueError, RuntimeError, DataReadinessError) as exc:
                if checkpoint.exists():
                    # A published checkpoint is immutable evidence. Corruption is
                    # a job refusal, never an ordinary provider-coverage gap.
                    raise
                failures.append({"cik": cik, "reason": str(exc), "decision_rows": str(len(decisions))})
                for row in decisions:
                    for feature in FEATURE_NAMES:
                        summary[(feature, row["decision_session"][:4], row["sector"], "issuer_failure")] += 1
        for row in rows:
            if row["status"] != "eligible":
                for feature in FEATURE_NAMES:
                    summary[(feature, row["decision_session"][:4], row["sector"] or "Unknown", row["status"])] += 1
        report = {
            "schema": "market_predictor.investment_coverage",
            "status": "coverage_complete_with_gaps" if failures else "coverage_complete",
            "job_identity": job_identity,
            "roster_manifest_sha256": roster_manifest_sha256,
            "roster_sha256": roster_digest,
            "declared_rows": len(rows),
            "eligible_rows": sum(len(v) for v in groups.values()),
            "eligible_issuers": len(groups),
            "completed_issuers": len(issuer_artifacts),
            "failed_issuers": len(failures),
            "feature_names": list(FEATURE_NAMES),
            "feature_contract_sha256": FEATURE_CONTRACT_SHA256,
            "status_counts": dict(sorted(Counter(r["status"] for r in rows).items())),
            "source_files": pins,
            "availability_basis": "reconstructed_research",
            **FLAGS,
        }

        def save(directory: Path) -> None:
            write_json_object(
                directory / "coverage.json",
                {
                    "counts": [
                        dict(zip(("feature", "year", "sector", "state", "rows"), (*key, value), strict=True))
                        for key, value in sorted(summary.items())
                    ],
                    "issuer_artifacts": issuer_artifacts,
                    "failures": failures,
                    "population_gaps": [r for r in rows if r["status"] != "eligible"],
                },
            )

        return _publish(output / "final", report, pins, save)
