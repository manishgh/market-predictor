"""Investment-only raw extension: verified benchmarks and quarantined stock candidates."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dividend_currencies import CurrencyCorroboration
from market_predictor.investment.raw_acquisition import QUARANTINE, verify_investment_raw_acquisition
from market_predictor.investment.stock_identity import (
    StockIdentityManifest,
    StockIdentityProof,
    StockIdentityUnavailable,
    replay_stock_identity,
)
from market_predictor.sources.ictax import verify_ictax_archive
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin


class InvestmentRawSourceExtension(HoldingContract):
    acquisition_request: SourcePin
    plan_directory: str
    plan_authority_sha256: Sha256
    archive_directory: str
    benchmark_tax_archive: SourcePin
    stock_identity: SourcePin | None = None


@dataclass(frozen=True)
class VerifiedInvestmentRawExtension:
    request: InvestmentRawSourceExtension
    acquisition: dict[str, Any]
    tax_archive: dict[str, Any]
    source_files: dict[str, str]
    stock_proofs: tuple[StockIdentityProof, ...] = ()

    @property
    def outcome_end(self) -> date:
        return date.fromisoformat(self.acquisition["outcome_end"])


def recheck_extension_sources(root: Path, files: Mapping[str, str]) -> None:
    for name, digest in files.items():
        guard("investment extension source hash verification")
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError("investment extension source changed")


def verify_investment_raw_extension(root: Path, request: InvestmentRawSourceExtension) -> VerifiedInvestmentRawExtension:
    """Replay owner plan/transport and official tax identity before holding the parent lease."""
    root = root.resolve()
    with heavy_job_lease("prepare-investment-raw-extension", runtime_dir=root / heavy_job_runtime_dir()):
        guard("investment raw extension request")
        if file_sha256(inside(root, request.acquisition_request.path)) != request.acquisition_request.sha256:
            raise DataReadinessError("investment extension acquisition request hash differs")
    raw = verify_investment_raw_acquisition(root, Path(request.acquisition_request.path), Path(request.plan_directory),
        Path(request.archive_directory), expected_plan_authority_sha256=request.plan_authority_sha256)
    tax = verify_ictax_archive(root=root, manifest=Path(request.benchmark_tax_archive.path),
        manifest_sha256=request.benchmark_tax_archive.sha256)
    files: dict[str, str] = {}
    for inventory in (raw["source_files"], tax["source_files"],
            {request.acquisition_request.path: request.acquisition_request.sha256,
             request.benchmark_tax_archive.path: request.benchmark_tax_archive.sha256}):
        for name, digest in inventory.items():
            normalized = str(inside(root, name))
            if normalized in files and files[normalized] != digest:
                raise DataReadinessError("investment extension has conflicting source pins")
            files[normalized] = digest
    proofs: tuple[StockIdentityProof, ...] = ()
    with heavy_job_lease("bind-investment-raw-extension", runtime_dir=root / heavy_job_runtime_dir()):
        if request.stock_identity is not None:
            path = inside(root, request.stock_identity.path)
            if file_sha256(path) != request.stock_identity.sha256 or path.stat().st_size > 16 * 1024**2:
                raise DataReadinessError("investment stock identity proof manifest differs")
            proofs = StockIdentityManifest.model_validate_json(path.read_bytes()).proofs
            if len(proofs) > 631 or len({(item.security_id, item.ticker) for item in proofs}) != len(proofs):
                raise DataReadinessError("investment stock identity proof population is ambiguous")
            files[str(path)] = request.stock_identity.sha256
        recheck_extension_sources(root, files)
    return VerifiedInvestmentRawExtension(request, raw, tax, files, proofs)


def compose_investment_raw_extension(*, root: Path, extension: VerifiedInvestmentRawExtension,
    parent_selection: dict[str, Any], parent_selection_pin: SourcePin, roster_rows_pin: SourcePin,
    action_query_pin: SourcePin, corroborations: Mapping[str, CurrencyCorroboration]
) -> dict[str, Any]:
    """Bind benchmarks and only positively proven stock observation intervals.

    Called inside the independently verified parent price/action source lease, after
    InvestmentCorporateActions has replayed all supplied corroborations. This does
    not waive any later corporate-action, currency or holding-accounting check.
    """
    root = root.resolve()
    recheck_extension_sources(root, extension.source_files)
    raw, tax = extension.acquisition, extension.tax_archive
    for key, expected in (("source_selection", parent_selection_pin), ("roster", roster_rows_pin),
                          ("action_query", action_query_pin)):
        actual = SourcePin.model_validate(raw["parent_pins"][key])
        if (inside(root, actual.path) != inside(root, expected.path) or actual.sha256 != expected.sha256):
            raise DataReadinessError("investment extension differs from verified parent " + key)
    if raw["identity_admission"] != {"stock": QUARANTINE, "benchmark": "captured_only"}:
        raise DataReadinessError("investment extension cannot assert stock identity admission")
    original = parent_selection["segments"]
    parent_benchmarks = {segment["artifact"]["ticker"] for segment in original if segment["artifact"]["role"] == "benchmark"}
    # This owner-replayed roster/plan scope cannot shrink when a capture fails.
    # Parent selections can also retain benchmarks for excluded investment sectors.
    planned_benchmarks = raw["planned_benchmark_symbols"]
    benchmarks = set(planned_benchmarks)
    if (not {"SPY", "QQQ"}.issubset(benchmarks) or len(benchmarks) != len(planned_benchmarks)
            or not benchmarks.issubset(parent_benchmarks)):
        raise DataReadinessError("investment extension planned benchmark scope differs from verified parent")
    end = extension.outcome_end
    identities: dict[str, str] = {}
    for symbol in benchmarks:
        proven = [item for item in corroborations.values() if item.symbol == symbol and item.evidence]
        isins = {item.tax.isin for item in proven}
        cusips = {item.identity.cusip for item in proven}
        if len(isins) != 1 or len(cusips) != 1:
            raise DataReadinessError("investment benchmark official ticker/CUSIP/ISIN identity is unproven")
        isin = next(iter(isins))
        rows = [row for row in tax["records"] if row["symbol"] == symbol and row["year"] == end.year]
        if len(rows) != 1 or rows[0]["status"] not in {"responses_archived", "retained_sources_verified"}:
            raise DataReadinessError("investment benchmark terminal-year identity source is missing")
        row = rows[0]
        if row["isin"] != isin:
            raise DataReadinessError("investment benchmark terminal-year ISIN differs from official identity")
        source = row["sources"]["structured"]["body"]
        path = inside(root, source["path"])
        if extension.source_files.get(str(path)) != source["sha256"] or file_sha256(path) != source["sha256"]:
            raise DataReadinessError("investment benchmark tax identity body is not verified")
        calculation = json.loads(path.read_bytes())["data"]["calculation"]
        if calculation["isin"] != isin or calculation["referenceDate"] != row["request"]["referenceDate"]:
            raise DataReadinessError("investment benchmark tax identity/date contradicts request")
        identities[symbol] = isin
    added: list[dict[str, Any]] = []
    quarantined_stocks: list[dict[str, Any]] = []
    stock_audit: list[dict[str, Any]] = []
    proof_lookup = {(item.security_id, item.ticker): item for item in extension.stock_proofs}
    if extension.stock_proofs and extension.request.stock_identity is None:
        raise DataReadinessError("investment stock proofs lack the independently pinned manifest")
    stock_cache: dict[str, Any] = {}
    covered_benchmarks: set[str] = set()
    for artifact in raw["completed_artifacts"]:
        guard("investment extension segment binding")
        if artifact["role"] not in {"stock", "benchmark"}:
            raise DataReadinessError("investment extension artifact role is unsupported")
        if artifact["role"] == "stock":
            if artifact["identity_admission"] != QUARANTINE:
                raise DataReadinessError("investment stock candidate lost quarantine")
            # Acquisition evidence remains hash-bound in the extension audit. A
            # quarantined candidate cannot own sessions in a usable selection.
            quarantined_stocks.append(artifact)
            proof = proof_lookup.get((artifact["security_id"], artifact["ticker"]))
            if artifact["status"] != "observed" or proof is None:
                stock_audit.append({"unit_id": artifact.get("unit_id"), "admitted_sessions": 0,
                    "missing_reasons": ["raw_stock_observations_unavailable" if artifact["status"] != "observed"
                                        else "raw_stock_extension_identity_unreviewed"]})
                continue
            from market_predictor.swing.labels.holding_paths import holding_calendar

            first, last = date.fromisoformat(artifact["start_date"]), date.fromisoformat(artifact["end_date"])
            if first > last or last > end:
                raise DataReadinessError("investment stock extension escapes declared end")
            parents = [segment for segment in original if segment["artifact"]["security_id"] == artifact["security_id"]]
            expected = artifact["candidate_requirements"]["candidate_parent_segment"]
            if expected not in parents or any(artifact[key] != expected["artifact"][key]
                    for key in ("security_id", "ticker", "provider_symbol", "role")):
                raise DataReadinessError("investment stock provider/security binding differs")
            if artifact["candidate_requirements"].get("asof") != str(last):
                raise DataReadinessError("investment stock provider asof differs from exact owner plan")
            sessions = holding_calendar(first, last)
            try:
                assert extension.request.stock_identity is not None
                replayed = replay_stock_identity(root, proof, sessions=sessions, source_files=extension.source_files,
                    interpretation_sha256=extension.request.stock_identity.sha256,
                    cache=stock_cache)
            except StockIdentityUnavailable as exc:
                stock_audit.append({"unit_id": artifact["unit_id"], "admitted_sessions": 0,
                    "missing_reasons": ["stock_identity_unavailable:" + str(exc)]})
                continue
            missing = set(artifact["missing_sessions"])
            admitted = {day for day in replayed.proven_sessions if day.isoformat() not in missing}
            # Each source owner is a contiguous exact run. Never overlap an old
            # selected span, including a parent span with an interior source gap.
            if any(date.fromisoformat(segment["first_session"]) <= day <= date.fromisoformat(segment["last_session"])
                    for day in admitted for segment in (*original, *added)
                    if segment["artifact"]["security_id"] == artifact["security_id"]):
                raise DataReadinessError("investment stock extension overlaps selected ownership")
            runs: list[list[date]] = []
            previous_admitted = False
            for day in sessions:
                if day in admitted:
                    if not previous_admitted:
                        runs.append([])
                    runs[-1].append(day)
                previous_admitted = day in admitted
            for run in runs:
                added.append({"archive": artifact["archive"], "first_session": str(run[0]), "last_session": str(run[-1]),
                    "missing_sessions": [], "source_admission": "stock_identity_reconstructed_research",
                    "artifact": {key: value for key, value in artifact.items() if key not in
                        {"archive", "candidate_requirements", "identity_admission", "ownership_admitted", "missing_sessions"}}})
            if len(admitted) == len(sessions):
                # This original capture remains unchanged; the composition has
                # now resolved every requested date for this candidate unit.
                quarantined_stocks.pop()
            stock_audit.append({"unit_id": artifact["unit_id"], "security_id": artifact["security_id"],
                "admitted_sessions": len(admitted), "requested_sessions": len(sessions),
                "unproven_identity_sessions": [str(day) for day in sessions if day not in replayed.proven_sessions],
                "missing_observation_sessions": sorted(missing), "missing_reasons": list(replayed.missing_reasons),
                "admission_basis": "reconstructed_research_current_verification"})
            continue
        if artifact["status"] != "observed":
            continue
        first, last = date.fromisoformat(artifact["start_date"]), date.fromisoformat(artifact["end_date"])
        if first > last or last > end:
            raise DataReadinessError("investment raw extension escapes declared end")
        identity, symbol = artifact["security_id"], artifact["ticker"]
        parents = [segment for segment in original if segment["artifact"]["security_id"] == identity]
        if not parents:
            raise DataReadinessError("investment extension changes the frozen security population")
        expected = artifact["candidate_requirements"]["candidate_parent_segment"]
        if expected not in parents or any(artifact[key] != expected["artifact"][key]
                for key in ("security_id", "ticker", "provider_symbol", "role")):
            raise DataReadinessError("investment extension provider/security binding differs")
        for segment in (*original, *added):
            if (segment["artifact"]["security_id"] == identity
                    and date.fromisoformat(segment["first_session"]) <= last
                    and date.fromisoformat(segment["last_session"]) >= first):
                # A source with an interior missing date needs an explicit split in
                # its selection; never silently prefer a newly fetched replacement.
                raise DataReadinessError("investment extension overlaps existing selected ownership")
        if identity != "benchmark:" + symbol or symbol not in identities or artifact["missing_sessions"]:
            raise DataReadinessError("investment benchmark raw extension identity or sessions are incomplete")
        admission = "benchmark_identity_verified"
        covered_benchmarks.add(symbol)
        added.append({"archive": artifact["archive"], "first_session": first.isoformat(), "last_session": last.isoformat(),
            "missing_sessions": artifact["missing_sessions"], "source_admission": admission,
            "artifact": {key: value for key, value in artifact.items() if key not in
                {"archive", "candidate_requirements", "identity_admission", "ownership_admitted", "missing_sessions"}}})
    if covered_benchmarks != benchmarks:
        raise DataReadinessError("investment raw extension lacks complete mandatory benchmark coverage")
    # Unsupported class facts remain local gaps, but every inspected source is
    # still part of the immutable publication's final hash recheck.
    for name, digest in stock_cache.get("source_files", {}).items():
        if name in extension.source_files and extension.source_files[name] != digest:
            raise DataReadinessError("investment stock identity cache has conflicting source pins")
        extension.source_files[name] = digest
    return {"schema": "market_predictor.investment_raw_source_selection",
            "parent_selection": parent_selection_pin.model_dump(mode="json"),
            "segments": [*copy.deepcopy(original), *added],
            "investment_outcome_end": end.isoformat(),
            "stock_extensions_quarantined": any(item["admitted_sessions"] < item.get("requested_sessions", 1)
                                                 for item in stock_audit),
            "stock_identity_admitted_artifact_count": sum(item["admitted_sessions"] > 0 for item in stock_audit),
            "stock_identity_admission_audit": stock_audit,
            "quarantined_stock_artifact_count": len(quarantined_stocks),
            "quarantined_stock_observed_artifact_count": sum(item["status"] == "observed" for item in quarantined_stocks),
            "quarantined_stock_security_ids": sorted({item["security_id"] for item in quarantined_stocks})}
