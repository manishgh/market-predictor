"""Resumable full-roster investment labels from independently replayed sources."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.corporate_actions import InvestmentCorporateActions
from market_predictor.investment.label_materialization import InvestmentSourceQualification, compile_investment_label
from market_predictor.investment.label_sources import InvestmentSourceReader
from market_predictor.research.investment_currencies import load_investment_currency_evidence
from market_predictor.research.investment_features import FLAGS, ArtifactReference, _object, _publish, _read
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Timestamp
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence
from market_predictor.swing.datasets.corrected_outcome_admission import corrected_ticker
from market_predictor.swing.datasets.corrected_outcomes import load_corrected_outcome_policy, verified_corrected_research_sources


class InvestmentLabelMaterializationRequest(HoldingContract):
    roster_manifest: ArtifactReference
    target_config: ArtifactReference
    currency_manifest: ArtifactReference | None = None
    nominal_unpaid_claim_valuation: Literal[True]
    evaluation_as_of_utc: Timestamp


def _path(root: Path, name: str) -> Path:
    return Path(name) if Path(name).is_absolute() else inside(root, name)


def _check(root: Path, pins: dict[str, str]) -> None:
    for name, digest in pins.items():
        if file_sha256(_path(root, name)) != digest:
            raise DataReadinessError(f"investment label source changed: {name}")


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = ("research/investment_labels.py", "research/investment_currencies.py", "research/investment_features.py",
        "investment/label_sources.py", "investment/label_materialization.py", "investment/corporate_actions.py",
        "investment/contracts.py", "investment/targets.py", "swing/contracts/holding_accounting.py",
        "swing/evaluation/holding_accounting.py", "swing/labels/holding_paths.py", "swing/labels/holding_identity.py",
        "swing/datasets/action_evidence.py", "swing/datasets/corrected_outcomes.py")
    return {str((package / name).resolve()): file_sha256(package / name) for name in names}


def _interval(row: dict[str, Any], horizon: Literal[63, 252]) -> tuple[tuple[Any, ...], str, str]:
    calendar = xcals.get_calendar("XNYS")
    dates = calendar.sessions_window(pd.Timestamp(row["decision_session"]), horizon + 1)[1:]
    sessions = tuple(day.date() for day in dates)
    return sessions, calendar.session_open(dates[0]).isoformat(), calendar.session_close(dates[-1]).isoformat()


def _outside_row(row: dict[str, Any], horizon: Literal[63, 252]) -> dict[str, Any]:
    _, entry, end = _interval(row, horizon)
    return {"decision_id": row["decision_id"], "security_id": row["security_id"],
        "decision_session": row["decision_session"], "horizon_sessions": horizon, "entry_time_utc": entry,
        "economic_maturity_utc": end, "label_available_at_utc": None, "status": "unresolved", "label_eligible": False,
        "missing_reasons": row["qualification_reasons"] or [row["status"]], "components": [],
        "request": None, "request_sha256": None, "target": None, **FLAGS}


def _membership_qualification(*, sources: dict[str, Any], row: dict[str, Any], horizon: Literal[63, 252],
    evidence: EvidenceReference) -> InvestmentSourceQualification:
    members = sources["memberships"]
    cutoff = pd.Timestamp(row["decision_time_utc"])
    start = pd.to_datetime(members.effective_from_utc, utc=True)
    end = pd.to_datetime(members.effective_to_utc, utc=True)
    available = pd.to_datetime(members.available_at_utc, utc=True)
    active = members.loc[start.le(cutoff) & (end.isna() | end.gt(cutoff)) & available.le(cutoff)]
    owners = active.loc[active.ticker.eq(row["ticker"])]
    exact = owners.loc[owners.security_id.eq(row["security_id"]) & owners.sector.eq(row["sector"])
        & owners.primary_benchmark.eq(row["sector_benchmark"])]
    missing = () if len(owners) == 1 and len(exact) == 1 else ("decision_membership_identity_sector_or_competing_owner_unproven",)
    # Universe membership determines admission at the decision only. The reader's
    # independently reconstructed exact source-plan segment owns the entire outcome
    # interval; leaving the index never creates a sale or an automatic label gap.
    return InvestmentSourceQualification((evidence,), missing)


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for horizon in (63, 252):
        selected = [row for row in rows if row["horizon_sessions"] == horizon]
        result[str(horizon)] = {"population_rows": len(selected),
            "scope_eligible_rows": sum(row["roster_status"] == "eligible" for row in selected),
            "economically_mature_rows": sum(row["economically_mature_at_evaluation"] for row in selected),
            "source_qualified_rows": sum(row["status"] == "qualified" for row in selected),
            "currently_available_rows": sum(row["available_by_evaluation_as_of"] for row in selected),
            "unresolved_rows": sum(row["status"] != "qualified" for row in selected),
            "missing_reason_counts": dict(sorted(Counter(reason for row in selected for reason in row["missing_reasons"]).items()))}
    return result


def _save_rows(directory: Path, records: list[dict[str, Any]]) -> None:
    write_json_object(directory / "labels.json", {"rows": records})


def materialize_investment_labels(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Preload action replay, then retain its one shared source lease through publication."""
    root, output = root.resolve(), output.resolve()
    if (output / "final").exists():
        raise FileExistsError("completed immutable investment labels already exist")
    # Preparation must refuse pressure/contention before even decoding the roster.
    # Release this lease before the independent action replay acquires its own.
    with heavy_job_lease("prepare-investment-labels", runtime_dir=root / heavy_job_runtime_dir()):
        guard("investment label preparation")
        pins = _implementation()
        request_ref = ArtifactReference(path=str(request), sha256=request_sha256)
        parsed = InvestmentLabelMaterializationRequest.model_validate_json(_read(root, request_ref, pins))
        roster = _object(root, parsed.roster_manifest, pins)
        if roster.get("schema") != "market_predictor.investment_coverage_roster" or roster.get("status") != "roster_complete":
            raise DataReadinessError("investment labels require complete frozen roster")
        for name, digest in roster["source_files"].items():
            pins[str(_path(root, name))] = digest
        _check(root, pins)
        parent = Path(parsed.roster_manifest.path).parent
        rows_file = ArtifactReference(path=str(parent / "roster.json"), sha256=roster["files"]["roster.json"])
        rows = _object(root, rows_file, pins)["rows"]
        if len(rows) != roster["declared_rows"] or len({row["decision_id"] for row in rows}) != len(rows):
            raise DataReadinessError("investment roster population count or unique decisions differ")
        coverage_request = _object(root, ArtifactReference(path=str(parent / "request.json"),
            sha256=roster["files"]["request.json"]), pins)
        membership_pin = ArtifactReference.model_validate_json(json.dumps(coverage_request["membership"]))
        _read(root, membership_pin, pins)
        policy = load_corrected_outcome_policy(root, Path(parsed.target_config.path), parsed.target_config.sha256)
    evidence = load_corporate_action_evidence(root=root, config=Path(policy.action_config.path), archive=Path(policy.action_archive),
        expected_audit_sha256=policy.action_audit_sha256, memory_guard=lambda: guard("investment action evidence"))
    with verified_corrected_research_sources(root, Path(parsed.target_config.path), parsed.target_config.sha256,
        policy, evidence, memory_guard=lambda: guard("investment label source replay")) as sources:
        pins.update({str(_path(root, name)): digest for name, digest in sources["source_files"].items()})
        job_identity = json_sha256({"request_sha256": request_sha256, "implementation": _implementation()})
        output.mkdir(parents=True, exist_ok=True)
        job_path = output / "_job.json"
        if job_path.exists():
            job = json.loads(job_path.read_text(encoding="utf-8"))
            if job.get("job_identity") != job_identity:
                raise DataReadinessError("investment label resume request or implementation differs")
        else:
            job = {"job_identity": job_identity, "verified_at_utc": datetime.now(UTC).isoformat()}
            write_json_object(job_path, job)
        verified_at = datetime.fromisoformat(job["verified_at_utc"])
        if verified_at.utcoffset() is None:
            raise DataReadinessError("investment job verification clock must be aware")
        pins[str(job_path)] = file_sha256(job_path)
        policy_ref = EvidenceReference(reference=str(inside(root, request)), artifact_sha256=request_sha256,
            record_locator="explicit nominal_unpaid_claim_valuation=true; research policy, not observed market claim price",
            interpretation_policy_sha256=request_sha256, retrieved_at=verified_at, available_at=verified_at)
        membership_ref = EvidenceReference(reference=membership_pin.path, artifact_sha256=membership_pin.sha256,
            record_locator="current independent verification of reconstructed historical security membership intervals",
            interpretation_policy_sha256=request_sha256, retrieved_at=verified_at, available_at=verified_at)
        currencies = {} if parsed.currency_manifest is None else load_investment_currency_evidence(root,
            Path(parsed.currency_manifest.path), parsed.currency_manifest.sha256, pins, interpretation_policy_sha256=request_sha256)
        actions = InvestmentCorporateActions(root, evidence, sources["action_index"], policy.official_windows,
            policy_ref, policy.numerical_end, currency_evidence=currencies)
        common_path = output / "_verified_sources.json"
        common_sources = {name: digest for name, digest in pins.items() if name != str(common_path)}
        if common_path.exists():
            if json.loads(common_path.read_text(encoding="utf-8")) != common_sources:
                raise DataReadinessError("investment resumed common source inventory differs")
        else:
            write_json_object(common_path, common_sources)
        common_digest = file_sha256(common_path)
        pins[str(common_path)] = common_digest
        reader = InvestmentSourceReader(root, sources["selection"], request_sha256)
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row["security_id"], []).append(row)
        checkpoints = []
        totals: dict[str, Counter[str]] = {str(horizon): Counter() for horizon in (63, 252)}
        reasons_by_horizon: dict[str, Counter[str]] = {str(horizon): Counter() for horizon in (63, 252)}
        availability_clocks: dict[str, Counter[str]] = {str(horizon): Counter() for horizon in (63, 252)}
        for security, decisions in sorted(groups.items()):
            guard("investment serial security labels")
            directory = output / "securities" / json_sha256(security)
            if not directory.exists():
                results = []
                for row in decisions:
                    row = dict(row)
                    if row["status"] == "eligible":
                        row["ticker"] = corrected_ticker(row["security_id"], row["ticker"],
                            pd.Timestamp(row["decision_session"]).date(), policy.decision_corrections)
                    for horizon in (63, 252):
                        guard("investment monthly horizon label")
                        if row["status"] == "eligible":
                            result = compile_investment_label(decision=row, horizon=horizon, reader=reader,
                                membership=_membership_qualification(sources=sources, row=row, horizon=horizon,
                                    evidence=membership_ref), action_qualification=actions.qualify)
                            result.pop("specifications", None)
                            result["request_sha256"] = None if result["request"] is None else json_sha256(result["request"])
                        else:
                            result = _outside_row(row, horizon)
                        result["roster_status"] = row["status"]
                        result["membership_basis"] = "reconstructed_research_current_verification"
                        result["membership_availability_basis"] = "reconstructed_research"
                        result["membership_source_capture_at_utc"] = None
                        result["membership_verified_at_utc"] = verified_at.isoformat()
                        result["economically_mature_at_evaluation"] = (
                            datetime.fromisoformat(result["economic_maturity_utc"]) <= parsed.evaluation_as_of_utc)
                        available = result["label_available_at_utc"]
                        result["available_by_evaluation_as_of"] = available is not None and (
                            datetime.fromisoformat(available.replace("Z", "+00:00")) <= parsed.evaluation_as_of_utc)
                        results.append(result)
                local_pins = {str(common_path): common_digest, str(job_path): pins[str(job_path)],
                    str(inside(root, request)): request_sha256}
                for result in results:
                    for component in result["components"]:
                        for source in component["sources"]:
                            local_pins[str(_path(root, source["reference"]))] = source["artifact_sha256"]
                _publish(directory, {"schema": "market_predictor.investment_security_labels", "status": "security_complete",
                    "job_identity": job_identity, "security_id": security, "declared_rows": len(results),
                    "source_files": local_pins, **FLAGS}, local_pins,
                    partial(_save_rows, records=results))
            checkpoint_ref = ArtifactReference(path=str(directory / "_manifest.json"), sha256=file_sha256(directory / "_manifest.json"))
            checkpoint = _object(root, checkpoint_ref, pins)
            if (checkpoint.get("job_identity") != job_identity or checkpoint.get("security_id") != security
                or checkpoint.get("declared_rows") != len(decisions) * 2):
                raise DataReadinessError("investment security checkpoint differs from frozen request")
            _check(root, checkpoint["source_files"])
            pins.update(checkpoint["source_files"])
            records = _object(root, ArtifactReference(path=str(directory / "labels.json"),
                sha256=checkpoint["files"]["labels.json"]), pins)["rows"]
            expected = {(row["decision_id"], horizon) for row in decisions for horizon in (63, 252)}
            if len(records) != len(expected) or {(row["decision_id"], row["horizon_sessions"]) for row in records} != expected:
                raise DataReadinessError("investment checkpoint row population differs")
            for horizon_key, counts in _counts(records).items():
                reasons_by_horizon[horizon_key].update(counts.pop("missing_reason_counts"))
                totals[horizon_key].update(counts)
            for record in records:
                if record["status"] == "qualified" and record["label_available_at_utc"] is not None:
                    availability_clocks[str(record["horizon_sessions"])][record["label_available_at_utc"]] += 1
            checkpoints.append(checkpoint_ref.model_dump(mode="json"))
        reader.recheck()
        actions.recheck()
        _check(root, pins)
        completed_at = datetime.now(UTC)
        available_by_completion = {horizon: sum(count for clock, count in clocks.items()
            if datetime.fromisoformat(clock.replace("Z", "+00:00")) <= completed_at)
            for horizon, clocks in availability_clocks.items()}
        counts_by_horizon = {horizon: {**dict(counts), "missing_reason_counts": dict(sorted(reasons_by_horizon[horizon].items()))}
            for horizon, counts in totals.items()}
        report = {"schema": "market_predictor.investment_label_corpus", "status": "labels_complete_with_explicit_gaps",
            "request_sha256": request_sha256, "job_identity": job_identity, "source_files": pins,
            "roster_rows": len(rows), "label_rows": sum(counts["population_rows"] for counts in totals.values()),
            "counts_by_horizon": counts_by_horizon,
            "security_checkpoints": checkpoints, "evaluation_as_of_utc": parsed.evaluation_as_of_utc.isoformat(),
            "completed_at_utc": completed_at.isoformat(), "available_by_completion_rows_by_horizon": dict(available_by_completion),
            "membership_basis": "reconstructed_research_current_verification", "membership_availability_basis": "reconstructed_research",
            "membership_source_capture_at_utc": None, "membership_verified_at_utc": verified_at.isoformat(), **FLAGS}
        return _publish(output / "final", report, pins,
            lambda directory: write_json_object(directory / "checkpoints.json", {"security_checkpoints": checkpoints}))
