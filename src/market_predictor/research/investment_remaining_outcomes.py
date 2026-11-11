"""Resolve additional retained stock/SPY facts for an immutable exclusion comparison."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.action_relevance import SUPPORTED_REASONS
from market_predictor.investment.dataset import add_pins, read_object, recheck_sources, require
from market_predictor.investment.label_sources import InvestmentSourceReader
from market_predictor.research.investment_exclusion_evaluation import (
    SCHEMA as EXCLUSION_SCHEMA,
)
from market_predictor.research.investment_exclusion_evaluation import (
    InvestmentExclusionEvaluationRequest,
    _coverage,
    _filter_cohorts,
    _parent_cohorts,
    comparison_sidecar_pins,
    exclusion_coverage,
    read_fixed_selection_outcomes,
)
from market_predictor.research.investment_features import _publish
from market_predictor.research.investment_label_reuse import producer_origin
from market_predictor.research.investment_labels import (
    prepare_investment_label_components,
    read_investment_label_inputs,
    verified_investment_label_sources,
)
from market_predictor.research.investment_selection_outcomes import (
    InvestmentSelectionOutcomeRequest,
    SelectedKey,
    _amended_label_request,
    _compare_cohorts,
    _reference,
    compile_selected_pair,
    publish_selected_pair,
)
from market_predictor.research.investment_selection_outcomes import (
    _implementation as selection_implementation,
)
from market_predictor.research.investment_training import FLAGS
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_outcome_admission import corrected_ticker


class InvestmentRemainingOutcomeRequest(HoldingContract):
    schema_version: Literal["market_predictor.investment_remaining_outcome_request"]
    scope: Literal["post_observation_sensitivity"]
    parent_manifest: SourcePin
    reviewed_action_dates: SourcePin | None = None
    reviewed_parent_units: SourcePin | None = None

    @model_validator(mode="after")
    def parent_units_require_dates(self) -> InvestmentRemainingOutcomeRequest:
        if self.reviewed_parent_units is not None and self.reviewed_action_dates is None:
            raise ValueError("reviewed parent share units require their reviewed action dates")
        return self


@dataclass(frozen=True)
class SurvivorOutcomes:
    manifest: dict[str, Any]
    outcomes_pin: SourcePin
    comparisons_pin: SourcePin
    ancestor_pin: SourcePin
    ancestor: dict[str, Any]
    original_unknowns: dict[SelectedKey, dict[str, Any]]
    rows: dict[SelectedKey, dict[str, Any]]
    comparisons: dict[str, Any]
    cohorts: dict[str, list[dict[str, Any]]]


def _read_survivors(root: Path, pin: SourcePin, pins: dict[str, str]) -> SurvivorOutcomes:
    manifest = read_object(root, pin, pins)
    require(manifest.get("schema") == EXCLUSION_SCHEMA and manifest.get("scope") == "post_observation_sensitivity"
        and manifest.get("status") in {"complete_with_explicit_gaps", "complete_research_only"}
        and manifest.get("round_trip_stock_cost_bps") == 20 and manifest.get("benchmark_basis") == "stock_net_minus_spy_gross"
        and manifest.get("outcome_verification_basis") == "pinned_parent_outcomes"
        and all(manifest.get(flag) is False for flag in (*FLAGS, "reranked", "replacements_added",
            "models_refitted", "original_labels_changed", "training_ready")),
        "remaining outcomes require the closed pinned exclusion comparison")
    exclusion = InvestmentExclusionEvaluationRequest.model_validate_json(json.dumps(
        read_object(root, SourcePin.model_validate(manifest["request"]), pins)))
    ancestor_pin = exclusion.parent_manifest
    require(ancestor_pin == SourcePin.model_validate(manifest["parent_manifest"]), "exclusion ancestor pin differs")
    ancestor = read_fixed_selection_outcomes(root, ancestor_pin, pins)
    require(ancestor.outcomes_pin == SourcePin.model_validate(manifest["parent_outcomes"])
        and ancestor.comparisons_pin == SourcePin.model_validate(manifest["parent_comparisons"])
        and manifest["parent_selection_sha256"] == ancestor.manifest["selection_sha256"],
        "exclusion ancestor outcomes, comparisons or ordered selections differ")
    retained, comparison, removed = _filter_cohorts(exclusion.exclusions, ancestor.cohorts, ancestor.outcomes, ancestor.comparisons)
    outcomes_pin, comparisons_pin = comparison_sidecar_pins(root, pin, manifest)
    actual_rows = read_object(root, outcomes_pin, pins)
    actual_comparisons = read_object(root, comparisons_pin, pins)
    require(json_sha256(actual_rows) == json_sha256({"rows": list(retained.values())})
        and json_sha256(actual_comparisons) == json_sha256({"views": comparison})
        and manifest["exclusions"] == removed
        and manifest["coverage"] == exclusion_coverage(ancestor.outcomes, ancestor.comparisons, retained, comparison)
        and manifest["comparison_views"] == len(comparison)
        and manifest["comparison_cohorts"] == sum(view["date_count"] for view in comparison.values())
        and manifest["empty_cohorts"] == sum(view["empty_dates"] for view in comparison.values()),
        "exclusion survivors, original clocks, coverage or ordered placements differ")
    original_unknowns = {key: {**row, "status": "unresolved"} for key, row in ancestor.outcomes.items()
        if row["original_status"] == "unresolved"}
    return SurvivorOutcomes(manifest, outcomes_pin, comparisons_pin, ancestor_pin, ancestor.manifest, original_unknowns,
        retained, comparison, _parent_cohorts(actual_comparisons, retained))


def _potential_action_ids(row: dict[str, Any]) -> frozenset[str]:
    return frozenset(parts[3] for reason in row["pair_missing_reasons"]
        if len(parts := reason.split(":")) == 5 and parts[:2] == ["stock", "action"]
        and parts[2] in {"name_changes", "spin_offs"} and parts[4] in SUPPORTED_REASONS)


def _reviewed_action_ids(root: Path, request: InvestmentRemainingOutcomeRequest, pins: dict[str, str]) -> frozenset[str]:
    return frozenset(record["action_id"]
        for pin in (request.reviewed_action_dates, request.reviewed_parent_units) if pin is not None
        for record in read_object(root, pin, pins)["records"])


def _comparison(parent: SurvivorOutcomes, rows: dict[SelectedKey, dict[str, Any]]) -> dict[str, Any]:
    """Keep all original removal evidence and replay only economics and completeness."""
    replayed = _compare_cohorts(parent.cohorts, rows)
    return {view: {**parent.comparisons[view], **result,
        "dates": [{**old, **new} for old, new in zip(parent.comparisons[view]["dates"], result["dates"], strict=True)]}
        for view, result in replayed.items()}


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {**selection_implementation(), **{str(package / name): file_sha256(package / name) for name in (
        "commands/investment_research.py", "research/investment_remaining_outcomes.py",
        "research/investment_exclusion_evaluation.py", "investment/action_relevance.py",
        "swing/datasets/reviewed_action_dates.py", "swing/datasets/reviewed_parent_units.py",
        "swing/datasets/reviewed_action_sources.py", "swing/datasets/action_windows.py",
    )}}


def resolve_remaining_investment_outcomes(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Compile only supported unknown survivors and keep all fixed cohort placements."""
    root = root.resolve()
    output = inside(root, output)
    if output.exists():
        raise FileExistsError(output)
    runtime = heavy_job_runtime_dir()
    request_pin = SourcePin(path=str(inside(root, request)), sha256=request_sha256)
    with heavy_job_lease("prepare-remaining-investment-outcomes", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("remaining investment outcome preparation")
        implementation = _implementation()
        pins = dict(implementation)
        parsed = InvestmentRemainingOutcomeRequest.model_validate_json(json.dumps(read_object(root, request_pin, pins)))
        parent = _read_survivors(root, parsed.parent_manifest, pins)
        ancestor_request_pin = SourcePin.model_validate(parent.ancestor["request"])
        ancestor_request = InvestmentSelectionOutcomeRequest.model_validate_json(json.dumps(
            read_object(root, ancestor_request_pin, pins)))
        require(parent.ancestor["request_sha256"] == ancestor_request_pin.sha256
            and all(parent.ancestor[name] == value
                for name, value in ancestor_request.model_dump(mode="json", exclude={"schema_version", "scope"}).items()),
            "ancestor selected outcome request does not bind its source amendments")
        inputs = read_investment_label_inputs(root, _reference(ancestor_request.label_request), pins)
        amended = _amended_label_request(root, ancestor_request, inputs.request, parent.original_unknowns, pins)
        roster = {row["decision_id"]: row for row in inputs.rows}
        unknown = {key: row for key, row in parent.rows.items() if row["status"] == "unresolved"}
        candidate_ids = {key: _potential_action_ids(row) for key, row in unknown.items()}
        action_ids = _reviewed_action_ids(root, parsed, pins).union(*candidate_ids.values())
        require(inputs.request.decision_ids is None and all(key[0] in roster for key in unknown),
            "remaining outcomes require the original full fixed decision roster")

    with verified_investment_label_sources(root, amended, inputs.policy) as (extension, evidence, sources):
        add_pins(root, pins, sources["source_files"])
        verified_at = datetime.now(UTC)
        policy_ref = EvidenceReference(reference=ancestor_request.label_request.path,
            artifact_sha256=ancestor_request.label_request.sha256,
            record_locator="original nominal-unpaid-claim valuation and 20 bp stock cost for fixed selected stock/SPY comparison",
            interpretation_policy_sha256=request_sha256, retrieved_at=verified_at, available_at=verified_at)
        membership_ref = EvidenceReference(reference=inputs.membership.path, artifact_sha256=inputs.membership.sha256,
            record_locator="current verification of the reconstructed security membership intervals",
            interpretation_policy_sha256=request_sha256, retrieved_at=verified_at, available_at=verified_at)
        actions, raw_selection = prepare_investment_label_components(root, amended, inputs.policy, inputs.roster_rows,
            extension, evidence, sources, pins, policy_ref, request_sha256, relevance_action_ids=action_ids,
            reviewed_action_dates=parsed.reviewed_action_dates, reviewed_parent_units=parsed.reviewed_parent_units)
        reader = InvestmentSourceReader(root, raw_selection, request_sha256)
        producer_export, producer_commit = producer_origin(implementation, pins)
        report: dict[str, Any] = {
            "schema": "market_predictor.investment_remaining_outcomes", "scope": parsed.scope,
            "status": "complete_with_explicit_gaps", "request": request_pin.model_dump(mode="json"),
            **parsed.model_dump(mode="json", exclude={"schema_version", "scope"}),
            "parent_outcomes": parent.outcomes_pin.model_dump(mode="json"),
            "parent_comparisons": parent.comparisons_pin.model_dump(mode="json"),
            "ancestor_manifest": parent.ancestor_pin.model_dump(mode="json"),
            "parent_selection_sha256": parent.manifest["parent_selection_sha256"], "exclusions": parent.manifest["exclusions"],
            "round_trip_stock_cost_bps": 20, "benchmark_basis": "stock_net_minus_spy_gross",
            "comparison_views": parent.manifest["comparison_views"], "comparison_cohorts": parent.manifest["comparison_cohorts"],
            "empty_cohorts": parent.manifest["empty_cohorts"], "verified_at_utc": verified_at.isoformat(),
            "source_files": pins, "implementation_files": implementation,
            "producer_export": None if producer_export is None else producer_export.model_dump(mode="json"),
            "producer_commit": producer_commit, "selection_changed": False, "models_refitted": False,
            "original_labels_changed": False, "training_ready": False, **FLAGS,
        }
        rows = dict(parent.rows)
        attempts = {key: {name: row[name] for name in ("decision_id", "horizon_sessions", "security_id", "decision_session")}
            for key, row in unknown.items()}
        reviewed_validator = partial(actions.validate_sources, reviewed_relevance=True)

        def save(staged: Path) -> None:
            proof_pins: dict[str, str] = {}
            previous_security = None
            for key in sorted(unknown, key=lambda key: (unknown[key]["security_id"], unknown[key]["decision_session"], key[1])):
                row, attempt = unknown[key], attempts[key]
                attempt.update(status="unattempted", relevance_action_ids=[], attempted_at_utc=None, missing_reasons=[])
                if not candidate_ids[key]:
                    continue
                decision = roster[key[0]]
                symbol = corrected_ticker(row["security_id"], row["ticker"], date.fromisoformat(row["decision_session"]),
                    inputs.policy.decision_corrections)
                eligible_ids = candidate_ids[key] & actions.reviewed_relevance_ids(
                    symbol, datetime.fromisoformat(row["entry_time_utc"]).date(),
                    datetime.fromisoformat(row["economic_maturity_utc"]).date(),
                )
                if not eligible_ids:
                    continue
                if row["security_id"] != previous_security:
                    guard("remaining investment security outcome recovery")
                    previous_security = row["security_id"]
                guard("remaining investment stock/SPY pair", reclaim_unused_memory=False)
                started = datetime.now(UTC)
                pair = compile_selected_pair(key, row, decision, reader, sources, membership_ref,
                    inputs.policy.decision_corrections, actions.validate_sources, reviewed_validator)
                attempt.update(status="unresolved", relevance_action_ids=sorted(eligible_ids),
                    attempted_at_utc=started.isoformat(), missing_reasons=pair["missing_reasons"])
                if pair["status"] != "source_validated":
                    continue
                proof_ids = {part.removeprefix("action_id=")
                    for ref in pair["specifications"]["stock"]["entry_evidence"] if ";relevance=" in ref["record_locator"]
                    for part in ref["record_locator"].split(";") if part.startswith("action_id=")}
                require(eligible_ids.issubset(proof_ids), "recovered holding lacks its selected action relevance proof")
                recovered = dict(row)
                publish_selected_pair(staged, output, key, recovered, pair, verified_at, proof_pins)
                rows[key] = recovered
                attempt["status"] = "recovered"

            comparisons = _comparison(parent, rows)
            attempted = sum(row["status"] != "unattempted" for row in attempts.values())
            recovered_count = len(proof_pins)
            report.update(attempted_keys=attempted, unattempted_keys=len(unknown) - attempted,
                newly_recovered_keys=recovered_count, still_unresolved_keys=len(unknown) - recovered_count,
                coverage={"before": _coverage(parent.rows.values(), parent.comparisons),
                    "after": _coverage(rows.values(), comparisons)})
            if all(view["date_count"] and view["complete_dates"] == view["date_count"] for view in comparisons.values()):
                report["status"] = "complete_research_only"
            reader.recheck()
            actions.recheck()
            add_pins(root, pins, reader.source_files)
            guard("remaining investment outcome publication")
            write_json_object(staged / "outcomes.json", {"rows": list(rows.values())})
            write_json_object(staged / "comparisons.json", {"views": comparisons})
            write_json_object(staged / "attempts.json", {"rows": list(attempts.values())})
            recheck_sources(root, pins)
            recheck_sources(staged, proof_pins)
            report["completed_at_utc"] = datetime.now(UTC).isoformat()

        result = _publish(output, report, pins, save)
        return {**result, **{name: report[name] for name in (
            "attempted_keys", "unattempted_keys", "newly_recovered_keys", "still_unresolved_keys", "coverage",
        )}}

