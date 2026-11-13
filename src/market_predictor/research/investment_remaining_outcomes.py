"""Resolve additional retained stock/SPY facts for an immutable exclusion comparison."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dataset import add_pins, pin_source_file, read_object, recheck_sources, require
from market_predictor.investment.label_sources import InvestmentSourceReader
from market_predictor.research.investment_exclusion_evaluation import _coverage
from market_predictor.research.investment_features import _publish
from market_predictor.research.investment_label_reuse import producer_origin
from market_predictor.research.investment_labels import (
    prepare_investment_label_components,
    read_investment_label_inputs,
    verified_investment_label_sources,
)
from market_predictor.research.investment_remaining_parent import (
    InvestmentRemainingOutcomeRequest,
    compare_survivor_cohorts,
    potential_action_ids,
    prepare_remaining_evidence,
    read_remaining_outcome_parent,
    verify_remaining_trigger_evidence,
)
from market_predictor.research.investment_selection_outcomes import (
    InvestmentSelectionOutcomeRequest,
    _amended_label_request,
    _reference,
    compile_selected_pair,
    publish_selected_pair,
)
from market_predictor.research.investment_selection_outcomes import (
    _implementation as selection_implementation,
)
from market_predictor.research.investment_training import FLAGS
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_outcome_admission import corrected_ticker


def _reviewed_action_ids(
    root: Path, request: InvestmentRemainingOutcomeRequest, pins: dict[str, str], *, paths: EvidencePaths,
) -> frozenset[str]:
    return frozenset(record["action_id"]
        for pin in (request.reviewed_action_dates, request.reviewed_parent_units) if pin is not None
        for record in read_object(root, pin, pins, paths=paths)["records"])


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {**selection_implementation(), **{str(package / name): file_sha256(package / name) for name in (
        "commands/investment_research.py", "research/investment_remaining_outcomes.py",
        "research/investment_exclusion_evaluation.py", "research/investment_remaining_parent.py",
        "investment/action_relevance.py", "investment/completed_cash_distributions.py",
        "investment/sec_document_evidence.py", "investment/reviewed_held_stock_actions.py",
        "swing/datasets/reviewed_action_dates.py", "swing/datasets/reviewed_parent_units.py",
        "swing/datasets/reviewed_action_sources.py", "swing/datasets/action_windows.py",
        "evidence/paths.py",
    )}}


def resolve_remaining_investment_outcomes(
    *, root: Path, request: Path, request_sha256: str, output: Path, data_root: Path | None = None,
    source_implementation_snapshot: SourcePin | None = None, validate_inputs_only: bool = False,
) -> dict[str, Any]:
    """Compile only supported unknown survivors and keep all fixed cohort placements."""
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    output = paths.identity(output)
    if paths.write(output).exists():
        raise FileExistsError(output)
    runtime = heavy_job_runtime_dir()
    request_pin = SourcePin(path=str(paths.identity(request)), sha256=request_sha256)
    with heavy_job_lease("prepare-remaining-investment-outcomes", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("remaining investment outcome preparation")
        implementation = _implementation()
        pins = dict(implementation)
        if source_implementation_snapshot is not None:
            pin_source_file(root, source_implementation_snapshot, pins, paths=paths)
        parsed = InvestmentRemainingOutcomeRequest.model_validate_json(json.dumps(read_object(root, request_pin, pins, paths=paths)))
        parent = read_remaining_outcome_parent(root, parsed.parent_manifest, pins, paths=paths)
        recovery_evidence = prepare_remaining_evidence(root, parent, parsed, pins, paths=paths)
        ancestor_request_pin = SourcePin.model_validate(parent.ancestor["request"])
        ancestor_request = InvestmentSelectionOutcomeRequest.model_validate_json(json.dumps(
            read_object(root, ancestor_request_pin, pins, paths=paths)))
        require(parent.ancestor["request_sha256"] == ancestor_request_pin.sha256
            and all(parent.ancestor[name] == value
                for name, value in ancestor_request.model_dump(mode="json", exclude={"schema_version", "scope"}).items()),
            "ancestor selected outcome request does not bind its source amendments")
        inputs = read_investment_label_inputs(root, _reference(ancestor_request.label_request), pins, paths=paths)
        amended = _amended_label_request(root, ancestor_request, inputs.request, parent.original_unknowns, pins, paths=paths)
        if recovery_evidence.currency_pin is not None:
            amended = amended.model_copy(update={
                "currency_corroboration_manifest": _reference(recovery_evidence.currency_pin),
            })
        roster = {row["decision_id"]: row for row in inputs.rows}
        unknown = {key: row for key, row in parent.rows.items() if row["status"] == "unresolved"}
        candidate_ids = {key: recovery_evidence.candidate_ids(parent, key) for key in unknown}
        action_ids = _reviewed_action_ids(root, parsed, pins, paths=paths).union(
            recovery_evidence.cash, recovery_evidence.held_actions, *candidate_ids.values(),
            *(potential_action_ids(unknown[key]) for key, ids in candidate_ids.items() if ids))
        require(inputs.request.decision_ids is None and all(key[0] in roster for key in unknown),
            "remaining outcomes require the original full fixed decision roster")

    # Raw and action archive preparation acquire this same non-reentrant lease.
    # The verified source context then holds it for holding replay and publication.
    with verified_investment_label_sources(root, amended, inputs.policy, paths=paths,
            source_implementation_snapshot=source_implementation_snapshot) as (extension, evidence, sources):
        add_pins(root, pins, sources["source_files"], paths=paths)
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
            reviewed_action_dates=parsed.reviewed_action_dates, reviewed_parent_units=parsed.reviewed_parent_units,
            completed_cash_manifest=parsed.completed_cash_distributions, completed_cash_facts=recovery_evidence.cash,
            existing_recipient_stock_identity=parsed.existing_recipient_stock_identity,
            reviewed_held_stock_actions_manifest=parsed.reviewed_held_stock_actions,
            reviewed_held_stock_actions=recovery_evidence.held_actions, paths=paths)
        reader = InvestmentSourceReader(root, raw_selection, request_sha256, paths=paths)
        prior_producer_sources = set(pins)
        producer_export, producer_commit = producer_origin(implementation, pins)
        producer_sources = implementation.keys() | (pins.keys() - prior_producer_sources)
        retained_external = inputs.retained_external | frozenset(
            name for name in producer_sources if not Path(name).is_relative_to(root))

        def recheck_inputs() -> None:
            reader.recheck()
            actions.recheck()
            add_pins(root, pins, actions.source_files, paths=paths)
            add_pins(root, pins, reader.source_files, paths=paths)
            recheck_sources(root, pins, paths=paths, retained_external=retained_external)

        if validate_inputs_only:
            recheck_inputs()
            return {
                "status": "inputs_verified", "data_root": str(paths.data_root),
                "fixed_holdings": len(parent.rows), "unknown_holdings": len(unknown),
                "source_pin_count": len(pins), "source_inventory_sha256": json_sha256(pins),
                "source_implementation_snapshot": (
                    None if source_implementation_snapshot is None else source_implementation_snapshot.model_dump(mode="json")),
                "holding_replay_started": False,
            }
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
        if source_implementation_snapshot is not None:
            report["source_implementation_snapshot"] = source_implementation_snapshot.model_dump(mode="json")
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
                first = datetime.fromisoformat(row["entry_time_utc"]).date()
                last = datetime.fromisoformat(row["economic_maturity_utc"]).date()
                supported = (actions.new_distribution_relevance_ids(row["security_id"], symbol, first, last,
                    new_cash_ids=recovery_evidence.added_cash_ids,
                    recipient_source_added=recovery_evidence.recipient_added) if recovery_evidence.incremental
                    else actions.reviewed_relevance_ids(symbol, first, last))
                if recovery_evidence.added_currency_ids:
                    supported = supported.union(actions.verified_currency_action_ids(
                        symbol, recovery_evidence.added_currency_ids))
                if recovery_evidence.added_held_action_ids:
                    supported = supported.union(actions.reviewed_held_stock_action_ids(row["security_id"], symbol, first, last)
                        & recovery_evidence.added_held_action_ids)
                eligible_ids = candidate_ids[key] & supported
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
                verify_remaining_trigger_evidence(root, request_pin, recovery_evidence,
                    pair["specifications"]["stock"], eligible_ids, pins, paths=paths)
                recovered = dict(row)
                publish_selected_pair(staged, output, key, recovered, pair, verified_at, proof_pins, paths=paths)
                rows[key] = recovered
                attempt["status"] = "recovered"

            comparisons = compare_survivor_cohorts(parent, rows)
            attempted = sum(row["status"] != "unattempted" for row in attempts.values())
            recovered_count = len(proof_pins)
            report.update(attempted_keys=attempted, unattempted_keys=len(unknown) - attempted,
                newly_recovered_keys=recovered_count, still_unresolved_keys=len(unknown) - recovered_count,
                coverage={"before": _coverage(parent.rows.values(), parent.comparisons),
                    "after": _coverage(rows.values(), comparisons)})
            if all(view["date_count"] and view["complete_dates"] == view["date_count"] for view in comparisons.values()):
                report["status"] = "complete_research_only"
            guard("remaining investment outcome publication")
            write_json_object(staged / "outcomes.json", {"rows": list(rows.values())})
            write_json_object(staged / "comparisons.json", {"views": comparisons})
            write_json_object(staged / "attempts.json", {"rows": list(attempts.values())})
            recheck_inputs()
            recheck_sources(staged, proof_pins)
            report["completed_at_utc"] = datetime.now(UTC).isoformat()

        result = _publish(output, report, pins, save, paths=paths, retained_external=retained_external)
        return {**result, **{name: report[name] for name in (
            "attempted_keys", "unattempted_keys", "newly_recovered_keys", "still_unresolved_keys", "coverage",
        )}}

