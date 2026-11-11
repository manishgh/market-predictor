"""Resolve additional retained stock/SPY facts for an immutable exclusion comparison."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from functools import partial
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import file_sha256
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.completed_cash_distributions import (
    new_completed_cash_ids,
    read_completed_cash_facts,
)
from market_predictor.investment.dataset import add_pins, read_object, recheck_sources, require
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
    distribution_action_ids,
    potential_action_ids,
    read_remaining_outcome_parent,
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


def _reviewed_action_ids(root: Path, request: InvestmentRemainingOutcomeRequest, pins: dict[str, str]) -> frozenset[str]:
    return frozenset(record["action_id"]
        for pin in (request.reviewed_action_dates, request.reviewed_parent_units) if pin is not None
        for record in read_object(root, pin, pins)["records"])


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {**selection_implementation(), **{str(package / name): file_sha256(package / name) for name in (
        "commands/investment_research.py", "research/investment_remaining_outcomes.py",
        "research/investment_exclusion_evaluation.py", "research/investment_remaining_parent.py",
        "investment/action_relevance.py", "investment/completed_cash_distributions.py",
        "investment/sec_document_evidence.py",
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
        parent = read_remaining_outcome_parent(root, parsed.parent_manifest, pins)
        previous = parent.request
        if previous is not None:
            require(parsed.reviewed_action_dates == previous.reviewed_action_dates
                and parsed.reviewed_parent_units == previous.reviewed_parent_units
                and (previous.existing_recipient_stock_identity is None
                    or parsed.existing_recipient_stock_identity == previous.existing_recipient_stock_identity),
                "remaining recovery changed a carried reviewed date, unit or recipient authority")
        prior_cash = read_completed_cash_facts(root, None if previous is None else previous.completed_cash_distributions, pins)
        cash = read_completed_cash_facts(root, parsed.completed_cash_distributions, pins)
        added_cash = new_completed_cash_ids(prior_cash, cash)
        recipient_added = parsed.existing_recipient_stock_identity is not None and (
            previous is None or previous.existing_recipient_stock_identity is None)
        additional = previous is not None or bool(cash) or recipient_added
        require(not additional or bool(added_cash) or recipient_added,
            "remaining recovery adds no new completed cash or existing-recipient evidence")
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
        new_cash = {action_id: cash[action_id] for action_id in added_cash}
        candidate_ids = {key: (distribution_action_ids(row, new_cash, recipient_added) if additional else potential_action_ids(row))
            for key, row in unknown.items()}
        action_ids = _reviewed_action_ids(root, parsed, pins).union(cash, *candidate_ids.values(),
            *(potential_action_ids(unknown[key]) for key, ids in candidate_ids.items() if ids))
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
            reviewed_action_dates=parsed.reviewed_action_dates, reviewed_parent_units=parsed.reviewed_parent_units,
            completed_cash_manifest=parsed.completed_cash_distributions, completed_cash_facts=cash,
            existing_recipient_stock_identity=parsed.existing_recipient_stock_identity)
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
                first = datetime.fromisoformat(row["entry_time_utc"]).date()
                last = datetime.fromisoformat(row["economic_maturity_utc"]).date()
                supported = (actions.new_distribution_relevance_ids(row["security_id"], symbol, first, last,
                    new_cash_ids=added_cash, recipient_source_added=recipient_added) if additional
                    else actions.reviewed_relevance_ids(symbol, first, last))
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
                proof_ids = {part.removeprefix("action_id=")
                    for ref in pair["specifications"]["stock"]["entry_evidence"]
                    if ref["record_locator"].startswith("family=") and ";relevance=" in ref["record_locator"]
                    for part in ref["record_locator"].split(";") if part.startswith("action_id=")}
                require(eligible_ids.issubset(proof_ids), "recovered holding lacks its selected action relevance proof")
                recovered = dict(row)
                publish_selected_pair(staged, output, key, recovered, pair, verified_at, proof_pins)
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
            reader.recheck()
            actions.recheck()
            add_pins(root, pins, actions.source_files)
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

