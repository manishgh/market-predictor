"""Read pinned survivor ledgers without replaying settled holding economics."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.investment.action_relevance import RECIPIENT_PRE_DISTRIBUTION_RELEVANCE, SUPPORTED_REASONS
from market_predictor.investment.completed_cash_distributions import (
    COMPLETED_CASH_RELEVANCE,
    ReviewedCompletedCashDistribution,
    new_completed_cash_ids,
    read_completed_cash_facts,
)
from market_predictor.investment.contracts import InvestmentSpyPairRequest, InvestmentSpyPairResult
from market_predictor.investment.dataset import pin_source_file, read_object, require
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
from market_predictor.research.investment_selection_outcomes import SelectedKey, _compare_cohorts
from market_predictor.research.investment_training import FLAGS
from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin

SCHEMA = "market_predictor.investment_remaining_outcomes"
SIDECARS = ("outcomes.json", "comparisons.json", "attempts.json")
ATTEMPT_IDENTITY = ("decision_id", "horizon_sessions", "security_id", "decision_session")
ATTEMPT_FIELDS = {*ATTEMPT_IDENTITY, "status", "relevance_action_ids", "attempted_at_utc", "missing_reasons"}
RECOVERY_FIELDS = {
    "status", "stock_net_return", "spy_gross_return", "net_excess_vs_spy", "verified_at_utc",
    "outcome_available_at_utc", "pair_missing_reasons", "pair_components", "pair_proof",
}


class InvestmentRemainingOutcomeRequest(HoldingContract):
    schema_version: Literal["market_predictor.investment_remaining_outcome_request"]
    scope: Literal["post_observation_sensitivity"]
    parent_manifest: SourcePin
    reviewed_action_dates: SourcePin | None = None
    reviewed_parent_units: SourcePin | None = None
    completed_cash_distributions: SourcePin | None = None
    existing_recipient_stock_identity: SourcePin | None = None

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
    request: InvestmentRemainingOutcomeRequest | None = None


def read_exclusion_survivors(root: Path, pin: SourcePin, pins: dict[str, str]) -> SurvivorOutcomes:
    return _read_exclusion_survivors(root, pin, pins, read_object(root, pin, pins))


def _read_exclusion_survivors(
    root: Path, pin: SourcePin, pins: dict[str, str], manifest: dict[str, Any],
) -> SurvivorOutcomes:
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


def compare_survivor_cohorts(parent: SurvivorOutcomes, rows: dict[SelectedKey, dict[str, Any]]) -> dict[str, Any]:
    """Keep all original removal evidence and replay only economics and completeness."""
    replayed = _compare_cohorts(parent.cohorts, rows)
    return {view: {**parent.comparisons[view], **result,
        "dates": [{**old, **new} for old, new in zip(parent.comparisons[view]["dates"], result["dates"], strict=True)]}
        for view, result in replayed.items()}


@dataclass(frozen=True)
class _Ledger:
    directory: Path
    manifest: dict[str, Any]
    request_pin: SourcePin
    request: InvestmentRemainingOutcomeRequest
    verified_at: datetime
    completed_at: datetime


def _clock(value: Any) -> datetime:
    require(isinstance(value, str), "remaining ledger requires an aware verification or attempt clock")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DataReadinessError("remaining ledger requires an aware verification or attempt clock") from exc
    require(parsed.utcoffset() is not None, "remaining ledger requires an aware verification or attempt clock")
    return parsed


def _read_ledger(root: Path, path: Path, manifest: dict[str, Any], pins: dict[str, str]) -> _Ledger:
    require(manifest.get("schema") == SCHEMA and manifest.get("scope") == "post_observation_sensitivity"
        and manifest.get("round_trip_stock_cost_bps") == 20 and manifest.get("benchmark_basis") == "stock_net_minus_spy_gross"
        and all(manifest.get(flag) is False for flag in (*FLAGS, "selection_changed", "models_refitted",
            "original_labels_changed", "training_ready"))
        and all(manifest.get(flag, False) is False for flag in ("reranked", "replacements_added")),
        "remaining ledger schema, scope, stock/SPY cost or research-only flags differ")
    request_pin = SourcePin.model_validate(manifest.get("request"))
    request = InvestmentRemainingOutcomeRequest.model_validate_json(json.dumps(read_object(root, request_pin, pins)))
    require(all(manifest.get(name) == value
        for name, value in request.model_dump(mode="json", exclude={"schema_version", "scope"}).items()),
        "remaining ledger request bindings differ")
    verified_at = _clock(manifest.get("verified_at_utc"))
    completed_at = _clock(manifest.get("completed_at_utc"))
    require(verified_at <= completed_at <= datetime.now(UTC), "remaining ledger verification or completion clock differs")
    return _Ledger(path.parent, manifest, request_pin, request, verified_at, completed_at)


def _ledger_sidecars(
    root: Path, ledger: _Ledger, pins: dict[str, str],
) -> tuple[tuple[SourcePin, dict[str, Any]], tuple[SourcePin, dict[str, Any]], list[dict[str, Any]]]:
    files = ledger.manifest.get("files")
    require(isinstance(files, dict) and set(files) == set(SIDECARS)
        and {child.name for child in ledger.directory.iterdir() if child.is_file()} == {*SIDECARS, "_manifest.json"},
        "remaining ledger requires an exact named sidecar inventory")
    assert isinstance(files, dict)
    loaded = {}
    for name in SIDECARS:
        child = ledger.directory / name
        require(inside(root, child) == child, "remaining ledger sidecar must be the named sibling file")
        pin = SourcePin(path=str(child), sha256=files[name])
        loaded[name] = (pin, read_object(root, pin, pins))
    attempts = loaded["attempts.json"][1]
    require(set(attempts) == {"rows"} and isinstance(attempts["rows"], list),
        "remaining ledger attempt sidecar requires ordered rows")
    return loaded["outcomes.json"], loaded["comparisons.json"], attempts["rows"]


def _validate_link(root: Path, ledger: _Ledger, parent: SurvivorOutcomes) -> None:
    manifest, request = ledger.manifest, ledger.request
    parent_pins = {
        "parent_outcomes": parent.outcomes_pin,
        "parent_comparisons": parent.comparisons_pin,
        "ancestor_manifest": parent.ancestor_pin,
    }
    require(all(SourcePin.model_validate(manifest.get(name)) == expected for name, expected in parent_pins.items())
        and all(json_sha256([manifest.get(name)]) == json_sha256([parent.manifest[name]]) for name in (
            "parent_selection_sha256", "exclusions", "comparison_views", "comparison_cohorts", "empty_cohorts",
        )), "remaining ledger parent or ancestor bindings, exclusions or ordered selections differ")
    sources = manifest.get("source_files")
    require(isinstance(sources, dict) and all(sources.get(str(inside(root, pin.path))) == pin.sha256 for pin in (
        ledger.request_pin, request.parent_manifest, *parent_pins.values(),
    )), "remaining ledger source inventory lacks an exact request, parent or ancestor pin")
    assert isinstance(sources, dict)
    before = parent.request
    if request.existing_recipient_stock_identity is not None:
        recipient = request.existing_recipient_stock_identity
        require(sources.get(str(inside(root, recipient.path))) == recipient.sha256,
            "remaining ledger source inventory lacks the exact recipient authority pin")
    if request.completed_cash_distributions is not None:
        cash = request.completed_cash_distributions
        require(sources.get(str(inside(root, cash.path))) == cash.sha256,
            "remaining ledger source inventory lacks the exact completed cash authority pin")
    if before is None:
        return
    require(before.reviewed_action_dates == request.reviewed_action_dates
        and before.reviewed_parent_units == request.reviewed_parent_units
        and (before.existing_recipient_stock_identity is None
            or before.existing_recipient_stock_identity == request.existing_recipient_stock_identity)
        and (before.completed_cash_distributions is None or request.completed_cash_distributions is not None),
        "remaining ledger replaced or removed a carried action authority")
    require(_clock(parent.manifest["completed_at_utc"]) <= ledger.verified_at,
        "remaining ledger verification precedes parent completion")


def potential_action_ids(row: dict[str, Any]) -> frozenset[str]:
    return frozenset(parts[3] for reason in row.get("pair_missing_reasons", [])
        if len(parts := reason.split(":")) == 5 and parts[:2] == ["stock", "action"]
        and parts[2] in {"name_changes", "spin_offs"} and parts[4] in SUPPORTED_REASONS)


def distribution_action_ids(
    row: dict[str, Any], new_cash: dict[str, ReviewedCompletedCashDistribution], recipient_source_added: bool,
) -> frozenset[str]:
    selected = set()
    for reason in row.get("pair_missing_reasons", []):
        parts = reason.split(":")
        if len(parts) != 5 or parts[:2] != ["stock", "action"]:
            continue
        _, _, family, action_id, gap = parts
        if family == "cash_dividends" and gap == "ordinary_domestic_cash_entitlement_unproven":
            fact = new_cash.get(action_id)
            if (fact is not None and (row["security_id"], row["ticker"]) == (fact.security_id, fact.ticker)
                    and _clock(row["entry_time_utc"]).date() > fact.completion.upper_bound):
                selected.add(action_id)
        elif family == "spin_offs" and recipient_source_added and gap in SUPPORTED_REASONS:
            selected.add(action_id)
    return frozenset(selected)


@dataclass(frozen=True)
class _ActionAdditions:
    cash: dict[str, ReviewedCompletedCashDistribution]
    recipient: bool
    incremental: bool

    def candidates(self, row: dict[str, Any]) -> frozenset[str]:
        if self.incremental:
            return distribution_action_ids(row, self.cash, self.recipient)
        return potential_action_ids(row)


def _canonical_text_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item and item == item.strip() for item in value) and (
        value == sorted(set(value)))


def _validate_attempt(
    attempt: dict[str, Any], previous: dict[str, Any], ledger: _Ledger, additions: _ActionAdditions,
) -> None:
    require(isinstance(attempt, dict) and set(attempt) == ATTEMPT_FIELDS
        and json_sha256([attempt[name] for name in ATTEMPT_IDENTITY]) == json_sha256([previous[name] for name in ATTEMPT_IDENTITY]),
        "remaining ledger attempt order, decision, security, date or horizon differs")
    status = attempt["status"]
    require(status in ("unattempted", "unresolved", "recovered"), "remaining ledger attempt has an unknown status")
    ids, reasons, clock = (attempt[name] for name in ("relevance_action_ids", "missing_reasons", "attempted_at_utc"))
    if status == "unattempted":
        require(ids == reasons == [] and clock is None, "unattempted holding has diagnostics or an attempt clock")
        return
    require(_canonical_text_list(ids) and bool(ids) and additions.candidates(previous).issuperset(ids)
        and _canonical_text_list(reasons) and bool(reasons) == (status == "unresolved"),
        "remaining ledger attempt lacks recorded eligible action IDs or coherent missing reasons")
    require(ledger.verified_at <= _clock(clock) <= ledger.completed_at,
        "remaining ledger attempt is outside the verification window")


def _recovered_proof(
    root: Path, ledger: _Ledger, key: SelectedKey, row: dict[str, Any], previous: dict[str, Any],
    attempt: dict[str, Any], additions: _ActionAdditions, pins: dict[str, str],
) -> Path:
    decision, horizon = key
    require(Path(decision).name == decision and decision not in {"", ".", ".."},
        "remaining ledger decision ID escapes the pair directory")
    expected_path = ledger.directory / "pairs" / decision / f"{horizon}.json"
    pin = SourcePin.model_validate(row.get("pair_proof"))
    require(inside(root, pin.path) == inside(root, expected_path) == expected_path and expected_path.is_file(),
        "remaining ledger recovery proof must be the named sibling pair file")
    guard("remaining ledger pair proof", reclaim_unused_memory=False)
    proof = read_object(root, pin, pins)
    require(set(proof) == {"schema", "verified_at_utc", "request", "result", "specifications"}
        and proof.get("schema") == "market_predictor.investment_selected_spy_pair_evidence",
        "remaining ledger recovery proof schema differs")
    request = InvestmentSpyPairRequest.model_validate_json(json.dumps(proof["request"]))
    result = InvestmentSpyPairResult.model_validate_json(json.dumps(proof["result"]))
    policy_sha256 = request.policy.sha256()
    require(result.status == "available" and request.decision_id == result.decision_id == decision
        and request.policy.horizon_sessions == result.horizon_sessions == horizon
        and request.decision_session.isoformat() == row["decision_session"]
        and request.sha256() == result.request_sha256 and policy_sha256 == result.policy_sha256
        and request.holdings[0].security_id == row["security_id"]
        and request.decision_time_utc < result.entry_time_utc == _clock(row["entry_time_utc"])
        and result.horizon_end_utc == _clock(row["economic_maturity_utc"]),
        "remaining ledger recovery proof request, policy, identity or interval differs")
    require(result.holdings[0].net_return == row.get("stock_net_return")
        and result.holdings[1].gross_return == row.get("spy_gross_return")
        and result.net_excess_vs_spy == row.get("net_excess_vs_spy")
        and row.get("pair_missing_reasons") == [],
        "remaining ledger recovery proof returns or availability differs from the row")
    available = _clock(row.get("outcome_available_at_utc"))
    require(proof["verified_at_utc"] == row.get("verified_at_utc") == ledger.manifest["verified_at_utc"]
        and _clock(previous["verified_at_utc"]) <= ledger.verified_at
        and result.outcome_available_at_utc == available
        and ledger.verified_at <= available <= ledger.completed_at
        and all(value.label_available_at_utc is not None and ledger.verified_at <= value.label_available_at_utc <= available
            for value in result.holdings), "remaining ledger recovery proof verification or availability clock differs")
    _validate_specifications(root, ledger, proof["specifications"], request, result, row, previous, attempt, additions)
    return expected_path


def _validate_specifications(
    root: Path, ledger: _Ledger, specifications: Any, request: InvestmentSpyPairRequest, result: InvestmentSpyPairResult,
    row: dict[str, Any], previous: dict[str, Any], attempt: dict[str, Any], additions: _ActionAdditions,
) -> None:
    components = row.get("pair_components")
    require(isinstance(specifications, dict) and set(specifications) == {"stock", "spy"}
        and isinstance(components, list) and len(components) == 2,
        "remaining ledger recovery proof lacks stock and SPY specifications or components")
    assert isinstance(components, list)
    stock_symbol = next((item["symbol"] for item in previous.get("pair_components", [])
        if item["role"] == "stock"), previous["ticker"])
    require(request.holdings[0].symbol == stock_symbol, "remaining ledger recovery stock symbol differs")
    for binding, value, component in zip(request.holdings, result.holdings, components, strict=True):
        spec = specifications[binding.role]
        require(isinstance(spec, dict) and isinstance(component, dict)
            and binding.role == value.role == component.get("role")
            and binding.security_id == value.security_id == spec.get("security_id") == component.get("security_id")
            and binding.symbol == component.get("symbol")
            and binding.specification_sha256 == value.specification_sha256 == json_sha256(spec)
            and spec.get("decision_id") == request.decision_id and spec.get("sector") == request.sector
            and spec.get("research_contract_sha256") == result.policy_sha256
            and spec.get("policy") == "fixed_horizon" and spec.get("price_basis") == "raw_with_no_adjustment"
            and spec.get("cost_prepaid_fraction") == 0.002
            and isinstance(spec.get("events"), list)
            and all(isinstance(event, dict) and event.get("kind") in ("corporate_action", "payment", "cash_available")
                for event in spec["events"])
            and _clock(spec.get("initial_entry_timestamp")) == result.entry_time_utc
            and isinstance(spec.get("session_end_timestamps"), list)
            and len(spec["session_end_timestamps"]) == result.horizon_sessions
            and _clock(spec["session_end_timestamps"][-1]) == result.horizon_end_utc
            and component.get("missing_reasons") == []
            and component.get("sources") == [source.model_dump(mode="json") for source in binding.sources],
            "remaining ledger recovery holding specification, cost, interval or component bindings differ")
    stock, spy = specifications["stock"], specifications["spy"]
    require(stock["session_end_timestamps"] == spy["session_end_timestamps"],
        "remaining ledger recovery stock and SPY holding intervals differ")
    refs = stock.get("entry_evidence")
    require(isinstance(refs, list) and all(isinstance(ref, dict) and isinstance(ref.get("record_locator"), str) for ref in refs),
        "remaining ledger recovery lacks its stock entry evidence")
    stock_sources = {source.reference: source.artifact_sha256 for source in request.holdings[0].sources}
    eligible = set()
    for ref in refs:
        locator = ref["record_locator"]
        if not locator.startswith("family=") or ";relevance=" not in locator:
            continue
        fields = dict(part.split("=", 1) for part in locator.split(";") if "=" in part)
        action_id = fields.get("action_id")
        if action_id not in attempt["relevance_action_ids"]:
            continue
        family, relevance = fields.get("family"), fields.get("relevance")
        if additions.incremental:
            expected = ("cash_dividends", COMPLETED_CASH_RELEVANCE) if action_id in additions.cash else (
                "spin_offs", RECIPIENT_PRE_DISTRIBUTION_RELEVANCE)
            if (family, relevance) != expected:
                continue
        reference, digest = ref.get("reference"), ref.get("artifact_sha256")
        if (ref.get("interpretation_policy_sha256") == ledger.request_pin.sha256
                and isinstance(reference, str) and isinstance(digest, str) and stock_sources.get(reference) == digest
                and ledger.manifest["source_files"].get(str(inside(root, reference))) == digest):
            eligible.add(action_id)
    require(eligible.issuperset(attempt["relevance_action_ids"]),
        "remaining ledger recovery lacks the triggering action relevance proof")


def _apply_ledger(
    root: Path, ledger: _Ledger, parent: SurvivorOutcomes, additions: _ActionAdditions, pins: dict[str, str],
) -> SurvivorOutcomes:
    guard("remaining ledger sidecars", reclaim_unused_memory=False)
    (outcomes_pin, outcomes), (comparisons_pin, comparisons), attempts = _ledger_sidecars(root, ledger, pins)
    rows = outcomes.get("rows")
    require(set(outcomes) == {"rows"} and isinstance(rows, list) and len(rows) == len(parent.rows),
        "remaining ledger survivor row inventory differs")
    prior_unknowns = {key: row for key, row in parent.rows.items() if row["status"] == "unresolved"}
    require(len(attempts) == len(prior_unknowns), "remaining ledger must account for each prior unknown exactly once")
    by_key = dict(zip(prior_unknowns, attempts, strict=True))
    assert isinstance(rows, list)
    current: dict[SelectedKey, dict[str, Any]] = {}
    new_proofs: set[Path] = set()
    attempted = 0
    for (key, previous), row in zip(parent.rows.items(), rows, strict=True):
        current[key] = row
        if key in by_key:
            attempt = by_key[key]
            _validate_attempt(attempt, previous, ledger, additions)
            attempted += int(attempt["status"] != "unattempted")
            if attempt["status"] == "recovered":
                require(isinstance(row, dict) and RECOVERY_FIELDS <= row.keys() and row["status"] == "recovered"
                    and json_sha256({name: value for name, value in row.items() if name not in RECOVERY_FIELDS})
                    == json_sha256({name: value for name, value in previous.items() if name not in RECOVERY_FIELDS}),
                    "remaining ledger changed a recovered survivor's frozen key, order or original metadata")
                new_proofs.add(_recovered_proof(root, ledger, key, row, previous, attempt, additions, pins))
                continue
        require(json_sha256([row]) == json_sha256([previous]),
            "remaining ledger changed an existing survivor key, order, outcome, proof or clock")
    proof_dir = ledger.directory / "pairs"
    require((not proof_dir.exists() or proof_dir.is_dir() and inside(root, proof_dir) == proof_dir)
        and {path for path in proof_dir.rglob("*") if path.is_file()} == new_proofs,
        "remaining ledger pair proof inventory differs from its recovered rows")

    expected = compare_survivor_cohorts(parent, current)
    require(json_sha256(comparisons) == json_sha256({"views": expected}),
        "remaining ledger ordered cohort placements, counts or means differ")
    status = ("complete_research_only"
        if all(view["date_count"] and view["complete_dates"] == view["date_count"] for view in expected.values())
        else "complete_with_explicit_gaps")
    counts = {"attempted_keys": attempted, "unattempted_keys": len(prior_unknowns) - attempted,
        "newly_recovered_keys": len(new_proofs), "still_unresolved_keys": len(prior_unknowns) - len(new_proofs)}
    coverage = {"before": _coverage(parent.rows.values(), parent.comparisons), "after": _coverage(current.values(), expected)}
    require(all(type(ledger.manifest.get(name)) is int and ledger.manifest[name] == count for name, count in counts.items())
        and ledger.manifest.get("status") == status
        and json_sha256([ledger.manifest.get("coverage")]) == json_sha256([coverage]),
        "remaining ledger attempt counters, coverage or completion status differ")
    return SurvivorOutcomes(ledger.manifest, outcomes_pin, comparisons_pin, parent.ancestor_pin, parent.ancestor,
        parent.original_unknowns, current, comparisons["views"], parent.cohorts, ledger.request)


def read_remaining_outcome_parent(root: Path, pin: SourcePin, pins: dict[str, str]) -> SurvivorOutcomes:
    """Validate a finite pinned lineage from the original exclusion to the direct parent."""
    visited: set[Path] = set()
    chain: list[_Ledger] = []
    while True:
        path = inside(root, pin.path)
        require(path.name == "_manifest.json", "remaining ledger parent must name its manifest")
        require(path not in visited, "remaining ledger parent chain contains a cycle")
        visited.add(path)
        manifest = read_object(root, pin, pins)
        if manifest.get("schema") == EXCLUSION_SCHEMA:
            parent = _read_exclusion_survivors(root, pin, pins, manifest)
            break
        ledger = _read_ledger(root, path, manifest, pins)
        chain.append(ledger)
        pin = ledger.request.parent_manifest
    cash_facts: dict[str, ReviewedCompletedCashDistribution] = {}
    previous_cash_pin = None
    for ledger in reversed(chain):
        _validate_link(root, ledger, parent)
        cash_pin = ledger.request.completed_cash_distributions
        supplied = read_completed_cash_facts(root, cash_pin, pins) if cash_pin != previous_cash_pin else cash_facts
        cash_ids = new_completed_cash_ids(cash_facts, supplied)
        recipient_pin = ledger.request.existing_recipient_stock_identity
        recipient_added = recipient_pin is not None and (
            parent.request is None or parent.request.existing_recipient_stock_identity is None)
        if recipient_added and recipient_pin is not None:
            pin_source_file(root, recipient_pin, pins)
        additional = parent.request is not None or bool(supplied) or recipient_added
        require(not additional or bool(cash_ids) or recipient_added,
            "remaining ledger added no completed cash facts or existing-recipient authority")
        additions = _ActionAdditions({action_id: supplied[action_id] for action_id in cash_ids}, recipient_added, additional)
        parent = _apply_ledger(root, ledger, parent, additions, pins)
        cash_facts, previous_cash_pin = supplied, cash_pin
    return parent

