"""Read pinned survivor ledgers without replaying settled holding economics."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import exchange_calendars as xcals
from pydantic import model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.path_integrity import verify_tree_containment
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.investment.action_relevance import RECIPIENT_PRE_DISTRIBUTION_RELEVANCE, SUPPORTED_REASONS
from market_predictor.investment.completed_cash_distributions import (
    COMPLETED_CASH_RELEVANCE,
    ReviewedCompletedCashDistribution,
    new_completed_cash_ids,
    read_completed_cash_facts,
)
from market_predictor.investment.contracts import InvestmentSpyPairRequest, InvestmentSpyPairResult
from market_predictor.investment.dataset import add_pins, pin_source_file, read_object, require
from market_predictor.investment.dividend_currencies import (
    MAX_CURRENCY_MANIFEST_BYTES,
    CurrencyArtifactPin,
    CurrencyCorroboration,
    StockCompositeIdentity,
    _decimal,
)
from market_predictor.investment.dividend_currencies import SCHEMA as CURRENCY_SCHEMA
from market_predictor.investment.dividend_currencies import (
    _read as read_currency_bytes,
)
from market_predictor.investment.reviewed_held_stock_actions import (
    HELD_STOCK_ACTION_LOCATOR,
    REVIEWED_SPLIT_UNITS,
    ReviewedHeldStockAction,
    ReviewedHoldingSplit,
    held_stock_action_covers,
    read_reviewed_held_stock_actions,
)
from market_predictor.investment.stock_identity import StockIdentityProof
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
    currency_corroborations: SourcePin | None = None
    reviewed_held_stock_actions: SourcePin | None = None

    @model_validator(mode="after")
    def parent_units_require_dates(self) -> InvestmentRemainingOutcomeRequest:
        if self.reviewed_parent_units is not None and self.reviewed_action_dates is None:
            raise ValueError("reviewed parent share units require their reviewed action dates")
        return self


@dataclass(frozen=True)
class FailedAttempt:
    attempts_pin: SourcePin
    attempted_at_utc: datetime
    missing_reasons: tuple[str, ...]


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
    latest_failed_attempts: dict[SelectedKey, FailedAttempt] = field(default_factory=dict)

    def latest_reasons(self, key: SelectedKey) -> tuple[str, ...]:
        attempt = self.latest_failed_attempts.get(key)
        if attempt is not None:
            return attempt.missing_reasons
        return tuple(self.rows[key].get("pair_missing_reasons", ()))


def read_exclusion_survivors(root: Path, pin: SourcePin, pins: dict[str, str], *, paths: EvidencePaths | None = None) -> SurvivorOutcomes:
    paths = paths or EvidencePaths(root)
    return _read_exclusion_survivors(root, pin, pins, read_object(root, pin, pins, paths=paths), paths=paths)


def _read_exclusion_survivors(
    root: Path, pin: SourcePin, pins: dict[str, str], manifest: dict[str, Any],
    *, paths: EvidencePaths,
) -> SurvivorOutcomes:
    require(manifest.get("schema") == EXCLUSION_SCHEMA and manifest.get("scope") == "post_observation_sensitivity"
        and manifest.get("status") in {"complete_with_explicit_gaps", "complete_research_only"}
        and manifest.get("round_trip_stock_cost_bps") == 20 and manifest.get("benchmark_basis") == "stock_net_minus_spy_gross"
        and manifest.get("outcome_verification_basis") == "pinned_parent_outcomes"
        and all(manifest.get(flag) is False for flag in (*FLAGS, "reranked", "replacements_added",
            "models_refitted", "original_labels_changed", "training_ready")),
        "remaining outcomes require the closed pinned exclusion comparison")
    exclusion = InvestmentExclusionEvaluationRequest.model_validate_json(json.dumps(
        read_object(root, SourcePin.model_validate(manifest["request"]), pins, paths=paths)))
    ancestor_pin = exclusion.parent_manifest
    require(ancestor_pin == SourcePin.model_validate(manifest["parent_manifest"]), "exclusion ancestor pin differs")
    ancestor = read_fixed_selection_outcomes(root, ancestor_pin, pins, paths=paths)
    require(ancestor.outcomes_pin == SourcePin.model_validate(manifest["parent_outcomes"])
        and ancestor.comparisons_pin == SourcePin.model_validate(manifest["parent_comparisons"])
        and manifest["parent_selection_sha256"] == ancestor.manifest["selection_sha256"],
        "exclusion ancestor outcomes, comparisons or ordered selections differ")
    retained, comparison, removed = _filter_cohorts(exclusion.exclusions, ancestor.cohorts, ancestor.outcomes, ancestor.comparisons)
    outcomes_pin, comparisons_pin = comparison_sidecar_pins(root, pin, manifest, paths=paths)
    actual_rows = read_object(root, outcomes_pin, pins, paths=paths)
    actual_comparisons = read_object(root, comparisons_pin, pins, paths=paths)
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


def _read_ledger(root: Path, path: Path, manifest: dict[str, Any], pins: dict[str, str], *, paths: EvidencePaths) -> _Ledger:
    require(manifest.get("schema") == SCHEMA and manifest.get("scope") == "post_observation_sensitivity"
        and manifest.get("round_trip_stock_cost_bps") == 20 and manifest.get("benchmark_basis") == "stock_net_minus_spy_gross"
        and all(manifest.get(flag) is False for flag in (*FLAGS, "selection_changed", "models_refitted",
            "original_labels_changed", "training_ready"))
        and all(manifest.get(flag, False) is False for flag in ("reranked", "replacements_added")),
        "remaining ledger schema, scope, stock/SPY cost or research-only flags differ")
    request_pin = SourcePin.model_validate(manifest.get("request"))
    request = InvestmentRemainingOutcomeRequest.model_validate_json(json.dumps(read_object(root, request_pin, pins, paths=paths)))
    require(all(manifest.get(name) == value
        for name, value in request.model_dump(mode="json", exclude={"schema_version", "scope"}).items()),
        "remaining ledger request bindings differ")
    if "source_implementation_snapshot" in manifest:
        snapshot = SourcePin.model_validate(manifest["source_implementation_snapshot"])
        sources = manifest.get("source_files")
        require(isinstance(sources, dict) and sources.get(str(paths.identity(snapshot.path))) == snapshot.sha256,
            "remaining ledger source inventory lacks the exact implementation snapshot pin")
        pin_source_file(root, snapshot, pins, paths=paths)
    verified_at = _clock(manifest.get("verified_at_utc"))
    completed_at = _clock(manifest.get("completed_at_utc"))
    require(verified_at <= completed_at <= datetime.now(UTC), "remaining ledger verification or completion clock differs")
    return _Ledger(path.parent, manifest, request_pin, request, verified_at, completed_at)


def _ledger_sidecars(
    root: Path, ledger: _Ledger, pins: dict[str, str],
    *, paths: EvidencePaths,
) -> tuple[tuple[SourcePin, dict[str, Any]], tuple[SourcePin, dict[str, Any]], tuple[SourcePin, list[dict[str, Any]]]]:
    files = ledger.manifest.get("files")
    require(isinstance(files, dict) and set(files) == set(SIDECARS)
        and {child.name for child in paths.directory(ledger.directory).iterdir() if child.is_file()} == {*SIDECARS, "_manifest.json"},
        "remaining ledger requires an exact named sidecar inventory")
    assert isinstance(files, dict)
    loaded = {}
    for name in SIDECARS:
        child = ledger.directory / name
        pin = SourcePin(path=str(child), sha256=files[name])
        loaded[name] = (pin, read_object(root, pin, pins, paths=paths))
    attempts_pin, attempts = loaded["attempts.json"]
    require(set(attempts) == {"rows"} and isinstance(attempts["rows"], list),
        "remaining ledger attempt sidecar requires ordered rows")
    return loaded["outcomes.json"], loaded["comparisons.json"], (attempts_pin, attempts["rows"])


def _validate_link(ledger: _Ledger, parent: SurvivorOutcomes, *, paths: EvidencePaths) -> None:
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
    require(isinstance(sources, dict) and all(sources.get(str(paths.identity(pin.path))) == pin.sha256 for pin in (
        ledger.request_pin, request.parent_manifest, *parent_pins.values(),
    )), "remaining ledger source inventory lacks an exact request, parent or ancestor pin")
    assert isinstance(sources, dict)
    for name in ("existing_recipient_stock_identity", "completed_cash_distributions",
                 "currency_corroborations", "reviewed_held_stock_actions"):
        pin = getattr(request, name)
        if pin is not None:
            require(sources.get(str(paths.identity(pin.path))) == pin.sha256,
                f"remaining ledger source inventory lacks the exact {name} authority pin")
    if parent.request is not None:
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
class _CurrencyAddition:
    assertion: CurrencyCorroboration
    owner: StockIdentityProof


@dataclass(frozen=True)
class RemainingEvidence:
    cash: dict[str, ReviewedCompletedCashDistribution]
    added_cash_ids: frozenset[str]
    recipient_added: bool
    incremental: bool
    currency_pin: SourcePin | None
    currency_additions: dict[str, _CurrencyAddition]
    held_pin: SourcePin | None
    held_actions: dict[str, ReviewedHeldStockAction]
    added_held_action_ids: frozenset[str]

    @property
    def added_currency_ids(self) -> frozenset[str]:
        return frozenset(self.currency_additions)

    def candidate_ids(self, parent: SurvivorOutcomes, key: SelectedKey) -> frozenset[str]:
        row = parent.rows[key]
        if not self.incremental:
            return potential_action_ids(row)
        # Historical distribution links used the frozen outcome row. New economic
        # evidence must address a blocker in the most recent attempted replay.
        cash = {action_id: self.cash[action_id] for action_id in self.added_cash_ids}
        candidates = set(distribution_action_ids(row, cash, self.recipient_added))
        if not self.currency_additions and not self.added_held_action_ids:
            return frozenset(candidates)
        first = _clock(row["entry_time_utc"]).date()
        last = _clock(row["economic_maturity_utc"]).date()
        for reason in parent.latest_reasons(key):
            parts = reason.split(":")
            if len(parts) != 5 or parts[:2] != ["stock", "action"]:
                continue
            _, _, family, action_id, gap = parts
            currency = self.currency_additions.get(action_id)
            if (currency is not None and family == "cash_dividends" and gap == "dividend_usd_currency_unproven"
                    and (row["security_id"], row["ticker"]) == (currency.owner.security_id, currency.assertion.symbol)
                    and first <= currency.assertion.ex_date <= last):
                candidates.add(action_id)
            if action_id not in self.added_held_action_ids:
                continue
            held = self.held_actions[action_id]
            expected = (("forward_splits", "corporate_terms_require_replay") if isinstance(held, ReviewedHoldingSplit)
                else ("cash_dividends", "ordinary_domestic_cash_entitlement_unproven"))
            if (family, gap) == expected and held_stock_action_covers(held, row["security_id"], row["ticker"], first, last):
                candidates.add(action_id)
        return frozenset(candidates)


def _read_currency_inventory(root: Path, pin: SourcePin, pins: dict[str, str], *, paths: EvidencePaths) -> list[dict[str, Any]]:
    guard("remaining ledger currency inventory", reclaim_unused_memory=False)
    value = parse_strict_json_object(read_currency_bytes(root, CurrencyArtifactPin(path=pin.path, sha256=pin.sha256),
        MAX_CURRENCY_MANIFEST_BYTES, paths=paths), label="remaining ledger currency manifest")
    add_pins(root, pins, {pin.path: pin.sha256}, paths=paths)
    records = value.get("corroborations")
    require(set(value) == {"schema", "corroborations"} and value["schema"] == CURRENCY_SCHEMA
        and isinstance(records, list) and 0 < len(records) <= 10_000
        and all(isinstance(record, dict) and isinstance(record.get("action_id"), str) and bool(record["action_id"])
            for record in records),
        "remaining ledger currency manifest shape differs")
    assert isinstance(records, list)
    require(len({record["action_id"] for record in records}) == len(records),
        "remaining ledger currency manifest repeats an action ID")
    return records


def _currency_source_pins(value: Any) -> Sequence[SourcePin]:
    if isinstance(value, list):
        return [pin for item in value for pin in _currency_source_pins(item)]
    if isinstance(value, dict):
        if set(value) == {"path", "sha256"}:
            return (SourcePin.model_validate(value),)
        return [pin for item in value.values() for pin in _currency_source_pins(item)]
    return ()


def _new_currency_additions(
    root: Path, ancestor: dict[str, Any], before: SourcePin | None, supplied: SourcePin | None, pins: dict[str, str],
    *, paths: EvidencePaths,
) -> dict[str, _CurrencyAddition]:
    require(before is None or supplied is not None, "remaining ledger removed carried currency corroborations")
    if supplied is None or supplied == before:
        return {}
    baseline = before or SourcePin.model_validate(ancestor.get("currency_corroborations"))
    previous = _read_currency_inventory(root, baseline, pins, paths=paths)
    current = _read_currency_inventory(root, supplied, pins, paths=paths)
    require(len(current) > len(previous) and json_sha256(current[:len(previous)]) == json_sha256(previous),
        "remaining ledger currency additions must preserve every exact prior assertion as an ordered prefix")
    declared: dict[str, str] = {}
    resolved: dict[str, str] = {}
    for record in current:
        for source in _currency_source_pins(record):
            path = resolved.get(source.path)
            if path is None:
                path = str(paths.identity(source.path))
                resolved[source.path] = path
            require(declared.setdefault(path, source.sha256) == source.sha256
                and pins.get(path, source.sha256) == source.sha256,
                "remaining ledger currency assertions declare conflicting source pins")
    additions: dict[str, _CurrencyAddition] = {}
    owners: dict[CurrencyArtifactPin, StockIdentityProof] = {}
    for record in current[len(previous):]:
        assertion = CurrencyCorroboration.model_validate_json(json.dumps(record))
        identity = assertion.identity
        require(isinstance(identity, StockCompositeIdentity) and assertion.method == "exact_tax_event"
            and assertion.special_cash_distribution is None and assertion.tax_resolved_value_difference is None,
            "remaining ledger currency additions require exact ordinary stock-class assertions")
        assert isinstance(identity, StockCompositeIdentity)
        owner = owners.get(identity.proof)
        if owner is None:
            pin = SourcePin(path=identity.proof.path, sha256=identity.proof.sha256)
            owner = StockIdentityProof.model_validate_json(json.dumps(read_object(root, pin, pins, paths=paths)))
            owners[identity.proof] = owner
        require(owner.ticker == assertion.symbol, "remaining ledger currency assertion differs from its exact stock class")
        additions[assertion.action_id] = _CurrencyAddition(assertion, owner)
    return additions


def prepare_remaining_evidence(
    root: Path, parent: SurvivorOutcomes, request: InvestmentRemainingOutcomeRequest, pins: dict[str, str],
    *, paths: EvidencePaths | None = None,
) -> RemainingEvidence:
    """Bind additive sources and preserve every authority active in the parent."""
    paths = paths or EvidencePaths(root)
    before = parent.request
    positive = request.currency_corroborations is not None or request.reviewed_held_stock_actions is not None
    if before is not None:
        require(before.reviewed_action_dates == request.reviewed_action_dates
            and before.reviewed_parent_units == request.reviewed_parent_units
            and (before.existing_recipient_stock_identity is None and not positive
                or before.existing_recipient_stock_identity == request.existing_recipient_stock_identity)
            and (before.completed_cash_distributions == request.completed_cash_distributions if positive
                else before.completed_cash_distributions is None or request.completed_cash_distributions is not None),
            "remaining ledger replaced or removed a carried action authority")
    prior_cash_pin = None if before is None else before.completed_cash_distributions
    prior_cash = read_completed_cash_facts(root, prior_cash_pin, pins, paths=paths)
    cash_pin = request.completed_cash_distributions
    cash = prior_cash if cash_pin == prior_cash_pin else read_completed_cash_facts(root, cash_pin, pins, paths=paths)
    cash_ids = new_completed_cash_ids(prior_cash, cash)
    recipient_pin = request.existing_recipient_stock_identity
    recipient_added = recipient_pin is not None and (before is None or before.existing_recipient_stock_identity is None)
    if recipient_pin is not None and recipient_added:
        pin_source_file(root, recipient_pin, pins, paths=paths)
    currency_pin = request.currency_corroborations
    currency_additions = _new_currency_additions(root, parent.ancestor,
        None if before is None else before.currency_corroborations, currency_pin, pins, paths=paths)
    prior_held_pin = None if before is None else before.reviewed_held_stock_actions
    prior_held = read_reviewed_held_stock_actions(root, prior_held_pin, pins, paths=paths)
    held_pin = request.reviewed_held_stock_actions
    held = prior_held if held_pin == prior_held_pin else read_reviewed_held_stock_actions(root, held_pin, pins, paths=paths)
    held_ids = frozenset(held.keys() - prior_held.keys())
    require(all(action_id in held and json_sha256(held[action_id].model_dump(mode="json")) == json_sha256(fact.model_dump(mode="json"))
        for action_id, fact in prior_held.items()) and (held_pin == prior_held_pin or bool(held_ids)),
        "remaining ledger held-stock actions must preserve every exact prior fact and source pin and add a new action")
    require(not (cash.keys() & held.keys()) and not (currency_additions.keys() & held.keys()),
        "remaining ledger gives one action conflicting cash or held-stock evidence roles")
    incremental = before is not None or bool(cash) or recipient_added or positive
    require(not incremental or bool(cash_ids) or recipient_added or bool(currency_additions) or bool(held_ids),
        "remaining ledger added no completed cash, existing-recipient, currency or held-stock evidence")
    return RemainingEvidence(cash, cash_ids, recipient_added, incremental, currency_pin, currency_additions,
        held_pin, held, held_ids)


def _canonical_text_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) and item and item == item.strip() for item in value) and (
        value == sorted(set(value)))


def _validate_attempt(
    attempt: dict[str, Any], parent: SurvivorOutcomes, key: SelectedKey, ledger: _Ledger, additions: RemainingEvidence,
) -> None:
    previous = parent.rows[key]
    require(isinstance(attempt, dict) and set(attempt) == ATTEMPT_FIELDS
        and json_sha256([attempt[name] for name in ATTEMPT_IDENTITY]) == json_sha256([previous[name] for name in ATTEMPT_IDENTITY]),
        "remaining ledger attempt order, decision, security, date or horizon differs")
    status = attempt["status"]
    require(status in ("unattempted", "unresolved", "recovered"), "remaining ledger attempt has an unknown status")
    ids, reasons, clock = (attempt[name] for name in ("relevance_action_ids", "missing_reasons", "attempted_at_utc"))
    if status == "unattempted":
        require(ids == reasons == [] and clock is None, "unattempted holding has diagnostics or an attempt clock")
        return
    require(_canonical_text_list(ids) and bool(ids) and additions.candidate_ids(parent, key).issuperset(ids)
        and _canonical_text_list(reasons) and bool(reasons) == (status == "unresolved"),
        "remaining ledger attempt lacks recorded eligible action IDs or coherent missing reasons")
    require(ledger.verified_at <= _clock(clock) <= ledger.completed_at,
        "remaining ledger attempt is outside the verification window")


def _recovered_proof(
    root: Path, ledger: _Ledger, key: SelectedKey, row: dict[str, Any], previous: dict[str, Any],
    attempt: dict[str, Any], additions: RemainingEvidence, pins: dict[str, str],
    *, paths: EvidencePaths,
) -> Path:
    decision, horizon = key
    require(Path(decision).name == decision and decision not in {"", ".", ".."},
        "remaining ledger decision ID escapes the pair directory")
    expected_path = ledger.directory / "pairs" / decision / f"{horizon}.json"
    pin = SourcePin.model_validate(row.get("pair_proof"))
    require(paths.identity(pin.path) == expected_path,
        "remaining ledger recovery proof must be the named sibling pair file")
    guard("remaining ledger pair proof", reclaim_unused_memory=False)
    proof = read_object(root, pin, pins, paths=paths)
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
    _validate_specifications(root, ledger, proof["specifications"], request, result, row, previous, attempt, additions, paths=paths)
    return expected_path


def _validate_specifications(
    root: Path, ledger: _Ledger, specifications: Any, request: InvestmentSpyPairRequest, result: InvestmentSpyPairResult,
    row: dict[str, Any], previous: dict[str, Any], attempt: dict[str, Any], additions: RemainingEvidence,
    *, paths: EvidencePaths,
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
    stock_sources = {source.reference: (source.artifact_sha256, source.interpretation_policy_sha256)
        for source in request.holdings[0].sources}
    verify_remaining_trigger_evidence(root, ledger.request_pin, additions, stock,
        attempt["relevance_action_ids"], ledger.manifest["source_files"], stock_sources=stock_sources,
        verified_at=ledger.verified_at, outcome_available_at=result.outcome_available_at_utc, paths=paths)


@dataclass(frozen=True)
class _PositiveTrigger:
    pin: SourcePin
    locator: str
    treatment: Literal["replace", "distribution"]
    security_id: str
    ex_date: date
    units: float
    allow_ex_open_entry: bool = False


def _positive_event_binding(evidence: RemainingEvidence, action_id: str) -> _PositiveTrigger | None:
    currency = evidence.currency_additions.get(action_id)
    if currency is not None:
        assert evidence.currency_pin is not None
        return _PositiveTrigger(evidence.currency_pin,
            f"currency corroboration;action_id={action_id};original_record_sha256={currency.assertion.original_record_sha256}",
            "distribution", currency.owner.security_id, currency.assertion.ex_date,
            float(_decimal(currency.assertion.tax.gross_amount)), allow_ex_open_entry=True)
    if action_id not in evidence.added_held_action_ids:
        return None
    fact = evidence.held_actions[action_id]
    assert evidence.held_pin is not None
    treatment: Literal["replace", "distribution"]
    if isinstance(fact, ReviewedHoldingSplit):
        treatment, units = "replace", REVIEWED_SPLIT_UNITS
    else:
        treatment, units = "distribution", float(_decimal(fact.gross_amount))
    return _PositiveTrigger(evidence.held_pin, f"{HELD_STOCK_ACTION_LOCATOR};kind={fact.kind};action_id={action_id};"
        f"security_id={fact.security_id};ticker={fact.ticker};retained_record_sha256={fact.retained_record_sha256}",
        treatment, fact.security_id, fact.ex_date, units)


def verify_remaining_trigger_evidence(
    root: Path, request_pin: SourcePin, evidence: RemainingEvidence, stock_specification: Mapping[str, Any],
    eligible_ids: Sequence[str] | frozenset[str], source_files: Mapping[str, str], *,
    stock_sources: Mapping[str, tuple[str, str]] | None = None,
    verified_at: datetime | None = None, outcome_available_at: datetime | None = None,
    paths: EvidencePaths | None = None,
) -> None:
    """Require each triggering fact in the stock entry or its matching economic event."""
    paths = paths or EvidencePaths(root)

    def bound(ref: Any, supplying_pin: SourcePin | None = None) -> bool:
        if not isinstance(ref, dict) or ref.get("interpretation_policy_sha256") != request_pin.sha256:
            return False
        reference, digest = ref.get("reference"), ref.get("artifact_sha256")
        if not isinstance(reference, str) or not isinstance(digest, str):
            return False
        path = paths.identity(reference)
        if source_files.get(str(path)) != digest or (
                stock_sources is not None and stock_sources.get(reference) != (digest, request_pin.sha256)):
            return False
        if supplying_pin is None:
            return True
        if path != paths.identity(supplying_pin.path) or digest != supplying_pin.sha256:
            return False
        available = _clock(ref.get("available_at"))
        retrieved = _clock(ref.get("retrieved_at"))
        return (available <= retrieved and (verified_at is None or verified_at <= available)
            and (outcome_available_at is None or retrieved <= outcome_available_at))

    def proves_binding(values: Any, trigger: _PositiveTrigger) -> bool:
        return isinstance(values, list) and any(isinstance(ref, dict) and ref.get("record_locator") == trigger.locator
            and bound(ref, trigger.pin) for ref in values)

    refs = stock_specification.get("entry_evidence")
    require(isinstance(refs, list) and all(isinstance(ref, dict) and isinstance(ref.get("record_locator"), str) for ref in refs),
        "remaining ledger recovery lacks its stock entry evidence")
    assert isinstance(refs, list)
    negative: dict[str, tuple[str | None, str | None]] = {}
    for ref in refs:
        locator = ref["record_locator"]
        if locator.startswith("family=") and ";relevance=" in locator:
            fields = dict(part.split("=", 1) for part in locator.split(";") if "=" in part)
            action_id = fields.get("action_id")
            if action_id is not None and action_id in eligible_ids and bound(ref):
                negative[action_id] = (fields.get("family"), fields.get("relevance"))
    events = stock_specification.get("events")
    require(isinstance(events, list), "remaining ledger recovery lacks its stock event inventory")
    assert isinstance(events, list)
    for action_id in eligible_ids:
        positive = _positive_event_binding(evidence, action_id)
        if positive is None:
            expected = ("cash_dividends", COMPLETED_CASH_RELEVANCE) if action_id in evidence.added_cash_ids else (
                "spin_offs", RECIPIENT_PRE_DISTRIBUTION_RELEVANCE)
            require(action_id in negative and (not evidence.incremental or negative[action_id] == expected),
                "remaining ledger recovery lacks the triggering action relevance proof")
            continue
        require(stock_specification.get("security_id") == positive.security_id and proves_binding(refs, positive),
            "remaining ledger recovery lacks the exact positive manifest proof on the stock entry")
        calendar = xcals.get_calendar("XNYS")
        session_open = calendar.session_open(positive.ex_date.isoformat()).to_pydatetime()
        entry = _clock(stock_specification.get("initial_entry_timestamp"))
        event_id = f"{stock_specification.get('initial_position_id')}:action:{action_id}"
        matches = [event for event in events if isinstance(event, dict) and event.get("event_id") == event_id]
        if positive.allow_ex_open_entry and entry == session_open:
            require(not matches, "remaining ledger ex-open entry cannot claim the same distribution")
            continue
        require(entry < session_open and len(matches) == 1,
            "remaining ledger recovery lacks exactly one triggering positive stock event")
        event = matches[0]
        legs = event.get("legs")
        split = positive.treatment == "replace"
        position_id = event_id + (":shares" if split else ":unpaid")
        require(event.get("kind") == "corporate_action" and event.get("treatment") == positive.treatment
            and event.get("owned_security_id") == positive.security_id and _clock(event.get("effective_at")) == session_open
            and isinstance(legs, list) and len(legs) == 1 and isinstance(legs[0], dict)
            and legs[0].get("kind") == ("tradable_shares" if split else "unpaid_proceeds")
            and legs[0].get("security_id") == positive.security_id and legs[0].get("position_id") == position_id
            and legs[0].get("currency") == "USD" and legs[0].get("face_value_per_unit") == (None if split else 1.0)
            and legs[0].get("units_per_owned_unit") == positive.units and proves_binding(event.get("evidence"), positive),
            "remaining ledger recovery lacks the exact positive manifest proof and terms on the triggering stock event")
        if not split:
            marks = stock_specification.get("marks")
            require(isinstance(marks, list), "remaining ledger recovery lacks the stock mark inventory")
            assert isinstance(marks, list)
            claim_marks = [mark for mark in marks if isinstance(mark, dict) and mark.get("position_id") == position_id]
            require(bool(claim_marks) and all(mark.get("status") == "known" and mark.get("currency") == "USD"
                and mark.get("value_per_unit") == 1.0 and proves_binding(mark.get("evidence"), positive) for mark in claim_marks),
                "remaining ledger recovery lacks the positive manifest proof on its unpaid claim marks")


def _apply_ledger(
    root: Path, ledger: _Ledger, parent: SurvivorOutcomes, additions: RemainingEvidence, pins: dict[str, str],
    *, paths: EvidencePaths,
) -> SurvivorOutcomes:
    guard("remaining ledger sidecars", reclaim_unused_memory=False)
    (outcomes_pin, outcomes), (comparisons_pin, comparisons), (attempts_pin, attempts) = _ledger_sidecars(root, ledger, pins, paths=paths)
    rows = outcomes.get("rows")
    require(set(outcomes) == {"rows"} and isinstance(rows, list) and len(rows) == len(parent.rows),
        "remaining ledger survivor row inventory differs")
    prior_unknowns = {key: row for key, row in parent.rows.items() if row["status"] == "unresolved"}
    require(len(attempts) == len(prior_unknowns), "remaining ledger must account for each prior unknown exactly once")
    by_key = dict(zip(prior_unknowns, attempts, strict=True))
    assert isinstance(rows, list)
    current: dict[SelectedKey, dict[str, Any]] = {}
    new_proofs: set[Path] = set()
    latest_failed = dict(parent.latest_failed_attempts)
    attempted = 0
    for (key, previous), row in zip(parent.rows.items(), rows, strict=True):
        current[key] = row
        if key in by_key:
            attempt = by_key[key]
            _validate_attempt(attempt, parent, key, ledger, additions)
            attempted += int(attempt["status"] != "unattempted")
            if attempt["status"] == "unresolved":
                latest_failed[key] = FailedAttempt(attempts_pin, _clock(attempt["attempted_at_utc"]), tuple(attempt["missing_reasons"]))
            if attempt["status"] == "recovered":
                latest_failed.pop(key, None)
                require(isinstance(row, dict) and RECOVERY_FIELDS <= row.keys() and row["status"] == "recovered"
                    and json_sha256({name: value for name, value in row.items() if name not in RECOVERY_FIELDS})
                    == json_sha256({name: value for name, value in previous.items() if name not in RECOVERY_FIELDS}),
                    "remaining ledger changed a recovered survivor's frozen key, order or original metadata")
                new_proofs.add(_recovered_proof(root, ledger, key, row, previous, attempt, additions, pins, paths=paths))
                continue
        require(json_sha256([row]) == json_sha256([previous]),
            "remaining ledger changed an existing survivor key, order, outcome, proof or clock")
    proof_dir = paths.write(ledger.directory / "pairs")
    if proof_dir.exists():
        verify_tree_containment(proof_dir, label="remaining ledger pair proof directory")
    require({paths.identity(path) for path in proof_dir.rglob("*") if path.is_file()} == new_proofs,
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
        parent.original_unknowns, current, comparisons["views"], parent.cohorts, ledger.request, latest_failed)


def read_remaining_outcome_parent(
    root: Path, pin: SourcePin, pins: dict[str, str], *, paths: EvidencePaths | None = None,
) -> SurvivorOutcomes:
    """Validate a finite pinned lineage from the original exclusion to the direct parent."""
    paths = paths or EvidencePaths(root)
    visited: set[Path] = set()
    chain: list[_Ledger] = []
    while True:
        path = paths.identity(pin.path)
        require(path.name == "_manifest.json", "remaining ledger parent must name its manifest")
        require(path not in visited, "remaining ledger parent chain contains a cycle")
        visited.add(path)
        manifest = read_object(root, pin, pins, paths=paths)
        if manifest.get("schema") == EXCLUSION_SCHEMA:
            parent = _read_exclusion_survivors(root, pin, pins, manifest, paths=paths)
            break
        ledger = _read_ledger(root, path, manifest, pins, paths=paths)
        chain.append(ledger)
        pin = ledger.request.parent_manifest
    for ledger in reversed(chain):
        _validate_link(ledger, parent, paths=paths)
        additions = prepare_remaining_evidence(root, parent, ledger.request, pins, paths=paths)
        parent = _apply_ledger(root, ledger, parent, additions, pins, paths=paths)
    return parent

