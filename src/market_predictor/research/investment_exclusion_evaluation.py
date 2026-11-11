"""Recompare retained selected stock/SPY outcomes after explicit exclusions."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from math import isclose, isfinite
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dataset import read_object, recheck_sources, require
from market_predictor.investment.training import HORIZONS
from market_predictor.research.investment_features import _publish
from market_predictor.research.investment_label_reuse import producer_origin
from market_predictor.research.investment_selection_outcomes import SelectedKey, _compare_cohorts
from market_predictor.research.investment_training import FLAGS
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier
from market_predictor.swing.contracts.holding_materialization import SourcePin

OUTCOME_STATUSES = ("reused_original", "recovered", "unresolved")
RETURN_FIELDS = ("stock_net_return", "spy_gross_return", "net_excess_vs_spy")
SCOPE = "post_observation_sensitivity"
SCHEMA = "market_predictor.investment_exclusion_evaluation"


class InvestmentExclusion(HoldingContract):
    ticker: Identifier
    security_id: Identifier
    reason: Identifier

    @model_validator(mode="after")
    def uppercase_ticker(self) -> Self:
        if self.ticker != self.ticker.upper():
            raise ValueError("exclusion tickers must use the original uppercase spelling")
        return self


class InvestmentExclusionEvaluationRequest(HoldingContract):
    schema_version: Literal["market_predictor.investment_exclusion_evaluation_request"]
    scope: Literal["post_observation_sensitivity"]
    parent_manifest: SourcePin
    exclusions: tuple[InvestmentExclusion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_exclusions(self) -> Self:
        if len({item.ticker for item in self.exclusions}) != len(self.exclusions) or len(
            {item.security_id for item in self.exclusions}
        ) != len(self.exclusions):
            raise ValueError("exclusion tickers and stable security identities must be unique")
        return self


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {str(package / name): file_sha256(package / name) for name in (
        "commands/investment_research.py", "research/investment_exclusion_evaluation.py",
        "research/investment_selection_outcomes.py", "research/investment_features.py",
        "research/investment_label_reuse.py", "investment/dataset.py", "evidence/hashing.py",
    )}


def _status_counts(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["status"] for row in rows)
    return {status: counts[status] for status in OUTCOME_STATUSES}


def _session(value: object) -> bool:
    return isinstance(value, str) and len(value) == 10 and date.fromisoformat(value).isoformat() == value


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _parent_rows(value: dict[str, Any]) -> dict[SelectedKey, dict[str, Any]]:
    rows = value.get("rows")
    require(isinstance(rows, list) and bool(rows), "parent selected outcomes must contain holding rows")
    assert isinstance(rows, list)
    outcomes: dict[SelectedKey, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict) and all(_text(row.get(name)) for name in ("decision_id", "security_id", "ticker"))
            and _session(row.get("decision_session")) and type(row.get("horizon_sessions")) is int
            and row["horizon_sessions"] in HORIZONS, "parent holding has an invalid decision, security, date or horizon")
        key = (row["decision_id"], row["horizon_sessions"])
        require(key not in outcomes, "parent selected outcomes duplicate a decision and horizon")
        status = row.get("status")
        require(status in OUTCOME_STATUSES and row.get("original_status") in ("source_validated", "unresolved")
            and (row.get("original_status") == "source_validated") == (status == "reused_original"),
            "parent holding has an invalid original or current status")
        require(all(name in row for name in (*RETURN_FIELDS, "outcome_available_at_utc")),
            "parent holding omits return values or availability")
        if status == "unresolved":
            require(all(row[name] is None for name in (*RETURN_FIELDS, "outcome_available_at_utc")),
                "unresolved parent holding has return values or availability")
        else:
            require(all(type(row[name]) in (int, float) and isfinite(row[name]) for name in RETURN_FIELDS)
                and isclose(row["stock_net_return"] - row["spy_gross_return"], row["net_excess_vs_spy"],
                            rel_tol=0, abs_tol=1e-12),
                "verified parent holding returns are invalid or do not reconcile to SPY")
            clock = row["outcome_available_at_utc"]
            require(isinstance(clock, str) and datetime.fromisoformat(clock).utcoffset() is not None,
                "verified parent holding has no aware outcome availability")
        outcomes[key] = row
    return outcomes


def _parent_cohorts(value: dict[str, Any], outcomes: dict[SelectedKey, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    views = value.get("views")
    require(isinstance(views, dict) and bool(views), "parent comparisons must contain views")
    assert isinstance(views, dict)
    cohorts: dict[str, list[dict[str, Any]]] = {}
    placed_keys: set[SelectedKey] = set()
    for view, comparison in views.items():
        parts = view.split("/")
        require(len(parts) == 4 and parts[0] in {str(horizon) for horizon in HORIZONS}
            and all(parts) and isinstance(comparison, dict) and isinstance(comparison.get("dates"), list),
            "parent comparison has an invalid view identity or date inventory")
        horizon = int(parts[0])
        cohorts[view] = []
        previous = ""
        for entry in comparison["dates"]:
            require(isinstance(entry, dict) and _session(entry.get("decision_session")),
                "parent comparison has an invalid cohort date")
            session = entry["decision_session"]
            decisions, securities = entry.get("decision_ids"), entry.get("security_ids")
            require(session > previous and isinstance(decisions, list) and isinstance(securities, list)
                and all(_text(item) for item in [*decisions, *securities])
                and len(decisions) == len(set(decisions)) == len(securities) == len(set(securities)),
                "parent cohort dates or ordered selections contain duplicates or mismatched counts")
            keys = [(decision, horizon) for decision in decisions]
            require(all(key in outcomes and outcomes[key]["security_id"] == security
                and outcomes[key]["decision_session"] == session for key, security in zip(keys, securities, strict=True)),
                "parent cohort placement does not bind the same decision, security, date and horizon")
            placed_keys.update(keys)
            cohorts[view].append({"decision_session": session, "security_ids": securities, "keys": keys})
            previous = session
    require(placed_keys == outcomes.keys(), "parent outcome rows and cohort placement keys differ")
    return cohorts


def _validate_parent_report(parent: dict[str, Any], outcomes: dict[SelectedKey, dict[str, Any]],
                            comparisons: dict[str, Any], cohorts: dict[str, list[dict[str, Any]]]) -> None:
    selection = {view: [{name: entry[name] for name in ("decision_session", "security_ids")} for entry in entries]
        for view, entries in cohorts.items()}
    require(json_sha256(selection) == parent.get("selection_sha256"), "parent ordered selection hash differs")
    require(json_sha256(_compare_cohorts(cohorts, outcomes)) == json_sha256(comparisons),
        "parent cohort counts or means differ from the original holdings")
    status = _status_counts(outcomes.values())
    expected = {
        "unique_selected_keys": len(outcomes), "reused_original_keys": status["reused_original"],
        "original_unknown_keys": status["recovered"] + status["unresolved"], "recovered_keys": status["recovered"],
        "unresolved_keys": status["unresolved"], "comparison_views": len(comparisons),
        "comparison_cohorts": sum(view["date_count"] for view in comparisons.values()),
        "selected_positions": sum(view["selected_positions"] for view in comparisons.values()),
    }
    require(all(type(parent.get(name)) is int and parent[name] == count for name, count in expected.items())
        and parent.get("status") == ("complete_with_explicit_gaps" if status["unresolved"] else "complete_research_only"),
        "parent manifest counts or completion status differ from the original holdings")


def _filter_cohorts(
    exclusions: tuple[InvestmentExclusion, ...], cohorts: dict[str, list[dict[str, Any]]],
    outcomes: dict[SelectedKey, dict[str, Any]], original: dict[str, Any],
) -> tuple[dict[SelectedKey, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    ids_by_ticker: dict[str, set[str]] = defaultdict(set)
    for row in outcomes.values():
        ids_by_ticker[row["ticker"]].add(row["security_id"])
    for item in exclusions:
        require(ids_by_ticker[item.ticker] == {item.security_id},
            f"exclusion ticker does not uniquely bind the supplied stable identity: {item.ticker}")
    exclusions_by_id = {item.security_id: item for item in exclusions}
    retained = {key: row for key, row in outcomes.items() if row["security_id"] not in exclusions_by_id}
    filtered: dict[str, list[dict[str, Any]]] = {}
    removed_by_date: dict[str, list[list[dict[str, Any]]]] = {}
    removed_by_security: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view, entries in cohorts.items():
        filtered[view], removed_by_date[view] = [], []
        for entry in entries:
            kept_keys = [key for key in entry["keys"] if key in retained]
            filtered[view].append({"decision_session": entry["decision_session"], "keys": kept_keys,
                "security_ids": [retained[key]["security_id"] for key in kept_keys]})
            removed = []
            for key in entry["keys"]:
                if key in retained:
                    continue
                row = outcomes[key]
                record = {name: row[name] for name in ("decision_id", "security_id", "ticker", "horizon_sessions", "status")}
                record["reason"] = exclusions_by_id[row["security_id"]].reason
                removed.append(record)
                removed_by_security[row["security_id"]].append(record)
            removed_by_date[view].append(removed)
    comparisons = _compare_cohorts(filtered, retained)
    for view, result in comparisons.items():
        before = original[view]
        for prior, after, removed in zip(before["dates"], result["dates"], removed_by_date[view], strict=True):
            after.update(original_selected_positions=prior["selected_positions"], original_complete=prior["complete"],
                removed_positions=removed)
        result.update(original_selected_positions=before["selected_positions"],
            removed_positions=before["selected_positions"] - result["selected_positions"],
            empty_dates=sum(not entry["selected_positions"] for entry in result["dates"]))
    removed_keys_by_security: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for key, row in outcomes.items():
        if key not in retained:
            removed_keys_by_security[row["security_id"]].append(row)
    removed_summary = [{**item.model_dump(mode="json"), "removed_keys": _status_counts(removed_keys_by_security[item.security_id]),
        "removed_positions": _status_counts(removed_by_security[item.security_id])} for item in exclusions]
    return retained, comparisons, removed_summary


def _coverage(rows: Iterable[dict[str, Any]], comparisons: dict[str, Any]) -> dict[str, Any]:
    keys = _status_counts(rows)
    positions = {status: sum(view[f"{column}_positions"] for view in comparisons.values())
        for status, column in zip(OUTCOME_STATUSES, ("reused", "recovered", "unresolved"), strict=True)}
    return {"unique_keys": sum(keys.values()), "selected_positions": sum(positions.values()), "key_status_counts": keys,
        "position_status_counts": positions, "complete_cohorts": sum(view["complete_dates"] for view in comparisons.values())}


def comparison_sidecar_pins(root: Path, manifest: SourcePin, value: dict[str, Any]) -> tuple[SourcePin, SourcePin]:
    parent_path = inside(root, manifest.path)
    files = value.get("files")
    require(parent_path.name == "_manifest.json" and isinstance(files, dict)
        and set(files) == {"outcomes.json", "comparisons.json"},
        "closed fixed-selection parent must have an exact manifest and named sidecar inventory")
    assert isinstance(files, dict)
    sidecars = []
    for name in ("outcomes.json", "comparisons.json"):
        child = parent_path.parent / name
        require(inside(root, child) == child, "parent selection sidecar must be the named sibling file")
        sidecars.append(SourcePin(path=str(child), sha256=files[name]))
    return sidecars[0], sidecars[1]


@dataclass(frozen=True)
class FixedSelectionOutcomes:
    manifest: dict[str, Any]
    outcomes_pin: SourcePin
    comparisons_pin: SourcePin
    outcomes: dict[SelectedKey, dict[str, Any]]
    comparisons: dict[str, Any]
    cohorts: dict[str, list[dict[str, Any]]]


def read_fixed_selection_outcomes(root: Path, pin: SourcePin, pins: dict[str, str]) -> FixedSelectionOutcomes:
    parent = read_object(root, pin, pins)
    require(parent.get("schema") == "market_predictor.investment_selection_outcomes"
        and parent.get("scope") == "retrospective_research"
        and parent.get("round_trip_stock_cost_bps") == 20 and parent.get("benchmark_basis") == "stock_net_minus_spy_gross"
        and all(parent.get(flag) is False for flag in (*FLAGS, "selection_changed", "models_refitted", "original_labels_changed")),
        "exclusions require a closed fixed-selection parent with the original stock/SPY cost policy")
    outcomes_pin, comparisons_pin = comparison_sidecar_pins(root, pin, parent)
    outcomes = _parent_rows(read_object(root, outcomes_pin, pins))
    original = read_object(root, comparisons_pin, pins)
    cohorts = _parent_cohorts(original, outcomes)
    _validate_parent_report(parent, outcomes, original["views"], cohorts)
    return FixedSelectionOutcomes(parent, outcomes_pin, comparisons_pin, outcomes, original["views"], cohorts)


def exclusion_coverage(
    outcomes: dict[SelectedKey, dict[str, Any]], original: dict[str, Any],
    retained: dict[SelectedKey, dict[str, Any]], comparisons: dict[str, Any],
) -> dict[str, Any]:
    before, after = _coverage(outcomes.values(), original), _coverage(retained.values(), comparisons)
    removed = {name: before[name] - after[name] for name in ("unique_keys", "selected_positions")}
    removed.update({name: {status: before[name][status] - after[name][status] for status in OUTCOME_STATUSES}
        for name in ("key_status_counts", "position_status_counts")})
    return {"original": before, "removed": removed, "retained": after}


def evaluate_investment_exclusions(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Publish a sensitivity evaluation derived entirely from the pinned parent outcomes."""
    root = root.resolve()
    output = inside(root, output)
    if output.exists():
        raise FileExistsError(output)
    runtime = heavy_job_runtime_dir()
    request_pin = SourcePin(path=str(inside(root, request)), sha256=request_sha256)
    with heavy_job_lease("evaluate-investment-exclusions", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("investment exclusion inputs")
        implementation = _implementation()
        pins = dict(implementation)
        parsed = InvestmentExclusionEvaluationRequest.model_validate_json(json.dumps(read_object(root, request_pin, pins)))
        fixed = read_fixed_selection_outcomes(root, parsed.parent_manifest, pins)
        parent = fixed.manifest
        retained, comparisons, removed = _filter_cohorts(parsed.exclusions, fixed.cohorts, fixed.outcomes, fixed.comparisons)
        coverage = exclusion_coverage(fixed.outcomes, fixed.comparisons, retained, comparisons)
        complete = all(view["date_count"] and view["complete_dates"] == view["date_count"] for view in comparisons.values())
        producer_export, producer_commit = producer_origin(implementation, pins)
        report = {
            "schema": SCHEMA, "scope": SCOPE, "status": "complete_research_only" if complete else "complete_with_explicit_gaps",
            "request": request_pin.model_dump(mode="json"), "parent_manifest": parsed.parent_manifest.model_dump(mode="json"),
            "parent_outcomes": fixed.outcomes_pin.model_dump(mode="json"),
            "parent_comparisons": fixed.comparisons_pin.model_dump(mode="json"),
            "parent_selection_sha256": parent["selection_sha256"], "exclusions": removed,
            "coverage": coverage,
            "comparison_views": len(comparisons), "comparison_cohorts": parent["comparison_cohorts"],
            "empty_cohorts": sum(view["empty_dates"] for view in comparisons.values()),
            "round_trip_stock_cost_bps": parent["round_trip_stock_cost_bps"], "benchmark_basis": parent["benchmark_basis"],
            "outcome_verification_basis": "pinned_parent_outcomes", "implementation_files": implementation, "source_files": pins,
            "producer_export": None if producer_export is None else producer_export.model_dump(mode="json"),
            "producer_commit": producer_commit, "reranked": False, "replacements_added": False, "models_refitted": False,
            "original_labels_changed": False, "training_ready": False, **FLAGS,
        }

        def save(staged: Path) -> None:
            write_json_object(staged / "outcomes.json", {"rows": list(retained.values())})
            write_json_object(staged / "comparisons.json", {"views": comparisons})
            recheck_sources(root, pins)
            report["completed_at_utc"] = datetime.now(UTC).isoformat()

        result = _publish(output, report, pins, save)
        return {**result, "coverage": report["coverage"], "empty_cohorts": report["empty_cohorts"]}

