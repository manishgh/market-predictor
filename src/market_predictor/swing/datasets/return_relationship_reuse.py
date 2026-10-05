"""Exact bounded source and saved-value comparison for an explicit reuse authority.

The old population and its targets are preserved evidence. Only the four shared
relationship formulas are replayed; neither targets nor original 120 bases are
certified as freshly reconstructed by this owner.
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd
import pyarrow.dataset as pds

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy
from market_predictor.swing.contracts.return_relationship_reuse import RelationshipReusePolicy
from market_predictor.swing.datasets import preserved_relationship_abstentions
from market_predictor.swing.datasets.adjusted_history_bindings import (
    bind_adjusted_history_decisions,
    expected_bound_history_sessions,
    load_adjusted_history_bindings,
)
from market_predictor.swing.datasets.corrected_outcomes import (
    load_corrected_decision_partition,
    load_corrected_outcome_policy,
    verified_corrected_price_sources,
)
from market_predictor.swing.datasets.initial_fit_raw_share_plan import MEMBERSHIP_COLUMNS, _projection
from market_predictor.swing.datasets.preserved_relationship_abstentions import (
    pinned_original_observation_report,
    preserved_abstention_evidence,
    preserved_physical_prefix,
)
from market_predictor.swing.datasets.relationship_historical_evidence import (
    HistoricalFeatureEvidence,
    historical_bars,
    historical_month,
    inspect_historical_publication,
)
from market_predictor.swing.datasets.research_dataset import _verify_decision_metadata
from market_predictor.swing.datasets.return_relationship_integrity import check_files, current_implementation, guard, pins, read_object
from market_predictor.swing.datasets.return_relationship_parent import VerifiedRelationshipParent
from market_predictor.swing.datasets.return_relationship_rows import ADDITIONS, IDENTITY_COLUMNS, build_group, read_spy, read_stock
from market_predictor.swing.datasets.return_relationship_sources import RelationshipSourceContext, relationship_sources
from market_predictor.swing.features.panel import swing_model_feature_columns

SCHEMA = "market_predictor.relationship_reuse_equivalence"
EXPECTED_ROWS = 586305
PEER_REPORT_SHA256 = "f51418cae90bb8fce520d4b8b2aff97eb344c0702ad9e9deadaba943b2c323f2"
CONFIGURATION_REPORT_SHA256 = "97fe1198f6dc7ed0ebd7e9401fa7ee9215c5d84268fa8990a801761954620b10"
SOURCE_COLUMNS = ("security_id", "ticker", "session_date_et", "bar_start_utc", "bar_end_utc", "available_at_utc",
    "open", "close", "volume", "price_feed", "adjustment", "source", "timeframe", "availability_policy")
CLOSED = {"training_eligible": False, "promotion_eligible": False, "serving_eligible": False}


@dataclass(frozen=True)
class ReuseEvidence:
    parent: HistoricalFeatureEvidence
    relationships: HistoricalFeatureEvidence
    files: dict[str, str]
    historical_implementation_files: dict[str, str]


def inspect_reuse_evidence(root: Path, policy: RelationshipReusePolicy) -> ReuseEvidence:
    if (policy.peer_reuse_report.sha256 != PEER_REPORT_SHA256
            or policy.configuration_difference_report.sha256 != CONFIGURATION_REPORT_SHA256):
        raise DataReadinessError("reuse requires the exact separately reviewed peer and configuration reports")
    parent = inspect_historical_publication(root, policy.historical_parent_publication,
        policy.historical_parent_receipt, profile="technical_market")
    relationships = inspect_historical_publication(root, policy.historical_relationship_publication,
        policy.historical_relationship_receipt, profile="technical_relationships")
    request = relationships.request
    peer = read_object(inside(root, policy.peer_reuse_report.path), policy.peer_reuse_report.sha256)
    differences = read_object(inside(root, policy.configuration_difference_report.path), policy.configuration_difference_report.sha256)
    if (request["parent_publication"] != policy.historical_parent_publication.model_dump(mode="json")
            or request["parent_saved_row_verification"] != policy.historical_parent_receipt.model_dump(mode="json")
            or request["parent_request_sha256"] != parent.manifest["request_sha256"]
            or parent.manifest["rows"] != relationships.manifest["rows"] or parent.manifest["rows"] != EXPECTED_ROWS
            or peer.get("schema") != "market_predictor.matched_peer_reuse_verification"
            or peer.get("status") != "passed_peer_replay_only" or peer.get("peer_numerical_replayed") is not True
            or peer.get("manifest_sha256") != policy.historical_parent_publication.sha256
            or peer.get("rows") != EXPECTED_ROWS or set(peer.get("months", {})) != set(parent.manifest["months"])
            or any(peer.get(key) is not False for key in ("base_price_features_replayed", "raw_news_aggregates_replayed",
                "targets_replayed", "training_eligible", "promotion_eligible"))
            or differences.get("schema") != "market_predictor.saved_configuration_differences"
            or differences.get("request_sha256") != parent.manifest["request_sha256"]
            or any(differences.get(key) is not False
                for key in ("source_equivalence_proven", "rewritten_historical_pins", "training_eligible"))):
        raise DataReadinessError("reuse historical authorities do not describe the same preserved population")
    for pin in (policy.feature_config, policy.strategy_contract):
        entry = differences["configurations"].get(pin.path)
        if entry is None or entry.get("current_sha256") != pin.sha256:
            raise DataReadinessError("current policy differs from the explicitly reviewed configuration comparison")
    peer_pins = pins(root, peer["source_files"])
    for name, digest in parent.source_files.items():
        if name == policy.historical_parent_receipt.path:
            continue
        if peer_pins.get(name) != digest:
            raise DataReadinessError("peer replay does not bind every inherited monthly artifact")
    files = pins(root, parent.source_files, relationships.source_files,
        {pin.path: pin.sha256 for pin in (policy.peer_reuse_report, policy.configuration_difference_report,
            policy.feature_config, policy.strategy_contract)})
    historical: dict[str, str] = {}
    for label, value in (("parent-request", parent.request), ("relationship-request", request),
            ("parent-receipt", parent.receipt), ("relationship-receipt", relationships.receipt)):
        historical.update({f"{label}/{name}": digest for name, digest in value.get("source_files", {}).items()
            if name.startswith("src/market_predictor/")})
    check_files(root, files)
    return ReuseEvidence(parent, relationships, files, historical)


def compare_frames(expected: pd.DataFrame, actual: pd.DataFrame, columns: list[str], *, scope: str,
    check_dtype: bool = False,
) -> dict[str, Any]:
    """Return bounded differences; nulls, clocks, row identity and absolute values matter."""
    if not set(columns).issubset(expected) or not set(columns).issubset(actual):
        return {"scope": scope, "equal": False, "reason": "missing_columns", "columns": columns}
    left, right = expected.loc[:, columns].reset_index(drop=True), actual.loc[:, columns].reset_index(drop=True)
    if len(left) != len(right):
        return {"scope": scope, "equal": False, "reason": "row_count", "expected_rows": len(left), "actual_rows": len(right)}
    changed: dict[str, int] = {}
    examples: list[dict[str, Any]] = []
    for column in columns:
        a, b = left[column], right[column]
        if check_dtype and a.dtype != b.dtype:
            changed[column] = len(a)
            if len(examples) < 8:
                examples.append({"column": column, "expected_dtype": str(a.dtype), "actual_dtype": str(b.dtype)})
            continue
        # Capture dtypes may differ, but never coerce numeric strings or clocks.
        equal = a.eq(b).fillna(False) | (a.isna() & b.isna())
        bool_a = a.map(lambda value: isinstance(value, bool))
        bool_b = b.map(lambda value: isinstance(value, bool))
        equal &= bool_a.eq(bool_b)
        count = int((~equal).sum())
        if count:
            changed[column] = count
            for index in equal.index[~equal][:max(0, 8 - len(examples))]:
                examples.append({"row": int(index), "column": column, "expected": str(a.loc[index]), "actual": str(b.loc[index])})
    return {"scope": scope, "equal": not changed, "rows": len(left), "difference_counts": changed, "examples": examples}


def consumed_sessions(decisions: pd.DataFrame, sessions: tuple[date, ...], *, benchmark: bool) -> tuple[date, ...]:
    offsets = {day: index for index, day in enumerate(sessions)}
    positions: set[int] = set()
    for day in decisions.session_date_et:
        if day not in offsets:
            raise DataReadinessError("reuse decision is outside the frozen exchange calendar")
        end = offsets[day]
        positions.update(range(max(0, end - (60 if benchmark else 252)), end + 1))
    return tuple(sessions[index] for index in sorted(positions))


def compare_source_inputs(old: pd.DataFrame, current: pd.DataFrame, required: tuple[date, ...], *, scope: str) -> dict[str, Any]:
    projected = []
    for frame in (old, current):
        selected = frame.loc[frame.session_date_et.isin(required)]
        if selected.session_date_et.duplicated().any():
            raise DataReadinessError("source equivalence cannot collapse duplicate session observations")
        indexed = selected.set_index("session_date_et").reindex(required)
        indexed["present"] = indexed.index.isin(selected.session_date_et)
        projected.append(indexed.rename_axis("session_date_et").reset_index())
    result = compare_frames(projected[0], projected[1], [*SOURCE_COLUMNS, "present"], scope=scope)
    result["required_sessions_sha256"] = json_sha256([str(day) for day in required])
    result["required_session_count"] = len(required)
    result["ingestion_clocks_compared"] = False
    result["ingestion_clock_role"] = "separately_pinned_capture_provenance_not_consumed_historical_proxy_clock"
    return result


def _historical_spy(root: Path, evidence: ReuseEvidence, files: dict[str, str]) -> pd.DataFrame:
    request = evidence.relationships.request
    candidates = [(name, digest) for name, digest in request["source_files"].items()
        if name.endswith("/combined_daily/_manifest.json")]
    if len(candidates) != 1:
        raise DataReadinessError("historical SPY requires one pinned combined inventory")
    name, digest = candidates[0]
    manifest = read_object(inside(root, name), digest)
    files[name] = digest
    records = [record for record in manifest["artifacts"] if record["ticker"] == "SPY"]
    if (len(records) != 1 or manifest["request_sha256"] != request["source_basis"]["combined_request_sha256"]):
        raise DataReadinessError("historical SPY query ownership differs")
    return historical_bars(root, request, {"kind": "combined", "artifact": records[0], "security_id": "benchmark:SPY"}, files)


def _old_group(evidence: ReuseEvidence, decisions: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    identity = str(decisions.security_id.iloc[0])
    candidates = [(key, item) for key, item in evidence.relationships.request["stock_inventory"].items()
        if item["security_id"] == identity and (item["kind"] == "corrected" or decisions.parent_ticker.eq(item["source_group"]).all())]
    if len(candidates) != 1:
        raise DataReadinessError("current query window lacks one historical relationship source owner")
    return candidates[0]


def _historical_group(evidence: HistoricalFeatureEvidence, decisions: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    arrow: Any = pds
    paths = [str(inside(evidence.path.parent, month["profiles"][evidence.profile]["path"]))
        for month in evidence.manifest["months"].values()]
    frame: pd.DataFrame = arrow.dataset(paths, format="parquet").to_table(columns=columns,
        filter=arrow.field("security_id") == str(decisions.security_id.iloc[0]), use_threads=False).to_pandas()
    frame = frame.loc[frame.decision_id.isin(decisions.decision_id)]
    if frame.decision_id.duplicated().any() or set(frame.decision_id) != set(decisions.decision_id):
        raise DataReadinessError("historical source group decision identities differ")
    return frame.set_index("decision_id").loc[decisions.decision_id].reset_index()


def _preserved_mapping(root: Path, evidence: ReuseEvidence, policy: RelationshipReusePolicy,
    context: RelationshipSourceContext, decisions: pd.DataFrame, unit_id: str,
) -> dict[str, Any] | None:
    old_key, old_item = _old_group(evidence, decisions)
    fact = old_item["quarantine"]
    if fact is None:
        return None
    if (len(decisions) != old_item["rows"] or json_sha256(sorted(decisions.decision_id)) != old_item["decision_ids_sha256"]):
        raise DataReadinessError("preserved historical abstention must retain its complete original decision group")
    observations = [pin for pin in fact["reviewed_evidence"]
        if pin["sha256"] == preserved_relationship_abstentions.HISTORICAL_OBSERVATION_SHA256]
    if len(observations) != 1:
        raise DataReadinessError("preserved abstention lacks one exact original observation authority")
    pinned_original_observation_report(root, SourcePin.model_validate(observations[0]))
    record = context.bindings.source.records[unit_id]
    source_path = inside(context.bindings.source.directory, record["bars_path"])
    expected = expected_bound_history_sessions(context.bindings, unit_id, context.memberships)
    return {"historical_group_key": old_key, "historical_fact": fact,
        "historical_request": {"path": (evidence.relationships.path.parent / "_request.json").relative_to(root).as_posix(),
            "sha256": evidence.relationships.manifest["request_sha256"]},
        "historical_receipt": policy.historical_relationship_receipt.model_dump(mode="json"),
        "historical_publication": policy.historical_relationship_publication.model_dump(mode="json"),
        "observation": observations[0], "current_unit_id": unit_id,
        "current_source": {"path": source_path.relative_to(root).as_posix(), "sha256": record["bars_sha256"]},
        "security_id": old_item["security_id"], "rows": len(decisions),
        "decision_ids_sha256": json_sha256(sorted(decisions.decision_id)),
        "parent_window": dict(context.bindings.windows[unit_id]["parent"]),
        "query_window": dict(context.bindings.windows[unit_id]["query"]),
        "expected_sessions_sha256": json_sha256([str(day) for day in expected]),
        "scope": ("whole_history_unavailable_identity_unresolved" if fact["first_invalid_session"] is None
            else "unchanged_valid_prefix_only"),
        "source_issuer_verified": False}


def _publish_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DataReadinessError("reuse authority refuses to overwrite an existing result")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        write_json_object(temporary, report)
        if os.name == "nt":
            os.rename(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_relationship_reuse(*, root: Path, config: Path, expected_config_sha256: str, output: Path) -> dict[str, Any]:
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    if not output.is_relative_to(root / "data/reports") or output.exists():
        raise DataReadinessError("reuse comparison requires a fresh report destination")
    policy = RelationshipReusePolicy.model_validate(read_object(config, expected_config_sha256))
    feature = ResearchFeaturePolicy.model_validate(tomllib.loads(inside(root, policy.feature_config.path).read_text(encoding="utf-8")))
    check_files(root, {policy.feature_config.path: policy.feature_config.sha256})
    outcome = load_corrected_outcome_policy(root, Path(feature.outcome_source_config.path), feature.outcome_source_config.sha256)
    if feature.strategy_contract != policy.strategy_contract:
        raise DataReadinessError("reuse feature and strategy authorities differ")
    implementations = current_implementation(root)
    files = {config.relative_to(root).as_posix(): expected_config_sha256, **implementations}
    failures: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
    preserved: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    all_decisions: list[pd.DataFrame] = []
    with verified_corrected_price_sources(root, Path(feature.outcome_source_config.path),
            feature.outcome_source_config.sha256, outcome) as source:
        guard(90.0)
        evidence = inspect_reuse_evidence(root, policy)
        files = pins(root, files, evidence.files, source["source_files"])
        bindings = load_adjusted_history_bindings(root=root, plan_authority=feature.adjusted_plan_authority,
            archive_authority=feature.adjusted_archive_authority)
        files = pins(root, files, bindings.source_files)
        memberships = _projection(source["membership_path"], MEMBERSHIP_COLUMNS)
        spy_ids = [key for key, record in bindings.source.records.items() if record["role"] == "benchmark" and record["ticker"] == "SPY"]
        if len(spy_ids) != 1:
            raise DataReadinessError("reuse current source must contain one SPY query unit")
        first = evidence.parent.manifest["months"]["2019-07"]["profiles"]["technical_market"]
        names, clocks = tuple(first["model_columns"]), dict(first["availability_columns"])
        parent = VerifiedRelationshipParent(evidence.parent.path, evidence.parent.manifest, evidence.parent.request,
            files, evidence.historical_implementation_files, names, clocks, evidence.parent)
        expected_names = tuple(swing_model_feature_columns(
            contract=load_strategy_contract(inside(root, policy.strategy_contract.path)), catalyst=False))
        if len(names) != 120 or names != expected_names:
            raise DataReadinessError("reuse requires exactly 120 unchanged baseline features")
        facts = None
        if policy.predictor_failure_facts is not None:
            from market_predictor.swing.datasets.predictor_abstention_derivation import PredictorFailureFacts

            facts = PredictorFailureFacts.model_validate_json(json.dumps(read_object(inside(root, policy.predictor_failure_facts.path),
                policy.predictor_failure_facts.sha256)))
            if facts.feature_config != policy.feature_config or facts.decision_config != feature.outcome_source_config:
                raise DataReadinessError("reuse failure facts belong to another current query authority")
            files = pins(root, files, {policy.predictor_failure_facts.path: policy.predictor_failure_facts.sha256})
            fact_pins = [facts.observations, *(pin for fact in facts.failures for pin in (*fact.source_artifacts, *fact.reviewed_evidence))]
            files = pins(root, files, {pin.path: pin.sha256 for pin in fact_pins})
            check_files(root, {pin.path: pin.sha256 for pin in fact_pins})
        basis = {"adjustment": "all", "price_feed": "sip", "source": "alpaca",
            "adjusted_plan_authority": feature.adjusted_plan_authority.model_dump(mode="json"),
            "adjusted_archive_authority": feature.adjusted_archive_authority.model_dump(mode="json"),
            "stream_policy": "one_verified_query_unit_per_parent_identity_window_never_splice",
            "historical_availability": "market_interval_close_not_first_seen"}
        context = RelationshipSourceContext(feature, outcome, facts, bindings, memberships, spy_ids[0], {}, {},
            pins(root, source["source_files"], bindings.source_files), basis, groups, "", preserved)
        month_results: dict[str, dict[str, Any]] = {}
        for record in source["manifest"]["files"]:
            if record["first_session"] > policy.decision_end or record["last_session"] < policy.decision_start:
                continue
            guard(90.0)
            decisions = load_corrected_decision_partition(root, source, record, outcome)
            if decisions.empty:
                continue
            month = record["partition_month"]
            baseline = historical_month(evidence.parent, month)
            original = historical_month(evidence.relationships, month)
            _verify_decision_metadata(baseline, decisions)
            compared = compare_frames(baseline, original, list(baseline.columns.drop("feature_profile")),
                scope=f"inherited/{month}", check_dtype=True)
            if not compared["equal"]:
                failures.append(compared)
            if (not baseline.feature_profile.eq("technical_market").all()
                    or not original.feature_profile.eq("technical_relationships").all()
                    or list(original.columns) != [*baseline.columns, *ADDITIONS]):
                failures.append({"scope": month, "equal": False, "reason": "historical_profile_or_column_order"})
            month_results[month] = {"rows": len(baseline), "decision_ids_sha256": json_sha256(sorted(baseline.decision_id)),
                "inherited_columns_equal": compared["equal"]}
            all_decisions.append(bind_adjusted_history_decisions(decisions, bindings))
            del baseline, original
            release_process_memory()
        population = pd.concat(all_decisions, ignore_index=True)
        del all_decisions
        if (len(population) != EXPECTED_ROWS or population.decision_id.duplicated().any()
                or set(month_results) != set(evidence.parent.manifest["months"])):
            raise DataReadinessError("reuse comparison did not cover the complete historical population")
        fact_map = {fact.group_key: fact for fact in facts.failures} if facts else {}
        for (identity, unit_id), decisions in population.groupby(["security_id", "source_group"], sort=True):
            key = json_sha256([str(identity), str(unit_id)])
            fact = fact_map.get(key)
            groups[key] = {"security_id": str(identity), "source_group": str(unit_id), "rows": len(decisions),
                "decision_ids_sha256": json_sha256(sorted(decisions.decision_id)), "artifact": dict(bindings.source.records[str(unit_id)]),
                "quarantine": fact.model_dump(mode="json") if fact is not None else None}
            mapping = _preserved_mapping(root, evidence, policy, context, decisions, str(unit_id))
            groups[key]["preserved_abstention_sha256"] = json_sha256(mapping) if mapping is not None else None
            if mapping is not None:
                if facts is not None:
                    raise DataReadinessError("preserved research abstentions require no invented current failure facts")
                preserved[key] = mapping
                _, _, preserved_pins = preserved_abstention_evidence(root, mapping)
                files = pins(root, files, preserved_pins)
                context.source_files.update(preserved_pins)
        basis["preserved_research_abstentions_sha256"] = json_sha256(preserved)
        basis["source_equivalence_scope"] = "consumed_inputs_only_with_explicit_unchanged_historical_abstentions"
        sources = relationship_sources(policy.historical_parent_publication.sha256, context, groups)
        spy = read_spy(context)
        spy["security_id"] = "benchmark:SPY"
        old_spy = _historical_spy(root, evidence, files)
        sessions = tuple(day.date() for day in xcals.get_calendar("XNYS").sessions_in_range(policy.source_start, policy.source_end))
        old_seen: dict[str, list[str]] = {}
        for (identity, unit_id), decisions in population.groupby(["security_id", "source_group"], sort=True):
            guard(90.0)
            key = json_sha256([str(identity), str(unit_id)])
            item = groups[key]
            old_key, old_item = _old_group(evidence, decisions)
            old_seen.setdefault(old_key, []).extend(decisions.decision_id)
            old = historical_bars(root, evidence.relationships.request, old_item, files)
            for rule in outcome.decision_corrections:
                if rule.security_id == identity and old_item["kind"] == "corrected":
                    old.loc[old.session_date_et.between(rule.first_session, rule.last_session), "ticker"] = rule.ticker
            old_quarantine = old_item["quarantine"]
            try:
                if old_quarantine is not None:
                    original_fact, observation, _ = preserved_abstention_evidence(root, preserved[key])
                    old = preserved_physical_prefix(old, original_fact, observation, tuple(sorted(old.session_date_et)))
                current = read_stock(root, context, item)
                required = consumed_sessions(decisions, sessions, benchmark=False)
                consumed = required
                if old_quarantine is not None:
                    boundary = old_quarantine["first_invalid_session"]
                    consumed = () if boundary is None else tuple(day for day in required if day < date.fromisoformat(boundary))
                stock_check = compare_source_inputs(old, current, consumed, scope=f"stock/{key}")
                stock_check["excluded_quarantined_session_positions"] = len(required) - len(consumed)
                stock_check["new_issuer_identity_proof"] = False
                stock_check["whole_history_identity_unresolved"] = (
                    old_quarantine is not None and old_quarantine["first_invalid_session"] is None)
                spy_check = compare_source_inputs(old_spy, spy, consumed_sessions(decisions, sessions, benchmark=True), scope=f"spy/{key}")
                columns = list(dict.fromkeys((*IDENTITY_COLUMNS, *names, *clocks.values(), "feature_profile")))
                baseline = _historical_group(evidence.parent, decisions, columns)
                actual = _historical_group(evidence.relationships, decisions, [*IDENTITY_COLUMNS, *ADDITIONS])
                replay = build_group(root, parent, context, item, sources, spy, baseline=baseline)
                additions = compare_frames(actual, replay, [*IDENTITY_COLUMNS, *ADDITIONS], scope=f"additions/{key}")
                comparisons[key] = {"source_inputs": stock_check, "spy_inputs": spy_check, "additions": additions,
                    "historical_inventory_key": old_key, "current_window": dict(bindings.windows[str(unit_id)]["query"]),
                    "parent_window": dict(bindings.windows[str(unit_id)]["parent"])}
                failures.extend(result for result in (stock_check, spy_check, additions) if not result["equal"])
            except DataReadinessError as error:
                failures.append({"scope": key, "equal": False, "reason": "unavailable_current_source", "detail": str(error)})
            release_process_memory()
        if set(old_seen) != set(evidence.relationships.request["stock_inventory"]):
            raise DataReadinessError("reuse comparison omitted historical source groups")
        for key, identifiers in old_seen.items():
            original = evidence.relationships.request["stock_inventory"][key]
            if len(identifiers) != original["rows"] or json_sha256(sorted(identifiers)) != original["decision_ids_sha256"]:
                raise DataReadinessError("reuse old query ownership changed decision population")
        report = {"schema": SCHEMA, "status": "passed_exact_reuse" if not failures else "failed_differences",
            "config": {"path": config.relative_to(root).as_posix(), "sha256": expected_config_sha256},
            "policy": policy.model_dump(mode="json"), "source_files": files,
            "current_implementation_files": implementations, "historical_implementation_files": evidence.historical_implementation_files,
            "rows": len(population), "decision_ids_sha256": json_sha256(sorted(population.decision_id)),
            "cohort_sha256": evidence.parent.request["cohort_sha256"], "months": month_results, "stock_inventory": groups,
            "comparisons": comparisons, "differences": failures, "source_inputs_equal": not failures,
            "additions_equal": all(value["additions"]["equal"] for value in comparisons.values()) and len(comparisons) == len(groups),
            "inherited_columns_equal": all(value["inherited_columns_equal"] for value in month_results.values()),
            "population_equal": True, "targets_replayed": False, "baseline_price_features_replayed": False,
            "raw_news_aggregates_replayed": False, "peer_evidence": "separately_pinned_saved_base_peer_replay",
            "source_window_bounds": [policy.source_start, policy.source_end], "source_basis": basis,
            "source_equivalence_scope": "consumed_inputs_only_with_explicit_unchanged_historical_abstentions",
            "preserved_research_abstentions": preserved, "quarantined_source_inputs_admitted": False,
            "context_source_files": dict(context.source_files), "membership_path": source["membership_path"].relative_to(root).as_posix(),
            "spy_unit_id": spy_ids[0], "sources": sources.model_dump(mode="json"), **CLOSED}
        check_files(root, files)
    # Finish the independently owned source context and its rechecks first. A new,
    # sequential lease protects final rechecks/publication; there is no nested lease.
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("publish-relationship-reuse-authority", runtime_dir=runtime):
        guard(90.0)
        check_files(root, files)
        if current_implementation(root) != implementations:
            raise DataReadinessError("reuse implementation changed before publication")
        _publish_report(output, report)
    return report


def _verified_authority(root: Path, policy: ReturnRelationshipPublicationPolicy) -> tuple[dict[str, Any], ReuseEvidence]:
    pin = policy.reuse_equivalence_authority
    if pin is None:
        raise DataReadinessError("historical reuse requires an explicit equivalence authority")
    report = read_object(inside(root, pin.path), pin.sha256)
    reuse = RelationshipReusePolicy.model_validate(report["policy"])
    config = SourcePin.model_validate(report["config"])
    if RelationshipReusePolicy.model_validate(read_object(inside(root, config.path), config.sha256)) != reuse:
        raise DataReadinessError("reuse report differs from its immutable request configuration")
    if (report.get("schema") != SCHEMA or report.get("status") != "passed_exact_reuse"
            or report.get("differences") != [] or report.get("rows") != EXPECTED_ROWS
            or any(report.get(name) is not True for name in ("source_inputs_equal", "additions_equal",
                "inherited_columns_equal", "population_equal"))
            or any(report.get(name) is not False for name in (*CLOSED, "targets_replayed",
                "baseline_price_features_replayed", "raw_news_aggregates_replayed", "quarantined_source_inputs_admitted"))
            or policy.parent_publication != reuse.historical_parent_publication
            or policy.parent_saved_row_verification != reuse.historical_parent_receipt
            or policy.feature_config != reuse.feature_config or policy.strategy_contract != reuse.strategy_contract
            or policy.predictor_failure_facts != reuse.predictor_failure_facts
            or report.get("source_window_bounds") != [policy.source_start, policy.source_end]
            or report.get("source_equivalence_scope") != "consumed_inputs_only_with_explicit_unchanged_historical_abstentions"
            or report.get("current_implementation_files") != current_implementation(root)):
        raise DataReadinessError("reuse authority is failed, stale or belongs to another current policy")
    evidence = inspect_reuse_evidence(root, reuse)
    files = pins(root, report["source_files"])
    required = pins(root, evidence.files, report["current_implementation_files"], report["context_source_files"],
        {config.path: config.sha256})
    if any(files.get(name) != digest for name, digest in required.items()):
        raise DataReadinessError("reuse authority omitted an independently bound source")
    groups, comparisons, months = report["stock_inventory"], report["comparisons"], report["months"]
    preserved = report.get("preserved_research_abstentions")
    old_scopes = {key for key, item in evidence.relationships.request["stock_inventory"].items() if item["quarantine"] is not None}
    if (not isinstance(preserved, dict) or not set(preserved).issubset(groups)
            or len(preserved) != len(old_scopes)
            or {item["historical_group_key"] for item in preserved.values()} != old_scopes
            or report["source_basis"].get("preserved_research_abstentions_sha256") != json_sha256(preserved)):
        raise DataReadinessError("reuse authority omitted or added a preserved historical abstention")
    for key, mapping in preserved.items():
        _, _, scope_files = preserved_abstention_evidence(root, mapping)
        source_pin = SourcePin.model_validate(mapping["current_source"])
        scope_files = pins(root, scope_files, {source_pin.path: source_pin.sha256})
        if (groups[key].get("preserved_abstention_sha256") != json_sha256(mapping)
                or groups[key]["security_id"] != mapping["security_id"] or groups[key]["source_group"] != mapping["current_unit_id"]
                or groups[key]["rows"] != mapping["rows"] or groups[key]["decision_ids_sha256"] != mapping["decision_ids_sha256"]
                or any(files.get(name) != digest for name, digest in scope_files.items())):
            raise DataReadinessError("reuse authority changed preserved abstention ownership or source pins")
    if (not groups or set(groups) != set(comparisons) or set(months) != set(evidence.parent.manifest["months"])
            or sum(item["rows"] for item in groups.values()) != EXPECTED_ROWS
            or sum(item["rows"] for item in months.values()) != EXPECTED_ROWS):
        raise DataReadinessError("reuse authority has incomplete group or monthly comparisons")
    for key, comparison in comparisons.items():
        if (key != json_sha256([groups[key]["security_id"], groups[key]["source_group"]])
                or groups[key].get("preserved_abstention_sha256") != (json_sha256(preserved[key]) if key in preserved else None)
                or any(comparison[name].get("equal") is not True for name in ("source_inputs", "spy_inputs", "additions"))):
            raise DataReadinessError("reuse authority contains an unequal source or addition group")
    for month, record in months.items():
        old = evidence.parent.manifest["months"][month]
        if (record.get("inherited_columns_equal") is not True or record["rows"] != old["rows"]
                or record["decision_ids_sha256"] != old["decision_ids_sha256"]):
            raise DataReadinessError("reuse authority changed a preserved monthly population")
    check_files(root, files)
    return report, evidence


def verified_reuse_parent(root: Path, policy: ReturnRelationshipPublicationPolicy) -> VerifiedRelationshipParent:
    report, evidence = _verified_authority(root, policy)
    assert policy.reuse_equivalence_authority is not None
    first = evidence.parent.manifest["months"]["2019-07"]["profiles"]["technical_market"]
    names, clocks = tuple(first["model_columns"]), dict(first["availability_columns"])
    for record in evidence.parent.manifest["months"].values():
        child = record["profiles"]["technical_market"]
        if child["model_columns"] != list(names) or child["availability_columns"] != clocks:
            raise DataReadinessError("historical baseline feature contract changes between months")
    expected_names = tuple(swing_model_feature_columns(
        contract=load_strategy_contract(inside(root, policy.strategy_contract.path)), catalyst=False))
    if len(names) != 120 or names != expected_names:
        raise DataReadinessError("historical reuse requires the original 120 ordered features")
    live = pins(root, report["source_files"], {policy.reuse_equivalence_authority.path: policy.reuse_equivalence_authority.sha256})
    return VerifiedRelationshipParent(evidence.parent.path, evidence.parent.manifest, evidence.parent.request,
        live, evidence.historical_implementation_files, names, clocks, evidence.parent)


def verified_reuse_context(root: Path, policy: ReturnRelationshipPublicationPolicy,
    parent: VerifiedRelationshipParent,
) -> RelationshipSourceContext:
    report, _ = _verified_authority(root, policy)
    if parent.historical_evidence is None or parent.path != inside(root, policy.parent_publication.path):
        raise DataReadinessError("reuse current source context lacks the separately verified historical parent")
    feature = ResearchFeaturePolicy.model_validate(tomllib.loads(inside(root, policy.feature_config.path).read_text(encoding="utf-8")))
    outcome = load_corrected_outcome_policy(root, Path(feature.outcome_source_config.path), feature.outcome_source_config.sha256)
    bindings = load_adjusted_history_bindings(root=root, plan_authority=feature.adjusted_plan_authority,
        archive_authority=feature.adjusted_archive_authority)
    files = pins(root, report["context_source_files"])
    if any(files.get(name) != digest for name, digest in pins(root, bindings.source_files).items()):
        raise DataReadinessError("reuse current adjusted source differs from the compared authority")
    membership_path = inside(root, report["membership_path"])
    if membership_path.relative_to(root).as_posix() not in files:
        raise DataReadinessError("reuse membership authority is not source pinned")
    memberships = _projection(membership_path, MEMBERSHIP_COLUMNS)
    facts = None
    if policy.predictor_failure_facts is not None:
        from market_predictor.swing.datasets.predictor_abstention_derivation import PredictorFailureFacts

        facts = PredictorFailureFacts.model_validate_json(json.dumps(read_object(inside(root, policy.predictor_failure_facts.path),
            policy.predictor_failure_facts.sha256)))
    groups = report["stock_inventory"]
    for key, item in groups.items():
        unit_id = item["source_group"]
        record = bindings.source.records.get(unit_id)
        comparison = report["comparisons"][key]
        if (record is None or dict(record) != item["artifact"] or record["role"] != "stock"
                or record["security_id"] != item["security_id"]
                or dict(bindings.windows[unit_id]["query"]) != comparison["current_window"]
                or dict(bindings.windows[unit_id]["parent"]) != comparison["parent_window"]):
            raise DataReadinessError("reuse current query ownership differs from source comparison")
        preserved = report["preserved_research_abstentions"].get(key)
        if preserved is not None:
            expected_sessions = expected_bound_history_sessions(bindings, unit_id, memberships)
            if preserved["expected_sessions_sha256"] != json_sha256([str(day) for day in expected_sessions]):
                raise DataReadinessError("reuse preserved abstention membership history changed")
    spy = bindings.source.records.get(report["spy_unit_id"])
    if spy is None or spy["role"] != "benchmark" or spy["ticker"] != "SPY":
        raise DataReadinessError("reuse current SPY query is missing or substituted")
    context = RelationshipSourceContext(feature, outcome, facts, bindings, memberships, report["spy_unit_id"], {}, {},
        files, report["source_basis"], groups, report["decision_ids_sha256"], report["preserved_research_abstentions"])
    if relationship_sources(policy.parent_publication.sha256, context, groups).model_dump(mode="json") != report["sources"]:
        raise DataReadinessError("reuse current relationship source identity differs")
    check_files(root, files)
    return context
