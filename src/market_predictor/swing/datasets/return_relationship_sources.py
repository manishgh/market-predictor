"""Complete inventory binding for the saved adjusted histories and reviewed absences."""
from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy
from market_predictor.swing.datasets.adjusted_history_bindings import (
    AdjustedHistoryBindings,
    bind_adjusted_history_decisions,
    load_adjusted_history_bindings,
)
from market_predictor.swing.datasets.initial_fit_raw_share_plan import MEMBERSHIP_COLUMNS, _projection
from market_predictor.swing.datasets.predictor_abstention_derivation import PredictorFailureFacts, _observation
from market_predictor.swing.datasets.return_relationship_integrity import check_files, pins, read_object
from market_predictor.swing.datasets.return_relationship_parent import VerifiedRelationshipParent


@dataclass(frozen=True)
class RelationshipSourceContext:
    feature_policy: ResearchFeaturePolicy
    outcome_policy: CorrectedOutcomePolicy
    facts: PredictorFailureFacts | None
    bindings: AdjustedHistoryBindings
    memberships: pd.DataFrame
    spy_unit_id: str
    predictor_manifest: dict[str, Any]
    predictor_request: dict[str, Any]
    source_files: dict[str, str]
    basis: dict[str, Any]


def _toml(root: Path, path: str, digest: str) -> dict[str, Any]:
    check_files(root, {path: digest})
    return tomllib.loads(inside(root, path).read_text(encoding="utf-8"))


def verify_source_context(root: Path, policy: ReturnRelationshipPublicationPolicy,
    parent: VerifiedRelationshipParent,
) -> RelationshipSourceContext:
    feature = ResearchFeaturePolicy.model_validate(_toml(root, policy.feature_config.path, policy.feature_config.sha256))
    if feature.strategy_contract != policy.strategy_contract:
        raise DataReadinessError("relationship source and strategy contracts differ")
    outcome = CorrectedOutcomePolicy.model_validate_json(json.dumps(
        _toml(root, feature.outcome_source_config.path, feature.outcome_source_config.sha256), default=str))
    facts = None
    if policy.predictor_failure_facts is not None:
        facts = PredictorFailureFacts.model_validate_json(json.dumps(read_object(
            inside(root, policy.predictor_failure_facts.path), policy.predictor_failure_facts.sha256)))
        if facts.feature_config != policy.feature_config or facts.decision_config != feature.outcome_source_config:
            raise DataReadinessError("relationship quarantine authorities differ")
    inherited = parent.source_files
    for pin in (feature.adjusted_archive_authority, feature.adjusted_plan_authority, outcome.research_contract, outcome.parent_config,
            policy.feature_config, feature.outcome_source_config, policy.predictor_failure_facts, policy.strategy_contract):
        if pin is None:
            continue
        if inherited.get(inside(root, pin.path).relative_to(root).as_posix()) != pin.sha256:
            raise DataReadinessError("relationship source authority substituted outside saved parent")
    parent_policy = _toml(root, outcome.parent_config.path, outcome.parent_config.sha256)
    preflight_name = inside(root, parent_policy["preflight_path"]).relative_to(root).as_posix()
    if inherited.get(preflight_name) != parent_policy["preflight_sha256"]:
        raise DataReadinessError("relationship membership preflight differs from saved parent")
    preflight = read_object(inside(root, preflight_name), parent_policy["preflight_sha256"])
    membership_name = inside(root, preflight["request"]["membership_path"]).relative_to(root).as_posix()
    if membership_name not in inherited:
        raise DataReadinessError("relationship membership lacks saved parent ownership")
    membership_pin = {membership_name: inherited[membership_name]}
    check_files(root, membership_pin)
    memberships = _projection(inside(root, membership_name), MEMBERSHIP_COLUMNS)
    check_files(root, membership_pin)
    bindings = load_adjusted_history_bindings(root=root, plan_authority=feature.adjusted_plan_authority,
        archive_authority=feature.adjusted_archive_authority)
    source = pins(root, bindings.source_files, membership_pin,
        {outcome.parent_config.path: outcome.parent_config.sha256, preflight_name: parent_policy["preflight_sha256"]})
    pins(root, inherited, source)
    spy = [key for key, record in bindings.source.records.items()
        if record["role"] == "benchmark" and record["ticker"] == "SPY"]
    if len(spy) != 1:
        raise DataReadinessError("relationship source requires one exact SPY query unit")
    predictors = []
    expected_facts = policy.predictor_failure_facts.model_dump(mode="json") if policy.predictor_failure_facts else None
    expected_declared = {pin.path: pin.sha256 for pin in (policy.feature_config, feature.outcome_source_config,
        feature.strategy_contract, feature.adjusted_plan_authority, feature.adjusted_archive_authority)}
    for name, digest in inherited.items():
        if not name.startswith("data/features/") or not name.endswith("/_manifest.json"):
            continue
        value = read_object(inside(root, name), digest)
        if value.get("schema") != "market_predictor.research_predictors" or value.get("rows") != parent.manifest["rows"]:
            continue
        lineage = value.get("derivation")
        if (lineage is not None and (expected_facts is None or lineage.get("approved_failure_facts") != expected_facts)
                or lineage is None and expected_facts is not None):
            raise DataReadinessError("relationship predictor derivation requires its exact failure facts")
        request_name = (inside(root, name).parent / "_request.json").relative_to(root).as_posix()
        if inherited.get(request_name) != value.get("request_sha256"):
            raise DataReadinessError("relationship predictor request lacks saved parent ownership")
        request = read_object(inside(root, request_name), value["request_sha256"])
        if (value.get("status") != "technical_inputs_complete_research_only" or value.get("failed_groups") != {}
                or value.get("training_eligible") is not False or value.get("promotion_eligible") is not False
                or value.get("exclusions_added") != []
                or request.get("schema") != "market_predictor.research_predictor_request"
                or request.get("config_sha256") != policy.feature_config.sha256
                or pins(root, request.get("declared_source_files", {})) != pins(root, expected_declared)
                or pins(root, request.get("adjusted_source_files", {})) != pins(root, bindings.source_files)
                or request.get("expected_rows") != parent.manifest["rows"]):
            raise DataReadinessError("relationship saved predictor source policy or population differs")
        predictors.append((value, request))
    if len(predictors) != 1:
        raise DataReadinessError("relationship source ownership lacks one exact saved predictor inventory")
    if facts is not None:
        report = read_object(inside(root, facts.observations.path), facts.observations.sha256)
        if report.get("adjusted_archive_authority") != feature.adjusted_archive_authority.model_dump(mode="json"):
            raise DataReadinessError("relationship quarantine report belongs to another adjusted archive")
        for fact in facts.failures:
            _observation(root, facts, fact)
    basis = {"adjustment": "all", "price_feed": "sip", "source": "alpaca",
        "adjusted_plan_authority": feature.adjusted_plan_authority.model_dump(mode="json"),
        "adjusted_archive_authority": feature.adjusted_archive_authority.model_dump(mode="json"),
        "stream_policy": "one_verified_query_unit_per_parent_identity_window_never_splice",
        "historical_availability": "market_interval_close_not_first_seen"}
    check_files(root, source)
    return RelationshipSourceContext(feature, outcome, facts, bindings, memberships, spy[0],
        predictors[0][0], predictors[0][1], source, basis)


def stock_inventory(population: pd.DataFrame, context: RelationshipSourceContext) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    data = bind_adjusted_history_decisions(population, context.bindings)
    if json_sha256(sorted(data.decision_id)) != context.predictor_request["decision_ids_sha256"]:
        raise DataReadinessError("relationship decision population differs from saved predictor request")
    failures = {fact.group_key: fact for fact in context.facts.failures} if context.facts else {}
    for (identity, symbol), group in data.groupby(["security_id", "source_group"], sort=True):
        key = json_sha256([identity, symbol])
        original = context.predictor_manifest["groups"].get(key)
        if original is None or original["rows"] != len(group) or original["decision_ids_sha256"] != json_sha256(sorted(group.decision_id)):
            raise DataReadinessError("relationship stock ownership differs from saved predictor group")
        fact = failures.get(key)
        record = dict(context.bindings.source.records[str(symbol)])
        result[key] = {"security_id": str(identity), "source_group": str(symbol), "rows": len(group),
            "decision_ids_sha256": original["decision_ids_sha256"],
            "artifact": record,
            "quarantine": fact.model_dump(mode="json") if fact is not None else None}
    if set(result) != set(context.predictor_manifest["groups"]):
        raise DataReadinessError("relationship stock inventory omits a saved predictor group")
    return result


def inventory_pins(root: Path, context: RelationshipSourceContext, inventory: dict[str, dict[str, Any]]) -> dict[str, str]:
    for item in inventory.values():
        record = context.bindings.source.records.get(item["source_group"])
        if (record is None or dict(record) != item["artifact"] or record["security_id"] != item["security_id"]
                or record["role"] != "stock"):
            raise DataReadinessError("relationship inventory differs from the verified query source")
    result = pins(root, context.source_files)
    check_files(root, result)
    return result


def relationship_sources(parent_sha256: str, context: RelationshipSourceContext,
    inventory: dict[str, dict[str, Any]],
) -> ReturnRelationshipSources:
    return ReturnRelationshipSources(baseline_authority_sha256=parent_sha256,
        stock_authority_sha256=json_sha256({"inventory": inventory, "basis": context.basis}),
        spy_authority_sha256=json_sha256({"artifact": dict(context.bindings.source.records[context.spy_unit_id]), "basis": context.basis}),
        availability_semantics="historical_proxy", availability_policy_id="market_interval_close",
        availability_policy_sha256=json_sha256({"policy": "market_interval_close", "basis": context.basis,
            "source_files": context.source_files}), price_adjustment="all",
        price_basis_and_vintage_id="saved-inventory-" + json_sha256(context.basis))
