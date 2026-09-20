"""Complete inventory binding for the saved adjusted histories and reviewed absences."""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.store import manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.corrected_outcomes import CorrectedOutcomePolicy
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy
from market_predictor.swing.datasets.feature_history_plan import verify_feature_plan_replay
from market_predictor.swing.datasets.history_archive import load_complete_swing_history_collection
from market_predictor.swing.datasets.predictor_abstention_derivation import PredictorFailureFacts, _observation
from market_predictor.swing.datasets.return_relationship_integrity import check_files, pins, read_object
from market_predictor.swing.datasets.return_relationship_parent import VerifiedRelationshipParent
from market_predictor.swing.features.adjusted_source import load_combined_adjusted_inventory

CORRECTED_SYMBOLS = {"cik:0000798354": "FI", "cik:0001415404": "SATS"}


@dataclass(frozen=True)
class RelationshipSourceContext:
    feature_policy: ResearchFeaturePolicy
    outcome_policy: CorrectedOutcomePolicy
    facts: PredictorFailureFacts
    combined_directory: Path
    corrected_directory: Path
    combined: dict[str, dict[str, Any]]
    corrected: dict[str, dict[str, Any]]
    predictor_manifest: dict[str, Any]
    source_files: dict[str, str]
    basis: dict[str, Any]


def _toml(root: Path, path: str, digest: str) -> dict[str, Any]:
    check_files(root, {path: digest})
    return tomllib.loads(inside(root, path).read_text(encoding="utf-8"))


def _collection_metadata(root: Path, record: dict[str, Any], family: str) -> tuple[dict[str, Any], dict[str, str]]:
    """Follow metadata byte pins in the already verified combined-panel request."""
    if family not in {"pre_collection", "post_collection"}:
        raise DataReadinessError("relationship collection family is unsupported")
    directory = inside(root, record["directory"])
    pre = family == "pre_collection"
    fields = {"_request.json": "request_sha256" if pre else "request_file_sha256",
        "_manifest.json": "manifest_sha256"}
    fields.update({"_authority.json": "authority_sha256"} if pre else {
        "_status.json": "status_sha256", "_source_collections.parquet": "source_collections_sha256"})
    if any(not isinstance(record.get(field), str) for field in fields.values()):
        raise DataReadinessError("relationship collection metadata lacks inherited byte pins")
    bound = pins(root, {inside(directory, name).relative_to(root).as_posix(): record[field]
        for name, field in fields.items()})
    check_files(root, bound)
    metadata = {name: read_object(inside(directory, name), record[field])
        for name, field in fields.items() if name.endswith(".json")}
    request, manifest = metadata["_request.json"], metadata["_manifest.json"]
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    # These two existing collection protocols use different JSON identity encodings.
    identity = json_sha256(payload) if pre else hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if (request.get("request_sha256") != identity or manifest.get("request_sha256") != identity
            or request.get("adjustment") != "all" or request.get("price_feed") != "sip"
            or request.get("provider" if pre else "source") != "alpaca"
            or request.get("timeframe") != ("1Day" if pre else "1d")):
        raise DataReadinessError("relationship collection identity or source basis differs")
    if pre:
        authority = metadata["_authority.json"]
        plan_hashes = request.get("plan_hashes")
        if (request.get("schema") != "edge_rebuild.swing_history_collection.v1"
                or manifest.get("schema") != request["schema"]
                or manifest.get("status") not in {"complete", "complete_with_unavailable"}
                or manifest.get("failed_units") != [] or manifest.get("unattempted_units") != []
                or not isinstance(plan_hashes, dict) or manifest.get("plan_hashes") != plan_hashes
                or authority.get("schema") != "edge_rebuild.swing_history_collection_authority.v1"
                or authority.get("state") != "complete" or authority.get("artifact") != "_manifest.json"
                or authority.get("artifact_sha256") != record["manifest_sha256"]
                or authority.get("request_sha256") != identity):
            raise DataReadinessError("relationship pre-collection authority differs")
        for key in ("authority", "units"):
            if not plan_hashes.get(key + "_sha256") or authority.get("plan_" + key + "_sha256") != plan_hashes[key + "_sha256"]:
                raise DataReadinessError("relationship pre-collection plan binding differs")
        for key in ("unit_set_sha256", "universe_sha256"):
            if not record.get(key) or any(value.get(key) != record[key] for value in (manifest, authority)):
                raise DataReadinessError("relationship pre-collection population binding differs")
        if request.get("universe_sha256") != record["universe_sha256"]:
            raise DataReadinessError("relationship pre-collection request universe differs")
    else:
        status = metadata["_status.json"]
        if request.get("schema") != "swing.daily_history_collection.v1" or record.get("request_identity_sha256") != identity:
            raise DataReadinessError("relationship post-collection request identity differs")
        for terminal in (status, manifest):
            if (terminal.get("schema") != "swing.daily_history_manifest.v1"
                    or terminal.get("status") not in {"complete", "complete_with_gaps"}
                    or terminal.get("request_sha256") != identity or terminal.get("failed_symbols") != {}
                    or terminal.get("source_collections_sha256") != record["source_collections_sha256"]):
                raise DataReadinessError("relationship post-collection terminal metadata differs")
        for key in ("status", "requested_symbols", "observed_symbols", "unavailable_symbols", "skipped_symbols"):
            if status.get(key) != manifest.get(key):
                raise DataReadinessError("relationship post-collection status and manifest disagree")
    return request, bound


def verify_source_context(root: Path, policy: ReturnRelationshipPublicationPolicy,
    parent: VerifiedRelationshipParent,
) -> RelationshipSourceContext:
    feature = ResearchFeaturePolicy.model_validate(_toml(root, policy.feature_config.path, policy.feature_config.sha256))
    if feature.strategy_contract != policy.strategy_contract:
        raise DataReadinessError("relationship source and strategy contracts differ")
    outcome = CorrectedOutcomePolicy.model_validate_json(json.dumps(
        _toml(root, feature.outcome_source_config.path, feature.outcome_source_config.sha256), default=str))
    facts = PredictorFailureFacts.model_validate_json(json.dumps(read_object(
        inside(root, policy.predictor_failure_facts.path), policy.predictor_failure_facts.sha256)))
    if facts.feature_config != policy.feature_config or facts.decision_config != feature.outcome_source_config:
        raise DataReadinessError("relationship quarantine authorities differ")
    inherited = parent.source_files
    source: dict[str, str] = {}
    directory = inside(root, feature.parent_request.path).parent
    combined = load_combined_adjusted_inventory(directory, request_sha256=feature.parent_request.sha256,
        final_manifest_sha256=feature.parent_manifest.sha256, final_authority_sha256=feature.parent_authority.sha256,
        combined_manifest_sha256=feature.combined_manifest.sha256,
        contract=load_strategy_contract(inside(root, policy.strategy_contract.path)))
    for pin in (feature.parent_request, feature.parent_manifest, feature.parent_authority, feature.combined_manifest,
            feature.adjusted_archive_authority, feature.adjusted_plan_authority, outcome.research_contract):
        key = inside(root, pin.path).relative_to(root).as_posix()
        if inherited.get(key) != pin.sha256:
            raise DataReadinessError("relationship source authority substituted outside saved parent")
        source[key] = pin.sha256
    combined_manifest = read_object(inside(root, feature.combined_manifest.path), feature.combined_manifest.sha256)
    parent_request = read_object(inside(root, feature.parent_request.path), feature.parent_request.sha256)
    inputs = parent_request["combined_daily_inputs"]
    if (inputs.get("source") != "alpaca" or inputs.get("timeframe") != "1Day" or inputs.get("adjustment") != "all"
            or inputs.get("price_feed") != "sip" or inputs.get("start_date") != policy.source_start
            or inputs.get("pre_end_date") != "2019-07-08" or inputs.get("post_start_date") != policy.decision_start
            or combined_manifest.get("source_lineage") != {key: inputs[key]
                for key in ("membership_authority", "post_collection", "pre_collection")}):
        raise DataReadinessError("relationship combined source basis or collection boundaries differ")
    for family in ("pre_collection", "post_collection"):
        record = inputs[family]
        _, bound = _collection_metadata(root, record, family)
        pins(root, inherited, bound)
        source = pins(root, source, bound)
    corrected_directory = inside(root, feature.adjusted_archive_authority.path).parent
    corrected_authority = read_object(inside(root, feature.adjusted_archive_authority.path), feature.adjusted_archive_authority.sha256)
    corrected_manifest = read_object(corrected_directory / "_manifest.json", corrected_authority["artifact_sha256"])
    source = pins(root, source, {(corrected_directory / "_manifest.json").relative_to(root).as_posix():
        corrected_authority["artifact_sha256"]})
    if policy.feature_plan_snapshot is not None:
        source = pins(root, source, verify_feature_plan_replay(root=root, authority=feature.adjusted_plan_authority,
            implementation_snapshot=policy.feature_plan_snapshot))
    replay = load_complete_swing_history_collection(corrected_directory,
        plan_directory=inside(root, feature.adjusted_plan_authority.path).parent, expected_adjustment="all",
        expected_plan_authority_sha256=feature.adjusted_plan_authority.sha256,
        feature_plan_snapshot=policy.feature_plan_snapshot)
    if replay != corrected_manifest:
        raise DataReadinessError("relationship corrected collection changed during replay")
    corrected = {item["security_id"]: item for item in replay["unit_artifacts"]}
    if len(corrected) != 2 or set(corrected) != set(CORRECTED_SYMBOLS):
        raise DataReadinessError("relationship corrected collection must contain exactly FI and SATS")
    corrected_basis: dict[str, Any] = {}
    for identity, item in corrected.items():
        if (item["provider_symbol"] != CORRECTED_SYMBOLS[identity] or item["start_date"] != policy.source_start
                or item["end_date"] != policy.source_end or item["status"] != "observed"):
            raise DataReadinessError("relationship corrected full-stream identity differs")
        unit = read_object(inside(corrected_directory, item["unit_manifest_path"]), item["unit_manifest_sha256"])
        for key, digest in ((item["bars_path"], item["bars_sha256"]),
                (item["unit_manifest_path"], item["unit_manifest_sha256"])):
            source = pins(root, source, {inside(corrected_directory, key).relative_to(root).as_posix(): digest})
        for page in unit["pages"]:
            source = pins(root, source, {inside(corrected_directory, page["raw_path"]).relative_to(root).as_posix(): page["raw_sha256"]})
            transport = page.get("transport")
            if not isinstance(transport, dict):
                raise DataReadinessError("relationship corrected stream lacks transport evidence")
            source = pins(root, source, {inside(corrected_directory, transport["body_path"]).relative_to(root).as_posix():
                transport["metadata"]["sha256"]})
        corrected_basis[identity] = {"unit_manifest_sha256": item["unit_manifest_sha256"],
            "provider_symbol": item["provider_symbol"], "asof": item["end_date"],
            "started_at_utc": unit["started_at_utc"], "completed_at_utc": unit["completed_at_utc"],
            "adjustment": unit["adjustment"], "price_feed": unit["price_feed"]}
    predictors = []
    for name, digest in inherited.items():
        if name.startswith("data/features/") and name.endswith("/_manifest.json"):
            value = read_object(inside(root, name), digest)
            if value.get("schema") == "market_predictor.research_predictors" and value.get("derivation", {}).get(
                    "approved_failure_facts") == policy.predictor_failure_facts.model_dump(mode="json"):
                predictors.append(value)
    if len(predictors) != 1 or predictors[0].get("rows") != parent.manifest["rows"]:
        raise DataReadinessError("relationship source ownership lacks one exact saved predictor inventory")
    for fact in facts.failures:
        _observation(root, facts, fact)
    basis = {"adjustment": "all", "price_feed": "sip", "source": "alpaca",
        "combined_request_sha256": combined_manifest["request_sha256"],
        "combined_source_lineage": combined_manifest["source_lineage"], "corrected_streams": corrected_basis,
        "stream_policy": "unchanged_parent_combined_or_single_full_corrected_stream_never_splice",
        "historical_availability": "market_interval_close_not_first_seen"}
    check_files(root, source)
    return RelationshipSourceContext(feature, outcome, facts, directory / "combined_daily", corrected_directory,
        combined, corrected, predictors[0], source, basis)


def stock_inventory(population: pd.DataFrame, context: RelationshipSourceContext) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    data = population.assign(source_group=[CORRECTED_SYMBOLS.get(str(identity), str(ticker))
        for identity, ticker in zip(population.security_id, population.parent_ticker, strict=True)])
    failures = {fact.security_id: fact for fact in context.facts.failures}
    for (identity, symbol), group in data.groupby(["security_id", "source_group"], sort=True):
        key = json_sha256([identity, symbol])
        original = context.predictor_manifest["groups"].get(key)
        if original is None or original["rows"] != len(group) or original["decision_ids_sha256"] != json_sha256(sorted(group.decision_id)):
            raise DataReadinessError("relationship stock ownership differs from saved predictor group")
        fact = failures.get(str(identity))
        corrected = context.corrected.get(str(identity))
        record = corrected if corrected is not None else context.combined.get(str(symbol))
        if record is None:
            raise DataReadinessError("relationship stock group has no admitted adjusted stream")
        result[key] = {"security_id": str(identity), "source_group": str(symbol), "rows": len(group),
            "decision_ids_sha256": original["decision_ids_sha256"],
            "kind": "corrected" if corrected is not None else "combined", "artifact": record,
            "quarantine": fact.model_dump(mode="json") if fact is not None else None}
    if set(result) != set(context.predictor_manifest["groups"]):
        raise DataReadinessError("relationship stock inventory omits a saved predictor group")
    return result


def inventory_pins(root: Path, context: RelationshipSourceContext, inventory: dict[str, dict[str, Any]]) -> dict[str, str]:
    result = dict(context.source_files)
    for item in (*inventory.values(), {"kind": "combined", "artifact": context.combined["SPY"]}):
        record = item["artifact"]
        if item["kind"] == "corrected":
            path = inside(context.corrected_directory, record["bars_path"])
            additions = {path.relative_to(root).as_posix(): record["bars_sha256"]}
        else:
            path = inside(context.combined_directory, record["path"])
            additions = {path.relative_to(root).as_posix(): record["sha256"],
                manifest_path_for(path).relative_to(root).as_posix(): record["canonical_manifest_sha256"]}
        result = pins(root, result, additions)
    check_files(root, result)
    return result


def relationship_sources(parent_sha256: str, context: RelationshipSourceContext,
    inventory: dict[str, dict[str, Any]],
) -> ReturnRelationshipSources:
    return ReturnRelationshipSources(baseline_authority_sha256=parent_sha256,
        stock_authority_sha256=json_sha256({"inventory": inventory, "basis": context.basis}),
        spy_authority_sha256=json_sha256({"artifact": context.combined["SPY"], "basis": context.basis}),
        availability_semantics="historical_proxy", availability_policy_id="market_interval_close",
        availability_policy_sha256=json_sha256({"policy": "market_interval_close", "basis": context.basis,
            "source_files": context.source_files}), price_adjustment="all",
        price_basis_and_vintage_id="saved-inventory-" + json_sha256(context.basis))
