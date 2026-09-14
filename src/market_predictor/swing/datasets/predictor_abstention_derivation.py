"""Explicit nullable completion of reviewed source failures, never issuer exclusion.

The parent remains immutable. Only a finished, exhaustive checkpoint can be
derived; failure approval is an independently pinned input, not inferred here.
"""
from __future__ import annotations

import json
import shutil
import tomllib
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq
from pydantic import Field, model_validator

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence
from market_predictor.swing.datasets.corrected_decisions import verified_corrected_decision_partitions
from market_predictor.swing.datasets.corrected_outcomes import (
    _projection,
    load_corrected_outcome_policy,
    verified_corrected_research_sources,
)
from market_predictor.swing.datasets.feature_history_plan import NUMERIC_END, WARMUP_START
from market_predictor.swing.datasets.initial_fit_raw_share_plan import MEMBERSHIP_COLUMNS
from market_predictor.swing.datasets.research_feature_sources import raw_dollar_volume_inputs
from market_predictor.swing.datasets.research_features import (
    CORRECTED_QUERY_SYMBOLS,
    _check_population,
    _guard,
    _pins,
    _publish,
    _verify_part,
)
from market_predictor.swing.datasets.session_requirements import session_abstentions_by_ticker
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.features.adjusted_source import (
    AdjustedTechnicalSource,
    _bars,
    build_adjusted_technical_source,
    expected_adjusted_history_sessions,
    load_combined_adjusted_inventory,
    read_combined_adjusted_bars,
)
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from market_predictor.swing.features.research_join import DECISION_KEYS

SCHEMA = "market_predictor.predictor_abstention_derivation.v1"
PIN_MAPS = ("declared_source_files", "source_files", "implementation_files", "adjusted_source_files")
PREFIX_IMPLEMENTATION_FILES = ("swing/datasets/action_evidence.py", "swing/datasets/holding_raw_sources.py")


class ReviewedPredictorFailure(HoldingContract):
    group_key: Sha256
    security_id: Identifier
    symbol: Identifier
    rows: Annotated[int, Field(gt=0)]
    parent_failure_sha256: Sha256
    reason_code: Literal["unverified_issuer_history", "invalid_observation_stream"]
    quarantine: Literal["entire_failed_group", "suffix_from_first_invalid"]
    first_invalid_session: date | None
    boundary_observation_sha256: Sha256 | None
    source_artifacts: Annotated[tuple[SourcePin, ...], Field(min_length=1)]
    reviewed_evidence: Annotated[tuple[SourcePin, ...], Field(min_length=1)]
    detail: Identifier

    @model_validator(mode="after")
    def exact_quarantine_scope(self) -> Self:
        if self.reason_code == "unverified_issuer_history":
            if (self.quarantine != "entire_failed_group" or self.first_invalid_session is not None
                    or self.boundary_observation_sha256 is not None):
                raise ValueError("unverified issuer history requires whole-stream quarantine")
        elif (self.quarantine != "suffix_from_first_invalid" or self.first_invalid_session is None
                or not WARMUP_START <= self.first_invalid_session <= NUMERIC_END or self.boundary_observation_sha256 is None):
            raise ValueError("observation failures require a pinned first-invalid boundary and prefix replay")
        return self


class PredictorFailureFacts(HoldingContract):
    schema_version: Literal["market_predictor.predictor_failure_facts.v1"]
    parent_checkpoint_sha256: Sha256
    parent_request_sha256: Sha256
    decision_config: SourcePin
    feature_config: SourcePin
    observations: SourcePin
    parent_run_finished: Literal[True]
    approval_scope: Literal["causal_prefix_replay_and_nullable_completion_only"]
    reviewed_by: Identifier
    failures: Annotated[tuple[ReviewedPredictorFailure, ...], Field(min_length=1)]


def _merge(root: Path, *maps: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for pins in maps:
        for name, digest in pins.items():
            pin = SourcePin(path=name, sha256=digest)
            key = inside(root, pin.path).relative_to(root).as_posix()
            if key in result and result[key] != digest:
                raise DataReadinessError(f"derivation source pins conflict: {key}")
            result[key] = digest
    return result


def _pin(root: Path, path: Path, digest: str) -> dict[str, str]:
    return {inside(root, path).relative_to(root).as_posix(): digest}


def _observation(root: Path, facts: PredictorFailureFacts, fact: ReviewedPredictorFailure) -> dict[str, Any]:
    report = pinned_object(inside(root, facts.observations.path), facts.observations.sha256)
    if (report.get("schema") != "market_predictor.predictor_source_failure_observations.v1"
            or report.get("numeric_first") != str(WARMUP_START) or report.get("numeric_last") != str(NUMERIC_END)):
        raise DataReadinessError("reviewed observations escape the frozen numeric window")
    records = [item for item in report["observations"] if item["security_id"] == fact.security_id and item["ticker"] == fact.symbol]
    if len(records) != 1:
        raise DataReadinessError("reviewed failure requires one exact issuer observation record")
    record: dict[str, Any] = records[0]
    if (len(fact.source_artifacts) != 1 or _merge(root, {record["source_path"]: record["source_sha256"]})
            != _merge(root, {fact.source_artifacts[0].path: fact.source_artifacts[0].sha256})):
        raise DataReadinessError("reviewed observation source bytes differ from failure facts")
    if fact.first_invalid_session is not None:
        invalid = record["invalid_rows"]
        if not invalid:
            raise DataReadinessError("reviewed first-invalid observation is absent")
        first = min(invalid, key=lambda row: date.fromisoformat(row["session_date_et"]))
        if (date.fromisoformat(first["session_date_et"]) != fact.first_invalid_session
                or json_sha256(first) != fact.boundary_observation_sha256 or first["invalid_fields"] != ["volume"]
                or first["ohlcv"]["volume"] != 0):
            raise DataReadinessError("reviewed first-invalid boundary evidence differs")
    return record


def _inputs(root: Path, checkpoint_pin: SourcePin, facts_pin: SourcePin,
    ) -> tuple[dict[str, Any], dict[str, Any], PredictorFailureFacts, dict[str, str]]:
    checkpoint_path = inside(root, checkpoint_pin.path)
    checkpoint = pinned_object(checkpoint_path, checkpoint_pin.sha256)
    facts_path = inside(root, facts_pin.path)
    facts_object = pinned_object(facts_path, facts_pin.sha256)
    facts = PredictorFailureFacts.model_validate_json(json.dumps(facts_object))
    request_path = checkpoint_path.parent / "_request.json"
    request = pinned_object(request_path, checkpoint["request_sha256"])
    if (facts.parent_checkpoint_sha256 != checkpoint_pin.sha256
            or facts.parent_request_sha256 != checkpoint["request_sha256"]
            or checkpoint.get("schema") != "market_predictor.research_predictors"
            or checkpoint.get("training_eligible") is not False
            or checkpoint.get("promotion_eligible") is not False or checkpoint.get("exclusions_added") != []
            or checkpoint.get("months") != {} or not checkpoint.get("groups")
            or request.get("schema") != "market_predictor.research_predictor_request"
            or request.get("decision_start") != "2019-07-09" or request.get("numeric_end") != "2024-05-28"
            or request.get("historical_first_seen_proven") is not False
            or request.get("adjusted_feature_columns") != [n for n in TECHNICAL_RANKING_FEATURES if n != "dollar_volume_log"]
            or request.get("raw_feature_columns") != ["dollar_volume_log"]):
        raise DataReadinessError("derivation parent or approved facts contract differs")
    files = _merge(root, *(request[name] for name in PIN_MAPS),
        {checkpoint_pin.path: checkpoint_pin.sha256, facts_pin.path: facts_pin.sha256,
            facts.observations.path: facts.observations.sha256},
        _pin(root, request_path, checkpoint["request_sha256"]))
    declared = _merge(root, request["declared_source_files"])
    if (any(declared.get(inside(root, pin.path).relative_to(root).as_posix()) != pin.sha256
            for pin in (facts.decision_config, facts.feature_config)) or facts.feature_config.sha256 != request["config_sha256"]):
        raise DataReadinessError("failure facts bind another decision or feature configuration")
    approved = {fact.group_key: fact for fact in facts.failures}
    if (len(approved) != len(facts.failures) or set(approved) != set(checkpoint["failed_groups"])
            or set(approved) & set(checkpoint["groups"])):
        raise DataReadinessError("failed groups require exact distinct reviewed approvals")
    adjusted = _merge(root, request["adjusted_source_files"])
    for key, fact in approved.items():
        failure = checkpoint["failed_groups"][key]
        if (json_sha256(failure) != fact.parent_failure_sha256
                or key != json_sha256([fact.security_id, fact.symbol])
                or failure.get("security_id") != fact.security_id or failure.get("symbol") != fact.symbol
                or type(failure.get("rows")) is not int or failure["rows"] != fact.rows
                or failure.get("error_type") != "DataReadinessError"):
            raise DataReadinessError("approved source failure differs from parent record")
        for source in fact.source_artifacts:
            if adjusted.get(inside(root, source.path).relative_to(root).as_posix()) != source.sha256:
                raise DataReadinessError("approved failure source is not a parent adjusted input")
        files = _merge(root, files, *({pin.path: pin.sha256}
            for pin in (*fact.source_artifacts, *fact.reviewed_evidence)))
        _observation(root, facts, fact)
    group_files: list[dict[str, str]] = []
    for part in checkpoint["groups"].values():
        path = inside(checkpoint_path.parent / "groups", part["path"])
        group_files.extend((_pin(root, path, part["sha256"]), _pin(root, manifest_path_for(path), part["manifest_sha256"])))
    files = _merge(root, files, *group_files)
    return checkpoint, request, facts, files


def validate_historical_predictor_derivation(*, root: Path, manifest: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, str]:
    """Validate historical declarations without substituting current code hashes.

    Returns all inherited/approval pins for the consumer's normal source rechecks.
    This metadata check does not load prices or independently grant source approval.
    """
    lineage = manifest.get("derivation")
    if lineage is None:
        if any(name in manifest or name in request for name in ("source_failures", "unavailable_groups", "recovered_groups", "derivation")):
            raise DataReadinessError("predictor failure lineage is incomplete")
        return {}
    if not isinstance(lineage, dict) or lineage.get("schema") != SCHEMA or request.get("derivation") != lineage:
        raise DataReadinessError("predictor derivation lineage differs")
    checkpoint_pin = SourcePin.model_validate(lineage["parent_checkpoint"])
    facts_pin = SourcePin.model_validate(lineage["approved_failure_facts"])
    parent, parent_request, facts, files = _inputs(root, checkpoint_pin, facts_pin)
    required = ["src/market_predictor/swing/datasets/predictor_abstention_derivation.py"]
    if any(fact.quarantine == "suffix_from_first_invalid" for fact in facts.failures):
        required.extend("src/market_predictor/" + name for name in PREFIX_IMPLEMENTATION_FILES)
    implementation = _merge(root, request["implementation_files"])
    if any(name not in implementation for name in required):
        raise DataReadinessError("derivation request omits inherited or approval source pins")
    files = _merge(root, files, {name: implementation[name] for name in required})
    if (manifest.get("failed_groups") != {} or manifest.get("source_failures") != parent["failed_groups"]
            or set(manifest["groups"]) != set(parent["groups"]) | set(parent["failed_groups"])
            or any(request.get(name) != parent_request.get(name) for name in parent_request if name not in PIN_MAPS)
            or manifest.get("rows") != parent_request["expected_rows"]):
        raise DataReadinessError("nullable completion lost parent population or source failures")
    for key, part in parent["groups"].items():
        if manifest["groups"][key] != {**part, "origin_request_sha256": parent["request_sha256"]}:
            raise DataReadinessError("inherited successful group differs")
    unavailable: dict[str, Any] = {}
    recovered: dict[str, Any] = {}
    for fact in facts.failures:
        part = manifest["groups"][fact.group_key]
        scope = part.get("completion_scope", {})
        prefix_rows, suffix_rows = scope.get("rebuilt_prefix_rows"), scope.get("quarantined_rows")
        eligible = part.get("feature_eligible_rows")
        if (part.get("rows") != fact.rows or part.get("source_failure_sha256") != fact.parent_failure_sha256
                or scope.get("quarantine") != fact.quarantine
                or scope.get("first_invalid_session") != (str(fact.first_invalid_session) if fact.first_invalid_session else None)
                or scope.get("boundary_observation_sha256") != fact.boundary_observation_sha256
                or type(prefix_rows) is not int or type(suffix_rows) is not int or type(eligible) is not int
                or prefix_rows < 0 or suffix_rows < 0 or prefix_rows + suffix_rows != fact.rows
                or not 0 <= eligible <= prefix_rows or (fact.quarantine == "entire_failed_group" and prefix_rows != 0)):
            raise DataReadinessError("completed prefix/suffix differs from approved source failure")
        for field in ("rebuilt_prefix_decision_ids_sha256", "quarantined_decision_ids_sha256"):
            SourcePin(path=field, sha256=scope[field])
        if suffix_rows:
            unavailable[fact.group_key] = scope
        if fact.quarantine == "suffix_from_first_invalid":
            recovered[fact.group_key] = scope
    if (manifest.get("unavailable_groups") != unavailable or manifest.get("recovered_groups") != recovered
            or manifest.get("unavailable_rows") != sum(scope["quarantined_rows"] for scope in unavailable.values())
            or manifest.get("rebuilt_prefix_rows") != sum(scope["rebuilt_prefix_rows"] for scope in recovered.values())):
        raise DataReadinessError("prefix/suffix completion counts or lineage differ")
    bound = _merge(root, *(request[name] for name in PIN_MAPS))
    if any(bound.get(name) != digest for name, digest in files.items()):
        raise DataReadinessError("derivation request omits inherited or approval source pins")
    return files


def validate_predictor_derivation(*, root: Path, manifest: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, str]:
    """Ordinary consumers still require the currently installed prefix helpers."""
    files = validate_historical_predictor_derivation(root=root, manifest=manifest, request=request)
    if manifest.get("recovered_groups"):
        for name in PREFIX_IMPLEMENTATION_FILES:
            path = Path(__file__).resolve().parents[2] / name
            files = _merge(root, files, _pin(root, path, file_sha256(path)))
        bound = _merge(root, *(request[name] for name in PIN_MAPS))
        if any(bound.get(name) != digest for name, digest in files.items()):
            raise DataReadinessError("derivation request omits inherited or approval source pins")
    return files


def _clocks(groups: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for part in groups.values():
        clocks = part["availability_columns"]
        if (not isinstance(clocks, dict) or set(clocks) != set(TECHNICAL_RANKING_FEATURES)
                or any(not isinstance(v, str) or not v.endswith("_at_utc") for v in clocks.values())
                or (result and clocks != result)):
            raise DataReadinessError("parent predictor availability maps differ")
        result = dict(clocks)
    return result


def _unavailable(group: pd.DataFrame, fact: ReviewedPredictorFailure, clocks: Mapping[str, str]) -> pd.DataFrame:
    if fact.first_invalid_session is not None and group.session_date_et.lt(fact.first_invalid_session).any():
        raise DataReadinessError("cannot quarantine causally earlier decision rows")
    columns = list(dict.fromkeys((*DECISION_KEYS, "sector", "session_date_et", "primary_benchmark", "parent_decision_id")))
    rows = group.loc[:, columns].copy()
    rows[list(TECHNICAL_RANKING_FEATURES)] = float("nan")
    for clock in set(clocks.values()):
        rows[clock] = pd.Series(pd.NaT, index=rows.index, dtype="datetime64[ns, UTC]")
    rows["feature_profile"] = "technical_market"
    rows["feature_eligible"] = False
    # No validated usable bars, not a claim that the provider observed zero bars.
    rows["daily_bar_count"] = 0
    rows["technical_missing_reasons"] = json.dumps([f"reviewed_source_failure:{fact.reason_code}:{fact.group_key}"])
    return rows


def _completion_scope(group: pd.DataFrame, fact: ReviewedPredictorFailure) -> dict[str, Any]:
    earlier = (group.session_date_et.lt(fact.first_invalid_session) if fact.first_invalid_session is not None
        else pd.Series(False, index=group.index))
    return {"quarantine": fact.quarantine,
        "first_invalid_session": str(fact.first_invalid_session) if fact.first_invalid_session is not None else None,
        "boundary_observation_sha256": fact.boundary_observation_sha256,
        "rebuilt_prefix_rows": int(earlier.sum()), "quarantined_rows": int((~earlier).sum()),
        "rebuilt_prefix_decision_ids_sha256": json_sha256(sorted(group.loc[earlier, "decision_id"])),
        "quarantined_decision_ids_sha256": json_sha256(sorted(group.loc[~earlier, "decision_id"]))}


def _validated_prefix(frame: pd.DataFrame, fact: ReviewedPredictorFailure, observation: Mapping[str, Any]) -> pd.DataFrame:
    """Prove the full earlier stream passes the canonical validator and the boundary fails."""
    if fact.first_invalid_session is None or len(frame) != observation["bounded_rows"]:
        raise DataReadinessError("prefix replay requires the complete bounded observed stream")
    days = pd.to_datetime(frame.bar_start_utc, utc=True, errors="raise").dt.tz_convert("America/New_York").dt.date
    if days.isna().any() or not days.between(WARMUP_START, NUMERIC_END).all():
        raise DataReadinessError("prefix boundary cannot be located in the bounded source clocks")
    boundary = frame.loc[days.eq(fact.first_invalid_session)]
    if len(boundary) != 1:
        raise DataReadinessError("first-invalid source session must have exactly one observation")
    reported = [row for row in observation["invalid_rows"] if row["session_date_et"] == str(fact.first_invalid_session)]
    if (len(reported) != 1 or json_sha256(reported[0]) != fact.boundary_observation_sha256
            or set(reported[0]["ohlcv"]) != {"open", "high", "low", "close", "volume"}
            or set(reported[0]["clocks"]) != {"bar_start_utc", "bar_end_utc", "available_at_utc"}):
        raise DataReadinessError("first-invalid source observation differs from pinned evidence")
    row = boundary.iloc[0]
    if (any(float(row[name]) != float(value) for name, value in reported[0]["ohlcv"].items())
            or any(pd.Timestamp(row[name]) != pd.Timestamp(value) for name, value in reported[0]["clocks"].items())
            or row.ticker != fact.symbol or row.timeframe != "1d" or row.price_feed != "sip" or row.adjustment != "all"):
        raise DataReadinessError("first-invalid source values or clocks differ from pinned evidence")
    # No bad rows are deleted from the middle of a warmup. Missing sessions remain
    # in the independently reconstructed requirements passed to the canonical builder.
    prefix = _bars(frame.loc[days.lt(fact.first_invalid_session)])
    try:
        _bars(boundary)
    except DataReadinessError as error:
        if str(error) != "adjusted history has invalid or placeholder OHLCV/clocks":
            raise DataReadinessError("boundary failed for a different canonical reason") from error
    else:
        raise DataReadinessError("reviewed first-invalid source observation is actually valid")
    return prefix


def _rebuild_prefix(group: pd.DataFrame, stock: pd.DataFrame, benchmarks: pd.DataFrame,
    memberships: pd.DataFrame, raw: pd.DataFrame, *, fact: ReviewedPredictorFailure,
    observation: Mapping[str, Any], expected_history_sessions: tuple[date, ...], contract: StrategyContract,
    clocks: Mapping[str, str],
) -> AdjustedTechnicalSource:
    prefix = _validated_prefix(stock, fact, observation)
    assert fact.first_invalid_session is not None
    earlier = group.loc[group.session_date_et.lt(fact.first_invalid_session)]
    suffix = group.loc[group.session_date_et.ge(fact.first_invalid_session)]
    parts = [_unavailable(suffix, fact, clocks)] if not suffix.empty else []
    if not earlier.empty:
        benchmark_days = pd.to_datetime(benchmarks.bar_start_utc, utc=True, errors="raise").dt.tz_convert("America/New_York").dt.date
        built = build_adjusted_technical_source(earlier, prefix,
            benchmarks.loc[benchmark_days.lt(fact.first_invalid_session)], memberships, raw,
            security_id=fact.security_id,
            expected_history_sessions=tuple(day for day in expected_history_sessions if day < fact.first_invalid_session),
            contract=contract)
        if dict(built.availability_columns) != dict(clocks):
            raise DataReadinessError("replayed prefix availability differs from parent feature contract")
        rows = built.rows.copy()
        _check_population(earlier, rows)
        rows["parent_decision_id"] = rows.decision_id.map(earlier.set_index("decision_id").parent_decision_id)
        rows["technical_missing_reasons"] = rows.technical_missing_reasons.map(json.dumps)
        parts.append(rows)
    result = pd.concat(parts, ignore_index=True).set_index("decision_id").reindex(group.decision_id).reset_index()
    _check_population(group, result)
    return AdjustedTechnicalSource(result, dict(clocks))


def _recover_prefixes(*, root: Path, facts: PredictorFailureFacts, expected: Mapping[str, pd.DataFrame],
    files: dict[str, str], clocks: Mapping[str, str],
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Replay source authority under its own lease, after the metadata lease exits."""
    recover = [fact for fact in facts.failures if fact.quarantine == "suffix_from_first_invalid"]
    if not recover:
        return {}, files
    config = inside(root, facts.feature_config.path)
    if config.stat().st_size > 65536 or file_sha256(config) != facts.feature_config.sha256:
        raise DataReadinessError("prefix feature policy differs from pinned parent configuration")
    policy = ResearchFeaturePolicy.model_validate(tomllib.loads(config.read_text(encoding="utf-8")))
    if policy.outcome_source_config != facts.decision_config:
        raise DataReadinessError("prefix source policy differs from frozen decision configuration")
    source_config = inside(root, facts.decision_config.path)
    source_policy = load_corrected_outcome_policy(root, source_config, facts.decision_config.sha256)
    # This established evidence loader acquires/releases its own lease. Do not
    # wrap it or the subsequent verified source context in another lease.
    evidence = load_corporate_action_evidence(root=root, config=inside(root, source_policy.action_config.path),
        archive=inside(root, source_policy.action_archive), expected_audit_sha256=source_policy.action_audit_sha256)
    recovered: dict[str, pd.DataFrame] = {}
    with verified_corrected_research_sources(root, source_config, facts.decision_config.sha256, source_policy, evidence) as sources:
        _guard()
        files = _merge(root, files, sources["source_files"])
        _pins(root, files)
        for name in type(policy).model_fields:
            if name != "schema_version":
                pin = getattr(policy, name)
                normalized = inside(root, pin.path).relative_to(root).as_posix()
                if files.get(normalized) != pin.sha256:
                    raise DataReadinessError("prefix policy dependency is not bound by the parent request")
        strategy = load_strategy_contract(inside(root, policy.strategy_contract.path))
        parent = inside(root, policy.parent_request.path).parent
        inventory = load_combined_adjusted_inventory(parent, request_sha256=policy.parent_request.sha256,
            final_manifest_sha256=policy.parent_manifest.sha256, final_authority_sha256=policy.parent_authority.sha256,
            combined_manifest_sha256=policy.combined_manifest.sha256, contract=strategy)
        combined = pinned_object(parent / "combined_daily/_manifest.json", policy.combined_manifest.sha256)
        report = pinned_object(inside(root, facts.observations.path), facts.observations.sha256)
        if report["combined_manifest_sha256"] != policy.combined_manifest.sha256:
            raise DataReadinessError("boundary observations refer to another combined source inventory")
        gap_record = combined["session_gap_audit"]
        gap_path = inside(parent / "combined_daily", gap_record["path"])
        if files.get(gap_path.relative_to(root).as_posix()) != gap_record["sha256"]:
            raise DataReadinessError("prefix session-gap authority is not bound by the parent request")
        gaps = session_abstentions_by_ticker(pinned_object(gap_path, gap_record["sha256"]))
        original_memberships = _projection(sources["membership_path"], MEMBERSHIP_COLUMNS)

        def read(symbol: str, identity: str) -> pd.DataFrame:
            record = inventory[symbol]
            path = inside(parent / "combined_daily", record["path"])
            if files.get(path.relative_to(root).as_posix()) != record["sha256"]:
                raise DataReadinessError("prefix stock or benchmark is not a pinned parent input")
            return read_combined_adjusted_bars(parent / "combined_daily", record, security_id=identity)

        for fact in recover:
            _guard()
            group = expected[fact.group_key]
            observation = _observation(root, facts, fact)
            record = inventory[fact.symbol]
            if (_pin(root, inside(parent / "combined_daily", record["path"]), record["sha256"])
                    != _merge(root, {observation["source_path"]: observation["source_sha256"]})):
                raise DataReadinessError("prefix issuer inventory differs from reviewed source bytes")
            stock = read(fact.symbol, fact.security_id)
            memberships = sources["memberships"].loc[sources["memberships"].security_id.eq(fact.security_id)]
            membership_start = pd.to_datetime(memberships.effective_from_utc, utc=True).dt.tz_convert("America/New_York").dt.date
            memberships = memberships.loc[membership_start.lt(fact.first_invalid_session)]
            symbols = {"SPY", "QQQ", *group.primary_benchmark, *memberships.primary_benchmark.dropna()}
            benchmarks = pd.concat([read(symbol, f"benchmark:{symbol}") for symbol in sorted(symbols)], ignore_index=True)
            history_sessions = expected_adjusted_history_sessions(security_id=fact.security_id,
                memberships=original_memberships.loc[original_memberships.ticker.eq(fact.symbol)],
                sparse_missing_sessions_by_ticker={ticker: tuple(sorted(days)) for ticker, days in gaps.items()})
            earlier = group.loc[group.session_date_et.lt(fact.first_invalid_session)]
            raw = (raw_dollar_volume_inputs(root=root, selection=sources["selection"], decisions=earlier,
                policy=source_policy, policy_sha256=facts.decision_config.sha256) if not earlier.empty else pd.DataFrame())
            recovered[fact.group_key] = _rebuild_prefix(group, stock, benchmarks, memberships, raw, fact=fact,
                observation=observation, expected_history_sessions=history_sessions, contract=strategy, clocks=clocks).rows
            del stock, benchmarks, raw
            release_process_memory()
        _pins(root, files)
    return recovered, files


def derive_predictors_with_abstentions(*, root: Path, parent_checkpoint: SourcePin,
    approved_failure_facts: SourcePin, output: Path,
) -> dict[str, Any]:
    """Preserve successful bytes, replay validated prefixes, and retain null suffixes.

    Metadata, canonical source replay, and publication use sequential workspace
    leases; publication follows both source exit checks. Existing output is never
    overwritten. Failed attempts leave an isolated .pending directory, not a final
    artifact. Only approved initial-fit prefixes are rebuilt; no labels or held-out
    numerics are loaded. The existing canonical feature builder owns all formulas.
    """
    root = root.resolve()
    output = inside(root, output)
    parent_dir = inside(root, parent_checkpoint.path).parent
    if (output.exists() or output == root / "data/features" or not output.is_relative_to(root / "data/features")
            or output.is_relative_to(parent_dir) or parent_dir.is_relative_to(output)):
        raise DataReadinessError("derivation requires a new disjoint data/features output")
    parent, parent_request, facts, files = _inputs(root, parent_checkpoint, approved_failure_facts)
    implementation_path = Path(__file__).resolve()
    implementation = _pin(root, implementation_path, file_sha256(implementation_path))
    if any(fact.quarantine == "suffix_from_first_invalid" for fact in facts.failures):
        for name in PREFIX_IMPLEMENTATION_FILES:
            path = implementation_path.parents[2] / name
            implementation = _merge(root, implementation, _pin(root, path, file_sha256(path)))
    files = _merge(root, files, implementation)
    stage = output.with_name(f".{output.name}.{uuid4().hex}.pending")
    with verified_corrected_decision_partitions(root=root, config=inside(root, facts.decision_config.path),
        expected_config_sha256=facts.decision_config.sha256) as projection:
        _guard()
        files = _merge(root, files, projection.source_files)
        _pins(root, files)
        partitions = list(projection.partitions)
        decisions = pd.concat([frame for _, frame in partitions], ignore_index=True)
        if (len(partitions) != len({month for month, _ in partitions}) or decisions.decision_id.duplicated().any()
                or len(decisions) != projection.expected_rows or len(decisions) != parent_request["expected_rows"]
                or parent_request["cohort_sha256"] != projection.cohort_sha256
                or parent_request["retained_security_ids"] != list(projection.retained_security_ids)
                or parent_request["decision_ids_sha256"] != json_sha256(sorted(decisions.decision_id))):
            raise DataReadinessError("derivation differs from the frozen decision population")
        decisions["source_group"] = [CORRECTED_QUERY_SYMBOLS.get(identity, ticker)
            for identity, ticker in zip(decisions.security_id, decisions.parent_ticker, strict=True)]
        expected = {json_sha256([identity, symbol]): group for (identity, symbol), group in
            decisions.groupby(["security_id", "source_group"], sort=True)}
        if set(expected) != set(parent["groups"]) | set(parent["failed_groups"]):
            raise DataReadinessError("parent checkpoint is not exhaustive for the frozen groups")
        clocks = _clocks(parent["groups"])
        for key, part in parent["groups"].items():
            _guard()
            _verify_part(parent_dir / "groups", part, parent["request_sha256"])
            path = inside(parent_dir / "groups", part["path"])
            context = list(dict.fromkeys((*DECISION_KEYS, "sector", "session_date_et", "primary_benchmark", "parent_decision_id")))
            frame, _ = load_canonical_artifact(path, expected_type="swing_research_technical_inputs",
                allow_research=True, columns=context)
            _check_population(expected[key], frame)
            if (part["decision_ids_sha256"] != json_sha256(sorted(expected[key].decision_id))
                    or not frame.set_index("decision_id").parent_decision_id.reindex(expected[key].decision_id).eq(
                        expected[key].parent_decision_id.to_numpy()).all()):
                raise DataReadinessError("inherited predictor parent decision identity differs")
            del frame
        for fact in facts.failures:
            if len(expected[fact.group_key]) != fact.rows:
                raise DataReadinessError("approved unavailable row count differs from frozen population")
        _pins(root, files)
    recovered, files = _recover_prefixes(root=root, facts=facts, expected=expected, files=files, clocks=clocks)
    with heavy_job_lease("predictor causal prefix completion", runtime_dir=inside(root, heavy_job_runtime_dir())):
        _guard()
        _pins(root, files)
        lineage = {"schema": SCHEMA, "parent_checkpoint": parent_checkpoint.model_dump(mode="json"),
            "approved_failure_facts": approved_failure_facts.model_dump(mode="json")}
        request = {**parent_request, "source_files": _merge(root, parent_request["source_files"], files),
            "implementation_files": _merge(root, parent_request["implementation_files"], implementation), "derivation": lineage}
        stage.mkdir(parents=True)
        write_json_object(stage / "_request.json", request)
        request_pin = file_sha256(stage / "_request.json")
        groups: dict[str, Any] = {}
        (stage / "groups").mkdir()
        for key, part in parent["groups"].items():
            _guard()
            source = inside(parent_dir / "groups", part["path"])
            destination = inside(stage / "groups", part["path"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            shutil.copyfile(manifest_path_for(source), manifest_path_for(destination))
            _verify_part(stage / "groups", part, parent["request_sha256"])
            groups[key] = {**part, "origin_request_sha256": parent["request_sha256"]}
        for fact in facts.failures:
            frame = (recovered[fact.group_key] if fact.quarantine == "suffix_from_first_invalid"
                else _unavailable(expected[fact.group_key], fact, clocks))
            part = _publish(frame, stage / "groups" / f"{fact.group_key}.parquet", request_pin)
            groups[fact.group_key] = {**part, "availability_columns": clocks,
                "source_failure_sha256": fact.parent_failure_sha256,
                "completion_scope": _completion_scope(expected[fact.group_key], fact)}
        arrow: Any = pds
        paths = [str(inside(stage / "groups", part["path"])) for part in groups.values()]
        # Null groups lack unavailable membership context; unify all schemas rather
        # than let the first (possibly null-only) file discard successful columns.
        parquet: Any = pq
        schema = pa.unify_schemas([parquet.read_schema(path) for path in paths], promote_options="permissive")
        dataset = arrow.dataset(paths, format="parquet", schema=schema)
        months: dict[str, Any] = {}
        for month, population in partitions:
            _guard()
            frame = dataset.to_table(filter=arrow.field("session_date_et").isin(list(population.session_date_et.unique())),
                use_threads=False).to_pandas()
            _check_population(population, frame)
            months[month] = _publish(frame.sort_values("decision_id"), stage / "months" / f"{month}.parquet", request_pin)
            del frame
            release_process_memory()
        scopes = {fact.group_key: groups[fact.group_key]["completion_scope"] for fact in facts.failures}
        result = {"schema": "market_predictor.research_predictors", "status": "technical_inputs_complete_research_only",
            "request_sha256": request_pin, "groups": groups, "months": months, "failed_groups": {},
            "source_failures": parent["failed_groups"],
            "unavailable_groups": {key: scope for key, scope in scopes.items() if scope["quarantined_rows"]},
            "recovered_groups": {key: scope for key, scope in scopes.items() if scope["quarantine"] == "suffix_from_first_invalid"},
            "unavailable_rows": sum(scope["quarantined_rows"] for scope in scopes.values()),
            "rebuilt_prefix_rows": sum(scope["rebuilt_prefix_rows"] for scope in scopes.values()), "derivation": lineage,
            "rows": len(decisions), "feature_eligible_rows": sum(p["feature_eligible_rows"] for p in months.values()),
            "training_eligible": False, "promotion_eligible": False, "exclusions_added": []}
        validate_predictor_derivation(root=root, manifest=result, request=request)
        _pins(root, files)
        if pinned_object(stage / "_request.json", request_pin) != request or output.exists():
            raise DataReadinessError("derivation staging or destination changed")
        for part in groups.values():
            _verify_part(stage / "groups", part, part.get("origin_request_sha256", request_pin))
        for part in months.values():
            _verify_part(stage / "months", part, request_pin)
        write_json_object(stage / "_manifest.json", result)
        stage.rename(output)
    return {**result, "manifest_sha256": file_sha256(output / "_manifest.json")}
