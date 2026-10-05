"""Preserve explicitly reviewed historical research abstentions without new failures.

Historical facts keep their original group/source identity. The separate mapping
binds a current query and population; it never claims a new failed predictor run.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, resolve_inside_authority
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.adjusted_history_bindings import AdjustedHistoryBindings
from market_predictor.swing.datasets.adjusted_history_source import read_adjusted_history_unit
from market_predictor.swing.datasets.holding_raw_sources import RAW_COLUMNS
from market_predictor.swing.datasets.predictor_abstention_derivation import ReviewedPredictorFailure, _validated_prefix
from market_predictor.swing.datasets.return_relationship_integrity import check_files, pins, read_object

HISTORICAL_OBSERVATION_SHA256 = "70c6f85029e64ba3354ba04b5aa936c2147272f2f527ca9757517b3d31178060"
HISTORICAL_OBSERVATION_SCHEMA = "market_predictor.predictor_source_failure_observations.v1"


def pinned_original_observation_report(root: Path, pin: SourcePin) -> dict[str, Any]:
    """Inspect one frozen historical report; never admit an old current contract."""
    if pin.sha256 != HISTORICAL_OBSERVATION_SHA256:
        raise DataReadinessError("preserved abstention requires the exact original observation report hash")
    report = read_object(inside(root, pin.path), pin.sha256)
    if (report.get("schema") != HISTORICAL_OBSERVATION_SCHEMA
            or report.get("numeric_first") != "2018-05-29" or report.get("numeric_last") != "2024-05-28"):
        raise DataReadinessError("preserved abstention original observation schema or numerical bounds differ")
    return report


def preserved_abstention_evidence(root: Path, mapping: dict[str, Any],
) -> tuple[ReviewedPredictorFailure, dict[str, Any], dict[str, str]]:
    required = {"historical_group_key", "historical_fact", "historical_request", "historical_receipt",
        "historical_publication", "observation", "current_unit_id", "current_source", "security_id", "rows",
        "decision_ids_sha256", "parent_window", "query_window", "expected_sessions_sha256", "scope", "source_issuer_verified"}
    if set(mapping) != required or mapping["source_issuer_verified"] is not False:
        raise DataReadinessError("preserved abstention mapping has an unsupported evidence shape")
    request_pin = SourcePin.model_validate(mapping["historical_request"])
    receipt_pin = SourcePin.model_validate(mapping["historical_receipt"])
    publication_pin = SourcePin.model_validate(mapping["historical_publication"])
    observation_pin = SourcePin.model_validate(mapping["observation"])
    request = read_object(inside(root, request_pin.path), request_pin.sha256)
    receipt = read_object(inside(root, receipt_pin.path), receipt_pin.sha256)
    publication = read_object(inside(root, publication_pin.path), publication_pin.sha256)
    item = request.get("stock_inventory", {}).get(mapping["historical_group_key"])
    if (publication.get("request_sha256") != request_pin.sha256 or receipt.get("manifest_sha256") != publication_pin.sha256
            or receipt.get("status") != "passed" or receipt.get("additions_source_replayed") is not True
            or item is None or item.get("quarantine") != mapping["historical_fact"]
            or item["security_id"] != mapping["security_id"] or item["rows"] != mapping["rows"]
            or item["decision_ids_sha256"] != mapping["decision_ids_sha256"]):
        raise DataReadinessError("preserved abstention changed its original fact or complete decision population")
    fact = ReviewedPredictorFailure.model_validate_json(json.dumps(mapping["historical_fact"]))
    if (fact.security_id != mapping["security_id"] or fact.rows != mapping["rows"]
            or fact.group_key != mapping["historical_group_key"]):
        raise DataReadinessError("preserved abstention issuer/count differs from its historical fact")
    references = (request_pin, receipt_pin, publication_pin, observation_pin, *fact.source_artifacts, *fact.reviewed_evidence)
    files = pins(root, {pin.path: pin.sha256 for pin in references})
    declared = pins(root, request["source_files"])
    for pin in (observation_pin, *fact.source_artifacts, *fact.reviewed_evidence):
        if declared.get(inside(root, pin.path).relative_to(root).as_posix()) != pin.sha256:
            raise DataReadinessError("preserved abstention observation/source lacks historical ownership")
    if observation_pin not in fact.reviewed_evidence or len(fact.source_artifacts) != 1:
        raise DataReadinessError("preserved abstention lacks one exact reviewed source observation")
    report = pinned_original_observation_report(root, observation_pin)
    source = fact.source_artifacts[0]
    records = [row for row in report.get("observations", ()) if row.get("security_id") == fact.security_id
        and row.get("ticker") == fact.symbol and row.get("source_path") == inside(root, source.path).relative_to(root).as_posix()
        and row.get("source_sha256") == source.sha256]
    if len(records) != 1:
        raise DataReadinessError("preserved abstention requires one immutable historical observation")
    observation = records[0]
    if fact.first_invalid_session is None:
        if mapping["scope"] != "whole_history_unavailable_identity_unresolved":
            raise DataReadinessError("whole-history abstention cannot imply resolved issuer identity")
    else:
        invalid = observation["invalid_rows"]
        if not invalid:
            raise DataReadinessError("preserved boundary is absent from its original observation")
        first = min(invalid, key=lambda row: row["session_date_et"])
        if (mapping["scope"] != "unchanged_valid_prefix_only" or first["session_date_et"] != str(fact.first_invalid_session)
                or json_sha256(first) != fact.boundary_observation_sha256 or first["invalid_fields"] != ["volume"]
                or first["ohlcv"]["volume"] != 0):
            raise DataReadinessError("preserved first-invalid scope differs from original evidence")
    check_files(root, files)
    return fact, observation, files


def preserved_physical_prefix(frame: pd.DataFrame, fact: ReviewedPredictorFailure,
    observation: dict[str, Any], expected_sessions: tuple[date, ...],
) -> pd.DataFrame:
    """Use the existing boundary evaluator; retain physical metadata on its prefix."""
    if fact.first_invalid_session is None:
        return frame.iloc[:0].copy()
    # Do not rewrite the original observation's row count to make a new query fit.
    _validated_prefix(frame, fact, observation)
    days = pd.to_datetime(frame.bar_start_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    return frame.loc[days.lt(fact.first_invalid_session) & days.isin(expected_sessions)].reset_index(drop=True)


def read_preserved_stock(root: Path, bindings: AdjustedHistoryBindings, unit_id: str,
    mapping: dict[str, Any], expected_sessions: tuple[date, ...],
) -> pd.DataFrame:
    fact, observation, files = preserved_abstention_evidence(root, mapping)
    record = bindings.source.records.get(unit_id)
    if record is None or unit_id not in bindings.windows:
        raise DataReadinessError("preserved abstention current query is absent")
    source_path = resolve_inside_authority(bindings.source.directory, record["bars_path"])
    expected = SourcePin(path=source_path.relative_to(root).as_posix(), sha256=record["bars_sha256"])
    if (mapping["current_unit_id"] != unit_id or mapping["current_source"] != expected.model_dump(mode="json")
            or record["security_id"] != fact.security_id or record["ticker"] != fact.symbol or record["role"] != "stock"
            or mapping["parent_window"] != dict(bindings.windows[unit_id]["parent"])
            or mapping["query_window"] != dict(bindings.windows[unit_id]["query"])
            or bindings.source_files.get(expected.path) != expected.sha256 or not expected_sessions
            or tuple(sorted(set(expected_sessions))) != expected_sessions
            or mapping["expected_sessions_sha256"] != json_sha256([str(day) for day in expected_sessions])):
        raise DataReadinessError("preserved abstention current query/issuer/window binding differs")
    # Strict current source validation precedes the explicit historical scope.
    unit = read_adjusted_history_unit(bindings.source, unit_id)
    if fact.first_invalid_session is None:
        frame = unit.bars.iloc[:0].copy()
    else:
        first, last = str(record["start_date"]), str(record["end_date"])
        if first < "2018-05-29" or last > "2024-05-28":
            raise DataReadinessError("preserved query escapes the initial-fit window")
        raw = pd.read_parquet(source_path, columns=RAW_COLUMNS,
            filters=[("session_date", ">=", first), ("session_date", "<=", last)])
        if file_sha256(source_path) != expected.sha256 or len(raw) != record["rows"]:
            raise DataReadinessError("preserved query changed during full boundary projection")
        canonical = canonicalize_bars(raw, timeframe="1d", availability_policy="market_interval_close")
        frame = preserved_physical_prefix(canonical, fact, observation, expected_sessions)
    check_files(root, files)
    if file_sha256(source_path) != expected.sha256:
        raise DataReadinessError("preserved query changed after boundary validation")
    return frame
