"""Exact documented spin terms and source-bound daughter prices; never model inputs."""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.sources.alpaca import decode_bars_page_response
from market_predictor.sources.alpaca_corporate_actions import corporate_action_parameters, decode_corporate_actions_page
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import PositionSourceBinding, SourcePin
from market_predictor.swing.datasets.corporate_action_collection import _response
from market_predictor.swing.datasets.history_archive import _validated_bars
from market_predictor.swing.datasets.holding_raw_sources import read_bound_observations
from market_predictor.swing.datasets.reviewed_action_dates import ReviewedActionDate
from market_predictor.swing.datasets.reviewed_action_sources import verify_reviewed_action_sources


class ReviewedSpinOff(HoldingContract):
    action_id: str = Field(min_length=1)
    retained_record_sha256: Sha256
    source_symbol: str = Field(min_length=1)
    source_cusip: str = Field(pattern=r"^[A-Z0-9]{9}$")
    daughter_symbol: str = Field(min_length=1)
    daughter_cusip: str = Field(pattern=r"^[A-Z0-9]{9}$")
    daughter_security_id: str = Field(min_length=1)
    ex_date: date
    daughter_units_per_parent: float = Field(gt=0, allow_inf_nan=False)
    price_selection: SourcePin
    price_response_receipt: SourcePin | None = None
    action_coverage_receipt: SourcePin
    official_receipts: tuple[SourcePin, ...] = Field(min_length=1)


class ReviewedSpinOffArtifact(HoldingContract):
    schema_version: Literal["market_predictor.reviewed_documented_spin_offs"]
    scope: Literal["documented_distribution_and_matched_daughter_sale_only"]
    production_eligible: Literal[False]
    source_files: dict[str, Sha256]
    records: tuple[ReviewedSpinOff, ...] = Field(min_length=1, max_length=256)


@dataclass(frozen=True)
class ReviewedSpinOffInputs:
    fact: ReviewedSpinOff
    observations: pd.DataFrame
    evidence: dict[date, EvidenceReference]
    actions: tuple[tuple[str, dict[str, Any]], ...]


def _object(root: Path, pin: SourcePin, pins: dict[str, str]) -> dict[str, Any]:
    if pins.get(pin.path) != pin.sha256 or file_sha256(inside(root, pin.path)) != pin.sha256:
        raise DataReadinessError("documented spin source pin differs")
    return parse_strict_json_object(inside(root, pin.path).read_bytes(), label="documented spin source")


def _transport(root: Path, pin: SourcePin, pins: dict[str, str], state: str) -> tuple[dict[str, Any], HttpByteResponse]:
    receipt = _object(root, pin, pins)
    directory = inside(root, pin.path).parent
    if receipt.get("state") != state or not isinstance(receipt.get("body_path"), str):
        raise DataReadinessError("documented daughter source lacks a complete acquired receipt")
    def source(leaf: str) -> Path:
        path = inside(directory, leaf)
        name = path.relative_to(root).as_posix()
        if name not in pins or file_sha256(path) != pins[name]:
            raise DataReadinessError("documented daughter transport/body is not pinned")
        return path
    query = parse_strict_json_object(source("_request.json").read_bytes(), label="documented daughter request")
    metadata = parse_strict_json_object(source("transport.json").read_bytes(), label="documented daughter transport")
    if receipt.get("query_sha256") != json_sha256(query):
        raise DataReadinessError("documented daughter query hash differs")
    response = _response(source(receipt["body_path"]).read_bytes(), metadata)
    if (response.sha256 != receipt.get("body_sha256")
            or query.get("url") != response.requested_url.partition("?")[0]):
        raise DataReadinessError("documented daughter receipt body differs")
    return query, response


def load_reviewed_spin_offs(root: Path, pin: SourcePin | None,
    records: dict[str, tuple[str, dict[str, Any]]], dates: dict[str, ReviewedActionDate], *, approved: bool,
    policy_sha256: str, retrieved_at: datetime, memory_guard: Callable[[], None],
) -> tuple[dict[str, ReviewedSpinOffInputs], dict[str, str]]:
    if pin is None:
        return {}, {}
    if not approved:
        raise DataReadinessError("documented spin treatment lacks explicit proportional-share/matched-exit approval")
    path = inside(root, pin.path)
    if path.stat().st_size > 1024**2 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("documented spin facts size or hash differs")
    raw = path.read_bytes()
    parse_strict_json_object(raw, label="documented spin facts")
    artifact = ReviewedSpinOffArtifact.model_validate_json(raw)
    pins = verify_reviewed_action_sources(root, pin, artifact.source_files,
        (receipt for fact in artifact.records for receipt in fact.official_receipts), memory_guard=memory_guard)
    result: dict[str, ReviewedSpinOffInputs] = {}
    for fact in artifact.records:
        memory_guard()
        match, official_date = records.get(fact.action_id), dates.get(fact.action_id)
        if (fact.action_id in result or match is None or match[0] != "spin_offs"
                or json_sha256(match[1]) != fact.retained_record_sha256 or official_date is None
                or official_date.family != "spin_offs"
                or official_date.economic_dates["ex_date"] != fact.ex_date):
            raise DataReadinessError("documented spin differs from exact retained action and official date")
        row = match[1]
        if ((row.get("source_symbol"), row.get("new_symbol"), row.get("source_cusip"), row.get("new_cusip"))
                != (fact.source_symbol, fact.daughter_symbol, fact.source_cusip, fact.daughter_cusip)
                or fact.source_cusip == fact.daughter_cusip):
            raise DataReadinessError("documented spin parent/daughter security class differs")
        parent_rate, child_rate = row.get("source_rate"), row.get("new_rate")
        if (isinstance(parent_rate, bool) or not isinstance(parent_rate, (int, float))
                or isinstance(child_rate, bool) or not isinstance(child_rate, (int, float))):
            raise DataReadinessError("documented spin ratio lacks positive numeric source terms")
        if (not 0 < parent_rate < math.inf or not 0 < child_rate < math.inf
                or child_rate / parent_rate != fact.daughter_units_per_parent):
            raise DataReadinessError("documented spin ratio differs from retained source terms")
        selection = _object(root, fact.price_selection, pins)
        segments = [segment for segment in selection["segments"]
            if segment["artifact"]["security_id"] == fact.daughter_security_id
            and segment["artifact"]["provider_symbol"] == fact.daughter_symbol]
        if len(segments) != 1:
            raise DataReadinessError("documented daughter price source is missing or ambiguous")
        segment = segments[0]
        unit = segment["artifact"]
        first, last = date.fromisoformat(segment["first_session"]), date.fromisoformat(segment["last_session"])
        price_path = inside(root, Path(segment["archive"]) / unit["bars_path"])
        if (unit["role"] != "stock" or unit["ticker"] != fact.daughter_symbol or first != fact.ex_date
                or not first <= last <= date(2024, 5, 28)
                or pins.get(price_path.relative_to(root).as_posix()) != unit["bars_sha256"]):
            raise DataReadinessError("documented daughter raw price binding differs")
        if fact.price_response_receipt is not None:
            query, response = _transport(root, fact.price_response_receipt, pins, "acquired_exact_nine_sessions")
            params = query.get("params", {})
            if (params.get("symbols") != fact.daughter_symbol or params.get("timeframe") != "1Day"
                    or params.get("feed") != "sip" or params.get("adjustment") != "raw"
                    or params.get("asof") != last.isoformat()):
                raise DataReadinessError("documented daughter raw price query differs")
            page = decode_bars_page_response(response, expected_params=params)
            if page.next_page_token or set(page.bars) != {fact.daughter_symbol}:
                raise DataReadinessError("documented daughter raw prices are incomplete or ambiguous")
            replay = _validated_bars(list(page.bars[fact.daughter_symbol]), unit=unit,
                ingested_at=response.retrieved_at_utc, adjustment="raw")
            projection = pd.read_parquet(price_path)
            if projection.to_dict("records") != replay.to_dict("records"):
                raise DataReadinessError("documented daughter raw projection differs from original provider bytes")
        reference = EvidenceReference(reference=pin.path, artifact_sha256=pin.sha256,
            interpretation_policy_sha256=policy_sha256, record_locator=f"documented spin={fact.action_id}",
            retrieved_at=retrieved_at, available_at=None)
        binding = PositionSourceBinding(position_id="daughter-source:" + fact.action_id, security_id=fact.daughter_security_id,
            unit_id=unit["unit_id"], provider_symbol=fact.daughter_symbol, bars_sha256=unit["bars_sha256"],
            first_session=first, last_session=last, ownership_evidence=(reference,))
        frame, refs = read_bound_observations(root, selection, binding, first=first, last=last,
            interpretation_sha256=policy_sha256)
        query, response = _transport(root, fact.action_coverage_receipt, pins, "acquired_complete_single_page")
        params = query.get("params", {})
        expected = corporate_action_parameters(ticker=fact.daughter_symbol,
            start=date(2019, 7, 9), end=date(2024, 5, 28))
        if params != expected:
            raise DataReadinessError("documented daughter action coverage query differs")
        families, token = decode_corporate_actions_page(response, expected_params=expected)
        if token:
            raise DataReadinessError("documented daughter action coverage is incomplete")
        actions = tuple((family, record) for family, values in families.items() for record in values)
        result[fact.action_id] = ReviewedSpinOffInputs(fact, frame, refs, actions)
    if file_sha256(path) != pin.sha256:
        raise DataReadinessError("documented spin facts changed while loading")
    return result, pins
