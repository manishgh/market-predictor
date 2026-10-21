"""Reviewed parent share units and regular-way entitlement boundaries; no payout replay."""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.reviewed_action_dates import ReviewedActionDate
from market_predictor.swing.datasets.reviewed_action_sources import verify_reviewed_action_sources


class ReviewedParentSpinUnits(HoldingContract):
    action_id: str = Field(min_length=1)
    retained_record_sha256: Sha256
    source_symbol: str = Field(min_length=1)
    ex_date: date
    parent_units_per_owned_unit: float = Field(ge=1.0, le=1.0, allow_inf_nan=False)
    regular_way_before_ex_includes_distribution: Literal[True]
    official_receipts: tuple[SourcePin, ...] = Field(min_length=1)


class ReviewedParentSpinUnitsArtifact(HoldingContract):
    schema_version: Literal["market_predictor.reviewed_parent_spin_units"]
    scope: Literal["raw_parent_units_and_new_purchase_relevance_only"]
    production_eligible: Literal[False]
    source_files: dict[str, Sha256]
    records: tuple[ReviewedParentSpinUnits, ...] = Field(min_length=1, max_length=256)


def load_reviewed_parent_spin_units(root: Path, pin: SourcePin | None,
    records: dict[str, tuple[str, dict[str, Any]]], dates: dict[str, ReviewedActionDate],
    *, memory_guard: Callable[[], None],
) -> tuple[dict[str, ReviewedParentSpinUnits], dict[str, str]]:
    if pin is None:
        return {}, {}
    path = inside(root, pin.path)
    if path.stat().st_size > 1024**2 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed parent-unit facts size or hash differs")
    raw = path.read_bytes()
    parse_strict_json_object(raw, label="reviewed parent-unit facts")
    artifact = ReviewedParentSpinUnitsArtifact.model_validate_json(raw)
    pins = verify_reviewed_action_sources(root, pin, artifact.source_files,
        (receipt for fact in artifact.records for receipt in fact.official_receipts), memory_guard=memory_guard)
    result: dict[str, ReviewedParentSpinUnits] = {}
    for fact in artifact.records:
        matched, reviewed_date = records.get(fact.action_id), dates.get(fact.action_id)
        if (fact.action_id in result or matched is None or matched[0] != "spin_offs"
                or json_sha256(matched[1]) != fact.retained_record_sha256
                or matched[1].get("source_symbol") != fact.source_symbol
                or reviewed_date is None or reviewed_date.family != "spin_offs"
                or reviewed_date.economic_dates["ex_date"] != fact.ex_date):
            raise DataReadinessError("reviewed parent units differ from exact retained spin and official dates")
        result[fact.action_id] = fact
    if file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed parent-unit facts changed while loading")
    return result, pins

