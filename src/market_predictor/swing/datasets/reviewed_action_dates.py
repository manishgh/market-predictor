"""Exact official economic dates for retrospective backtest relevance only."""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.reviewed_action_sources import verify_reviewed_action_sources

DateField = Literal["ex_date", "record_date", "payable_date", "effective_date"]


class ReviewedActionDate(HoldingContract):
    action_id: str
    family: Literal["name_changes", "spin_offs"]
    retained_record_sha256: Sha256
    economic_dates: dict[DateField, date]
    official_receipts: tuple[SourcePin, ...]

    @model_validator(mode="after")
    def supported_dates(self) -> Self:
        expected = {"effective_date"} if self.family == "name_changes" else {"ex_date", "record_date", "payable_date"}
        if (set(self.economic_dates) != expected or not self.action_id.strip() or not self.official_receipts
                or len({pin.path for pin in self.official_receipts}) != len(self.official_receipts)):
            raise ValueError("reviewed dates require one exact identity and complete supported economic anchors")
        return self


class ReviewedActionDateArtifact(HoldingContract):
    schema_version: Literal["market_predictor.reviewed_backtest_action_dates"]
    scope: Literal["retrospective_temporal_relevance_only"]
    production_eligible: Literal[False]
    source_files: dict[str, Sha256]
    records: tuple[ReviewedActionDate, ...]


def load_reviewed_action_dates(root: Path, pin: SourcePin | None,
    records: dict[str, tuple[str, dict[str, Any]]], *, memory_guard: Callable[[], None],
) -> tuple[dict[str, ReviewedActionDate], dict[str, str]]:
    if pin is None:
        return {}, {}
    path = inside(root, pin.path)
    if path.stat().st_size > 1024**2 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed action dates size or hash differs")
    raw = path.read_bytes()
    parse_strict_json_object(raw, label="reviewed action dates")
    artifact = ReviewedActionDateArtifact.model_validate_json(raw)
    if not artifact.records or len(artifact.records) > 256:
        raise DataReadinessError("reviewed action dates exceed their exact-action bound")
    pins = verify_reviewed_action_sources(root, pin, artifact.source_files,
        (receipt for fact in artifact.records for receipt in fact.official_receipts), memory_guard=memory_guard)
    result: dict[str, ReviewedActionDate] = {}
    for fact in artifact.records:
        matched = records.get(fact.action_id)
        if (fact.action_id in result or matched is None or matched[0] != fact.family
                or json_sha256(matched[1]) != fact.retained_record_sha256):
            raise DataReadinessError("reviewed action dates do not match one exact retained action record")
        for field, value in fact.economic_dates.items():
            if matched[1].get(field) not in (None, "", value.isoformat()):
                raise DataReadinessError("reviewed action dates contradict a retained economic date")
        result[fact.action_id] = fact
    if file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed action dates changed while loading")
    return result, pins
