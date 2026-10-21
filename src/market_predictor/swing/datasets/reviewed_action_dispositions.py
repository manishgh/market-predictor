"""Explicit source-bound refutations of exact retained corporate-action records."""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.reviewed_action_sources import verify_reviewed_action_sources


class ReviewedActionRecordDisposition(HoldingContract):
    action_id: str = Field(min_length=1)
    family: Literal["stock_dividends"]
    retained_record_sha256: Sha256
    symbol: str = Field(min_length=1)
    disposition: Literal["refuted_by_audited_issuer_share_reconciliation"]
    fiscal_first: date
    fiscal_last: date
    official_receipts: tuple[SourcePin, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def period_order(self) -> ReviewedActionRecordDisposition:
        if not 0 < (self.fiscal_last - self.fiscal_first).days <= 366:
            raise ValueError("audited share reconciliation needs a bounded ordered fiscal period")
        return self


class ReviewedActionRecordDispositionsArtifact(HoldingContract):
    schema_version: Literal["market_predictor.reviewed_action_record_dispositions"]
    scope: Literal["exact_retained_record_refutation_only"]
    production_eligible: Literal[False]
    source_files: dict[str, Sha256]
    records: tuple[ReviewedActionRecordDisposition, ...] = Field(min_length=1, max_length=256)


def load_reviewed_record_dispositions(root: Path, pin: SourcePin | None,
    records: dict[str, tuple[str, dict[str, Any]]], *, memory_guard: Callable[[], None],
) -> tuple[dict[str, ReviewedActionRecordDisposition], dict[str, str]]:
    if pin is None:
        return {}, {}
    path = inside(root, pin.path)
    if path.stat().st_size > 1024**2 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed action dispositions size or hash differs")
    raw = path.read_bytes()
    parse_strict_json_object(raw, label="reviewed action dispositions")
    artifact = ReviewedActionRecordDispositionsArtifact.model_validate_json(raw)
    pins = verify_reviewed_action_sources(root, pin, artifact.source_files,
        (receipt for fact in artifact.records for receipt in fact.official_receipts), memory_guard=memory_guard)
    result: dict[str, ReviewedActionRecordDisposition] = {}
    for fact in artifact.records:
        matched = records.get(fact.action_id)
        if (fact.action_id in result or matched is None or matched[0] != fact.family
                or json_sha256(matched[1]) != fact.retained_record_sha256 or matched[1].get("symbol") != fact.symbol):
            raise DataReadinessError("reviewed disposition differs from exact retained record")
        try:
            ex_date = date.fromisoformat(matched[1]["ex_date"])
        except (KeyError, TypeError, ValueError) as error:
            raise DataReadinessError("refuted action lacks its exact stated ex-date") from error
        if not fact.fiscal_first <= ex_date <= fact.fiscal_last:
            raise DataReadinessError("refuted action is outside the audited share period")
        result[fact.action_id] = fact
    if file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed dispositions changed while loading")
    return result, pins
