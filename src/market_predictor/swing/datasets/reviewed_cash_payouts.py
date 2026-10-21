"""Exact declared USD cash entitlements, separate from their assumed cash clock."""
from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self

import exchange_calendars as xcals
from pydantic import Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.reviewed_action_sources import verify_reviewed_action_sources


class CashRecordBinding(HoldingContract):
    action_id: str
    record_sha256: Sha256


class DeclaredCashComponent(HoldingContract):
    kind: Literal["ordinary", "special", "supplemental"]
    declared_rate_usd: float = Field(gt=0, allow_inf_nan=False)


class ReviewedCashPayout(HoldingContract):
    entitlement_id: str
    symbol: str
    ex_date: date
    currency: Literal["USD"]
    amount_basis: Literal["issuer_declared_gross"]
    components: tuple[DeclaredCashComponent, ...]
    provider_records: tuple[CashRecordBinding, ...]
    official_receipts: tuple[SourcePin, ...]

    @model_validator(mode="after")
    def exact_group(self) -> Self:
        if (not self.entitlement_id.strip() or not self.symbol.strip() or not self.components
                or not self.provider_records or not self.official_receipts
                or len({row.action_id for row in self.provider_records}) != len(self.provider_records)
                or len({row.path for row in self.official_receipts}) != len(self.official_receipts)
                or not xcals.get_calendar("XNYS").is_session(self.ex_date.isoformat())):
            raise ValueError("reviewed cash requires exact unique records, components, official receipts and an ex-session")
        return self

    @property
    def declared_rate_usd(self) -> float:
        return math.fsum(component.declared_rate_usd for component in self.components)


class ReviewedCashArtifact(HoldingContract):
    schema_version: Literal["market_predictor.reviewed_backtest_cash_payouts"]
    scope: Literal["retrospective_declared_cash_only"]
    production_eligible: Literal[False]
    source_files: dict[str, Sha256]
    entitlements: tuple[ReviewedCashPayout, ...]


def load_reviewed_cash_payouts(root: Path, pin: SourcePin | None,
    records: dict[str, tuple[str, dict[str, Any]]], *, documented_cash_approved: bool,
    memory_guard: Callable[[], None],
) -> tuple[tuple[ReviewedCashPayout, ...], dict[str, str]]:
    if pin is None:
        return (), {}
    path = inside(root, pin.path)
    if path.stat().st_size > 1024**2 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed cash facts size or hash differs")
    raw = path.read_bytes()
    parse_strict_json_object(raw, label="reviewed cash facts")
    artifact = ReviewedCashArtifact.model_validate_json(raw)
    if not artifact.entitlements or len(artifact.entitlements) > 256:
        raise DataReadinessError("reviewed cash facts exceed their exact-action bound")
    pins = verify_reviewed_action_sources(root, pin, artifact.source_files,
        (receipt for fact in artifact.entitlements for receipt in fact.official_receipts), memory_guard=memory_guard)
    ids: set[str] = set()
    groups: set[tuple[str, date]] = set()
    bindings: set[str] = set()
    for fact in artifact.entitlements:
        group = fact.symbol, fact.ex_date
        if fact.entitlement_id in ids or group in groups:
            raise DataReadinessError("reviewed cash entitlement or symbol/ex-date group duplicated")
        ids.add(fact.entitlement_id)
        groups.add(group)
        if not math.isfinite(fact.declared_rate_usd):
            raise DataReadinessError("reviewed cash total is not finite")
        members = {identity: record for identity, (family, record) in records.items() if family == "cash_dividends"
            and record.get("symbol") == fact.symbol and record.get("ex_date") == fact.ex_date.isoformat()}
        if set(members) != {row.action_id for row in fact.provider_records}:
            raise DataReadinessError("reviewed cash does not bind every same-session provider cash record")
        cusips = {record.get("cusip") for record in members.values()}
        if len(cusips) != 1 or not all(isinstance(cusip, str) and len(cusip) == 9 and cusip.isalnum() for cusip in cusips):
            raise DataReadinessError("reviewed cash security class is missing or inconsistent")
        special = any(component.kind != "ordinary" for component in fact.components)
        for row in fact.provider_records:
            if row.action_id in bindings or json_sha256(members[row.action_id]) != row.record_sha256:
                raise DataReadinessError("reviewed cash record is changed or bound more than once")
            bindings.add(row.action_id)
            special |= members[row.action_id].get("special") is not False
        if special and not documented_cash_approved:
            raise DataReadinessError("documented special/supplemental cash timing lacks explicit approval")
        for family, record in records.values():
            if (family in {"forward_splits", "reverse_splits", "stock_dividends", "spin_offs"}
                    and fact.symbol in (record.get("symbol"), record.get("source_symbol"), record.get("new_symbol"))
                    and record.get("ex_date") == fact.ex_date.isoformat()):
                raise DataReadinessError("same-session share transformation leaves reviewed cash unit basis unresolved")
    if file_sha256(path) != pin.sha256:
        raise DataReadinessError("reviewed cash facts changed while loading")
    return artifact.entitlements, pins
