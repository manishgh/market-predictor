"""Pinned reviewed symbol-correction facts, independent of archive orchestration."""
from __future__ import annotations

import hashlib
import tomllib
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import resolve_inside_authority


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SymbolCorrection(_Strict):
    security_id: str = Field(pattern=r"^cik:\d{10}$")
    ticker: str = Field(pattern=r"^[A-Z][A-Z.]{0,9}$")
    provider_symbol: str = Field(pattern=r"^[A-Z][A-Z.]{0,9}$")
    parent_unit_id: str = Field(pattern=r"^swing-daily-[0-9a-f]{24}$")
    start_date: date
    end_date: date
    document_ids: list[str] = Field(min_length=2)
    interpretation: str = Field(min_length=40)
    record_locators: list[str] = Field(min_length=2)


class SymbolCorrectionPolicy(_Strict):
    schema_version: Literal["market_predictor.swing_symbol_correction_policy"]
    parent_plan: str
    parent_plan_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_archive: str
    parent_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_inventory: str
    document_archive: str
    document_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corrections: list[SymbolCorrection] = Field(min_length=2, max_length=2)


def load_symbol_correction_policy(root: Path, config: Path, expected_sha256: str) -> SymbolCorrectionPolicy:
    path = resolve_inside_authority(root, str(config))
    if path.stat().st_size > 1024**2:
        raise DataReadinessError("symbol correction policy exceeds bound")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DataReadinessError("symbol correction requires its independent reviewed policy pin")
    return SymbolCorrectionPolicy.model_validate(tomllib.loads(payload.decode("utf-8")))
