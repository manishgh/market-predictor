"""Metadata-only action query scope; neither holdings nor absence admission."""
from __future__ import annotations

import json
import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.swing.contracts.research_cohort import Sha256, SwingResearchCohort

BENCHMARKS = frozenset("SPY QQQ XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY".split())
SUCCESSORS = frozenset("DISH RTX AZN CP FBIN".split())


class ScopePin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    path: str = Field(min_length=1)
    sha256: Sha256


class SuccessorQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    symbol: str = Field(pattern=r"^[A-Z0-9][A-Z0-9.-]{0,14}$")
    inventory: ScopePin
    document_id: str = Field(min_length=1)
    record_locator: str = Field(min_length=1)
    retained_document: ScopePin | None = None


class CorporateActionScope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["market_predictor.swing_corporate_action_scope"]
    source_selection: ScopePin
    cohort: ScopePin
    successors: tuple[SuccessorQuery, ...] = Field(min_length=5, max_length=5)
    process_start: date
    process_end: date
    numeric_end: date
    maximum_pages_per_ticker: int = Field(ge=1, le=100)
    page_limit: int = Field(ge=1, le=1000)
    maximum_response_bytes: int = Field(ge=1024, le=4 * 1024**2)


def _pinned_bytes(root: Path, pin: ScopePin, bound: dict[str, str]) -> bytes:
    path = resolve_inside_authority(root, pin.path)
    if path.stat().st_size > 8 * 1024**2 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("corporate-action scope source pin differs")
    data = path.read_bytes()
    if file_sha256(path) != pin.sha256:
        raise DataReadinessError("corporate-action scope source changed during read")
    bound[path.relative_to(root).as_posix()] = pin.sha256
    return data


def _signed(data: bytes) -> dict[str, Any]:
    value = parse_strict_json_object(data, label="corporate-action scope authority")
    if value.get("audit_sha256") != json_sha256({k: v for k, v in value.items() if k != "audit_sha256"}):
        raise DataReadinessError("corporate-action scope semantic identity differs")
    return value


def prepare_corporate_action_scope(root: Path, config: Path, raw: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct query symbols from pinned metadata without opening price files."""
    policy = CorporateActionScope.model_validate_json(json.dumps(raw))
    if (policy.process_start != date(2018, 5, 29) or policy.process_end != date(2026, 9, 11)
            or policy.numeric_end != date(2024, 5, 28)):
        raise DataReadinessError("corporate-action scope differs from frozen query/numeric boundaries")
    if {item.symbol for item in policy.successors} != SUCCESSORS:
        raise DataReadinessError("corporate-action scope requires the five reviewed source-only successors")
    bound = {config.relative_to(root).as_posix(): file_sha256(config)}
    selection = _signed(_pinned_bytes(root, policy.source_selection, bound))
    report = _signed(_pinned_bytes(root, policy.cohort, bound))
    cohort = SwingResearchCohort.model_validate_json(json.dumps(report["cohort"]))
    if (report.get("cohort_sha256") != cohort.sha256() or not cohort.within_cap
            or len(cohort.original_security_ids) != 631 or len(cohort.excluded_security_ids) != 45
            or len(cohort.retained_security_ids) != 586 or cohort.maximum_exclusion_bps != 1000):
        raise DataReadinessError("corporate-action scope differs from frozen cohort")
    if (selection.get("schema") != "market_predictor.swing_symbol_corrected_sources"
            or selection.get("status") != "source_selection_verified"
            or selection.get("exclusions_added") != []):
        raise DataReadinessError("corporate-action scope requires corrected source selection")
    stocks, benchmarks, symbols, units = set(), set(), set(), set()
    for segment in selection["segments"]:
        artifact = segment["artifact"]
        symbol, identity, unit = artifact["provider_symbol"], artifact["security_id"], artifact["unit_id"]
        first, last = date.fromisoformat(segment["first_session"]), date.fromisoformat(segment["last_session"])
        if (not re.fullmatch(r"[A-Z0-9][A-Z0-9.-]{0,14}", symbol) or unit in units
                or not date(2019, 7, 9) <= first <= last <= policy.numeric_end):
            raise DataReadinessError("corporate-action selected symbol/interval differs")
        archive = (root / segment["archive"]).resolve()
        if not archive.is_relative_to(root) or not (archive / artifact["bars_path"]).resolve().is_relative_to(archive):
            raise DataReadinessError("corporate-action selected source escapes root")
        units.add(unit)
        symbols.add(symbol)
        if artifact["role"] == "stock":
            stocks.add(identity)
        elif artifact["role"] == "benchmark" and identity == f"benchmark:{symbol}":
            benchmarks.add(symbol)
        else:
            raise DataReadinessError("corporate-action selected role/identity differs")
    if len(stocks) != 545 or not stocks.issubset(cohort.retained_security_ids) or benchmarks != BENCHMARKS:
        raise DataReadinessError("corporate-action scope requires 545 retained stocks and all 13 ETFs")
    for successor in policy.successors:
        inventory = tomllib.loads(_pinned_bytes(root, successor.inventory, bound).decode("utf-8"))
        documents = [item for item in inventory.get("documents", []) if item.get("document_id") == successor.document_id]
        if inventory.get("schema_version") != "market_predictor.official_document_inventory" or len(documents) != 1:
            raise DataReadinessError("corporate-action successor requires one pinned completion inventory reference")
        if successor.retained_document is not None:
            _pinned_bytes(root, successor.retained_document, bound)
        symbols.add(successor.symbol)
    return {"policy": policy.model_dump(mode="json"), "tickers": sorted(symbols), "bound_files": bound,
        "scope": "corrected_cohort_provider_process_date_evidence_only"}
