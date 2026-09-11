"""Frozen archival query scope, not current membership or issuer authority."""
from __future__ import annotations

import re
import tomllib
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.provider_symbols import PROVIDER_ALPACA, provider_symbol

START = date(2026, 7, 9)
FAMILIES = ("bars_raw", "bars_all", "news")


class Pin(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    start: date = START
    output: str = Field(min_length=1)
    source_manifests: list[Pin] = Field(min_length=1, max_length=100)
    provenance: list[Pin] = Field(default_factory=list, max_length=100)
    additional_symbols: list[str] = Field(default_factory=list, max_length=10000)
    batch_size: int = Field(default=50, ge=1, le=50)
    bars_limit: int = Field(default=10000, ge=1, le=10000)
    news_limit: int = Field(default=50, ge=1, le=50)
    max_pages_per_unit: int = Field(default=10000, ge=1, le=10000)
    revision_overlap_days: int = Field(default=3, ge=0, le=7)


def read_object(path: Path, *, maximum_bytes: int = 16 * 1024**2) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise ValueError("incremental metadata exceeds bounded size")
    return parse_strict_json_object(path.read_bytes(), label="incremental metadata")


def load_config(path: Path) -> tuple[Config, Path, dict[str, Any]]:
    if path.stat().st_size > 1024**2:
        raise ValueError("incremental config exceeds bounded size")
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw.get("start"), str):
        raw["start"] = date.fromisoformat(raw["start"])
    config = Config.model_validate(raw)
    if config.start < date(2019, 7, 9):
        raise ValueError("incremental start must be on or after 2019-07-09")
    output = (path.parent / config.output).resolve()
    symbols: set[str] = set(config.additional_symbols)
    pins: list[dict[str, str]] = []
    for kind, references in (("source_manifest", config.source_manifests), ("provenance", config.provenance)):
        for reference in references:
            source = (path.parent / reference.path).resolve()
            if source.is_relative_to(output):
                raise ValueError("incremental output cannot contain input provenance")
            if file_sha256(source) != reference.sha256:
                raise ValueError("incremental provenance hash mismatch")
            pins.append({"kind": kind, "path": str(source), "sha256": reference.sha256})
            if kind == "source_manifest":
                manifest = read_object(source)
                artifacts = manifest.get("artifacts")
                if not isinstance(artifacts, list) or not artifacts:
                    raise ValueError("source manifest requires artifacts[].ticker")
                for artifact in artifacts:
                    if not isinstance(artifact, dict) or not isinstance(artifact.get("ticker"), str):
                        raise ValueError("source manifest requires artifacts[].ticker")
                    symbols.add(artifact["ticker"])
                if file_sha256(source) != reference.sha256:
                    raise ValueError("source manifest changed during read")
    if not 1 <= len(symbols) <= 10000 or any(not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,31}", s) for s in symbols):
        raise ValueError("invalid or excessive archival symbols")
    mapping = {symbol: provider_symbol(symbol, PROVIDER_ALPACA) for symbol in sorted(symbols)}
    request: dict[str, Any] = {
        "schema": "market_predictor.alpaca_incremental.v1",
        "start": config.start.isoformat(), "symbols": sorted(set(mapping.values())),
        "archival_symbol_mapping": mapping, "provenance": pins,
        "batch_size": config.batch_size, "bars_limit": config.bars_limit,
        "news_limit": config.news_limit, "max_pages_per_unit": config.max_pages_per_unit,
        "revision_overlap_days": config.revision_overlap_days,
        "families": list(FAMILIES), "feed": "sip", "timeframe": "1Day",
        "asof_policy": "unit_utc_date", "end_policy": "next_utc_midnight_minus_one_microsecond",
        "scope": "archival_provider_query_codes_only", "active_membership_authority": False,
        "issuer_attribution": False, "training_ready": False, "survivor_filtering": False,
    }
    request["request_sha256"] = json_sha256(request)
    return config, output, request
