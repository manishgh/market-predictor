"""Hash-bound Alpaca bar acquisition plans: request units, plan files and the complete-plan verifier."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.collection.alpaca_bars.contracts import REGULAR_BAR_HISTORY_PLAN_SCHEMA, SESSION_BENCHMARK_PLAN_SCHEMA
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.provider_symbols import provider_symbol

REGULAR_BAR_PLAN_AUTHORITY_SCHEMA = "edge_rebuild.intraday_history_plan_authority.v1"
SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA = "edge_rebuild.selected_session_benchmark_one_minute_plan_authority.v1"
# The plan layers the retained collectors write; each maps to the only authority schema that may sign it.
RETAINED_PLAN_SCHEMAS = {
    REGULAR_BAR_HISTORY_PLAN_SCHEMA: REGULAR_BAR_PLAN_AUTHORITY_SCHEMA,
    SESSION_BENCHMARK_PLAN_SCHEMA: SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA,
}


def load_complete_bar_plan(
    directory: Path,
    *,
    accepted_schemas: Mapping[str, str] = RETAINED_PLAN_SCHEMAS,
) -> dict[str, Any]:
    """Verify a five-minute acquisition plan against its complete authority.

    `accepted_schemas` maps each recognised plan schema to the exact authority
    schema that may sign it. An unregistered schema fails closed, so one
    layer's plan can never be replayed under another layer's identity.
    """

    request = load_plan_json(directory / "_request.json")
    manifest = load_plan_json(directory / "_manifest.json")
    authority = load_plan_json(directory / "_authority.json")
    fingerprint = str(request.get("plan_fingerprint", ""))
    plan_schema = str(manifest.get("schema", ""))
    unhashed_request = {key: value for key, value in request.items() if key != "plan_fingerprint"}
    if (
        not fingerprint
        or json_sha256(unhashed_request) != fingerprint
        or plan_schema not in accepted_schemas
        or manifest.get("plan_fingerprint") != fingerprint
        or authority.get("schema") != accepted_schemas[plan_schema]
        or authority.get("state") != "complete"
        or authority.get("plan_fingerprint") != fingerprint
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
    ):
        raise DataReadinessError("five-minute acquisition plan lacks matching complete authority")
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("ER1A plan has no registered files")
    expected = {"_manifest.json", "_authority.json"}
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("ER1A plan file record is malformed")
        relative = str(raw.get("path", ""))
        path = directory / relative
        expected.add(relative.replace("/", "\\"))
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_file()
            or path.stat().st_size != int(raw.get("bytes", -1))
            or file_sha256(path) != raw.get("sha256")
        ):
            raise DataReadinessError(f"ER1A plan artifact does not verify: {path}")
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if actual != expected:
        raise DataReadinessError("ER1A plan artifact file set differs")
    return manifest


def chunk_request_symbols(
    symbols: list[str],
    *,
    expected_bars_per_symbol: int,
    maximum_symbols_per_unit: int,
    maximum_expected_rows_per_unit: int,
    label: str,
) -> Iterator[tuple[list[str], dict[str, str]]]:
    """Chunk one session cross-section into provider-legal request units."""

    symbols_per_unit = min(
        maximum_symbols_per_unit,
        maximum_expected_rows_per_unit // expected_bars_per_symbol,
    )
    if symbols_per_unit < 1:
        raise DataReadinessError(f"unit row cap cannot fit {label}")
    for offset in range(0, len(symbols), symbols_per_unit):
        chunk = symbols[offset : offset + symbols_per_unit]
        mapping = {provider_symbol(ticker, "alpaca"): ticker for ticker in chunk}
        if len(mapping) != len(chunk):
            raise DataReadinessError(f"provider-symbol collision on {label}")
        yield chunk, mapping


def request_unit_record(
    *,
    unit_id: str,
    session_date: object,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk: list[str],
    mapping: dict[str, str],
    expected_bars_per_symbol: int,
    plan_fingerprint: str,
    timeframe: str,
) -> dict[str, object]:
    """Build one canonical SIP request-unit record."""

    if timeframe not in {"1Min", "5Min"}:
        raise ValueError("request-unit timeframe must be 1Min or 5Min")

    return {
        "unit_id": unit_id,
        "session_date_et": session_date,
        "requested_start_utc": start,
        "requested_end_utc": end,
        "canonical_symbols_json": json.dumps(chunk, separators=(",", ":")),
        "provider_symbols_json": json.dumps(
            sorted(mapping),
            separators=(",", ":"),
        ),
        "provider_to_canonical_json": json.dumps(
            mapping,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "symbol_count": len(chunk),
        "expected_bars_per_symbol": expected_bars_per_symbol,
        "maximum_expected_rows": expected_bars_per_symbol * len(chunk),
        "timeframe": timeframe,
        "price_feed": "sip",
        "adjustment": "all",
        "sort": "asc",
        "limit": 10_000,
        "plan_fingerprint": plan_fingerprint,
    }


def file_record(path: Path, root: Path, rows: int) -> dict[str, object]:
    """Describe one plan artifact the way `_manifest.json` registers it."""

    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def load_plan_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"ER1A JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"ER1A JSON is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def write_plan_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def stable_identity_hash(*parts: str) -> str:
    """Hash ordered identity parts with an unambiguous separator."""

    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
