"""Bounded synthetic scope and archive identity tests; no provider calls or bars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.research_cohort import SwingResearchCohort
from market_predictor.swing.datasets import corporate_action_collection as collector
from market_predictor.swing.datasets.corporate_action_scope import BENCHMARKS, SUCCESSORS, prepare_corporate_action_scope


def _write(path: Path, data: dict[str, Any]) -> dict[str, str]:
    path.write_text(json.dumps(data), encoding="utf-8")
    return {"path": path.name, "sha256": file_sha256(path)}


def _signed(path: Path, data: dict[str, Any]) -> dict[str, str]:
    unsigned = {k: v for k, v in data.items() if k != "audit_sha256"}
    return _write(path, {**unsigned, "audit_sha256": json_sha256(unsigned)})


@pytest.fixture
def scope(tmp_path: Path) -> dict[str, Any]:
    ids = [f"synthetic:{i:04d}" for i in range(631)]
    cohort = SwingResearchCohort.model_validate_json(json.dumps({
        "schema_version": "market_predictor.swing_research_cohort", "scope": "retrospective_development_restriction",
        "price_basis_status": "not_certified_by_cohort", "combined_daily_inputs_sha256": "a" * 64,
        "original_security_ids": ids, "inherited_excluded_security_ids": ids[:27], "warmup_only_security_ids": [],
        "exclusions": [{"security_id": identity, "tickers": ["EXCLUDED"], "reason": "unresolved_corporate_action"}
            for identity in ids[27:45]], "maximum_exclusion_bps": 1000, "cap_approval_reference": "synthetic-approval",
        "source_files": {"synthetic-parent": "b" * 64},
    }))
    cohort_pin = _signed(tmp_path / "cohort.json", {"cohort": cohort.model_dump(mode="json"), "cohort_sha256": cohort.sha256()})
    segments = []
    for identity, symbol, role in [*( (identity, f"S{i}", "stock") for i, identity in enumerate(ids[45:590])),
            *((f"benchmark:{symbol}", symbol, "benchmark") for symbol in sorted(BENCHMARKS))]:
        segments.append({"first_session": "2019-07-09", "last_session": "2024-05-28", "archive": "raw",
            "artifact": {"security_id": identity, "provider_symbol": symbol, "unit_id": symbol, "role": role,
                "bars_path": f"{symbol}/bars.parquet"}})
    selection_pin = _signed(tmp_path / "selection.json", {
        "schema": "market_predictor.swing_symbol_corrected_sources", "status": "source_selection_verified",
        "exclusions_added": [], "segments": segments,
    })
    inventory = tmp_path / "completion.toml"
    inventory.write_text('schema_version = "market_predictor.official_document_inventory"\n'
        + "".join(f'[[documents]]\ndocument_id = "completion_{symbol}"\n' for symbol in sorted(SUCCESSORS)), encoding="utf-8")
    config = tmp_path / "scope.toml"
    config.write_text("# synthetic config pinned by helper\n", encoding="utf-8")
    return {"root": tmp_path, "config": config, "raw": {
        "schema_version": "market_predictor.swing_corporate_action_scope", "source_selection": selection_pin,
        "cohort": cohort_pin, "successors": [{"symbol": symbol,
            "inventory": {"path": inventory.name, "sha256": file_sha256(inventory)},
            "document_id": f"completion_{symbol}", "record_locator": "Reviewed common-share conversion"}
            for symbol in sorted(SUCCESSORS)],
        "process_start": "2018-05-29", "process_end": "2026-09-11", "numeric_end": "2024-05-28",
        "maximum_pages_per_ticker": 3, "page_limit": 10, "maximum_response_bytes": 2048,
    }}


def _prepare(scope: dict[str, Any]) -> dict[str, Any]:
    return prepare_corporate_action_scope(scope["root"], scope["config"], scope["raw"])


def test_full_scope_uses_metadata_only_and_retains_source_only_successors(scope: dict[str, Any]) -> None:
    assert not (scope["root"] / "raw").exists()
    request = _prepare(scope)
    assert len(request["tickers"]) == 545 + 13 + 5
    assert BENCHMARKS | SUCCESSORS <= set(request["tickers"])
    assert request["scope"] == "corrected_cohort_provider_process_date_evidence_only"
    assert request["policy"]["numeric_end"] == "2024-05-28"
    assert len(request["bound_files"]) == 4
    assert not (scope["root"] / "raw").exists()


@pytest.mark.parametrize("field,value", [
    ("process_start", "2019-07-09"), ("process_end", "2024-05-28"), ("numeric_end", "2026-09-11"),
    ("observation_inventory", "old-inventory.json"), ("expected_securities", 545),
])
def test_scope_mode_and_dates_are_frozen(scope: dict[str, Any], field: str, value: Any) -> None:
    scope["raw"][field] = value
    with pytest.raises((DataReadinessError, ValueError)):
        _prepare(scope)


@pytest.mark.parametrize("kind", ["etf", "stock", "excluded", "duplicate", "date", "archive_escape", "bars_escape", "role"])
def test_resigned_bad_selection_is_not_admitted_as_query_scope(scope: dict[str, Any], kind: str) -> None:
    path = scope["root"] / "selection.json"
    value = json.loads(path.read_text())
    if kind == "etf":
        value["segments"].pop()
    elif kind == "stock":
        value["segments"].pop(0)
    elif kind == "excluded":
        value["segments"][0]["artifact"]["security_id"] = "synthetic:0000"
    elif kind == "duplicate":
        value["segments"].append(value["segments"][0])
    elif kind == "date":
        value["segments"][0]["last_session"] = "2024-05-29"
    elif kind == "archive_escape":
        value["segments"][0]["archive"] = "../outside"
    elif kind == "bars_escape":
        value["segments"][0]["artifact"]["bars_path"] = "../outside.parquet"
    else:
        value["segments"][0]["artifact"]["role"] = "ignored"
    scope["raw"]["source_selection"] = _signed(path, value)
    with pytest.raises(DataReadinessError):
        _prepare(scope)


@pytest.mark.parametrize("kind", ["pin", "document", "symbol", "missing", "escape"])
def test_successor_reference_is_required_and_pinned(scope: dict[str, Any], kind: str) -> None:
    item = scope["raw"]["successors"][0]
    if kind == "pin":
        item["inventory"]["sha256"] = "0" * 64
    elif kind == "document":
        item["document_id"] = "not-in-inventory"
    elif kind == "symbol":
        item["symbol"] = "UNAPPROVED"
    elif kind == "missing":
        scope["raw"]["successors"].pop()
    else:
        item["inventory"]["path"] = "../outside.toml"
    with pytest.raises((DataReadinessError, ValueError)):
        _prepare(scope)


def test_scope_dispatch_uses_exclusive_schema(scope: dict[str, Any]) -> None:
    # JSON-compatible TOML inline tables avoid a test dependency on a TOML writer.
    def toml(value: Any) -> str:
        if isinstance(value, dict):
            return "{ " + ", ".join(f"{key} = {toml(item)}" for key, item in value.items()) + " }"
        if isinstance(value, list):
            return "[" + ", ".join(toml(item) for item in value) + "]"
        return json.dumps(value)
    scope["config"].write_text("".join(f"{key} = {toml(value)}\n" for key, value in scope["raw"].items()), encoding="utf-8")
    request = collector._prepare(scope["root"], scope["config"])
    assert len(request["tickers"]) == 563
    assert "swing/datasets/corporate_action_scope.py" in request["implementation_files"]


def test_retained_completion_document_is_bound_without_invented_receipt(scope: dict[str, Any]) -> None:
    path = scope["root"] / "retained.html"
    path.write_text("<html>synthetic reviewed completion</html>", encoding="utf-8")
    scope["raw"]["successors"][0]["retained_document"] = {"path": path.name, "sha256": file_sha256(path)}
    assert _prepare(scope)["bound_files"][path.name] == file_sha256(path)
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="source pin"):
        _prepare(scope)


def test_archived_acquisition_identity_can_differ_but_not_scope(tmp_path: Path) -> None:
    current = {"policy": {"process_end": "2024-05-28"}, "tickers": ["AAA"], "bound_files": {},
        "scope": "initial_fit_provider_process_date_evidence_only", "implementation_files": collector._implementation()}
    current["request_sha256"] = json_sha256(current)
    original = {**current, "implementation_files": {name: "a" * 64 for name in current["implementation_files"]}}
    original.pop("request_sha256")
    original["request_sha256"] = json_sha256(original)
    _write(tmp_path / "_request.json", original)
    assert collector._archived_request(tmp_path, current) == original
    changed = {**current, "tickers": ["BBB"]}
    with pytest.raises(DataReadinessError, match="request differs"):
        collector._archived_request(tmp_path, changed)


def test_page_memory_guard_retains_partial_receipt_and_stops(scope: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    request = {**_prepare(scope), "request_sha256": "a" * 64}
    def pressure() -> None:
        raise MemoryBudgetError("synthetic physical memory pressure")
    def forbidden(*args: Any) -> Any:
        pytest.fail("memory guard must precede provider call")
    monkeypatch.setattr(collector, "assert_system_memory_available", pressure)
    output = scope["root"] / "collection"
    result = collector._collect(output, request, "AAA", forbidden)
    assert isinstance(result, MemoryBudgetError)
    receipts = list(output.glob("tickers/AAA/*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["state"] == "failed" and receipt["pages"] == []
