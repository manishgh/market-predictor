from pathlib import Path
from typing import Any

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.datasets import action_evidence
from market_predictor.swing.datasets.action_evidence import load_corporate_action_evidence
from tests.test_swing_corporate_action_collection import _empty, _page, _run
from tests.test_swing_corporate_action_collection import inventory as inventory


@pytest.fixture(autouse=True)
def available_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(action_evidence, "assert_system_memory_available", lambda: None)


def _load(case: dict[str, Any], digest: str) -> Any:
    return load_corporate_action_evidence(root=case["root"], config=case["config"],
        archive=case["output"], expected_audit_sha256=digest)


def test_real_verifier_precedes_record_projection(inventory: dict[str, Any]) -> None:
    def fetch(params: dict[str, Any], limit: int) -> Any:
        return _page(params, {"cash_dividends": [{"id": params["symbols"],
            "symbol": params["symbols"], "ex_date": "2024-01-05", "rate": 0.5, "special": False}]})

    report = _run(inventory, fetch)
    evidence = _load(inventory, report["audit_sha256"])
    assert evidence.records_by_symbol["AAA"]["cash_dividends"][0]["rate"] == 0.5
    assert evidence.unavailable_symbols == ()
    evidence.recheck(inventory["root"])
    body = next(inventory["output"].glob("tickers/AAA/*/*.bin"))
    body.write_bytes(b"tamper")
    with pytest.raises(DataReadinessError, match="changed"):
        evidence.recheck(inventory["root"])
    with pytest.raises(DataReadinessError):
        _load(inventory, report["audit_sha256"])


def test_empty_queries_are_observed_but_not_missing(inventory: dict[str, Any]) -> None:
    report = _run(inventory, _empty)
    result = _load(inventory, report["audit_sha256"])
    assert result.records_by_symbol == {"AAA": {}, "BBB": {}}
    assert not result.unavailable_symbols


def test_wrong_external_pin_fails(inventory: dict[str, Any]) -> None:
    _run(inventory, _empty)
    with pytest.raises(DataReadinessError, match="audit pin"):
        _load(inventory, "0" * 64)


def test_directory_escape_rejected(inventory: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="escapes"):
        load_corporate_action_evidence(root=tmp_path, config=inventory["config"], archive=tmp_path.parent,
            expected_audit_sha256="0" * 64)


def test_accidental_cached_record_mutation_rejected(inventory: dict[str, Any]) -> None:
    report = _run(inventory, _empty)
    evidence = _load(inventory, report["audit_sha256"])
    evidence.records_by_symbol.pop("AAA")
    with pytest.raises(DataReadinessError, match="in memory"):
        evidence.recheck(inventory["root"])
