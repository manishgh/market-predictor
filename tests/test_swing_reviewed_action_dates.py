"""Synthetic UNIT official receipts for bounded retrospective date relevance."""
from __future__ import annotations

import json
import tomllib
from datetime import date
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_windows import action_window
from market_predictor.swing.datasets.backtest_corporate_actions import BacktestCorporateActions, load_backtest_corporate_action_context
from tests.test_swing_backtest_corporate_actions import _bridge
from tests.test_swing_funded_policy_inputs import provider as provider


def _name() -> dict[str, Any]:
    return {"id": "exact-name", "old_symbol": "AAA", "new_symbol": "AAA", "old_cusip": "012345678",
        "new_cusip": "987654321", "process_date": "2024-01-02"}


def _spin() -> dict[str, Any]:
    return {"id": "exact-spin", "source_symbol": "AAA", "new_symbol": "CHILD", "ex_date": "2024-01-10",
        "source_rate": 1.0, "new_rate": 0.25}


def _with_facts(provider: Any, family: str, record: dict[str, Any], dates: dict[str, str],
    *, fault: str | None = None) -> BacktestCorporateActions:
    _bridge(provider, {family: [record]})
    directory = provider.root / "official/selected"
    directory.mkdir(parents=True)
    body = directory / "body.bin"
    body.write_bytes(b"UNIT issuer economic date statement, not operational evidence")
    query = {"url": "https://issuer.example/exact-action", "scope_sha256": "a" * 64}
    request = directory / "_request.json"
    request.write_text(json.dumps(query), encoding="utf-8")
    transport = directory / "transport.json"
    transport.write_text(json.dumps({"status_code": 200, "redirect_chain": [], "requested_url": query["url"],
        "final_url": query["url"] if fault != "redirect" else "https://issuer.example/other",
        "sha256": file_sha256(body), "body_length": body.stat().st_size}), encoding="utf-8")
    receipt = directory / "receipt.json"
    receipt.write_text(json.dumps({"state": "archived_unreviewed" if fault != "failed_receipt" else "response_rejected",
        "status_code": 200, "request_sha256": json_sha256(query) if fault != "query" else "e" * 64,
        "url": query["url"], "body_path": body.name, "body_sha256": file_sha256(body),
        "transport_sha256": file_sha256(transport)}), encoding="utf-8")
    pins = {p.relative_to(provider.root).as_posix(): file_sha256(p) for p in (body, request, transport, receipt)}
    fact = {"action_id": record["id"], "family": family,
        "retained_record_sha256": json_sha256(record) if fault != "record" else "e" * 64,
        "economic_dates": dates, "official_receipts": [{"path": receipt.relative_to(provider.root).as_posix(),
            "sha256": file_sha256(receipt)}]}
    artifact = {"schema_version": "market_predictor.reviewed_backtest_action_dates",
        "scope": "retrospective_temporal_relevance_only", "production_eligible": False, "source_files": pins,
        "records": [fact, fact] if fault == "duplicate" else [fact]}
    facts = provider.root / "reviewed-dates.json"
    facts.write_text(json.dumps(artifact), encoding="utf-8")
    policy = provider.root / "reviewed-backtest.toml"
    policy.write_text((provider.root / "backtest.toml").read_text(encoding="utf-8")
        + f'\n[reviewed_action_dates]\npath="{facts.name}"\nsha256="{file_sha256(facts)}"\n', encoding="utf-8")
    if fault == "body_tamper":
        body.write_bytes(b"changed")
    context = load_backtest_corporate_action_context(provider.root, SourcePin(path=policy.name, sha256=file_sha256(policy)))
    return BacktestCorporateActions(provider.root, provider.unit_action_evidence, context, memory_guard=lambda: None)


def test_official_effective_date_excludes_future_identity_change_without_cash_or_mapping(provider: Any) -> None:
    record = _name()
    bridge = _with_facts(provider, "name_changes", record, {"effective_date": "2024-02-01"})
    window = action_window("name_changes", record)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 3), date(2024, 1, 17))
    assert bridge.irrelevance_reason("OTHER", window, date(2024, 1, 3), date(2024, 1, 17)) is None
    assert not bridge.terms and not bridge.split_ratios("AAA", date(2024, 1, 1), date(2024, 3, 1))
    assert record == _name()


@pytest.mark.parametrize("first,last", [("2024-01-03", "2024-02-01"), ("2024-02-01", "2024-02-02")])
def test_identity_boundary_crossing_and_effective_day_remain_unavailable(provider: Any, first: str, last: str) -> None:
    record = _name()
    bridge = _with_facts(provider, "name_changes", record, {"effective_date": "2024-02-01"})
    assert bridge.irrelevance_reason("AAA", action_window("name_changes", record),
        date.fromisoformat(first), date.fromisoformat(last)) is None



@pytest.mark.parametrize("price_history", [False, True])
def test_new_same_symbol_raw_window_after_reviewed_change_needs_no_prior_ownership_mapping(
    provider: Any, price_history: bool,
) -> None:
    record = _name()
    bridge = _with_facts(provider, "name_changes", record, {"effective_date": "2024-02-01"})
    window = action_window("name_changes", record)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 2, 2), date(2024, 2, 12),
        price_history=price_history) == "new_same_symbol_raw_window_after_official_identity_change"
    assert bridge.irrelevance_reason("OTHER", window, date(2024, 2, 2), date(2024, 2, 12),
        price_history=price_history) is None
    assert not bridge.terms and not bridge.split_ratios("AAA", date(2024, 1, 1), date(2024, 3, 1))
    assert record == _name()


def test_post_change_symbol_alias_and_unsourced_date_remain_unavailable(provider: Any) -> None:
    record = _name() | {"new_symbol": "BBB"}
    bridge = _with_facts(provider, "name_changes", record, {"effective_date": "2024-02-01"})
    window = action_window("name_changes", record)
    for symbol in ("AAA", "BBB"):
        assert bridge.irrelevance_reason(symbol, window, date(2024, 2, 2), date(2024, 2, 12)) is None
    bridge.reviewed_dates.clear()
    assert bridge.irrelevance_reason("AAA", window, date(2024, 2, 2), date(2024, 2, 12)) is None


def test_official_spin_dates_bound_prior_holding_but_do_not_create_daughter_shares(provider: Any) -> None:
    record = _spin()
    bridge = _with_facts(provider, "spin_offs", record,
        {"ex_date": "2024-01-10", "record_date": "2024-01-08", "payable_date": "2024-01-09"})
    window = action_window("spin_offs", record)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 3), date(2024, 1, 5))
    assert bridge.irrelevance_reason("CHILD", window, date(2024, 1, 3), date(2024, 1, 5))
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 8), date(2024, 1, 10)) is None
    assert not bridge.terms


def test_known_later_due_bill_is_retained_beyond_official_payable_date(provider: Any) -> None:
    record = _spin() | {"due_bill_redemption_date": "2024-01-19"}
    bridge = _with_facts(provider, "spin_offs", record,
        {"ex_date": "2024-01-10", "record_date": "2024-01-08", "payable_date": "2024-01-09"})
    assert bridge.irrelevance_reason("AAA", action_window("spin_offs", record), date(2024, 1, 11), date(2024, 1, 17)) is None


@pytest.mark.parametrize("fault", ["record", "duplicate", "query", "redirect", "failed_receipt", "body_tamper"])
def test_source_record_and_receipt_failures_cannot_authorize_dates(provider: Any, fault: str) -> None:
    with pytest.raises(DataReadinessError):
        _with_facts(provider, "name_changes", _name(), {"effective_date": "2024-02-01"}, fault=fault)


def test_official_date_cannot_silently_replace_supplied_economic_date(provider: Any) -> None:
    with pytest.raises(DataReadinessError, match="contradict"):
        _with_facts(provider, "spin_offs", _spin(),
            {"ex_date": "2024-01-11", "record_date": "2024-01-08", "payable_date": "2024-01-09"})


def test_original_policy_semantic_hash_stays_exact_when_optional_fact_pin_absent(provider: Any) -> None:
    bridge = _bridge(provider, {"name_changes": [_name()]})
    raw = tomllib.loads((provider.root / "backtest.toml").read_text(encoding="utf-8"))
    assert bridge.context.policy.sha256() == json_sha256(raw)
    assert bridge.reviewed_dates == {}
