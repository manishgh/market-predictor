from __future__ import annotations

import sys
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild import swing_history_collection as collector
from tests.test_swing_history_collection import _FakeSource, _json, _plan, _reauthorize_plan, _write_json

SCOPE = "historical_symbol_correction"
PROVIDER_SYMBOLS = {"ECHO": "SATS", "FISV": "FI"}


@pytest.fixture
def owner_validator(monkeypatch: pytest.MonkeyPatch) -> Mock:
    # Transport-only fixtures deliberately do not claim real policy/document admission.
    name = "market_predictor.swing.datasets.symbol_corrections"
    module = ModuleType(name)
    validator = Mock(return_value=None)
    monkeypatch.setattr(module, "validate_symbol_correction_collection_plan", validator, raising=False)
    monkeypatch.setitem(sys.modules, name, module)
    return validator


def _resign(directory: Path) -> str:
    manifest = _json(directory / "_manifest.json")
    manifest["request_sha256"] = file_sha256(directory / "_request.json")
    _write_json(directory / "_manifest.json", manifest)
    _reauthorize_plan(directory)
    authority = _json(directory / "_authority.json")
    authority["request_sha256"] = manifest["request_sha256"]
    _write_json(directory / "_authority.json", authority)
    return file_sha256(directory / "_authority.json")


def _correction_plan(tmp_path: Path) -> tuple[Path, str]:
    directory = _plan(tmp_path, adjustment="raw")
    # Synthetic one-session geometry isolates collector transport from owner policy.
    units = pd.DataFrame([
        {"security_id": security_id, "ticker": ticker, "start_date": "2020-01-02",
         "end_date": "2020-01-02", "role": "stock"}
        for security_id, ticker in [("cik:0001415404", "ECHO"), ("cik:0000798354", "FISV")]
    ])
    units.to_csv(directory / "daily_bar_units.csv", index=False, lineterminator="\n")
    request = _json(directory / "_request.json")
    request.update(scope=SCOPE, provider_symbols=PROVIDER_SYMBOLS,
                   asof_policy="inclusive_unit_end_date_entity_mapping_not_ownership")
    _write_json(directory / "_request.json", request)
    manifest = _json(directory / "_manifest.json")
    manifest["scope"] = SCOPE
    manifest["daily_bars"].update(planned_units=2, stock_units=2, benchmark_units=0)
    _write_json(directory / "_manifest.json", manifest)
    return directory, _resign(directory)


def _collect(plan: Path, output: Path, source: _FakeSource, pin: str | None, **kwargs: Any) -> dict[str, Any]:
    return collector.collect_swing_history_plan(
        plan_directory=plan, output_directory=output, source_factory=lambda: source,
        provider_symbol_for=PROVIDER_SYMBOLS.__getitem__, expected_plan_authority_sha256=pin, **kwargs,
    )


def test_correction_collects_and_replays_through_owner_with_raw_mapping(
    tmp_path: Path, owner_validator: Mock,
) -> None:
    plan, pin = _correction_plan(tmp_path)
    output, source = tmp_path / "collection", _FakeSource()
    result = _collect(plan, output, source, pin)
    assert result["status"] == "complete"
    assert result["requested_units"] == result["observed_units"] == 2
    assert {call["symbol"] for call in source.calls} == {"SATS", "FI"}
    assert all(call["adjustment"] == "raw" and call["asof"] == date(2020, 1, 2) for call in source.calls)
    assert all(call["start"] == datetime(2020, 1, 2, 5, tzinfo=UTC) for call in source.calls)
    assert all(call["end_exclusive"] == datetime(2020, 1, 3, 5, tzinfo=UTC) for call in source.calls)
    assert owner_validator.call_count == 2  # Online admission and publication replay.
    for call in owner_validator.call_args_list:
        assert call.args == ()
        assert set(call.kwargs) == {"directory", "request", "manifest", "units", "parent_archive_loader"}
        assert call.kwargs["directory"] == plan
        assert call.kwargs["request"] == _json(plan / "_request.json")
        assert call.kwargs["manifest"] == _json(plan / "_manifest.json")
        assert call.kwargs["units"]["role"].eq("stock").all()
        assert set(call.kwargs["units"]["ticker"]) == {"ECHO", "FISV"}
        assert call.kwargs["parent_archive_loader"] is collector.load_complete_swing_history_collection
    source.calls.clear()
    assert collector.load_complete_swing_history_collection(
        output, plan_directory=plan, expected_adjustment="raw", expected_plan_authority_sha256=pin,
    ) == result
    assert owner_validator.call_count == 3
    assert source.calls == []
    request = _json(output / "_request.json")
    assert request["provider_symbols"] == PROVIDER_SYMBOLS
    assert request["transport_receipts_required"] is True
    for artifact in result["unit_artifacts"]:
        unit = _json(output / artifact["unit_manifest_path"])
        assert unit["provider_symbol"] == PROVIDER_SYMBOLS[unit["ticker"]]
        assert unit["pages"][0]["transport"]


@pytest.mark.parametrize("pin", [None, "0" * 64])
@pytest.mark.parametrize("operation", ["collect", "replay"])
def test_correction_requires_independent_pin_before_owner_or_output(
    tmp_path: Path, owner_validator: Mock, pin: str | None, operation: str,
) -> None:
    plan, _ = _correction_plan(tmp_path)
    output, source = tmp_path / "collection", _FakeSource()
    with pytest.raises(DataReadinessError, match="independent plan authority pin"):
        if operation == "collect":
            _collect(plan, output, source, pin)
        else:
            collector.load_complete_swing_history_collection(
                output, plan_directory=plan, expected_adjustment="raw", expected_plan_authority_sha256=pin,
            )
    owner_validator.assert_not_called()
    assert source.calls == []
    assert not output.exists()


@pytest.mark.parametrize("target", ["request", "manifest", "both"])
@pytest.mark.parametrize("scope", [None, "initial_fit_raw_share_acquisition", "generic_history_acquisition"])
def test_scope_downgrade_cannot_bypass_pin_or_benchmarks(
    tmp_path: Path, owner_validator: Mock, target: str, scope: str | None,
) -> None:
    plan, original_pin = _correction_plan(tmp_path)
    for name in ("request", "manifest") if target == "both" else (target,):
        payload = _json(plan / f"_{name}.json")
        payload["scope"] = scope
        _write_json(plan / f"_{name}.json", payload)
    replacement_pin = _resign(plan)
    source, output = _FakeSource(), tmp_path / "collection"
    for pin in (None, original_pin, replacement_pin):
        with pytest.raises(DataReadinessError):
            _collect(plan, output, source, pin)
    owner_validator.assert_not_called()
    assert not output.exists()
    assert source.calls == []


@pytest.mark.parametrize("field,value", [
    ("provider_symbols", {}), ("provider_symbols", {"ECHO": "SATS", "FISV": ""}),
    ("provider_symbols", {"ECHO": "SATS", "FISV": 123}),
    ("provider_symbols", {**PROVIDER_SYMBOLS, "SPY": "SPY"}),
    ("asof_policy", "current_symbol"),
])
def test_correction_provider_policy_poison_fails_before_owner(
    tmp_path: Path, owner_validator: Mock, field: str, value: Any,
) -> None:
    plan, _ = _correction_plan(tmp_path)
    request = _json(plan / "_request.json")
    request[field] = value
    _write_json(plan / "_request.json", request)
    pin = _resign(plan)
    source, output = _FakeSource(), tmp_path / "collection"
    with pytest.raises(DataReadinessError, match="provider identity policy"):
        _collect(plan, output, source, pin)
    owner_validator.assert_not_called()
    assert source.calls == []
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("adjustment", "all"), ("adjustment", "split"), ("source", "other"),
    ("price_feed", "iex"), ("timeframe", "1Min"),
])
def test_correction_raw_sip_plan_contract_cannot_be_weakened(
    tmp_path: Path, owner_validator: Mock, field: str, value: str,
) -> None:
    plan, _ = _correction_plan(tmp_path)
    manifest = _json(plan / "_manifest.json")
    manifest["daily_bars"][field] = value
    _write_json(plan / "_manifest.json", manifest)
    pin = _resign(plan)
    source, output = _FakeSource(), tmp_path / "collection"
    with pytest.raises(DataReadinessError):
        _collect(plan, output, source, pin)
    owner_validator.assert_not_called()
    assert source.calls == []
    assert not output.exists()


def test_runtime_provider_mapping_must_match_pinned_correction_plan(tmp_path: Path, owner_validator: Mock) -> None:
    plan, pin = _correction_plan(tmp_path)
    source, output = _FakeSource(), tmp_path / "collection"
    with pytest.raises(DataReadinessError, match="provider symbols differ"):
        collector.collect_swing_history_plan(
            plan_directory=plan, output_directory=output, source_factory=lambda: source,
            provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=pin,
        )
    owner_validator.assert_called_once()
    assert source.calls == []
    assert not output.exists()


@pytest.mark.parametrize("reason", ["parent archive replay failed", "inherited benchmark missing", "document pin differs"])
def test_owner_rejection_stops_dispatch_and_offline_replay(
    tmp_path: Path, owner_validator: Mock, reason: str,
) -> None:
    plan, pin = _correction_plan(tmp_path)
    output, source = tmp_path / "collection", _FakeSource()
    owner_validator.side_effect = DataReadinessError(reason)
    with pytest.raises(DataReadinessError, match=reason):
        _collect(plan, output, source, pin)
    assert not output.exists()
    assert source.calls == []
    owner_validator.side_effect = None
    _collect(plan, output, source, pin)
    owner_validator.side_effect = DataReadinessError(reason)
    with pytest.raises(DataReadinessError, match=reason):
        collector.load_complete_swing_history_collection(
            output, plan_directory=plan, expected_adjustment="raw", expected_plan_authority_sha256=pin,
        )


@pytest.mark.parametrize("fault", ["planned_units", "stock_units", "benchmark_units", "ranges", "outside_range"])
def test_owner_success_does_not_skip_common_counts_or_ranges(
    tmp_path: Path, owner_validator: Mock, fault: str,
) -> None:
    plan, _ = _correction_plan(tmp_path)
    manifest = _json(plan / "_manifest.json")
    if fault == "ranges":
        manifest["missing_session_ranges"] = []
    elif fault == "outside_range":
        manifest["missing_session_ranges"][0].update(first_session="2020-01-03", last_session="2020-01-03")
    else:
        manifest["daily_bars"][fault] += 1
    _write_json(plan / "_manifest.json", manifest)
    pin = _resign(plan)
    output, source = tmp_path / "collection", _FakeSource()
    with pytest.raises(DataReadinessError, match="counts|ranges|escapes"):
        _collect(plan, output, source, pin)
    owner_validator.assert_called_once()
    assert source.calls == []
    assert not output.exists()


@pytest.mark.parametrize("scope", [None, "initial_fit_raw_share_acquisition"])
@pytest.mark.parametrize("fault", ["missing_spy", "partial_benchmark_range"])
def test_ordinary_plans_still_require_local_complete_benchmarks(
    tmp_path: Path, owner_validator: Mock, scope: str | None, fault: str,
) -> None:
    plan = _plan(tmp_path, adjustment="raw")
    units_path = plan / "daily_bar_units.csv"
    units = pd.read_csv(units_path, dtype=str)
    manifest = _json(plan / "_manifest.json")
    request = _json(plan / "_request.json")
    if fault == "missing_spy":
        units = units.loc[units.ticker.ne("SPY")]
        units.to_csv(units_path, index=False, lineterminator="\n")
        manifest["daily_bars"].update(planned_units=21, benchmark_units=1)
    else:
        manifest["missing_session_ranges"][0]["first_session"] = "2020-01-01"
    if scope is not None:
        request.update(scope=scope, provider_symbols={ticker: ticker for ticker in units.ticker},
                       asof_policy="inclusive_unit_end_date_entity_mapping_not_ownership")
        manifest["scope"] = scope
    _write_json(plan / "_request.json", request)
    _write_json(plan / "_manifest.json", manifest)
    pin = _resign(plan)
    output, source = tmp_path / "collection", _FakeSource()
    with pytest.raises(DataReadinessError, match="benchmark"):
        collector.collect_swing_history_plan(
            plan_directory=plan, output_directory=output, source_factory=lambda: source,
            provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=pin,
        )
    owner_validator.assert_not_called()
    assert source.calls == []
    assert not output.exists()


@pytest.mark.parametrize("fault", ["symbol", "feed", "adjustment", "query_adjustment", "missing_receipt"])
def test_correction_response_poison_never_publishes_authority(tmp_path: Path, owner_validator: Mock, fault: str) -> None:
    plan, pin = _correction_plan(tmp_path)
    output = tmp_path / "collection"
    result = _collect(plan, output, _FakeSource(fault_symbol="SATS", fault=fault), pin)
    assert result["status"] == "incomplete"
    assert [unit["ticker"] for unit in result["failed_units"]] == ["ECHO"]
    assert not (output / "_authority.json").exists()
    owner_validator.assert_called_once()


def test_correction_rejects_wrong_transport_asof(tmp_path: Path, owner_validator: Mock) -> None:
    class WrongAsOfSource(_FakeSource):
        def fetch_daily_page(self, *args: Any, **kwargs: Any) -> collector.SwingDailyPage:
            page = super().fetch_daily_page(*args, **kwargs)
            receipt = page.transport_response
            assert receipt is not None
            wrong_url = receipt.requested_url.replace("asof=2020-01-02", "asof=2020-01-03")
            return replace(page, transport_response=replace(receipt, requested_url=wrong_url, final_url=wrong_url))

    plan, pin = _correction_plan(tmp_path)
    output = tmp_path / "collection"
    result = _collect(plan, output, WrongAsOfSource(), pin)
    assert result["status"] == "incomplete"
    assert len(result["failed_units"]) == 2
    assert all("transport" in row["error"] for row in result["failed_units"])
    assert not (output / "_authority.json").exists()
    owner_validator.assert_called_once()


def test_correction_resume_reuses_only_verified_units(tmp_path: Path, owner_validator: Mock) -> None:
    plan, pin = _correction_plan(tmp_path)
    output, source = tmp_path / "collection", _FakeSource()
    partial = _collect(plan, output, source, pin, maximum_units_this_run=1)
    assert partial["status"] == "incomplete"
    assert partial["terminal_units"] == 1
    complete = _collect(plan, output, source, pin)
    assert complete["status"] == "complete"
    assert complete["resumed_units"] == 1
    assert len(source.calls) == 2
    assert owner_validator.call_count == 3


@pytest.mark.parametrize("fault", ["provider_mapping", "raw_body", "consumer_adjustment"])
def test_correction_offline_replay_rejects_poison(tmp_path: Path, owner_validator: Mock, fault: str) -> None:
    plan, pin = _correction_plan(tmp_path)
    output = tmp_path / "collection"
    result = _collect(plan, output, _FakeSource(), pin)
    if fault == "provider_mapping":
        request = _json(output / "_request.json")
        request["provider_symbols"]["ECHO"] = "ECHO"
        _write_json(output / "_request.json", request)
    elif fault == "raw_body":
        unit = _json(output / result["unit_artifacts"][0]["unit_manifest_path"])
        path = output / unit["pages"][0]["transport"]["body_path"]
        path.write_bytes(path.read_bytes() + b"poison")
    with pytest.raises(DataReadinessError, match="provider symbols|transport|consumer adjustment"):
        collector.load_complete_swing_history_collection(
            output, plan_directory=plan, expected_adjustment="all" if fault == "consumer_adjustment" else "raw",
            expected_plan_authority_sha256=pin,
        )
    assert owner_validator.call_count == 3
