from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.swing_history_acquisition import (
    AUTHORITY_SCHEMA as PLAN_AUTHORITY_SCHEMA,
)
from market_predictor.edge_rebuild.swing_history_acquisition import PLAN_SCHEMA
from market_predictor.edge_rebuild.swing_history_collection import (
    COLLECTION_AUTHORITY_SCHEMA,
    SwingDailyPage,
    collect_swing_history_plan,
    load_complete_swing_history_collection,
)
from market_predictor.v3.errors import DataReadinessError


def test_exact_plan_collection_publishes_verified_unit_authority(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeSource()

    manifest = collect_swing_history_plan(
        plan_directory=plan,
        output_directory=tmp_path / "collection",
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
    )

    assert manifest["status"] == "complete"
    assert manifest["requested_units"] == 22
    assert manifest["observed_units"] == 22
    assert manifest["failed_units"] == []
    assert manifest["unavailable_units"] == []
    assert manifest["unattempted_units"] == []
    assert len(source.calls) == 22
    assert {"AAA", "QQQ", "SPY"}.issubset({call["symbol"] for call in source.calls})
    assert all(call["start"] == datetime(2020, 1, 2, 5, tzinfo=UTC) for call in source.calls)
    assert all(call["end_exclusive"] == datetime(2020, 1, 3, 5, tzinfo=UTC) for call in source.calls)
    assert all(call["asof"] == date(2020, 1, 2) for call in source.calls)
    request = _json(tmp_path / "collection" / "_request.json")
    assert request["workers"] == 2
    assert request["price_feed"] == "sip"
    assert request["adjustment"] == "all"
    authority = _json(tmp_path / "collection" / "_authority.json")
    assert authority["schema"] == COLLECTION_AUTHORITY_SCHEMA
    assert authority["artifact_sha256"] == file_sha256(tmp_path / "collection" / "_manifest.json")
    verified = load_complete_swing_history_collection(
        tmp_path / "collection",
        plan_directory=plan,
    )
    assert verified == manifest


def test_operational_limit_resumes_only_missing_units(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeSource()
    output = tmp_path / "collection"

    partial = collect_swing_history_plan(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
        maximum_units_this_run=1,
    )

    assert partial["status"] == "incomplete"
    assert partial["terminal_units"] == 1
    assert len(partial["unattempted_units"]) == 21
    assert not (output / "_authority.json").exists()
    completed = collect_swing_history_plan(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
    )
    assert completed["status"] == "complete"
    assert completed["resumed_units"] == 1
    assert len(source.calls) == 22


def test_reauthorized_plan_drift_is_refused_on_resume(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeSource()
    output = tmp_path / "collection"
    collect_swing_history_plan(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
        maximum_units_this_run=1,
    )
    units_path = plan / "daily_bar_units.csv"
    units = pd.read_csv(units_path, dtype=str)
    units.loc[units["ticker"].eq("AAA"), "security_id"] = "security:DRIFT"
    units.to_csv(units_path, index=False, lineterminator="\n")
    _reauthorize_plan(plan)

    with pytest.raises(DataReadinessError, match="request drifted"):
        collect_swing_history_plan(
            plan_directory=plan,
            output_directory=output,
            source_factory=lambda: source,
            provider_symbol_for=lambda ticker: ticker,
        )


@pytest.mark.parametrize(
    "fault",
    [
        "symbol",
        "timeframe",
        "feed",
        "adjustment",
        "outside_date",
        "timestamp_alignment",
        "ohlcv",
        "missing_field",
    ],
)
def test_response_contract_or_bar_poison_prevents_authority(
    tmp_path: Path,
    fault: str,
) -> None:
    plan = _plan(tmp_path)
    source = _FakeSource(fault_symbol="AAA", fault=fault)
    output = tmp_path / "collection"

    status = collect_swing_history_plan(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
        maximum_units_this_run=3,
    )

    assert status["status"] == "incomplete"
    assert [record["ticker"] for record in status["failed_units"]] == ["AAA"]
    assert status["stop_reason"] == "non_allowed_failure"
    assert not (output / "_manifest.json").exists()
    assert not (output / "_authority.json").exists()


def test_empty_stock_is_explicit_allowed_unavailable(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeSource(empty_symbols={"AAA"})

    manifest = collect_swing_history_plan(
        plan_directory=plan,
        output_directory=tmp_path / "collection",
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
    )

    assert manifest["status"] == "complete_with_unavailable"
    assert manifest["unavailable_units"][0]["ticker"] == "AAA"
    assert manifest["unavailable_units"][0]["allowed"] is True
    assert _json(tmp_path / "collection" / "_status.json")["status"] == "complete_with_unavailable"
    assert (tmp_path / "collection" / "_authority.json").is_file()


def test_empty_benchmark_is_non_allowed_and_prevents_authority(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeSource(empty_symbols={"SPY"})
    output = tmp_path / "collection"

    status = collect_swing_history_plan(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
        maximum_units_this_run=2,
    )

    assert status["status"] == "incomplete"
    assert [record["ticker"] for record in status["failed_units"]] == ["SPY"]
    assert not (output / "_authority.json").exists()


def test_stock_unavailability_above_five_percent_prevents_authority(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    source = _FakeSource(empty_symbols={"AAA", "T000"})

    status = collect_swing_history_plan(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: source,
        provider_symbol_for=lambda ticker: ticker,
    )

    assert status["status"] == "incomplete"
    assert status["unavailable_security_fraction"] == 0.1
    assert {record["allowed"] for record in status["unavailable_units"]} == {False}
    assert status["stop_reason"] == "non_allowed_failure"
    assert not (output / "_authority.json").exists()


def test_raw_unit_poison_invalidates_complete_collection(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    output = tmp_path / "collection"
    collect_swing_history_plan(
        plan_directory=plan,
        output_directory=output,
        source_factory=lambda: _FakeSource(),
        provider_symbol_for=lambda ticker: ticker,
    )
    manifest = _json(output / "_manifest.json")
    first_unit = manifest["unit_artifacts"][0]
    unit_manifest = _json(output / first_unit["unit_manifest_path"])
    raw_path = output / unit_manifest["pages"][0]["raw_path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"poison")

    with pytest.raises(DataReadinessError, match="raw page does not verify"):
        load_complete_swing_history_collection(output, plan_directory=plan)


def test_plan_unit_hash_poison_fails_before_source_call(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    source = _FakeSource()
    units_path = plan / "daily_bar_units.csv"
    units_path.write_text(units_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="plan authority is invalid"):
        collect_swing_history_plan(
            plan_directory=plan,
            output_directory=tmp_path / "collection",
            source_factory=lambda: source,
            provider_symbol_for=lambda ticker: ticker,
        )
    assert source.calls == []


class _FakeSource:
    def __init__(
        self,
        *,
        empty_symbols: set[str] | None = None,
        fault_symbol: str | None = None,
        fault: str | None = None,
    ) -> None:
        self.empty_symbols = empty_symbols or set()
        self.fault_symbol = fault_symbol
        self.fault = fault
        self.calls: list[dict[str, Any]] = []

    def fetch_daily_page(
        self,
        symbol: str,
        start: datetime,
        end_exclusive: datetime,
        *,
        page_token: str | None,
        asof: date,
    ) -> SwingDailyPage:
        self.calls.append(
            {
                "symbol": symbol,
                "start": start,
                "end_exclusive": end_exclusive,
                "page_token": page_token,
                "asof": asof,
            }
        )
        bars: tuple[dict[str, Any], ...] = ()
        if symbol not in self.empty_symbols:
            bars = (
                {
                    "t": "2020-01-02T05:00:00Z",
                    "o": 10.0,
                    "h": 12.0,
                    "l": 9.0,
                    "c": 11.0,
                    "v": 1000,
                },
            )
        page = SwingDailyPage(
            request_page_token=page_token,
            next_page_token=None,
            response_symbol=symbol,
            response_timeframe="1Day",
            response_feed="sip",
            response_adjustment="all",
            bars=bars,
            response_headers={},
            raw_payload={"bars": {symbol: list(bars)}},
        )
        if symbol != self.fault_symbol:
            return page
        if self.fault == "symbol":
            return replace(page, response_symbol="WRONG")
        if self.fault == "timeframe":
            return replace(page, response_timeframe="1Min")
        if self.fault == "feed":
            return replace(page, response_feed="iex")
        if self.fault == "adjustment":
            return replace(page, response_adjustment="raw")
        poisoned = dict(page.bars[0])
        if self.fault == "outside_date":
            poisoned["t"] = "2020-01-03T05:00:00Z"
        elif self.fault == "timestamp_alignment":
            poisoned["t"] = "2020-01-02T12:00:00Z"
        elif self.fault == "ohlcv":
            poisoned["h"] = 8.0
        elif self.fault == "missing_field":
            del poisoned["v"]
        return replace(page, bars=(poisoned,))


def _plan(tmp_path: Path) -> Path:
    directory = tmp_path / "plan"
    directory.mkdir()
    request = {
        "schema": PLAN_SCHEMA,
        "temporal_manifest_sha256": "1" * 64,
        "temporal_authority_sha256": "2" * 64,
        "membership_authority": {
            "authority_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "request_sha256": "5" * 64,
            "membership_artifact_sha256": "6" * 64,
            "universe_sha256": "7" * 64,
            "parent_lineage": {"raw_authority_sha256": "8" * 64},
        },
    }
    _write_json(directory / "_request.json", request)
    stock_units = [
        {
            "security_id": f"security:{ticker}",
            "ticker": ticker,
            "start_date": "2020-01-02",
            "end_date": "2020-01-02",
            "role": "stock",
        }
        for ticker in ["AAA", *(f"T{number:03d}" for number in range(19))]
    ]
    units = pd.DataFrame(
        [
            *stock_units,
            {
                "security_id": "benchmark:QQQ",
                "ticker": "QQQ",
                "start_date": "2020-01-02",
                "end_date": "2020-01-02",
                "role": "benchmark",
            },
            {
                "security_id": "benchmark:SPY",
                "ticker": "SPY",
                "start_date": "2020-01-02",
                "end_date": "2020-01-02",
                "role": "benchmark",
            },
        ]
    )
    units.to_csv(directory / "daily_bar_units.csv", index=False, lineterminator="\n")
    manifest = {
        "schema": PLAN_SCHEMA,
        "status": "ready_for_daily_history_collection",
        "outcomes_read": False,
        "request_sha256": file_sha256(directory / "_request.json"),
        "missing_session_ranges": [
            {
                "first_session": "2020-01-02",
                "last_session": "2020-01-02",
                "sessions": 1,
            }
        ],
        "membership": {
            "universe_sha256": "7" * 64,
            "parent_lineage": {"raw_authority_sha256": "8" * 64},
        },
        "daily_bars": {
            "status": "ready",
            "planned_units": 22,
            "stock_units": 20,
            "benchmark_units": 2,
            "source": "alpaca",
            "timeframe": "1Day",
            "price_feed": "sip",
            "adjustment": "all",
            "units_artifact": {
                "path": "daily_bar_units.csv",
                "bytes": (directory / "daily_bar_units.csv").stat().st_size,
                "sha256": file_sha256(directory / "daily_bar_units.csv"),
            },
        },
    }
    _write_json(directory / "_manifest.json", manifest)
    _write_json(
        directory / "_authority.json",
        {
            "schema": PLAN_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(directory / "_manifest.json"),
            "request_sha256": file_sha256(directory / "_request.json"),
            "units_sha256": file_sha256(directory / "daily_bar_units.csv"),
            "universe_sha256": "7" * 64,
        },
    )
    return directory


def _reauthorize_plan(directory: Path) -> None:
    manifest = _json(directory / "_manifest.json")
    units_path = directory / "daily_bar_units.csv"
    manifest["daily_bars"]["units_artifact"] = {
        "path": "daily_bar_units.csv",
        "bytes": units_path.stat().st_size,
        "sha256": file_sha256(units_path),
    }
    _write_json(directory / "_manifest.json", manifest)
    authority = _json(directory / "_authority.json")
    authority["artifact_sha256"] = file_sha256(directory / "_manifest.json")
    authority["units_sha256"] = file_sha256(units_path)
    _write_json(directory / "_authority.json", authority)


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
