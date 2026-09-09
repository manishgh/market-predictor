from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any
from unittest.mock import Mock
from urllib.parse import urlencode

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.swing_history_collection import (
    COLLECTION_AUTHORITY_SCHEMA,
    AlpacaSwingDailyPageSource,
    SwingDailyPage,
    collect_swing_history_plan,
    load_complete_swing_history_collection,
)
from market_predictor.sources.alpaca import AlpacaSource, decode_bars_page_response
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.datasets.history_plan_publication import (
    AUTHORITY_SCHEMA as PLAN_AUTHORITY_SCHEMA,
)
from market_predictor.swing.datasets.history_plan_publication import PLAN_SCHEMA


@pytest.mark.parametrize("adjustment", ["all", "raw"])
def test_exact_plan_collection_publishes_verified_unit_authority(tmp_path: Path, adjustment: str) -> None:
    plan = _plan(tmp_path, adjustment=adjustment)
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
    assert request["adjustment"] == adjustment
    assert request["transport_receipts_required"] is True
    assert {call["adjustment"] for call in source.calls} == {adjustment}
    for unit in manifest["unit_artifacts"]:
        unit_manifest = _json(tmp_path / "collection" / unit["unit_manifest_path"])
        assert unit_manifest["adjustment"] == adjustment
        assert unit_manifest["pages"][0]["transport"]
        bars = pd.read_parquet(tmp_path / "collection" / unit_manifest["bars_path"])
        assert set(bars.adjustment) == {adjustment}
    authority = _json(tmp_path / "collection" / "_authority.json")
    assert authority["schema"] == COLLECTION_AUTHORITY_SCHEMA
    assert authority["artifact_sha256"] == file_sha256(tmp_path / "collection" / "_manifest.json")
    verified = load_complete_swing_history_collection(
        tmp_path / "collection",
        plan_directory=plan,
        expected_adjustment=adjustment,
    )
    assert verified == manifest
    with pytest.raises(DataReadinessError, match="consumer adjustment differs"):
        load_complete_swing_history_collection(
            tmp_path / "collection", plan_directory=plan,
            expected_adjustment="raw" if adjustment == "all" else "all",
        )


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
        load_complete_swing_history_collection(output, plan_directory=plan, expected_adjustment="all")


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


@pytest.mark.parametrize("adjustment", ["raw", "all"])
def test_transport_adapter_collects_and_replays_exact_basis(tmp_path: Path, adjustment: str) -> None:
    plan = _plan(tmp_path, adjustment=adjustment)
    output = tmp_path / "collection"
    source = AlpacaSource(Settings(ALPACA_API_KEY_ID="test-key", ALPACA_API_SECRET_KEY="test-secret"))
    client = Mock()

    def respond(url: str, *, params: dict[str, Any], **kwargs: Any) -> HttpByteResponse:
        assert url == "https://data.alpaca.markets/v2/stocks/bars"
        assert params == {
            "symbols": params["symbols"], "timeframe": "1Day",
            "start": "2020-01-02T05:00:00+00:00", "end": "2020-01-03T04:59:59.999999+00:00",
            "feed": "sip", "limit": 10_000, "adjustment": adjustment, "sort": "asc", "asof": "2020-01-02",
        }
        assert kwargs["allow_redirects"] is False
        payload = {"bars": {params["symbols"]: [
            {"t": "2020-01-02T05:00:00Z", "o": 10.0, "h": 12.0, "l": 9.0, "c": 11.0, "v": 1000},
        ]}, "next_page_token": None}
        return _test_transport_response(payload, params)

    client.get_bytes_with_metadata.side_effect = respond
    source.client = client
    manifest = collect_swing_history_plan(
        plan_directory=plan, output_directory=output,
        source_factory=lambda: AlpacaSwingDailyPageSource(source), provider_symbol_for=lambda ticker: ticker,
    )
    assert manifest["status"] == "complete"
    assert client.get_bytes_with_metadata.call_count == 22
    client.reset_mock()
    assert load_complete_swing_history_collection(
        output, plan_directory=plan, expected_adjustment=adjustment,
    ) == manifest
    client.get_bytes_with_metadata.assert_not_called()


@pytest.mark.parametrize("adjustment", ["raw", "all"])
def test_cross_basis_resume_fails_without_source_calls(tmp_path: Path, adjustment: str) -> None:
    plan, output, _ = _partial_unit(tmp_path, adjustment=adjustment)
    manifest = _json(plan / "_manifest.json")
    manifest["daily_bars"]["adjustment"] = "all" if adjustment == "raw" else "raw"
    _write_json(plan / "_manifest.json", manifest)
    _reauthorize_plan(plan)
    source = _FakeSource()
    with pytest.raises(DataReadinessError, match="request drifted"):
        collect_swing_history_plan(
            plan_directory=plan, output_directory=output,
            source_factory=lambda: source, provider_symbol_for=lambda ticker: ticker,
        )
    assert source.calls == []


@pytest.mark.parametrize("adjustment", ["raw", "all"])
@pytest.mark.parametrize("fault", ["query_adjustment", "missing_receipt"])
def test_new_acquisition_requires_matching_original_receipt(
    tmp_path: Path, adjustment: str, fault: str,
) -> None:
    plan = _plan(tmp_path, adjustment=adjustment)
    source = _FakeSource(fault_symbol="QQQ", fault=fault)
    page = source.fetch_daily_page(
        "QQQ", datetime(2020, 1, 2, 5, tzinfo=UTC), datetime(2020, 1, 3, 5, tzinfo=UTC),
        page_token=None, asof=date(2020, 1, 2), adjustment=adjustment,
    )
    assert page.response_adjustment == adjustment
    source.calls.clear()
    output = tmp_path / "collection"
    status = collect_swing_history_plan(
        plan_directory=plan, output_directory=output,
        source_factory=lambda: source, provider_symbol_for=lambda ticker: ticker, maximum_units_this_run=2,
    )
    assert status["status"] == "incomplete"
    assert [row["ticker"] for row in status["failed_units"]] == ["QQQ"]
    assert "transport" in status["failed_units"][0]["error"]
    assert status["observed_units"] == 1
    assert not (output / "_authority.json").exists()
    if fault == "query_adjustment":
        failed_unit = output / "units" / status["failed_units"][0]["unit_id"]
        attempts = list((failed_unit / "attempts").glob("*.json"))
        assert len(attempts) == 1
        attempt = _json(attempts[0])
        assert attempt["status"] == "failed"
        assert len(attempt["pages"]) == 1
        receipt = attempt["pages"][0]["transport"]
        metadata = receipt["metadata"]
        original = page.transport_response
        assert original is not None
        assert metadata["requested_url"] == metadata["final_url"] == original.requested_url
        body_path = output / receipt["body_path"]
        assert body_path.is_file()
        assert body_path.is_relative_to(failed_unit / "raw" / attempt["attempt_id"])
        assert body_path.read_bytes() == original.body
        assert file_sha256(body_path) == metadata["sha256"] == original.sha256
        assert body_path.stat().st_size == metadata["body_length"] == original.body_length
        response = HttpByteResponse(**{
            **metadata, "body": body_path.read_bytes(),
            "retrieved_at_utc": datetime.fromisoformat(metadata["retrieved_at_utc"]),
            "redirect_chain": tuple(metadata["redirect_chain"]),
            "safe_headers": tuple(tuple(row) for row in metadata["safe_headers"]),
        })
        expected_params = {
            "symbols": "QQQ", "timeframe": "1Day", "start": "2020-01-02T05:00:00+00:00",
            "end": "2020-01-03T04:59:59.999999+00:00", "feed": "sip", "limit": 10_000,
            "adjustment": adjustment, "sort": "asc", "asof": "2020-01-02",
        }
        calls_before_replay = list(source.calls)
        with pytest.raises(RuntimeError, match="query"):
            decode_bars_page_response(response, expected_params=expected_params)
        decoded = decode_bars_page_response(response, expected_params={
            **expected_params, "adjustment": "all" if adjustment == "raw" else "raw",
        })
        assert decoded.raw_payload == page.raw_payload
        assert source.calls == calls_before_replay

        siblings = list((output / "units").glob("*/_manifest.json"))
        assert len(siblings) == 1
        assert _json(siblings[0])["ticker"] == "SPY"
        sibling_hash = file_sha256(siblings[0])
        attempt_hash = file_sha256(attempts[0])
        retry = _FakeSource()
        resumed = collect_swing_history_plan(
            plan_directory=plan, output_directory=output,
            source_factory=lambda: retry, provider_symbol_for=lambda ticker: ticker, maximum_units_this_run=1,
        )
        assert resumed["resumed_units"] == 1
        assert resumed["terminal_units"] == resumed["observed_units"] == 2
        assert resumed["failed_units"] == []
        assert [call["symbol"] for call in retry.calls] == ["QQQ"]
        assert file_sha256(siblings[0]) == sibling_hash
        assert file_sha256(attempts[0]) == attempt_hash
        assert file_sha256(body_path) == original.sha256


@pytest.mark.parametrize("adjustment", ["raw", "all"])
@pytest.mark.parametrize("location", ["manifest", "parquet"])
def test_wrong_basis_unit_fails_resume_even_with_rehashed_local_manifest(
    tmp_path: Path, adjustment: str, location: str,
) -> None:
    plan, output, path = _partial_unit(tmp_path, adjustment=adjustment)
    manifest = _json(path)
    wrong = "all" if adjustment == "raw" else "raw"
    if location == "manifest":
        manifest["adjustment"] = wrong
    else:
        bars_path = output / manifest["bars_path"]
        bars = pd.read_parquet(bars_path)
        bars["adjustment"] = wrong
        bars.to_parquet(bars_path, index=False)
        manifest["bars_sha256"] = file_sha256(bars_path)
        manifest["bars_bytes"] = bars_path.stat().st_size
    _write_json(path, manifest)
    source = _FakeSource()
    with pytest.raises(DataReadinessError, match="resume identity differs|adjustment differs"):
        collect_swing_history_plan(
            plan_directory=plan, output_directory=output,
            source_factory=lambda: source, provider_symbol_for=lambda ticker: ticker,
        )
    assert source.calls == []


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_rehashed_parquet_value_must_equal_original_response(tmp_path: Path, column: str) -> None:
    plan, output, path = _partial_unit(tmp_path, adjustment="raw")
    manifest = _json(path)
    bars_path = output / manifest["bars_path"]
    original = file_sha256(output / manifest["pages"][0]["transport"]["body_path"])
    bars = pd.read_parquet(bars_path)
    bars.loc[0, column] += 0.25 if column != "volume" else 1
    bars.to_parquet(bars_path, index=False)
    manifest["bars_sha256"] = file_sha256(bars_path)
    manifest["bars_bytes"] = bars_path.stat().st_size
    _write_json(path, manifest)
    source = _FakeSource()
    with pytest.raises(DataReadinessError, match="Parquet differs from response replay"):
        collect_swing_history_plan(
            plan_directory=plan, output_directory=output,
            source_factory=lambda: source, provider_symbol_for=lambda ticker: ticker,
        )
    assert source.calls == []
    assert file_sha256(output / manifest["pages"][0]["transport"]["body_path"]) == original


@pytest.mark.parametrize("adjustment", ["raw", "all"])
@pytest.mark.parametrize("missing", ["receipt", "body"])
def test_missing_transport_evidence_fails_resume(tmp_path: Path, adjustment: str, missing: str) -> None:
    plan, output, path = _partial_unit(tmp_path, adjustment=adjustment)
    manifest = _json(path)
    if missing == "receipt":
        del manifest["pages"][0]["transport"]
    else:
        (output / manifest["pages"][0]["transport"]["body_path"]).unlink()
    _write_json(path, manifest)
    source = _FakeSource()
    with pytest.raises(DataReadinessError, match="original transport receipt|transport body missing"):
        collect_swing_history_plan(
            plan_directory=plan, output_directory=output,
            source_factory=lambda: source, provider_symbol_for=lambda ticker: ticker,
        )
    assert source.calls == []


def _partial_unit(tmp_path: Path, *, adjustment: str) -> tuple[Path, Path, Path]:
    plan = _plan(tmp_path, adjustment=adjustment)
    output = tmp_path / "collection"
    result = collect_swing_history_plan(
        plan_directory=plan, output_directory=output,
        source_factory=_FakeSource, provider_symbol_for=lambda ticker: ticker, maximum_units_this_run=1,
    )
    assert result["terminal_units"] == result["observed_units"] == 1
    paths = list((output / "units").glob("*/_manifest.json"))
    assert len(paths) == 1
    return plan, output, paths[0]


def _test_transport_response(payload: dict[str, Any], params: dict[str, Any]) -> HttpByteResponse:
    """Synthetic test evidence only; never a provider observation."""
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    url = "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(params)
    return HttpByteResponse(
        body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=200,
        retrieved_at_utc=datetime(2020, 1, 3, 6, tzinfo=UTC), content_type="application/json",
        content_encoding=None, etag=None, last_modified=None, body_length=len(body),
        sha256=sha256(body).hexdigest(), body_representation="http_entity_encoded", safe_headers=(),
    )


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
        adjustment: str,
    ) -> SwingDailyPage:
        self.calls.append(
            {
                "symbol": symbol,
                "start": start,
                "end_exclusive": end_exclusive,
                "page_token": page_token,
                "asof": asof,
                "adjustment": adjustment,
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
            response_adjustment=adjustment,
            bars=bars,
            response_headers={},
            raw_payload={"bars": {symbol: list(bars)}},
        )
        fault = self.fault if symbol == self.fault_symbol else None
        if fault == "symbol":
            page = replace(page, response_symbol="WRONG")
        elif fault == "timeframe":
            page = replace(page, response_timeframe="1Min")
        elif fault == "feed":
            page = replace(page, response_feed="iex")
        elif fault == "adjustment":
            page = replace(page, response_adjustment="raw" if adjustment == "all" else "all")
        elif fault in {"outside_date", "timestamp_alignment", "ohlcv", "missing_field"}:
            poisoned = dict(page.bars[0])
            if fault == "outside_date":
                poisoned["t"] = "2020-01-03T05:00:00Z"
            elif fault == "timestamp_alignment":
                poisoned["t"] = "2020-01-02T12:00:00Z"
            elif fault == "ohlcv":
                poisoned["h"] = 8.0
            else:
                del poisoned["v"]
            page = replace(page, bars=(poisoned,))
        payload = {"bars": {symbol: list(page.bars)}, "next_page_token": page.next_page_token}
        params = {
            "symbols": symbol, "timeframe": "1Day", "start": start.isoformat(),
            "end": (end_exclusive - timedelta(microseconds=1)).isoformat(),
            "feed": "sip", "limit": 10_000, "adjustment": adjustment,
            "sort": "asc", "asof": asof.isoformat(),
        }
        if page_token is not None:
            params["page_token"] = page_token
        if fault == "query_adjustment":
            params["adjustment"] = "raw" if adjustment == "all" else "all"
        return replace(
            page, raw_payload=payload,
            transport_response=None if fault == "missing_receipt" else _test_transport_response(payload, params),
        )


def _plan(tmp_path: Path, *, adjustment: str = "all") -> Path:
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
            "adjustment": adjustment,
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
