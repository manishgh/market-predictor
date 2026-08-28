from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.history_collection import (
    collect_intraday_history,
    load_complete_intraday_history_collection,
)
from market_predictor.intraday.contracts.history_collection import (
    INTRADAY_HISTORY_PLAN_SCHEMA,
    load_intraday_history_config,
)
from market_predictor.intraday.datasets.history import (
    PLAN_AUTHORITY_SCHEMA,
)
from market_predictor.sources.alpaca import AlpacaBarsPage

POLICY_PATH = Path("configs/edge_rebuild_intraday_history.toml")


def test_collector_publishes_raw_lineage_and_complete_authority(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path / "plan")
    output = tmp_path / "collection"
    source = _FakeAlpacaSource()

    result = collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=lambda: source,
    )
    verified = load_complete_intraday_history_collection(output)

    assert result["status"] == "transport_complete"
    assert result["completed_units"] == 1
    assert result["total_rows"] == 4
    assert verified["request_sha256"] == result["request_sha256"]
    artifact = result["artifacts"][0]
    bars = pd.read_parquet(output / artifact["path"])
    assert set(bars["ticker"]) == {"AAA", "BBB"}
    assert set(bars["timeframe"]) == {"5m"}
    assert set(bars["price_feed"]) == {"sip"}
    assert set(bars["adjustment"]) == {"all"}
    assert bars["bar_start_utc"].dt.minute.tolist() == [30, 30, 35, 35]
    raw_page = output / artifact["pages"][0]["raw_page_path"]
    assert artifact["pages"][0]["raw_page_sha256"] == file_sha256(raw_page)
    payload = json.loads(raw_page.read_bytes())
    assert set(payload["bars"]) == {"AAA", "BBB"}
    assert artifact["pages"][0]["body_representation"] == "http_entity_encoded"
    assert artifact["pages"][0]["status_code"] == 200
    assert artifact["pages"][0]["final_url"] == artifact["pages"][0]["requested_url"]
    assert set(artifact["symbol_coverage"]) == {"AAA", "BBB"}
    assert source.timeframes == ["5Min"]


def test_collector_resumes_verified_unit_without_network(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path / "plan")
    output = tmp_path / "collection"
    config = load_intraday_history_config(POLICY_PATH)
    first = collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY_PATH,
        output_directory=output,
        config=config,
        source_factory=_FakeAlpacaSource,
    )
    (output / "_authority.json").unlink()
    (output / "_manifest.json").unlink()

    def unexpected_source() -> _FakeAlpacaSource:
        raise AssertionError("verified resume must not call Alpaca")

    resumed = collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY_PATH,
        output_directory=output,
        config=config,
        source_factory=unexpected_source,
    )

    assert resumed["resumed_units"] == 1
    assert resumed["total_rows"] == first["total_rows"]


def test_collector_operational_batch_limit_resumes_same_identity(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path / "plan", unit_count=2)
    output = tmp_path / "collection"
    config = load_intraday_history_config(POLICY_PATH)
    first = collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY_PATH,
        output_directory=output,
        config=config,
        source_factory=_FakeAlpacaSource,
        maximum_units_this_run=1,
    )

    assert first["status"] == "transport_incomplete"
    assert first["stop_reason"] == "operational_batch_limit"
    assert first["completed_units"] == 1
    assert first["unattempted_units"] == 1

    second = collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY_PATH,
        output_directory=output,
        config=config,
        source_factory=_FakeAlpacaSource,
    )

    assert second["status"] == "transport_complete"
    assert second["resumed_units"] == 1
    assert second["completed_units"] == 2


def test_collector_fails_closed_on_repeated_page_token(
    tmp_path: Path,
) -> None:
    result = collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=tmp_path / "collection",
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_RepeatingTokenSource,
    )

    assert result["status"] == "transport_incomplete"
    assert result["completed_units"] == 0
    error = next(iter(result["failed_units"].values()))
    assert "repeated a page token" in error
    assert not (tmp_path / "collection" / "_authority.json").exists()


def test_collector_failure_circuit_leaves_units_unattempted(
    tmp_path: Path,
) -> None:
    plan = _write_plan(tmp_path / "plan", unit_count=8)
    config = load_intraday_history_config(POLICY_PATH)

    result = collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY_PATH,
        output_directory=tmp_path / "collection",
        config=config,
        source_factory=_AlwaysFailingSource,
    )

    assert result["status"] == "transport_incomplete"
    assert len(result["failed_units"]) == 5
    assert result["unattempted_units"] == 3


def test_collector_rejects_mutated_resume_unit(tmp_path: Path) -> None:
    plan = _write_plan(tmp_path / "plan")
    output = tmp_path / "collection"
    config = load_intraday_history_config(POLICY_PATH)
    result = collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY_PATH,
        output_directory=output,
        config=config,
        source_factory=_FakeAlpacaSource,
    )
    artifact = output / result["artifacts"][0]["path"]
    bars = pd.read_parquet(artifact)
    bars.loc[0, "close"] = 999.0
    bars.to_parquet(artifact, index=False)
    (output / "_authority.json").unlink()
    (output / "_manifest.json").unlink()

    with pytest.raises(DataReadinessError, match="integrity failed"):
        collect_intraday_history(
            plan_directory=plan,
            policy_path=POLICY_PATH,
            output_directory=output,
            config=config,
            source_factory=_FakeAlpacaSource,
        )


def test_complete_authority_rejects_missing_raw_provider_page(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    result = collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_FakeAlpacaSource,
    )
    raw_page = output / result["artifacts"][0]["pages"][0]["raw_page_path"]
    raw_page.unlink()

    with pytest.raises(DataReadinessError, match="raw provider page"):
        load_complete_intraday_history_collection(output)


def test_complete_authority_rejects_changed_exact_body(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    result = collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_FakeAlpacaSource,
    )
    raw_page = output / result["artifacts"][0]["pages"][0]["raw_page_path"]
    raw_page.write_bytes(raw_page.read_bytes() + b" ")

    with pytest.raises(DataReadinessError, match="raw provider page"):
        load_complete_intraday_history_collection(output)


def test_complete_authority_rejects_changed_transport_sidecar(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    result = collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_FakeAlpacaSource,
    )
    sidecar = output / result["artifacts"][0]["pages"][0]["raw_sidecar_path"]
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["status_code"] = 206
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="sidecar changed"):
        load_complete_intraday_history_collection(output)


def test_complete_authority_rejects_extra_raw_inventory_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_FakeAlpacaSource,
    )
    extra = output / "raw_pages" / "unexpected.body"
    extra.write_bytes(b"{}")

    with pytest.raises(DataReadinessError, match="inventory changed"):
        load_complete_intraday_history_collection(output)


def test_complete_authority_rejects_canonical_bar_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    result = collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_FakeAlpacaSource,
    )
    artifact = result["artifacts"][0]
    bars_path = output / artifact["path"]
    bars = pd.read_parquet(bars_path)
    bars.loc[0, "close"] = 999.0
    bars.to_parquet(bars_path, index=False)
    artifact["sha256"] = file_sha256(bars_path)
    bars_path.with_suffix(".manifest.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    manifest_path = output / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0] = artifact
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = output / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="do not replay"):
        load_complete_intraday_history_collection(output)


def test_complete_authority_rejects_canonical_availability_mutation(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    result = collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_FakeAlpacaSource,
    )
    artifact = result["artifacts"][0]
    bars_path = output / artifact["path"]
    bars = pd.read_parquet(bars_path)
    bars.loc[0, "available_at_utc"] += pd.Timedelta(minutes=1)
    bars.to_parquet(bars_path, index=False)
    _resign_collection_artifact(output, artifact, bars_path)

    with pytest.raises(DataReadinessError, match="do not replay"):
        load_complete_intraday_history_collection(output)


def test_complete_authority_rejects_mixed_schema_generations(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    result = collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_FakeAlpacaSource,
    )
    artifact = result["artifacts"][0]
    artifact["schema"] = "edge_rebuild.intraday_history_unit.v1"
    bars_path = output / artifact["path"]
    _resign_collection_artifact(output, artifact, bars_path)

    with pytest.raises(DataReadinessError, match="mixes authority schema"):
        load_complete_intraday_history_collection(output)


def test_complete_authority_rejects_non_alpaca_request_endpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "collection"
    collect_intraday_history(
        plan_directory=_write_plan(tmp_path / "plan"),
        policy_path=POLICY_PATH,
        output_directory=output,
        config=load_intraday_history_config(POLICY_PATH),
        source_factory=_WrongEndpointSource,
    )

    with pytest.raises(DataReadinessError, match="endpoint changed"):
        load_complete_intraday_history_collection(output)


def _resign_collection_artifact(
    output: Path,
    artifact: dict[str, object],
    bars_path: Path,
) -> None:
    artifact["sha256"] = file_sha256(bars_path)
    bars_path.with_suffix(".manifest.json").write_text(
        json.dumps(artifact), encoding="utf-8"
    )
    manifest_path = output / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][0] = artifact
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = output / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")


class _FakeAlpacaSource:
    def __init__(self) -> None:
        self.settings = SimpleNamespace(alpaca_stock_feed="sip")
        self.client = SimpleNamespace(timeout=30)
        self.timeframes: list[str] = []

    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        self.timeframes.append(str(kwargs["timeframe"]))
        timestamps = [
            pd.Timestamp(start),
            pd.Timestamp(start) + pd.Timedelta(minutes=5),
        ]
        bars = {
            symbol: tuple(
                {
                    "t": timestamp.isoformat(),
                    "o": 100.0,
                    "h": 101.0,
                    "l": 99.0,
                    "c": 100.5,
                    "v": 1000,
                }
                for timestamp in timestamps
            )
            for symbol in symbols
        }
        next_page_token = None
        payload = {
            "bars": {symbol: list(values) for symbol, values in bars.items()},
            "next_page_token": next_page_token,
        }
        query = {
            "symbols": ",".join(symbols),
            "timeframe": str(kwargs["timeframe"]),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": "sip",
            "limit": str(kwargs["limit"]),
            "adjustment": "all",
            "sort": "asc",
            "asof": kwargs["asof"].isoformat(),
        }
        if kwargs.get("page_token") is not None:
            query["page_token"] = str(kwargs["page_token"])
        requested_url = "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(query)
        return AlpacaBarsPage(
            request_page_token=kwargs.get("page_token"),
            next_page_token=next_page_token,
            bars=bars,
            response_headers={
                "Content-Type": "application/json",
                "X-RateLimit-Remaining": "100",
            },
            raw_payload=payload,
            raw_body=json.dumps(payload, separators=(",", ":")).encode(),
            requested_url=requested_url,
            status_code=200,
            retrieved_at_utc=datetime.now(UTC),
            final_url=requested_url,
        )


class _RepeatingTokenSource(_FakeAlpacaSource):
    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        page = super().fetch_bars_page(
            symbols,
            start,
            end,
            **kwargs,
        )
        bars = (
            {symbol: () for symbol in symbols}
            if kwargs.get("page_token") is not None
            else page.bars
        )
        payload = {
            "bars": {symbol: list(values) for symbol, values in bars.items()},
            "next_page_token": "repeat",
        }
        return replace(
            page,
            next_page_token="repeat",
            bars=bars,
            raw_payload=payload,
            raw_body=json.dumps(payload, separators=(",", ":")).encode(),
        )


class _WrongEndpointSource(_FakeAlpacaSource):
    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        page = super().fetch_bars_page(symbols, start, end, **kwargs)
        assert page.requested_url is not None
        wrong = page.requested_url.replace(
            "https://data.alpaca.markets",
            "https://example.invalid",
        )
        return replace(page, requested_url=wrong, final_url=wrong)


class _AlwaysFailingSource(_FakeAlpacaSource):
    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        del symbols, start, end, kwargs
        raise ConnectionError("provider unavailable")


def _write_plan(path: Path, *, unit_count: int = 1) -> Path:
    config = load_intraday_history_config(POLICY_PATH)
    units_path = path / "units" / "5Min" / "2024-07.parquet"
    units_path.parent.mkdir(parents=True)
    units = [
            {
                "unit_id": f"{index:064d}",
                "session_date_et": pd.Timestamp("2024-07-08").date(),
                "requested_start_utc": pd.Timestamp(
                    "2024-07-08 13:30:00",
                    tz="UTC",
                ),
                "requested_end_utc": pd.Timestamp(
                    "2024-07-08 13:40:00",
                    tz="UTC",
                ),
                "canonical_symbols_json": '["AAA","BBB"]',
                "provider_symbols_json": '["AAA","BBB"]',
                "provider_to_canonical_json": (
                    '{"AAA":"AAA","BBB":"BBB"}'
                ),
                "symbol_count": 2,
                "expected_bars_per_symbol": 2,
                "maximum_expected_rows": 4,
                "timeframe": "5Min",
                "price_feed": "sip",
                "adjustment": "all",
                "sort": "asc",
                "limit": 10_000,
                "plan_fingerprint": "placeholder",
            }
            for index in range(unit_count)
        ]
    pd.DataFrame(units).to_parquet(units_path, index=False)
    request_payload = {
        "schema": INTRADAY_HISTORY_PLAN_SCHEMA,
        "policy_sha256": config.sha256(),
        "test": True,
    }
    fingerprint = _json_sha256(request_payload)
    frame = pd.read_parquet(units_path)
    frame["plan_fingerprint"] = fingerprint
    frame.to_parquet(units_path, index=False)
    request = {**request_payload, "plan_fingerprint": fingerprint}
    request_path = path / "_request.json"
    request_path.write_text(
        json.dumps(request, sort_keys=True),
        encoding="utf-8",
    )
    files = [
        _file_record(request_path, path, 1),
        _file_record(units_path, path, unit_count),
    ]
    manifest_path = path / "_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": INTRADAY_HISTORY_PLAN_SCHEMA,
                "plan_fingerprint": fingerprint,
                "policy_sha256": config.sha256(),
                "files": files,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (path / "_authority.json").write_text(
        json.dumps(
            {
                "schema": PLAN_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(manifest_path),
                "plan_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return path


def _file_record(path: Path, root: Path, rows: int) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _json_sha256(value: object) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
