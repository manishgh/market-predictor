from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.history_collection import (
    collect_intraday_history,
    load_complete_intraday_history_collection,
)
from market_predictor.edge_rebuild.history_contracts import (
    INTRADAY_HISTORY_PLAN_SCHEMA,
    load_intraday_history_config,
)
from market_predictor.edge_rebuild.intraday_history import (
    PLAN_AUTHORITY_SCHEMA,
)
from market_predictor.sources.alpaca import AlpacaBarsPage
from market_predictor.v3.errors import DataReadinessError

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
    payload = json.loads(gzip.decompress(raw_page.read_bytes()))
    assert set(payload["bars"]) == {"AAA", "BBB"}
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
        del end
        self.timeframes.append(str(kwargs["timeframe"]))
        timestamps = [
            pd.Timestamp(start),
            pd.Timestamp(start) + pd.Timedelta(minutes=5),
        ]
        return AlpacaBarsPage(
            request_page_token=kwargs.get("page_token"),
            next_page_token=None,
            bars={
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
            },
            response_headers={"X-RateLimit-Remaining": "100"},
        )


class _RepeatingTokenSource(_FakeAlpacaSource):
    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        if kwargs.get("page_token") is not None:
            return AlpacaBarsPage(
                request_page_token=kwargs.get("page_token"),
                next_page_token="repeat",
                bars={symbol: () for symbol in symbols},
                response_headers={},
            )
        page = super().fetch_bars_page(
            symbols,
            start,
            end,
            **kwargs,
        )
        return AlpacaBarsPage(
            request_page_token=kwargs.get("page_token"),
            next_page_token="repeat",
            bars=page.bars,
            response_headers=page.response_headers,
        )


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
