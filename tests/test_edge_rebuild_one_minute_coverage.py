from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from urllib.parse import urlencode

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.history_collection import collect_intraday_history
from market_predictor.edge_rebuild.one_minute_coverage import (
    load_complete_one_minute_coverage,
    publish_selected_session_one_minute_coverage,
)
from market_predictor.edge_rebuild.selected_session_history import (
    build_selected_session_history_plan,
)
from market_predictor.intraday.contracts.history_collection import (
    load_collection_transport_config,
    load_selected_session_one_minute_config,
)
from market_predictor.intraday.datasets.history import write_plan_json
from market_predictor.intraday.datasets.selection import (
    IntradaySelectionResult,
    publish_intraday_selection,
)
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.sources.alpaca import AlpacaBarsPage

POLICY = Path("configs/edge_rebuild_selected_session_one_minute.toml")
CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")
SESSION = "2024-07-05"
EARLY_CLOSE_SESSION = "2024-07-03"


def test_coverage_publishes_hash_bound_ready_evidence(tmp_path: Path) -> None:
    plan, collection, canonical, _ = _collect(tmp_path, ["AAA", "BBB"])
    output = tmp_path / "coverage"

    result = _publish(plan, collection, canonical, output)
    verified = load_complete_one_minute_coverage(output)

    assert result["ready_for_feature_build"] is True
    assert result["summary"]["required_stock_sessions"] == 2
    assert result["summary"]["empty_stock_sessions"] == 0
    assert result["summary"]["complete_stock_sessions"] == 2
    assert result["five_minute_canonical_manifest_sha256"] == file_sha256(canonical / "_manifest.json")
    assert result["five_minute_canonical_authority_sha256"] == file_sha256(canonical / "_authority.json")
    assert verified["summary"] == result["summary"]

    coverage = output / "stock_session_coverage.parquet"
    frame = pd.read_parquet(coverage)
    frame.loc[0, "observed_rows"] = 0
    frame.to_parquet(coverage, index=False)
    with pytest.raises(DataReadinessError, match="failed hash"):
        load_complete_one_minute_coverage(output)


def test_coverage_excludes_whole_security_below_five_percent(
    tmp_path: Path,
) -> None:
    symbols = [f"S{index:02d}" for index in range(30)]
    plan, collection, canonical, _ = _collect(
        tmp_path,
        symbols,
        empty_one_minute_symbols={"S00"},
    )

    result = _publish(plan, collection, canonical, tmp_path / "coverage")
    exclusions = pd.read_parquet(tmp_path / "coverage" / "excluded_securities.parquet")

    assert result["ready_for_feature_build"] is True
    assert result["summary"]["excluded_securities"] == 1
    assert result["summary"]["excluded_security_share"] == pytest.approx(1 / 30)
    assert exclusions["ticker"].tolist() == ["S00"]


def test_missing_canonical_session_excludes_whole_security(
    tmp_path: Path,
) -> None:
    symbols = [f"S{index:02d}" for index in range(30)]
    plan, collection, canonical, _ = _collect(
        tmp_path,
        symbols,
        missing_canonical_symbols={"S00"},
    )

    result = _publish(plan, collection, canonical, tmp_path / "coverage")
    coverage = pd.read_parquet(tmp_path / "coverage" / "stock_session_coverage.parquet")
    excluded = pd.read_parquet(tmp_path / "coverage" / "excluded_securities.parquet")

    missing = coverage.loc[coverage["ticker"].eq("S00")].iloc[0]
    assert missing["observed_five_minute_rows"] == 0
    assert missing["coverage_status"] == "incomplete"
    assert excluded["ticker"].tolist() == ["S00"]
    assert result["ready_for_feature_build"] is True


def test_early_close_uses_exact_xnys_counts(tmp_path: Path) -> None:
    plan, collection, canonical, _ = _collect(
        tmp_path,
        ["AAA"],
        session=EARLY_CLOSE_SESSION,
    )

    _publish(plan, collection, canonical, tmp_path / "coverage")
    coverage = pd.read_parquet(tmp_path / "coverage" / "stock_session_coverage.parquet")

    assert coverage.loc[0, "expected_rows"] == 210
    assert coverage.loc[0, "observed_rows"] == 210
    assert coverage.loc[0, "expected_five_minute_rows"] == 42
    assert coverage.loc[0, "observed_five_minute_rows"] == 42
    assert coverage.loc[0, "five_minute_bar_continuity"] == 1.0


def test_corrupt_canonical_file_is_rejected(tmp_path: Path) -> None:
    plan, collection, canonical, _ = _collect(tmp_path, ["AAA"])
    path = canonical / "regular" / "5m" / "AAA.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "volume"] = 99_999
    changed.to_parquet(path, index=False)

    with pytest.raises(DataReadinessError, match="failed hash"):
        _publish(plan, collection, canonical, tmp_path / "coverage")


def test_canonical_manifest_row_count_poison_is_rejected(tmp_path: Path) -> None:
    plan, collection, canonical, _ = _collect(tmp_path, ["AAA"])
    manifest = _read_json(canonical / "_manifest.json")
    manifest["files"][0]["rows"] += 1
    manifest["total_rows"] += 1
    write_plan_json(canonical / "_manifest.json", manifest)
    _resign_authority(canonical)

    with pytest.raises(DataReadinessError, match="row count differs"):
        _publish(plan, collection, canonical, tmp_path / "coverage")


def test_future_regular_bar_is_rejected_even_when_resigned(tmp_path: Path) -> None:
    plan, collection, canonical, _ = _collect(
        tmp_path,
        ["AAA"],
        future_bar_symbols={"AAA"},
    )

    with pytest.raises(DataReadinessError, match="not causal and exact"):
        _publish(plan, collection, canonical, tmp_path / "coverage")


def test_plan_lineage_rejects_legacy_selection_schema(tmp_path: Path) -> None:
    plan, collection, canonical, screen = _collect(tmp_path, ["AAA"])
    manifest = _read_json(screen / "_manifest.json")
    manifest["schema"] = "edge_rebuild.intraday_universe_selection.v2"
    write_plan_json(screen / "_manifest.json", manifest)
    authority = _read_json(screen / "_authority.json")
    authority["artifact_sha256"] = file_sha256(screen / "_manifest.json")
    write_plan_json(screen / "_authority.json", authority)

    with pytest.raises(DataReadinessError, match="trusted current selection"):
        _publish(plan, collection, canonical, tmp_path / "coverage")


def _publish(
    plan: Path,
    collection: Path,
    canonical: Path,
    output: Path,
) -> dict[str, Any]:
    return publish_selected_session_one_minute_coverage(
        plan_directory=plan,
        collection_directory=collection,
        five_minute_canonical_directory=canonical,
        strategy_contract=load_strategy_contract(CONTRACT_PATH),
        strategy_contract_path=CONTRACT_PATH,
        output_directory=output,
    )


def _collect(
    root: Path,
    symbols: list[str],
    *,
    session: str = SESSION,
    empty_one_minute_symbols: set[str] | None = None,
    missing_canonical_symbols: set[str] | None = None,
    future_bar_symbols: set[str] | None = None,
) -> tuple[Path, Path, Path, Path]:
    contract = load_strategy_contract(CONTRACT_PATH)
    selection = pd.DataFrame(
        {
            "ticker": symbols,
            "session_date_et": [session] * len(symbols),
            "average_volume_prior_sessions": [1_500_000.0] * len(symbols),
            "median_volume_prior_sessions": [1_400_000.0] * len(symbols),
            "relative_volume_at_activation": [3.3] * len(symbols),
            "price_at_activation": [42.5] * len(symbols),
            "activation_time_utc": [pd.Timestamp(f"{session} 14:36:00+00:00") for _ in symbols],
            "activation_rank": range(1, len(symbols) + 1),
        }
    )
    screen = root / "screen"
    publish_intraday_selection(
        IntradaySelectionResult(
            liquidity=selection,
            selection=selection,
            audit={
                "schema": "edge_rebuild.intraday_universe_selection.v3",
                "strategy_id": contract.intraday.strategy_id,
                "strategy_contract_sha256": contract.sha256(),
                "canonical_dir": str(root / "screen-source-canonical"),
                "membership_authority_dir": str(root / "memberships"),
                "membership_authority_sha256": "1" * 64,
                "membership_manifest_sha256": "2" * 64,
                "membership_table_sha256": "3" * 64,
                "membership_universe_sha256": "4" * 64,
                "membership_universe_snapshot_id": "test-sp500",
                "membership_parent_lineage": {
                    "raw_authority_sha256": "5" * 64,
                    "raw_manifest_sha256": "6" * 64,
                },
                "membership_cold_start_policy": "reset_on_each_membership_entry",
                "first_session_et": session,
                "last_session_et": session,
                "excluded_tickers": [],
            },
        ),
        output_directory=screen,
    )
    plan = root / "plan"
    build_selected_session_history_plan(
        selection_directory=screen,
        policy_path=POLICY,
        output_directory=plan,
        config=load_selected_session_one_minute_config(POLICY),
        strategy_contract=contract,
        strategy_contract_path=CONTRACT_PATH,
    )
    collection = root / "collection"
    collect_intraday_history(
        plan_directory=plan,
        policy_path=POLICY,
        output_directory=collection,
        config=load_collection_transport_config(POLICY),
        source_factory=lambda: cast(Any, _FakeSource(empty_one_minute_symbols or set())),
    )
    canonical = _canonical_store(
        root,
        symbols=symbols,
        session=session,
        missing_symbols=missing_canonical_symbols or set(),
        future_bar_symbols=future_bar_symbols or set(),
    )
    return plan, collection, canonical, screen


def _canonical_store(
    root: Path,
    *,
    symbols: list[str],
    session: str,
    missing_symbols: set[str],
    future_bar_symbols: set[str],
) -> Path:
    calendar = xcals.get_calendar("XNYS")
    label = pd.Timestamp(session)
    opened = pd.Timestamp(calendar.session_open(label)).tz_convert("UTC")
    closed = pd.Timestamp(calendar.session_close(label)).tz_convert("UTC")
    canonical = root / "canonical"
    records: list[dict[str, Any]] = []
    for ticker in symbols:
        if ticker in missing_symbols:
            continue
        starts = list(pd.date_range(opened, closed, freq="5min", inclusive="left"))
        if ticker in future_bar_symbols:
            starts.append(closed)
        frame = pd.DataFrame(
            {
                "ticker": ticker,
                "session_date_et": session,
                "session_segment": "regular",
                "history_era": "collected",
                "timeframe": "5m",
                "bar_start_utc": starts,
                "bar_end_utc": [value + pd.Timedelta(minutes=5) for value in starts],
                "available_at_utc": [value + pd.Timedelta(minutes=6) for value in starts],
                "ingested_at_utc": [value + pd.Timedelta(minutes=6) for value in starts],
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 1_000,
                "source": "alpaca",
                "price_feed": "sip",
                "adjustment": "all",
            }
        )
        path = canonical / "regular" / "5m" / f"{ticker}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        records.append(
            {
                "store": "regular",
                "ticker": ticker,
                "path": f"regular/5m/{ticker}.parquet",
                "rows": len(frame),
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        "schema": "edge_rebuild.intraday_materialization.v1",
        "window_first_session": session,
        "window_last_session": session,
        "integrity": {
            "blocking_defect_count": 0,
            "identity_breaks": [],
            "fabricated_bars": [],
            "truncated_ticker_sessions": [],
        },
        "files": records,
        "total_rows": sum(int(record["rows"]) for record in records),
    }
    write_plan_json(canonical / "_manifest.json", manifest)
    write_plan_json(
        canonical / "_authority.json",
        {
            "schema": "edge_rebuild.intraday_materialization_authority.v1",
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(canonical / "_manifest.json"),
        },
    )
    return canonical


def _resign_authority(canonical: Path) -> None:
    authority = _read_json(canonical / "_authority.json")
    authority["artifact_sha256"] = file_sha256(canonical / "_manifest.json")
    write_plan_json(canonical / "_authority.json", authority)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


class _FakeSource:
    def __init__(self, empty_symbols: set[str]) -> None:
        self.settings = SimpleNamespace(alpaca_stock_feed="sip")
        self.client = SimpleNamespace(timeout=30)
        self.empty_symbols = empty_symbols

    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        **kwargs: object,
    ) -> AlpacaBarsPage:
        timeframe = str(kwargs["timeframe"])
        minutes = 1 if timeframe == "1Min" else 5
        bars = int((pd.Timestamp(end) - pd.Timestamp(start)) / pd.Timedelta(minutes=minutes)) + 1
        timestamps = [pd.Timestamp(start) + pd.Timedelta(minutes=minutes * offset) for offset in range(bars)]
        raw_bars = {
            symbol: (
                ()
                if symbol in self.empty_symbols
                else tuple(
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
            )
            for symbol in symbols
        }
        payload = {
            "bars": {
                symbol: list(values) for symbol, values in raw_bars.items()
            },
            "next_page_token": None,
        }
        query = {
            "symbols": ",".join(symbols),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": "sip",
            "limit": str(kwargs["limit"]),
            "adjustment": "all",
            "sort": "asc",
            "asof": kwargs["asof"].isoformat(),
        }
        requested_url = "https://data.alpaca.markets/v2/stocks/bars?" + urlencode(query)
        return AlpacaBarsPage(
            request_page_token=None,
            next_page_token=None,
            bars=raw_bars,
            response_headers={"Content-Type": "application/json"},
            raw_payload=payload,
            raw_body=json.dumps(payload, separators=(",", ":")).encode(),
            requested_url=requested_url,
            status_code=200,
            retrieved_at_utc=datetime.now(UTC),
            final_url=requested_url,
        )
