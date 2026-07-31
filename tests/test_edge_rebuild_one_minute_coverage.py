from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from market_predictor.edge_rebuild.history_collection import collect_intraday_history
from market_predictor.edge_rebuild.history_contracts import (
    load_collection_transport_config,
    load_selected_session_history_config,
    load_selected_session_one_minute_config,
)
from market_predictor.edge_rebuild.intraday_selection import (
    IntradaySelectionResult,
    publish_intraday_selection,
)
from market_predictor.edge_rebuild.one_minute_coverage import (
    load_complete_one_minute_coverage,
    publish_selected_session_one_minute_coverage,
)
from market_predictor.edge_rebuild.selected_session_history import (
    build_selected_session_history_plan,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.sources.alpaca import AlpacaBarsPage
from market_predictor.v3.errors import DataReadinessError

POLICY = Path("configs/edge_rebuild_selected_session_one_minute.toml")
FIVE_MINUTE_POLICY = Path("configs/edge_rebuild_selected_session_history.toml")
CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")
SESSION = "2024-07-05"


def test_coverage_publishes_hash_bound_ready_evidence(tmp_path: Path) -> None:
    plan, collection, five_minute = _collect(tmp_path, ["AAA", "BBB"])
    output = tmp_path / "coverage"

    result = publish_selected_session_one_minute_coverage(
        plan_directory=plan,
        collection_directory=collection,
        five_minute_collection_directory=five_minute,
        strategy_contract=load_strategy_contract(CONTRACT_PATH),
        strategy_contract_path=CONTRACT_PATH,
        output_directory=output,
    )
    verified = load_complete_one_minute_coverage(output)

    assert result["ready_for_feature_build"] is True
    assert result["summary"]["required_stock_sessions"] == 2
    assert result["summary"]["empty_stock_sessions"] == 0
    assert result["summary"]["complete_stock_sessions"] == 2
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
    plan, collection, five_minute = _collect(
        tmp_path, symbols, empty_symbols={"S00"}
    )

    result = publish_selected_session_one_minute_coverage(
        plan_directory=plan,
        collection_directory=collection,
        five_minute_collection_directory=five_minute,
        strategy_contract=load_strategy_contract(CONTRACT_PATH),
        strategy_contract_path=CONTRACT_PATH,
        output_directory=tmp_path / "coverage",
    )
    exclusions = pd.read_parquet(
        tmp_path / "coverage" / "excluded_securities.parquet"
    )

    assert result["ready_for_feature_build"] is True
    assert result["summary"]["excluded_securities"] == 1
    assert result["summary"]["excluded_security_share"] == pytest.approx(1 / 30)
    assert exclusions["ticker"].tolist() == ["S00"]


def test_coverage_rejects_five_minute_input_in_one_minute_slot(
    tmp_path: Path,
) -> None:
    plan, _, five_minute = _collect(tmp_path, ["AAA", "BBB"])

    with pytest.raises(DataReadinessError, match="1Min collection identity"):
        publish_selected_session_one_minute_coverage(
            plan_directory=plan,
            collection_directory=five_minute,
            five_minute_collection_directory=five_minute,
            strategy_contract=load_strategy_contract(CONTRACT_PATH),
            strategy_contract_path=CONTRACT_PATH,
            output_directory=tmp_path / "coverage",
        )


def _collect(
    root: Path,
    symbols: list[str],
    *,
    empty_symbols: set[str] | None = None,
) -> tuple[Path, Path, Path]:
    contract = load_strategy_contract(CONTRACT_PATH)
    selection = pd.DataFrame(
        {
            "ticker": symbols,
            "session_date_et": [SESSION] * len(symbols),
            "average_volume_prior_sessions": [1_500_000.0] * len(symbols),
            "median_volume_prior_sessions": [1_400_000.0] * len(symbols),
            "relative_volume_at_activation": [3.3] * len(symbols),
            "price_at_activation": [42.5] * len(symbols),
            "activation_time_utc": [
                pd.Timestamp(f"{SESSION} 14:36:00+00:00") for _ in symbols
            ],
            "activation_rank": range(1, len(symbols) + 1),
        }
    )
    screen = root / "screen"
    publish_intraday_selection(
        IntradaySelectionResult(
            liquidity=selection,
            selection=selection,
            audit={
                "schema": "edge_rebuild.intraday_universe_selection.v2",
                "strategy_id": contract.intraday.strategy_id,
                "strategy_contract_sha256": contract.sha256(),
                "canonical_dir": str(root / "canonical"),
                "first_session_et": SESSION,
                "last_session_et": SESSION,
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
        source_factory=lambda: _FakeSource(empty_symbols or set()),
    )
    five_minute_plan = root / "five-minute-plan"
    build_selected_session_history_plan(
        selection_directory=screen,
        policy_path=FIVE_MINUTE_POLICY,
        output_directory=five_minute_plan,
        config=load_selected_session_history_config(FIVE_MINUTE_POLICY),
        strategy_contract=contract,
        strategy_contract_path=CONTRACT_PATH,
    )
    five_minute_collection = root / "five-minute-collection"
    collect_intraday_history(
        plan_directory=five_minute_plan,
        policy_path=FIVE_MINUTE_POLICY,
        output_directory=five_minute_collection,
        config=load_collection_transport_config(FIVE_MINUTE_POLICY),
        source_factory=lambda: _FakeSource(set()),
    )
    return plan, collection, five_minute_collection


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
        del end
        timeframe = str(kwargs["timeframe"])
        minutes = 1 if timeframe == "1Min" else 5
        bars = 390 if timeframe == "1Min" else 78
        timestamps = [
            pd.Timestamp(start) + pd.Timedelta(minutes=minutes * offset)
            for offset in range(bars)
        ]
        return AlpacaBarsPage(
            request_page_token=None,
            next_page_token=None,
            bars={
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
            },
            response_headers={},
        )
