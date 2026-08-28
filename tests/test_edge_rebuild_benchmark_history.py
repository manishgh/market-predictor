from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.edge_rebuild import benchmark_history
from market_predictor.edge_rebuild.benchmark_history import (
    build_selected_session_benchmark_plan,
)
from market_predictor.edge_rebuild.history_contracts import (
    SelectedSessionBenchmarkConfig,
    load_selected_session_benchmark_config,
)
from market_predictor.intraday.datasets.history import (
    load_complete_intraday_history_plan,
)
from market_predictor.modeling.strategy_contract import load_strategy_contract

POLICY = Path("configs/edge_rebuild_selected_session_benchmarks.toml")
CONTRACT = Path("configs/edge_rebuild_strategy_contract.toml")


def test_benchmark_plan_covers_every_session_with_all_benchmarks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_strategy_contract(CONTRACT)
    selection = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "session_date_et": ["2024-07-03", "2024-07-05"],
        }
    )
    identity = {
        "strategy_id": contract.intraday.strategy_id,
        "strategy_contract_sha256": contract.sha256(),
        "manifest_sha256": "a" * 64,
        "request_sha256": "b" * 64,
        "table_sha256": "c" * 64,
        "stock_sessions": 2,
        "symbols": 2,
        "sessions": 2,
        "first_session_et": "2024-07-03",
        "last_session_et": "2024-07-05",
        "research_only": True,
        "path": "test-selection",
    }
    monkeypatch.setattr(
        benchmark_history,
        "verify_selected_stock_sessions",
        lambda _directory: (selection, identity),
    )
    output = tmp_path / "plan"

    manifest = build_selected_session_benchmark_plan(
        selection_directory=tmp_path / "selection",
        policy_path=POLICY,
        output_directory=output,
        config=load_selected_session_benchmark_config(POLICY),
        strategy_contract=contract,
        strategy_contract_path=CONTRACT,
    )
    verified = load_complete_intraday_history_plan(output)
    units = pd.read_parquet(output / "units" / "1Min" / "2024-07.parquet")
    symbols = [json.loads(value) for value in units["canonical_symbols_json"]]

    assert verified["schema"] == (
        "edge_rebuild.selected_session_benchmark_one_minute_plan.v1"
    )
    assert manifest["summary"]["planned_history_sessions"] == 2
    assert manifest["summary"]["benchmark_tickers"] == 13
    assert manifest["summary"]["early_close_sessions"] == 1
    assert units["timeframe"].eq("1Min").all()
    assert {len(value) for value in symbols} == {13}
    assert set(units["expected_bars_per_symbol"]) == {210, 390}


def test_benchmark_contract_requires_qqq_and_every_sector() -> None:
    config = load_selected_session_benchmark_config(POLICY)
    payload = config.model_dump(mode="python")
    payload["benchmark_tickers"] = tuple(
        value for value in config.benchmark_tickers if value != "QQQ"
    )

    with pytest.raises(ValueError, match="SPY, QQQ"):
        SelectedSessionBenchmarkConfig.model_validate(payload)
