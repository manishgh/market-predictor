import pytest
import pandas as pd
import numpy as np
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.intraday_features import build_causal_intraday_features, CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS
from tests.test_edge_rebuild_intraday_live import _inputs, ROOT

completed_volume_bars, stock_one_minute_bars, benchmark_one_minute_bars, point_in_time_memberships = _inputs()
contract = load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")
built = build_causal_intraday_features(
    completed_volume_bars,
    stock_one_minute_bars,
    benchmark_one_minute_bars,
    point_in_time_memberships,
    contract=contract,
    strategy_contract_sha256=contract.sha256(),
)

# Find the nan columns for the last row
last_row = built.iloc[-1]
for col in CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS:
    if pd.isna(last_row[col]):
        print(f"NaN Found in {col}")
