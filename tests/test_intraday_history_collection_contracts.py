from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from market_predictor.intraday.contracts.history_collection import (
    BROAD_INTRADAY_HISTORY_SCHEMA,
    EXTENDED_CONTEXT_SCHEMA,
    INTRADAY_HISTORY_SCHEMA,
    SELECTED_SESSION_BENCHMARK_SCHEMA,
    SELECTED_SESSION_HISTORY_SCHEMA,
    SELECTED_SESSION_ONE_MINUTE_SCHEMA,
    BroadIntradayHistoryConfig,
    ExtendedSessionContextConfig,
    IntradayHistoryConfig,
    IntradayTransportConfig,
    PointInTimeUniverseConfig,
    SelectedSessionBenchmarkConfig,
    SelectedSessionHistoryConfig,
    SelectedSessionOneMinuteConfig,
    load_broad_intraday_history_config,
    load_collection_transport_config,
    load_extended_session_context_config,
    load_intraday_history_config,
    load_selected_session_benchmark_config,
    load_selected_session_history_config,
    load_selected_session_one_minute_config,
)

_CONFIG_CASES: tuple[
    tuple[
        str,
        Path,
        Callable[[Path], IntradayTransportConfig],
        type[IntradayTransportConfig],
        str,
    ],
    ...,
] = (
    (
        INTRADAY_HISTORY_SCHEMA,
        Path("configs/edge_rebuild_intraday_history.toml"),
        load_intraday_history_config,
        IntradayHistoryConfig,
        "252886fb7b7fcfca19917a1daa8e1ea43d950e006287adca12796525c911a830",
    ),
    (
        EXTENDED_CONTEXT_SCHEMA,
        Path("configs/edge_rebuild_extended_session_context.toml"),
        load_extended_session_context_config,
        ExtendedSessionContextConfig,
        "2fb6118c448438c5ffe59a1cb3319b39f4e80bf47bca5c77df55948e204700d6",
    ),
    (
        SELECTED_SESSION_HISTORY_SCHEMA,
        Path("configs/edge_rebuild_selected_session_history.toml"),
        load_selected_session_history_config,
        SelectedSessionHistoryConfig,
        "536a8194d376cf2e6925d90b8bf22e7f071fc2854c793c9b5a365b78b2841c22",
    ),
    (
        SELECTED_SESSION_ONE_MINUTE_SCHEMA,
        Path("configs/edge_rebuild_selected_session_one_minute.toml"),
        load_selected_session_one_minute_config,
        SelectedSessionOneMinuteConfig,
        "0c2896b7e40a5c0afb502c65b6ce167f16705d1b36aa44d0a90c94f2ffe1e318",
    ),
    (
        SELECTED_SESSION_BENCHMARK_SCHEMA,
        Path("configs/edge_rebuild_selected_session_benchmarks.toml"),
        load_selected_session_benchmark_config,
        SelectedSessionBenchmarkConfig,
        "4215b3f63b7b5ff0cf30c6415d35362653f4f492510a9cab9a04b971be14c2cf",
    ),
    (
        BROAD_INTRADAY_HISTORY_SCHEMA,
        Path("configs/edge_rebuild_broad_intraday_history.toml"),
        load_broad_intraday_history_config,
        BroadIntradayHistoryConfig,
        "07bd5c64ef9c1b66b09cec7122e62c3abd4cda83e3ebea1d34742b497e993832",
    ),
)


@pytest.mark.parametrize("model", (
    IntradayTransportConfig,
    PointInTimeUniverseConfig,
    IntradayHistoryConfig,
    ExtendedSessionContextConfig,
    SelectedSessionHistoryConfig,
    SelectedSessionOneMinuteConfig,
    SelectedSessionBenchmarkConfig,
    BroadIntradayHistoryConfig,
))
def test_intraday_history_contracts_have_one_horizon_owner(model: type[object]) -> None:
    assert model.__module__ == "market_predictor.intraday.contracts.history_collection"


@pytest.mark.parametrize("schema,path,loader,model,expected_sha256", _CONFIG_CASES)
def test_intraday_history_contract_schema_and_hash_are_stable(
    schema: str,
    path: Path,
    loader: Callable[[Path], IntradayTransportConfig],
    model: type[IntradayTransportConfig],
    expected_sha256: str,
) -> None:
    loaded = loader(path)

    assert loaded.schema_version == schema
    assert isinstance(loaded, model)
    assert loaded.sha256() == expected_sha256
    assert load_collection_transport_config(path) == loaded
