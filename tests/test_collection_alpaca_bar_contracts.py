"""Retained Alpaca bar collection contracts keep every recorded policy identity."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.collection.alpaca_bars import contracts
from market_predictor.collection.alpaca_bars.contracts import (
    REGULAR_BAR_HISTORY_SCHEMA,
    SESSION_BENCHMARK_SCHEMA,
    AlpacaTransportConfig,
    PointInTimeUniverseConfig,
    RegularBarHistoryConfig,
    SessionBenchmarkConfig,
    load_regular_bar_history_config,
    load_session_benchmark_config,
)
from market_predictor.collection.retained_inputs import RETAINED_CONFIGS
from market_predictor.core.errors import DataReadinessError

FIVE_MINUTE_POLICY = Path("configs/edge_rebuild_intraday_history.toml")
BENCHMARK_POLICY = Path("configs/edge_rebuild_selected_session_benchmarks.toml")
# The request of the last prospective SIP session (data/raw/prospective_sip_sessions/session_20260820_v1).
SESSION_REQUEST = Path(__file__).parent / "fixtures" / "collection" / "sip_session_20260820_request.json"


@pytest.mark.parametrize("model", (AlpacaTransportConfig, PointInTimeUniverseConfig, RegularBarHistoryConfig,
                                   SessionBenchmarkConfig))
def test_retained_contracts_have_one_owner(model: type[object]) -> None:
    assert model.__module__ == contracts.__name__


@pytest.mark.parametrize(("path", "loader", "model", "schema", "expected"), [
    (FIVE_MINUTE_POLICY, load_regular_bar_history_config, RegularBarHistoryConfig, REGULAR_BAR_HISTORY_SCHEMA,
     "252886fb7b7fcfca19917a1daa8e1ea43d950e006287adca12796525c911a830"),
    (BENCHMARK_POLICY, load_session_benchmark_config, SessionBenchmarkConfig, SESSION_BENCHMARK_SCHEMA,
     "4215b3f63b7b5ff0cf30c6415d35362653f4f492510a9cab9a04b971be14c2cf"),
])
def test_policy_hashes_are_stable(path: Path, loader: Callable[[Path], AlpacaTransportConfig], model: type[object],
                                  schema: str, expected: str) -> None:
    loaded = loader(path)
    assert isinstance(loaded, model) and loaded.schema_version == schema and loaded.sha256() == expected


def test_the_recorded_sip_session_request_reproduces_from_todays_configs() -> None:
    """The next session appends to the same lineage only if today's files and models reproduce the recorded identity."""
    request = json.loads(SESSION_REQUEST.read_text(encoding="utf-8"))
    assert Path(request["five_minute_policy_path"].replace("\\", "/")) == FIVE_MINUTE_POLICY
    assert Path(request["benchmark_policy_path"].replace("\\", "/")) == BENCHMARK_POLICY
    assert file_sha256(FIVE_MINUTE_POLICY) == request["five_minute_policy_file_sha256"]
    assert file_sha256(BENCHMARK_POLICY) == request["benchmark_policy_file_sha256"]
    assert load_regular_bar_history_config(FIVE_MINUTE_POLICY).sha256() == request["five_minute_policy_sha256"]
    assert load_session_benchmark_config(BENCHMARK_POLICY).sha256() == request["benchmark_policy_sha256"]


def test_a_renamed_field_would_break_the_recorded_identity(tmp_path: Path) -> None:
    changed = tmp_path / "policy.toml"
    changed.write_text(FIVE_MINUTE_POLICY.read_text(encoding="utf-8").replace(
        "intraday_finalization_delay_seconds", "finalization_delay_seconds"), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="policy is invalid"):
        load_regular_bar_history_config(changed)


def test_every_config_the_recorded_session_binds_is_retained() -> None:
    request = json.loads(SESSION_REQUEST.read_text(encoding="utf-8"))
    bound = {request[key].replace("\\", "/") for key in ("five_minute_policy_path", "benchmark_policy_path")}
    assert bound == set(RETAINED_CONFIGS) and all(Path(path).is_file() for path in RETAINED_CONFIGS)
