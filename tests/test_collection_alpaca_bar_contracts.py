"""Canonical collection policies must not reuse historical request identities."""
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
     "40b15ae647c48162618f01e08cbb41a0251b701574c84605eefec59c308e83e1"),
    (BENCHMARK_POLICY, load_session_benchmark_config, SessionBenchmarkConfig, SESSION_BENCHMARK_SCHEMA,
     "ff86c2d82df0598ca9dff41f29287652def8cca6bf2ad17608050c594806c5d2"),
])
def test_policy_hashes_are_stable(path: Path, loader: Callable[[Path], AlpacaTransportConfig], model: type[object],
                                  schema: str, expected: str) -> None:
    loaded = loader(path)
    assert isinstance(loaded, model) and loaded.schema_version == schema and loaded.sha256() == expected


def test_recorded_sip_session_retains_distinct_historical_policy_identities() -> None:
    """Canonical policy changes cannot be presented as the recorded session's inputs."""
    request = json.loads(SESSION_REQUEST.read_text(encoding="utf-8"))
    assert Path(request["five_minute_policy_path"].replace("\\", "/")) == FIVE_MINUTE_POLICY
    assert Path(request["benchmark_policy_path"].replace("\\", "/")) == BENCHMARK_POLICY
    assert request["schema"] == "edge_rebuild.prospective_sip_session_request.v1"
    assert request["five_minute_policy_file_sha256"] == "31f6df00072eebe35e0f0169a12b07782dc6bb919edec8568abb57c2d54b0ada"
    assert request["benchmark_policy_file_sha256"] == "22524a6a9cd20ccd840c12984134452ad4de50515fe0e6d91d8c4f94089ea9f5"
    assert request["five_minute_policy_sha256"] == "252886fb7b7fcfca19917a1daa8e1ea43d950e006287adca12796525c911a830"
    assert request["benchmark_policy_sha256"] == "4215b3f63b7b5ff0cf30c6415d35362653f4f492510a9cab9a04b971be14c2cf"
    assert file_sha256(FIVE_MINUTE_POLICY) != request["five_minute_policy_file_sha256"]
    assert file_sha256(BENCHMARK_POLICY) != request["benchmark_policy_file_sha256"]
    assert load_regular_bar_history_config(FIVE_MINUTE_POLICY).sha256() != request["five_minute_policy_sha256"]
    assert load_session_benchmark_config(BENCHMARK_POLICY).sha256() != request["benchmark_policy_sha256"]


@pytest.mark.parametrize("retired_version", ["v1", "v2"])
@pytest.mark.parametrize(("path", "loader", "schema"), [
    (FIVE_MINUTE_POLICY, load_regular_bar_history_config, REGULAR_BAR_HISTORY_SCHEMA),
    (BENCHMARK_POLICY, load_session_benchmark_config, SESSION_BENCHMARK_SCHEMA),
])
def test_canonical_policy_loaders_reject_retired_schemas(
    tmp_path: Path, path: Path, loader: Callable[[Path], AlpacaTransportConfig], schema: str,
    retired_version: str,
) -> None:
    changed = tmp_path / "policy.toml"
    changed.write_text(path.read_text(encoding="utf-8").replace(
        f'"{schema}"', f'"{schema}.{retired_version}"'), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="policy is invalid"):
        loader(changed)


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
