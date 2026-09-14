import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES, swing_model_feature_columns
from market_predictor.swing.features.research_join import rebuild_research_peer_features
from tests.test_swing_features import _panel
from tests.test_swing_features import contract as contract


def _features() -> pd.DataFrame:
    data = _panel(sessions=1)
    data["decision_id"] = data.security_id + ":decision"
    data["decision_time_utc"] = pd.Timestamp("2024-01-02T23:00:00Z")
    data["base_available_at"] = pd.Timestamp("2024-01-02T21:00:00Z")
    data["return_5d_xs_z"] = 9999.0
    return data


def _rebuild(data: pd.DataFrame, contract: StrategyContract) -> tuple[pd.DataFrame, dict[str, str]]:
    return rebuild_research_peer_features(data, contract=contract,
        retained_security_ids=frozenset(f"sec:{i:03d}" for i in range(59)),
        availability_columns={name: "base_available_at" for name in TECHNICAL_RANKING_FEATURES})


def test_excluded_security_and_old_targets_cannot_change_peer_scores(contract: StrategyContract) -> None:
    data = _features()
    before, clocks = _rebuild(data, contract)
    data.loc[59, list(TECHNICAL_RANKING_FEATURES)] = 1e9
    data["forward_return"] = -1000.0
    data["return_5d_xs_z"] = -9999.0
    after, _ = _rebuild(data, contract)
    columns = list(swing_model_feature_columns(contract=contract, catalyst=False))
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert len(after) == 59
    assert "forward_return" not in after and "barrier_label" not in after
    assert set(clocks) == set(columns)
    assert after[clocks["return_5d_xs_z"]].eq(pd.Timestamp("2024-01-02T21:00:00Z")).all()


def test_peer_clock_includes_later_available_input(contract: StrategyContract) -> None:
    data = _features()
    data.loc[0, "base_available_at"] = pd.Timestamp("2024-01-02T21:04:00Z")
    after, clocks = _rebuild(data, contract)
    assert after[clocks["return_5d_xs_z"]].eq(pd.Timestamp("2024-01-02T21:04:00Z")).all()


@pytest.mark.parametrize("poison", ["eligibility", "warmup", "session", "future_clock"])
def test_peer_context_poison_rejected(contract: StrategyContract, poison: str) -> None:
    data = _features()
    if poison == "eligibility":
        data["feature_eligible"] = "False"
    elif poison == "warmup":
        data["daily_bar_count"] = -1
    elif poison == "session":
        data["session_date_et"] = pd.Timestamp("2024-01-03").date()
    else:
        data.loc[0, "base_available_at"] = pd.Timestamp("2024-01-02T23:01:00Z")
    with pytest.raises(DataReadinessError):
        _rebuild(data, contract)


def test_mixed_cutoffs_cannot_leak_peer_values(contract: StrategyContract) -> None:
    data = _features()
    data.loc[0, "decision_time_utc"] = pd.Timestamp("2024-01-02T21:01:00Z")
    data.loc[1, "base_available_at"] = pd.Timestamp("2024-01-02T21:04:00Z")
    with pytest.raises(DataReadinessError, match="canonical swing"):
        _rebuild(data, contract)


def test_equivalent_session_formats_share_value_and_clock_groups(contract: StrategyContract) -> None:
    data = _features()
    data["session_date_et"] = data.session_date_et.astype(object)
    data.loc[0, "session_date_et"] = "2024-01-02"
    data.loc[0, "base_available_at"] = pd.Timestamp("2024-01-02T22:59:00Z")
    after, clocks = _rebuild(data, contract)
    assert after.sector_peer_count.eq(59).all()
    assert after[clocks["return_5d_xs_z"]].eq(pd.Timestamp("2024-01-02T22:59:00Z")).all()
