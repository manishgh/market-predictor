"""Independent abstention audit; fixtures do not establish source admission."""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

import market_predictor.swing.datasets.predictor_abstention_derivation as owner
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.adjusted_source import build_adjusted_technical_source
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES, swing_model_feature_columns
from market_predictor.swing.features.research_join import rebuild_research_peer_features
from tests.test_swing_features import _panel
from tests.test_swing_features import contract as contract
from tests.test_swing_predictor_abstention_derivation import _prefix_case, _rebuild, _refreeze, _run
from tests.test_swing_predictor_abstention_derivation import example as example


def _late_observation_failure(example: dict[str, Any]) -> owner.ReviewedPredictorFailure:
    fact = example["facts"]["failures"][0]
    fact["reason_code"] = "invalid_observation_stream"
    fact["quarantine"] = "suffix_from_first_invalid"
    fact["first_invalid_session"] = "2023-10-13"
    fact["boundary_observation_sha256"] = "a" * 64
    fact["detail"] = "Synthetic isolated invalid volume on 2023-10-13; earlier windows independently valid."
    _refreeze(example)
    return owner.ReviewedPredictorFailure.model_validate_json(json.dumps(fact))


def test_late_bar_date_without_boundary_evidence_cannot_authorize_completion(example: dict[str, Any]) -> None:
    _late_observation_failure(example)
    with pytest.raises(DataReadinessError, match="first-invalid observation is absent"):
        _run(example)
    assert not example["output"].exists()


def test_whole_group_quarantine_of_earlier_peer_is_now_rejected(
    example: dict[str, Any], contract: StrategyContract,
) -> None:
    fact = _late_observation_failure(example)
    panel = _panel(sessions=1, securities=60)
    cutoff = pd.Timestamp("2019-07-09T22:00:00Z")
    panel["session_date_et"] = cutoff.date()
    panel["decision_time_utc"] = cutoff
    panel["technical_available_at_utc"] = cutoff - pd.Timedelta(hours=1)
    panel.loc[59, ["security_id", "ticker"]] = [fact.security_id, fact.symbol]
    panel["decision_id"] = panel.security_id + "-2019-07-09"
    panel["parent_decision_id"] = "parent-" + panel.decision_id
    panel["primary_benchmark"] = "XLK"
    clocks = dict.fromkeys(TECHNICAL_RANKING_FEATURES, "technical_available_at_utc")
    retained = frozenset(panel.security_id)
    baseline, _ = rebuild_research_peer_features(panel, contract=contract,
        retained_security_ids=retained, availability_columns=clocks)
    with pytest.raises(DataReadinessError, match="causally earlier"):
        owner._unavailable(panel.iloc[59:], fact, clocks)
    actual, _ = rebuild_research_peer_features(panel, contract=contract, retained_security_ids=retained, availability_columns=clocks)
    columns = list(swing_model_feature_columns(contract=contract, catalyst=False))
    assert actual.decision_id.tolist() == baseline.decision_id.tolist()
    assert baseline.iloc[:59].sector_peer_count.eq(60).all()
    assert actual.iloc[:59].sector_peer_count.eq(60).all()
    pd.testing.assert_frame_equal(baseline[columns], actual[columns])
    assert "forward_return" not in actual and "barrier_label" not in actual


def test_recovered_prefix_and_peers_equal_clean_history_despite_late_poison(contract: StrategyContract) -> None:
    args, fact, observation = _prefix_case()
    clean = {**args, "adjusted_bars": args["adjusted_bars"].copy()}
    clean["adjusted_bars"].loc[290, "volume"] = 1_000_000.0
    baseline = build_adjusted_technical_source(**clean, contract=contract).rows
    args["adjusted_bars"].loc[291:, ["open", "high", "low", "close"]] *= 1_000_000
    args["adjusted_bars"].loc[291:, "volume"] = -1
    recovered = _rebuild(args, fact, observation, contract)
    earlier = baseline.session_date_et.lt(fact.first_invalid_session)
    columns = [*TECHNICAL_RANKING_FEATURES, "feature_eligible", "daily_bar_count",
        "technical_available_at_utc", "raw_dollar_volume_available_at_utc"]
    pd.testing.assert_frame_equal(baseline.loc[earlier, columns], recovered.loc[earlier, columns])
    assert recovered.decision_id.tolist() == baseline.decision_id.tolist()
    assert recovered.loc[~earlier, list(TECHNICAL_RANKING_FEATURES)].isna().all().all()
    assert recovered.loc[~earlier, columns[-2:]].isna().all().all()
    assert recovered.loc[~earlier, "session_date_et"].min() == fact.first_invalid_session

    panel = _panel(sessions=1, securities=60)
    cutoff = baseline.loc[earlier, "decision_time_utc"].iloc[0]
    panel["decision_time_utc"] = cutoff
    panel["session_date_et"] = cutoff.date()
    panel["decision_id"] = panel.security_id + "-decision"
    for clock in columns[-2:]:
        panel[clock] = cutoff - pd.Timedelta(hours=1)
    clocks = {name: "raw_dollar_volume_available_at_utc" if name == "dollar_volume_log"
        else "technical_available_at_utc" for name in TECHNICAL_RANKING_FEATURES}
    retained = frozenset(panel.security_id)
    panel.loc[59, columns] = baseline.loc[earlier, columns].iloc[0]
    clean_peers, _ = rebuild_research_peer_features(panel, contract=contract,
        retained_security_ids=retained, availability_columns=clocks)
    panel.loc[59, columns] = recovered.loc[earlier, columns].iloc[0]
    recovered_peers, _ = rebuild_research_peer_features(panel, contract=contract,
        retained_security_ids=retained, availability_columns=clocks)
    pd.testing.assert_frame_equal(clean_peers, recovered_peers)
    assert recovered_peers.sector_peer_count.eq(60).all()


def test_retry_ignores_incomplete_staging_and_preserves_parent(example: dict[str, Any]) -> None:
    parent = {path: path.read_bytes() for path in example["parent"].rglob("*") if path.is_file()}
    example["fail_exit"] = True
    with pytest.raises(DataReadinessError, match="source exit failed"):
        _run(example)
    assert not list(example["output"].parent.glob(".*.pending"))
    abandoned = example["output"].parent / ".abandoned.pending"
    (abandoned / "months").mkdir(parents=True)
    (abandoned / "months" / "ignored.parquet").write_bytes(b"corrupt abandoned test staging")
    example["fail_exit"] = False
    result = _run(example)
    assert result["rows"] == 4 and example["output"].joinpath("_manifest.json").is_file()
    assert not (abandoned / "_manifest.json").exists()
    assert all(path.read_bytes() == payload for path, payload in parent.items())


@pytest.mark.parametrize("name", ["source", "evidence", "producer"])
def test_source_mutation_at_projection_exit_blocks_finalization(example: dict[str, Any], name: str) -> None:
    example["before_exit"] = lambda: example[name].write_bytes(b"changed after month publication")
    with pytest.raises(DataReadinessError, match="changed"):
        _run(example)
    assert example["exited"] and not example["output"].exists()
    assert not list(example["output"].parent.glob(".*.pending/_manifest.json"))
