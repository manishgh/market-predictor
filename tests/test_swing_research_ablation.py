from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.reconciliation import event_feature_columns
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.catalyst_decision_authority import (
    RANKING_SOURCE_FAMILIES,
    WINDOWS,
    CatalystDecisionAuthority,
)
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from market_predictor.swing.features.research_ablation import assemble_research_ablations
from tests.test_swing_features import contract as contract
from tests.test_swing_research_partition import _inputs


def _authority(expected: pd.DataFrame, unknown: bool = False) -> CatalystDecisionAuthority:
    events = expected.copy()
    for name in event_feature_columns(WINDOWS, source_families=RANKING_SOURCE_FAMILIES):
        events[name] = pd.NaT if name.endswith("_utc") else 0.0
    events["evidence_lineage_count"] = 1
    events["evidence_lineage_sha256s"] = "[\"test\"]"
    coverage = pd.DataFrame([{
        "collection_id": "fixture", "chunk_id": identity, "security_id": identity, "ticker": ticker,
        "source_family": family, "requested_start_utc": cutoff - pd.Timedelta(days=4),
        "requested_end_utc": cutoff, "completed_at_utc": cutoff + pd.Timedelta(days=2),
        "status": "failed" if unknown else "observed_empty", "row_count": 0,
        "coverage_state": "failed_or_unobserved" if unknown else "observed_empty",
        "missingness_known": not unknown, "training_eligible": not unknown,
        "zero_event_semantics": "unknown_failed" if unknown else "known_zero_events",
        "schema_version": "swing.catalyst_source_coverage.v1",
    } for identity, ticker, cutoff in zip(expected.security_id, expected.ticker, expected.decision_time_utc, strict=True)
        for family in RANKING_SOURCE_FAMILIES])
    return CatalystDecisionAuthority(Path("test-only"), events, coverage, {"production_ready": False}, {})


def _run(contract: StrategyContract, unknown: bool = False) -> dict:
    expected, technical, outcomes = _inputs()
    technical = technical.iloc[:59].drop(columns="label_eligible", errors="ignore")
    return assemble_research_ablations(expected_decisions=expected, technical=technical, outcomes=outcomes,
        catalyst=_authority(expected, unknown), contract=contract,
        retained_security_ids=frozenset(expected.security_id),
        technical_availability={name: "base_available_at" for name in TECHNICAL_RANKING_FEATURES})


def test_research_ablation_keeps_matched_population_and_nullable_returns(contract: StrategyContract) -> None:
    result = _run(contract)
    assert set(result) == {"technical_market", "catalyst_full"}
    for value in result.values():
        assert len(value.rows) == 59
        assert pd.isna(value.rows.loc[0, "fixed_horizon_net_return"])
        assert not value.rows.training_eligible.any()
    assert result["catalyst_full"].rows.event_count_3d.eq(0).all()
    assert result["catalyst_full"].rows.catalyst_required_source_complete.all()


def test_unknown_news_stays_unknown_without_dropping_stock(contract: StrategyContract) -> None:
    result = _run(contract, unknown=True)
    for value in result.values():
        assert len(value.rows) == 59
        assert not value.rows.feature_eligible.any()
    assert result["catalyst_full"].rows.event_count_3d.isna().all()


def test_foreign_catalyst_decision_is_rejected(contract: StrategyContract) -> None:
    expected, technical, outcomes = _inputs()
    authority = _authority(expected)
    authority.decisions.loc[0, "decision_id"] = "foreign-decision"
    with pytest.raises(DataReadinessError, match="foreign or duplicate"):
        assemble_research_ablations(expected_decisions=expected, technical=technical, outcomes=outcomes,
            catalyst=authority, contract=contract,
            retained_security_ids=frozenset(expected.security_id), technical_availability={})


def test_sparse_event_authority_does_not_drop_observed_empty_decisions(contract: StrategyContract) -> None:
    expected, technical, outcomes = _inputs()
    original = _authority(expected)
    sparse = CatalystDecisionAuthority(original.directory, original.decisions.iloc[1:].copy(), original.coverage,
        original.manifest, original.authority)
    result = assemble_research_ablations(expected_decisions=expected,
        technical=technical.iloc[:59].drop(columns="label_eligible", errors="ignore"), outcomes=outcomes,
        catalyst=sparse, contract=contract, retained_security_ids=frozenset(expected.security_id),
        technical_availability={name: "base_available_at" for name in TECHNICAL_RANKING_FEATURES})
    assert len(result["catalyst_full"].rows) == 59
    assert result["catalyst_full"].rows.loc[0, "event_count_3d"] == 0
