"""Synthetic candidate composition through the actual relationship/reaction kernels."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.features.issuer_reaction_profile import build_issuer_reaction_profile
from market_predictor.swing.features.return_candidate import build_return_candidate_features
from market_predictor.swing.features.return_relationships import build_return_relationship_profile
from tests.test_swing_issuer_reaction_profile import bundle as reaction_bundle


@pytest.fixture
def candidate() -> dict[str, Any]:
    inputs = cast(Any, reaction_bundle).__wrapped__()
    parent = inputs.pop("baseline")
    sources = inputs.pop("sources")
    rows = parent.rows.drop(columns=list(RETURN_RELATIONSHIP_COLUMNS)).copy()
    rows["feature_profile"] = "technical_market"
    rows["cross_section_eligible"] = True
    rows["sector"] = "technology"
    rows["primary_benchmark"] = "SPY"
    names = parent.model_columns[:120]
    baseline = replace(parent, rows=rows, model_columns=names,
        availability_columns={name: parent.availability_columns[name] for name in names})
    return {
        **inputs,
        "baseline": baseline,
        "baseline_availability_semantics": "historical_proxy",
        "expected_decisions": rows.loc[:, [
            "decision_id", "security_id", "ticker", "decision_time_utc", "session_date_et",
        ]],
        "contract": load_strategy_contract(
            Path(__file__).resolve().parents[1] / "configs/edge_rebuild_strategy_contract.toml"),
        "relationship_sources": sources.bars,
        "reaction_sources": sources,
    }


def _observed(candidate: dict[str, Any]) -> dict[str, Any]:
    bars = candidate["relationship_sources"].model_copy(update={"availability_semantics": "observed"})
    sources = candidate["reaction_sources"].model_copy(update={
        "bars": bars, "event_availability_semantics": "observed", "identity_availability_semantics": "observed",
    })
    return {**candidate, "baseline_availability_semantics": "observed", "relationship_sources": bars,
        "reaction_sources": sources, "stock_bars": candidate["stock_bars"].assign(availability_policy="observed"),
        "spy_bars": candidate["spy_bars"].assign(availability_policy="observed")}


def test_exact_kernel_composition_and_metadata_preservation(candidate: dict[str, Any]) -> None:
    before = candidate["baseline"].rows.copy(deep=True)
    relationships = build_return_relationship_profile(**{key: candidate[key] for key in (
        "baseline", "expected_decisions", "stock_bars", "spy_bars", "history_sessions", "contract",
    )}, sources=candidate["relationship_sources"])
    direct = build_issuer_reaction_profile(**{key: candidate[key] for key in (
        "qualified_events", "coverage", "stock_bars", "spy_bars", "history_sessions",
    )}, baseline=relationships, sources=candidate["reaction_sources"])
    result = build_return_candidate_features(**candidate)
    pd.testing.assert_frame_equal(result.rows, direct.rows)
    assert result.model_columns == (*candidate["baseline"].model_columns, *RETURN_RELATIONSHIP_COLUMNS, *REACTION_COLUMNS)
    assert len(result.model_columns) == 126
    assert result.availability_columns == direct.availability_columns
    retained = before.columns.drop("feature_profile")
    pd.testing.assert_frame_equal(result.rows[retained], before[retained])
    pd.testing.assert_frame_equal(candidate["baseline"].rows, before)
    assert result.rows[REACTION_COLUMNS[0]].iloc[0] == pytest.approx(0.08)
    assert result.rows[REACTION_COLUMNS[1]].iloc[0] == pytest.approx(4.0)
    assert all(result.audit[key] is False for key in (
        "source_eligible", "training_eligible", "promotion_eligible", "serving_eligible", "economic_acceptance",
    ))


def test_observed_batch_live_and_single_decision_equality(candidate: dict[str, Any]) -> None:
    inputs = _observed(candidate)
    parent = inputs["baseline"]
    later = parent.rows.copy()
    day = later.session_date_et.iloc[0]
    later["session_date_et"] = inputs["history_sessions"][inputs["history_sessions"].index(day) + 1]
    later["decision_time_utc"] = swing_prediction_cutoffs(later.session_date_et)
    later["decision_id"] = "later-decision"
    later["parent_clock"] = later.decision_time_utc - pd.Timedelta(minutes=5)
    rows = pd.concat([later, parent.rows], ignore_index=True)
    inputs["baseline"] = replace(parent, rows=rows)
    inputs["expected_decisions"] = rows.loc[:, inputs["expected_decisions"].columns]
    inputs["coverage"] = pd.DataFrame({"decision_id": rows.decision_id, "coverage_status": "known",
        "available_at_utc": rows.decision_time_utc})
    batch = build_return_candidate_features(**inputs)
    live = build_return_candidate_features(**inputs, purpose="live_construction")
    pd.testing.assert_frame_equal(batch.rows, live.rows)
    assert batch.audit["candidate_composition"] == live.audit["candidate_composition"]
    singles = []
    for position in range(2):
        single = {**inputs, "baseline": replace(parent, rows=rows.iloc[[position]]),
            "expected_decisions": inputs["expected_decisions"].iloc[[position]],
            "coverage": inputs["coverage"].iloc[[position]]}
        singles.append(build_return_candidate_features(**single, purpose="live_construction").rows)
    pd.testing.assert_frame_equal(batch.rows, pd.concat(singles, ignore_index=True))


@pytest.mark.parametrize("mode", ["absent", "unavailable", "future"])
def test_nullable_reaction_preserves_decision(candidate: dict[str, Any], mode: str) -> None:
    if mode == "absent":
        candidate["qualified_events"] = candidate["qualified_events"].iloc[:0]
    elif mode == "unavailable":
        day = candidate["baseline"].rows.session_date_et.iloc[0]
        candidate["stock_bars"] = candidate["stock_bars"].loc[~candidate["stock_bars"].session_date_et.eq(day)]
    else:
        candidate["qualified_events"] = candidate["qualified_events"].assign(
            event_available_at_utc=candidate["baseline"].rows.decision_time_utc + pd.Timedelta(seconds=1))
    result = build_return_candidate_features(**candidate)
    assert result.rows.decision_id.tolist() == candidate["baseline"].rows.decision_id.tolist()
    assert result.rows[list(REACTION_COLUMNS)].isna().all(axis=None)
    assert not result.rows.feature_eligible.any()


def test_wrong_issuer_rejected(candidate: dict[str, Any]) -> None:
    candidate["qualified_events"] = candidate["qualified_events"].assign(security_id="foreign-issuer")
    with pytest.raises(DataReadinessError, match="foreign security"):
        build_return_candidate_features(**candidate)


def test_newer_rejected_revision_suppresses_old_qualified_text(candidate: dict[str, Any]) -> None:
    revision = candidate["qualified_events"].copy()
    revision["event_version_sha256"] = "b" * 64
    revision["event_available_at_utc"] += pd.Timedelta(minutes=1)
    revision["qualification_status"] = "rejected"
    revision["event_family"] = None
    candidate["qualified_events"] = pd.concat([candidate["qualified_events"], revision], ignore_index=True)
    result = build_return_candidate_features(**candidate)
    assert result.rows.selected_event_id.isna().all()
    assert result.rows[list(REACTION_COLUMNS)].isna().all(axis=None)


@pytest.mark.parametrize("component", ["baseline", "bars", "events", "identity"])
def test_live_rejects_each_proxy_source(candidate: dict[str, Any], component: str) -> None:
    inputs = _observed(candidate)
    if component == "baseline":
        inputs["baseline_availability_semantics"] = "historical_proxy"
    elif component == "bars":
        inputs["relationship_sources"] = candidate["relationship_sources"]
        inputs["reaction_sources"] = inputs["reaction_sources"].model_copy(
            update={"bars": candidate["relationship_sources"]})
    else:
        field = "event_availability_semantics" if component == "events" else "identity_availability_semantics"
        inputs["reaction_sources"] = inputs["reaction_sources"].model_copy(
            update={field: "historical_proxy"})
    with pytest.raises(DataReadinessError, match="proxy|observed"):
        build_return_candidate_features(**inputs, purpose="live_construction")


@pytest.mark.parametrize(("field", "value"), [
    ("stock_authority_sha256", "a" * 64), ("spy_authority_sha256", "a" * 64),
    ("availability_semantics", "observed"), ("availability_policy_id", "another-policy"),
    ("availability_policy_sha256", "a" * 64), ("price_adjustment", "raw"),
    ("price_basis_and_vintage_id", "another-vintage"),
])
def test_mixed_component_authorities_rejected(candidate: dict[str, Any], field: str, value: str) -> None:
    bars = candidate["reaction_sources"].bars.model_copy(update={field: value})
    candidate["reaction_sources"] = candidate["reaction_sources"].model_copy(update={"bars": bars})
    with pytest.raises(DataReadinessError, match="identical stock/SPY"):
        build_return_candidate_features(**candidate)


def test_distinct_baseline_lineages_are_disclosed_and_hashed(candidate: dict[str, Any]) -> None:
    original = build_return_candidate_features(**candidate)
    bars = candidate["reaction_sources"].bars.model_copy(update={"baseline_authority_sha256": "a" * 64})
    candidate["reaction_sources"] = candidate["reaction_sources"].model_copy(update={"bars": bars})
    changed = build_return_candidate_features(**candidate)
    pd.testing.assert_frame_equal(original.rows, changed.rows)
    lineage = cast(Mapping[str, object], changed.audit["candidate_composition"])
    previous = cast(Mapping[str, object], original.audit["candidate_composition"])
    assert lineage["relationship_baseline_authority_sha256"] == "1" * 64
    assert lineage["reaction_baseline_authority_sha256"] == "a" * 64
    assert lineage["relationship_profile_sha256"] == previous["relationship_profile_sha256"]
    assert lineage["reaction_profile_sha256"] != previous["reaction_profile_sha256"]
    assert changed.audit["candidate_composition_sha256"] != original.audit["candidate_composition_sha256"]


def test_live_future_baseline_value_rejected(candidate: dict[str, Any]) -> None:
    inputs = _observed(candidate)
    parent = inputs["baseline"]
    rows = parent.rows.assign(parent_clock=parent.rows.decision_time_utc + pd.Timedelta(seconds=1))
    inputs["baseline"] = replace(parent, rows=rows)
    with pytest.raises(DataReadinessError, match="unavailable at its cutoff"):
        build_return_candidate_features(**inputs, purpose="live_construction")


@pytest.mark.parametrize(("field", "value"), [("price_feed", "iex"), ("adjustment", "raw")])
def test_physical_bars_must_match_contract(candidate: dict[str, Any], field: str, value: str) -> None:
    candidate["stock_bars"] = candidate["stock_bars"].assign(**{field: value})
    with pytest.raises(DataReadinessError, match="SIP and the bound price adjustment"):
        build_return_candidate_features(**candidate)
