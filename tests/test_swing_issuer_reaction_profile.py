"""Synthetic decision projection tests; reuse the closed reaction kernel."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.issuer_reaction_profile import ISSUER_REACTION_PROFILE, issuer_reaction_profile_sha256
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.features.issuer_reaction_profile import build_issuer_reaction_profile
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from tests.test_swing_issuer_reaction import _inputs


@pytest.fixture
def bundle() -> dict[str, Any]:
    kernel = _inputs()
    contract = load_strategy_contract(Path(__file__).resolve().parents[1] / "configs/edge_rebuild_strategy_contract.toml")
    names = (*swing_model_feature_columns(contract=contract, catalyst=False), *RETURN_RELATIONSHIP_COLUMNS)
    decisions = kernel.events.loc[:, ["decision_id", "security_id", "ticker"]].copy()
    decisions["session_date_et"] = kernel.sessions[kernel.position]
    decisions["decision_time_utc"] = swing_prediction_cutoffs(decisions.session_date_et)
    rows = pd.concat([decisions, pd.DataFrame(np.full((1, 124), 0.125, dtype=np.float32), columns=names)], axis=1)
    rows["parent_clock"] = decisions.decision_time_utc - pd.Timedelta(minutes=5)
    rows["feature_profile"] = "technical_relationships"
    rows["feature_eligible"] = False
    rows["spy_fixed_horizon_excess_return"] = np.nan
    rows["parent_decision_id"] = "unchanged-parent"
    baseline = ResearchFeaturePartition(rows, tuple(names), dict.fromkeys(names, "parent_clock"), {})
    events = kernel.events.drop(columns=["decision_id", "decision_time_utc"]).copy()
    events["source_family"] = "alpaca"
    events["event_family"] = "earnings"
    events["qualification_status"] = "qualified"
    events["qualification_authority_sha256"] = kernel.sources.event_authority_sha256
    events["duplicate_group_id"] = "episode-a"
    coverage = pd.DataFrame({"decision_id": rows.decision_id, "coverage_status": "known",
        "available_at_utc": rows.decision_time_utc})
    return {"baseline": baseline, "qualified_events": events, "coverage": coverage, "stock_bars": kernel.stocks,
        "spy_bars": kernel.spy, "history_sessions": kernel.sessions, "sources": kernel.sources}


def build(bundle: dict[str, Any], **changes: Any) -> ResearchFeaturePartition:
    return build_issuer_reaction_profile(**{**bundle, **changes})


def test_parent_preservation_and_real_kernel_hand_calculation(bundle: dict[str, Any]) -> None:
    parent = bundle["baseline"]
    before = parent.rows.copy(deep=True)
    result = build(bundle)
    assert result.model_columns == (*parent.model_columns, *REACTION_COLUMNS)
    assert len(result.model_columns) == 126
    retained = parent.rows.columns.drop("feature_profile")
    pd.testing.assert_frame_equal(result.rows[retained], before[retained])
    pd.testing.assert_frame_equal(parent.rows, before)
    assert result.rows.feature_profile.eq(ISSUER_REACTION_PROFILE).all()
    assert result.rows[REACTION_COLUMNS[0]].iloc[0] == pytest.approx(0.08)
    assert result.rows[REACTION_COLUMNS[1]].iloc[0] == pytest.approx(4.0)
    for name in REACTION_COLUMNS:
        assert result.rows[name].dtype == np.dtype("float32")
        assert str(result.rows[f"available_at_{name}"].dtype) == "datetime64[ns, UTC]"
    assert all(result.audit[name] is False for name in ("training_eligible", "promotion_eligible", "serving_eligible"))


@pytest.mark.parametrize("status", ["rejected", "unclassified"])
def test_known_revision_suppresses_old_qualified_version(bundle: dict[str, Any], status: str) -> None:
    revision = bundle["qualified_events"].copy()
    revision["event_version_sha256"] = "b" * 64
    revision["event_available_at_utc"] += pd.Timedelta(minutes=1)
    revision["qualification_status"] = status
    revision["event_family"] = None
    result = build(bundle, qualified_events=pd.concat([bundle["qualified_events"], revision], ignore_index=True))
    assert result.rows.selected_event_id.isna().all()
    assert result.rows[f"missing_reason_{REACTION_COLUMNS[0]}"].eq("no_qualified_event_in_lookback").all()


def test_future_revision_cannot_erase_asof_version(bundle: dict[str, Any]) -> None:
    revision = bundle["qualified_events"].copy()
    revision["event_version_sha256"] = "b" * 64
    revision["event_available_at_utc"] = bundle["baseline"].rows.decision_time_utc + pd.Timedelta(seconds=1)
    revision["qualification_status"] = "rejected"
    result = build(bundle, qualified_events=pd.concat([revision, bundle["qualified_events"]], ignore_index=True))
    assert result.rows.selected_event_version_sha256.iloc[0] == "a" * 64


def test_latest_event_selected_before_reaction_completeness(bundle: dict[str, Any]) -> None:
    newer = bundle["qualified_events"].copy()
    newer["event_id"], newer["duplicate_group_id"], newer["event_version_sha256"] = "newer", "new", "b" * 64
    newer["event_available_at_utc"] = bundle["baseline"].rows.decision_time_utc - pd.Timedelta(minutes=1)
    result = build(bundle, qualified_events=pd.concat([bundle["qualified_events"], newer], ignore_index=True))
    assert result.rows.selected_event_id.iloc[0] == "newer"
    assert result.rows[list(REACTION_COLUMNS)].isna().all(axis=None)
    assert result.rows[f"missing_reason_{REACTION_COLUMNS[0]}"].iloc[0] == "reaction_session_not_complete_at_decision"


def test_duplicate_earliest_availability_and_order_invariance(bundle: dict[str, Any]) -> None:
    duplicate = bundle["qualified_events"].copy()
    duplicate["source_family"], duplicate["event_id"] = "sec", "duplicate"
    duplicate["event_available_at_utc"] += pd.Timedelta(hours=5)
    events = pd.concat([duplicate, bundle["qualified_events"], duplicate], ignore_index=True)
    first, second = build(bundle, qualified_events=events), build(bundle, qualified_events=events.iloc[::-1])
    assert first.rows.selected_event_id.iloc[0] == "event-a"
    pd.testing.assert_frame_equal(first.rows, second.rows)


def test_old_duplicate_cannot_extend_lookback(bundle: dict[str, Any]) -> None:
    older = bundle["qualified_events"].copy()
    older["event_id"] = "original"
    older["event_available_at_utc"] = bundle["baseline"].rows.decision_time_utc - pd.Timedelta(days=4)
    result = build(bundle, qualified_events=pd.concat([older, bundle["qualified_events"]], ignore_index=True))
    assert result.rows.selected_event_id.isna().all()


@pytest.mark.parametrize(("offset", "selected"), [(pd.Timedelta(days=3), False),
    (pd.Timedelta(days=3) - pd.Timedelta(nanoseconds=1), True), (pd.Timedelta(0), True),
    (-pd.Timedelta(nanoseconds=1), False)])
def test_lookback_endpoints(bundle: dict[str, Any], offset: pd.Timedelta, selected: bool) -> None:
    events = bundle["qualified_events"].copy()
    events["event_available_at_utc"] = bundle["baseline"].rows.decision_time_utc - offset
    result = build(bundle, qualified_events=events)
    assert bool(result.rows.selected_event_id.notna().iloc[0]) == selected


@pytest.mark.parametrize("status", ["known", "unknown"])
def test_no_event_is_null_with_explicit_coverage_reason(bundle: dict[str, Any], status: str) -> None:
    coverage = bundle["coverage"].assign(coverage_status=status)
    result = build(bundle, qualified_events=bundle["qualified_events"].iloc[:0], coverage=coverage)
    assert result.rows[list(REACTION_COLUMNS)].isna().all(axis=None)
    expected = "no_qualified_event_in_lookback" if status == "known" else "unknown_source_coverage"
    assert result.rows[f"missing_reason_{REACTION_COLUMNS[0]}"].iloc[0] == expected


def test_unknown_coverage_retains_individually_qualified_event(bundle: dict[str, Any]) -> None:
    coverage = bundle["coverage"].assign(coverage_status="unknown", available_at_utc=pd.NaT)
    result = build(bundle, coverage=coverage)
    assert result.rows[REACTION_COLUMNS[0]].notna().all()
    assert result.rows.reaction_coverage_status.eq("unknown").all()


@pytest.mark.parametrize(("field", "value"), [("qualification_authority_sha256", "0" * 64),
    ("security_id", "foreign"), ("ticker", "WRONG"), ("source_family", "finviz"),
    ("event_version_sha256", "bad"), ("qualification_status", "admitted"), ("event_family", "analyst_revision")])
def test_invalid_source_and_identity_rejected(bundle: dict[str, Any], field: str, value: str) -> None:
    events = bundle["qualified_events"].copy()
    events[field] = value
    with pytest.raises(DataReadinessError):
        build(bundle, qualified_events=events)


def test_conflicting_same_version_rejected(bundle: dict[str, Any]) -> None:
    conflict = bundle["qualified_events"].assign(qualification_status="rejected")
    with pytest.raises(DataReadinessError, match="contradictory"):
        build(bundle, qualified_events=pd.concat([bundle["qualified_events"], conflict], ignore_index=True))


@pytest.mark.parametrize("poison", ["duplicate", "missing", "future", "known_without_clock"])
def test_coverage_population_and_clocks(bundle: dict[str, Any], poison: str) -> None:
    coverage = bundle["coverage"].copy()
    if poison == "duplicate":
        coverage = pd.concat([coverage, coverage], ignore_index=True)
    elif poison == "missing":
        coverage = coverage.iloc[:0]
    elif poison == "future":
        coverage["available_at_utc"] += pd.Timedelta(seconds=1)
    else:
        coverage["available_at_utc"] = pd.NaT
    with pytest.raises(DataReadinessError):
        build(bundle, coverage=coverage)


def test_live_refuses_proxy_even_without_events(bundle: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="live"):
        build(bundle, qualified_events=bundle["qualified_events"].iloc[:0], purpose="live_construction")


def test_shared_live_and_historical_observed_values(bundle: dict[str, Any]) -> None:
    sources = bundle["sources"].model_copy(update={
        "bars": bundle["sources"].bars.model_copy(update={"availability_semantics": "observed"}),
        "event_availability_semantics": "observed", "identity_availability_semantics": "observed"})
    changes = {"sources": sources, "stock_bars": bundle["stock_bars"].assign(availability_policy="observed"),
        "spy_bars": bundle["spy_bars"].assign(availability_policy="observed")}
    pd.testing.assert_frame_equal(build(bundle, **changes).rows, build(bundle, **changes, purpose="live_construction").rows)


@pytest.mark.parametrize("poison", ["duplicate", "cutoff", "clock", "names", "collision"])
def test_parent_identity_and_inputs_are_strict(bundle: dict[str, Any], poison: str) -> None:
    parent = bundle["baseline"]
    rows = parent.rows.copy()
    if poison == "duplicate":
        rows = pd.concat([rows, rows], ignore_index=True)
    elif poison == "cutoff":
        rows["decision_time_utc"] += pd.Timedelta(seconds=1)
    elif poison == "clock":
        rows["parent_clock"] = rows.decision_time_utc + pd.Timedelta(seconds=1)
    elif poison == "collision":
        rows[REACTION_COLUMNS[0]] = 0.0
    else:
        parent = replace(parent, model_columns=parent.model_columns[::-1])
    with pytest.raises((DataReadinessError, ValueError)):
        build(bundle, baseline=replace(parent, rows=rows))


def test_profile_hash_binds_sources(bundle: dict[str, Any]) -> None:
    names, sources = bundle["baseline"].model_columns, bundle["sources"]
    original = issuer_reaction_profile_sha256(names, sources)
    changed = sources.model_copy(update={"event_authority_sha256": "a" * 64})
    assert original != issuer_reaction_profile_sha256(names, changed)


@pytest.mark.parametrize("infer_string", [False, True])
def test_multiple_decisions_preserve_index_and_match_single_decision_calls(bundle: dict[str, Any], infer_string: bool) -> None:
    parent = bundle["baseline"]
    next_day = bundle["history_sessions"][bundle["history_sessions"].index(parent.rows.session_date_et.iloc[0]) + 1]
    later = parent.rows.copy()
    later["decision_id"] = "later-decision"
    later["session_date_et"] = next_day
    later["decision_time_utc"] = swing_prediction_cutoffs(later.session_date_et)
    later["parent_clock"] = later.decision_time_utc - pd.Timedelta(minutes=5)
    rows = pd.concat([later, parent.rows], ignore_index=True)
    rows.index = [19, 7]
    coverage = pd.DataFrame({"decision_id": rows.decision_id, "coverage_status": "known",
        "available_at_utc": rows.decision_time_utc})
    with pd.option_context("future.infer_string", infer_string):
        full = build(bundle, baseline=replace(parent, rows=rows), coverage=coverage.iloc[::-1])
        singles = [build(bundle, baseline=replace(parent, rows=rows.iloc[[position]]),
            coverage=coverage.iloc[[position]]).rows for position in range(2)]
    pd.testing.assert_frame_equal(full.rows, pd.concat(singles))
    with pd.option_context("future.infer_string", not infer_string):
        other = build(bundle, baseline=replace(parent, rows=rows), coverage=coverage.iloc[::-1])
    pd.testing.assert_frame_equal(full.rows, other.rows)
    for name in ("feature_profile", "reaction_coverage_status", "selected_source_family", "selected_event_id",
                 "selected_event_version_sha256", "selected_event_family", "selected_duplicate_group_id",
                 "selected_qualification_authority_sha256", *(f"missing_reason_{name}" for name in REACTION_COLUMNS)):
        assert full.rows[name].dtype == pd.StringDtype(storage="python")
    assert full.rows.reaction_session_date_et.dtype == np.dtype("object")
    assert full.rows.index.tolist() == [19, 7]
    assert full.rows.selected_event_id.isna().tolist() == [True, False]


def test_missing_selected_bar_never_substitutes_an_older_event(bundle: dict[str, Any]) -> None:
    older = bundle["qualified_events"].copy()
    older["event_id"], older["duplicate_group_id"] = "older", "older-episode"
    older["event_available_at_utc"] -= pd.Timedelta(days=2)
    day = bundle["baseline"].rows.session_date_et.iloc[0]
    stock = bundle["stock_bars"].loc[~bundle["stock_bars"].session_date_et.eq(day)]
    result = build(bundle, qualified_events=pd.concat([older, bundle["qualified_events"]], ignore_index=True), stock_bars=stock)
    assert result.rows.selected_event_id.iloc[0] == "event-a"
    assert result.rows[list(REACTION_COLUMNS)].isna().all(axis=None)
    assert result.rows[f"missing_reason_{REACTION_COLUMNS[0]}"].iloc[0] == "missing_required_session_or_value"


def test_identity_after_decision_cannot_authorize_event(bundle: dict[str, Any]) -> None:
    events = bundle["qualified_events"].copy()
    events["identity_available_at_utc"] = bundle["baseline"].rows.decision_time_utc + pd.Timedelta(seconds=1)
    assert build(bundle, qualified_events=events).rows.selected_event_id.isna().all()


def test_equal_time_event_selection_is_independent_of_input_order(bundle: dict[str, Any]) -> None:
    other = bundle["qualified_events"].assign(event_id="aaa-earliest-tie", duplicate_group_id="independent")
    events = pd.concat([bundle["qualified_events"], other], ignore_index=True)
    first, second = build(bundle, qualified_events=events), build(bundle, qualified_events=events.iloc[::-1])
    assert first.rows.selected_event_id.iloc[0] == "aaa-earliest-tie"
    pd.testing.assert_frame_equal(first.rows, second.rows)
