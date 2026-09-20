from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.swing.contracts.return_feature_profiles import (
    RETURN_RELATIONSHIP_COLUMNS,
    ReturnRelationshipSources,
    return_relationship_profile_sha256,
)
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.features.research_join import DECISION_KEYS, join_research_features_and_outcomes
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from market_predictor.swing.features.return_relationships import build_return_relationship_profile
from tests.test_swing_research_partition import _assemble, _inputs


@dataclass
class Inputs:
    decisions: pd.DataFrame
    baseline: ResearchFeaturePartition
    stocks: pd.DataFrame
    spy: pd.DataFrame
    sessions: tuple[date, ...]
    contract: StrategyContract
    sources: ReturnRelationshipSources


@pytest.fixture(scope="module")
def contract() -> StrategyContract:
    return load_strategy_contract(Path(__file__).resolve().parents[1] / "configs/edge_rebuild_strategy_contract.toml")


def _bars(sessions: tuple[date, ...], ticker: str) -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    position = np.arange(len(sessions), dtype=float)
    close = (100.0 + 0.3 * position + np.sin(position / 9.0)) if ticker == "AAA" else (300.0 + 0.2 * position)
    end = pd.to_datetime([calendar.session_close(pd.Timestamp(day)) for day in sessions], utc=True)
    frame = pd.DataFrame({
        "ticker": ticker, "session_date_et": sessions,
        "bar_start_utc": pd.to_datetime([calendar.session_open(pd.Timestamp(day)) for day in sessions], utc=True),
        "bar_end_utc": end, "available_at_utc": end + pd.Timedelta(minutes=15),
        "open": close * (0.99 if ticker == "AAA" else 0.995), "close": close,
        "volume": 1000.0 + position * 10.0, "price_feed": "sip", "adjustment": "all",
        "source": "alpaca", "timeframe": "1d", "availability_policy": "market_interval_close",
        "ingested_at_utc": end + pd.Timedelta(minutes=15),
    })
    if ticker == "AAA":
        frame["security_id"] = "security-a"
    return frame


@pytest.fixture
def inputs(contract: StrategyContract) -> Inputs:
    sessions = tuple(stamp.date() for stamp in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2023-06-30"))[:280]
    days = [sessions[index] for index in (249, 250, 251, 252, 259)]
    decisions = pd.DataFrame({
        "decision_id": [f"decision-{index}" for index in range(len(days))],
        "security_id": "security-a", "ticker": "AAA", "session_date_et": days,
        "decision_time_utc": swing_prediction_cutoffs(pd.Series(days)),
    })
    names = tuple(swing_model_feature_columns(contract=contract, catalyst=False))
    values = pd.DataFrame(np.full((len(days), len(names)), 0.125, dtype=np.float32), columns=names)
    base = pd.concat([decisions, values], axis=1)
    base["baseline_available_at_utc"] = (decisions.decision_time_utc - pd.Timedelta(minutes=30)).dt.as_unit("ns")
    base["feature_profile"] = "technical_market"
    base["feature_eligible"] = True
    base["fixed_horizon_net_return"] = [np.nan, 0.1, -0.2, 0.3, 0.0]
    partition = ResearchFeaturePartition(base, names, dict.fromkeys(names, "baseline_available_at_utc"), {"training_eligible": False})
    sources = ReturnRelationshipSources(
        baseline_authority_sha256="1" * 64, stock_authority_sha256="2" * 64, spy_authority_sha256="3" * 64,
        availability_semantics="historical_proxy", availability_policy_id="test-approved-historical-proxy",
        availability_policy_sha256="4" * 64, price_adjustment="all", price_basis_and_vintage_id="test-frozen-vintage",
    )
    return Inputs(decisions, partition, _bars(sessions, "AAA"), _bars(sessions, "SPY"), sessions, contract, sources)


def _build(inputs: Inputs, *, live: bool = False) -> ResearchFeaturePartition:
    return build_return_relationship_profile(
        expected_decisions=inputs.decisions, baseline=inputs.baseline,
        stock_bars=inputs.stocks, spy_bars=inputs.spy, history_sessions=inputs.sessions,
        contract=inputs.contract, sources=inputs.sources,
        purpose="live_construction" if live else "historical_research",
    )


def test_exact_four_formulas_and_frozen_order(inputs: Inputs) -> None:
    result = _build(inputs)
    t = 259
    stock, spy = inputs.stocks, inputs.spy
    expected = (
        stock.close.iloc[t - 21] / stock.close.iloc[t - 126] - 1,
        stock.close.iloc[t - 21] / stock.close.iloc[t - 252] - 1,
        stock.volume.iloc[t] / stock.volume.iloc[t - 20:t].mean()
        * (stock.close.iloc[t] / stock.open.iloc[t] - spy.close.iloc[t] / spy.open.iloc[t]),
        (stock.close.iloc[t] / stock.close.iloc[t - 5] - spy.close.iloc[t] / spy.close.iloc[t - 5])
        * (spy.close.iloc[t] / spy.close.iloc[t - 60] - 1),
    )
    assert result.model_columns == (*inputs.baseline.model_columns, *RETURN_RELATIONSHIP_COLUMNS)
    assert len(result.model_columns) == 124
    assert all(result.rows[name].dtype == np.dtype("float32") for name in RETURN_RELATIONSHIP_COLUMNS)
    np.testing.assert_allclose(result.rows.iloc[-1][list(RETURN_RELATIONSHIP_COLUMNS)].to_numpy(float), expected)
    for name in RETURN_RELATIONSHIP_COLUMNS:
        assert result.rows.iloc[-1][f"available_at_{name}"] == stock.available_at_utc.iloc[t]


def test_250_warmup_does_not_shorten_253_position_momentum(inputs: Inputs) -> None:
    result = _build(inputs)
    name = RETURN_RELATIONSHIP_COLUMNS[1]
    assert result.rows[name].iloc[:3].isna().all()
    assert result.rows[f"available_at_{name}"].iloc[:3].isna().all()
    assert result.rows[f"missing_reason_{name}"].iloc[:3].eq("baseline_warmup_250_does_not_supply_253_positions").all()
    assert result.rows[name].iloc[3:].notna().all()
    assert result.audit["minimum_session_positions"][name] == 253
    assert result.audit["coverage"][name]["missing_rows"] == 3


def test_baseline_population_values_clocks_labels_and_inputs_preserved(inputs: Inputs) -> None:
    original = inputs.baseline.rows.copy(deep=True)
    original_stock = inputs.stocks.copy(deep=True)
    inputs.baseline = replace(inputs.baseline, rows=inputs.baseline.rows.iloc[::-1])
    result = _build(inputs)
    columns = original.columns.drop("feature_profile")
    pd.testing.assert_frame_equal(result.rows.loc[:, columns], original.loc[:, columns])
    pd.testing.assert_frame_equal(inputs.stocks, original_stock)
    pd.testing.assert_frame_equal(inputs.baseline.rows, original.iloc[::-1])
    assert result.rows.feature_profile.eq("technical_relationships").all()
    assert result.audit["population_preserved"] is True
    assert result.audit["outcome_filtered_rows"] == 0
    for name in ("training_eligible", "promotion_eligible", "serving_eligible"):
        assert result.audit[name] is False


def test_shared_batch_live_builder_and_generic_outcome_join(inputs: Inputs) -> None:
    inputs.sources = inputs.sources.model_copy(update={"availability_semantics": "observed"})
    for frame in (inputs.stocks, inputs.spy):
        frame["availability_policy"] = "observed"
    batch = _build(inputs)
    inputs.decisions = inputs.decisions.iloc[[-1]].reset_index(drop=True)
    inputs.baseline = replace(inputs.baseline, rows=inputs.baseline.rows.iloc[[-1]].reset_index(drop=True))
    live = _build(inputs, live=True)
    pd.testing.assert_frame_equal(live.rows, batch.rows.iloc[[-1]].reset_index(drop=True))
    outcomes = inputs.decisions.loc[:, list(DECISION_KEYS)].assign(fixed_horizon_net_return=np.nan)
    joined = join_research_features_and_outcomes(
        live.rows, outcomes, feature_columns=live.model_columns,
        availability_columns=live.availability_columns, retained_security_ids=frozenset({"security-a"}),
    )
    assert len(joined) == 1 and joined.fixed_horizon_net_return.isna().all()
    assert joined[list(live.model_columns)].notna().all(axis=None)


def test_historical_proxy_is_research_usable_not_live_admission(inputs: Inputs) -> None:
    assert _build(inputs).rows[RETURN_RELATIONSHIP_COLUMNS[0]].notna().all()
    with pytest.raises(DataReadinessError, match="historical proxy"):
        _build(inputs, live=True)


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_missing_session_is_not_a_surviving_row_shift(inputs: Inputs, source: str) -> None:
    frame = getattr(inputs, source)
    setattr(inputs, source, frame.drop(index=248))
    result = _build(inputs)
    affected = RETURN_RELATIONSHIP_COLUMNS[:3] if source == "stocks" else (RETURN_RELATIONSHIP_COLUMNS[3],)
    for name in affected:
        assert result.rows[name].isna().all()
    if source == "spy":
        assert result.rows[RETURN_RELATIONSHIP_COLUMNS[2]].notna().all()
    else:
        assert pd.notna(result.rows.iloc[-1][RETURN_RELATIONSHIP_COLUMNS[3]])
    assert len(result.rows) == len(inputs.decisions)


@pytest.mark.parametrize("source,column", [("stocks", "close"), ("stocks", "volume"), ("spy", "close")])
def test_missing_individual_values_remain_null(inputs: Inputs, source: str, column: str) -> None:
    getattr(inputs, source).loc[259, column] = np.nan
    result = _build(inputs)
    name = RETURN_RELATIONSHIP_COLUMNS[2]
    assert pd.isna(result.rows.iloc[-1][name])
    assert result.rows.iloc[-1][f"missing_reason_{name}"] == "missing_required_session_or_value"


@pytest.mark.parametrize("source", ["stocks", "spy"])
@pytest.mark.parametrize("clock", ["available_at_utc", "bar_start_utc", "bar_end_utc", "ingested_at_utc"])
def test_missing_consumed_clocks_are_not_invented(inputs: Inputs, source: str, clock: str) -> None:
    getattr(inputs, source).loc[259, clock] = pd.NaT
    result = _build(inputs)
    name = RETURN_RELATIONSHIP_COLUMNS[2]
    assert pd.isna(result.rows.iloc[-1][name])
    assert pd.isna(result.rows.iloc[-1][f"available_at_{name}"])
    assert result.rows.iloc[-1][f"missing_reason_{name}"] == "missing_required_bar_clock"


@pytest.mark.parametrize("source,index", [("stocks", 245), ("spy", 259), ("spy", 210)])
def test_every_consumed_clock_respects_cutoff_including_nanoseconds(inputs: Inputs, source: str, index: int) -> None:
    getattr(inputs, source).loc[index, "available_at_utc"] = inputs.decisions.decision_time_utc.iloc[-1] + pd.Timedelta(nanoseconds=1)
    result = _build(inputs)
    name = RETURN_RELATIONSHIP_COLUMNS[3 if source == "spy" and index == 210 else 2]
    assert pd.isna(result.rows.iloc[-1][name])
    assert result.rows.iloc[-1][f"missing_reason_{name}"] == "required_bar_available_after_decision"


def test_future_values_and_outcomes_cannot_change_past_features(inputs: Inputs) -> None:
    original = _build(inputs)
    for frame in (inputs.stocks, inputs.spy):
        frame.loc[260:, ["open", "close", "volume"]] = 1e10
    inputs.baseline.rows["fixed_horizon_net_return"] = -99999.0
    result = _build(inputs)
    columns = [*result.model_columns, *set(result.availability_columns.values())]
    pd.testing.assert_frame_equal(original.rows[columns], result.rows[columns])


def test_excluded_recent_prices_do_not_enter_skip_month_momentum(inputs: Inputs) -> None:
    original = _build(inputs)
    inputs.stocks.loc[239:259, "close"] *= 2
    result = _build(inputs)
    for name in RETURN_RELATIONSHIP_COLUMNS[:2]:
        assert result.rows.iloc[-1][name] == original.rows.iloc[-1][name]


def test_current_volume_excluded_from_baseline_and_zero_semantics(inputs: Inputs) -> None:
    name = RETURN_RELATIONSHIP_COLUMNS[2]
    original = _build(inputs).rows.iloc[-1][name]
    inputs.stocks.loc[259, "volume"] *= 2
    assert _build(inputs).rows.iloc[-1][name] == pytest.approx(2 * original)
    inputs.stocks.loc[259, "volume"] = 0
    assert _build(inputs).rows.iloc[-1][name] == 0
    inputs.stocks.loc[239:258, "volume"] = 0
    result = _build(inputs)
    assert pd.isna(result.rows.iloc[-1][name])
    assert result.rows.iloc[-1][f"missing_reason_{name}"] == "zero_lagged_volume_baseline"


@pytest.mark.parametrize("source,column", [("stocks", "security_id"), ("stocks", "available_at_utc"),
    ("spy", "open"), ("spy", "volume"), ("spy", "price_feed")])
def test_missing_required_columns_error(inputs: Inputs, source: str, column: str) -> None:
    setattr(inputs, source, getattr(inputs, source).drop(columns=column))
    with pytest.raises(DataReadinessError, match="requires unique columns"):
        _build(inputs)


@pytest.mark.parametrize("source,column,value", [("stocks", "price_feed", "iex"), ("spy", "adjustment", "raw"),
    ("spy", "ticker", "QQQ"), ("stocks", "security_id", "foreign"), ("stocks", "ticker", "OTHER")])
def test_source_or_identity_substitution_rejected(inputs: Inputs, source: str, column: str, value: str) -> None:
    getattr(inputs, source).loc[259, column] = value
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_nonmatching_open_close_interval_rejected(inputs: Inputs, source: str) -> None:
    getattr(inputs, source).loc[259, "bar_start_utc"] += pd.Timedelta(minutes=1)
    with pytest.raises(DataReadinessError, match="complete exchange session"):
        _build(inputs)


def test_calendar_cannot_skip_an_exchange_session(inputs: Inputs) -> None:
    inputs.sessions = inputs.sessions[:248] + inputs.sessions[249:]
    with pytest.raises(DataReadinessError, match="every XNYS session"):
        _build(inputs)


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_duplicate_bar_rejected(inputs: Inputs, source: str) -> None:
    frame = getattr(inputs, source)
    setattr(inputs, source, pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(DataReadinessError, match="duplicate"):
        _build(inputs)


def test_empty_but_schema_present_stock_history_preserves_null_decisions(inputs: Inputs) -> None:
    inputs.stocks = inputs.stocks.iloc[:0]
    result = _build(inputs)
    assert len(result.rows) == len(inputs.decisions)
    assert result.rows[list(RETURN_RELATIONSHIP_COLUMNS)].isna().all(axis=None)


def test_baseline_missingness_is_not_imputed(inputs: Inputs) -> None:
    column = inputs.baseline.model_columns[0]
    inputs.baseline.rows.loc[0, column] = np.nan
    result = _build(inputs)
    assert pd.isna(result.rows.loc[0, column])
    assert result.rows[RETURN_RELATIONSHIP_COLUMNS[0]].notna().all()


def test_baseline_order_population_and_clock_poison_rejected(inputs: Inputs) -> None:
    original = inputs.baseline
    inputs.baseline = replace(original, model_columns=original.model_columns[::-1])
    with pytest.raises(DataReadinessError, match="120 ordered"):
        _build(inputs)
    inputs.baseline = replace(original, rows=original.rows.iloc[1:])
    with pytest.raises(DataReadinessError, match="complete expected population"):
        _build(inputs)
    inputs.baseline = original
    original.rows.loc[0, "baseline_available_at_utc"] = inputs.decisions.decision_time_utc.iloc[0] + pd.Timedelta(nanoseconds=1)
    with pytest.raises(DataReadinessError, match="unavailable at its cutoff"):
        _build(inputs)


def test_existing_output_collision_rejected(inputs: Inputs) -> None:
    inputs.baseline.rows[RETURN_RELATIONSHIP_COLUMNS[0]] = 0.5
    with pytest.raises(DataReadinessError, match="already exist"):
        _build(inputs)


def test_profile_hash_binds_order_sources_and_availability_semantics(inputs: Inputs) -> None:
    columns, sources = inputs.baseline.model_columns, inputs.sources
    original = return_relationship_profile_sha256(columns, sources)
    assert original == return_relationship_profile_sha256(columns, sources)
    assert original != return_relationship_profile_sha256(columns[::-1], sources)
    for change in ({"stock_authority_sha256": "a" * 64}, {"availability_semantics": "observed"},
        {"price_basis_and_vintage_id": "different-vintage"}):
        assert original != return_relationship_profile_sha256(columns, sources.model_copy(update=change))
    with pytest.raises(ValueError, match="120 distinct"):
        return_relationship_profile_sha256(columns[:-1], sources)
    with pytest.raises(ValidationError):
        ReturnRelationshipSources.model_validate({**sources.model_dump(), "stock_authority_sha256": "unverified"})


@pytest.mark.parametrize("source", ["stocks", "spy"])
@pytest.mark.parametrize("column", ["open", "close", "volume"])
@pytest.mark.parametrize("value", [True, np.bool_(False), "123.0"])
def test_numeric_type_masquerades_rejected(inputs: Inputs, source: str, column: str, value: object) -> None:
    frame = getattr(inputs, source)
    frame[column] = frame[column].astype(object)
    frame.loc[259, column] = value
    with pytest.raises(DataReadinessError, match="never boolean or string"):
        _build(inputs)


def test_relationship_float32_overflow_is_explicit_null_not_infinity(inputs: Inputs) -> None:
    inputs.stocks.loc[259, "volume"] = 1e50
    name = RETURN_RELATIONSHIP_COLUMNS[2]
    result = _build(inputs)
    assert pd.isna(result.rows.iloc[-1][name])
    assert result.rows.iloc[-1][f"missing_reason_{name}"] == "relationship_exceeds_float32_range"
    assert not np.isinf(result.rows[list(RETURN_RELATIONSHIP_COLUMNS)].to_numpy()).any()


@pytest.mark.parametrize("value", [True, "0.25", 1e50])
def test_baseline_numeric_poison_rejected_without_conversion(inputs: Inputs, value: object) -> None:
    name = inputs.baseline.model_columns[0]
    inputs.baseline.rows[name] = inputs.baseline.rows[name].astype(object)
    inputs.baseline.rows.loc[0, name] = value
    with pytest.raises(DataReadinessError):
        _build(inputs)


def test_midnight_provider_boundaries_are_not_silently_remapped(inputs: Inputs) -> None:
    day = inputs.sessions[259]
    inputs.stocks.loc[259, "bar_start_utc"] = pd.Timestamp(day, tz="America/New_York").tz_convert("UTC")
    with pytest.raises(DataReadinessError, match="complete exchange session"):
        _build(inputs)


def test_existing_canonicalizer_intervals_are_accepted_without_remapping(inputs: Inputs) -> None:
    original = _build(inputs)
    for source in ("stocks", "spy"):
        frame = getattr(inputs, source)
        raw = frame.loc[:, ["ticker", "open", "close", "volume"]].copy()
        raw["timestamp"] = pd.to_datetime([
            pd.Timestamp(day, tz="America/New_York").tz_convert("UTC") for day in inputs.sessions
        ], utc=True)
        raw["high"] = raw[["open", "close"]].max(axis=1) * 1.01
        raw["low"] = raw[["open", "close"]].min(axis=1) * 0.99
        raw["ingested_at_utc"] = frame.available_at_utc + pd.Timedelta(days=1)
        canonical = canonicalize_bars(raw, timeframe="1d", price_feed="sip", adjustment="all")
        canonical["session_date_et"] = canonical.bar_start_utc.dt.tz_convert("America/New_York").dt.date
        if source == "stocks":
            canonical["security_id"] = "security-a"
        setattr(inputs, source, canonical)
    result = _build(inputs)
    pd.testing.assert_frame_equal(result.rows, original.rows)


def test_two_security_batch_keeps_groups_and_decision_order_separate(inputs: Inputs) -> None:
    first = _build(inputs)
    other_stock = inputs.stocks.copy()
    other_stock["security_id"] = "security-b"
    other_stock["ticker"] = "BBB"
    other_stock["close"] *= 1 + np.arange(len(other_stock)) / 1000
    other_stock["open"] *= 1 + np.arange(len(other_stock)) / 1500
    other_decisions = inputs.decisions.assign(security_id="security-b", ticker="BBB")
    other_decisions["decision_id"] = other_decisions.decision_id + "-b"
    other_base = inputs.baseline.rows.assign(security_id="security-b", ticker="BBB")
    other_base["decision_id"] = other_base.decision_id + "-b"
    second_inputs = replace(inputs, decisions=other_decisions, stocks=other_stock,
        baseline=replace(inputs.baseline, rows=other_base))
    second = _build(second_inputs)
    inputs.decisions = pd.concat([other_decisions, inputs.decisions], ignore_index=True)
    inputs.stocks = pd.concat([inputs.stocks, other_stock], ignore_index=True).iloc[::-1]
    inputs.baseline = replace(inputs.baseline, rows=pd.concat([inputs.baseline.rows, other_base], ignore_index=True))
    combined = _build(inputs)
    pd.testing.assert_frame_equal(combined.rows, pd.concat([second.rows, first.rows], ignore_index=True))


def test_missing_spy_history_does_not_hide_stock_momentum(inputs: Inputs) -> None:
    inputs.spy = inputs.spy.iloc[:0]
    result = _build(inputs)
    assert result.rows[RETURN_RELATIONSHIP_COLUMNS[0]].notna().all()
    assert result.rows[list(RETURN_RELATIONSHIP_COLUMNS[2:])].isna().all(axis=None)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), -1.0])
def test_invalid_price_values_rejected(inputs: Inputs, value: float) -> None:
    inputs.stocks.loc[259, "close"] = value
    with pytest.raises(DataReadinessError):
        _build(inputs)


def test_real_research_assembler_auxiliary_clocks_are_preserved(inputs: Inputs) -> None:
    expected, features, outcomes = _inputs()
    partition = _assemble(expected, features, outcomes, inputs.contract)
    assert len(partition.model_columns) == 120
    assert len(partition.availability_columns) == 160
    sessions = tuple(stamp.date() for stamp in xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2024-01-02"))[-253:]
    stock = _bars(sessions, "AAA")
    stocks = pd.concat([
        stock.assign(security_id=row.security_id, ticker=row.ticker) for row in expected.itertuples()
    ], ignore_index=True)
    result = build_return_relationship_profile(
        expected_decisions=expected, baseline=partition, stock_bars=stocks, spy_bars=_bars(sessions, "SPY"),
        history_sessions=sessions, contract=inputs.contract, sources=inputs.sources,
    )
    assert len(result.availability_columns) == 164
    for name, clock in partition.availability_columns.items():
        assert result.availability_columns[name] == clock
        pd.testing.assert_series_equal(result.rows[clock], partition.rows[clock])
    preserved = partition.rows.columns.drop("feature_profile")
    pd.testing.assert_frame_equal(result.rows[preserved], partition.rows[preserved])
    assert result.rows[list(RETURN_RELATIONSHIP_COLUMNS)].notna().all(axis=None)


@pytest.mark.parametrize("source", ["stocks", "spy"])
@pytest.mark.parametrize("column,value", [("source", "yahoo"), ("timeframe", "1h"),
    ("availability_policy", "provider_publication_proxy")])
def test_physical_canonical_source_substitution_rejected(inputs: Inputs, source: str, column: str, value: str) -> None:
    getattr(inputs, source).loc[259, column] = value
    with pytest.raises(DataReadinessError, match="physical"):
        _build(inputs)


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_claimed_observed_cannot_hide_physical_proxy(inputs: Inputs, source: str) -> None:
    inputs.sources = inputs.sources.model_copy(update={"availability_semantics": "observed"})
    for frame in (inputs.stocks, inputs.spy):
        frame["availability_policy"] = "observed"
    getattr(inputs, source).loc[259, "availability_policy"] = "market_interval_close"
    with pytest.raises(DataReadinessError, match="physical availability"):
        _build(inputs, live=True)


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_observed_bar_availability_cannot_precede_ingestion(inputs: Inputs, source: str) -> None:
    inputs.sources = inputs.sources.model_copy(update={"availability_semantics": "observed"})
    for frame in (inputs.stocks, inputs.spy):
        frame["availability_policy"] = "observed"
    getattr(inputs, source).loc[259, "ingested_at_utc"] += pd.Timedelta(days=1)
    with pytest.raises(DataReadinessError, match="precedes ingestion"):
        _build(inputs, live=True)


def test_truncated_prefix_matches_full_batch_without_using_future_rows(inputs: Inputs) -> None:
    full = _build(inputs)
    cutoff = inputs.sessions[259]
    inputs.sessions = inputs.sessions[:260]
    inputs.stocks = inputs.stocks.loc[inputs.stocks.session_date_et.le(cutoff)]
    inputs.spy = inputs.spy.loc[inputs.spy.session_date_et.le(cutoff)]
    prefix = _build(inputs)
    pd.testing.assert_frame_equal(prefix.rows, full.rows)
