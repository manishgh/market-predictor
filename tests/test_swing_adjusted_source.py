from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.adjusted_source import (
    BAR_COLUMNS,
    build_adjusted_technical_source,
    expected_adjusted_history_sessions,
    load_combined_adjusted_inventory,
    read_combined_adjusted_bars,
)
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from tests.test_swing_features import contract as contract
from tests.test_swing_predictor_history import _inputs


def _source_inputs() -> dict:
    stock, benchmarks = _inputs()
    stock["decision_id"] = [f"decision-{index}" for index in stock.index]
    parent = stock.iloc[-20:].copy()
    parent["future_net_return_10d"] = 999
    parent["return_20d_xs_rank"] = 999
    member = pd.DataFrame([dict(ticker="AAA", security_id="issuer-a", sector="Technology", industry="software",
        market_cap_bucket="large", liquidity_bucket="liquid", primary_benchmark="XLK", universe_snapshot_id="membership",
        source="synthetic", available_at_utc=pd.Timestamp("2021-12-31T00:00:00Z"),
        effective_from_utc=pd.Timestamp("2021-12-31T00:00:00Z"), effective_to_utc=pd.NaT)])
    raw = parent.copy()
    raw["adjustment"] = "raw"
    raw["close"] = 123.0
    raw["volume"] = 456.0
    return dict(decisions=parent, adjusted_bars=stock.loc[:, list(BAR_COLUMNS)], benchmark_bars=benchmarks,
        memberships=member, raw_decision_bars=raw, security_id="issuer-a",
        expected_history_sessions=tuple(stock.session_date_et))


def test_exact_population_raw_dollar_volume_and_no_old_features(contract: StrategyContract) -> None:
    args = _source_inputs()
    result = build_adjusted_technical_source(**args, contract=contract)
    rows = result.rows
    assert rows.decision_id.tolist() == args["decisions"].decision_id.tolist()
    assert set(result.availability_columns) == set(TECHNICAL_RANKING_FEATURES)
    assert rows.daily_bar_count.max() == 300
    np.testing.assert_allclose(rows.dollar_volume_log, np.log1p(123 * 456))
    assert rows.return_20d.notna().all()
    assert "future_net_return_10d" not in rows and "return_20d_xs_rank" not in rows


@pytest.mark.parametrize("kind", ["stock", "benchmark"])
def test_midpath_gap_preserves_parent_and_nulls_state(kind: str, contract: StrategyContract) -> None:
    args = _source_inputs()
    if kind == "stock":
        args["adjusted_bars"] = args["adjusted_bars"].drop(index=285)
    else:
        args["benchmark_bars"] = args["benchmark_bars"].drop(index=285)
    rows = build_adjusted_technical_source(**args, contract=contract).rows
    assert len(rows) == 20
    assert rows.return_20d.iloc[:5].notna().all()
    assert rows.return_20d.iloc[5:].isna().all()
    assert rows.technical_missing_reasons.iloc[5:].map(bool).all()
    assert not rows.feature_eligible.iloc[5:].any()


def test_missing_raw_keeps_row_and_null_dollar_volume(contract: StrategyContract) -> None:
    args = _source_inputs()
    args["raw_decision_bars"] = args["raw_decision_bars"].iloc[:-1]
    rows = build_adjusted_technical_source(**args, contract=contract).rows
    assert len(rows) == 20 and pd.isna(rows.dollar_volume_log.iloc[-1])
    assert "missing_or_invalid_raw_dollar_volume" in rows.technical_missing_reasons.iloc[-1]


@pytest.mark.parametrize("poison", ["raw", "feed", "membership", "identity", "heldout"])
def test_bad_binding_refused(poison: str, contract: StrategyContract) -> None:
    args = _source_inputs()
    if poison == "raw":
        args["adjusted_bars"]["adjustment"] = "raw"
    elif poison == "feed":
        args["adjusted_bars"]["price_feed"] = "iex"
    elif poison == "membership":
        args["memberships"]["security_id"] = "wrong-issuer"
    elif poison == "identity":
        args["raw_decision_bars"]["ticker"] = "WRONG"
    else:
        args["adjusted_bars"].loc[0, "bar_start_utc"] = pd.Timestamp("2025-01-02T14:30:00Z")
    with pytest.raises(DataReadinessError):
        build_adjusted_technical_source(**args, contract=contract)


def test_future_price_poison_does_not_change_earlier_rows(contract: StrategyContract) -> None:
    args = _source_inputs()
    before = build_adjusted_technical_source(**args, contract=contract).rows
    for name in ("adjusted_bars", "benchmark_bars"):
        bars = args[name]
        cutoff = bars.bar_start_utc.sort_values().iloc[-1]
        bars.loc[bars.bar_start_utc.eq(cutoff), ["open", "high", "low", "close"]] *= 100
    after = build_adjusted_technical_source(**args, contract=contract).rows
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_projected_read_excludes_heldout_numeric_and_labels(tmp_path: Path) -> None:
    args = _source_inputs()
    bars = args["adjusted_bars"].copy()
    later = bars.iloc[:1].copy()
    later["bar_start_utc"] = pd.Timestamp("2025-01-02T14:30:00Z")
    bars = pd.concat([bars, later], ignore_index=True)
    bars["future_net_return_10d"] = 1e99
    path = tmp_path / "bars.parquet"
    bars.to_parquet(path)
    record = dict(path=path.name, ticker="AAA", sha256=file_sha256(path))
    result = read_combined_adjusted_bars(tmp_path, record, security_id="issuer-a")
    assert len(result) == 300 and set(result) == set(BAR_COLUMNS)
    assert result.bar_start_utc.max().date() <= date(2024, 5, 28)
    record["sha256"] = "0" * 64
    with pytest.raises(DataReadinessError, match="hash mismatch"):
        read_combined_adjusted_bars(tmp_path, record, security_id="issuer-a")


@pytest.mark.parametrize("security_id", ["cik:0001415404", "cik:0000798354"])
def test_wrong_old_corrected_stream_never_read(tmp_path: Path, security_id: str) -> None:
    with pytest.raises(DataReadinessError, match="newly replayed"):
        read_combined_adjusted_bars(tmp_path, {"ticker": "AAA"}, security_id=security_id)


@pytest.mark.parametrize("ticker,security_id", [("SATS", "cik:0001415404"), ("FI", "cik:0000798354")])
def test_corrected_stream_keeps_canonical_parent_symbol(ticker: str, security_id: str, contract: StrategyContract) -> None:
    args = _source_inputs()
    args["security_id"] = security_id
    args["adjusted_bars"]["ticker"] = ticker
    for name in ("decisions", "memberships", "raw_decision_bars"):
        args[name]["security_id"] = security_id
    rows = build_adjusted_technical_source(**args, contract=contract).rows
    assert rows.security_id.eq(security_id).all()
    assert rows.ticker.eq("AAA").all()
    assert rows.return_20d.notna().all()


def test_missing_entire_stock_stream_preserves_population(contract: StrategyContract) -> None:
    args = _source_inputs()
    args["adjusted_bars"] = args["adjusted_bars"].iloc[:0]
    rows = build_adjusted_technical_source(**args, contract=contract).rows
    assert len(rows) == 20 and rows.return_20d.isna().all()
    assert rows.technical_missing_reasons.map(lambda value: "adjusted_history_warmup_crosses_missing_session" in value).all()


def test_inventory_metadata_only_and_tamper(tmp_path: Path, contract: StrategyContract, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.evidence.hashing import json_sha256
    from market_predictor.evidence.io import write_json_object

    inputs = dict(price_feed="sip", adjustment="all")
    request = dict(combined_daily_inputs=inputs, request_sha256="request", strategy_contract_sha256=contract.sha256())
    combined_request = json_sha256({**inputs, "parent_materialization_request_sha256": "request"})
    (tmp_path / "combined_daily").mkdir()
    (tmp_path / "final").mkdir()
    write_json_object(tmp_path / "_request.json", request)
    write_json_object(tmp_path / "combined_daily/_manifest.json", dict(request_sha256=combined_request,
        artifacts=[dict(ticker="AAA", path="bars/missing.parquet", sha256="a" * 64)]))
    combined_pin = file_sha256(tmp_path / "combined_daily/_manifest.json")
    write_json_object(tmp_path / "combined_daily/_authority.json", dict(state="complete", artifact_sha256=combined_pin,
        request_sha256=combined_request))
    write_json_object(tmp_path / "final/_manifest.json", dict(request_sha256="request", strategy_contract_sha256=contract.sha256(),
        source=dict(combined_daily_authority_sha256=file_sha256(tmp_path / "combined_daily/_authority.json"))))
    final_pin = file_sha256(tmp_path / "final/_manifest.json")
    write_json_object(tmp_path / "final/_authority.json", dict(state="complete", artifact_sha256=final_pin,
        request_sha256="request", strategy_contract_sha256=contract.sha256()))
    kwargs = dict(request_sha256=file_sha256(tmp_path / "_request.json"), final_manifest_sha256=final_pin,
        final_authority_sha256=file_sha256(tmp_path / "final/_authority.json"), combined_manifest_sha256=combined_pin,
        contract=contract)
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: pytest.fail("numeric read forbidden"))
    assert set(load_combined_adjusted_inventory(tmp_path, **kwargs)) == {"AAA"}
    request["combined_daily_inputs"]["adjustment"] = "raw"
    (tmp_path / "_request.json").unlink()
    write_json_object(tmp_path / "_request.json", request)
    with pytest.raises(DataReadinessError, match="hash mismatch"):
        load_combined_adjusted_inventory(tmp_path, **kwargs)


def test_single_gap_recovers_at_configured_clean_warmup(contract: StrategyContract) -> None:
    args = _source_inputs()
    args["adjusted_bars"] = args["adjusted_bars"].drop(index=35)
    rows = build_adjusted_technical_source(**args, contract=contract).rows
    assert contract.swing.minimum_warmup_sessions == 250
    assert rows.return_20d.iloc[:5].isna().all()
    assert rows.return_20d.iloc[5:].notna().all()
    assert rows.feature_eligible.iloc[5:].all()
    assert rows.technical_missing_reasons.iloc[5:].map(lambda value: not value).all()


def test_xlc_preinception_prefix_does_not_permanently_block(contract: StrategyContract) -> None:
    args = _source_inputs()
    args["memberships"]["primary_benchmark"] = "XLC"
    benchmarks = args["benchmark_bars"]
    benchmarks.loc[benchmarks.ticker.eq("XLK"), "ticker"] = "XLC"
    sessions = args["expected_history_sessions"]
    missing_prefix = benchmarks.ticker.eq("XLC") & benchmarks.bar_start_utc.dt.date.lt(sessions[36])
    args["benchmark_bars"] = benchmarks.loc[~missing_prefix]
    rows = build_adjusted_technical_source(**args, contract=contract).rows
    assert rows.feature_eligible.iloc[:5].eq(False).all()
    assert rows.feature_eligible.iloc[5:].all()
    assert rows.technical_missing_reasons.iloc[5:].map(lambda value: not value).all()


def test_expected_sessions_use_original_ipo_membership_and_retain_sparse_gap() -> None:
    args = _source_inputs()
    members = args["memberships"].copy()
    members["effective_from_utc"] = pd.Timestamp("2022-01-03T05:00:00Z")
    members["effective_to_utc"] = pd.Timestamp("2023-03-15T04:00:00Z")
    gap = args["expected_history_sessions"][35]
    expected = expected_adjusted_history_sessions(security_id="issuer-a", memberships=members,
        sparse_missing_sessions_by_ticker={"AAA": (gap,)})
    assert expected == args["expected_history_sessions"]
    assert gap in expected and expected[0] == date(2022, 1, 3)
    with pytest.raises(DataReadinessError, match="outside ticker membership"):
        expected_adjusted_history_sessions(security_id="issuer-a", memberships=members,
            sparse_missing_sessions_by_ticker={"AAA": (date(2020, 1, 2),)})


@pytest.mark.parametrize("historical_evidence", [True, False])
def test_pre2019_context_comes_only_from_known_historical_membership(
    historical_evidence: bool, contract: StrategyContract, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import exchange_calendars as xcals

    import market_predictor.swing.features.adjusted_source as owner
    from market_predictor.canonical.joins import decisions_from_completed_bars
    from tests.test_swing_features import _canonical_bars

    sessions = xcals.get_calendar("XNYS").sessions_in_range("2018-05-29", "2019-12-31")[:360]
    args = _source_inputs()
    stock = decisions_from_completed_bars(_canonical_bars("AAA", sessions, drift=0.001), mode="swing-nightly")
    stock["security_id"] = "issuer-a"
    stock["decision_id"] = [f"decision-{index}" for index in stock.index]
    args["decisions"] = stock.iloc[-20:].copy()
    args["raw_decision_bars"] = stock.iloc[-20:].assign(adjustment="raw")
    args["adjusted_bars"] = stock.loc[:, list(BAR_COLUMNS)]
    args["expected_history_sessions"] = tuple(stock.session_date_et)
    args["benchmark_bars"] = pd.concat([_canonical_bars(ticker, sessions, drift=0.0005)
        for ticker in ("SPY", "QQQ", "XLK", "XLF")], ignore_index=True)
    modern = args["memberships"].copy()
    modern["effective_from_utc"] = pd.Timestamp("2019-01-02T05:00:00Z")
    modern["available_at_utc"] = pd.Timestamp("2019-01-02T05:00:00Z")
    old = modern.copy()
    old["ticker"] = "OLD"
    old["sector"] = "Financials"
    old["primary_benchmark"] = "XLF"
    old["effective_from_utc"] = pd.Timestamp("2018-05-29T04:00:00Z")
    old["effective_to_utc"] = pd.Timestamp("2019-01-02T05:00:00Z")
    old["available_at_utc"] = pd.Timestamp("2018-05-29T04:00:00Z") if historical_evidence else pd.Timestamp("2019-01-03T05:00:00Z")
    args["memberships"] = pd.concat([old, modern], ignore_index=True)
    original = owner.build_swing_predictor_history
    captured = []

    def capture(history: pd.DataFrame, benchmarks: pd.DataFrame, *, contract: StrategyContract) -> pd.DataFrame:
        captured.append(history.copy())
        return original(history, benchmarks, contract=contract)

    monkeypatch.setattr(owner, "build_swing_predictor_history", capture)
    rows = owner.build_adjusted_technical_source(**args, contract=contract).rows
    history = captured[0]
    warmup = history.loc[history.session_date_et.lt(date(2019, 1, 2))]
    if historical_evidence:
        assert warmup.primary_benchmark.eq("XLF").all()
        assert warmup.ticker.eq("OLD").all()
        assert warmup.membership_available_at_utc.notna().all()
    else:
        assert warmup.primary_benchmark.isna().all()
        assert warmup.membership_available_at_utc.isna().all()
    assert rows.primary_benchmark.eq("XLK").all()
    assert rows.decision_id.tolist() == args["decisions"].decision_id.tolist()
