"""Unit tests for Step 5f: OPRA OCC parser, Black-Scholes IV/Delta solver, ATM Straddle, 25d Skew, and Falling-Knife Veto."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pydantic import SecretStr

from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.research_lane import forward_record, opra_options
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError
from market_predictor.research_lane.prospective_contracts import (
    latest_settled_xnys_session,
    resolve_xnys_session_clock,
)


def test_occ_parser_and_black_scholes_iv_delta_inversion_accuracy() -> None:
    occ = opra_options.parse_occ_option_symbol("CROX261218P00095500")
    assert occ is not None
    assert occ.underlying == "CROX"
    assert occ.expiration == date(2026, 12, 18)
    assert occ.option_type == "put"
    assert occ.strike == pytest.approx(95.5)
    padded = opra_options.parse_occ_option_symbol("AAPL  261218C00200000")
    assert padded is not None and padded.underlying == "AAPL" and padded.strike == pytest.approx(200.0)
    assert opra_options.parse_occ_option_symbol("BADSYMBOL") is None

    for otype, true_iv, k_val in (("call", 0.32, 108.0), ("put", 0.48, 91.0)):
        px, true_delta = opra_options.bs_price_and_delta(100.0, k_val, 30.0 / 365.0, true_iv, option_type=otype)
        solved_iv, solved_delta = opra_options.solve_iv_and_delta(px, 100.0, k_val, 30.0 / 365.0, option_type=otype)
        assert solved_iv is not None and solved_delta is not None
        assert solved_iv == pytest.approx(true_iv, abs=1e-6)
        assert solved_delta == pytest.approx(true_delta, abs=1e-6)


def test_straddle_25d_skew_falling_knife_veto_and_sealed_bar_rejection() -> None:
    as_of = date(2026, 11, 17)
    spot = 100.0
    # Front 10d ATM straddle at K=100 (`iv = 0.35`), plus 30d OTM put (`iv = 0.55`, steep put skew!) vs OTM call (`iv = 0.30`)
    c_atm = opra_options.OccContract("TK261127C00100000", "TK", date(2026, 11, 27), "call", 100.0)
    p_atm = opra_options.OccContract("TK261127P00100000", "TK", date(2026, 11, 27), "put", 100.0)
    c_otm = opra_options.OccContract("TK261218C00107000", "TK", date(2026, 12, 18), "call", 107.0)
    p_otm = opra_options.OccContract("TK261218P00090000", "TK", date(2026, 12, 18), "put", 90.0)

    px_c_atm, _ = opra_options.bs_price_and_delta(spot, 100.0, 10.0 / 365.0, 0.35, option_type="call")
    px_p_atm, _ = opra_options.bs_price_and_delta(spot, 100.0, 10.0 / 365.0, 0.35, option_type="put")
    px_c_otm, _ = opra_options.bs_price_and_delta(spot, 107.0, 31.0 / 365.0, 0.30, option_type="call")
    px_p_otm, _ = opra_options.bs_price_and_delta(spot, 90.0, 31.0 / 365.0, 0.55, option_type="put")

    m = opra_options.compute_symbol_option_metrics(
        [(c_atm, px_c_atm, 25), (p_atm, px_p_atm, 25), (c_otm, px_c_otm, 20), (p_otm, px_p_otm, 20)],
        spot_close=spot,
        as_of_date=as_of,
        atr_pct_14=0.022,
        latest_catalyst_rx_1d=0.08,
    )
    assert m["atm_straddle_pct"] == pytest.approx((px_c_atm + px_p_atm) / spot, abs=1e-6)
    assert m["put_call_skew_25d"] == pytest.approx(0.25, abs=1e-4)
    assert float(m["catalyst_shock_to_straddle"]) > 1.5

    # Sealed date (`<= 2026-07-08`) must raise SealedPeriodError immediately
    with pytest.raises(SealedPeriodError):
        opra_options._fetch_session_option_bars({}, [c_atm], date(2025, 10, 15))

    # Post-as_of_date bar leaked by provider must raise SealedPeriodError immediately
    future_bar_resp = MagicMock(
        status_code=200,
        json=lambda: {"bars": {"TK261127C00100000": [{"t": "2026-11-18T21:00:00Z", "vw": 3.5, "c": 3.5, "n": 10}]}},
    )
    with patch("market_predictor.research_lane.opra_options.requests.get", return_value=future_bar_resp), pytest.raises(
        SealedPeriodError, match="after as_of_date"
    ):
        opra_options._fetch_session_option_bars({}, [c_atm], as_of)

    # Verify `collect_live_opra_options_overlay` triggers `falling_knife_put_skew_veto` on worst-skew stock
    sessions = pd.bdate_range("2025-10-15", "2026-11-17").to_numpy().astype("datetime64[D]")
    rng = np.random.default_rng(19)
    n_s, n_c = len(sessions), 6
    close = np.full((n_s, n_c), 100.0, dtype=np.float32) * np.cumprod(
        1.0 + rng.normal(0.0, 0.01, size=(n_s, n_c)), axis=0
    ).astype(np.float32)
    close[-1, :] = 100.0
    spy_c = np.full(n_s, 500.0, dtype=np.float32)
    sids = tuple(f"ticker:S{i}" for i in range(n_c))
    panel = ResearchPanel(
        sessions=sessions,
        security_ids=sids,
        open=close * np.float32(0.998),
        high=close * np.float32(1.012),
        low=close * np.float32(0.988),
        close=close,
        volume=np.full((n_s, n_c), 1e6, dtype=np.float32),
        eligible=np.ones((n_s, n_c), dtype=bool),
        sector=np.zeros((n_s, n_c), dtype=np.int16),
        sector_names=("Tech",),
        spy_open=spy_c,
        spy_close=spy_c,
        qqq_open=spy_c,
        qqq_close=spy_c,
        development_end=as_of,
        load_report={},
    )
    scores_df = forward_record._score_panel_last_session(panel)
    scores_df["score_reversal_short"] = [0.45, 0.30, 0.15, 0.0, -0.20, -0.40]

    def _fake_get(url: str, **kwargs: Any) -> MagicMock:
        params = kwargs.get("params", {})
        if "/v2/options/contracts" in url:
            u = str(params.get("underlying_symbols", ""))
            if u in ("SPY", "QQQ"):
                return MagicMock(status_code=200, json=lambda: {"option_contracts": [
                    {"symbol": f"{u}261218C00500000"}, {"symbol": f"{u}261218P00500000"},
                    {"symbol": f"{u}270219C00500000"}, {"symbol": f"{u}270219P00500000"},
                ]})
            return MagicMock(status_code=200, json=lambda: {"option_contracts": [
                {"symbol": f"{u}261218C00107000"}, {"symbol": f"{u}261218P00090000"},
            ]})
        if "/v2/stocks/bars" in url:
            u_bars: dict[str, list[dict[str, Any]]] = {}
            for sym in str(params.get("symbols", "")).split(","):
                if not sym:
                    continue
                px_u = 500.0 if sym in ("SPY", "QQQ") else 100.0
                u_bars[sym] = [{"t": "2026-11-17T20:55:00Z", "o": px_u, "h": px_u, "l": px_u, "c": px_u, "v": 10000}]
            return MagicMock(status_code=200, json=lambda: {"bars": u_bars})
        # Option bars (`5Min` at `20:55:00Z` = `15:55..16:00 ET` and `1Day`): S0 has IV=0.65 put skew, S1..S5 have IV=0.31
        tf = str(params.get("timeframe", "1Day"))
        t_stamp = "2026-11-17T20:55:00Z" if tf == "5Min" else "2026-11-17T05:00:00Z"
        bars_out: dict[str, list[dict[str, Any]]] = {}
        for sym in str(params.get("symbols", "")).split(","):
            if not sym:
                continue
            if sym.startswith(("SPY", "QQQ")):
                dte_b = 31 if "261218" in sym else 94
                iv_b = 0.24 if dte_b == 31 else 0.20  # Backwardation (`0.24 > 0.20`)!
                p_b, _ = opra_options.bs_price_and_delta(500.0, 500.0, dte_b / 365.0, iv_b, option_type="call")
                bars_out[sym] = [{"t": t_stamp, "vw": p_b, "c": p_b, "n": 50}]
            elif sym.endswith("C00107000"):
                bars_out[sym] = [{"t": t_stamp, "vw": px_c_otm, "c": px_c_otm, "n": 20}]
            elif sym.startswith("S0") and sym.endswith("P00090000"):
                p_hi, _ = opra_options.bs_price_and_delta(100.0, 90.0, 31.0 / 365.0, 0.65, option_type="put")
                bars_out[sym] = [{"t": t_stamp, "vw": p_hi, "c": p_hi, "n": 30}]
            elif sym.endswith("P00090000"):
                p_norm, _ = opra_options.bs_price_and_delta(100.0, 90.0, 31.0 / 365.0, 0.31, option_type="put")
                bars_out[sym] = [{"t": t_stamp, "vw": p_norm, "c": p_norm, "n": 20}]
        return MagicMock(status_code=200, json=lambda: {"bars": bars_out})

    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))
    with patch("market_predictor.research_lane.opra_options.requests.get", side_effect=_fake_get), patch(
        "market_predictor.research_lane.opra_options.guard"
    ):
        enriched, opt_df, opt_sum = opra_options.collect_live_opra_options_overlay(panel, scores_df, cfg, options_top_k=6)

    assert opt_sum["opra_regime"] == "backwardation"
    assert opt_sum["opra_backwardation"] is True
    assert opt_sum["benchmark_observation_kind"] == "synchronized_close"
    assert float(opt_sum["spy_iv_term_30_90"]) > 1.0
    s0 = enriched.loc[enriched["security_id"] == "ticker:S0"].iloc[0]
    assert bool(s0["falling_knife_put_skew_veto"]) is True
    assert pd.isna(s0["score_reversal_skew_vetoed"])
    assert len(opt_df) == 6


def test_latest_completed_session_cutoff_and_idempotent_skip_existing(tmp_path: Path) -> None:
    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))
    fake_page = MagicMock(
        bars={
            "SPY": [
                {"t": "2026-11-18T05:00:00Z", "o": 590, "h": 592, "l": 588, "c": 591, "v": 1e7},
                {"t": "2026-11-19T05:00:00Z", "o": 591, "h": 595, "l": 590, "c": 594, "v": 1e7},
                {"t": "2026-11-20T05:00:00Z", "o": 594, "h": 596, "l": 592, "c": 595, "v": 1e7},
            ]
        }
    )
    with patch("market_predictor.research_lane.forward_record.AlpacaSource") as mock_alp:
        mock_alp.return_value.fetch_bars_page.return_value = fake_page
        # Intraday Friday 14:00 ET (19:00 UTC < 16:20 ET) must NOT pick incomplete Friday 2026-11-20 -> picks 2026-11-19!
        d_intra = forward_record.resolve_latest_completed_session(cfg, now_utc=datetime(2026, 11, 20, 19, 0, tzinfo=UTC))
        assert d_intra == date(2026, 11, 19)
        # Post-close Friday 16:30 ET (21:30 UTC >= 16:20 ET) picks completed Friday 2026-11-20!
        d_after = forward_record.resolve_latest_completed_session(cfg, now_utc=datetime(2026, 11, 20, 21, 30, tzinfo=UTC))
        assert d_after == date(2026, 11, 20)
        # Sunday afternoon picks completed Friday 2026-11-20!
        d_sun = forward_record.resolve_latest_completed_session(cfg, now_utc=datetime(2026, 11, 22, 17, 0, tzinfo=UTC))
        assert d_sun == date(2026, 11, 20)
        # Sealed date <= 2026-07-08 must raise SealedPeriodError!
        with pytest.raises(SealedPeriodError):
            forward_record.resolve_latest_completed_session(cfg, now_utc=datetime(2026, 7, 8, 22, 0, tzinfo=UTC))

    sessions = pd.bdate_range("2026-07-15", "2026-11-19").to_numpy().astype("datetime64[D]")
    n_s, n_c = len(sessions), 4
    close = np.full((n_s, n_c), 50.0, dtype=np.float32) * np.linspace(1.0, 1.15, n_s, dtype=np.float32)[:, None]
    spy_c = np.full(n_s, 500.0, dtype=np.float32)
    panel = ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"ticker:T{i}" for i in range(n_c)),
        open=close,
        high=close * np.float32(1.01),
        low=close * np.float32(0.99),
        close=close,
        volume=np.full((n_s, n_c), 5e5, dtype=np.float32),
        eligible=np.ones((n_s, n_c), dtype=bool),
        sector=np.zeros((n_s, n_c), dtype=np.int16),
        sector_names=("Tech",),
        spy_open=spy_c,
        spy_close=spy_c,
        qqq_open=spy_c,
        qqq_close=spy_c,
        development_end=date(2026, 11, 19),
        load_report={},
    )
    with patch(
        "market_predictor.research_lane.forward_record.resolve_latest_completed_session",
        return_value=date(2026, 11, 19),
    ), patch(
        "market_predictor.research_lane.forward_record._fetch_live_prospective_panel",
        return_value=panel,
    ) as mock_panel, patch("market_predictor.research_lane.forward_record.guard"), patch.dict(
        "os.environ", {"MARKET_PREDICTOR_RUNTIME_DIR": str(tmp_path / "rt")}
    ):
        m1 = forward_record.record_forward_scores(
            root=tmp_path,
            as_of_date=None,
            live_prospective=True,
            include_live_catalysts=False,
            include_live_options=False,
            skip_existing=True,
            settings=cfg,
            clock_fn=lambda: datetime(2026, 11, 19, 22, 0, tzinfo=UTC),
        )
        assert m1["status"] == "recorded"
        assert m1["record_mode"] == "prospective"
        assert m1["requested_as_of_date"] == "latest"
        assert m1["scored_session"] == "2026-11-19"
        assert m1["published_before_entry"] is True
        assert m1["next_tradable_session"] == "2026-11-20"
        assert m1["next_tradable_open_utc"] == "2026-11-20T14:30:00+00:00"
        assert m1["sealed_holdout_untouched"] is True
        assert len(m1["frozen_prospective_candidates"]) == 3
        assert m1["frozen_prospective_candidates"][0]["candidate_id"] == "mid_small_bn_sec_xgb_10d"
        assert mock_panel.call_count == 1

        # Second call with skip_existing=True short-circuits before panel fetch & does not duplicate log!
        m2 = forward_record.record_forward_scores(
            root=tmp_path,
            as_of_date=None,
            live_prospective=True,
            include_live_catalysts=False,
            include_live_options=False,
            skip_existing=True,
            settings=cfg,
            clock_fn=lambda: datetime(2026, 11, 19, 22, 5, tzinfo=UTC),
        )
        assert m2["status"] == "already_recorded"
        assert mock_panel.call_count == 1
        log_lines = (tmp_path / forward_record.FORWARD_RECORD_DIRECTORY / forward_record.FORWARD_LOG_NAME).read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(log_lines) == 1

        # Default skip_existing=False remains fail-closed (`FileExistsError`)!
        with pytest.raises(FileExistsError):
            forward_record.record_forward_scores(
                root=tmp_path,
                as_of_date=date(2026, 11, 19),
                live_prospective=True,
                include_live_catalysts=False,
                include_live_options=False,
                settings=cfg,
                clock_fn=lambda: datetime(2026, 11, 19, 22, 10, tzinfo=UTC),
            )

        # Late scheduler retry on a new unrecorded session AFTER next_tradable_open_utc (`14:30 UTC`) must fail closed!
        with pytest.raises(DataReadinessError, match="missed next tradable open"):
            forward_record.record_forward_scores(
                root=tmp_path,
                as_of_date=date(2026, 11, 18),
                live_prospective=True,
                include_live_catalysts=False,
                include_live_options=False,
                settings=cfg,
                clock_fn=lambda: datetime(2026, 11, 19, 17, 13, 57, tzinfo=UTC),
            )


def test_xnys_session_clock_weekend_holiday_and_early_close_settlement() -> None:
    # Friday 2026-11-20 -> next tradable open is Monday 2026-11-23 at 14:30 UTC (weekend skipped)
    fri_clk = resolve_xnys_session_clock(date(2026, 11, 20))
    assert fri_clk.is_early_close is False
    assert fri_clk.source_cutoff_utc == datetime(2026, 11, 20, 21, 0, tzinfo=UTC)
    assert fri_clk.next_tradable_session == date(2026, 11, 23)
    assert fri_clk.next_tradable_open_utc == datetime(2026, 11, 23, 14, 30, tzinfo=UTC)

    # Black Friday 2026-11-27 is an early close (13:00 EST = 18:00 UTC, settlement 18:20 UTC, next open Monday 2026-11-30)
    bf_clk = resolve_xnys_session_clock(date(2026, 11, 27))
    assert bf_clk.is_early_close is True
    assert bf_clk.session_close_utc == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    assert bf_clk.source_cutoff_utc == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    assert bf_clk.settlement_ready_utc == datetime(2026, 11, 27, 18, 20, tzinfo=UTC)
    assert bf_clk.next_tradable_session == date(2026, 11, 30)
    assert bf_clk.next_tradable_open_utc == datetime(2026, 11, 30, 14, 30, tzinfo=UTC)

    # Before 18:20 UTC on Black Friday (`18:10 UTC`), latest settled session is Wednesday 2026-11-25 (skipping Thanksgiving!)
    pre_bf = latest_settled_xnys_session(datetime(2026, 11, 27, 18, 10, tzinfo=UTC))
    assert pre_bf.as_of_date == date(2026, 11, 25)
    # After 18:20 UTC on Black Friday (`18:25 UTC`), Black Friday 2026-11-27 itself is settled!
    post_bf = latest_settled_xnys_session(datetime(2026, 11, 27, 18, 25, tzinfo=UTC))
    assert post_bf.as_of_date == date(2026, 11, 27)


def test_r7_tristate_opra_regime_decisive_positive_indeterminate_partial_and_outage(tmp_path: Path) -> None:
    """Verify tri-state OPRA regime (`backwardation`, `contango`, `unknown`) across recording & evaluation (`R7`)."""
    import json

    from market_predictor.research_lane.forward_evaluate import evaluate_forward_scores
    from market_predictor.research_lane.prospective_contracts import (
        extract_snapshot_opra_regime,
        resolve_opra_regime,
    )

    # 1. Pure tri-state truth table for the frozen either-index (`spy > 1.0 or qqq > 1.0`) rule
    assert resolve_opra_regime(1.08, None) == "backwardation"
    assert resolve_opra_regime(None, 1.04) == "backwardation"
    assert resolve_opra_regime(0.93, 0.96) == "contango"
    assert resolve_opra_regime(0.93, None) == "unknown"
    assert resolve_opra_regime(None, 0.96) == "unknown"
    assert resolve_opra_regime(None, None) == "unknown"

    # Legacy manifest with `opra_backwardation: false` when both term ratios were `null` MUST infer `"unknown"`!
    legacy_outage = {
        "options_summary": {
            "opra_enabled": True,
            "spy_iv_term_30_90": None,
            "qqq_iv_term_30_90": None,
            "opra_backwardation": False,
        }
    }
    assert extract_snapshot_opra_regime(legacy_outage) == ("unknown", "legacy_unspecified")

    # 2. Exercise `collect_live_opra_options_overlay` under:
    #    (a) decisive positive (`SPY > 1.0`, `QQQ` outage),
    #    (b) indeterminate partial (`SPY <= 1.0`, `QQQ` outage),
    #    (c) total outage (`SPY` & `QQQ` both fail).
    as_of = date(2026, 11, 17)
    sessions = pd.bdate_range("2025-10-15", "2026-11-17").to_numpy().astype("datetime64[D]")
    n_s, n_c = len(sessions), 6
    close = np.full((n_s, n_c), 100.0, dtype=np.float32)
    spy_c = np.full(n_s, 500.0, dtype=np.float32)
    panel = ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"ticker:S{i}" for i in range(n_c)),
        open=close,
        high=close * np.float32(1.01),
        low=close * np.float32(0.99),
        close=close,
        volume=np.full((n_s, n_c), 1e6, dtype=np.float32),
        eligible=np.ones((n_s, n_c), dtype=bool),
        sector=np.zeros((n_s, n_c), dtype=np.int16),
        sector_names=("Tech",),
        spy_open=spy_c,
        spy_close=spy_c,
        qqq_open=spy_c,
        qqq_close=spy_c,
        development_end=as_of,
        load_report={},
    )
    scores_df = forward_record._score_panel_last_session(panel)
    scores_df["score_mid_small_bn_sec_xgb_10d"] = [0.45, 0.30, 0.10, -0.10, -0.25, -0.45]
    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))

    def _run_scenario(spy_front_iv: float | None, qqq_front_iv: float | None) -> tuple[pd.DataFrame, dict[str, Any]]:
        def _mock_http(url: str, **kwargs: Any) -> MagicMock:
            params = kwargs.get("params", {})
            if "/v2/options/contracts" in url:
                u = str(params.get("underlying_symbols", ""))
                if u in ("SPY", "QQQ"):
                    return MagicMock(status_code=200, json=lambda: {"option_contracts": [
                        {"symbol": f"{u}261218C00500000"}, {"symbol": f"{u}270219C00500000"},
                    ]})
                return MagicMock(status_code=200, json=lambda: {"option_contracts": []})
            if "/v2/stocks/bars" in url:
                u_bars = {
                    s: [{"t": "2026-11-17T20:55:00Z", "o": 500.0, "h": 500.0, "l": 500.0, "c": 500.0, "v": 5000}]
                    for s in str(params.get("symbols", "")).split(",") if s in ("SPY", "QQQ")
                }
                return MagicMock(status_code=200, json=lambda: {"bars": u_bars})
            bars_out: dict[str, list[dict[str, Any]]] = {}
            t_stamp = "2026-11-17T20:55:00Z" if str(params.get("timeframe")) == "5Min" else "2026-11-17T05:00:00Z"
            for sym in str(params.get("symbols", "")).split(","):
                if sym.startswith("SPY") and spy_front_iv is not None:
                    dte_b = 31 if "261218" in sym else 94
                    iv_b = spy_front_iv if dte_b == 31 else 0.20
                    px_b, _ = opra_options.bs_price_and_delta(500.0, 500.0, dte_b / 365.0, iv_b, option_type="call")
                    bars_out[sym] = [{"t": t_stamp, "vw": px_b, "c": px_b, "n": 25}]
                elif sym.startswith("QQQ") and qqq_front_iv is not None:
                    dte_b = 31 if "261218" in sym else 94
                    iv_b = qqq_front_iv if dte_b == 31 else 0.20
                    px_b, _ = opra_options.bs_price_and_delta(500.0, 500.0, dte_b / 365.0, iv_b, option_type="call")
                    bars_out[sym] = [{"t": t_stamp, "vw": px_b, "c": px_b, "n": 25}]
            return MagicMock(status_code=200, json=lambda: {"bars": bars_out})

        with patch("market_predictor.research_lane.opra_options.requests.get", side_effect=_mock_http), patch(
            "market_predictor.research_lane.opra_options.guard"
        ):
            enr, _, summ = opra_options.collect_live_opra_options_overlay(panel, scores_df, cfg, options_top_k=6)
        return enr, summ

    # (a) Decisive positive (`SPY = 0.24 / 0.20 > 1.0`, `QQQ = None`) -> `"backwardation"`
    enr_pos, sum_pos = _run_scenario(0.24, None)
    assert sum_pos["opra_regime"] == "backwardation"
    assert sum_pos["opra_backwardation"] is True
    assert sum_pos["qqq_iv_term_30_90"] is None
    assert enr_pos["score_mid_small_bn_sec_xgb_opra_10d"].notna().all()

    # (b) Indeterminate partial (`SPY = 0.18 / 0.20 < 1.0`, `QQQ = None`) -> `"unknown"` (`abstain`!)
    enr_ind, sum_ind = _run_scenario(0.18, None)
    assert sum_ind["opra_regime"] == "unknown"
    assert sum_ind["opra_backwardation"] is None
    assert enr_ind["score_r2_opra_conditioned"].isna().all()
    assert enr_ind["score_mid_small_bn_sec_xgb_opra_10d"].isna().all()

    # (c) Total outage (`SPY = None`, `QQQ = None`) -> `"unknown"`, and evaluator claims NO measured-regime portfolio!
    enr_out, sum_out = _run_scenario(None, None)
    assert sum_out["opra_regime"] == "unknown"
    assert sum_out["opra_backwardation"] is None
    assert enr_out["score_r2_opra_conditioned"].isna().all()
    assert enr_out["score_mid_small_bn_sec_xgb_opra_10d"].isna().all()

    snap_dir = tmp_path / forward_record.FORWARD_RECORD_DIRECTORY / "2026-11-17"
    snap_dir.mkdir(parents=True)
    enr_out.to_parquet(snap_dir / "scores.parquet", index=False)
    (snap_dir / "_manifest.json").write_text(
        json.dumps({
            "scored_session": "2026-11-17",
            "record_mode": "prospective",
            "published_at_utc": "2026-11-17T22:00:00+00:00",
            "options_summary": sum_out,
        }),
        encoding="utf-8",
    )
    fake_bars = pd.DataFrame([
        {"ticker": tk, "session": d_s, "open": 100.0, "close": 102.0}
        for d_s in ("2026-11-17", "2026-11-18")
        for tk in ("SPY", "S0", "S1", "S2", "S3", "S4", "S5")
    ])
    with patch("market_predictor.research_lane.forward_evaluate._fetch_post_holdout_bars", return_value=fake_bars), patch(
        "market_predictor.research_lane.forward_evaluate.guard"
    ), patch.dict("os.environ", {"MARKET_PREDICTOR_RUNTIME_DIR": str(tmp_path / "rt")}):
        eval_res = evaluate_forward_scores(root=tmp_path, as_of_date=date(2026, 11, 18), horizons=(1,), settings=cfg)
    rep = json.loads((tmp_path / eval_res["output"] / "report.json").read_text(encoding="utf-8"))
    assert rep["pooled_candidates_by_horizon"] == {}
    assert rep["opra_diagnostics"][0]["opra_regime"] == "unknown"
    assert rep["opra_diagnostics"][0]["opra_backwardation"] is None


def test_r10_intraday_underlying_move_synchronized_close_vs_daily_vwap_proxy() -> None:
    """Verify a `$100 -> $120` intraday spot jump at constant `30%` volatility cannot fabricate closing IV/skew from daily VWAP (`R10`)."""
    as_of = date(2026, 11, 17)
    sessions = pd.bdate_range("2025-10-15", "2026-11-17").to_numpy().astype("datetime64[D]")
    n_s, n_c = len(sessions), 6
    close = np.full((n_s, n_c), 100.0, dtype=np.float32)
    # Both S0 and S1 rallied intraday from $100.00 in the morning to $120.00 at the close!
    close[-1, 0] = 120.0
    close[-1, 1] = 120.0
    spy_c = np.full(n_s, 500.0, dtype=np.float32)
    panel = ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"ticker:S{i}" for i in range(n_c)),
        open=np.full((n_s, n_c), 100.0, dtype=np.float32),
        high=close * np.float32(1.01),
        low=np.full((n_s, n_c), 99.0, dtype=np.float32),
        close=close,
        volume=np.full((n_s, n_c), 1e6, dtype=np.float32),
        eligible=np.ones((n_s, n_c), dtype=bool),
        sector=np.zeros((n_s, n_c), dtype=np.int16),
        sector_names=("Tech",),
        spy_open=spy_c,
        spy_close=spy_c,
        qqq_open=spy_c,
        qqq_close=spy_c,
        development_end=as_of,
        load_report={},
    )
    scores_df = forward_record._score_panel_last_session(panel)
    scores_df["score_reversal_short"] = [0.40, 0.35, 0.10, 0.0, -0.15, -0.35]

    true_sigma = 0.30
    dte_t = 31.0 / 365.0
    morn_spot = 112.0
    # Morning (`$112` spot) option prices at constant 30% vol vs Close (`$120` spot) option prices at constant 30% vol:
    morn_put_114, _ = opra_options.bs_price_and_delta(morn_spot, 114.0, dte_t, true_sigma, option_type="put")
    morn_call_126, _ = opra_options.bs_price_and_delta(morn_spot, 126.0, dte_t, true_sigma, option_type="call")
    morn_call_120, _ = opra_options.bs_price_and_delta(morn_spot, 120.0, dte_t, true_sigma, option_type="call")
    morn_put_120, _ = opra_options.bs_price_and_delta(morn_spot, 120.0, dte_t, true_sigma, option_type="put")

    close_put_114, _ = opra_options.bs_price_and_delta(120.0, 114.0, dte_t, true_sigma, option_type="put")
    close_call_126, _ = opra_options.bs_price_and_delta(120.0, 126.0, dte_t, true_sigma, option_type="call")
    close_call_120, _ = opra_options.bs_price_and_delta(120.0, 120.0, dte_t, true_sigma, option_type="call")
    close_put_120, _ = opra_options.bs_price_and_delta(120.0, 120.0, dte_t, true_sigma, option_type="put")

    def _mock_intraday_jump(url: str, **kwargs: Any) -> MagicMock:
        params = kwargs.get("params", {})
        if "/v2/options/contracts" in url:
            u = str(params.get("underlying_symbols", ""))
            if u in ("SPY", "QQQ"):
                return MagicMock(status_code=200, json=lambda: {"option_contracts": [
                    {"symbol": f"{u}261218C00500000"}, {"symbol": f"{u}270219C00500000"},
                ]})
            if u in ("S0", "S1"):
                return MagicMock(status_code=200, json=lambda: {"option_contracts": [
                    {"symbol": f"{u}261218P00114000"},
                    {"symbol": f"{u}261218C00126000"},
                    {"symbol": f"{u}261218C00120000"},
                    {"symbol": f"{u}261218P00120000"},
                ]})
            return MagicMock(status_code=200, json=lambda: {"option_contracts": []})

        if "/v2/stocks/bars" in url:
            u_bars: dict[str, list[dict[str, Any]]] = {}
            for sym in str(params.get("symbols", "")).split(","):
                if sym in ("SPY", "QQQ"):
                    u_bars[sym] = [{"t": "2026-11-17T20:55:00Z", "c": 500.0, "v": 10000}]
                elif sym in ("S0", "S1"):
                    u_bars[sym] = [{"t": "2026-11-17T20:55:00Z", "c": 120.0, "v": 10000}]
            return MagicMock(status_code=200, json=lambda: {"bars": u_bars})

        tf = str(params.get("timeframe", "1Day"))
        bars_out: dict[str, list[dict[str, Any]]] = {}
        for sym in str(params.get("symbols", "")).split(","):
            if sym.startswith(("SPY", "QQQ")):
                dte_b = 31 if "261218" in sym else 94
                px_b, _ = opra_options.bs_price_and_delta(500.0, 500.0, dte_b / 365.0, 0.18, option_type="call")
                t_b = "2026-11-17T20:55:00Z" if tf == "5Min" else "2026-11-17T05:00:00Z"
                bars_out[sym] = [{"t": t_b, "vw": px_b, "c": px_b, "n": 40}]
                continue
            is_s0 = sym.startswith("S0")
            if sym.endswith("P00114000"):
                m_px, c_px = morn_put_114, close_put_114
            elif sym.endswith("C00126000"):
                m_px, c_px = morn_call_126, close_call_126
            elif sym.endswith("C00120000"):
                m_px, c_px = morn_call_120, close_call_120
            elif sym.endswith("P00120000"):
                m_px, c_px = morn_put_120, close_put_120
            else:
                continue
            if tf == "1Day":
                bars_out[sym] = [{"t": "2026-11-17T05:00:00Z", "vw": m_px, "c": c_px if is_s0 else m_px, "n": 200}]
            elif tf == "5Min" and is_s0:
                bars_out[sym] = [
                    {"t": "2026-11-17T20:55:00Z", "vw": m_px, "c": c_px, "n": 15},
                    {"t": "2026-11-17T21:00:00Z", "vw": 99.0, "c": 99.0, "n": 15},  # post-close bar -> ignored!
                ]
        return MagicMock(status_code=200, json=lambda: {"bars": bars_out})

    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))
    with patch("market_predictor.research_lane.opra_options.requests.get", side_effect=_mock_intraday_jump), patch(
        "market_predictor.research_lane.opra_options.guard"
    ):
        enriched, opt_df, opt_sum = opra_options.collect_live_opra_options_overlay(panel, scores_df, cfg, options_top_k=6)

    assert opt_sum["opra_regime"] == "contango"
    assert opt_sum["opra_backwardation"] is False

    # S0 (`synchronized_close` at `20:55Z` with `$120` spot) recovers exact constant `30%` IV & `0.0` skew!
    s0 = enriched.loc[enriched["security_id"] == "ticker:S0"].iloc[0]
    assert s0["option_observation_kind"] == "synchronized_close"
    assert float(s0["atm_iv_30d"]) == pytest.approx(true_sigma, abs=1e-3)
    assert float(s0["put_call_skew_25d"]) == pytest.approx(0.0, abs=1e-3)
    assert bool(s0["falling_knife_put_skew_veto"]) is False
    # Whereas S0's morning-distorted 1Day VWAP proxy would have fabricated > +20% fake put skew!
    assert float(s0["vwap_proxy_put_call_skew_25d"]) > 0.20

    # S1 (morning-only trades, zero closing 5Min bars) CANNOT report daily VWAP as synchronized closing IV/skew!
    s1 = enriched.loc[enriched["security_id"] == "ticker:S1"].iloc[0]
    assert s1["option_observation_kind"] == "daily_vwap_proxy_only"
    assert pd.isna(s1["atm_iv_30d"])
    assert pd.isna(s1["put_call_skew_25d"])
    assert pd.isna(s1["atm_straddle_pct"])
    assert bool(s1["falling_knife_put_skew_veto"]) is False
    assert float(s1["vwap_proxy_put_call_skew_25d"]) > 0.20
    assert len(opt_df) == 6
