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
from market_predictor.research_lane import forward_record, opra_options
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError


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
    sessions = pd.bdate_range("2026-07-15", "2026-11-17").to_numpy().astype("datetime64[D]")
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
        # Bars: S0 has extreme put skew (`p_otm` priced at IV=0.65), S1..S5 have normal skew (`IV=0.31`)
        bars_out: dict[str, list[dict[str, Any]]] = {}
        for sym in str(params.get("symbols", "")).split(","):
            if not sym:
                continue
            if sym.startswith(("SPY", "QQQ")):
                dte_b = 31 if "261218" in sym else 94
                iv_b = 0.24 if dte_b == 31 else 0.20  # Backwardation (`0.24 > 0.20`)!
                p_b, _ = opra_options.bs_price_and_delta(500.0, 500.0, dte_b / 365.0, iv_b, option_type="call")
                bars_out[sym] = [{"t": "2026-11-17T21:00:00Z", "vw": p_b, "c": p_b, "n": 50}]
            elif sym.endswith("C00107000"):
                bars_out[sym] = [{"t": "2026-11-17T21:00:00Z", "vw": px_c_otm, "c": px_c_otm, "n": 20}]
            elif sym.startswith("S0") and sym.endswith("P00090000"):
                p_hi, _ = opra_options.bs_price_and_delta(100.0, 90.0, 31.0 / 365.0, 0.65, option_type="put")
                bars_out[sym] = [{"t": "2026-11-17T21:00:00Z", "vw": p_hi, "c": p_hi, "n": 30}]
            elif sym.endswith("P00090000"):
                p_norm, _ = opra_options.bs_price_and_delta(100.0, 90.0, 31.0 / 365.0, 0.31, option_type="put")
                bars_out[sym] = [{"t": "2026-11-17T21:00:00Z", "vw": p_norm, "c": p_norm, "n": 20}]
        return MagicMock(status_code=200, json=lambda: {"bars": bars_out})

    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))  # type: ignore[call-arg]
    with patch("market_predictor.research_lane.opra_options.requests.get", side_effect=_fake_get), patch(
        "market_predictor.research_lane.opra_options.guard"
    ):
        enriched, opt_df, opt_sum = opra_options.collect_live_opra_options_overlay(panel, scores_df, cfg, options_top_k=6)

    assert opt_sum["opra_backwardation"] is True
    assert float(opt_sum["spy_iv_term_30_90"]) > 1.0
    s0 = enriched.loc[enriched["security_id"] == "ticker:S0"].iloc[0]
    assert bool(s0["falling_knife_put_skew_veto"]) is True
    assert pd.isna(s0["score_reversal_skew_vetoed"])
    assert len(opt_df) == 6


def test_latest_completed_session_cutoff_and_idempotent_skip_existing(tmp_path: Path) -> None:
    cfg = Settings(_env_file=None, ALPACA_API_KEY_ID="k", ALPACA_API_SECRET_KEY=SecretStr("s"))  # type: ignore[call-arg]
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
        )
        assert m1["status"] == "recorded"
        assert m1["requested_as_of_date"] == "latest"
        assert m1["scored_session"] == "2026-11-19"
        assert m1["sealed_holdout_untouched"] is True
        assert len(m1["frozen_prospective_candidates"]) == 2
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
            )
