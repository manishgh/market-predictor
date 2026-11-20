"""Unit tests for Step 5e: CBOE volatility term-structure truncation, causal regime features, and R1/R2 invariance."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane import benchmark_tilt, vol_regime
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError


def test_parse_cboe_csv_truncates_at_development_end_and_blocks_sealed_dates() -> None:
    csv_text = (
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "05/24/2024,12.1,12.5,11.9,11.93\n"
        "05/28/2024,12.4,13.1,12.2,12.92\n"
        "05/29/2024,13.5,14.5,13.4,14.28\n"
        "08/05/2024,38.0,65.7,28.0,38.57\n"
    )
    df = vol_regime.parse_cboe_csv_truncated(csv_text, series_name="vix", development_end=date(2024, 5, 28))
    assert df["session_date"].tolist() == ["2024-05-24", "2024-05-28"]
    assert df["vix"].tolist() == [11.93, 12.92]

    with pytest.raises(SealedPeriodError):
        vol_regime.parse_cboe_csv_truncated(csv_text, series_name="vix", development_end=date(2024, 5, 29))


def test_vol_regime_features_and_r1_r2_future_poison_invariance() -> None:
    sessions = pd.bdate_range("2022-01-03", "2023-06-30").to_numpy().astype("datetime64[D]")
    n_s = len(sessions)
    rng = np.random.default_rng(101)
    spy_c = (400.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.01, size=n_s))).astype(np.float32)

    vix = rng.uniform(14.0, 28.0, size=n_s)
    vix3m = np.full(n_s, 22.0)
    # Create explicit backwardation spikes (`vix > vix3m`) around rows 80..95 and 220..240
    vix[80:95] = 31.0
    vix[220:240] = 34.0

    cboe_df = pd.DataFrame({
        "session_date": [str(s) for s in sessions],
        "vix": vix,
        "vix3m": vix3m,
        "vix9d": vix * rng.uniform(0.90, 1.08, size=n_s),
        "vxn": vix * rng.uniform(1.15, 1.35, size=n_s),
        "skew": rng.uniform(115.0, 145.0, size=n_s),
    })

    # Sealed row in `cboe_df` must raise SealedPeriodError
    bad_df = pd.concat([cboe_df, pd.DataFrame([{"session_date": "2024-06-03", "vix": 15.0}])], ignore_index=True)
    with pytest.raises(SealedPeriodError):
        vol_regime.build_causal_vol_regime_features(sessions, spy_c, bad_df, development_end=date(2024, 5, 28))

    feats_full = vol_regime.build_causal_vol_regime_features(
        sessions, spy_c, cboe_df, development_end=date(2024, 5, 28),
    )

    # Truncated & future-poisoned after row 200: rows `<= 200` must match bit-for-bit!
    cboe_poison = cboe_df.copy()
    cboe_poison.loc[201:, "vix"] = 85.0
    cboe_poison.loc[201:, "skew"] = 190.0
    spy_poison = spy_c.copy()
    spy_poison[201:] *= np.float32(0.5)
    feats_poison = vol_regime.build_causal_vol_regime_features(
        sessions, spy_poison, cboe_poison, development_end=date(2024, 5, 28),
    )
    for k_name in feats_full:
        np.testing.assert_allclose(feats_poison[k_name][:201], feats_full[k_name][:201], rtol=0, atol=0, equal_nan=True)

    # Verify R1 shifts 100% into SPY on backwardation rebalance dates in `core_spy` mode
    n_c = 60
    close = (50.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.015, size=(n_s, n_c)), axis=0)).astype(np.float32)
    panel = ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"cik:{i:010d}" for i in range(n_c)),
        open=close * np.float32(0.999),
        high=close * np.float32(1.01),
        low=close * np.float32(0.99),
        close=close,
        volume=np.full((n_s, n_c), 1e6, dtype=np.float32),
        eligible=np.ones((n_s, n_c), dtype=bool),
        sector=np.zeros((n_s, n_c), dtype=np.int16),
        sector_names=("Ind",),
        spy_open=spy_c * np.float32(0.9995),
        spy_close=spy_c,
        qqq_open=spy_c * np.float32(0.9995),
        qqq_close=spy_c,
        development_end=date(2023, 6, 30),
        load_report={},
    )
    raw_s = rng.uniform(-0.5, 0.5, size=(n_s, n_c))
    bn_s = rng.uniform(-0.5, 0.5, size=(n_s, n_c))
    rv_s = rng.uniform(-0.5, 0.5, size=(n_s, n_c))
    r2_full = vol_regime.build_r2_adaptive_hedge_signal(
        panel, raw_score=raw_s, beta_neutral_score=bn_s, reversal_5d_rank=rv_s, is_backwardation=feats_full["is_backwardation"],
    )
    r2_poison = vol_regime.build_r2_adaptive_hedge_signal(
        panel, raw_score=raw_s, beta_neutral_score=bn_s, reversal_5d_rank=rv_s, is_backwardation=feats_poison["is_backwardation"],
    )
    np.testing.assert_allclose(r2_poison[:201], r2_full[:201], rtol=0, atol=0, equal_nan=True)

    win = benchmark_tilt.continuous_rebalance_windows(panel, 25, 10)
    ws_dec, _ = benchmark_tilt.build_skill_weights(panel, bn_s, win.decision_rows, buffered=True, minimum_stocks=20)
    ws = benchmark_tilt._apply_open_fills(ws_dec, win.tradable)
    back_dec = feats_full["is_backwardation"][win.decision_rows] > 0.5
    r1_res = vol_regime._simulate_r1_backwardation_to_spy(
        core_mode="core_spy",
        core_weight=0.80,
        sp_tilt_weights=None,
        mid_sat_weights=ws,
        is_backwardation_at_decisions=back_dec,
        sp_windows=win,
        mid_windows=win,
        horizon=10,
    )
    assert r1_res["backwardation_rebalance_share"] > 0.0
    assert r1_res["compounded_net_20bps"] is not None
