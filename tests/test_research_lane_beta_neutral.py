"""Unit, exact-orthogonality, boilerplate-filtering, and causality tests for Step 5c (`beta_neutral`)."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane import beta_neutral, two_stage
from market_predictor.research_lane.panel import MARKET_TIMEZONE, SealedPeriodError
from tests.support.research_lane_fundamentals import SyntheticFundamentals, synthetic_fundamental_inputs
from tests.support.research_lane_panel import POST_END_PRICE, synthetic_panel


def test_cross_sectional_neutralize_exact_zero_sector_beta_and_vol_loadings_and_causality() -> None:
    rng = np.random.default_rng(101)
    n_sess, n_sec = 40, 60
    sector = np.tile(np.arange(n_sec, dtype=np.int16) % 4, (n_sess, 1))
    beta_rk = rng.uniform(-0.5, 0.5, size=(n_sess, n_sec)).astype(np.float32)
    ivol_rk = rng.uniform(-0.5, 0.5, size=(n_sess, n_sec)).astype(np.float32)
    eligible = rng.random((n_sess, n_sec)) > 0.10

    # Inject strong sector + beta + vol confounding into `raw_score`.
    raw_score = (
        0.8 * sector.astype(np.float64)
        - 1.5 * beta_rk.astype(np.float64)
        - 0.9 * ivol_rk.astype(np.float64)
        + rng.normal(0.0, 0.2, size=(n_sess, n_sec))
    )

    orth = beta_neutral.cross_sectional_neutralize(
        raw_score, eligible, sector, beta_rk, ivol_rk, rerank=False,
    )
    for r in range(n_sess):
        m = eligible[r]
        res_r = orth[r, m]
        assert abs(float(res_r.mean())) < 1e-8
        assert abs(float(np.dot(res_r, beta_rk[r, m]))) < 1e-8
        assert abs(float(np.dot(res_r, ivol_rk[r, m]))) < 1e-8
        for s_code in range(4):
            sm = m & (sector[r] == s_code)
            assert abs(float(orth[r, sm].mean())) < 1e-8

    ranked_full = beta_neutral.cross_sectional_neutralize(
        raw_score, eligible, sector, beta_rk, ivol_rk, rerank=True,
    )
    cut = 25
    ranked_trunc = beta_neutral.cross_sectional_neutralize(
        raw_score[: cut + 1], eligible[: cut + 1], sector[: cut + 1], beta_rk[: cut + 1], ivol_rk[: cut + 1], rerank=True,
    )
    np.testing.assert_allclose(ranked_trunc, ranked_full[: cut + 1], rtol=0, atol=0, equal_nan=True)

    p_raw = raw_score.copy()
    p_beta = beta_rk.copy()
    p_raw[cut + 1 :] = 99999.0
    p_beta[cut + 1 :] = 0.5
    ranked_poison = beta_neutral.cross_sectional_neutralize(
        p_raw, eligible, sector, p_beta, ivol_rk, rerank=True,
    )
    np.testing.assert_allclose(ranked_poison[: cut + 1], ranked_full[: cut + 1], rtol=0, atol=0, equal_nan=True)


def test_precise_sec_catalyst_features_drops_507_801_noise_honors_clock_and_poison_invariance() -> None:
    panel = synthetic_panel(sessions=340, securities=20, seed=31)
    sec_ids = tuple(f"cik:{i + 1:010d}" for i in range(len(panel.security_ids)))
    end_day = pd.Timestamp(panel.sessions[-1]).date()
    panel_elig = replace(panel, security_ids=sec_ids, eligible=np.ones_like(panel.eligible), development_end=end_day)

    day_290 = pd.Timestamp(panel_elig.sessions[290]).date()
    after_close_290 = pd.Timestamp(f"{day_290.isoformat()} 16:10:00", tz=MARKET_TIMEZONE).tz_convert("UTC")
    day_285 = pd.Timestamp(panel_elig.sessions[285]).date()
    in_hours_285 = pd.Timestamp(f"{day_285.isoformat()} 09:00:00", tz=MARKET_TIMEZONE).tz_convert("UTC")

    events_8k = pd.DataFrame({
        "security_id": [sec_ids[0], sec_ids[1], sec_ids[2], sec_ids[3]],
        "accession_number": ["acc-1", "acc-2", "acc-3", "acc-4"],
        # `5.07` annual meeting vote and `8.01,9.01` boilerplate must be dropped!
        "item_codes": ["2.02", "5.07,9.01", "8.01,9.01", "4.02"],
        "available_at_utc": [after_close_290, in_hours_285, in_hours_285, in_hours_285],
    })
    non8k = pd.DataFrame({
        "security_id": [sec_ids[4], sec_ids[5]],
        "accession_number": ["acc-5", "acc-6"],
        "catalyst_family": ["rx_activist_13d", "rx_dilutive_offering"],
        "available_at_utc": [in_hours_285, after_close_290],
    })

    feats_full, rep = beta_neutral.build_precise_sec_catalyst_features(panel_elig, events_8k, non8k)
    assert rep["dropped_boilerplate_8k_rows"] == 2
    assert rep["qualified_catalyst_tuples"] == 4
    # After-hours 16:10 ET on row 290 maps to row 291 -> row 290 stays 0.0!
    assert feats_full["rx_earnings_202"][290, 0] == pytest.approx(0.0, abs=1e-7)
    assert feats_full["rx_earnings_202"][291, 0] != 0.0
    assert feats_full["rx_dilutive_offering"][290, 5] == pytest.approx(0.0, abs=1e-7)
    assert feats_full["rx_dilutive_offering"][291, 5] != 0.0

    # Future poison invariance
    p_close = panel_elig.close.copy()
    p_vol = panel_elig.volume.copy()
    p_close[300:] = POST_END_PRICE
    p_vol[300:] = 1e12
    poisoned = replace(panel_elig, close=p_close, volume=p_vol)
    feats_poison, _ = beta_neutral.build_precise_sec_catalyst_features(poisoned, events_8k, non8k)
    for name in beta_neutral.PRECISE_STAGE2_FEATURE_NAMES:
        np.testing.assert_allclose(feats_poison[name][:300], feats_full[name][:300], rtol=0, atol=0, equal_nan=True)

    # Sealed row refusal
    sealed_non8k = pd.DataFrame({
        "security_id": [sec_ids[0]],
        "accession_number": ["acc-sealed"],
        "catalyst_family": ["rx_activist_13d"],
        "available_at_utc": [pd.Timestamp("2024-06-01 12:00:00+00:00")],
    })
    panel_2024 = replace(panel_elig, development_end=date(2024, 5, 28))
    with pytest.raises(SealedPeriodError):
        beta_neutral.build_precise_sec_catalyst_features(panel_2024, events_8k.iloc[:0], sealed_non8k)


def test_beta_neutral_stage1_walk_forward_runs() -> None:
    syn: SyntheticFundamentals = synthetic_fundamental_inputs()
    inp = syn.inputs
    bundle = two_stage.build_stage1_features(inp.panel, inp)
    res = beta_neutral.run_beta_neutral_stage1_walk_forward(
        inp.panel, bundle, inp.security_issuer, first_row=260, horizons=(5, 21),
    )
    assert set(res) == {5, 21}
    assert res[21]["models"]["xgb_shallow_5seed"]["unseen_oos_scores"] is not None
    assert res[5]["models"]["xgb_shallow_5seed"]["unseen_oos_scores"] is None
