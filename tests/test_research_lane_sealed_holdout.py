"""Unit tests for Step 6: fail-closed sealed guard, purged holdout fold boundaries, and zero pre-holdout leakage."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane import vol_regime
from market_predictor.research_lane.fitted import target_ranks
from market_predictor.research_lane.labels import research_labels
from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError
from market_predictor.research_lane.sealed_holdout import (
    SEALED_HOLDOUT_END,
    SEALED_HOLDOUT_START,
    _render_holdout_markdown,
    build_holdout_folds,
    evaluate_sealed_holdout_2024_2026,
    fit_stage2_xgb_on_holdout,
)


def test_sealed_holdout_fails_closed_without_unseal_and_purges_pre_holdout(tmp_path: Path) -> None:
    with pytest.raises(SealedPeriodError, match="unseal-final-test-period"):
        evaluate_sealed_holdout_2024_2026(root=tmp_path, unseal_final_test_period=False)

    sessions = pd.bdate_range("2019-07-01", "2026-07-08").to_numpy().astype("datetime64[D]")
    folds = build_holdout_folds(sessions)
    assert [f.year for f in folds] == [2024, 2025, 2026]
    first_hold_day = pd.Timestamp(sessions[folds[0].test_start]).date()
    assert first_hold_day >= SEALED_HOLDOUT_START
    assert pd.Timestamp(sessions[folds[0].test_start - 1]).date() < SEALED_HOLDOUT_START

    rng = np.random.default_rng(42)
    n_s, n_c = len(sessions), 55
    close = np.full((n_s, n_c), 50.0, dtype=np.float32) * np.cumprod(
        1.0 + rng.normal(0.0002, 0.012, size=(n_s, n_c)), axis=0
    ).astype(np.float32)
    spy_c = np.full(n_s, 450.0, dtype=np.float32)
    panel = ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"cik:{i:010d}" for i in range(n_c)),
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
        development_end=SEALED_HOLDOUT_END,
        load_report={},
    )
    base_scores = rng.uniform(-0.5, 0.5, size=(n_s, n_c))
    targets = {10: target_ranks(research_labels(panel, 10).stock, panel.eligible)}
    cat_feats = {"cat_f1": rng.uniform(-0.5, 0.5, size=(n_s, n_c)).astype(np.float32)}

    comb_wf, paired_wf = fit_stage2_xgb_on_holdout(
        panel,
        base_scores,
        targets_by_horizon=targets,
        catalyst_features=cat_feats,
        catalyst_feature_names=("cat_f1",),
        horizons=(10,),
        refit_annually=True,
    )
    comb_frz, _ = fit_stage2_xgb_on_holdout(
        panel,
        base_scores,
        targets_by_horizon=targets,
        catalyst_features=cat_feats,
        catalyst_feature_names=("cat_f1",),
        horizons=(10,),
        refit_annually=False,
    )
    hold_row = folds[0].test_start
    # Zero pre-holdout predictions allowed in holdout matrix!
    assert np.isnan(comb_wf[10][:hold_row]).all()
    assert np.isnan(comb_frz[10][:hold_row]).all()
    assert np.isfinite(comb_wf[10][hold_row:-15]).any()
    assert np.isfinite(comb_frz[10][hold_row:-15]).any()
    assert set(paired_wf[10]["by_year"].keys()) == {"2024", "2025", "2026"}

    mid_panel = ResearchPanel(
        sessions=sessions,
        security_ids=tuple(f"cik:{1000 + i:010d}" for i in range(n_c)),
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
        development_end=SEALED_HOLDOUT_END,
        load_report={},
    )
    vr = vol_regime.evaluate_vol_regime_core_satellite(
        sp500_panel=panel,
        sp500_mv=np.full((n_s, n_c), 1e9, dtype=np.float64),
        sp500_eq24=base_scores,
        sp500_eq24_news_by_h={10: comb_wf[10]},
        mid_panel=mid_panel,
        mid_raw_eq24=base_scores,
        mid_bn_eq24=base_scores,
        mid_raw_sec_by_h={10: comb_wf[10]},
        mid_bn_sec_by_h={10: comb_wf[10]},
        mid_r3_by_h={10: comb_wf[10]},
        mid_rev5_rank=base_scores.astype(np.float32),
        vol_feats={"is_backwardation": np.zeros(n_s, dtype=np.float64)},
        stage1_oos_start_row=hold_row,
        stage2_oos_start_row=hold_row,
        horizons=(10,),
    )
    md = _render_holdout_markdown({
        "notice": "research label, synthetic test",
        "settings": {"holdout_first_decision_session": "2024-05-29", "holdout_end": "2026-07-08"},
        "regime_diagnostics": {
            "holdout_sessions": n_s - hold_row,
            "holdout_backwardation_sessions": 0,
            "holdout_backwardation_share": 0.0,
        },
        "holdout_paired_catalyst_ic": {"sp500_alpaca_news": {"10": paired_wf[10]}},
        "frozen_20240528_no_refit_candidates": vr["vol_regime_2021_2024"],
        "holdout_vol_regime": vr,
    })
    assert "90_10__sp500_tilt_news__R1_backwardation_to_spy" in md
