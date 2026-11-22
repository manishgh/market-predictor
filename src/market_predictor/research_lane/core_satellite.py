"""Step 5d: Unified Core `SPY` / S&P 500 Active Tilt + Mid/Small SEC-Catalyst Satellite vs. `SPY`.

Evaluates long-only joint portfolios (`w >= 0`, `sum(w) == 1.0`) combining:
1. **Core Sleeve (`80%` or `90%`):**
   - `core_spy`: passive `SPY` (`0%` internal stock-replacement turnover), or
   - `core_sp500_tilt`: point-in-time S&P 500 cap-weighted `sector_active_slice` (`<= +-2.0%`
     per-stock active bound, exact sector neutrality, `10 bps` deadband) driven by Step 5a's
     `equal_rank_24` (and `equal_rank_24 + news_xgboost` Alpaca news-reaction residuals).
2. **Satellite Sleeve (`20%` or `10%` on disjoint `mid_small_ex_sp500`):**
   - Buffered top-10% / hold-top-33.3% equal-weight basket (`skill_buffered_hold33`) across the
     `1,559` survivorship-checked active + delisted non-S&P-500 US common stocks driven by:
     - `sat_raw_eq24` (Step 5b `equal_rank_24`),
     - `sat_beta_neutral_eq24` (Step 5c `[sector, beta_252d, idio_vol_60d]`-orthogonalized `equal_rank_24`),
     - `sat_raw_eq24_sec_xgb` (`sat_raw_eq24` + Step 5c walk-forward `news_xgboost` precise SEC catalysts),
     - `sat_beta_neutral_eq24_sec_xgb` (`sat_beta_neutral_eq24` + Step 5c walk-forward `news_xgboost` precise SEC catalysts).

All portfolio weights are decided strictly after session `r_k` close, filled at `open[r_k + 1]`,
and held continuously to `open[r_{k+1} + 1]` with exact post-drift turnover `0.5 * sum(|w_new - w_drifted|)`
charged across the joint `[sp500_stocks, mid_small_stocks, SPY]` vector (`20 bps` primary, `10 bps` informational).
"""
from __future__ import annotations

import gc
from collections.abc import Callable, Mapping
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from market_predictor.research_lane import benchmark_tilt, beta_neutral, two_stage
from market_predictor.research_lane.fitted import date_weights, session_years, training_rows
from market_predictor.research_lane.labels import research_labels
from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.scoreboard import (
    daily_rank_correlation,
    newey_west,
    scored_cells,
)

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

CORE_SATELLITE_COMMAND: Final = "score-core-satellite-spy"
HORIZONS: Final = (5, 10, 21, 63)
ALLOCATIONS: Final = (("80_20", 0.80), ("90_10", 0.90))
REQUIRED_SPY_NW_T: Final = 2.50
STRICT_SPY_NW_T: Final = 3.00
REQUIRED_SAT_SKILL_NW_T: Final = 2.00

CORE_SATELLITE_READING_RULE: Final = (
    "A unified core-satellite specification earns one test on the sealed period (2024-05-29+) only if, "
    "at one of the four pre-registered horizons (5, 10, 21, 63 sessions) on OOS 2021..2024: "
    "(1) mean net excess return against SPY after 20 bps post-drift turnover cost is positive with "
    "Newey-West t >= 2.50 (reporting t >= 3.00 alongside); (2) net excess vs SPY is positive in >= 75% "
    "of scored calendar years (>= 3/4 in 2021..2024); and (3) its standalone mid_small_ex_sp500 buffered "
    "satellite beats equal-weight scored mid/small caps net of 20 bps with Newey-West t >= 2.00."
)


def verify_disjoint_core_satellite_universes(
    sp500_panel: ResearchPanel,
    sp500_cik_map: Mapping[str, str],
    mid_panel: ResearchPanel,
    mid_cik_map: Mapping[str, str],
) -> dict[str, Any]:
    """Assert identical session/SPY timelines and zero simultaneous active CIK overlap across sessions."""
    if not np.array_equal(sp500_panel.sessions, mid_panel.sessions):
        raise ValueError("sp500_panel and mid_panel sessions are not identical")
    if not np.allclose(sp500_panel.spy_open, mid_panel.spy_open, rtol=1e-6, atol=1e-6, equal_nan=True):
        raise ValueError("sp500_panel and mid_panel spy_open series differ")
    if not np.allclose(sp500_panel.spy_close, mid_panel.spy_close, rtol=1e-6, atol=1e-6, equal_nan=True):
        raise ValueError("sp500_panel and mid_panel spy_close series differ")

    sp_ciks_by_col = [sp500_cik_map.get(sid, f"unmapped:{sid}") for sid in sp500_panel.security_ids]
    mid_ciks_by_col = [mid_cik_map.get(sid, sid.removeprefix("cik:")) for sid in mid_panel.security_ids]

    shared_ciks = set(sp_ciks_by_col) & set(mid_ciks_by_col)
    simultaneous_overlaps = 0
    if shared_ciks:
        sp_idx_map: dict[str, list[int]] = {}
        for idx, c in enumerate(sp_ciks_by_col):
            if c in shared_ciks:
                sp_idx_map.setdefault(c, []).append(idx)
        mid_idx_map: dict[str, list[int]] = {}
        for idx, c in enumerate(mid_ciks_by_col):
            if c in shared_ciks:
                mid_idx_map.setdefault(c, []).append(idx)
        for c in shared_ciks:
            sp_any = sp500_panel.eligible[:, sp_idx_map[c]].any(axis=1)
            mid_any = mid_panel.eligible[:, mid_idx_map[c]].any(axis=1)
            simultaneous_overlaps += int((sp_any & mid_any).sum())

    if simultaneous_overlaps > 0:
        raise ValueError(
            f"core S&P 500 and satellite mid_small_ex_sp500 share {simultaneous_overlaps} active (session, CIK) cells"
        )
    return {
        "aligned_sessions": len(sp500_panel.sessions),
        "first_session": str(sp500_panel.sessions[0]),
        "last_session": str(sp500_panel.sessions[-1]),
        "sp500_securities": len(sp500_panel.security_ids),
        "mid_small_securities": len(mid_panel.security_ids),
        "ciks_ever_switching_tiers": len(shared_ciks),
        "simultaneous_active_cik_overlaps": simultaneous_overlaps,
    }


def fit_eq24_stage2_xgb_by_horizon(
    panel: ResearchPanel,
    base_scores: FloatArray,
    *,
    targets_by_horizon: Mapping[int, FloatArray],
    catalyst_features: Mapping[str, Float32Array],
    catalyst_feature_names: tuple[str, ...],
    horizons: tuple[int, ...] = HORIZONS,
    checkpoint: Callable[[str], None] | None = None,
) -> tuple[dict[int, Float32Array], int, dict[int, dict[str, Any]]]:
    """Fit walk-forward `news_xgboost` residuals strictly on `target - base_scores` for `2021..2024`."""
    news_start_row = int(np.searchsorted(panel.sessions, np.datetime64(two_stage.NEWS_START_DATE, "D"), side="left"))
    news_folds = two_stage._dynamic_folds(panel.sessions, news_start_row, min_train_sessions=120)
    if not news_folds:
        raise ValueError("insufficient sessions after NEWS_START_DATE to form Stage-2 walk-forward folds")

    news_scored_start = news_folds[0].test_start
    decision_rows_news = np.arange(len(panel.sessions)) >= news_scored_start
    combined_by_h: dict[int, Float32Array] = {}
    paired_by_h: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        labels = research_labels(panel, horizon)
        target = targets_by_horizon[horizon]
        residual = target - base_scores
        comb = np.full(panel.eligible.shape, np.nan, dtype=np.float64)

        for fold in news_folds:
            tr_rows = training_rows(news_start_row, fold.test_start, horizon)
            tr_cells = panel.eligible[tr_rows] & np.isfinite(residual[tr_rows])
            te_rows = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
            te_cells = panel.eligible[te_rows] & np.isfinite(base_scores[te_rows])
            if int(tr_cells.sum()) < 50 or not bool(te_cells.any()):
                continue
            r_pred, _ = two_stage._fit_predict_tabular(
                "xgboost",
                two_stage._design_matrix(catalyst_features, catalyst_feature_names, tr_rows, tr_cells),
                residual[tr_rows][tr_cells].astype(np.float64),
                date_weights(tr_cells),
                two_stage._design_matrix(catalyst_features, catalyst_feature_names, te_rows, te_cells),
                xgb_params=two_stage.STAGE2_XGB_PARAMS,
                seeds=two_stage.STAGE2_XGB_SEEDS,
            )
            blk = np.full(te_cells.shape, np.nan, dtype=np.float64)
            blk[te_cells] = base_scores[te_rows][te_cells] + r_pred
            comb[te_rows] = blk

        cells_pair = scored_cells(comb, labels.stock, panel.eligible, decision_rows_news) & np.isfinite(base_scores)
        ic_base = daily_rank_correlation(base_scores, labels.stock, cells_pair)
        ic_comb = daily_rank_correlation(comb, labels.stock, cells_pair)
        delta_ic = ic_comb - ic_base
        nw_delta = newey_west(delta_ic, horizon)
        yrs = session_years(panel.sessions)
        pos_yrs = 0
        tot_yrs = 0
        for yr in np.unique(yrs[np.isfinite(delta_ic)]):
            vals = delta_ic[np.isfinite(delta_ic) & (yrs == yr)]
            if vals.size:
                tot_yrs += 1
                if float(vals.mean()) > 0.0:
                    pos_yrs += 1

        combined_by_h[horizon] = comb.astype(np.float32)
        paired_by_h[horizon] = {
            **nw_delta,
            "positive_years": pos_yrs,
            "years_scored": tot_yrs,
        }
        del comb, residual
        gc.collect()
        if checkpoint is not None:
            checkpoint(f"core_satellite stage2 h={horizon}")

    return combined_by_h, news_scored_start, paired_by_h


def _risk_metrics(returns: FloatArray, horizon: int) -> dict[str, float | None]:
    valid = returns[np.isfinite(returns)]
    if valid.size < 2:
        return {"sharpe_annualized": None, "volatility_annualized": None, "max_drawdown": None}
    periods_per_yr = 252.0 / float(horizon)
    std = float(np.std(valid, ddof=1))
    vol_ann = std * float(np.sqrt(periods_per_yr))
    sharpe = float(valid.mean() / std * np.sqrt(periods_per_yr)) if std > 1e-12 else None
    eq = np.cumprod(1.0 + valid)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    mdd = float(np.min(eq / np.maximum(peaks, 1e-12) - 1.0))
    return {
        "sharpe_annualized": sharpe,
        "volatility_annualized": vol_ann,
        "max_drawdown": mdd,
    }


def _pack_joint_summary(
    path: benchmark_tilt.PathMetrics,
    matched_bench: FloatArray,
    matched_name: str,
    windows: benchmark_tilt.ContinuousWindows,
    horizon: int,
) -> dict[str, Any]:
    base = benchmark_tilt._pack_policy_summary(path, matched_bench, matched_name, windows, horizon)
    base["risk_net_20bps"] = _risk_metrics(path.net_returns_20bps, horizon)
    base["risk_spy"] = _risk_metrics(windows.spy_returns, horizon)
    return base


def simulate_joint_core_satellite(
    *,
    core_mode: str,
    core_weight: float,
    sp_tilt_weights: FloatArray | None,
    mid_sat_weights: FloatArray,
    sp_windows: benchmark_tilt.ContinuousWindows,
    mid_windows: benchmark_tilt.ContinuousWindows,
    horizon: int,
) -> dict[str, Any]:
    """Simulate exact joint `[sp500_stocks, mid_small_stocks, SPY]` path with post-drift turnover & `20 bps` cost."""
    if not (0.0 < core_weight < 1.0):
        raise ValueError(f"core_weight must be in (0, 1); got {core_weight}")
    sat_weight = 1.0 - core_weight
    k_len, n_sp = sp_windows.stock_returns.shape
    k_mid, n_mid = mid_windows.stock_returns.shape
    if k_len != k_mid or not np.array_equal(sp_windows.decision_rows, mid_windows.decision_rows):
        raise ValueError("sp_windows and mid_windows decision_rows must align")

    total_cols = n_sp + n_mid + 1
    spy_col = n_sp + n_mid
    w_joint = np.full((k_len, total_cols), np.nan, dtype=np.float64)

    sat_ok = np.isfinite(mid_sat_weights).all(axis=1) & (mid_sat_weights.sum(axis=1) > 0.0)
    spy_ok = np.isfinite(sp_windows.spy_returns)

    if core_mode == "core_spy":
        active = sat_ok & spy_ok
        w_joint[active, :] = 0.0
        w_joint[active, n_sp:spy_col] = sat_weight * mid_sat_weights[active]
        w_joint[active, spy_col] = core_weight
    elif core_mode == "core_sp500_tilt":
        if sp_tilt_weights is None:
            raise ValueError("sp_tilt_weights required for core_sp500_tilt")
        core_ok = np.isfinite(sp_tilt_weights).all(axis=1) & (sp_tilt_weights.sum(axis=1) > 0.0)
        active = core_ok & sat_ok & spy_ok
        w_joint[active, :] = 0.0
        w_joint[active, :n_sp] = core_weight * sp_tilt_weights[active]
        w_joint[active, n_sp:spy_col] = sat_weight * mid_sat_weights[active]
    else:
        raise ValueError(f"unsupported core_mode: {core_mode}")

    if bool(np.any(w_joint[np.isfinite(w_joint[:, 0])] < -1e-12)):
        raise AssertionError("negative weight encountered in long-only core-satellite portfolio")
    active_rows = np.isfinite(w_joint).all(axis=1)
    if bool(active_rows.any()):
        row_sums = w_joint[active_rows].sum(axis=1)
        if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1e-9):
            raise AssertionError("joint core-satellite weights failed to sum to 1.0")

    spy_bench_w = np.full((k_len, total_cols), np.nan, dtype=np.float64)
    spy_bench_w[active_rows, :] = 0.0
    spy_bench_w[active_rows, spy_col] = 1.0

    joint_returns = np.column_stack([
        sp_windows.stock_returns,
        mid_windows.stock_returns,
        sp_windows.spy_returns,
    ])
    path = benchmark_tilt.simulate_portfolio_path(w_joint, joint_returns, spy_bench_w)
    return _pack_joint_summary(path, sp_windows.spy_returns, "spy", sp_windows, horizon)


def evaluate_core_satellite_grid(
    *,
    sp500_panel: ResearchPanel,
    sp500_mv: FloatArray,
    sp500_eq24: FloatArray,
    sp500_eq24_news_by_h: Mapping[int, Float32Array],
    mid_panel: ResearchPanel,
    mid_signals_stage1: Mapping[str, FloatArray],
    mid_signals_stage2_by_h: Mapping[int, Mapping[str, Float32Array]],
    stage1_oos_start_row: int,
    stage2_oos_start_row: int,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    """Evaluate controls + all pre-registered Core x Satellite x Allocation specifications vs. `SPY`."""
    controls_2021_2024: dict[str, dict[str, Any]] = {}
    satellites_2021_2024: dict[str, dict[str, Any]] = {}
    grid_2021_2024: dict[str, dict[str, Any]] = {}
    grid_2020_2024: dict[str, dict[str, Any]] = {}
    passing_specs: list[str] = []
    passing_strict_t3: list[str] = []

    for horizon in horizons:
        h_key = str(horizon)
        # --- Window 1: OOS 2021..2024 (`stage2_oos_start_row`, identical dates for Stage-1 and Stage-2) ---
        sp_win = benchmark_tilt.continuous_rebalance_windows(sp500_panel, stage2_oos_start_row, horizon)
        mid_win = benchmark_tilt.continuous_rebalance_windows(mid_panel, stage2_oos_start_row, horizon)
        sp_sec_dec = sp500_panel.sector[sp_win.decision_rows]

        # Passive SPY reference
        controls_2021_2024.setdefault("passive_spy_100", {})[h_key] = {
            "periods": int(len(sp_win.decision_rows)),
            "first_entry_session": str(sp500_panel.sessions[int(sp_win.entry_rows[0])]),
            "last_exit_session": str(sp500_panel.sessions[int(sp_win.exit_rows[-1])]),
            "compounded_spy": benchmark_tilt._compounded_return(sp_win.spy_returns),
            "risk_spy": _risk_metrics(sp_win.spy_returns, horizon),
        }

        # 100% S&P 500 Cap-Tilt controls (`eq24` and `eq24 + news_xgboost`)
        sp_eq_dec, sp_cap_bench_dec = benchmark_tilt.build_cap_tilt_weights(
            sp500_panel, sp500_eq24, sp500_mv, sp_win.decision_rows, mode="sector_active_slice",
        )
        sp_eq_w = benchmark_tilt._apply_open_fills(sp_eq_dec, sp_win.tradable)
        sp_cap_bench_w = benchmark_tilt._apply_open_fills(sp_cap_bench_dec, sp_win.tradable)
        sp_cap_bench_path = benchmark_tilt.simulate_portfolio_path(sp_cap_bench_w, sp_win.stock_returns)
        sp_eq_path = benchmark_tilt.simulate_portfolio_path(sp_eq_w, sp_win.stock_returns, sp_cap_bench_w, sp_sec_dec)
        controls_2021_2024.setdefault("sp500_cap_tilt_eq24_100", {})[h_key] = _pack_joint_summary(
            sp_eq_path, sp_cap_bench_path.gross_returns, "sp500_cap_weighted", sp_win, horizon,
        )

        sp_news_mat = sp500_eq24_news_by_h[horizon].astype(np.float64)
        sp_news_dec, _ = benchmark_tilt.build_cap_tilt_weights(
            sp500_panel, sp_news_mat, sp500_mv, sp_win.decision_rows, mode="sector_active_slice",
        )
        sp_news_w = benchmark_tilt._apply_open_fills(sp_news_dec, sp_win.tradable)
        sp_news_path = benchmark_tilt.simulate_portfolio_path(
            sp_news_w, sp_win.stock_returns, sp_cap_bench_w, sp_sec_dec,
        )
        controls_2021_2024.setdefault("sp500_cap_tilt_news_xgb_100", {})[h_key] = _pack_joint_summary(
            sp_news_path, sp_cap_bench_path.gross_returns, "sp500_cap_weighted", sp_win, horizon,
        )

        # Standalone mid_small_ex_sp500 satellites (`2021..2024`)
        all_mid_2021: dict[str, FloatArray] = {
            "sat_raw_eq24": mid_signals_stage1["sat_raw_eq24"],
            "sat_beta_neutral_eq24": mid_signals_stage1["sat_beta_neutral_eq24"],
            "sat_raw_eq24_sec_xgb": mid_signals_stage2_by_h[horizon]["sat_raw_eq24_sec_xgb"].astype(np.float64),
            "sat_beta_neutral_eq24_sec_xgb": mid_signals_stage2_by_h[horizon]["sat_beta_neutral_eq24_sec_xgb"].astype(np.float64),
        }
        sat_w_map: dict[str, FloatArray] = {}
        sat_ok_gate: dict[str, bool] = {}

        for sat_name, sat_mat in all_mid_2021.items():
            w_s_dec, b_s_dec = benchmark_tilt.build_skill_weights(
                mid_panel, sat_mat, mid_win.decision_rows, buffered=True,
            )
            w_s = benchmark_tilt._apply_open_fills(w_s_dec, mid_win.tradable)
            b_s = benchmark_tilt._apply_open_fills(b_s_dec, mid_win.tradable)
            p_s = benchmark_tilt.simulate_portfolio_path(w_s, mid_win.stock_returns, b_s)
            b_s_gross = benchmark_tilt.simulate_portfolio_path(b_s, mid_win.stock_returns).gross_returns
            s_sum = _pack_joint_summary(p_s, b_s_gross, "equal_weight_mid_small", mid_win, horizon)
            satellites_2021_2024.setdefault(sat_name, {})[h_key] = s_sum
            sat_w_map[sat_name] = w_s

            s_ex = s_sum["vs_matched_benchmark"]["net_excess_20bps_per_period"]
            s_t = s_sum["vs_matched_benchmark"]["newey_west_20bps"]["t"]
            sat_ok_gate[sat_name] = bool(
                s_ex is not None and float(s_ex) > 0.0 and s_t is not None and float(s_t) >= REQUIRED_SAT_SKILL_NW_T
            )

        # Joint 16 Core x Satellite x Allocation specifications (`2021..2024`)
        for alloc_tag, c_weight in ALLOCATIONS:
            for core_label, c_mode, sp_w_arg in (
                ("spy", "core_spy", None),
                ("sp500_tilt_news", "core_sp500_tilt", sp_news_w),
            ):
                for sat_name, sat_w in sat_w_map.items():
                    spec_id = f"{alloc_tag}__{core_label}__{sat_name}"
                    j_sum = simulate_joint_core_satellite(
                        core_mode=c_mode,
                        core_weight=c_weight,
                        sp_tilt_weights=sp_w_arg,
                        mid_sat_weights=sat_w,
                        sp_windows=sp_win,
                        mid_windows=mid_win,
                        horizon=horizon,
                    )
                    spy_ex = j_sum["vs_spy"]["net_excess_20bps_per_period"]
                    spy_t = j_sum["vs_spy"]["newey_west_20bps"]["t"]
                    pos_y = int(j_sum["vs_spy"]["positive_years_20bps"])
                    tot_y = int(j_sum["vs_spy"]["years_scored"])
                    yr_share_ok = tot_y > 0 and (pos_y / float(tot_y)) >= 0.75

                    passes = bool(
                        spy_ex is not None
                        and float(spy_ex) > 0.0
                        and spy_t is not None
                        and float(spy_t) >= REQUIRED_SPY_NW_T
                        and yr_share_ok
                        and sat_ok_gate[sat_name]
                    )
                    passes_strict = bool(passes and spy_t is not None and float(spy_t) >= STRICT_SPY_NW_T)
                    j_sum["satellite_passes_gate"] = sat_ok_gate[sat_name]
                    j_sum["passes_reading_rule"] = passes
                    j_sum["passes_strict_t3"] = passes_strict
                    grid_2021_2024.setdefault(spec_id, {})[h_key] = j_sum
                    if passes:
                        passing_specs.append(f"{spec_id}@h={horizon}")
                    if passes_strict:
                        passing_strict_t3.append(f"{spec_id}@h={horizon}")

        # --- Window 2: Full 5-Year OOS 2020..2024 (`stage1_oos_start_row`, Stage-1-only specs) ---
        sp_win20 = benchmark_tilt.continuous_rebalance_windows(sp500_panel, stage1_oos_start_row, horizon)
        mid_win20 = benchmark_tilt.continuous_rebalance_windows(mid_panel, stage1_oos_start_row, horizon)
        sp_eq20_dec, _ = benchmark_tilt.build_cap_tilt_weights(
            sp500_panel, sp500_eq24, sp500_mv, sp_win20.decision_rows, mode="sector_active_slice",
        )
        sp_eq20_w = benchmark_tilt._apply_open_fills(sp_eq20_dec, sp_win20.tradable)

        sat20_w_map: dict[str, FloatArray] = {}
        for sat_name in ("sat_raw_eq24", "sat_beta_neutral_eq24"):
            w20_dec, _ = benchmark_tilt.build_skill_weights(
                mid_panel, mid_signals_stage1[sat_name], mid_win20.decision_rows, buffered=True,
            )
            sat20_w_map[sat_name] = benchmark_tilt._apply_open_fills(w20_dec, mid_win20.tradable)

        for alloc_tag, c_weight in ALLOCATIONS:
            for core_label, c_mode, sp_w_arg in (
                ("spy", "core_spy", None),
                ("sp500_tilt_eq24", "core_sp500_tilt", sp_eq20_w),
            ):
                for sat_name, sat_w in sat20_w_map.items():
                    spec_id = f"{alloc_tag}__{core_label}__{sat_name}"
                    grid_2020_2024.setdefault(spec_id, {})[h_key] = simulate_joint_core_satellite(
                        core_mode=c_mode,
                        core_weight=c_weight,
                        sp_tilt_weights=sp_w_arg,
                        mid_sat_weights=sat_w,
                        sp_windows=sp_win20,
                        mid_windows=mid_win20,
                        horizon=horizon,
                    )

    return {
        "controls_2021_2024": controls_2021_2024,
        "satellites_standalone_2021_2024": satellites_2021_2024,
        "core_satellite_2021_2024": grid_2021_2024,
        "core_satellite_2020_2024": grid_2020_2024,
        "reading_summary": {
            "rule": CORE_SATELLITE_READING_RULE,
            "passing_specifications": passing_specs,
            "passing_strict_t3": passing_strict_t3,
            "earns_sealed_period_test": len(passing_specs) > 0,
        },
    }


def _pct(val: Any) -> str:
    return "n/a" if val is None else f"{float(val) * 100:.2f}%"


def _num(val: Any, digits: int = 2) -> str:
    return "n/a" if val is None else f"{float(val):.{digits}f}"


def render_core_satellite_markdown(document: Mapping[str, Any]) -> str:
    """Render readable markdown scoreboard for Step 5d Core-Satellite vs. `SPY`."""
    settings = document["settings"]
    res = document["core_satellite"]
    notice = document["notice"]
    horizons = [str(h) for h in settings["horizons_sessions"]]

    lines = [
        "# Step 5d: Unified Core `SPY` / S&P 500 Active Tilt + Mid/Small SEC-Catalyst Satellite vs. `SPY`",
        "",
        f"Every figure below: {notice}.",
        "",
        "## Method & boundaries",
        "",
        f"- **Disjoint universes:** S&P 500 (`{document['universe_alignment']['sp500_securities']}` securities) + "
        f"`mid_small_ex_sp500` (`{document['universe_alignment']['mid_small_securities']}` survivorship-checked "
        f"active + delisted mid/small-cap securities; `{document['universe_alignment']['simultaneous_active_cik_overlaps']}` "
        "active CIK overlaps across sessions).",
        "- **Execution & costs:** Long-only (`w >= 0`, `sum(w) == 1.0`), continuous `[open[r_k + 1], open[r_{k+1} + 1]]` "
        "windows, exact post-drift turnover `0.5 * sum(|w_new - w_drifted|)` across both core and satellite legs "
        "charged at `20 bps` (`10 bps` informational).",
        "",
        "## Section A: Reference Controls & Standalone Mid/Small Satellites (OOS 2021..2024)",
        "",
        "| Portfolio / Sleeve | H | Turnover | Compounded net 20bps | SPY compounded | "
        "Net vs Matched/yr [NW t] | Net vs SPY/yr [NW t] | Sharpe (port / SPY) | MaxDD (port / SPY) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c_name in ("sp500_cap_tilt_eq24_100", "sp500_cap_tilt_news_xgb_100"):
        for h in horizons:
            e = res["controls_2021_2024"][c_name][h]
            vm = e["vs_matched_benchmark"]
            vs = e["vs_spy"]
            lines.append(
                f"| {c_name} | {h} | {_pct(e['mean_steady_one_way_turnover'])} | {_pct(e['compounded_net_20bps'])} | "
                f"{_pct(vs['compounded_benchmark'])} | {_pct(vm['net_excess_20bps_annualized'])} [{_num(vm['newey_west_20bps']['t'])}] | "
                f"{_pct(vs['net_excess_20bps_annualized'])} [{_num(vs['newey_west_20bps']['t'])}] | "
                f"{_num(e['risk_net_20bps']['sharpe_annualized'])} / {_num(e['risk_spy']['sharpe_annualized'])} | "
                f"{_pct(e['risk_net_20bps']['max_drawdown'])} / {_pct(e['risk_spy']['max_drawdown'])} |"
            )
    for s_name, h_map in res["satellites_standalone_2021_2024"].items():
        for h in horizons:
            e = h_map[h]
            vm = e["vs_matched_benchmark"]
            vs = e["vs_spy"]
            lines.append(
                f"| {s_name} (100% mid/small) | {h} | {_pct(e['mean_steady_one_way_turnover'])} | {_pct(e['compounded_net_20bps'])} | "
                f"{_pct(vs['compounded_benchmark'])} | {_pct(vm['net_excess_20bps_annualized'])} [{_num(vm['newey_west_20bps']['t'])}] | "
                f"{_pct(vs['net_excess_20bps_annualized'])} [{_num(vs['newey_west_20bps']['t'])}] | "
                f"{_num(e['risk_net_20bps']['sharpe_annualized'])} / {_num(e['risk_spy']['sharpe_annualized'])} | "
                f"{_pct(e['risk_net_20bps']['max_drawdown'])} / {_pct(e['risk_spy']['max_drawdown'])} |"
            )

    lines += [
        "",
        "## Section B: Unified Core + Mid/Small SEC-Catalyst Satellite vs. `SPY` (OOS 2021..2024, Net of 20 bps)",
        "",
        "| Specification | H | Turnover | Port net 20bps | SPY | Net vs SPY/yr [NW t] | "
        "Net 10bps vs SPY/yr | 2021 | 2022 | 2023 | 2024 | Pos yrs | Sharpe | Passes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for spec_id, h_map in res["core_satellite_2021_2024"].items():
        for h in horizons:
            e = h_map[h]
            vs = e["vs_spy"]
            by_y = vs["by_year_net_20bps"]
            y_cells = [_pct(by_y.get(y, {}).get("mean")) for y in ("2021", "2022", "2023", "2024")]
            lines.append(
                f"| {spec_id} | {h} | {_pct(e['mean_steady_one_way_turnover'])} | {_pct(e['compounded_net_20bps'])} | "
                f"{_pct(vs['compounded_benchmark'])} | {_pct(vs['net_excess_20bps_annualized'])} [{_num(vs['newey_west_20bps']['t'])}] | "
                f"{_pct(vs['net_excess_10bps_annualized'])} | " + " | ".join(y_cells)
                + f" | {vs['positive_years_20bps']}/{vs['years_scored']} | {_num(e['risk_net_20bps']['sharpe_annualized'])} | "
                f"{'yes' if e['passes_reading_rule'] else 'no'} |"
            )

    lines += [
        "",
        "## Section C: Full 5-Year OOS 2020..2024 Stage-1 Core + Satellite Check (Including 2020 COVID Shock)",
        "",
        "| Specification | H | Turnover | Port net 20bps | SPY | Net vs SPY/yr [NW t] | 2020 | 2021 | 2022 | 2023 | 2024 | Pos yrs |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for spec_id, h_map in res["core_satellite_2020_2024"].items():
        for h in horizons:
            e = h_map[h]
            vs = e["vs_spy"]
            by_y = vs["by_year_net_20bps"]
            y_cells = [_pct(by_y.get(y, {}).get("mean")) for y in ("2020", "2021", "2022", "2023", "2024")]
            lines.append(
                f"| {spec_id} | {h} | {_pct(e['mean_steady_one_way_turnover'])} | {_pct(e['compounded_net_20bps'])} | "
                f"{_pct(vs['compounded_benchmark'])} | {_pct(vs['net_excess_20bps_annualized'])} [{_num(vs['newey_west_20bps']['t'])}] | "
                + " | ".join(y_cells) + f" | {vs['positive_years_20bps']}/{vs['years_scored']} |"
            )

    r_sum = res["reading_summary"]
    lines += [
        "",
        "## Section D: Pre-registered reading rule",
        "",
        r_sum["rule"],
        "",
        f"- Passing specifications (`NW t >= 2.50`): `{r_sum['passing_specifications']}`.",
        f"- Passing strict `NW t >= 3.00`: `{r_sum['passing_strict_t3']}`.",
        "Verdict: **"
        + (
            "candidate earned sealed test"
            if r_sum["earns_sealed_period_test"]
            else "no specification passed — 2024-05-29+ remains sealed"
        )
        + "**.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "ALLOCATIONS",
    "CORE_SATELLITE_COMMAND",
    "CORE_SATELLITE_READING_RULE",
    "HORIZONS",
    "beta_neutral",
    "evaluate_core_satellite_grid",
    "fit_eq24_stage2_xgb_by_horizon",
    "render_core_satellite_markdown",
    "simulate_joint_core_satellite",
    "verify_disjoint_core_satellite_universes",
]
