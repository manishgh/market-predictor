"""Step 5e (Option B): $0 Public CBOE Index Volatility Term-Structure & Skew Regime Conditioning.

Collects free daily CBOE implied-volatility indices (`VIX` 30d, `VIX3M` 90d, `VIX9D` 9d,
`VXN` Nasdaq-100 30d, `SKEW`), truncating strictly in memory to `DATE <= development_end`
(`2024-05-28`) before writing to disk, and evaluates three pre-registered volatility-regime
policies (`R1_backwardation_to_spy`, `R2_adaptive_hedge_switch`, `R3_walk_forward_vol_interaction_xgb`)
on the unified Core `SPY` / S&P 500 Active Tilt + `mid_small_ex_sp500` SEC-Catalyst Satellite
architecture against passive `SPY` net of `20 bps` exact post-drift turnover.
"""
from __future__ import annotations

import csv
import gc
import io
import json
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research_lane import benchmark_tilt, core_satellite, two_stage
from market_predictor.research_lane.fitted import date_weights, training_rows
from market_predictor.research_lane.panel import (
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
)
from market_predictor.research_lane.predictors import centered_rank

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

CBOE_DIRECTORY: Final = Path("data/research/research_lane/cboe_vol_indices")
CBOE_PARQUET_NAME: Final = "cboe_daily.parquet"
COLLECT_CBOE_COMMAND: Final = "collect-research-lane-cboe-term-structure"
VOL_REGIME_COMMAND: Final = "score-vol-regime-core-satellite"

CBOE_CSV_URLS: Final[Mapping[str, str]] = {
    "vix": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
    "vix3m": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv",
    "vix9d": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX9D_History.csv",
    "vxn": "https://cdn.cboe.com/api/global/us_indices/daily_prices/VXN_History.csv",
    "skew": "https://cdn.cboe.com/api/global/us_indices/daily_prices/SKEW_History.csv",
}

INTERACTION_FEATURE_NAMES: Final = (
    "base_raw_eq24",
    "base_bn_eq24",
    "sec_xgb_residual",
    "px_beta_252d",
    "px_reversal_5d",
    "fn_operating_profitability",
    "vix_term_30_90_centered",
    "vxn_to_vix_centered",
    "beta_x_term",
    "rev5_x_term",
)


def parse_cboe_csv_truncated(
    csv_text: str,
    *,
    series_name: str,
    development_end: date,
    min_date: date = date(2014, 1, 1),
    unseal_final_test_period: bool = False,
) -> pd.DataFrame:
    """Parse one CBOE daily CSV and drop every row dated `> development_end` immediately in memory."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    end_iso = development_end.isoformat()
    min_iso = min_date.isoformat()

    reader = csv.DictReader(io.StringIO(csv_text))
    rows: list[tuple[str, float]] = []
    for raw in reader:
        d_raw = str(raw.get("DATE") or raw.get("Date") or "").strip()
        if not d_raw:
            continue
        parts = d_raw.split("/")
        if len(parts) == 3:
            iso = f"{parts[2].strip()}-{parts[0].strip().zfill(2)}-{parts[1].strip().zfill(2)}"
        else:
            iso = d_raw[:10]
        if iso > end_iso or iso < min_iso:
            continue
        val_raw = raw.get("CLOSE")
        if val_raw is None:
            val_raw = raw.get("SKEW") or raw.get("Close")
        val_str = str(val_raw or "").strip()
        if not val_str or val_str == ".":
            continue
        try:
            val_f = float(val_str)
        except ValueError:
            continue
        if np.isfinite(val_f) and val_f > 0.0:
            rows.append((iso, val_f))

    df = pd.DataFrame(rows, columns=["session_date", series_name]).drop_duplicates("session_date", keep="last")
    return df.sort_values("session_date").reset_index(drop=True)


def collect_cboe_vol_indices(
    *,
    root: Path,
    development_end: date,
    data_root: Path | None = None,
    output: Path = CBOE_DIRECTORY,
    unseal_final_test_period: bool = False,
) -> dict[str, Any]:
    """Download public CBOE daily volatility indices (`<= development_end` strictly) to parquet."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(COLLECT_CBOE_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("cboe vol index collection start")
        dest = paths.write(paths.identity(output))
        dest.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": "Mozilla/5.0 market-predictor/0.1"}

        merged: pd.DataFrame | None = None
        series_counts: dict[str, int] = {}
        for name, url in CBOE_CSV_URLS.items():
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            part = parse_cboe_csv_truncated(
                resp.text,
                series_name=name,
                development_end=development_end,
                unseal_final_test_period=unseal_final_test_period,
            )
            series_counts[name] = len(part)
            merged = part if merged is None else merged.merge(part, on="session_date", how="outer")

        assert merged is not None
        merged = merged.sort_values("session_date").reset_index(drop=True)
        if len(merged) and str(merged["session_date"].max()) > development_end.isoformat():
            raise SealedPeriodError("CBOE table exceeded development_end")

        out_pq = dest / CBOE_PARQUET_NAME
        cast(Any, pq).write_table(pa.Table.from_pandas(merged, preserve_index=False), out_pq)
        manifest = {
            "collected_at_utc": datetime.now(UTC).isoformat(),
            "development_end": development_end.isoformat(),
            "rows": len(merged),
            "first_date": str(merged["session_date"].min()) if len(merged) else None,
            "last_date": str(merged["session_date"].max()) if len(merged) else None,
            "series_counts": series_counts,
            "urls": dict(CBOE_CSV_URLS),
        }
        (dest / "_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest


def build_causal_vol_regime_features(
    sessions: npt.NDArray[np.datetime64],
    spy_close: npt.NDArray[np.floating[Any]],
    cboe_df: pd.DataFrame,
    *,
    development_end: date,
    unseal_final_test_period: bool = False,
) -> dict[str, FloatArray]:
    """Align CBOE daily series onto `sessions` (`<= t`, forward-fill `<= 3` sessions) and compute causal ratios."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    if len(cboe_df) and str(cboe_df["session_date"].max()) > development_end.isoformat():
        raise SealedPeriodError(
            f"CBOE volatility input contains date {cboe_df['session_date'].max()} > {development_end.isoformat()}"
        )

    n_s = len(sessions)
    aligned: dict[str, FloatArray] = {}
    c_df = cboe_df.sort_values("session_date").reset_index(drop=True)
    c_days = pd.to_datetime(c_df["session_date"]).to_numpy().astype("datetime64[D]")

    for col in ("vix", "vix3m", "vix9d", "vxn", "skew"):
        arr = np.full(n_s, np.nan, dtype=np.float64)
        if col in c_df.columns and len(c_df):
            vals = pd.to_numeric(c_df[col], errors="coerce").to_numpy(dtype=np.float64)
            ok = np.isfinite(vals) & (vals > 0.0)
            ok_days = c_days[ok]
            ok_vals = vals[ok]
            if ok_days.size > 0:
                pos = np.searchsorted(ok_days, sessions, side="right") - 1
                valid_p = pos >= 0
                pk = np.where(valid_p, pos, 0)
                gap_days = (sessions - ok_days[pk]).astype("timedelta64[D]").astype(np.int64)
                use = valid_p & (gap_days >= 0) & (gap_days <= 5)
                arr[use] = ok_vals[pk[use]]
        aligned[col] = arr

    with np.errstate(invalid="ignore", divide="ignore"):
        term_30_90 = np.where(
            np.isfinite(aligned["vix"]) & np.isfinite(aligned["vix3m"]) & (aligned["vix3m"] > 0.0),
            aligned["vix"] / aligned["vix3m"],
            np.nan,
        )
        term_9_30 = np.where(
            np.isfinite(aligned["vix9d"]) & np.isfinite(aligned["vix"]) & (aligned["vix"] > 0.0),
            aligned["vix9d"] / aligned["vix"],
            np.nan,
        )
        vxn_vix = np.where(
            np.isfinite(aligned["vxn"]) & np.isfinite(aligned["vix"]) & (aligned["vix"] > 0.0),
            aligned["vxn"] / aligned["vix"],
            np.nan,
        )

    # Causal 60-session trailing z-score of CBOE SKEW (`<= t`).
    skew = aligned["skew"]
    skew_z60 = np.full(n_s, np.nan, dtype=np.float64)
    for t in range(19, n_s):
        win = skew[max(0, t - 59) : t + 1]
        fin = win[np.isfinite(win)]
        if fin.size >= 20 and np.isfinite(skew[t]):
            sd = float(np.std(fin, ddof=1))
            if sd > 1e-8:
                skew_z60[t] = (skew[t] - float(fin.mean())) / sd

    # Causal 20-session realized SPY annualized volatility & VRP ratio `VIX / (100 * rvol_20d)`.
    sc = spy_close.astype(np.float64)
    spy_ret = np.full(n_s, np.nan, dtype=np.float64)
    ok_sc = np.isfinite(sc[1:]) & (sc[1:] > 0.0) & np.isfinite(sc[:-1]) & (sc[:-1] > 0.0)
    spy_ret[1:][ok_sc] = sc[1:][ok_sc] / sc[:-1][ok_sc] - 1.0
    spy_vrp_20 = np.full(n_s, np.nan, dtype=np.float64)
    for t in range(20, n_s):
        rw = spy_ret[t - 19 : t + 1]
        fin_r = rw[np.isfinite(rw)]
        if fin_r.size >= 15 and np.isfinite(aligned["vix"][t]):
            rv = float(np.std(fin_r, ddof=1) * np.sqrt(252.0) * 100.0)
            if rv > 1e-6:
                spy_vrp_20[t] = aligned["vix"][t] / rv

    is_back = (
        (np.isfinite(term_30_90) & (term_30_90 > 1.0))
        | (np.isfinite(term_9_30) & (term_9_30 > 1.05))
    ).astype(np.float64)

    return {
        "vix": aligned["vix"],
        "vix3m": aligned["vix3m"],
        "vix9d": aligned["vix9d"],
        "vxn": aligned["vxn"],
        "skew": skew,
        "vix_term_30_90": term_30_90,
        "vix_term_9_30": term_9_30,
        "vxn_to_vix": vxn_vix,
        "cboe_skew_z60": skew_z60,
        "spy_vrp_20": spy_vrp_20,
        "is_backwardation": is_back,
    }


def build_r2_adaptive_hedge_signal(
    panel: ResearchPanel,
    *,
    raw_score: FloatArray,
    beta_neutral_score: FloatArray,
    reversal_5d_rank: FloatArray,
    is_backwardation: FloatArray,
) -> FloatArray:
    """Zero-parameter regime state machine: `raw_score` in contango, `0.7*beta_neutral + 0.3*rev5` in backwardation."""
    n_s, n_c = raw_score.shape
    out = np.full((n_s, n_c), np.nan, dtype=np.float64)
    back_rows = is_backwardation > 0.5
    cont_rows = ~back_rows
    out[cont_rows] = raw_score[cont_rows]
    if bool(back_rows.any()):
        bn_sub = beta_neutral_score[back_rows]
        rv_sub = np.nan_to_num(reversal_5d_rank[back_rows], nan=0.0)
        blend = np.where(np.isfinite(bn_sub), 0.70 * bn_sub + 0.30 * rv_sub, np.nan)
        out[back_rows] = centered_rank(blend, panel.eligible[back_rows])
    return out


def fit_r3_vol_interaction_xgb_by_horizon(
    panel: ResearchPanel,
    *,
    targets_by_horizon: Mapping[int, FloatArray],
    base_raw_eq24: FloatArray,
    base_bn_eq24: FloatArray,
    sec_xgb_by_h: Mapping[int, Float32Array],
    px_beta_252d: Float32Array,
    px_reversal_5d: Float32Array,
    fn_op_prof: Float32Array,
    vol_feats: Mapping[str, FloatArray],
    horizons: tuple[int, ...] = core_satellite.HORIZONS,
    checkpoint: Callable[[str], None] | None = None,
) -> dict[int, Float32Array]:
    """Fit purged walk-forward shallow XGBoost with cross-sectional x volatility-term-structure interactions."""
    news_start_row = int(np.searchsorted(panel.sessions, np.datetime64(two_stage.NEWS_START_DATE, "D"), side="left"))
    folds = two_stage._dynamic_folds(panel.sessions, news_start_row, min_train_sessions=120)
    shape = panel.eligible.shape

    term_c = np.nan_to_num(vol_feats["vix_term_30_90"] - 1.0, nan=0.0).astype(np.float32)[:, None]
    vxn_c = np.nan_to_num(vol_feats["vxn_to_vix"] - 1.0, nan=0.0).astype(np.float32)[:, None]
    term_mat = np.broadcast_to(term_c, shape).copy()
    vxn_mat = np.broadcast_to(vxn_c, shape).copy()
    beta_x_term = (px_beta_252d * term_mat).astype(np.float32)
    rev5_x_term = (px_reversal_5d * term_mat).astype(np.float32)

    out_by_h: dict[int, Float32Array] = {}
    for horizon in horizons:
        comb_sec = sec_xgb_by_h[horizon].astype(np.float64)
        sec_res = np.where(np.isfinite(comb_sec) & np.isfinite(base_raw_eq24), comb_sec - base_raw_eq24, 0.0).astype(np.float32)
        feat_map: dict[str, Float32Array] = {
            "base_raw_eq24": base_raw_eq24.astype(np.float32),
            "base_bn_eq24": base_bn_eq24.astype(np.float32),
            "sec_xgb_residual": sec_res,
            "px_beta_252d": px_beta_252d,
            "px_reversal_5d": px_reversal_5d,
            "fn_operating_profitability": fn_op_prof,
            "vix_term_30_90_centered": term_mat,
            "vxn_to_vix_centered": vxn_mat,
            "beta_x_term": beta_x_term,
            "rev5_x_term": rev5_x_term,
        }
        target = targets_by_horizon[horizon]
        preds_mat = np.full(shape, np.nan, dtype=np.float64)
        for fold in folds:
            tr_rows = training_rows(news_start_row, fold.test_start, horizon)
            tr_cells = panel.eligible[tr_rows] & np.isfinite(target[tr_rows])
            te_rows = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
            te_cells = panel.eligible[te_rows]
            if int(tr_cells.sum()) < 50 or not bool(te_cells.any()):
                continue
            p_vals, _ = two_stage._fit_predict_tabular(
                "xgboost",
                two_stage._design_matrix(feat_map, INTERACTION_FEATURE_NAMES, tr_rows, tr_cells),
                target[tr_rows][tr_cells].astype(np.float64),
                date_weights(tr_cells),
                two_stage._design_matrix(feat_map, INTERACTION_FEATURE_NAMES, te_rows, te_cells),
                xgb_params=two_stage.STAGE2_XGB_PARAMS,
                seeds=two_stage.STAGE2_XGB_SEEDS,
            )
            blk = np.full(te_cells.shape, np.nan, dtype=np.float64)
            blk[te_cells] = p_vals
            preds_mat[te_rows] = blk

        out_by_h[horizon] = centered_rank(preds_mat, panel.eligible).astype(np.float32)
        del feat_map, preds_mat, sec_res, comb_sec
        gc.collect()
        if checkpoint is not None:
            checkpoint(f"vol_regime r3 h={horizon}")

    return out_by_h


def _simulate_r1_backwardation_to_spy(
    *,
    core_mode: str,
    core_weight: float,
    sp_tilt_weights: FloatArray | None,
    mid_sat_weights: FloatArray,
    is_backwardation_at_decisions: npt.NDArray[np.bool_],
    sp_windows: benchmark_tilt.ContinuousWindows,
    mid_windows: benchmark_tilt.ContinuousWindows,
    horizon: int,
) -> dict[str, Any]:
    """Shift satellite weight into `SPY` on rebalance dates with front-month IV backwardation (`VIX > VIX3M`)."""
    sat_weight = 1.0 - core_weight
    k_len, n_sp = sp_windows.stock_returns.shape
    _, n_mid = mid_windows.stock_returns.shape
    total_cols = n_sp + n_mid + 1
    spy_col = n_sp + n_mid
    w_joint = np.full((k_len, total_cols), np.nan, dtype=np.float64)

    sat_ok = np.isfinite(mid_sat_weights).all(axis=1) & (mid_sat_weights.sum(axis=1) > 0.0)
    spy_ok = np.isfinite(sp_windows.spy_returns)

    if core_mode == "core_spy":
        active = sat_ok & spy_ok
        w_joint[active, :] = 0.0
        cont = active & (~is_backwardation_at_decisions)
        back = active & is_backwardation_at_decisions
        w_joint[cont, n_sp:spy_col] = sat_weight * mid_sat_weights[cont]
        w_joint[cont, spy_col] = core_weight
        w_joint[back, spy_col] = 1.0
    elif core_mode == "core_sp500_tilt":
        if sp_tilt_weights is None:
            raise ValueError("sp_tilt_weights required")
        core_ok = np.isfinite(sp_tilt_weights).all(axis=1) & (sp_tilt_weights.sum(axis=1) > 0.0)
        active = core_ok & sat_ok & spy_ok
        w_joint[active, :] = 0.0
        w_joint[active, :n_sp] = core_weight * sp_tilt_weights[active]
        cont = active & (~is_backwardation_at_decisions)
        back = active & is_backwardation_at_decisions
        w_joint[cont, n_sp:spy_col] = sat_weight * mid_sat_weights[cont]
        w_joint[back, spy_col] = sat_weight
    else:
        raise ValueError(f"unsupported core_mode: {core_mode}")

    active_rows = np.isfinite(w_joint).all(axis=1)
    spy_bench_w = np.full((k_len, total_cols), np.nan, dtype=np.float64)
    spy_bench_w[active_rows, :] = 0.0
    spy_bench_w[active_rows, spy_col] = 1.0
    joint_returns = np.column_stack([sp_windows.stock_returns, mid_windows.stock_returns, sp_windows.spy_returns])
    path = benchmark_tilt.simulate_portfolio_path(w_joint, joint_returns, spy_bench_w)
    res = core_satellite._pack_joint_summary(path, sp_windows.spy_returns, "spy", sp_windows, horizon)
    res["backwardation_rebalance_share"] = float(is_backwardation_at_decisions[active_rows].mean()) if bool(active_rows.any()) else 0.0
    return res


def evaluate_vol_regime_core_satellite(
    *,
    sp500_panel: ResearchPanel,
    sp500_mv: FloatArray,
    sp500_eq24: FloatArray,
    sp500_eq24_news_by_h: Mapping[int, Float32Array],
    mid_panel: ResearchPanel,
    mid_raw_eq24: FloatArray,
    mid_bn_eq24: FloatArray,
    mid_raw_sec_by_h: Mapping[int, Float32Array],
    mid_bn_sec_by_h: Mapping[int, Float32Array],
    mid_r3_by_h: Mapping[int, Float32Array],
    mid_rev5_rank: Float32Array,
    vol_feats: Mapping[str, FloatArray],
    stage1_oos_start_row: int,
    stage2_oos_start_row: int,
    horizons: tuple[int, ...] = core_satellite.HORIZONS,
) -> dict[str, Any]:
    """Evaluate `R1`, `R2`, and `R3` volatility-regime Core-Satellite portfolios vs. `SPY`."""
    is_back = vol_feats["is_backwardation"]
    sat_standalone_2021: dict[str, dict[str, Any]] = {}
    grid_2021_2024: dict[str, dict[str, Any]] = {}
    grid_2020_2024: dict[str, dict[str, Any]] = {}
    passing_specs: list[str] = []
    passing_strict_t3: list[str] = []

    rev5_f64 = mid_rev5_rank.astype(np.float64)
    r2_stage1_sig = build_r2_adaptive_hedge_signal(
        mid_panel,
        raw_score=mid_raw_eq24,
        beta_neutral_score=mid_bn_eq24,
        reversal_5d_rank=rev5_f64,
        is_backwardation=is_back,
    )

    for horizon in horizons:
        h_key = str(horizon)
        sp_win = benchmark_tilt.continuous_rebalance_windows(sp500_panel, stage2_oos_start_row, horizon)
        mid_win = benchmark_tilt.continuous_rebalance_windows(mid_panel, stage2_oos_start_row, horizon)
        back_at_dec = is_back[sp_win.decision_rows] > 0.5

        sp_news_dec, _ = benchmark_tilt.build_cap_tilt_weights(
            sp500_panel, sp500_eq24_news_by_h[horizon].astype(np.float64), sp500_mv, sp_win.decision_rows, mode="sector_active_slice",
        )
        sp_news_w = benchmark_tilt._apply_open_fills(sp_news_dec, sp_win.tradable)

        r2_s2_sig = build_r2_adaptive_hedge_signal(
            mid_panel,
            raw_score=mid_raw_sec_by_h[horizon].astype(np.float64),
            beta_neutral_score=mid_bn_sec_by_h[horizon].astype(np.float64),
            reversal_5d_rank=rev5_f64,
            is_backwardation=is_back,
        )
        sat_candidates_2021: dict[str, FloatArray] = {
            "R1_base_bn_sec_xgb": mid_bn_sec_by_h[horizon].astype(np.float64),
            "R2_adaptive_hedge_sec_xgb": r2_s2_sig,
            "R3_vol_interaction_xgb": mid_r3_by_h[horizon].astype(np.float64),
        }
        sat_w_map: dict[str, FloatArray] = {}
        sat_ok_gate: dict[str, bool] = {}

        for s_name, s_mat in sat_candidates_2021.items():
            ws_dec, bs_dec = benchmark_tilt.build_skill_weights(mid_panel, s_mat, mid_win.decision_rows, buffered=True)
            ws = benchmark_tilt._apply_open_fills(ws_dec, mid_win.tradable)
            bs = benchmark_tilt._apply_open_fills(bs_dec, mid_win.tradable)
            ps = benchmark_tilt.simulate_portfolio_path(ws, mid_win.stock_returns, bs)
            bs_g = np.einsum("ki,ki->k", np.where(np.isfinite(bs), bs, 0.0), mid_win.stock_returns)
            bs_g[~np.isfinite(bs).all(axis=1)] = np.nan
            s_sum = core_satellite._pack_joint_summary(ps, bs_g, "equal_weight_mid_small", mid_win, horizon)
            sat_standalone_2021.setdefault(s_name, {})[h_key] = s_sum
            sat_w_map[s_name] = ws
            s_ex = s_sum["vs_matched_benchmark"]["net_excess_20bps_per_period"]
            s_t = s_sum["vs_matched_benchmark"]["newey_west_20bps"]["t"]
            sat_ok_gate[s_name] = bool(
                s_ex is not None and float(s_ex) > 0.0 and s_t is not None and float(s_t) >= core_satellite.REQUIRED_SAT_SKILL_NW_T
            )

        for alloc_tag, c_weight in core_satellite.ALLOCATIONS:
            for core_label, c_mode, sp_w_arg in (
                ("spy", "core_spy", None),
                ("sp500_tilt_news", "core_sp500_tilt", sp_news_w),
            ):
                # R1: park satellite in SPY on backwardation rebalance dates
                spec_r1 = f"{alloc_tag}__{core_label}__R1_backwardation_to_spy"
                sum_r1 = _simulate_r1_backwardation_to_spy(
                    core_mode=c_mode,
                    core_weight=c_weight,
                    sp_tilt_weights=sp_w_arg,
                    mid_sat_weights=sat_w_map["R1_base_bn_sec_xgb"],
                    is_backwardation_at_decisions=back_at_dec,
                    sp_windows=sp_win,
                    mid_windows=mid_win,
                    horizon=horizon,
                )
                # R2 & R3: continuous joint simulation
                for pol_key, s_ref in (
                    ("R2_adaptive_hedge_sec_xgb", "R2_adaptive_hedge_sec_xgb"),
                    ("R3_vol_interaction_xgb", "R3_vol_interaction_xgb"),
                ):
                    spec_id = f"{alloc_tag}__{core_label}__{pol_key}"
                    j_sum = core_satellite.simulate_joint_core_satellite(
                        core_mode=c_mode,
                        core_weight=c_weight,
                        sp_tilt_weights=sp_w_arg,
                        mid_sat_weights=sat_w_map[s_ref],
                        sp_windows=sp_win,
                        mid_windows=mid_win,
                        horizon=horizon,
                    )
                    grid_2021_2024.setdefault(spec_id, {})[h_key] = j_sum

                grid_2021_2024.setdefault(spec_r1, {})[h_key] = sum_r1

                for spec_id, sat_gate_key in (
                    (spec_r1, "R1_base_bn_sec_xgb"),
                    (f"{alloc_tag}__{core_label}__R2_adaptive_hedge_sec_xgb", "R2_adaptive_hedge_sec_xgb"),
                    (f"{alloc_tag}__{core_label}__R3_vol_interaction_xgb", "R3_vol_interaction_xgb"),
                ):
                    entry = grid_2021_2024[spec_id][h_key]
                    spy_ex = entry["vs_spy"]["net_excess_20bps_per_period"]
                    spy_t = entry["vs_spy"]["newey_west_20bps"]["t"]
                    pos_y = int(entry["vs_spy"]["positive_years_20bps"])
                    tot_y = int(entry["vs_spy"]["years_scored"])
                    yr_ok = tot_y > 0 and (pos_y / float(tot_y)) >= 0.75
                    passes = bool(
                        spy_ex is not None
                        and float(spy_ex) > 0.0
                        and spy_t is not None
                        and float(spy_t) >= core_satellite.REQUIRED_SPY_NW_T
                        and yr_ok
                        and sat_ok_gate[sat_gate_key]
                    )
                    passes_t3 = bool(passes and spy_t is not None and float(spy_t) >= core_satellite.STRICT_SPY_NW_T)
                    entry["satellite_passes_gate"] = sat_ok_gate[sat_gate_key]
                    entry["passes_reading_rule"] = passes
                    entry["passes_strict_t3"] = passes_t3
                    if passes:
                        passing_specs.append(f"{spec_id}@h={horizon}")
                    if passes_t3:
                        passing_strict_t3.append(f"{spec_id}@h={horizon}")

        # Full 5-year OOS 2020..2024 (`R1` and `R2` zero-fit volatility regime rules across COVID 2020!)
        sp_win20 = benchmark_tilt.continuous_rebalance_windows(sp500_panel, stage1_oos_start_row, horizon)
        mid_win20 = benchmark_tilt.continuous_rebalance_windows(mid_panel, stage1_oos_start_row, horizon)
        back_dec20 = is_back[sp_win20.decision_rows] > 0.5
        sp_eq20_dec, _ = benchmark_tilt.build_cap_tilt_weights(
            sp500_panel, sp500_eq24, sp500_mv, sp_win20.decision_rows, mode="sector_active_slice",
        )
        sp_eq20_w = benchmark_tilt._apply_open_fills(sp_eq20_dec, sp_win20.tradable)
        w_bn20_dec, _ = benchmark_tilt.build_skill_weights(mid_panel, mid_bn_eq24, mid_win20.decision_rows, buffered=True)
        w_bn20 = benchmark_tilt._apply_open_fills(w_bn20_dec, mid_win20.tradable)
        w_r2_20_dec, _ = benchmark_tilt.build_skill_weights(mid_panel, r2_stage1_sig, mid_win20.decision_rows, buffered=True)
        w_r2_20 = benchmark_tilt._apply_open_fills(w_r2_20_dec, mid_win20.tradable)

        for alloc_tag, c_weight in core_satellite.ALLOCATIONS:
            for core_label, c_mode, sp_w_arg in (
                ("spy", "core_spy", None),
                ("sp500_tilt_eq24", "core_sp500_tilt", sp_eq20_w),
            ):
                grid_2020_2024.setdefault(f"{alloc_tag}__{core_label}__R1_backwardation_to_spy", {})[h_key] = (
                    _simulate_r1_backwardation_to_spy(
                        core_mode=c_mode,
                        core_weight=c_weight,
                        sp_tilt_weights=sp_w_arg,
                        mid_sat_weights=w_bn20,
                        is_backwardation_at_decisions=back_dec20,
                        sp_windows=sp_win20,
                        mid_windows=mid_win20,
                        horizon=horizon,
                    )
                )
                grid_2020_2024.setdefault(f"{alloc_tag}__{core_label}__R2_adaptive_hedge_eq24", {})[h_key] = (
                    core_satellite.simulate_joint_core_satellite(
                        core_mode=c_mode,
                        core_weight=c_weight,
                        sp_tilt_weights=sp_w_arg,
                        mid_sat_weights=w_r2_20,
                        sp_windows=sp_win20,
                        mid_windows=mid_win20,
                        horizon=horizon,
                    )
                )

    return {
        "satellites_standalone_2021_2024": sat_standalone_2021,
        "vol_regime_2021_2024": grid_2021_2024,
        "vol_regime_2020_2024": grid_2020_2024,
        "reading_summary": {
            "rule": core_satellite.CORE_SATELLITE_READING_RULE,
            "passing_specifications": passing_specs,
            "passing_strict_t3": passing_strict_t3,
            "earns_sealed_period_test": len(passing_specs) > 0,
        },
    }


def render_vol_regime_markdown(document: Mapping[str, Any]) -> str:
    """Render readable markdown scoreboard for Step 5e CBOE Volatility Term-Structure Regime Conditioning."""
    settings = document["settings"]
    res = document["vol_regime"]
    notice = document["notice"]
    horizons = [str(h) for h in settings["horizons_sessions"]]
    pct = core_satellite._pct
    num = core_satellite._num

    lines = [
        "# Step 5e: CBOE Index Volatility Term-Structure & Skew Regime Conditioning vs. `SPY`",
        "",
        f"Every figure below: {notice}.",
        "",
        "## Method & boundaries",
        "",
        f"- **CBOE daily indices (`<= {settings['development_end']}`):** `VIX` (30d), `VIX3M` (90d), `VIX9D` (9d), "
        f"`VXN` (Nasdaq-100 30d), `SKEW`. Backwardation share across `2016–2024` sessions: "
        f"`{pct(document['regime_diagnostics']['backwardation_session_share'])}`.",
        "- **Policies:** `R1_backwardation_to_spy` (satellite shifts to `SPY` when `VIX > VIX3M`), "
        "`R2_adaptive_hedge` (`raw` in contango, `0.7*beta_neutral + 0.3*reversal_5d` in backwardation), "
        "`R3_vol_interaction_xgb` (purged walk-forward shallow XGBoost with `beta * term` & `rev5 * term` interactions).",
        "",
        "## Section A: Standalone Volatility-Conditioned Mid/Small Satellites (OOS 2021..2024)",
        "",
        "| Satellite | H | Turnover | Net 20bps | SPY | Net vs Matched/yr [NW t] | Net vs SPY/yr [NW t] | Sharpe | MaxDD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s_name, h_map in res["satellites_standalone_2021_2024"].items():
        for h in horizons:
            e = h_map[h]
            vm = e["vs_matched_benchmark"]
            vs = e["vs_spy"]
            lines.append(
                f"| {s_name} | {h} | {pct(e['mean_steady_one_way_turnover'])} | {pct(e['compounded_net_20bps'])} | "
                f"{pct(vs['compounded_benchmark'])} | {pct(vm['net_excess_20bps_annualized'])} [{num(vm['newey_west_20bps']['t'])}] | "
                f"{pct(vs['net_excess_20bps_annualized'])} [{num(vs['newey_west_20bps']['t'])}] | "
                f"{num(e['risk_net_20bps']['sharpe_annualized'])} | {pct(e['risk_net_20bps']['max_drawdown'])} |"
            )

    lines += [
        "",
        "## Section B: Volatility-Regime Core-Satellite Portfolios vs. `SPY` (OOS 2021..2024, Net of 20 bps)",
        "",
        "| Specification | H | Turnover | Port net 20bps | SPY | Net vs SPY/yr [NW t] | "
        "2021 | 2022 | 2023 | 2024 | Pos yrs | Sharpe | MaxDD | Passes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for spec_id, h_map in res["vol_regime_2021_2024"].items():
        for h in horizons:
            e = h_map[h]
            vs = e["vs_spy"]
            by_y = vs["by_year_net_20bps"]
            y_cells = [pct(by_y.get(y, {}).get("mean")) for y in ("2021", "2022", "2023", "2024")]
            lines.append(
                f"| {spec_id} | {h} | {pct(e['mean_steady_one_way_turnover'])} | {pct(e['compounded_net_20bps'])} | "
                f"{pct(vs['compounded_benchmark'])} | {pct(vs['net_excess_20bps_annualized'])} [{num(vs['newey_west_20bps']['t'])}] | "
                + " | ".join(y_cells)
                + f" | {vs['positive_years_20bps']}/{vs['years_scored']} | "
                f"{num(e['risk_net_20bps']['sharpe_annualized'])} | {pct(e['risk_net_20bps']['max_drawdown'])} | "
                f"{'yes' if e['passes_reading_rule'] else 'no'} |"
            )

    lines += [
        "",
        "## Section C: Full 5-Year OOS 2020..2024 (`R1` & `R2` Including March 2020 VIX Backwardation Spike)",
        "",
        "| Specification | H | Turnover | Port net 20bps | SPY | Net vs SPY/yr [NW t] | "
        "2020 | 2021 | 2022 | 2023 | 2024 | Pos yrs | MaxDD (port / SPY) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for spec_id, h_map in res["vol_regime_2020_2024"].items():
        for h in horizons:
            e = h_map[h]
            vs = e["vs_spy"]
            by_y = vs["by_year_net_20bps"]
            y_cells = [pct(by_y.get(y, {}).get("mean")) for y in ("2020", "2021", "2022", "2023", "2024")]
            lines.append(
                f"| {spec_id} | {h} | {pct(e['mean_steady_one_way_turnover'])} | {pct(e['compounded_net_20bps'])} | "
                f"{pct(vs['compounded_benchmark'])} | {pct(vs['net_excess_20bps_annualized'])} [{num(vs['newey_west_20bps']['t'])}] | "
                + " | ".join(y_cells)
                + f" | {vs['positive_years_20bps']}/{vs['years_scored']} | "
                f"{pct(e['risk_net_20bps']['max_drawdown'])} / {pct(e['risk_spy']['max_drawdown'])} |"
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
    "CBOE_CSV_URLS",
    "CBOE_DIRECTORY",
    "CBOE_PARQUET_NAME",
    "COLLECT_CBOE_COMMAND",
    "INTERACTION_FEATURE_NAMES",
    "VOL_REGIME_COMMAND",
    "build_causal_vol_regime_features",
    "build_r2_adaptive_hedge_signal",
    "collect_cboe_vol_indices",
    "evaluate_vol_regime_core_satellite",
    "fit_r3_vol_interaction_xgb_by_horizon",
    "parse_cboe_csv_truncated",
    "render_vol_regime_markdown",
]
