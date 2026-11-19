"""Stage-2 purged walk-forward meta-labeling and continuous SPY-excess models on candidate setups.

Evaluates both deterministic Stage-1 rules (Setups A, B, C, SPY bull regime, and ADR% screens)
and Stage-2 machine-learning filters over the out-of-sample scored years 2021..2024:
- Track 1 (Calibrated XGBClassifier): ``binary:logistic`` on Triple-Barrier ``y = 1`` (+2.5R
  before -1.0R in 10 sessions), calibrated with chronological out-of-fold Platt scaling inside
  the purged training history (never touching test rows).
- Track 2 (Continuous SPY-Excess Regressors): shallow ``XGBRegressor`` and standardized ``Ridge``
  predicting 10-session volatility-scaled excess return over SPY.

Strict Purge Invariant: a candidate event on session ``t`` enters fold ``Y``'s training set only
when ``t >= first_row``, ``outcomes.valid`` is True, ``t + horizon < fold.test_start``, and
``outcomes.exit_row < fold.test_start``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from market_predictor.research_lane.barrier_labels import (
    BARRIER_HORIZON_SESSIONS,
    FROZEN_ROUND_TRIP_COST,
    GOVERNED_STOP_MULTIPLE,
    GOVERNED_TARGET_MULTIPLE,
    OUTCOME_STOP,
    OUTCOME_TARGET,
    OUTCOME_TIMEOUT,
    CandidateOutcomes,
    evaluate_candidate_barriers,
)
from market_predictor.research_lane.fitted import SCORED_YEARS, Fold, scored_folds, session_years
from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.scoreboard import newey_west
from market_predictor.research_lane.setups import (
    SETUP_FEATURE_NAMES,
    SHOCK_RATIO_THRESHOLD,
    SetupDetection,
)

CLASSIFIER_PROB_THRESHOLD: Final = 0.42
TOP_QUANTILE_CUTOFF: Final = 0.75
RIDGE_META_ALPHA: Final = 100.0
META_SEEDS: Final = (0, 1, 2)

XGB_CLASSIFIER_PARAMS: Final[Mapping[str, Any]] = MappingProxyType({
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "n_estimators": 200,
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.02,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "n_jobs": 4,
})

XGB_REGRESSOR_PARAMS: Final[Mapping[str, Any]] = MappingProxyType({
    "objective": "reg:squarederror",
    "n_estimators": 200,
    "max_depth": 3,
    "min_child_weight": 50,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "learning_rate": 0.02,
    "reg_lambda": 10.0,
    "tree_method": "hist",
    "n_jobs": 4,
})

MINIMUM_OOS_TRADES: Final = 100
MINIMUM_OOS_ENTRY_DATES: Final = 40
REQUIRED_EXCESS_NW_T: Final = 2.5
REQUIRED_POSITIVE_OOS_YEARS: Final = 3

STEP3A_READING_RULE: Final = (
    "A candidate specification earns one test on the sealed period only if, across out-of-sample years "
    "2021..2024 (to 2024-05-28), it selects at least 100 trades across at least 40 distinct entry dates, "
    "achieves positive standalone net return and positive net R after the 0.20% round-trip cost, and beats "
    "matched-interval SPY after the 0.20% cost with entry-date-clustered Newey-West (lag 10) t >= 2.5 and "
    "positive net SPY excess in at least 3 of 4 scored calendar years."
)

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class CandidateTable:
    """Extracted candidate events for a Stage-1 mask."""

    rows: IntArray
    cols: IntArray
    years: IntArray
    matrix: Float32Array
    spy_bull: BoolArray
    shock_bucket_b: BoolArray
    vol_pct: FloatArray
    setup_a: BoolArray
    setup_b: BoolArray
    setup_c: BoolArray
    outcomes: CandidateOutcomes
    governed_outcomes: CandidateOutcomes


def extract_candidate_table(
    panel: ResearchPanel,
    detection: SetupDetection,
    mask: BoolArray,
    *,
    horizon: int = BARRIER_HORIZON_SESSIONS,
) -> CandidateTable:
    """Extract candidate events, stationary feature rows and barrier outcomes for ``mask``."""
    rows, cols = np.nonzero(mask)
    rows_i = rows.astype(np.int64)
    cols_i = cols.astype(np.int64)
    n = int(rows_i.size)
    matrix = np.empty((n, len(SETUP_FEATURE_NAMES)), dtype=np.float32)
    for j, name in enumerate(SETUP_FEATURE_NAMES):
        matrix[:, j] = detection.features[name][rows_i, cols_i]

    all_years = session_years(panel.sessions)
    years = all_years[rows_i]
    spy_bull = detection.spy_bull_regime[rows_i]
    shock_vals = detection.shock_ratio_14[rows_i, cols_i]
    shock_b = np.isfinite(shock_vals) & (shock_vals > SHOCK_RATIO_THRESHOLD)
    close_vals = panel.close.astype(np.float64)[rows_i, cols_i]
    vol_vals = detection.volatility[rows_i, cols_i]
    vol_pct = np.where(close_vals > 0.0, vol_vals / close_vals, np.nan)

    outcomes = evaluate_candidate_barriers(panel, detection.volatility, rows_i, cols_i, horizon=horizon)
    governed = evaluate_candidate_barriers(
        panel,
        detection.volatility,
        rows_i,
        cols_i,
        target_multiple=GOVERNED_TARGET_MULTIPLE,
        stop_multiple=GOVERNED_STOP_MULTIPLE,
        horizon=horizon,
    )
    return CandidateTable(
        rows=rows_i,
        cols=cols_i,
        years=years,
        matrix=matrix,
        spy_bull=spy_bull,
        shock_bucket_b=shock_b,
        vol_pct=vol_pct,
        setup_a=detection.setup_a[rows_i, cols_i],
        setup_b=detection.setup_b[rows_i, cols_i],
        setup_c=detection.setup_c[rows_i, cols_i],
        outcomes=outcomes,
        governed_outcomes=governed,
    )


def purged_train_mask(table: CandidateTable, *, first_row: int, test_start: int,
                      horizon: int = BARRIER_HORIZON_SESSIONS) -> BoolArray:
    """True only for valid candidate rows whose barrier exit AND fixed horizon end strictly before ``test_start``."""
    return np.asarray(
        table.outcomes.valid
        & (table.rows >= first_row)
        & ((table.rows + horizon) < test_start)
        & (table.outcomes.exit_row >= 0)
        & (table.outcomes.exit_row < test_start),
        dtype=bool,
    )


def event_date_weights(rows: IntArray) -> FloatArray:
    """Equal total weight per active training session, rescaled so mean weight is 1."""
    if rows.size == 0:
        return np.zeros(0, dtype=np.float64)
    unique_rows, inverse, counts = np.unique(rows, return_inverse=True, return_counts=True)
    raw = 1.0 / counts[inverse].astype(np.float64)
    return np.asarray(raw * (rows.size / unique_rows.size), dtype=np.float64)


def _logit(prob: FloatArray) -> FloatArray:
    clipped = np.clip(prob, 1e-6, 1.0 - 1e-6)
    return np.asarray(np.log(clipped / (1.0 - clipped)), dtype=np.float64)


def _sigmoid(z: FloatArray) -> FloatArray:
    clipped = np.clip(z, -30.0, 30.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _fit_xgb_classifier_ensemble(x: Float32Array, y: npt.NDArray[np.int8], weights: FloatArray) -> list[XGBClassifier]:
    pos = int((y == 1).sum())
    if pos == 0 or pos == int(y.size):
        return []
    # Adapt min_child_weight safely if early inner folds have fewer than 250 candidate rows.
    mcw = min(int(XGB_CLASSIFIER_PARAMS["min_child_weight"]), max(10, int(y.size // 8)))
    params = {**dict(XGB_CLASSIFIER_PARAMS), "min_child_weight": mcw}
    models: list[XGBClassifier] = []
    for seed in META_SEEDS:
        clf = XGBClassifier(**params, random_state=seed, importance_type="gain")
        clf.fit(x, y, sample_weight=weights)
        models.append(clf)
    return models


def _predict_raw_prob(models: Sequence[XGBClassifier], x: Float32Array, fallback_rate: float) -> FloatArray:
    if not models or x.shape[0] == 0:
        return np.full(x.shape[0], fallback_rate, dtype=np.float64)
    probs = [np.asarray(m.predict_proba(x)[:, 1], dtype=np.float64) for m in models]
    return np.asarray(np.mean(probs, axis=0), dtype=np.float64)


def _fit_platt_calibrator(
    x_train: Float32Array,
    y_train: npt.NDArray[np.int8],
    train_rows: IntArray,
    *,
    horizon: int,
) -> tuple[float, float]:
    """Learn monotone Platt scaling (slope >= 0, intercept) from chronological purged inner splits."""
    unique_sessions = np.unique(train_rows)
    if unique_sessions.size < 40:
        return 1.0, 0.0
    # Two forward-chaining inner calibration blocks (50%->75% and 75%->100% of training sessions).
    s1 = int(unique_sessions[int(unique_sessions.size * 0.50)])
    s2 = int(unique_sessions[int(unique_sessions.size * 0.75)])
    oof_logits: list[FloatArray] = []
    oof_y: list[npt.NDArray[np.int8]] = []
    for cal_start, cal_stop in ((s1, s2), (s2, int(unique_sessions[-1]) + 1)):
        inner_tr = (train_rows + horizon) < cal_start
        inner_cal = (train_rows >= cal_start) & (train_rows < cal_stop)
        if int(inner_tr.sum()) < 60 or int(inner_cal.sum()) < 20:
            continue
        y_sub = y_train[inner_tr]
        if int((y_sub == 1).sum()) < 5 or int((y_sub == 0).sum()) < 5:
            continue
        w_sub = event_date_weights(train_rows[inner_tr])
        models = _fit_xgb_classifier_ensemble(x_train[inner_tr], y_sub, w_sub)
        if not models:
            continue
        raw_p = _predict_raw_prob(models, x_train[inner_cal], float(y_sub.mean()))
        oof_logits.append(_logit(raw_p))
        oof_y.append(y_train[inner_cal])
    if not oof_logits:
        return 1.0, 0.0
    z_all = np.concatenate(oof_logits).reshape(-1, 1)
    y_all = np.concatenate(oof_y)
    if int((y_all == 1).sum()) < 5 or int((y_all == 0).sum()) < 5:
        return 1.0, 0.0
    lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
    lr.fit(z_all, y_all)
    slope = float(lr.coef_[0, 0])
    intercept = float(lr.intercept_[0])
    # Monotonicity guard: never allow noisy small-sample calibration to invert ranking direction.
    if not np.isfinite(slope) or slope <= 0.05:
        return 1.0, 0.0
    return slope, intercept


def _encode_ridge(x_train: Float32Array, x_test: Float32Array, weights: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Train-only median imputation (sessions_since_catalyst=99 when absent) + weighted standardization."""
    tr = x_train.astype(np.float64, copy=True)
    te = x_test.astype(np.float64, copy=True)
    since_idx = SETUP_FEATURE_NAMES.index("sessions_since_catalyst")
    tr[:, since_idx] = np.where(np.isnan(tr[:, since_idx]), 99.0, tr[:, since_idx])
    te[:, since_idx] = np.where(np.isnan(te[:, since_idx]), 99.0, te[:, since_idx])
    for col in range(tr.shape[1]):
        col_vals = tr[:, col]
        finite = np.isfinite(col_vals)
        med = float(np.median(col_vals[finite])) if bool(finite.any()) else 0.0
        tr[:, col] = np.where(finite, col_vals, med)
        te[:, col] = np.where(np.isfinite(te[:, col]), te[:, col], med)
    scaler = StandardScaler()
    scaler.fit(tr, sample_weight=weights)
    return np.asarray(scaler.transform(tr), dtype=np.float64), np.asarray(scaler.transform(te), dtype=np.float64)


def summarize_trades(
    table: CandidateTable,
    selected: BoolArray,
    *,
    panel_sessions: npt.NDArray[np.datetime64],
    scored_years: Sequence[int] = SCORED_YEARS,
    use_governed_barriers: bool = False,
    horizon: int = BARRIER_HORIZON_SESSIONS,
) -> dict[str, Any]:
    """Compute trade-level and entry-date-clustered Newey-West metrics for ``selected`` candidates."""
    out = table.governed_outcomes if use_governed_barriers else table.outcomes
    chosen = selected & out.valid
    n_trades = int(chosen.sum())
    if n_trades == 0:
        return {
            "trades": 0,
            "entry_dates": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "timeout_rate": None,
            "positive_net_20bps_win_rate": None,
            "mean_holding_sessions": None,
            "mean_gross_return": None,
            "mean_net_return_20bps": None,
            "mean_net_return_10bps": None,
            "mean_net_r_20bps": None,
            "mean_spy_matched_return": None,
            "mean_net_excess_spy_20bps": None,
            "mean_fixed_10d_net_20bps": None,
            "mean_fixed_10d_net_excess_spy_20bps": None,
            "daily_cohort_net_20bps_nw": {"mean": None, "standard_error": None, "t": None, "count": 0},
            "daily_cohort_excess_spy_20bps_nw": {"mean": None, "standard_error": None, "t": None, "count": 0},
            "by_year": {},
            "positive_excess_years": 0,
            "years_with_trades": 0,
            "passes_reading_rule": False,
        }

    codes = out.outcome_code[chosen]
    net20 = out.net_return_20bps[chosen]
    net10 = out.net_return_10bps[chosen]
    net_r = out.net_r_20bps[chosen]
    gross = out.gross_return[chosen]
    spy_m = out.spy_matched_return[chosen]
    ex20 = out.net_excess_spy_20bps[chosen]
    f_net20 = out.fixed_net_20bps[chosen]
    f_ex20 = out.fixed_net_excess_spy_20bps[chosen]
    hold = out.holding_sessions[chosen].astype(np.float64)
    c_rows = table.rows[chosen]
    c_years = table.years[chosen]

    # Entry-session equal-weighted daily cohorts on the full panel calendar (preserving session spacing for NW lag).
    daily_net = np.full(len(panel_sessions), np.nan, dtype=np.float64)
    daily_ex = np.full(len(panel_sessions), np.nan, dtype=np.float64)
    for u_row in np.unique(c_rows):
        in_day = c_rows == u_row
        daily_net[int(u_row)] = float(net20[in_day].mean())
        daily_ex[int(u_row)] = float(ex20[in_day].mean())

    nw_net = newey_west(daily_net, horizon)
    nw_ex = newey_west(daily_ex, horizon)
    n_dates = int(np.isfinite(daily_ex).sum())

    by_year: dict[str, dict[str, Any]] = {}
    pos_ex_years = 0
    years_present = 0
    for yr in scored_years:
        in_yr = c_years == int(yr)
        cnt = int(in_yr.sum())
        if cnt == 0:
            by_year[str(yr)] = {"trades": 0, "mean_net_20bps": None, "mean_net_r_20bps": None,
                                "mean_net_excess_spy_20bps": None, "target_hit_rate": None}
            continue
        years_present += 1
        yr_ex = float(ex20[in_yr].mean())
        if yr_ex > 0.0:
            pos_ex_years += 1
        by_year[str(yr)] = {
            "trades": cnt,
            "mean_net_20bps": float(net20[in_yr].mean()),
            "mean_net_r_20bps": float(net_r[in_yr].mean()),
            "mean_net_excess_spy_20bps": yr_ex,
            "target_hit_rate": float((codes[in_yr] == OUTCOME_TARGET).mean()),
        }

    mean_net20 = float(net20.mean())
    mean_net_r = float(net_r.mean())
    mean_ex20 = float(ex20.mean())
    nw_t = nw_ex.get("t")
    passes = bool(
        n_trades >= MINIMUM_OOS_TRADES
        and n_dates >= MINIMUM_OOS_ENTRY_DATES
        and mean_net20 > 0.0
        and mean_net_r > 0.0
        and mean_ex20 > 0.0
        and nw_t is not None
        and float(nw_t) >= REQUIRED_EXCESS_NW_T
        and pos_ex_years >= REQUIRED_POSITIVE_OOS_YEARS
    )

    return {
        "trades": n_trades,
        "entry_dates": n_dates,
        "target_hit_rate": float((codes == OUTCOME_TARGET).mean()),
        "stop_hit_rate": float((codes == OUTCOME_STOP).mean()),
        "timeout_rate": float((codes == OUTCOME_TIMEOUT).mean()),
        "positive_net_20bps_win_rate": float((net20 > 0.0).mean()),
        "mean_holding_sessions": float(hold.mean()),
        "mean_gross_return": float(gross.mean()),
        "mean_net_return_20bps": mean_net20,
        "mean_net_return_10bps": float(net10.mean()),
        "mean_net_r_20bps": mean_net_r,
        "mean_spy_matched_return": float(spy_m.mean()),
        "mean_net_excess_spy_20bps": mean_ex20,
        "mean_fixed_10d_net_20bps": float(f_net20.mean()),
        "mean_fixed_10d_net_excess_spy_20bps": float(f_ex20.mean()),
        "daily_cohort_net_20bps_nw": nw_net,
        "daily_cohort_excess_spy_20bps_nw": nw_ex,
        "by_year": by_year,
        "positive_excess_years": pos_ex_years,
        "years_with_trades": years_present,
        "passes_reading_rule": passes,
    }


@dataclass(frozen=True)
class WalkForwardMetaResult:
    """Out-of-sample meta-model predictions, diagnostics and specification scorecards."""

    prob_calibrated: FloatArray
    prob_raw: FloatArray
    pred_xgb_vol_excess: FloatArray
    pred_ridge_vol_excess: FloatArray
    prob_top_q_selected: BoolArray
    folds: list[dict[str, Any]]
    classifier_importance: list[dict[str, Any]]
    regressor_importance: list[dict[str, Any]]
    specifications: dict[str, dict[str, Any]]
    calibration_buckets: list[dict[str, Any]]
    shock_stratification: dict[str, Any]


def run_walk_forward_meta_models(
    panel: ResearchPanel,
    detection: SetupDetection,
    *,
    first_row: int,
    folds: Sequence[Fold] | None = None,
    horizon: int = BARRIER_HORIZON_SESSIONS,
    checkpoint: Callable[[str], None] | None = None,
) -> tuple[CandidateTable, WalkForwardMetaResult]:
    """Run purged walk-forward Stage-1 and Stage-2 evaluations on primary candidate setups."""
    if folds is not None:
        active_folds = tuple(folds)
    else:
        present_years = [y for y in SCORED_YEARS if bool(np.any(session_years(panel.sessions) == y))]
        active_folds = scored_folds(panel.sessions, present_years)
    table = extract_candidate_table(panel, detection, detection.any_setup_primary, horizon=horizon)
    n = int(table.rows.size)

    prob_cal = np.full(n, np.nan, dtype=np.float64)
    prob_raw = np.full(n, np.nan, dtype=np.float64)
    pred_xgb_ex = np.full(n, np.nan, dtype=np.float64)
    pred_ridge_ex = np.full(n, np.nan, dtype=np.float64)
    prob_top_q = np.zeros(n, dtype=bool)
    oos_mask = np.zeros(n, dtype=bool)

    clf_imp: list[FloatArray] = []
    reg_imp: list[FloatArray] = []
    fold_reports: list[dict[str, Any]] = []

    for fold in active_folds:
        tr_mask = purged_train_mask(table, first_row=first_row, test_start=fold.test_start, horizon=horizon)
        te_mask = (table.rows >= fold.test_start) & (table.rows < fold.test_stop) & table.outcomes.valid
        oos_mask |= te_mask
        n_tr = int(tr_mask.sum())
        n_te = int(te_mask.sum())

        x_tr = table.matrix[tr_mask]
        x_te = table.matrix[te_mask]
        y_tr_bin = table.outcomes.y_barrier[tr_mask]
        y_tr_ex = table.outcomes.vol_scaled_fixed_excess_spy[tr_mask]
        tr_rows = table.rows[tr_mask]
        weights = event_date_weights(tr_rows)

        base_rate = float(y_tr_bin.mean()) if n_tr > 0 else 0.0
        slope, intercept = (1.0, 0.0)
        q75_cut = CLASSIFIER_PROB_THRESHOLD

        if n_tr >= 40 and n_te > 0:
            slope, intercept = _fit_platt_calibrator(x_tr, y_tr_bin, tr_rows, horizon=horizon)
            clf_models = _fit_xgb_classifier_ensemble(x_tr, y_tr_bin, weights)
            raw_te = _predict_raw_prob(clf_models, x_te, base_rate)
            cal_te = _sigmoid(slope * _logit(raw_te) + intercept)
            raw_tr = _predict_raw_prob(clf_models, x_tr, base_rate)
            cal_tr = _sigmoid(slope * _logit(raw_tr) + intercept)
            q75_cut = float(np.quantile(cal_tr, TOP_QUANTILE_CUTOFF))
            prob_raw[te_mask] = raw_te
            prob_cal[te_mask] = cal_te
            prob_top_q[te_mask] = cal_te >= q75_cut
            if clf_models:
                clf_imp.append(
                    np.mean([np.asarray(m.feature_importances_, dtype=np.float64) for m in clf_models], axis=0)
                )

            mcw_reg = min(int(XGB_REGRESSOR_PARAMS["min_child_weight"]), max(10, n_tr // 8))
            reg_params = {**dict(XGB_REGRESSOR_PARAMS), "min_child_weight": mcw_reg}
            reg_models: list[XGBRegressor] = []
            for seed in META_SEEDS:
                reg = XGBRegressor(**reg_params, random_state=seed, importance_type="gain")
                reg.fit(x_tr, y_tr_ex, sample_weight=weights)
                reg_models.append(reg)
            pred_xgb_ex[te_mask] = np.mean(
                [np.asarray(m.predict(x_te), dtype=np.float64) for m in reg_models], axis=0
            )
            reg_imp.append(
                np.mean([np.asarray(m.feature_importances_, dtype=np.float64) for m in reg_models], axis=0)
            )

            tr_std, te_std = _encode_ridge(x_tr, x_te, weights)
            ridge = Ridge(alpha=RIDGE_META_ALPHA, fit_intercept=True)
            ridge.fit(tr_std, y_tr_ex, sample_weight=weights)
            pred_ridge_ex[te_mask] = np.asarray(ridge.predict(te_std), dtype=np.float64)

        fold_reports.append({
            "scored_year": fold.year,
            "train_candidates": n_tr,
            "train_entry_dates": int(np.unique(tr_rows).size) if n_tr else 0,
            "train_positive_barrier_rate": base_rate if n_tr else None,
            "last_train_decision_session": str(panel.sessions[int(tr_rows.max())]) if n_tr else None,
            "last_train_barrier_exit_session": (
                str(panel.sessions[int(table.outcomes.exit_row[tr_mask].max())]) if n_tr else None
            ),
            "first_test_session": str(panel.sessions[fold.test_start]),
            "last_test_session": str(panel.sessions[fold.test_stop - 1]),
            "test_valid_candidates": n_te,
            "platt_slope": slope,
            "platt_intercept": intercept,
            "train_q75_calibrated_prob": q75_cut,
        })
        if checkpoint is not None:
            checkpoint(f"research lane step 3a meta-models fold {fold.year}")

    # Evaluate pre-registered Stage-1 and Stage-2 specifications strictly on out-of-sample years (2021..2024).
    xgb_expected_excess = pred_xgb_ex * table.vol_pct
    ridge_expected_excess = pred_ridge_ex * table.vol_pct
    specs_masks: dict[str, tuple[BoolArray, bool]] = {
        "stage1_all_primary_adr15": (oos_mask, False),
        "stage1_all_spy_bull_regime": (oos_mask & table.spy_bull, False),
        "stage1_setup_a_catalyst": (oos_mask & table.setup_a, False),
        "stage1_setup_b_pullback": (oos_mask & table.setup_b, False),
        "stage1_setup_c_squeeze": (oos_mask & table.setup_c, False),
        "stage1_all_governed_3_0_1_5_atr": (oos_mask, True),
        "stage2_xgb_barrier_p_gt_042": (
            oos_mask & np.isfinite(prob_cal) & (prob_cal > CLASSIFIER_PROB_THRESHOLD),
            False,
        ),
        "stage2_xgb_barrier_top_q25": (oos_mask & prob_top_q, False),
        "stage2_xgb_vol_excess_gt_20bps": (
            oos_mask & np.isfinite(xgb_expected_excess) & (xgb_expected_excess > FROZEN_ROUND_TRIP_COST),
            False,
        ),
        "stage2_ridge_vol_excess_gt_20bps": (
            oos_mask & np.isfinite(ridge_expected_excess) & (ridge_expected_excess > FROZEN_ROUND_TRIP_COST),
            False,
        ),
    }

    scored_years = tuple(f.year for f in active_folds)
    specifications = {
        name: summarize_trades(
            table,
            sel_mask,
            panel_sessions=panel.sessions,
            scored_years=scored_years,
            use_governed_barriers=gov_flag,
            horizon=horizon,
        )
        for name, (sel_mask, gov_flag) in specs_masks.items()
    }

    # Out-of-sample probability calibration table across probability bins.
    cal_valid = oos_mask & np.isfinite(prob_cal)
    cal_buckets: list[dict[str, Any]] = []
    edges = (0.0, 0.25, 0.35, 0.42, 1.01)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = cal_valid & (prob_cal >= lo) & (prob_cal < hi)
        cnt = int(in_bin.sum())
        cal_buckets.append({
            "bin": f"[{lo:.2f}, {min(hi, 1.0):.2f}]",
            "candidates": cnt,
            "mean_predicted_prob": float(prob_cal[in_bin].mean()) if cnt else None,
            "empirical_target_hit_rate": float(table.outcomes.y_barrier[in_bin].mean()) if cnt else None,
            "mean_net_r_20bps": float(table.outcomes.net_r_20bps[in_bin].mean()) if cnt else None,
            "mean_net_excess_spy_20bps": float(table.outcomes.net_excess_spy_20bps[in_bin].mean()) if cnt else None,
        })

    # Shock-regime diagnostic (Bucket A normal <= 3x MTR vs Bucket B post-shock > 3x MTR) on OOS Stage-1 events.
    shock_strat = {
        "bucket_a_normal_vol": summarize_trades(
            table, oos_mask & ~table.shock_bucket_b, panel_sessions=panel.sessions, scored_years=scored_years,
            horizon=horizon,
        ),
        "bucket_b_post_shock_gt_3x_mtr": summarize_trades(
            table, oos_mask & table.shock_bucket_b, panel_sessions=panel.sessions, scored_years=scored_years,
            horizon=horizon,
        ),
    }

    def top_imp(items: list[FloatArray]) -> list[dict[str, Any]]:
        if not items:
            return []
        mean_imp = np.mean(items, axis=0)
        order = np.argsort(-mean_imp, kind="stable")[:12]
        return [{"feature": SETUP_FEATURE_NAMES[int(i)], "importance": float(mean_imp[int(i)])} for i in order]

    return table, WalkForwardMetaResult(
        prob_calibrated=prob_cal,
        prob_raw=prob_raw,
        pred_xgb_vol_excess=pred_xgb_ex,
        pred_ridge_vol_excess=pred_ridge_ex,
        prob_top_q_selected=prob_top_q,
        folds=fold_reports,
        classifier_importance=top_imp(clf_imp),
        regressor_importance=top_imp(reg_imp),
        specifications=specifications,
        calibration_buckets=cal_buckets,
        shock_stratification=shock_strat,
    )
