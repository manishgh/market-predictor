"""Recommendation Brief §5 & §6 Steps 2-4: Combined Stage-1 Base Models & Stage-2 News Residual Model.

Implements:
- Stage 1 (§5.3, §5.4, §6 Step 2): 24 pre-signed price + SEC financial/earnings predictors (raw and
  sector-neutral centered ranks = 48 features, no duplicate missingness indicator columns), evaluated
  with `equal_rank_24` (zero-fit baseline), `ridge_rank`, and `xgb_shallow_5seed` across 5, 10, 21,
  and 63 sessions on extended history (`2018-05-29..2024-05-28`), including side-by-side 5-fold
  unseen-issuer cross-validation (§5.6 rule 3).
- Stage 2 (§5.2, §6 Step 3): continuous news-reaction & abnormal-volume features built from retained
  Alpaca direct-issuer events (`2019-07-09..2024-05-28`), fitted strictly on out-of-sample Stage-1
  residuals and tested by paired daily rank-correlation improvement (`Newey-West t >= 2.0`).
- §5.5 & §5.6 unified Skill (buffered `buy top 10% / hold top 33%`), Product (cap-weighted tilt vs
  matched benchmark and SPY), and 5-point base acceptance rule.
"""
from __future__ import annotations

import gc
import hashlib
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from market_predictor.research_lane import benchmark_tilt, fundamentals, predictors
from market_predictor.research_lane.fitted import (
    RIDGE_ALPHA,
    Fold,
    date_weights,
    session_years,
    target_ranks,
    training_rows,
)
from market_predictor.research_lane.indicators import SECTOR_MINIMUM_STOCKS
from market_predictor.research_lane.labels import HORIZONS, research_labels
from market_predictor.research_lane.news_catalysts import (
    CATEGORY_CODES,
    EARLY_NEWS_FAMILY_DIR,
    LATER_NEWS_FAMILY_DIR,
    _pit_session_ticker_map,
)
from market_predictor.research_lane.panel import (
    MARKET_TIMEZONE,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
    file_sha256,
)
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.scoreboard import (
    daily_rank_correlation,
    newey_west,
    score_predictor,
    scored_cells,
)
from market_predictor.research_lane.sec_filings import event_rows

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

PRIMARY_HORIZON: Final = 21
SECONDARY_HORIZON: Final = 10
UNSEEN_CV_HORIZONS: Final = (10, 21)
UNSEEN_ISSUER_FOLDS: Final = 5

STAGE1_SEEDS: Final = (0, 1, 2, 3, 4)  # §5.4: averaged over five random seeds
UNSEEN_XGB_SEEDS: Final = (0, 1, 2)
STAGE1_XGB_PARAMS: Final[Mapping[str, Any]] = MappingProxyType({
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.5,
    "min_child_weight": 200,
    "reg_lambda": 10.0,
    "max_bin": 64,
    "tree_method": "hist",
    "n_jobs": 1,
})

STAGE2_XGB_SEEDS: Final = (0, 1, 2)
STAGE2_XGB_PARAMS: Final[Mapping[str, Any]] = MappingProxyType({
    "n_estimators": 150,
    "max_depth": 2,
    "learning_rate": 0.02,
    "subsample": 0.7,
    "colsample_bytree": 0.6,
    "min_child_weight": 200,
    "reg_lambda": 20.0,
    "max_bin": 64,
    "tree_method": "hist",
    "n_jobs": 1,
})

NEWS_DECAY_HALFLIFE_SESSIONS: Final = 5.0
NEWS_EXPIRY_SESSIONS: Final = 21
NEWS_START_DATE: Final = date(2019, 7, 9)
EQUAL_RANK_MIN_PRESENT: Final = 12

SECTION_5_6_ACCEPTANCE_RULE: Final = (
    "Base model acceptance (§5.6, fixed before fitting): (1) 21-session later-date mean rank correlation >= 0.03 "
    "with Newey-West t >= 3.0; (2) positive in >= 70% of scored calendar years; (3) unseen-issuer 21-session rank "
    "correlation >= 50% of the later-date figure; (4) buffered skill basket (buy top 10% / hold top 33%) positive "
    "after 20 bps turnover cost; (5) beats the zero-fit equal-weighted average of the 24 ranked inputs (or is that "
    "baseline when no fitted model beats it and 1-4 hold). Stage-2 news is accepted only when paired daily rank-"
    "correlation improvement achieves Newey-West t >= 2.0 and is positive in most scored years."
)

TIER1_FAMILIES: Final = tuple(CATEGORY_CODES.keys())

NEWS_FEATURE_NAMES: Final = (
    *(f"news_rx_{fam}" for fam in TIER1_FAMILIES),
    "news_rx_offering",
    "news_rx_any",
    "news_abn_vol_20",
    "news_recency_21",
    "news_unusual_count_5_60",
)


@dataclass(frozen=True)
class Stage1FeatureBundle:
    """48 centered-rank base features (24 cross-sectional + 24 sector-neutral) and `equal_rank_24`."""

    features: dict[str, Float32Array]
    feature_names: tuple[str, ...]
    equal_rank_24: FloatArray
    market_value: FloatArray
    coverage_report: dict[str, Any] = field(default_factory=dict)


def _sector_centered_rank(
    values: FloatArray, eligible: BoolArray, sector: npt.NDArray[np.int16], minimum: int = SECTOR_MINIMUM_STOCKS,
) -> Float32Array:
    out = np.full(values.shape, np.nan, dtype=np.float32)
    valid_base = eligible & np.isfinite(values) & (sector >= 0)
    for code in np.unique(sector[valid_base]):
        in_sec = sector == code
        cols = np.flatnonzero(in_sec.any(axis=0))
        if cols.size < minimum:
            continue
        sub_mask = eligible[:, cols] & in_sec[:, cols]
        sub_vals = values[:, cols]
        ranked = centered_rank(sub_vals, sub_mask)
        enough = ((sub_mask & np.isfinite(sub_vals)).sum(axis=1) >= minimum)[:, None]
        cur = out[:, cols]
        out[:, cols] = np.where(sub_mask & enough, ranked.astype(np.float32), cur)
    return out


def build_stage1_features(
    panel: ResearchPanel,
    fundamental_inputs: fundamentals.FundamentalInputs,
) -> Stage1FeatureBundle:
    """Build §5.3's 24 signed base predictors (cross-sectional + sector-neutral ranks) and market value."""
    shape = panel.eligible.shape
    present = np.zeros(shape, dtype=np.int32)
    total = np.zeros(shape, dtype=np.float64)
    feat_map: dict[str, Float32Array] = {}
    ordered_cs: list[str] = []
    ordered_sn: list[str] = []
    elig_total = max(1, int(panel.eligible.sum()))
    cov: dict[str, float] = {}

    def _consume(key: str, arr: FloatArray) -> None:
        nonlocal present, total
        cov[key] = float((panel.eligible & np.isfinite(arr)).sum() / elig_total)
        cs = centered_rank(arr, panel.eligible)
        fin = np.isfinite(cs)
        present += fin.astype(np.int32)
        total += np.where(fin, cs, 0.0)
        feat_map[key] = cs.astype(np.float32)
        ordered_cs.append(key)
        del cs

        sn_key = f"{key}_sector"
        feat_map[sn_key] = _sector_centered_rank(arr, panel.eligible, panel.sector)
        ordered_sn.append(sn_key)

    # Rank and free the 12 price predictors BEFORE computing the 12 fundamental predictors to halve peak RAM.
    price_raw = predictors.compute_predictors(panel)
    for name in predictors.PREDICTOR_NAMES:
        raw_p = predictors.EXPECTED_SIGNS[name] * price_raw.pop(name)
        _consume(f"px_{name}", raw_p)
        del raw_p
    del price_raw
    gc.collect()

    fund_res = fundamentals.compute_fundamental_predictors(fundamental_inputs)
    mv = fund_res.market_value
    fund_report = fund_res.report
    fund_res.reasons.clear()
    for name in fundamentals.PREDICTOR_NAMES:
        raw_f = fundamentals.EXPECTED_SIGNS[name] * fund_res.values.pop(name)
        _consume(f"fn_{name}", raw_f)
        del raw_f
    del fund_res
    gc.collect()

    with np.errstate(invalid="ignore", divide="ignore"):
        eq_mean = total / np.maximum(present, 1)
    equal_rank_24 = np.where(panel.eligible & (present >= EQUAL_RANK_MIN_PRESENT), eq_mean, np.nan)
    del present, total, eq_mean
    cov["equal_rank_24"] = float(np.isfinite(equal_rank_24).sum() / elig_total)

    return Stage1FeatureBundle(
        features=feat_map,
        feature_names=(*ordered_cs, *ordered_sn),
        equal_rank_24=equal_rank_24,
        market_value=mv,
        coverage_report={"feature_coverage_on_eligible_cells": cov, "fundamental_report": fund_report},
    )


def _dynamic_folds(sessions: npt.NDArray[np.datetime64], first_row: int, min_train_sessions: int = 120) -> tuple[Fold, ...]:
    """Chronological calendar-year test folds that have at least `min_train_sessions` prior sessions."""
    years = session_years(sessions)
    unique_years = sorted(int(y) for y in np.unique(years[first_row:]))
    out: list[Fold] = []
    for yr in unique_years:
        idx = np.flatnonzero(years == yr)
        if idx.size == 0:
            continue
        t_start = int(idx[0])
        if t_start - first_row >= min_train_sessions:
            out.append(Fold(year=yr, test_start=t_start, test_stop=int(idx[-1]) + 1))
    return tuple(out)


def _design_matrix(
    features: Mapping[str, Float32Array], names: tuple[str, ...], rows: IntArray, mask: BoolArray,
) -> Float32Array:
    out = np.empty((int(mask.sum()), len(names)), dtype=np.float32)
    for idx, name in enumerate(names):
        out[:, idx] = features[name][rows][mask]
    return out


def _fit_predict_tabular(
    learner: str,
    x_train: Float32Array,
    y_train: FloatArray,
    w_train: FloatArray,
    x_test: Float32Array,
    *,
    xgb_params: Mapping[str, Any] = STAGE1_XGB_PARAMS,
    seeds: tuple[int, ...] = STAGE1_SEEDS,
) -> tuple[FloatArray, FloatArray]:
    """Fit `ridge` or multi-seed `xgboost` and return `(test_predictions, feature_importance)`."""
    n_feat = x_train.shape[1]
    if x_train.shape[0] < 20 or x_test.shape[0] == 0:
        return np.full(x_test.shape[0], np.nan, dtype=np.float64), np.zeros(n_feat, dtype=np.float64)

    if learner == "ridge":
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True, solver="cholesky")
        model.fit(np.nan_to_num(x_train, nan=0.0).astype(np.float64), y_train, sample_weight=w_train)
        preds = np.asarray(model.predict(np.nan_to_num(x_test, nan=0.0).astype(np.float64)), dtype=np.float64)
        imp = np.abs(np.asarray(model.coef_, dtype=np.float64)).ravel()
        return preds, imp

    if learner == "xgboost":
        pred_acc = np.zeros(x_test.shape[0], dtype=np.float64)
        imp_acc = np.zeros(n_feat, dtype=np.float64)
        for seed in seeds:
            est = XGBRegressor(**xgb_params, random_state=seed)
            est.fit(x_train, y_train, sample_weight=w_train)
            pred_acc += np.asarray(est.predict(x_test), dtype=np.float64)
            imp_acc += np.asarray(est.feature_importances_, dtype=np.float64)
            del est
        gc.collect()
        return pred_acc / float(len(seeds)), imp_acc / float(len(seeds))

    raise ValueError(f"unsupported learner: {learner}")


def _issuer_fold_ids(
    security_ids: tuple[str, ...], security_cik_map: Mapping[str, str], n_folds: int = UNSEEN_ISSUER_FOLDS,
) -> IntArray:
    """Deterministic issuer-grouped fold index `0..n_folds-1` (keeps dual-class shares together)."""
    buckets = np.empty(len(security_ids), dtype=np.int64)
    for i, sec_id in enumerate(security_ids):
        key = security_cik_map.get(sec_id, sec_id)
        digest = int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=4).digest(), "little")
        buckets[i] = digest % n_folds
    return buckets


def run_stage1_walk_forward(
    panel: ResearchPanel,
    bundle: Stage1FeatureBundle,
    security_cik_map: Mapping[str, str],
    *,
    first_row: int,
    horizons: tuple[int, ...] = HORIZONS,
    checkpoint: Callable[[str], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run Stage-1 (`equal_rank_24`, `ridge_rank`, `xgb_shallow_5seed`) + unseen-issuer CV."""
    folds = _dynamic_folds(panel.sessions, first_row)
    if not folds:
        raise ValueError("insufficient session span to form walk-forward folds")

    issuer_buckets = _issuer_fold_ids(panel.security_ids, security_cik_map)
    by_horizon: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        labels = research_labels(panel, horizon)
        target = target_ranks(labels.stock, panel.eligible)
        shape = panel.eligible.shape

        # Also generate 2019 H2 out-of-sample Stage-1 predictions (trained on 2018..mid-2019) for Stage-2 residuals!
        news_start_row = int(np.searchsorted(panel.sessions, np.datetime64(NEWS_START_DATE, "D"), side="left"))
        first_fold_start = folds[0].test_start
        pre_fold_slice = (
            Fold(year=0, test_start=news_start_row, test_stop=first_fold_start)
            if news_start_row < first_fold_start and (news_start_row - first_row) >= 60
            else None
        )

        # 1. Zero-fit baseline `equal_rank_24` restricted to scored OOS window `>= folds[0].test_start` for fair comparison,
        # plus full `>= news_start_row` copy for Stage-2 residuals.
        eq_oos = np.full(shape, np.nan, dtype=np.float64)
        eq_oos[first_fold_start:] = bundle.equal_rank_24[first_fold_start:]
        eq_for_stage2 = np.full(shape, np.nan, dtype=np.float64)
        eq_for_stage2[min(news_start_row, first_fold_start):] = bundle.equal_rank_24[min(news_start_row, first_fold_start):]

        learner_outputs: dict[str, dict[str, Any]] = {
            "equal_rank_24": {
                "oos_scores": eq_oos,
                "stage2_base_scores": eq_for_stage2,
                "unseen_oos_scores": eq_oos if horizon in UNSEEN_CV_HORIZONS else None,
                "importance": np.full(len(bundle.feature_names), 1.0 / len(bundle.feature_names), dtype=np.float64),
            }
        }

        for spec_name, learner_kind in (("ridge_rank", "ridge"), ("xgb_shallow_5seed", "xgboost")):
            oos_scores = np.full(shape, np.nan, dtype=np.float64)
            stage2_base = np.full(shape, np.nan, dtype=np.float64)
            imps: list[FloatArray] = []

            if pre_fold_slice is not None:
                tr_rows = training_rows(first_row, pre_fold_slice.test_start, horizon)
                if tr_rows.size >= 10:
                    tr_cells = panel.eligible[tr_rows] & np.isfinite(target[tr_rows])
                    te_rows = np.arange(pre_fold_slice.test_start, pre_fold_slice.test_stop, dtype=np.int64)
                    te_cells = panel.eligible[te_rows]
                    preds_pre, _ = _fit_predict_tabular(
                        learner_kind,
                        _design_matrix(bundle.features, bundle.feature_names, tr_rows, tr_cells),
                        target[tr_rows][tr_cells].astype(np.float64),
                        date_weights(tr_cells),
                        _design_matrix(bundle.features, bundle.feature_names, te_rows, te_cells),
                    )
                    blk = np.full(te_cells.shape, np.nan, dtype=np.float64)
                    blk[te_cells] = preds_pre
                    stage2_base[te_rows] = blk

            for fold in folds:
                tr_rows = training_rows(first_row, fold.test_start, horizon)
                tr_cells = panel.eligible[tr_rows] & np.isfinite(target[tr_rows])
                te_rows = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
                te_cells = panel.eligible[te_rows]
                preds, imp = _fit_predict_tabular(
                    learner_kind,
                    _design_matrix(bundle.features, bundle.feature_names, tr_rows, tr_cells),
                    target[tr_rows][tr_cells].astype(np.float64),
                    date_weights(tr_cells),
                    _design_matrix(bundle.features, bundle.feature_names, te_rows, te_cells),
                )
                blk = np.full(te_cells.shape, np.nan, dtype=np.float64)
                blk[te_cells] = preds
                oos_scores[te_rows] = blk
                stage2_base[te_rows] = blk
                imps.append(imp)
                if checkpoint is not None:
                    checkpoint(f"stage1 {spec_name} h={horizon} fold={fold.year}")

            unseen_scores: FloatArray | None = None
            if horizon in UNSEEN_CV_HORIZONS:
                unseen_scores = np.full(shape, np.nan, dtype=np.float64)
                for fold in folds:
                    tr_rows = training_rows(first_row, fold.test_start, horizon)
                    te_rows = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
                    for b_id in range(UNSEEN_ISSUER_FOLDS):
                        in_bucket = (issuer_buckets == b_id)[None, :]
                        tr_cells = panel.eligible[tr_rows] & np.isfinite(target[tr_rows]) & ~in_bucket
                        te_cells = panel.eligible[te_rows] & in_bucket
                        if not bool(te_cells.any()) or int(tr_cells.sum()) < 50:
                            continue
                        u_preds, _ = _fit_predict_tabular(
                            learner_kind,
                            _design_matrix(bundle.features, bundle.feature_names, tr_rows, tr_cells),
                            target[tr_rows][tr_cells].astype(np.float64),
                            date_weights(tr_cells),
                            _design_matrix(bundle.features, bundle.feature_names, te_rows, te_cells),
                            seeds=UNSEEN_XGB_SEEDS,
                        )
                        sub_blk = unseen_scores[te_rows]
                        sub_blk[te_cells] = u_preds
                        unseen_scores[te_rows] = sub_blk
                    if checkpoint is not None:
                        checkpoint(f"stage1 unseen {spec_name} h={horizon} fold={fold.year}")

            learner_outputs[spec_name] = {
                "oos_scores": oos_scores,
                "stage2_base_scores": stage2_base,
                "unseen_oos_scores": unseen_scores,
                "importance": np.mean(imps, axis=0) if imps else np.zeros(len(bundle.feature_names), dtype=np.float64),
            }

        by_horizon[horizon] = {
            "labels": labels,
            "target": target,
            "folds": folds,
            "scored_start": first_fold_start,
            "models": learner_outputs,
        }

    return by_horizon


def load_news_reaction_features(
    panel: ResearchPanel,
    *,
    panel_root: Path,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    unseal_final_test_period: bool = False,
) -> tuple[dict[str, Float32Array], dict[str, str], dict[str, Any]]:
    """Build §5.3's continuous News-Reaction features (`<= development_end` strictly)."""
    check_development_end(panel.development_end, unseal_final_test_period=unseal_final_test_period)
    cutoff_utc = pd.Timestamp(panel.development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)
    cutoff_utc = cutoff_utc.tz_convert("UTC")

    pit_map = _pit_session_ticker_map(panel_root, panel.sessions, panel.development_end)
    col_of = {sec_id: idx for idx, sec_id in enumerate(panel.security_ids)}
    cik_counts = Counter(sec_id.split(":")[1] for sec_id in panel.security_ids if sec_id.startswith("cik:"))
    unique_cik_col = {
        f"cik:{sec_id.split(':')[1]}": idx
        for idx, sec_id in enumerate(panel.security_ids)
        if sec_id.startswith("cik:") and sec_id.count(":") == 1 and cik_counts[sec_id.split(":")[1]] == 1
    }

    hashes: dict[str, str] = {}
    frames: list[pd.DataFrame] = []
    for label, d_path in (("early", early_news_dir), ("later", later_news_dir)):
        p_file = d_path / "family_events.parquet"
        m_file = d_path / "_manifest.json"
        if not p_file.is_file():
            continue
        hashes[f"{label}_news_family_events_sha256"] = file_sha256(p_file)
        if m_file.is_file():
            hashes[f"{label}_news_family_manifest_sha256"] = file_sha256(m_file)
        tbl = cast(Any, pq).read_table(
            p_file,
            columns=[
                "security_id", "ticker", "event_family", "feature_available_at_utc",
                "relation_channel", "research_eligible",
            ],
            partitioning=None,
            filters=[("feature_available_at_utc", "<", cutoff_utc.to_pydatetime())],
            use_threads=False,
        )
        df = tbl.to_pandas()
        del tbl
        if len(df):
            if bool((pd.to_datetime(df["feature_available_at_utc"], utc=True) >= cutoff_utc).any()):
                raise SealedPeriodError(f"sealed news event reached memory from {p_file}")
            df = df.loc[df["research_eligible"].fillna(False).astype(bool) & df["relation_channel"].eq("direct_issuer")]
            frames.append(df)

    n_sess, n_sec = len(panel.sessions), len(panel.security_ids)
    shape = (n_sess, n_sec)
    raw_feats: dict[str, FloatArray] = {name: np.full(shape, np.nan, dtype=np.float64) for name in NEWS_FEATURE_NAMES}
    count_matrix = np.zeros(shape, dtype=np.float64)

    mapped_events = 0
    if frames:
        events = pd.concat(frames, ignore_index=True)
        ev_rows = event_rows(panel.sessions, pd.to_datetime(events["feature_available_at_utc"], utc=True))
        inside = (ev_rows >= 1) & (ev_rows < n_sess)
        events = events.loc[inside].reset_index(drop=True)
        ev_rows = ev_rows[inside]

        cols_mapped = np.full(len(events), -1, dtype=np.int64)
        for idx, (s_id, tk, r_idx) in enumerate(
            zip(events["security_id"].astype(str), events["ticker"].astype(str), ev_rows, strict=False)
        ):
            pit_hit = pit_map.get((int(r_idx), tk.upper()))
            if pit_hit is not None and pit_hit in col_of:
                cols_mapped[idx] = col_of[pit_hit]
            elif s_id in col_of:
                cols_mapped[idx] = col_of[s_id]
            elif s_id.startswith("cik:"):
                prefix = f"cik:{s_id.split(':')[1]}"
                if prefix in unique_cik_col:
                    cols_mapped[idx] = unique_cik_col[prefix]

        keep_m = cols_mapped >= 0
        events = events.loc[keep_m].reset_index(drop=True)
        ev_rows = ev_rows[keep_m]
        cols_mapped = cols_mapped[keep_m]
        mapped_events = len(events)

        closes = panel.close.astype(np.float64)
        vols = panel.volume.astype(np.float64)
        spy_c = panel.spy_close.astype(np.float64)

        # Precompute 20-day lagged mean volume (`mean(vol[t-20:t])`).
        vol_clean = np.where(np.isfinite(vols) & (vols >= 0.0), vols, 0.0)
        vol_fin = np.isfinite(vols) & (vols >= 0.0)
        vol_csum = np.vstack([np.zeros((1, n_sec)), np.cumsum(vol_clean, axis=0)])
        cnt_csum = np.vstack([np.zeros((1, n_sec), dtype=np.int32), np.cumsum(vol_fin.astype(np.int32), axis=0)])

        ev_df = pd.DataFrame({
            "row": ev_rows,
            "col": cols_mapped,
            "family": events["event_family"].astype(str).to_numpy(),
        })
        for r_i, c_i in zip(ev_rows, cols_mapped, strict=False):
            count_matrix[int(r_i), int(c_i)] += 1.0

        decay_lut = np.power(0.5, np.arange(NEWS_EXPIRY_SESSIONS + 1, dtype=np.float64) / NEWS_DECAY_HALFLIFE_SESSIONS)
        tracked_families = (*TIER1_FAMILIES, "offering")

        for col_id, grp in ev_df.groupby("col"):
            c = int(col_id)
            by_row_fam = grp.drop_duplicates(["row", "family"])
            # Overall reaction per distinct event session on column `c`
            distinct_r = np.sort(by_row_fam["row"].unique().astype(np.int64))
            prev_r = distinct_r - 1
            px_ok = (
                np.isfinite(closes[distinct_r, c]) & (closes[distinct_r, c] > 0.0)
                & np.isfinite(closes[prev_r, c]) & (closes[prev_r, c] > 0.0)
                & np.isfinite(spy_c[distinct_r]) & (spy_c[distinct_r] > 0.0)
                & np.isfinite(spy_c[prev_r]) & (spy_c[prev_r] > 0.0)
            )
            rx_vals = np.full(distinct_r.shape, np.nan, dtype=np.float64)
            rx_vals[px_ok] = (
                (closes[distinct_r[px_ok], c] / closes[prev_r[px_ok], c] - 1.0)
                - (spy_c[distinct_r[px_ok]] / spy_c[prev_r[px_ok]] - 1.0)
            )

            v_start = np.maximum(0, distinct_r - 20)
            v_cnt = cnt_csum[distinct_r, c] - cnt_csum[v_start, c]
            v_sum = vol_csum[distinct_r, c] - vol_csum[v_start, c]
            abn_vals = np.full(distinct_r.shape, np.nan, dtype=np.float64)
            v_ok = (v_cnt >= 10) & (v_sum > 0.0) & np.isfinite(vols[distinct_r, c])
            abn_vals[v_ok] = vols[distinct_r[v_ok], c] / (v_sum[v_ok] / v_cnt[v_ok])

            sess_idx = np.arange(n_sess, dtype=np.int64)
            pos = np.searchsorted(distinct_r, sess_idx, side="right") - 1
            valid_pos = pos >= 0
            picked = np.where(valid_pos, pos, 0)
            age = sess_idx - distinct_r[picked]
            active = valid_pos & (age <= NEWS_EXPIRY_SESSIONS)
            age_clipped = np.clip(age, 0, NEWS_EXPIRY_SESSIONS)
            w_decay = decay_lut[age_clipped]

            raw_feats["news_rx_any"][active, c] = rx_vals[picked[active]] * w_decay[active]
            raw_feats["news_abn_vol_20"][active, c] = np.log1p(np.maximum(0.0, abn_vals[picked[active]])) * w_decay[active]
            raw_feats["news_recency_21"][active, c] = (NEWS_EXPIRY_SESSIONS + 1 - age[active]) / float(
                NEWS_EXPIRY_SESSIONS + 1
            )

            rx_by_row = dict(zip(distinct_r.tolist(), rx_vals.tolist(), strict=False))
            for fam in tracked_families:
                f_rows = np.sort(by_row_fam.loc[by_row_fam["family"] == fam, "row"].to_numpy(dtype=np.int64))
                if f_rows.size == 0:
                    continue
                f_rx = np.asarray([rx_by_row.get(int(r), np.nan) for r in f_rows], dtype=np.float64)
                f_pos = np.searchsorted(f_rows, sess_idx, side="right") - 1
                f_valid = f_pos >= 0
                f_pick = np.where(f_valid, f_pos, 0)
                f_age = sess_idx - f_rows[f_pick]
                f_act = f_valid & (f_age <= NEWS_EXPIRY_SESSIONS)
                f_dec = decay_lut[np.clip(f_age, 0, NEWS_EXPIRY_SESSIONS)]
                raw_feats[f"news_rx_{fam}"][f_act, c] = f_rx[f_pick[f_act]] * f_dec[f_act]

    # Unusual 5d article count vs 60d trailing baseline (strictly causal `<= t`).
    c_csum = np.vstack([np.zeros((1, n_sec)), np.cumsum(count_matrix, axis=0)])
    rows_arange = np.arange(1, n_sess + 1)
    c5 = c_csum[rows_arange] - c_csum[np.maximum(0, rows_arange - 5)]
    c60 = c_csum[rows_arange] - c_csum[np.maximum(0, rows_arange - 60)]
    news_start_row = int(np.searchsorted(panel.sessions, np.datetime64(NEWS_START_DATE, "D"), side="left"))
    warm_news = np.arange(n_sess)[:, None] >= (news_start_row + 20)
    raw_feats["news_unusual_count_5_60"] = np.where(
        warm_news & panel.eligible, c5 / np.maximum(c60 / 12.0, 0.25), np.nan,
    )

    # Fill absent-event stocks with 0.0 (zero news shock / zero abnormal event volume / zero recency) BEFORE
    # cross-sectional ranking so single-event sessions and same-sign event clusters rank against the neutral
    # zero-news cross-section instead of collapsing or inverting against each other.
    ranked_news: dict[str, Float32Array] = {}
    active_news_era = (np.arange(n_sess)[:, None] >= news_start_row) & panel.eligible
    for name in NEWS_FEATURE_NAMES:
        if name == "news_unusual_count_5_60":
            base_arr = np.where(active_news_era, raw_feats[name], np.nan)
        else:
            base_arr = np.where(active_news_era, np.nan_to_num(raw_feats[name], nan=0.0), np.nan)
        rk = centered_rank(base_arr, active_news_era)
        ranked_news[name] = rk.astype(np.float32)

    return ranked_news, hashes, {"mapped_direct_issuer_events": mapped_events, "features": list(NEWS_FEATURE_NAMES)}


def run_stage2_news_residuals(
    panel: ResearchPanel,
    stage1_by_horizon: Mapping[int, Mapping[str, Any]],
    news_features: Mapping[str, Float32Array],
    *,
    horizons: tuple[int, ...] = HORIZONS,
    feature_names: tuple[str, ...] = NEWS_FEATURE_NAMES,
    checkpoint: Callable[[str], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Fit Stage-2 residual models (`ridge`, `xgboost`) on `target_rank - stage1_oos_pred` for `2021..2024`."""
    news_start_row = int(np.searchsorted(panel.sessions, np.datetime64(NEWS_START_DATE, "D"), side="left"))
    news_folds = _dynamic_folds(panel.sessions, news_start_row, min_train_sessions=120)
    out_by_horizon: dict[int, dict[str, Any]] = {}
    if not news_folds:
        return out_by_horizon

    n_names = feature_names
    news_scored_start = news_folds[0].test_start
    decision_rows_news = np.arange(len(panel.sessions)) >= news_scored_start

    for horizon in horizons:
        s1_h = stage1_by_horizon[horizon]
        labels = s1_h["labels"]
        target = s1_h["target"]
        pairs: dict[str, dict[str, Any]] = {}

        for base_name, base_entry in s1_h["models"].items():
            base_s2_scores: FloatArray = base_entry["stage2_base_scores"]
            residual = target - base_s2_scores

            for s2_learner in ("ridge", "xgboost"):
                comb_scores = np.full(panel.eligible.shape, np.nan, dtype=np.float64)
                for fold in news_folds:
                    tr_rows = training_rows(news_start_row, fold.test_start, horizon)
                    tr_cells = panel.eligible[tr_rows] & np.isfinite(residual[tr_rows])
                    te_rows = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
                    te_cells = panel.eligible[te_rows] & np.isfinite(base_s2_scores[te_rows])
                    if int(tr_cells.sum()) < 50 or not bool(te_cells.any()):
                        continue
                    r_pred, _ = _fit_predict_tabular(
                        s2_learner,
                        _design_matrix(news_features, n_names, tr_rows, tr_cells),
                        residual[tr_rows][tr_cells].astype(np.float64),
                        date_weights(tr_cells),
                        _design_matrix(news_features, n_names, te_rows, te_cells),
                        xgb_params=STAGE2_XGB_PARAMS,
                        seeds=STAGE2_XGB_SEEDS,
                    )
                    blk = np.full(te_cells.shape, np.nan, dtype=np.float64)
                    blk[te_cells] = base_s2_scores[te_rows][te_cells] + r_pred
                    comb_scores[te_rows] = blk

                # Paired daily rank-correlation improvement over `base_name` on identical `news_scored_start:` dates.
                cells_pair = scored_cells(comb_scores, labels.stock, panel.eligible, decision_rows_news) & np.isfinite(
                    base_s2_scores
                )
                ic_base = daily_rank_correlation(base_s2_scores, labels.stock, cells_pair)
                ic_comb = daily_rank_correlation(comb_scores, labels.stock, cells_pair)
                delta_ic = ic_comb - ic_base
                nw_delta = newey_west(delta_ic, horizon)
                yrs = session_years(panel.sessions)
                pos_yrs = 0
                tot_yrs = 0
                by_yr_delta: dict[str, float] = {}
                for yr in np.unique(yrs[np.isfinite(delta_ic)]):
                    vals = delta_ic[np.isfinite(delta_ic) & (yrs == yr)]
                    if vals.size:
                        tot_yrs += 1
                        m_val = float(vals.mean())
                        by_yr_delta[str(int(yr))] = m_val
                        if m_val > 0.0:
                            pos_yrs += 1

                t_val = nw_delta["t"]
                accepted_news = bool(
                    t_val is not None and float(t_val) >= 2.0 and tot_yrs > 0 and pos_yrs > tot_yrs / 2.0
                )
                pairs[f"{base_name}+news_{s2_learner}"] = {
                    "base_model": base_name,
                    "stage2_learner": s2_learner,
                    "combined_scores": comb_scores.astype(np.float32),
                    "news_scored_start": news_scored_start,
                    "paired_ic_improvement": {
                        **nw_delta,
                        "by_year": by_yr_delta,
                        "positive_years": pos_yrs,
                        "years_scored": tot_yrs,
                        "passes_paired_news_gate": accepted_news,
                    },
                }
                del comb_scores
                gc.collect()
                if checkpoint is not None:
                    checkpoint(f"stage2 {base_name}+{s2_learner} h={horizon}")
            del residual
            gc.collect()

        out_by_horizon[horizon] = {"folds": news_folds, "specifications": pairs}

    return out_by_horizon


def evaluate_two_stage_scoreboard(
    panel: ResearchPanel,
    bundle: Stage1FeatureBundle,
    stage1_by_horizon: Mapping[int, Mapping[str, Any]],
    stage2_by_horizon: Mapping[int, Mapping[str, Any]],
    *,
    horizons: tuple[int, ...] = HORIZONS,
) -> dict[str, Any]:
    """Build the unified §5.5 & §5.6 scoreboard across Stage-1 and Stage-2 models."""
    years = session_years(panel.sessions)
    stage1_table: dict[str, dict[str, Any]] = {}
    stage2_table: dict[str, dict[str, Any]] = {}

    for horizon in horizons:
        h_key = str(horizon)
        s1_h = stage1_by_horizon[horizon]
        labels = s1_h["labels"]
        scored_start = int(s1_h["scored_start"])
        dec_mask = np.arange(len(panel.sessions)) >= scored_start

        # Continuous rebalance evaluation (`skill_top10_hold33` & `product_sector_active_slice`) on OOS window.
        windows = benchmark_tilt.continuous_rebalance_windows(panel, scored_start, horizon)
        sec_at_dec = panel.sector[windows.decision_rows]
        _, cap_bench_dec = benchmark_tilt.build_cap_tilt_weights(
            panel, bundle.equal_rank_24, bundle.market_value, windows.decision_rows, mode="sector_active_slice",
        )
        cap_bench_w = benchmark_tilt._apply_open_fills(cap_bench_dec, windows.tradable)
        cap_bench_path = benchmark_tilt.simulate_portfolio_path(cap_bench_w, windows.stock_returns)

        for model_name, m_entry in s1_h["models"].items():
            scores: FloatArray = m_entry["oos_scores"]
            board = score_predictor(
                scores, labels, eligible=panel.eligible, decision_rows=dec_mask,
                sector=panel.sector, years=years, first_row=scored_start,
            )

            unseen_ic: dict[str, Any] | None = None
            if m_entry["unseen_oos_scores"] is not None:
                u_scores: FloatArray = m_entry["unseen_oos_scores"]
                u_cells = scored_cells(u_scores, labels.stock, panel.eligible, dec_mask)
                u_corr = daily_rank_correlation(u_scores, labels.stock, u_cells)
                unseen_ic = newey_west(u_corr, horizon)

            # §5.5 Buffered Skill (`buy top 10% / hold top 33%`) & Cap-Tilt Product
            w_s1_dec, b_s1_dec = benchmark_tilt.build_skill_weights(
                panel, scores, windows.decision_rows, buffered=True,
            )
            w_s1 = benchmark_tilt._apply_open_fills(w_s1_dec, windows.tradable)
            b_s1 = benchmark_tilt._apply_open_fills(b_s1_dec, windows.tradable)
            p_s1 = benchmark_tilt.simulate_portfolio_path(w_s1, windows.stock_returns, b_s1)
            b_s1_gross = benchmark_tilt.simulate_portfolio_path(b_s1, windows.stock_returns).gross_returns
            skill_buffered = benchmark_tilt._pack_policy_summary(
                p_s1, b_s1_gross, "equal_weight_scored_stocks", windows, horizon,
            )

            w_prod_dec, b_prod_dec = benchmark_tilt.build_cap_tilt_weights(
                panel, scores, bundle.market_value, windows.decision_rows, mode="sector_active_slice",
            )
            w_prod = benchmark_tilt._apply_open_fills(w_prod_dec, windows.tradable)
            b_prod = benchmark_tilt._apply_open_fills(b_prod_dec, windows.tradable)
            p_prod = benchmark_tilt.simulate_portfolio_path(w_prod, windows.stock_returns, b_prod, sec_at_dec)
            product_tilt = benchmark_tilt._pack_policy_summary(
                p_prod, cap_bench_path.gross_returns, "own_cap_weighted_constituents", windows, horizon,
            )

            top_imp_idx = np.argsort(-m_entry["importance"], kind="stable")[:12]
            stage1_table.setdefault(model_name, {})[h_key] = {
                "score": board,
                "unseen_issuer_rank_correlation": unseen_ic,
                "skill_buffered_hold33": skill_buffered,
                "product_sector_active_slice": product_tilt,
                "top_features": [
                    {"feature": bundle.feature_names[int(i)], "importance": float(m_entry["importance"][int(i)])}
                    for i in top_imp_idx
                ],
            }

        if horizon in stage2_by_horizon:
            for spec_key, s2_entry in stage2_by_horizon[horizon]["specifications"].items():
                n_start = int(s2_entry["news_scored_start"])
                n_mask = np.arange(len(panel.sessions)) >= n_start
                c_scores = np.asarray(s2_entry["combined_scores"], dtype=np.float64)
                c_board = score_predictor(
                    c_scores, labels, eligible=panel.eligible, decision_rows=n_mask,
                    sector=panel.sector, years=years, first_row=n_start,
                )
                del c_scores
                stage2_table.setdefault(spec_key, {})[h_key] = {
                    "base_model": s2_entry["base_model"],
                    "stage2_learner": s2_entry["stage2_learner"],
                    "paired_ic_improvement": s2_entry["paired_ic_improvement"],
                    "score": c_board,
                }

    # Evaluate §5.6 5-point acceptance rule at PRIMARY_HORIZON (21 sessions) if present.
    acceptance: dict[str, Any] = {"rule": SECTION_5_6_ACCEPTANCE_RULE, "candidates": {}, "passing_models": []}
    p_key = str(PRIMARY_HORIZON)
    if p_key in stage1_table.get("equal_rank_24", {}):
        eq_ic_21 = stage1_table["equal_rank_24"][p_key]["score"]["rank_correlation"]["mean"]
        for model_name, horizons_map in stage1_table.items():
            entry_21 = horizons_map[p_key]
            rc = entry_21["score"]["rank_correlation"]
            mean_ic, nw_t = rc["mean"], rc["t"]
            pos_share = entry_21["score"]["share_of_years_positive"]
            u_rc = entry_21["unseen_issuer_rank_correlation"]
            u_mean = None if u_rc is None else u_rc["mean"]
            sk_net = entry_21["skill_buffered_hold33"]["vs_matched_benchmark"]["net_excess_20bps_per_period"]

            c1 = mean_ic is not None and float(mean_ic) >= 0.03 and nw_t is not None and float(nw_t) >= 3.0
            c2 = pos_share is not None and float(pos_share) >= 0.70
            c3 = (
                mean_ic is not None
                and u_mean is not None
                and float(u_mean) >= 0.5 * float(mean_ic)
            )
            c4 = sk_net is not None and float(sk_net) > 0.0
            c5 = (
                True
                if model_name == "equal_rank_24"
                else (mean_ic is not None and eq_ic_21 is not None and float(mean_ic) > float(eq_ic_21))
            )
            passes = bool(c1 and c2 and c3 and c4 and c5)
            acceptance["candidates"][model_name] = {
                "mean_ic_21d": mean_ic,
                "newey_west_t_21d": nw_t,
                "share_of_years_positive_21d": pos_share,
                "unseen_issuer_ic_21d": u_mean,
                "buffered_skill_net_20bps_per_21d": sk_net,
                "beats_or_is_equal_rank_24": c5,
                "passes_section_5_6": passes,
            }
            if passes:
                acceptance["passing_models"].append(model_name)

    accepted_stage2 = [
        f"{spec_key}@h={h_k}"
        for spec_key, h_map in stage2_table.items()
        for h_k, item in h_map.items()
        if item["paired_ic_improvement"]["passes_paired_news_gate"]
    ]
    acceptance["accepted_stage2_news_overlays"] = accepted_stage2
    acceptance["earns_sealed_period_test"] = bool(acceptance["passing_models"])

    return {
        "stage1_models": stage1_table,
        "stage2_news_models": stage2_table,
        "acceptance_summary": acceptance,
    }
