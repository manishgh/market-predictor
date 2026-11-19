"""Step 5c (Option 1): Beta/Volatility/Sector-Neutralized Stage-1 & High-Precision $0 SEC Catalyst Stage-2.

Addresses the two empirical bottlenecks diagnosed in Step 5b on `mid_small_ex_sp500`:
1. Stage-1 unhedged negative loading on `beta_252d` and `idio_vol_60d` (which caused `-0.0714` IC during
   2020's post-lockdown high-beta rally, inflating IC standard error to `0.0211` and dragging top-10%
   bull-market returns by ~3%/yr) via closed-form per-session cross-sectional OLS orthogonalization
   against `[1, sector_dummies, centered_rank(beta_252d), centered_rank(idio_vol_60d)]` at session `t`
   close (zero lookahead, zero tuned parameters), plus purged residual-target training for Ridge and
   5-seed shallow XGBoost.
2. Stage-2 catalyst signal dilution (where Step 5b's all-8K dump included routine `Item 5.07` annual-meeting
   votes and `Item 8.01` boilerplate, while Alpaca `/v1beta1/news` purged historical 2019-2024 Benzinga
   articles upon migrating to `fiscal.ai`) by collecting free SEC EDGAR `SC 13D/A` activist stakes,
   `S-3`/`424B` dilutive equity offerings, and `NT 10-K/Q` late-filing notices alongside filtered material
   `8-K` items (`2.02`, `1.01`/`2.01`, non-earnings `7.01`, non-5.07 `5.02`, and `1.03`/`2.04`/`2.06`/
   `3.01`/`4.01`/`4.02` distress) and volume-confirmed signed reaction (`rx * log1p(abn_vol_20)`).
"""
from __future__ import annotations

import gc
import json
from collections.abc import Callable, Mapping
from datetime import date
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.config import Settings, get_settings
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.process_memory import release_unused_process_memory
from market_predictor.research_lane import benchmark_tilt, two_stage
from market_predictor.research_lane.fitted import Fold, date_weights, session_years, target_ranks, training_rows
from market_predictor.research_lane.free_backfill import EARLY_DECISION_START, _conservative_edgar_availability_utc
from market_predictor.research_lane.labels import HORIZONS, research_labels
from market_predictor.research_lane.panel import (
    MARKET_TIMEZONE,
    PANEL_DIRECTORY,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
)
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.scoreboard import (
    daily_rank_correlation,
    newey_west,
    score_predictor,
    scored_cells,
)
from market_predictor.research_lane.sec_filings import event_rows
from market_predictor.research_lane.wider_universe import WIDER_UNIVERSE_DIRECTORY
from market_predictor.sources.sec import SecSource

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

COLLECT_SEC_CATALYSTS_COMMAND: Final = "collect-research-lane-sec-catalysts"
NON8K_CATALYST_PARQUET: Final = "wider_non8k_catalysts.parquet"

_ACTIVIST_FORMS: Final = frozenset({"SC 13D", "SC 13D/A", "SCHEDULE 13D", "SCHEDULE 13D/A"})
_OFFERING_FORMS: Final = frozenset({"S-1", "S-3", "S-3ASR", "424B2", "424B4", "424B5"})
_LATE_FILING_FORMS: Final = frozenset({"NT 10-K", "NT 10-Q", "NT 20-F"})
_DISTRESS_8K_ITEMS: Final = frozenset({"1.03", "2.04", "2.06", "3.01", "4.01", "4.02"})

PRECISE_CATALYST_FAMILIES: Final = (
    "rx_earnings_202",
    "rx_deal_101_201",
    "rx_regfd_701",
    "rx_exec_502",
    "rx_activist_13d",
    "rx_dilutive_offering",
    "rx_accounting_distress",
)
PRECISE_STAGE2_FEATURE_NAMES: Final = (
    *PRECISE_CATALYST_FAMILIES,
    "rx_material_any",
    "rx_vol_confirmed",
    "material_abn_vol_20",
    "material_recency_21",
    "material_unusual_count_5_60",
)


def cross_sectional_neutralize(
    values: FloatArray,
    eligible: BoolArray,
    sector: npt.NDArray[np.int16],
    beta_signed_rank: Float32Array | FloatArray,
    idio_vol_signed_rank: Float32Array | FloatArray,
    *,
    rerank: bool = True,
    min_stocks: int = 20,
) -> FloatArray:
    """Project `values[t]` orthogonal to `[1, sector_dummies[t], beta_rank[t], idio_vol_rank[t]]` per session `t`.

    Uses only cross-sectional quantities at row `t` (`<= t` strictly; zero cross-date leakage).
    When `rerank=False`, the returned residuals have exact 0.0 mean within every populated sector and
    exact 0.0 dot product with `beta_signed_rank[t]` and `idio_vol_signed_rank[t]` on `eligible & finite(values)`.
    When `rerank=True`, those residuals are mapped to centered cross-sectional ranks in `[-0.5, +0.5]`.
    """
    n_sess, n_sec = values.shape
    out = np.full((n_sess, n_sec), np.nan, dtype=np.float64)
    beta_f = np.asarray(beta_signed_rank, dtype=np.float64)
    ivol_f = np.asarray(idio_vol_signed_rank, dtype=np.float64)

    for r in range(n_sess):
        mask = eligible[r] & np.isfinite(values[r])
        n = int(mask.sum())
        if n < min_stocks:
            continue
        y = values[r, mask].astype(np.float64)
        sec_r = sector[r, mask]
        uniq_s = [int(s) for s in np.unique(sec_r) if int(s) >= 0]

        cols: list[FloatArray] = [np.ones(n, dtype=np.float64)]
        for s_code in uniq_s:
            cols.append((sec_r == s_code).astype(np.float64))
        b_vec = np.where(np.isfinite(beta_f[r, mask]), beta_f[r, mask], 0.0)
        v_vec = np.where(np.isfinite(ivol_f[r, mask]), ivol_f[r, mask], 0.0)
        cols.append(b_vec)
        cols.append(v_vec)

        x_mat = np.column_stack(cols)
        xtx = x_mat.T @ x_mat
        xtx.flat[:: xtx.shape[0] + 1] += 1e-9
        coef = np.linalg.solve(xtx, x_mat.T @ y)
        resid = y - x_mat @ coef
        out[r, mask] = resid

    if rerank:
        return centered_rank(out, eligible)
    return out


def _extract_non8k_catalysts_from_submissions(
    sec: SecSource,
    cik: str,
    *,
    start_day: date,
    development_end: date,
) -> list[dict[str, Any]]:
    """Extract `SC 13D/A`, `S-1/S-3/424B`, and `NT 10-K/Q` filings in `[start_day, development_end]`."""
    check_development_end(development_end)
    start_iso, end_iso = start_day.isoformat(), development_end.isoformat()
    cutoff_utc = (pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")

    root = sec.client.get_json(sec.submissions_url.format(cik=cik.zfill(10)))
    if not isinstance(root, dict):
        return []
    filings = root.get("filings", {})
    if not isinstance(filings, dict):
        return []

    sections: list[dict[str, Any]] = []
    rec = filings.get("recent")
    if isinstance(rec, dict):
        sections.append(rec)
    for desc in filings.get("files", []):
        if not isinstance(desc, dict):
            continue
        f_from = str(desc.get("filingFrom", ""))
        f_to = str(desc.get("filingTo", ""))
        fname = str(desc.get("name", "")).strip()
        if fname and f_from <= end_iso and f_to >= start_iso:
            try:
                hist = sec.client.get_json(sec.historical_submission_url.format(name=fname))
                if isinstance(hist, dict):
                    sub = hist.get("filings", hist)
                    if isinstance(sub, dict):
                        sections.append(sub)
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    seen_accn: set[str] = set()
    for block in sections:
        forms = block.get("form", [])
        f_dates = block.get("filingDate", [])
        accns = block.get("accessionNumber", [])
        accepts = block.get("acceptanceDateTime", [])
        n = min(len(forms), len(f_dates), len(accns))
        for idx in range(n):
            fd = str(f_dates[idx]).strip()
            if fd < start_iso or fd > end_iso:
                continue
            fm = str(forms[idx]).strip().upper()
            fam: str | None = None
            if fm in _ACTIVIST_FORMS:
                fam = "rx_activist_13d"
            elif fm in _OFFERING_FORMS:
                fam = "rx_dilutive_offering"
            elif fm in _LATE_FILING_FORMS:
                fam = "rx_accounting_distress"
            if fam is None:
                continue
            accn = str(accns[idx]).strip()
            if not accn or accn in seen_accn:
                continue
            raw_acc = accepts[idx] if idx < len(accepts) else ""
            avail = _conservative_edgar_availability_utc(raw_acc, fd)
            if avail >= cutoff_utc:
                continue
            seen_accn.add(accn)
            rows.append({
                "security_id": f"cik:{cik.zfill(10)}",
                "sec_cik": cik.zfill(10),
                "accession_number": accn,
                "sec_form": fm,
                "catalyst_family": fam,
                "available_at_utc": avail.isoformat(),
            })
    return rows


def collect_wider_sec_catalysts(
    *,
    root: Path,
    development_end: date,
    data_root: Path | None = None,
    wider_dir: Path = WIDER_UNIVERSE_DIRECTORY,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Resumably collect `$0` SEC `SC 13D/A`, `S-3/424B`, and `NT 10-K/Q` catalysts (`<= development_end`)."""
    check_development_end(development_end)
    cfg = settings or get_settings()
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(COLLECT_SEC_CATALYSTS_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("sec non-8k catalyst collection start")
        dest = paths.write(paths.identity(wider_dir))
        selected_ciks = sorted(json.loads((dest / "selected_ciks.json").read_text(encoding="utf-8"))["selected_ciks"])

        cache_jsonl = dest / "sec_non8k_catalysts_cache.jsonl"
        cached_by_cik: dict[str, list[dict[str, Any]]] = {}
        if cache_jsonl.is_file():
            for line in cache_jsonl.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = json.loads(line)
                    cached_by_cik[str(item["cik"])] = list(item["rows"])

        todo = [c for c in selected_ciks if c not in cached_by_cik]
        sec = SecSource(cfg)
        if todo:
            with cache_jsonl.open("a", encoding="utf-8") as handle:
                for idx, cik in enumerate(todo):
                    try:
                        c_rows = _extract_non8k_catalysts_from_submissions(
                            sec, cik, start_day=date(2018, 1, 1), development_end=development_end,
                        )
                    except Exception as exc:
                        if "404" in str(exc):
                            c_rows = []
                        else:
                            raise
                    cached_by_cik[cik] = c_rows
                    handle.write(json.dumps({"cik": cik, "rows": c_rows}, sort_keys=True) + "\n")
                    if (idx + 1) % 40 == 0:
                        handle.flush()
                        release_unused_process_memory()
                        guard("sec non-8k catalyst collection")

        flat_rows = [r for c in selected_ciks for r in cached_by_cik.get(c, [])]
        if flat_rows:
            out_df = pd.DataFrame(flat_rows).drop_duplicates(["security_id", "accession_number"])
            out_df["available_at_utc"] = pd.to_datetime(out_df["available_at_utc"], utc=True)
        else:
            out_df = pd.DataFrame(columns=[
                "security_id", "sec_cik", "accession_number", "sec_form", "catalyst_family", "available_at_utc",
            ])
        out_parquet = dest / NON8K_CATALYST_PARQUET
        cast(Any, pq).write_table(pa.Table.from_pandas(out_df, preserve_index=False), out_parquet)
        fam_counts = out_df["catalyst_family"].value_counts().to_dict() if len(out_df) else {}
        return {
            "development_end": development_end.isoformat(),
            "ciks_processed": len(selected_ciks),
            "newly_queried_ciks": len(todo),
            "non8k_catalyst_rows": len(out_df),
            "by_family": {str(k): int(v) for k, v in sorted(fam_counts.items())},
            "alpaca_historical_news_2019_2024_available": False,
            "alpaca_news_note": "Alpaca /v1beta1/news migrated to fiscal.ai and returns 0 historical 2019-2024 rows.",
        }


def build_precise_sec_catalyst_features(
    panel: ResearchPanel,
    all_8k_events: pd.DataFrame,
    non8k_events: pd.DataFrame,
    *,
    unseal_final_test_period: bool = False,
) -> tuple[dict[str, Float32Array], dict[str, Any]]:
    """Build noise-free Stage-2 continuous reaction & volume-confirmed SEC catalyst features (`<= development_end`)."""
    check_development_end(panel.development_end, unseal_final_test_period=unseal_final_test_period)
    cutoff_utc = (pd.Timestamp(panel.development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")
    n_sess, n_sec = len(panel.sessions), len(panel.security_ids)
    col_of = {sid: idx for idx, sid in enumerate(panel.security_ids)}

    ev_records: list[tuple[int, int, str]] = []
    dropped_boilerplate_8k = 0
    fam_counter: dict[str, int] = {f: 0 for f in PRECISE_CATALYST_FAMILIES}

    if len(all_8k_events):
        av_8k = pd.to_datetime(all_8k_events["available_at_utc"], utc=True)
        if bool((av_8k >= cutoff_utc).any()):
            raise SealedPeriodError("8-K events at or after development_end reached precise catalyst builder")
        r_8k = event_rows(panel.sessions, av_8k)
        for r_i, sid_raw, items_raw in zip(
            r_8k, all_8k_events["security_id"].astype(str), all_8k_events["item_codes"].fillna("").astype(str), strict=False,
        ):
            ri = int(r_i)
            c_i = col_of.get(sid_raw, -1)
            if c_i < 0 or ri < 1 or ri >= n_sess:
                continue
            iset = {x.strip() for x in items_raw.split(",") if x.strip()}
            matched: list[str] = []
            if "2.02" in iset:
                matched.append("rx_earnings_202")
            if "1.01" in iset or "2.01" in iset:
                matched.append("rx_deal_101_201")
            if "7.01" in iset and "2.02" not in iset:
                matched.append("rx_regfd_701")
            if "5.02" in iset and "5.07" not in iset:
                matched.append("rx_exec_502")
            if iset & _DISTRESS_8K_ITEMS:
                matched.append("rx_accounting_distress")
            if not matched:
                dropped_boilerplate_8k += 1
                continue
            for mf in matched:
                ev_records.append((ri, c_i, mf))
                fam_counter[mf] += 1

    if len(non8k_events):
        av_n8 = pd.to_datetime(non8k_events["available_at_utc"], utc=True)
        if bool((av_n8 >= cutoff_utc).any()):
            raise SealedPeriodError("non-8K SEC catalyst events after development_end reached builder")
        r_n8 = event_rows(panel.sessions, av_n8)
        for r_i, sid_raw, fam_raw in zip(
            r_n8, non8k_events["security_id"].astype(str), non8k_events["catalyst_family"].astype(str), strict=False,
        ):
            ri = int(r_i)
            c_i = col_of.get(sid_raw, -1)
            if c_i >= 0 and 1 <= ri < n_sess and fam_raw in fam_counter:
                ev_records.append((ri, c_i, fam_raw))
                fam_counter[fam_raw] += 1

    count_mat = np.zeros((n_sess, n_sec), dtype=np.float32)
    raw_feats: dict[str, Float32Array] = {
        name: np.full((n_sess, n_sec), np.nan, dtype=np.float32) for name in PRECISE_STAGE2_FEATURE_NAMES
    }

    if ev_records:
        closes = panel.close.astype(np.float64)
        vols = panel.volume.astype(np.float64)
        spy_c = panel.spy_close.astype(np.float64)
        v_ok = np.isfinite(vols) & (vols >= 0.0)
        v_csum = np.vstack([np.zeros((1, n_sec)), np.cumsum(np.where(v_ok, vols, 0.0), axis=0)])
        c_csum = np.vstack([np.zeros((1, n_sec), dtype=np.int32), np.cumsum(v_ok.astype(np.int32), axis=0)])
        decay_lut = np.power(0.5, np.arange(22, dtype=np.float64) / 5.0)
        sess_idx = np.arange(n_sess, dtype=np.int64)

        ev_df = pd.DataFrame(ev_records, columns=["row", "col", "fam"]).drop_duplicates()
        for r_u, c_u in ev_df[["row", "col"]].drop_duplicates().to_numpy():
            count_mat[int(r_u), int(c_u)] += 1.0

        for col_id, grp in ev_df.groupby("col", sort=False):
            c = int(col_id)
            dist_r = np.sort(grp["row"].unique().astype(np.int64))
            prev_r = dist_r - 1
            px_ok = (
                np.isfinite(closes[dist_r, c]) & (closes[dist_r, c] > 0.0)
                & np.isfinite(closes[prev_r, c]) & (closes[prev_r, c] > 0.0)
                & np.isfinite(spy_c[dist_r]) & (spy_c[dist_r] > 0.0)
                & np.isfinite(spy_c[prev_r]) & (spy_c[prev_r] > 0.0)
            )
            rx_v = np.full(dist_r.shape, np.nan, dtype=np.float64)
            rx_v[px_ok] = (
                (closes[dist_r[px_ok], c] / closes[prev_r[px_ok], c] - 1.0)
                - (spy_c[dist_r[px_ok]] / spy_c[prev_r[px_ok]] - 1.0)
            )
            vs = np.maximum(0, dist_r - 20)
            vc = c_csum[dist_r, c] - c_csum[vs, c]
            vsum = v_csum[dist_r, c] - v_csum[vs, c]
            abn = np.full(dist_r.shape, np.nan, dtype=np.float64)
            vok = (vc >= 10) & (vsum > 0.0) & np.isfinite(vols[dist_r, c])
            abn[vok] = vols[dist_r[vok], c] / (vsum[vok] / vc[vok])

            pos = np.searchsorted(dist_r, sess_idx, side="right") - 1
            vp = pos >= 0
            pk = np.where(vp, pos, 0)
            age = sess_idx - dist_r[pk]
            act = vp & (age <= 21)
            wd = decay_lut[np.clip(age, 0, 21)]

            rx_dec = rx_v[pk[act]] * wd[act]
            log_abn = np.log1p(np.maximum(0.0, abn[pk[act]]))
            raw_feats["rx_material_any"][act, c] = rx_dec.astype(np.float32)
            raw_feats["rx_vol_confirmed"][act, c] = (
                rx_v[pk[act]] * np.where(np.isfinite(log_abn), log_abn, 1.0) * wd[act]
            ).astype(np.float32)
            raw_feats["material_abn_vol_20"][act, c] = (log_abn * wd[act]).astype(np.float32)
            raw_feats["material_recency_21"][act, c] = ((22 - age[act]) / 22.0).astype(np.float32)

            rx_lookup = dict(zip(dist_r.tolist(), rx_v.tolist(), strict=False))
            for fam_name in PRECISE_CATALYST_FAMILIES:
                fr = np.sort(grp.loc[grp["fam"] == fam_name, "row"].to_numpy(dtype=np.int64))
                if fr.size == 0:
                    continue
                frx = np.asarray([rx_lookup.get(int(r), np.nan) for r in fr], dtype=np.float64)
                fp = np.searchsorted(fr, sess_idx, side="right") - 1
                fv = fp >= 0
                fpk = np.where(fv, fp, 0)
                fage = sess_idx - fr[fpk]
                fact = fv & (fage <= 21)
                fwd = decay_lut[np.clip(fage, 0, 21)]
                raw_feats[fam_name][fact, c] = (frx[fpk[fact]] * fwd[fact]).astype(np.float32)

    cnt_cs = np.vstack([np.zeros((1, n_sec), dtype=np.float64), np.cumsum(count_mat.astype(np.float64), axis=0)])
    ra = np.arange(1, n_sess + 1)
    c5 = cnt_cs[ra] - cnt_cs[np.maximum(0, ra - 5)]
    c60 = cnt_cs[ra] - cnt_cs[np.maximum(0, ra - 60)]
    start_r = int(np.searchsorted(panel.sessions, np.datetime64(EARLY_DECISION_START, "D"), side="left"))
    warm_c = np.arange(n_sess)[:, None] >= (start_r + 20)
    raw_feats["material_unusual_count_5_60"] = np.where(
        warm_c & panel.eligible, c5 / np.maximum(c60 / 12.0, 0.25), np.nan,
    ).astype(np.float32)

    ranked_out: dict[str, Float32Array] = {}
    active_era = (np.arange(n_sess)[:, None] >= start_r) & panel.eligible
    for name in PRECISE_STAGE2_FEATURE_NAMES:
        arr_f = raw_feats.pop(name).astype(np.float64)
        base_a = (
            np.where(active_era, arr_f, np.nan)
            if name == "material_unusual_count_5_60"
            else np.where(active_era, np.nan_to_num(arr_f, nan=0.0), np.nan)
        )
        del arr_f
        ranked_out[name] = centered_rank(base_a, active_era).astype(np.float32)
        del base_a
    release_unused_process_memory()
    return ranked_out, {
        "qualified_catalyst_tuples": len(ev_records),
        "dropped_boilerplate_8k_rows": dropped_boilerplate_8k,
        "by_catalyst_family": fam_counter,
        "features": list(PRECISE_STAGE2_FEATURE_NAMES),
    }


def run_beta_neutral_stage1_walk_forward(
    panel: ResearchPanel,
    bundle: two_stage.Stage1FeatureBundle,
    security_cik_map: Mapping[str, str],
    *,
    first_row: int,
    horizons: tuple[int, ...] = HORIZONS,
    checkpoint: Callable[[str], None] | None = None,
) -> dict[int, dict[str, Any]]:
    """Run beta/vol/sector-neutralized Stage-1 (`equal_rank_24_beta_neutral`, `ridge_beta_neutral`, `xgb_shallow_5seed_beta_neutral`)."""
    folds = two_stage._dynamic_folds(panel.sessions, first_row)
    if not folds:
        raise ValueError("insufficient session span to form walk-forward folds")

    beta_rk = bundle.features["px_beta_252d"]
    ivol_rk = bundle.features["px_idio_vol_60d"]
    eq_neutral = cross_sectional_neutralize(
        bundle.equal_rank_24, panel.eligible, panel.sector, beta_rk, ivol_rk, rerank=True,
    )

    issuer_buckets = two_stage._issuer_fold_ids(panel.security_ids, security_cik_map)
    by_horizon: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        labels = research_labels(panel, horizon)
        raw_target = target_ranks(labels.stock, panel.eligible)
        # Purged training target: forward rank stripped of realized session-(s..s+h) sector, beta & idio_vol factor returns!
        neutral_target = cross_sectional_neutralize(
            raw_target, panel.eligible, panel.sector, beta_rk, ivol_rk, rerank=True,
        )
        shape = panel.eligible.shape

        news_start_row = int(np.searchsorted(panel.sessions, np.datetime64(two_stage.NEWS_START_DATE, "D"), side="left"))
        first_fold_start = folds[0].test_start
        pre_fold_slice = (
            Fold(year=0, test_start=news_start_row, test_stop=first_fold_start)
            if news_start_row < first_fold_start and (news_start_row - first_row) >= 60
            else None
        )

        eq_oos = np.full(shape, np.nan, dtype=np.float64)
        eq_oos[first_fold_start:] = eq_neutral[first_fold_start:]
        eq_for_s2 = np.full(shape, np.nan, dtype=np.float64)
        eq_for_s2[min(news_start_row, first_fold_start) :] = eq_neutral[min(news_start_row, first_fold_start) :]

        # Keep key `"equal_rank_24"` as the primary baseline key so `evaluate_two_stage_scoreboard` §5.6 logic works identically!
        learner_outputs: dict[str, dict[str, Any]] = {
            "equal_rank_24": {
                "oos_scores": eq_oos,
                "stage2_base_scores": eq_for_s2,
                "unseen_oos_scores": eq_oos if horizon in two_stage.UNSEEN_CV_HORIZONS else None,
                "importance": np.full(len(bundle.feature_names), 1.0 / len(bundle.feature_names), dtype=np.float64),
            }
        }

        for spec_name, learner_kind in (("ridge_rank", "ridge"), ("xgb_shallow_5seed", "xgboost")):
            oos_raw = np.full(shape, np.nan, dtype=np.float64)
            s2_raw = np.full(shape, np.nan, dtype=np.float64)
            imps: list[FloatArray] = []

            if pre_fold_slice is not None:
                tr_rows = training_rows(first_row, pre_fold_slice.test_start, horizon)
                if tr_rows.size >= 10:
                    tr_cells = panel.eligible[tr_rows] & np.isfinite(neutral_target[tr_rows])
                    te_rows = np.arange(pre_fold_slice.test_start, pre_fold_slice.test_stop, dtype=np.int64)
                    te_cells = panel.eligible[te_rows]
                    preds_pre, _ = two_stage._fit_predict_tabular(
                        learner_kind,
                        two_stage._design_matrix(bundle.features, bundle.feature_names, tr_rows, tr_cells),
                        neutral_target[tr_rows][tr_cells].astype(np.float64),
                        date_weights(tr_cells),
                        two_stage._design_matrix(bundle.features, bundle.feature_names, te_rows, te_cells),
                    )
                    blk = np.full(te_cells.shape, np.nan, dtype=np.float64)
                    blk[te_cells] = preds_pre
                    s2_raw[te_rows] = blk

            for fold in folds:
                tr_rows = training_rows(first_row, fold.test_start, horizon)
                tr_cells = panel.eligible[tr_rows] & np.isfinite(neutral_target[tr_rows])
                te_rows = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
                te_cells = panel.eligible[te_rows]
                preds, imp = two_stage._fit_predict_tabular(
                    learner_kind,
                    two_stage._design_matrix(bundle.features, bundle.feature_names, tr_rows, tr_cells),
                    neutral_target[tr_rows][tr_cells].astype(np.float64),
                    date_weights(tr_cells),
                    two_stage._design_matrix(bundle.features, bundle.feature_names, te_rows, te_cells),
                )
                blk = np.full(te_cells.shape, np.nan, dtype=np.float64)
                blk[te_cells] = preds
                oos_raw[te_rows] = blk
                s2_raw[te_rows] = blk
                imps.append(imp)
                if checkpoint is not None:
                    checkpoint(f"beta_neutral stage1 {spec_name} h={horizon} fold={fold.year}")

            oos_scores = cross_sectional_neutralize(
                oos_raw, panel.eligible, panel.sector, beta_rk, ivol_rk, rerank=True,
            )
            stage2_base = cross_sectional_neutralize(
                s2_raw, panel.eligible, panel.sector, beta_rk, ivol_rk, rerank=True,
            )
            del oos_raw, s2_raw
            gc.collect()

            unseen_scores: FloatArray | None = None
            if horizon in two_stage.UNSEEN_CV_HORIZONS:
                unseen_raw = np.full(shape, np.nan, dtype=np.float64)
                for fold in folds:
                    tr_rows = training_rows(first_row, fold.test_start, horizon)
                    te_rows = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
                    for b_id in range(two_stage.UNSEEN_ISSUER_FOLDS):
                        in_bucket = (issuer_buckets == b_id)[None, :]
                        tr_cells = panel.eligible[tr_rows] & np.isfinite(neutral_target[tr_rows]) & ~in_bucket
                        te_cells = panel.eligible[te_rows] & in_bucket
                        if not bool(te_cells.any()) or int(tr_cells.sum()) < 50:
                            continue
                        u_preds, _ = two_stage._fit_predict_tabular(
                            learner_kind,
                            two_stage._design_matrix(bundle.features, bundle.feature_names, tr_rows, tr_cells),
                            neutral_target[tr_rows][tr_cells].astype(np.float64),
                            date_weights(tr_cells),
                            two_stage._design_matrix(bundle.features, bundle.feature_names, te_rows, te_cells),
                            seeds=two_stage.UNSEEN_XGB_SEEDS,
                        )
                        sub_blk = unseen_raw[te_rows]
                        sub_blk[te_cells] = u_preds
                        unseen_raw[te_rows] = sub_blk
                    if checkpoint is not None:
                        checkpoint(f"beta_neutral unseen {spec_name} h={horizon} fold={fold.year}")
                unseen_scores = cross_sectional_neutralize(
                    unseen_raw, panel.eligible, panel.sector, beta_rk, ivol_rk, rerank=True,
                )
                del unseen_raw
                gc.collect()

            learner_outputs[spec_name] = {
                "oos_scores": oos_scores,
                "stage2_base_scores": stage2_base,
                "unseen_oos_scores": unseen_scores,
                "importance": np.mean(imps, axis=0) if imps else np.zeros(len(bundle.feature_names), dtype=np.float64),
            }

        by_horizon[horizon] = {
            "labels": labels,
            "target": neutral_target,
            "folds": folds,
            "scored_start": first_fold_start,
            "models": learner_outputs,
        }
    return by_horizon


__all__ = [
    "COLLECT_SEC_CATALYSTS_COMMAND",
    "NON8K_CATALYST_PARQUET",
    "PANEL_DIRECTORY",
    "PRECISE_CATALYST_FAMILIES",
    "PRECISE_STAGE2_FEATURE_NAMES",
    "benchmark_tilt",
    "build_precise_sec_catalyst_features",
    "collect_wider_sec_catalysts",
    "cross_sectional_neutralize",
    "daily_rank_correlation",
    "newey_west",
    "run_beta_neutral_stage1_walk_forward",
    "score_predictor",
    "scored_cells",
    "session_years",
]
