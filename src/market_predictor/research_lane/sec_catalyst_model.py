"""Step 7: Train, persist, verify, and score the surviving `mid_small_beta_neutral_sec` Stage-2 XGBoost bundle.

Trains the 3-seed shallow `XGBRegressor` ensemble (`STAGE2_XGB_PARAMS`) on `mid_small_ex_sp500`
beta/sector/idio-vol-neutralized forward-rank residuals (`neutral_target[h] - mid_bn_eq24`) across
`2019-07-09..2026-07-08` (strictly purged so `t + h < len(sessions)` and every training label matures
on or before `2026-07-08`), persists native `.ubj` trees + SHA-256 `_manifest.json`, verifies
artifact integrity on load (`ArtifactIntegrityError`), and scores live prospective panels using the
exact batch `build_precise_sec_catalyst_features` transformation alongside `OPRA` straddle/skew rules.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBRegressor

from market_predictor.config import Settings, get_settings
from market_predictor.core.errors import ArtifactIntegrityError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.process_memory import process_memory_snapshot, release_unused_process_memory
from market_predictor.research_lane import beta_neutral, fundamentals, two_stage
from market_predictor.research_lane.company_facts import IssuerFacts, read_company_facts
from market_predictor.research_lane.fitted import (
    SAMPLE_EVERY_SESSIONS,
    date_weights,
    target_ranks,
    training_rows,
)
from market_predictor.research_lane.free_backfill import (
    BACKFILL_DIRECTORY,
    _conservative_edgar_availability_utc,
)
from market_predictor.research_lane.labels import research_labels
from market_predictor.research_lane.panel import (
    PANEL_DIRECTORY,
    ResearchPanel,
    SealedPeriodError,
    file_sha256,
)
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.sealed_holdout import (
    SEALED_EXTENSION_DIR,
    SEALED_HOLDOUT_END,
    _load_unsealed_sp500_and_mid_small,
    collect_sealed_holdout_extension,
)
from market_predictor.research_lane.trial_log import TRIAL_LOG_NAME, append_trial, git_identity
from market_predictor.research_lane.wider_universe import WIDER_UNIVERSE_DIRECTORY

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

TRAIN_SEC_MODEL_COMMAND: Final = "train-research-lane-sec-catalyst-model"
MODEL_SCHEMA_VERSION: Final = "market_predictor.research_lane.sec_catalyst_model.v1"
MODEL_ID: Final = "mid_small_beta_neutral_sec_xgb"
DEFAULT_SEC_MODEL_BUNDLE_DIR: Final = Path("data/research/research_lane/models/mid_small_beta_neutral_sec_xgb")
MODEL_HORIZONS: Final[tuple[int, ...]] = (10, 21)
LIVE_SEC_LOOKBACK_DAYS: Final = 105


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def feature_contract_sha256(feature_names: tuple[str, ...]) -> str:
    """Deterministic SHA-256 of the ordered Stage-2 feature contract."""
    payload = json.dumps(list(feature_names), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class LoadedSecCatalystBundle:
    """Integrity-verified Stage-2 SEC Catalyst `XGBRegressor` ensemble loaded from disk."""

    bundle_dir: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    feature_names: tuple[str, ...]
    horizons: tuple[int, ...]
    estimators_by_horizon: Mapping[int, tuple[XGBRegressor, ...]]

    def predict_residual(self, horizon: int, x_matrix: Float32Array) -> FloatArray:
        if x_matrix.ndim != 2 or x_matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected 2D matrix with {len(self.feature_names)} columns, got shape {x_matrix.shape}"
            )
        estimators = self.estimators_by_horizon[horizon]
        if x_matrix.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        acc = np.zeros(x_matrix.shape[0], dtype=np.float64)
        for est in estimators:
            acc += np.asarray(est.predict(x_matrix), dtype=np.float64)
        return acc / float(len(estimators))


def fit_and_save_sec_catalyst_bundle_from_panel(
    panel: ResearchPanel,
    base_bn_scores: FloatArray,
    *,
    beta_rank: Float32Array | FloatArray,
    idio_vol_rank: Float32Array | FloatArray,
    catalyst_features: Mapping[str, Float32Array],
    destination: Path,
    root: Path,
    input_sha256: Mapping[str, str],
    catalyst_report: Mapping[str, Any],
    horizons: tuple[int, ...] = MODEL_HORIZONS,
    xgb_params: Mapping[str, Any] = two_stage.STAGE2_XGB_PARAMS,
    seeds: tuple[int, ...] = two_stage.STAGE2_XGB_SEEDS,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit and atomically persist `.ubj` XGBoost models + SHA-256 `_manifest.json` from prepared matrices."""
    if destination.exists() and not overwrite:
        raise FileExistsError(f"SEC catalyst model bundle already exists: {destination}")

    feat_names = beta_neutral.PRECISE_STAGE2_FEATURE_NAMES
    for fn in feat_names:
        if fn not in catalyst_features:
            raise ValueError(f"missing required Stage-2 feature: {fn}")

    news_start_row = int(
        np.searchsorted(panel.sessions, np.datetime64(two_stage.NEWS_START_DATE, "D"), side="left")
    )
    n_sess = len(panel.sessions)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
    staging.mkdir(parents=True)

    artifact_files: dict[str, dict[str, Any]] = {}
    horizon_reports: dict[str, Any] = {}

    try:
        for horizon in horizons:
            labels = research_labels(panel, horizon)
            raw_target = target_ranks(labels.stock, panel.eligible)
            neutral_target = beta_neutral.cross_sectional_neutralize(
                raw_target,
                panel.eligible,
                panel.sector,
                beta_rank,
                idio_vol_rank,
                rerank=True,
            )
            residual = neutral_target - base_bn_scores
            tr_rows = training_rows(news_start_row, n_sess, horizon)
            if tr_rows.size == 0:
                raise ValueError(f"no matured training rows for horizon {horizon}")
            # Strictly enforce that every training row matures on or before `n_sess - 1`
            if int(tr_rows.max()) + horizon >= n_sess:
                raise SealedPeriodError(f"unmatured label row leaked into training for h={horizon}")

            tr_cells = panel.eligible[tr_rows] & np.isfinite(residual[tr_rows])
            n_examples = int(tr_cells.sum())
            if n_examples < 50:
                raise ValueError(f"insufficient training examples ({n_examples} < 50) for horizon {horizon}")

            x_train = two_stage._design_matrix(catalyst_features, feat_names, tr_rows, tr_cells)
            y_train = residual[tr_rows][tr_cells].astype(np.float64)
            w_train = date_weights(tr_cells)

            imp_acc = np.zeros(len(feat_names), dtype=np.float64)
            seed_files: list[str] = []
            for seed in seeds:
                est = XGBRegressor(**xgb_params, random_state=seed)
                est.fit(x_train, y_train, sample_weight=w_train)
                imp_acc += np.asarray(est.feature_importances_, dtype=np.float64)
                rel_name = f"xgb_h{horizon}_seed{seed}.ubj"
                ubj_path = staging / rel_name
                est.save_model(ubj_path)
                artifact_files[rel_name] = {
                    "bytes": int(ubj_path.stat().st_size),
                    "sha256": file_sha256(ubj_path),
                    "horizon_sessions": int(horizon),
                    "seed": int(seed),
                }
                seed_files.append(rel_name)
                del est
            gc.collect()

            mean_imp = imp_acc / float(len(seeds))
            order_imp = np.argsort(-mean_imp)
            horizon_reports[str(horizon)] = {
                "sampled_decision_sessions": int(tr_rows.size),
                "first_decision_session": str(panel.sessions[int(tr_rows[0])]),
                "last_matured_decision_session": str(panel.sessions[int(tr_rows[-1])]),
                "last_outcome_session_used": str(panel.sessions[int(tr_rows[-1]) + horizon]),
                "training_examples_stock_sessions": n_examples,
                "distinct_eligible_securities": int(tr_cells.any(axis=0).sum()),
                "model_files": seed_files,
                "feature_importance": [
                    {"feature": feat_names[int(i)], "importance": float(mean_imp[int(i)])}
                    for i in order_imp
                ],
            }

        mem_snap = process_memory_snapshot()
        mem_gib = {"current_gib": mem_snap[0] / 2**30, "peak_gib": mem_snap[1] / 2**30} if mem_snap else None
        manifest = _json_safe({
            "schema": MODEL_SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "research_only": True,
            "actionable": False,
            "promotion_permitted": False,
            "label_notice": (
                "research label on provider split- and dividend-adjusted SIP daily bars; "
                "not a promoted production release"
            ),
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "cohort": "mid_small_ex_sp500",
            "neutralization": "per_session_ols_orthogonal_to_sector_beta252_idiovol60",
            "train_window": {
                "warmup_first_session": str(panel.sessions[0]),
                "catalyst_train_start": two_stage.NEWS_START_DATE.isoformat(),
                "data_end_session": str(panel.sessions[-1]),
                "sample_every_sessions": SAMPLE_EVERY_SESSIONS,
            },
            "feature_names": list(feat_names),
            "feature_contract_sha256": feature_contract_sha256(feat_names),
            "horizons_sessions": [int(h) for h in horizons],
            "seeds": [int(s) for s in seeds],
            "xgb_params": dict(xgb_params),
            "artifact_files": artifact_files,
            "horizons": horizon_reports,
            "catalyst_report": dict(catalyst_report),
            "inputs": dict(sorted(input_sha256.items())),
            "code": git_identity(root),
            "process_memory_gib": mem_gib,
        })
        (staging / "_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
        )
        if destination.exists() and overwrite:
            shutil.rmtree(destination)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return cast(dict[str, Any], manifest)


def verify_and_load_sec_catalyst_bundle(bundle_dir: Path) -> LoadedSecCatalystBundle:
    """Verify `_manifest.json`, ordered feature hash, and every `.ubj` SHA-256 before loading models."""
    manifest_path = bundle_dir / "_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"missing SEC catalyst model manifest: {manifest_path}")

    manifest_sha = file_sha256(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArtifactIntegrityError(f"unreadable SEC catalyst model manifest: {manifest_path}") from exc

    if manifest.get("schema") != MODEL_SCHEMA_VERSION:
        raise ArtifactIntegrityError(
            f"unsupported SEC catalyst model schema {manifest.get('schema')!r}; expected {MODEL_SCHEMA_VERSION!r}"
        )
    raw_feats = manifest.get("feature_names")
    if not isinstance(raw_feats, list) or tuple(str(x) for x in raw_feats) != beta_neutral.PRECISE_STAGE2_FEATURE_NAMES:
        raise ArtifactIntegrityError("SEC catalyst model feature_names mismatch against PRECISE_STAGE2_FEATURE_NAMES")

    feat_tuple = beta_neutral.PRECISE_STAGE2_FEATURE_NAMES
    expected_contract_sha = feature_contract_sha256(feat_tuple)
    if manifest.get("feature_contract_sha256") != expected_contract_sha:
        raise ArtifactIntegrityError("SEC catalyst model feature_contract_sha256 tamper detected")

    artifact_files = manifest.get("artifact_files")
    if not isinstance(artifact_files, dict) or not artifact_files:
        raise ArtifactIntegrityError("SEC catalyst model manifest contains no artifact_files")

    horizons = tuple(int(h) for h in manifest.get("horizons_sessions", []))
    seeds = tuple(int(s) for s in manifest.get("seeds", []))
    if not horizons or not seeds:
        raise ArtifactIntegrityError("SEC catalyst model manifest missing horizons_sessions or seeds")

    loaded_by_h: dict[int, tuple[XGBRegressor, ...]] = {}
    for h in horizons:
        ests: list[XGBRegressor] = []
        for s in seeds:
            rel_name = f"xgb_h{h}_seed{s}.ubj"
            meta = artifact_files.get(rel_name)
            if not isinstance(meta, dict):
                raise ArtifactIntegrityError(f"manifest missing entry for {rel_name}")
            ubj_path = bundle_dir / rel_name
            if not ubj_path.is_file():
                raise ArtifactIntegrityError(f"missing model payload file: {ubj_path}")
            actual_bytes = int(ubj_path.stat().st_size)
            if actual_bytes != int(meta.get("bytes", -1)):
                raise ArtifactIntegrityError(
                    f"byte-size mismatch on {rel_name}: expected {meta.get('bytes')}, got {actual_bytes}"
                )
            actual_sha = file_sha256(ubj_path)
            if actual_sha != str(meta.get("sha256", "")):
                raise ArtifactIntegrityError(
                    f"SHA-256 tamper mismatch on {rel_name}: expected {meta.get('sha256')}, got {actual_sha}"
                )
            reg = XGBRegressor()
            reg.load_model(ubj_path)
            ests.append(reg)
        loaded_by_h[h] = tuple(ests)

    return LoadedSecCatalystBundle(
        bundle_dir=bundle_dir,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        feature_names=feat_tuple,
        horizons=horizons,
        estimators_by_horizon=loaded_by_h,
    )


def extract_live_sec_batch_tables_for_cik(
    payload: dict[str, Any],
    *,
    security_id: str,
    cik: str,
    window_start_utc: datetime,
    cutoff_utc: datetime,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract `all_8k_events` and `non8k_events` rows from live SEC `submissions` matching batch schema."""
    recent = payload.get("filings", {}).get("recent", {})
    if not isinstance(recent, dict):
        return [], []
    forms = recent.get("form", [])
    f_dates = recent.get("filingDate", [])
    accepts = recent.get("acceptanceDateTime", [])
    accs = recent.get("accessionNumber", [])
    items_list = recent.get("items", [])

    win_start_ts = pd.Timestamp(window_start_utc).tz_convert("UTC")
    cutoff_ts = pd.Timestamp(cutoff_utc).tz_convert("UTC")
    cik_10 = str(cik).zfill(10)

    ev_8k: list[dict[str, Any]] = []
    ev_non8k: list[dict[str, Any]] = []
    seen_8k: set[str] = set()
    seen_non8k: set[str] = set()
    n = min(len(forms), len(f_dates), len(accs))
    for i in range(n):
        f_str = str(f_dates[i] or "").strip()
        acc_no = str(accs[i] or "").strip()
        if not f_str or not acc_no:
            continue
        raw_acc = accepts[i] if i < len(accepts) else ""
        avail_ts = _conservative_edgar_availability_utc(raw_acc, f_str)
        if avail_ts < win_start_ts or avail_ts >= cutoff_ts:
            continue
        form = str(forms[i] or "").upper().strip()
        if form == "8-K" and acc_no not in seen_8k and i < len(items_list):
            raw_it = items_list[i]
            iset = (
                {str(x).strip() for x in raw_it if str(x).strip()}
                if isinstance(raw_it, list)
                else {x.strip() for x in str(raw_it or "").split(",") if x.strip()}
            )
            if iset:
                seen_8k.add(acc_no)
                ev_8k.append({
                    "security_id": security_id,
                    "sec_cik": cik_10,
                    "accession_number": acc_no,
                    "sec_form": "8-K",
                    "item_codes": ",".join(sorted(iset)),
                    "accepted_at_utc": avail_ts,
                    "available_at_utc": avail_ts,
                })
        elif acc_no not in seen_non8k:
            fam: str | None = None
            if form in beta_neutral._ACTIVIST_FORMS:
                fam = "rx_activist_13d"
            elif form in beta_neutral._OFFERING_FORMS:
                fam = "rx_dilutive_offering"
            elif form in beta_neutral._LATE_FILING_FORMS:
                fam = "rx_accounting_distress"
            if fam is not None:
                seen_non8k.add(acc_no)
                ev_non8k.append({
                    "security_id": security_id,
                    "sec_cik": cik_10,
                    "accession_number": acc_no,
                    "sec_form": form,
                    "catalyst_family": fam,
                    "available_at_utc": avail_ts,
                })
    return ev_8k, ev_non8k


def predict_live_sec_catalyst_scores(
    panel: ResearchPanel,
    scores_df: pd.DataFrame,
    *,
    all_8k_events: pd.DataFrame,
    non8k_events: pd.DataFrame,
    queried_security_ids: set[str],
    bundle: LoadedSecCatalystBundle,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score `panel`'s final session using the exact batch `build_precise_sec_catalyst_features` + verified XGBoost."""
    enriched = scores_df.copy()
    n_sess, n_sec = len(panel.sessions), len(panel.security_ids)
    last_idx = n_sess - 1

    # Restrict `eligible` at `last_idx` strictly to SEC-queried securities so unqueried stocks stay `None` (never `0.0`).
    queried_elig = np.zeros((n_sess, n_sec), dtype=bool)
    for col_i, sid in enumerate(panel.security_ids):
        if bool(panel.eligible[last_idx, col_i]) and sid in queried_security_ids:
            queried_elig[last_idx, col_i] = True

    n_queried_elig = int(queried_elig[last_idx].sum())
    if n_queried_elig < 2:
        for h in bundle.horizons:
            enriched[f"sec_xgb_residual_{h}d"] = None
            enriched[f"score_mid_small_bn_sec_xgb_{h}d"] = None
        return enriched, {
            "model_scored": False,
            "reason": f"insufficient_sec_queried_stocks ({n_queried_elig} < 2)",
            "model_id": MODEL_ID,
            "manifest_sha256": bundle.manifest_sha256,
        }

    queried_panel = replace(panel, eligible=queried_elig)
    sec_feats, cat_rep = beta_neutral.build_precise_sec_catalyst_features(
        queried_panel,
        all_8k_events,
        non8k_events,
        unseal_final_test_period=True,
    )

    te_rows = np.asarray([last_idx], dtype=np.int64)
    te_cells = queried_elig[te_rows]
    x_last = two_stage._design_matrix(sec_feats, bundle.feature_names, te_rows, te_cells)
    active_cols = np.flatnonzero(te_cells[0])
    sid_by_col = {int(c): panel.security_ids[int(c)] for c in active_cols}

    sids_df = enriched["security_id"].astype(str).tolist()
    base_bn = pd.to_numeric(enriched["score_beta_neutral_price_12"], errors="coerce").to_numpy(dtype=np.float64)
    base_by_sid = {sid: base_bn[idx] for idx, sid in enumerate(sids_df)}

    scored_counts: dict[str, int] = {}
    for h in bundle.horizons:
        r_pred = bundle.predict_residual(h, x_last)
        res_by_sid: dict[str, float] = {}
        raw_comb = np.full(len(sids_df), np.nan, dtype=np.float64)
        sid_index = {sid: idx for idx, sid in enumerate(sids_df)}
        for pos_i, col_arr in enumerate(active_cols):
            sid = sid_by_col[int(col_arr)]
            rv = float(r_pred[pos_i])
            res_by_sid[sid] = rv
            b_val = base_by_sid.get(sid, np.nan)
            if sid in sid_index and np.isfinite(b_val) and np.isfinite(rv):
                raw_comb[sid_index[sid]] = b_val + rv

        comb_mask = np.isfinite(raw_comb)
        ranked_comb = (
            centered_rank(raw_comb[None, :], comb_mask[None, :])[0]
            if int(comb_mask.sum()) >= 2
            else raw_comb
        )
        enriched[f"sec_xgb_residual_{h}d"] = [
            None if s not in res_by_sid or not np.isfinite(res_by_sid[s]) else float(res_by_sid[s])
            for s in sids_df
        ]
        enriched[f"score_mid_small_bn_sec_xgb_{h}d"] = [
            None if not np.isfinite(v) else float(v) for v in ranked_comb
        ]
        scored_counts[f"{h}d"] = int(np.isfinite(ranked_comb).sum())

    return enriched, {
        "model_scored": True,
        "model_id": MODEL_ID,
        "manifest_sha256": bundle.manifest_sha256,
        "feature_contract_sha256": str(bundle.manifest["feature_contract_sha256"]),
        "sec_queried_eligible_stocks": n_queried_elig,
        "scored_stocks_by_horizon": scored_counts,
        "precise_sec_feature_report": cat_rep,
    }


def apply_opra_straddle_skew_overlay(
    scores_df: pd.DataFrame,
    *,
    base_col: str = "score_mid_small_bn_sec_xgb_10d",
    out_col: str = "score_mid_small_bn_sec_xgb_opra_10d",
) -> list[float | None]:
    """Combine trained `mid_small_bn_sec_xgb` rank with live `OPRA` `25d` put-skew veto & `ATM straddle` shock rule.

    1. Veto (`None`) any stock flagged by `falling_knife_put_skew_veto` (`25d` Put-Call IV skew >= max(5%, p90) on dip).
    2. Conviction boost (`+0.10 * clip(shock_to_straddle - 1.0, 0.0, 1.5)`) when a positive catalyst's 1d market-adjusted
       move exceeded the `OPRA` front-expiry `ATM Straddle %` (`shock_to_straddle > 1.0`).
    3. Under-reaction fade (`-0.05`) when a stock gapped > +2% on a catalyst (`latest_catalyst_rx_1d > 0.02`) yet moved
       less than half of its priced-in `ATM Straddle %` (`0 < shock_to_straddle < 0.50`).
    4. Re-rank surviving finite stocks into `[-0.5, +0.5]`.
    """
    if base_col not in scores_df.columns:
        return [None] * len(scores_df)

    base = pd.to_numeric(scores_df[base_col], errors="coerce").to_numpy(dtype=np.float64)
    veto = (
        scores_df["falling_knife_put_skew_veto"].fillna(False).astype(bool).to_numpy()
        if "falling_knife_put_skew_veto" in scores_df.columns
        else np.zeros(len(scores_df), dtype=bool)
    )
    shock = (
        pd.to_numeric(scores_df["catalyst_shock_to_straddle"], errors="coerce").to_numpy(dtype=np.float64)
        if "catalyst_shock_to_straddle" in scores_df.columns
        else np.full(len(scores_df), np.nan, dtype=np.float64)
    )
    rx1 = (
        pd.to_numeric(scores_df["latest_catalyst_rx_1d"], errors="coerce").to_numpy(dtype=np.float64)
        if "latest_catalyst_rx_1d" in scores_df.columns
        else np.full(len(scores_df), np.nan, dtype=np.float64)
    )

    adj = base.copy()
    adj[veto] = np.nan
    pos_breakout = (~veto) & np.isfinite(adj) & np.isfinite(shock) & (shock > 1.0) & np.isfinite(rx1) & (rx1 > 0.0)
    adj[pos_breakout] += 0.10 * np.clip(shock[pos_breakout] - 1.0, 0.0, 1.5)

    under_react = (
        (~veto) & np.isfinite(adj) & np.isfinite(shock) & (shock > 0.0) & (shock < 0.50)
        & np.isfinite(rx1) & (rx1 > 0.02)
    )
    adj[under_react] -= 0.05

    fin = np.isfinite(adj)
    ranked = centered_rank(adj[None, :], fin[None, :])[0] if int(fin.sum()) >= 2 else adj
    _ = out_col
    return [None if not np.isfinite(v) else float(v) for v in ranked]


def train_and_persist_sec_catalyst_model(
    *,
    root: Path,
    output: Path = DEFAULT_SEC_MODEL_BUNDLE_DIR,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    sp500_backfill_dir: Path = BACKFILL_DIRECTORY,
    wider_dir: Path = WIDER_UNIVERSE_DIRECTORY,
    ext_dir: Path = SEALED_EXTENSION_DIR,
    horizons: tuple[int, ...] = MODEL_HORIZONS,
    overwrite: bool = False,
    unseal_final_test_period: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Train & persist the `mid_small_beta_neutral_sec` Stage-2 XGBoost bundle on `2019-07-09..2026-07-08`."""
    if not unseal_final_test_period:
        raise SealedPeriodError(
            f"{TRAIN_SEC_MODEL_COMMAND} trains through {SEALED_HOLDOUT_END.isoformat()} and requires "
            "--unseal-final-test-period"
        )

    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()
    identity = paths.identity(output)

    with heavy_job_lease(TRAIN_SEC_MODEL_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("sec catalyst model training start")
        destination = paths.write(identity)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"SEC catalyst model bundle already exists: {destination}")

        cfg = settings or get_settings()
        ext_files = collect_sealed_holdout_extension(
            paths, cfg, panel_rel=panel, sp500_backfill_rel=sp500_backfill_dir, wider_rel=wider_dir, ext_rel=ext_dir,
        )
        ext_sp, wdata, all_non8k_mid, extra_sha = _load_unsealed_sp500_and_mid_small(
            paths, ext_files, panel_rel=panel, sp500_backfill_rel=sp500_backfill_dir, wider_rel=wider_dir,
        )
        del ext_sp
        release_unused_process_memory()
        guard("sec catalyst model mid_small panel loaded")

        mid_facts = wdata.company_facts_paths

        def _load_mid_f(cik: str) -> IssuerFacts | None:
            p = mid_facts.get(cik)
            return read_company_facts(p, SEALED_HOLDOUT_END)[0] if (p is not None and p.is_file()) else None

        mid_bundle = two_stage.build_stage1_features(
            wdata.panel,
            fundamentals.FundamentalInputs(
                panel=wdata.panel,
                unadjusted_close=wdata.unadjusted_close,
                releases=wdata.releases_202,
                security_issuer=wdata.security_cik_map,
                load_issuer=_load_mid_f,
            ),
        )
        mid_panel = wdata.panel
        all_8k_mid = wdata.all_8k_events
        input_sha = {**wdata.input_sha256, **extra_sha}
        del wdata

        beta_rk = mid_bundle.features["px_beta_252d"].copy()
        ivol_rk = mid_bundle.features["px_idio_vol_60d"].copy()
        mid_bn_eq24 = beta_neutral.cross_sectional_neutralize(
            mid_bundle.equal_rank_24, mid_panel.eligible, mid_panel.sector, beta_rk, ivol_rk, rerank=True,
        )
        del mid_bundle
        release_unused_process_memory()
        guard("sec catalyst model stage1 computed")

        sec_cat_feats, sec_cat_rep = beta_neutral.build_precise_sec_catalyst_features(
            mid_panel, all_8k_mid, all_non8k_mid, unseal_final_test_period=True,
        )
        del all_8k_mid, all_non8k_mid
        release_unused_process_memory()
        guard("sec catalyst model stage2 features computed")

        manifest = fit_and_save_sec_catalyst_bundle_from_panel(
            mid_panel,
            mid_bn_eq24,
            beta_rank=beta_rk,
            idio_vol_rank=ivol_rk,
            catalyst_features=sec_cat_feats,
            destination=destination,
            root=root,
            input_sha256=input_sha,
            catalyst_report=sec_cat_rep,
            horizons=horizons,
            overwrite=overwrite,
        )
        # Self-verify bundle integrity immediately after atomic write!
        loaded = verify_and_load_sec_catalyst_bundle(destination)

        append_trial(
            paths.write(paths.identity(Path("data/research/research_lane"))) / TRIAL_LOG_NAME,
            root=root,
            command=TRAIN_SEC_MODEL_COMMAND,
            output=identity.relative_to(root),
            input_sha256=input_sha,
            settings={
                "model_id": MODEL_ID,
                "horizons_sessions": list(horizons),
                "train_start": two_stage.NEWS_START_DATE.isoformat(),
                "data_end": SEALED_HOLDOUT_END.isoformat(),
                "unseal_final_test_period": True,
            },
            scoreboard=manifest,
        )

    return {
        "bundle_directory": identity.relative_to(root).as_posix(),
        "manifest_sha256": loaded.manifest_sha256,
        "feature_contract_sha256": manifest["feature_contract_sha256"],
        "horizons": manifest["horizons"],
        "process_memory_gib": manifest["process_memory_gib"],
    }


__all__ = [
    "DEFAULT_SEC_MODEL_BUNDLE_DIR",
    "LIVE_SEC_LOOKBACK_DAYS",
    "LoadedSecCatalystBundle",
    "MODEL_HORIZONS",
    "MODEL_ID",
    "MODEL_SCHEMA_VERSION",
    "TRAIN_SEC_MODEL_COMMAND",
    "apply_opra_straddle_skew_overlay",
    "extract_live_sec_batch_tables_for_cik",
    "feature_contract_sha256",
    "fit_and_save_sec_catalyst_bundle_from_panel",
    "predict_live_sec_catalyst_scores",
    "train_and_persist_sec_catalyst_model",
    "verify_and_load_sec_catalyst_bundle",
]
