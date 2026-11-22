"""Step 6: Pre-Registered Single Final-Test Evaluation on the `2024-05-29..2026-07-08` Holdout Window.

Evaluates the pre-declared frozen Core-Satellite volatility-regime candidates (`R1` and `R2` from
`FROZEN_PROSPECTIVE_CANDIDATES`) strictly over the previously untouched `2024-05-29..2026-07-08`
holdout window (`unseal_final_test_period=True` required; fails closed when `False`):

1. Preserves all `<= 2024-05-28` development artifacts byte-for-byte unchanged on disk, writing
   incremental `2024-05-28..2026-07-08` CBOE, Alpaca SIP (`all` & `raw` with exact `2024-05-28`
   overlap stitching so `2024–2026` splits/dividends create zero price jumps), and SEC EDGAR
   `8-K` + non-`8-K` (`SC 13D/A`, `S-1/S-3/424B`, `NT 10-K/Q`) extension artifacts into
   `data/research/research_lane/sealed_holdout_extension_20240529_20260708/`.
2. Scores strictly out-of-sample rebalance windows starting at `2024-05-29` (`entry open[2024-05-30]`
   through `2026-07-08`, zero pre-`2024-05-29` development windows included) under both:
   - **Purged Walk-Forward Folds (`2024H2` `2024-05-29..2024-12-31`, `2025`, `2026H1` `2026-01-02..2026-07-08`)**
     where every Stage-2 catalyst `XGBoost` training row satisfies `t + h < fold.test_start`, and
   - **Strictly Frozen `2024-05-28` Model Weights (`zero refits` after `2024-05-28`)** for the primary
     candidates (`90_10__sp500_tilt_news__R1_backwardation_to_spy` and `80_20__spy__R2_adaptive_hedge_sec_xgb`).
"""
from __future__ import annotations

import gc
import json
import math
import os
import shutil
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from market_predictor.config import Settings, get_settings
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.process_memory import process_memory_snapshot, release_unused_process_memory
from market_predictor.research_lane import (
    benchmark_tilt,
    beta_neutral,
    core_satellite,
    free_backfill,
    fundamentals,
    two_stage,
    vol_regime,
    wider_universe,
)
from market_predictor.research_lane.company_facts import (
    COMPANY_FACTS_DIRECTORY,
    COMPANY_FACTS_FILE,
    COMPANY_FACTS_SOURCES,
    IssuerFacts,
    read_company_facts,
)
from market_predictor.research_lane.fitted import Fold, date_weights, session_years, target_ranks, training_rows
from market_predictor.research_lane.free_backfill import (
    BACKFILL_DIRECTORY,
    EARLY_DECISION_START,
    EXTRA_FACTS_DIR,
    NEWS_DECISION_START,
    SEC_IDENTITY_RELATIONS,
    _conservative_edgar_availability_utc,
    _fetch_alpaca_daily_batch,
    _read_panel_membership,
    existing_company_facts_paths,
    resolve_all_sp500_ciks,
)
from market_predictor.research_lane.labels import HORIZONS, research_labels
from market_predictor.research_lane.news_catalysts import EARLY_NEWS_FAMILY_DIR, LATER_NEWS_FAMILY_DIR
from market_predictor.research_lane.panel import (
    BAR_MANIFEST,
    DEFAULT_DEVELOPMENT_END,
    MARKET_TIMEZONE,
    PANEL_DIRECTORY,
    PRICE_FIELDS,
    ResearchPanel,
    SealedPeriodError,
    file_sha256,
)
from market_predictor.research_lane.scoreboard import daily_rank_correlation, newey_west, scored_cells
from market_predictor.research_lane.sec_filings import FILINGS_DIRECTORY, FILINGS_FILE, acceptance_cutoff
from market_predictor.research_lane.trial_log import TRIAL_LOG_NAME, append_trial
from market_predictor.research_lane.unadjusted_prices import UNADJUSTED_DIRECTORY
from market_predictor.research_lane.wider_universe import (
    TOP_LIQUID_COUNT,
    WIDER_UNIVERSE_DIRECTORY,
    _load_sessions_and_benchmarks,
    compute_pit_liquidity_eligibility,
)
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.sec import SecSource

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

EVALUATE_SEALED_HOLDOUT_COMMAND: Final = "evaluate-sealed-holdout-2024-2026"
SEALED_EXTENSION_DIR: Final = Path("data/research/research_lane/sealed_holdout_extension_20240529_20260708")
LATER_SEALED_SEC_FILINGS: Final = Path("data/research/swing_later_sealed_sec_form_inventory_corrected_clock/filings.parquet")
OUTPUT_PARENT: Final = Path("data/research/research_lane")
OVERLAP_STITCH_DATE: Final = DEFAULT_DEVELOPMENT_END  # 2024-05-28 exact overlap session
SEALED_HOLDOUT_START: Final = date(2024, 5, 29)
SEALED_HOLDOUT_END: Final = date(2026, 7, 8)


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


def _pct(val: Any) -> str:
    return "n/a" if val is None else f"{100.0 * float(val):.2f}%"


def _num(val: Any, digits: int = 2) -> str:
    return "n/a" if val is None else f"{float(val):.{digits}f}"


def _fetch_name_change_cutoffs(settings: Settings, start_day: date, end_day: date) -> dict[str, date]:
    """Query Alpaca `/v1/corporate-actions` (`name_change`) in `[start_day, end_day]` to quarantine symbol reuse."""
    if not settings.has_alpaca:
        return {}
    headers = {
        "APCA-API-KEY-ID": settings.alpaca_api_key_id or "",
        "APCA-API-SECRET-KEY": settings.alpaca_api_secret_value or "",
    }
    url = "https://data.alpaca.markets/v1/corporate-actions"
    cutoffs: dict[str, date] = {}
    token: str | None = None
    while True:
        params: dict[str, Any] = {
            "types": "name_change",
            "start": start_day.isoformat(),
            "end": end_day.isoformat(),
            "limit": 1000,
        }
        if token:
            params["page_token"] = token
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=20)
        except Exception:
            break
        if resp.status_code != 200:
            break
        js = resp.json()
        ca = js.get("corporate_actions", {}) if isinstance(js, dict) else {}
        for item in ca.get("name_changes", []) if isinstance(ca, dict) else []:
            if not isinstance(item, dict):
                continue
            ex_raw = str(item.get("process_date") or item.get("ex_date") or "").strip()
            if len(ex_raw) != 10:
                continue
            ex_d = date.fromisoformat(ex_raw)
            prev_d = ex_d - timedelta(days=1)
            for k_sym in ("old_symbol", "new_symbol"):
                sym = str(item.get(k_sym) or "").upper().strip()
                if sym:
                    cutoffs[sym] = prev_d if sym not in cutoffs else min(cutoffs[sym], prev_d)
        token = js.get("next_page_token") if isinstance(js, dict) else None
        if not token:
            break
    return cutoffs


def _extract_holdout_sec_events_for_cik(
    sec: SecSource,
    cik: str,
    *,
    start_day: date,
    end_day: date,
) -> dict[str, Any]:
    """Single-call extraction of `2024-05-29..2026-07-08` `8-K` items, non-`8-K` catalysts, & `last_filing_date`."""
    payload = sec.client.get_json(sec.submissions_url.format(cik=cik.zfill(10)))
    if not isinstance(payload, dict):
        return {"cik": cik, "last_filing_date": None, "events_8k": [], "events_non8k": []}

    start_iso = start_day.isoformat()
    end_iso = end_day.isoformat()
    cutoff_utc = (pd.Timestamp(end_day, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")

    filings_obj = payload.get("filings", {})
    recent = filings_obj.get("recent", {}) if isinstance(filings_obj, dict) else {}
    sections: list[dict[str, Any]] = [recent] if isinstance(recent, dict) else []
    for desc in (filings_obj.get("files", []) if isinstance(filings_obj, dict) else []):
        if isinstance(desc, dict):
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

    last_f: str | None = None
    ev_8k: list[dict[str, Any]] = []
    ev_non8k: list[dict[str, Any]] = []
    seen_8k: set[str] = set()
    seen_non8k: set[str] = set()
    cik_10 = cik.zfill(10)

    for block in sections:
        forms = block.get("form", [])
        f_dates = block.get("filingDate", [])
        accns = block.get("accessionNumber", [])
        accepts = block.get("acceptanceDateTime", [])
        items_col = block.get("items", [])
        n = min(len(forms), len(f_dates), len(accns))
        for idx in range(n):
            fd = str(f_dates[idx]).strip()
            if not fd or fd > end_iso:
                continue
            fm = str(forms[idx]).strip().upper()
            if fm in ("10-K", "10-Q", "8-K", "20-F", "4"):
                last_f = fd if last_f is None else max(last_f, fd)
            if fd < start_iso:
                continue
            accn = str(accns[idx]).strip()
            if not accn:
                continue
            raw_acc = accepts[idx] if idx < len(accepts) else ""
            avail = _conservative_edgar_availability_utc(raw_acc, fd)
            if avail >= cutoff_utc:
                continue

            if fm == "8-K" and accn not in seen_8k and idx < len(items_col):
                raw_it = items_col[idx]
                item_set = (
                    {str(x).strip() for x in raw_it if str(x).strip()}
                    if isinstance(raw_it, list)
                    else {x.strip() for x in str(raw_it or "").split(",") if x.strip()}
                )
                if item_set:
                    seen_8k.add(accn)
                    ev_8k.append({
                        "security_id": f"cik:{cik_10}",
                        "sec_cik": cik_10,
                        "accession_number": accn,
                        "sec_form": "8-K",
                        "item_codes": ",".join(sorted(item_set)),
                        "accepted_at_utc": avail.isoformat(),
                        "available_at_utc": avail.isoformat(),
                    })
            elif accn not in seen_non8k:
                fam: str | None = None
                if fm in beta_neutral._ACTIVIST_FORMS:
                    fam = "rx_activist_13d"
                elif fm in beta_neutral._OFFERING_FORMS:
                    fam = "rx_dilutive_offering"
                elif fm in beta_neutral._LATE_FILING_FORMS:
                    fam = "rx_accounting_distress"
                if fam is not None:
                    seen_non8k.add(accn)
                    ev_non8k.append({
                        "security_id": f"cik:{cik_10}",
                        "sec_cik": cik_10,
                        "accession_number": accn,
                        "sec_form": fm,
                        "catalyst_family": fam,
                        "available_at_utc": avail.isoformat(),
                    })

    return {"cik": cik_10, "last_filing_date": last_f, "events_8k": ev_8k, "events_non8k": ev_non8k}


def collect_sealed_holdout_extension(
    paths: EvidencePaths,
    settings: Settings,
    *,
    panel_rel: Path = PANEL_DIRECTORY,
    sp500_backfill_rel: Path = BACKFILL_DIRECTORY,
    wider_rel: Path = WIDER_UNIVERSE_DIRECTORY,
    ext_rel: Path = SEALED_EXTENSION_DIR,
) -> dict[str, Any]:
    """Resumably populate `sealed_holdout_extension_20240529_20260708/` without touching `<= 2024-05-28` files."""
    ext_dir = paths.write(paths.identity(ext_rel))
    ext_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = paths.directory(panel_rel)
    wider_dir = paths.directory(wider_rel)
    sp500_bf_dir = paths.directory(sp500_backfill_rel)

    # 1. CBOE daily volatility indices through 2026-07-08
    cboe_pq = ext_dir / vol_regime.CBOE_PARQUET_NAME
    if not cboe_pq.is_file():
        cboe_rows: list[pd.DataFrame] = []
        headers_cb = {"User-Agent": "Mozilla/5.0 market-predictor-research/0.1"}
        for col_name, url in vol_regime.CBOE_CSV_URLS.items():
            resp = requests.get(url, headers=headers_cb, timeout=30)
            resp.raise_for_status()
            cboe_rows.append(
                vol_regime.parse_cboe_csv_truncated(
                    resp.text,
                    series_name=col_name,
                    development_end=SEALED_HOLDOUT_END,
                    unseal_final_test_period=True,
                )
            )
        merged_cb: pd.DataFrame | None = None
        for sub_cb in cboe_rows:
            merged_cb = sub_cb if merged_cb is None else merged_cb.merge(sub_cb, on="session_date", how="outer")
        assert merged_cb is not None
        merged_cb = merged_cb.sort_values("session_date").reset_index(drop=True)
        cast(Any, pq).write_table(pa.Table.from_pandas(merged_cb, preserve_index=False), cboe_pq)

    # 2. Resumable SEC `8-K` + non-`8-K` (`2024-05-29..2026-07-08`) for active `wider_us_universe` CIKs
    roster_orig = cast(Any, pq).read_table(wider_dir / "roster_spans.parquet", use_threads=False).to_pandas()
    roster_orig["valid_from"] = pd.to_datetime(roster_orig["valid_from"]).dt.date
    roster_orig["valid_to"] = pd.to_datetime(roster_orig["valid_to"]).dt.date
    selected_ciks = sorted(json.loads((wider_dir / "selected_ciks.json").read_text(encoding="utf-8"))["selected_ciks"])

    membership_full = _read_panel_membership(panel_dir, SEALED_HOLDOUT_END, unseal_final_test_period=True)
    sp500_sec_ids = tuple(sorted(membership_full["security_id"].astype(str).unique()))
    sp500_cik_map, _ = resolve_all_sp500_ciks(
        sp500_sec_ids,
        identity_relations_path=paths.read(SEC_IDENTITY_RELATIONS),
        filings_path=paths.read(FILINGS_DIRECTORY / FILINGS_FILE),
        development_end=SEALED_HOLDOUT_END,
        unseal_final_test_period=True,
    )
    sp500_cik_set = set(sp500_cik_map.values())
    active_ciks_at_boundary = set(
        roster_orig.loc[roster_orig["valid_to"] >= OVERLAP_STITCH_DATE, "cik"].astype(str)
    )
    mid_active_ciks = sorted((set(selected_ciks) - sp500_cik_set) & active_ciks_at_boundary)

    sec_cache_jsonl = ext_dir / "sec_holdout_cache.jsonl"
    sec_cached: dict[str, dict[str, Any]] = {}
    if sec_cache_jsonl.is_file():
        for line in sec_cache_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                sec_cached[str(obj["cik"])] = obj

    todo_ciks = [c for c in mid_active_ciks if c not in sec_cached]
    if todo_ciks:
        sec = SecSource(settings)
        with sec_cache_jsonl.open("a", encoding="utf-8") as handle:
            for idx_c, cik in enumerate(todo_ciks):
                try:
                    item_sec = _extract_holdout_sec_events_for_cik(
                        sec, cik, start_day=SEALED_HOLDOUT_START, end_day=SEALED_HOLDOUT_END,
                    )
                except Exception as exc:
                    if "404" in str(exc):
                        item_sec = {"cik": cik, "last_filing_date": None, "events_8k": [], "events_non8k": []}
                    else:
                        raise
                sec_cached[cik] = item_sec
                handle.write(json.dumps(item_sec, sort_keys=True) + "\n")
                if (idx_c + 1) % 40 == 0:
                    handle.flush()
                    release_unused_process_memory()
                    guard("sealed holdout sec submissions")

    holdout_8k_pq = ext_dir / "mid_small_8k_20240529_20260708.parquet"
    holdout_non8k_pq = ext_dir / "mid_small_non8k_20240529_20260708.parquet"
    if not holdout_8k_pq.is_file() or not holdout_non8k_pq.is_file():
        rows_8k = [r for v in sec_cached.values() for r in v.get("events_8k", [])]
        rows_n8k = [r for v in sec_cached.values() for r in v.get("events_non8k", [])]
        df_8k = pd.DataFrame(rows_8k) if rows_8k else pd.DataFrame(
            columns=["security_id", "sec_cik", "accession_number", "sec_form", "item_codes", "accepted_at_utc", "available_at_utc"]
        )
        df_n8k = pd.DataFrame(rows_n8k) if rows_n8k else pd.DataFrame(
            columns=["security_id", "sec_cik", "accession_number", "sec_form", "catalyst_family", "available_at_utc"]
        )
        cast(Any, pq).write_table(pa.Table.from_pandas(df_8k, preserve_index=False), holdout_8k_pq)
        cast(Any, pq).write_table(pa.Table.from_pandas(df_n8k, preserve_index=False), holdout_non8k_pq)

    # 3. Build `extended_roster_spans.parquet` (`2024-05-29..2026-07-08` ticker-reuse & SEC-life checked)
    ext_roster_pq = ext_dir / "extended_roster_spans.parquet"
    if not ext_roster_pq.is_file():
        nc_cutoffs = _fetch_name_change_cutoffs(settings, SEALED_HOLDOUT_START, SEALED_HOLDOUT_END)
        new_spans: list[dict[str, Any]] = []
        for rec in roster_orig.to_dict(orient="records"):
            v_from: date = rec["valid_from"]
            v_to: date = rec["valid_to"]
            tk = str(rec["ticker"]).upper().strip()
            cik_s = str(rec["cik"]).zfill(10)
            if v_to >= OVERLAP_STITCH_DATE:
                target_end = SEALED_HOLDOUT_END
                if tk in nc_cutoffs and nc_cutoffs[tk] >= OVERLAP_STITCH_DATE:
                    target_end = min(target_end, nc_cutoffs[tk])
                if cik_s in sec_cached:
                    lf = sec_cached[cik_s].get("last_filing_date")
                    if isinstance(lf, str) and len(lf) == 10:
                        # Allow 120 calendar days after final SEC filing before marking delisted/merged
                        lf_cap = min(SEALED_HOLDOUT_END, date.fromisoformat(lf) + timedelta(days=120))
                        target_end = min(target_end, max(OVERLAP_STITCH_DATE, lf_cap))
                v_to = max(v_to, target_end)
            new_spans.append({**rec, "valid_from": v_from, "valid_to": v_to})
        ext_r_df = pd.DataFrame(new_spans)
        cast(Any, pq).write_table(pa.Table.from_pandas(ext_r_df, preserve_index=False), ext_roster_pq)
    else:
        ext_r_df = cast(Any, pq).read_table(ext_roster_pq, use_threads=False).to_pandas()
        ext_r_df["valid_from"] = pd.to_datetime(ext_r_df["valid_from"]).dt.date
        ext_r_df["valid_to"] = pd.to_datetime(ext_r_df["valid_to"]).dt.date

    # 4. Fetch `2024-05-28..2026-07-08` adjusted (`all`) & raw (`raw`) SIP daily bars (`2024-05-28` overlap for split-stitching!)
    bars_adj_pq = ext_dir / "bars_adj_20240528_20260708.parquet"
    bars_raw_pq = ext_dir / "bars_raw_20240528_20260708.parquet"
    sp500_holdout_tks = set(
        membership_full.loc[
            membership_full["session"] >= np.datetime64(OVERLAP_STITCH_DATE, "D"), "ticker"
        ].astype(str).str.upper()
    )
    wider_holdout_tks = set(
        ext_r_df.loc[
            ext_r_df["cik"].isin(selected_ciks) & (ext_r_df["valid_to"] >= SEALED_HOLDOUT_START), "ticker"
        ].astype(str).str.upper()
    )
    need_bar_syms = sorted((sp500_holdout_tks | wider_holdout_tks) - {"SPY", "QQQ"})

    alpaca = AlpacaSource(settings)
    if not bars_adj_pq.is_file():
        adj_df = _fetch_alpaca_daily_batch(
            alpaca,
            need_bar_syms,
            OVERLAP_STITCH_DATE,
            SEALED_HOLDOUT_END,
            adjustment="all",
            asof_day=SEALED_HOLDOUT_END,
            unseal_final_test_period=True,
        )
        cast(Any, pq).write_table(pa.Table.from_pandas(adj_df, preserve_index=False), bars_adj_pq)
        del adj_df
        release_unused_process_memory()

    if not bars_raw_pq.is_file():
        raw_df = _fetch_alpaca_daily_batch(
            alpaca,
            need_bar_syms,
            OVERLAP_STITCH_DATE,
            SEALED_HOLDOUT_END,
            adjustment="raw",
            asof_day=SEALED_HOLDOUT_END,
            unseal_final_test_period=True,
        )
        cast(Any, pq).write_table(
            pa.Table.from_pandas(raw_df[["ticker", "session_date", "close"]], preserve_index=False), bars_raw_pq,
        )
        del raw_df
        release_unused_process_memory()

    del membership_full, sp500_bf_dir
    return {
        "cboe_parquet": cboe_pq,
        "holdout_8k_parquet": holdout_8k_pq,
        "holdout_non8k_parquet": holdout_non8k_pq,
        "ext_roster_parquet": ext_roster_pq,
        "bars_adj_parquet": bars_adj_pq,
        "bars_raw_parquet": bars_raw_pq,
        "mid_active_ciks_queried": len(mid_active_ciks),
        "bar_symbols_queried": len(need_bar_syms),
    }


def _load_unsealed_sp500_and_mid_small(
    paths: EvidencePaths,
    ext_files: Mapping[str, Any],
    *,
    panel_rel: Path = PANEL_DIRECTORY,
    sp500_backfill_rel: Path = BACKFILL_DIRECTORY,
    wider_rel: Path = WIDER_UNIVERSE_DIRECTORY,
) -> tuple[
    free_backfill.ExtendedSP500Data,
    wider_universe.WiderUniverseData,
    pd.DataFrame,
    dict[str, str],
]:
    """Load `2016-01-04..2026-07-08` S&P 500 and `mid_small_ex_sp500` panels with exact `2024-05-28` overlap stitching."""
    panel_dir = paths.directory(panel_rel)

    bars_adj_ext = cast(Any, pq).read_table(ext_files["bars_adj_parquet"], use_threads=False).to_pandas()
    bars_adj_ext["session"] = pd.to_datetime(bars_adj_ext["session_date"]).to_numpy().astype("datetime64[D]")
    bars_raw_ext = cast(Any, pq).read_table(ext_files["bars_raw_parquet"], use_threads=False).to_pandas()
    bars_raw_ext["session"] = pd.to_datetime(bars_raw_ext["session_date"]).to_numpy().astype("datetime64[D]")

    # 1. Load S&P 500 through 2026-07-08, then overlay `2024-05-29..2026-07-08` missing bars, raw closes, and sealed 8-K 2.02!
    ext_sp = free_backfill.load_extended_sp500_data(
        paths,
        development_end=SEALED_HOLDOUT_END,
        panel_rel=panel_rel,
        backfill_rel=sp500_backfill_rel,
        unseal_final_test_period=True,
    )
    sp_panel = ext_sp.panel
    sp_sessions = sp_panel.sessions
    sp_col_idx = pd.Index(sp_panel.security_ids)
    stitch_np = np.datetime64(OVERLAP_STITCH_DATE, "D")
    stitch_row = int(np.searchsorted(sp_sessions, stitch_np))

    mem_full = _read_panel_membership(panel_dir, SEALED_HOLDOUT_END, unseal_final_test_period=True)
    key_sec = mem_full.groupby(["ticker", "session"])["security_id"].transform("nunique")
    dup_sec = mem_full.duplicated(["security_id", "session"], keep=False)
    usable_mem_holdout = mem_full.loc[
        key_sec.eq(1)
        & (~dup_sec)
        & mem_full["session"].isin(sp_sessions)
        & (mem_full["session"] >= np.datetime64(SEALED_HOLDOUT_START, "D")),
        ["ticker", "session", "security_id"],
    ]

    # Fill any missing adjusted S&P 500 cells in `2024-05-29..2026-07-08` (rescaling pre-2024-05-29 if ticker was only in missing_members)
    adj_hold = bars_adj_ext.loc[bars_adj_ext["session"] >= np.datetime64(SEALED_HOLDOUT_START, "D")].merge(
        usable_mem_holdout, on=["ticker", "session"], how="inner",
    )
    if len(adj_hold):
        rr_h = np.searchsorted(sp_sessions, adj_hold["session"].to_numpy().astype("datetime64[D]"))
        cc_h = sp_col_idx.get_indexer(adj_hold["security_id"].astype(str))
        miss_cells = (cc_h >= 0) & (~np.isfinite(sp_panel.close[rr_h, cc_h]))
        if bool(miss_cells.any()):
            # Rescale pre-holdout rows of those specific columns using 2024-05-28 overlap so no split jump occurs
            ov_map = dict(
                zip(
                    bars_adj_ext.loc[bars_adj_ext["session"] == stitch_np, "ticker"],
                    bars_adj_ext.loc[bars_adj_ext["session"] == stitch_np, "close"],
                    strict=False,
                )
            )
            mem_st = mem_full.loc[key_sec.eq(1) & (~dup_sec) & (mem_full["session"] == stitch_np), ["ticker", "security_id"]]
            tk_by_sid = dict(zip(mem_st["security_id"].astype(str), mem_st["ticker"].astype(str), strict=False))
            for col_fix in sorted(set(cc_h[miss_cells].tolist())):
                sid_f = sp_panel.security_ids[col_fix]
                tk_f = tk_by_sid.get(sid_f, "")
                old_px = float(sp_panel.close[stitch_row, col_fix])
                new_px = float(ov_map.get(tk_f, np.nan))
                if np.isfinite(old_px) and old_px > 0.0 and np.isfinite(new_px) and new_px > 0.0:
                    sc_f = new_px / old_px
                    if abs(sc_f - 1.0) > 1e-4:
                        for mat_px in (sp_panel.open, sp_panel.high, sp_panel.low, sp_panel.close):
                            mat_px[: stitch_row + 1, col_fix] = (mat_px[: stitch_row + 1, col_fix] * sc_f).astype(np.float32)
                        sp_panel.volume[: stitch_row + 1, col_fix] = (
                            sp_panel.volume[: stitch_row + 1, col_fix] / sc_f
                        ).astype(np.float32)
            rr_m = rr_h[miss_cells]
            cc_m = cc_h[miss_cells]
            sp_panel.open[rr_m, cc_m] = adj_hold["open"].to_numpy(dtype=np.float32)[miss_cells]
            sp_panel.high[rr_m, cc_m] = adj_hold["high"].to_numpy(dtype=np.float32)[miss_cells]
            sp_panel.low[rr_m, cc_m] = adj_hold["low"].to_numpy(dtype=np.float32)[miss_cells]
            sp_panel.close[rr_m, cc_m] = adj_hold["close"].to_numpy(dtype=np.float32)[miss_cells]
            sp_panel.volume[rr_m, cc_m] = adj_hold["volume"].to_numpy(dtype=np.float32)[miss_cells]

    # Fill `2024-05-29..2026-07-08` S&P 500 `unadjusted_close` (`market_value = unadjusted_close * shares`)
    sp_unadj = ext_sp.unadjusted_close.copy()
    raw_hold = bars_raw_ext.loc[bars_raw_ext["session"] >= np.datetime64(SEALED_HOLDOUT_START, "D")].merge(
        usable_mem_holdout, on=["ticker", "session"], how="inner",
    ).drop_duplicates(["security_id", "session"])
    if len(raw_hold):
        rr_r = np.searchsorted(sp_sessions, raw_hold["session"].to_numpy().astype("datetime64[D]"))
        cc_r = sp_col_idx.get_indexer(raw_hold["security_id"].astype(str))
        ok_r = (cc_r >= 0) & (~np.isfinite(sp_unadj[rr_r, cc_r]))
        sp_unadj[rr_r[ok_r], cc_r[ok_r]] = raw_hold["close"].to_numpy(dtype=np.float64)[ok_r]

    # Append retained `swing_later_sealed_sec_form_inventory_corrected_clock/filings.parquet` 8-K Item 2.02 releases
    sp_releases = ext_sp.releases
    try:
        sealed_sec_path = paths.read(LATER_SEALED_SEC_FILINGS)
    except Exception:
        sealed_sec_path = paths.project_root / LATER_SEALED_SEC_FILINGS
    extra_sha: dict[str, str] = {
        "cboe_20260708_sha256": file_sha256(Path(ext_files["cboe_parquet"])),
        "bars_adj_ext_sha256": file_sha256(Path(ext_files["bars_adj_parquet"])),
        "bars_raw_ext_sha256": file_sha256(Path(ext_files["bars_raw_parquet"])),
        "holdout_8k_ext_sha256": file_sha256(Path(ext_files["holdout_8k_parquet"])),
        "holdout_non8k_ext_sha256": file_sha256(Path(ext_files["holdout_non8k_parquet"])),
    }
    if sealed_sec_path.is_file():
        extra_sha["later_sealed_sec_filings_sha256"] = file_sha256(sealed_sec_path)
        cutoff_s = acceptance_cutoff(SEALED_HOLDOUT_END)
        s_tbl = cast(Any, pq).read_table(
            sealed_sec_path,
            columns=["sec_cik", "accession_number", "sec_form", "item_codes", "accepted_at_utc"],
            partitioning=None,
            filters=[("accepted_at_utc", "<", cutoff_s.to_pydatetime())],
            use_threads=False,
        ).to_pandas()
        m_202 = (
            s_tbl["sec_form"].eq("8-K")
            & s_tbl["item_codes"].fillna("").astype(str).map(
                lambda t: "2.02" in {c.strip() for c in t.split(",")}
            )
            & s_tbl["accepted_at_utc"].notna()
        )
        sub_202 = s_tbl.loc[m_202, ["sec_cik", "accession_number", "accepted_at_utc"]].copy()
        if len(sub_202):
            cik_to_sp_secs: dict[str, list[str]] = {}
            for s_id_sp, cik_sp in ext_sp.security_cik_map.items():
                cik_to_sp_secs.setdefault(str(cik_sp).zfill(10), []).append(s_id_sp)
            mapped_202_rows: list[dict[str, Any]] = []
            for cik_v, accn_v, acc_ts in zip(
                sub_202["sec_cik"].astype(str),
                sub_202["accession_number"].astype(str),
                pd.to_datetime(sub_202["accepted_at_utc"], utc=True),
                strict=False,
            ):
                for s_id_sp in cik_to_sp_secs.get(cik_v.strip().zfill(10), ()):
                    mapped_202_rows.append({
                        "security_id": s_id_sp,
                        "accession_number": accn_v,
                        "accepted_at_utc": acc_ts,
                        "available_at_utc": acc_ts,
                    })
            if mapped_202_rows:
                rel_s = pd.DataFrame(mapped_202_rows)
                sp_releases = pd.concat([sp_releases, rel_s], ignore_index=True).drop_duplicates(
                    ["security_id", "accession_number"]
                )

    ext_sp_full = free_backfill.ExtendedSP500Data(
        panel=sp_panel,
        unadjusted_close=sp_unadj,
        releases=sp_releases,
        security_cik_map=ext_sp.security_cik_map,
        company_facts_paths=ext_sp.company_facts_paths,
        input_sha256={**ext_sp.input_sha256, **extra_sha},
        report=ext_sp.report,
    )

    del bars_adj_ext, bars_raw_ext, mem_full
    release_unused_process_memory()
    wdata_full, all_non8k_mid = load_unsealed_mid_small_through_date(
        paths,
        ext_files,
        as_of_date=SEALED_HOLDOUT_END,
        panel_rel=panel_rel,
        sp500_backfill_rel=sp500_backfill_rel,
        wider_rel=wider_rel,
    )
    return ext_sp_full, wdata_full, all_non8k_mid, extra_sha


def load_unsealed_mid_small_through_date(
    paths: EvidencePaths,
    ext_files: Mapping[str, Any],
    *,
    as_of_date: date = SEALED_HOLDOUT_END,
    source_cutoff_utc: pd.Timestamp | None = None,
    panel_rel: Path = PANEL_DIRECTORY,
    sp500_backfill_rel: Path = BACKFILL_DIRECTORY,
    wider_rel: Path = WIDER_UNIVERSE_DIRECTORY,
) -> tuple[wider_universe.WiderUniverseData, pd.DataFrame]:
    """Load `mid_small_ex_sp500` across `2016-01-04..as_of_date` (`<= 2026-07-08`) with zero post-`as_of_date` reads."""
    if as_of_date > SEALED_HOLDOUT_END:
        raise ValueError(f"load_unsealed_mid_small_through_date requires as_of_date <= {SEALED_HOLDOUT_END.isoformat()}")
    end_iso = as_of_date.isoformat()
    panel_dir = paths.directory(panel_rel)
    sp500_bf_dir = paths.directory(sp500_backfill_rel)
    wider_dir = paths.directory(wider_rel)
    stitch_np = np.datetime64(OVERLAP_STITCH_DATE, "D")

    bars_adj_ext = cast(Any, pq).read_table(
        ext_files["bars_adj_parquet"],
        filters=[("session_date", "<=", end_iso)],
        use_threads=False,
    ).to_pandas()
    if len(bars_adj_ext):
        bars_adj_ext["session"] = pd.to_datetime(bars_adj_ext["session_date"]).to_numpy().astype("datetime64[D]")
    else:
        bars_adj_ext["session"] = np.empty(0, dtype="datetime64[D]")

    bars_raw_ext = cast(Any, pq).read_table(
        ext_files["bars_raw_parquet"],
        filters=[("session_date", "<=", end_iso)],
        use_threads=False,
    ).to_pandas()
    if len(bars_raw_ext):
        bars_raw_ext["session"] = pd.to_datetime(bars_raw_ext["session_date"]).to_numpy().astype("datetime64[D]")
    else:
        bars_raw_ext["session"] = np.empty(0, dtype="datetime64[D]")

    mem_full = _read_panel_membership(panel_dir, as_of_date, unseal_final_test_period=True)
    selected_ciks = tuple(sorted(json.loads((wider_dir / "selected_ciks.json").read_text(encoding="utf-8"))["selected_ciks"]))
    col_of_cik = {c: idx for idx, c in enumerate(selected_ciks)}
    ext_roster = cast(Any, pq).read_table(ext_files["ext_roster_parquet"], use_threads=False).to_pandas()
    ext_roster["valid_from"] = pd.to_datetime(ext_roster["valid_from"]).dt.date
    ext_roster["valid_to"] = pd.to_datetime(ext_roster["valid_to"]).dt.date

    sessions, spy_open, spy_close, qqq_open, qqq_close, warm_stitch, _ = _load_sessions_and_benchmarks(
        panel_dir, sp500_bf_dir, as_of_date,
    )
    matrices, in_span = wider_universe._stream_adjusted_bars_into_matrices(
        sessions=sessions,
        ciks=selected_ciks,
        roster_df=ext_roster,
        panel_dir=panel_dir,
        sp500_bf_dir=sp500_bf_dir,
        adj_dir=wider_dir / "bars_adjusted",
        warm_stitch=warm_stitch,
        development_end=as_of_date,
        fields=PRICE_FIELDS,
    )

    n_sess, n_sec = len(sessions), len(selected_ciks)
    st_idx_w = int(np.searchsorted(sessions, stitch_np))
    sub_rost = ext_roster.loc[ext_roster["cik"].isin(col_of_cik)]
    tk_ivs: dict[str, list[tuple[int, int, int]]] = {}
    for rec in sub_rost.to_dict(orient="records"):
        c_i = col_of_cik[str(rec["cik"])]
        r0 = int(np.searchsorted(sessions, np.datetime64(rec["valid_from"], "D"), side="left"))
        r1 = int(np.searchsorted(sessions, np.datetime64(rec["valid_to"], "D"), side="right"))
        if r0 < r1:
            tk_ivs.setdefault(str(rec["ticker"]), []).append((r0, r1, c_i))

    if len(bars_adj_ext):
        for tk_e, grp_e in bars_adj_ext.groupby("ticker", sort=False):
            ivs = tk_ivs.get(str(tk_e))
            if not ivs:
                continue
            ov_row = grp_e.loc[grp_e["session"] == stitch_np, "close"]
            new_ov = float(ov_row.iloc[0]) if len(ov_row) else np.nan
            post_e = grp_e.loc[grp_e["session"] >= np.datetime64(SEALED_HOLDOUT_START, "D")]
            if post_e.empty:
                continue
            b_s = post_e["session"].to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if not bool(ok.any()):
                continue
            rr_all = rc[ok]
            src_all = np.flatnonzero(ok)
            for r0, r1, col_i in ivs:
                wm = (rr_all >= r0) & (rr_all < r1) & (~np.isfinite(matrices["close"][rr_all, col_i]))
                if not bool(wm.any()):
                    continue
                old_ov = float(matrices["close"][st_idx_w, col_i]) if r0 <= st_idx_w < r1 else np.nan
                if np.isfinite(old_ov) and old_ov > 0.0 and np.isfinite(new_ov) and new_ov > 0.0:
                    sc_w = new_ov / old_ov
                    if abs(sc_w - 1.0) > 1e-4:
                        pre_stop = min(r1, st_idx_w + 1)
                        for f_px in ("open", "high", "low", "close"):
                            matrices[f_px][r0:pre_stop, col_i] = (
                                matrices[f_px][r0:pre_stop, col_i].astype(np.float64) * sc_w
                            ).astype(np.float32)
                        matrices["volume"][r0:pre_stop, col_i] = (
                            matrices["volume"][r0:pre_stop, col_i].astype(np.float64) / sc_w
                        ).astype(np.float32)
                rr_put = rr_all[wm]
                src_put = src_all[wm]
                for f_k in PRICE_FIELDS:
                    matrices[f_k][rr_put, col_i] = post_e[f_k].to_numpy(dtype=np.float32)[src_put]

    unadj_w = np.full((n_sess, n_sec), np.nan, dtype=np.float64)

    def _place_raw(df_r: pd.DataFrame) -> None:
        if df_r.empty:
            return
        if "session" not in df_r.columns:
            df_r = df_r.copy()
            df_r["session"] = pd.to_datetime(df_r["session_date"]).to_numpy().astype("datetime64[D]")
        for tk_r, grp_r in df_r.groupby("ticker", sort=False):
            ivs = tk_ivs.get(str(tk_r))
            if not ivs:
                continue
            b_s = grp_r["session"].to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if not bool(ok.any()):
                continue
            rr_a = rc[ok]
            cl_a = grp_r["close"].to_numpy(dtype=np.float64)[ok]
            for r0, r1, col_i in ivs:
                wm = (rr_a >= r0) & (rr_a < r1)
                if bool(wm.any()):
                    unadj_w[rr_a[wm], col_i] = cl_a[wm]

    sp500_sec_ids = tuple(sorted(mem_full["security_id"].astype(str).unique()))
    sp500_cik_all, _ = resolve_all_sp500_ciks(
        sp500_sec_ids,
        identity_relations_path=paths.read(SEC_IDENTITY_RELATIONS),
        filings_path=paths.read(FILINGS_DIRECTORY / FILINGS_FILE),
        development_end=as_of_date,
        unseal_final_test_period=True,
    )
    raw_dir_can = paths.directory(UNADJUSTED_DIRECTORY)
    for item in json.loads((raw_dir_can / "_manifest.json").read_text(encoding="utf-8"))["unit_artifacts"]:
        if item.get("role") != "stock":
            continue
        c_hit = sp500_cik_all.get(str(item["security_id"]))
        if c_hit is None or c_hit not in col_of_cik:
            continue
        c_col = col_of_cik[c_hit]
        u_df = cast(Any, pq).read_table(
            raw_dir_can / Path(str(item["bars_path"]).replace("\\", "/")),
            columns=["session_date", "close"],
            filters=[("session_date", "<=", end_iso)],
            use_threads=False,
        ).to_pandas()
        if not u_df.empty:
            b_s = pd.to_datetime(u_df["session_date"]).to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if bool(ok.any()):
                unadj_w[rc[ok], c_col] = u_df["close"].to_numpy(dtype=np.float64)[ok]

    for r_sh in [
        sp500_bf_dir / "bars" / "early_and_missing_raw_20180529_20240528.parquet",
        *sorted((wider_dir / "bars_raw").glob("raw_shard_*.parquet")),
    ]:
        if r_sh.is_file():
            _place_raw(
                cast(Any, pq).read_table(
                    r_sh,
                    columns=["ticker", "session_date", "close"],
                    filters=[("session_date", "<=", end_iso)],
                    use_threads=False,
                ).to_pandas()
            )
    _place_raw(bars_raw_ext)

    del bars_adj_ext, bars_raw_ext
    release_unused_process_memory()

    sp500_mask = np.zeros((n_sess, n_sec), dtype=bool)
    mem_known = mem_full.loc[
        (mem_full["membership_available_at_utc"] <= mem_full["decision_time_utc"]).fillna(False)
        & mem_full["session"].isin(sessions)
        & mem_full["security_id"].isin(sp500_cik_all)
    ]
    mr = np.searchsorted(sessions, mem_known["session"].to_numpy().astype("datetime64[D]"))
    mc_ciks = mem_known["security_id"].map(sp500_cik_all)
    valid_mc = mc_ciks.isin(col_of_cik).to_numpy()
    if bool(valid_mc.any()):
        mc_idx = np.asarray([col_of_cik[str(c)] for c in mc_ciks.loc[valid_mc]], dtype=np.int32)
        sp500_mask[mr[valid_mc], mc_idx] = True
    del mem_full, mem_known

    elig_top1200 = compute_pit_liquidity_eligibility(
        matrices["close"], matrices["volume"], unadj_w, in_span, sessions, top_n=TOP_LIQUID_COUNT,
    )
    eligible = elig_top1200 & (~sp500_mask)
    del elig_top1200, in_span

    keep_cols = np.flatnonzero(eligible.any(axis=0))
    kept_ciks = tuple(selected_ciks[int(i)] for i in keep_cols)
    sec_ids_w = tuple(f"cik:{c}" for c in kept_ciks)
    for f_k in PRICE_FIELDS:
        matrices[f_k] = np.ascontiguousarray(matrices[f_k][:, keep_cols])
    unadj_w = np.ascontiguousarray(unadj_w[:, keep_cols])
    eligible = np.ascontiguousarray(eligible[:, keep_cols])
    sp500_mask = np.ascontiguousarray(sp500_mask[:, keep_cols])
    release_unused_process_memory()

    cik_sectors = json.loads((wider_dir / "cik_sectors.json").read_text(encoding="utf-8"))
    sec_per_col = [str(cik_sectors.get(c, "Industrials")) for c in kept_ciks]
    sec_names_w = tuple(sorted(set(sec_per_col)))
    sec_codes_w = pd.Index(sec_names_w).get_indexer(sec_per_col).astype(np.int16)
    sector_mat_w = np.broadcast_to(sec_codes_w[None, :], (n_sess, len(kept_ciks))).copy()

    first_dec_row = int(np.searchsorted(sessions, np.datetime64(EARLY_DECISION_START, "D"), side="left"))
    mid_panel = ResearchPanel(
        sessions=sessions,
        security_ids=sec_ids_w,
        open=matrices["open"],
        high=matrices["high"],
        low=matrices["low"],
        close=matrices["close"],
        volume=matrices["volume"],
        eligible=eligible,
        sector=sector_mat_w,
        sector_names=sec_names_w,
        spy_open=spy_open,
        spy_close=spy_close,
        qqq_open=qqq_open,
        qqq_close=qqq_close,
        development_end=as_of_date,
        load_report={
            "subset": "mid_small_ex_sp500",
            "development_end": end_iso,
            "sessions": {"count": n_sess, "first": str(sessions[0]), "last": str(sessions[-1])},
            "panel_securities_total": len(kept_ciks),
            "eligible_cells_total": int(eligible.sum()),
            "mean_eligible_stocks_per_decision_session": float(eligible[first_dec_row:].sum(axis=1).mean()),
        },
        input_sha256={
            "combined_daily_manifest_sha256": file_sha256(panel_dir / BAR_MANIFEST),
            "extended_roster_spans_sha256": file_sha256(Path(ext_files["ext_roster_parquet"])),
        },
    )

    cutoff_ts = (
        source_cutoff_utc.tz_convert("UTC")
        if source_cutoff_utc is not None
        else acceptance_cutoff(as_of_date)
    )
    w_8k_pre = cast(Any, pq).read_table(wider_dir / "wider_8k_events.parquet", use_threads=False).to_pandas()
    w_8k_post = cast(Any, pq).read_table(ext_files["holdout_8k_parquet"], use_threads=False).to_pandas()
    for df_ev in (w_8k_pre, w_8k_post):
        if len(df_ev):
            df_ev["available_at_utc"] = pd.to_datetime(df_ev["available_at_utc"], utc=True)
            df_ev["accepted_at_utc"] = pd.to_datetime(df_ev["accepted_at_utc"], utc=True)
    all_8k_mid = pd.concat(
        [
            w_8k_pre[["security_id", "accession_number", "item_codes", "accepted_at_utc", "available_at_utc"]],
            w_8k_post[["security_id", "accession_number", "item_codes", "accepted_at_utc", "available_at_utc"]],
        ],
        ignore_index=True,
    ).drop_duplicates(["security_id", "accession_number"])
    if len(all_8k_mid):
        all_8k_mid = all_8k_mid.loc[all_8k_mid["available_at_utc"] <= cutoff_ts].copy()

    has_202_w = all_8k_mid["item_codes"].fillna("").astype(str).map(
        lambda s: "2.02" in {x.strip() for x in s.split(",")}
    )
    rel_202_mid = all_8k_mid.loc[
        has_202_w,
        ["security_id", "accession_number", "accepted_at_utc", "available_at_utc"],
    ].reset_index(drop=True)

    non8k_pre = cast(Any, pq).read_table(wider_dir / beta_neutral.NON8K_CATALYST_PARQUET, use_threads=False).to_pandas()
    non8k_post = cast(Any, pq).read_table(ext_files["holdout_non8k_parquet"], use_threads=False).to_pandas()
    all_non8k_mid = pd.concat([non8k_pre, non8k_post], ignore_index=True).drop_duplicates(["security_id", "accession_number"])
    if len(all_non8k_mid):
        all_non8k_mid["available_at_utc"] = pd.to_datetime(all_non8k_mid["available_at_utc"], utc=True)
        all_non8k_mid = all_non8k_mid.loc[all_non8k_mid["available_at_utc"] <= cutoff_ts].copy()

    extra_f: Path | None = None
    try:
        cand_extra = paths.directory(EXTRA_FACTS_DIR)
        if cand_extra.is_dir():
            extra_f = cand_extra
    except Exception:
        fallback_extra = paths.project_root / EXTRA_FACTS_DIR
        if fallback_extra.is_dir():
            extra_f = fallback_extra
    facts_paths_w = existing_company_facts_paths(
        set(kept_ciks),
        primary_sources=paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES),
        extra_sources=extra_f,
        backfill_sources=sp500_bf_dir / "company_facts",
    )
    w_facts_dir = wider_dir / "company_facts"
    for c_id in kept_ciks:
        pw = w_facts_dir / c_id / COMPANY_FACTS_FILE
        if c_id not in facts_paths_w and pw.is_file():
            facts_paths_w[c_id] = pw

    wdata_full = wider_universe.WiderUniverseData(
        panel=mid_panel,
        unadjusted_close=unadj_w,
        releases_202=rel_202_mid,
        all_8k_events=all_8k_mid,
        security_cik_map={f"cik:{c}": c for c in kept_ciks if c in facts_paths_w},
        company_facts_paths=facts_paths_w,
        sp500_member_mask=sp500_mask,
        input_sha256=mid_panel.input_sha256,
        report=mid_panel.load_report,
    )
    return wdata_full, all_non8k_mid


def build_holdout_folds(sessions: npt.NDArray[np.datetime64]) -> tuple[Fold, ...]:
    """Exact purged holdout folds starting strictly at `2024-05-29`: `2024H2`, `2025`, and `2026H1`."""
    r_hold = int(np.searchsorted(sessions, np.datetime64(SEALED_HOLDOUT_START, "D"), side="left"))
    r_2025 = int(np.searchsorted(sessions, np.datetime64("2025-01-01", "D"), side="left"))
    r_2026 = int(np.searchsorted(sessions, np.datetime64("2026-01-01", "D"), side="left"))
    n_s = len(sessions)
    folds: list[Fold] = []
    if r_hold < min(r_2025, n_s):
        folds.append(Fold(year=2024, test_start=r_hold, test_stop=min(r_2025, n_s)))
    if r_2025 < min(r_2026, n_s):
        folds.append(Fold(year=2025, test_start=r_2025, test_stop=min(r_2026, n_s)))
    if r_2026 < n_s:
        folds.append(Fold(year=2026, test_start=r_2026, test_stop=n_s))
    return tuple(folds)


def fit_stage2_xgb_on_holdout(
    panel: ResearchPanel,
    base_scores: FloatArray,
    *,
    targets_by_horizon: Mapping[int, FloatArray],
    catalyst_features: Mapping[str, Float32Array],
    catalyst_feature_names: tuple[str, ...],
    horizons: tuple[int, ...] = HORIZONS,
    refit_annually: bool = True,
) -> tuple[dict[int, Float32Array], dict[int, dict[str, Any]]]:
    """Predict Stage-2 catalyst residuals strictly on `2024-05-29..2026-07-08` with exact `t + h < test_start` purge."""
    news_start_row = int(np.searchsorted(panel.sessions, np.datetime64(NEWS_DECISION_START, "D"), side="left"))
    hold_start_row = int(np.searchsorted(panel.sessions, np.datetime64(SEALED_HOLDOUT_START, "D"), side="left"))
    folds = (
        build_holdout_folds(panel.sessions)
        if refit_annually
        else (Fold(year=2024, test_start=hold_start_row, test_stop=len(panel.sessions)),)
    )
    decision_rows_hold = np.arange(len(panel.sessions)) >= hold_start_row
    combined_by_h: dict[int, Float32Array] = {}
    paired_by_h: dict[int, dict[str, Any]] = {}
    yrs = session_years(panel.sessions)

    for horizon in horizons:
        labels = research_labels(panel, horizon)
        target = targets_by_horizon[horizon]
        residual = target - base_scores
        comb = np.full(panel.eligible.shape, np.nan, dtype=np.float64)

        for fold in folds:
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

        cells_pair = scored_cells(comb, labels.stock, panel.eligible, decision_rows_hold) & np.isfinite(base_scores)
        ic_base = daily_rank_correlation(base_scores, labels.stock, cells_pair)
        ic_comb = daily_rank_correlation(comb, labels.stock, cells_pair)
        delta_ic = ic_comb - ic_base
        nw_d = newey_west(delta_ic, horizon)
        by_yr: dict[str, float] = {}
        for y in sorted(np.unique(yrs[decision_rows_hold])):
            ym = (yrs == y) & np.isfinite(delta_ic)
            if bool(ym.any()):
                by_yr[str(int(y))] = float(np.mean(delta_ic[ym]))
        paired_by_h[horizon] = {
            "base_ic_mean": newey_west(ic_base, horizon)["mean"],
            "combined_ic_mean": newey_west(ic_comb, horizon)["mean"],
            "paired_delta_ic_mean": nw_d["mean"],
            "paired_delta_ic_nw_t": nw_d["t"],
            "by_year": by_yr,
        }
        combined_by_h[horizon] = comb.astype(np.float32)
        gc.collect()

    return combined_by_h, paired_by_h


def _render_holdout_markdown(doc: Mapping[str, Any]) -> str:
    vr = doc["holdout_vol_regime"]
    grid_s2 = vr["vol_regime_2021_2024"]
    grid_s1 = vr["vol_regime_2020_2024"]
    sat_st = vr["satellites_standalone_2021_2024"]
    frozen_cands = doc["frozen_20240528_no_refit_candidates"]
    years = ["2024", "2025", "2026"]

    lines = [
        "# Step 6: Unsealed Final-Test Holdout Evaluation (`2024-05-29` to `2026-07-08`) vs. `SPY`",
        "",
        f"Every figure below: {doc['notice']}.",
        "",
        "## Method & boundaries",
        "",
        f"- **Holdout window:** strictly `{doc['settings']['holdout_first_decision_session']}` "
        f"through `{doc['settings']['holdout_end']}` "
        f"(`{doc['regime_diagnostics']['holdout_sessions']}` sessions; "
        f"`{doc['regime_diagnostics']['holdout_backwardation_sessions']}` "
        f"backwardation sessions = `{_pct(doc['regime_diagnostics']['holdout_backwardation_share'])}`). "
        "Zero pre-`2024-05-29` windows included.",
        "- **Costs & execution:** `open[r_k + 1]` to `open[r_{k+1} + 1]` continuous windows, "
        "exact post-drift turnover charged at `20 bps` (`10 bps` informational).",
        "",
        "## Section A: Pre-Declared Primary & Secondary Candidates "
        "(Purged Annual Walk-Forward Refit vs. Frozen `2024-05-28` Weights)",
        "",
        "| Mode | Candidate | H | Turnover | Port Net 20bps | `SPY` | Net vs `SPY`/yr [NW t] | "
        "2024H2 | 2025 | 2026H1 | Pos Yrs | Sharpe (Port / `SPY`) | MaxDD (Port / `SPY`) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    focus = [
        ("walk_forward_refit", "90_10__sp500_tilt_news__R1_backwardation_to_spy", 10, grid_s2),
        ("frozen_20240528", "90_10__sp500_tilt_news__R1_backwardation_to_spy", 10, frozen_cands),
        ("walk_forward_refit", "90_10__spy__R1_backwardation_to_spy", 10, grid_s2),
        ("zero_fit_stage1", "90_10__sp500_tilt_eq24__R1_backwardation_to_spy", 10, grid_s1),
        ("walk_forward_refit", "80_20__spy__R2_adaptive_hedge_sec_xgb", 63, grid_s2),
        ("frozen_20240528", "80_20__spy__R2_adaptive_hedge_sec_xgb", 63, frozen_cands),
        ("walk_forward_refit", "80_20__spy__R2_adaptive_hedge_sec_xgb", 21, grid_s2),
        ("walk_forward_refit", "80_20__spy__R2_adaptive_hedge_sec_xgb", 10, grid_s2),
        ("walk_forward_refit", "80_20__spy__R1_backwardation_to_spy", 21, grid_s2),
        ("zero_fit_stage1", "80_20__spy__R2_adaptive_hedge_eq24", 63, grid_s1),
    ]
    for mode_lbl, spec_id, h_int, src_grid in focus:
        if spec_id not in src_grid or str(h_int) not in src_grid[spec_id]:
            continue
        e = src_grid[spec_id][str(h_int)]
        vs = e["vs_spy"]
        by_y = vs["by_year_net_20bps"]
        yr_cells = [_pct(by_y.get(y, {}).get("mean")) for y in years]
        lines.append(
            f"| {mode_lbl} | {spec_id} | {h_int} | {_pct(e['mean_steady_one_way_turnover'])} | "
            f"{_pct(e['compounded_net_20bps'])} | {_pct(vs['compounded_benchmark'])} | "
            f"{_pct(vs['net_excess_20bps_annualized'])} [{_num(vs['newey_west_20bps']['t'])}] | "
            + " | ".join(yr_cells)
            + f" | {vs['positive_years_20bps']}/{vs['years_scored']} | "
            f"{_num(e['risk_net_20bps']['sharpe_annualized'])} / {_num(e['risk_spy']['sharpe_annualized'])} | "
            f"{_pct(e['risk_net_20bps']['max_drawdown'])} / {_pct(e['risk_spy']['max_drawdown'])} |"
        )

    lines += [
        "",
        "## Section B: Standalone `mid_small_ex_sp500` Satellites & Stage-2 Catalyst Paired IC on Holdout (`2024-05-29..2026-07-08`)",
        "",
        "| Satellite (`100% mid/small`) | H | Turnover | Net 20bps | `SPY` | "
        "Net vs Matched Mid/Small/yr [NW t] | Net vs `SPY`/yr [NW t] | Sharpe | MaxDD |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s_name in ("R1_base_bn_sec_xgb", "R2_adaptive_hedge_sec_xgb"):
        if s_name not in sat_st:
            continue
        for h_k in ("5", "10", "21", "63"):
            if h_k not in sat_st[s_name]:
                continue
            e = sat_st[s_name][h_k]
            vm = e["vs_matched_benchmark"]
            vs = e["vs_spy"]
            lines.append(
                f"| {s_name} | {h_k} | {_pct(e['mean_steady_one_way_turnover'])} | "
                f"{_pct(e['compounded_net_20bps'])} | {_pct(vs['compounded_benchmark'])} | "
                f"{_pct(vm['net_excess_20bps_annualized'])} [{_num(vm['newey_west_20bps']['t'])}] | "
                f"{_pct(vs['net_excess_20bps_annualized'])} [{_num(vs['newey_west_20bps']['t'])}] | "
                f"{_num(e['risk_net_20bps']['sharpe_annualized'])} | {_pct(e['risk_net_20bps']['max_drawdown'])} |"
            )

    lines += [
        "",
        "### Stage-2 Catalyst Residual Paired Daily IC Improvement on Holdout (`2024-05-29..2026-07-08`)",
        "",
        "| Cohort | H | Base IC | Combined IC | Paired Delta IC [NW t] | 2024H2 | 2025 | 2026H1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for coh_key, p_map in doc["holdout_paired_catalyst_ic"].items():
        for h_k, pv in p_map.items():
            by_y = pv["by_year"]
            lines.append(
                f"| {coh_key} | {h_k} | {_num(pv['base_ic_mean'], 4)} | {_num(pv['combined_ic_mean'], 4)} | "
                f"{_num(pv['paired_delta_ic_mean'], 4)} [{_num(pv['paired_delta_ic_nw_t'])}] | "
                f"{_num(by_y.get('2024'), 4)} | {_num(by_y.get('2025'), 4)} | {_num(by_y.get('2026'), 4)} |"
            )

    lines += [
        "",
        "## Section C: Full Volatility-Regime Core-Satellite Grid on Holdout (`2024-05-29..2026-07-08`)",
        "",
        "| Specification | H | Turnover | Port Net 20bps | `SPY` | Net vs `SPY`/yr [NW t] | "
        "2024H2 | 2025 | 2026H1 | Pos Yrs | Sharpe | MaxDD |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for spec_id, h_map in grid_s2.items():
        for h_k, e in h_map.items():
            vs = e["vs_spy"]
            by_y = vs["by_year_net_20bps"]
            yr_cells = [_pct(by_y.get(y, {}).get("mean")) for y in years]
            lines.append(
                f"| {spec_id} | {h_k} | {_pct(e['mean_steady_one_way_turnover'])} | "
                f"{_pct(e['compounded_net_20bps'])} | {_pct(vs['compounded_benchmark'])} | "
                f"{_pct(vs['net_excess_20bps_annualized'])} [{_num(vs['newey_west_20bps']['t'])}] | "
                + " | ".join(yr_cells)
                + f" | {vs['positive_years_20bps']}/{vs['years_scored']} | "
                f"{_num(e['risk_net_20bps']['sharpe_annualized'])} | {_pct(e['risk_net_20bps']['max_drawdown'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def evaluate_sealed_holdout_2024_2026(
    *,
    root: Path,
    output: Path | None = None,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    sp500_backfill_dir: Path = BACKFILL_DIRECTORY,
    wider_dir: Path = WIDER_UNIVERSE_DIRECTORY,
    ext_dir: Path = SEALED_EXTENSION_DIR,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    horizons: tuple[int, ...] = HORIZONS,
    unseal_final_test_period: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Execute the one-time governed `2024-05-29..2026-07-08` holdout test (`unseal_final_test_period=True` required)."""
    if not unseal_final_test_period:
        raise SealedPeriodError(
            "evaluate-sealed-holdout-2024-2026 requires explicit --unseal-final-test-period authorization"
        )

    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    out_rel = output or (OUTPUT_PARENT / f"sealed_holdout_20240529_20260708_{stamp}")
    identity = paths.identity(out_rel)

    with heavy_job_lease(EVALUATE_SEALED_HOLDOUT_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("sealed holdout preparation")
        destination = paths.write(identity)
        if destination.exists():
            raise FileExistsError(f"holdout output already exists: {destination}")

        cfg = settings or get_settings()
        ext_files = collect_sealed_holdout_extension(
            paths, cfg, panel_rel=panel, sp500_backfill_rel=sp500_backfill_dir, wider_rel=wider_dir, ext_rel=ext_dir,
        )
        guard("sealed holdout extension inputs ready")

        ext_sp, wdata, all_non8k_mid, extra_sha = _load_unsealed_sp500_and_mid_small(
            paths, ext_files, panel_rel=panel, sp500_backfill_rel=sp500_backfill_dir, wider_rel=wider_dir,
        )
        alignment_report = core_satellite.verify_disjoint_core_satellite_universes(
            ext_sp.panel, ext_sp.security_cik_map, wdata.panel, wdata.security_cik_map,
        )
        guard("sealed holdout panels loaded and disjointness verified")

        # Build S&P 500 Stage-1 + Stage-2 (`2016..2026-07-08`)
        sp_facts = ext_sp.company_facts_paths

        def _load_sp_f(cik: str) -> IssuerFacts | None:
            p = sp_facts.get(cik)
            return read_company_facts(p, SEALED_HOLDOUT_END)[0] if (p is not None and p.is_file()) else None

        sp_bundle = two_stage.build_stage1_features(
            ext_sp.panel,
            fundamentals.FundamentalInputs(
                panel=ext_sp.panel,
                unadjusted_close=ext_sp.unadjusted_close,
                releases=ext_sp.releases,
                security_issuer=ext_sp.security_cik_map,
                load_issuer=_load_sp_f,
            ),
        )
        sp500_panel = ext_sp.panel
        sp500_mv = sp_bundle.market_value.copy()
        sp500_eq24 = sp_bundle.equal_rank_24.copy()
        sp_sha = dict(ext_sp.input_sha256)
        del sp_bundle, ext_sp
        release_unused_process_memory()

        cboe_df = cast(Any, pq).read_table(ext_files["cboe_parquet"], use_threads=False).to_pandas()
        vol_feats = vol_regime.build_causal_vol_regime_features(
            sp500_panel.sessions,
            sp500_panel.spy_close,
            cboe_df,
            development_end=SEALED_HOLDOUT_END,
            unseal_final_test_period=True,
        )
        del cboe_df

        sp_news_feats, sp_news_sha, _ = two_stage.load_news_reaction_features(
            sp500_panel,
            panel_root=paths.directory(panel),
            early_news_dir=paths.directory(early_news_dir),
            later_news_dir=paths.directory(later_news_dir),
            unseal_final_test_period=True,
        )
        sp_targets = {h: target_ranks(research_labels(sp500_panel, h).stock, sp500_panel.eligible) for h in horizons}
        sp_news_wf_by_h, sp_paired_ic = fit_stage2_xgb_on_holdout(
            sp500_panel,
            sp500_eq24,
            targets_by_horizon=sp_targets,
            catalyst_features=sp_news_feats,
            catalyst_feature_names=two_stage.NEWS_FEATURE_NAMES,
            horizons=horizons,
            refit_annually=True,
        )
        sp_news_frz_10, _ = fit_stage2_xgb_on_holdout(
            sp500_panel,
            sp500_eq24,
            targets_by_horizon=sp_targets,
            catalyst_features=sp_news_feats,
            catalyst_feature_names=two_stage.NEWS_FEATURE_NAMES,
            horizons=(10,),
            refit_annually=False,
        )
        del sp_news_feats, sp_targets
        release_unused_process_memory()
        guard("sealed holdout sp500 stage2 fitted")

        # Build `mid_small_ex_sp500` Stage-1 + Stage-2 (`2016..2026-07-08`)
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
        w_sha = dict(wdata.input_sha256)
        del wdata

        mid_raw_eq24 = mid_bundle.equal_rank_24.copy()
        beta_rk = mid_bundle.features["px_beta_252d"].copy()
        ivol_rk = mid_bundle.features["px_idio_vol_60d"]
        rev5_rk = mid_bundle.features["px_reversal_5d"].copy()
        mid_bn_eq24 = beta_neutral.cross_sectional_neutralize(
            mid_raw_eq24, mid_panel.eligible, mid_panel.sector, beta_rk, ivol_rk, rerank=True,
        )
        mid_raw_targets = {h: target_ranks(research_labels(mid_panel, h).stock, mid_panel.eligible) for h in horizons}
        mid_bn_targets = {
            h: beta_neutral.cross_sectional_neutralize(
                mid_raw_targets[h], mid_panel.eligible, mid_panel.sector, beta_rk, ivol_rk, rerank=True,
            )
            for h in horizons
        }
        del mid_bundle, beta_rk, ivol_rk
        release_unused_process_memory()

        sec_cat_feats, sec_cat_rep = beta_neutral.build_precise_sec_catalyst_features(
            mid_panel, all_8k_mid, all_non8k_mid, unseal_final_test_period=True,
        )
        del all_8k_mid, all_non8k_mid
        release_unused_process_memory()

        mid_raw_sec_wf, mid_raw_paired_ic = fit_stage2_xgb_on_holdout(
            mid_panel,
            mid_raw_eq24,
            targets_by_horizon=mid_raw_targets,
            catalyst_features=sec_cat_feats,
            catalyst_feature_names=beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
            horizons=horizons,
            refit_annually=True,
        )
        mid_bn_sec_wf, mid_bn_paired_ic = fit_stage2_xgb_on_holdout(
            mid_panel,
            mid_bn_eq24,
            targets_by_horizon=mid_bn_targets,
            catalyst_features=sec_cat_feats,
            catalyst_feature_names=beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
            horizons=horizons,
            refit_annually=True,
        )
        # Also fit strictly frozen `2024-05-28` weights (`refit_annually=False`) for `10d` & `63d` primary checks!
        mid_raw_sec_frz, _ = fit_stage2_xgb_on_holdout(
            mid_panel,
            mid_raw_eq24,
            targets_by_horizon=mid_raw_targets,
            catalyst_features=sec_cat_feats,
            catalyst_feature_names=beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
            horizons=(10, 63),
            refit_annually=False,
        )
        mid_bn_sec_frz, _ = fit_stage2_xgb_on_holdout(
            mid_panel,
            mid_bn_eq24,
            targets_by_horizon=mid_bn_targets,
            catalyst_features=sec_cat_feats,
            catalyst_feature_names=beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
            horizons=(10, 63),
            refit_annually=False,
        )
        del sec_cat_feats, mid_raw_targets, mid_bn_targets
        release_unused_process_memory()
        guard("sealed holdout mid_small stage2 fitted")

        holdout_start_row = int(np.searchsorted(sp500_panel.sessions, np.datetime64(SEALED_HOLDOUT_START, "D"), side="left"))

        holdout_eval = vol_regime.evaluate_vol_regime_core_satellite(
            sp500_panel=sp500_panel,
            sp500_mv=sp500_mv,
            sp500_eq24=sp500_eq24,
            sp500_eq24_news_by_h=sp_news_wf_by_h,
            mid_panel=mid_panel,
            mid_raw_eq24=mid_raw_eq24,
            mid_bn_eq24=mid_bn_eq24,
            mid_raw_sec_by_h=mid_raw_sec_wf,
            mid_bn_sec_by_h=mid_bn_sec_wf,
            mid_rev5_rank=rev5_rk,
            vol_feats=vol_feats,
            stage1_oos_start_row=holdout_start_row,
            stage2_oos_start_row=holdout_start_row,
            horizons=horizons,
        )

        # Evaluate zero-refit (`frozen at 2024-05-28`) versions of the two primary candidates (`10d` R1 & `63d` R2)
        frozen_cands: dict[str, dict[str, Any]] = {}
        is_back = vol_feats["is_backwardation"]
        # 1. `90_10__sp500_tilt_news__R1_backwardation_to_spy` at 10d (frozen 2024-05-28 weights)
        sp_win10 = benchmark_tilt.continuous_rebalance_windows(sp500_panel, holdout_start_row, 10)
        mid_win10 = benchmark_tilt.continuous_rebalance_windows(mid_panel, holdout_start_row, 10)
        sp_frz10_dec, _ = benchmark_tilt.build_cap_tilt_weights(
            sp500_panel, sp_news_frz_10[10].astype(np.float64), sp500_mv, sp_win10.decision_rows, mode="sector_active_slice",
        )
        sp_frz10_w = benchmark_tilt._apply_open_fills(sp_frz10_dec, sp_win10.tradable)
        ws_frz10_dec, _ = benchmark_tilt.build_skill_weights(
            mid_panel, mid_bn_sec_frz[10].astype(np.float64), mid_win10.decision_rows, buffered=True,
        )
        ws_frz10 = benchmark_tilt._apply_open_fills(ws_frz10_dec, mid_win10.tradable)
        frozen_cands.setdefault("90_10__sp500_tilt_news__R1_backwardation_to_spy", {})["10"] = (
            vol_regime._simulate_r1_backwardation_to_spy(
                core_mode="core_sp500_tilt",
                core_weight=0.90,
                sp_tilt_weights=sp_frz10_w,
                mid_sat_weights=ws_frz10,
                is_backwardation_at_decisions=(is_back[sp_win10.decision_rows] > 0.5),
                sp_windows=sp_win10,
                mid_windows=mid_win10,
                horizon=10,
            )
        )
        # 2. `80_20__spy__R2_adaptive_hedge_sec_xgb` at 63d (frozen 2024-05-28 weights)
        sp_win63 = benchmark_tilt.continuous_rebalance_windows(sp500_panel, holdout_start_row, 63)
        mid_win63 = benchmark_tilt.continuous_rebalance_windows(mid_panel, holdout_start_row, 63)
        r2_frz63 = vol_regime.build_r2_adaptive_hedge_signal(
            mid_panel,
            raw_score=mid_raw_sec_frz[63].astype(np.float64),
            beta_neutral_score=mid_bn_sec_frz[63].astype(np.float64),
            reversal_5d_rank=rev5_rk.astype(np.float64),
            is_backwardation=is_back,
        )
        ws_frz63_dec, _ = benchmark_tilt.build_skill_weights(mid_panel, r2_frz63, mid_win63.decision_rows, buffered=True)
        ws_frz63 = benchmark_tilt._apply_open_fills(ws_frz63_dec, mid_win63.tradable)
        frozen_cands.setdefault("80_20__spy__R2_adaptive_hedge_sec_xgb", {})["63"] = (
            core_satellite.simulate_joint_core_satellite(
                core_mode="core_spy",
                core_weight=0.80,
                sp_tilt_weights=None,
                mid_sat_weights=ws_frz63,
                sp_windows=sp_win63,
                mid_windows=mid_win63,
                horizon=63,
            )
        )

        hold_back = is_back[holdout_start_row:] > 0.5
        regime_diag = {
            "holdout_sessions": int(len(sp500_panel.sessions) - holdout_start_row),
            "holdout_backwardation_sessions": int(hold_back.sum()),
            "holdout_backwardation_share": float(hold_back.mean()),
            "mean_holdout_vix_term_30_90": float(np.nanmean(vol_feats["vix_term_30_90"][holdout_start_row:])),
        }
        mem_snap = process_memory_snapshot()
        mem_gib = {"current_gib": mem_snap[0] / 2**30, "peak_gib": mem_snap[1] / 2**30} if mem_snap else None

        all_sha = {**sp_sha, **sp_news_sha, **w_sha, **extra_sha}
        settings_doc: dict[str, Any] = {
            "holdout_first_decision_session": str(sp500_panel.sessions[holdout_start_row]),
            "holdout_end": SEALED_HOLDOUT_END.isoformat(),
            "unseal_final_test_period": True,
            "horizons_sessions": list(horizons),
            "cost_per_one_way_turnover_primary": benchmark_tilt.FROZEN_COST_PER_TURNOVER,
            "label": "research label on provider split- and dividend-adjusted SIP daily bars",
        }
        document = _json_safe({
            "notice": (
                f"research label, adjusted prices, unsealed final-test holdout "
                f"{settings_doc['holdout_first_decision_session']} to {settings_doc['holdout_end']}"
            ),
            "settings": settings_doc,
            "inputs": all_sha,
            "extension_summary": {
                "mid_active_ciks_queried": ext_files["mid_active_ciks_queried"],
                "bar_symbols_queried": ext_files["bar_symbols_queried"],
                "mid_sec_catalyst_report": sec_cat_rep,
            },
            "universe_alignment": alignment_report,
            "regime_diagnostics": regime_diag,
            "holdout_paired_catalyst_ic": {
                "sp500_alpaca_news": {str(k): v for k, v in sp_paired_ic.items()},
                "mid_small_raw_sec": {str(k): v for k, v in mid_raw_paired_ic.items()},
                "mid_small_beta_neutral_sec": {str(k): v for k, v in mid_bn_paired_ic.items()},
            },
            "frozen_20240528_no_refit_candidates": frozen_cands,
            "holdout_vol_regime": holdout_eval,
            "process_memory_after_scoring": mem_gib,
        })

        staging = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        staging.mkdir(parents=True)
        try:
            (staging / "scoreboard.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
            )
            (staging / "scoreboard.md").write_text(_render_holdout_markdown(document), encoding="utf-8")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        append_trial(
            destination.parent / TRIAL_LOG_NAME,
            root=root,
            command=EVALUATE_SEALED_HOLDOUT_COMMAND,
            output=identity.relative_to(root),
            input_sha256=all_sha,
            settings=settings_doc,
            scoreboard=document,
        )

    return {
        "output": identity.relative_to(root).as_posix(),
        "universe_alignment": alignment_report,
        "regime_diagnostics": regime_diag,
        "process_memory_after_scoring": mem_gib,
    }


__all__ = [
    "EVALUATE_SEALED_HOLDOUT_COMMAND",
    "SEALED_EXTENSION_DIR",
    "SEALED_HOLDOUT_END",
    "SEALED_HOLDOUT_START",
    "build_holdout_folds",
    "evaluate_sealed_holdout_2024_2026",
    "fit_stage2_xgb_on_holdout",
    "load_unsealed_mid_small_through_date",
]
