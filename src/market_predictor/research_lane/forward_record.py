"""Append-only prospective forward score & live multi-source catalyst recorder (§5.6 & Step 5d).

Publishes one dated cross-sectional score snapshot (`score_price_equal_12`,
`score_beta_neutral_price_12`, `score_reversal_short`) for a single decision session
`as_of_date` without reading any future outcomes, allowing prospective forward evidence to
mature over calendar time.

Historical sealed dates (`2024-05-29` through `2026-07-08`) are strictly refused unless
explicitly unsealed after freezing one candidate. Live prospective dates (`>= 2026-07-09`)
fetch trailing daily SIP bars from Alpaca (`[max(2026-07-09, as_of_date - 410d), as_of_date]`)
and — when `include_live_catalysts=True` — capture point-in-time live catalysts strictly
inside `[max(2026-07-09, close_utc - 30d), close_utc(as_of_date)]` from Finviz Elite
(`news_export.ashx`), Alpaca (`fiscal.ai` `/v1beta1/news`), and SEC EDGAR (`8-K`, `SC 13D/A`,
`S-1/S-3/424B`, `NT 10-K/Q`), classified deterministically via `classify_event_families`.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import time as time_mod
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from market_predictor.catalysts.issuer_events.classification import classify_event_families
from market_predictor.config import Settings, get_settings
from market_predictor.core.errors import ArtifactIntegrityError, DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research_lane import (
    benchmark_tilt,
    indicators,
    live_cohort,
    opra_options,
    predictors,
    reversal,
    sec_catalyst_model,
)
from market_predictor.research_lane.beta_neutral import (
    _ACTIVIST_FORMS,
    _DISTRESS_8K_ITEMS,
    _LATE_FILING_FORMS,
    _OFFERING_FORMS,
    cross_sectional_neutralize,
)
from market_predictor.research_lane.free_backfill import _conservative_edgar_availability_utc
from market_predictor.research_lane.panel import (
    MARKET_TIMEZONE,
    PANEL_DIRECTORY,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
)
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.prospective_contracts import (
    FROZEN_PROSPECTIVE_CANDIDATE_SPECS,
    PERSISTED_SEC_MODEL_MANIFEST_SHA256,
    SNAPSHOT_MANIFEST_SCHEMA,
    ProspectiveCandidateSpec,
    RecordMode,
    build_payload_artifacts_manifest,
    extract_snapshot_opra_regime,
    latest_settled_xnys_session,
    resolve_xnys_session_clock,
    verify_snapshot_payload_hashes,
)
from market_predictor.research_lane.sec_filings import event_rows
from market_predictor.research_lane.trial_log import git_identity
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.sec import SecSource

FORWARD_RECORD_DIRECTORY: Final = Path("data/research/research_lane/forward_records")
FORWARD_LOG_NAME: Final = "forward_log.jsonl"
HISTORICAL_SEALED_END: Final = date(2026, 7, 8)
POST_CLOSE_SETTLEMENT_ET: Final = time(16, 20)
RECORD_FORWARD_COMMAND: Final = "record-forward-research-scores"
FINVIZ_ELITE_NEWS_URL: Final = "https://elite.finviz.com/news_export.ashx"
SEC_COMPANY_TICKERS_URL: Final = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_CATALYST_TOP_K: Final = 100
FROZEN_PROSPECTIVE_CANDIDATES: Final[tuple[dict[str, Any], ...]] = tuple(
    spec.to_dict() for spec in FROZEN_PROSPECTIVE_CANDIDATE_SPECS
)


def redact_finviz_auth(value: object, token: str | None = None) -> str:
    """Scrub `auth=` query tokens and explicit secret strings from transport errors or URLs."""
    text = re.sub(r"(?i)(auth=)[^&\s\"']+", r"\1<REDACTED>", str(value))
    if token:
        text = text.replace(token, "<REDACTED>")
    return text


def _default_ticker_for_security_id(sec_id: str, ticker_map: Mapping[str, str] | None = None) -> str:
    if ticker_map is not None and sec_id in ticker_map:
        return str(ticker_map[sec_id]).upper().strip()
    if sec_id.startswith("ticker:"):
        return sec_id.split(":", 1)[1].upper().strip()
    return sec_id.split(":", 1)[1].upper().strip() if ":" in sec_id else sec_id.upper().strip()


def _score_panel_last_session(
    panel: ResearchPanel,
    *,
    active_ticker_by_security_id: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Compute cross-sectional scores strictly at the final session row `-1` of `panel`."""
    if len(panel.sessions) == 0:
        raise ValueError("cannot score an empty panel")
    last_idx = len(panel.sessions) - 1
    session_str = str(panel.sessions[last_idx])
    elig_row = panel.eligible[last_idx]

    price_raw = predictors.compute_predictors(panel)
    ranked_last: dict[str, npt.NDArray[np.float64]] = {}
    signed_ranks: list[npt.NDArray[np.float64]] = []
    for name in predictors.PREDICTOR_NAMES:
        rk_row = centered_rank(price_raw[name], panel.eligible)[last_idx]
        ranked_last[name] = rk_row
        signed_ranks.append(predictors.EXPECTED_SIGNS[name] * rk_row)

    stack = np.stack(signed_ranks)
    fin = np.isfinite(stack)
    cnt = fin.sum(axis=0)
    avail_preds = int(sum(1 for rk_row in ranked_last.values() if np.isfinite(rk_row[elig_row]).any()))
    min_req = max(4, predictors.minimum_present(avail_preds)) if avail_preds > 0 else 8
    with np.errstate(invalid="ignore", divide="ignore"):
        price_eq_12 = np.where(
            elig_row & (cnt >= min_req),
            np.where(fin, stack, 0.0).sum(axis=0) / np.maximum(cnt, 1),
            np.nan,
        )

    beta_rk = ranked_last["beta_252d"]
    has_beta_252 = bool(np.isfinite(beta_rk[elig_row]).any())
    beta_neutral_12 = (
        cross_sectional_neutralize(
            price_eq_12[None, :],
            elig_row[None, :],
            panel.sector[last_idx : last_idx + 1],
            beta_rk[None, :],
            ranked_last["idio_vol_60d"][None, :],
            rerank=True,
        )[0]
        if has_beta_252
        else np.full(len(panel.security_ids), np.nan, dtype=np.float64)
    )

    ind = indicators.compute_indicators(panel)
    rev_short = reversal.reversal_signals(ind, panel.eligible)[reversal.REVERSAL_COMPOSITE][last_idx]

    rows: list[dict[str, Any]] = []
    for col, sec_id in enumerate(panel.security_ids):
        if not bool(elig_row[col]):
            continue
        sec_code = int(panel.sector[last_idx, col])
        sec_name = panel.sector_names[sec_code] if 0 <= sec_code < len(panel.sector_names) else ""
        rows.append({
            "session_date": session_str,
            "security_id": sec_id,
            "ticker": _default_ticker_for_security_id(sec_id, active_ticker_by_security_id),
            "sector": sec_name,
            "close_adjusted": float(panel.close[last_idx, col]),
            "score_price_equal_12": None if not np.isfinite(price_eq_12[col]) else float(price_eq_12[col]),
            "score_beta_neutral_price_12": None if not np.isfinite(beta_neutral_12[col]) else float(beta_neutral_12[col]),
            "score_reversal_short": None if not np.isfinite(rev_short[col]) else float(rev_short[col]),
        })
    return pd.DataFrame(rows)


def _fetch_live_prospective_panel(
    as_of_date: date,
    settings: Settings,
    *,
    paths: EvidencePaths | None = None,
    root: Path | None = None,
    unseal_final_test_period: bool = False,
) -> live_cohort.DecisionDateMidSmallCohort:
    """Build the verified live `mid_small_ex_sp500` cohort via `EvidencePaths` & `roster_spans.parquet`."""
    resolved_paths = paths if paths is not None else EvidencePaths(root or Path("."))
    clk = resolve_xnys_session_clock(as_of_date)
    return live_cohort.load_decision_date_mid_small_cohort(
        resolved_paths,
        as_of_date=as_of_date,
        source_cutoff_utc=clk.source_cutoff_utc,
        live_prospective=True,
        unseal_final_test_period=unseal_final_test_period,
        settings=settings,
    )


def parse_finviz_elite_news_csv(
    csv_text: str,
    *,
    ticker: str,
    security_id: str | None = None,
    window_start_utc: datetime,
    cutoff_utc: datetime,
    unseal_final_test_period: bool = False,
) -> list[dict[str, Any]]:
    """Parse Finviz Elite `news_export.ashx` (`v=3`) CSV rows strictly inside `[window_start_utc, cutoff_utc]`."""
    sealed_end_utc = datetime(2026, 7, 8, 23, 59, 59, tzinfo=UTC)
    if not unseal_final_test_period and window_start_utc <= sealed_end_utc:
        raise SealedPeriodError("Finviz Elite live window cannot overlap sealed period <= 2026-07-08")

    eastern = ZoneInfo(MARKET_TIMEZONE)
    symbol = ticker.upper().strip()
    resolved_sid = security_id or f"ticker:{symbol}"
    reader = csv.DictReader(io.StringIO(csv_text))
    items: list[dict[str, Any]] = []
    for row in reader:
        title = str(row.get("Title") or "").strip()
        dt_raw = str(row.get("Date") or "").strip()
        if not title or not dt_raw:
            continue
        parsed_ny: datetime | None = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed_ny = datetime.strptime(dt_raw, fmt).replace(tzinfo=eastern)
                break
            except ValueError:
                continue
        if parsed_ny is None:
            continue
        pub_utc = parsed_ny.astimezone(UTC)
        if not unseal_final_test_period and pub_utc <= sealed_end_utc:
            continue
        if pub_utc < window_start_utc or pub_utc > cutoff_utc:
            continue
        url = str(row.get("Url") or "").strip()
        category = str(row.get("Category") or "").strip()
        provider = str(row.get("Source") or "").strip()
        ev_hash = hashlib.sha256(f"finviz|{symbol}|{pub_utc.isoformat()}|{title}|{url}".encode()).hexdigest()[:20]
        items.append({
            "event_id": f"finviz:{ev_hash}",
            "security_id": resolved_sid,
            "ticker": symbol,
            "source_family": "finviz",
            "provider_source": provider,
            "provider_category": category,
            "sec_form": "",
            "structured_catalyst": "",
            "feature_available_at_utc": pub_utc.isoformat(),
            "title": title,
            "url": url,
        })
    return items


def _extract_live_sec_catalysts_and_company(
    payload: dict[str, Any],
    *,
    ticker: str,
    cik: str,
    security_id: str | None = None,
    fallback_company: str,
    window_start_utc: datetime,
    cutoff_utc: datetime,
) -> tuple[list[dict[str, Any]], str, str | None]:
    """Extract live SEC catalyst filings in `[window_start_utc, cutoff_utc]` plus causal pre-window company name."""
    company_name = str(payload.get("name") or fallback_company or "").strip()
    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    f_dates = recent.get("filingDate", [])
    accepts = recent.get("acceptanceDateTime", [])
    accs = recent.get("accessionNumber", [])
    items_list = recent.get("items", [])
    descs = recent.get("primaryDocDescription", [])

    earliest_pre_utc: pd.Timestamp | None = None
    out: list[dict[str, Any]] = []
    n = min(len(forms), len(f_dates), len(accs))
    for i in range(n):
        f_str = str(f_dates[i] or "").strip()
        if not f_str:
            continue
        acc_dt_str = str(accepts[i] or "") if i < len(accepts) else ""
        avail_ts = _conservative_edgar_availability_utc(acc_dt_str, f_str)
        if avail_ts is None:
            continue
        if avail_ts < pd.Timestamp(window_start_utc):
            if earliest_pre_utc is None or avail_ts < earliest_pre_utc:
                earliest_pre_utc = avail_ts
            continue
        if avail_ts > pd.Timestamp(cutoff_utc):
            continue

        form = str(forms[i] or "").upper().strip()
        acc_no = str(accs[i] or "").strip()
        if not acc_no:
            continue

        structured_fam = ""
        if form in ("8-K", "8-K/A"):
            raw_it = items_list[i] if i < len(items_list) else ""
            it_str = ",".join(str(x) for x in raw_it) if isinstance(raw_it, list) else str(raw_it or "")
            iset = {x.strip() for x in it_str.split(",") if x.strip()}
            if "2.02" in iset:
                structured_fam = "rx_earnings_202"
            elif "1.01" in iset or "2.01" in iset:
                structured_fam = "rx_deal_101_201"
            elif "7.01" in iset:
                structured_fam = "rx_regfd_701"
            elif "5.02" in iset and "5.07" not in iset:
                structured_fam = "rx_exec_502"
            elif iset & _DISTRESS_8K_ITEMS:
                structured_fam = "rx_accounting_distress"
            else:
                continue
        elif form in _ACTIVIST_FORMS:
            structured_fam = "rx_activist_13d"
        elif form in _OFFERING_FORMS:
            structured_fam = "rx_dilutive_offering"
        elif form in _LATE_FILING_FORMS:
            structured_fam = "rx_accounting_distress"
        else:
            continue

        desc = str(descs[i] or "").strip() if i < len(descs) else ""
        title_txt = f"{company_name} ({ticker}) SEC {form} {structured_fam} {desc}".strip()
        out.append({
            "event_id": f"sec:{cik}:{acc_no}",
            "security_id": security_id or f"ticker:{ticker}",
            "ticker": ticker,
            "source_family": "sec",
            "provider_source": "sec_edgar",
            "provider_category": form,
            "sec_form": form,
            "structured_catalyst": structured_fam,
            "feature_available_at_utc": avail_ts.isoformat(),
            "title": title_txt,
            "url": "",
        })

    comp_avail_iso = earliest_pre_utc.isoformat() if earliest_pre_utc is not None else None
    return out, company_name, comp_avail_iso


def collect_live_prospective_catalysts(
    panel: ResearchPanel,
    scores_df: pd.DataFrame,
    settings: Settings,
    *,
    catalyst_top_k: int = DEFAULT_CATALYST_TOP_K,
    sec_model_bundle_dir: Path | None = None,
    expected_sec_model_manifest_sha256: str | None = PERSISTED_SEC_MODEL_MANIFEST_SHA256,
    cohort_for_stage1: live_cohort.DecisionDateMidSmallCohort | None = None,
    unseal_final_test_period: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Collect & classify live Finviz Elite, Alpaca (`fiscal.ai`), and SEC catalysts `<= close_utc(scored_session)`."""
    scored_day = date.fromisoformat(str(scores_df["session_date"].iloc[0]))
    if scored_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(f"cannot collect live catalysts on or before {HISTORICAL_SEALED_END.isoformat()}")

    loaded_sec_bundle: sec_catalyst_model.LoadedSecCatalystBundle | None = None
    if sec_model_bundle_dir is not None and (sec_model_bundle_dir / "_manifest.json").is_file():
        loaded_sec_bundle = sec_catalyst_model.verify_and_load_sec_catalyst_bundle(
            sec_model_bundle_dir,
            expected_manifest_sha256=expected_sec_model_manifest_sha256,
        )

    session_clock = resolve_xnys_session_clock(scored_day)
    cutoff_utc = session_clock.source_cutoff_utc
    window_start_utc = cutoff_utc - timedelta(days=30)
    sec_batch_start_utc = cutoff_utc - timedelta(days=sec_catalyst_model.LIVE_SEC_LOOKBACK_DAYS)
    if not unseal_final_test_period:
        window_start_utc = max(window_start_utc, datetime(2026, 7, 9, 0, 0, tzinfo=UTC))
        sec_batch_start_utc = max(sec_batch_start_utc, datetime(2026, 7, 9, 0, 0, tzinfo=UTC))

    rank_col = (
        "score_beta_neutral_price_12"
        if "score_beta_neutral_price_12" in scores_df.columns and scores_df["score_beta_neutral_price_12"].notna().any()
        else "score_price_equal_12"
    )
    ordered = scores_df.sort_values(rank_col, ascending=False, na_position="last")
    sid_to_tk: dict[str, str] = {}
    tk_to_sid: dict[str, str] = {}
    for _, r_tk in scores_df.iterrows():
        s_val = str(r_tk["security_id"])
        t_val = (
            str(r_tk["ticker"]).upper().strip()
            if "ticker" in scores_df.columns and pd.notna(r_tk["ticker"]) and str(r_tk["ticker"]).strip()
            else _default_ticker_for_security_id(s_val)
        )
        sid_to_tk[s_val] = t_val
        tk_to_sid[t_val] = s_val

    top_sec_ids = ordered["security_id"].head(max(1, int(catalyst_top_k))).astype(str).tolist()
    top_tickers = [sid_to_tk[sid] for sid in top_sec_ids if sid in sid_to_tk]

    raw_events: list[dict[str, Any]] = []
    batch_8k_rows: list[dict[str, Any]] = []
    batch_non8k_rows: list[dict[str, Any]] = []
    sec_queried_sids: set[str] = set()
    company_meta: dict[str, tuple[str, str | None]] = {}
    source_errors: dict[str, int] = {"finviz": 0, "alpaca": 0, "sec": 0}

    # 1. SEC EDGAR CIK submissions (`8-K`, `SC 13D/A`, `S-1/S-3/424B`, `NT 10-K/Q`) across full cohort when SEC bundle active
    sec_target_sids = (
        scores_df["security_id"].astype(str).tolist()
        if loaded_sec_bundle is not None
        else top_sec_ids
    )
    if settings.sec_user_agent.strip():
        sec = SecSource(settings)
        tk_to_cik_title: dict[str, tuple[str, str]] = {}
        need_ticker_lookup = any(not sid.startswith("cik:") for sid in sec_target_sids)
        if need_ticker_lookup:
            try:
                ct_payload = sec.client.get_json(SEC_COMPANY_TICKERS_URL)
                if isinstance(ct_payload, dict):
                    for entry in ct_payload.values():
                        if isinstance(entry, dict) and entry.get("ticker") and entry.get("cik_str") is not None:
                            tk_u = str(entry["ticker"]).upper().strip()
                            tk_to_cik_title[tk_u] = (
                                str(entry["cik_str"]).zfill(10),
                                str(entry.get("title") or "").strip(),
                            )
            except Exception:
                source_errors["sec"] += 1

        for sid_q in sec_target_sids:
            tk = sid_to_tk.get(sid_q, _default_ticker_for_security_id(sid_q))
            if sid_q.startswith("cik:"):
                cik_10 = sid_q.removeprefix("cik:").zfill(10)
                fallback_title = ""
            else:
                mapped = tk_to_cik_title.get(tk) or tk_to_cik_title.get(tk.replace(".", "-"))
                if mapped is None:
                    continue
                cik_10, fallback_title = mapped
            try:
                sub = sec.client.get_json(sec.submissions_url.format(cik=cik_10))
                if not isinstance(sub, dict):
                    continue
                sec_queried_sids.add(sid_q)
                s_rows, comp_name, comp_avail = _extract_live_sec_catalysts_and_company(
                    sub,
                    ticker=tk,
                    cik=cik_10,
                    security_id=sid_q,
                    fallback_company=fallback_title,
                    window_start_utc=window_start_utc,
                    cutoff_utc=cutoff_utc,
                )
                company_meta[tk] = (comp_name, comp_avail)
                raw_events.extend(s_rows)
                if loaded_sec_bundle is not None:
                    b8, bn8 = sec_catalyst_model.extract_live_sec_batch_tables_for_cik(
                        sub,
                        security_id=sid_q,
                        cik=cik_10,
                        window_start_utc=sec_batch_start_utc,
                        cutoff_utc=cutoff_utc,
                    )
                    batch_8k_rows.extend(b8)
                    batch_non8k_rows.extend(bn8)
            except Exception:
                source_errors["sec"] += 1

    # 2. Finviz Elite per-ticker `news_export.ashx` (`v=3`) using `FINVIZ_ELITE_AUTH` / `FINVIZ_API_KEY`
    finviz_token = settings.finviz_elite_auth_value
    finviz_enabled = bool(finviz_token)
    if finviz_token:
        req_headers = {"User-Agent": "Mozilla/5.0 market-predictor/0.1"}
        for idx, tk in enumerate(top_tickers):
            try:
                resp = requests.get(
                    FINVIZ_ELITE_NEWS_URL,
                    params={"v": "3", "t": tk, "auth": finviz_token},
                    headers=req_headers,
                    timeout=20,
                )
                if resp.status_code == 200 and resp.text:
                    raw_events.extend(
                        parse_finviz_elite_news_csv(
                            resp.text,
                            ticker=tk,
                            security_id=tk_to_sid.get(tk, f"ticker:{tk}"),
                            window_start_utc=window_start_utc,
                            cutoff_utc=cutoff_utc,
                            unseal_final_test_period=unseal_final_test_period,
                        )
                    )
                elif resp.status_code != 200:
                    source_errors["finviz"] += 1
            except Exception as exc:
                _ = redact_finviz_auth(exc, finviz_token)
                source_errors["finviz"] += 1
            if (idx + 1) % 10 == 0:
                time_mod.sleep(0.15)
                guard("forward live Finviz Elite news capture")

    # 3. Alpaca live `/v1beta1/news` (`fiscal.ai`) batch query across top candidates
    alpaca_enabled = settings.has_alpaca
    if alpaca_enabled:
        alpaca = AlpacaSource(settings)
        top_set = set(top_tickers)
        for off in range(0, len(top_tickers), 40):
            batch_syms = tuple(top_tickers[off : off + 40])
            page_token: str | None = None
            pages = 0
            try:
                while pages < 5:
                    n_page = alpaca.fetch_news_page(
                        ",".join(batch_syms),
                        window_start_utc,
                        cutoff_utc,
                        limit=50,
                        page_token=page_token,
                        include_content=False,
                    )
                    pages += 1
                    for item in n_page.news:
                        syms = [str(s).upper().strip() for s in (item.get("symbols") or []) if str(s).strip()]
                        if len(syms) != 1 or syms[0] not in top_set:
                            continue
                        tk = syms[0]
                        created_raw = str(item.get("created_at") or "").strip()
                        title = str(item.get("headline") or "").strip()
                        if not created_raw or not title:
                            continue
                        pub_ts = pd.Timestamp(created_raw)
                        pub_utc = (pub_ts.tz_localize("UTC") if pub_ts.tzinfo is None else pub_ts.tz_convert("UTC")).to_pydatetime()
                        if pub_utc < window_start_utc or pub_utc > cutoff_utc:
                            continue
                        nid = str(item.get("id") or hashlib.sha256(f"{tk}|{created_raw}|{title}".encode()).hexdigest()[:16])
                        raw_events.append({
                            "event_id": f"alpaca:{nid}",
                            "security_id": tk_to_sid.get(tk, f"ticker:{tk}"),
                            "ticker": tk,
                            "source_family": "alpaca",
                            "provider_source": str(item.get("source") or "alpaca"),
                            "provider_category": "direct_issuer_news",
                            "sec_form": "",
                            "structured_catalyst": "",
                            "feature_available_at_utc": pub_utc.isoformat(),
                            "title": title,
                            "url": str(item.get("url") or ""),
                        })
                    page_token = n_page.next_page_token
                    if not page_token:
                        break
            except Exception:
                source_errors["alpaca"] += 1

    enriched = scores_df.copy()
    if not raw_events:
        for c_int in ("finviz_news_21d", "alpaca_news_21d", "sec_catalysts_21d", "classified_catalysts_21d"):
            enriched[c_int] = 0
        enriched["latest_catalyst_family"] = None
        enriched["latest_catalyst_rx_1d"] = None
        sec_model_rep_empty: dict[str, Any] = {"model_scored": False, "reason": "bundle_not_present"}
        if loaded_sec_bundle is not None:
            df_8k_e = pd.DataFrame(batch_8k_rows, columns=[
                "security_id", "sec_cik", "accession_number", "sec_form", "item_codes", "accepted_at_utc", "available_at_utc",
            ])
            df_n8_e = pd.DataFrame(batch_non8k_rows, columns=[
                "security_id", "sec_cik", "accession_number", "sec_form", "catalyst_family", "available_at_utc",
            ])
            if cohort_for_stage1 is not None and len(df_8k_e):
                enriched, _ = _attach_stage1_equal_rank_24_baseline(
                    enriched, cohort_for_stage1, scored_day, extra_8k_events=df_8k_e,
                )
            enriched, sec_model_rep_empty = sec_catalyst_model.predict_live_sec_catalyst_scores(
                panel,
                enriched,
                all_8k_events=df_8k_e,
                non8k_events=df_n8_e,
                queried_security_ids=sec_queried_sids,
                bundle=loaded_sec_bundle,
            )
        empty_cat = pd.DataFrame(columns=[
            "event_id", "security_id", "ticker", "source_family", "provider_source", "provider_category",
            "sec_form", "structured_catalyst", "classified_families", "feature_available_at_utc",
            "event_session", "rx_1d", "title", "url",
        ])
        return enriched, empty_cat, {
            "window_start_utc": window_start_utc.isoformat(),
            "decision_cutoff_utc": cutoff_utc.isoformat(),
            "candidate_stocks_queried": len(top_tickers),
            "finviz_elite_enabled": finviz_enabled,
            "alpaca_news_enabled": alpaca_enabled,
            "total_events": 0,
            "by_source_family": {},
            "by_classified_family": {},
            "sec_xgb_model": sec_model_rep_empty,
            "source_errors": source_errors,
        }

    ev_df = pd.DataFrame(raw_events).drop_duplicates("event_id", keep="first").reset_index(drop=True)
    ev_df["issuer_company"] = [company_meta.get(tk, ("", None))[0] for tk in ev_df["ticker"]]
    ev_df["issuer_company_available_at_utc"] = [company_meta.get(tk, ("", None))[1] for tk in ev_df["ticker"]]
    # Blank `issuer_company` where causal pre-window availability timestamp is absent so `classify_event_families` never fails.
    no_avail = ev_df["issuer_company_available_at_utc"].isna()
    ev_df.loc[no_avail, "issuer_company"] = ""

    classified_df = classify_event_families(ev_df)
    fam_by_event: dict[str, list[str]] = {}
    if not classified_df.empty:
        for eid, grp in classified_df.groupby("event_id", sort=False):
            fam_by_event[str(eid)] = sorted(set(grp["event_family"].astype(str)))

    ev_df["classified_families"] = [
        ",".join(fam_by_event.get(str(eid), [])) for eid in ev_df["event_id"]
    ]

    # Map each event's `feature_available_at_utc` to its causal regular session `e` in `panel` and compute `rx_1d`.
    av_series = pd.to_datetime(ev_df["feature_available_at_utc"], utc=True)
    e_rows = event_rows(panel.sessions, av_series)
    col_of = {sid: idx for idx, sid in enumerate(panel.security_ids)}
    closes = panel.close.astype(np.float64)
    spy_c = panel.spy_close.astype(np.float64)
    n_sess = len(panel.sessions)
    sess_strs: list[str | None] = []
    rx_list: list[float | None] = []
    age_sessions: list[int] = []
    for r_val, sid in zip(e_rows, ev_df["security_id"].astype(str), strict=False):
        ri = int(r_val)
        ci = col_of.get(sid, -1)
        if 0 <= ri < n_sess:
            sess_strs.append(str(panel.sessions[ri]))
            age_sessions.append(n_sess - 1 - ri)
        else:
            sess_strs.append(None)
            age_sessions.append(999)
        if ci >= 0 and 1 <= ri < n_sess:
            c0, c1 = closes[ri - 1, ci], closes[ri, ci]
            s0, s1 = spy_c[ri - 1], spy_c[ri]
            if np.isfinite(c0) and c0 > 0 and np.isfinite(c1) and c1 > 0 and np.isfinite(s0) and s0 > 0 and np.isfinite(s1) and s1 > 0:
                rx_list.append(float((c1 / c0 - 1.0) - (s1 / s0 - 1.0)))
            else:
                rx_list.append(None)
        else:
            rx_list.append(None)

    ev_df["event_session"] = sess_strs
    ev_df["rx_1d"] = rx_list
    ev_df["_age_sessions"] = age_sessions

    in_21d = ev_df.loc[ev_df["_age_sessions"] <= 21].copy()
    fv_cnt = in_21d.loc[in_21d["source_family"] == "finviz"].groupby("security_id").size().to_dict()
    al_cnt = in_21d.loc[in_21d["source_family"] == "alpaca"].groupby("security_id").size().to_dict()
    sc_cnt = in_21d.loc[in_21d["source_family"] == "sec"].groupby("security_id").size().to_dict()
    cl_mask = (in_21d["classified_families"] != "") | (in_21d["structured_catalyst"] != "")
    cl_cnt = in_21d.loc[cl_mask].groupby("security_id").size().to_dict()

    latest_fam: dict[str, str] = {}
    latest_rx: dict[str, float | None] = {}
    if len(in_21d):
        in_sorted = in_21d.sort_values("feature_available_at_utc")
        for sid, grp in in_sorted.groupby("security_id", sort=False):
            qual = grp.loc[(grp["classified_families"] != "") | (grp["structured_catalyst"] != "")]
            pick = qual.iloc[-1] if len(qual) else grp.iloc[-1]
            fam_lbl = str(pick["classified_families"] or pick["structured_catalyst"] or pick["provider_category"] or pick["source_family"])
            latest_fam[str(sid)] = fam_lbl
            rx_v = pick["rx_1d"]
            latest_rx[str(sid)] = None if rx_v is None or pd.isna(rx_v) else float(rx_v)

    sids = enriched["security_id"].astype(str)
    enriched["finviz_news_21d"] = [int(fv_cnt.get(s, 0)) for s in sids]
    enriched["alpaca_news_21d"] = [int(al_cnt.get(s, 0)) for s in sids]
    enriched["sec_catalysts_21d"] = [int(sc_cnt.get(s, 0)) for s in sids]
    enriched["classified_catalysts_21d"] = [int(cl_cnt.get(s, 0)) for s in sids]
    enriched["latest_catalyst_family"] = [latest_fam.get(s) for s in sids]
    enriched["latest_catalyst_rx_1d"] = [latest_rx.get(s) for s in sids]

    sec_model_rep: dict[str, Any] = {"model_scored": False, "reason": "bundle_not_present"}
    if loaded_sec_bundle is not None:
        df_8k = (
            pd.DataFrame(batch_8k_rows).drop_duplicates(["security_id", "accession_number"])
            if batch_8k_rows
            else pd.DataFrame(columns=[
                "security_id", "sec_cik", "accession_number", "sec_form", "item_codes", "accepted_at_utc", "available_at_utc",
            ])
        )
        df_n8 = (
            pd.DataFrame(batch_non8k_rows).drop_duplicates(["security_id", "accession_number"])
            if batch_non8k_rows
            else pd.DataFrame(columns=[
                "security_id", "sec_cik", "accession_number", "sec_form", "catalyst_family", "available_at_utc",
            ])
        )
        if cohort_for_stage1 is not None and len(df_8k):
            enriched, _ = _attach_stage1_equal_rank_24_baseline(
                enriched, cohort_for_stage1, scored_day, extra_8k_events=df_8k,
            )
        enriched, sec_model_rep = sec_catalyst_model.predict_live_sec_catalyst_scores(
            panel,
            enriched,
            all_8k_events=df_8k,
            non8k_events=df_n8,
            queried_security_ids=sec_queried_sids,
            bundle=loaded_sec_bundle,
        )

    keep_cols = [
        "event_id", "security_id", "ticker", "source_family", "provider_source", "provider_category",
        "sec_form", "structured_catalyst", "classified_families", "feature_available_at_utc",
        "event_session", "rx_1d", "title", "url",
    ]
    cat_out = ev_df[keep_cols].sort_values(["feature_available_at_utc", "event_id"], kind="stable").reset_index(drop=True)
    by_src = {str(k): int(v) for k, v in cat_out["source_family"].value_counts().to_dict().items()}
    by_cfam: dict[str, int] = {}
    if not classified_df.empty:
        by_cfam = {str(k): int(v) for k, v in classified_df["event_family"].value_counts().to_dict().items()}

    return enriched, cat_out, {
        "window_start_utc": window_start_utc.isoformat(),
        "decision_cutoff_utc": cutoff_utc.isoformat(),
        "candidate_stocks_queried": len(top_tickers),
        "finviz_elite_enabled": finviz_enabled,
        "alpaca_news_enabled": alpaca_enabled,
        "total_events": len(cat_out),
        "by_source_family": dict(sorted(by_src.items())),
        "by_classified_family": dict(sorted(by_cfam.items())),
        "sec_xgb_model": sec_model_rep,
        "source_errors": source_errors,
    }


def resolve_latest_completed_session(
    settings: Settings,
    *,
    now_utc: datetime | None = None,
    unseal_final_test_period: bool = False,
) -> date:
    """Resolve the latest `XNYS` session whose post-close settlement cutoff (`close + 20m`) has passed."""
    ts_utc = now_utc.astimezone(UTC) if now_utc is not None else datetime.now(UTC)
    settled_clock = latest_settled_xnys_session(ts_utc)
    cutoff_day = settled_clock.as_of_date
    if cutoff_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(
            f"latest completed session {cutoff_day.isoformat()} is inside sealed window <= {HISTORICAL_SEALED_END.isoformat()}"
        )

    start_day = cutoff_day - timedelta(days=14)
    if not unseal_final_test_period:
        start_day = max(start_day, HISTORICAL_SEALED_END + timedelta(days=1))
    start_utc = datetime.combine(start_day, time(0, 0), tzinfo=UTC)
    end_utc = datetime.combine(cutoff_day, time(23, 59, 59), tzinfo=UTC)

    alpaca = AlpacaSource(settings)
    page = alpaca.fetch_bars_page(
        ("SPY",), start_utc, end_utc, timeframe="1Day", asof=cutoff_day, adjustment="all",
    )
    spy_days: list[date] = []
    for b in page.bars.get("SPY", []):
        b_day = pd.Timestamp(str(b["t"])).tz_convert(MARKET_TIMEZONE).date()
        if b_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
            raise SealedPeriodError(f"SPY session {b_day.isoformat()} leaked from sealed window")
        if start_day <= b_day <= cutoff_day:
            spy_days.append(b_day)
    if not spy_days:
        raise ValueError(f"no completed SPY trading session found in [{start_day.isoformat()}, {cutoff_day.isoformat()}]")
    return max(spy_days)


def _enforce_prospective_prediction_clock(session_clock_date: date, now_utc: datetime) -> None:
    """Require `session_close_utc <= now_utc < next_tradable_open_utc` before recording prospectively."""
    clk = resolve_xnys_session_clock(session_clock_date)
    ts = now_utc.astimezone(UTC) if now_utc.tzinfo is not None else now_utc.replace(tzinfo=UTC)
    if ts < clk.session_close_utc:
        raise DataReadinessError(
            f"prospective recording for session {clk.as_of_date.isoformat()} attempted at {ts.isoformat()} "
            f"before session close {clk.session_close_utc.isoformat()}"
        )
    if ts >= clk.next_tradable_open_utc:
        raise DataReadinessError(
            f"prospective recording for session {clk.as_of_date.isoformat()} missed next tradable open "
            f"{clk.next_tradable_open_utc.isoformat()} on {clk.next_tradable_session.isoformat()} "
            f"(completed at {ts.isoformat()}); late post-open records cannot be admitted as prospective"
        )


def _verify_existing_snapshot_for_skip(
    snapshot_dir: Path,
    *,
    expected_candidates: tuple[ProspectiveCandidateSpec, ...] = FROZEN_PROSPECTIVE_CANDIDATE_SPECS,
) -> dict[str, Any]:
    """Verify payload SHA-256s and candidate registry identity before returning `--skip-existing`."""
    manifest_path = snapshot_dir / "_manifest.json"
    if not manifest_path.is_file():
        raise ArtifactIntegrityError(f"existing snapshot directory missing _manifest.json: {snapshot_dir}")
    try:
        manifest = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise ArtifactIntegrityError(f"unreadable snapshot _manifest.json in {snapshot_dir}") from exc

    payload_status = verify_snapshot_payload_hashes(snapshot_dir, manifest)
    if payload_status != "verified":
        raise ArtifactIntegrityError(
            f"existing snapshot {snapshot_dir} lacks verified payload_artifacts ({payload_status}); refusing --skip-existing"
        )

    raw_cands = manifest.get("frozen_prospective_candidates")
    if not isinstance(raw_cands, list) or len(raw_cands) != len(expected_candidates):
        raise ArtifactIntegrityError(
            f"existing snapshot {snapshot_dir} candidate registry length mismatch against FROZEN_PROSPECTIVE_CANDIDATE_SPECS"
        )
    for raw_item, expected in zip(raw_cands, expected_candidates, strict=True):
        if not isinstance(raw_item, dict):
            raise ArtifactIntegrityError(f"malformed candidate spec in {snapshot_dir}")
        try:
            persisted = ProspectiveCandidateSpec.from_mapping(raw_item)
        except ValueError as exc:
            raise ArtifactIntegrityError(f"incomplete candidate spec in {snapshot_dir}: {exc}") from exc
        if persisted.candidate_spec_sha256 != expected.candidate_spec_sha256:
            raise ArtifactIntegrityError(
                f"candidate identity mismatch on {expected.candidate_id} in {snapshot_dir}: "
                f"expected {expected.candidate_spec_sha256}, got {persisted.candidate_spec_sha256}"
            )
    return manifest


def _attach_stage1_equal_rank_24_baseline(
    scores_df: pd.DataFrame,
    cohort: live_cohort.DecisionDateMidSmallCohort,
    scored_day: date,
    *,
    extra_8k_events: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach `score_equal_rank_24_beta_neutral` computed by the shared batch Stage-1 builder (`R3`)."""
    releases_202 = cohort.releases_202
    if extra_8k_events is not None and len(extra_8k_events):
        has_202 = extra_8k_events["item_codes"].fillna("").astype(str).map(
            lambda s: "2.02" in {x.strip() for x in s.split(",")}
        )
        sub_202 = extra_8k_events.loc[
            has_202,
            ["security_id", "accession_number", "accepted_at_utc", "available_at_utc"],
        ].copy()
        if len(sub_202):
            releases_202 = pd.concat([releases_202, sub_202], ignore_index=True).drop_duplicates(
                ["security_id", "accession_number"]
            )

    mid_bn_eq24, _, _, stage1_rep = sec_catalyst_model.compute_mid_small_stage1_baseline(
        cohort.panel,
        unadjusted_close=cohort.unadjusted_close,
        releases_202=releases_202,
        security_cik_map=cohort.security_cik_map,
        company_facts_paths=cohort.company_facts_paths,
        as_of_date=scored_day,
    )
    out = scores_df.copy()
    last_row = mid_bn_eq24[-1]
    val_by_sid = {
        sid: (None if not np.isfinite(last_row[idx]) else float(last_row[idx]))
        for idx, sid in enumerate(cohort.panel.security_ids)
    }
    out["score_equal_rank_24_beta_neutral"] = [
        val_by_sid.get(str(s)) for s in out["security_id"].astype(str)
    ]
    return out, stage1_rep


def build_snapshot_decision_holdings(
    scores_df: pd.DataFrame,
    *,
    panel_sessions: npt.NDArray[Any],
    dest_parent: Path,
    candidate_specs: tuple[ProspectiveCandidateSpec, ...] = FROZEN_PROSPECTIVE_CANDIDATE_SPECS,
    options_summary: Mapping[str, Any] | None = None,
    catalyst_summary: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Select and freeze candidate holdings & target weights from decision-time inputs ONLY (`R5`, `R6`).

    Reuses `benchmark_tilt.select_buffered_top_indices` (`buy top 10% / hold top 33%`) and reads
    prior rebalance holdings from `dest_parent / prev_rebalance_session / holdings.parquet` when
    recorded, never consulting any future bar or outcome.
    """
    scored_session = str(scores_df["session_date"].iloc[0])
    sess_strs = [str(s) for s in panel_sessions]
    sids = scores_df["security_id"].astype(str).tolist()
    tickers = (
        [str(t).upper().strip() for t in scores_df["ticker"].fillna("").astype(str)]
        if "ticker" in scores_df.columns
        else [_default_ticker_for_security_id(s) for s in sids]
    )

    all_cik = bool(scores_df["security_id"].astype(str).str.startswith("cik:").all()) if len(scores_df) else False
    unknown_sectors = (
        int((scores_df["sector"].fillna("Unknown").astype(str) == "Unknown").sum())
        if "sector" in scores_df.columns
        else len(scores_df)
    )
    sec_meta = catalyst_summary.get("sec_xgb_model") if isinstance(catalyst_summary, Mapping) else None
    sec_scored = bool(isinstance(sec_meta, Mapping) and sec_meta.get("model_scored") is True)
    actual_model_sha = str(sec_meta.get("manifest_sha256") or "") if isinstance(sec_meta, Mapping) else ""
    actual_feat_sha = str(sec_meta.get("feature_contract_sha256") or "") if isinstance(sec_meta, Mapping) else ""
    opra_reg, opra_obs_kind = extract_snapshot_opra_regime({"options_summary": options_summary})

    rows: list[dict[str, Any]] = []
    summary_by_cand: dict[str, Any] = {}

    for spec in candidate_specs:
        h = spec.horizon_sessions
        prev_sess = sess_strs[-1 - h] if len(sess_strs) > h and sess_strs[-1] == scored_session else None
        prior_held_sids: set[str] = set()
        if prev_sess is not None:
            prev_dir = dest_parent / prev_sess
            prev_mf = prev_dir / "_manifest.json"
            prev_hp = prev_dir / "holdings.parquet"
            if prev_mf.is_file() and prev_hp.is_file():
                prev_m_dict = cast(dict[str, Any], json.loads(prev_mf.read_text(encoding="utf-8")))
                verify_snapshot_payload_hashes(prev_dir, prev_m_dict)
                prev_h_df = cast(Any, pq).read_table(prev_hp, use_threads=False).to_pandas()
                mask_prev = (
                    (prev_h_df["candidate_id"].astype(str) == spec.candidate_id)
                    & (~prev_h_df["abstained"].astype(bool))
                    & (prev_h_df["asset_role"].astype(str) == "satellite_equity")
                    & (pd.to_numeric(prev_h_df["target_weight"], errors="coerce") > 0.0)
                )
                prior_held_sids = set(prev_h_df.loc[mask_prev, "security_id"].astype(str).tolist())

        abstain_reason: str | None = None
        if spec.cohort == "mid_small_ex_sp500" and (not all_cik or unknown_sectors > 0):
            abstain_reason = "cohort_identity_or_sector_not_verified"
        elif not sec_scored:
            detail = (
                str(sec_meta.get("reason") or "sec_xgb_model_not_scored")
                if isinstance(sec_meta, Mapping)
                else "sec_xgb_model_not_scored"
            )
            abstain_reason = f"missing_sec_model:{detail}"
        elif actual_model_sha != spec.expected_model_manifest_sha256:
            abstain_reason = f"model_manifest_sha256_mismatch:{actual_model_sha}!={spec.expected_model_manifest_sha256}"
        elif actual_feat_sha != spec.expected_feature_contract_sha256:
            abstain_reason = f"feature_contract_sha256_mismatch:{actual_feat_sha}!={spec.expected_feature_contract_sha256}"
        elif spec.regime_rule != "none":
            if spec.opra_observation_policy == "synchronized_close_only" and opra_obs_kind != "synchronized_close":
                abstain_reason = f"opra_observation_not_synchronized_close:{opra_obs_kind}"
            elif opra_reg == "unknown" and spec.missing_regime_policy == "abstain":
                abstain_reason = "opra_regime_unknown"

        scores_arr = (
            pd.to_numeric(scores_df[spec.score_column], errors="coerce").to_numpy(dtype=np.float64)
            if spec.score_column in scores_df.columns
            else np.full(len(scores_df), np.nan, dtype=np.float64)
        )
        valid = np.flatnonzero(np.isfinite(scores_arr))
        if abstain_reason is None and valid.size < spec.min_eligible_stocks:
            abstain_reason = f"insufficient_scored_stocks_in_{spec.score_column}:{int(valid.size)}<{spec.min_eligible_stocks}"

        cand_reg_str = "not_applicable" if spec.regime_rule == "none" else opra_reg
        if abstain_reason is not None:
            rows.append({
                "session_date": scored_session,
                "candidate_id": spec.candidate_id,
                "candidate_spec_sha256": spec.candidate_spec_sha256,
                "horizon_sessions": h,
                "abstained": True,
                "abstain_reason": abstain_reason,
                "opra_regime": cand_reg_str,
                "security_id": "",
                "ticker": "",
                "asset_role": "abstained",
                "target_weight": 0.0,
                "decision_score": None,
                "selection_rank": None,
            })
            summary_by_cand[spec.candidate_id] = {
                "abstained": True,
                "abstain_reason": abstain_reason,
                "selected_equities": 0,
                "prior_rebalance_session": prev_sess,
                "retained_from_prior_rebalance": 0,
            }
            continue

        if (
            spec.regime_rule == "shift_satellite_to_spy_when_either_index_iv_term_30_90_gt_1"
            and opra_reg == "backwardation"
        ):
            rows.append({
                "session_date": scored_session,
                "candidate_id": spec.candidate_id,
                "candidate_spec_sha256": spec.candidate_spec_sha256,
                "horizon_sessions": h,
                "abstained": False,
                "abstain_reason": None,
                "opra_regime": cand_reg_str,
                "security_id": "benchmark:SPY",
                "ticker": "SPY",
                "asset_role": "regime_spy_shift",
                "target_weight": 1.0,
                "decision_score": None,
                "selection_rank": 1,
            })
            summary_by_cand[spec.candidate_id] = {
                "abstained": False,
                "abstain_reason": None,
                "opra_regime": cand_reg_str,
                "selected_equities": 0,
                "spy_weight": 1.0,
                "prior_rebalance_session": prev_sess,
                "retained_from_prior_rebalance": 0,
            }
            continue

        prior_idx = {idx for idx, sid in enumerate(sids) if sid in prior_held_sids}
        chosen = benchmark_tilt.select_buffered_top_indices(
            scores_arr,
            valid,
            prior_held=prior_idx,
            buffered=(spec.selection_rule == "buffered_top_decile_equal_weight_buy10_hold33"),
            buy_top_fraction=spec.buy_top_fraction,
            hold_top_fraction=spec.hold_top_fraction,
        )
        eq_weight = float(spec.satellite_weight) / float(chosen.size)
        retained_cnt = sum(1 for col in chosen if sids[int(col)] in prior_held_sids)
        for rk_pos, col_i in enumerate(chosen):
            idx_c = int(col_i)
            rows.append({
                "session_date": scored_session,
                "candidate_id": spec.candidate_id,
                "candidate_spec_sha256": spec.candidate_spec_sha256,
                "horizon_sessions": h,
                "abstained": False,
                "abstain_reason": None,
                "opra_regime": cand_reg_str,
                "security_id": sids[idx_c],
                "ticker": tickers[idx_c],
                "asset_role": "satellite_equity",
                "target_weight": eq_weight,
                "decision_score": float(scores_arr[idx_c]),
                "selection_rank": rk_pos + 1,
            })
        if spec.core_spy_weight > 0.0:
            rows.append({
                "session_date": scored_session,
                "candidate_id": spec.candidate_id,
                "candidate_spec_sha256": spec.candidate_spec_sha256,
                "horizon_sessions": h,
                "abstained": False,
                "abstain_reason": None,
                "opra_regime": cand_reg_str,
                "security_id": "benchmark:SPY",
                "ticker": "SPY",
                "asset_role": "core_spy",
                "target_weight": float(spec.core_spy_weight),
                "decision_score": None,
                "selection_rank": None,
            })
        summary_by_cand[spec.candidate_id] = {
            "abstained": False,
            "abstain_reason": None,
            "opra_regime": cand_reg_str,
            "selected_equities": int(chosen.size),
            "spy_weight": float(spec.core_spy_weight),
            "prior_rebalance_session": prev_sess,
            "retained_from_prior_rebalance": int(retained_cnt),
        }

    return pd.DataFrame(rows), summary_by_cand


def record_forward_scores(
    *,
    root: Path,
    as_of_date: date | None = None,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    output_parent: Path = FORWARD_RECORD_DIRECTORY,
    sec_model_bundle: Path = sec_catalyst_model.DEFAULT_SEC_MODEL_BUNDLE_DIR,
    expected_sec_model_manifest_sha256: str | None = PERSISTED_SEC_MODEL_MANIFEST_SHA256,
    live_prospective: bool = False,
    record_mode: RecordMode | None = None,
    include_live_catalysts: bool = True,
    include_live_options: bool = True,
    catalyst_top_k: int = DEFAULT_CATALYST_TOP_K,
    skip_existing: bool = False,
    unseal_final_test_period: bool = False,
    settings: Settings | None = None,
    clock_fn: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Publish one immutable dated forward-score snapshot (plus live catalysts & OPRA options) and append to log."""
    effective_mode: RecordMode = (
        record_mode if record_mode is not None else ("prospective" if live_prospective else "retrospective_replay")
    )
    if effective_mode == "prospective" and not live_prospective:
        raise ValueError("record_mode='prospective' requires live_prospective=True")

    if as_of_date is not None:
        if not live_prospective:
            check_development_end(as_of_date, unseal_final_test_period=unseal_final_test_period)
        elif as_of_date <= HISTORICAL_SEALED_END and not unseal_final_test_period:
            raise SealedPeriodError(
                f"cannot record live prospective scores inside sealed window ({as_of_date.isoformat()} <= "
                f"{HISTORICAL_SEALED_END.isoformat()})"
            )
    elif not live_prospective:
        raise ValueError("as_of_date=None ('latest') requires live_prospective=True")

    now_fn = clock_fn or (lambda: datetime.now(UTC))
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(RECORD_FORWARD_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("forward score recorder start")
        cfg = settings or get_settings()
        resolved_date = (
            resolve_latest_completed_session(
                cfg,
                now_utc=now_fn(),
                unseal_final_test_period=unseal_final_test_period,
            )
            if as_of_date is None
            else as_of_date
        )
        dest_parent = paths.write(paths.identity(output_parent))
        pre_dir = dest_parent / resolved_date.isoformat()
        if skip_existing and (pre_dir / "_manifest.json").is_file():
            existing_pre = _verify_existing_snapshot_for_skip(pre_dir)
            return {**existing_pre, "status": "already_recorded"}

        session_clock = resolve_xnys_session_clock(resolved_date)
        if effective_mode == "prospective":
            _enforce_prospective_prediction_clock(resolved_date, now_fn())

        cohort_data: live_cohort.DecisionDateMidSmallCohort | None = None
        if live_prospective:
            live_loaded = _fetch_live_prospective_panel(
                resolved_date,
                cfg,
                paths=paths,
                root=root,
                unseal_final_test_period=unseal_final_test_period,
            )
            if isinstance(live_loaded, live_cohort.DecisionDateMidSmallCohort):
                cohort_data = live_loaded
                loaded_panel = live_loaded.panel
            else:
                loaded_panel = live_loaded
        else:
            cohort_data = live_cohort.load_decision_date_mid_small_cohort(
                paths,
                as_of_date=resolved_date,
                source_cutoff_utc=session_clock.source_cutoff_utc,
                panel_rel=panel,
                live_prospective=False,
                unseal_final_test_period=unseal_final_test_period,
                settings=cfg,
            )
            loaded_panel = cohort_data.panel

        scores_df = _score_panel_last_session(
            loaded_panel,
            active_ticker_by_security_id=(
                cohort_data.active_ticker_by_security_id if cohort_data is not None else None
            ),
        )
        if scores_df.empty:
            raise ValueError(f"no eligible stocks scored on {resolved_date.isoformat()}")
        scored_session = str(scores_df["session_date"].iloc[0])
        scored_day = date.fromisoformat(scored_session)
        scored_clock = resolve_xnys_session_clock(scored_day)

        dest_dir = dest_parent / scored_session
        if dest_dir.exists():
            if skip_existing and (dest_dir / "_manifest.json").is_file():
                existing_post = _verify_existing_snapshot_for_skip(dest_dir)
                return {**existing_post, "status": "already_recorded"}
            raise FileExistsError(f"forward score snapshot already exists for session {scored_session}: {dest_dir}")

        stage1_baseline_report: dict[str, Any] = {
            "baseline_scored": False,
            "abstained": True,
            "reason": "missing_fundamental_cohort_inputs",
        }
        if cohort_data is not None:
            scores_df, stage1_baseline_report = _attach_stage1_equal_rank_24_baseline(
                scores_df, cohort_data, scored_day,
            )
        else:
            scores_df["score_equal_rank_24_beta_neutral"] = None

        catalysts_df: pd.DataFrame | None = None
        catalyst_summary: dict[str, Any] | None = None
        bundle_abs = sec_model_bundle if sec_model_bundle.is_absolute() else paths.write(paths.identity(sec_model_bundle))
        if live_prospective and include_live_catalysts:
            scores_df, catalysts_df, catalyst_summary = collect_live_prospective_catalysts(
                loaded_panel,
                scores_df,
                cfg,
                catalyst_top_k=catalyst_top_k,
                sec_model_bundle_dir=bundle_abs,
                expected_sec_model_manifest_sha256=expected_sec_model_manifest_sha256,
                cohort_for_stage1=cohort_data,
                unseal_final_test_period=unseal_final_test_period,
            )
        elif not live_prospective and cohort_data is not None and (bundle_abs / "_manifest.json").is_file():
            loaded_sec_bundle = sec_catalyst_model.verify_and_load_sec_catalyst_bundle(
                bundle_abs,
                expected_manifest_sha256=expected_sec_model_manifest_sha256,
            )
            scores_df, sec_model_rep = sec_catalyst_model.predict_live_sec_catalyst_scores(
                loaded_panel,
                scores_df,
                all_8k_events=cohort_data.all_8k_events,
                non8k_events=cohort_data.non8k_events,
                queried_security_ids=set(loaded_panel.security_ids),
                bundle=loaded_sec_bundle,
            )
            catalyst_summary = {
                "mode": "retrospective_replay_retained_sec_events",
                "decision_cutoff_utc": scored_clock.source_cutoff_utc.isoformat(),
                "sec_xgb_model": sec_model_rep,
            }

        options_df: pd.DataFrame | None = None
        options_summary: dict[str, Any] | None = None
        if live_prospective and include_live_options:
            scores_df, options_df, options_summary = opra_options.collect_live_opra_options_overlay(
                loaded_panel,
                scores_df,
                cfg,
                options_top_k=catalyst_top_k,
                unseal_final_test_period=unseal_final_test_period,
            )

        holdings_df, holdings_summary = build_snapshot_decision_holdings(
            scores_df,
            panel_sessions=loaded_panel.sessions,
            dest_parent=dest_parent,
            options_summary=options_summary,
            catalyst_summary=catalyst_summary,
        )

        dest_parent.mkdir(parents=True, exist_ok=True)
        staging = dest_parent / f".{scored_session}.{uuid4().hex}.partial"
        staging.mkdir(parents=True)
        try:
            cast(Any, pq).write_table(pa.Table.from_pandas(scores_df, preserve_index=False), staging / "scores.parquet")
            cast(Any, pq).write_table(pa.Table.from_pandas(holdings_df, preserve_index=False), staging / "holdings.parquet")
            if catalysts_df is not None:
                cast(Any, pq).write_table(
                    pa.Table.from_pandas(catalysts_df, preserve_index=False), staging / "catalysts.parquet",
                )
            if options_df is not None and len(options_df):
                cast(Any, pq).write_table(
                    pa.Table.from_pandas(options_df, preserve_index=False), staging / "options_metrics.parquet",
                )

            payload_artifacts = build_payload_artifacts_manifest(staging)
            published_at = now_fn().astimezone(UTC)
            if effective_mode == "prospective":
                _enforce_prospective_prediction_clock(scored_day, published_at)

            manifest: dict[str, Any] = {
                "schema": SNAPSHOT_MANIFEST_SCHEMA,
                "status": "recorded",
                "record_mode": effective_mode,
                "as_of_date": scored_session,
                "source_cutoff_utc": scored_clock.source_cutoff_utc.isoformat(),
                "session_close_utc": scored_clock.session_close_utc.isoformat(),
                "is_early_close": scored_clock.is_early_close,
                "next_tradable_session": scored_clock.next_tradable_session.isoformat(),
                "next_tradable_open_utc": scored_clock.next_tradable_open_utc.isoformat(),
                "published_at_utc": published_at.isoformat(),
                "recorded_at_utc": published_at.isoformat(),
                "published_before_entry": bool(published_at < scored_clock.next_tradable_open_utc),
                "requested_as_of_date": "latest" if as_of_date is None else resolved_date.isoformat(),
                "resolved_as_of_date": resolved_date.isoformat(),
                "scored_session": scored_session,
                "session_clock": session_clock.to_dict() if scored_day == resolved_date else scored_clock.to_dict(),
                "live_prospective": live_prospective,
                "sealed_holdout_window": "2024-05-29..2026-07-08",
                "sealed_holdout_untouched": not unseal_final_test_period,
                "cohort_authority_report": cohort_data.authority_report if cohort_data is not None else None,
                "stage1_baseline_report": stage1_baseline_report,
                "frozen_prospective_candidates": list(FROZEN_PROSPECTIVE_CANDIDATES),
                "candidate_registry_sha256": json_sha256(list(FROZEN_PROSPECTIVE_CANDIDATES)),
                "decision_holdings_summary": holdings_summary,
                "payload_artifacts": payload_artifacts,
                "eligible_stocks_scored": len(scores_df),
                "catalyst_summary": catalyst_summary,
                "options_summary": options_summary,
                "code": git_identity(root),
                "label_notice": (
                    "research forward score snapshot (zero future outcomes read)"
                    if effective_mode == "prospective"
                    else "retrospective replay research snapshot (non-prospective diagnostic)"
                ),
            }
            (staging / "_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
            )
            os.rename(staging, dest_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        log_path = dest_parent / FORWARD_LOG_NAME
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, sort_keys=True) + "\n")

    return manifest


__all__ = [
    "DEFAULT_CATALYST_TOP_K",
    "FINVIZ_ELITE_NEWS_URL",
    "FORWARD_LOG_NAME",
    "FORWARD_RECORD_DIRECTORY",
    "FROZEN_PROSPECTIVE_CANDIDATES",
    "HISTORICAL_SEALED_END",
    "POST_CLOSE_SETTLEMENT_ET",
    "RECORD_FORWARD_COMMAND",
    "build_snapshot_decision_holdings",
    "collect_live_prospective_catalysts",
    "parse_finviz_elite_news_csv",
    "record_forward_scores",
    "redact_finviz_auth",
    "resolve_latest_completed_session",
]
