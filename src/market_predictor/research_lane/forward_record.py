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
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research_lane import indicators, opra_options, predictors, reversal
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
    load_research_panel,
)
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.sec_filings import event_rows
from market_predictor.research_lane.trial_log import git_identity
from market_predictor.research_lane.wider_universe import WIDER_UNIVERSE_DIRECTORY
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
FROZEN_PROSPECTIVE_CANDIDATES: Final[tuple[dict[str, Any], ...]] = (
    {
        "candidate_id": "90_10__sp500_tilt_news__R1_backwardation_to_spy",
        "horizon_sessions": 10,
        "regime_rule": "shift_satellite_to_spy_when_iv_term_30_90_gt_1",
        "satellite_score_column": "score_beta_neutral_price_12",
    },
    {
        "candidate_id": "80_20__spy__R2_adaptive_hedge_sec_xgb",
        "horizon_sessions": 63,
        "regime_rule": "contango_raw_else_70_beta_neutral_30_skew_vetoed_reversal",
        "satellite_score_column": "score_r2_opra_conditioned",
    },
)


def redact_finviz_auth(value: object, token: str | None = None) -> str:
    """Scrub `auth=` query tokens and explicit secret strings from transport errors or URLs."""
    text = re.sub(r"(?i)(auth=)[^&\s\"']+", r"\1<REDACTED>", str(value))
    if token:
        text = text.replace(token, "<REDACTED>")
    return text


def _score_panel_last_session(panel: ResearchPanel) -> pd.DataFrame:
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
    if not np.isfinite(beta_rk[elig_row]).any() and len(panel.sessions) >= 61:
        ret_stock = predictors.daily_returns(panel.close)
        ret_mkt = predictors.daily_returns(panel.spy_close)
        beta_60 = predictors._regression(ret_stock, ret_mkt, 60).beta
        beta_rk = centered_rank(beta_60, panel.eligible)[last_idx]

    beta_neutral_12 = cross_sectional_neutralize(
        price_eq_12[None, :],
        elig_row[None, :],
        panel.sector[last_idx : last_idx + 1],
        beta_rk[None, :],
        ranked_last["idio_vol_60d"][None, :],
        rerank=True,
    )[0]

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
            "sector": sec_name,
            "close_adjusted": float(panel.close[last_idx, col]),
            "score_price_equal_12": None if not np.isfinite(price_eq_12[col]) else float(price_eq_12[col]),
            "score_beta_neutral_price_12": None if not np.isfinite(beta_neutral_12[col]) else float(beta_neutral_12[col]),
            "score_reversal_short": None if not np.isfinite(rev_short[col]) else float(rev_short[col]),
        })
    return pd.DataFrame(rows)


def _lookup_local_sectors_by_ticker(root: Path, tickers: tuple[str, ...]) -> tuple[npt.NDArray[np.int16], tuple[str, ...]]:
    """Map tickers to SIC sector codes using retained `wider_us_universe` metadata if present on disk."""
    roster_pq = root / WIDER_UNIVERSE_DIRECTORY / "wider_roster_spans.parquet"
    sectors_json = root / WIDER_UNIVERSE_DIRECTORY / "cik_sectors.json"
    if not (roster_pq.is_file() and sectors_json.is_file()):
        return np.zeros(len(tickers), dtype=np.int16), ("Unknown",)

    roster_df = cast(Any, pq).read_table(roster_pq, columns=["cik", "ticker", "valid_to"], use_threads=False).to_pandas()
    cik_to_name: dict[str, str] = json.loads(sectors_json.read_text(encoding="utf-8"))
    roster_df = roster_df.sort_values("valid_to")
    tk_to_name: dict[str, str] = {}
    for _, r in roster_df.iterrows():
        tk = str(r["ticker"]).upper().strip()
        cik = str(r["cik"]).zfill(10)
        if cik in cik_to_name:
            tk_to_name[tk] = str(cik_to_name[cik])

    per_tk = [tk_to_name.get(tk, "Unknown") for tk in tickers]
    sec_names = tuple(sorted(set(per_tk)))
    codes = pd.Index(sec_names).get_indexer(per_tk).astype(np.int16)
    return codes, sec_names


def _fetch_live_prospective_panel(
    as_of_date: date,
    settings: Settings,
    *,
    root: Path | None = None,
    unseal_final_test_period: bool = False,
) -> ResearchPanel:
    """Build a trailing live panel ending at `as_of_date` (`> 2026-07-08`), clamped after the sealed window."""
    if as_of_date <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(
            f"prospective live mode requires as_of_date > {HISTORICAL_SEALED_END.isoformat()}; got {as_of_date}"
        )
    alpaca = AlpacaSource(settings)
    assets = alpaca.fetch_ticker_universe()
    asset_syms = {
        s for s in (set(assets["symbol"].astype(str).str.upper()) - {"SPY", "QQQ"})
        if len(s) <= 5 and s.replace(".", "").isalpha()
    }
    roster_pq = (root / WIDER_UNIVERSE_DIRECTORY / "wider_roster_spans.parquet") if root is not None else None
    if roster_pq is not None and roster_pq.is_file():
        r_df = cast(Any, pq).read_table(roster_pq, columns=["ticker"], use_threads=False).to_pandas()
        sec_common = sorted({str(t).upper().strip() for t in r_df["ticker"]} & asset_syms)
        symbols = sec_common[:2000] if len(sec_common) >= 500 else sorted(asset_syms)[:1500]
    else:
        all_sorted = sorted(asset_syms)
        step = max(1, len(all_sorted) // 1500)
        symbols = all_sorted[::step][:1500]
    query_syms = ["SPY", "QQQ", *symbols]

    start_day = as_of_date - timedelta(days=410)
    if not unseal_final_test_period:
        # Strictly forbid reading any historical bar inside `2024-05-29..2026-07-08` while sealed!
        start_day = max(start_day, HISTORICAL_SEALED_END + timedelta(days=1))
    start_utc = datetime.combine(start_day, time(0, 0), tzinfo=UTC)
    end_utc = datetime.combine(as_of_date, time(23, 59, 59), tzinfo=UTC)

    bar_rows: list[dict[str, Any]] = []

    def _fetch_chunk(chunk: tuple[str, ...]) -> None:
        if not chunk:
            return
        token: str | None = None
        try:
            while True:
                page = alpaca.fetch_bars_page(
                    chunk, start_utc, end_utc, timeframe="1Day", page_token=token, asof=as_of_date, adjustment="all",
                )
                for sym, blist in page.bars.items():
                    for b in blist:
                        b_day = pd.Timestamp(str(b["t"])).tz_convert(MARKET_TIMEZONE).date()
                        if start_day <= b_day <= as_of_date:
                            bar_rows.append({
                                "ticker": sym,
                                "session": np.datetime64(b_day, "D"),
                                "open": float(b["o"]),
                                "high": float(b["h"]),
                                "low": float(b["l"]),
                                "close": float(b["c"]),
                                "volume": float(b["v"]),
                            })
                token = page.next_page_token
                if not token:
                    break
        except Exception as exc:
            if "status=400" in str(exc):
                if len(chunk) == 1:
                    return
                for single in chunk:
                    _fetch_chunk((single,))
                return
            raise

    for offset in range(0, len(query_syms), 45):
        _fetch_chunk(tuple(query_syms[offset : offset + 45]))
        guard("forward live panel fetch")

    df = pd.DataFrame(bar_rows).drop_duplicates(["ticker", "session"], keep="last")
    spy = df.loc[df["ticker"] == "SPY"].sort_values("session")
    qqq = df.loc[df["ticker"] == "QQQ"].sort_values("session")
    sessions = spy["session"].to_numpy().astype("datetime64[D]")
    qqq_al = pd.DataFrame({"session": sessions}).merge(qqq[["session", "open", "close"]], on="session", how="left")

    stocks_df = df.loc[~df["ticker"].isin(["SPY", "QQQ"]) & df["session"].isin(sessions)]
    sec_tickers = tuple(sorted(stocks_df["ticker"].unique()))
    sec_ids = tuple(f"ticker:{tk}" for tk in sec_tickers)
    col_idx = pd.Index(sec_tickers)

    shape = (len(sessions), len(sec_ids))
    r_pos = np.searchsorted(sessions, stocks_df["session"].to_numpy().astype("datetime64[D]"))
    c_pos = col_idx.get_indexer(stocks_df["ticker"])
    mats: dict[str, npt.NDArray[np.float32]] = {}
    for f_name in ("open", "high", "low", "close", "volume"):
        m = np.full(shape, np.nan, dtype=np.float32)
        m[r_pos, c_pos] = stocks_df[f_name].to_numpy(dtype=np.float32)
        mats[f_name] = m

    req_win = min(len(sessions), 252)
    min_valid = max(60, int(0.95 * req_win))
    fin_win = (
        (np.isfinite(mats["close"][-req_win:]) & (mats["close"][-req_win:] > 0)).sum(axis=0) >= min_valid
        if len(sessions) >= 60
        else np.zeros(len(sec_ids), dtype=bool)
    )
    dv_slice = mats["close"][-60:].astype(np.float64) * mats["volume"][-60:].astype(np.float64)
    dv_fin = np.isfinite(dv_slice)
    dv_cnt = dv_fin.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dvol_60 = np.where(dv_cnt > 0, np.where(dv_fin, dv_slice, 0.0).sum(axis=0) / np.maximum(dv_cnt, 1), np.nan)
    elig_last = fin_win & np.isfinite(dvol_60) & np.isfinite(mats["close"][-1])
    above_5 = elig_last & (mats["close"][-1] >= 5.0)
    if int(above_5.sum()) >= 50:
        elig_last = above_5
    if int(elig_last.sum()) > 1000:
        cutoff_dv = float(np.sort(dvol_60[elig_last])[-1000])
        elig_last = elig_last & (dvol_60 >= cutoff_dv)
    eligible = np.zeros(shape, dtype=bool)
    eligible[-1] = elig_last

    if root is not None:
        sec_codes_row, sector_names = _lookup_local_sectors_by_ticker(root, sec_tickers)
        sector_mat = np.broadcast_to(sec_codes_row[None, :], shape).copy()
    else:
        sector_mat = np.zeros(shape, dtype=np.int16)
        sector_names = ("Unknown",)

    return ResearchPanel(
        sessions=sessions,
        security_ids=sec_ids,
        open=mats["open"],
        high=mats["high"],
        low=mats["low"],
        close=mats["close"],
        volume=mats["volume"],
        eligible=eligible,
        sector=sector_mat,
        sector_names=sector_names,
        spy_open=spy["open"].to_numpy(dtype=np.float32),
        spy_close=spy["close"].to_numpy(dtype=np.float32),
        qqq_open=qqq_al["open"].to_numpy(dtype=np.float32),
        qqq_close=qqq_al["close"].to_numpy(dtype=np.float32),
        development_end=as_of_date,
        load_report={"mode": "live_prospective_alpaca_sip", "as_of_date": as_of_date.isoformat()},
    )


def parse_finviz_elite_news_csv(
    csv_text: str,
    *,
    ticker: str,
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
            "security_id": f"ticker:{symbol}",
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
            "security_id": f"ticker:{ticker}",
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
    unseal_final_test_period: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Collect & classify live Finviz Elite, Alpaca (`fiscal.ai`), and SEC catalysts `<= close_utc(scored_session)`."""
    scored_day = date.fromisoformat(str(scores_df["session_date"].iloc[0]))
    if scored_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(f"cannot collect live catalysts on or before {HISTORICAL_SEALED_END.isoformat()}")

    cutoff_utc = datetime.combine(scored_day, time(16, 0), tzinfo=ZoneInfo(MARKET_TIMEZONE)).astimezone(UTC)
    window_start_utc = cutoff_utc - timedelta(days=30)
    if not unseal_final_test_period:
        window_start_utc = max(window_start_utc, datetime(2026, 7, 9, 0, 0, tzinfo=UTC))

    rank_col = (
        "score_beta_neutral_price_12"
        if "score_beta_neutral_price_12" in scores_df.columns and scores_df["score_beta_neutral_price_12"].notna().any()
        else "score_price_equal_12"
    )
    ordered = scores_df.sort_values(rank_col, ascending=False, na_position="last")
    top_sec_ids = ordered["security_id"].head(max(1, int(catalyst_top_k))).astype(str).tolist()
    top_tickers = [sid.split(":", 1)[1] if ":" in sid else sid for sid in top_sec_ids]

    raw_events: list[dict[str, Any]] = []
    company_meta: dict[str, tuple[str, str | None]] = {}
    source_errors: dict[str, int] = {"finviz": 0, "alpaca": 0, "sec": 0}

    # 1. SEC EDGAR ticker -> CIK mapping + recent submissions (`8-K`, `SC 13D/A`, `S-1/S-3/424B`, `NT 10-K/Q`)
    if settings.sec_user_agent.strip():
        sec = SecSource(settings)
        tk_to_cik_title: dict[str, tuple[str, str]] = {}
        try:
            ct_payload = sec.client.get_json(SEC_COMPANY_TICKERS_URL)
            if isinstance(ct_payload, dict):
                for entry in ct_payload.values():
                    if isinstance(entry, dict) and entry.get("ticker") and entry.get("cik_str") is not None:
                        tk_u = str(entry["ticker"]).upper().strip()
                        tk_to_cik_title[tk_u] = (str(entry["cik_str"]).zfill(10), str(entry.get("title") or "").strip())
        except Exception:
            source_errors["sec"] += 1

        for tk in top_tickers:
            mapped = tk_to_cik_title.get(tk) or tk_to_cik_title.get(tk.replace(".", "-"))
            if mapped is None:
                continue
            cik_10, fallback_title = mapped
            try:
                sub = sec.client.get_json(sec.submissions_url.format(cik=cik_10))
                if not isinstance(sub, dict):
                    continue
                s_rows, comp_name, comp_avail = _extract_live_sec_catalysts_and_company(
                    sub,
                    ticker=tk,
                    cik=cik_10,
                    fallback_company=fallback_title,
                    window_start_utc=window_start_utc,
                    cutoff_utc=cutoff_utc,
                )
                company_meta[tk] = (comp_name, comp_avail)
                raw_events.extend(s_rows)
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
                            "security_id": f"ticker:{tk}",
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
        "source_errors": source_errors,
    }


def resolve_latest_completed_session(
    settings: Settings,
    *,
    now_utc: datetime | None = None,
    unseal_final_test_period: bool = False,
) -> date:
    """Resolve the latest regular US session whose `16:20 ET` post-close settlement cutoff has passed."""
    ts_utc = now_utc.astimezone(UTC) if now_utc is not None else datetime.now(UTC)
    ts_ny = ts_utc.astimezone(ZoneInfo(MARKET_TIMEZONE))
    cutoff_day = ts_ny.date() if ts_ny.time() >= POST_CLOSE_SETTLEMENT_ET else (ts_ny.date() - timedelta(days=1))
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


def record_forward_scores(
    *,
    root: Path,
    as_of_date: date | None = None,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    output_parent: Path = FORWARD_RECORD_DIRECTORY,
    live_prospective: bool = False,
    include_live_catalysts: bool = True,
    include_live_options: bool = True,
    catalyst_top_k: int = DEFAULT_CATALYST_TOP_K,
    skip_existing: bool = False,
    unseal_final_test_period: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Publish one immutable dated forward-score snapshot (plus live catalysts & OPRA options) and append to log."""
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

    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(RECORD_FORWARD_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("forward score recorder start")
        cfg = settings or get_settings()
        resolved_date = (
            resolve_latest_completed_session(cfg, unseal_final_test_period=unseal_final_test_period)
            if as_of_date is None
            else as_of_date
        )
        dest_parent = paths.write(paths.identity(output_parent))
        pre_manifest = dest_parent / resolved_date.isoformat() / "_manifest.json"
        if skip_existing and pre_manifest.is_file():
            existing_pre = cast(dict[str, Any], json.loads(pre_manifest.read_text(encoding="utf-8")))
            return {**existing_pre, "status": "already_recorded"}

        if live_prospective:
            loaded_panel = _fetch_live_prospective_panel(
                resolved_date, cfg, root=root, unseal_final_test_period=unseal_final_test_period,
            )
        else:
            loaded_panel = load_research_panel(
                paths.directory(panel),
                development_end=resolved_date,
                unseal_final_test_period=unseal_final_test_period,
            )

        scores_df = _score_panel_last_session(loaded_panel)
        if scores_df.empty:
            raise ValueError(f"no eligible stocks scored on {resolved_date.isoformat()}")
        scored_session = str(scores_df["session_date"].iloc[0])

        dest_dir = dest_parent / scored_session
        if dest_dir.exists():
            if skip_existing and (dest_dir / "_manifest.json").is_file():
                existing_post = cast(dict[str, Any], json.loads((dest_dir / "_manifest.json").read_text(encoding="utf-8")))
                return {**existing_post, "status": "already_recorded"}
            raise FileExistsError(f"forward score snapshot already exists for session {scored_session}: {dest_dir}")

        catalysts_df: pd.DataFrame | None = None
        catalyst_summary: dict[str, Any] | None = None
        if live_prospective and include_live_catalysts:
            scores_df, catalysts_df, catalyst_summary = collect_live_prospective_catalysts(
                loaded_panel,
                scores_df,
                cfg,
                catalyst_top_k=catalyst_top_k,
                unseal_final_test_period=unseal_final_test_period,
            )

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

        dest_parent.mkdir(parents=True, exist_ok=True)
        staging = dest_parent / f".{scored_session}.{uuid4().hex}.partial"
        staging.mkdir(parents=True)
        manifest: dict[str, Any] = {
            "status": "recorded",
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "requested_as_of_date": "latest" if as_of_date is None else resolved_date.isoformat(),
            "resolved_as_of_date": resolved_date.isoformat(),
            "scored_session": scored_session,
            "live_prospective": live_prospective,
            "sealed_holdout_window": "2024-05-29..2026-07-08",
            "sealed_holdout_untouched": not unseal_final_test_period,
            "frozen_prospective_candidates": list(FROZEN_PROSPECTIVE_CANDIDATES),
            "eligible_stocks_scored": len(scores_df),
            "catalyst_summary": catalyst_summary,
            "options_summary": options_summary,
            "code": git_identity(root),
            "label_notice": "research forward score snapshot (zero future outcomes read)",
        }
        try:
            cast(Any, pq).write_table(pa.Table.from_pandas(scores_df, preserve_index=False), staging / "scores.parquet")
            if catalysts_df is not None:
                cast(Any, pq).write_table(
                    pa.Table.from_pandas(catalysts_df, preserve_index=False), staging / "catalysts.parquet",
                )
            if options_df is not None and len(options_df):
                cast(Any, pq).write_table(
                    pa.Table.from_pandas(options_df, preserve_index=False), staging / "options_metrics.parquet",
                )
            (staging / "_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
            )
            os.replace(staging, dest_dir)
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
    "collect_live_prospective_catalysts",
    "parse_finviz_elite_news_csv",
    "record_forward_scores",
    "redact_finviz_auth",
    "resolve_latest_completed_session",
]
