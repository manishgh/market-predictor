"""Zero-cost ($0) survivorship-checked wider US common-stock universe (`2016..2024-05-28`) for Step 5b.

Builds a point-in-time liquid US common-stock panel (~Top 1,200 each session by trailing 60-day dollar
volume, plus the mid/small-cap ex-S&P-500 slice) using only free SEC EDGAR (`SecSource`, 6 req/s) and
existing Alpaca SIP (`AlpacaSource`), without paid vendor data and without reading any row dated after
`development_end` (`2024-05-28`):

1. Enumerates active and later-delisted/bankrupt/acquired US common-stock issuers via SEC EDGAR XBRL
   frames (`us-gaap/Assets/USD` >= $600M and `dei/EntityCommonStockSharesOutstanding` >= 5M shares across
   `CY2016Q2I..CY2024Q1I`, dropping any accession filed after `2024` or period ending after `2024-04-15`),
   unioned with all mapped S&P 500 CIKs.
2. Resolves historical tickers and valid `[valid_from, valid_to]` spans (`<= 2024-05-28`) via SEC
   `company_tickers_exchange.json` (including OTC `Q`-suffix bankruptcies), retained S&P 500 identity
   relations, Alpaca corporate-action `name_changes` (`<= 2024-05-28`), and SEC `submissions/CIK{cik}.json`
   (`tickers`, `10-K/10-Q/8-K` filename ticker prefixes, and Form 4 XML `<issuerTradingSymbol>` fallback),
   quarantining overlapping ticker reuse so a symbol never crosses issuers.
3. Resumably collects `2016-01-04..valid_to` adjusted (`all`) and unadjusted (`raw`) SIP daily bars,
   determines the exact point-in-time Top-1,200 liquid common stocks (`unadjusted_close >= $5.00`,
   `>= 252` trailing closes, ranked by trailing 60-session dollar volume on `2018-05-29..2024-05-28`),
   and fetches SEC `submissions` (`sic` sector + all `8-K` item catalyst events `<= 2024-05-28`) and
   `company_facts.json` strictly for CIKs that enter the eligible universe.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, Literal, cast

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
from market_predictor.research_lane.company_facts import (
    COMPANY_FACTS_DIRECTORY,
    COMPANY_FACTS_FILE,
    COMPANY_FACTS_SOURCES,
)
from market_predictor.research_lane.free_backfill import (
    BACKFILL_DIRECTORY,
    EARLY_DECISION_START,
    EXTRA_FACTS_DIR,
    MINIMUM_WARMUP_CLOSES,
    NEWS_DECISION_START,
    SEC_IDENTITY_RELATIONS,
    _conservative_edgar_availability_utc,
    _fetch_alpaca_daily_batch,
    _read_panel_membership,
    _trailing_finite_count,
    existing_company_facts_paths,
    resolve_all_sp500_ciks,
)
from market_predictor.research_lane.news_catalysts import EARLY_NEWS_FAMILY_DIR, LATER_NEWS_FAMILY_DIR
from market_predictor.research_lane.panel import (
    BAR_DIRECTORY,
    BAR_MANIFEST,
    MARKET_TIMEZONE,
    PANEL_DIRECTORY,
    PRICE_FIELDS,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
    file_sha256,
)
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.sec_filings import FILINGS_DIRECTORY, FILINGS_FILE, event_rows
from market_predictor.research_lane.unadjusted_prices import UNADJUSTED_DIRECTORY
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.sec import SecSource

WIDER_UNIVERSE_DIRECTORY: Final = Path("data/research/research_lane/wider_us_universe")
COLLECT_WIDER_UNIVERSE_COMMAND: Final = "collect-research-lane-wider-universe"

MIN_FRAME_ASSETS_USD: Final = 600_000_000.0
MIN_FRAME_SHARES: Final = 5_000_000.0
MIN_UNADJUSTED_PRICE: Final = 5.0
MIN_DOLLAR_VOLUME_60D: Final = 1_000_000.0
TOP_LIQUID_COUNT: Final = 1_200

UniverseSubset = Literal["all_top1200", "mid_small_ex_sp500"]

_DOC_TICKER_RE: Final = re.compile(r"^([a-z]{1,5})[-_]\d{6,8}", re.IGNORECASE)
_FORM4_SYM_RE: Final = re.compile(r"<issuerTradingSymbol>\s*([A-Za-z0-9.\-]+)\s*</issuerTradingSymbol>")
_EXCLUDED_DOC_PREFIXES: Final = frozenset({
    "FORM", "DEF", "EX", "DOC", "D", "A", "R", "Q", "K", "S", "FILE", "MAIN", "BODY", "INDEX", "EXHIBIT",
})
_VALID_COMMON_EXCHANGES: Final = frozenset({"NYSE", "Nasdaq", "TXSE", "CBOE"})

WIDER_CATALYST_ITEMS: Final = (
    ("1.01", "rx_8k_101_material_agreement"),
    ("2.01", "rx_8k_201_acquisition"),
    ("2.02", "rx_8k_202_earnings"),
    ("2.05", "rx_8k_205_restructuring"),
    ("5.02", "rx_8k_502_officer_change"),
    ("7.01", "rx_8k_701_reg_fd"),
    ("8.01", "rx_8k_801_other_event"),
)
WIDER_STAGE2_FEATURE_NAMES: Final = (
    *(feat_name for _, feat_name in WIDER_CATALYST_ITEMS),
    "rx_catalyst_any",
    "catalyst_abn_vol_20",
    "catalyst_recency_21",
    "catalyst_unusual_count_5_60",
)


@dataclass(frozen=True)
class WiderUniverseData:
    """Survivorship-checked wider US common-stock panel, unadjusted closes, SEC releases/events, and CIK facts."""

    panel: ResearchPanel
    unadjusted_close: npt.NDArray[np.float64]
    releases_202: pd.DataFrame
    all_8k_events: pd.DataFrame
    security_cik_map: dict[str, str]
    company_facts_paths: dict[str, Path]
    sp500_member_mask: npt.NDArray[np.bool_]
    input_sha256: dict[str, str]
    report: dict[str, Any]


def sic_to_sector(sic_value: str | int | None) -> str:
    """Deterministic mapping of 4-digit SEC SIC codes into the 11 GICS-equivalent equity sectors."""
    digits = "".join(ch for ch in str(sic_value or "") if ch.isdigit())
    if not digits:
        return "Industrials"
    code = int(digits[:4])
    if (1300 <= code <= 1399) or (2900 <= code <= 2999) or (4600 <= code <= 4699):
        return "Energy"
    if (2830 <= code <= 2836) or (3840 <= code <= 3851) or (8000 <= code <= 8099):
        return "Health Care"
    if (3570 <= code <= 3579) or (3660 <= code <= 3679) or (3810 <= code <= 3829) or (7370 <= code <= 7379):
        return "Information Technology"
    if (4800 <= code <= 4899) or (7800 <= code <= 7849) or (2700 <= code <= 2749):
        return "Communication Services"
    if 4900 <= code <= 4949:
        return "Utilities"
    if code == 6798 or (6500 <= code <= 6599):
        return "Real Estate"
    if (6000 <= code <= 6499) or (6700 <= code <= 6799):
        return "Financials"
    if (
        (100 <= code <= 999)
        or (2000 <= code <= 2199)
        or (2840 <= code <= 2844)
        or (5140 <= code <= 5159)
        or (5400 <= code <= 5499)
        or code == 5912
    ):
        return "Consumer Staples"
    if (
        (1000 <= code <= 1299)
        or (1400 <= code <= 1499)
        or (2400 <= code <= 2499)
        or (2600 <= code <= 2699)
        or (2800 <= code <= 2899)
        or (3000 <= code <= 3499)
    ):
        return "Materials"
    if (
        (2200 <= code <= 2399)
        or (2500 <= code <= 2599)
        or (3710 <= code <= 3716)
        or code == 3751
        or (3900 <= code <= 3999)
        or (5010 <= code <= 5015)
        or (5200 <= code <= 5399)
        or (5500 <= code <= 5999)
        or (7000 <= code <= 7299)
        or (7500 <= code <= 7999)
    ):
        return "Consumer Discretionary"
    return "Industrials"


def _is_clean_common_symbol(raw_ticker: str) -> str | None:
    """Normalize common-stock ticker (`BRK-B` kept canonical); reject preferreds, units, warrants, rights."""
    sym = raw_ticker.strip().upper().replace(".", "-")
    if not sym:
        return None
    if "-" in sym:
        parts = sym.split("-")
        if len(parts) != 2 or not parts[0].isalpha() or parts[1] not in ("A", "B", "C"):
            return None
        return sym if len(parts[0]) <= 4 else None
    if not sym.isalpha() or len(sym) > 5:
        return None
    if len(sym) == 5 and sym[-1] in ("W", "U", "R", "P", "X", "Y", "F"):
        return None
    return sym


def quarantine_ticker_spans(spans: pd.DataFrame) -> pd.DataFrame:
    """Guarantee zero date overlap across different CIKs sharing a `ticker` (`valid_from <= valid_to`).

    Expected columns: `cik`, `ticker`, `valid_from`, `valid_to`, `source_priority` (lower = higher priority).
    When two distinct CIKs claim the same `ticker` with overlapping `[valid_from, valid_to]`:
    - The chronologically earlier span's `valid_to` is clamped to `later.valid_from - 1 day`.
    - Any resulting reversed span (`valid_from > valid_to`) is dropped.
    """
    if spans.empty:
        return spans.copy()
    df = spans.copy()
    df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
    df["valid_to"] = pd.to_datetime(df["valid_to"]).dt.date
    df = df.sort_values(["ticker", "valid_from", "source_priority", "cik"], kind="stable").reset_index(drop=True)

    kept_rows: list[dict[str, Any]] = []
    for _, grp in df.groupby("ticker", sort=False):
        recs = grp.to_dict(orient="records")
        # Merge duplicate rows of the exact same CIK on the same ticker first.
        merged_cik: list[dict[str, Any]] = []
        for r in recs:
            if merged_cik and merged_cik[-1]["cik"] == r["cik"]:
                merged_cik[-1]["valid_from"] = min(merged_cik[-1]["valid_from"], r["valid_from"])
                merged_cik[-1]["valid_to"] = max(merged_cik[-1]["valid_to"], r["valid_to"])
            else:
                merged_cik.append(dict(r))

        for idx in range(len(merged_cik)):
            cur = merged_cik[idx]
            if idx + 1 < len(merged_cik):
                nxt = merged_cik[idx + 1]
                if cur["valid_to"] >= nxt["valid_from"]:
                    cur["valid_to"] = nxt["valid_from"] - timedelta(days=1)
            if cur["valid_from"] <= cur["valid_to"]:
                kept_rows.append(cur)

    if not kept_rows:
        return df.iloc[:0].copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def compute_pit_liquidity_eligibility(
    close: npt.NDArray[np.float32],
    volume: npt.NDArray[np.float32],
    unadjusted_close: npt.NDArray[np.float64],
    in_span: npt.NDArray[np.bool_],
    sessions: npt.NDArray[np.datetime64],
    *,
    decision_start: date = EARLY_DECISION_START,
    min_warmup_closes: int = MINIMUM_WARMUP_CLOSES,
    min_unadjusted_price: float = MIN_UNADJUSTED_PRICE,
    min_dollar_volume_60d: float = MIN_DOLLAR_VOLUME_60D,
    top_n: int = TOP_LIQUID_COUNT,
) -> npt.NDArray[np.bool_]:
    """Point-in-time causal Top-`top_n` US common-stock eligibility at session `t` close (`<= t` strictly)."""
    n_sess, n_sec = close.shape
    trailing_cnt = _trailing_finite_count(close, min_warmup_closes)
    has_warmup = (trailing_cnt >= min_warmup_closes) & np.isfinite(close) & (close > 0.0)

    # Causal 60-session mean daily dollar volume (`close * volume` on `[t-59..t]`, requiring >= 45 valid bars).
    dv = close.astype(np.float64) * volume.astype(np.float64)
    dv_ok = np.isfinite(dv) & (dv >= 0.0)
    dv_clean = np.where(dv_ok, dv, 0.0)
    dv_csum = np.vstack([np.zeros((1, n_sec), dtype=np.float64), np.cumsum(dv_clean, axis=0)])
    cnt_csum = np.vstack([np.zeros((1, n_sec), dtype=np.int32), np.cumsum(dv_ok.astype(np.int32), axis=0)])
    r_idx = np.arange(1, n_sess + 1)
    r_prev = np.maximum(0, r_idx - 60)
    win_sum = dv_csum[r_idx] - dv_csum[r_prev]
    win_cnt = cnt_csum[r_idx] - cnt_csum[r_prev]
    with np.errstate(invalid="ignore", divide="ignore"):
        dvol_60 = np.where(win_cnt >= 45, win_sum / np.maximum(win_cnt, 1), np.nan)

    # Unadjusted price >= $5.00 where raw close is observed; if raw close is not yet loaded (Stage B pre-screen),
    # pass `np.inf` in `unadjusted_close`.
    px_ok = np.isfinite(unadjusted_close) & (unadjusted_close >= min_unadjusted_price)
    qualifies = in_span & has_warmup & px_ok & np.isfinite(dvol_60) & (dvol_60 >= min_dollar_volume_60d)

    first_row = int(np.searchsorted(sessions, np.datetime64(decision_start, "D"), side="left"))
    eligible = np.zeros((n_sess, n_sec), dtype=bool)
    for r in range(first_row, n_sess):
        q_cols = np.flatnonzero(qualifies[r])
        if q_cols.size == 0:
            continue
        if q_cols.size <= top_n:
            eligible[r, q_cols] = True
        else:
            row_dv = dvol_60[r, q_cols]
            # Stable top-`top_n` selection (ties broken deterministically by column index).
            order = np.argsort(-row_dv, kind="stable")[:top_n]
            eligible[r, q_cols[order]] = True
    return eligible


def _parse_submissions_payload(
    sec: SecSource,
    cik: str,
    payload: dict[str, Any],
    *,
    development_end: date,
    fetch_overflow_8k: bool,
    allow_form4_xml: bool,
) -> dict[str, Any]:
    """Extract `(ticker, sic, first_filing, last_filing, events_8k)` strictly `<= development_end`."""
    end_iso = development_end.isoformat()
    cutoff_utc = (pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")

    sic = str(payload.get("sic", "") or "").strip()
    name = str(payload.get("name", "") or "").strip()
    raw_tickers = payload.get("tickers", [])
    raw_exchanges = payload.get("exchanges", [])
    chosen_ticker: str | None = None
    if isinstance(raw_tickers, list):
        for idx, tk in enumerate(raw_tickers):
            ex = str(raw_exchanges[idx]) if isinstance(raw_exchanges, list) and idx < len(raw_exchanges) else ""
            if ex in ("OTC",):
                s_str = str(tk).strip().upper()
                if len(s_str) in (4, 5) and s_str.endswith("Q"):
                    cand = _is_clean_common_symbol(s_str[:-1])
                    if cand:
                        chosen_ticker = cand
                        break
                continue
            cand = _is_clean_common_symbol(str(tk))
            if cand:
                chosen_ticker = cand
                break

    filings = payload.get("filings", {})
    sections: list[dict[str, Any]] = []
    if isinstance(filings, dict):
        rec = filings.get("recent")
        if isinstance(rec, dict):
            sections.append(rec)
        if fetch_overflow_8k:
            for desc in filings.get("files", []):
                if not isinstance(desc, dict):
                    continue
                f_from = str(desc.get("filingFrom", ""))
                f_to = str(desc.get("filingTo", ""))
                fname = str(desc.get("name", "")).strip()
                if fname and f_from <= end_iso and f_to >= "2018-01-01":
                    try:
                        hist = sec.client.get_json(sec.historical_submission_url.format(name=fname))
                        if isinstance(hist, dict):
                            sub_b = hist.get("filings", hist)
                            if isinstance(sub_b, dict):
                                sections.append(sub_b)
                    except Exception:
                        pass

    doc_counter: Counter[str] = Counter()
    first_f: str | None = None
    last_f: str | None = None
    form4_candidate: tuple[str, str] | None = None
    events_8k: list[dict[str, Any]] = []
    seen_accn: set[str] = set()

    for block in sections:
        forms = block.get("form", [])
        f_dates = block.get("filingDate", [])
        accns = block.get("accessionNumber", [])
        accepts = block.get("acceptanceDateTime", [])
        items_col = block.get("items", [])
        docs = block.get("primaryDocument", [])
        n = min(len(forms), len(f_dates), len(accns))
        for idx in range(n):
            fd = str(f_dates[idx]).strip()
            if not fd or fd > end_iso:
                continue
            fm = str(forms[idx]).strip().upper()
            if fm in ("10-K", "10-Q", "8-K", "20-F", "4"):
                first_f = fd if first_f is None else min(first_f, fd)
                last_f = fd if last_f is None else max(last_f, fd)
            if chosen_ticker is None and idx < len(docs) and fm in ("10-K", "10-Q", "8-K"):
                m = _DOC_TICKER_RE.match(str(docs[idx]).strip())
                if m:
                    cand_p = _is_clean_common_symbol(m.group(1))
                    if cand_p and cand_p not in _EXCLUDED_DOC_PREFIXES:
                        doc_counter[cand_p] += 1
            if chosen_ticker is None and form4_candidate is None and fm == "4" and idx < len(docs) and fd >= "2015-01-01":
                raw_doc = str(docs[idx]).strip().split("/")[-1]
                if raw_doc.lower().endswith(".xml"):
                    form4_candidate = (str(accns[idx]).strip(), raw_doc)

            if fm == "8-K" and fd >= "2018-01-01" and idx < len(items_col) and idx < len(accepts):
                accn = str(accns[idx]).strip()
                if not accn or accn in seen_accn:
                    continue
                raw_it = items_col[idx]
                item_set = (
                    {str(x).strip() for x in raw_it if str(x).strip()}
                    if isinstance(raw_it, list)
                    else {x.strip() for x in str(raw_it or "").split(",") if x.strip()}
                )
                if not item_set:
                    continue
                avail = _conservative_edgar_availability_utc(accepts[idx], fd)
                if avail >= cutoff_utc:
                    continue
                seen_accn.add(accn)
                events_8k.append({
                    "sec_cik": cik,
                    "accession_number": accn,
                    "sec_form": "8-K",
                    "item_codes": ",".join(sorted(item_set)),
                    "accepted_at_utc": avail.isoformat(),
                    "available_at_utc": avail.isoformat(),
                })

    if chosen_ticker is None and doc_counter:
        chosen_ticker = doc_counter.most_common(1)[0][0]

    if chosen_ticker is None and allow_form4_xml and form4_candidate is not None:
        acc_f4, xml_name = form4_candidate
        url_f4 = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_f4.replace('-', '')}/{xml_name}"
        try:
            resp = sec.client.get_bytes_with_metadata(url_f4, maximum_body_bytes=512 * 1024)
            text_xml = resp.body.decode("utf-8", errors="ignore")
            m4 = _FORM4_SYM_RE.search(text_xml)
            if m4:
                chosen_ticker = _is_clean_common_symbol(m4.group(1))
        except Exception:
            pass

    return {
        "cik": cik,
        "name": name,
        "ticker": chosen_ticker,
        "sic": sic,
        "sector": sic_to_sector(sic),
        "first_filing_date": first_f,
        "last_filing_date": last_f,
        "events_8k": events_8k,
    }


def collect_wider_us_universe(
    *,
    root: Path,
    development_end: date,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    sp500_backfill: Path = BACKFILL_DIRECTORY,
    output: Path = WIDER_UNIVERSE_DIRECTORY,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Resumably build the $0 survivorship-checked wider US liquid common-stock universe (`<= development_end`)."""
    check_development_end(development_end)
    cfg = settings or get_settings()
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(COLLECT_WIDER_UNIVERSE_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("wider universe collection start")
        dest = paths.write(paths.identity(output))
        dest.mkdir(parents=True, exist_ok=True)
        adj_dir = dest / "bars_adjusted"
        raw_dir = dest / "bars_raw"
        facts_dir = dest / "company_facts"
        adj_dir.mkdir(exist_ok=True)
        raw_dir.mkdir(exist_ok=True)
        facts_dir.mkdir(exist_ok=True)

        sec = SecSource(cfg)
        alpaca = AlpacaSource(cfg)
        panel_dir = paths.directory(panel)

        # Phase 1: Candidate CIK enumeration (`CY2016..CY2024Q1I`) + delisted historical ticker resolution.
        frames_cache_path = dest / "xbrl_frame_candidates.json"
        if frames_cache_path.is_file():
            cand_meta = json.loads(frames_cache_path.read_text(encoding="utf-8"))
        else:
            asset_ciks: dict[str, dict[str, Any]] = {}
            share_ciks: set[str] = set()
            frame_tags = [*(f"CY{yr}Q2I" for yr in range(2016, 2024)), "CY2024Q1I"]
            for ftag in frame_tags:
                fa = sec.client.get_json(f"https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/{ftag}.json")
                if isinstance(fa, dict):
                    for r in fa.get("data", []):
                        if not isinstance(r, dict):
                            continue
                        acc_parts = str(r.get("accn", "")).split("-")
                        if len(acc_parts) < 2 or not acc_parts[1].isdigit():
                            continue
                        acc_yr = 2000 + int(acc_parts[1])
                        end_d = str(r.get("end", ""))
                        if acc_yr > development_end.year or end_d > "2024-04-15":
                            continue
                        if float(r.get("val", 0.0) or 0.0) >= MIN_FRAME_ASSETS_USD:
                            c_str = str(r["cik"]).zfill(10)
                            cur = asset_ciks.get(c_str)
                            asset_ciks[c_str] = {
                                "name": str(r.get("entityName", "")),
                                "first_end": end_d if cur is None else min(str(cur["first_end"]), end_d),
                                "last_end": end_d if cur is None else max(str(cur["last_end"]), end_d),
                            }
                fs = sec.client.get_json(
                    f"https://data.sec.gov/api/xbrl/frames/dei/EntityCommonStockSharesOutstanding/shares/{ftag}.json"
                )
                if isinstance(fs, dict):
                    for r in fs.get("data", []):
                        if not isinstance(r, dict):
                            continue
                        acc_parts = str(r.get("accn", "")).split("-")
                        if len(acc_parts) < 2 or not acc_parts[1].isdigit():
                            continue
                        acc_yr = 2000 + int(acc_parts[1])
                        end_d = str(r.get("end", ""))
                        if acc_yr > development_end.year or end_d > "2024-04-15":
                            continue
                        if float(r.get("val", 0.0) or 0.0) >= MIN_FRAME_SHARES:
                            share_ciks.add(str(r["cik"]).zfill(10))

            exch_raw = sec.client.get_json("https://www.sec.gov/files/company_tickers_exchange.json")
            exch_rows = exch_raw.get("data", []) if isinstance(exch_raw, dict) else []
            exch_map: dict[str, list[tuple[str, str]]] = {}
            for row_e in exch_rows:
                if isinstance(row_e, list) and len(row_e) >= 4:
                    exch_map.setdefault(str(row_e[0]).zfill(10), []).append((str(row_e[2]), str(row_e[3])))

            transitions = alpaca.fetch_security_transitions(date(2016, 1, 1), development_end)
            name_changes = transitions.loc[
                transitions["transition_type"].eq("name_change") & transitions["identity_continuity"].eq(True)
            ].to_dict(orient="records")

            cand_meta = {
                "asset_and_share_ciks": {c: asset_ciks[c] for c in sorted(set(asset_ciks) & share_ciks)},
                "exchange_map": exch_map,
                "name_changes": name_changes,
            }
            frames_cache_path.write_text(json.dumps(cand_meta, sort_keys=True), encoding="utf-8")
        guard("wider universe xbrl frames cached")

        membership = _read_panel_membership(panel_dir, development_end)
        sp500_sec_ids = tuple(sorted(membership["security_id"].astype(str).unique()))
        sp500_cik_map, _ = resolve_all_sp500_ciks(
            sp500_sec_ids,
            identity_relations_path=paths.read(SEC_IDENTITY_RELATIONS),
            filings_path=paths.read(FILINGS_DIRECTORY / FILINGS_FILE),
            development_end=development_end,
        )
        sp500_ciks = set(sp500_cik_map.values())

        # Known `(cik, ticker, valid_from, valid_to)` from S&P 500 membership + `sec_identity_relations`.
        sp500_tk_spans: dict[tuple[str, str], tuple[date, date, str]] = {}
        mem_with_cik = membership.loc[membership["security_id"].isin(sp500_cik_map)].copy()
        mem_with_cik["cik"] = mem_with_cik["security_id"].map(sp500_cik_map)
        for (c_val, tk_val), grp_sp in mem_with_cik.groupby(["cik", "ticker"]):
            clean_tk = _is_clean_common_symbol(str(tk_val))
            if clean_tk is None:
                continue
            d_min = date(2016, 1, 4)
            d_max = min(pd.Timestamp(grp_sp["session"].max()).date() + timedelta(days=30), development_end)
            sec_str = str(grp_sp["sector"].dropna().iloc[0]) if len(grp_sp["sector"].dropna()) else "Industrials"
            sp500_tk_spans[(str(c_val), clean_tk)] = (d_min, d_max, sec_str)
        del mem_with_cik

        cand_ciks = sorted(set(cand_meta["asset_and_share_ciks"]) | sp500_ciks)
        exch_map_loaded: dict[str, list[tuple[str, str]]] = cand_meta["exchange_map"]

        # Load resumable submissions metadata cache (`cik -> parsed dict`).
        sub_cache_path = dest / "submissions_meta.jsonl"
        sub_cache: dict[str, dict[str, Any]] = {}
        if sub_cache_path.is_file():
            for line in sub_cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    sub_cache[str(obj["cik"])] = obj

        # Resolve delisted/unmapped candidate CIKs first (`~750` CIKs not on active NYSE/Nasdaq and not in S&P 500).
        active_exch_ticker: dict[str, str] = {}
        otc_bankrupt_ticker: dict[str, str] = {}
        for c_id in cand_ciks:
            entries = exch_map_loaded.get(c_id, [])
            for tk_raw, ex_raw in entries:
                if ex_raw in _VALID_COMMON_EXCHANGES:
                    clean_s = _is_clean_common_symbol(tk_raw)
                    if clean_s:
                        active_exch_ticker[c_id] = clean_s
                        break
            if c_id not in active_exch_ticker:
                for tk_raw, ex_raw in entries:
                    s_up = tk_raw.strip().upper()
                    if ex_raw == "OTC" and len(s_up) in (4, 5) and s_up.endswith("Q"):
                        clean_q = _is_clean_common_symbol(s_up[:-1])
                        if clean_q:
                            otc_bankrupt_ticker[c_id] = clean_q
                            break

        need_delisted_sub = [
            c_id for c_id in cand_ciks
            if c_id not in active_exch_ticker
            and c_id not in sp500_ciks
            and c_id not in sub_cache
        ]
        if need_delisted_sub:
            with sub_cache_path.open("a", encoding="utf-8") as handle:
                for idx_d, c_id in enumerate(need_delisted_sub):
                    try:
                        sub_json = sec.client.get_json(sec.submissions_url.format(cik=c_id))
                        parsed = (
                            _parse_submissions_payload(
                                sec, c_id, sub_json, development_end=development_end,
                                fetch_overflow_8k=True, allow_form4_xml=True,
                            )
                            if isinstance(sub_json, dict)
                            else {"cik": c_id, "ticker": None, "events_8k": []}
                        )
                    except Exception as exc:
                        if "404" in str(exc):
                            parsed = {"cik": c_id, "ticker": None, "events_8k": []}
                        else:
                            raise
                    if parsed.get("ticker") is None and c_id in otc_bankrupt_ticker:
                        parsed["ticker"] = otc_bankrupt_ticker[c_id]
                    sub_cache[c_id] = parsed
                    handle.write(json.dumps(parsed, sort_keys=True) + "\n")
                    if (idx_d + 1) % 30 == 0:
                        handle.flush()
                        release_unused_process_memory()
                        guard("wider universe delisted submissions")

        # Assemble `(cik, ticker, valid_from, valid_to, source_priority)` spans and apply rename chains + quarantine.
        new_to_old: dict[str, list[tuple[str, date]]] = {}
        for nc in cand_meta["name_changes"]:
            old_s = _is_clean_common_symbol(str(nc.get("old_symbol", "")))
            new_s = _is_clean_common_symbol(str(nc.get("new_symbol", "")))
            eff_str = str(nc.get("effective_date", ""))
            if old_s and new_s and old_s != new_s and "2016-01-04" < eff_str <= development_end.isoformat():
                new_to_old.setdefault(new_s, []).append((old_s, date.fromisoformat(eff_str)))

        raw_spans: list[dict[str, Any]] = []
        for (c_sp, tk_sp), (d0, d1, _) in sp500_tk_spans.items():
            raw_spans.append({
                "cik": c_sp, "ticker": tk_sp, "valid_from": d0, "valid_to": d1, "source_priority": 1,
            })
        for c_id, tk_act in active_exch_ticker.items():
            f_info = cand_meta["asset_and_share_ciks"].get(c_id)
            first_y = int(str(f_info["first_end"])[:4]) if f_info else 2016
            v_from = date(2016, 1, 4) if first_y <= 2017 else date(max(2016, first_y - 1), 1, 1)
            if tk_act in new_to_old:
                # Chain single rename (`old_sym` before `eff_d`, `tk_act` from `eff_d` onward).
                old_sym, eff_d = sorted(new_to_old[tk_act], key=lambda x: x[1])[-1]
                if v_from < eff_d:
                    raw_spans.append({
                        "cik": c_id, "ticker": old_sym, "valid_from": v_from,
                        "valid_to": eff_d - timedelta(days=1), "source_priority": 2,
                    })
                    v_from = eff_d
            raw_spans.append({
                "cik": c_id, "ticker": tk_act, "valid_from": v_from, "valid_to": development_end, "source_priority": 2,
            })
        for c_id, meta_d in sub_cache.items():
            tk_d = meta_d.get("ticker")
            last_fd = meta_d.get("last_filing_date")
            first_fd = meta_d.get("first_filing_date")
            if not tk_d or not last_fd or str(last_fd) < "2017-01-01":
                continue
            v0 = max(date(2016, 1, 4), date.fromisoformat(str(first_fd)) - timedelta(days=365)) if first_fd else date(2016, 1, 4)
            v1 = min(development_end, date.fromisoformat(str(last_fd)) + timedelta(days=45))
            if v0 <= v1:
                raw_spans.append({
                    "cik": c_id, "ticker": str(tk_d), "valid_from": v0, "valid_to": v1, "source_priority": 3,
                })

        roster_df = quarantine_ticker_spans(pd.DataFrame(raw_spans))
        roster_path = dest / "roster_spans.parquet"
        cast(Any, pq).write_table(pa.Table.from_pandas(roster_df, preserve_index=False), roster_path)
        release_unused_process_memory()

        # Phase 2: Fetch adjusted (`all`) daily SIP bars for non-S&P-500 candidate tickers in resumable shards.
        bar_manifest = json.loads((panel_dir / BAR_MANIFEST).read_text(encoding="utf-8"))["artifacts"]
        sp500_bar_tickers = {str(item["ticker"]) for item in bar_manifest}
        extra_cand_spans = roster_df.loc[~roster_df["ticker"].isin(sp500_bar_tickers)].copy()

        # Group extra tickers into deterministic shards of 90 tickers by `valid_to` year so `asof = max(valid_to)`.
        extra_by_tk = (
            extra_cand_spans.groupby("ticker", as_index=False)
            .agg({"valid_from": "min", "valid_to": "max"})
            .sort_values(["valid_to", "ticker"], kind="stable")
            .reset_index(drop=True)
        )
        shard_size = 90
        for s_idx, offset in enumerate(range(0, len(extra_by_tk), shard_size)):
            shard_file = adj_dir / f"shard_{s_idx:03d}.parquet"
            if shard_file.is_file():
                continue
            sub_sh = extra_by_tk.iloc[offset : offset + shard_size]
            # Group within shard by year of `valid_to` so delisted symbols query an active `asof <= valid_to`!
            sh_frames: list[pd.DataFrame] = []
            sub_sh_years = sub_sh["valid_to"].map(lambda d: d.year)
            for yr_end, grp_y in sub_sh.groupby(sub_sh_years, sort=True):
                del yr_end
                # Further group if `valid_to < development_end` by exact month-end / max valid_to in small sub-buckets.
                still_open = grp_y.loc[grp_y["valid_to"] >= development_end]
                closed_early = grp_y.loc[grp_y["valid_to"] < development_end]
                if len(still_open):
                    sh_frames.append(
                        _fetch_alpaca_daily_batch(
                            alpaca, sorted(still_open["ticker"].astype(str)),
                            date(2016, 1, 4), development_end, adjustment="all", asof_day=development_end,
                        )
                    )
                for v_end_d, grp_cl in closed_early.groupby("valid_to", sort=True):
                    sh_frames.append(
                        _fetch_alpaca_daily_batch(
                            alpaca, sorted(grp_cl["ticker"].astype(str)),
                            date(2016, 1, 4), v_end_d, adjustment="all", asof_day=v_end_d,
                        )
                    )
            sh_df = (
                pd.concat(sh_frames, ignore_index=True).drop_duplicates(["ticker", "session_date"], keep="last")
                if sh_frames
                else pd.DataFrame(columns=["ticker", "session_date", "open", "high", "low", "close", "volume"])
            )
            cast(Any, pq).write_table(pa.Table.from_pandas(sh_df, preserve_index=False), shard_file)
            del sh_frames, sh_df
            release_unused_process_memory()
            guard(f"wider universe adj shard {s_idx}")

        # Pre-screen candidate CIKs via Phase-2 adjusted bars (`dvol_60` top 1,350 before $5 raw filter) to identify
        # the exact ~1,650 CIKs needing unadjusted raw bars, SEC `submissions` (`sic` + `8-K`), and `company_facts`.
        selected_ciks_path = dest / "selected_ciks.json"
        if selected_ciks_path.is_file():
            selected_ciks = sorted(json.loads(selected_ciks_path.read_text(encoding="utf-8"))["selected_ciks"])
        else:
            selected_ciks = _prescreen_liquid_ciks(
                paths=paths,
                panel_dir=panel_dir,
                sp500_backfill_dir=paths.directory(sp500_backfill),
                adj_dir=adj_dir,
                roster_df=roster_df,
                sp500_ciks=sp500_ciks,
                development_end=development_end,
            )
            selected_ciks_path.write_text(
                json.dumps({"selected_ciks": selected_ciks}, indent=2, sort_keys=True), encoding="utf-8",
            )
        release_unused_process_memory()
        guard("wider universe selected ciks determined")

        # Phase 3a: Fetch unadjusted (`raw`) daily closes for non-S&P-500 `selected_ciks` (`2018-05-29..valid_to`).
        raw_manifest = json.loads(
            (paths.directory(UNADJUSTED_DIRECTORY) / "_manifest.json").read_text(encoding="utf-8")
        )["unit_artifacts"]
        sp500_raw_tickers = {str(item["ticker"]) for item in raw_manifest if item.get("role") == "stock"}
        sel_spans = roster_df.loc[
            roster_df["cik"].isin(selected_ciks)
            & (~roster_df["ticker"].isin(sp500_raw_tickers))
            & (roster_df["valid_to"] >= EARLY_DECISION_START)
        ]
        raw_by_tk = (
            sel_spans.groupby("ticker", as_index=False)
            .agg({"valid_to": "max"})
            .sort_values(["valid_to", "ticker"], kind="stable")
            .reset_index(drop=True)
        )
        for r_idx_sh, offset in enumerate(range(0, len(raw_by_tk), shard_size)):
            r_shard_file = raw_dir / f"raw_shard_{r_idx_sh:03d}.parquet"
            if r_shard_file.is_file():
                continue
            sub_r = raw_by_tk.iloc[offset : offset + shard_size]
            r_frames: list[pd.DataFrame] = []
            open_r = sub_r.loc[sub_r["valid_to"] >= development_end]
            closed_r = sub_r.loc[sub_r["valid_to"] < development_end]
            if len(open_r):
                r_frames.append(
                    _fetch_alpaca_daily_batch(
                        alpaca, sorted(open_r["ticker"].astype(str)),
                        EARLY_DECISION_START, development_end, adjustment="raw", asof_day=development_end,
                    )
                )
            for v_end_d, grp_rc in closed_r.groupby("valid_to", sort=True):
                r_frames.append(
                    _fetch_alpaca_daily_batch(
                        alpaca, sorted(grp_rc["ticker"].astype(str)),
                        EARLY_DECISION_START, v_end_d, adjustment="raw", asof_day=v_end_d,
                    )
                )
            r_df = (
                pd.concat(r_frames, ignore_index=True).drop_duplicates(["ticker", "session_date"], keep="last")
                if r_frames
                else pd.DataFrame(columns=["ticker", "session_date", "close"])
            )
            cast(Any, pq).write_table(
                pa.Table.from_pandas(r_df[["ticker", "session_date", "close"]], preserve_index=False), r_shard_file,
            )
            del r_frames, r_df
            release_unused_process_memory()
            guard(f"wider universe raw shard {r_idx_sh}")

        # Phase 3b: Fetch SEC `submissions/CIK{cik}.json` (`sic` + all `8-K` items `<= 2024-05-28`) for `selected_ciks`.
        need_sel_sub = [c_id for c_id in selected_ciks if c_id not in sub_cache]
        if need_sel_sub:
            with sub_cache_path.open("a", encoding="utf-8") as handle:
                for idx_s, c_id in enumerate(need_sel_sub):
                    try:
                        sub_json = sec.client.get_json(sec.submissions_url.format(cik=c_id))
                        parsed = (
                            _parse_submissions_payload(
                                sec, c_id, sub_json, development_end=development_end,
                                fetch_overflow_8k=True, allow_form4_xml=False,
                            )
                            if isinstance(sub_json, dict)
                            else {"cik": c_id, "sic": "", "sector": "Industrials", "events_8k": []}
                        )
                    except Exception as exc:
                        if "404" in str(exc):
                            parsed = {"cik": c_id, "sic": "", "sector": "Industrials", "events_8k": []}
                        else:
                            raise
                    sub_cache[c_id] = parsed
                    handle.write(json.dumps(parsed, sort_keys=True) + "\n")
                    if (idx_s + 1) % 30 == 0:
                        handle.flush()
                        release_unused_process_memory()
                        guard("wider universe selected submissions")

        # Materialize consolidated `wider_8k_events.parquet` & `cik_sectors.json`.
        events_8k_path = dest / "wider_8k_events.parquet"
        sectors_path = dest / "cik_sectors.json"
        if not events_8k_path.is_file() or not sectors_path.is_file():
            ev_rows: list[dict[str, Any]] = []
            sec_map_out: dict[str, str] = {}
            for c_id in selected_ciks:
                info_c = sub_cache.get(c_id, {})
                sec_map_out[c_id] = str(info_c.get("sector") or "Industrials")
                for ev in info_c.get("events_8k", []):
                    ev_rows.append({
                        "security_id": f"cik:{c_id}",
                        "sec_cik": c_id,
                        "accession_number": str(ev["accession_number"]),
                        "sec_form": "8-K",
                        "item_codes": str(ev["item_codes"]),
                        "accepted_at_utc": pd.Timestamp(str(ev["accepted_at_utc"])),
                        "available_at_utc": pd.Timestamp(str(ev["available_at_utc"])),
                    })
            ev_df = (
                pd.DataFrame(ev_rows).drop_duplicates(["security_id", "accession_number"])
                if ev_rows
                else pd.DataFrame(columns=[
                    "security_id", "sec_cik", "accession_number", "sec_form",
                    "item_codes", "accepted_at_utc", "available_at_utc",
                ])
            )
            cast(Any, pq).write_table(pa.Table.from_pandas(ev_df, preserve_index=False), events_8k_path)
            sectors_path.write_text(json.dumps(sec_map_out, indent=2, sort_keys=True), encoding="utf-8")
            del ev_rows, ev_df
            release_unused_process_memory()

        # Phase 3c: Fetch missing SEC `company_facts.json` for `selected_ciks`.
        status_path = dest / "_status.json"
        prior_st = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        sec_404: set[str] = set(prior_st.get("sec_404_ciks", []))

        sp500_bf_facts = paths.directory(sp500_backfill) / "company_facts"
        extra_f = paths.project_root / EXTRA_FACTS_DIR
        already_have = existing_company_facts_paths(
            set(selected_ciks),
            primary_sources=paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES),
            extra_sources=extra_f if extra_f.is_dir() else None,
            backfill_sources=sp500_bf_facts,
        )
        for c_id in selected_ciks:
            p_w = facts_dir / c_id / COMPANY_FACTS_FILE
            if c_id not in already_have and p_w.is_file():
                already_have[c_id] = p_w

        missing_facts = [c_id for c_id in selected_ciks if c_id not in already_have and c_id not in sec_404]
        fetched_new = 0
        for c_id in missing_facts:
            target_f = facts_dir / c_id / COMPANY_FACTS_FILE
            if target_f.is_file():
                continue
            try:
                _, raw_resp = sec.fetch_cik_company_facts(c_id)
            except Exception as exc:
                if "404" in str(exc):
                    sec_404.add(c_id)
                    continue
                raise
            target_f.parent.mkdir(parents=True, exist_ok=True)
            tmp_f = target_f.with_suffix(".json.tmp")
            tmp_f.write_bytes(raw_resp.body)
            tmp_f.replace(target_f)
            already_have[c_id] = target_f
            fetched_new += 1
            if fetched_new % 25 == 0:
                release_unused_process_memory()
                guard("wider universe sec company facts")

        summary = {
            "development_end": development_end.isoformat(),
            "candidate_ciks_screened": len(cand_ciks),
            "quarantined_roster_spans": len(roster_df),
            "selected_liquid_ciks": len(selected_ciks),
            "sp500_ciks_included": len(set(selected_ciks) & sp500_ciks),
            "non_sp500_mid_small_ciks": len(set(selected_ciks) - sp500_ciks),
            "company_facts_available_ciks": len(already_have),
            "company_facts_newly_fetched": fetched_new,
            "sec_404_ciks": sorted(sec_404),
        }
        status_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary


def _load_sessions_and_benchmarks(
    panel_dir: Path, sp500_bf_dir: Path, development_end: date,
) -> tuple[
    npt.NDArray[np.datetime64],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    dict[str, float],
    dict[str, float],
]:
    """Load extended `2016-01-04..development_end` SPY/QQQ benchmarks + `2018-05-29` overlap stitch dictionaries."""
    end_iso = development_end.isoformat()
    cutoff_ts = (pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")
    stitch_day = np.datetime64(EARLY_DECISION_START, "D")
    artifacts = json.loads((panel_dir / BAR_MANIFEST).read_text(encoding="utf-8"))["artifacts"]

    bench_post: dict[str, pd.DataFrame] = {}
    for item in artifacts:
        tk = str(item["ticker"])
        if tk not in ("SPY", "QQQ"):
            continue
        tbl = cast(Any, pq).read_table(
            panel_dir / BAR_DIRECTORY / str(item["path"]),
            columns=["bar_start_utc", "open", "close"],
            partitioning=None,
            filters=[("bar_start_utc", "<", cutoff_ts.to_pydatetime())],
            use_threads=False,
        )
        df = tbl.to_pandas()
        del tbl
        sess = pd.to_datetime(df["bar_start_utc"], utc=True).dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None).dt.floor("D")
        df["session"] = sess.to_numpy().astype("datetime64[D]")
        bench_post[tk] = df.loc[df["session"] <= np.datetime64(development_end, "D"), ["session", "open", "close"]]

    warm_path = sp500_bf_dir / "bars" / "warmup_adjusted_20160104_20180529.parquet"
    w_tbl = cast(Any, pq).read_table(
        warm_path,
        columns=["ticker", "session_date", "close", "open"],
        partitioning=None,
        filters=[("session_date", "<=", end_iso)],
        use_threads=False,
    )
    w_df = w_tbl.to_pandas()
    del w_tbl
    w_df["session"] = pd.to_datetime(w_df["session_date"]).to_numpy().astype("datetime64[D]")
    w_on = w_df.loc[w_df["session"] == stitch_day].drop_duplicates("ticker", keep="first")
    warm_stitch = {
        str(tk): float(cl)
        for tk, cl in zip(w_on["ticker"], w_on["close"], strict=False)
        if np.isfinite(cl) and float(cl) > 0.0
    }

    bench_frames: dict[str, list[pd.DataFrame]] = {k: [v] for k, v in bench_post.items()}
    for b_sym in ("SPY", "QQQ"):
        b_post_on = bench_post[b_sym].loc[bench_post[b_sym]["session"] == stitch_day]
        sub_b = w_df.loc[(w_df["ticker"] == b_sym) & (w_df["session"] < stitch_day), ["session", "open", "close"]].copy()
        if len(sub_b) and len(b_post_on) and b_sym in warm_stitch:
            b_sc = float(b_post_on["close"].iloc[0]) / warm_stitch[b_sym]
            sub_b["open"] = sub_b["open"] * b_sc
            sub_b["close"] = sub_b["close"] * b_sc
        bench_frames[b_sym].append(sub_b)
    del w_df, bench_post

    spy_df = pd.concat(bench_frames["SPY"], ignore_index=True).drop_duplicates("session", keep="first").sort_values("session")
    qqq_df = pd.concat(bench_frames["QQQ"], ignore_index=True).drop_duplicates("session", keep="first").sort_values("session")
    sessions = spy_df["session"].to_numpy().astype("datetime64[D]")
    qqq_al = pd.DataFrame({"session": sessions}).merge(qqq_df, on="session", how="left")
    return (
        sessions,
        spy_df["open"].to_numpy(dtype=np.float32),
        spy_df["close"].to_numpy(dtype=np.float32),
        qqq_al["open"].to_numpy(dtype=np.float32),
        qqq_al["close"].to_numpy(dtype=np.float32),
        warm_stitch,
        {},
    )


def _stream_adjusted_bars_into_matrices(
    *,
    sessions: npt.NDArray[np.datetime64],
    ciks: tuple[str, ...],
    roster_df: pd.DataFrame,
    panel_dir: Path,
    sp500_bf_dir: Path,
    adj_dir: Path,
    warm_stitch: dict[str, float],
    development_end: date,
    fields: tuple[str, ...] = PRICE_FIELDS,
) -> tuple[dict[str, npt.NDArray[np.float32]], npt.NDArray[np.bool_]]:
    """Stream adjusted bars from S&P 500 + backfill + wider shards directly into `(sessions, ciks)` `float32` matrices."""
    n_sess, n_sec = len(sessions), len(ciks)
    col_of = {c: idx for idx, c in enumerate(ciks)}
    end_iso = development_end.isoformat()
    cutoff_ts = (pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")
    stitch_day = np.datetime64(EARLY_DECISION_START, "D")

    # Build `ticker -> list[(r_start, r_stop, col_idx)]` from quarantined `roster_df` restricted to `ciks`.
    sub_r = roster_df.loc[roster_df["cik"].isin(col_of)].copy()
    in_span = np.zeros((n_sess, n_sec), dtype=bool)
    tk_intervals: dict[str, list[tuple[int, int, int]]] = {}
    for rec in sub_r.to_dict(orient="records"):
        c_idx = col_of[str(rec["cik"])]
        r0 = int(np.searchsorted(sessions, np.datetime64(rec["valid_from"], "D"), side="left"))
        r1 = int(np.searchsorted(sessions, np.datetime64(rec["valid_to"], "D"), side="right"))
        if r0 < r1:
            in_span[r0:r1, c_idx] = True
            tk_intervals.setdefault(str(rec["ticker"]), []).append((r0, r1, c_idx))

    matrices: dict[str, npt.NDArray[np.float32]] = {
        f: np.full((n_sess, n_sec), np.nan, dtype=np.float32) for f in fields
    }
    base_stitch: dict[str, float] = {}

    def _place_df(tk: str, df_b: pd.DataFrame, *, min_session: np.datetime64 | None = None, scale: float = 1.0) -> None:
        ivs = tk_intervals.get(tk)
        if not ivs or df_b.empty:
            return
        if min_session is not None:
            on_st = df_b.loc[df_b["session"] == stitch_day, "close"]
            if len(on_st) and np.isfinite(on_st.iloc[0]) and float(on_st.iloc[0]) > 0.0 and tk not in base_stitch:
                base_stitch[tk] = float(on_st.iloc[0])
            df_b = df_b.loc[df_b["session"] >= min_session]
            if df_b.empty:
                return
        b_s = df_b["session"].to_numpy().astype("datetime64[D]")
        rc = np.searchsorted(sessions, b_s)
        ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
        if not bool(ok.any()):
            return
        rr_all = rc[ok]
        idx_all = np.flatnonzero(ok)
        for r0, r1, col_i in ivs:
            w_mask = (rr_all >= r0) & (rr_all < r1)
            if not bool(w_mask.any()):
                continue
            rr = rr_all[w_mask]
            src_i = idx_all[w_mask]
            for f in fields:
                vals = df_b[f].to_numpy(dtype=np.float64)[src_i]
                if scale != 1.0:
                    vals = vals / scale if f == "volume" else vals * scale
                matrices[f][rr, col_i] = vals.astype(np.float32)

    skipped_etfs = {"SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"}
    artifacts = json.loads((panel_dir / BAR_MANIFEST).read_text(encoding="utf-8"))["artifacts"]
    for idx_a, item in enumerate(artifacts):
        tk = str(item["ticker"])
        if tk in skipped_etfs or tk not in tk_intervals:
            continue
        tbl = cast(Any, pq).read_table(
            panel_dir / BAR_DIRECTORY / str(item["path"]),
            columns=["bar_start_utc", *fields],
            partitioning=None,
            filters=[("bar_start_utc", "<", cutoff_ts.to_pydatetime())],
            use_threads=False,
        )
        df = tbl.to_pandas()
        del tbl
        if not df.empty:
            sess = pd.to_datetime(df["bar_start_utc"], utc=True).dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None).dt.floor("D")
            df["session"] = sess.to_numpy().astype("datetime64[D]")
            df = df.loc[df["session"] <= np.datetime64(development_end, "D")]
            _place_df(tk, df, min_session=stitch_day)
        del df
        if (idx_a + 1) % 100 == 0:
            release_unused_process_memory()

    # Missing S&P 500 members + wider shards (`2016..development_end` single-pass adjusted bars).
    shard_paths = [
        sp500_bf_dir / "bars" / "missing_members_adjusted_20160104_20240528.parquet",
        *sorted(adj_dir.glob("shard_*.parquet")),
    ]
    for sp_path in shard_paths:
        if not sp_path.is_file():
            continue
        tbl = cast(Any, pq).read_table(
            sp_path,
            columns=["ticker", "session_date", *fields],
            partitioning=None,
            filters=[("session_date", "<=", end_iso)],
            use_threads=False,
        )
        df = tbl.to_pandas()
        del tbl
        if not df.empty:
            if str(df["session_date"].max()) > end_iso:
                raise SealedPeriodError(f"adjusted bars after {end_iso} reached memory from {sp_path}")
            df["session"] = pd.to_datetime(df["session_date"]).to_numpy().astype("datetime64[D]")
            for tk_s, grp_s in df.groupby("ticker", sort=False):
                _place_df(str(tk_s), grp_s, min_session=None)
        del df
        release_unused_process_memory()

    # Rescaled S&P 500 `2016-01-04..2018-05-28` warm-up bars.
    warm_path = sp500_bf_dir / "bars" / "warmup_adjusted_20160104_20180529.parquet"
    w_tbl = cast(Any, pq).read_table(
        warm_path,
        columns=["ticker", "session_date", *fields],
        partitioning=None,
        filters=[("session_date", "<=", end_iso)],
        use_threads=False,
    )
    w_df = w_tbl.to_pandas()
    del w_tbl
    if not w_df.empty:
        w_df["session"] = pd.to_datetime(w_df["session_date"]).to_numpy().astype("datetime64[D]")
        w_pre = w_df.loc[w_df["session"] < stitch_day]
        for tk_w, grp_w in w_pre.groupby("ticker", sort=False):
            tk_str = str(tk_w)
            sc = (
                base_stitch[tk_str] / warm_stitch[tk_str]
                if tk_str in base_stitch and tk_str in warm_stitch and warm_stitch[tk_str] > 0.0
                else 1.0
            )
            _place_df(tk_str, grp_w, min_session=None, scale=sc)
    del w_df
    release_unused_process_memory()
    return matrices, in_span


def _prescreen_liquid_ciks(
    *,
    paths: EvidencePaths,
    panel_dir: Path,
    sp500_backfill_dir: Path,
    adj_dir: Path,
    roster_df: pd.DataFrame,
    sp500_ciks: set[str],
    development_end: date,
) -> list[str]:
    """Stream `close` and `volume` only (`2` fields) across all candidate CIKs to select Top-1,300 liquid CIKs."""
    del paths
    sessions, _, _, _, _, warm_stitch, _ = _load_sessions_and_benchmarks(panel_dir, sp500_backfill_dir, development_end)
    all_ciks = tuple(sorted(set(roster_df["cik"].astype(str)) | sp500_ciks))
    mats, in_span = _stream_adjusted_bars_into_matrices(
        sessions=sessions,
        ciks=all_ciks,
        roster_df=roster_df,
        panel_dir=panel_dir,
        sp500_bf_dir=sp500_backfill_dir,
        adj_dir=adj_dir,
        warm_stitch=warm_stitch,
        development_end=development_end,
        fields=("close", "volume"),
    )
    dummy_raw = np.full(mats["close"].shape, 100.0, dtype=np.float64)
    elig_pre = compute_pit_liquidity_eligibility(
        mats["close"], mats["volume"], dummy_raw, in_span, sessions,
        top_n=TOP_LIQUID_COUNT + 150,  # slight buffer so after $5 raw-price check full 1,200 remain!
    )
    ever_cols = np.flatnonzero(elig_pre.any(axis=0))
    chosen = sorted({all_ciks[int(i)] for i in ever_cols} | sp500_ciks)
    del mats, in_span, dummy_raw, elig_pre
    release_unused_process_memory()
    return chosen


def load_wider_universe_data(
    paths: EvidencePaths,
    *,
    development_end: date,
    subset: UniverseSubset = "all_top1200",
    panel_rel: Path = PANEL_DIRECTORY,
    sp500_backfill_rel: Path = BACKFILL_DIRECTORY,
    wider_rel: Path = WIDER_UNIVERSE_DIRECTORY,
    unseal_final_test_period: bool = False,
) -> WiderUniverseData:
    """Load the survivorship-checked wider US panel (`all_top1200` or `mid_small_ex_sp500`) in low memory."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    panel_dir = paths.directory(panel_rel)
    sp500_bf_dir = paths.directory(sp500_backfill_rel)
    wider_dir = paths.directory(wider_rel)
    end_iso = development_end.isoformat()

    roster_path = wider_dir / "roster_spans.parquet"
    selected_path = wider_dir / "selected_ciks.json"
    events_8k_path = wider_dir / "wider_8k_events.parquet"
    sectors_path = wider_dir / "cik_sectors.json"

    hashes = {
        "combined_daily_manifest_sha256": file_sha256(panel_dir / BAR_MANIFEST),
        "wider_roster_spans_sha256": file_sha256(roster_path),
        "wider_selected_ciks_sha256": file_sha256(selected_path),
        "wider_8k_events_sha256": file_sha256(events_8k_path),
        "wider_cik_sectors_sha256": file_sha256(sectors_path),
    }

    selected_ciks = tuple(sorted(json.loads(selected_path.read_text(encoding="utf-8"))["selected_ciks"]))
    security_ids = tuple(f"cik:{c}" for c in selected_ciks)
    col_of_cik = {c: idx for idx, c in enumerate(selected_ciks)}

    roster_df = cast(Any, pq).read_table(roster_path, use_threads=False).to_pandas()
    roster_df["valid_from"] = pd.to_datetime(roster_df["valid_from"]).dt.date
    roster_df["valid_to"] = pd.to_datetime(roster_df["valid_to"]).dt.date

    sessions, spy_open, spy_close, qqq_open, qqq_close, warm_stitch, _ = _load_sessions_and_benchmarks(
        panel_dir, sp500_bf_dir, development_end,
    )
    matrices, in_span = _stream_adjusted_bars_into_matrices(
        sessions=sessions,
        ciks=selected_ciks,
        roster_df=roster_df,
        panel_dir=panel_dir,
        sp500_bf_dir=sp500_bf_dir,
        adj_dir=wider_dir / "bars_adjusted",
        warm_stitch=warm_stitch,
        development_end=development_end,
        fields=PRICE_FIELDS,
    )

    # Stream unadjusted closes (`swing_initial_fit_raw_share_canonical` + `sp500_bf` + `wider/bars_raw`).
    n_sess, n_sec = len(sessions), len(selected_ciks)
    unadj = np.full((n_sess, n_sec), np.nan, dtype=np.float64)

    sub_r = roster_df.loc[roster_df["cik"].isin(col_of_cik)]
    tk_ivs: dict[str, list[tuple[int, int, int]]] = {}
    for rec in sub_r.to_dict(orient="records"):
        c_i = col_of_cik[str(rec["cik"])]
        r0 = int(np.searchsorted(sessions, np.datetime64(rec["valid_from"], "D"), side="left"))
        r1 = int(np.searchsorted(sessions, np.datetime64(rec["valid_to"], "D"), side="right"))
        if r0 < r1:
            tk_ivs.setdefault(str(rec["ticker"]), []).append((r0, r1, c_i))

    def _place_raw_by_ticker(df_r: pd.DataFrame) -> None:
        if df_r.empty:
            return
        df_r = df_r.copy()
        df_r["session"] = pd.to_datetime(df_r["session_date"]).to_numpy().astype("datetime64[D]")
        for tk_k, grp_r in df_r.groupby("ticker", sort=False):
            ivs = tk_ivs.get(str(tk_k))
            if not ivs:
                continue
            b_s = grp_r["session"].to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if not bool(ok.any()):
                continue
            rr_a = rc[ok]
            cl_a = grp_r["close"].to_numpy(dtype=np.float64)[ok]
            for r0, r1, c_i in ivs:
                wm = (rr_a >= r0) & (rr_a < r1)
                if bool(wm.any()):
                    unadj[rr_a[wm], c_i] = cl_a[wm]

    # Canonical S&P 500 raw shares (`security_id -> cik` via `resolve_all_sp500_ciks`).
    membership = _read_panel_membership(panel_dir, development_end)
    sp500_sec_ids = tuple(sorted(membership["security_id"].astype(str).unique()))
    sp500_cik_map, _ = resolve_all_sp500_ciks(
        sp500_sec_ids,
        identity_relations_path=paths.read(SEC_IDENTITY_RELATIONS),
        filings_path=paths.read(FILINGS_DIRECTORY / FILINGS_FILE),
        development_end=development_end,
    )
    raw_dir_can = paths.directory(UNADJUSTED_DIRECTORY)
    raw_units = json.loads((raw_dir_can / "_manifest.json").read_text(encoding="utf-8"))["unit_artifacts"]
    for idx_u, item in enumerate(raw_units):
        if item.get("role") != "stock":
            continue
        c_hit = sp500_cik_map.get(str(item["security_id"]))
        if c_hit is None or c_hit not in col_of_cik:
            continue
        c_col = col_of_cik[c_hit]
        u_tbl = cast(Any, pq).read_table(
            raw_dir_can / Path(str(item["bars_path"]).replace("\\", "/")),
            columns=["session_date", "close"],
            partitioning=None,
            filters=[("session_date", "<=", end_iso)],
            use_threads=False,
        )
        u_df = u_tbl.to_pandas()
        del u_tbl
        if not u_df.empty:
            b_s = pd.to_datetime(u_df["session_date"]).to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if bool(ok.any()):
                unadj[rc[ok], c_col] = u_df["close"].to_numpy(dtype=np.float64)[ok]
        del u_df
        if (idx_u + 1) % 120 == 0:
            release_unused_process_memory()

    for r_sh_path in [
        sp500_bf_dir / "bars" / "early_and_missing_raw_20180529_20240528.parquet",
        *sorted((wider_dir / "bars_raw").glob("raw_shard_*.parquet")),
    ]:
        if not r_sh_path.is_file():
            continue
        rt = cast(Any, pq).read_table(
            r_sh_path,
            columns=["ticker", "session_date", "close"],
            partitioning=None,
            filters=[("session_date", "<=", end_iso)],
            use_threads=False,
        )
        rdf = rt.to_pandas()
        del rt
        if len(rdf) and str(rdf["session_date"].max()) > end_iso:
            raise SealedPeriodError(f"raw bars after {end_iso} reached memory from {r_sh_path}")
        _place_raw_by_ticker(rdf)
        del rdf
        release_unused_process_memory()

    # Point-in-time S&P 500 membership mask on `(sessions, selected_ciks)` (`membership_available <= decision_time`).
    sp500_mask = np.zeros((n_sess, n_sec), dtype=bool)
    mem_known = membership.loc[
        (membership["membership_available_at_utc"] <= membership["decision_time_utc"]).fillna(False)
        & membership["session"].isin(sessions)
        & membership["security_id"].isin(sp500_cik_map)
    ]
    mr = np.searchsorted(sessions, mem_known["session"].to_numpy().astype("datetime64[D]"))
    mc_ciks = mem_known["security_id"].map(sp500_cik_map)
    valid_mc = mc_ciks.isin(col_of_cik).to_numpy()
    if bool(valid_mc.any()):
        mc_idx = np.asarray([col_of_cik[str(c)] for c in mc_ciks.loc[valid_mc]], dtype=np.int32)
        sp500_mask[mr[valid_mc], mc_idx] = True
    del membership, mem_known
    release_unused_process_memory()

    elig_top1200 = compute_pit_liquidity_eligibility(
        matrices["close"], matrices["volume"], unadj, in_span, sessions, top_n=TOP_LIQUID_COUNT,
    )
    eligible = elig_top1200 & (~sp500_mask) if subset == "mid_small_ex_sp500" else elig_top1200
    del elig_top1200, in_span

    # Prune columns that are never eligible in `subset` (e.g. permanent S&P 500 members in `mid_small_ex_sp500`)
    # so downstream feature/score matrices and `company_facts` parsing allocate zero memory for inactive securities.
    keep_cols = np.flatnonzero(eligible.any(axis=0))
    selected_ciks = tuple(selected_ciks[int(i)] for i in keep_cols)
    security_ids = tuple(f"cik:{c}" for c in selected_ciks)
    n_sec = len(selected_ciks)
    for f_k in PRICE_FIELDS:
        matrices[f_k] = np.ascontiguousarray(matrices[f_k][:, keep_cols])
    unadj = np.ascontiguousarray(unadj[:, keep_cols])
    eligible = np.ascontiguousarray(eligible[:, keep_cols])
    sp500_mask = np.ascontiguousarray(sp500_mask[:, keep_cols])
    release_unused_process_memory()

    # Sectors from `cik_sectors.json`.
    cik_sectors = json.loads(sectors_path.read_text(encoding="utf-8"))
    sec_per_col = [str(cik_sectors.get(c, "Industrials")) for c in selected_ciks]
    sector_names = tuple(sorted(set(sec_per_col)))
    sec_codes_1d = pd.Index(sector_names).get_indexer(sec_per_col).astype(np.int16)
    sector_mat = np.broadcast_to(sec_codes_1d[None, :], (n_sess, n_sec)).copy()

    first_dec_row = int(np.searchsorted(sessions, np.datetime64(EARLY_DECISION_START, "D"), side="left"))
    wider_panel = ResearchPanel(
        sessions=sessions,
        security_ids=security_ids,
        open=matrices["open"],
        high=matrices["high"],
        low=matrices["low"],
        close=matrices["close"],
        volume=matrices["volume"],
        eligible=eligible,
        sector=sector_mat,
        sector_names=sector_names,
        spy_open=spy_open,
        spy_close=spy_close,
        qqq_open=qqq_open,
        qqq_close=qqq_close,
        development_end=development_end,
        load_report={
            "subset": subset,
            "development_end": end_iso,
            "sessions": {"count": n_sess, "first": str(sessions[0]), "last": str(sessions[-1])},
            "first_eligible_decision_session": str(sessions[first_dec_row]),
            "panel_securities_total": n_sec,
            "eligible_cells_total": int(eligible.sum()),
            "mean_eligible_stocks_per_decision_session": float(eligible[first_dec_row:].sum(axis=1).mean()),
            "eligible_cells_pre_20190709": int(
                eligible[: int(np.searchsorted(sessions, np.datetime64(NEWS_DECISION_START, "D")))].sum()
            ),
        },
        input_sha256=hashes,
    )

    # Merge SEC 8-K events (`wider_8k_events.parquet` + S&P 500 `filings.parquet` + `pre2019_8k_releases.parquet`).
    cutoff_utc = (pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")
    ev_frames: list[pd.DataFrame] = []
    w_ev = cast(Any, pq).read_table(events_8k_path, use_threads=False).to_pandas()
    if len(w_ev):
        w_ev["available_at_utc"] = pd.to_datetime(w_ev["available_at_utc"], utc=True)
        w_ev["accepted_at_utc"] = pd.to_datetime(w_ev["accepted_at_utc"], utc=True)
        if bool((w_ev["available_at_utc"] >= cutoff_utc).any()):
            raise SealedPeriodError("wider_8k_events contains sealed timestamp")
        ev_frames.append(w_ev[["security_id", "accession_number", "item_codes", "accepted_at_utc", "available_at_utc"]])

    for sp_ev_path in (
        paths.read(FILINGS_DIRECTORY / FILINGS_FILE),
        sp500_bf_dir / "pre2019_8k_releases.parquet",
    ):
        if not sp_ev_path.is_file():
            continue
        st_tbl = cast(Any, pq).read_table(
            sp_ev_path,
            columns=["security_id", "sec_cik", "accession_number", "sec_form", "item_codes", "accepted_at_utc", "available_at_utc"],
            filters=[("available_at_utc", "<", cutoff_utc.to_pydatetime())],
            use_threads=False,
        )
        sdf = st_tbl.to_pandas()
        del st_tbl
        sdf = sdf.loc[sdf["sec_form"].eq("8-K")].copy()
        if len(sdf):
            sdf["security_id"] = "cik:" + sdf["sec_cik"].astype(str).str.zfill(10)
            sdf = sdf.loc[sdf["security_id"].isin(security_ids)]
            ev_frames.append(sdf[["security_id", "accession_number", "item_codes", "accepted_at_utc", "available_at_utc"]])

    all_8k = (
        pd.concat(ev_frames, ignore_index=True).drop_duplicates(["security_id", "accession_number"])
        if ev_frames
        else pd.DataFrame(columns=["security_id", "accession_number", "item_codes", "accepted_at_utc", "available_at_utc"])
    )
    has_202 = all_8k["item_codes"].fillna("").astype(str).map(lambda s: "2.02" in {x.strip() for x in s.split(",")})
    releases_202 = all_8k.loc[has_202, ["security_id", "accession_number", "accepted_at_utc", "available_at_utc"]].reset_index(drop=True)

    # Resolve all available `company_facts.json` paths across the 4 local directories.
    extra_f = paths.project_root / EXTRA_FACTS_DIR
    facts_paths = existing_company_facts_paths(
        set(selected_ciks),
        primary_sources=paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES),
        extra_sources=extra_f if extra_f.is_dir() else None,
        backfill_sources=sp500_bf_dir / "company_facts",
    )
    w_facts_dir = wider_dir / "company_facts"
    for c_id in selected_ciks:
        pw = w_facts_dir / c_id / COMPANY_FACTS_FILE
        if c_id not in facts_paths and pw.is_file():
            facts_paths[c_id] = pw

    active_cik_map = {f"cik:{c}": c for c in selected_ciks if c in facts_paths}
    release_unused_process_memory()

    return WiderUniverseData(
        panel=wider_panel,
        unadjusted_close=unadj,
        releases_202=releases_202,
        all_8k_events=all_8k,
        security_cik_map=active_cik_map,
        company_facts_paths=facts_paths,
        sp500_member_mask=sp500_mask,
        input_sha256=hashes,
        report={
            "panel": wider_panel.load_report,
            "securities_with_company_facts": len(active_cik_map),
            "securities_with_unadjusted_close": int(np.isfinite(unadj).any(axis=0).sum()),
            "total_8k_events": len(all_8k),
            "total_8k_202_earnings_releases": len(releases_202),
        },
    )


def build_wider_catalyst_reaction_features(
    panel: ResearchPanel,
    all_8k_events: pd.DataFrame,
    *,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    unseal_final_test_period: bool = False,
) -> tuple[dict[str, npt.NDArray[np.float32]], dict[str, Any]]:
    """Build continuous Stage-2 catalyst-reaction features across all 1,200+ issuers from SEC 8-K + Alpaca news."""
    check_development_end(panel.development_end, unseal_final_test_period=unseal_final_test_period)
    n_sess, n_sec = len(panel.sessions), len(panel.security_ids)
    col_of = {sid: idx for idx, sid in enumerate(panel.security_ids)}
    cutoff_utc = (pd.Timestamp(panel.development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")

    # Map SEC 8-K filings (100% universe coverage from `2018-05-29` onward!) + retained Alpaca news where mapped.
    ev_records: list[tuple[int, int, str]] = []
    count_mat = np.zeros((n_sess, n_sec), dtype=np.float64)

    if len(all_8k_events):
        r_8k = event_rows(panel.sessions, pd.to_datetime(all_8k_events["available_at_utc"], utc=True))
        for r_i, sid_raw, items_raw in zip(
            r_8k, all_8k_events["security_id"].astype(str), all_8k_events["item_codes"].fillna("").astype(str), strict=False,
        ):
            ri = int(r_i)
            c_i = col_of.get(sid_raw, -1)
            if c_i < 0 or ri < 1 or ri >= n_sess:
                continue
            count_mat[ri, c_i] += 1.0
            iset = {x.strip() for x in items_raw.split(",") if x.strip()}
            matched_any = False
            for item_code, feat_name in WIDER_CATALYST_ITEMS:
                if item_code in iset:
                    ev_records.append((ri, c_i, feat_name))
                    matched_any = True
            if not matched_any:
                ev_records.append((ri, c_i, "rx_8k_801_other_event"))

    alpaca_mapped = 0
    for d_path in (early_news_dir, later_news_dir):
        pf = d_path / "family_events.parquet"
        if not pf.is_file():
            continue
        tbl = cast(Any, pq).read_table(
            pf,
            columns=["security_id", "feature_available_at_utc", "relation_channel", "research_eligible"],
            filters=[("feature_available_at_utc", "<", cutoff_utc.to_pydatetime())],
            use_threads=False,
        )
        ndf = tbl.to_pandas()
        del tbl
        ndf = ndf.loc[ndf["research_eligible"].fillna(False).astype(bool) & ndf["relation_channel"].eq("direct_issuer")]
        if len(ndf):
            rn = event_rows(panel.sessions, pd.to_datetime(ndf["feature_available_at_utc"], utc=True))
            for r_i, sid_n in zip(rn, ndf["security_id"].astype(str), strict=False):
                ri = int(r_i)
                parts = sid_n.split(":")
                c_key = f"cik:{parts[1]}" if len(parts) >= 2 and sid_n.startswith("cik:") else sid_n
                c_i = col_of.get(c_key, -1)
                if c_i >= 0 and 1 <= ri < n_sess:
                    count_mat[ri, c_i] += 1.0
                    ev_records.append((ri, c_i, "rx_8k_801_other_event"))
                    alpaca_mapped += 1
        del ndf

    raw_feats: dict[str, npt.NDArray[np.float64]] = {
        name: np.full((n_sess, n_sec), np.nan, dtype=np.float64) for name in WIDER_STAGE2_FEATURE_NAMES
    }
    if ev_records:
        closes = panel.close.astype(np.float64)
        vols = panel.volume.astype(np.float64)
        spy_c = panel.spy_close.astype(np.float64)
        v_ok_m = np.isfinite(vols) & (vols >= 0.0)
        v_csum = np.vstack([np.zeros((1, n_sec)), np.cumsum(np.where(v_ok_m, vols, 0.0), axis=0)])
        c_csum = np.vstack([np.zeros((1, n_sec), dtype=np.int32), np.cumsum(v_ok_m.astype(np.int32), axis=0)])
        decay_lut = np.power(0.5, np.arange(22, dtype=np.float64) / 5.0)
        sess_idx = np.arange(n_sess, dtype=np.int64)

        ev_df = pd.DataFrame(ev_records, columns=["row", "col", "feat"]).drop_duplicates()
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
            raw_feats["rx_catalyst_any"][act, c] = rx_v[pk[act]] * wd[act]
            raw_feats["catalyst_abn_vol_20"][act, c] = np.log1p(np.maximum(0.0, abn[pk[act]])) * wd[act]
            raw_feats["catalyst_recency_21"][act, c] = (22 - age[act]) / 22.0

            rx_lookup = dict(zip(dist_r.tolist(), rx_v.tolist(), strict=False))
            for _, f_name in WIDER_CATALYST_ITEMS:
                fr = np.sort(grp.loc[grp["feat"] == f_name, "row"].to_numpy(dtype=np.int64))
                if fr.size == 0:
                    continue
                frx = np.asarray([rx_lookup.get(int(r), np.nan) for r in fr], dtype=np.float64)
                fp = np.searchsorted(fr, sess_idx, side="right") - 1
                fv = fp >= 0
                fpk = np.where(fv, fp, 0)
                fage = sess_idx - fr[fpk]
                fact = fv & (fage <= 21)
                fwd = decay_lut[np.clip(fage, 0, 21)]
                raw_feats[f_name][fact, c] = frx[fpk[fact]] * fwd[fact]

    cnt_cs = np.vstack([np.zeros((1, n_sec)), np.cumsum(count_mat, axis=0)])
    ra = np.arange(1, n_sess + 1)
    c5 = cnt_cs[ra] - cnt_cs[np.maximum(0, ra - 5)]
    c60 = cnt_cs[ra] - cnt_cs[np.maximum(0, ra - 60)]
    start_r = int(np.searchsorted(panel.sessions, np.datetime64(EARLY_DECISION_START, "D"), side="left"))
    warm_c = np.arange(n_sess)[:, None] >= (start_r + 20)
    raw_feats["catalyst_unusual_count_5_60"] = np.where(
        warm_c & panel.eligible, c5 / np.maximum(c60 / 12.0, 0.25), np.nan,
    )

    ranked_out: dict[str, npt.NDArray[np.float32]] = {}
    active_era = (np.arange(n_sess)[:, None] >= start_r) & panel.eligible
    for name in WIDER_STAGE2_FEATURE_NAMES:
        base_a = (
            np.where(active_era, raw_feats.pop(name), np.nan)
            if name == "catalyst_unusual_count_5_60"
            else np.where(active_era, np.nan_to_num(raw_feats.pop(name), nan=0.0), np.nan)
        )
        ranked_out[name] = centered_rank(base_a, active_era).astype(np.float32)
    release_unused_process_memory()
    return ranked_out, {
        "total_mapped_catalyst_tuples": len(ev_records),
        "alpaca_news_events_mapped": alpaca_mapped,
        "features": list(WIDER_STAGE2_FEATURE_NAMES),
    }
