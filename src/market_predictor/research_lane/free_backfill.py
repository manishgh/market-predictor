"""Zero-cost ($0) S&P 500 data completeness & 2016-2019 warm-up backfill for Research Lane Step 5a.

Uses only free SEC EDGAR (`SecSource`, 6 req/s) and existing Alpaca SIP (`AlpacaSource`) to:
1. Fetch missing S&P 500 `company_facts.json` files (lifting financial coverage from ~76% to ~96%+),
2. Backfill `2016-01-04..2019-07-08` adjusted (`all`) and unadjusted (`raw`) SIP daily bars plus any
   S&P 500 member tickers absent from `combined_daily/bars` strictly `<= development_end` (`2024-05-28`),
3. Backfill `2018-01-01..2019-07-08` SEC `8-K` Item `2.02` earnings releases with conservative
   `max(UTC, America/New_York)` acceptance timestamps so zero lookahead is possible.

Nothing dated `>= 2024-05-29` (the sealed final-test period) is ever requested or read.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.catalysts.sec_filings.acceptance_clock import wall_clock
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
from market_predictor.research_lane.panel import (
    BAR_DIRECTORY,
    BAR_MANIFEST,
    ELIGIBILITY_COLUMN,
    MARKET_TIMEZONE,
    PANEL_COLUMNS,
    PANEL_DIRECTORY,
    PANEL_MANIFEST,
    PRICE_FIELDS,
    PROFILE_DIRECTORY,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
    file_sha256,
)
from market_predictor.research_lane.sec_filings import (
    FILING_COLUMNS,
    FILINGS_DIRECTORY,
    FILINGS_FILE,
    read_sec_filings,
    security_issuers,
)
from market_predictor.research_lane.unadjusted_prices import (
    UNADJUSTED_DIRECTORY,
    load_unadjusted_close,
)
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.sec import SecSource

BACKFILL_DIRECTORY: Final = Path("data/research/research_lane/sp500_free_backfill")
SEC_IDENTITY_RELATIONS: Final = Path(
    "data/canonical/sec_identity/sec_identity_20190709_20260708_v2/sec_identity_relations.parquet"
)
EXTRA_FACTS_DIR: Final = Path("data/raw/investment_company_facts")

WARMUP_START_DATE: Final = date(2016, 1, 4)
EARLY_DECISION_START: Final = date(2018, 5, 29)
NEWS_DECISION_START: Final = date(2019, 7, 9)
MINIMUM_WARMUP_CLOSES: Final = 252
SYMBOL_BATCH_SIZE: Final = 40

COLLECT_FREE_SP500_COMMAND: Final = "collect-research-lane-free-sp500"


@dataclass(frozen=True)
class ExtendedSP500Data:
    """Extended `2016..2024-05-28` S&P 500 panel, unadjusted closes, SEC releases, and CIK map."""

    panel: ResearchPanel
    unadjusted_close: npt.NDArray[np.float64]
    releases: pd.DataFrame
    security_cik_map: dict[str, str]
    company_facts_paths: dict[str, Path]
    input_sha256: dict[str, str]
    report: dict[str, Any]


def _read_panel_membership(
    panel_root: Path, development_end: date, *, unseal_final_test_period: bool = False,
) -> pd.DataFrame:
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    profile_dir = panel_root / PROFILE_DIRECTORY
    end_text = development_end.isoformat()
    frames: list[pd.DataFrame] = []
    for child in sorted(profile_dir.iterdir()):
        if not child.is_dir() or not child.name.startswith("year="):
            continue
        yr = int(child.name.split("=", 1)[1])
        if yr > development_end.year:
            continue
        table = cast(Any, pq).read_table(
            child / "part.parquet",
            columns=list(PANEL_COLUMNS),
            partitioning=None,
            filters=[("session_date_et", "<=", development_end)],
        )
        df = table.to_pandas()
        if len(df) and str(df["session_date_et"].astype(str).max()) > end_text:
            raise SealedPeriodError(f"membership rows after {end_text} reached memory from {child}")
        frames.append(df)
    if not frames:
        raise ValueError(f"no membership rows <= {end_text} in {profile_dir}")
    out = pd.concat(frames, ignore_index=True)
    out["session"] = pd.to_datetime(out["session_date_et"]).to_numpy().astype("datetime64[D]")
    return out


def resolve_all_sp500_ciks(
    security_ids: tuple[str, ...],
    *,
    identity_relations_path: Path,
    filings_path: Path,
    development_end: date,
    unseal_final_test_period: bool = False,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Resolve `security_id -> 10-digit CIK` using `cik:` IDs, canonical SEC relations, and filings."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    cutoff = pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)

    rel_ciks: dict[str, set[str]] = {}
    if identity_relations_path.is_file():
        rel_tbl = cast(Any, pq).read_table(
            identity_relations_path,
            columns=["security_id", "sec_cik", "available_at_utc"],
            partitioning=None,
        )
        rel_df = rel_tbl.to_pandas()
        rel_df = rel_df.loc[pd.to_datetime(rel_df["available_at_utc"], utc=True) < cutoff.tz_convert("UTC")]
        for sec_id, cik_val in zip(rel_df["security_id"].astype(str), rel_df["sec_cik"].astype(str), strict=False):
            digits = cik_val.strip().zfill(10)
            if digits.isdigit() and len(digits) == 10:
                rel_ciks.setdefault(sec_id, set()).add(digits)

    filings = read_sec_filings(filings_path, development_end)
    filing_ciks = security_issuers(filings)
    del filings

    mapping: dict[str, str] = {}
    sources: Counter[str] = Counter()
    for sec_id in security_ids:
        parts = sec_id.split(":")
        if parts[0] == "cik" and len(parts) >= 2 and parts[1].isdigit():
            mapping[sec_id] = parts[1].zfill(10)
            sources["cik_prefix"] += 1
            continue
        rel_set = rel_ciks.get(sec_id, set())
        if len(rel_set) == 1:
            mapping[sec_id] = next(iter(rel_set))
            sources["sec_identity_relations"] += 1
            continue
        f_ciks = filing_ciks.get(sec_id, ())
        if len(f_ciks) == 1:
            mapping[sec_id] = f_ciks[0].zfill(10)
            sources["sec_filings_inventory"] += 1
            continue
        sources["unmapped"] += 1

    return mapping, {
        "total_securities": len(security_ids),
        "mapped_securities": len(mapping),
        "distinct_ciks": len(set(mapping.values())),
        "by_source": dict(sorted(sources.items())),
    }


def existing_company_facts_paths(
    ciks: set[str],
    *,
    primary_sources: Path,
    extra_sources: Path | None,
    backfill_sources: Path,
) -> dict[str, Path]:
    """Locate `company_facts.json` across primary, raw extra, and backfill directories."""
    found: dict[str, Path] = {}
    for cik in sorted(ciks):
        p1 = primary_sources / cik / COMPANY_FACTS_FILE
        if p1.is_file():
            found[cik] = p1
            continue
        if extra_sources is not None and (extra_sources / cik).is_dir():
            hit = None
            for child in sorted((extra_sources / cik).iterdir()):
                cand = child / COMPANY_FACTS_FILE
                if cand.is_file():
                    hit = cand
                    break
            if hit is not None:
                found[cik] = hit
                continue
        p3 = backfill_sources / cik / COMPANY_FACTS_FILE
        if p3.is_file():
            found[cik] = p3
    return found


def _to_alpaca_symbol(ticker: str) -> str:
    """Convert canonical hyphenated class symbols (`BRK-B`, `BF-B`) to Alpaca SIP dot notation (`BRK.B`)."""
    return ticker.strip().upper().replace("-", ".")


def _fetch_alpaca_daily_batch(
    alpaca: AlpacaSource,
    symbols: list[str],
    start_day: date,
    end_day: date,
    *,
    adjustment: str,
    asof_day: date,
    unseal_final_test_period: bool = False,
) -> pd.DataFrame:
    """Fetch daily SIP bars for `symbols` across `[start_day, end_day]`."""
    asof_eff = max(asof_day, end_day)
    check_development_end(end_day, unseal_final_test_period=unseal_final_test_period)
    check_development_end(asof_eff, unseal_final_test_period=unseal_final_test_period)
    start_utc = datetime.combine(start_day, time(0, 0), tzinfo=UTC)
    end_utc = datetime.combine(end_day, time(23, 59, 59), tzinfo=UTC)

    prov_to_orig: dict[str, str] = {}
    for raw_sym in symbols:
        clean = raw_sym.strip()
        if clean:
            prov_to_orig[_to_alpaca_symbol(clean)] = clean
    prov_symbols = sorted(prov_to_orig)

    rows_out: list[dict[str, Any]] = []

    def _run_chunk(chunk: tuple[str, ...]) -> None:
        if not chunk:
            return
        token: str | None = None
        seen_tokens: set[str] = set()
        try:
            while True:
                page = alpaca.fetch_bars_page(
                    chunk,
                    start_utc,
                    end_utc,
                    timeframe="1Day",
                    page_token=token,
                    asof=asof_eff,
                    limit=10_000,
                    adjustment=adjustment,
                )
                for sym, bar_list in page.bars.items():
                    orig_sym = prov_to_orig.get(sym, sym)
                    for bar in bar_list:
                        ts = pd.Timestamp(str(bar["t"]))
                        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
                        bar_day = ts.tz_convert(MARKET_TIMEZONE).date()
                        if bar_day > end_day:
                            raise SealedPeriodError(f"Alpaca returned bar after {end_day}: {orig_sym} {bar_day}")
                        avail_utc = pd.Timestamp(
                            datetime.combine(bar_day, time(16, 15)), tz=MARKET_TIMEZONE,
                        ).tz_convert("UTC")
                        rows_out.append({
                            "ticker": orig_sym,
                            "session_date": bar_day.isoformat(),
                            "bar_start_utc": ts,
                            "available_at_utc": avail_utc,
                            "open": float(bar["o"]),
                            "high": float(bar["h"]),
                            "low": float(bar["l"]),
                            "close": float(bar["c"]),
                            "volume": float(bar["v"]),
                            "source": "alpaca",
                            "timeframe": "1Day",
                            "price_feed": "sip",
                            "adjustment": adjustment,
                        })
                token = page.next_page_token
                if not token:
                    break
                if token in seen_tokens:
                    raise RuntimeError("Alpaca bars pagination repeated a page token")
                seen_tokens.add(token)
        except Exception as exc:
            if "status=400" in str(exc):
                if len(chunk) == 1:
                    return
                for single in chunk:
                    _run_chunk((single,))
                return
            raise

    for offset in range(0, len(prov_symbols), SYMBOL_BATCH_SIZE):
        _run_chunk(tuple(prov_symbols[offset : offset + SYMBOL_BATCH_SIZE]))
        guard(f"research lane backfill bars {adjustment}")

    if not rows_out:
        return pd.DataFrame(columns=[
            "ticker", "session_date", "bar_start_utc", "available_at_utc",
            "open", "high", "low", "close", "volume", "source", "timeframe", "price_feed", "adjustment",
        ])
    return pd.DataFrame(rows_out).drop_duplicates(["ticker", "session_date"], keep="last").reset_index(drop=True)


def _conservative_edgar_availability_utc(raw_acceptance: object, filing_date_str: str) -> pd.Timestamp:
    """Return `max(UTC reading, America/New_York reading)` so neither EDGAR clock convention leaks."""
    text = str(raw_acceptance or "").strip()
    if text.endswith(".000Z") and len(text) == 24:
        wc = wall_clock(text)
        as_utc = wc.tz_localize("UTC")
        try:
            as_ny = wc.tz_localize(MARKET_TIMEZONE, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
        except ValueError:
            as_ny = pd.NaT
        if pd.notna(as_ny) and as_ny > as_utc:
            return as_ny
        return as_utc
    if len(text) == 14 and text.isdigit():
        dt = datetime.strptime(text, "%Y%m%d%H%M%S")
        as_utc = pd.Timestamp(dt, tz="UTC")
        as_ny = pd.Timestamp(dt).tz_localize(MARKET_TIMEZONE, ambiguous="NaT", nonexistent="NaT").tz_convert("UTC")
        return as_ny if pd.notna(as_ny) and as_ny > as_utc else as_utc
    # Fallback to end of filing_date after market close (17:00 ET).
    f_day = date.fromisoformat(filing_date_str)
    return pd.Timestamp(datetime.combine(f_day, time(17, 0)), tz=MARKET_TIMEZONE).tz_convert("UTC")


def _extract_8k_202_from_submissions(
    sec: SecSource,
    cik: str,
    start_day: date,
    end_day: date,
) -> list[dict[str, Any]]:
    """Extract `8-K` (non-amendment) Item `2.02` filings in `[start_day, end_day]` for `cik`."""
    check_development_end(end_day)
    url = sec.submissions_url.format(cik=cik.zfill(10))
    root = sec.client.get_json(url)
    if not isinstance(root, dict):
        return []
    filings = root.get("filings", {})
    if not isinstance(filings, dict):
        return []
    sections: list[dict[str, Any]] = []
    recent = filings.get("recent")
    if isinstance(recent, dict):
        sections.append(recent)
    for desc in filings.get("files", []):
        if not isinstance(desc, dict):
            continue
        f_from = str(desc.get("filingFrom", ""))
        f_to = str(desc.get("filingTo", ""))
        name = str(desc.get("name", "")).strip()
        if name and f_from <= end_day.isoformat() and f_to >= start_day.isoformat():
            hist = sec.client.get_json(sec.historical_submission_url.format(name=name))
            if isinstance(hist, dict):
                sub = hist.get("filings", hist)
                if isinstance(sub, dict):
                    sections.append(sub)

    start_iso, end_iso = start_day.isoformat(), end_day.isoformat()
    cutoff_utc = pd.Timestamp(datetime.combine(end_day, time(23, 59, 59)), tz=MARKET_TIMEZONE).tz_convert("UTC")
    results: list[dict[str, Any]] = []
    seen_accn: set[str] = set()

    for sec_block in sections:
        forms = sec_block.get("form", [])
        f_dates = sec_block.get("filingDate", [])
        accns = sec_block.get("accessionNumber", [])
        accepts = sec_block.get("acceptanceDateTime", [])
        items_col = sec_block.get("items", [])
        n = min(len(forms), len(f_dates), len(accns), len(accepts), len(items_col))
        for idx in range(n):
            form = str(forms[idx]).strip().upper()
            if form != "8-K":
                continue
            f_date = str(f_dates[idx]).strip()
            if f_date < start_iso or f_date > end_iso:
                continue
            raw_items = items_col[idx]
            if isinstance(raw_items, list):
                item_set = {str(x).strip() for x in raw_items if str(x).strip()}
            else:
                item_set = {x.strip() for x in str(raw_items or "").split(",") if x.strip()}
            if "2.02" not in item_set:
                continue
            accn = str(accns[idx]).strip()
            if not accn or accn in seen_accn:
                continue
            avail = _conservative_edgar_availability_utc(accepts[idx], f_date)
            if avail > cutoff_utc:
                continue
            seen_accn.add(accn)
            results.append({
                "sec_cik": cik.zfill(10),
                "accession_number": accn,
                "sec_form": "8-K",
                "item_codes": ",".join(sorted(item_set)),
                "accepted_at_utc": avail,
                "available_at_utc": avail,
            })
    return results


def collect_free_sp500_backfill(
    *,
    root: Path,
    development_end: date,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    output: Path = BACKFILL_DIRECTORY,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Resumably collect free SEC company_facts, 2016-2019 daily SIP bars, and pre-2019 8-K releases."""
    check_development_end(development_end)
    cfg = settings or get_settings()
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(COLLECT_FREE_SP500_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("research lane free backfill start")
        dest = paths.write(paths.identity(output))
        dest.mkdir(parents=True, exist_ok=True)
        facts_dest = dest / "company_facts"
        bars_dest = dest / "bars"
        facts_dest.mkdir(parents=True, exist_ok=True)
        bars_dest.mkdir(parents=True, exist_ok=True)

        status_path = dest / "_status.json"
        prior_status: dict[str, Any] = (
            json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        )
        not_found_ciks: set[str] = set(prior_status.get("sec_404_ciks", []))

        panel_dir = paths.directory(panel)
        membership = _read_panel_membership(panel_dir, development_end)
        security_ids = tuple(sorted(membership["security_id"].astype(str).unique()))

        sec_mapping, map_report = resolve_all_sp500_ciks(
            security_ids,
            identity_relations_path=paths.read(SEC_IDENTITY_RELATIONS),
            filings_path=paths.read(FILINGS_DIRECTORY / FILINGS_FILE),
            development_end=development_end,
        )
        all_ciks = set(sec_mapping.values())

        extra_dir = paths.project_root / EXTRA_FACTS_DIR
        found_facts = existing_company_facts_paths(
            all_ciks,
            primary_sources=paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES),
            extra_sources=extra_dir if extra_dir.is_dir() else None,
            backfill_sources=facts_dest,
        )
        missing_ciks = sorted(all_ciks - set(found_facts) - not_found_ciks)

        sec = SecSource(cfg)
        fetched_facts = 0
        for cik in missing_ciks:
            target_file = facts_dest / cik / COMPANY_FACTS_FILE
            if target_file.is_file():
                continue
            try:
                _, raw_resp = sec.fetch_cik_company_facts(cik)
            except Exception as exc:
                if "404" in str(exc):
                    not_found_ciks.add(cik)
                    continue
                raise
            target_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = target_file.with_suffix(".json.tmp")
            tmp_file.write_bytes(raw_resp.body)
            tmp_file.replace(target_file)
            fetched_facts += 1
            if fetched_facts % 20 == 0:
                guard("research lane free backfill sec facts")

        # Step 2: Alpaca SIP daily bars backfill (`<= development_end` strictly).
        alpaca = AlpacaSource(cfg)
        bar_manifest = json.loads((panel_dir / BAR_MANIFEST).read_text(encoding="utf-8"))["artifacts"]
        present_adj_tickers = {str(item["ticker"]) for item in bar_manifest}

        # Tickers active in 2018-2019 need 2016-01-04..2018-05-29 warm-up bars (`adjustment="all"`), including the
        # single overlap day `2018-05-29` so `load_extended_sp500_data` can rescale warm-up bars to match the exact
        # split/dividend adjustment basis of `combined_daily/bars` without any stitch jump.
        day0_syms = sorted(
            set(membership.loc[membership["session"] == np.datetime64("2018-05-29", "D"), "ticker"].astype(str))
            | {"SPY", "QQQ"}
        )
        added_early_syms = sorted(
            set(membership.loc[membership["session"] <= np.datetime64("2019-07-09", "D"), "ticker"].astype(str))
            - set(day0_syms)
        )
        warmup_adj_path = bars_dest / "warmup_adjusted_20160104_20180529.parquet"
        if not warmup_adj_path.is_file():
            frames_w: list[pd.DataFrame] = [
                _fetch_alpaca_daily_batch(
                    alpaca, day0_syms, date(2016, 1, 4), date(2018, 5, 29),
                    adjustment="all", asof_day=date(2018, 5, 29),
                ),
            ]
            if added_early_syms:
                frames_w.append(
                    _fetch_alpaca_daily_batch(
                        alpaca, added_early_syms, date(2016, 1, 4), date(2018, 5, 29),
                        adjustment="all", asof_day=date(2019, 7, 8),
                    )
                )
            w_df = pd.concat(frames_w, ignore_index=True).drop_duplicates(["ticker", "session_date"], keep="first")
            cast(Any, pq).write_table(pa.Table.from_pandas(w_df, preserve_index=False), warmup_adj_path)

        # Any membership ticker absent from `combined_daily/bars`: fetch 2016-01-04..last_seen_session (`<= 2024-05-28`).
        all_member_tickers = sorted(set(membership["ticker"].astype(str)) - present_adj_tickers)
        missing_adj_path = bars_dest / "missing_members_adjusted_20160104_20240528.parquet"
        if not missing_adj_path.is_file():
            if all_member_tickers:
                frames_m: list[pd.DataFrame] = []
                last_by_tk = (
                    membership.loc[membership["ticker"].isin(all_member_tickers)]
                    .groupby("ticker")["session"]
                    .max()
                )
                for last_dt64, grp_s in last_by_tk.groupby(last_by_tk):
                    end_tk = min(pd.Timestamp(last_dt64).date(), development_end)
                    syms_tk = sorted(str(k) for k in grp_s.index)
                    frames_m.append(
                        _fetch_alpaca_daily_batch(
                            alpaca, syms_tk, date(2016, 1, 4), end_tk, adjustment="all", asof_day=end_tk,
                        )
                    )
                m_df = pd.concat(frames_m, ignore_index=True).drop_duplicates(["ticker", "session_date"], keep="last")
            else:
                m_df = _fetch_alpaca_daily_batch(
                    alpaca, [], date(2018, 5, 29), development_end, adjustment="all", asof_day=development_end,
                )
            cast(Any, pq).write_table(pa.Table.from_pandas(m_df, preserve_index=False), missing_adj_path)

        # Unadjusted daily closes (`adjustment="raw"`) for 2018-05-29..2019-07-08 + securities missing raw bars.
        early_raw_path = bars_dest / "early_and_missing_raw_20180529_20240528.parquet"
        if not early_raw_path.is_file():
            raw_manifest = json.loads(
                (paths.directory(UNADJUSTED_DIRECTORY) / "_manifest.json").read_text(encoding="utf-8")
            )["unit_artifacts"]
            raw_covered_sec = {str(item["security_id"]) for item in raw_manifest if item.get("role") == "stock"}
            early_mem = membership.loc[membership["session"] < np.datetime64("2019-07-09", "D")]
            last_early = early_mem.groupby("ticker")["session"].max()
            still_listed_syms = sorted(
                str(tk) for tk, mx in last_early.items() if pd.Timestamp(mx).date() >= date(2019, 7, 1)
            )
            exited_early = {
                str(tk): pd.Timestamp(mx).date()
                for tk, mx in last_early.items()
                if pd.Timestamp(mx).date() < date(2019, 7, 1)
            }
            frames_r: list[pd.DataFrame] = []
            if still_listed_syms:
                frames_r.append(
                    _fetch_alpaca_daily_batch(
                        alpaca, still_listed_syms, date(2018, 5, 29), date(2019, 7, 8),
                        adjustment="raw", asof_day=date(2019, 7, 8),
                    )
                )
            for sym_ex, end_ex in sorted(exited_early.items()):
                frames_r.append(
                    _fetch_alpaca_daily_batch(
                        alpaca, [sym_ex], date(2018, 5, 29), end_ex,
                        adjustment="raw", asof_day=end_ex,
                    )
                )

            uncovered_rows = membership.loc[
                (membership["session"] >= np.datetime64("2019-07-09", "D"))
                & (~membership["security_id"].astype(str).isin(raw_covered_sec))
            ]
            if len(uncovered_rows):
                last_uncov = uncovered_rows.groupby("ticker")["session"].max()
                for last_dt64, grp_u in last_uncov.groupby(last_uncov):
                    end_u = min(pd.Timestamp(last_dt64).date(), development_end)
                    syms_u = sorted(str(k) for k in grp_u.index)
                    frames_r.append(
                        _fetch_alpaca_daily_batch(
                            alpaca, syms_u, date(2019, 7, 9), end_u, adjustment="raw", asof_day=end_u,
                        )
                    )
            r_df = (
                pd.concat(frames_r, ignore_index=True).drop_duplicates(["ticker", "session_date"], keep="last")
                if frames_r
                else _fetch_alpaca_daily_batch(
                    alpaca, [], date(2018, 5, 29), development_end, adjustment="raw", asof_day=development_end,
                )
            )
            cast(Any, pq).write_table(pa.Table.from_pandas(r_df, preserve_index=False), early_raw_path)

        # Step 3: Pre-2019-07-09 SEC 8-K Item 2.02 releases (`2018-01-01..2019-07-08`).
        pre2019_8k_path = dest / "pre2019_8k_releases.parquet"
        if not pre2019_8k_path.is_file():
            cik_to_secs: dict[str, list[str]] = {}
            for sec_id, cik_val in sec_mapping.items():
                cik_to_secs.setdefault(cik_val, []).append(sec_id)

            release_rows: list[dict[str, Any]] = []
            for idx_c, cik_val in enumerate(sorted(cik_to_secs)):
                if cik_val in not_found_ciks:
                    continue
                try:
                    items_8k = _extract_8k_202_from_submissions(
                        sec, cik_val, date(2018, 1, 1), date(2019, 7, 8),
                    )
                except Exception as exc:
                    if "404" in str(exc):
                        continue
                    raise
                for row_8k in items_8k:
                    for sec_id in cik_to_secs[cik_val]:
                        release_rows.append({"security_id": sec_id, **row_8k})
                if (idx_c + 1) % 40 == 0:
                    guard("research lane free backfill sec 8k")

            rel_df = (
                pd.DataFrame(release_rows, columns=list(FILING_COLUMNS))
                if release_rows
                else pd.DataFrame(columns=list(FILING_COLUMNS))
            )
            cast(Any, pq).write_table(pa.Table.from_pandas(rel_df, preserve_index=False), pre2019_8k_path)

        final_found = existing_company_facts_paths(
            all_ciks,
            primary_sources=paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES),
            extra_sources=extra_dir if extra_dir.is_dir() else None,
            backfill_sources=facts_dest,
        )
        summary = {
            "development_end": development_end.isoformat(),
            "security_cik_mapping": map_report,
            "company_facts_total_available_ciks": len(final_found),
            "company_facts_newly_fetched": fetched_facts,
            "sec_404_ciks": sorted(not_found_ciks),
            "files": {
                "warmup_adjusted": warmup_adj_path.name,
                "missing_members_adjusted": missing_adj_path.name,
                "early_and_missing_raw": early_raw_path.name,
                "pre2019_8k_releases": pre2019_8k_path.name,
            },
        }
        status_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return summary


def _trailing_finite_count(matrix: npt.NDArray[np.float32], window: int) -> npt.NDArray[np.int32]:
    """Per-cell count of finite positive prices in `matrix[max(0, r - window + 1) : r + 1]`."""
    valid = (np.isfinite(matrix) & (matrix > 0.0)).astype(np.int32)
    csum = np.cumsum(valid, axis=0, dtype=np.int32)
    out = csum.copy()
    if matrix.shape[0] > window:
        out[window:] = csum[window:] - csum[:-window]
    return out


def load_extended_sp500_data(
    paths: EvidencePaths,
    *,
    development_end: date,
    panel_rel: Path = PANEL_DIRECTORY,
    backfill_rel: Path = BACKFILL_DIRECTORY,
    unseal_final_test_period: bool = False,
) -> ExtendedSP500Data:
    """Load the `2016..development_end` extended S&P 500 panel, raw closes, releases, and company facts."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    panel_root = paths.directory(panel_rel)
    backfill_root = paths.directory(backfill_rel)
    end_iso = development_end.isoformat()

    bar_manifest_path = panel_root / BAR_MANIFEST
    panel_manifest_path = panel_root / PANEL_MANIFEST
    warmup_adj_path = backfill_root / "bars" / "warmup_adjusted_20160104_20180529.parquet"
    missing_adj_path = backfill_root / "bars" / "missing_members_adjusted_20160104_20240528.parquet"
    early_raw_path = backfill_root / "bars" / "early_and_missing_raw_20180529_20240528.parquet"
    pre2019_8k_path = backfill_root / "pre2019_8k_releases.parquet"
    filings_path = paths.read(FILINGS_DIRECTORY / FILINGS_FILE)
    relations_path = paths.read(SEC_IDENTITY_RELATIONS)

    hashes = {
        "combined_daily_manifest_sha256": file_sha256(bar_manifest_path),
        "final_manifest_sha256": file_sha256(panel_manifest_path),
        "warmup_adjusted_sha256": file_sha256(warmup_adj_path),
        "missing_members_adjusted_sha256": file_sha256(missing_adj_path),
        "early_and_missing_raw_sha256": file_sha256(early_raw_path),
        "pre2019_8k_releases_sha256": file_sha256(pre2019_8k_path),
        "sec_filings_parquet_sha256": file_sha256(filings_path),
        "sec_identity_relations_sha256": file_sha256(relations_path),
    }

    try:
        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
    except Exception:
        pass

    membership = _read_panel_membership(
        panel_root, development_end, unseal_final_test_period=unseal_final_test_period,
    )
    release_unused_process_memory()
    security_ids = tuple(sorted(membership["security_id"].astype(str).unique()))
    col_index = pd.Index(security_ids)

    artifacts = json.loads(bar_manifest_path.read_text(encoding="utf-8"))["artifacts"]
    cutoff_ts = (pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)).tz_convert("UTC")
    stitch_day = np.datetime64(EARLY_DECISION_START, "D")

    # Step A: Read ONLY SPY and QQQ first so `sessions` calendar is fixed before streaming stock files.
    bench_post: dict[str, pd.DataFrame] = {}
    for item in artifacts:
        ticker = str(item["ticker"])
        if ticker not in ("SPY", "QQQ"):
            continue
        tbl = cast(Any, pq).read_table(
            panel_root / BAR_DIRECTORY / str(item["path"]),
            columns=["bar_start_utc", "open", "close"],
            partitioning=None,
            filters=[("bar_start_utc", "<", cutoff_ts.to_pydatetime())],
            use_threads=False,
        )
        df = tbl.to_pandas()
        del tbl
        sess = pd.to_datetime(df["bar_start_utc"], utc=True).dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None).dt.floor("D")
        df["session"] = sess.to_numpy().astype("datetime64[D]")
        bench_post[ticker] = df.loc[df["session"] <= np.datetime64(development_end, "D"), ["session", "open", "close"]]

    warm_bench_tbl = cast(Any, pq).read_table(
        warmup_adj_path,
        columns=["ticker", "session_date", "open", "close"],
        partitioning=None,
        filters=[("ticker", "in", ["SPY", "QQQ"]), ("session_date", "<=", end_iso)],
        use_threads=False,
    )
    warm_bench_df = warm_bench_tbl.to_pandas()
    del warm_bench_tbl
    bench_frames: dict[str, list[pd.DataFrame]] = {k: [v] for k, v in bench_post.items()}
    if len(warm_bench_df):
        warm_bench_df["session"] = pd.to_datetime(warm_bench_df["session_date"]).to_numpy().astype("datetime64[D]")
        for b_sym in ("SPY", "QQQ"):
            b_post_on = bench_post[b_sym].loc[bench_post[b_sym]["session"] == stitch_day]
            b_warm_on = warm_bench_df.loc[(warm_bench_df["ticker"] == b_sym) & (warm_bench_df["session"] == stitch_day)]
            sub_b = warm_bench_df.loc[
                (warm_bench_df["ticker"] == b_sym) & (warm_bench_df["session"] < stitch_day),
                ["session", "open", "close"],
            ].copy()
            if len(sub_b):
                if len(b_post_on) and len(b_warm_on) and float(b_warm_on["close"].iloc[0]) > 0.0:
                    b_scale = float(b_post_on["close"].iloc[0]) / float(b_warm_on["close"].iloc[0])
                    sub_b["open"] = sub_b["open"] * b_scale
                    sub_b["close"] = sub_b["close"] * b_scale
                bench_frames[b_sym].append(sub_b)
    del warm_bench_df, bench_post

    spy_df = pd.concat(bench_frames["SPY"], ignore_index=True).drop_duplicates("session", keep="first").sort_values("session")
    qqq_df = pd.concat(bench_frames["QQQ"], ignore_index=True).drop_duplicates("session", keep="first").sort_values("session")
    sessions = spy_df["session"].to_numpy().astype("datetime64[D]")
    qqq_aligned = pd.DataFrame({"session": sessions}).merge(qqq_df, on="session", how="left")
    spy_open_arr = spy_df["open"].to_numpy(dtype=np.float32)
    spy_close_arr = spy_df["close"].to_numpy(dtype=np.float32)
    qqq_open_arr = qqq_aligned["open"].to_numpy(dtype=np.float32)
    qqq_close_arr = qqq_aligned["close"].to_numpy(dtype=np.float32)
    del bench_frames, spy_df, qqq_df, qqq_aligned
    release_unused_process_memory()

    # Step B: Build compact integer index lookups (`ticker -> (sorted_r_idx, c_idx)`) for streaming matrix placement.
    earliest_mem = (
        membership.sort_values("session")
        .drop_duplicates("security_id", keep="first")[["ticker", "security_id"]]
    )
    dup_early_tk = earliest_mem["ticker"].duplicated(keep=False)
    unambiguous_early = earliest_mem.loc[~dup_early_tk]
    early_col_lookup: dict[str, int] = {
        str(tk): int(col_index.get_indexer([str(sid)])[0])
        for tk, sid in zip(unambiguous_early["ticker"], unambiguous_early["security_id"], strict=False)
    }

    key_sec = membership.groupby(["ticker", "session"])["security_id"].transform("nunique")
    dup_sec = membership.duplicated(["security_id", "session"], keep=False)
    usable_mem = membership.loc[
        key_sec.eq(1) & ~dup_sec & membership["session"].isin(sessions),
        ["ticker", "session", "security_id"],
    ].sort_values(["ticker", "session"])

    mem_lookup: dict[str, tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]] = {}
    for tk_key, grp_m in usable_mem.groupby("ticker", sort=False):
        r_m = np.searchsorted(sessions, grp_m["session"].to_numpy().astype("datetime64[D]")).astype(np.int32)
        c_m = col_index.get_indexer(grp_m["security_id"].astype(str)).astype(np.int32)
        mem_lookup[str(tk_key)] = (r_m, c_m)

    shape = (len(sessions), len(security_ids))
    matrices: dict[str, npt.NDArray[np.float32]] = {
        field_name: np.full(shape, np.nan, dtype=np.float32) for field_name in PRICE_FIELDS
    }
    base_stock_stitch: dict[str, float] = {}
    n_sess = len(sessions)

    def _place_post_bars(tk: str, df_b: pd.DataFrame, *, overwrite: bool) -> None:
        on_st = df_b.loc[df_b["session"] == stitch_day, "close"]
        if len(on_st) and np.isfinite(on_st.iloc[0]) and float(on_st.iloc[0]) > 0.0 and tk not in base_stock_stitch:
            base_stock_stitch[tk] = float(on_st.iloc[0])
        pair = mem_lookup.get(tk)
        if pair is None:
            return
        mem_r, mem_c = pair
        sub = df_b.loc[df_b["session"] >= stitch_day]
        if sub.empty:
            return
        b_sess = sub["session"].to_numpy().astype("datetime64[D]")
        r_cand = np.searchsorted(sessions, b_sess)
        in_cal_mask = (r_cand < n_sess) & (sessions[np.minimum(r_cand, n_sess - 1)] == b_sess)
        if not bool(in_cal_mask.any()):
            return
        r_sub = r_cand[in_cal_mask].astype(np.int32)
        pos = np.searchsorted(mem_r, r_sub)
        hit = (pos < len(mem_r)) & (mem_r[np.minimum(pos, len(mem_r) - 1)] == r_sub)
        if not bool(hit.any()):
            return
        rr = r_sub[hit]
        cc = mem_c[pos[hit]]
        idx_valid = np.flatnonzero(in_cal_mask)[hit]
        if not overwrite:
            empty_cells = ~np.isfinite(matrices["close"][rr, cc])
            if not bool(empty_cells.any()):
                return
            rr = rr[empty_cells]
            cc = cc[empty_cells]
            idx_valid = idx_valid[empty_cells]
        for field_name in PRICE_FIELDS:
            matrices[field_name][rr, cc] = sub[field_name].to_numpy(dtype=np.float32)[idx_valid]

    skipped_etfs = {"SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"}
    for idx_art, item in enumerate(artifacts):
        ticker = str(item["ticker"])
        if ticker in skipped_etfs:
            continue
        tbl = cast(Any, pq).read_table(
            panel_root / BAR_DIRECTORY / str(item["path"]),
            columns=["bar_start_utc", "open", "high", "low", "close", "volume"],
            partitioning=None,
            filters=[("bar_start_utc", "<", cutoff_ts.to_pydatetime())],
            use_threads=False,
        )
        df = tbl.to_pandas()
        del tbl
        if not df.empty:
            sess = (
                pd.to_datetime(df["bar_start_utc"], utc=True)
                .dt.tz_convert(MARKET_TIMEZONE)
                .dt.tz_localize(None)
                .dt.floor("D")
            )
            df["session"] = sess.to_numpy().astype("datetime64[D]")
            df = df.loc[df["session"] <= np.datetime64(development_end, "D")]
            if not df.empty:
                _place_post_bars(ticker, df, overwrite=True)
        del df
        if (idx_art + 1) % 80 == 0:
            release_unused_process_memory()
    release_unused_process_memory()

    # Step C: Place missing_adj_path (`overwrite=False`) and rescaled pre-2018-05-29 warmup_adj_path directly.
    miss_tbl = cast(Any, pq).read_table(
        missing_adj_path,
        columns=["ticker", "session_date", "open", "high", "low", "close", "volume"],
        partitioning=None,
        filters=[("session_date", "<=", end_iso)],
        use_threads=False,
    )
    miss_df = miss_tbl.to_pandas()
    del miss_tbl
    if len(miss_df):
        if str(miss_df["session_date"].max()) > end_iso:
            raise SealedPeriodError(f"adjusted bars after {end_iso} reached memory from {missing_adj_path}")
        miss_df["session"] = pd.to_datetime(miss_df["session_date"]).to_numpy().astype("datetime64[D]")
        for tk_m, grp_ms in miss_df.groupby("ticker", sort=False):
            if str(tk_m) not in ("SPY", "QQQ"):
                _place_post_bars(str(tk_m), grp_ms, overwrite=False)
                # Also place pre-warmup rows for missing members if unambiguous in early_col_lookup!
                c_early = early_col_lookup.get(str(tk_m), -1)
                if c_early >= 0:
                    pre_ms = grp_ms.loc[grp_ms["session"] < stitch_day]
                    if len(pre_ms):
                        b_s = pre_ms["session"].to_numpy().astype("datetime64[D]")
                        rc = np.searchsorted(sessions, b_s)
                        ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
                        if bool(ok.any()):
                            for field_name in PRICE_FIELDS:
                                matrices[field_name][rc[ok], c_early] = pre_ms[field_name].to_numpy(dtype=np.float32)[ok]
    del miss_df
    release_unused_process_memory()

    warm_tbl = cast(Any, pq).read_table(
        warmup_adj_path,
        columns=["ticker", "session_date", "open", "high", "low", "close", "volume"],
        partitioning=None,
        filters=[("session_date", "<=", end_iso)],
        use_threads=False,
    )
    warm_df = warm_tbl.to_pandas()
    del warm_tbl
    if len(warm_df):
        if str(warm_df["session_date"].max()) > end_iso:
            raise SealedPeriodError(f"adjusted bars after {end_iso} reached memory from {warmup_adj_path}")
        warm_df["session"] = pd.to_datetime(warm_df["session_date"]).to_numpy().astype("datetime64[D]")
        w_on = warm_df.loc[warm_df["session"] == stitch_day].drop_duplicates("ticker", keep="first")
        warm_stitch = {
            str(tk): float(cl)
            for tk, cl in zip(w_on["ticker"], w_on["close"], strict=False)
            if np.isfinite(cl) and float(cl) > 0.0
        }
        del w_on
        for tk_w, grp_w in warm_df.groupby("ticker", sort=False):
            tk_str = str(tk_w)
            c_early = early_col_lookup.get(tk_str, -1)
            if c_early < 0 or tk_str in ("SPY", "QQQ"):
                continue
            sub_w = grp_w.loc[grp_w["session"] < stitch_day]
            if sub_w.empty:
                continue
            b_s = sub_w["session"].to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if not bool(ok.any()):
                continue
            s_factor = (
                base_stock_stitch[tk_str] / warm_stitch[tk_str]
                if tk_str in base_stock_stitch and tk_str in warm_stitch and warm_stitch[tk_str] > 0.0
                else 1.0
            )
            rr_w = rc[ok]
            for px_col in ("open", "high", "low", "close"):
                matrices[px_col][rr_w, c_early] = (sub_w[px_col].to_numpy(dtype=np.float64)[ok] * s_factor).astype(
                    np.float32
                )
            matrices["volume"][rr_w, c_early] = (sub_w["volume"].to_numpy(dtype=np.float64)[ok] / s_factor).astype(
                np.float32
            )
    del warm_df
    release_unused_process_memory()

    # Causal eligibility on >= 2018-05-29: retain original `pre_catalyst_feature_eligible` where True, AND enable
    # 2018-05-29..2019-07-08 rows where membership was known at decision time and >= 252 trailing closes exist!
    mem_in_cal = membership.loc[membership["session"].isin(sessions)]
    m_row = np.searchsorted(sessions, mem_in_cal["session"].to_numpy().astype("datetime64[D]"))
    m_col = col_index.get_indexer(mem_in_cal["security_id"].astype(str))
    member_known = (mem_in_cal["membership_available_at_utc"] <= mem_in_cal["decision_time_utc"]).fillna(False).to_numpy()
    clean_id = (
        mem_in_cal.groupby(["ticker", "session"])["security_id"].transform("nunique").eq(1)
        & ~mem_in_cal.duplicated(["security_id", "session"], keep=False)
    ).to_numpy()
    orig_elig = mem_in_cal[ELIGIBILITY_COLUMN].fillna(False).astype(bool).to_numpy()

    trailing_252 = _trailing_finite_count(matrices["close"], MINIMUM_WARMUP_CLOSES)
    has_warmup = (trailing_252[m_row, m_col] >= MINIMUM_WARMUP_CLOSES) & np.isfinite(matrices["close"][m_row, m_col])
    keep_elig = member_known & clean_id & (orig_elig | has_warmup)

    eligible = np.zeros(shape, dtype=bool)
    eligible[m_row[keep_elig], m_col[keep_elig]] = True
    # Enforce zero eligible cells before 2018-05-29 (warm-up only).
    first_mem_row = int(np.searchsorted(sessions, np.datetime64(EARLY_DECISION_START, "D"), side="left"))
    eligible[:first_mem_row, :] = False

    sector_values = mem_in_cal["sector"].astype("string")
    sector_names = tuple(sorted(str(s) for s in sector_values.dropna().unique()))
    sector = np.full(shape, -1, dtype=np.int16)
    sector[m_row, m_col] = pd.Index(sector_names).get_indexer(sector_values.fillna("")).astype(np.int16)

    ext_panel = ResearchPanel(
        sessions=sessions,
        security_ids=security_ids,
        open=matrices["open"],
        high=matrices["high"],
        low=matrices["low"],
        close=matrices["close"],
        volume=matrices["volume"],
        eligible=eligible,
        sector=sector,
        sector_names=sector_names,
        spy_open=spy_open_arr,
        spy_close=spy_close_arr,
        qqq_open=qqq_open_arr,
        qqq_close=qqq_close_arr,
        development_end=development_end,
        load_report={
            "development_end": end_iso,
            "eligibility_column": f"{ELIGIBILITY_COLUMN}_or_252d_warmup_with_pit_membership",
            "sessions": {"count": len(sessions), "first": str(sessions[0]), "last": str(sessions[-1])},
            "first_eligible_decision_session": str(sessions[first_mem_row]),
            "eligible_cells_total": int(eligible.sum()),
            "eligible_cells_pre_20190709": int(
                eligible[: int(np.searchsorted(sessions, np.datetime64(NEWS_DECISION_START, "D")))].sum()
            ),
        },
        input_sha256=hashes,
    )

    # Unadjusted closes: merge canonical `2019-07-09+` raw bars with backfill `2018-05-29+` raw bars.
    del membership, mem_in_cal
    release_unused_process_memory()
    unadj_base, unadj_rep, unadj_sha = load_unadjusted_close(paths.directory(UNADJUSTED_DIRECTORY), ext_panel)
    release_unused_process_memory()
    hashes.update(unadj_sha)
    raw_extra_tbl = cast(Any, pq).read_table(
        early_raw_path,
        columns=["ticker", "session_date", "close"],
        partitioning=None,
        filters=[("session_date", "<=", end_iso)],
        use_threads=False,
    )
    raw_extra_df = raw_extra_tbl.to_pandas()
    del raw_extra_tbl
    if len(raw_extra_df):
        if str(raw_extra_df["session_date"].max()) > end_iso:
            raise SealedPeriodError(f"raw bars after {end_iso} reached memory from {early_raw_path}")
        raw_extra_df["session"] = pd.to_datetime(raw_extra_df["session_date"]).to_numpy().astype("datetime64[D]")
        raw_mapped = raw_extra_df.merge(usable_mem, on=["ticker", "session"], how="inner")
        raw_mapped = raw_mapped.loc[raw_mapped["session"].isin(sessions)].drop_duplicates(["security_id", "session"])
        rr = np.searchsorted(sessions, raw_mapped["session"].to_numpy().astype("datetime64[D]"))
        rc = col_index.get_indexer(raw_mapped["security_id"].astype(str))
        fill_mask = ~np.isfinite(unadj_base[rr, rc])
        unadj_base[rr[fill_mask], rc[fill_mask]] = raw_mapped["close"].to_numpy(dtype=np.float64)[fill_mask]
        del raw_mapped
    del raw_extra_df, usable_mem
    release_unused_process_memory()

    # Merge SEC 8-K Item 2.02 releases: pre-2019 backfill + retained 2019-2024 inventory.
    filings_post = read_sec_filings(filings_path, development_end)
    post_mask = filings_post["sec_form"].eq("8-K") & filings_post["item_codes"].fillna("").astype(str).map(
        lambda t: "2.02" in {c.strip() for c in t.split(",")}
    )
    rel_post = filings_post.loc[post_mask, ["security_id", "accession_number", "accepted_at_utc", "available_at_utc"]]
    pre_tbl = cast(Any, pq).read_table(pre2019_8k_path, partitioning=None)
    rel_pre_df = pre_tbl.to_pandas()
    if len(rel_pre_df):
        cutoff_8k = pd.Timestamp(development_end, tz=MARKET_TIMEZONE) + pd.Timedelta(days=1)
        if bool((pd.to_datetime(rel_pre_df["available_at_utc"], utc=True) >= cutoff_8k.tz_convert("UTC")).any()):
            raise SealedPeriodError("pre2019_8k_releases contains sealed timestamp")
        rel_pre = rel_pre_df[["security_id", "accession_number", "accepted_at_utc", "available_at_utc"]]
        all_releases = pd.concat([rel_pre, rel_post], ignore_index=True).drop_duplicates(["security_id", "accession_number"])
    else:
        all_releases = rel_post.reset_index(drop=True)

    sec_cik_map, map_rep = resolve_all_sp500_ciks(
        security_ids,
        identity_relations_path=relations_path,
        filings_path=filings_path,
        development_end=development_end,
        unseal_final_test_period=unseal_final_test_period,
    )
    extra_dir = paths.project_root / EXTRA_FACTS_DIR
    facts_paths = existing_company_facts_paths(
        set(sec_cik_map.values()),
        primary_sources=paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES),
        extra_sources=extra_dir if extra_dir.is_dir() else None,
        backfill_sources=backfill_root / "company_facts",
    )
    # Keep only securities whose CIK actually has a `company_facts.json` file.
    active_cik_map = {sec_id: cik for sec_id, cik in sec_cik_map.items() if cik in facts_paths}

    return ExtendedSP500Data(
        panel=ext_panel,
        unadjusted_close=unadj_base,
        releases=all_releases,
        security_cik_map=active_cik_map,
        company_facts_paths=facts_paths,
        input_sha256=hashes,
        report={
            "panel": ext_panel.load_report,
            "security_cik_mapping": map_rep,
            "securities_with_company_facts": len(active_cik_map),
            "distinct_ciks_with_company_facts": len(facts_paths),
            "securities_with_unadjusted_close": int(np.isfinite(unadj_base).any(axis=0).sum()),
            "unadjusted_base_report": unadj_rep,
            "total_8k_202_release_rows": len(all_releases),
        },
    )
