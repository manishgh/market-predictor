"""Point-in-time `mid_small_ex_sp500` cohort builder & authority verifier (`R2`).

Rebuilds the decision-date `mid_small_ex_sp500` cohort from `EvidencePaths`, `roster_spans.parquet`
(or `extended_roster_spans.parquet`), `cik_sectors.json`, and point-in-time S&P 500 membership
authorities, applying issuer CIK identity, effective ticker spans (including ticker renames), S&P 500
constituent exclusions, unadjusted price (`>= $5.00`), `252`-session warm-up, and `60`-session
dollar-volume Top-`1,200` eligibility before any peer calculation.

Fails closed (`DataReadinessError`) naming the missing authority and affected date whenever S&P 500
membership or SEC roster span coverage ends before `as_of_date` or was observed after `source_cutoff_utc`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.config import Settings, get_settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.research_lane import beta_neutral, sealed_holdout, wider_universe
from market_predictor.research_lane.company_facts import (
    COMPANY_FACTS_DIRECTORY,
    COMPANY_FACTS_FILE,
    COMPANY_FACTS_SOURCES,
)
from market_predictor.research_lane.free_backfill import (
    BACKFILL_DIRECTORY,
    EXTRA_FACTS_DIR,
    SEC_IDENTITY_RELATIONS,
    _fetch_alpaca_daily_batch,
    _read_panel_membership,
    existing_company_facts_paths,
    resolve_all_sp500_ciks,
)
from market_predictor.research_lane.panel import (
    DEFAULT_DEVELOPMENT_END,
    PANEL_DIRECTORY,
    PRICE_FIELDS,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
)
from market_predictor.research_lane.prospective_contracts import parse_utc_timestamp
from market_predictor.research_lane.sec_filings import FILINGS_DIRECTORY, FILINGS_FILE
from market_predictor.research_lane.wider_universe import (
    TOP_LIQUID_COUNT,
    WIDER_UNIVERSE_DIRECTORY,
    compute_pit_liquidity_eligibility,
)
from market_predictor.sources.alpaca import AlpacaSource

CANONICAL_INDEX_MEMBERSHIP_REL: Final = Path("data/canonical/index_membership")
OBSERVED_INDEX_MEMBERSHIP_REL: Final = Path("data/raw/index_membership")


@dataclass(frozen=True)
class DecisionDateMidSmallCohort:
    """Verified decision-date `mid_small_ex_sp500` cohort, auxiliary inputs, and ticker bridge."""

    panel: ResearchPanel
    unadjusted_close: npt.NDArray[np.float64]
    releases_202: pd.DataFrame
    all_8k_events: pd.DataFrame
    non8k_events: pd.DataFrame
    security_cik_map: dict[str, str]
    company_facts_paths: dict[str, Path]
    active_ticker_by_security_id: dict[str, str]
    security_id_by_active_ticker: dict[str, str]
    authority_report: dict[str, Any]


def _resolve_roster_parquet(
    paths: EvidencePaths,
    *,
    as_of_date: date,
    wider_rel: Path = WIDER_UNIVERSE_DIRECTORY,
    ext_rel: Path = sealed_holdout.SEALED_EXTENSION_DIR,
) -> tuple[Path, str, pd.DataFrame]:
    """Load `roster_spans.parquet` (or `extended_roster_spans.parquet` when covering `> 2024-05-28`) via `EvidencePaths`."""
    wider_dir = paths.directory(wider_rel)
    base_roster_pq = wider_dir / "roster_spans.parquet"
    if not base_roster_pq.is_file():
        raise DataReadinessError(
            f"missing SEC wider roster authority for {as_of_date.isoformat()}: {base_roster_pq} does not exist"
        )

    chosen_pq = base_roster_pq
    chosen_label = (wider_rel / "roster_spans.parquet").as_posix()
    if as_of_date > DEFAULT_DEVELOPMENT_END:
        try:
            ext_dir = paths.directory(ext_rel)
            ext_pq = ext_dir / "extended_roster_spans.parquet"
            if ext_pq.is_file():
                chosen_pq = ext_pq
                chosen_label = (ext_rel / "extended_roster_spans.parquet").as_posix()
        except Exception:
            pass

    df = cast(Any, pq).read_table(chosen_pq, use_threads=False).to_pandas()
    if df.empty:
        raise DataReadinessError(f"roster authority {chosen_label} is empty for {as_of_date.isoformat()}")
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["valid_from"] = pd.to_datetime(df["valid_from"]).dt.date
    df["valid_to"] = pd.to_datetime(df["valid_to"]).dt.date
    return chosen_pq, chosen_label, df


def _inspect_latest_post_panel_sp500_membership(
    paths: EvidencePaths,
    *,
    source_cutoff_utc: datetime,
) -> tuple[date | None, str, Path | None, datetime | None]:
    """Inspect canonical & observed S&P 500 membership authorities strictly observed `<= source_cutoff_utc`."""
    best_date: date | None = None
    best_label = "none"
    best_dir: Path | None = None
    later_post_cutoff_obs: datetime | None = None

    for rel_parent, date_key in (
        (CANONICAL_INDEX_MEMBERSHIP_REL, "cutoff_date"),
        (OBSERVED_INDEX_MEMBERSHIP_REL, "effective_horizon_date"),
    ):
        try:
            abs_parent = paths.directory(rel_parent)
        except Exception:
            continue
        if not abs_parent.is_dir():
            continue
        for child in sorted(abs_parent.iterdir()):
            if not child.is_dir():
                continue
            man_path = child / "_manifest.json"
            mem_pq = child / "memberships.parquet"
            if not (man_pq_ok := (man_path.is_file() and mem_pq.is_file())):
                _ = man_pq_ok
                continue
            try:
                man = json.loads(man_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(man, dict) or man.get("status") != "complete":
                continue
            obs_raw = man.get("observed_at_utc")
            if obs_raw is not None:
                obs_dt = parse_utc_timestamp(obs_raw)
                if obs_dt is None:
                    continue
                if obs_dt > source_cutoff_utc:
                    if later_post_cutoff_obs is None or obs_dt < later_post_cutoff_obs:
                        later_post_cutoff_obs = obs_dt
                    continue
            cov_str = man.get(date_key) or man.get("cutoff_date")
            if not isinstance(cov_str, str) or len(cov_str) != 10:
                continue
            cov_day = date.fromisoformat(cov_str)
            if best_date is None or cov_day >= best_date:
                best_date = cov_day
                best_label = (rel_parent / child.name).as_posix()
                best_dir = child
    return best_date, best_label, best_dir, later_post_cutoff_obs


def verify_mid_small_cohort_authorities(
    paths: EvidencePaths,
    *,
    as_of_date: date,
    source_cutoff_utc: datetime,
    panel_rel: Path = PANEL_DIRECTORY,
    wider_rel: Path = WIDER_UNIVERSE_DIRECTORY,
    ext_rel: Path = sealed_holdout.SEALED_EXTENSION_DIR,
    unseal_final_test_period: bool = False,
) -> tuple[pd.DataFrame, Path | None, dict[str, Any]]:
    """Verify that both S&P 500 membership and SEC roster spans causally cover `as_of_date`."""
    wider_dir = paths.directory(wider_rel)
    selected_path = wider_dir / "selected_ciks.json"
    sectors_path = wider_dir / "cik_sectors.json"
    if not selected_path.is_file():
        raise DataReadinessError(f"missing wider selected_ciks.json authority: {selected_path}")
    if not sectors_path.is_file():
        raise DataReadinessError(f"missing wider cik_sectors.json sector authority: {sectors_path}")

    _, roster_label, roster_df = _resolve_roster_parquet(
        paths, as_of_date=as_of_date, wider_rel=wider_rel, ext_rel=ext_rel,
    )
    roster_max_to: date = max(roster_df["valid_to"])

    panel_dir = paths.directory(panel_rel)
    panel_probe_end = min(as_of_date, sealed_holdout.SEALED_HOLDOUT_END)
    allow_panel_read = unseal_final_test_period or (as_of_date > sealed_holdout.SEALED_HOLDOUT_END)
    mem_panel = _read_panel_membership(
        panel_dir, panel_probe_end, unseal_final_test_period=allow_panel_read,
    )
    panel_max_day = pd.Timestamp(mem_panel["session"].max()).date()

    sp500_authority_label = (panel_rel / "final/panel/feature_profile=technical_market").as_posix()
    sp500_covered_through = panel_max_day
    post_panel_mem_dir: Path | None = None

    if as_of_date <= panel_max_day:
        on_day = mem_panel.loc[
            (mem_panel["session"] == np.datetime64(as_of_date, "D"))
            & (mem_panel["membership_available_at_utc"] <= mem_panel["decision_time_utc"]).fillna(False)
        ]
        if on_day.empty:
            raise DataReadinessError(
                f"missing S&P 500 point-in-time membership authority for {as_of_date.isoformat()}: "
                f"{sp500_authority_label} has zero causal membership rows on {as_of_date.isoformat()}"
            )
    else:
        ext_cov_day, ext_label, ext_mem_dir, post_cutoff_obs = _inspect_latest_post_panel_sp500_membership(
            paths, source_cutoff_utc=source_cutoff_utc,
        )
        if ext_cov_day is not None and ext_cov_day > sp500_covered_through:
            sp500_covered_through = ext_cov_day
            sp500_authority_label = ext_label
            post_panel_mem_dir = ext_mem_dir

        if sp500_covered_through < as_of_date:
            post_note = (
                f" (a snapshot observed at {post_cutoff_obs.isoformat()} after cutoff {source_cutoff_utc.isoformat()} "
                "was rightly excluded to prevent lookahead)"
                if post_cutoff_obs is not None
                else ""
            )
            roster_gap_note = (
                f"; SEC roster authority {roster_label} also ends at {roster_max_to.isoformat()}"
                if roster_max_to < as_of_date
                else ""
            )
            raise DataReadinessError(
                f"missing S&P 500 point-in-time membership authority for decision date {as_of_date.isoformat()}: "
                f"latest causal authority {sp500_authority_label} ends at {sp500_covered_through.isoformat()} "
                f"(< {as_of_date.isoformat()}){post_note}{roster_gap_note}"
            )

    if roster_max_to < as_of_date:
        raise DataReadinessError(
            f"missing SEC roster span coverage for decision date {as_of_date.isoformat()}: "
            f"{roster_label} ends at {roster_max_to.isoformat()} (< {as_of_date.isoformat()})"
        )

    return roster_df, post_panel_mem_dir, {
        "as_of_date": as_of_date.isoformat(),
        "source_cutoff_utc": source_cutoff_utc.isoformat(),
        "roster_authority": roster_label,
        "roster_covered_through": roster_max_to.isoformat(),
        "sp500_membership_authority": sp500_authority_label,
        "sp500_covered_through": sp500_covered_through.isoformat(),
    }


def _active_ticker_maps_on_date(
    roster_df: pd.DataFrame,
    security_ids: tuple[str, ...],
    as_of_date: date,
) -> tuple[dict[str, str], dict[str, str]]:
    """Map `cik:0000123456 <-> active ticker on as_of_date` from quarantined `roster_df`."""
    ciks_wanted = {sid.removeprefix("cik:") for sid in security_ids if sid.startswith("cik:")}
    active = roster_df.loc[
        roster_df["cik"].isin(ciks_wanted)
        & (roster_df["valid_from"] <= as_of_date)
        & (roster_df["valid_to"] >= as_of_date)
    ].sort_values(["source_priority", "valid_to"], kind="stable")
    sid_to_tk: dict[str, str] = {}
    tk_to_sid: dict[str, str] = {}
    for rec in active.to_dict(orient="records"):
        sid = f"cik:{str(rec['cik']).zfill(10)}"
        tk = str(rec["ticker"]).upper().strip()
        if sid not in sid_to_tk and tk:
            sid_to_tk[sid] = tk
            tk_to_sid[tk] = sid
    return sid_to_tk, tk_to_sid


def load_decision_date_mid_small_cohort(
    paths: EvidencePaths,
    *,
    as_of_date: date,
    source_cutoff_utc: datetime,
    panel_rel: Path = PANEL_DIRECTORY,
    sp500_backfill_rel: Path = BACKFILL_DIRECTORY,
    wider_rel: Path = WIDER_UNIVERSE_DIRECTORY,
    ext_rel: Path = sealed_holdout.SEALED_EXTENSION_DIR,
    live_prospective: bool = False,
    unseal_final_test_period: bool = False,
    settings: Settings | None = None,
) -> DecisionDateMidSmallCohort:
    """Build or load the verified `mid_small_ex_sp500` cohort and SEC inputs through `as_of_date`."""
    if not live_prospective:
        check_development_end(as_of_date, unseal_final_test_period=unseal_final_test_period)
    elif as_of_date <= sealed_holdout.SEALED_HOLDOUT_END and not unseal_final_test_period:
        raise SealedPeriodError(
            f"live_prospective cohort refuses sealed date {as_of_date.isoformat()} <= "
            f"{sealed_holdout.SEALED_HOLDOUT_END.isoformat()}"
        )

    roster_df, post_panel_mem_dir, auth_report = verify_mid_small_cohort_authorities(
        paths,
        as_of_date=as_of_date,
        source_cutoff_utc=source_cutoff_utc,
        panel_rel=panel_rel,
        wider_rel=wider_rel,
        ext_rel=ext_rel,
        unseal_final_test_period=unseal_final_test_period,
    )
    wider_dir = paths.directory(wider_rel)
    cutoff_ts = pd.Timestamp(source_cutoff_utc).tz_convert("UTC")

    if as_of_date <= DEFAULT_DEVELOPMENT_END:
        wdata = wider_universe.load_wider_universe_data(
            paths,
            development_end=as_of_date,
            subset="mid_small_ex_sp500",
            panel_rel=panel_rel,
            sp500_backfill_rel=sp500_backfill_rel,
            wider_rel=wider_rel,
            unseal_final_test_period=unseal_final_test_period,
        )
        all_8k = wdata.all_8k_events
        if len(all_8k):
            all_8k = all_8k.loc[pd.to_datetime(all_8k["available_at_utc"], utc=True) <= cutoff_ts].copy()
        rel_202 = wdata.releases_202
        if len(rel_202):
            rel_202 = rel_202.loc[pd.to_datetime(rel_202["available_at_utc"], utc=True) <= cutoff_ts].copy()

        non8k_pq = wider_dir / beta_neutral.NON8K_CATALYST_PARQUET
        if non8k_pq.is_file():
            non8k = cast(Any, pq).read_table(non8k_pq, use_threads=False).to_pandas()
            if len(non8k):
                non8k["available_at_utc"] = pd.to_datetime(non8k["available_at_utc"], utc=True)
                non8k = non8k.loc[non8k["available_at_utc"] <= cutoff_ts].copy()
        else:
            non8k = pd.DataFrame(
                columns=["security_id", "sec_cik", "accession_number", "sec_form", "catalyst_family", "available_at_utc"]
            )
        sid_to_tk, tk_to_sid = _active_ticker_maps_on_date(roster_df, wdata.panel.security_ids, as_of_date)
        return DecisionDateMidSmallCohort(
            panel=wdata.panel,
            unadjusted_close=wdata.unadjusted_close,
            releases_202=rel_202,
            all_8k_events=all_8k,
            non8k_events=non8k,
            security_cik_map=wdata.security_cik_map,
            company_facts_paths=wdata.company_facts_paths,
            active_ticker_by_security_id=sid_to_tk,
            security_id_by_active_ticker=tk_to_sid,
            authority_report=auth_report,
        )

    if as_of_date <= sealed_holdout.SEALED_HOLDOUT_END:
        ext_dir = paths.directory(ext_rel)
        ext_files = {
            "ext_roster_parquet": ext_dir / "extended_roster_spans.parquet",
            "bars_adj_parquet": ext_dir / "bars_adj_20240528_20260708.parquet",
            "bars_raw_parquet": ext_dir / "bars_raw_20240528_20260708.parquet",
            "holdout_8k_parquet": ext_dir / "mid_small_8k_20240529_20260708.parquet",
            "holdout_non8k_parquet": ext_dir / "mid_small_non8k_20240529_20260708.parquet",
        }
        wdata_u, non8k_u = sealed_holdout.load_unsealed_mid_small_through_date(
            paths,
            ext_files,
            as_of_date=as_of_date,
            source_cutoff_utc=cutoff_ts,
            panel_rel=panel_rel,
            sp500_backfill_rel=sp500_backfill_rel,
            wider_rel=wider_rel,
        )
        sid_to_tk, tk_to_sid = _active_ticker_maps_on_date(roster_df, wdata_u.panel.security_ids, as_of_date)
        return DecisionDateMidSmallCohort(
            panel=wdata_u.panel,
            unadjusted_close=wdata_u.unadjusted_close,
            releases_202=wdata_u.releases_202,
            all_8k_events=wdata_u.all_8k_events,
            non8k_events=non8k_u,
            security_cik_map=wdata_u.security_cik_map,
            company_facts_paths=wdata_u.company_facts_paths,
            active_ticker_by_security_id=sid_to_tk,
            security_id_by_active_ticker=tk_to_sid,
            authority_report=auth_report,
        )

    # Live post-2026-07-08 path (reached ONLY when both extended roster spans and causal S&P 500 membership cover `as_of_date`)
    cfg = settings or get_settings()
    alpaca = AlpacaSource(cfg)
    selected_ciks = tuple(
        sorted(json.loads((wider_dir / "selected_ciks.json").read_text(encoding="utf-8"))["selected_ciks"])
    )
    cik_sectors: dict[str, str] = json.loads((wider_dir / "cik_sectors.json").read_text(encoding="utf-8"))
    start_day = as_of_date - timedelta(days=430)
    if not unseal_final_test_period:
        start_day = max(start_day, sealed_holdout.SEALED_HOLDOUT_END + timedelta(days=1))

    sub_rost = roster_df.loc[
        roster_df["cik"].isin(selected_ciks) & (roster_df["valid_to"] >= start_day)
    ].copy()
    need_syms = sorted(set(sub_rost["ticker"].astype(str)) | {"SPY", "QQQ"})
    adj_df = _fetch_alpaca_daily_batch(
        alpaca,
        need_syms,
        start_day,
        as_of_date,
        adjustment="all",
        asof_day=as_of_date,
        unseal_final_test_period=True,
    )
    raw_df = _fetch_alpaca_daily_batch(
        alpaca,
        [s for s in need_syms if s not in ("SPY", "QQQ")],
        start_day,
        as_of_date,
        adjustment="raw",
        asof_day=as_of_date,
        unseal_final_test_period=True,
    )

    spy_df = adj_df.loc[adj_df["ticker"] == "SPY"].sort_values("session_date").drop_duplicates("session_date", keep="last")
    qqq_df = adj_df.loc[adj_df["ticker"] == "QQQ"].sort_values("session_date").drop_duplicates("session_date", keep="last")
    sessions = pd.to_datetime(spy_df["session_date"]).to_numpy().astype("datetime64[D]")
    qqq_al = pd.DataFrame({"session_date": spy_df["session_date"].tolist()}).merge(
        qqq_df[["session_date", "open", "close"]], on="session_date", how="left",
    )

    n_sess, n_sec = len(sessions), len(selected_ciks)
    col_of_cik = {c: idx for idx, c in enumerate(selected_ciks)}
    in_span = np.zeros((n_sess, n_sec), dtype=bool)
    tk_ivs: dict[str, list[tuple[int, int, int]]] = {}
    for rec in sub_rost.to_dict(orient="records"):
        c_i = col_of_cik[str(rec["cik"])]
        r0 = int(np.searchsorted(sessions, np.datetime64(rec["valid_from"], "D"), side="left"))
        r1 = int(np.searchsorted(sessions, np.datetime64(rec["valid_to"], "D"), side="right"))
        if r0 < r1:
            in_span[r0:r1, c_i] = True
            tk_ivs.setdefault(str(rec["ticker"]), []).append((r0, r1, c_i))

    matrices: dict[str, npt.NDArray[np.float32]] = {
        f: np.full((n_sess, n_sec), np.nan, dtype=np.float32) for f in PRICE_FIELDS
    }
    stocks_adj = adj_df.loc[~adj_df["ticker"].isin(["SPY", "QQQ"])].copy()
    if len(stocks_adj):
        stocks_adj["session"] = pd.to_datetime(stocks_adj["session_date"]).to_numpy().astype("datetime64[D]")
        for tk_a, grp_a in stocks_adj.groupby("ticker", sort=False):
            ivs = tk_ivs.get(str(tk_a))
            if not ivs:
                continue
            b_s = grp_a["session"].to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if not bool(ok.any()):
                continue
            rr_a = rc[ok]
            idx_a = np.flatnonzero(ok)
            for r0, r1, col_i in ivs:
                wm = (rr_a >= r0) & (rr_a < r1)
                if bool(wm.any()):
                    for f_k in PRICE_FIELDS:
                        matrices[f_k][rr_a[wm], col_i] = grp_a[f_k].to_numpy(dtype=np.float32)[idx_a[wm]]

    unadj = np.full((n_sess, n_sec), np.nan, dtype=np.float64)
    if len(raw_df):
        raw_c = raw_df.copy()
        raw_c["session"] = pd.to_datetime(raw_c["session_date"]).to_numpy().astype("datetime64[D]")
        for tk_r, grp_r in raw_c.groupby("ticker", sort=False):
            ivs = tk_ivs.get(str(tk_r))
            if not ivs:
                continue
            b_s = grp_r["session"].to_numpy().astype("datetime64[D]")
            rc = np.searchsorted(sessions, b_s)
            ok = (rc < n_sess) & (sessions[np.minimum(rc, n_sess - 1)] == b_s)
            if not bool(ok.any()):
                continue
            rr_r = rc[ok]
            cl_r = grp_r["close"].to_numpy(dtype=np.float64)[ok]
            for r0, r1, col_i in ivs:
                wm = (rr_r >= r0) & (rr_r < r1)
                if bool(wm.any()):
                    unadj[rr_r[wm], col_i] = cl_r[wm]

    # Point-in-time S&P 500 constituent exclusion from verified post-panel membership authority
    sp500_mask = np.zeros((n_sess, n_sec), dtype=bool)
    if post_panel_mem_dir is not None and (post_panel_mem_dir / "memberships.parquet").is_file():
        m_tbl = cast(Any, pq).read_table(post_panel_mem_dir / "memberships.parquet", use_threads=False).to_pandas()
        sp_sec_ids = tuple(sorted(m_tbl["security_id"].astype(str).unique()))
        sp_cik_map, _ = resolve_all_sp500_ciks(
            sp_sec_ids,
            identity_relations_path=paths.read(SEC_IDENTITY_RELATIONS),
            filings_path=paths.read(FILINGS_DIRECTORY / FILINGS_FILE),
            development_end=sealed_holdout.SEALED_HOLDOUT_END,
            unseal_final_test_period=True,
        )
        eff_from = pd.to_datetime(m_tbl["effective_from_utc"], utc=True)
        eff_to = pd.to_datetime(m_tbl["effective_to_utc"], utc=True)
        avail = pd.to_datetime(m_tbl["available_at_utc"], utc=True)
        for idx_m, row_m in m_tbl.iterrows():
            sid_m = str(row_m["security_id"])
            cik_m = sp_cik_map.get(sid_m)
            if cik_m is None or cik_m not in col_of_cik:
                continue
            if pd.notna(avail.loc[idx_m]) and avail.loc[idx_m] > cutoff_ts:
                continue
            d0 = eff_from.loc[idx_m].tz_convert("America/New_York").date()
            d1 = eff_to.loc[idx_m].tz_convert("America/New_York").date() if pd.notna(eff_to.loc[idx_m]) else as_of_date
            r0 = int(np.searchsorted(sessions, np.datetime64(d0, "D"), side="left"))
            r1 = int(np.searchsorted(sessions, np.datetime64(d1, "D"), side="right"))
            if r0 < r1:
                sp500_mask[r0:r1, col_of_cik[cik_m]] = True

    elig_top1200 = compute_pit_liquidity_eligibility(
        matrices["close"],
        matrices["volume"],
        unadj,
        in_span,
        sessions,
        decision_start=pd.Timestamp(sessions[0]).date() if n_sess else as_of_date,
        top_n=TOP_LIQUID_COUNT,
    )
    eligible = elig_top1200 & (~sp500_mask)
    keep_cols = np.flatnonzero(eligible.any(axis=0))
    if keep_cols.size == 0:
        # Keep non-S&P active in-span columns with `eligible = all False` so caller can report exact warmup/history gap
        keep_cols = np.flatnonzero((in_span & (~sp500_mask)).any(axis=0))
    kept_ciks = tuple(selected_ciks[int(i)] for i in keep_cols)
    sec_ids_live = tuple(f"cik:{c}" for c in kept_ciks)
    for f_k in PRICE_FIELDS:
        matrices[f_k] = np.ascontiguousarray(matrices[f_k][:, keep_cols])
    unadj = np.ascontiguousarray(unadj[:, keep_cols])
    eligible = np.ascontiguousarray(eligible[:, keep_cols])

    sec_per_col = [str(cik_sectors.get(c, "Industrials")) for c in kept_ciks]
    sec_names_live = tuple(sorted(set(sec_per_col))) if sec_per_col else ("Industrials",)
    sec_codes_live = (
        pd.Index(sec_names_live).get_indexer(sec_per_col).astype(np.int16)
        if sec_per_col
        else np.zeros(0, dtype=np.int16)
    )
    sector_mat_live = np.broadcast_to(sec_codes_live[None, :], (n_sess, len(kept_ciks))).copy()

    live_panel = ResearchPanel(
        sessions=sessions,
        security_ids=sec_ids_live,
        open=matrices["open"],
        high=matrices["high"],
        low=matrices["low"],
        close=matrices["close"],
        volume=matrices["volume"],
        eligible=eligible,
        sector=sector_mat_live,
        sector_names=sec_names_live,
        spy_open=spy_df["open"].to_numpy(dtype=np.float32),
        spy_close=spy_df["close"].to_numpy(dtype=np.float32),
        qqq_open=qqq_al["open"].to_numpy(dtype=np.float32),
        qqq_close=qqq_al["close"].to_numpy(dtype=np.float32),
        development_end=as_of_date,
        load_report={
            "mode": "live_prospective_mid_small_ex_sp500",
            "as_of_date": as_of_date.isoformat(),
            "authority_report": auth_report,
        },
    )

    sp500_bf_dir = paths.directory(sp500_backfill_rel)
    extra_f: Path | None = None
    try:
        cand_extra = paths.directory(EXTRA_FACTS_DIR)
        if cand_extra.is_dir():
            extra_f = cand_extra
    except Exception:
        pass
    facts_paths_live = existing_company_facts_paths(
        set(kept_ciks),
        primary_sources=paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES),
        extra_sources=extra_f,
        backfill_sources=sp500_bf_dir / "company_facts",
    )
    w_facts_dir = wider_dir / "company_facts"
    for c_id in kept_ciks:
        pw = w_facts_dir / c_id / COMPANY_FACTS_FILE
        if c_id not in facts_paths_live and pw.is_file():
            facts_paths_live[c_id] = pw

    sid_to_tk, tk_to_sid = _active_ticker_maps_on_date(roster_df, sec_ids_live, as_of_date)
    empty_8k = pd.DataFrame(columns=["security_id", "accession_number", "item_codes", "accepted_at_utc", "available_at_utc"])
    empty_202 = pd.DataFrame(columns=["security_id", "accession_number", "accepted_at_utc", "available_at_utc"])
    empty_n8 = pd.DataFrame(columns=["security_id", "sec_cik", "accession_number", "sec_form", "catalyst_family", "available_at_utc"])
    return DecisionDateMidSmallCohort(
        panel=live_panel,
        unadjusted_close=unadj,
        releases_202=empty_202,
        all_8k_events=empty_8k,
        non8k_events=empty_n8,
        security_cik_map={f"cik:{c}": c for c in kept_ciks if c in facts_paths_live},
        company_facts_paths=facts_paths_live,
        active_ticker_by_security_id=sid_to_tk,
        security_id_by_active_ticker=tk_to_sid,
        authority_report={**auth_report, "utc_checked": datetime.now(UTC).isoformat()},
    )


__all__ = [
    "CANONICAL_INDEX_MEMBERSHIP_REL",
    "DecisionDateMidSmallCohort",
    "OBSERVED_INDEX_MEMBERSHIP_REL",
    "load_decision_date_mid_small_cohort",
    "verify_mid_small_cohort_authorities",
]
