"""Step 5h: Prospective Forward-Score & OPRA Option Outcome Maturation Evaluator.

Grades immutable dated forward snapshots (`data/research/research_lane/forward_records/YYYY-MM-DD`)
against realized post-decision open-to-close research-label returns (`close[t+h] / open[t+1] - 1`)
and matched-interval `SPY` returns strictly up to `as_of_date` (`> 2026-07-08`), keeping matured
`(snapshot_date, horizon)` pairs explicitly separate from `pending_entry_bar` and
`pending_maturation` pairs (`zero` sealed-window access and `zero` post-`as_of_date` reads).
"""
from __future__ import annotations

import json
import math
import os
import shutil
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.config import Settings, get_settings
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research_lane.forward_record import (
    FORWARD_RECORD_DIRECTORY,
    FROZEN_PROSPECTIVE_CANDIDATES,
    HISTORICAL_SEALED_END,
    resolve_latest_completed_session,
)
from market_predictor.research_lane.panel import MARKET_TIMEZONE, SealedPeriodError
from market_predictor.research_lane.prospective_contracts import audit_snapshot_admission
from market_predictor.research_lane.trial_log import git_identity
from market_predictor.sources.alpaca import AlpacaSource

EVALUATE_FORWARD_COMMAND: Final = "evaluate-forward-research-scores"
FORWARD_EVAL_DIRECTORY: Final = Path("data/research/research_lane/forward_evaluations")
DEFAULT_EVAL_HORIZONS: Final[tuple[int, ...]] = (1, 2, 3, 5, 10, 21, 63)
SIGNAL_COLUMNS: Final[tuple[str, ...]] = (
    "score_mid_small_bn_sec_xgb_opra_10d",
    "score_mid_small_bn_sec_xgb_10d",
    "score_mid_small_bn_sec_xgb_21d",
    "score_price_equal_12",
    "score_beta_neutral_price_12",
    "score_r2_opra_conditioned",
    "score_reversal_short",
    "score_reversal_skew_vetoed",
)
ONE_WAY_ENTRY_COST: Final = 0.0010  # 0.5 * 20 bps on initial entry; 20 bps round-trip = 0.0020
ROUND_TRIP_COST_20BPS: Final = 0.0020


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


def _pct(val: float | None) -> str:
    return "n/a" if val is None or not math.isfinite(val) else f"{100.0 * val:+.2f}%"


def _num(val: float | None, digits: int = 4) -> str:
    return "n/a" if val is None or not math.isfinite(val) else f"{val:+.{digits}f}"


def _fetch_post_holdout_bars(
    tickers: tuple[str, ...],
    start_day: date,
    end_day: date,
    settings: Settings,
    *,
    unseal_final_test_period: bool = False,
) -> pd.DataFrame:
    """Fetch `1Day` adjusted SIP `open`/`close` bars strictly inside `[start_day, end_day]` (`> 2026-07-08`)."""
    if (start_day <= HISTORICAL_SEALED_END or end_day <= HISTORICAL_SEALED_END) and not unseal_final_test_period:
        raise SealedPeriodError(
            f"forward evaluation refuses bar window touching <= {HISTORICAL_SEALED_END.isoformat()}: "
            f"[{start_day.isoformat()}, {end_day.isoformat()}]"
        )
    if end_day < start_day:
        raise ValueError(f"end_day {end_day} < start_day {start_day}")

    alpaca = AlpacaSource(settings)
    start_utc = datetime.combine(start_day, time(0, 0), tzinfo=UTC)
    end_utc = datetime.combine(end_day, time(23, 59, 59), tzinfo=UTC)
    query_syms = ["SPY", *sorted(set(tickers) - {"SPY"})]
    rows: list[dict[str, Any]] = []

    def _fetch_chunk(chunk: tuple[str, ...]) -> None:
        if not chunk:
            return
        token: str | None = None
        try:
            while True:
                page = alpaca.fetch_bars_page(
                    chunk, start_utc, end_utc, timeframe="1Day", page_token=token, asof=end_day, adjustment="all",
                )
                for sym, blist in page.bars.items():
                    for b in blist:
                        b_day = pd.Timestamp(str(b["t"])).tz_convert(MARKET_TIMEZONE).date()
                        if b_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
                            raise SealedPeriodError(f"bar {b_day.isoformat()} leaked from sealed window")
                        if b_day > end_day:
                            raise SealedPeriodError(f"bar {b_day.isoformat()} leaked after as_of_date {end_day.isoformat()}")
                        if start_day <= b_day <= end_day:
                            rows.append({
                                "ticker": str(sym).upper(),
                                "session": b_day.isoformat(),
                                "open": float(b["o"]),
                                "close": float(b["c"]),
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

    for off in range(0, len(query_syms), 45):
        _fetch_chunk(tuple(query_syms[off : off + 45]))
        guard("forward outcome bar fetch")

    if not rows:
        return pd.DataFrame(columns=["ticker", "session", "open", "close"])
    return pd.DataFrame(rows).drop_duplicates(["ticker", "session"], keep="last").reset_index(drop=True)


def _score_single_signal_cohort(
    sig_vals: np.ndarray,
    ret_vals: np.ndarray,
    spy_ret: float,
) -> dict[str, Any] | None:
    """Compute Spearman IC, top-10% return, net-20bps spread vs equal-weight & vs SPY on one snapshot-horizon."""
    mask = np.isfinite(sig_vals) & np.isfinite(ret_vals)
    n_valid = int(mask.sum())
    if n_valid < 10:
        return None
    s = sig_vals[mask]
    r = ret_vals[mask]
    if float(np.std(s)) <= 1e-12 or float(np.std(r)) <= 1e-12:
        ic_val = 0.0
    else:
        rk_s = pd.Series(s).rank(method="average").to_numpy(dtype=np.float64)
        rk_r = pd.Series(r).rank(method="average").to_numpy(dtype=np.float64)
        corr = float(np.corrcoef(rk_s, rk_r)[0, 1])
        ic_val = corr if np.isfinite(corr) else 0.0

    k_top = max(1, int(round(0.10 * n_valid)))
    top_idx = np.argsort(-s, kind="stable")[:k_top]
    top_gross = float(np.mean(r[top_idx]))
    top_net_20bps = top_gross - ROUND_TRIP_COST_20BPS
    ew_gross = float(np.mean(r))
    return {
        "stocks_scored_with_outcome": n_valid,
        "top_decile_count": k_top,
        "rank_ic": ic_val,
        "equal_weight_return": ew_gross,
        "spy_return": spy_ret,
        "top10_gross_return": top_gross,
        "top10_net_20bps_return": top_net_20bps,
        "top10_net_minus_ew": top_net_20bps - ew_gross,
        "top10_net_minus_spy": top_net_20bps - spy_ret,
    }


def _render_forward_eval_markdown(doc: dict[str, Any]) -> str:
    lines = [
        "# Prospective Forward Research Score & OPRA Option Maturation Report",
        "",
        f"- **Evaluation mode:** `{doc['evaluation_mode']}`",
        f"- **Evaluated through completed session:** `{doc['evaluated_as_of_date']}`",
        f"- **Sealed holdout (`2024-05-29..2026-07-08`) untouched:** `{doc['sealed_holdout_untouched']}`",
        f"- **Snapshots discovered:** `{doc['snapshots_total']}` (`{', '.join(doc['snapshot_dates'])}`)",
        f"- **Snapshots recorded after intended entry (`rejected`):** `{len(doc['snapshots_recorded_after_entry_dates'])}` "
        f"(`{', '.join(doc['snapshots_recorded_after_entry_dates']) or 'none'}`)",
        f"- **Snapshots qualifying for persisted SEC model:** `{len(doc['snapshots_qualifying_for_persisted_sec_model_dates'])}` "
        f"(`{', '.join(doc['snapshots_qualifying_for_persisted_sec_model_dates']) or 'none'}`)",
        f"- **Label convention:** `{doc['label_notice']}` (`20 bps` round-trip cost applied to active top-10% sleeves).",
        "",
        "## Section 0: Snapshot Prediction-Clock & Candidate Admission Audit",
        "",
        "| Snapshot `t` | Mode | Source Cutoff UTC | Published UTC | Next Open UTC | Clock Status | SEC Model Qualified |",
        "|---|---|---|---|---|---|---|",
    ]
    for aud in doc["snapshot_admission_audit"]:
        lines.append(
            f"| {aud['snapshot_date']} | {aud['record_mode']} | {aud['source_cutoff_utc']} | "
            f"{aud['published_at_utc'] or '-'} | {aud['next_tradable_open_utc']} | "
            f"{aud['clock_status']} | {aud['qualifies_for_persisted_sec_model']} |"
        )

    lines += [
        "",
        "## Section A: Snapshot & Horizon Maturation Matrix",
        "",
        "| Snapshot `t` | Horizon `h` | Status | Entry `t+1` | Exit `t+h` | Remaining Sessions | Known Outcomes / Scored | `SPY` Return |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in doc["maturation_schedule"]:
        known_str = (
            f"{row['known_outcomes']} / {row['eligible_scored']}"
            if row["status"] == "matured"
            else f"- / {row['eligible_scored']}"
        )
        lines.append(
            f"| {row['snapshot_date']} | {row['horizon_sessions']}d | {row['status']} | "
            f"{row['entry_session'] or '-'} | {row['exit_session'] or '-'} | "
            f"{row['sessions_remaining']} | {known_str} | {_pct(row['spy_return'])} |"
        )

    lines += [
        "",
        "## Section B: Pooled Matured Signal Performance by Horizon",
        "",
        "| Signal | H | Matured Dates | Mean IC | Mean Top-10% Net 20bps | Mean `SPY` | Net vs `SPY` | Net vs EW |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if not doc["pooled_signals_by_horizon"]:
        lines.append("| *(no matured horizons yet)* | - | 0 | n/a | n/a | n/a | n/a | n/a |")
    else:
        for h_key, sig_map in doc["pooled_signals_by_horizon"].items():
            for sig_name, agg in sig_map.items():
                lines.append(
                    f"| {sig_name} | {h_key}d | {agg['matured_dates']} | {_num(agg['mean_rank_ic'])} | "
                    f"{_pct(agg['mean_top10_net_20bps'])} | {_pct(agg['mean_spy_return'])} | "
                    f"{_pct(agg['mean_top10_net_minus_spy'])} | {_pct(agg['mean_top10_net_minus_ew'])} |"
                )

    lines += [
        "",
        "## Section C: Frozen Core-Satellite Prospective Candidates (Matured Windows)",
        "",
        "| Candidate | H | Matured Dates | Mean Portfolio Net 20bps | Mean `SPY` | Mean Net Excess vs `SPY` | Backwardation Dates |",
        "|---|---|---|---|---|---|---|",
    ]
    if not doc["pooled_candidates_by_horizon"]:
        lines.append("| *(no matured horizons yet)* | - | 0 | n/a | n/a | n/a | 0 |")
    else:
        for h_key, cand_map in doc["pooled_candidates_by_horizon"].items():
            for cand_id, agg in cand_map.items():
                lines.append(
                    f"| {cand_id} | {h_key}d | {agg['matured_dates']} | {_pct(agg['mean_portfolio_net_20bps'])} | "
                    f"{_pct(agg['mean_spy_return'])} | {_pct(agg['mean_net_excess_vs_spy'])} | "
                    f"{agg['backwardation_dates']} |"
                )

    lines += [
        "",
        "## Section D: Live `OPRA` Option-Overlay Cohort Diagnostics (Matured Windows)",
        "",
        "| Snapshot `t` | H | `OPRA` Backwardation | Vetoed Falling-Knife (`Mean vs SPY`) | "
        "Shock > Straddle (`Mean vs SPY`) |",
        "|---|---|---|---|---|",
    ]
    if not doc["opra_diagnostics"]:
        lines.append("| *(no matured OPRA snapshots yet)* | - | - | - | - |")
    else:
        for item in doc["opra_diagnostics"]:
            veto_txt = (
                f"{item['vetoed_count']} ({_pct(item['vetoed_mean_excess_vs_spy'])})"
                if item["vetoed_count"] > 0
                else "0 (n/a)"
            )
            shock_txt = (
                f"{item['shock_gt_straddle_count']} ({_pct(item['shock_gt_straddle_mean_excess_vs_spy'])})"
                if item["shock_gt_straddle_count"] > 0
                else "0 (n/a)"
            )
            lines.append(
                f"| {item['snapshot_date']} | {item['horizon_sessions']}d | {item['opra_backwardation']} | "
                f"{veto_txt} | {shock_txt} |"
            )
    lines.append("")
    return "\n".join(lines)


def evaluate_forward_scores(
    *,
    root: Path,
    as_of_date: date | None = None,
    data_root: Path | None = None,
    forward_dir: Path = FORWARD_RECORD_DIRECTORY,
    output: Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_EVAL_HORIZONS,
    allow_retrospective_replay: bool = False,
    unseal_final_test_period: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Evaluate all recorded prospective forward snapshots up to `as_of_date` without reading future bars."""
    if as_of_date is not None and as_of_date <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(
            f"forward evaluation refuses as_of_date <= {HISTORICAL_SEALED_END.isoformat()}: {as_of_date.isoformat()}"
        )

    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(EVALUATE_FORWARD_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("forward evaluator start")
        cfg = settings or get_settings()
        eval_day = (
            resolve_latest_completed_session(cfg, unseal_final_test_period=unseal_final_test_period)
            if as_of_date is None
            else as_of_date
        )
        if eval_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
            raise SealedPeriodError(f"resolved eval_day {eval_day.isoformat()} <= {HISTORICAL_SEALED_END.isoformat()}")

        records_abs = paths.directory(forward_dir)
        snapshots: list[tuple[date, pd.DataFrame, dict[str, Any], dict[str, Any]]] = []
        if records_abs.is_dir():
            for child in sorted(records_abs.iterdir()):
                if not child.is_dir() or child.name.startswith("."):
                    continue
                scores_pq = child / "scores.parquet"
                manifest_js = child / "_manifest.json"
                if not (scores_pq.is_file() and manifest_js.is_file()):
                    continue
                snap_day = date.fromisoformat(child.name)
                if snap_day <= HISTORICAL_SEALED_END and not unseal_final_test_period:
                    raise SealedPeriodError(
                        f"forward snapshot {snap_day.isoformat()} is inside sealed window <= {HISTORICAL_SEALED_END.isoformat()}"
                    )
                if snap_day > eval_day:
                    continue
                s_df = cast(Any, pq).read_table(scores_pq, use_threads=False).to_pandas()
                m_dict = cast(dict[str, Any], json.loads(manifest_js.read_text(encoding="utf-8")))
                audit = audit_snapshot_admission(
                    child,
                    s_df,
                    m_dict,
                    allow_retrospective_replay=allow_retrospective_replay,
                )
                snapshots.append((snap_day, s_df, m_dict, audit))

        if not snapshots:
            raise ValueError(f"no prospective forward snapshots found <= {eval_day.isoformat()} in {records_abs}")

        clock_admitted_snapshots = [
            (d, s_df, m_dict, aud)
            for d, s_df, m_dict, aud in snapshots
            if (
                (allow_retrospective_replay and aud["record_mode"] == "retrospective_replay")
                or (aud["record_mode"] != "retrospective_replay" and bool(aud["published_before_entry"]))
            )
        ]

        all_tickers: set[str] = set()
        for _, s_df, _, _ in clock_admitted_snapshots:
            for sid in s_df["security_id"].astype(str):
                all_tickers.add(sid.split(":", 1)[1] if ":" in sid else sid)

        earliest_snap = min(d for d, _, _, _ in snapshots)
        bars_df = (
            _fetch_post_holdout_bars(
                tuple(sorted(all_tickers)),
                earliest_snap,
                eval_day,
                cfg,
                unseal_final_test_period=unseal_final_test_period,
            )
            if clock_admitted_snapshots
            else pd.DataFrame(columns=["ticker", "session", "open", "close"])
        )
        spy_bars = bars_df.loc[bars_df["ticker"] == "SPY"].sort_values("session").reset_index(drop=True)
        sessions = spy_bars["session"].astype(str).tolist()
        sess_to_idx = {s: idx for idx, s in enumerate(sessions)}
        spy_open = spy_bars["open"].to_numpy(dtype=np.float64)
        spy_close = spy_bars["close"].to_numpy(dtype=np.float64)

        open_piv = bars_df.pivot(index="session", columns="ticker", values="open").reindex(sessions) if len(bars_df) else pd.DataFrame()
        close_piv = bars_df.pivot(index="session", columns="ticker", values="close").reindex(sessions) if len(bars_df) else pd.DataFrame()

        maturation_schedule: list[dict[str, Any]] = []
        per_cell_signals: dict[int, dict[str, list[dict[str, Any]]]] = {}
        per_cell_candidates: dict[int, dict[str, list[dict[str, Any]]]] = {}
        opra_diagnostics: list[dict[str, Any]] = []

        for snap_day, s_df, m_dict, aud in snapshots:
            s_iso = snap_day.isoformat()
            eligible_cnt = len(s_df)
            clock_ok = (
                (allow_retrospective_replay and aud["record_mode"] == "retrospective_replay")
                or (aud["record_mode"] != "retrospective_replay" and bool(aud["published_before_entry"]))
            )
            if not clock_ok:
                rej_status = (
                    "rejected_retrospective_replay"
                    if aud["record_mode"] == "retrospective_replay"
                    else f"rejected_{aud['clock_status']}"
                )
                for h in horizons:
                    maturation_schedule.append({
                        "snapshot_date": s_iso,
                        "horizon_sessions": h,
                        "status": rej_status,
                        "entry_session": aud["next_tradable_session"],
                        "exit_session": None,
                        "sessions_remaining": h,
                        "eligible_scored": eligible_cnt,
                        "known_outcomes": 0,
                        "spy_return": None,
                    })
                continue
            if s_iso not in sess_to_idx:
                for h in horizons:
                    maturation_schedule.append({
                        "snapshot_date": s_iso,
                        "horizon_sessions": h,
                        "status": "snapshot_not_in_spy_calendar",
                        "entry_session": None,
                        "exit_session": None,
                        "sessions_remaining": h,
                        "eligible_scored": eligible_cnt,
                        "known_outcomes": 0,
                        "spy_return": None,
                    })
                continue

            t_idx = sess_to_idx[s_iso]
            entry_idx = t_idx + 1
            tks = [sid.split(":", 1)[1] if ":" in sid else sid for sid in s_df["security_id"].astype(str)]
            opt_sum = m_dict.get("options_summary") or {}
            has_opra = bool(opt_sum.get("opra_enabled"))
            is_backw = bool(opt_sum.get("opra_backwardation", False))

            for h in horizons:
                exit_idx = t_idx + h
                if entry_idx >= len(sessions):
                    maturation_schedule.append({
                        "snapshot_date": s_iso,
                        "horizon_sessions": h,
                        "status": "pending_entry_bar",
                        "entry_session": None,
                        "exit_session": None,
                        "sessions_remaining": exit_idx - (len(sessions) - 1),
                        "eligible_scored": eligible_cnt,
                        "known_outcomes": 0,
                        "spy_return": None,
                    })
                    continue
                if exit_idx >= len(sessions):
                    maturation_schedule.append({
                        "snapshot_date": s_iso,
                        "horizon_sessions": h,
                        "status": "pending_maturation",
                        "entry_session": sessions[entry_idx],
                        "exit_session": None,
                        "sessions_remaining": exit_idx - (len(sessions) - 1),
                        "eligible_scored": eligible_cnt,
                        "known_outcomes": 0,
                        "spy_return": None,
                    })
                    continue

                spy_r = float(spy_close[exit_idx] / spy_open[entry_idx] - 1.0)
                o_row = open_piv.iloc[entry_idx]
                c_row = close_piv.iloc[exit_idx]
                rets = np.empty(len(tks), dtype=np.float64)
                for i_tk, tk in enumerate(tks):
                    o_v = float(o_row[tk]) if tk in o_row and pd.notna(o_row[tk]) else np.nan
                    c_v = float(c_row[tk]) if tk in c_row and pd.notna(c_row[tk]) else np.nan
                    rets[i_tk] = (c_v / o_v - 1.0) if (np.isfinite(o_v) and o_v > 0.0 and np.isfinite(c_v)) else np.nan

                known_n = int(np.isfinite(rets).sum())
                maturation_schedule.append({
                    "snapshot_date": s_iso,
                    "horizon_sessions": h,
                    "status": "matured",
                    "entry_session": sessions[entry_idx],
                    "exit_session": sessions[exit_idx],
                    "sessions_remaining": 0,
                    "eligible_scored": eligible_cnt,
                    "known_outcomes": known_n,
                    "spy_return": spy_r,
                })

                sig_results_here: dict[str, dict[str, Any]] = {}
                for col in SIGNAL_COLUMNS:
                    if col not in s_df.columns:
                        continue
                    arr = pd.to_numeric(s_df[col], errors="coerce").to_numpy(dtype=np.float64)
                    res_c = _score_single_signal_cohort(arr, rets, spy_r)
                    if res_c is not None:
                        sig_results_here[col] = res_c
                        per_cell_signals.setdefault(h, {}).setdefault(col, []).append({"snapshot_date": s_iso, **res_c})

                # Frozen Prospective Candidate 1: `90% SPY + 10% R1` (`shifts 10% sleeve into SPY when backwardation`)
                bn_res = sig_results_here.get("score_beta_neutral_price_12")
                if bn_res is not None:
                    r1_port = spy_r if is_backw else (0.90 * spy_r + 0.10 * float(bn_res["top10_net_20bps_return"]))
                    per_cell_candidates.setdefault(h, {}).setdefault(
                        "90_10__spy_core__R1_backwardation_to_spy", []
                    ).append({
                        "snapshot_date": s_iso,
                        "portfolio_net_20bps": r1_port,
                        "spy_return": spy_r,
                        "net_excess_vs_spy": r1_port - spy_r,
                        "opra_backwardation": is_backw,
                    })

                # Frozen Prospective Candidate 2: `80% SPY + 20% R2 OPRA-conditioned` (fallback `score_price_equal_12` if no OPRA)
                r2_res = sig_results_here.get("score_r2_opra_conditioned") or sig_results_here.get("score_price_equal_12")
                if r2_res is not None:
                    r2_port = 0.80 * spy_r + 0.20 * float(r2_res["top10_net_20bps_return"])
                    per_cell_candidates.setdefault(h, {}).setdefault(
                        "80_20__spy_core__R2_opra_adaptive_hedge", []
                    ).append({
                        "snapshot_date": s_iso,
                        "portfolio_net_20bps": r2_port,
                        "spy_return": spy_r,
                        "net_excess_vs_spy": r2_port - spy_r,
                        "opra_backwardation": is_backw,
                        "used_live_opra": "score_r2_opra_conditioned" in sig_results_here,
                    })

                if has_opra:
                    veto_mask = (
                        s_df["falling_knife_put_skew_veto"].fillna(False).to_numpy(dtype=bool)
                        if "falling_knife_put_skew_veto" in s_df.columns
                        else np.zeros(len(s_df), dtype=bool)
                    ) & np.isfinite(rets)
                    shock_vals = (
                        pd.to_numeric(s_df["catalyst_shock_to_straddle"], errors="coerce").to_numpy(dtype=np.float64)
                        if "catalyst_shock_to_straddle" in s_df.columns
                        else np.full(len(s_df), np.nan)
                    )
                    shock_mask = np.isfinite(shock_vals) & (shock_vals > 1.0) & np.isfinite(rets)
                    opra_diagnostics.append({
                        "snapshot_date": s_iso,
                        "horizon_sessions": h,
                        "opra_backwardation": is_backw,
                        "vetoed_count": int(veto_mask.sum()),
                        "vetoed_mean_excess_vs_spy": (
                            float(np.mean(rets[veto_mask] - spy_r)) if int(veto_mask.sum()) > 0 else None
                        ),
                        "shock_gt_straddle_count": int(shock_mask.sum()),
                        "shock_gt_straddle_mean_excess_vs_spy": (
                            float(np.mean(rets[shock_mask] - spy_r)) if int(shock_mask.sum()) > 0 else None
                        ),
                    })

        pooled_signals: dict[str, dict[str, Any]] = {}
        for h in sorted(per_cell_signals):
            pooled_signals[str(h)] = {}
            for sig_name, items in per_cell_signals[h].items():
                pooled_signals[str(h)][sig_name] = {
                    "matured_dates": len(items),
                    "mean_rank_ic": float(np.mean([x["rank_ic"] for x in items])),
                    "mean_top10_net_20bps": float(np.mean([x["top10_net_20bps_return"] for x in items])),
                    "mean_spy_return": float(np.mean([x["spy_return"] for x in items])),
                    "mean_top10_net_minus_spy": float(np.mean([x["top10_net_minus_spy"] for x in items])),
                    "mean_top10_net_minus_ew": float(np.mean([x["top10_net_minus_ew"] for x in items])),
                    "by_snapshot": items,
                }

        pooled_candidates: dict[str, dict[str, Any]] = {}
        for h in sorted(per_cell_candidates):
            pooled_candidates[str(h)] = {}
            for cand_id, items in per_cell_candidates[h].items():
                pooled_candidates[str(h)][cand_id] = {
                    "matured_dates": len(items),
                    "mean_portfolio_net_20bps": float(np.mean([x["portfolio_net_20bps"] for x in items])),
                    "mean_spy_return": float(np.mean([x["spy_return"] for x in items])),
                    "mean_net_excess_vs_spy": float(np.mean([x["net_excess_vs_spy"] for x in items])),
                    "backwardation_dates": sum(1 for x in items if x["opra_backwardation"]),
                    "by_snapshot": items,
                }

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_rel = output or (FORWARD_EVAL_DIRECTORY / f"forward_eval_{stamp}")
        dest = paths.write(paths.identity(out_rel))
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.parent / f".{dest.name}.{uuid4().hex}.partial"
        staging.mkdir(parents=True)

        admission_audits = [aud for _, _, _, aud in snapshots]
        after_entry_dates = [a["snapshot_date"] for a in admission_audits if a["clock_status"] == "recorded_after_entry"]
        before_entry_dates = [a["snapshot_date"] for a in admission_audits if bool(a["published_before_entry"])]
        sec_qualified_dates = [a["snapshot_date"] for a in admission_audits if bool(a["qualifies_for_persisted_sec_model"])]

        doc = _json_safe({
            "evaluation_mode": "retrospective_replay" if allow_retrospective_replay else "prospective",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_as_of_date": eval_day.isoformat(),
            "sealed_holdout_window": "2024-05-29..2026-07-08",
            "sealed_holdout_untouched": not unseal_final_test_period,
            "frozen_prospective_candidates": list(FROZEN_PROSPECTIVE_CANDIDATES),
            "snapshots_total": len(snapshots),
            "snapshot_dates": [d.isoformat() for d, _, _, _ in snapshots],
            "snapshots_recorded_after_entry_dates": after_entry_dates,
            "snapshots_published_before_entry_dates": before_entry_dates,
            "snapshots_qualifying_for_persisted_sec_model_dates": sec_qualified_dates,
            "snapshot_admission_audit": admission_audits,
            "matured_snapshot_horizon_pairs": sum(1 for r in maturation_schedule if r["status"] == "matured"),
            "pending_snapshot_horizon_pairs": sum(
                1 for r in maturation_schedule if r["status"] in ("pending_entry_bar", "pending_maturation")
            ),
            "rejected_snapshot_horizon_pairs": sum(
                1 for r in maturation_schedule if str(r["status"]).startswith("rejected_")
            ),
            "maturation_schedule": maturation_schedule,
            "pooled_signals_by_horizon": pooled_signals,
            "pooled_candidates_by_horizon": pooled_candidates,
            "opra_diagnostics": opra_diagnostics,
            "code": git_identity(root),
            "label_notice": "research label: close[t+h] / open[t+1] - 1 on provider split- and dividend-adjusted SIP daily bars",
        })
        try:
            (staging / "report.json").write_text(json.dumps(doc, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
            (staging / "report.md").write_text(_render_forward_eval_markdown(doc), encoding="utf-8")
            os.replace(staging, dest)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    return {
        "output": paths.identity(out_rel).relative_to(root).as_posix(),
        "evaluation_mode": doc["evaluation_mode"],
        "evaluated_as_of_date": eval_day.isoformat(),
        "snapshots_total": len(snapshots),
        "snapshots_recorded_after_entry_dates": after_entry_dates,
        "snapshots_qualifying_for_persisted_sec_model_dates": sec_qualified_dates,
        "matured_pairs": doc["matured_snapshot_horizon_pairs"],
        "pending_pairs": doc["pending_snapshot_horizon_pairs"],
        "rejected_pairs": doc["rejected_snapshot_horizon_pairs"],
    }


__all__ = [
    "DEFAULT_EVAL_HORIZONS",
    "EVALUATE_FORWARD_COMMAND",
    "FORWARD_EVAL_DIRECTORY",
    "evaluate_forward_scores",
]
