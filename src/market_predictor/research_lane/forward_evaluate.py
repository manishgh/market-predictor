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
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.config import Settings, get_settings
from market_predictor.core.errors import ArtifactIntegrityError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research_lane import benchmark_tilt, live_cohort
from market_predictor.research_lane.forward_record import (
    FORWARD_RECORD_DIRECTORY,
    FROZEN_PROSPECTIVE_CANDIDATES,
    HISTORICAL_SEALED_END,
    resolve_latest_completed_session,
)
from market_predictor.research_lane.panel import MARKET_TIMEZONE, SealedPeriodError
from market_predictor.research_lane.prospective_contracts import (
    ProspectiveCandidateSpec,
    audit_snapshot_admission,
    extract_snapshot_opra_regime,
    resolve_xnys_session_clock,
)
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
    *,
    security_ids: tuple[str, ...] | None = None,
) -> dict[str, Any] | None:
    """Select top-10% cohort strictly from decision-time `sig_vals` before inspecting outcomes (`R5`).

    If any selected top-10% stock has an unresolved (`NaN`) future return, the cohort return is reported
    as incomplete (`top10_complete=False`, `top10_gross_return=None`, `top10_net_20bps_return=None`)
    without ever substituting a lower-ranked stock. Observed-subset Spearman rank correlation is kept
    explicitly labelled under `observed_subset_rank_ic`.
    """
    decision_valid = np.flatnonzero(np.isfinite(sig_vals))
    if decision_valid.size < 10:
        return None

    k_top = max(1, int(decision_valid.size * 0.10))
    desc_decision = decision_valid[np.argsort(-sig_vals[decision_valid], kind="stable")]
    selected_cols = desc_decision[:k_top]
    selected_ids = (
        [security_ids[int(c)] for c in selected_cols]
        if security_ids is not None
        else [str(int(c)) for c in selected_cols]
    )

    sel_rets = ret_vals[selected_cols]
    unresolved_sel = int((~np.isfinite(sel_rets)).sum())
    top10_complete = unresolved_sel == 0

    all_elig_rets = ret_vals[decision_valid]
    ew_complete = bool(np.isfinite(all_elig_rets).all())
    ew_gross = float(np.mean(all_elig_rets)) if ew_complete else None

    if top10_complete:
        gross_f = float(np.mean(sel_rets))
        net_f = float(gross_f - ROUND_TRIP_COST_20BPS)
        top_gross: float | None = gross_f
        top_net_20bps: float | None = net_f
        top_minus_spy: float | None = float(net_f - spy_ret)
        top_minus_ew: float | None = (
            float(net_f - ew_gross) if ew_gross is not None else None
        )
    else:
        top_gross = None
        top_net_20bps = None
        top_minus_spy = None
        top_minus_ew = None

    obs_mask = np.isfinite(sig_vals) & np.isfinite(ret_vals)
    n_obs = int(obs_mask.sum())
    obs_ic: float | None
    if n_obs < 10:
        obs_ic = None
    else:
        s_obs = sig_vals[obs_mask]
        r_obs = ret_vals[obs_mask]
        if float(np.std(s_obs)) <= 1e-12 or float(np.std(r_obs)) <= 1e-12:
            obs_ic = 0.0
        else:
            rk_s = pd.Series(s_obs).rank(method="average").to_numpy(dtype=np.float64)
            rk_r = pd.Series(r_obs).rank(method="average").to_numpy(dtype=np.float64)
            corr = float(np.corrcoef(rk_s, rk_r)[0, 1])
            obs_ic = corr if np.isfinite(corr) else 0.0

    return {
        "decision_scored_stocks": int(decision_valid.size),
        "stocks_scored_with_outcome": n_obs,
        "top_decile_count": k_top,
        "selected_security_ids": selected_ids,
        "selected_unresolved_outcomes": unresolved_sel,
        "top10_complete": top10_complete,
        "equal_weight_complete": ew_complete,
        "rank_ic_scope": "observed_subset_diagnostic",
        "observed_subset_rank_ic": obs_ic,
        "rank_ic": obs_ic,
        "equal_weight_return": ew_gross,
        "spy_return": spy_ret,
        "top10_gross_return": top_gross,
        "top10_net_20bps_return": top_net_20bps,
        "top10_net_minus_ew": top_minus_ew,
        "top10_net_minus_spy": top_minus_spy,
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
        "## Section C: Persisted Frozen Candidate Portfolios (`open[t+1] -> open[t+h+1]`, Post-Drift Turnover)",
        "",
        "| Candidate | H | Complete | Matured Rebalances | Compounded Net | Compounded `SPY` | Mean Turnover | Mean Net vs `SPY` |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if not doc["pooled_candidates_by_horizon"]:
        lines.append("| *(no matured candidate rebalances yet)* | - | - | 0 | n/a | n/a | n/a | n/a |")
    else:
        for h_key, cand_map in doc["pooled_candidates_by_horizon"].items():
            for cand_id, agg in cand_map.items():
                lines.append(
                    f"| {cand_id} | {h_key}d | {agg['portfolio_complete']} | {agg['matured_dates']} | "
                    f"{_pct(agg['compounded_net_20bps'])} | {_pct(agg['compounded_spy_return'])} | "
                    f"{_pct(agg['mean_one_way_turnover'])} | {_pct(agg['mean_net_excess_vs_spy'])} |"
                )

    lines += [
        "",
        "### Section C.1: Candidate Rebalance Maturation & Abstention Schedule",
        "",
        "| Snapshot `t` | Candidate | H | Status | Entry Open `t+1` | Exit Open `t+h+1` | Remaining | Detail |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if not doc["candidate_rebalance_schedule"]:
        lines.append("| *(no persisted candidate rebalances)* | - | - | - | - | - | - | - |")
    else:
        for r_item in doc["candidate_rebalance_schedule"]:
            lines.append(
                f"| {r_item['snapshot_date']} | {r_item['candidate_id']} | {r_item['horizon_sessions']}d | "
                f"{r_item['status']} | {r_item['entry_session'] or '-'} | {r_item['exit_session'] or '-'} | "
                f"{r_item['sessions_remaining_to_exit_open']} | {r_item['abstain_reason'] or '-'} |"
            )

    if doc["candidate_abstentions_and_rejections"]:
        lines += [
            "",
            "### Section C.2: Incomplete Candidate Definitions & Recorded Abstentions",
            "",
            "| Snapshot `t` | Candidate | H | Status | Reason |",
            "|---|---|---|---|---|",
        ]
        for rej in doc["candidate_abstentions_and_rejections"]:
            lines.append(
                f"| {rej['snapshot_date']} | {rej['candidate_id'] or '-'} | "
                f"{rej['horizon_sessions'] if rej['horizon_sessions'] is not None else '-'} | "
                f"{rej['status']} | {rej['reason']} |"
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


@dataclass(frozen=True, slots=True)
class _OutcomeBarLookup:
    """Session-indexed `open`/`close` bar lookup by canonical `security_id` (falling back to `ticker`)."""

    sessions: list[str]
    sess_to_idx: dict[str, int]
    spy_open: np.ndarray
    spy_close: np.ndarray
    open_by_sid: dict[str, np.ndarray]
    close_by_sid: dict[str, np.ndarray]
    open_by_tk: dict[str, np.ndarray]
    close_by_tk: dict[str, np.ndarray]

    def open_close_at(self, security_id: str, ticker: str, row_idx: int) -> tuple[float, float]:
        if security_id == "benchmark:SPY" or ticker == "SPY":
            return float(self.spy_open[row_idx]), float(self.spy_close[row_idx])
        if security_id in self.open_by_sid:
            return float(self.open_by_sid[security_id][row_idx]), float(self.close_by_sid[security_id][row_idx])
        tk_up = ticker.upper().strip()
        if tk_up and tk_up in self.open_by_tk:
            return float(self.open_by_tk[tk_up][row_idx]), float(self.close_by_tk[tk_up][row_idx])
        return np.nan, np.nan


def _resolve_outcome_bars(
    *,
    paths: EvidencePaths,
    all_tickers: tuple[str, ...],
    earliest_snap: date,
    eval_day: date,
    cfg: Settings,
    allow_retrospective_replay: bool,
    unseal_final_test_period: bool,
) -> _OutcomeBarLookup:
    wider_dir_exists = (paths.data_root / "research/research_lane/wider_us_universe").is_dir() or (
        paths.project_root / "data/research/research_lane/wider_us_universe"
    ).is_dir()
    if allow_retrospective_replay and eval_day <= HISTORICAL_SEALED_END and wider_dir_exists:
        eval_clk = resolve_xnys_session_clock(eval_day)
        coh = live_cohort.load_decision_date_mid_small_cohort(
            paths,
            as_of_date=eval_day,
            source_cutoff_utc=eval_clk.source_cutoff_utc,
            live_prospective=False,
            unseal_final_test_period=unseal_final_test_period,
            settings=cfg,
        )
        p = coh.panel
        s_row = int(np.searchsorted(p.sessions, np.datetime64(earliest_snap.isoformat(), "D"), side="left"))
        sess_list = [str(s) for s in p.sessions[s_row:]]
        open_sid = {sid: p.open[s_row:, idx].astype(np.float64) for idx, sid in enumerate(p.security_ids)}
        close_sid = {sid: p.close[s_row:, idx].astype(np.float64) for idx, sid in enumerate(p.security_ids)}
        return _OutcomeBarLookup(
            sessions=sess_list,
            sess_to_idx={s: i for i, s in enumerate(sess_list)},
            spy_open=p.spy_open[s_row:].astype(np.float64),
            spy_close=p.spy_close[s_row:].astype(np.float64),
            open_by_sid=open_sid,
            close_by_sid=close_sid,
            open_by_tk={},
            close_by_tk={},
        )

    bars_df = _fetch_post_holdout_bars(
        all_tickers,
        earliest_snap,
        eval_day,
        cfg,
        unseal_final_test_period=unseal_final_test_period,
    )
    spy_bars = bars_df.loc[bars_df["ticker"] == "SPY"].sort_values("session").reset_index(drop=True)
    sessions = spy_bars["session"].astype(str).tolist()
    spy_open = spy_bars["open"].to_numpy(dtype=np.float64)
    spy_close = spy_bars["close"].to_numpy(dtype=np.float64)
    open_piv = bars_df.pivot(index="session", columns="ticker", values="open").reindex(sessions) if len(bars_df) else pd.DataFrame()
    close_piv = bars_df.pivot(index="session", columns="ticker", values="close").reindex(sessions) if len(bars_df) else pd.DataFrame()
    open_tk = {str(col).upper(): open_piv[col].to_numpy(dtype=np.float64) for col in open_piv.columns}
    close_tk = {str(col).upper(): close_piv[col].to_numpy(dtype=np.float64) for col in close_piv.columns}
    return _OutcomeBarLookup(
        sessions=sessions,
        sess_to_idx={s: idx for idx, s in enumerate(sessions)},
        spy_open=spy_open,
        spy_close=spy_close,
        open_by_sid={},
        close_by_sid={},
        open_by_tk=open_tk,
        close_by_tk=close_tk,
    )


def _evaluate_persisted_candidates(
    admitted_snapshots: list[tuple[date, pd.DataFrame, pd.DataFrame | None, dict[str, Any], dict[str, Any]]],
    bars: _OutcomeBarLookup,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Grade persisted `ProspectiveCandidateSpec`s and `holdings.parquet` via shared `benchmark_tilt` accounting (`R6`)."""
    pooled_candidates: dict[str, dict[str, Any]] = {}
    rebalance_schedule: list[dict[str, Any]] = []
    abstentions_and_rejections: list[dict[str, Any]] = []

    records_by_cand: dict[
        str,
        list[tuple[date, pd.DataFrame, pd.DataFrame | None, dict[str, Any], dict[str, Any], ProspectiveCandidateSpec]],
    ] = {}
    for snap_day, s_df, h_df, m_dict, aud in admitted_snapshots:
        s_iso = snap_day.isoformat()
        raw_cands = m_dict.get("frozen_prospective_candidates")
        if not isinstance(raw_cands, list) or not raw_cands:
            abstentions_and_rejections.append({
                "snapshot_date": s_iso,
                "candidate_id": None,
                "horizon_sessions": None,
                "status": "no_frozen_candidates_declared",
                "reason": "snapshot manifest contains no frozen_prospective_candidates",
            })
            continue
        for raw_c in raw_cands:
            if not isinstance(raw_c, Mapping):
                continue
            cid_guess = str(raw_c.get("candidate_id") or "<unnamed>")
            h_guess = raw_c.get("horizon_sessions")
            try:
                spec = ProspectiveCandidateSpec.from_mapping(raw_c)
            except ValueError as exc:
                abstentions_and_rejections.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cid_guess,
                    "horizon_sessions": h_guess,
                    "status": "incomplete_candidate_definition",
                    "reason": str(exc),
                })
                continue
            records_by_cand.setdefault(spec.candidate_id, []).append((snap_day, s_df, h_df, m_dict, aud, spec))

    sessions = bars.sessions
    sess_to_idx = bars.sess_to_idx

    for cand_id, cand_records in sorted(records_by_cand.items()):
        spec0 = cand_records[0][5]
        h = spec0.horizon_sessions
        anchor_r_idx: int | None = None
        rebalance_nodes: list[tuple[int, date, pd.DataFrame | None, dict[str, Any], dict[str, Any], bool, str | None]] = []

        for snap_day, _, h_df, m_dict, aud, spec in cand_records:
            s_iso = snap_day.isoformat()
            if spec.candidate_spec_sha256 != spec0.candidate_spec_sha256:
                raise ArtifactIntegrityError(
                    f"candidate_spec_sha256 changed across snapshots for {cand_id}: "
                    f"{spec0.candidate_spec_sha256} != {spec.candidate_spec_sha256}"
                )
            if s_iso not in sess_to_idx:
                rebalance_schedule.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "candidate_spec_sha256": spec.candidate_spec_sha256,
                    "horizon_sessions": h,
                    "execution_interval": "open[t+1]->open[t+h+1]",
                    "status": "snapshot_not_in_spy_calendar",
                    "entry_session": aud["next_tradable_session"],
                    "exit_session": None,
                    "sessions_remaining_to_exit_open": h + 1,
                    "selected_equities": 0,
                    "abstain_reason": "snapshot_not_in_spy_calendar",
                })
                continue

            r_idx = sess_to_idx[s_iso]
            if h_df is None:
                msg = "holdings.parquet missing from snapshot"
                abstentions_and_rejections.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "horizon_sessions": h,
                    "status": "missing_persisted_holdings",
                    "reason": msg,
                })
                rebalance_schedule.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "candidate_spec_sha256": spec.candidate_spec_sha256,
                    "horizon_sessions": h,
                    "execution_interval": "open[t+1]->open[t+h+1]",
                    "status": "missing_persisted_holdings",
                    "entry_session": aud["next_tradable_session"],
                    "exit_session": None,
                    "sessions_remaining_to_exit_open": 0,
                    "selected_equities": 0,
                    "abstain_reason": msg,
                })
                continue

            cand_h = h_df.loc[h_df["candidate_id"].astype(str) == cand_id].reset_index(drop=True)
            if cand_h.empty:
                msg = f"{cand_id} missing from holdings.parquet"
                abstentions_and_rejections.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "horizon_sessions": h,
                    "status": "candidate_missing_from_holdings_parquet",
                    "reason": msg,
                })
                continue

            persisted_sha = str(cand_h["candidate_spec_sha256"].iloc[0] or "")
            if persisted_sha and persisted_sha != spec.candidate_spec_sha256:
                raise ArtifactIntegrityError(
                    f"holdings.parquet candidate_spec_sha256 mismatch for {cand_id} on {s_iso}: "
                    f"{persisted_sha} != {spec.candidate_spec_sha256}"
                )

            if anchor_r_idx is not None and (r_idx - anchor_r_idx) % h != 0:
                rebalance_schedule.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "candidate_spec_sha256": spec.candidate_spec_sha256,
                    "horizon_sessions": h,
                    "execution_interval": "open[t+1]->open[t+h+1]",
                    "status": "non_rebalance_observation_session",
                    "entry_session": aud["next_tradable_session"],
                    "exit_session": None,
                    "sessions_remaining_to_exit_open": 0,
                    "selected_equities": int((cand_h["asset_role"].astype(str) == "satellite_equity").sum()),
                    "abstain_reason": None,
                })
                continue

            if bool(cand_h["abstained"].iloc[0]):
                abs_reason = str(cand_h["abstain_reason"].iloc[0] or "abstained_without_reason")
                abstentions_and_rejections.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "horizon_sessions": h,
                    "status": "abstained",
                    "reason": abs_reason,
                })
                rebalance_schedule.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "candidate_spec_sha256": spec.candidate_spec_sha256,
                    "horizon_sessions": h,
                    "execution_interval": "open[t+1]->open[t+h+1]",
                    "status": "abstained",
                    "entry_session": aud["next_tradable_session"],
                    "exit_session": None,
                    "sessions_remaining_to_exit_open": 0,
                    "selected_equities": 0,
                    "abstain_reason": abs_reason,
                })
                if anchor_r_idx is not None:
                    rebalance_nodes.append((r_idx, snap_day, cand_h, m_dict, aud, True, abs_reason))
                continue

            if anchor_r_idx is None:
                anchor_r_idx = r_idx
            rebalance_nodes.append((r_idx, snap_day, cand_h, m_dict, aud, False, None))

        # Separate matured vs pending rebalance nodes (`exit_idx = r_idx + h + 1` for `open[r+1]->open[r+h+1]`)
        matured_chain: list[tuple[int, date | None, pd.DataFrame | None, dict[str, Any] | None, dict[str, Any] | None, bool]] = []
        prev_r_idx: int | None = None
        for r_idx, node_day, cand_h, node_m_dict, aud, is_abs, node_abs_reason in rebalance_nodes:
            s_iso = node_day.isoformat()
            entry_idx = r_idx + 1
            exit_idx = r_idx + h + 1
            eq_cnt = int((cand_h["asset_role"].astype(str) == "satellite_equity").sum()) if cand_h is not None else 0
            if is_abs:
                if prev_r_idx is not None and r_idx - prev_r_idx > h:
                    matured_chain.append((prev_r_idx + h, None, None, None, None, True))
                matured_chain.append((r_idx, node_day, cand_h, node_m_dict, aud, True))
                prev_r_idx = r_idx
                continue

            if entry_idx >= len(sessions):
                rebalance_schedule.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "candidate_spec_sha256": spec0.candidate_spec_sha256,
                    "horizon_sessions": h,
                    "execution_interval": "open[t+1]->open[t+h+1]",
                    "status": "pending_entry_bar",
                    "entry_session": aud["next_tradable_session"],
                    "exit_session": None,
                    "sessions_remaining_to_exit_open": exit_idx - (len(sessions) - 1),
                    "selected_equities": eq_cnt,
                    "abstain_reason": node_abs_reason,
                })
                continue
            if exit_idx >= len(sessions):
                rebalance_schedule.append({
                    "snapshot_date": s_iso,
                    "candidate_id": cand_id,
                    "candidate_spec_sha256": spec0.candidate_spec_sha256,
                    "horizon_sessions": h,
                    "execution_interval": "open[t+1]->open[t+h+1]",
                    "status": "pending_maturation",
                    "entry_session": sessions[entry_idx],
                    "exit_session": None,
                    "sessions_remaining_to_exit_open": exit_idx - (len(sessions) - 1),
                    "selected_equities": eq_cnt,
                    "abstain_reason": node_abs_reason,
                })
                continue

            if prev_r_idx is not None and r_idx - prev_r_idx > h:
                # Unrecorded intermediate rebalance gap -> insert abstention row so drift resets cleanly
                matured_chain.append((prev_r_idx + h, None, None, None, None, True))
            matured_chain.append((r_idx, node_day, cand_h, node_m_dict, aud, False))
            prev_r_idx = r_idx

        active_matured = [node for node in matured_chain if not node[5] and node[2] is not None]
        if not active_matured:
            continue

        # Build canonical asset index (`[satellite_equities..., benchmark:SPY]`) across matured chain
        tk_by_sid: dict[str, str] = {}
        for _, _, cand_h, _, _, is_abs in matured_chain:
            if is_abs or cand_h is None:
                continue
            eq_rows = cand_h.loc[cand_h["asset_role"].astype(str) == "satellite_equity"]
            for _, hr in eq_rows.iterrows():
                tk_by_sid[str(hr["security_id"])] = str(hr["ticker"] or "").upper().strip()
        eq_sids = sorted(tk_by_sid.keys())
        eq_col_of = {sid: idx for idx, sid in enumerate(eq_sids)}
        n_eq = len(eq_sids)
        k_len = len(matured_chain)

        eq_w_dec = np.full((k_len, max(1, n_eq)), np.nan, dtype=np.float64)
        eq_tradable = np.zeros((k_len, max(1, n_eq)), dtype=bool)
        asset_returns = np.full((k_len, n_eq + 1), np.nan, dtype=np.float64)
        spy_open_rets = np.full(k_len, np.nan, dtype=np.float64)
        is_regime_spy_shift = np.zeros(k_len, dtype=bool)

        for k, (r_idx, m_day, cand_h, _, _, is_abs) in enumerate(matured_chain):
            if is_abs or cand_h is None or m_day is None:
                continue
            entry_idx = r_idx + 1
            exit_idx = r_idx + h + 1
            spy_in, _ = bars.open_close_at("benchmark:SPY", "SPY", entry_idx)
            spy_out, _ = bars.open_close_at("benchmark:SPY", "SPY", exit_idx)
            if np.isfinite(spy_in) and spy_in > 0.0 and np.isfinite(spy_out) and spy_out > 0.0:
                spy_ret_k = float(spy_out / spy_in - 1.0)
                spy_open_rets[k] = spy_ret_k
                asset_returns[k, n_eq] = spy_ret_k

            for j, sid in enumerate(eq_sids):
                o_in, _ = bars.open_close_at(sid, tk_by_sid[sid], entry_idx)
                o_out, _ = bars.open_close_at(sid, tk_by_sid[sid], exit_idx)
                in_ok = bool(np.isfinite(o_in) and o_in > 0.0)
                out_ok = bool(np.isfinite(o_out) and o_out > 0.0)
                eq_tradable[k, j] = in_ok
                if in_ok and out_ok:
                    asset_returns[k, j] = float(o_out / o_in - 1.0)

            roles = set(cand_h["asset_role"].astype(str).tolist())
            if "regime_spy_shift" in roles:
                is_regime_spy_shift[k] = True
            else:
                row_u = np.zeros(max(1, n_eq), dtype=np.float64)
                eq_rows = cand_h.loc[cand_h["asset_role"].astype(str) == "satellite_equity"]
                for _, hr in eq_rows.iterrows():
                    sid_h = str(hr["security_id"])
                    if sid_h in eq_col_of:
                        row_u[eq_col_of[sid_h]] = float(hr["target_weight"])
                s_u = float(row_u.sum())
                if s_u > 0.0:
                    eq_w_dec[k] = row_u / s_u

        eq_w_filled = benchmark_tilt._apply_open_fills(eq_w_dec, eq_tradable)
        w_joint = np.full((k_len, n_eq + 1), np.nan, dtype=np.float64)
        for k in range(k_len):
            if not np.isfinite(spy_open_rets[k]):
                continue
            if is_regime_spy_shift[k]:
                w_joint[k, :] = 0.0
                w_joint[k, n_eq] = 1.0
            elif n_eq > 0 and bool(np.isfinite(eq_w_filled[k]).all()) and float(eq_w_filled[k].sum()) > 0.0:
                w_joint[k, :n_eq] = float(spec0.satellite_weight) * eq_w_filled[k]
                w_joint[k, n_eq] = float(spec0.core_spy_weight)

        path = benchmark_tilt.simulate_portfolio_path(w_joint, asset_returns)
        cost_rate = float(spec0.cost_bps) / 10_000.0

        snap_items: list[dict[str, Any]] = []
        net_vec: list[float] = []
        spy_vec: list[float] = []
        gross_vec: list[float] = []
        tau_vec: list[float] = []
        steady_vec: list[float] = []

        for k, (r_idx, m_day2, cand_h, m_man, _, is_abs) in enumerate(matured_chain):
            if is_abs or m_day2 is None or cand_h is None or m_man is None:
                continue
            s_iso = m_day2.isoformat()
            entry_idx = r_idx + 1
            exit_idx = r_idx + h + 1
            w_row = w_joint[k]
            r_row = asset_returns[k]
            tau_k = float(path.one_way_turnovers[k]) if np.isfinite(path.one_way_turnovers[k]) else None
            steady_k = float(path.steady_turnovers[k]) if np.isfinite(path.steady_turnovers[k]) else None
            cost_k = float(cost_rate * tau_k) if tau_k is not None else None
            gross_k = float(path.gross_returns[k]) if np.isfinite(path.gross_returns[k]) else None
            net_k = float(gross_k - cost_k) if (gross_k is not None and cost_k is not None) else None
            spy_k = float(spy_open_rets[k]) if np.isfinite(spy_open_rets[k]) else None
            excess_k = float(net_k - spy_k) if (net_k is not None and spy_k is not None) else None
            period_complete = gross_k is not None

            filled_map: dict[str, float] = {}
            drifted_map: dict[str, float] = {}
            if bool(np.isfinite(w_row).all()):
                for j, sid in enumerate(eq_sids):
                    if w_row[j] > 0.0:
                        filled_map[sid] = float(w_row[j])
                if w_row[n_eq] > 0.0:
                    filled_map["benchmark:SPY"] = float(w_row[n_eq])
                if gross_k is not None and (1.0 + gross_k) > 1e-12:
                    for j, sid in enumerate(eq_sids):
                        if w_row[j] > 0.0:
                            drifted_map[sid] = float(w_row[j] * (1.0 + r_row[j]) / (1.0 + gross_k))
                    if w_row[n_eq] > 0.0:
                        drifted_map["benchmark:SPY"] = float(w_row[n_eq] * (1.0 + r_row[n_eq]) / (1.0 + gross_k))

            opra_reg, _ = extract_snapshot_opra_regime(m_man)
            opra_back: bool | None = (
                True if opra_reg == "backwardation" else (False if opra_reg == "contango" else None)
            )
            eq_cnt = int((cand_h["asset_role"].astype(str) == "satellite_equity").sum())

            rebalance_schedule.append({
                "snapshot_date": s_iso,
                "candidate_id": cand_id,
                "candidate_spec_sha256": spec0.candidate_spec_sha256,
                "horizon_sessions": h,
                "execution_interval": "open[t+1]->open[t+h+1]",
                "status": "matured_complete" if period_complete else "matured_incomplete_exit",
                "entry_session": sessions[entry_idx],
                "exit_session": sessions[exit_idx],
                "sessions_remaining_to_exit_open": 0,
                "selected_equities": eq_cnt,
                "abstain_reason": None,
            })

            snap_items.append({
                "snapshot_date": s_iso,
                "entry_session": sessions[entry_idx],
                "exit_session": sessions[exit_idx],
                "execution_interval": "open[t+1]->open[t+h+1]",
                "period_complete": period_complete,
                "opra_regime": "not_applicable" if spec0.regime_rule == "none" else opra_reg,
                "opra_backwardation": None if spec0.regime_rule == "none" else opra_back,
                "selected_equities": eq_cnt,
                "filled_holdings": filled_map,
                "drifted_end_weights": drifted_map,
                "one_way_turnover": tau_k,
                "steady_one_way_turnover": steady_k,
                "transaction_cost": cost_k,
                "portfolio_gross_return": gross_k,
                "portfolio_net_20bps": net_k,
                "spy_return": spy_k,
                "net_excess_vs_spy": excess_k,
            })
            if tau_k is not None:
                tau_vec.append(tau_k)
            if steady_k is not None:
                steady_vec.append(steady_k)
            if period_complete and net_k is not None and gross_k is not None and spy_k is not None:
                net_vec.append(net_k)
                gross_vec.append(gross_k)
                spy_vec.append(spy_k)

        if not snap_items:
            continue

        all_complete = bool(path.portfolio_complete) and len(net_vec) == len(snap_items)
        comp_gross = float(np.prod(1.0 + np.asarray(gross_vec)) - 1.0) if (all_complete and gross_vec) else None
        comp_net = float(np.prod(1.0 + np.asarray(net_vec)) - 1.0) if (all_complete and net_vec) else None
        comp_spy = float(np.prod(1.0 + np.asarray(spy_vec)) - 1.0) if (all_complete and spy_vec) else None

        pooled_candidates.setdefault(str(h), {})[cand_id] = {
            "candidate_id": cand_id,
            "candidate_spec_sha256": spec0.candidate_spec_sha256,
            "horizon_sessions": h,
            "execution_interval": "open[t+1]->open[t+h+1]",
            "cost_bps": spec0.cost_bps,
            "portfolio_complete": all_complete,
            "unresolved_held_periods": path.unresolved_held_periods,
            "unresolved_held_positions": path.unresolved_held_positions,
            "matured_dates": len(snap_items),
            "complete_matured_dates": len(net_vec),
            "compounded_gross": comp_gross,
            "compounded_net_20bps": comp_net,
            "compounded_spy_return": comp_spy,
            "compounded_net_excess_vs_spy": (
                float(comp_net - comp_spy) if (comp_net is not None and comp_spy is not None) else None
            ),
            "mean_one_way_turnover": float(np.mean(tau_vec)) if tau_vec else None,
            "mean_steady_one_way_turnover": float(np.mean(steady_vec)) if steady_vec else None,
            "mean_portfolio_gross": float(np.mean(gross_vec)) if all_complete and gross_vec else None,
            "mean_portfolio_net_20bps": float(np.mean(net_vec)) if all_complete and net_vec else None,
            "mean_spy_return": float(np.mean(spy_vec)) if all_complete and spy_vec else None,
            "mean_net_excess_vs_spy": (
                float(np.mean(np.asarray(net_vec) - np.asarray(spy_vec))) if all_complete and net_vec else None
            ),
            "backwardation_dates": sum(1 for x in snap_items if x["opra_backwardation"] is True),
            "by_snapshot": snap_items,
        }

    return pooled_candidates, rebalance_schedule, abstentions_and_rejections


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
        snapshots: list[tuple[date, pd.DataFrame, pd.DataFrame | None, dict[str, Any], dict[str, Any]]] = []
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
                holdings_pq = child / "holdings.parquet"
                h_df = (
                    cast(Any, pq).read_table(holdings_pq, use_threads=False).to_pandas()
                    if holdings_pq.is_file()
                    else None
                )
                snapshots.append((snap_day, s_df, h_df, m_dict, audit))

        if not snapshots:
            raise ValueError(f"no prospective forward snapshots found <= {eval_day.isoformat()} in {records_abs}")

        clock_admitted_snapshots = [
            (d, s_df, h_df, m_dict, aud)
            for d, s_df, h_df, m_dict, aud in snapshots
            if (
                (allow_retrospective_replay and aud["record_mode"] == "retrospective_replay")
                or (aud["record_mode"] != "retrospective_replay" and bool(aud["published_before_entry"]))
            )
        ]

        all_tickers: set[str] = set()
        for _, s_df, h_df, _, _ in clock_admitted_snapshots:
            if "ticker" in s_df.columns:
                for tk_val in s_df["ticker"].dropna().astype(str):
                    if tk_val.strip():
                        all_tickers.add(tk_val.upper().strip())
            else:
                for sid in s_df["security_id"].astype(str):
                    all_tickers.add(sid.split(":", 1)[1] if ":" in sid else sid)
            if h_df is not None and "ticker" in h_df.columns:
                for tk_h in h_df["ticker"].dropna().astype(str):
                    if tk_h.strip() and tk_h.strip() != "SPY":
                        all_tickers.add(tk_h.upper().strip())

        earliest_snap = min(d for d, _, _, _, _ in snapshots)
        bars = (
            _resolve_outcome_bars(
                paths=paths,
                all_tickers=tuple(sorted(all_tickers)),
                earliest_snap=earliest_snap,
                eval_day=eval_day,
                cfg=cfg,
                allow_retrospective_replay=allow_retrospective_replay,
                unseal_final_test_period=unseal_final_test_period,
            )
            if clock_admitted_snapshots
            else _OutcomeBarLookup(
                sessions=[],
                sess_to_idx={},
                spy_open=np.empty(0, dtype=np.float64),
                spy_close=np.empty(0, dtype=np.float64),
                open_by_sid={},
                close_by_sid={},
                open_by_tk={},
                close_by_tk={},
            )
        )
        sessions = bars.sessions
        sess_to_idx = bars.sess_to_idx
        spy_open = bars.spy_open
        spy_close = bars.spy_close

        maturation_schedule: list[dict[str, Any]] = []
        per_cell_signals: dict[int, dict[str, list[dict[str, Any]]]] = {}
        opra_diagnostics: list[dict[str, Any]] = []

        for snap_day, s_df, _, m_dict, aud in snapshots:
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
            sids_row = tuple(s_df["security_id"].astype(str).tolist())
            if "ticker" in s_df.columns:
                tks = [str(t).upper().strip() for t in s_df["ticker"].fillna("").astype(str)]
            else:
                tks = [sid.split(":", 1)[1] if ":" in sid else sid for sid in sids_row]
            opt_sum = m_dict.get("options_summary") or {}
            has_opra = bool(opt_sum.get("opra_enabled"))
            opra_regime, opra_obs_kind = extract_snapshot_opra_regime(m_dict)
            is_backw: bool | None = (
                True if opra_regime == "backwardation" else (False if opra_regime == "contango" else None)
            )

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
                rets = np.empty(len(tks), dtype=np.float64)
                for i_tk, (sid_i, tk_i) in enumerate(zip(sids_row, tks, strict=False)):
                    o_v, _ = bars.open_close_at(sid_i, tk_i, entry_idx)
                    _, c_v = bars.open_close_at(sid_i, tk_i, exit_idx)
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

                for col in SIGNAL_COLUMNS:
                    if col not in s_df.columns:
                        continue
                    arr = pd.to_numeric(s_df[col], errors="coerce").to_numpy(dtype=np.float64)
                    res_c = _score_single_signal_cohort(arr, rets, spy_r, security_ids=sids_row)
                    if res_c is not None:
                        per_cell_signals.setdefault(h, {}).setdefault(col, []).append({"snapshot_date": s_iso, **res_c})

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
                        "opra_regime": opra_regime,
                        "opra_observation_kind": opra_obs_kind,
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
                ic_vals = [float(x["observed_subset_rank_ic"]) for x in items if x["observed_subset_rank_ic"] is not None]
                all_top10_ok = all(bool(x["top10_complete"]) for x in items)
                all_ew_ok = all(x["top10_net_minus_ew"] is not None for x in items)
                pooled_signals[str(h)][sig_name] = {
                    "matured_dates": len(items),
                    "complete_top10_dates": sum(1 for x in items if bool(x["top10_complete"])),
                    "incomplete_top10_dates": sum(1 for x in items if not bool(x["top10_complete"])),
                    "top10_complete": all_top10_ok,
                    "rank_ic_scope": "observed_subset_diagnostic",
                    "observed_subset_mean_rank_ic": float(np.mean(ic_vals)) if ic_vals else None,
                    "mean_rank_ic": float(np.mean(ic_vals)) if ic_vals else None,
                    "mean_top10_net_20bps": (
                        float(np.mean([float(x["top10_net_20bps_return"]) for x in items])) if all_top10_ok else None
                    ),
                    "mean_spy_return": float(np.mean([float(x["spy_return"]) for x in items])),
                    "mean_top10_net_minus_spy": (
                        float(np.mean([float(x["top10_net_minus_spy"]) for x in items])) if all_top10_ok else None
                    ),
                    "mean_top10_net_minus_ew": (
                        float(np.mean([float(x["top10_net_minus_ew"]) for x in items])) if (all_top10_ok and all_ew_ok) else None
                    ),
                    "by_snapshot": items,
                }

        pooled_candidates, candidate_rebalance_schedule, candidate_abstentions_and_rejections = (
            _evaluate_persisted_candidates(clock_admitted_snapshots, bars)
            if clock_admitted_snapshots
            else ({}, [], [])
        )

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        out_rel = output or (FORWARD_EVAL_DIRECTORY / f"forward_eval_{stamp}")
        dest = paths.write(paths.identity(out_rel))
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.parent / f".{dest.name}.{uuid4().hex}.partial"
        staging.mkdir(parents=True)

        admission_audits = [aud for _, _, _, _, aud in snapshots]
        after_entry_dates = [a["snapshot_date"] for a in admission_audits if a["clock_status"] == "recorded_after_entry"]
        before_entry_dates = [a["snapshot_date"] for a in admission_audits if bool(a["published_before_entry"])]
        sec_qualified_dates = [a["snapshot_date"] for a in admission_audits if bool(a["qualifies_for_persisted_sec_model"])]

        doc = _json_safe({
            "evaluation_mode": "retrospective_replay" if allow_retrospective_replay else "prospective",
            "evaluated_at_utc": datetime.now(UTC).isoformat(),
            "evaluated_as_of_date": eval_day.isoformat(),
            "sealed_holdout_window": "2024-05-29..2026-07-08",
            "sealed_holdout_untouched": not unseal_final_test_period,
            "current_reference_candidates": list(FROZEN_PROSPECTIVE_CANDIDATES),
            "snapshots_total": len(snapshots),
            "snapshot_dates": [d.isoformat() for d, _, _, _, _ in snapshots],
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
            "candidate_rebalance_schedule": candidate_rebalance_schedule,
            "candidate_abstentions_and_rejections": candidate_abstentions_and_rejections,
            "opra_diagnostics": opra_diagnostics,
            "code": git_identity(root),
            "label_notice": (
                "research label: signal diagnostics use close[t+h]/open[t+1] - 1; "
                "candidate portfolios use contiguous open[t+1]->open[t+h+1] with post-drift turnover on adjusted SIP daily bars"
            ),
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
        "matured_candidate_rebalances": sum(
            1 for r in candidate_rebalance_schedule if str(r["status"]).startswith("matured_")
        ),
        "pending_candidate_rebalances": sum(
            1 for r in candidate_rebalance_schedule if r["status"] in ("pending_entry_bar", "pending_maturation")
        ),
        "candidate_abstentions_or_incomplete": len(candidate_abstentions_and_rejections),
    }


__all__ = [
    "DEFAULT_EVAL_HORIZONS",
    "EVALUATE_FORWARD_COMMAND",
    "FORWARD_EVAL_DIRECTORY",
    "evaluate_forward_scores",
]
