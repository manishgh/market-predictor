"""Score validated wider beta-neutral + precise SEC catalyst research-lane models under the heavy-job lease."""
from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import pyarrow.parquet as pq

from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.process_memory import process_memory_snapshot, release_unused_process_memory
from market_predictor.research_lane import (
    beta_neutral,
    free_backfill,
    fundamentals,
    two_stage,
    wider_universe,
)
from market_predictor.research_lane.company_facts import (
    IssuerFacts,
    read_company_facts,
)
from market_predictor.research_lane.fitted import (
    RIDGE_ALPHA,
    SAMPLE_EVERY_SESSIONS,
)
from market_predictor.research_lane.labels import HORIZONS
from market_predictor.research_lane.panel import (
    PANEL_DIRECTORY,
    check_development_end,
    file_sha256,
)
from market_predictor.research_lane.trial_log import TRIAL_LOG_NAME, append_trial

WIDER_BETA_NEUTRAL_COMMAND = "score-wider-beta-neutral-models"
OUTPUT_PARENT = Path("data/research/research_lane")
LABEL_NOTICE = (
    "research label, adjusted prices, development history {first} to {end}, "
    "not proof of outperformance"
)
READING_RULE = (
    "With about 0.03 uncertainty per out-of-sample rank correlation, the best of 24 would reach roughly "
    "0.05 by chance; the conclusion rests on the median and on agreement across learners, horizons and "
    "years. Technical features solve the problem only if the median is at or above 0.03 and most "
    "specifications are positive in at least 3 of 4 scored years."
)


def _process_memory_gib() -> dict[str, float] | None:
    snapshot = process_memory_snapshot()
    if snapshot is None:
        return None
    current, peak = snapshot
    return {"current_gib": current / 2**30, "peak_gib": peak / 2**30}


def default_output(now: datetime | None = None, prefix: str = "wider_beta_neutral") -> Path:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_PARENT / f"{prefix}_{stamp}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float | np.floating):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _number(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def fitted_summary(specifications: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Median/min/max out-of-sample mean rank correlation and year agreement across specifications."""
    means = {
        key: spec["rank_correlation_mean"]
        for key, spec in specifications.items()
        if spec["rank_correlation_mean"] is not None
    }
    values = list(means.values())
    consistent = sum(1 for spec in specifications.values() if spec["positive_years"] >= 3)

    def grouped(field_name: str) -> dict[str, float | None]:
        groups: dict[str, list[float]] = {}
        for key, spec in specifications.items():
            if key in means:
                groups.setdefault(str(spec[field_name]), []).append(means[key])
        return {name: _median(items) for name, items in groups.items()}

    median = _median(values)
    return {
        "specifications": len(specifications),
        "median_mean_rank_correlation": median,
        "minimum_mean_rank_correlation": min(values) if values else None,
        "maximum_mean_rank_correlation": max(values) if values else None,
        "specifications_positive_in_at_least_3_of_4_years": consistent,
        "median_by_learner": grouped("learner"),
        "median_by_feature_set": grouped("feature_set"),
        "median_by_horizon_sessions": grouped("horizon_sessions"),
        "reading_rule": READING_RULE,
        "rule_met": bool(median is not None and median >= 0.03 and consistent > len(specifications) / 2),
    }


def _render_two_stage_markdown(document: Mapping[str, Any]) -> str:
    settings = document["settings"]
    ts = document["two_stage"]
    notice = document["notice"]
    horizons = [str(h) for h in settings["horizons_sessions"]]
    years = ["2020", "2021", "2022", "2023", "2024"]

    lines = [
        "# Recommendation Brief §5-§6 Steps 2-4: Combined Stage-1 & Stage-2 News Residual Scoreboard",
        "",
        f"Every figure below: {notice}.",
        "",
        "## Method & boundaries",
        "",
        f"- **Stage-1 base history:** warm-up daily bars from `2016-01-04`; eligible decisions "
        f"from `{settings['stage1_first_decision_session']}` to `{settings['development_end']}` (out-of-sample walk-forward "
        "scored across `2020..2024`, purged so `t + h < fold.test_start`).",
        "- **Stage-1 inputs (`48` features, zero duplicate missingness indicators):** `24` pre-signed cross-sectional "
        "centered ranks (`12` Step-1 price + `12` Step-2a SEC earnings/surprise/valuation/quality) + `24` sector-neutral "
        "centered ranks.",
        "- **Stage-2 news-reaction residual model (`2019-07-09..end`, OOS `2021..2024`):** continuous per-category "
        "market-adjusted 1d reaction, 20d abnormal volume, recency, and 5d/60d unusual article volume fitted strictly "
        "on out-of-sample Stage-1 residuals.",
        "",
        "## Section A: Stage-1 models — Later-date OOS, Unseen-issuer CV, Buffered Skill & Cap-Tilt Product",
        "",
        "| Model | H | Later-date IC [NW t] | Unseen-issuer IC [NW t] | Sector-neutral [t] | "
        + " | ".join(years)
        + " | Pos yrs | Skill hold33 net 20bps/yr [t] | Cap-tilt vs SPY/yr [t] |",
        "|---|---|---|---|---|" + "---|" * len(years) + "---|---|---|",
    ]
    for model_name, h_map in ts["stage1_models"].items():
        for h in horizons:
            entry = h_map[h]
            sc = entry["score"]
            rc = sc["rank_correlation"]
            sn = sc["sector_neutral_rank_correlation"]
            u_rc = entry["unseen_issuer_rank_correlation"]
            u_txt = "n/a" if u_rc is None else f"{_number(u_rc['mean'])} [{_number(u_rc['t'], 2)}]"
            by_yr = sc["rank_correlation_by_year"]
            yr_cells = [_number(by_yr.get(y, {}).get("mean")) for y in years]
            pos_yrs = sum(1 for v in by_yr.values() if v["mean"] > 0)
            sk = entry["skill_buffered_hold33"]["vs_matched_benchmark"]
            pr = entry["product_sector_active_slice"]["vs_spy"]
            lines.append(
                f"| {model_name} | {h} | {_number(rc['mean'])} [{_number(rc['t'], 2)}] | {u_txt} | "
                f"{_number(sn['mean'])} [{_number(sn['t'], 2)}] | "
                + " | ".join(yr_cells)
                + f" | {pos_yrs}/{sc['years_scored']} | "
                f"{_percent(sk['net_excess_20bps_annualized'])} [{_number(sk['newey_west_20bps']['t'], 2)}] | "
                f"{_percent(pr['net_excess_20bps_annualized'])} [{_number(pr['newey_west_20bps']['t'], 2)}] |"
            )

    lines += [
        "",
        "## Section B: Stage-2 news-reaction residual models — Paired daily IC improvement (OOS 2021..2024)",
        "",
        "| Specification | H | Combined IC [NW t] | Paired delta IC [NW t] | Pos delta yrs | Passes paired gate (t >= 2.0) |",
        "|---|---|---|---|---|---|",
    ]
    for spec_key, h_map in ts["stage2_news_models"].items():
        for h in horizons:
            entry = h_map[h]
            rc = entry["score"]["rank_correlation"]
            pi = entry["paired_ic_improvement"]
            lines.append(
                f"| {spec_key} | {h} | {_number(rc['mean'])} [{_number(rc['t'], 2)}] | "
                f"{_number(pi['mean'])} [{_number(pi['t'], 2)}] | {pi['positive_years']}/{pi['years_scored']} | "
                f"{'yes' if pi['passes_paired_news_gate'] else 'no'} |"
            )

    lines += [
        "",
        "## Section C: Top 12 Stage-1 features at 21 sessions",
        "",
    ]
    for model_name in ("ridge_rank", "xgb_shallow_5seed"):
        if model_name in ts["stage1_models"] and "21" in ts["stage1_models"][model_name]:
            feats = ts["stage1_models"][model_name]["21"]["top_features"]
            lines.append(f"- **{model_name}:** " + ", ".join(f"{f['feature']} {f['importance']:.4g}" for f in feats))

    acc = ts["acceptance_summary"]
    lines += [
        "",
        "## Section D: §5.6 Pre-registered acceptance rule (21-session primary horizon)",
        "",
        acc["rule"],
        "",
        "| Candidate | 21d IC [NW t] | Pos yrs share | Unseen-issuer 21d IC | "
        "Buffered skill net 20bps/21d | Beats/is baseline | Passes §5.6 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c_name, c_val in acc["candidates"].items():
        lines.append(
            f"| {c_name} | {_number(c_val['mean_ic_21d'])} [{_number(c_val['newey_west_t_21d'], 2)}] | "
            f"{_percent(c_val['share_of_years_positive_21d'])} | {_number(c_val['unseen_issuer_ic_21d'])} | "
            f"{_percent(c_val['buffered_skill_net_20bps_per_21d'])} | "
            f"{'yes' if c_val['beats_or_is_equal_rank_24'] else 'no'} | "
            f"{'yes' if c_val['passes_section_5_6'] else 'no'} |"
        )
    lines += [
        "",
        f"- Passing Stage-1 base models: `{acc['passing_models']}`.",
        f"- Accepted Stage-2 news overlays (`paired t >= 2.0`): `{acc['accepted_stage2_news_overlays']}`.",
        (
            "Verdict: **earns one sealed-period test**."
            if acc["earns_sealed_period_test"]
            else "Verdict: **no candidate passed §5.6**."
        ),
        "",
    ]
    return "\n".join(lines)


def score_wider_beta_neutral_models(
    *,
    root: Path,
    output: Path | None = None,
    development_end: date,
    subset: wider_universe.UniverseSubset = "mid_small_ex_sp500",
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    sp500_backfill_dir: Path = free_backfill.BACKFILL_DIRECTORY,
    wider_dir: Path = wider_universe.WIDER_UNIVERSE_DIRECTORY,
    horizons: tuple[int, ...] = HORIZONS,
    unseal_final_test_period: bool = False,
) -> dict[str, Any]:
    """Run Step 5c beta/vol/sector-neutral Stage-1 + high-precision $0 SEC catalyst Stage-2 on wider US equities."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    output = output or default_output(prefix=f"wider_beta_neutral_{subset}")
    identity = paths.identity(output)
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(WIDER_BETA_NEUTRAL_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("wider beta-neutral preparation")
        destination = paths.write(identity)
        if destination.exists():
            raise FileExistsError(f"research lane output already exists: {destination}")

        wdata = wider_universe.load_wider_universe_data(
            paths,
            development_end=development_end,
            subset=subset,
            panel_rel=panel,
            sp500_backfill_rel=sp500_backfill_dir,
            wider_rel=wider_dir,
            unseal_final_test_period=unseal_final_test_period,
        )
        guard("wider beta-neutral panel loaded")

        first_row = int(
            np.searchsorted(wdata.panel.sessions, np.datetime64(free_backfill.EARLY_DECISION_START, "D"), side="left")
        )
        facts_counts: Counter[str] = Counter()
        issuers_loaded = 0
        facts_paths = wdata.company_facts_paths

        def _load_fact(cik: str) -> IssuerFacts | None:
            nonlocal issuers_loaded
            p = facts_paths.get(cik)
            if p is None or not p.is_file():
                return None
            facts, cnt = read_company_facts(p, development_end)
            facts_counts.update(cnt)
            issuers_loaded += 1
            if issuers_loaded % 40 == 0:
                guard("wider beta-neutral company facts")
            return facts

        fund_inputs = fundamentals.FundamentalInputs(
            panel=wdata.panel,
            unadjusted_close=wdata.unadjusted_close,
            releases=wdata.releases_202,
            security_issuer=wdata.security_cik_map,
            load_issuer=_load_fact,
        )
        bundle = two_stage.build_stage1_features(wdata.panel, fund_inputs)
        w_panel = wdata.panel
        sec_cik_map = wdata.security_cik_map
        all_8k_events = wdata.all_8k_events
        w_sha = dict(wdata.input_sha256)
        w_rep = dict(wdata.report)
        del fund_inputs, wdata
        release_unused_process_memory()
        guard("wider beta-neutral stage1 features computed")

        non8k_path = paths.directory(wider_dir) / beta_neutral.NON8K_CATALYST_PARQUET
        w_sha["wider_non8k_catalysts_sha256"] = file_sha256(non8k_path)
        non8k_df = cast(Any, pq).read_table(non8k_path, use_threads=False).to_pandas()

        cat_feats, cat_rep = beta_neutral.build_precise_sec_catalyst_features(
            w_panel,
            all_8k_events,
            non8k_df,
            unseal_final_test_period=unseal_final_test_period,
        )
        del all_8k_events, non8k_df
        release_unused_process_memory()
        guard("wider beta-neutral precise SEC catalyst features computed")

        stage1_merged: dict[str, dict[str, Any]] = {}
        stage2_merged: dict[str, dict[str, Any]] = {}
        acceptance_merged: dict[str, Any] = {
            "rule": two_stage.SECTION_5_6_ACCEPTANCE_RULE,
            "candidates": {},
            "passing_models": [],
            "accepted_stage2_news_overlays": [],
            "earns_sealed_period_test": False,
        }
        for h in horizons:
            s1_h = beta_neutral.run_beta_neutral_stage1_walk_forward(
                w_panel, bundle, sec_cik_map, first_row=first_row, horizons=(h,), checkpoint=guard,
            )
            s2_h = two_stage.run_stage2_news_residuals(
                w_panel,
                s1_h,
                cat_feats,
                horizons=(h,),
                feature_names=beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
                checkpoint=guard,
            )
            ev_h = two_stage.evaluate_two_stage_scoreboard(
                w_panel, bundle, s1_h, s2_h, horizons=(h,),
            )
            for m_name, h_dict in ev_h["stage1_models"].items():
                stage1_merged.setdefault(m_name, {}).update(h_dict)
            for sp_key, h_dict in ev_h["stage2_news_models"].items():
                stage2_merged.setdefault(sp_key, {}).update(h_dict)
            if h == two_stage.PRIMARY_HORIZON:
                acceptance_merged["candidates"] = ev_h["acceptance_summary"]["candidates"]
                acceptance_merged["passing_models"] = ev_h["acceptance_summary"]["passing_models"]
                acceptance_merged["earns_sealed_period_test"] = ev_h["acceptance_summary"]["earns_sealed_period_test"]
            acceptance_merged["accepted_stage2_news_overlays"].extend(
                ev_h["acceptance_summary"]["accepted_stage2_news_overlays"]
            )
            del s1_h, s2_h, ev_h
            release_unused_process_memory()
            guard(f"wider beta-neutral horizon {h} completed")

        del cat_feats
        release_unused_process_memory()
        evaluation: dict[str, Any] = {
            "stage1_models": stage1_merged,
            "stage2_news_models": stage2_merged,
            "acceptance_summary": acceptance_merged,
        }

        settings: dict[str, Any] = {
            "universe_subset": subset,
            "neutralization": "per_session_ols_orthogonal_to_sector_beta252_idiovol60",
            "development_end": development_end.isoformat(),
            "unseal_final_test_period": unseal_final_test_period,
            "stage1_first_decision_session": str(w_panel.sessions[first_row]),
            "stage2_news_start_date": two_stage.NEWS_START_DATE.isoformat(),
            "horizons_sessions": list(horizons),
            "primary_horizon_sessions": two_stage.PRIMARY_HORIZON,
            "secondary_horizon_sessions": two_stage.SECONDARY_HORIZON,
            "stage1_feature_count": len(bundle.feature_names),
            "stage2_catalyst_feature_count": len(beta_neutral.PRECISE_STAGE2_FEATURE_NAMES),
            "sample_every_sessions": SAMPLE_EVERY_SESSIONS,
            "ridge_alpha": RIDGE_ALPHA,
            "stage1_xgb_params": dict(two_stage.STAGE1_XGB_PARAMS),
            "stage1_seeds": list(two_stage.STAGE1_SEEDS),
            "stage2_xgb_params": dict(two_stage.STAGE2_XGB_PARAMS),
            "stage2_seeds": list(two_stage.STAGE2_XGB_SEEDS),
            "unseen_issuer_folds": two_stage.UNSEEN_ISSUER_FOLDS,
            "label": "research label on provider split- and dividend-adjusted SIP daily bars",
        }
        document = _json_safe({
            "notice": LABEL_NOTICE.format(
                first=settings["stage1_first_decision_session"], end=settings["development_end"],
            ),
            "settings": settings,
            "inputs": w_sha,
            "extended_data_report": {
                **w_rep,
                "company_facts_issuers_loaded": issuers_loaded,
                "company_facts_rows_kept": dict(sorted(facts_counts.items())),
            },
            "stage1_coverage": bundle.coverage_report,
            "stage2_news_report": cat_rep,
            "two_stage": evaluation,
            "process_memory_after_scoring": _process_memory_gib(),
        })

        staging = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        staging.mkdir(parents=True)
        try:
            (staging / "scoreboard.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
            )
            (staging / "scoreboard.md").write_text(_render_two_stage_markdown(document), encoding="utf-8")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        append_trial(
            destination.parent / TRIAL_LOG_NAME,
            root=root,
            command=WIDER_BETA_NEUTRAL_COMMAND,
            output=identity.relative_to(root),
            input_sha256=w_sha,
            settings=settings,
            scoreboard=document,
        )
    return {
        "output": identity.relative_to(root).as_posix(),
        "subset": subset,
        "inputs": w_sha,
        "acceptance_summary": evaluation["acceptance_summary"],
        "process_memory_after_scoring": document["process_memory_after_scoring"],
    }
