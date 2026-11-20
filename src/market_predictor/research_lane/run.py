"""Score frozen research-lane predictor sets on development history under the heavy-job lease."""
from __future__ import annotations

import json
import math
import os
import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pyarrow.parquet as pq

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
    reversal,
    two_stage,
    wider_universe,
)
from market_predictor.research_lane.barrier_labels import (
    BARRIER_HORIZON_SESSIONS,
    DEFAULT_STOP_MULTIPLE,
    DEFAULT_TARGET_MULTIPLE,
    FROZEN_ROUND_TRIP_COST,
    INFO_ROUND_TRIP_COST,
)
from market_predictor.research_lane.company_facts import (
    COMPANY_FACTS_DIRECTORY,
    COMPANY_FACTS_FILE,
    COMPANY_FACTS_MANIFEST,
    COMPANY_FACTS_SOURCES,
    FLOW_TAGS,
    INSTANT_TAGS,
    MISSING_REASONS,
    IssuerFacts,
    read_company_facts,
)
from market_predictor.research_lane.fitted import (
    LEARNERS,
    RIDGE_ALPHA,
    SAMPLE_EVERY_SESSIONS,
    SCORED_YEARS,
    XGBOOST_PARAMETERS,
    XGBOOST_SEEDS,
    scored_folds,
    session_years,
    target_ranks,
    walk_forward,
)
from market_predictor.research_lane.indicators import (
    INDICATOR_NAMES,
    SECTOR_MINIMUM_STOCKS,
    build_feature_sets,
    compute_indicators,
)
from market_predictor.research_lane.labels import FIRST_DECISION_DATE, HORIZONS, first_decision_row, research_labels
from market_predictor.research_lane.meta_model import (
    CLASSIFIER_PROB_THRESHOLD,
    META_SEEDS,
    RIDGE_META_ALPHA,
    STEP3A_READING_RULE,
    TOP_QUANTILE_CUTOFF,
    XGB_CLASSIFIER_PARAMS,
    XGB_REGRESSOR_PARAMS,
    extract_candidate_table,
    run_walk_forward_meta_models,
    summarize_trades,
)
from market_predictor.research_lane.news_catalysts import (
    EARLY_NEWS_FAMILY_DIR,
    LATER_NEWS_FAMILY_DIR,
    load_catalyst_matrix,
)
from market_predictor.research_lane.panel import (
    PANEL_DIRECTORY,
    ResearchPanel,
    check_development_end,
    file_sha256,
    load_research_panel,
)
from market_predictor.research_lane.predictors import (
    COMPOSITE_GROUPS,
    EXPECTED_SIGNS,
    PREDICTOR_NAMES,
    compute_composites,
    compute_predictors,
    minimum_present,
)
from market_predictor.research_lane.scoreboard import (
    COST_PER_TURNOVER,
    MINIMUM_STOCKS,
    TOP_FRACTION,
    build_scoreboard,
    score_predictor,
)
from market_predictor.research_lane.sec_filings import (
    FILINGS_DIRECTORY,
    FILINGS_FILE,
    FILINGS_MANIFEST,
    earnings_releases,
    read_sec_filings,
    security_issuers,
)
from market_predictor.research_lane.setups import (
    PRIMARY_MIN_ADR_PCT,
    SETUP_FEATURE_NAMES,
    STRICT_MIN_ADR_PCT,
    VolatilityMode,
    build_setups_and_features,
)
from market_predictor.research_lane.trial_log import TRIAL_LOG_NAME, append_trial
from market_predictor.research_lane.unadjusted_prices import UNADJUSTED_DIRECTORY, load_unadjusted_close

COMMAND = "score-price-predictors"
FUNDAMENTAL_COMMAND = "score-fundamental-predictors"
TECHNICAL_COMMAND = "score-technical-models"
REVERSAL_COMMAND = "score-short-reversal"
SWING_SETUPS_COMMAND = "score-swing-setups"
BENCHMARK_TILT_COMMAND = "score-benchmark-tilt"
TWO_STAGE_COMMAND = "score-two-stage-models"
WIDER_TWO_STAGE_COMMAND = "score-wider-two-stage-models"
WIDER_BETA_NEUTRAL_COMMAND = "score-wider-beta-neutral-models"
CORE_SATELLITE_COMMAND = core_satellite.CORE_SATELLITE_COMMAND
OUTPUT_PARENT = Path("data/research/research_lane")
LABEL_NOTICE = ("research label, adjusted prices, development history {first} to {end}, "
                "not proof of outperformance")
FloatArray = npt.NDArray[np.float64]


def _process_memory_gib() -> dict[str, float] | None:
    snapshot = process_memory_snapshot()
    if snapshot is None:
        return None
    current, peak = snapshot
    return {"current_gib": current / 2**30, "peak_gib": peak / 2**30}


def default_output(now: datetime | None = None, prefix: str = "price_predictors") -> Path:
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


def render_markdown(scoreboard: dict[str, Any], settings: dict[str, Any], load_report: dict[str, Any], *,
                    title: str = "Price predictor scoreboard", reference: str = "composite_all",
                    extra: list[str] | None = None) -> str:
    notice = LABEL_NOTICE.format(first=settings["first_decision_date"], end=settings["development_end"])
    horizons = [str(h) for h in settings["horizons_sessions"]]
    lines = [
        f"# {title}", "",
        f"Every figure below: {notice}.", "",
        "Universe: point-in-time S&P 500 member rows whose technical features were eligible on the decision date "
        f"(`{settings['eligibility_column']}`). Decision after the close of session t; entry at the adjusted open of "
        "t+1; exit at the adjusted close of t+h sessions. Returns are shown as percent of entry price.", "",
        f"A date is scored only when at least {MINIMUM_STOCKS} eligible stocks have both a predictor value and a known "
        "label. Rank correlation is the Spearman coefficient between the signed predictor and the forward return "
        "across stocks on one date (range -1 to +1), averaged over dates. The t-statistic uses a Newey-West "
        "standard error with lag equal to the horizon in sessions, because overlapping holding windows make "
        "neighbouring dates dependent.", "",
        "## Label coverage", "",
        "| Horizon (sessions) | Eligible decisions (stock-sessions) | Known label (stock-sessions) | Known share | "
        "Window past end | Entry bar missing | Exit bar missing |",
        "|---|---|---|---|---|---|---|",
    ]
    for horizon in horizons:
        facts = scoreboard["universe"][horizon]
        missing = facts["missing_label_stock_sessions_by_reason"]
        lines.append(
            f"| {horizon} | {facts['eligible_decisions_stock_sessions']} | {facts['known_label_stock_sessions']} | "
            f"{_percent(facts['known_label_share'])} | {missing['window_runs_past_development_end']} | "
            f"{missing['entry_bar_missing']} | {missing['exit_bar_missing']} |")
    lines += ["", "## Mean daily rank correlation (Newey-West t-statistic in brackets)", "",
              "| Predictor (sign) | " + " | ".join(f"{h} sessions" for h in horizons) + " |",
              "|---|" + "---|" * len(horizons)]
    for name, entries in scoreboard["predictors"].items():
        sign = settings["expected_signs"].get(name, 1)
        sign_text = "unsigned" if settings.get("predictors_unsigned") else ("+" if sign > 0 else "-")
        cells = [f"{_number(entries[h]['rank_correlation']['mean'])} [{_number(entries[h]['rank_correlation']['t'], 2)}]"
                 for h in horizons]
        lines.append(f"| {name} ({sign_text}) | " + " | ".join(cells) + " |")
    lines += ["", "Scored dates per horizon (dates with at least the minimum stock count): " + ", ".join(
        f"{h} sessions: {scoreboard['predictors'][reference][h]['rank_correlation']['count']} dates"
        for h in horizons) + f" ({reference}; single predictors may differ).", "",
        "## Cross-checks: non-overlapping dates, sector-neutral, share of positive years", "",
        "| Predictor | Horizon | Non-overlapping mean [t] (dates) | Sector-neutral mean [t] | Positive years / years |",
        "|---|---|---|---|---|"]
    for name, entries in scoreboard["predictors"].items():
        for h in horizons:
            entry = entries[h]
            non = entry["rank_correlation"]["non_overlapping"]
            neutral = entry["sector_neutral_rank_correlation"]
            share = entry["share_of_years_positive"]
            positive = "n/a" if share is None else f"{round(share * entry['years_scored'])} / {entry['years_scored']}"
            lines.append(f"| {name} | {h} | {_number(non['mean'])} [{_number(non['t'], 2)}] ({non['count']}) | "
                         f"{_number(neutral['mean'])} [{_number(neutral['t'], 2)}] | {positive} |")
    lines += ["", "## Top tenth minus all scored stocks, non-overlapping rebalances", "",
              f"Each rebalance holds the top {int(TOP_FRACTION * 100)}% of scored stocks by signed predictor for h "
              "sessions. Spread = mean research-label return of the top tenth minus the mean of all scored stocks "
              f"on that date, per holding period. Annualized = per-period mean x 252 / h (arithmetic). Net subtracts "
              f"{COST_PER_TURNOVER * 100:.2f}% x mean turnover per period; turnover is the fraction of top-tenth names "
              "not held at the previous rebalance.", "",
              "| Predictor | Horizon | Periods | Spread per period [t] | Annualized | Turnover | Net annualized |",
              "|---|---|---|---|---|---|---|"]
    for name, entries in scoreboard["predictors"].items():
        for h in horizons:
            top = entries[h]["top_tenth"]
            lines.append(
                f"| {name} | {h} | {top['rebalance_periods']} | {_percent(top['mean_spread_per_period_fraction'])} "
                f"[{_number(top['spread_t'], 2)}] | {_percent(top['annualized_spread_fraction'])} | "
                f"{_percent(top['mean_turnover_fraction'])} | {_percent(top['net_annualized_spread_fraction'])} |")
    lines += ["", "## Average eligible member versus the index", "",
              "Per decision date: equal-weighted mean research-label return of eligible stocks with a known label, "
              "minus the SPY (or QQQ) return over the same open-to-close interval; averaged over decision dates.", "",
              "| Horizon | Dates | Minus SPY per period [t] | Minus QQQ per period [t] |", "|---|---|---|---|"]
    for horizon in horizons:
        facts = scoreboard["universe"][horizon]
        spy, qqq = facts["average_stock_minus_spy"], facts["average_stock_minus_qqq"]
        lines.append(f"| {horizon} | {facts['decision_dates_with_average']} | {_percent(spy['mean'])} "
                     f"[{_number(spy['t'], 2)}] | {_percent(qqq['mean'])} [{_number(qqq['t'], 2)}] |")
    lines += ["", "By calendar year (mean per period, percent):", "",
              "| Horizon | Year | Minus SPY | Minus QQQ | Dates |", "|---|---|---|---|---|"]
    for horizon in horizons:
        facts = scoreboard["universe"][horizon]
        for year, item in facts["average_stock_minus_spy"]["by_year"].items():
            qqq_item = facts["average_stock_minus_qqq"]["by_year"].get(year, {})
            lines.append(f"| {horizon} | {year} | {_percent(item['mean'])} | {_percent(qqq_item.get('mean'))} | "
                         f"{item['dates']} |")
    lines += extra or []
    lines += ["", "## Panel load facts", "", "```json", json.dumps(_json_safe(load_report), indent=2, sort_keys=True),
              "```", ""]
    return "\n".join(lines)


@dataclass(frozen=True)
class PredictorSet:
    """Signed predictors plus the settings, input hashes and report sections specific to one command."""

    signed: dict[str, FloatArray]
    settings: dict[str, Any]
    input_sha256: dict[str, str] = field(default_factory=dict)
    sections: dict[str, Any] = field(default_factory=dict)
    markdown: list[str] = field(default_factory=list)
    # Optional: extra sections and markdown (placed before ``markdown``) computed from the finished scoreboard.
    summarize: Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]] | None = None


Builder = Callable[[ResearchPanel, EvidencePaths, int], PredictorSet]


def _score(*, command: str, title: str, reference: str, output_prefix: str, build: Builder, root: Path,
           output: Path | None, development_end: date, data_root: Path | None, panel: Path,
           unseal_final_test_period: bool, horizons: tuple[int, ...] = HORIZONS) -> dict[str, Any]:
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    output = output or default_output(prefix=output_prefix)
    identity = paths.identity(output)
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease(command, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("research lane preparation")
        destination = paths.write(identity)
        if destination.exists():
            raise FileExistsError(f"research lane output already exists: {destination}")
        research_panel = load_research_panel(paths.directory(panel), development_end=development_end,
                                             unseal_final_test_period=unseal_final_test_period)
        guard("research lane panel loaded")
        first_row = first_decision_row(research_panel)
        if first_row >= len(research_panel.sessions):
            raise ValueError(f"no session on or after the first decision date {FIRST_DECISION_DATE}")
        predictor_set = build(research_panel, paths, first_row)
        guard("research lane predictors computed")
        labels = [research_labels(research_panel, horizon) for horizon in horizons]
        scoreboard = build_scoreboard(predictor_set.signed, labels, eligible=research_panel.eligible,
                                      sessions=research_panel.sessions, sector=research_panel.sector,
                                      first_row=first_row)
        guard("research lane scoreboard computed")
        sections = dict(predictor_set.sections)
        markdown = list(predictor_set.markdown)
        if predictor_set.summarize is not None:
            extra_sections, extra_markdown = predictor_set.summarize(scoreboard)
            sections.update(extra_sections)
            markdown = extra_markdown + markdown
        settings = {
            "development_end": development_end.isoformat(),
            "unseal_final_test_period": unseal_final_test_period,
            "first_decision_date": FIRST_DECISION_DATE.isoformat(),
            "first_decision_session": str(research_panel.sessions[first_row]),
            "horizons_sessions": list(horizons),
            "panel": paths.relative(panel).as_posix(),
            "eligibility_column": research_panel.load_report["eligibility_column"],
            **predictor_set.settings,
            "minimum_stocks_per_date": MINIMUM_STOCKS,
            "top_fraction": TOP_FRACTION,
            "cost_per_unit_turnover_fraction": COST_PER_TURNOVER,
            "label": "close[t+h] / open[t+1] - 1, provider split- and dividend-adjusted SIP daily bars",
        }
        input_sha256 = {**research_panel.input_sha256, **predictor_set.input_sha256}
        document = _json_safe({
            "notice": LABEL_NOTICE.format(first=settings["first_decision_date"], end=settings["development_end"]),
            "units": {"returns": "fraction (0.01 = 1 percent)", "rank_correlation": "Spearman coefficient, -1..+1",
                      "counts": "stocks, dates or stock-sessions as named in each field"},
            "settings": settings, "inputs": input_sha256, "panel_load": research_panel.load_report,
            **sections,
            "process_memory_after_scoring": _process_memory_gib(),
            **scoreboard,
        })
        staging = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        staging.mkdir(parents=True)
        try:
            (staging / "scoreboard.json").write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False),
                                                     encoding="utf-8")
            (staging / "scoreboard.md").write_text(render_markdown(
                document, settings, document["panel_load"], title=title, reference=reference,
                extra=markdown), encoding="utf-8")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        append_trial(destination.parent / TRIAL_LOG_NAME, root=root, command=command, output=identity.relative_to(root),
                     input_sha256=input_sha256, settings=settings, scoreboard=document)
    return {"output": identity.relative_to(root).as_posix(), "inputs": input_sha256,
            "process_memory_after_scoring": document["process_memory_after_scoring"]}


def _price_predictor_set(research_panel: ResearchPanel, paths: EvidencePaths, first_row: int) -> PredictorSet:
    raw = compute_predictors(research_panel)
    composites = compute_composites(raw, research_panel.eligible)
    signed = {name: EXPECTED_SIGNS[name] * raw[name] for name in PREDICTOR_NAMES}
    signed.update(composites)
    del raw
    return PredictorSet(signed=signed, settings={
        "expected_signs": dict(EXPECTED_SIGNS),
        "composite_groups": {name: list(members) for name, members in COMPOSITE_GROUPS.items()},
        "composite_minimum_present": {name: minimum_present(len(m)) for name, m in COMPOSITE_GROUPS.items()},
    })


def score_price_predictors(*, root: Path, output: Path | None = None, development_end: date, data_root: Path | None = None,
                           panel: Path = PANEL_DIRECTORY, unseal_final_test_period: bool = False) -> dict[str, Any]:
    """Load development history, compute the frozen predictors and write scoreboard.json/.md plus a trial line."""
    return _score(command=COMMAND, title="Price predictor scoreboard", reference="composite_all",
                  output_prefix="price_predictors", build=_price_predictor_set, root=root, output=output,
                  development_end=development_end, data_root=data_root, panel=panel,
                  unseal_final_test_period=unseal_final_test_period)


@dataclass
class _CompanyFactsReader:
    """Reads one issuer at a time (rule 2 applied while parsing); nothing is cached between issuers."""

    sources: Path
    development_end: date
    counts: Counter[str] = field(default_factory=Counter)
    issuers_read: int = 0

    def path(self, cik: str) -> Path:
        return self.sources / cik / COMPANY_FACTS_FILE

    def __call__(self, cik: str) -> IssuerFacts | None:
        path = self.path(cik)
        if not path.is_file():
            return None
        facts, counts = read_company_facts(path, self.development_end)
        self.counts.update(counts)
        self.issuers_read += 1
        if self.issuers_read % 25 == 0:
            guard("research lane company facts")
        return facts


def security_issuer_map(security_ids: tuple[str, ...], recorded: Mapping[str, tuple[str, ...]],
                        has_facts: Callable[[str], bool]) -> tuple[dict[str, str], dict[str, Any]]:
    """CIK per panel security: the CIK in a ``cik:`` security id, otherwise the single CIK recorded for the
    security's filings by the SEC form inventory. No other mapping is attempted."""
    mapping: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for security in security_ids:
        kind = security.split(":", 1)[0]
        if kind == "cik":
            cik, source = security.split(":", 1)[1], "cik_security_id"
        else:
            ciks = recorded.get(security, ())
            if len(ciks) != 1:
                counts[f"{kind}:no_single_recorded_cik"] += 1
                continue
            cik, source = ciks[0], "filing_inventory_recorded_cik"
        if not has_facts(cik):
            counts[f"{kind}:{source}:no_company_facts_file"] += 1
            continue
        counts[f"{kind}:{source}:mapped"] += 1
        mapping[security] = cik
    return mapping, {"panel_securities": len(security_ids), "mapped_securities": len(mapping),
                     "by_id_kind_and_source": dict(sorted(counts.items()))}


def _coverage_markdown(coverage: Mapping[str, Any], sections: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Coverage of the step 2a inputs", "",
             "Coverage = share of eligible decision stock-sessions (eligible member rows on or after the first "
             "decision date) that have a value. Securities = panel securities with a value on at least one of those "
             "stock-sessions.", "",
             "| Input | Stock-sessions with value | Eligible decision stock-sessions | Coverage | Securities | "
             "Missing by reason (stock-sessions) |", "|---|---|---|---|---|---|"]
    for name, entry in coverage.items():
        if name.startswith("_"):
            continue
        reasons = ", ".join(f"{key}: {value}" for key, value in entry.get("missing_stock_sessions_by_reason", {}).items())
        lines.append(f"| {name} | {entry['stock_sessions_with_value']} | {entry['eligible_decision_stock_sessions']} | "
                     f"{_percent(entry['coverage_share'])} | {entry['securities_with_any_value']} | {reasons or '-'} |")
    lines += ["", "```json", json.dumps(_json_safe({key: value for key, value in sections.items()
                                                     if key != "fundamental_input_coverage"}),
                                         indent=2, sort_keys=True), "```"]
    return lines


def _fundamental_predictor_set(research_panel: ResearchPanel, paths: EvidencePaths, first_row: int) -> PredictorSet:
    development_end = research_panel.development_end
    filings_path = paths.read(FILINGS_DIRECTORY / FILINGS_FILE)
    filings_manifest = paths.read(FILINGS_DIRECTORY / FILINGS_MANIFEST)
    facts_manifest = paths.read(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_MANIFEST)
    unadjusted_directory = paths.directory(UNADJUSTED_DIRECTORY)
    input_sha256 = {"sec_filings_parquet_sha256": file_sha256(filings_path),
                    "sec_filings_manifest_sha256": file_sha256(filings_manifest),
                    "company_facts_final_manifest_sha256": file_sha256(facts_manifest)}
    filings = read_sec_filings(filings_path, development_end)
    filings_report: dict[str, Any] = {"rows_read": len(filings),
                                      "first_acceptance_utc": str(filings["accepted_at_utc"].min()),
                                      "last_acceptance_utc": str(filings["accepted_at_utc"].max())}
    releases, release_report = earnings_releases(filings)
    recorded = security_issuers(filings)
    del filings
    reader = _CompanyFactsReader(paths.directory(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES), development_end)
    mapping, identity_report = security_issuer_map(research_panel.security_ids, recorded,
                                                   lambda cik: reader.path(cik).is_file())
    unadjusted, unadjusted_report, unadjusted_sha = load_unadjusted_close(unadjusted_directory, research_panel)
    input_sha256.update(unadjusted_sha)
    guard("research lane step 2a inputs read")
    result = fundamentals.compute_fundamental_predictors(fundamentals.FundamentalInputs(
        panel=research_panel, unadjusted_close=unadjusted, releases=releases, security_issuer=mapping,
        load_issuer=reader))
    del unadjusted
    guard("research lane step 2a inputs computed")
    composites = fundamentals.compute_composites(result.values, research_panel.eligible)
    decision_rows = np.arange(len(research_panel.sessions)) >= first_row
    coverage = fundamentals.coverage(result, composites, research_panel.eligible, decision_rows)
    signed = {name: fundamentals.EXPECTED_SIGNS[name] * result.values[name] for name in fundamentals.PREDICTOR_NAMES}
    signed.update(composites)
    sections = {
        "fundamental_input_coverage": coverage,
        "fundamental_inputs_report": {
            "sec_filings": {**filings_report, **release_report},
            "security_issuer_mapping": identity_report,
            "company_facts_rows_kept_filed_on_or_before_development_end": dict(sorted(reader.counts.items())),
            "company_facts_issuers_read": reader.issuers_read,
            "unadjusted_close": unadjusted_report,
            **result.report,
        },
    }
    settings = {
        "expected_signs": dict(fundamentals.EXPECTED_SIGNS),
        "composite_groups": {name: list(m) for name, m in fundamentals.COMPOSITE_GROUPS.items()},
        "composite_minimum_present": {name: minimum_present(len(m)) for name, m in fundamentals.COMPOSITE_GROUPS.items()},
        "sec_filings": paths.relative(filings_path).as_posix(),
        "company_facts": paths.relative(COMPANY_FACTS_DIRECTORY / COMPANY_FACTS_SOURCES).as_posix(),
        "unadjusted_close": paths.relative(UNADJUSTED_DIRECTORY).as_posix(),
        "flow_tags": {item: list(tags) for item, tags in FLOW_TAGS.items()},
        "instant_tags": {item: list(tags) for item, tags in INSTANT_TAGS.items()},
        "missing_reasons": {str(code): name for code, name in MISSING_REASONS.items()},
        "availability": ("company fact usable from the first session after its filing date; earliest filing per "
                         "period kept; release event session = first session whose 16:00 New York close is at or "
                         "after the inventory availability time; reaction usable from e+1 through e+63; market value "
                         "= unadjusted close x share count; fundamentals expire 200 days after the latest period end"),
    }
    return PredictorSet(signed=signed, settings=settings, input_sha256=input_sha256, sections=sections,
                        markdown=_coverage_markdown(coverage, sections["fundamental_inputs_report"]))


def score_fundamental_predictors(*, root: Path, output: Path | None = None, development_end: date,
                                 data_root: Path | None = None, panel: Path = PANEL_DIRECTORY,
                                 unseal_final_test_period: bool = False) -> dict[str, Any]:
    """Score the frozen earnings-reaction and financial-statement inputs on development history (no model fit)."""
    return _score(command=FUNDAMENTAL_COMMAND, title="Earnings and financial-statement input scoreboard",
                  reference="composite_fundamental_all", output_prefix="fundamental_predictors",
                  build=_fundamental_predictor_set, root=root, output=output, development_end=development_end,
                  data_root=data_root, panel=panel, unseal_final_test_period=unseal_final_test_period)


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def indicator_summary(scoreboard: Mapping[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """Per horizon: indicators with |Newey-West t| >= 2 versus 5% expected by chance, and the ten largest |t|."""
    result: dict[str, Any] = {}
    for horizon in (str(h) for h in HORIZONS):
        rows = [(name, scoreboard["predictors"][name][horizon]["rank_correlation"]) for name in names]
        with_t = [(name, entry) for name, entry in rows if entry["t"] is not None]
        largest = sorted(with_t, key=lambda item: -abs(item[1]["t"]))[:10]
        result[horizon] = {
            "indicators_scored": len(with_t),
            "indicators_with_abs_t_at_least_2": sum(1 for _, entry in with_t if abs(entry["t"]) >= 2.0),
            "expected_by_chance_5_percent": 0.05 * len(with_t),
            "ten_largest_abs_t_picked_after_looking": [
                {"indicator": name, "mean_rank_correlation": entry["mean"], "newey_west_t": entry["t"]}
                for name, entry in largest],
        }
    return result


def _indicator_markdown(summary: Mapping[str, Any]) -> list[str]:
    lines = ["", "## Section C: each new indicator alone (unsigned centered rank)", "",
             "Count of indicators with |Newey-West t| >= 2 per horizon, against the number expected by chance if no "
             "indicator had any ordering ability (5% of the indicators scored). The ten largest |t| are picked after "
             "looking at all results, so they overstate what a fresh sample would show.", "",
             "| Horizon (sessions) | Indicators scored | abs(t) >= 2 | Expected by chance |", "|---|---|---|---|"]
    for horizon, entry in summary.items():
        lines.append(f"| {horizon} | {entry['indicators_scored']} | {entry['indicators_with_abs_t_at_least_2']} | "
                     f"{entry['expected_by_chance_5_percent']:.2f} |")
    for horizon, entry in summary.items():
        lines += ["", f"Ten largest |t| at {horizon} sessions (picked after looking): " + ", ".join(
            f"{item['indicator']} {_number(item['mean_rank_correlation'])} [{_number(item['newey_west_t'], 2)}]"
            for item in entry["ten_largest_abs_t_picked_after_looking"])]
    return lines


def fitted_summary(specifications: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Median/min/max out-of-sample mean rank correlation and year agreement across specifications."""
    means = {key: spec["rank_correlation_mean"] for key, spec in specifications.items()
             if spec["rank_correlation_mean"] is not None}
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


READING_RULE = ("With about 0.03 uncertainty per out-of-sample rank correlation, the best of 24 would reach roughly "
                "0.05 by chance; the conclusion rests on the median and on agreement across learners, horizons and "
                "years. Technical features solve the problem only if the median is at or above 0.03 and most "
                "specifications are positive in at least 3 of 4 scored years.")


def _fitted_markdown(section: Mapping[str, Any]) -> list[str]:
    years = [str(year) for year in SCORED_YEARS]
    lines = ["", "## Section D: walk-forward fitted combinations (out-of-sample scored years only)", "",
             "Each scored year is predicted by a model trained only on sampled decision dates (every 5th session from "
             "2019-07-09) whose label window ends before the first session of that year. Scores are research-label "
             "rank correlations of the out-of-sample prediction with the forward return, over decision dates from "
             "the first session of 2021 to the development end. Spread columns: top tenth minus all scored stocks, "
             "non-overlapping holding periods, percent per period.", "",
             "| Feature set | Learner | Horizon | Mean rank corr [NW t] | Dates | " + " | ".join(years)
             + " | Positive years | Sector-neutral [t] | Top-tenth spread [t] | Turnover | Net spread |",
             "|---|---|---|---|---|" + "---|" * len(years) + "---|---|---|---|---|"]
    for spec in section["specifications"].values():
        score = spec["score"]
        rank, top = score["rank_correlation"], score["top_tenth"]
        neutral = score["sector_neutral_rank_correlation"]
        by_year = score["rank_correlation_by_year"]
        lines.append(
            f"| {spec['feature_set']} | {spec['learner']} | {spec['horizon_sessions']} | {_number(rank['mean'])} "
            f"[{_number(rank['t'], 2)}] | {rank['count']} | "
            + " | ".join(_number(by_year.get(year, {}).get("mean")) for year in years)
            + f" | {spec['positive_years']} / {score['years_scored']} | {_number(neutral['mean'])} "
            f"[{_number(neutral['t'], 2)}] | {_percent(top['mean_spread_per_period_fraction'])} "
            f"[{_number(top['spread_t'], 2)}] | {_percent(top['mean_turnover_fraction'])} | "
            f"{_percent(top['net_mean_spread_per_period_fraction'])} |")
    summary = section["summary"]
    lines += ["", "### Summary across the 24 specifications", "",
              f"- Median mean rank correlation: {_number(summary['median_mean_rank_correlation'])}; minimum "
              f"{_number(summary['minimum_mean_rank_correlation'])}; maximum {_number(summary['maximum_mean_rank_correlation'])}.",
              f"- Specifications positive in at least 3 of 4 scored years: "
              f"{summary['specifications_positive_in_at_least_3_of_4_years']} of {summary['specifications']}.",
              "- Median by learner: " + ", ".join(f"{k} {_number(v)}" for k, v in summary["median_by_learner"].items()),
              "- Median by feature set: " + ", ".join(f"{k} {_number(v)}" for k, v in summary["median_by_feature_set"].items()),
              "- Median by horizon (sessions): " + ", ".join(
                  f"{k} {_number(v)}" for k, v in summary["median_by_horizon_sessions"].items()),
              f"- Pre-registered reading rule: \"{summary['reading_rule']}\"",
              f"- Rule met: {'yes' if summary['rule_met'] else 'no'}.", "",
              "### Top 15 features by importance at 21 sessions (information only; mean over folds)", "",
              "Ridge: absolute coefficient on the centered rank. Trees: gain share, mean over seeds and folds.", ""]
    for key, items in section["top_features_21_sessions"].items():
        lines.append(f"- {key}: " + ", ".join(f"{item['feature']} {item['importance']:.4g}" for item in items))
    return lines


def _technical_model_set(research_panel: ResearchPanel, paths: EvidencePaths, first_row: int) -> PredictorSet:
    feature_sets = build_feature_sets(research_panel)
    features = feature_sets.features
    guard("research lane technical features computed")
    sessions, eligible = research_panel.sessions, research_panel.eligible
    folds = scored_folds(sessions)
    scored_start = folds[0].test_start
    decision_rows = np.arange(len(sessions)) >= scored_start
    years = session_years(sessions)
    specifications: dict[str, dict[str, Any]] = {}
    importance: dict[str, list[dict[str, Any]]] = {}
    for horizon in HORIZONS:
        labels = research_labels(research_panel, horizon)
        target = target_ranks(labels.stock, eligible)
        for set_name, names in feature_sets.sets.items():
            for learner in LEARNERS:
                result = walk_forward(features, names, target, eligible=eligible, sessions=sessions,
                                      first_row=first_row, horizon=horizon, learner=learner, folds=folds,
                                      checkpoint=guard)
                score = score_predictor(result.scores, labels, eligible=eligible, decision_rows=decision_rows,
                                        sector=research_panel.sector, years=years, first_row=scored_start)
                positive = sum(1 for item in score["rank_correlation_by_year"].values() if item["mean"] > 0)
                specifications[f"{set_name}/{learner}/{horizon}"] = {
                    "feature_set": set_name, "learner": learner, "horizon_sessions": horizon, "features": len(names),
                    "rank_correlation_mean": score["rank_correlation"]["mean"], "positive_years": positive,
                    "score": score, "folds": result.folds,
                }
                if horizon == 21:
                    order = np.argsort(-result.importance, kind="stable")[:15]
                    importance[f"{learner}/{set_name}"] = [
                        {"feature": names[index], "importance": float(result.importance[index])} for index in order]
                del result
        del labels, target
    section = {"specifications": specifications, "summary": fitted_summary(specifications),
               "top_features_21_sessions": importance,
               "scored_folds": [{"year": fold.year, "first_session": str(sessions[fold.test_start]),
                                 "last_session": str(sessions[fold.test_stop - 1])} for fold in folds]}
    signed = {name: features.pop(name).astype(np.float64) for name in INDICATOR_NAMES}
    features.clear()
    settings = {
        "expected_signs": {}, "predictors_unsigned": True,
        "indicator_names": list(INDICATOR_NAMES),
        "feature_sets": {name: len(members) for name, members in feature_sets.sets.items()},
        "feature_transform": "centered cross-sectional rank among eligible stocks (-0.5..+0.5)",
        "sector_neutral_transform": ("centered rank within the stock's sector among eligible stocks with a value; NaN "
                                     f"when fewer than {SECTOR_MINIMUM_STOCKS} such stocks"),
        "target": "centered rank of the research-label forward return among eligible stocks with a known label",
        "scored_years": list(SCORED_YEARS), "sample_every_sessions": SAMPLE_EVERY_SESSIONS,
        "training_purge": "training decision row t kept only when t + h < first session row of the scored year",
        "row_weights": "1 / rows on the date, rescaled by one constant so the mean weight is 1",
        "learners": {"ridge": {"alpha": RIDGE_ALPHA, "fit_intercept": True, "missing_feature": 0.0},
                     "xgboost": {**dict(XGBOOST_PARAMETERS), "seeds": list(XGBOOST_SEEDS), "missing_feature": "NaN"}},
        "fitted_score_first_decision_session": str(sessions[scored_start]),
    }
    return PredictorSet(signed=signed, settings=settings, sections={"technical_models": section},
                        markdown=_fitted_markdown(section),
                        summarize=lambda board: (
                            {"indicator_summary": indicator_summary(board, INDICATOR_NAMES)},
                            _indicator_markdown(indicator_summary(board, INDICATOR_NAMES))))


def score_technical_models(*, root: Path, output: Path | None = None, development_end: date,
                           data_root: Path | None = None, panel: Path = PANEL_DIRECTORY,
                           unseal_final_test_period: bool = False) -> dict[str, Any]:
    """Score ~50 more technical indicators alone, then walk-forward ridge and boosted-tree combinations."""
    return _score(command=TECHNICAL_COMMAND, title="Technical indicators and fitted combinations scoreboard",
                  reference="return_1d", output_prefix="technical_models", build=_technical_model_set, root=root,
                  output=output, development_end=development_end, data_root=data_root, panel=panel,
                  unseal_final_test_period=unseal_final_test_period)


def _reversal_set(research_panel: ResearchPanel, paths: EvidencePaths, first_row: int) -> PredictorSet:
    indicators = compute_indicators(research_panel)
    signed = reversal.reversal_signals(indicators, research_panel.eligible)
    del indicators

    def summarize(board: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        result = reversal.reading(board)
        return {"reversal_reading": result}, reversal.reading_markdown(result)

    return PredictorSet(signed=signed, settings={
        "expected_signs": {**{name: reversal.REVERSAL_SIGN for name in reversal.REVERSAL_INPUTS},
                           reversal.REVERSAL_COMPOSITE: 1},
        "composite_groups": {reversal.REVERSAL_COMPOSITE: list(reversal.REVERSAL_INPUTS)},
        "composite_minimum_present": {reversal.REVERSAL_COMPOSITE: reversal.REVERSAL_MINIMUM_PRESENT},
        "reading_rule": reversal.READING_RULE,
    }, summarize=summarize)


def score_short_reversal(*, root: Path, output: Path | None = None, development_end: date,
                         data_root: Path | None = None, panel: Path = PANEL_DIRECTORY,
                         unseal_final_test_period: bool = False) -> dict[str, Any]:
    """Score the frozen 1-3 session reversal inputs and composite and apply the pre-registered reading rule."""
    return _score(command=REVERSAL_COMMAND, title="Very-short-term reversal scoreboard",
                  reference=reversal.REVERSAL_COMPOSITE, output_prefix="short_reversal", build=_reversal_set,
                  root=root, output=output, development_end=development_end, data_root=data_root, panel=panel,
                  unseal_final_test_period=unseal_final_test_period, horizons=reversal.REVERSAL_HORIZONS)


def _render_swing_setups_markdown(document: Mapping[str, Any]) -> str:
    settings = document["settings"]
    sec = document["swing_setups"]
    notice = document["notice"]
    years = [str(y) for y in settings["scored_years"]]

    def r_fmt(val: Any) -> str:
        return "n/a" if val is None else f"{float(val):+.3f}R"

    lines = [
        f"# Event-driven swing setups & Stage-2 meta-labeling scoreboard ({settings['volatility_mode']})",
        "",
        f"Every figure below: {notice}.",
        "",
        "## Method & boundaries",
        "",
        f"- **Population & entry timing:** point-in-time S&P 500 eligible rows (`{settings['eligibility_column']}`) "
        f"from `{settings['first_decision_date']}` to `{settings['development_end']}`. Decision after session `t` "
        f"close; entry at `open[t+1]`; conservative `stop_first` daily-bar barrier resolution (`+{settings['target_atr_multiple']}x` "
        f"profit target, `-{settings['stop_atr_multiple']}x` stop-loss, `{settings['horizon_sessions']}`-session time stop).",
        f"- **Volatility estimator:** `{settings['volatility_mode']}`.",
        f"- **Costs:** `{settings['round_trip_cost_fraction'] * 100:.2f}%` round-trip per trade (`0.10%` informational); "
        "SPY matched-interval return is gross over the exact `[open[t+1], close[exit_row]]` window.",
        "",
        "## Section A: Full-development Stage-1 zero-ML base rates across ADR% screens (2019-07-09..end)",
        "",
        "| Screen / setup | Trades | Dates | Target % | Stop % | Timeout % | Mean hold | Net 20bps | Net R (20bps) | "
        "Matched SPY | Net excess vs SPY [NW t] |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label, entry in sec["stage1_full_period_screens"].items():
        nw_t = entry["daily_cohort_excess_spy_20bps_nw"]["t"]
        lines.append(
            f"| {label} | {entry['trades']} | {entry['entry_dates']} | {_percent(entry['target_hit_rate'])} | "
            f"{_percent(entry['stop_hit_rate'])} | {_percent(entry['timeout_rate'])} | "
            f"{_number(entry['mean_holding_sessions'], 2)} | {_percent(entry['mean_net_return_20bps'])} | "
            f"{r_fmt(entry['mean_net_r_20bps'])} | {_percent(entry['mean_spy_matched_return'])} | "
            f"{_percent(entry['mean_net_excess_spy_20bps'])} [{_number(nw_t, 2)}] |"
        )

    lines += [
        "",
        "## Section B: Out-of-sample walk-forward Stage-1 & Stage-2 specifications (2021..end)",
        "",
        "| Specification | Trades | Dates | Target % | Net 20bps | Net R | Net excess vs SPY [NW t] | "
        "Fixed-10d excess | " + " | ".join(f"{y} ex" for y in years) + " | Pos yrs | Passes |",
        "|---|---|---|---|---|---|---|---|" + "---|" * len(years) + "---|---|",
    ]
    for name, entry in sec["oos_specifications"].items():
        nw_t = entry["daily_cohort_excess_spy_20bps_nw"]["t"]
        yr_cells = [_percent(entry["by_year"].get(y, {}).get("mean_net_excess_spy_20bps")) for y in years]
        lines.append(
            f"| {name} | {entry['trades']} | {entry['entry_dates']} | {_percent(entry['target_hit_rate'])} | "
            f"{_percent(entry['mean_net_return_20bps'])} | {r_fmt(entry['mean_net_r_20bps'])} | "
            f"{_percent(entry['mean_net_excess_spy_20bps'])} [{_number(nw_t, 2)}] | "
            f"{_percent(entry['mean_fixed_10d_net_excess_spy_20bps'])} | "
            + " | ".join(yr_cells)
            + f" | {entry['positive_excess_years']}/{entry['years_with_trades']} | "
            f"{'yes' if entry['passes_reading_rule'] else 'no'} |"
        )

    lines += [
        "",
        "## Section C: Out-of-sample XGBClassifier probability calibration",
        "",
        "| Probability bin | Candidates | Mean predicted p | Empirical target rate | Mean net R (20bps) | Mean net excess vs SPY |",
        "|---|---|---|---|---|---|",
    ]
    for b in sec["oos_calibration_buckets"]:
        lines.append(
            f"| {b['bin']} | {b['candidates']} | {_number(b['mean_predicted_prob'], 3)} | "
            f"{_percent(b['empirical_target_hit_rate'])} | {r_fmt(b['mean_net_r_20bps'])} | "
            f"{_percent(b['mean_net_excess_spy_20bps'])} |"
        )

    lines += [
        "",
        "## Section D: Shock-regime stratification on OOS Stage-1 setups (Bucket A <= 3x MTR vs Bucket B > 3x MTR)",
        "",
        "| Regime bucket | Trades | Target % | Stop % | Timeout % | Net 20bps | Net R (20bps) | Net excess vs SPY [NW t] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for b_name, entry in sec["oos_shock_stratification"].items():
        nw_t = entry["daily_cohort_excess_spy_20bps_nw"]["t"]
        lines.append(
            f"| {b_name} | {entry['trades']} | {_percent(entry['target_hit_rate'])} | "
            f"{_percent(entry['stop_hit_rate'])} | {_percent(entry['timeout_rate'])} | "
            f"{_percent(entry['mean_net_return_20bps'])} | {r_fmt(entry['mean_net_r_20bps'])} | "
            f"{_percent(entry['mean_net_excess_spy_20bps'])} [{_number(nw_t, 2)}] |"
        )

    lines += [
        "",
        "## Section E: Top 12 features by walk-forward importance",
        "",
        "- **XGBClassifier (+2.5R/-1.0R):** "
        + ", ".join(f"{item['feature']} {item['importance']:.4g}" for item in sec["classifier_top_features"]),
        "- **XGBRegressor (vol-scaled SPY excess):** "
        + ", ".join(f"{item['feature']} {item['importance']:.4g}" for item in sec["regressor_top_features"]),
        "",
        "## Pre-registered reading rule",
        "",
        sec["reading_summary"]["rule"],
        "",
        f"Passing specifications: `{sec['reading_summary']['passing_specifications']}`.",
        (
            "Verdict: **earns one sealed-period test**."
            if sec["reading_summary"]["earns_sealed_period_test"]
            else "Verdict: **no specification passed**."
        ),
        "",
    ]
    return "\n".join(lines)


def score_swing_setups(
    *,
    root: Path,
    output: Path | None = None,
    development_end: date,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    filings_dir: Path = FILINGS_DIRECTORY,
    volatility_mode: VolatilityMode = "wilder_atr_14",
    unseal_final_test_period: bool = False,
) -> dict[str, Any]:
    """Run Stage-1 deterministic swing scanners and Stage-2 purged walk-forward meta-models."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    prefix = "swing_setups" if volatility_mode == "wilder_atr_14" else f"swing_setups_{volatility_mode}"
    output = output or default_output(prefix=prefix)
    identity = paths.identity(output)
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(SWING_SETUPS_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("research lane swing setups preparation")
        destination = paths.write(identity)
        if destination.exists():
            raise FileExistsError(f"research lane output already exists: {destination}")

        panel_dir = paths.directory(panel)
        research_panel = load_research_panel(
            panel_dir, development_end=development_end, unseal_final_test_period=unseal_final_test_period,
        )
        guard("research lane swing setups panel loaded")
        first_row = first_decision_row(research_panel)
        if first_row >= len(research_panel.sessions):
            raise ValueError(f"no session on or after the first decision date {FIRST_DECISION_DATE}")

        catalysts = load_catalyst_matrix(
            research_panel,
            panel_root=panel_dir,
            early_news_dir=paths.directory(early_news_dir),
            later_news_dir=paths.directory(later_news_dir),
            filings_dir=paths.directory(filings_dir),
            unseal_final_test_period=unseal_final_test_period,
        )
        guard("research lane swing setups catalysts loaded")

        detection = build_setups_and_features(
            research_panel, catalysts, first_row=first_row, volatility_mode=volatility_mode,
        )
        guard("research lane swing setups features computed")

        # Full development period (2019-07-09..development_end) zero-ML screen diagnostics.
        all_years = tuple(sorted(int(y) for y in np.unique(session_years(research_panel.sessions[first_row:]))))
        unfiltered_table = extract_candidate_table(research_panel, detection, detection.any_setup_unfiltered)
        primary_table = extract_candidate_table(research_panel, detection, detection.any_setup_primary)
        strict_table = extract_candidate_table(research_panel, detection, detection.any_setup_strict_adr)

        stage1_screens = {
            "unfiltered_sp500_any_setup": summarize_trades(
                unfiltered_table, np.ones(len(unfiltered_table.rows), dtype=bool),
                panel_sessions=research_panel.sessions, scored_years=all_years,
            ),
            "adr_ge_1_5pct_any_setup": summarize_trades(
                primary_table, np.ones(len(primary_table.rows), dtype=bool),
                panel_sessions=research_panel.sessions, scored_years=all_years,
            ),
            "adr_ge_2_5pct_any_setup": summarize_trades(
                strict_table, np.ones(len(strict_table.rows), dtype=bool),
                panel_sessions=research_panel.sessions, scored_years=all_years,
            ),
            "adr_ge_1_5pct_setup_a_catalyst": summarize_trades(
                primary_table, primary_table.setup_a,
                panel_sessions=research_panel.sessions, scored_years=all_years,
            ),
            "adr_ge_1_5pct_setup_b_pullback": summarize_trades(
                primary_table, primary_table.setup_b,
                panel_sessions=research_panel.sessions, scored_years=all_years,
            ),
            "adr_ge_1_5pct_setup_c_squeeze": summarize_trades(
                primary_table, primary_table.setup_c,
                panel_sessions=research_panel.sessions, scored_years=all_years,
            ),
        }
        del unfiltered_table, primary_table, strict_table
        guard("research lane swing setups stage1 diagnostics computed")

        _, meta = run_walk_forward_meta_models(
            research_panel, detection, first_row=first_row, checkpoint=guard,
        )
        guard("research lane swing setups meta-models scored")

        passing_specs = [name for name, spec in meta.specifications.items() if spec["passes_reading_rule"]]
        reading_summary = {
            "rule": STEP3A_READING_RULE,
            "passing_specifications": passing_specs,
            "earns_sealed_period_test": bool(passing_specs),
            "volatility_mode": volatility_mode,
        }

        settings: dict[str, Any] = {
            "development_end": development_end.isoformat(),
            "unseal_final_test_period": unseal_final_test_period,
            "first_decision_date": FIRST_DECISION_DATE.isoformat(),
            "first_decision_session": str(research_panel.sessions[first_row]),
            "panel": paths.relative(panel).as_posix(),
            "early_news_dir": paths.relative(early_news_dir).as_posix(),
            "later_news_dir": paths.relative(later_news_dir).as_posix(),
            "sec_filings_dir": paths.relative(filings_dir).as_posix(),
            "eligibility_column": research_panel.load_report["eligibility_column"],
            "volatility_mode": volatility_mode,
            "primary_min_adr_pct": PRIMARY_MIN_ADR_PCT,
            "strict_min_adr_pct": STRICT_MIN_ADR_PCT,
            "target_atr_multiple": DEFAULT_TARGET_MULTIPLE,
            "stop_atr_multiple": DEFAULT_STOP_MULTIPLE,
            "horizon_sessions": BARRIER_HORIZON_SESSIONS,
            "same_bar_resolution": "stop_first",
            "round_trip_cost_fraction": FROZEN_ROUND_TRIP_COST,
            "info_round_trip_cost_fraction": INFO_ROUND_TRIP_COST,
            "feature_names": list(SETUP_FEATURE_NAMES),
            "scored_years": [f["scored_year"] for f in meta.folds],
            "classifier_prob_threshold": CLASSIFIER_PROB_THRESHOLD,
            "classifier_top_quantile_cutoff": TOP_QUANTILE_CUTOFF,
            "xgb_classifier_params": dict(XGB_CLASSIFIER_PARAMS),
            "xgb_regressor_params": dict(XGB_REGRESSOR_PARAMS),
            "ridge_alpha": RIDGE_META_ALPHA,
            "seeds": list(META_SEEDS),
            "label": (
                "research label on provider split- and dividend-adjusted SIP daily bars; "
                "entry at open[t+1], conservative stop_first barrier resolution, matched-interval SPY comparison"
            ),
        }
        input_sha256 = {**research_panel.input_sha256, **catalysts.input_sha256}
        document = _json_safe({
            "notice": LABEL_NOTICE.format(first=settings["first_decision_date"], end=settings["development_end"]),
            "settings": settings,
            "inputs": input_sha256,
            "panel_load": research_panel.load_report,
            "catalyst_load": catalysts.report,
            "setup_detection": detection.report,
            "swing_setups": {
                "stage1_full_period_screens": stage1_screens,
                "walk_forward_folds": meta.folds,
                "oos_specifications": meta.specifications,
                "oos_calibration_buckets": meta.calibration_buckets,
                "oos_shock_stratification": meta.shock_stratification,
                "classifier_top_features": meta.classifier_importance,
                "regressor_top_features": meta.regressor_importance,
                "reading_summary": reading_summary,
            },
            "process_memory_after_scoring": _process_memory_gib(),
        })

        staging = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        staging.mkdir(parents=True)
        try:
            (staging / "scoreboard.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
            )
            (staging / "scoreboard.md").write_text(_render_swing_setups_markdown(document), encoding="utf-8")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        append_trial(
            destination.parent / TRIAL_LOG_NAME,
            root=root,
            command=SWING_SETUPS_COMMAND,
            output=identity.relative_to(root),
            input_sha256=input_sha256,
            settings=settings,
            scoreboard=document,
        )
    return {
        "output": identity.relative_to(root).as_posix(),
        "inputs": input_sha256,
        "reading_summary": reading_summary,
        "process_memory_after_scoring": document["process_memory_after_scoring"],
    }


@dataclass
class _BenchmarkTiltFactsReader:
    """Reads primary SEC company facts or falls back to retained `data/raw/investment_company_facts`."""

    primary_sources: Path
    extra_sources: Path | None
    development_end: date
    counts: Counter[str] = field(default_factory=Counter)
    issuers_read: int = 0
    extra_ciks_read: list[str] = field(default_factory=list)
    extra_sha256: dict[str, str] = field(default_factory=dict)

    def resolve_path(self, cik: str) -> tuple[Path | None, bool]:
        primary = self.primary_sources / cik / COMPANY_FACTS_FILE
        if primary.is_file():
            return primary, False
        if self.extra_sources is not None and (self.extra_sources / cik).is_dir():
            for child in sorted((self.extra_sources / cik).iterdir()):
                cand = child / COMPANY_FACTS_FILE
                if cand.is_file():
                    return cand, True
        return None, False

    def has_facts(self, cik: str) -> bool:
        path, _ = self.resolve_path(cik)
        return path is not None

    def __call__(self, cik: str) -> IssuerFacts | None:
        path, is_extra = self.resolve_path(cik)
        if path is None:
            return None
        if is_extra:
            self.extra_ciks_read.append(cik)
            self.extra_sha256[f"extra_company_facts_{cik}_sha256"] = file_sha256(path)
        facts, counts = read_company_facts(path, self.development_end)
        self.counts.update(counts)
        self.issuers_read += 1
        if self.issuers_read % 25 == 0:
            guard("research lane benchmark tilt company facts")
        return facts


def _render_benchmark_tilt_markdown(document: Mapping[str, Any]) -> str:
    settings = document["settings"]
    bt = document["benchmark_tilt"]
    notice = document["notice"]
    horizons = [str(h) for h in settings["horizons_sessions"]]
    years = ["2019", "2020", "2021", "2022", "2023", "2024"]

    lines = [
        "# Cap-weighted benchmark tilt & turnover-buffered reversal/quality scoreboard (Option C)",
        "",
        f"Every figure below: {notice}.",
        "",
        "## Method & boundaries",
        "",
        f"- **Population & holding clock:** point-in-time S&P 500 eligible rows (`{settings['eligibility_column']}`) "
        f"from `{settings['first_decision_date']}` to `{settings['development_end']}`. Rebalance decisions after "
        "session `r_k` close; trades execute at `open[r_k + 1]` and hold continuously to `open[r_{k+1} + 1]` "
        "with exact post-drift weights and one-way turnover `0.5 * sum(|w_new - w_drifted|)`.",
        f"- **Costs:** `{settings['frozen_cost_per_turnover'] * 100:.2f}%` per unit of one-way replaced weight "
        f"(`{settings['info_cost_per_turnover'] * 100:.2f}%` informational). Initial entry charges `0.5` one-way turn.",
        f"- **Benchmark-tilt bounds:** long-only (`w >= 0`), per-stock active deviation `<= +-2.0%` relative to "
        "point-in-time `unadjusted_close * shares` cap weight, exact sector neutrality (`0.0% <= +-5.0%`), "
        f"and `{settings['trade_deadband'] * 10000:.0f} bps` unchanged-state trade deadband.",
        "",
        "## Section A: Universe & passive benchmark alignment by horizon",
        "",
        "| Horizon | Periods | Tradable stocks | Cap-covered stocks | SPY compounded | Cap-subset compounded | "
        "Cap-subset turnover | Cap-subset vs SPY/yr [NW t] | Equal-weight vs SPY/yr [NW t] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for h in horizons:
        u = bt["universe_by_horizon"][h]
        cap_vs = u["cap_subset_minus_spy"]
        ew_vs = u["equal_weight_eligible_minus_spy"]
        lines.append(
            f"| {h} | {u['rebalance_periods']} | {_number(u['mean_tradable_eligible_stocks'], 1)} | "
            f"{_number(u['mean_cap_covered_stocks'], 1)} | {_percent(u['spy_compounded_return'])} | "
            f"{_percent(u['cap_subset_compounded_gross_return'])} | {_percent(u['cap_subset_mean_steady_turnover'])} | "
            f"{_percent(cap_vs['net_excess_20bps_annualized'])} [{_number(cap_vs['newey_west_20bps']['t'], 2)}] | "
            f"{_percent(ew_vs['net_excess_20bps_annualized'])} [{_number(ew_vs['newey_west_20bps']['t'], 2)}] |"
        )

    lines += [
        "",
        "## Section B: Skill report (equal-weight top-10% vs equal-weight scored universe)",
        "",
        "| Signal | Horizon | Policy | Turnover | Gross excess/period | Net 20bps/period [NW t] | "
        "Net 10bps/period | Net 20bps/yr | Pos yrs |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for sig_name, sig_horizons in bt["cells"].items():
        for h in horizons:
            for pol in ("skill_top10_unbuffered", "skill_top10_hold33"):
                entry = sig_horizons[h][pol]
                ex = entry["vs_matched_benchmark"]
                lines.append(
                    f"| {sig_name} | {h} | {pol} | {_percent(entry['mean_steady_one_way_turnover'])} | "
                    f"{_percent(ex['gross_excess_per_period'])} | {_percent(ex['net_excess_20bps_per_period'])} "
                    f"[{_number(ex['newey_west_20bps']['t'], 2)}] | {_percent(ex['net_excess_10bps_per_period'])} | "
                    f"{_percent(ex['net_excess_20bps_annualized'])} | "
                    f"{ex['positive_years_20bps']}/{ex['years_scored']} |"
                )

    lines += [
        "",
        "## Section C: Product report (cap-weighted tilts & 80% SPY + 20% satellite, net at 20bps)",
        "",
        "| Signal | H | Policy | Act share | Turnover | Max stk/sec | Net vs matched/yr [NW t] | "
        "Net vs SPY/yr [NW t] | Net 10bps vs SPY/yr | "
        + " | ".join(years)
        + " | SPY pos yrs | Passes |",
        "|---|---|---|---|---|---|---|---|---|" + "---|" * len(years) + "---|---|",
    ]
    for sig_name, sig_horizons in bt["cells"].items():
        for h in horizons:
            for pol in (
                "product_cap_mult_buffered",
                "product_sector_active_slice",
                "product_core_spy_80_sat_20",
            ):
                entry = sig_horizons[h][pol]
                ex_m = entry["vs_matched_benchmark"]
                ex_spy = entry["vs_spy"]
                yr_cells = [_percent(ex_spy["by_year_net_20bps"].get(y, {}).get("mean")) for y in years]
                max_bounds = f"{_percent(entry['max_stock_active_abs'])}/{_percent(entry['max_sector_active_abs'])}"
                lines.append(
                    f"| {sig_name} | {h} | {pol} | {_percent(entry['mean_active_share'])} | "
                    f"{_percent(entry['mean_steady_one_way_turnover'])} | {max_bounds} | "
                    f"{_percent(ex_m['net_excess_20bps_annualized'])} [{_number(ex_m['newey_west_20bps']['t'], 2)}] | "
                    f"{_percent(ex_spy['net_excess_20bps_annualized'])} [{_number(ex_spy['newey_west_20bps']['t'], 2)}] | "
                    f"{_percent(ex_spy['net_excess_10bps_annualized'])} | "
                    + " | ".join(yr_cells)
                    + f" | {ex_spy['positive_years_20bps']}/{ex_spy['years_scored']} | "
                    f"{'yes' if entry['passes_reading_rule'] else 'no'} |"
                )

    rs = bt["reading_summary"]
    lines += [
        "",
        "## Pre-registered reading rule",
        "",
        rs["rule"],
        "",
        f"- Passing specifications (`NW t >= 2.50`): `{rs['passing_specifications']}`.",
        f"- Passing strict `NW t >= 3.00`: `{rs['passing_strict_t_3_0_specifications']}`.",
        (
            "Verdict: **earns one sealed-period test**."
            if rs["earns_sealed_period_test"]
            else "Verdict: **no specification passed**."
        ),
        "",
    ]
    return "\n".join(lines)


def score_benchmark_tilt(
    *,
    root: Path,
    output: Path | None = None,
    development_end: date,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    filings_dir: Path = FILINGS_DIRECTORY,
    company_facts_dir: Path = COMPANY_FACTS_DIRECTORY,
    extra_company_facts_dir: Path | None = benchmark_tilt.EXTRA_COMPANY_FACTS_DIRECTORY,
    unadjusted_dir: Path = UNADJUSTED_DIRECTORY,
    horizons: tuple[int, ...] = benchmark_tilt.TILT_HORIZONS,
    unseal_final_test_period: bool = False,
) -> dict[str, Any]:
    """Evaluate Option C cap-weighted benchmark tilts and turnover-buffered skill baskets."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    output = output or default_output(prefix="benchmark_tilt")
    identity = paths.identity(output)
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(BENCHMARK_TILT_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("research lane benchmark tilt preparation")
        destination = paths.write(identity)
        if destination.exists():
            raise FileExistsError(f"research lane output already exists: {destination}")

        research_panel = load_research_panel(
            paths.directory(panel), development_end=development_end, unseal_final_test_period=unseal_final_test_period,
        )
        guard("research lane benchmark tilt panel loaded")
        first_row = first_decision_row(research_panel)
        if first_row >= len(research_panel.sessions):
            raise ValueError(f"no session on or after the first decision date {FIRST_DECISION_DATE}")

        filings_path = paths.read(filings_dir / FILINGS_FILE)
        filings_manifest = paths.read(filings_dir / FILINGS_MANIFEST)
        facts_manifest = paths.read(company_facts_dir / COMPANY_FACTS_MANIFEST)
        input_sha256: dict[str, str] = {
            **research_panel.input_sha256,
            "sec_filings_parquet_sha256": file_sha256(filings_path),
            "sec_filings_manifest_sha256": file_sha256(filings_manifest),
            "company_facts_final_manifest_sha256": file_sha256(facts_manifest),
        }

        filings = read_sec_filings(filings_path, development_end)
        recorded_ciks = security_issuers(filings)
        del filings

        extra_resolved: Path | None = None
        if extra_company_facts_dir is not None:
            cand_extra = paths.project_root / extra_company_facts_dir
            if data_root is not None and (data_root / extra_company_facts_dir).is_dir():
                cand_extra = data_root / extra_company_facts_dir
            if cand_extra.is_dir():
                extra_resolved = cand_extra

        reader = _BenchmarkTiltFactsReader(
            primary_sources=paths.directory(company_facts_dir / COMPANY_FACTS_SOURCES),
            extra_sources=extra_resolved,
            development_end=development_end,
        )
        mapping, identity_report = benchmark_tilt.map_security_ciks(
            research_panel.security_ids, recorded_ciks, reader.has_facts,
        )
        unadjusted, unadjusted_report, unadjusted_sha = load_unadjusted_close(
            paths.directory(unadjusted_dir), research_panel,
        )
        input_sha256.update(unadjusted_sha)
        guard("research lane benchmark tilt raw inputs read")

        qc = benchmark_tilt.build_quality_and_market_cap(research_panel, unadjusted, mapping, reader)
        del unadjusted
        input_sha256.update(reader.extra_sha256)
        guard("research lane benchmark tilt quality & market value computed")

        signals = benchmark_tilt.build_option_c_signals(research_panel, qc.quality_slow)
        guard("research lane benchmark tilt signals computed")

        evaluation = benchmark_tilt.evaluate_option_c(
            research_panel, signals, qc.market_value, first_row=first_row, horizons=horizons,
        )
        guard("research lane benchmark tilt evaluation completed")

        settings: dict[str, Any] = {
            "development_end": development_end.isoformat(),
            "unseal_final_test_period": unseal_final_test_period,
            "first_decision_date": FIRST_DECISION_DATE.isoformat(),
            "first_decision_session": str(research_panel.sessions[first_row]),
            "horizons_sessions": list(horizons),
            "signals": list(benchmark_tilt.SIGNAL_NAMES),
            "policies": list(benchmark_tilt.POLICY_NAMES),
            "panel": paths.relative(panel).as_posix(),
            "sec_filings": paths.relative(filings_path).as_posix(),
            "company_facts": paths.relative(company_facts_dir / COMPANY_FACTS_SOURCES).as_posix(),
            "extra_company_facts": (
                None if extra_company_facts_dir is None else extra_company_facts_dir.as_posix()
            ),
            "unadjusted_close": paths.relative(unadjusted_dir).as_posix(),
            "eligibility_column": research_panel.load_report["eligibility_column"],
            "buy_top_fraction": benchmark_tilt.BUY_TOP_FRACTION,
            "hold_top_fraction": benchmark_tilt.HOLD_TOP_FRACTION,
            "mult_enter_top_pct": benchmark_tilt.MULT_ENTER_TOP_PCT,
            "mult_exit_top_pct": benchmark_tilt.MULT_EXIT_TOP_PCT,
            "mult_enter_bot_pct": benchmark_tilt.MULT_ENTER_BOT_PCT,
            "mult_exit_bot_pct": benchmark_tilt.MULT_EXIT_BOT_PCT,
            "sector_active_slice_share": benchmark_tilt.SECTOR_ACTIVE_SLICE_SHARE,
            "core_spy_weight": benchmark_tilt.CORE_SPY_WEIGHT,
            "stock_active_limit": benchmark_tilt.STOCK_ACTIVE_LIMIT,
            "sector_active_limit": benchmark_tilt.SECTOR_ACTIVE_LIMIT,
            "trade_deadband": benchmark_tilt.TRADE_DEADBAND,
            "frozen_cost_per_turnover": benchmark_tilt.FROZEN_COST_PER_TURNOVER,
            "info_cost_per_turnover": benchmark_tilt.INFO_COST_PER_TURNOVER,
            "label": (
                "research label on provider split- and dividend-adjusted SIP daily bars; continuous open-to-open "
                "holding windows [open[r_k + 1], open[r_{k+1} + 1]] with post-drift turnover accounting"
            ),
        }

        document = _json_safe({
            "notice": LABEL_NOTICE.format(first=settings["first_decision_date"], end=settings["development_end"]),
            "settings": settings,
            "inputs": input_sha256,
            "panel_load": research_panel.load_report,
            "quality_and_cap_inputs": {
                "security_issuer_mapping": identity_report,
                "company_facts_issuers_read": reader.issuers_read,
                "extra_company_facts_ciks_read": reader.extra_ciks_read,
                "company_facts_rows_kept_filed_on_or_before_development_end": dict(sorted(reader.counts.items())),
                "unadjusted_close": unadjusted_report,
                **qc.report,
            },
            "benchmark_tilt": evaluation,
            "process_memory_after_scoring": _process_memory_gib(),
        })

        staging = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        staging.mkdir(parents=True)
        try:
            (staging / "scoreboard.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
            )
            (staging / "scoreboard.md").write_text(_render_benchmark_tilt_markdown(document), encoding="utf-8")
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        append_trial(
            destination.parent / TRIAL_LOG_NAME,
            root=root,
            command=BENCHMARK_TILT_COMMAND,
            output=identity.relative_to(root),
            input_sha256=input_sha256,
            settings=settings,
            scoreboard=document,
        )
    return {
        "output": identity.relative_to(root).as_posix(),
        "inputs": input_sha256,
        "reading_summary": evaluation["reading_summary"],
        "process_memory_after_scoring": document["process_memory_after_scoring"],
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
        f"- **Stage-1 base history:** warm-up daily bars from `2016-01-04`; point-in-time S&P 500 eligible decisions "
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


def score_two_stage_models(
    *,
    root: Path,
    output: Path | None = None,
    development_end: date,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    backfill_dir: Path = free_backfill.BACKFILL_DIRECTORY,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    horizons: tuple[int, ...] = HORIZONS,
    unseal_final_test_period: bool = False,
) -> dict[str, Any]:
    """Run Recommendation Brief §6 Steps 2-3 (Stage-1 combined base + Stage-2 news residual) on extended S&P 500."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    output = output or default_output(prefix="two_stage_models")
    identity = paths.identity(output)
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(TWO_STAGE_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("two-stage preparation")
        destination = paths.write(identity)
        if destination.exists():
            raise FileExistsError(f"research lane output already exists: {destination}")

        ext = free_backfill.load_extended_sp500_data(
            paths,
            development_end=development_end,
            panel_rel=panel,
            backfill_rel=backfill_dir,
            unseal_final_test_period=unseal_final_test_period,
        )
        guard("two-stage extended panel loaded")

        first_row = int(
            np.searchsorted(ext.panel.sessions, np.datetime64(free_backfill.EARLY_DECISION_START, "D"), side="left")
        )
        facts_counts: Counter[str] = Counter()
        issuers_loaded = 0
        facts_paths = ext.company_facts_paths

        def _load_fact(cik: str) -> IssuerFacts | None:
            nonlocal issuers_loaded
            p = facts_paths.get(cik)
            if p is None or not p.is_file():
                return None
            facts, cnt = read_company_facts(p, development_end)
            facts_counts.update(cnt)
            issuers_loaded += 1
            if issuers_loaded % 30 == 0:
                guard("two-stage company facts")
            return facts

        fund_inputs = fundamentals.FundamentalInputs(
            panel=ext.panel,
            unadjusted_close=ext.unadjusted_close,
            releases=ext.releases,
            security_issuer=ext.security_cik_map,
            load_issuer=_load_fact,
        )
        bundle = two_stage.build_stage1_features(ext.panel, fund_inputs)
        ext_panel = ext.panel
        sec_cik_map = ext.security_cik_map
        ext_sha = dict(ext.input_sha256)
        ext_rep = dict(ext.report)
        del fund_inputs, ext
        release_unused_process_memory()
        guard("two-stage stage1 features computed")

        news_feats, news_sha, news_rep = two_stage.load_news_reaction_features(
            ext_panel,
            panel_root=paths.directory(panel),
            early_news_dir=paths.directory(early_news_dir),
            later_news_dir=paths.directory(later_news_dir),
            unseal_final_test_period=unseal_final_test_period,
        )
        release_unused_process_memory()
        guard("two-stage news features computed")

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
            s1_h = two_stage.run_stage1_walk_forward(
                ext_panel, bundle, sec_cik_map, first_row=first_row, horizons=(h,), checkpoint=guard,
            )
            s2_h = two_stage.run_stage2_news_residuals(
                ext_panel, s1_h, news_feats, horizons=(h,), checkpoint=guard,
            )
            ev_h = two_stage.evaluate_two_stage_scoreboard(
                ext_panel, bundle, s1_h, s2_h, horizons=(h,),
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
            guard(f"two-stage horizon {h} completed")

        del news_feats
        release_unused_process_memory()
        evaluation: dict[str, Any] = {
            "stage1_models": stage1_merged,
            "stage2_news_models": stage2_merged,
            "acceptance_summary": acceptance_merged,
        }

        input_sha256 = {**ext_sha, **news_sha}
        settings: dict[str, Any] = {
            "development_end": development_end.isoformat(),
            "unseal_final_test_period": unseal_final_test_period,
            "stage1_first_decision_session": str(ext_panel.sessions[first_row]),
            "stage2_news_start_date": two_stage.NEWS_START_DATE.isoformat(),
            "horizons_sessions": list(horizons),
            "primary_horizon_sessions": two_stage.PRIMARY_HORIZON,
            "secondary_horizon_sessions": two_stage.SECONDARY_HORIZON,
            "stage1_feature_count": len(bundle.feature_names),
            "stage2_news_feature_count": len(two_stage.NEWS_FEATURE_NAMES),
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
            "inputs": input_sha256,
            "extended_data_report": {
                **ext_rep,
                "company_facts_issuers_loaded": issuers_loaded,
                "company_facts_rows_kept": dict(sorted(facts_counts.items())),
            },
            "stage1_coverage": bundle.coverage_report,
            "stage2_news_report": news_rep,
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
            command=TWO_STAGE_COMMAND,
            output=identity.relative_to(root),
            input_sha256=input_sha256,
            settings=settings,
            scoreboard=document,
        )
    return {
        "output": identity.relative_to(root).as_posix(),
        "inputs": input_sha256,
        "acceptance_summary": evaluation["acceptance_summary"],
        "process_memory_after_scoring": document["process_memory_after_scoring"],
    }


def score_wider_two_stage_models(
    *,
    root: Path,
    output: Path | None = None,
    development_end: date,
    subset: wider_universe.UniverseSubset = "all_top1200",
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    sp500_backfill_dir: Path = free_backfill.BACKFILL_DIRECTORY,
    wider_dir: Path = wider_universe.WIDER_UNIVERSE_DIRECTORY,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    horizons: tuple[int, ...] = HORIZONS,
    unseal_final_test_period: bool = False,
) -> dict[str, Any]:
    """Run Step 5b Stage-1 combined base + Stage-2 catalyst residual models on the $0 wider US universe."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    output = output or default_output(prefix=f"wider_two_stage_{subset}")
    identity = paths.identity(output)
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(WIDER_TWO_STAGE_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("wider two-stage preparation")
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
        guard("wider universe panel loaded")

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
                guard("wider two-stage company facts")
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
        guard("wider two-stage stage1 features computed")

        cat_feats, cat_rep = wider_universe.build_wider_catalyst_reaction_features(
            w_panel,
            all_8k_events,
            early_news_dir=paths.directory(early_news_dir),
            later_news_dir=paths.directory(later_news_dir),
            unseal_final_test_period=unseal_final_test_period,
        )
        del all_8k_events
        release_unused_process_memory()
        guard("wider two-stage catalyst features computed")

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
            s1_h = two_stage.run_stage1_walk_forward(
                w_panel, bundle, sec_cik_map, first_row=first_row, horizons=(h,), checkpoint=guard,
            )
            s2_h = two_stage.run_stage2_news_residuals(
                w_panel,
                s1_h,
                cat_feats,
                horizons=(h,),
                feature_names=wider_universe.WIDER_STAGE2_FEATURE_NAMES,
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
            guard(f"wider two-stage horizon {h} completed")

        del cat_feats
        release_unused_process_memory()
        evaluation: dict[str, Any] = {
            "stage1_models": stage1_merged,
            "stage2_news_models": stage2_merged,
            "acceptance_summary": acceptance_merged,
        }

        settings: dict[str, Any] = {
            "universe_subset": subset,
            "development_end": development_end.isoformat(),
            "unseal_final_test_period": unseal_final_test_period,
            "stage1_first_decision_session": str(w_panel.sessions[first_row]),
            "stage2_news_start_date": two_stage.NEWS_START_DATE.isoformat(),
            "horizons_sessions": list(horizons),
            "primary_horizon_sessions": two_stage.PRIMARY_HORIZON,
            "secondary_horizon_sessions": two_stage.SECONDARY_HORIZON,
            "stage1_feature_count": len(bundle.feature_names),
            "stage2_catalyst_feature_count": len(wider_universe.WIDER_STAGE2_FEATURE_NAMES),
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
            command=WIDER_TWO_STAGE_COMMAND,
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


def score_core_satellite_spy(
    *,
    root: Path,
    output: Path | None = None,
    development_end: date,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    sp500_backfill_dir: Path = free_backfill.BACKFILL_DIRECTORY,
    wider_dir: Path = wider_universe.WIDER_UNIVERSE_DIRECTORY,
    early_news_dir: Path = EARLY_NEWS_FAMILY_DIR,
    later_news_dir: Path = LATER_NEWS_FAMILY_DIR,
    horizons: tuple[int, ...] = HORIZONS,
    unseal_final_test_period: bool = False,
) -> dict[str, Any]:
    """Run Step 5d unified Core SPY / S&P 500 active tilt + mid/small SEC-catalyst satellite vs. SPY."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    output = output or default_output(prefix="core_satellite_spy")
    identity = paths.identity(output)
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(CORE_SATELLITE_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("core-satellite preparation")
        destination = paths.write(identity)
        if destination.exists():
            raise FileExistsError(f"research lane output already exists: {destination}")

        # Phase 1: S&P 500 Core signals (`equal_rank_24` + Stage-2 Alpaca `news_xgboost`)
        ext = free_backfill.load_extended_sp500_data(
            paths,
            development_end=development_end,
            panel_rel=panel,
            backfill_rel=sp500_backfill_dir,
            unseal_final_test_period=unseal_final_test_period,
        )
        first_row = int(
            np.searchsorted(ext.panel.sessions, np.datetime64(free_backfill.EARLY_DECISION_START, "D"), side="left")
        )
        sp_facts_paths = ext.company_facts_paths

        def _load_sp_fact(cik: str) -> IssuerFacts | None:
            p = sp_facts_paths.get(cik)
            if p is None or not p.is_file():
                return None
            facts, _ = read_company_facts(p, development_end)
            return facts

        sp_bundle = two_stage.build_stage1_features(
            ext.panel,
            fundamentals.FundamentalInputs(
                panel=ext.panel,
                unadjusted_close=ext.unadjusted_close,
                releases=ext.releases,
                security_issuer=ext.security_cik_map,
                load_issuer=_load_sp_fact,
            ),
        )
        sp500_panel = ext.panel
        sp500_cik_map = dict(ext.security_cik_map)
        sp_sha = dict(ext.input_sha256)
        sp500_mv = sp_bundle.market_value.copy()
        sp500_eq24 = sp_bundle.equal_rank_24.copy()
        del sp_bundle, ext
        release_unused_process_memory()
        guard("core-satellite sp500 stage1 computed")

        sp_news_feats, sp_news_sha, sp_news_rep = two_stage.load_news_reaction_features(
            sp500_panel,
            panel_root=paths.directory(panel),
            early_news_dir=paths.directory(early_news_dir),
            later_news_dir=paths.directory(later_news_dir),
            unseal_final_test_period=unseal_final_test_period,
        )
        sp_targets = {
            h: target_ranks(research_labels(sp500_panel, h).stock, sp500_panel.eligible)
            for h in horizons
        }
        sp500_eq24_news_by_h, stage2_oos_start_row, sp_paired_ic = core_satellite.fit_eq24_stage2_xgb_by_horizon(
            sp500_panel,
            sp500_eq24,
            targets_by_horizon=sp_targets,
            catalyst_features=sp_news_feats,
            catalyst_feature_names=two_stage.NEWS_FEATURE_NAMES,
            horizons=horizons,
            checkpoint=guard,
        )
        del sp_news_feats, sp_targets
        release_unused_process_memory()
        guard("core-satellite sp500 stage2 computed")

        # Phase 2: Disjoint `mid_small_ex_sp500` Satellite signals (`raw` & `beta_neutral` + precise SEC `news_xgboost`)
        wdata = wider_universe.load_wider_universe_data(
            paths,
            development_end=development_end,
            subset="mid_small_ex_sp500",
            panel_rel=panel,
            sp500_backfill_rel=sp500_backfill_dir,
            wider_rel=wider_dir,
            unseal_final_test_period=unseal_final_test_period,
        )
        alignment_report = core_satellite.verify_disjoint_core_satellite_universes(
            sp500_panel, sp500_cik_map, wdata.panel, wdata.security_cik_map,
        )
        mid_facts_paths = wdata.company_facts_paths
        mid_loaded = 0

        def _load_mid_fact(cik: str) -> IssuerFacts | None:
            nonlocal mid_loaded
            p = mid_facts_paths.get(cik)
            if p is None or not p.is_file():
                return None
            facts, _ = read_company_facts(p, development_end)
            mid_loaded += 1
            if mid_loaded % 40 == 0:
                guard("core-satellite mid_small company facts")
            return facts

        mid_bundle = two_stage.build_stage1_features(
            wdata.panel,
            fundamentals.FundamentalInputs(
                panel=wdata.panel,
                unadjusted_close=wdata.unadjusted_close,
                releases=wdata.releases_202,
                security_issuer=wdata.security_cik_map,
                load_issuer=_load_mid_fact,
            ),
        )
        mid_panel = wdata.panel
        all_8k_events = wdata.all_8k_events
        w_sha = dict(wdata.input_sha256)
        del wdata

        mid_raw_eq24 = mid_bundle.equal_rank_24.copy()
        beta_rk = mid_bundle.features["px_beta_252d"]
        ivol_rk = mid_bundle.features["px_idio_vol_60d"]
        mid_bn_eq24 = beta_neutral.cross_sectional_neutralize(
            mid_raw_eq24, mid_panel.eligible, mid_panel.sector, beta_rk, ivol_rk, rerank=True,
        )
        mid_raw_targets = {
            h: target_ranks(research_labels(mid_panel, h).stock, mid_panel.eligible)
            for h in horizons
        }
        mid_bn_targets = {
            h: beta_neutral.cross_sectional_neutralize(
                mid_raw_targets[h], mid_panel.eligible, mid_panel.sector, beta_rk, ivol_rk, rerank=True,
            )
            for h in horizons
        }
        del mid_bundle, beta_rk, ivol_rk
        release_unused_process_memory()
        guard("core-satellite mid_small stage1 computed")

        wider_abs = paths.directory(wider_dir)
        non8k_pq = wider_abs / beta_neutral.NON8K_CATALYST_PARQUET
        if not non8k_pq.is_file():
            raise FileNotFoundError(f"missing {non8k_pq}; run {beta_neutral.COLLECT_SEC_CATALYSTS_COMMAND} first")
        w_sha["wider_non8k_catalysts_sha256"] = file_sha256(non8k_pq)
        non8k_df = cast(Any, pq).read_table(non8k_pq, use_threads=False).to_pandas()
        sec_cat_feats, sec_cat_rep = beta_neutral.build_precise_sec_catalyst_features(
            mid_panel, all_8k_events, non8k_df, unseal_final_test_period=unseal_final_test_period,
        )
        del all_8k_events, non8k_df
        release_unused_process_memory()

        mid_raw_sec_by_h, _, mid_raw_paired_ic = core_satellite.fit_eq24_stage2_xgb_by_horizon(
            mid_panel,
            mid_raw_eq24,
            targets_by_horizon=mid_raw_targets,
            catalyst_features=sec_cat_feats,
            catalyst_feature_names=beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
            horizons=horizons,
            checkpoint=guard,
        )
        mid_bn_sec_by_h, _, mid_bn_paired_ic = core_satellite.fit_eq24_stage2_xgb_by_horizon(
            mid_panel,
            mid_bn_eq24,
            targets_by_horizon=mid_bn_targets,
            catalyst_features=sec_cat_feats,
            catalyst_feature_names=beta_neutral.PRECISE_STAGE2_FEATURE_NAMES,
            horizons=horizons,
            checkpoint=guard,
        )
        del sec_cat_feats, mid_raw_targets, mid_bn_targets
        release_unused_process_memory()
        guard("core-satellite mid_small stage2 computed")

        # Phase 3: Joint Core-Satellite simulation vs. SPY
        s1_folds = two_stage._dynamic_folds(sp500_panel.sessions, first_row)
        stage1_oos_start_row = s1_folds[0].test_start

        evaluation = core_satellite.evaluate_core_satellite_grid(
            sp500_panel=sp500_panel,
            sp500_mv=sp500_mv,
            sp500_eq24=sp500_eq24,
            sp500_eq24_news_by_h=sp500_eq24_news_by_h,
            mid_panel=mid_panel,
            mid_signals_stage1={
                "sat_raw_eq24": mid_raw_eq24,
                "sat_beta_neutral_eq24": mid_bn_eq24,
            },
            mid_signals_stage2_by_h={
                h: {
                    "sat_raw_eq24_sec_xgb": mid_raw_sec_by_h[h],
                    "sat_beta_neutral_eq24_sec_xgb": mid_bn_sec_by_h[h],
                }
                for h in horizons
            },
            stage1_oos_start_row=stage1_oos_start_row,
            stage2_oos_start_row=stage2_oos_start_row,
            horizons=horizons,
        )
        guard("core-satellite joint simulation completed")

        all_sha = {**sp_sha, **sp_news_sha, **w_sha}
        settings: dict[str, Any] = {
            "development_end": development_end.isoformat(),
            "unseal_final_test_period": unseal_final_test_period,
            "stage1_oos_first_session": str(sp500_panel.sessions[stage1_oos_start_row]),
            "stage2_oos_first_session": str(sp500_panel.sessions[stage2_oos_start_row]),
            "horizons_sessions": list(horizons),
            "allocations": {tag: w for tag, w in core_satellite.ALLOCATIONS},
            "cost_per_one_way_turnover_primary": benchmark_tilt.FROZEN_COST_PER_TURNOVER,
            "cost_per_one_way_turnover_info": benchmark_tilt.INFO_COST_PER_TURNOVER,
            "label": "research label on provider split- and dividend-adjusted SIP daily bars",
        }
        document = _json_safe({
            "notice": LABEL_NOTICE.format(
                first=settings["stage1_oos_first_session"], end=settings["development_end"],
            ),
            "settings": settings,
            "inputs": all_sha,
            "universe_alignment": alignment_report,
            "catalyst_reports": {
                "sp500_alpaca_news": sp_news_rep,
                "sp500_paired_ic": {str(k): v for k, v in sp_paired_ic.items()},
                "mid_small_precise_sec": sec_cat_rep,
                "mid_small_raw_paired_ic": {str(k): v for k, v in mid_raw_paired_ic.items()},
                "mid_small_beta_neutral_paired_ic": {str(k): v for k, v in mid_bn_paired_ic.items()},
            },
            "core_satellite": evaluation,
            "process_memory_after_scoring": _process_memory_gib(),
        })

        staging = destination.with_name(f".{destination.name}.{uuid4().hex}.partial")
        staging.mkdir(parents=True)
        try:
            (staging / "scoreboard.json").write_text(
                json.dumps(document, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
            )
            (staging / "scoreboard.md").write_text(
                core_satellite.render_core_satellite_markdown(document), encoding="utf-8",
            )
            os.replace(staging, destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        append_trial(
            destination.parent / TRIAL_LOG_NAME,
            root=root,
            command=CORE_SATELLITE_COMMAND,
            output=identity.relative_to(root),
            input_sha256=all_sha,
            settings=settings,
            scoreboard=document,
        )
    return {
        "output": identity.relative_to(root).as_posix(),
        "inputs": all_sha,
        "universe_alignment": alignment_report,
        "reading_summary": evaluation["reading_summary"],
        "process_memory_after_scoring": document["process_memory_after_scoring"],
    }
