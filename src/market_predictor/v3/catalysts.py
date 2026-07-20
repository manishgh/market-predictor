from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from sklearn.metrics import ndcg_score

from market_predictor.data_quality import sanitize_events_frame
from market_predictor.features import add_event_taxonomy, source_family_for_source
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.schema import FrozenContract

AvailabilityPolicy = Literal["strict_ingestion", "provider_publication_backfill"]
MATERIAL_EVENT_TYPES = ("earnings", "guidance", "analyst", "ma", "fda", "contract", "offering", "insider")
NEGATIVE_VETO_EVENT_TYPES = ("guidance", "fda", "offering")
O1_WINDOWS = {"2h": pd.Timedelta(hours=2), "1d": pd.Timedelta(days=1)}


class O1OverlayConfig(FrozenContract):
    coverage_start_utc: datetime
    coverage_end_utc: datetime
    availability_policy: AvailabilityPolicy = "strict_ingestion"
    base_family: str = "R1"
    overlay_weight: float = Field(default=0.15, ge=0, le=1)
    ticker_veto_penalty: float = Field(default=0.50, ge=0, le=2)
    minimum_relevance: float = Field(default=1.25, ge=0)
    negative_veto_sentiment: float = Field(default=-0.35, ge=-1, le=0)
    global_veto_sentiment: float = Field(default=-0.60, ge=-1, le=0)
    minimum_global_veto_events: int = Field(default=3, ge=1)
    minimum_ticker_file_coverage: float = Field(default=0.90, ge=0, le=1)
    minimum_sentiment_coverage: float = Field(default=0.95, ge=0, le=1)
    maximum_market_context_boundary_gap_hours: float = Field(default=24.0, ge=0)
    minimum_decision_rows: int = Field(default=10_000, ge=1)

    @field_validator("coverage_start_utc", "coverage_end_utc")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise ValueError("O1 coverage timestamps must be timezone-aware")
        return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())

    @model_validator(mode="after")
    def validate_window(self) -> O1OverlayConfig:
        if self.coverage_start_utc >= self.coverage_end_utc:
            raise ValueError("O1 coverage start must precede coverage end")
        return self


class O1AuditConfig(FrozenContract):
    top_k: int = Field(default=10, ge=1)
    bootstrap_iterations: int = Field(default=1_000, ge=100, le=100_000)
    bootstrap_seed: int = 42
    minimum_sessions: int = Field(default=20, ge=2)


def build_o1_overlay_evidence(
    predictions: pd.DataFrame,
    *,
    event_directories: list[Path],
    market_context_path: Path | None,
    config: O1OverlayConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {
        "ticker",
        "decision_time_utc",
        "decision_group_id",
        "audit_scope",
        "family",
        "model_run_id",
        "score",
        "ranking_target",
        "ranking_grade",
        "session_date_et",
        "entry_time_utc",
        "primary_exit_time_utc",
        "path_realized_return_net",
        "independent_event_id",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataReadinessError(f"O1 prediction evidence missing columns: {', '.join(missing)}")
    directories = [Path(path) for path in event_directories]
    missing_directories = [str(path) for path in directories if not path.is_dir()]
    if not directories or missing_directories:
        raise DataReadinessError(f"O1 event directories are missing: {missing_directories or 'none provided'}")

    data = predictions[predictions["family"].astype(str).eq(config.base_family)].copy()
    if data.empty:
        raise DataReadinessError(f"O1 requires {config.base_family} OOF predictions")
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["decision_time_utc"] = _utc_series(data["decision_time_utc"])
    if bool(data["decision_time_utc"].isna().any()):
        raise DataReadinessError("O1 predictions contain invalid decision timestamps")
    run_ids = set(data["model_run_id"].astype(str))
    if len(run_ids) != 1:
        raise DataReadinessError("O1 requires exactly one base model_run_id")
    identity = ["ticker", "decision_time_utc", "decision_group_id", "audit_scope"]
    if bool(data.duplicated(identity).any()):
        raise DataReadinessError("O1 predictions contain duplicate decision identities")

    coverage_start = pd.Timestamp(config.coverage_start_utc)
    coverage_end = pd.Timestamp(config.coverage_end_utc)
    in_window = data["decision_time_utc"].between(coverage_start, coverage_end, inclusive="both")
    input_rows = len(data)
    data = data.loc[in_window].copy()
    prediction_tickers = sorted(data["ticker"].unique())
    ticker_files = _ticker_file_map(prediction_tickers, directories)
    covered_tickers = sorted(ticker_files)
    missing_tickers = sorted(set(prediction_tickers).difference(covered_tickers))
    data = data[data["ticker"].isin(covered_tickers)].copy()
    if data.empty:
        raise DataReadinessError("O1 has no prediction rows inside declared source coverage")

    decision_columns = ["ticker", "decision_time_utc"]
    decisions = data[decision_columns].drop_duplicates(["ticker", "decision_time_utc"]).copy()
    pieces: list[pd.DataFrame] = []
    event_audits: list[dict[str, Any]] = []
    for ticker, group in decisions.groupby("ticker", sort=False, observed=True):
        events, event_audit = _load_ticker_events(
            str(ticker),
            ticker_files[str(ticker)],
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            config=config,
        )
        pieces.append(_aggregate_event_windows(group, events, prefix="ticker_"))
        event_audits.append(event_audit)
    catalyst = pd.concat(pieces, ignore_index=True)

    market_audit: dict[str, Any] = {"available": False, "rows": 0, "sentiment_coverage": 0.0}
    if market_context_path is not None:
        if not market_context_path.exists():
            raise DataReadinessError(f"O1 market context file is missing: {market_context_path}")
        market_events, market_audit = _prepare_event_frame(
            pd.read_parquet(market_context_path),
            ticker=None,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            config=config,
            minimum_relevance=0.0,
        )
        market_decisions = decisions[["decision_time_utc"]].drop_duplicates().copy()
        market_decisions["ticker"] = "MARKET"
        market_features = _aggregate_event_windows(market_decisions, market_events, prefix="global_").drop(columns="ticker")
        catalyst = catalyst.merge(market_features, on="decision_time_utc", how="left", validate="many_to_one")
    else:
        catalyst = _empty_global_features(catalyst)

    data = data.merge(
        catalyst,
        on=["ticker", "decision_time_utc"],
        how="inner",
        validate="many_to_one",
    )
    data["r1_rank_percentile"] = data.groupby(["audit_scope", "decision_group_id"], observed=True)["score"].rank(
        method="average",
        pct=True,
    )
    ticker_signal = 0.65 * data["ticker_signal_2h"] + 0.35 * data["ticker_signal_1d"]
    data["o1_ticker_signal"] = ticker_signal.clip(-1, 1)
    data["o1_ticker_veto"] = data["ticker_negative_material_count_1d"].gt(0)
    data["o1_global_veto"] = data["global_event_count_2h"].ge(config.minimum_global_veto_events) & data[
        "global_sentiment_mean_2h"
    ].le(config.global_veto_sentiment)
    data["o1_eligible"] = ~(data["o1_ticker_veto"] | data["o1_global_veto"])
    data["o1_score"] = (
        data["r1_rank_percentile"]
        + config.overlay_weight * data["o1_ticker_signal"]
        - config.ticker_veto_penalty * data["o1_ticker_veto"].astype(float)
    )
    data["o1_status"] = np.select(
        [
            data["o1_global_veto"],
            data["o1_ticker_veto"],
            data["o1_ticker_signal"].ge(0.15),
            data["o1_ticker_signal"].le(-0.15),
            data["ticker_event_count_1d"].gt(0),
        ],
        ["global_veto", "ticker_veto", "confirmed", "conflicting", "mixed"],
        default="absent",
    )

    event_rows = int(sum(record["rows_after_dedup"] for record in event_audits))
    relevant_rows = int(sum(record["relevant_rows"] for record in event_audits))
    sentiment_rows = int(sum(record["sentiment_rows"] for record in event_audits))
    sentiment_coverage = sentiment_rows / relevant_rows if relevant_rows else 0.0
    ticker_file_coverage = len(covered_tickers) / len(prediction_tickers) if prediction_tickers else 0.0
    future_matches = int(
        (
            data["ticker_latest_event_at_utc"].notna()
            & (data["ticker_latest_event_at_utc"] > data["decision_time_utc"])
        ).sum()
    )
    readiness_failures: list[str] = []
    if ticker_file_coverage < config.minimum_ticker_file_coverage:
        readiness_failures.append(
            f"ticker event-file coverage {ticker_file_coverage:.4f} < {config.minimum_ticker_file_coverage:.4f}"
        )
    if sentiment_coverage < config.minimum_sentiment_coverage:
        readiness_failures.append(
            f"relevant-event sentiment coverage {sentiment_coverage:.4f} < {config.minimum_sentiment_coverage:.4f}"
        )
    if len(data) < config.minimum_decision_rows:
        readiness_failures.append(f"covered decision rows {len(data)} < {config.minimum_decision_rows}")
    if future_matches:
        readiness_failures.append(f"future catalyst matches detected: {future_matches}")
    if market_context_path is not None and float(market_audit["sentiment_coverage"]) < config.minimum_sentiment_coverage:
        readiness_failures.append("market-context sentiment coverage is incomplete")
    if market_context_path is not None:
        maximum_gap = pd.Timedelta(hours=config.maximum_market_context_boundary_gap_hours)
        first_market_event = _optional_timestamp(market_audit.get("first_available_at_utc"))
        last_market_event = _optional_timestamp(market_audit.get("last_available_at_utc"))
        if first_market_event is None or first_market_event > coverage_start + maximum_gap:
            readiness_failures.append("market-context archive does not cover the declared start boundary")
        if last_market_event is None or last_market_event < coverage_end - maximum_gap:
            readiness_failures.append("market-context archive does not cover the declared end boundary")

    audit = {
        "schema": "ml_v3.o1_catalyst_readiness.v1",
        "config": config.model_dump(mode="json"),
        "base_model_run_id": next(iter(run_ids)),
        "availability_policy": config.availability_policy,
        "research_only": config.availability_policy == "provider_publication_backfill",
        "input_prediction_rows": input_rows,
        "covered_prediction_rows": len(data),
        "prediction_tickers_in_window": len(prediction_tickers),
        "covered_tickers": len(covered_tickers),
        "ticker_file_coverage": ticker_file_coverage,
        "missing_tickers": missing_tickers,
        "event_rows_after_dedup": event_rows,
        "relevant_event_rows": relevant_rows,
        "sentiment_rows": sentiment_rows,
        "sentiment_coverage": sentiment_coverage,
        "rows_with_ticker_catalyst_1d": int(data["ticker_event_count_1d"].gt(0).sum()),
        "rows_with_global_context_2h": int(data["global_event_count_2h"].gt(0).sum()),
        "future_matches": future_matches,
        "market_context": market_audit,
        "readiness_failures": readiness_failures,
        "ready": not readiness_failures,
    }
    return data.sort_values(["audit_scope", "decision_time_utc", "ticker"]).reset_index(drop=True), audit


def evaluate_o1_ablation(
    evidence: pd.DataFrame,
    *,
    config: O1AuditConfig = O1AuditConfig(),
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "ticker",
        "audit_scope",
        "decision_group_id",
        "session_date_et",
        "decision_time_utc",
        "entry_time_utc",
        "primary_exit_time_utc",
        "ranking_grade",
        "ranking_target",
        "path_realized_return_net",
        "independent_event_id",
        "r1_rank_percentile",
        "o1_score",
        "o1_eligible",
        "o1_status",
    }
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise DataReadinessError(f"O1 ablation evidence missing columns: {', '.join(missing)}")
    reports: dict[str, Any] = {}
    selections: list[pd.DataFrame] = []
    for scope, scope_data in evidence.groupby("audit_scope", sort=False, observed=True):
        scope_report, scope_selected = _evaluate_scope(scope_data, scope=str(scope), config=config)
        reports[str(scope)] = scope_report
        selections.append(scope_selected)
    required_scopes = {"walk_forward", "ticker_holdout"}
    missing_scopes = sorted(required_scopes.difference(reports))
    if missing_scopes:
        raise DataReadinessError(f"O1 ablation missing audit scopes: {', '.join(missing_scopes)}")
    passed = all(
        reports[scope]["o1"]["mean_top_k_excess_return"] > 0
        and reports[scope]["paired_excess_return_delta_interval"]["low"] >= 0
        for scope in required_scopes
    )
    report = {
        "schema": "ml_v3.o1_ablation.v1",
        "config": config.model_dump(mode="json"),
        "scopes": reports,
        "passed_development_floor": passed,
        "promotion_eligible": False,
        "promotion_blocker": "O1 requires strict catalyst provenance, a valid downside model, and frozen gates",
    }
    selected = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    return report, selected


def _evaluate_scope(
    data: pd.DataFrame,
    *,
    scope: str,
    config: O1AuditConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    baseline_rows: list[pd.DataFrame] = []
    overlay_rows: list[pd.DataFrame] = []
    paired: list[dict[str, Any]] = []
    baseline_ndcg: list[float] = []
    overlay_ndcg: list[float] = []
    for group_id, group in data.groupby("decision_group_id", sort=False, observed=True):
        if len(group) < 2:
            continue
        k = min(config.top_k, len(group))
        grade = pd.to_numeric(group["ranking_grade"], errors="coerce")
        valid = grade.notna()
        if int(valid.sum()) < 2:
            continue
        baseline_ndcg.append(
            float(ndcg_score([grade[valid].to_numpy()], [group.loc[valid, "r1_rank_percentile"].to_numpy()], k=k))
        )
        overlay_ndcg.append(float(ndcg_score([grade[valid].to_numpy()], [group.loc[valid, "o1_score"].to_numpy()], k=k)))
        baseline = group.nlargest(k, "r1_rank_percentile").copy()
        eligible = group[group["o1_eligible"].astype(bool)]
        overlay = eligible.nlargest(min(k, len(eligible)), "o1_score").copy() if not eligible.empty else eligible.copy()
        baseline["strategy"] = "R1"
        overlay["strategy"] = "O1"
        baseline_rows.append(baseline)
        overlay_rows.append(overlay)
        paired.append(
            {
                "session_date_et": group["session_date_et"].iloc[0],
                "decision_group_id": group_id,
                "r1_excess_return": float(pd.to_numeric(baseline["ranking_target"], errors="coerce").mean()),
                "o1_excess_return": (
                    float(pd.to_numeric(overlay["ranking_target"], errors="coerce").mean()) if not overlay.empty else 0.0
                ),
            }
        )
    if not paired:
        raise DataReadinessError(f"O1 {scope} scope contains no auditable ranking groups")
    baseline_selected = pd.concat(baseline_rows, ignore_index=True)
    overlay_selected = pd.concat(overlay_rows, ignore_index=True) if overlay_rows else pd.DataFrame(columns=data.columns)
    paired_frame = pd.DataFrame(paired)
    paired_frame["delta"] = paired_frame["o1_excess_return"] - paired_frame["r1_excess_return"]
    selected = pd.concat([baseline_selected, overlay_selected], ignore_index=True)
    session_count = int(pd.Series(paired_frame["session_date_et"]).nunique())
    report = {
        "scope": scope,
        "ranking_groups": len(paired_frame),
        "sessions": session_count,
        "r1": _selection_summary(baseline_selected, baseline_ndcg),
        "o1": _selection_summary(overlay_selected, overlay_ndcg),
        "paired_excess_return_delta": float(paired_frame["delta"].mean()),
        "paired_excess_return_delta_interval": _session_bootstrap_interval(
            paired_frame,
            value_column="delta",
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
        ),
    }
    if session_count < config.minimum_sessions:
        report["readiness_failure"] = f"sessions {session_count} < {config.minimum_sessions}"
    return report, selected


def _selection_summary(selected: pd.DataFrame, ndcg_values: list[float]) -> dict[str, Any]:
    if selected.empty:
        return {
            "selected_rows": 0,
            "independent_trades": 0,
            "mean_ndcg_at_k": float(np.mean(ndcg_values)) if ndcg_values else None,
            "mean_top_k_excess_return": 0.0,
            "positive_top_k_rate": 0.0,
            "average_trade_return": 0.0,
            "win_rate": 0.0,
            "profit_factor": None,
            "max_drawdown": 0.0,
            "catalyst_selected_rate": 0.0,
        }
    independent = selected.dropna(subset=["independent_event_id"]).drop_duplicates("independent_event_id")
    independent = _non_overlapping_groups(independent)
    excess = pd.to_numeric(selected["ranking_target"], errors="coerce").dropna()
    realized = pd.to_numeric(independent["path_realized_return_net"], errors="coerce").dropna()
    catalyst_rate = float(selected["o1_status"].astype(str).isin({"confirmed", "mixed", "conflicting"}).mean())
    return {
        "selected_rows": len(selected),
        "independent_trades": len(independent),
        "mean_ndcg_at_k": float(np.mean(ndcg_values)) if ndcg_values else None,
        "mean_top_k_excess_return": float(excess.mean()),
        "positive_top_k_rate": float(excess.gt(0).mean()),
        "average_trade_return": float(realized.mean()) if not realized.empty else 0.0,
        "win_rate": float(realized.gt(0).mean()) if not realized.empty else 0.0,
        "profit_factor": _profit_factor(realized),
        "max_drawdown": _max_drawdown(_portfolio_returns(independent)),
        "catalyst_selected_rate": catalyst_rate,
    }


def _load_ticker_events(
    ticker: str,
    paths: list[Path],
    *,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
    config: O1OverlayConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = [pd.read_parquet(path) for path in paths]
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    events, audit = _prepare_event_frame(
        raw,
        ticker=ticker,
        coverage_start=coverage_start,
        coverage_end=coverage_end,
        config=config,
        minimum_relevance=config.minimum_relevance,
    )
    audit["ticker"] = ticker
    audit["files"] = [str(path) for path in paths]
    return events, audit


def _prepare_event_frame(
    frame: pd.DataFrame,
    *,
    ticker: str | None,
    coverage_start: pd.Timestamp,
    coverage_end: pd.Timestamp,
    config: O1OverlayConfig,
    minimum_relevance: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    clean, verification = sanitize_events_frame(frame)
    if clean.empty:
        return _empty_events(), {
            "available": True,
            "rows": 0,
            "rows_after_dedup": 0,
            "relevant_rows": 0,
            "sentiment_rows": 0,
            "sentiment_coverage": 0.0,
            "duplicates_removed": verification.duplicate_rows_removed,
        }
    clean["published_at_utc"] = _utc_series(clean["timestamp"])
    if config.availability_policy == "strict_ingestion":
        availability_column = next(
            (column for column in ("available_at_utc", "ingested_at_utc") if column in clean.columns),
            None,
        )
        if availability_column is None:
            raise DataReadinessError("strict O1 catalyst input requires available_at_utc or ingested_at_utc")
        available = _utc_series(clean[availability_column])
        clean["available_at_utc"] = pd.concat([clean["published_at_utc"], available], axis=1).max(axis=1)
    else:
        clean["available_at_utc"] = clean["published_at_utc"]
    clean = clean.dropna(subset=["available_at_utc"])
    clean = clean[
        clean["available_at_utc"].between(coverage_start - max(O1_WINDOWS.values()), coverage_end, inclusive="both")
    ].copy()
    clean["source_family"] = clean["source"].map(source_family_for_source)
    clean = add_event_taxonomy(clean)
    text = _event_text(clean)
    if ticker is None:
        clean["relevance"] = 1.0
    else:
        title_match = clean["title"].map(lambda value: _contains_ticker(value, ticker))
        text_match = text.map(lambda value: _contains_ticker(value, ticker))
        generic = clean["title"].map(_is_generic_headline)
        relevance = 1.0 + 0.75 * title_match.astype(float) + 0.35 * (~title_match & text_match).astype(float)
        relevance -= 0.60 * generic.astype(float)
        relevance += 0.30 * clean["source_family"].eq("sec").astype(float)
        clean["relevance"] = relevance.clip(lower=0.1)
    clean["sentiment_numeric"] = pd.to_numeric(clean.get("sentiment_numeric"), errors="coerce")
    clean = clean.drop_duplicates(["ticker", "published_at_utc", "source", "title", "url"], keep="first")
    clean = clean[clean["relevance"].ge(minimum_relevance)].copy()
    sentiment_rows = int(clean["sentiment_numeric"].notna().sum())
    clean["sentiment_numeric"] = clean["sentiment_numeric"].fillna(0.0).clip(-1, 1)
    material_columns = [f"event_{name}" for name in MATERIAL_EVENT_TYPES if f"event_{name}" in clean.columns]
    veto_columns = [f"event_{name}" for name in NEGATIVE_VETO_EVENT_TYPES if f"event_{name}" in clean.columns]
    clean["material_event"] = clean[material_columns].max(axis=1).fillna(0) if material_columns else 0
    veto_event = clean[veto_columns].max(axis=1).fillna(0).gt(0) if veto_columns else pd.Series(False, index=clean.index)
    clean["negative_material"] = veto_event & clean["sentiment_numeric"].le(config.negative_veto_sentiment)
    clean["event_id"] = clean.apply(_event_id, axis=1)
    clean = clean.sort_values("available_at_utc").reset_index(drop=True)
    audit = {
        "available": True,
        "rows": len(frame),
        "rows_after_dedup": len(clean),
        "relevant_rows": len(clean),
        "sentiment_rows": sentiment_rows,
        "sentiment_coverage": sentiment_rows / len(clean) if len(clean) else 0.0,
        "duplicates_removed": verification.duplicate_rows_removed,
        "first_available_at_utc": clean["available_at_utc"].min().isoformat() if not clean.empty else None,
        "last_available_at_utc": clean["available_at_utc"].max().isoformat() if not clean.empty else None,
    }
    return clean, audit


def _aggregate_event_windows(decisions: pd.DataFrame, events: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    output = decisions.sort_values("decision_time_utc").copy()
    for window in O1_WINDOWS:
        output[f"{prefix}event_count_{window}"] = 0.0
        output[f"{prefix}sentiment_mean_{window}"] = 0.0
        output[f"{prefix}relevance_mean_{window}"] = 0.0
        output[f"{prefix}source_diversity_{window}"] = 0.0
        output[f"{prefix}material_count_{window}"] = 0.0
        output[f"{prefix}negative_material_count_{window}"] = 0.0
        output[f"{prefix}signal_{window}"] = 0.0
    output[f"{prefix}latest_event_at_utc"] = pd.NaT
    if events.empty:
        return output
    decision_ns = _timestamp_ns(output["decision_time_utc"])
    event_ns = _timestamp_ns(events["available_at_utc"])
    end = np.searchsorted(event_ns, decision_ns, side="right")
    latest = end - 1
    has_latest = latest >= 0
    latest_values = np.full(len(output), np.datetime64("NaT"), dtype="datetime64[ns]")
    latest_values[has_latest] = events["available_at_utc"].to_numpy(dtype="datetime64[ns]")[latest[has_latest]]
    output[f"{prefix}latest_event_at_utc"] = pd.to_datetime(latest_values, utc=True)
    sentiment = events["sentiment_numeric"].to_numpy(dtype=float)
    relevance = events["relevance"].to_numpy(dtype=float)
    weighted_sentiment = sentiment * relevance
    material = pd.to_numeric(events["material_event"], errors="coerce").fillna(0).to_numpy(dtype=float)
    negative_material = events["negative_material"].astype(float).to_numpy()
    source_families = sorted(events["source_family"].astype(str).unique())
    for name, window in O1_WINDOWS.items():
        start = np.searchsorted(event_ns, decision_ns - int(window.value), side="left")
        count = end - start
        relevance_sum = _window_sum(relevance, start, end)
        sentiment_sum = _window_sum(weighted_sentiment, start, end)
        divisor = np.where(relevance_sum > 0, relevance_sum, np.nan)
        sentiment_mean = np.divide(sentiment_sum, divisor, out=np.zeros_like(sentiment_sum), where=np.isfinite(divisor))
        event_divisor = np.where(count > 0, count, np.nan)
        relevance_mean = np.divide(
            relevance_sum,
            event_divisor,
            out=np.zeros_like(relevance_sum),
            where=np.isfinite(event_divisor),
        )
        diversity = np.zeros(len(output), dtype=float)
        for family in source_families:
            family_values = events["source_family"].astype(str).eq(family).astype(float).to_numpy()
            diversity += (_window_sum(family_values, start, end) > 0).astype(float)
        attention = np.clip(np.log1p(count) / math.log(4), 0, 1)
        diversity_weight = 0.75 + 0.25 * np.clip(diversity / 3, 0, 1)
        relevance_weight = np.clip(relevance_mean / 1.75, 0, 1)
        signal = np.clip(sentiment_mean * attention * diversity_weight * relevance_weight, -1, 1)
        output[f"{prefix}event_count_{name}"] = count.astype(float)
        output[f"{prefix}sentiment_mean_{name}"] = sentiment_mean
        output[f"{prefix}relevance_mean_{name}"] = relevance_mean
        output[f"{prefix}source_diversity_{name}"] = diversity
        output[f"{prefix}material_count_{name}"] = _window_sum(material, start, end)
        output[f"{prefix}negative_material_count_{name}"] = _window_sum(negative_material, start, end)
        output[f"{prefix}signal_{name}"] = signal
    return output


def _ticker_file_map(tickers: list[str], directories: list[Path]) -> dict[str, list[Path]]:
    mapping: dict[str, list[Path]] = {}
    for ticker in tickers:
        paths = [directory / f"{ticker}_events.parquet" for directory in directories]
        existing = [path for path in paths if path.exists()]
        if existing:
            mapping[ticker] = existing
    return mapping


def _empty_global_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for window in O1_WINDOWS:
        output[f"global_event_count_{window}"] = 0.0
        output[f"global_sentiment_mean_{window}"] = 0.0
        output[f"global_relevance_mean_{window}"] = 0.0
        output[f"global_source_diversity_{window}"] = 0.0
        output[f"global_material_count_{window}"] = 0.0
        output[f"global_negative_material_count_{window}"] = 0.0
        output[f"global_signal_{window}"] = 0.0
    output["global_latest_event_at_utc"] = pd.NaT
    return output


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "available_at_utc",
            "source_family",
            "sentiment_numeric",
            "relevance",
            "material_event",
            "negative_material",
            "event_id",
        ]
    )


def _event_text(frame: pd.DataFrame) -> pd.Series:
    text = frame["title"].fillna("").astype(str)
    for column in ("summary", "text"):
        if column in frame.columns:
            text = text + " " + frame[column].fillna("").astype(str)
    return text


def _contains_ticker(value: object, ticker: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9])\$?{re.escape(ticker.upper())}(?![A-Z0-9])", str(value).upper()))


def _is_generic_headline(value: object) -> bool:
    lowered = str(value or "").lower()
    return any(
        pattern in lowered
        for pattern in (
            "stocks moving",
            "stock moving",
            "moving higher",
            "moving lower",
            "premarket",
            "pre-market",
            "after-market",
            "biggest stock movers",
            "market summary",
            "trending stocks",
        )
    )


def _event_id(row: pd.Series) -> str:
    payload = "|".join(
        str(row.get(column, ""))
        for column in ("ticker", "published_at_utc", "source", "title", "url")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _window_sum(values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
    return np.asarray(cumulative[end] - cumulative[start], dtype=float)


def _timestamp_ns(series: pd.Series) -> np.ndarray:
    values = pd.to_datetime(series, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    return np.asarray(values, dtype=np.int64)


def _utc_series(series: pd.Series) -> pd.Series:
    return pd.Series(pd.to_datetime(series, errors="coerce", utc=True), index=series.index)


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(timestamp) else pd.Timestamp(timestamp)


def _session_bootstrap_interval(
    frame: pd.DataFrame,
    *,
    value_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    sessions = list(pd.Series(frame["session_date_et"]).dropna().unique())
    if len(sessions) < 2:
        raise DataReadinessError("O1 session bootstrap requires at least two sessions")
    values = {
        session: pd.to_numeric(frame.loc[frame["session_date_et"].eq(session), value_column], errors="coerce").dropna()
        for session in sessions
    }
    random = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled_sessions = random.choice(sessions, size=len(sessions), replace=True)
        samples[index] = float(pd.concat([values[session] for session in sampled_sessions], ignore_index=True).mean())
    point = float(pd.to_numeric(frame[value_column], errors="coerce").mean())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"point": point, "low": float(low), "high": float(high), "iterations": float(iterations), "seed": float(seed)}


def _non_overlapping_groups(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    frame = selected.copy()
    frame["entry_time_utc"] = _utc_series(frame["entry_time_utc"])
    frame["primary_exit_time_utc"] = _utc_series(frame["primary_exit_time_utc"])
    groups = frame.groupby("decision_group_id", observed=True).agg(
        entry_time_utc=("entry_time_utc", "first"),
        primary_exit_time_utc=("primary_exit_time_utc", "first"),
    )
    keep: list[str] = []
    last_exit: pd.Timestamp | None = None
    for group_id, row in groups.sort_values("entry_time_utc").iterrows():
        entry = pd.Timestamp(row["entry_time_utc"])
        exit_time = pd.Timestamp(row["primary_exit_time_utc"])
        if last_exit is not None and entry <= last_exit:
            continue
        keep.append(str(group_id))
        last_exit = exit_time
    return frame[frame["decision_group_id"].astype(str).isin(keep)].copy()


def _portfolio_returns(selected: pd.DataFrame) -> pd.Series:
    if selected.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(
        selected.groupby(["session_date_et", "decision_group_id"], observed=True)["path_realized_return_net"].mean(),
        errors="coerce",
    ).dropna()


def _profit_factor(returns: pd.Series) -> float | None:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    if losses == 0:
        return None if gains == 0 else gains / 1e-12
    return gains / losses


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns.clip(lower=-0.999999)).cumprod()
    return float((1 - equity / equity.cummax()).max())
