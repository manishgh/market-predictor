from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.partitions import assert_development_only
from market_predictor.v3.schema import FrozenContract

FAILURE_ATTRIBUTION_SCHEMA = "ml_v3.failure_attribution.v1"
PRIMARY_OUTCOME = "net_excess_qqq_60m"
OUTCOME_COLUMNS = (
    "net_excess_qqq_30m",
    "net_excess_qqq_60m",
    "net_excess_qqq_120m",
    "net_excess_qqq_to_close",
    "net_excess_sector_30m",
    "net_excess_sector_60m",
    "net_excess_sector_120m",
    "net_excess_sector_to_close",
    "net_return_30m",
    "net_return_60m",
    "net_return_120m",
    "net_return_to_close",
)

FAILURE_ATTRIBUTION_DATASET_COLUMNS = [
    "ticker",
    "decision_time_utc",
    "decision_group_id",
    "session_date_et",
    "sector",
    "industry",
    "market_cap_bucket",
    "liquidity_bucket",
    "session_progress",
    "regime_risk_on",
    "regime_risk_off",
    "regime_high_volatility",
    "xs_rank_dollar_volume",
    "xs_rank_atr_pct",
    "mfe_60m",
    "mae_60m",
    "ranking_target",
    *OUTCOME_COLUMNS,
]


class FailureAttributionConfig(FrozenContract):
    family: str = "R1"
    top_k: int = Field(default=10, ge=1)
    bootstrap_iterations: int = Field(default=1_000, ge=100, le=100_000)
    bootstrap_seed: int = 42
    minimum_stratum_rows: int = Field(default=100, ge=1)
    minimum_stratum_sessions: int = Field(default=20, ge=1)


def build_failure_attribution(
    predictions: pd.DataFrame,
    development: pd.DataFrame,
    *,
    dataset_fingerprint: str,
    config: FailureAttributionConfig = FailureAttributionConfig(),
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Explain a rejected ranker without selecting a replacement on inspected data."""
    prediction_required = {
        "ticker",
        "decision_time_utc",
        "decision_group_id",
        "session_date_et",
        "audit_scope",
        "family",
        "model_run_id",
        "score",
        "ranking_target",
        "fold",
    }
    missing_predictions = sorted(prediction_required.difference(predictions.columns))
    if missing_predictions:
        raise DataReadinessError(
            f"failure-attribution predictions missing columns: {', '.join(missing_predictions)}"
        )
    missing_development = sorted(set(FAILURE_ATTRIBUTION_DATASET_COLUMNS).difference(development.columns))
    if missing_development:
        raise DataReadinessError(
            f"failure-attribution dataset missing columns: {', '.join(missing_development)}"
        )

    data = predictions.loc[predictions["family"].astype(str).eq(config.family)].copy()
    if data.empty:
        raise DataReadinessError(f"failure attribution requires {config.family} predictions")
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["decision_time_utc"] = _utc_series(data["decision_time_utc"])
    if bool(data["decision_time_utc"].isna().any()):
        raise DataReadinessError("failure-attribution predictions contain invalid decision timestamps")
    assert_development_only(data)
    scopes = set(data["audit_scope"].astype(str))
    required_scopes = {"walk_forward", "ticker_holdout"}
    if scopes != required_scopes:
        raise DataReadinessError(
            f"failure attribution requires exactly {sorted(required_scopes)}; received {sorted(scopes)}"
        )
    run_ids = set(data["model_run_id"].astype(str))
    if len(run_ids) != 1:
        raise DataReadinessError("failure attribution requires exactly one model_run_id")
    identity = ["ticker", "decision_time_utc", "decision_group_id", "audit_scope"]
    if bool(data.duplicated(identity).any()):
        raise DataReadinessError("failure-attribution predictions contain duplicate decision identities")
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["ranking_target"] = pd.to_numeric(data["ranking_target"], errors="coerce")
    if not bool(np.isfinite(data[["score", "ranking_target"]].to_numpy(dtype=float)).all()):
        raise DataReadinessError("failure-attribution scores and ranking targets must be finite")

    context = development.loc[:, FAILURE_ATTRIBUTION_DATASET_COLUMNS].copy()
    context["ticker"] = context["ticker"].astype(str).str.upper().str.strip()
    context["decision_time_utc"] = _utc_series(context["decision_time_utc"])
    if bool(context["decision_time_utc"].isna().any()):
        raise DataReadinessError("failure-attribution dataset contains invalid decision timestamps")
    assert_development_only(context)
    context_identity = ["ticker", "decision_time_utc", "decision_group_id"]
    if bool(context.duplicated(context_identity).any()):
        raise DataReadinessError("failure-attribution dataset contains duplicate decision identities")
    context = context.rename(
        columns={
            "session_date_et": "dataset_session_date_et",
            "ranking_target": "dataset_ranking_target",
        }
    )
    joined = data.merge(context, on=context_identity, how="left", validate="many_to_one", indicator=True)
    unmatched = int(joined["_merge"].ne("both").sum())
    if unmatched:
        raise DataReadinessError(f"failure attribution has {unmatched} predictions without frozen dataset rows")
    joined = joined.drop(columns="_merge")
    if not bool(joined["session_date_et"].astype(str).eq(joined["dataset_session_date_et"].astype(str)).all()):
        raise DataReadinessError("prediction and frozen dataset session identities differ")
    _validate_joined_targets(joined)
    joined = _add_diagnostic_dimensions(joined)

    selected = _select_top_k(joined, config.top_k)
    if selected.empty:
        raise DataReadinessError("failure attribution selected no top-k rows")
    scope_reports = {
        str(scope): _scope_report(scope_data, selected[selected["audit_scope"].eq(scope)], config=config)
        for scope, scope_data in joined.groupby("audit_scope", sort=True, observed=True)
    }
    strata = _build_strata(joined, selected, config=config)
    report: dict[str, Any] = {
        "schema": FAILURE_ATTRIBUTION_SCHEMA,
        "diagnostic_only": True,
        "promotion_eligible": False,
        "shadow_data_accessed": False,
        "config": config.model_dump(mode="json"),
        "dataset_fingerprint": dataset_fingerprint,
        "model_run_id": next(iter(run_ids)),
        "readiness": {
            "prediction_rows": len(data),
            "joined_rows": len(joined),
            "selected_rows": len(selected),
            "tickers": int(joined["ticker"].nunique()),
            "sessions": int(joined["session_date_et"].nunique()),
            "decision_groups": int(joined["decision_group_id"].nunique()),
            "future_or_shadow_rows": 0,
            "duplicate_prediction_identities": 0,
            "unmatched_dataset_rows": 0,
        },
        "scope_summary": scope_reports,
        "strata_summary": {
            "rows": len(strata),
            "reliable_rows": int(strata["meets_minimum_evidence"].sum()),
            "positive_reliable_rows": int(
                (strata["meets_minimum_evidence"] & strata["selected_mean_excess_60m"].gt(0)).sum()
            ),
            "warning": "Strata are explanatory diagnostics and must not become filters on this inspected evidence.",
        },
        "next_decision": "Use this report to freeze one new development hypothesis; do not open C9 shadow data.",
    }
    selected = selected.sort_values(["audit_scope", "decision_time_utc", "ticker"]).reset_index(drop=True)
    return report, strata, selected


def _validate_joined_targets(joined: pd.DataFrame) -> None:
    prediction_target = pd.to_numeric(joined["ranking_target"], errors="coerce")
    dataset_target = pd.to_numeric(joined["dataset_ranking_target"], errors="coerce")
    if bool(dataset_target.isna().any()):
        raise DataReadinessError("frozen dataset ranking target is incomplete")
    maximum_difference = float((prediction_target - dataset_target).abs().max())
    if maximum_difference > 1e-12:
        raise DataReadinessError(
            f"prediction and frozen dataset ranking targets differ by as much as {maximum_difference}"
        )
    for column in OUTCOME_COLUMNS:
        values = pd.to_numeric(joined[column], errors="coerce")
        if bool(values.isna().any()) or not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
            raise DataReadinessError(f"failure-attribution outcome is incomplete or non-finite: {column}")


def _add_diagnostic_dimensions(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    decision_et = output["decision_time_utc"].dt.tz_convert("America/New_York")
    minute = decision_et.dt.hour * 60 + decision_et.dt.minute
    output["month"] = decision_et.dt.strftime("%Y-%m")
    output["time_bucket"] = pd.cut(
        minute,
        bins=[570, 630, 720, 840, 960],
        labels=["opening", "late_morning", "midday", "afternoon"],
        right=False,
    ).astype("object")
    if bool(output["time_bucket"].isna().any()):
        raise DataReadinessError("failure attribution found decision rows outside configured market time buckets")
    output["market_regime"] = np.select(
        [
            pd.to_numeric(output["regime_high_volatility"], errors="coerce").fillna(0).gt(0),
            pd.to_numeric(output["regime_risk_off"], errors="coerce").fillna(0).gt(0),
            pd.to_numeric(output["regime_risk_on"], errors="coerce").fillna(0).gt(0),
        ],
        ["high_volatility", "risk_off", "risk_on"],
        default="neutral",
    )
    output["liquidity_rank_quartile"] = _rank_quartile(output["xs_rank_dollar_volume"])
    output["volatility_rank_quartile"] = _rank_quartile(output["xs_rank_atr_pct"])
    output["fold_bucket"] = output["fold"].fillna(-1).astype(str)
    for column in ("sector", "market_cap_bucket", "liquidity_bucket"):
        output[column] = output[column].fillna("unknown").astype(str)
    return output


def _rank_quartile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if bool(numeric.isna().any()):
        raise DataReadinessError("failure attribution requires complete cross-sectional rank features")
    return pd.cut(
        numeric,
        bins=[-np.inf, 0.25, 0.50, 0.75, np.inf],
        labels=["q1_low", "q2", "q3", "q4_high"],
        include_lowest=True,
    ).astype(str)


def _select_top_k(frame: pd.DataFrame, top_k: int) -> pd.DataFrame:
    selections: list[pd.DataFrame] = []
    for _, group in frame.groupby(["audit_scope", "decision_group_id"], sort=False, observed=True):
        selections.append(group.nlargest(min(top_k, len(group)), "score"))
    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(columns=frame.columns)


def _scope_report(
    scope_data: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    config: FailureAttributionConfig,
) -> dict[str, Any]:
    horizon_reports = {
        outcome: _outcome_report(scope_data, selected, outcome=outcome, config=config)
        for outcome in OUTCOME_COLUMNS
    }
    rank_correlations: list[float] = []
    for _, group in scope_data.groupby("decision_group_id", sort=False, observed=True):
        if len(group) < 2:
            continue
        score_rank = group["score"].rank(method="average").to_numpy(dtype=float)
        target_rank = group[PRIMARY_OUTCOME].rank(method="average").to_numpy(dtype=float)
        correlation = float(np.corrcoef(score_rank, target_rank)[0, 1])
        if np.isfinite(correlation):
            rank_correlations.append(correlation)

    score_percentile = scope_data.groupby("decision_group_id", observed=True)["score"].rank(
        method="first", pct=True
    )
    decile = np.ceil(score_percentile * 10).clip(1, 10).astype(int)
    decile_frame = scope_data.assign(score_decile=decile)
    deciles = [
        {
            "score_decile": int(value),
            "rows": len(group),
            "mean_excess_60m": float(pd.to_numeric(group[PRIMARY_OUTCOME]).mean()),
            "positive_rate_60m": float(pd.to_numeric(group[PRIMARY_OUTCOME]).gt(0).mean()),
        }
        for value, group in decile_frame.groupby("score_decile", sort=True, observed=True)
    ]
    return {
        "rows": len(scope_data),
        "selected_rows": len(selected),
        "tickers": int(scope_data["ticker"].nunique()),
        "sessions": int(scope_data["session_date_et"].nunique()),
        "decision_groups": int(scope_data["decision_group_id"].nunique()),
        "mean_cross_section_rank_correlation_60m": (
            float(np.mean(rank_correlations)) if rank_correlations else None
        ),
        "outcomes": horizon_reports,
        "score_deciles": deciles,
    }


def _outcome_report(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    outcome: str,
    config: FailureAttributionConfig,
) -> dict[str, Any]:
    population_groups = population.groupby("decision_group_id", observed=True)[outcome].mean().rename("population")
    selected_groups = selected.groupby("decision_group_id", observed=True)[outcome].mean().rename("selected")
    oracle_groups = (
        population.groupby("decision_group_id", observed=True)[outcome]
        .nlargest(config.top_k)
        .groupby(level=0)
        .mean()
        .rename("oracle")
    )
    sessions = population.groupby("decision_group_id", observed=True)["session_date_et"].first()
    paired = pd.concat([sessions, population_groups, selected_groups, oracle_groups], axis=1).dropna()
    paired["delta"] = paired["selected"] - paired["population"]
    return {
        "groups": len(paired),
        "selected_mean": float(paired["selected"].mean()),
        "population_group_mean": float(paired["population"].mean()),
        "selection_delta": float(paired["delta"].mean()),
        "oracle_top_k_mean": float(paired["oracle"].mean()),
        "selected_positive_group_rate": float(paired["selected"].gt(0).mean()),
        "selected_mean_interval": _session_bootstrap_interval(
            paired,
            value_column="selected",
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
        ),
        "selection_delta_interval": _session_bootstrap_interval(
            paired,
            value_column="delta",
            iterations=config.bootstrap_iterations,
            seed=config.bootstrap_seed,
        ),
    }


def _build_strata(
    population: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    config: FailureAttributionConfig,
) -> pd.DataFrame:
    dimensions = (
        "month",
        "fold_bucket",
        "time_bucket",
        "market_regime",
        "sector",
        "market_cap_bucket",
        "liquidity_bucket",
        "liquidity_rank_quartile",
        "volatility_rank_quartile",
    )
    rows: list[dict[str, Any]] = []
    for scope, scope_selected in selected.groupby("audit_scope", sort=True, observed=True):
        scope_population = population[population["audit_scope"].eq(scope)]
        for dimension in dimensions:
            for value, selected_stratum in scope_selected.groupby(dimension, sort=True, observed=True):
                population_stratum = scope_population[scope_population[dimension].eq(value)]
                selected_excess = pd.to_numeric(selected_stratum[PRIMARY_OUTCOME], errors="coerce")
                population_excess = pd.to_numeric(population_stratum[PRIMARY_OUTCOME], errors="coerce")
                sessions = int(selected_stratum["session_date_et"].nunique())
                rows.append(
                    {
                        "audit_scope": str(scope),
                        "dimension": dimension,
                        "value": str(value),
                        "selected_rows": len(selected_stratum),
                        "population_rows": len(population_stratum),
                        "decision_groups": int(selected_stratum["decision_group_id"].nunique()),
                        "sessions": sessions,
                        "selected_mean_excess_60m": float(selected_excess.mean()),
                        "population_mean_excess_60m": float(population_excess.mean()),
                        "selection_delta_excess_60m": float(selected_excess.mean() - population_excess.mean()),
                        "selected_positive_rate_60m": float(selected_excess.gt(0).mean()),
                        "selected_mean_return_60m": float(
                            pd.to_numeric(selected_stratum["net_return_60m"], errors="coerce").mean()
                        ),
                        "selected_mean_mfe_60m": float(
                            pd.to_numeric(selected_stratum["mfe_60m"], errors="coerce").mean()
                        ),
                        "selected_mean_mae_60m": float(
                            pd.to_numeric(selected_stratum["mae_60m"], errors="coerce").mean()
                        ),
                        "meets_minimum_evidence": (
                            len(selected_stratum) >= config.minimum_stratum_rows
                            and sessions >= config.minimum_stratum_sessions
                        ),
                    }
                )
    return pd.DataFrame(rows).sort_values(["audit_scope", "dimension", "value"]).reset_index(drop=True)


def _session_bootstrap_interval(
    frame: pd.DataFrame,
    *,
    value_column: str,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    values = pd.DataFrame(
        {
            "session_date_et": frame["session_date_et"],
            "value": pd.to_numeric(frame[value_column], errors="coerce"),
        }
    ).dropna()
    blocks = values.groupby("session_date_et", sort=False, observed=True)["value"].agg(["sum", "count"])
    if len(blocks) < 2:
        raise DataReadinessError("failure attribution requires at least two sessions for bootstrap")
    block_sums = blocks["sum"].to_numpy(dtype=float)
    block_counts = blocks["count"].to_numpy(dtype=float)
    random = np.random.default_rng(seed)
    sampled_indices = random.integers(0, len(blocks), size=(iterations, len(blocks)))
    sample_sums = block_sums[sampled_indices].sum(axis=1)
    sample_counts = block_counts[sampled_indices].sum(axis=1)
    samples = np.divide(sample_sums, sample_counts, out=np.zeros_like(sample_sums), where=sample_counts > 0)
    point = float(values["value"].mean())
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "iterations": float(iterations),
        "seed": float(seed),
    }


def _utc_series(series: pd.Series) -> pd.Series:
    return pd.Series(pd.to_datetime(series, errors="coerce", utc=True), index=series.index)
