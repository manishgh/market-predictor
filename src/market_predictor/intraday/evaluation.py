from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from market_predictor.execution_policy import (
    ExecutionCostPolicy,
    execution_cost_fraction,
    flat_stress_surcharge,
)
from market_predictor.intraday.contracts import (
    INTRADAY_REQUIRED_MARKET_REGIMES,
    downside_target_column,
    excess_return_column,
    net_return_column,
    opportunity_target_column,
)
from market_predictor.prediction_policy import (
    PredictionSelectionPolicy,
    expected_calibration_error,
    finite_or_none,
    select_intraday_candidates,
)
from market_predictor.regime_evidence import (
    REGIME_BOOTSTRAP_ITERATIONS,
    REGIME_BOOTSTRAP_SEED,
    session_block_mean_interval,
)


def prediction_evidence(
    frame: pd.DataFrame,
    *,
    opportunity_raw: np.ndarray,
    opportunity_probability: np.ndarray,
    downside_raw: np.ndarray,
    downside_probability: np.ndarray,
    scope: str,
    horizon_minutes: int,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "session_date_et",
        "decision_group_id",
        "decision_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "label_window_end_utc",
        "independent_event_id",
        "concurrent_label_count",
        "overlap_weight",
        "market_regime",
        "sector",
        "primary_benchmark",
        "catalyst_eligible",
        "event_count_2h",
        "event_relevance_mean_2h",
        "low_relevance_event_fraction_2h",
        opportunity_target_column(horizon_minutes),
        downside_target_column(horizon_minutes),
        net_return_column(horizon_minutes),
        f"path_realized_return_gross_{horizon_minutes}m",
        f"path_spy_return_{horizon_minutes}m",
        f"path_qqq_return_{horizon_minutes}m",
        f"path_sector_return_{horizon_minutes}m",
        "entry_price",
        "entry_dollar_volume",
        "entry_atr_pct",
        excess_return_column(horizon_minutes, "spy"),
        excess_return_column(horizon_minutes, "qqq"),
        excess_return_column(horizon_minutes, "sector"),
    ]
    output = frame.loc[:, [column for column in columns if column in frame.columns]].copy()
    output = output.reset_index(drop=True)
    output["opportunity_raw_probability"] = opportunity_raw
    output["intraday_opportunity_probability"] = opportunity_probability
    output["downside_raw_probability"] = downside_raw
    output["intraday_downside_probability"] = downside_probability
    output["intraday_opportunity_prediction"] = (opportunity_probability >= 0.5).astype("int8")
    output["intraday_downside_prediction"] = (downside_probability >= 0.5).astype("int8")
    output["validation_scope"] = scope
    return output


def classification_metrics(target: pd.Series, probability: pd.Series) -> dict[str, float]:
    y = pd.to_numeric(target, errors="coerce").astype(int)
    p = pd.to_numeric(probability, errors="coerce").clip(0, 1)
    prediction = p.ge(0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y,
        prediction,
        average="binary",
        zero_division=0,
    )
    auc = float(roc_auc_score(y, p)) if y.nunique() > 1 else float("nan")
    average_precision = float(average_precision_score(y, p)) if y.nunique() > 1 else float("nan")
    base = float(y.mean())
    cutoff = float(p.quantile(0.9))
    top_rate = float(y[p.ge(cutoff)].mean())
    return {
        "roc_auc": auc,
        "average_precision": average_precision,
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "expected_calibration_error": expected_calibration_error(y, p),
        "base_positive_rate": base,
        "top_decile_positive_rate": top_rate,
        "top_decile_lift": top_rate / base if base > 0 else float("nan"),
    }


def overlap_evidence_summary(
    weights: pd.Series,
    independent_event_ids: pd.Series,
) -> dict[str, float | int]:
    """Summarize overlapping-label evidence conservatively.

    Summed average uniqueness measures label-equivalent information. The
    independent-event count is the greedy non-overlapping subset emitted by
    label construction. Effective evidence is bounded by both; independent
    sessions are reported and gated separately by the trainer.
    """

    if len(weights) != len(independent_event_ids):
        raise ValueError("overlap weights and independent event IDs must align")
    values = pd.to_numeric(weights, errors="coerce")
    if bool((values.isna() | values.le(0) | values.gt(1)).any()):
        raise ValueError("overlap weights must be finite and in the interval (0, 1]")
    event_ids = independent_event_ids.astype("string")
    populated = event_ids[event_ids.notna() & event_ids.str.strip().ne("")]
    summed_uniqueness = float(values.sum())
    independent_events = int(populated.nunique())
    return {
        "summed_label_uniqueness": summed_uniqueness,
        "independent_event_count": independent_events,
        "effective_sample_size": min(summed_uniqueness, float(independent_events)),
    }


def phase_economics(
    predictions: pd.DataFrame,
    *,
    horizon_minutes: int,
    decision_interval_minutes: int,
    top_k: int,
    downside_ceiling: float,
    max_trades_per_session: int,
    scope: str,
    cohort_column: str | None = None,
    policy: ExecutionCostPolicy | None = None,
    cost_stress: float = 1.0,
) -> pd.DataFrame:
    phase_count = max(1, math.ceil(horizon_minutes / decision_interval_minutes))
    groups = (
        predictions[["session_date_et", "decision_group_id", "decision_time_utc"]]
        .drop_duplicates("decision_group_id")
        .sort_values(["session_date_et", "decision_time_utc"], kind="stable")
    )
    groups["group_ordinal"] = groups.groupby("session_date_et", sort=False).cumcount()
    ordinal = groups.set_index("decision_group_id")["group_ordinal"]
    records: list[dict[str, object]] = []
    selection_policy = PredictionSelectionPolicy(
        intraday_top_k=top_k,
        intraday_downside_ceiling=downside_ceiling,
        intraday_max_trades_per_session=max_trades_per_session,
    )
    selected_all = select_intraday_candidates(
        predictions,
        policy=selection_policy,
        opportunity_column="intraday_opportunity_probability",
        downside_column="intraday_downside_probability",
    )
    for phase in range(phase_count):
        phase_groups = set(ordinal[ordinal.mod(phase_count).eq(phase)].index)
        selected = selected_all[
            selected_all["decision_group_id"].isin(phase_groups)
        ].copy()
        records.append(
            _economic_record(
                selected,
                horizon_minutes=horizon_minutes,
                scope=scope,
                phase=phase,
                policy=policy,
                cost_stress=cost_stress,
            )
        )
        if cohort_column is not None:
            if cohort_column not in selected.columns:
                raise ValueError(f"cohort column {cohort_column} is missing")
            for cohort, cohort_rows in selected.groupby(cohort_column, sort=True):
                records.append(
                    _economic_record(
                        cohort_rows,
                        horizon_minutes=horizon_minutes,
                        scope=f"{scope}:{cohort}",
                        phase=phase,
                        policy=policy,
                        cost_stress=cost_stress,
                    )
                )
    return pd.DataFrame(records)


def conservative_economics(economics: pd.DataFrame) -> pd.DataFrame:
    finite_ratio = pd.to_numeric(economics["return_drawdown_ratio"], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    return pd.DataFrame(
        [
            {
                "scope": "all_validation_scopes",
                "phase": "conservative",
                "selected_trades": int(pd.to_numeric(economics["selected_trades"], errors="coerce").min()),
                "selected_decision_groups": int(pd.to_numeric(economics["selected_decision_groups"], errors="coerce").min()),
                "avg_trade_return": float(pd.to_numeric(economics["avg_trade_return"], errors="coerce").min()),
                "avg_excess_return_vs_spy": float(pd.to_numeric(economics["avg_excess_return_vs_spy"], errors="coerce").min()),
                "avg_excess_return_vs_qqq": float(pd.to_numeric(economics["avg_excess_return_vs_qqq"], errors="coerce").min()),
                "avg_excess_return_vs_sector": float(pd.to_numeric(economics["avg_excess_return_vs_sector"], errors="coerce").min()),
                "avg_trade_return_ci_low": float(
                    pd.to_numeric(
                        economics["avg_trade_return_ci_low"],
                        errors="coerce",
                    ).min()
                ),
                "avg_excess_return_vs_spy_ci_low": float(
                    pd.to_numeric(
                        economics["avg_excess_return_vs_spy_ci_low"],
                        errors="coerce",
                    ).min()
                ),
                "win_rate": float(pd.to_numeric(economics["win_rate"], errors="coerce").min()),
                "profit_factor": float(pd.to_numeric(economics["profit_factor"], errors="coerce").min()),
                "cumulative_return": float(pd.to_numeric(economics["cumulative_return"], errors="coerce").min()),
                "max_drawdown": float(pd.to_numeric(economics["max_drawdown"], errors="coerce").max()),
                "return_drawdown_ratio": (
                    float(finite_ratio.min()) if finite_ratio.notna().any() else 0.0
                ),
                "negative_session_rate": float(pd.to_numeric(economics["negative_session_rate"], errors="coerce").max()),
                "average_turnover": float(pd.to_numeric(economics["average_turnover"], errors="coerce").max()),
                "sessions": int(pd.to_numeric(economics["sessions"], errors="coerce").min()),
            }
        ]
    )


def regime_audit(
    predictions: pd.DataFrame,
    *,
    horizon_minutes: int,
    decision_interval_minutes: int,
    top_k: int,
    downside_ceiling: float,
    max_trades_per_session: int,
    target_column: str,
    min_regime_sessions: int = 5,
    min_regime_trades: int = 20,
    required_regimes: tuple[str, ...] = INTRADAY_REQUIRED_MARKET_REGIMES,
    policy: ExecutionCostPolicy | None = None,
) -> pd.DataFrame:
    """Per-regime selected-policy economics, calibration, and evidence status.

    Sparse regimes (too few independent sessions or selected trades) are
    ``insufficient_evidence`` and never counted as passing; populated losing
    regimes remain visible to the worst-regime economic gate.
    """

    labelled = predictions.assign(_regime=predictions["market_regime"].fillna("unknown").astype(str))
    counts = labelled["_regime"].value_counts()
    total = int(counts.sum())
    summary: dict[str, object] = {
        "scope": "summary",
        "regimes_present": int(counts.size),
        "max_single_regime_share": float(counts.max() / total) if total else 1.0,
        "rows": total,
        "evidence_status": "summary",
        "required_regime": False,
        "minimum_sessions": min_regime_sessions,
        "minimum_trades": min_regime_trades,
        "selected_trades": float("nan"),
        "sessions": float("nan"),
        "avg_trade_return": float("nan"),
        "avg_excess_return_vs_spy": float("nan"),
        "avg_trade_return_ci_low": float("nan"),
        "avg_excess_return_vs_spy_ci_low": float("nan"),
        "max_drawdown": float("nan"),
        "calibration_error": float("nan"),
    }
    details: list[dict[str, object]] = []
    observed_regimes = list(labelled["_regime"].drop_duplicates())
    audit_regimes = [*required_regimes, *(regime for regime in observed_regimes if regime not in required_regimes)]
    for regime in audit_regimes:
        subset = labelled[labelled["_regime"].eq(regime)]
        required = regime in required_regimes
        if subset.empty:
            details.append(
                {
                    "scope": f"regime:{regime}",
                    "regimes_present": int(counts.size),
                    "max_single_regime_share": 0.0,
                    "rows": 0,
                    "evidence_status": "missing_required" if required else "insufficient_evidence",
                    "required_regime": required,
                    "minimum_sessions": min_regime_sessions,
                    "minimum_trades": min_regime_trades,
                    "selected_trades": 0,
                    "sessions": 0,
                    "avg_trade_return": None,
                    "avg_excess_return_vs_spy": None,
                    "avg_trade_return_ci_low": None,
                    "avg_excess_return_vs_spy_ci_low": None,
                    "max_drawdown": None,
                    "calibration_error": None,
                }
            )
            continue
        selection_policy = PredictionSelectionPolicy(
            intraday_top_k=top_k,
            intraday_downside_ceiling=downside_ceiling,
            intraday_max_trades_per_session=max_trades_per_session,
        )
        selected_regime = select_intraday_candidates(
            subset,
            policy=selection_policy,
            opportunity_column="intraday_opportunity_probability",
            downside_column="intraday_downside_probability",
        )
        regime_economics = phase_economics(
            subset,
            horizon_minutes=horizon_minutes,
            decision_interval_minutes=decision_interval_minutes,
            top_k=top_k,
            downside_ceiling=downside_ceiling,
            max_trades_per_session=max_trades_per_session,
            scope=f"regime:{regime}",
            policy=policy,
        )
        populated_economics = regime_economics[
            pd.to_numeric(
                regime_economics["selected_trades"],
                errors="coerce",
            ).gt(0)
        ]
        record = (
            conservative_economics(populated_economics).iloc[0]
            if not populated_economics.empty
            else pd.Series(dtype=object)
        )
        sessions = int(selected_regime["session_date_et"].nunique())
        trades = int(len(selected_regime))
        status = "sufficient" if sessions >= min_regime_sessions and trades >= min_regime_trades else "insufficient_evidence"
        details.append(
            {
                "scope": f"regime:{regime}",
                "regimes_present": int(counts.size),
                "max_single_regime_share": float(counts.get(regime, 0) / total) if total else 1.0,
                "rows": int(counts.get(regime, 0)),
                "evidence_status": status,
                "required_regime": required,
                "minimum_sessions": min_regime_sessions,
                "minimum_trades": min_regime_trades,
                "selected_trades": trades,
                "sessions": sessions,
                "avg_trade_return": finite_or_none(record.get("avg_trade_return")),
                "avg_excess_return_vs_spy": finite_or_none(record.get("avg_excess_return_vs_spy")),
                "avg_trade_return_ci_low": finite_or_none(
                    record.get("avg_trade_return_ci_low")
                ),
                "avg_excess_return_vs_spy_ci_low": finite_or_none(
                    record.get("avg_excess_return_vs_spy_ci_low")
                ),
                "max_drawdown": finite_or_none(record.get("max_drawdown")),
                "calibration_error": (
                    expected_calibration_error(subset[target_column], subset["intraday_opportunity_probability"])
                    if len(subset)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame([summary, *details])


def catalyst_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    eligible = predictions.get("catalyst_eligible", pd.Series(False, index=predictions.index))
    catalyst_eligible = eligible.fillna(False).astype(bool)
    event_count = pd.to_numeric(predictions.get("event_count_2h"), errors="coerce").fillna(0.0)
    low_fraction = pd.to_numeric(
        predictions.get("low_relevance_event_fraction_2h"),
        errors="coerce",
    ).fillna(0.0)
    total_events = float(event_count.sum())
    return pd.DataFrame(
        [
            {
                "has_catalyst_features": "event_count_2h" in predictions.columns,
                "rows": len(predictions),
                "catalyst_eligible_rows": int(catalyst_eligible.sum()),
                "catalyst_coverage_rate": (float(catalyst_eligible.mean()) if len(predictions) else 0.0),
                "rows_with_catalyst": int(event_count.gt(0).sum()),
                "catalyst_event_rate": float(event_count.gt(0).mean()) if len(predictions) else 0.0,
                "low_relevance_event_rate": (float((event_count * low_fraction).sum() / total_events) if total_events > 0 else 0.0),
                "included_in_estimators": False,
                "alignment_error_total": 0,
            }
        ]
    )


def _economic_record(
    selected: pd.DataFrame,
    *,
    horizon_minutes: int,
    scope: str,
    phase: int,
    policy: ExecutionCostPolicy | None = None,
    cost_stress: float = 1.0,
) -> dict[str, object]:
    gross_column = f"path_realized_return_gross_{horizon_minutes}m"
    cost = execution_cost_fraction(
        selected,
        price_column="entry_price",
        atr_pct_column="entry_atr_pct",
        policy=policy,
        stress=cost_stress,
    )
    if cost is not None and gross_column in selected.columns:
        base_return = pd.to_numeric(selected[gross_column], errors="coerce")
        cost_series = cost
        net = base_return - cost_series
        excess = {
            benchmark: (
                net
                - pd.to_numeric(
                    selected[
                        f"path_{benchmark}_return_{horizon_minutes}m"
                    ],
                    errors="coerce",
                )
            ).dropna()
            for benchmark in ("spy", "qqq", "sector")
        }
    else:
        base_return = pd.to_numeric(selected.get(net_return_column(horizon_minutes)), errors="coerce")
        cost_series = pd.Series(flat_stress_surcharge(cost_stress, policy), index=selected.index)
        net = base_return - cost_series
        excess = {
            benchmark: (
                pd.to_numeric(
                    selected.get(
                        excess_return_column(
                            horizon_minutes,
                            benchmark,
                        )
                    ),
                    errors="coerce",
                )
                - cost_series
            ).dropna()
            for benchmark in ("spy", "qqq", "sector")
        }
    work = selected.assign(_net=net)
    returns = work["_net"].dropna()
    group_returns = (
        work.groupby(["session_date_et", "decision_group_id"], sort=False)["_net"].mean().dropna()
    )
    # Allocation-aware: each decision group is one sequential full-capital, equal-weighted
    # deployment (phases are non-overlapping in time), so a session compounds its groups
    # rather than summing unconstrained overlapping trade returns.
    session_returns = (1.0 + group_returns).groupby(level="session_date_et").prod() - 1.0
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    equity = (1.0 + session_returns).cumprod()
    cumulative = float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
    drawdown = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else float("nan")
    drawdown_abs = abs(drawdown) if np.isfinite(drawdown) else float("nan")
    return_interval = session_block_mean_interval(
        work.loc[returns.index, "session_date_et"],
        returns,
    )
    spy_excess = excess["spy"]
    spy_interval = session_block_mean_interval(
        work.loc[spy_excess.index, "session_date_et"],
        spy_excess,
    )
    return {
        "scope": scope,
        "phase": phase,
        "selected_trades": int(len(returns)),
        "selected_decision_groups": int(selected["decision_group_id"].nunique()),
        "avg_trade_return": float(returns.mean()) if not returns.empty else float("nan"),
        "avg_trade_return_ci_low": return_interval["low"],
        "avg_excess_return_vs_spy_ci_low": spy_interval["low"],
        "confidence_method": "session_block_bootstrap_95pct",
        "confidence_iterations": REGIME_BOOTSTRAP_ITERATIONS,
        "confidence_seed": REGIME_BOOTSTRAP_SEED,
        **{
            f"avg_excess_return_vs_{benchmark}": (float(values.mean()) if not values.empty else float("nan"))
            for benchmark, values in excess.items()
        },
        "win_rate": float(returns.gt(0).mean()) if not returns.empty else float("nan"),
        "profit_factor": gains / losses if losses > 0 else float("inf") if gains > 0 else float("nan"),
        "cumulative_return": cumulative,
        "max_drawdown": drawdown_abs,
        "return_drawdown_ratio": (cumulative / drawdown_abs if drawdown_abs > 0 else float("inf") if cumulative > 0 else 0.0),
        "negative_session_rate": (float(session_returns.lt(0).mean()) if not session_returns.empty else float("nan")),
        "average_turnover": _average_turnover(selected),
        "sessions": int(len(session_returns)),
    }


def _average_turnover(selected: pd.DataFrame) -> float:
    if selected.empty:
        return float("nan")
    turnovers: list[float] = []
    for _, session in selected.groupby("session_date_et", sort=False):
        prior: set[str] | None = None
        for _, group in session.sort_values("decision_time_utc").groupby(
            "decision_group_id",
            sort=False,
        ):
            current = set(group["ticker"].astype(str))
            if prior is not None:
                union = prior | current
                turnovers.append(len(prior ^ current) / len(union) if union else 0.0)
            prior = current
    return float(np.mean(turnovers)) if turnovers else 0.0
