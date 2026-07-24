from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
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
from market_predictor.prediction_policy import (
    PredictionSelectionPolicy,
    expected_calibration_error,
    finite_or_none,
    select_swing_candidates,
)
from market_predictor.swing.contracts import (
    swing_excess_column,
    swing_net_return_column,
    swing_target_column,
)


def prediction_evidence(
    frame: pd.DataFrame,
    *,
    raw_probability: np.ndarray,
    probability: np.ndarray,
    scope: str,
    horizon: int,
) -> pd.DataFrame:
    columns = [
        "ticker",
        "session_date_et",
        "decision_group_id",
        "decision_time_utc",
        "market_regime",
        "sector",
        "primary_benchmark",
        "event_count_3d",
        "event_relevance_mean_3d",
        "low_relevance_event_fraction_3d",
        swing_target_column(horizon),
        swing_net_return_column(horizon),
        f"future_gross_return_{horizon}d",
        "close",
        "atr_pct_14",
        swing_excess_column(horizon, "spy"),
        swing_excess_column(horizon, "qqq"),
        swing_excess_column(horizon, "sector"),
    ]
    available = [column for column in columns if column in frame.columns]
    output = frame.loc[:, available].copy().reset_index(drop=True)
    output["raw_probability"] = raw_probability
    output["swing_probability"] = probability
    output["swing_prediction"] = (probability >= 0.5).astype("int8")
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
    base = float(y.mean())
    cutoff = float(p.quantile(0.9))
    top_rate = float(y[p.ge(cutoff)].mean())
    return {
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
        "base_positive_rate": base,
        "top_decile_positive_rate": top_rate,
        "top_decile_lift": top_rate / base if base > 0 else float("nan"),
    }


def phase_economics(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    top_k: int,
    scope: str,
    policy: ExecutionCostPolicy | None = None,
    cost_stress: float = 1.0,
) -> pd.DataFrame:
    return_column = swing_net_return_column(horizon)
    gross_column = f"future_gross_return_{horizon}d"
    excess_columns = {
        benchmark: swing_excess_column(horizon, benchmark)
        for benchmark in ("spy", "qqq", "sector")
    }
    sessions = sorted(pd.to_datetime(predictions["session_date_et"]).dt.date.unique())
    records: list[dict[str, object]] = []
    selection_policy = PredictionSelectionPolicy(swing_top_k=top_k)
    for phase in range(horizon):
        selected_sessions = set(sessions[phase::horizon])
        phase_rows = predictions[pd.to_datetime(predictions["session_date_et"]).dt.date.isin(selected_sessions)]
        selected = select_swing_candidates(
            phase_rows,
            policy=selection_policy,
            probability_column="swing_probability",
        )
        cost = execution_cost_fraction(
            selected,
            price_column="close",
            atr_pct_column="atr_pct_14",
            policy=policy,
            stress=cost_stress,
        )
        if cost is not None and gross_column in selected.columns:
            base_return = pd.to_numeric(selected[gross_column], errors="coerce")
            cost_series = cost
            selected = selected.assign(_net=base_return - cost_series)
            excess = {
                benchmark: (
                    selected["_net"]
                    - pd.to_numeric(
                        selected[f"future_{benchmark}_return_{horizon}d"],
                        errors="coerce",
                    )
                ).dropna()
                for benchmark in excess_columns
            }
        else:
            base_return = pd.to_numeric(selected[return_column], errors="coerce")
            cost_series = pd.Series(flat_stress_surcharge(cost_stress, policy), index=selected.index)
            selected = selected.assign(_net=base_return - cost_series)
            excess = {
                benchmark: (
                    pd.to_numeric(selected[column], errors="coerce")
                    - cost_series
                ).dropna()
                for benchmark, column in excess_columns.items()
            }
        returns = selected["_net"].dropna()
        period = selected.groupby("session_date_et")["_net"].mean().dropna()
        records.append(_economic_record(returns, excess, period, scope=scope, phase=phase))
    return pd.DataFrame(records)


def conservative_economics(economics: pd.DataFrame) -> pd.DataFrame:
    finite_ratio = pd.to_numeric(economics["return_drawdown_ratio"], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    record = {
        "scope": "all_validation_scopes",
        "phase": "conservative",
        "selected_trades": int(pd.to_numeric(economics["selected_trades"], errors="coerce").min()),
        "avg_trade_return": float(pd.to_numeric(economics["avg_trade_return"], errors="coerce").min()),
        "avg_excess_return_vs_spy": float(
            pd.to_numeric(economics["avg_excess_return_vs_spy"], errors="coerce").min()
        ),
        "avg_excess_return_vs_qqq": float(
            pd.to_numeric(economics["avg_excess_return_vs_qqq"], errors="coerce").min()
        ),
        "avg_excess_return_vs_sector": float(
            pd.to_numeric(economics["avg_excess_return_vs_sector"], errors="coerce").min()
        ),
        "win_rate": float(pd.to_numeric(economics["win_rate"], errors="coerce").min()),
        "profit_factor": float(pd.to_numeric(economics["profit_factor"], errors="coerce").min()),
        "cumulative_return": float(pd.to_numeric(economics["cumulative_return"], errors="coerce").min()),
        "max_drawdown": float(pd.to_numeric(economics["max_drawdown"], errors="coerce").max()),
        "return_drawdown_ratio": float(finite_ratio.min()) if finite_ratio.notna().any() else float("inf"),
        "negative_period_rate": float(
            pd.to_numeric(economics["negative_period_rate"], errors="coerce").max()
        ),
        "periods": int(pd.to_numeric(economics["periods"], errors="coerce").min()),
    }
    return pd.DataFrame([record])


def regime_audit(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    top_k: int,
    target_column: str,
    min_regime_sessions: int = 5,
    min_regime_trades: int = 20,
    policy: ExecutionCostPolicy | None = None,
) -> pd.DataFrame:
    """Per-regime selected-policy economics, calibration, and evidence status.

    Beyond representation (regimes present, dominant share), each frozen regime
    reports the conservative selected-policy return, benchmark excess, drawdown,
    calibration error, and independent sessions. A regime with too few sessions
    or selected trades is ``insufficient_evidence`` and can never be counted as
    passing; a populated regime that loses is visible to the worst-regime gate.
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
        "selected_trades": float("nan"),
        "sessions": float("nan"),
        "avg_trade_return": float("nan"),
        "avg_excess_return_vs_spy": float("nan"),
        "max_drawdown": float("nan"),
        "calibration_error": float("nan"),
    }
    details: list[dict[str, object]] = []
    for regime, subset in labelled.groupby("_regime", sort=False):
        record = conservative_economics(
            phase_economics(subset, horizon=horizon, top_k=top_k, scope=f"regime:{regime}", policy=policy)
        ).iloc[0]
        sessions = int(finite_or_none(record.get("periods")) or 0)
        trades = int(finite_or_none(record.get("selected_trades")) or 0)
        status = "sufficient" if sessions >= min_regime_sessions and trades >= min_regime_trades else "insufficient_evidence"
        details.append(
            {
                "scope": f"regime:{regime}",
                "regimes_present": int(counts.size),
                "max_single_regime_share": float(counts.get(regime, 0) / total) if total else 1.0,
                "rows": int(counts.get(regime, 0)),
                "evidence_status": status,
                "selected_trades": trades,
                "sessions": sessions,
                "avg_trade_return": finite_or_none(record.get("avg_trade_return")),
                "avg_excess_return_vs_spy": finite_or_none(record.get("avg_excess_return_vs_spy")),
                "max_drawdown": finite_or_none(record.get("max_drawdown")),
                "calibration_error": (
                    expected_calibration_error(subset[target_column], subset["swing_probability"])
                    if len(subset)
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame([summary, *details])


def catalyst_audit(predictions: pd.DataFrame) -> pd.DataFrame:
    event_count = pd.to_numeric(predictions.get("event_count_3d"), errors="coerce").fillna(0.0)
    low_fraction = pd.to_numeric(
        predictions.get("low_relevance_event_fraction_3d"),
        errors="coerce",
    ).fillna(0.0)
    total_events = float(event_count.sum())
    return pd.DataFrame(
        [
            {
                "has_catalyst_features": "event_count_3d" in predictions.columns,
                "rows": len(predictions),
                "rows_with_catalyst": int(event_count.gt(0).sum()),
                "catalyst_row_rate": float(event_count.gt(0).mean()) if len(predictions) else 0.0,
                "low_relevance_event_rate": (
                    float((event_count * low_fraction).sum() / total_events) if total_events > 0 else 0.0
                ),
                "alignment_error_total": 0,
            }
        ]
    )


def _economic_record(
    returns: pd.Series,
    excess: dict[str, pd.Series],
    period_returns: pd.Series,
    *,
    scope: str,
    phase: int | str,
) -> dict[str, object]:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    equity = (1.0 + period_returns).cumprod()
    cumulative = float(equity.iloc[-1] - 1.0) if not equity.empty else float("nan")
    drawdown = float((equity / equity.cummax() - 1.0).min()) if not equity.empty else float("nan")
    drawdown_abs = abs(drawdown) if np.isfinite(drawdown) else float("nan")
    return {
        "scope": scope,
        "phase": phase,
        "selected_trades": int(len(returns)),
        "avg_trade_return": float(returns.mean()) if not returns.empty else float("nan"),
        **{
            f"avg_excess_return_vs_{benchmark}": (
                float(values.mean()) if not values.empty else float("nan")
            )
            for benchmark, values in excess.items()
        },
        "win_rate": float(returns.gt(0).mean()) if not returns.empty else float("nan"),
        "profit_factor": gains / losses if losses > 0 else float("inf") if gains > 0 else float("nan"),
        "cumulative_return": cumulative,
        "max_drawdown": drawdown_abs,
        "return_drawdown_ratio": (
            cumulative / drawdown_abs if drawdown_abs > 0 else float("inf") if cumulative > 0 else 0.0
        ),
        "negative_period_rate": float(period_returns.lt(0).mean()) if not period_returns.empty else float("nan"),
        "periods": int(len(period_returns)),
    }
