from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, ndcg_score

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract

CalibrationMethod = Literal["sigmoid", "isotonic"]


@dataclass(frozen=True, slots=True)
class DisjointCalibrator:
    method: CalibrationMethod
    family: str
    model_run_id: str
    model: Any

    def predict(self, score: pd.Series | np.ndarray) -> np.ndarray:
        values = np.asarray(score, dtype=float)
        if self.method == "sigmoid":
            probability = np.asarray(self.model.predict_proba(values.reshape(-1, 1)))[:, 1]
        else:
            probability = np.asarray(self.model.predict(values), dtype=float)
        return np.asarray(np.clip(probability, 1e-6, 1 - 1e-6), dtype=float)


class RankingAuditConfig(FrozenContract):
    top_k: int = Field(default=10, ge=1)
    maximum_downside_probability: float = Field(default=0.5, ge=0, le=1)
    bootstrap_iterations: int = Field(default=1_000, ge=100, le=100_000)
    bootstrap_seed: int = 42
    minimum_sessions: int = Field(default=20, ge=2)
    independent_events_only: bool = True
    require_calibrated_downside: bool = True
    schema_version: str = ML_V3_SCHEMA_VERSION


class V3PromotionGateConfig(FrozenContract):
    minimum_sessions: int = Field(default=20, ge=2)
    minimum_selected_trades: int = Field(default=100, ge=1)
    minimum_mean_ndcg_at_k: float = Field(default=0.55, ge=0, le=1)
    minimum_holdout_ndcg_at_k: float = Field(default=0.50, ge=0, le=1)
    minimum_average_trade_return: float = 0.0
    minimum_average_return_ci_low: float = 0.0
    minimum_profit_factor: float = Field(default=1.05, ge=0)
    maximum_drawdown: float = Field(default=0.25, ge=0, le=1)
    maximum_calibration_ece: float = Field(default=0.10, ge=0, le=1)
    required_calibration_families: tuple[str, ...] = ("D1",)

def fit_disjoint_calibrator(
    predictions: pd.DataFrame,
    *,
    family: str,
    method: CalibrationMethod = "sigmoid",
    fit_fraction: float = 0.5,
    minimum_sessions: int = 6,
    random_seed: int = 42,
) -> tuple[DisjointCalibrator, dict[str, Any], pd.DataFrame]:
    if family not in {"B1", "B2", "D1"}:
        raise DataReadinessError("only probabilistic classifier families B1, B2, and D1 can be calibrated")
    if not 0.2 <= fit_fraction <= 0.8:
        raise ValueError("fit_fraction must be between 0.2 and 0.8")
    required = {"family", "audit_scope", "session_date_et", "score", "target", "model_run_id"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise DataReadinessError(f"calibration input missing columns: {', '.join(missing)}")
    data = predictions[
        predictions["family"].astype(str).eq(family) & predictions["audit_scope"].astype(str).eq("walk_forward")
    ].copy()
    data["_session"] = pd.to_datetime(data["session_date_et"], errors="coerce").dt.date
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["target"] = pd.to_numeric(data["target"], errors="coerce")
    data = data.dropna(subset=["_session", "score", "target"])
    run_ids = set(data["model_run_id"].astype(str))
    if len(run_ids) != 1:
        raise DataReadinessError("calibration input must contain exactly one model_run_id")
    if not set(data["target"].unique()).issubset({0, 1}):
        raise DataReadinessError("calibration target must be binary")
    sessions = sorted(data["_session"].unique())
    if len(sessions) < minimum_sessions:
        raise DataReadinessError(f"calibration requires at least {minimum_sessions} OOF sessions")
    split = min(len(sessions) - 2, max(2, int(len(sessions) * fit_fraction)))
    fit_sessions = set(sessions[:split])
    evaluation_sessions = set(sessions[split:])
    fit_data = data[data["_session"].isin(fit_sessions)].copy()
    evaluation = data[data["_session"].isin(evaluation_sessions)].copy()
    if fit_data["target"].nunique() < 2 or evaluation["target"].nunique() < 2:
        raise DataReadinessError("calibration fit and evaluation partitions must each contain both classes")
    if method == "sigmoid":
        model: Any = LogisticRegression(random_state=random_seed)
        model.fit(fit_data["score"].to_numpy().reshape(-1, 1), fit_data["target"].astype(int))
    else:
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(fit_data["score"], fit_data["target"].astype(int))
    calibrator = DisjointCalibrator(
        method=method,
        family=family,
        model_run_id=next(iter(run_ids)),
        model=model,
    )
    evaluation["calibrated_probability"] = calibrator.predict(evaluation["score"])
    before = np.clip(evaluation["score"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    after = evaluation["calibrated_probability"].to_numpy(dtype=float)
    target = evaluation["target"].to_numpy(dtype=int)
    report: dict[str, Any] = {
        "schema": "ml_v3.calibration_audit.v1",
        "family": family,
        "model_run_id": calibrator.model_run_id,
        "method": method,
        "fit_rows": len(fit_data),
        "evaluation_rows": len(evaluation),
        "fit_sessions": len(fit_sessions),
        "evaluation_sessions": len(evaluation_sessions),
        "fit_start": min(fit_sessions).isoformat(),
        "fit_end": max(fit_sessions).isoformat(),
        "evaluation_start": min(evaluation_sessions).isoformat(),
        "evaluation_end": max(evaluation_sessions).isoformat(),
        "before": _calibration_metrics(target, before),
        "after": _calibration_metrics(target, after),
    }
    return calibrator, report, evaluation.drop(columns="_session")


def build_multi_output_evidence(
    predictions: pd.DataFrame,
    *,
    opportunity_family: str,
    downside_family: str = "D1",
    audit_scope: str = "walk_forward",
    downside_calibration: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if opportunity_family == downside_family:
        raise ValueError("opportunity and downside families must differ")
    keys = ["ticker", "decision_time_utc", "decision_group_id", "audit_scope"]
    opportunity = predictions[
        predictions["family"].astype(str).eq(opportunity_family)
        & predictions["audit_scope"].astype(str).eq(audit_scope)
    ].copy()
    downside = predictions[
        predictions["family"].astype(str).eq(downside_family)
        & predictions["audit_scope"].astype(str).eq(audit_scope)
    ].copy()
    if opportunity.empty or downside.empty:
        raise DataReadinessError("multi-output evidence requires both opportunity and downside predictions")
    if bool(opportunity.duplicated(keys).any() or downside.duplicated(keys).any()):
        raise DataReadinessError("multi-output predictions contain duplicate decision identities")
    opportunity = opportunity.rename(
        columns={"score": "raw_opportunity_score", "opportunity_score": "opportunity_score"}
    )
    downside = downside[keys + ["score", "model_run_id"]]
    if downside_calibration is not None:
        calibration_required = {*keys, "calibrated_probability", "model_run_id"}
        calibration_missing = sorted(calibration_required.difference(downside_calibration.columns))
        if calibration_missing:
            raise DataReadinessError(f"downside calibration missing columns: {', '.join(calibration_missing)}")
        calibrated = downside_calibration[list(calibration_required)].copy()
        if bool(calibrated.duplicated(keys).any()):
            raise DataReadinessError("downside calibration contains duplicate decision identities")
        downside = downside.merge(
            calibrated,
            on=[*keys, "model_run_id"],
            how="inner",
            validate="one_to_one",
        ).drop(columns="score")
        downside = downside.rename(
            columns={"calibrated_probability": "downside_probability", "model_run_id": "downside_model_run_id"}
        )
        downside["downside_calibrated"] = 1
    else:
        downside = downside.rename(
            columns={"score": "downside_probability", "model_run_id": "downside_model_run_id"}
        )
        downside["downside_calibrated"] = 0
    keep = list(
        dict.fromkeys(
            [
                *keys,
                "session_date_et",
                "entry_time_utc",
                "primary_exit_time_utc",
                "opportunity_score",
                "raw_opportunity_score",
                "model_run_id",
                "ranking_target",
                "ranking_grade",
                "path_realized_return_net",
                "independent_event_id",
                "market_regime",
            ]
        )
    )
    keep = [column for column in keep if column in opportunity.columns]
    evidence = opportunity[keep].merge(downside, on=keys, how="inner", validate="one_to_one")
    if len(evidence) != len(downside):
        raise DataReadinessError("calibrated downside identities are missing matching opportunity rows")
    evidence = evidence.rename(columns={"model_run_id": "opportunity_model_run_id"})
    evidence["opportunity_family"] = opportunity_family
    evidence["downside_family"] = downside_family
    return evidence


def evaluate_ranking_economics(
    evidence: pd.DataFrame,
    *,
    config: RankingAuditConfig = RankingAuditConfig(),
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        "ticker",
        "decision_time_utc",
        "session_date_et",
        "entry_time_utc",
        "primary_exit_time_utc",
        "decision_group_id",
        "opportunity_score",
        "downside_probability",
        "ranking_target",
        "ranking_grade",
        "path_realized_return_net",
    }
    missing = sorted(required.difference(evidence.columns))
    if missing:
        raise DataReadinessError(f"ranking evidence missing columns: {', '.join(missing)}")
    data = evidence.copy()
    numeric = [
        "opportunity_score",
        "downside_probability",
        "ranking_target",
        "ranking_grade",
        "path_realized_return_net",
    ]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=numeric)
    if config.require_calibrated_downside:
        if "downside_calibrated" not in data.columns or not bool(data["downside_calibrated"].eq(1).all()):
            raise DataReadinessError("ranking economics requires disjointly calibrated downside probabilities")
    if config.independent_events_only and "independent_event_id" not in data.columns:
        raise DataReadinessError("independent_event_id is required for non-overlapping economics")
    ndcg_by_session: dict[object, list[float]] = {}
    selections: list[pd.DataFrame] = []
    for _, group in data.groupby("decision_group_id", sort=False):
        if len(group) < 2:
            continue
        k = min(config.top_k, len(group))
        ndcg = float(
            ndcg_score(
                [group["ranking_grade"].to_numpy()],
                [group["opportunity_score"].to_numpy()],
                k=k,
            )
        )
        session = group["session_date_et"].iloc[0]
        ndcg_by_session.setdefault(session, []).append(ndcg)
        candidates = group[group["downside_probability"] <= config.maximum_downside_probability]
        if config.independent_events_only:
            candidates = candidates.dropna(subset=["independent_event_id"])
        if candidates.empty:
            continue
        selections.append(candidates.nlargest(min(config.top_k, len(candidates)), "opportunity_score"))
    selected = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame(columns=data.columns)
    if config.independent_events_only:
        selected = selected.drop_duplicates("independent_event_id")
    selected = _globally_non_overlapping_groups(selected)
    selected = selected.sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)
    returns = pd.to_numeric(selected["path_realized_return_net"], errors="coerce").dropna()
    sessions = pd.Series(selected["session_date_et"]).nunique()
    readiness_failures: list[str] = []
    if sessions < config.minimum_sessions:
        readiness_failures.append(f"selected sessions {sessions} < {config.minimum_sessions}")
    if returns.empty:
        raise DataReadinessError("ranking audit selected no independent trades")
    average_return_interval = _session_block_interval(
        selected,
        metric=lambda frame: float(pd.to_numeric(frame["path_realized_return_net"]).mean()),
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    win_rate_interval = _session_block_interval(
        selected,
        metric=lambda frame: float(pd.to_numeric(frame["path_realized_return_net"]).gt(0).mean()),
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    ndcg_sessions = pd.DataFrame(
        {
            "session_date_et": list(ndcg_by_session),
            "session_ndcg": [float(np.mean(values)) for values in ndcg_by_session.values()],
        }
    )
    ndcg_interval = _session_block_interval(
        ndcg_sessions,
        metric=lambda frame: float(frame["session_ndcg"].mean()),
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    report: dict[str, Any] = {
        "schema": "ml_v3.ranking_economics_audit.v1",
        "config": config.model_dump(mode="json"),
        "ranking_groups": int(sum(len(values) for values in ndcg_by_session.values())),
        "selected_trades": len(selected),
        "selected_decision_groups": int(selected["decision_group_id"].nunique()),
        "selected_sessions": int(sessions),
        "mean_ndcg_at_k": ndcg_interval["point"],
        "mean_top_k_excess_return": float(pd.to_numeric(selected["ranking_target"]).mean()),
        "average_trade_return": float(returns.mean()),
        "win_rate": float(returns.gt(0).mean()),
        "profit_factor": _profit_factor(returns),
        "max_drawdown": _maximum_drawdown(_portfolio_group_returns(selected)),
        "negative_session_rate": _negative_session_rate(selected),
        "average_trade_return_interval": average_return_interval,
        "win_rate_interval": win_rate_interval,
        "mean_ndcg_interval": ndcg_interval,
        "readiness_failures": readiness_failures,
        "ready": not readiness_failures,
    }
    return report, selected


def evaluate_v3_promotion_evidence(
    *,
    ranking_audit: dict[str, Any] | None,
    holdout_metrics: dict[str, Any] | None,
    calibration_audits: dict[str, dict[str, Any]] | None,
    config: V3PromotionGateConfig = V3PromotionGateConfig(),
) -> dict[str, Any]:
    failures: list[str] = []
    if ranking_audit is None:
        failures.append("ranking economics audit is required")
    else:
        failures.extend(str(item) for item in ranking_audit.get("readiness_failures", []))
        _minimum_gate(failures, ranking_audit, "selected_sessions", config.minimum_sessions)
        _minimum_gate(failures, ranking_audit, "selected_trades", config.minimum_selected_trades)
        _minimum_gate(failures, ranking_audit, "mean_ndcg_at_k", config.minimum_mean_ndcg_at_k)
        _minimum_gate(failures, ranking_audit, "average_trade_return", config.minimum_average_trade_return)
        interval = ranking_audit.get("average_trade_return_interval", {})
        _minimum_gate(failures, interval, "low", config.minimum_average_return_ci_low, prefix="average return CI")
        _minimum_gate(failures, ranking_audit, "profit_factor", config.minimum_profit_factor)
        _maximum_gate(failures, ranking_audit, "max_drawdown", config.maximum_drawdown)
    if holdout_metrics is None:
        failures.append("ticker-holdout metrics are required")
    else:
        _minimum_gate(failures, holdout_metrics, "mean_ndcg_at_k", config.minimum_holdout_ndcg_at_k, prefix="holdout")
    for family in config.required_calibration_families:
        audit = calibration_audits.get(family) if calibration_audits else None
        if audit is None:
            failures.append(f"calibration audit is required for {family}")
            continue
        after = audit.get("after", {})
        _maximum_gate(failures, after, "expected_calibration_error", config.maximum_calibration_ece, prefix=family)
    return {
        "schema": "ml_v3.promotion_evidence.v1",
        "passed": not failures,
        "failures": failures,
        "thresholds": config.model_dump(mode="json"),
    }


def _calibration_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    return {
        "brier_score": float(brier_score_loss(target, probability)),
        "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        "expected_calibration_error": _expected_calibration_error(target, probability),
    }


def _expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    assignments = np.digitize(probability, edges[1:-1], right=True)
    error = 0.0
    for index in range(bins):
        mask = assignments == index
        if not bool(mask.any()):
            continue
        error += float(mask.mean()) * abs(float(target[mask].mean()) - float(probability[mask].mean()))
    return error


def _session_block_interval(
    frame: pd.DataFrame,
    *,
    metric: Any,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    sessions = list(pd.Series(frame["session_date_et"]).dropna().unique())
    if len(sessions) < 2:
        raise DataReadinessError("session-block bootstrap requires at least two sessions")
    blocks = {session: frame[pd.Series(frame["session_date_et"], index=frame.index).eq(session)] for session in sessions}
    random = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        selected_sessions = random.choice(sessions, size=len(sessions), replace=True)
        sampled = pd.concat([blocks[session] for session in selected_sessions], ignore_index=True)
        samples[iteration] = float(metric(sampled))
    point = float(metric(frame))
    low, high = np.quantile(samples, [0.025, 0.975])
    return {"point": point, "low": float(low), "high": float(high), "iterations": float(iterations), "seed": float(seed)}


def session_block_interval(
    frame: pd.DataFrame,
    *,
    metric: Any,
    iterations: int,
    seed: int,
) -> dict[str, float]:
    """Public, deterministic session-block bootstrap used by promotion evidence."""

    return _session_block_interval(frame, metric=metric, iterations=iterations, seed=seed)


def _profit_factor(returns: pd.Series) -> float | None:
    gains = float(returns[returns > 0].sum())
    losses = abs(float(returns[returns < 0].sum()))
    if losses == 0:
        return None if gains == 0 else gains / 1e-12
    return gains / losses


def _maximum_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns.clip(lower=-0.999999)).cumprod()
    drawdown = 1 - equity / equity.cummax()
    return float(drawdown.max()) if not drawdown.empty else 0.0


def _negative_session_rate(selected: pd.DataFrame) -> float:
    group_returns = _portfolio_group_returns(selected)
    session_return = group_returns.groupby(level="session_date_et").sum()
    return float(pd.to_numeric(session_return).lt(0).mean())


def _globally_non_overlapping_groups(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected
    frame = selected.copy()
    frame["entry_time_utc"] = frame["entry_time_utc"].map(_aware_timestamp)
    frame["primary_exit_time_utc"] = frame["primary_exit_time_utc"].map(_aware_timestamp)
    if bool(frame[["entry_time_utc", "primary_exit_time_utc"]].isna().any(axis=None)):
        raise DataReadinessError("selected evidence contains invalid entry or exit timestamps")
    groups = frame.groupby("decision_group_id", sort=False).agg(
        entry_time_utc=("entry_time_utc", "first"),
        primary_exit_time_utc=("primary_exit_time_utc", "first"),
        entry_count=("entry_time_utc", "nunique"),
        exit_count=("primary_exit_time_utc", "nunique"),
    )
    if bool(groups[["entry_count", "exit_count"]].ne(1).any(axis=None)):
        raise DataReadinessError("decision group contains inconsistent entry or exit timestamps")
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


def _portfolio_group_returns(selected: pd.DataFrame) -> pd.Series:
    grouped = selected.groupby(["session_date_et", "decision_group_id"], sort=False)["path_realized_return_net"].mean()
    return pd.to_numeric(grouped, errors="coerce").dropna()


def _aware_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return pd.NaT
    return timestamp.tz_convert("UTC")


def _minimum_gate(
    failures: list[str],
    record: dict[str, Any],
    key: str,
    threshold: float,
    *,
    prefix: str = "ranking",
) -> None:
    value = _finite_float(record.get(key))
    if value is None or value < threshold:
        failures.append(f"{prefix} {key} {value} < {threshold}")


def _maximum_gate(
    failures: list[str],
    record: dict[str, Any],
    key: str,
    threshold: float,
    *,
    prefix: str = "ranking",
) -> None:
    value = _finite_float(record.get(key))
    if value is None or value > threshold:
        failures.append(f"{prefix} {key} {value} > {threshold}")


def _finite_float(value: object) -> float | None:
    try:
        converted = float(str(value))
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None
