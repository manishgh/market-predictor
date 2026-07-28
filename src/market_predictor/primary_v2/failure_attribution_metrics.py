"""Pure cohort metrics for primary V2 failure attribution."""

from __future__ import annotations

import json
import math
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_predictor.primary_v2.contracts import INTRADAY_V2_ID
from market_predictor.primary_v2.failure_attribution_contracts import (
    FailureAttributionConfig,
    FailureAttributionStrategyConfig,
)
from market_predictor.regime_evidence import session_block_mean_interval
from market_predictor.v3.errors import DataReadinessError

UNKNOWN_CATEGORY = "unknown"
_CONSERVATIVE_MINIMUM_METRICS = (
    "average_net_return",
    "average_net_return_ci_low",
    "average_excess_return_vs_spy",
    "average_excess_return_vs_spy_ci_low",
    "profit_factor",
)
_REPLICATED_METRICS = (
    "rows",
    "sessions",
    *_CONSERVATIVE_MINIMUM_METRICS,
    "maximum_drawdown",
)


def build_cohort_evidence(
    evidence_rows: pd.DataFrame,
    *,
    strategy_id: str,
    config: FailureAttributionConfig,
) -> pd.DataFrame:
    """Calculate frozen one-dimensional cohort evidence."""

    strategy = config.strategy(strategy_id)
    rows = _prepare_dimensions(
        evidence_rows,
        strategy=strategy,
        strategy_id=strategy_id,
    )
    records: list[dict[str, object]] = []
    for scope in config.validation_scopes:
        scope_rows = _assign_evaluation_phases(
            rows.loc[rows["validation_scope"].eq(scope)].copy(),
            strategy=strategy,
        )
        if scope_rows.empty:
            raise DataReadinessError(
                f"failure-attribution scope has no rows: {scope}"
            )
        for dimension in strategy.dimensions:
            groups = _dimension_groups(
                scope_rows,
                dimension=dimension,
            )
            for cohort_value, cohort in groups:
                for phase in range(strategy.non_overlapping_phases):
                    phase_rows = cohort.loc[
                        cohort["evaluation_phase"].eq(phase)
                    ]
                    if phase_rows.empty:
                        continue
                    records.append(
                        _cohort_record(
                            phase_rows,
                            validation_scope=scope,
                            evaluation_phase=phase,
                            dimension=dimension,
                            cohort_value=cohort_value,
                            strategy=strategy,
                            strategy_id=strategy_id,
                            config=config,
                        )
                    )
    evidence = pd.DataFrame(records)
    return evidence.sort_values(
        [
            "dimension",
            "cohort_value",
            "validation_scope",
            "evaluation_phase",
        ],
        kind="stable",
    ).reset_index(drop=True)


def build_replicated_viability(
    cohort_evidence: pd.DataFrame,
    *,
    strategy_id: str,
    config: FailureAttributionConfig,
) -> pd.DataFrame:
    """Require the same cohort to pass every phase in both scopes."""

    required = {
        "validation_scope",
        "evaluation_phase",
        "dimension",
        "cohort_value",
        "scope_pass",
        "failure_reasons_json",
        *_REPLICATED_METRICS,
    }
    if missing := sorted(required.difference(cohort_evidence.columns)):
        raise DataReadinessError(
            "cohort evidence is missing required columns: " + ", ".join(missing)
        )
    strategy = config.strategy(strategy_id)
    expected_phases = set(range(strategy.non_overlapping_phases))
    records = [
        _replicated_record(
            group,
            dimension=str(dimension),
            cohort_value=str(value),
            expected_phases=expected_phases,
            config=config,
        )
        for (dimension, value), group in cohort_evidence.groupby(
            ["dimension", "cohort_value"],
            sort=True,
            observed=True,
        )
    ]
    return pd.DataFrame(records).sort_values(
        ["dimension", "cohort_value"],
        kind="stable",
    ).reset_index(drop=True)


def normalized_category(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna(UNKNOWN_CATEGORY).str.strip()
    return (
        normalized.mask(normalized.eq(""), UNKNOWN_CATEGORY)
        .astype(str)
    )


def _dimension_groups(
    rows: pd.DataFrame,
    *,
    dimension: str,
) -> tuple[tuple[str, pd.DataFrame], ...]:
    if dimension == "overall":
        return (("all", rows),)
    return tuple(
        (str(value), group)
        for value, group in rows.groupby(
            dimension,
            sort=True,
            observed=True,
            dropna=False,
        )
    )


def _replicated_record(
    group: pd.DataFrame,
    *,
    dimension: str,
    cohort_value: str,
    expected_phases: set[int],
    config: FailureAttributionConfig,
) -> dict[str, object]:
    record: dict[str, object] = {
        "dimension": dimension,
        "cohort_value": cohort_value,
    }
    failures: list[str] = []
    passes: list[bool] = []
    for scope in config.validation_scopes:
        scope_rows = group.loc[group["validation_scope"].eq(scope)]
        actual_phases = set(
            pd.to_numeric(
                scope_rows["evaluation_phase"],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
        record[f"{scope}_phases_present"] = len(actual_phases)
        record[f"{scope}_phases_passed"] = int(
            scope_rows["scope_pass"].astype(bool).sum()
        )
        if scope_rows.empty or actual_phases != expected_phases:
            passes.append(False)
            failures.append(
                f"{scope}: expected phases {sorted(expected_phases)}, "
                f"found {sorted(actual_phases)}"
            )
            for metric in _REPLICATED_METRICS:
                record[f"{scope}_{metric}"] = float("nan")
            record[f"{scope}_pass"] = False
            continue
        scope_pass = bool(scope_rows["scope_pass"].astype(bool).all())
        passes.append(scope_pass)
        record[f"{scope}_pass"] = scope_pass
        for metric in ("rows", "sessions"):
            record[f"{scope}_{metric}"] = int(
                pd.to_numeric(scope_rows[metric], errors="coerce").min()
            )
        for metric in _CONSERVATIVE_MINIMUM_METRICS:
            record[f"{scope}_{metric}"] = float(
                pd.to_numeric(scope_rows[metric], errors="coerce").min()
            )
        record[f"{scope}_maximum_drawdown"] = float(
            pd.to_numeric(
                scope_rows["maximum_drawdown"],
                errors="coerce",
            ).max()
        )
        if not scope_pass:
            _append_phase_failures(
                failures,
                scope=scope,
                scope_rows=scope_rows,
            )
    record["replicated_viable"] = bool(passes) and all(passes)
    record["failure_reasons_json"] = json.dumps(
        failures,
        sort_keys=True,
        separators=(",", ":"),
    )
    return record


def _append_phase_failures(
    failures: list[str],
    *,
    scope: str,
    scope_rows: pd.DataFrame,
) -> None:
    for _, phase_row in scope_rows.iterrows():
        reasons = json.loads(str(phase_row["failure_reasons_json"]))
        phase = int(phase_row["evaluation_phase"])
        failures.extend(
            f"{scope}/phase-{phase}: {reason}"
            for reason in reasons
        )


def _prepare_dimensions(
    evidence_rows: pd.DataFrame,
    *,
    strategy: FailureAttributionStrategyConfig,
    strategy_id: str,
) -> pd.DataFrame:
    rows = evidence_rows.copy()
    for column in ("market_regime", "sector"):
        rows[column] = normalized_category(rows[column])
    volatility = pd.to_numeric(
        rows[strategy.volatility_column],
        errors="coerce",
    )
    if bool(volatility.lt(0).any()) or not np.isfinite(
        volatility.to_numpy(dtype=float)
    ).all():
        raise DataReadinessError("volatility bucket input is invalid")
    rows["volatility_bucket"] = np.select(
        [
            volatility.lt(strategy.volatility_low_upper_bound),
            volatility.lt(strategy.volatility_medium_upper_bound),
        ],
        ["low", "medium"],
        default="high",
    )
    if strategy_id == INTRADAY_V2_ID:
        for column in ("market_cap_bucket", "liquidity_bucket"):
            rows[column] = normalized_category(rows[column])
        rows["time_of_day"] = _time_of_day(rows["decision_time_utc"])
    return rows


def _assign_evaluation_phases(
    rows: pd.DataFrame,
    *,
    strategy: FailureAttributionStrategyConfig,
) -> pd.DataFrame:
    periods = pd.to_datetime(
        rows[strategy.period_column],
        errors="coerce",
    )
    if periods.isna().any():
        raise DataReadinessError(
            "validation period is invalid for non-overlapping phases"
        )
    normalized = periods.dt.normalize()
    ordered = sorted(normalized.unique())
    phase_by_period = {
        period: index % strategy.non_overlapping_phases
        for index, period in enumerate(ordered)
    }
    rows["evaluation_phase"] = normalized.map(phase_by_period)
    if rows["evaluation_phase"].isna().any():
        raise DataReadinessError(
            "validation period could not be assigned to an evaluation phase"
        )
    rows["evaluation_phase"] = rows["evaluation_phase"].astype(int)
    return rows


def _cohort_record(
    rows: pd.DataFrame,
    *,
    validation_scope: str,
    evaluation_phase: int,
    dimension: str,
    cohort_value: str,
    strategy: FailureAttributionStrategyConfig,
    strategy_id: str,
    config: FailureAttributionConfig,
) -> dict[str, object]:
    gross = rows[strategy.gross_return_column].astype(float)
    net = rows[strategy.net_return_column].astype(float)
    spy = rows[strategy.spy_excess_return_column].astype(float)
    cost = rows["stamped_round_trip_cost_fraction"].astype(float)
    sessions = rows[strategy.period_column]
    net_interval = session_block_mean_interval(
        sessions,
        net,
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    spy_interval = session_block_mean_interval(
        sessions,
        spy,
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    period_returns = (
        rows.assign(_net=net)
        .groupby(strategy.period_column, observed=True)["_net"]
        .mean()
        .sort_index()
    )
    equity = (1 + period_returns).cumprod()
    maximum_drawdown = (
        abs(float((equity / equity.cummax() - 1).min()))
        if not equity.empty
        else float("nan")
    )
    gains = float(net.loc[net.gt(0)].sum())
    losses = abs(float(net.loc[net.lt(0)].sum()))
    profit_factor = (
        gains / losses
        if losses > 0
        else float("inf")
        if gains > 0
        else float("nan")
    )
    record: dict[str, object] = {
        "validation_scope": validation_scope,
        "evaluation_phase": evaluation_phase,
        "dimension": dimension,
        "cohort_value": cohort_value,
        "rows": len(rows),
        "sessions": int(sessions.nunique()),
        "tickers": int(rows["ticker"].nunique()),
        "average_gross_return": float(gross.mean()),
        "average_stamped_round_trip_cost_fraction": float(cost.mean()),
        "average_stamped_round_trip_cost_bps": float(cost.mean() * 10_000),
        "minimum_stamped_round_trip_cost_bps": float(cost.min() * 10_000),
        "maximum_stamped_round_trip_cost_bps": float(cost.max() * 10_000),
        "average_net_return": float(net.mean()),
        "average_net_return_ci_low": net_interval["low"],
        "average_net_return_ci_high": net_interval["high"],
        "average_excess_return_vs_spy": float(spy.mean()),
        "average_excess_return_vs_spy_ci_low": spy_interval["low"],
        "average_excess_return_vs_spy_ci_high": spy_interval["high"],
        "win_rate": float(net.gt(0).mean()),
        "profit_factor": profit_factor,
        "maximum_drawdown": maximum_drawdown,
        "average_mfe": float(rows[strategy.mfe_column].mean()),
        "average_mae": float(rows[strategy.mae_column].mean()),
        "target_first_rate": float("nan"),
        "stop_first_rate": float("nan"),
        "timeout_rate": float("nan"),
        "average_resolution_minutes": float("nan"),
    }
    if strategy_id == INTRADAY_V2_ID:
        record.update(_intraday_path_metrics(rows))
    failures = _scope_failures(record, config=config)
    record["scope_pass"] = not failures
    record["failure_reasons_json"] = json.dumps(
        failures,
        sort_keys=True,
        separators=(",", ":"),
    )
    return record


def _intraday_path_metrics(rows: pd.DataFrame) -> dict[str, float]:
    return {
        "target_first_rate": float(
            pd.to_numeric(
                rows["target_before_stop_30m"],
                errors="coerce",
            ).mean()
        ),
        "stop_first_rate": float(
            pd.to_numeric(
                rows["stop_before_target_30m"],
                errors="coerce",
            ).mean()
        ),
        "timeout_rate": float(
            pd.to_numeric(
                rows["path_timeout_30m"],
                errors="coerce",
            ).mean()
        ),
        "average_resolution_minutes": float(
            pd.to_numeric(
                rows["path_outcome_bar"],
                errors="coerce",
            ).mean()
        ),
    }


def _scope_failures(
    record: dict[str, object],
    *,
    config: FailureAttributionConfig,
) -> list[str]:
    checks = (
        (
            "rows",
            _number(record["rows"]),
            float(config.minimum_rows_per_scope),
            "minimum_inclusive",
        ),
        (
            "sessions",
            _number(record["sessions"]),
            float(config.minimum_sessions_per_scope),
            "minimum_inclusive",
        ),
        (
            "average_net_return",
            _number(record["average_net_return"]),
            config.minimum_average_net_return,
            "minimum",
        ),
        (
            "average_net_return_ci_low",
            _number(record["average_net_return_ci_low"]),
            config.minimum_average_net_return_ci_low,
            "minimum",
        ),
        (
            "average_excess_return_vs_spy",
            _number(record["average_excess_return_vs_spy"]),
            config.minimum_average_excess_return_vs_spy,
            "minimum",
        ),
        (
            "average_excess_return_vs_spy_ci_low",
            _number(record["average_excess_return_vs_spy_ci_low"]),
            config.minimum_average_excess_return_vs_spy_ci_low,
            "minimum",
        ),
        (
            "profit_factor",
            _number(record["profit_factor"], allow_positive_infinity=True),
            config.minimum_profit_factor,
            "minimum_inclusive",
        ),
        (
            "maximum_drawdown",
            _number(record["maximum_drawdown"]),
            config.maximum_drawdown,
            "maximum",
        ),
        (
            "average_stamped_round_trip_cost_bps",
            _number(record["average_stamped_round_trip_cost_bps"]),
            config.minimum_stamped_round_trip_cost_bps,
            "minimum_inclusive",
        ),
    )
    failures: list[str] = []
    for label, value, threshold, direction in checks:
        if value is None:
            failures.append(f"{label} is not finite")
        elif direction == "minimum" and value <= threshold:
            failures.append(
                f"{label} {value:.10g} is not above {threshold:.10g}"
            )
        elif direction == "minimum_inclusive" and value < threshold:
            failures.append(
                f"{label} {value:.10g} is below {threshold:.10g}"
            )
        elif direction == "maximum" and value > threshold:
            failures.append(
                f"{label} {value:.10g} exceeds {threshold:.10g}"
            )
    return failures


def _time_of_day(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, utc=True, errors="coerce")
    if timestamps.isna().any():
        raise DataReadinessError(
            "intraday decision time is invalid for time-of-day attribution"
        )
    eastern = timestamps.dt.tz_convert(ZoneInfo("America/New_York"))
    minutes = eastern.dt.hour * 60 + eastern.dt.minute
    return pd.Series(
        np.select(
            [minutes.lt(11 * 60), minutes.lt(14 * 60)],
            ["opening_before_11_et", "midday_11_to_14_et"],
            default="late_after_14_et",
        ),
        index=values.index,
        dtype="object",
    )


def _number(
    value: object,
    *,
    allow_positive_infinity: bool = False,
) -> float | None:
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    if math.isfinite(number):
        return number
    if allow_positive_infinity and number == float("inf"):
        return number
    return None
