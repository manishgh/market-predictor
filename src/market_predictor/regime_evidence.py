from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

REGIME_BOOTSTRAP_ITERATIONS = 1_000
REGIME_BOOTSTRAP_SEED = 47


def session_block_mean_interval(
    session_ids: pd.Series,
    values: pd.Series,
    *,
    iterations: int = REGIME_BOOTSTRAP_ITERATIONS,
    seed: int = REGIME_BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Deterministic session-block bootstrap interval for a row-level mean."""

    if len(session_ids) != len(values):
        raise ValueError("session IDs and values must align")
    if iterations < 100:
        raise ValueError("session bootstrap requires at least 100 iterations")
    frame = pd.DataFrame(
        {
            "session": session_ids.reset_index(drop=True),
            "value": pd.to_numeric(values.reset_index(drop=True), errors="coerce"),
        }
    ).dropna()
    blocks = frame.groupby("session", sort=False, observed=True)["value"].agg(["sum", "count"])
    point = float(frame["value"].mean()) if not frame.empty else float("nan")
    if len(blocks) < 2:
        return {
            "point": point,
            "low": float("nan"),
            "high": float("nan"),
            "sessions": int(len(blocks)),
            "iterations": iterations,
            "seed": seed,
        }
    block_sums = blocks["sum"].to_numpy(dtype=float)
    block_counts = blocks["count"].to_numpy(dtype=float)
    random = np.random.default_rng(seed)
    sampled_indices = random.integers(
        0,
        len(blocks),
        size=(iterations, len(blocks)),
    )
    sample_sums = block_sums[sampled_indices].sum(axis=1)
    sample_counts = block_counts[sampled_indices].sum(axis=1)
    samples = np.divide(
        sample_sums,
        sample_counts,
        out=np.full(iterations, np.nan, dtype=float),
        where=sample_counts > 0,
    )
    low, high = np.nanquantile(samples, [0.025, 0.975])
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "sessions": int(len(blocks)),
        "iterations": iterations,
        "seed": seed,
    }


def regime_promotion_failures(
    regime_audit: pd.DataFrame,
    *,
    required_regimes: tuple[str, ...],
    min_required_sessions: int,
    min_required_trades: int,
    min_avg_excess_return_vs_spy: float,
    min_avg_trade_return_ci_low: float,
    min_avg_excess_return_vs_spy_ci_low: float,
    max_drawdown: float,
    max_calibration_error: float,
) -> list[str]:
    """Apply one fail-closed required-regime contract to either model family."""

    required_columns = {
        "scope",
        "evidence_status",
        "required_regime",
        "minimum_sessions",
        "minimum_trades",
        "sessions",
        "selected_trades",
        "avg_trade_return_ci_low",
        "avg_excess_return_vs_spy_ci_low",
    }
    if missing := sorted(required_columns.difference(regime_audit.columns)):
        return [f"regime audit is missing required columns: {', '.join(missing)}"]
    failures: list[str] = []
    details = regime_audit[regime_audit["scope"].astype(str).str.startswith("regime:")]
    by_scope = {
        str(detail["scope"]): detail
        for _, detail in details.iterrows()
    }
    for regime in required_regimes:
        scope = f"regime:{regime}"
        detail = by_scope.get(scope)
        if detail is None:
            failures.append(f"{scope} required regime evidence is missing")
            continue
        if not _strict_bool(detail.get("required_regime")):
            failures.append(f"{scope} is not marked as a required regime")
        _append_minimum_failure(
            failures,
            scope=scope,
            label="audit minimum_sessions",
            value=_finite_number(detail.get("minimum_sessions")),
            minimum=float(min_required_sessions),
        )
        _append_minimum_failure(
            failures,
            scope=scope,
            label="audit minimum_trades",
            value=_finite_number(detail.get("minimum_trades")),
            minimum=float(min_required_trades),
        )
        _append_minimum_failure(
            failures,
            scope=scope,
            label="sessions",
            value=_finite_number(detail.get("sessions")),
            minimum=float(min_required_sessions),
        )
        _append_minimum_failure(
            failures,
            scope=scope,
            label="selected_trades",
            value=_finite_number(detail.get("selected_trades")),
            minimum=float(min_required_trades),
        )
    for _, detail in details.iterrows():
        scope = str(detail.get("scope"))
        sessions = _finite_number(detail.get("sessions"))
        trades = _finite_number(detail.get("selected_trades"))
        if (
            sessions is None
            or sessions < min_required_sessions
            or trades is None
            or trades < min_required_trades
        ):
            continue
        _append_minimum_failure(
            failures,
            scope=scope,
            label="avg_excess_return_vs_spy",
            value=_finite_number(detail.get("avg_excess_return_vs_spy")),
            minimum=min_avg_excess_return_vs_spy,
        )
        _append_maximum_failure(
            failures,
            scope=scope,
            label="max_drawdown",
            value=_finite_number(detail.get("max_drawdown")),
            maximum=max_drawdown,
            missing_fails=False,
        )
        _append_maximum_failure(
            failures,
            scope=scope,
            label="calibration_error",
            value=_finite_number(detail.get("calibration_error")),
            maximum=max_calibration_error,
            missing_fails=False,
        )
        _append_minimum_failure(
            failures,
            scope=scope,
            label="avg_trade_return_ci_low",
            value=_finite_number(detail.get("avg_trade_return_ci_low")),
            minimum=min_avg_trade_return_ci_low,
        )
        _append_minimum_failure(
            failures,
            scope=scope,
            label="avg_excess_return_vs_spy_ci_low",
            value=_finite_number(detail.get("avg_excess_return_vs_spy_ci_low")),
            minimum=min_avg_excess_return_vs_spy_ci_low,
        )
    return failures


def _append_minimum_failure(
    failures: list[str],
    *,
    scope: str,
    label: str,
    value: float | None,
    minimum: float,
) -> None:
    if value is None or value < minimum:
        failures.append(f"{scope} {label} {value} < {minimum}")


def _append_maximum_failure(
    failures: list[str],
    *,
    scope: str,
    label: str,
    value: float | None,
    maximum: float,
    missing_fails: bool,
) -> None:
    if (value is None and missing_fails) or (value is not None and value > maximum):
        failures.append(f"{scope} {label} {value} > {maximum}")


def _finite_number(value: object) -> float | None:
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"
