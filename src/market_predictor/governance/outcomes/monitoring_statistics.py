"""Session-gap-aware overlap uncertainty for monitoring means and return ratios."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Literal

import exchange_calendars as xcals
import pandas as pd


@dataclass(frozen=True)
class MonitoringEstimate:
    mean: float
    standard_error: float
    t_statistic: float | None
    method: Literal["hansen_hodrick", "newey_west"]
    sample_count: int


def exchange_session_ordinals(sessions: Sequence[date | str]) -> tuple[int, ...]:
    """Map exact XNYS session dates to positions, without compressing absent sessions."""
    calendar = xcals.get_calendar("XNYS")
    positions = []
    for value in sessions:
        if isinstance(value, datetime):
            raise ValueError("session dates must not contain times")
        if isinstance(value, str):
            parsed = date.fromisoformat(value)
            if parsed.isoformat() != value:
                raise ValueError("session dates must use canonical ISO dates")
        elif isinstance(value, date):
            parsed = value
        else:
            raise ValueError("session dates must be dates or canonical ISO strings")
        position = int(calendar.sessions.get_indexer([pd.Timestamp(parsed)])[0])
        if position < 0:
            raise ValueError("date is outside the available XNYS sessions")
        positions.append(position)
    if len(set(positions)) != len(positions):
        raise ValueError("duplicate session ordinals")
    return tuple(positions)


def _inputs(
    session_ordinals: Sequence[int], values: Sequence[float], horizon_sessions: int,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if isinstance(horizon_sessions, bool) or not isinstance(horizon_sessions, Integral) or horizon_sessions < 1:
        raise ValueError("horizon_sessions must be a positive integer")
    if not len(values) or len(session_ordinals) != len(values):
        raise ValueError("nonempty values must match session ordinals")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in session_ordinals):
        raise ValueError("session ordinals must be integers")
    ordinals = tuple(int(value) for value in session_ordinals)
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("duplicate session ordinals")
    if any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) for value in values):
        raise ValueError("values must be finite real numbers")
    return ordinals, tuple(float(value) for value in values)


def _estimate(
    ordinals: tuple[int, ...], residuals: tuple[float, ...], *, mean: float,
    denominator: float, horizon_sessions: int,
) -> MonitoringEstimate:
    # Covariance products use exact exchange-session distances; absent sessions
    # add no pair. The uncorrected sandwich is sum(products) / denominator**2.
    if not math.isfinite(mean) or not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("estimate and positive denominator must be finite")
    scores = {ordinal: value / denominator for ordinal, value in zip(ordinals, residuals, strict=True)}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("normalized estimating scores must be finite")

    def variance(lags: int, *, bartlett: bool) -> float:
        terms = [value * value for value in scores.values()]
        for lag in range(1, lags + 1):
            weight = 1.0 - lag / (lags + 1) if bartlett else 1.0
            terms.extend(2.0 * weight * value * scores[ordinal - lag]
                for ordinal, value in scores.items() if ordinal - lag in scores)
        if not all(math.isfinite(term) for term in terms):
            return math.inf
        try:
            return math.fsum(terms)
        except OverflowError:
            return math.inf

    method: Literal["hansen_hodrick", "newey_west"] = "hansen_hodrick"
    estimate_variance = variance(horizon_sessions - 1, bartlett=False)
    if not math.isfinite(estimate_variance) or estimate_variance < 0:
        method = "newey_west"
        estimate_variance = variance(2 * horizon_sessions, bartlett=True)
    if not math.isfinite(estimate_variance) or estimate_variance < 0:
        raise ValueError("overlap variance cannot be represented as a finite nonnegative number")
    standard_error = math.sqrt(estimate_variance)
    t_statistic = mean / standard_error if standard_error > 0 else None
    if t_statistic is not None and not math.isfinite(t_statistic):
        raise ValueError("t statistic cannot be represented as a finite number")
    return MonitoringEstimate(mean, standard_error, t_statistic, method, len(ordinals))


def mean_statistics(
    session_ordinals: Sequence[int], values: Sequence[float], *, horizon_sessions: int,
) -> MonitoringEstimate:
    """Equal-observation mean; HH N-1 covariance, NW 2N fallback if HH is invalid.

    Zero variance has no finite t statistic. Missing observations must be omitted
    with their original session ordinals retained, never imputed with zero.
    """
    ordinals, numbers = _inputs(session_ordinals, values, horizon_sessions)
    mean = math.fsum(value / len(numbers) for value in numbers)
    return _estimate(ordinals, tuple(value - mean for value in numbers), mean=mean,
        denominator=float(len(numbers)), horizon_sessions=horizon_sessions)


def ratio_statistics(
    session_ordinals: Sequence[int], numerators: Sequence[float], denominators: Sequence[float], *,
    horizon_sessions: int,
) -> MonitoringEstimate:
    """Ratio of sums, with covariance on numerator minus ratio*denominator.

    Inputs are daily sums of excess returns and actual holding sessions. A genuine
    zero-exposure session may contribute (0, 0); the overall denominator must be positive.
    """
    ordinals, top = _inputs(session_ordinals, numerators, horizon_sessions)
    _, bottom = _inputs(session_ordinals, denominators, horizon_sessions)
    if any(value < 0 for value in bottom) or any(d == 0 and n != 0 for n, d in zip(top, bottom, strict=True)):
        raise ValueError("denominators must be nonnegative and zero exposure must have zero numerator")
    try:
        total_bottom = math.fsum(bottom)
        total_top = math.fsum(top)
    except OverflowError as exc:
        raise ValueError("ratio sums must be finite") from exc
    if not math.isfinite(total_bottom) or total_bottom <= 0:
        raise ValueError("total denominator must be finite and positive")
    mean = total_top / total_bottom
    residuals = tuple(n - mean * d for n, d in zip(top, bottom, strict=True))
    return _estimate(ordinals, residuals, mean=mean, denominator=total_bottom, horizon_sessions=horizon_sessions)
