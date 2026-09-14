"""Separate observable input completeness from future supervised-label usability."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime

import numpy as np
import pandas as pd

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError

TARGET_COLUMN = "spy_fixed_horizon_excess_return"
ROLES = ("spy", "qqq", "sector")
RETURN_COLUMNS = (
    "fixed_horizon_gross_return", "fixed_horizon_net_return",
    *(f"{role}_horizon_gross_return" for role in ROLES),
    *(f"{role}_fixed_horizon_excess_return" for role in ROLES),
)
CONTEXT_COLUMNS = (
    "decision_id", "security_id", "ticker", "sector", "session_date_et", "decision_time_utc",
    "feature_eligible", "daily_bar_count", "research_label_mature_at", "stock_source_admitted",
    "stock_component_id", "stock_missing_reasons", "fixed_comparisons_complete",
    "training_eligible", "production_eligible",
    *(f"{role}_{suffix}" for role in ROLES for suffix in ("component_id", "missing_reasons")),
)


def utc_clocks(values: pd.Series, name: str) -> pd.Series:
    """Validate each representation before conversion, including equal mixed offsets."""
    if isinstance(values.dtype, pd.DatetimeTZDtype) and str(values.dtype.tz) == "UTC":
        return pd.to_datetime(values, utc=True)
    for value in values.dropna():
        if not isinstance(value, (str, datetime, pd.Timestamp)):
            raise DataReadinessError(f"{name} requires explicit UTC timestamps")
        try:
            stamp = pd.Timestamp(value)
            if stamp.tzinfo is None or stamp.utcoffset() != pd.Timedelta(0):
                raise ValueError("not UTC")
        except (TypeError, ValueError) as error:
            raise DataReadinessError(f"{name} requires explicit UTC timestamps") from error
    return pd.to_datetime(values, utc=True, errors="raise")


def _numeric(values: pd.Series, name: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if (values.notna() & numeric.isna()).any() or np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise DataReadinessError(f"{name} must be finite numeric or missing")
    return numeric


def _boolean(values: pd.Series, name: str) -> pd.Series:
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataReadinessError(f"{name} must contain explicit booleans")
    return values.astype(bool)


def _no_gaps(values: pd.Series, name: str) -> pd.Series:
    result = []
    for value in values:
        try:
            decoded = json.loads(value) if isinstance(value, str) else None
        except ValueError as error:
            raise DataReadinessError(f"{name} must contain JSON reason arrays") from error
        if not isinstance(decoded, list) or any(not isinstance(reason, str) or not reason for reason in decoded):
            raise DataReadinessError(f"{name} must contain JSON reason arrays")
        result.append(not decoded)
    return pd.Series(result, index=values.index, dtype=bool)


def fixed_horizon_readiness(
    frame: pd.DataFrame, *, model_columns: Sequence[str], availability_columns: Mapping[str, str],
    maturity_by_session: Mapping[date, pd.Timestamp], fit_end: pd.Timestamp,
    minimum_warmup: int, round_trip_cost: float,
) -> pd.DataFrame:
    """Return diagnostic masks without filtering inputs or inferring live admission.

    Complete-case counts are descriptive only, not an imputation/selection policy.
    Missing selected outcomes must never be removed from later portfolio evaluation.
    Historical horizon maturity is not a claim of historical provider first receipt.
    """
    columns = tuple(model_columns)
    required = {*CONTEXT_COLUMNS, *RETURN_COLUMNS, *columns, *availability_columns.values()}
    if (frame.empty or not frame.columns.is_unique or not frame.index.is_unique or not columns
            or len(set(columns)) != len(columns) or set(availability_columns) != set(columns)
            or not required.issubset(frame)):
        raise DataReadinessError("fixed-horizon audit lacks unique rows or declared columns")
    if set(columns).intersection((*CONTEXT_COLUMNS, *RETURN_COLUMNS)):
        raise DataReadinessError("model inputs cannot include outcome or decision metadata")
    for name in ("decision_id", "security_id", "ticker", "sector"):
        if not frame[name].map(lambda v: isinstance(v, str) and bool(v) and v.strip() == v).all():
            raise DataReadinessError(f"invalid {name}")
    if frame.decision_id.duplicated().any() or frame.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("duplicate fixed-horizon decisions")
    days = frame.session_date_et
    if not days.map(lambda value: type(value) is date and value in maturity_by_session).all():
        raise DataReadinessError("decisions outside frozen initial-fit sessions")
    cutoff = utc_clocks(frame.decision_time_utc, "decision cutoff")
    if cutoff.isna().any() or not cutoff.eq(swing_prediction_cutoffs(days)).all():
        raise DataReadinessError("decision cutoff differs from canonical session cutoff")
    for name in ("training_eligible", "production_eligible"):
        if _boolean(frame[name], name).any():
            raise DataReadinessError("publication cannot claim training or production permission")
    warmup = _numeric(frame.daily_bar_count, "daily_bar_count")
    if warmup.isna().any() or warmup.lt(0).any() or warmup.mod(1).ne(0).any():
        raise DataReadinessError("invalid warm-up counts")
    eligible = _boolean(frame.feature_eligible, "feature_eligible")
    if (eligible & warmup.lt(minimum_warmup)).any():
        raise DataReadinessError("feature-eligible row is under-warm")
    complete = pd.Series(True, index=frame.index)
    clocks = {name: utc_clocks(frame[name], name) for name in set(availability_columns.values())}
    for name in columns:
        values = _numeric(frame[name], name)
        clock = clocks[availability_columns[name]]
        if (values.notna() & (clock.isna() | clock.gt(cutoff))).any():
            raise DataReadinessError(f"future or absent feature clock: {name}")
        complete &= values.notna()
    returns = {name: _numeric(frame[name], name) for name in RETURN_COLUMNS}
    stock = _boolean(frame.stock_source_admitted, "stock_source_admitted")
    if not stock.eq(_no_gaps(frame.stock_missing_reasons, "stock_missing_reasons")).all():
        raise DataReadinessError("stock admission differs from missing reasons")
    maturity = utc_clocks(frame.research_label_mature_at, "research label maturity")
    expected = pd.to_datetime(days.map(maturity_by_session), utc=True)
    if (maturity.notna() & maturity.ne(expected)).any() or (stock & (maturity.isna() | maturity.gt(fit_end))).any():
        raise DataReadinessError("research label maturity differs from exact tenth close or fit boundary")
    gross, net = returns["fixed_horizon_gross_return"], returns["fixed_horizon_net_return"]
    if not gross.notna().eq(stock).all() or not net.notna().eq(stock).all():
        raise DataReadinessError("stock return availability differs from source admission")
    if (stock & net.ne(gross - round_trip_cost)).any():
        raise DataReadinessError("stock cost arithmetic differs; cost must occur exactly once")
    component = frame.stock_component_id
    if (stock & ~component.map(lambda v: isinstance(v, str) and bool(v))).any():
        raise DataReadinessError("admitted stock lacks a bound component")
    comparable = stock.copy()
    for role in ROLES:
        known = _no_gaps(frame[f"{role}_missing_reasons"], f"{role}_missing_reasons")
        if (known & expected.gt(fit_end)).any():
            raise DataReadinessError(f"{role} benchmark extends beyond the fit boundary")
        benchmark, excess = returns[f"{role}_horizon_gross_return"], returns[f"{role}_fixed_horizon_excess_return"]
        if not benchmark.notna().eq(known).all():
            raise DataReadinessError(f"{role} benchmark availability differs from source reasons")
        if (known & ~frame[f"{role}_component_id"].map(lambda v: isinstance(v, str) and bool(v))).any():
            raise DataReadinessError(f"{role} benchmark lacks a bound component")
        paired = stock & known
        if not excess.notna().eq(paired).all() or (paired & excess.ne(net - benchmark)).any():
            raise DataReadinessError(f"{role} excess arithmetic or availability differs")
        comparable &= known
    if not comparable.eq(_boolean(frame.fixed_comparisons_complete, "fixed_comparisons_complete")).all():
        raise DataReadinessError("fixed comparison flag differs from independent components")
    return pd.DataFrame({"feature_eligible": eligible, "model_inputs_complete": complete,
        "decision_inputs_complete": eligible & complete,
        "fixed_horizon_supervision_available": comparable & maturity.le(fit_end),
        "complete_case_supervision": eligible & complete & comparable & maturity.le(fit_end)}, index=frame.index)
