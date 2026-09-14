"""Exact decision joins for separately verified research features and outcomes."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import numpy as np
import pandas as pd

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract

DECISION_KEYS = ("decision_id", "security_id", "ticker", "decision_time_utc")
_OUTCOME_PREFIXES = ("future_", "managed_", "label_", "exit_", "entry_", "barrier_", "forward_",
    "rank_", "ranking_", "target_", "stop_")


def _clocks(values: pd.Series, name: str) -> pd.Series:
    for value in values.dropna():
        if not isinstance(value, (str, datetime, pd.Timestamp)):
            raise DataReadinessError(f"{name} requires explicit timezone-aware timestamps")
        try:
            if pd.Timestamp(value).tzinfo is None:
                raise ValueError("naive clock")
        except (ValueError, TypeError) as exc:
            raise DataReadinessError(f"{name} requires explicit timezone-aware timestamps") from exc
    return pd.to_datetime(values, utc=True, errors="coerce")


def _identities(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if not frame.columns.is_unique or not set(DECISION_KEYS).issubset(frame.columns):
        raise DataReadinessError(f"{name} requires unique columns and complete decision identities")
    result = frame.loc[:, list(DECISION_KEYS)].copy()
    for key in DECISION_KEYS[:-1]:
        if not result[key].map(lambda value: isinstance(value, str) and bool(value.strip()) and value == value.strip()).all():
            raise DataReadinessError(f"{name} has missing or non-canonical {key}")
    if result.decision_id.duplicated().any():
        raise DataReadinessError(f"{name} duplicates a decision")
    clocks = _clocks(result.decision_time_utc, f"{name} cutoffs")
    if clocks.isna().any():
        raise DataReadinessError(f"{name} has invalid decision cutoffs")
    expected = swing_prediction_cutoffs(clocks.dt.tz_convert("America/New_York").dt.date)
    if clocks.ne(expected).any():
        raise DataReadinessError(f"{name} stock identity, ticker or cutoff differs from canonical swing decisions")
    result["decision_time_utc"] = clocks
    if result.duplicated(["security_id", "decision_time_utc"]).any():
        raise DataReadinessError(f"{name} duplicates a security/cutoff")
    return result.set_index("decision_id")


def join_research_features_and_outcomes(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    availability_columns: Mapping[str, str],
    retained_security_ids: frozenset[str],
) -> pd.DataFrame:
    """Join one partition without dropping unresolved decisions or admitting sources.

    Callers must replay the source authorities before invoking this transformation.
    It does not turn a calculated label into admitted data. The input projection is
    explicit, so superseded labels in a historical feature panel cannot leak through.
    Cohort restriction must already precede peer transforms; filtering here is an error.
    """
    left, right = _identities(features, "features"), _identities(outcomes, "outcomes")
    columns = tuple(feature_columns)
    if (not columns or len(columns) != len(set(columns)) or set(columns).intersection(DECISION_KEYS)
            or any(name.startswith(_OUTCOME_PREFIXES) for name in columns)
            or not set(columns).issubset(features.columns)):
        raise DataReadinessError("research join requires distinct predictor columns, never outcome columns")
    if set(availability_columns) != set(columns) or not set(availability_columns.values()).issubset(features.columns):
        raise DataReadinessError("every predictor requires an explicit availability column")
    if any(name.startswith(_OUTCOME_PREFIXES) for name in availability_columns.values()):
        raise DataReadinessError("outcomes cannot be projected as predictor availability clocks")
    if not set(left.security_id).issubset(retained_security_ids):
        raise DataReadinessError("research peer population contains an excluded security")
    if set(left.index) != set(right.index):
        raise DataReadinessError("outcomes must cover every feature decision, including explicit unavailable rows")
    aligned = right.reindex(left.index)
    if not left.equals(aligned):
        raise DataReadinessError("stock identity, ticker or cutoff differs between features and outcomes")
    payload = [name for name in outcomes if name not in DECISION_KEYS]
    if set(payload).intersection((*columns, *availability_columns.values())):
        raise DataReadinessError("outcomes would overwrite predictor values or their availability clocks")
    cutoff = pd.to_datetime(features.decision_time_utc, utc=True)
    for name, clock_column in availability_columns.items():
        values = pd.to_numeric(features[name], errors="coerce")
        if (features[name].notna() & values.isna()).any() or np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            raise DataReadinessError(f"predictor {name} must be numeric, finite or explicitly unavailable")
        available = _clocks(features[clock_column], clock_column)
        if (values.notna() & (available.isna() | available.gt(cutoff))).any():
            raise DataReadinessError(f"predictor {name} is unavailable at its decision cutoff")
    selected = list(dict.fromkeys((*DECISION_KEYS, *columns, *availability_columns.values())))
    result = features.loc[:, selected].merge(
        outcomes.loc[:, ["decision_id", *payload]], on="decision_id", how="left", validate="one_to_one", sort=False,
    )
    if result.decision_id.tolist() != features.decision_id.tolist():
        raise DataReadinessError("research join changed the decision order or population")
    return result


def rebuild_research_peer_features(
    features: pd.DataFrame, *, contract: StrategyContract, retained_security_ids: frozenset[str],
    availability_columns: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Restrict verified base rows before the existing population-relative transform.

    No old target, old peer score or outcome eligibility enters the calculation.
    This cannot repair wrong-issuer base indicators: the caller must rebuild those
    from corrected observations before supplying this source-verified partition.
    """
    from market_predictor.swing.features.panel import (
        CATALYST_RANKING_FEATURES,
        TECHNICAL_RANKING_FEATURES,
        swing_model_feature_columns,
    )
    from market_predictor.swing.features.pipeline import SectorRelativeScalingStep

    _identities(features, "base features")
    context = ("sector", "session_date_et", "feature_profile", "feature_eligible", "daily_bar_count")
    if not set(context).issubset(features):
        raise DataReadinessError("peer rebuild lacks sector, profile or warm-up evidence")
    data = features.loc[features.security_id.isin(retained_security_ids)].copy()
    if data.empty:
        raise DataReadinessError("peer rebuild has no retained decisions")
    profiles = set(data.feature_profile)
    if profiles not in ({"technical_market"}, {"catalyst_full"}):
        raise DataReadinessError("peer rebuild requires one declared feature profile")
    catalyst = profiles == {"catalyst_full"}
    inputs = (*TECHNICAL_RANKING_FEATURES, *(CATALYST_RANKING_FEATURES if catalyst else ()))
    if not data.feature_eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataReadinessError("peer feature eligibility must be explicit booleans")
    counts = pd.to_numeric(data.daily_bar_count, errors="coerce")
    if counts.isna().any() or (~np.isfinite(counts)).any() or counts.lt(0).any() or counts.mod(1).ne(0).any():
        raise DataReadinessError("peer warm-up counts must be nonnegative integers")
    sessions = pd.to_datetime(data.session_date_et, errors="coerce")
    cutoff_dates = pd.to_datetime(data.decision_time_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    if sessions.isna().any() or sessions.dt.date.ne(cutoff_dates).any() or data.sector.isna().any():
        raise DataReadinessError("peer session/sector context differs from decision identity")
    # Reuse predictor validation/projection to strip all legacy labels and peer scores.
    projected = join_research_features_and_outcomes(data, data.loc[:, list(DECISION_KEYS)],
        feature_columns=inputs, availability_columns=availability_columns, retained_security_ids=retained_security_ids)
    projected.index = data.index
    projected.loc[:, list(context)] = data.loc[:, list(context)]
    projected["session_date_et"] = sessions.dt.date
    projected["daily_bar_count"] = counts
    result = SectorRelativeScalingStep(contract).transform(projected)
    columns = swing_model_feature_columns(contract=contract, catalyst=catalyst)
    clocks: dict[str, str] = {}
    clock_values: dict[str, pd.Series] = {}
    eligible = data.feature_eligible & counts.ge(contract.swing.minimum_warmup_sessions)
    for name in columns:
        suffix = next(value for value in ("_sector_z", "_xs_z", "_xs_rank") if name.endswith(value))
        base = name.removesuffix(suffix)
        availability = _clocks(data[availability_columns[base]], availability_columns[base])
        availability = availability.where(eligible & data[base].notna())
        groups = [projected.session_date_et, projected.sector] if suffix == "_sector_z" else projected.session_date_et
        clock_name = f"available_at_{name}"
        clock_values[clock_name] = availability.groupby(groups, sort=False).transform("max")
        clocks[name] = clock_name
    return pd.concat((result, pd.DataFrame(clock_values, index=result.index)), axis=1), clocks
