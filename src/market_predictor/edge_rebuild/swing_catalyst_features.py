from __future__ import annotations

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.features.catalyst_decision_authority import (
    COVERAGE_FLAG_COLUMNS,
    REQUIRED_MODEL_SOURCE_FAMILIES,
    TRACKED_SOURCE_FAMILIES,
    CatalystDecisionAuthority,
    attach_catalyst_decision_features,
)


def build_swing_ablation_rows(
    technical_rows: pd.DataFrame,
    catalyst_authority: CatalystDecisionAuthority,
) -> dict[str, pd.DataFrame]:
    """Create matched technical and catalyst populations from one row authority."""
    from market_predictor.edge_rebuild.swing_features import (
        CATALYST_AUDIT_FEATURES,
        CATALYST_RANKING_FEATURES,
        SWING_CATALYST_FEATURE_PROFILE,
        SWING_FEATURE_PROFILE,
    )

    attached = attach_catalyst_decision_features(
        technical_rows,
        catalyst_authority,
    )
    required_flags = {
        f"source_coverage_known_{family}_{window}"
        for family in REQUIRED_MODEL_SOURCE_FAMILIES
        for window in ("1d", "3d")
    }
    required = {
        "feature_eligible",
        "label_eligible",
        "catalyst_source_complete_1d",
        "catalyst_source_complete_3d",
        *COVERAGE_FLAG_COLUMNS,
        *CATALYST_AUDIT_FEATURES,
        *CATALYST_RANKING_FEATURES,
        *required_flags,
    }
    missing = sorted(required.difference(attached.columns))
    if missing:
        raise DataReadinessError(
            f"catalyst ablation rows are missing authority columns: {missing}"
        )

    required_complete = pd.Series(True, index=attached.index)
    for window in ("1d", "3d"):
        declared = (
            attached[f"catalyst_source_complete_{window}"]
            .fillna(False)
            .astype(bool)
        )
        observed = pd.concat(
            [
                attached[f"source_coverage_known_{family}_{window}"]
                .fillna(False)
                .astype(bool)
                for family in REQUIRED_MODEL_SOURCE_FAMILIES
            ],
            axis=1,
        ).all(axis=1)
        if bool(declared.ne(observed).any()):
            raise DataReadinessError(
                f"catalyst required-source completeness conflicts in {window}"
            )
        required_complete &= observed
    attached = _scope_catalyst_aggregates_to_required_sources(attached)
    comparable = (
        attached["feature_eligible"].fillna(False).astype(bool)
        & required_complete
    )
    base = attached.copy()
    base["pre_catalyst_feature_eligible"] = (
        base["feature_eligible"].fillna(False).astype(bool)
    )
    base["catalyst_required_source_complete"] = required_complete
    base["ablation_population_eligible"] = comparable
    base["feature_eligible"] = comparable
    base["label_eligible"] = (
        base["label_eligible"].fillna(False).astype(bool) & comparable
    )

    technical = base.copy()
    technical["feature_profile"] = SWING_FEATURE_PROFILE
    catalyst = base.copy()
    catalyst["feature_profile"] = SWING_CATALYST_FEATURE_PROFILE
    return {
        SWING_FEATURE_PROFILE: technical,
        SWING_CATALYST_FEATURE_PROFILE: catalyst,
    }



def _scope_catalyst_aggregates_to_required_sources(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Expose aggregate model fields only when counts prove Alpaca-only evidence."""

    if REQUIRED_MODEL_SOURCE_FAMILIES != ("alpaca",):
        raise DataReadinessError(
            "swing aggregate scoping currently requires Alpaca as the sole "
            "historical model source"
        )
    output = rows.copy()
    aggregate_bases = (
        "sentiment_mean",
        "sentiment_coverage",
        "event_relevance_mean",
        "low_relevance_event_fraction",
    )
    for window in ("1d", "3d"):
        event_column = f"event_count_{window}"
        alpaca_column = f"source_count_alpaca_{window}"
        event_count = pd.to_numeric(output[event_column], errors="coerce")
        alpaca_count = pd.to_numeric(output[alpaca_column], errors="coerce")
        optional_positive = pd.Series(False, index=output.index)
        for family in TRACKED_SOURCE_FAMILIES:
            if family in REQUIRED_MODEL_SOURCE_FAMILIES:
                continue
            optional_count = pd.to_numeric(
                output[f"source_count_{family}_{window}"],
                errors="coerce",
            )
            optional_positive |= optional_count.gt(0).fillna(False)
        if bool((event_count < alpaca_count).fillna(False).any()):
            raise DataReadinessError(
                f"catalyst event count is below Alpaca evidence in {window}"
            )
        alpaca_only = (
            event_count.notna()
            & alpaca_count.notna()
            & np.isclose(event_count, alpaca_count)
            & ~optional_positive
        )
        output[event_column] = alpaca_count
        for base in aggregate_bases:
            column = f"{base}_{window}"
            numeric = pd.to_numeric(output[column], errors="coerce")
            output[column] = numeric.where(alpaca_only)
        output[f"alpaca_aggregate_complete_{window}"] = alpaca_only
    return output



