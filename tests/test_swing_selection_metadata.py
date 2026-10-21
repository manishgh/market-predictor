"""Synthetic UNIT proof of omitted selection context; no operational evidence."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.research import swing_oof_policy_evaluation as owner
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from market_predictor.swing.features.pipeline import SectorRelativeScalingStep


@pytest.fixture
def cohort() -> tuple[pd.DataFrame, StrategyContract]:
    strategy = load_strategy_contract(Path(__file__).parents[1] / "configs/edge_rebuild_strategy_contract.toml")
    minimum = strategy.labels.minimum_cross_section_for_ranking
    rows = pd.DataFrame([(day, sector, number) for day in (date(2024, 1, 2), date(2024, 1, 3))
                         for sector in ("A", "B", "C") for number in range(minimum)],
                        columns=["session_date_et", "sector", "number"])
    rows["security_id"] = rows.sector + rows.number.astype(str)
    rows["decision_id"] = rows.session_date_et.astype(str) + rows.security_id
    rows["ticker"] = rows.security_id
    rows["primary_benchmark"] = "XLF"
    rows["decision_time_utc"] = swing_prediction_cutoffs(rows.session_date_et)
    rows["feature_profile"] = "technical_market"
    rows["feature_eligible"] = True
    rows["daily_bar_count"] = strategy.swing.minimum_warmup_sessions
    rows.loc[rows.sector.eq("B") & rows.number.eq(0), "daily_bar_count"] -= 1
    rows.loc[rows.sector.eq("C") & rows.number.eq(0), "feature_eligible"] = False
    rows = rows.drop(columns="number")
    return rows, strategy


def _compose(rows: pd.DataFrame, strategy: StrategyContract) -> pd.DataFrame:
    return owner._selection_metadata(owner._metadata(rows), strategy)


def test_exact_canonical_peer_gate_from_missing_selection_columns(cohort: tuple[pd.DataFrame, StrategyContract]) -> None:
    rows, strategy = cohort
    assert not {"cross_section_eligible", "decision_group_id"}.intersection(rows)
    numerical = rows.assign(**{name: np.arange(len(rows), dtype="float32") for name in TECHNICAL_RANKING_FEATURES})
    reference = SectorRelativeScalingStep(strategy).transform(numerical)
    result = _compose(rows, strategy)
    pd.testing.assert_series_equal(result.cross_section_eligible, reference.sector_rank_eligible,
                                   check_names=False, check_dtype=False)
    assert result.cross_section_eligible.eq(result.sector.eq("A")).all()
    assert result.decision_group_id.equals(result.decision_time_utc.map(lambda value: value.isoformat()))
    pd.testing.assert_frame_equal(result[rows.columns], rows, check_dtype=False)


def test_full_cohort_precedes_scope_and_targets_never_change_selection(cohort: tuple[pd.DataFrame, StrategyContract]) -> None:
    rows, strategy = cohort
    rows = rows.assign(scope_eligible=False, spy_fixed_horizon_excess_return=np.nan,
                       predicted_excess_return=-0.5, forward_return=np.inf, barrier_label=None)
    rows.loc[rows.sector.eq("A") & rows.security_id.str.endswith("0"), "scope_eligible"] = True
    result = _compose(rows, strategy)
    assert result.loc[result.scope_eligible, "cross_section_eligible"].all()
    changed = rows.assign(spy_fixed_horizon_excess_return=999, forward_return=-999,
                          barrier_label=1, predicted_excess_return=999)
    pd.testing.assert_frame_equal(result[["decision_group_id", "cross_section_eligible"]],
                                   _compose(changed, strategy)[["decision_group_id", "cross_section_eligible"]])
    subset = _compose(rows.loc[rows.scope_eligible], strategy)
    assert not subset.cross_section_eligible.any()  # A premature scope filter would change the result.


@pytest.mark.parametrize("value", [None, True, np.inf, -1, 250.5, "250"])
def test_invalid_warmup_refuses(cohort: tuple[pd.DataFrame, StrategyContract], value: Any) -> None:
    rows, strategy = cohort
    rows["daily_bar_count"] = rows.daily_bar_count.astype(object)
    rows.loc[0, "daily_bar_count"] = value
    with pytest.raises(DataReadinessError, match="warm-up counts"):
        _compose(rows, strategy)


@pytest.mark.parametrize("poison", ["duplicate_security", "duplicate_id", "clock", "group", "cross_section", "boolean"])
def test_identity_clock_and_supplied_metadata_poison_refuses(cohort: tuple[pd.DataFrame, StrategyContract], poison: str) -> None:
    rows, strategy = cohort
    if poison == "duplicate_security":
        rows.loc[1, "security_id"] = rows.loc[0, "security_id"]
    elif poison == "duplicate_id":
        rows.loc[1, "decision_id"] = rows.loc[0, "decision_id"]
    elif poison == "clock":
        rows.loc[0, "decision_time_utc"] += pd.Timedelta(minutes=1)
    elif poison == "group":
        rows["decision_group_id"] = "wrong"
    elif poison == "cross_section":
        rows["cross_section_eligible"] = True
    else:
        rows["feature_eligible"] = "true"
    with pytest.raises(DataReadinessError):
        _compose(rows, strategy)


def test_consistent_supplied_metadata_and_selection_zero_days(cohort: tuple[pd.DataFrame, StrategyContract]) -> None:
    rows, strategy = cohort
    first = _compose(rows, strategy)
    pd.testing.assert_frame_equal(_compose(first, strategy), first)
    first = first.assign(scope_eligible=True, predicted_excess_return=-0.01)
    days = tuple(sorted(first.session_date_et.astype(str).unique()))
    complete, selected, counts = owner._select(first, days, days, strategy)
    # Only sector A passes its peer gate, so canonical minimum-three-sectors selection abstains.
    assert selected.empty and len(complete) == len(rows)
    assert len(counts) == len(days) and all(item["selected_rows"] == 0 for item in counts)
