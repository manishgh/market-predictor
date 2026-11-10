"""Synthetic unit fixtures for retrospective split and economic reporting contracts."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.training import (
    TARGET,
    compare_feature_profiles,
    investment_diagnostics,
    investment_fold,
)
from market_predictor.swing.training.return_estimators import fit_return_regressor, predict_return
from market_predictor.swing.training.return_validation import security_transfer_mask


@pytest.fixture
def metadata() -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    days = calendar.sessions_in_range("2019-07-01", "2024-05-28")
    months = pd.Series(days, index=days.strftime("%Y-%m")).groupby(level=0).last()
    rows = []
    for day in months:
        for horizon in (63, 252):
            end = calendar.session_close(calendar.session_offset(day, horizon))
            for security in range(80):
                rows.append({"decision_id": f"{day.date()}-{security}", "security_id": f"stock-{security}",
                    "decision_session": day.date(), "decision_time_utc": calendar.session_close(day),
                    "horizon_sessions": horizon, "economic_maturity_at_utc": end,
                    "label_available_at_utc": datetime(2026, 11, 10, tzinfo=UTC),
                    "research_eligible": True, "feature_eligible": True})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("horizon", [63, 252])
def test_exchange_session_embargo_and_strict_economic_maturity(metadata: pd.DataFrame, horizon: int) -> None:
    fold = investment_fold(metadata, horizon, 2022, "temporal")
    assert not fold.missing_reasons
    assert fold.information_cutoff_utc == pd.Timestamp("2022-01-31T21:00:00Z")
    calendar = xcals.get_calendar("XNYS")
    assert fold.training_maturity_cutoff_utc == calendar.session_close(calendar.session_offset("2022-01-31", -21))
    assert metadata.iloc[fold.train_indices].economic_maturity_at_utc.lt(fold.training_maturity_cutoff_utc).all()
    assert metadata.iloc[fold.train_indices].label_available_at_utc.gt(fold.information_cutoff_utc).all()
    # Historical folds use economic maturity while retaining real 2026 verification.
    poisoned_index = fold.train_indices[0]
    metadata.loc[poisoned_index, "economic_maturity_at_utc"] = fold.training_maturity_cutoff_utc
    assert poisoned_index not in investment_fold(metadata, horizon, 2022, "temporal").train_indices


def test_security_transfer_never_trains_on_held_out_ids(metadata: pd.DataFrame) -> None:
    fold = investment_fold(metadata, 252, 2023, "security_transfer")
    assert not fold.missing_reasons
    train = metadata.iloc[fold.train_indices]
    score = metadata.iloc[fold.score_indices]
    assert not set(train.security_id).intersection(score.security_id)
    assert not security_transfer_mask(train.security_id).any()
    assert security_transfer_mask(score.security_id).all()


def test_unknown_outcomes_do_not_change_prediction_population(metadata: pd.DataFrame) -> None:
    before = investment_fold(metadata, 63, 2023, "temporal")
    metadata.loc[before.prediction_indices[:50], "research_eligible"] = False
    after = investment_fold(metadata, 63, 2023, "temporal")
    assert np.array_equal(before.prediction_indices, after.prediction_indices)
    assert np.array_equal(before.score_indices, after.score_indices)


def test_insufficient_scope_is_explicit_not_threshold_relaxation(metadata: pd.DataFrame) -> None:
    metadata["research_eligible"] = False
    fold = investment_fold(metadata, 252, 2022, "temporal")
    assert "fewer_than_100_training_rows" in fold.missing_reasons
    assert "fewer_than_12_training_months" in fold.missing_reasons
    assert "fewer_than_20_training_securities" in fold.missing_reasons
    assert "fewer_than_30_known_scoring_rows" in fold.missing_reasons
    assert len(fold.prediction_indices) > 0


def predictions() -> pd.DataFrame:
    rows = []
    for month, count in (("2023-01-31", 11), ("2023-02-28", 20)):
        for i in range(count):
            rows.append({"decision_id": f"{month}-{i}", "security_id": f"stock-{i:02}",
                "decision_session": month, "predicted_excess_return": float(count - i),
                TARGET: 0.198, "stock_net_return": 0.298, "spy_gross_return": 0.1,
                "qqq_gross_return": 0.2, "sector_gross_return": 0.15, "research_eligible": True})
    return pd.DataFrame(rows)


def test_decile_rounding_ties_cost_once_and_matched_spy() -> None:
    frame = predictions()
    frame["predicted_excess_return"] = 0.0
    result = investment_diagnostics(frame)
    selected = result["top_decile_by_date"]
    assert isinstance(selected, list)
    assert [day["selected_rows"] for day in selected] == [2, 2]
    assert all(day["selected_security_ids"] == ["stock-00", "stock-01"] for day in selected)
    assert result["top_decile_mean_net_excess_vs_spy"] == pytest.approx(0.198)
    assert result["zero_excess_baseline_mse"] == pytest.approx(0.198**2)
    assert result["portfolio_evaluated"] is False


def test_selected_unknown_outcome_never_replaced_or_claimed_complete() -> None:
    frame = predictions()
    frame.loc[0, "research_eligible"] = False
    frame.loc[0, TARGET] = np.nan
    result = investment_diagnostics(frame)
    selected: Any = result["top_decile_by_date"]
    assert selected[0]["selected_security_ids"] == ["stock-00", "stock-01"]
    assert selected[0]["known_selected_outcomes"] == 1
    assert selected[0]["mean_net_excess_vs_spy"] is None
    assert result["top_decile_mean_net_excess_vs_spy"] is None
    assert result["unknown_scored_outcomes"] == 1


def test_metrics_equal_date_weights_not_equal_rows_or_folds() -> None:
    frame = predictions()
    frame["predicted_excess_return"] = 0.0
    frame[TARGET] = np.where(frame.decision_session.eq("2023-01-31"), 1.0, 3.0)
    frame["stock_net_return"] = frame[TARGET] + frame.spy_gross_return
    result = investment_diagnostics(frame)
    assert result["mae"] == pytest.approx(2.0)
    assert result["mse"] == pytest.approx(5.0)


@pytest.mark.parametrize("field,value", [(TARGET, 9.0), ("spy_gross_return", np.nan), ("qqq_gross_return", np.nan)])
def test_complete_economics_require_consistent_finite_holdings(field: str, value: float) -> None:
    frame = predictions()
    frame.loc[0, field] = value
    with pytest.raises(DataReadinessError, match="complete matched holding returns"):
        investment_diagnostics(frame)


def test_profile_comparison_refuses_different_denominators() -> None:
    base = investment_diagnostics(predictions())
    full = dict(base, scored_rows=0)
    with pytest.raises(DataReadinessError, match="populations differ"):
        compare_feature_profiles({"252/technical_relationships/regularized_linear_return/temporal": base,
            "252/all_features/regularized_linear_return/temporal": full})


def test_train_only_imputation_and_scaling_ignore_scoring_values() -> None:
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0, 5.0], "b": [np.nan] * 4})
    model = fit_return_regressor("regularized_linear_return", train, np.arange(4, dtype=float), np.ones(4),
        {"alpha": 1.0, "fit_intercept": True, "solver": "lsqr", "tol": 1e-6, "max_iter": 10000})
    before = model.medians.copy()
    scaler_before = model.scaler.mean_.copy()
    predict_return(model, pd.DataFrame({"a": [1e10, np.nan], "b": [1e8, np.nan]}))
    assert np.array_equal(model.medians, before)
    assert np.array_equal(model.scaler.mean_, scaler_before)
    assert before.tolist() == [3.0, 0.0]


def test_custom_shared_lease_refuses_before_training_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
    from market_predictor.research import investment_training as owner

    runtime = tmp_path / "shared-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime))
    with heavy_job_lease("other-job", runtime_dir=runtime), pytest.raises(HeavyJobBusyError):
        owner.train_investment_models(root=tmp_path, request=tmp_path / "does-not-exist.json",
            request_sha256="0" * 64, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_actual_fit_clock_cannot_precede_label_verification(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.research import investment_training as owner

    monkeypatch.setattr(owner, "guard", lambda stage: None)
    metadata = pd.DataFrame({"label_available_at_utc": [pd.Timestamp("2099-01-01T00:00:00Z")]})
    with pytest.raises(DataReadinessError, match="actual investment fit"):
        owner._publish_fit(tmp_path, "future", features=pd.DataFrame({"x": [1.0]}), metadata=metadata,
            columns=("x",), family="regularized_linear_return", parameters={}, train_indices=np.array([0]),
            score_indices=np.array([], dtype=int), prediction_indices=np.array([], dtype=int), identity={},
            request_sha256="0" * 64, admitted_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert not (tmp_path / "future").exists()


def test_saved_fit_preserves_actual_clock_and_missing_predictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.research import investment_training as owner

    monkeypatch.setattr(owner, "guard", lambda stage: None)
    meta = predictions().iloc[:8].copy()
    meta["label_available_at_utc"] = pd.Timestamp("2025-01-01T00:00:00Z")
    meta["economic_maturity_at_utc"] = pd.Timestamp("2024-01-01T00:00:00Z")
    features = pd.DataFrame({"x": np.arange(8, dtype=float)})
    record, scored = owner._publish_fit(tmp_path, "unit", features=features, metadata=meta, columns=("x",),
        family="regularized_linear_return", parameters={"alpha": 1.0, "fit_intercept": True, "solver": "lsqr",
            "tol": 1e-6, "max_iter": 10000}, train_indices=np.array([0, 1, 2, 3]),
        score_indices=np.array([4, 5, 6, 7]), prediction_indices=np.array([4, 5, 6]), identity={},
        request_sha256="0" * 64, admitted_at=datetime(2025, 1, 1, tzinfo=UTC))
    assert record["persistence_prediction_parity"] is True
    assert record["serving_eligible"] is False
    assert scored is not None and scored.predicted_excess_return.isna().sum() == 1
    assert scored.prediction_generated_at_utc.ge(scored.actual_fitted_at_utc).all()
    assert scored.actual_fitted_at_utc.gt(pd.to_datetime(scored.decision_session, utc=True)).all()
    assert (tmp_path / "unit/model.joblib").is_file()
