"""Synthetic unit cases for paired diagnostics, not measured investment evidence."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from market_predictor.investment.contracts import InvestmentTargetRequest, InvestmentTargetResult
from market_predictor.investment.evaluation import (
    InvestmentPredictionProvenance,
    InvestmentPredictionRow,
    evaluate_paired_investment_predictions,
)
from market_predictor.investment.targets import project_investment_targets
from tests.test_investment_targets import _fixture


def _case() -> tuple[InvestmentTargetRequest, InvestmentTargetResult, InvestmentPredictionRow]:
    request, holdings = _fixture()
    target = project_investment_targets(request, holdings)
    row = InvestmentPredictionRow(request_sha256=request.sha256(), policy_sha256=request.policy.sha256(),
        decision_id=request.decision_id, security_id=request.holdings[0].security_id,
        decision_session=request.decision_session, horizon_sessions=request.policy.horizon_sessions,
        decision_time_utc=request.decision_time_utc,
        provenance=InvestmentPredictionProvenance(model_id="saved-model", artifact_sha256="e" * 64,
            fitted_at_utc=request.decision_time_utc - timedelta(days=1),
            latest_training_label_available_at_utc=request.decision_time_utc - timedelta(days=2),
            evidence_scope="development"), prediction=.05, missing_reason=None)
    return request, target, row


def _evaluate(request: InvestmentTargetRequest, target: InvestmentTargetResult,
              baseline: InvestmentPredictionRow, candidate: InvestmentPredictionRow,
              clock: datetime | None = None) -> dict[str, object]:
    return evaluate_paired_investment_predictions(requests=[request], targets=[target], baseline=[baseline],
        candidate=[candidate], evaluation_as_of_utc=clock or target.horizon_end_utc + timedelta(days=1))


def test_paired_errors_use_net_stock_minus_gross_spy_without_compounding() -> None:
    request, target, row = _case()
    result = _evaluate(request, target, row, row.model_copy(update={"prediction": .058}))
    assert result["paired_rows"] == 1
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["baseline"]["mae"] == pytest.approx(.008)
    assert metrics["candidate"]["mse"] == pytest.approx(0)
    assert metrics["zero_spy_excess"]["mae"] == pytest.approx(.058)
    assert metrics["candidate"]["spearman"] is None
    assert result["confidence_intervals"] is None
    assert not result["promotion_eligible"] and not result["untouched_test_evidence"]


def test_full_label_maturity_clock_is_inclusive() -> None:
    request, target, row = _case()
    assert target.label_available_at_utc is not None
    before = _evaluate(request, target, row, row, target.label_available_at_utc - timedelta(microseconds=1))
    assert before["paired_rows"] == 0
    assert before["exclusion_reason_counts"] == {"target_not_matured": 1}
    metrics = before["metrics"]
    assert isinstance(metrics, dict) and metrics["candidate"]["mae"] is None
    assert _evaluate(request, target, row, row, target.label_available_at_utc)["paired_rows"] == 1


def test_missing_prediction_is_explicit_and_denominator_preserved() -> None:
    request, target, row = _case()
    missing = row.model_copy(update={"prediction": None, "missing_reason": "feature_unavailable"})
    result = _evaluate(request, target, row, missing)
    assert result["cohort_rows"] == result["outcome_eligible_rows"] == 1
    assert result["pairing_coverage"] == 0
    assert result["missing_reason_counts"] == {"candidate:feature_unavailable": 1}
    with pytest.raises(ValueError, match="full identical"):
        evaluate_paired_investment_predictions(requests=[request], targets=[target], baseline=[row], candidate=[],
            evaluation_as_of_utc=target.horizon_end_utc)


@pytest.mark.parametrize("field,value", [("prediction", float("nan")), ("prediction", float("inf")),
    ("decision_id", "wrong"), ("security_id", "wrong"), ("policy_sha256", "f" * 64)])
def test_forecast_copy_tampering_rejected(field: str, value: object) -> None:
    request, target, row = _case()
    with pytest.raises(ValueError):
        _evaluate(request, target, row, row.model_copy(update={field: value}))


@pytest.mark.parametrize("field", ["fitted_at_utc", "latest_training_label_available_at_utc"])
def test_future_fit_and_training_label_poison_rejected(field: str) -> None:
    request, target, row = _case()
    provenance = row.provenance.model_copy(update={field: request.decision_time_utc})
    with pytest.raises(ValueError):
        _evaluate(request, target, row, row.model_copy(update={"provenance": provenance}))


def test_training_label_at_fit_and_naive_provenance_are_rejected() -> None:
    request, target, row = _case()
    for change in ({"latest_training_label_available_at_utc": row.provenance.fitted_at_utc},
                   {"fitted_at_utc": row.provenance.fitted_at_utc.replace(tzinfo=None)}):
        with pytest.raises(ValueError):
            _evaluate(request, target, row, row.model_copy(update={
                "provenance": row.provenance.model_copy(update=change)}))


def test_target_binding_clock_and_spy_tampering_rejected() -> None:
    request, target, row = _case()
    for change in ({"request_sha256": "f" * 64}, {"policy_sha256": "f" * 64},
                   {"horizon_end_utc": target.horizon_end_utc + timedelta(days=1)},
                   {"net_excess_vs_spy": .5}):
        with pytest.raises(ValueError):
            _evaluate(request, target.model_copy(update=change), row, row)


def test_unavailable_fourth_holding_excludes_without_fabricating_zero() -> None:
    request, target, row = _case()
    sector = target.holdings[-1].model_copy(update={"missing_reasons": ("missing_mark",),
        "label_available_at_utc": None})
    unavailable = target.model_copy(update={"holdings": (*target.holdings[:-1], sector),
        "missing_reasons": ("sector:missing_mark",), "label_eligible": False,
        "status": "unavailable", "label_available_at_utc": None})
    result = _evaluate(request, unavailable, row, row)
    assert result["paired_rows"] == 0
    assert result["exclusion_reason_counts"] == {"target_unavailable": 1}


def test_naive_clock_duplicate_rows_and_unmatched_target_rejected() -> None:
    request, target, row = _case()
    with pytest.raises(ValueError, match="timezone aware"):
        _evaluate(request, target, row, row, datetime(2026, 11, 1))
    with pytest.raises(ValueError, match="duplicate forecast"):
        evaluate_paired_investment_predictions(requests=[request], targets=[target], baseline=[row, row],
            candidate=[row], evaluation_as_of_utc=target.horizon_end_utc)


def test_equal_date_errors_and_cross_section_spearman() -> None:
    import exchange_calendars as xcals
    import pandas as pd

    request, target, row = _case()
    requests = []
    targets = []
    baseline = []
    candidate = []
    calendar = xcals.get_calendar("XNYS")
    for index, (offset, prediction, stock_return) in enumerate(((0, 0., .08), (0, 0., .1), (0, 0., .12), (1, 1., .1))):
        day = request.decision_session + timedelta(days=offset)
        bindings = (request.holdings[0].model_copy(update={"security_id": f"stock:{index}"}), *request.holdings[1:])
        revised = request.model_copy(update={"decision_id": f"decision:{index}", "holdings": bindings,
            "decision_session": day, "decision_time_utc": request.decision_time_utc + timedelta(days=offset)})
        sessions = calendar.sessions_window(pd.Timestamp(day), 64)[1:]
        end = calendar.session_close(sessions[62]).to_pydatetime()
        values = (target.holdings[0].model_copy(update={"security_id": f"stock:{index}",
            "gross_return": stock_return, "net_return": stock_return - .002,
            "label_available_at_utc": end}), *(value.model_copy(update={"label_available_at_utc": end})
            for value in target.holdings[1:]))
        outcome = target.model_copy(update={"request_sha256": revised.sha256(), "decision_id": revised.decision_id,
            "holdings": values, "entry_time_utc": calendar.session_open(sessions[0]).to_pydatetime(),
            "horizon_end_utc": end, "label_available_at_utc": end,
            "net_excess_vs_spy": stock_return - .002 - .04,
            "net_excess_vs_qqq": stock_return - .002 - .06,
            "net_excess_vs_sector": stock_return - .002 - .08})
        forecast = row.model_copy(update={"request_sha256": revised.sha256(), "decision_id": revised.decision_id,
            "security_id": f"stock:{index}", "decision_session": day,
            "decision_time_utc": revised.decision_time_utc, "prediction": prediction})
        requests.append(revised)
        targets.append(outcome)
        baseline.append(forecast)
        candidate.append(forecast.model_copy(update={"prediction": stock_return - .042}))
    result = evaluate_paired_investment_predictions(requests=requests, targets=targets, baseline=baseline,
        candidate=candidate, evaluation_as_of_utc=target.horizon_end_utc + timedelta(days=20))
    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert metrics["baseline"]["mae"] == pytest.approx((.058 + .942) / 2)
    assert metrics["candidate"]["spearman"] == pytest.approx(1)
    assert metrics["candidate"]["spearman_eligible_dates"] == 1
    assert result["paired_rows"] == 4 and result["paired_decision_dates"] == 2
