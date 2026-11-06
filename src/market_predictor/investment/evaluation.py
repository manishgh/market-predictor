"""Paired saved investment forecasts: descriptive errors, never promotion evidence."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from typing import Literal, Self

import exchange_calendars as xcals
import pandas as pd
from pydantic import model_validator

from market_predictor.investment.contracts import InvestmentTargetRequest, InvestmentTargetResult
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256, Timestamp


class InvestmentPredictionProvenance(HoldingContract):
    model_id: Identifier
    artifact_sha256: Sha256
    fitted_at_utc: Timestamp
    latest_training_label_available_at_utc: Timestamp
    evidence_scope: Literal["development", "validation", "prospective"]

    @model_validator(mode="after")
    def chronological_fit(self) -> Self:
        if self.latest_training_label_available_at_utc >= self.fitted_at_utc:
            raise ValueError("training labels must be available strictly before fit")
        return self


class InvestmentPredictionRow(HoldingContract):
    request_sha256: Sha256
    policy_sha256: Sha256
    decision_id: Identifier
    security_id: Identifier
    decision_session: date
    horizon_sessions: Literal[63, 252]
    decision_time_utc: Timestamp
    provenance: InvestmentPredictionProvenance
    prediction: float | None
    missing_reason: Identifier | None

    @model_validator(mode="after")
    def valid_prediction(self) -> Self:
        if (self.prediction is None) != (self.missing_reason is not None):
            raise ValueError("missing predictions require exactly one explicit reason")
        if self.provenance.fitted_at_utc >= self.decision_time_utc:
            raise ValueError("fit must precede scoring decision strictly")
        return self


def _unique(rows: Sequence[InvestmentPredictionRow]) -> dict[str, InvestmentPredictionRow]:
    result = {}
    for original in rows:
        row = InvestmentPredictionRow.model_validate_json(original.model_dump_json())
        if row.request_sha256 in result:
            raise ValueError("duplicate forecast request")
        result[row.request_sha256] = row
    return result


def _rank_correlation(predicted: list[float], observed: list[float]) -> float | None:
    if len(predicted) < 3 or len(set(predicted)) < 2 or len(set(observed)) < 2:
        return None
    return float(pd.Series(predicted).rank().corr(pd.Series(observed).rank()))


def evaluate_paired_investment_predictions(
    *, requests: Sequence[InvestmentTargetRequest], targets: Sequence[InvestmentTargetResult],
    baseline: Sequence[InvestmentPredictionRow], candidate: Sequence[InvestmentPredictionRow],
    evaluation_as_of_utc: datetime,
) -> dict[str, object]:
    """Compare the entire declared cohort; explicit missing rows cannot improve coverage.

    Supplied source facts are not independently admitted here. Input hashes and policy
    bindings prove consistency, not provider authority or untouched-test status.
    """
    if evaluation_as_of_utc.tzinfo is None or evaluation_as_of_utc.utcoffset() is None:
        raise ValueError("evaluation clock must be timezone aware")
    request_map: dict[str, InvestmentTargetRequest] = {}
    identities: set[tuple[str, date, int]] = set()
    for original in requests:
        request = InvestmentTargetRequest.model_validate_json(original.model_dump_json())
        identity = (request.holdings[0].security_id, request.decision_session, request.policy.horizon_sessions)
        if request.sha256() in request_map or identity in identities:
            raise ValueError("duplicate decision/security/horizon")
        identities.add(identity)
        request_map[request.sha256()] = request
    if len({request.policy.horizon_sessions for request in request_map.values()}) > 1:
        raise ValueError("evaluate horizons separately")
    target_map: dict[str, InvestmentTargetResult] = {}
    for original_target in targets:
        target = InvestmentTargetResult.model_validate_json(original_target.model_dump_json())
        if target.request_sha256 in target_map:
            raise ValueError("duplicate target")
        target_map[target.request_sha256] = target
    arms = {"baseline": _unique(baseline), "candidate": _unique(candidate)}
    if set(target_map) != set(request_map) or any(set(rows) != set(request_map) for rows in arms.values()):
        raise ValueError("targets and both forecast arms must contain the full identical request cohort")
    calendar = xcals.get_calendar("XNYS")
    excluded: Counter[str] = Counter()
    missing: Counter[str] = Counter()
    groups: dict[date, list[tuple[float, float, float]]] = defaultdict(list)
    eligible_count = 0
    model_dates: dict[tuple[str, date], tuple[str, str, datetime, datetime, str]] = {}
    for key, request in request_map.items():
        target = target_map[key]
        session = pd.Timestamp(request.decision_session)
        if not calendar.is_session(session):
            raise ValueError("decision must be an XNYS session")
        dates = calendar.sessions_window(session, request.policy.horizon_sessions + 1)[1:]
        entry = calendar.session_open(dates[0]).to_pydatetime()
        end = calendar.session_close(dates[request.policy.horizon_sessions - 1]).to_pydatetime()
        if not calendar.session_close(session).to_pydatetime() <= request.decision_time_utc < entry:
            raise ValueError("decision clock differs from canonical next-open policy")
        if (target.policy_sha256 != request.policy.sha256() or target.decision_id != request.decision_id
                or target.horizon_sessions != request.policy.horizon_sessions
                or target.entry_time_utc != entry or target.horizon_end_utc != end
                or any((value.security_id, value.specification_sha256) !=
                    (binding.security_id, binding.specification_sha256)
                    for value, binding in zip(target.holdings, request.holdings, strict=True))):
            raise ValueError("target binding or canonical holding interval differs from request")
        for name, rows in arms.items():
            row = rows[key]
            if (row.policy_sha256 != request.policy.sha256() or row.decision_id != request.decision_id
                    or row.security_id != request.holdings[0].security_id
                    or row.decision_session != request.decision_session
                    or row.decision_time_utc != request.decision_time_utc
                    or row.horizon_sessions != request.policy.horizon_sessions):
                raise ValueError("forecast binding differs from request")
            provenance = row.provenance
            model_identity = (provenance.model_id, provenance.artifact_sha256, provenance.fitted_at_utc,
                              provenance.latest_training_label_available_at_utc, provenance.evidence_scope)
            model_key = (name, request.decision_session)
            if model_key in model_dates and model_dates[model_key] != model_identity:
                raise ValueError("each arm must use one model provenance per decision date")
            model_dates[model_key] = model_identity
            if row.missing_reason is not None:
                missing[f"{name}:{row.missing_reason}"] += 1
        reasons = []
        if not target.label_eligible:
            reasons.append("target_unavailable")
            for reason in target.missing_reasons:
                missing[f"target:{reason}"] += 1
        elif target.label_available_at_utc is None or target.label_available_at_utc > evaluation_as_of_utc:
            reasons.append("target_not_matured")
        else:
            eligible_count += 1
        for name, rows in arms.items():
            if rows[key].prediction is None:
                reasons.append(f"{name}_prediction_missing")
        if reasons:
            excluded.update(reasons)
            continue
        observed = target.net_excess_vs_spy
        left, right = arms["baseline"][key].prediction, arms["candidate"][key].prediction
        if observed is None or left is None or right is None:
            raise ValueError("eligible paired target requires finite values")
        groups[request.decision_session].append((observed, left, right))
    daily: list[dict[str, object]] = []
    for day, values in sorted(groups.items()):
        metrics: dict[str, object] = {"decision_session": day.isoformat(), "paired_rows": len(values)}
        observed_outcomes = [row[0] for row in values]
        for name, position in (("baseline", 1), ("candidate", 2), ("zero_spy_excess", 0)):
            predictions = [0.0 if position == 0 else row[position] for row in values]
            errors = [prediction - outcome for prediction, outcome in zip(predictions, observed_outcomes, strict=True)]
            metrics[name] = {"mae": sum(abs(error) for error in errors) / len(errors),
                             "mse": sum(error * error for error in errors) / len(errors),
                             "spearman": _rank_correlation(predictions, observed_outcomes)}
        daily.append(metrics)
    aggregate: dict[str, object] = {}
    for name in ("baseline", "candidate", "zero_spy_excess"):
        summary: dict[str, object] = {}
        for metric in ("mae", "mse", "spearman"):
            scores = []
            for day_metrics in daily:
                block = day_metrics[name]
                assert isinstance(block, dict)
                score = block[metric]
                if score is not None:
                    scores.append(float(score))
            summary[metric] = sum(scores) / len(scores) if scores else None
            if metric == "spearman":
                summary["spearman_eligible_dates"] = len(scores)
        aggregate[name] = summary
    paired_count = sum(len(values) for values in groups.values())
    return {"scope": "research_only_saved_forecast_diagnostics", "horizon_sessions":
            next(iter(request_map.values())).policy.horizon_sessions if request_map else None,
            "primary_horizon_sessions": 252, "diagnostic_horizon_sessions": 63,
            "evaluation_as_of_utc": evaluation_as_of_utc.isoformat(),
            "cohort_rows": len(request_map), "outcome_eligible_rows": eligible_count,
            "paired_rows": paired_count, "excluded_rows": len(request_map) - paired_count,
            "exclusion_reason_counts": dict(sorted(excluded.items())),
            "missing_reason_counts": dict(sorted(missing.items())),
            "pairing_coverage": paired_count / len(request_map) if request_map else None,
            "eligible_outcome_pairing_coverage": paired_count / eligible_count if eligible_count else None,
            "paired_decision_dates": len(daily), "aggregation": "equal_weight_per_decision_date",
            "model_provenance": [
                {"arm": arm, "decision_session": day.isoformat(), "model_id": identity[0],
                 "artifact_sha256": identity[1], "fitted_at_utc": identity[2].isoformat(),
                 "latest_training_label_available_at_utc": identity[3].isoformat(),
                 "evidence_scope": identity[4]}
                for (arm, day), identity in sorted(model_dates.items())],
            "metrics": aggregate, "per_decision_date": daily,
            "overlap_notice": "Long-horizon outcomes overlap; rows and dates are not independent samples.",
            "confidence_intervals": None, "annual_compounded_returns": None,
            "source_authority_verified": False, "untouched_test_evidence": False,
            "profitability_established": False, "training_ready": False,
            "promotion_eligible": False, "production_eligible": False}
