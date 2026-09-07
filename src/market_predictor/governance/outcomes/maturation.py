"""Governed orchestration for causal prediction-outcome maturation."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeAlias

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    round_trip_cost_bps,
)
from market_predictor.governance.outcomes.contracts import (
    MaturationAttemptV1,
    MaturedOutcomeV2,
    PredictionMaturationIntentV2,
    content_sha256,
)
from market_predictor.intraday.evaluation.outcome_maturation import (
    evaluate_intraday_maturation,
)
from market_predictor.modeling.maturation import MaturedPath, PendingPath
from market_predictor.swing.evaluation.outcome_maturation import (
    evaluate_swing_maturation,
)

MaturationResult: TypeAlias = MaturationAttemptV1 | MaturedOutcomeV2
_BAR_COLUMNS = {
    "ticker",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_feed",
    "adjustment",
    "timeframe",
}


def mature_prediction(
    intent: PredictionMaturationIntentV2,
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    source_artifact_sha256: str,
) -> tuple[MaturationResult, list[dict[str, object]]]:
    observed = _aware_utc(observed_as_of)
    data = _prepare_bars(
        bars,
        observed_as_of=observed,
        source_artifact_sha256=source_artifact_sha256,
        required_price_feed=intent.price_feed,
        required_timeframe="1d" if intent.view == "swing" else "1m",
    )
    evaluated = (
        evaluate_swing_maturation(intent, data)
        if intent.view == "swing"
        else evaluate_intraday_maturation(intent, data, observed_at=observed)
    )
    if isinstance(evaluated, PendingPath):
        return (
            maturation_attempt(
                intent,
                observed_as_of=observed,
                status="pending",
                reasons=evaluated.reasons,
                missing_intervals=evaluated.missing_intervals,
            ),
            [],
        )
    return _outcome_from_path(intent, evaluated), evaluated.evidence_rows


def maturation_attempt(
    intent: PredictionMaturationIntentV2,
    *,
    observed_as_of: datetime,
    status: str,
    reasons: tuple[str, ...],
    missing_intervals: tuple[str, ...] = (),
) -> MaturationAttemptV1:
    if status not in {"pending", "blocked"}:
        raise ValueError("maturation attempt status must be pending or blocked")
    observed = _aware_utc(observed_as_of)
    base = {
        "contract_version": "market_predictor.maturation_attempt.v1",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "observed_as_of_utc": observed,
        "status": status,
        "reasons": reasons,
        "missing_intervals": missing_intervals,
    }
    return MaturationAttemptV1.model_validate(
        {**base, "attempt_id": content_sha256(base)}
    )


def _outcome_from_path(
    intent: PredictionMaturationIntentV2,
    path: MaturedPath,
) -> MaturedOutcomeV2:
    assert intent.decision_atr is not None
    participation = 0.0
    execution_cost_bps = round_trip_cost_bps(
        price=path.entry_price,
        atr_pct=intent.decision_atr / path.entry_price,
        participation=participation,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    net_return = path.gross_return - execution_cost_bps / 10_000.0
    base = {
        "contract_version": "market_predictor.matured_outcome.v2",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "snapshot_id": intent.snapshot_id,
        "ticker": intent.ticker,
        "view": intent.view,
        "horizon": intent.horizon,
        "entry_time_utc": path.entry_time,
        "exit_time_utc": path.exit_time,
        "label_available_at_utc": path.label_available,
        "matured_at_utc": path.label_available,
        "entry_price": path.entry_price,
        "exit_price": path.exit_price,
        "gross_return": path.gross_return,
        "label_round_trip_cost_bps": path.label_round_trip_cost_bps,
        "label_net_return": path.label_net_return,
        "execution_policy_sha256": intent.execution_policy_sha256,
        "decision_atr": intent.decision_atr,
        "execution_participation_fraction": participation,
        "execution_cost_bps": execution_cost_bps,
        "net_return": net_return,
        "mfe": path.mfe,
        "mae": path.mae,
        "path_outcome": path.path_outcome,
        "opportunity_target": path.opportunity_target,
        "downside_target": path.downside_target,
        "spy_return": path.spy_return,
        "qqq_return": path.qqq_return,
        "sector_return": path.sector_return,
        "excess_return_vs_spy": net_return - path.spy_return,
        "excess_return_vs_qqq": net_return - path.qqq_return,
        "excess_return_vs_sector": net_return - path.sector_return,
        "evidence_sha256": content_sha256(path.evidence_rows),
    }
    return MaturedOutcomeV2.model_validate(
        {**base, "outcome_id": content_sha256(base)}
    )


def _prepare_bars(
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    source_artifact_sha256: str,
    required_price_feed: str,
    required_timeframe: str,
) -> pd.DataFrame:
    missing = sorted(_BAR_COLUMNS.difference(bars.columns))
    if missing:
        raise DataReadinessError(
            f"maturation bars are missing columns: {', '.join(missing)}"
        )
    if len(source_artifact_sha256) != 64 or any(
        character not in "0123456789abcdef"
        for character in source_artifact_sha256
    ):
        raise DataReadinessError("maturation source artifact identity is invalid")
    data = bars.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        data[column] = pd.to_datetime(data[column], errors="coerce", utc=True)
    if data[["bar_start_utc", "bar_end_utc", "available_at_utc"]].isna().any().any():
        raise DataReadinessError("maturation bars contain invalid timestamps")
    if "session_date_et" not in data:
        data["session_date_et"] = (
            data["bar_start_utc"].dt.tz_convert("America/New_York").dt.date
        )
    else:
        data["session_date_et"] = pd.to_datetime(
            data["session_date_et"],
            errors="coerce",
        ).dt.date
    if data["session_date_et"].isna().any():
        raise DataReadinessError("maturation bars contain invalid sessions")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data[numeric].isna().any().any() or not np.isfinite(
        data[numeric].to_numpy(float)
    ).all():
        raise DataReadinessError("maturation bars contain invalid OHLCV")
    if bool(data["available_at_utc"].lt(data["bar_end_utc"]).any()):
        raise DataReadinessError("maturation bar availability precedes bar completion")
    if bool(
        data["price_feed"]
        .astype(str)
        .str.upper()
        .ne(required_price_feed.upper())
        .any()
    ):
        raise DataReadinessError("maturation bars use an unexpected price feed")
    if bool(data["adjustment"].astype(str).str.lower().ne("all").any()):
        raise DataReadinessError("maturation bars are not fully adjusted")
    if bool(
        data["timeframe"]
        .astype(str)
        .str.lower()
        .str.strip()
        .ne(required_timeframe)
        .any()
    ):
        raise DataReadinessError("maturation bars use an unexpected timeframe")
    if bool(data.duplicated(["ticker", "bar_start_utc"]).any()):
        raise DataReadinessError("maturation bars contain duplicate ticker intervals")
    data = data[data["available_at_utc"].le(observed_as_of)].copy()
    data["source_artifact_sha256"] = source_artifact_sha256
    return data.sort_values(["ticker", "bar_start_utc"], kind="stable")


def _aware_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("observed_as_of must be timezone-aware")
    return value.astimezone(UTC)
