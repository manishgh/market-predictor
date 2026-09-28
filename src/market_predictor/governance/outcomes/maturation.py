"""Governed orchestration for causal swing prediction-outcome maturation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import TypeAlias

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    round_trip_cost_bps,
)
from market_predictor.governance.outcomes.contracts import (
    MaturationAttempt,
    MaturedOutcome,
    PredictionMaturationIntent,
    SensitivityFill,
    content_sha256,
)
from market_predictor.modeling.maturation import MaturedPath, PendingPath
from market_predictor.swing.evaluation.outcome_maturation import (
    evaluate_swing_maturation,
)

MaturationResult: TypeAlias = MaturationAttempt | MaturedOutcome
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
    "source_artifact_sha256",
}


@dataclass(frozen=True)
class AttemptContext:
    """The drift policy terms an attempt ran under and the receipts it read."""

    grace_days: int
    settlement_days: int
    drift_policy_sha256: str
    receipt_ids: Sequence[str] = ()


def mature_prediction(
    intent: PredictionMaturationIntent,
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    proven_stock_gaps: frozenset[date],
) -> tuple[MaturedOutcome | PendingPath, list[dict[str, object]]]:
    """Mature one intent on `bars`, each of which names its source in `source_artifact_sha256`.

    `proven_stock_gaps` holds the sessions in which the stock is proven to have no usable
    bar; a target or stop reached before such a session matures without waiting for it.
    """
    data = _prepare_bars(
        bars,
        observed_as_of=_aware_utc(observed_as_of),
        required_price_feed=intent.price_feed,
        required_timeframe="1d",
    )
    evaluated = evaluate_swing_maturation(intent, data, proven_stock_gaps=proven_stock_gaps)
    if isinstance(evaluated, PendingPath):
        return evaluated, []
    return _outcome_from_path(intent, evaluated), evaluated.evidence_rows


def maturation_attempt(
    intent: PredictionMaturationIntent,
    *,
    observed_as_of: datetime,
    status: str,
    reasons: tuple[str, ...],
    context: AttemptContext,
    missing_intervals: tuple[str, ...] = (),
    operator_id: str | None = None,
    operator_reference: str | None = None,
    sensitivity: tuple[SensitivityFill, ...] = (),
) -> MaturationAttempt:
    if status not in {"pending", "blocked", "unresolvable"}:
        raise ValueError("maturation attempt status must be pending, blocked or unresolvable")
    observed = _aware_utc(observed_as_of)
    base = {
        "contract": "market_predictor.maturation_attempt",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "observed_as_of_utc": observed,
        "status": status,
        "reasons": reasons,
        "missing_intervals": missing_intervals,
        "grace_days": context.grace_days,
        "settlement_days": context.settlement_days,
        "drift_policy_sha256": context.drift_policy_sha256,
        "receipt_ids": tuple(sorted(set(context.receipt_ids))),
        "operator_id": operator_id,
        "operator_reference": operator_reference,
        "sensitivity": [fill.model_dump(mode="json") for fill in sensitivity],
    }
    return MaturationAttempt.model_validate(
        {**base, "attempt_id": content_sha256(base)}
    )


def _outcome_from_path(
    intent: PredictionMaturationIntent,
    path: MaturedPath,
) -> MaturedOutcome:
    participation = 0.0
    execution_cost_bps = round_trip_cost_bps(
        price=path.entry_price,
        atr_pct=intent.decision_atr_fraction * path.decision_close / path.entry_price,
        participation=participation,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    net_return = path.gross_return - execution_cost_bps / 10_000.0
    base = {
        "contract": "market_predictor.matured_outcome",
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
        "decision_atr_fraction": intent.decision_atr_fraction,
        "decision_close": path.decision_close,
        "execution_participation_fraction": participation,
        "execution_cost_bps": execution_cost_bps,
        "net_return": net_return,
        "mfe": path.mfe,
        "mae": path.mae,
        "path_outcome": path.path_outcome,
        "spy_return": path.spy_return,
        "qqq_return": path.qqq_return,
        "sector_return": path.sector_return,
        "excess_return_vs_spy": net_return - path.spy_return,
        "excess_return_vs_qqq": net_return - path.qqq_return,
        "excess_return_vs_sector": net_return - path.sector_return,
        "holding_sessions": path.holding_sessions,
        "fixed_horizon_net_return": path.fixed_horizon_net_return,
        "fixed_horizon_excess_return_vs_sector": (
            None
            if path.fixed_horizon_net_return is None or path.fixed_horizon_sector_return is None
            else path.fixed_horizon_net_return - path.fixed_horizon_sector_return
        ),
        "evidence_sha256": content_sha256(path.evidence_rows),
    }
    return MaturedOutcome.model_validate(
        {**base, "outcome_id": content_sha256(base)}
    )


def _prepare_bars(
    bars: pd.DataFrame,
    *,
    observed_as_of: datetime,
    required_price_feed: str,
    required_timeframe: str,
) -> pd.DataFrame:
    missing = sorted(_BAR_COLUMNS.difference(bars.columns))
    if missing:
        raise DataReadinessError(
            f"maturation bars are missing columns: {', '.join(missing)}"
        )
    if not bars["source_artifact_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all():
        raise DataReadinessError("maturation bar source identity is invalid")
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
    # A missing or non-finite price keeps its bar; the observation validator marks it unusable.
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
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
    return data.sort_values(["ticker", "bar_start_utc"], kind="stable")


def _aware_utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("observed_as_of must be timezone-aware")
    return value.astimezone(UTC)
