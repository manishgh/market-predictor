from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from market_predictor.hypothesis_registry import TEST_CLOCK_ENV
from market_predictor.locking import file_lock
from market_predictor.outcome_contracts import (
    MaturedOutcomeV1,
    PredictionMaturationIntentV2,
)
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.evaluation import session_block_interval

CAUSAL_SHADOW_SCHEMA = "market_predictor.causal_shadow_evidence.v2"


def write_causal_shadow_bundle(
    root: Path,
    repository: OutcomeRepository,
    *,
    hypothesis: dict[str, Any],
    generated_at: datetime | None = None,
    bootstrap_iterations: int = 1_000,
    bootstrap_seed: int = 42,
) -> Path:
    """Derive immutable paired shadow economics from matured prediction rows."""

    if bootstrap_iterations < 100:
        raise ValueError("shadow bootstrap requires at least 100 iterations")
    if generated_at is not None and os.environ.get(TEST_CLOCK_ENV) != "1":
        raise DataReadinessError("caller-supplied shadow timestamps are test-only")
    source_rows = _derive_source_rows(repository, hypothesis)
    session_returns = _session_returns(source_rows)
    created = _utc(generated_at or datetime.now(UTC))
    declared = _parse_utc(
        str(hypothesis.get("declared_at_utc") or ""),
        "hypothesis declared_at_utc",
    )
    first_session = date.fromisoformat(session_returns[0]["session_date_et"])
    last_session = date.fromisoformat(session_returns[-1]["session_date_et"])
    if first_session <= declared.date():
        raise DataReadinessError(
            "every shadow session must follow hypothesis declaration"
        )
    if created.date() < last_session:
        raise DataReadinessError(
            "shadow evidence cannot be generated before its last session"
        )
    if created > datetime.now(UTC) + timedelta(minutes=5):
        raise DataReadinessError(
            "shadow evidence generation time cannot be in the future"
        )
    interval_frame = pd.DataFrame(session_returns)
    interval_frame["paired_improvement"] = (
        pd.to_numeric(
            interval_frame["candidate_benchmark_excess_return"],
            errors="raise",
        )
        - pd.to_numeric(
            interval_frame["baseline_benchmark_excess_return"],
            errors="raise",
        )
    )
    interval = session_block_interval(
        interval_frame,
        metric=lambda frame: float(
            pd.to_numeric(frame["paired_improvement"], errors="raise").mean()
        ),
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    content: dict[str, Any] = {
        "schema": CAUSAL_SHADOW_SCHEMA,
        "hypothesis_id": hypothesis.get("hypothesis_id"),
        "hypothesis_family": hypothesis.get("hypothesis_family"),
        "hypothesis_record_sha256": hypothesis.get("record_sha256"),
        "candidate_artifact_sha256": hypothesis.get(
            "candidate_artifact_sha256"
        ),
        "baseline_id": hypothesis.get("baseline_id"),
        "baseline_artifact_sha256": hypothesis.get(
            "baseline_artifact_sha256"
        ),
        "prediction_policy_sha256": hypothesis.get(
            "prediction_policy_sha256"
        ),
        "execution_policy_sha256": hypothesis.get(
            "execution_policy_sha256"
        ),
        "shadow_workload": hypothesis.get("shadow_workload"),
        "source_rows_sha256": _json_sha256(source_rows),
        "source_evidence_sha256": _json_sha256(
            {
                "hypothesis_record_sha256": hypothesis.get("record_sha256"),
                "source_rows": source_rows,
                "session_returns": session_returns,
            }
        ),
        "generated_at_utc": created.isoformat(),
        "first_session_date_et": first_session.isoformat(),
        "last_session_date_et": last_session.isoformat(),
        "independent_sessions": len(session_returns),
        "bootstrap": {
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
        },
        "paired_improvement_interval": interval,
        "source_rows": source_rows,
        "session_returns": session_returns,
    }
    fingerprint = _json_sha256(content)
    payload = {**content, "shadow_fingerprint": fingerprint}
    path = root / "shadow" / f"{fingerprint}.json"
    with file_lock(root / ".shadow-bundles"):
        if path.exists():
            existing = load_causal_shadow_bundle(
                path,
                repository=repository,
                hypothesis=hypothesis,
            )
            if existing != payload:
                raise DataReadinessError(
                    "shadow fingerprint collision or immutable bundle mismatch"
                )
            return path
        _write_json_atomic(path, payload)
    return path


def load_causal_shadow_bundle(
    path: Path,
    *,
    repository: OutcomeRepository,
    hypothesis: dict[str, Any],
) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"causal shadow evidence is unavailable or invalid: {path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError("causal shadow evidence must contain an object")
    payload = {str(key): value for key, value in loaded.items()}
    fingerprint = str(payload.pop("shadow_fingerprint", ""))
    if (
        payload.get("schema") != CAUSAL_SHADOW_SCHEMA
        or _json_sha256(payload) != fingerprint
    ):
        raise DataReadinessError(
            "causal shadow evidence integrity check failed"
        )
    if payload.get("hypothesis_record_sha256") != hypothesis.get(
        "record_sha256"
    ):
        raise DataReadinessError(
            "causal shadow evidence does not match the hypothesis"
        )
    source_rows = _derive_source_rows(repository, hypothesis)
    if (
        payload.get("source_rows") != source_rows
        or payload.get("source_rows_sha256") != _json_sha256(source_rows)
    ):
        raise DataReadinessError(
            "causal shadow source rows do not reproduce from outcomes"
        )
    session_returns = _session_returns(source_rows)
    if payload.get("session_returns") != session_returns:
        raise DataReadinessError(
            "causal shadow session returns do not reproduce from source rows"
        )
    _verify_interval(payload, session_returns)
    return {**payload, "shadow_fingerprint": fingerprint}


def _derive_source_rows(
    repository: OutcomeRepository,
    hypothesis: dict[str, Any],
) -> list[dict[str, Any]]:
    workload = hypothesis.get("shadow_workload")
    if not isinstance(workload, dict):
        raise DataReadinessError("hypothesis has no frozen shadow workload")
    expected_groups = tuple(
        str(value) for value in workload.get("decision_group_ids", [])
    )
    view = str(workload.get("view") or "")
    horizon = str(workload.get("horizon") or "")
    minimum_tickers = int(workload.get("minimum_tickers_per_group") or 0)
    candidate_sha = str(hypothesis.get("candidate_artifact_sha256") or "")
    baseline_sha = str(hypothesis.get("baseline_artifact_sha256") or "")
    intents = repository.intents()
    candidate = _workload_intents(
        intents,
        artifact_sha256=candidate_sha,
        view=view,
        horizon=horizon,
        expected_groups=expected_groups,
    )
    baseline = _workload_intents(
        intents,
        artifact_sha256=baseline_sha,
        view=view,
        horizon=horizon,
        expected_groups=expected_groups,
    )
    records: list[dict[str, Any]] = []
    for group_id in expected_groups:
        candidate_group = {
            intent.ticker: intent
            for intent in candidate
            if intent.decision_group_id == group_id
        }
        baseline_group = {
            intent.ticker: intent
            for intent in baseline
            if intent.decision_group_id == group_id
        }
        if (
            len(candidate_group) < minimum_tickers
            or set(candidate_group) != set(baseline_group)
        ):
            raise DataReadinessError(
                f"shadow group {group_id} has incomplete paired cross-sections"
            )
        for ticker in sorted(candidate_group):
            candidate_intent = candidate_group[ticker]
            baseline_intent = baseline_group[ticker]
            _validate_intent_pair(
                candidate_intent,
                baseline_intent,
                hypothesis=hypothesis,
            )
            records.append(
                {
                    "session_date_et": (
                        candidate_intent.decision_session_et.isoformat()
                    ),
                    "decision_group_id": group_id,
                    "decision_time_utc": (
                        candidate_intent.decision_time_utc.isoformat()
                    ),
                    "ticker": ticker,
                    "feature_artifact_sha256": (
                        candidate_intent.feature_artifact_sha256
                    ),
                    "candidate": _side_record(
                        repository,
                        candidate_intent,
                    ),
                    "baseline": _side_record(
                        repository,
                        baseline_intent,
                    ),
                }
            )
    _validate_non_overlapping_groups(records)
    return records


def _workload_intents(
    intents: Sequence[PredictionMaturationIntentV2],
    *,
    artifact_sha256: str,
    view: str,
    horizon: str,
    expected_groups: Sequence[str],
) -> list[PredictionMaturationIntentV2]:
    selected = [
        intent
        for intent in intents
        if intent.model_artifact_sha256 == artifact_sha256
        and intent.view == view
        and intent.horizon == horizon
        and intent.decision_group_id in expected_groups
    ]
    groups = {intent.decision_group_id for intent in selected}
    if groups != set(expected_groups):
        raise DataReadinessError(
            "shadow repository does not contain every frozen decision group"
        )
    identities = [
        (intent.decision_group_id, intent.ticker)
        for intent in selected
    ]
    if len(identities) != len(set(identities)):
        raise DataReadinessError(
            "shadow repository contains duplicate model rows"
        )
    return selected


def _validate_intent_pair(
    candidate: PredictionMaturationIntentV2,
    baseline: PredictionMaturationIntentV2,
    *,
    hypothesis: dict[str, Any],
) -> None:
    if (
        candidate.ticker != baseline.ticker
        or candidate.canonical_security_id
        != baseline.canonical_security_id
        or candidate.decision_time_utc != baseline.decision_time_utc
        or candidate.decision_session_et != baseline.decision_session_et
        or candidate.feature_artifact_sha256
        != baseline.feature_artifact_sha256
        or candidate.prediction_policy_sha256
        != baseline.prediction_policy_sha256
        or candidate.execution_policy_sha256
        != baseline.execution_policy_sha256
    ):
        raise DataReadinessError(
            "candidate and baseline shadow rows are not point-in-time paired"
        )
    if (
        candidate.prediction_policy_sha256
        != hypothesis.get("prediction_policy_sha256")
        or candidate.execution_policy_sha256
        != hypothesis.get("execution_policy_sha256")
    ):
        raise DataReadinessError(
            "shadow row policy identity does not match the hypothesis"
        )


def _side_record(
    repository: OutcomeRepository,
    intent: PredictionMaturationIntentV2,
) -> dict[str, Any]:
    outcome: MaturedOutcomeV1 | None = None
    if intent.selected_for_policy:
        try:
            outcome = repository.load_outcome(intent.maturation_key)
        except FileNotFoundError as exc:
            raise DataReadinessError(
                "selected shadow prediction has no matured outcome"
            ) from exc
    return {
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "snapshot_id": intent.snapshot_id,
        "rank": intent.rank,
        "selection_eligible": intent.selection_eligible,
        "selected_for_policy": intent.selected_for_policy,
        "outcome_id": outcome.outcome_id if outcome else None,
        "outcome_evidence_sha256": (
            outcome.evidence_sha256 if outcome else None
        ),
        "entry_time_utc": (
            outcome.entry_time_utc.isoformat() if outcome else None
        ),
        "exit_time_utc": (
            outcome.exit_time_utc.isoformat() if outcome else None
        ),
        "net_return": outcome.net_return if outcome else None,
        "spy_return": outcome.spy_return if outcome else None,
    }


def _validate_non_overlapping_groups(
    records: Sequence[dict[str, Any]],
) -> None:
    frame = pd.DataFrame(records)
    for side in ("candidate", "baseline"):
        for _, session in frame.groupby("session_date_et", sort=True):
            intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
            for group_id, group in session.groupby(
                "decision_group_id",
                sort=False,
            ):
                selected = [
                    cast(dict[str, Any], value)
                    for value in group[side]
                    if cast(dict[str, Any], value)["selected_for_policy"]
                ]
                if not selected:
                    continue
                starts = [
                    pd.Timestamp(value["entry_time_utc"])
                    for value in selected
                ]
                ends = [
                    pd.Timestamp(value["exit_time_utc"])
                    for value in selected
                ]
                intervals.append((min(starts), max(ends), str(group_id)))
            intervals.sort()
            for previous, current in zip(
                intervals,
                intervals[1:],
                strict=False,
            ):
                if current[0] < previous[1]:
                    raise DataReadinessError(
                        "shadow workload contains overlapping selected groups"
                    )


def _session_returns(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(records)
    output: list[dict[str, Any]] = []
    for session_date, session in frame.groupby(
        "session_date_et",
        sort=True,
    ):
        values: dict[str, float] = {}
        for side in ("candidate", "baseline"):
            portfolio_growth = 1.0
            benchmark_growth = 1.0
            for _, group in session.groupby(
                "decision_group_id",
                sort=False,
            ):
                selected = [
                    cast(dict[str, Any], value)
                    for value in group[side]
                    if cast(dict[str, Any], value)["selected_for_policy"]
                ]
                if not selected:
                    continue
                net = float(
                    pd.Series(
                        [value["net_return"] for value in selected]
                    ).mean()
                )
                spy = float(
                    pd.Series(
                        [value["spy_return"] for value in selected]
                    ).mean()
                )
                if not math.isfinite(net) or not math.isfinite(spy):
                    raise DataReadinessError(
                        "shadow selected returns must be finite"
                    )
                portfolio_growth *= 1.0 + net
                benchmark_growth *= 1.0 + spy
            values[f"{side}_benchmark_excess_return"] = (
                portfolio_growth - benchmark_growth
            )
        output.append(
            {
                "session_date_et": str(session_date),
                **values,
            }
        )
    if len(output) < 2:
        raise DataReadinessError(
            "shadow evidence requires at least two complete sessions"
        )
    return output


def _verify_interval(
    payload: dict[str, Any],
    session_returns: list[dict[str, Any]],
) -> None:
    bootstrap = payload.get("bootstrap")
    declared = payload.get("paired_improvement_interval")
    if not isinstance(bootstrap, dict) or not isinstance(declared, dict):
        raise DataReadinessError("shadow confidence evidence is incomplete")
    iterations = int(bootstrap.get("iterations") or 0)
    seed = int(bootstrap.get("seed") or 0)
    if iterations < 100:
        raise DataReadinessError(
            "shadow bootstrap configuration is invalid"
        )
    frame = pd.DataFrame(session_returns)
    frame["paired_improvement"] = (
        pd.to_numeric(
            frame["candidate_benchmark_excess_return"],
            errors="raise",
        )
        - pd.to_numeric(
            frame["baseline_benchmark_excess_return"],
            errors="raise",
        )
    )
    recomputed = session_block_interval(
        frame,
        metric=lambda rows: float(
            pd.to_numeric(rows["paired_improvement"], errors="raise").mean()
        ),
        iterations=iterations,
        seed=seed,
    )
    if declared != recomputed:
        raise DataReadinessError(
            "shadow confidence interval does not reproduce"
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("shadow timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _parse_utc(value: str, name: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError(f"{name} must be timezone-aware")
    return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())
