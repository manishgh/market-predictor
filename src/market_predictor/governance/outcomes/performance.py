from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal, Self, cast
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.outcomes.contracts import (
    OPERATOR_VERIFIED,
    SWING_HORIZON_PATTERN,
    MaturationAttempt,
    MaturedOutcome,
    PredictionMaturationIntent,
    PredictionMonitoringObservation,
    content_sha256,
    refuse_retired_intraday,
    swing_horizon_sessions,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.session_coverage import SessionCoverage, build_session_coverage
from market_predictor.governance.outcomes.session_records import SessionRecord, SessionRecordStore, expected_sessions
from market_predictor.governance.outcomes.sessions import horizon_last_close
from market_predictor.locking import file_lock

PERFORMANCE_REPORT_VERSION = "market_predictor.selected_policy_performance"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
_IDENTITY_COLUMNS = [
    "model_release_id",
    "model_artifact_sha256",
    "prediction_policy_sha256",
    "label_policy_sha256",
    "execution_policy_sha256",
    "view",
    "horizon",
]


class SelectedPolicyCohort(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def refuse_retired_view(cls, data: object) -> object:
        return refuse_retired_intraday(data)

    cohort_id: str = Field(pattern=SHA256_PATTERN)
    model_release_id: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    prediction_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    label_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_artifact_set_sha256: str = Field(pattern=SHA256_PATTERN)
    source_intent_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    source_observation_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    source_outcome_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    source_session_record_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    source_attempt_ids_sha256: str = Field(pattern=SHA256_PATTERN)
    view: Literal["swing"]
    horizon: str = Field(pattern=SWING_HORIZON_PATTERN)
    cohort_type: Literal[
        "all",
        "market_regime",
        "sector",
        "market_cap_bucket",
        "liquidity_bucket",
        "calibration_bin",
    ]
    cohort_value: str = Field(min_length=1, max_length=128)
    window_start_utc: datetime
    window_end_utc: datetime
    total_predictions: int = Field(ge=1)
    eligible_predictions: int = Field(ge=0)
    selected_predictions: int = Field(ge=0)
    actionable_predictions: int = Field(ge=0)
    matured_selected_samples: int = Field(ge=0)
    pending_selected_samples: int = Field(ge=0)
    # Selected decisions whose latest attempt is `unresolvable`: out of pending, counted apart,
    # with the distinct securities they belong to (one acquired stock can fill several).
    unresolvable_selected_samples: int = Field(ge=0)
    unresolvable_selected_securities: int = Field(ge=0)
    # Of those, the ones an operator resolved, audited apart, and the ones never entered, which
    # have no return and stay out of the sensitivity means.
    operator_verified_selected_samples: int = Field(ge=0)
    unresolvable_never_entered_samples: int = Field(ge=0)
    # Diagnostics, never gates: the mean excess return vs the sector over matured outcomes and
    # every unresolvable one filled at its last usable close, or at its stress value. None when
    # nothing is unresolvable or some unresolvable outcome has no such fill.
    sensitivity_mean_excess_last_close: float | None = None
    sensitivity_mean_excess_stress: float | None = None
    oldest_pending_decision_time_utc: datetime | None = None
    oldest_pending_decision_session_et: date | None = None
    # Over every stored intent of the route, not only this window: nothing leaves monitoring uncounted.
    route_oldest_pending_decision_session_et: date | None = None
    independent_decision_groups: int = Field(ge=0)
    evidence_status: Literal["sufficient", "insufficient_evidence"]
    selection_rate: float = Field(ge=0, le=1)
    actionable_rate: float = Field(ge=0, le=1)
    mean_probability: float | None = Field(default=None, ge=0, le=1)
    probability_p10: float | None = Field(default=None, ge=0, le=1)
    probability_p50: float | None = Field(default=None, ge=0, le=1)
    probability_p90: float | None = Field(default=None, ge=0, le=1)
    mean_decision_score: float | None = None
    decision_score_p10: float | None = None
    decision_score_p50: float | None = None
    decision_score_p90: float | None = None
    mean_selected_rank: float | None = Field(default=None, ge=1)
    selected_rank_p90: float | None = Field(default=None, ge=1)
    average_net_return: float | None = None
    average_excess_return_vs_spy: float | None = None
    average_excess_return_vs_qqq: float | None = None
    average_excess_return_vs_sector: float | None = None
    cumulative_net_return: float | None = None
    win_rate: float | None = Field(default=None, ge=0, le=1)
    max_drawdown: float | None = Field(default=None, ge=0)
    first_decision_time_utc: datetime
    last_decision_time_utc: datetime
    last_matured_outcome_utc: datetime | None = None

    @field_validator(
        "window_start_utc",
        "window_end_utc",
        "first_decision_time_utc",
        "last_decision_time_utc",
        "last_matured_outcome_utc",
        "oldest_pending_decision_time_utc",
    )
    @classmethod
    def aware_times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() is None:
            raise ValueError(
                "selected-policy performance timestamps must be timezone-aware"
            )
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if not (
            self.actionable_predictions
            <= self.selected_predictions
            <= self.eligible_predictions
            <= self.total_predictions
        ):
            raise ValueError("selected-policy cohort counts are inconsistent")
        if (
            self.matured_selected_samples + self.pending_selected_samples + self.unresolvable_selected_samples
            != self.actionable_predictions
        ):
            raise ValueError("selected-policy maturation counts are inconsistent")
        if (self.unresolvable_selected_samples == 0) != (self.unresolvable_selected_securities == 0) or (
            self.unresolvable_selected_securities > self.unresolvable_selected_samples
            or self.operator_verified_selected_samples > self.unresolvable_selected_samples
            or self.unresolvable_never_entered_samples > self.unresolvable_selected_samples
        ):
            raise ValueError("selected-policy unresolvable counts are inconsistent")
        if self.unresolvable_selected_samples == 0 and (
            self.sensitivity_mean_excess_last_close is not None or self.sensitivity_mean_excess_stress is not None
        ):
            raise ValueError("selected-policy sensitivity needs an unresolvable outcome")
        pending_evidence = (self.oldest_pending_decision_time_utc, self.oldest_pending_decision_session_et)
        if any((self.pending_selected_samples > 0) != (value is not None) for value in pending_evidence):
            raise ValueError("selected-policy pending timestamp is inconsistent")
        if self.cohort_type != "all" and self.route_oldest_pending_decision_session_et is not None:
            raise ValueError("only the route's all cohort carries its oldest pending decision")
        if (
            self.route_oldest_pending_decision_session_et is not None
            and self.oldest_pending_decision_session_et is not None
            and self.route_oldest_pending_decision_session_et > self.oldest_pending_decision_session_et
        ):
            raise ValueError("the route's oldest pending decision cannot follow the window's")
        if self.window_start_utc >= self.window_end_utc:
            raise ValueError("selected-policy report window is invalid")
        if not (
            self.window_start_utc
            <= self.first_decision_time_utc
            <= self.last_decision_time_utc
            <= self.window_end_utc
        ):
            raise ValueError("selected-policy decision times are outside the window")
        if not math.isclose(
            self.selection_rate,
            self.selected_predictions / self.total_predictions,
            abs_tol=1e-12,
        ):
            raise ValueError("selected-policy selection rate is inconsistent")
        if not math.isclose(
            self.actionable_rate,
            self.actionable_predictions / self.total_predictions,
            abs_tol=1e-12,
        ):
            raise ValueError("selected-policy actionable rate is inconsistent")
        probability_fields = (
            self.mean_probability,
            self.probability_p10,
            self.probability_p50,
            self.probability_p90,
            self.mean_decision_score,
            self.decision_score_p10,
            self.decision_score_p50,
            self.decision_score_p90,
            self.mean_selected_rank,
            self.selected_rank_p90,
        )
        if self.actionable_predictions == 0 and any(
            value is not None for value in probability_fields
        ):
            raise ValueError("empty selected-policy cohort has score evidence")
        economic_outcome_fields = (
            self.average_net_return,
            self.average_excess_return_vs_spy,
            self.average_excess_return_vs_qqq,
            self.average_excess_return_vs_sector,
            self.cumulative_net_return,
            self.win_rate,
            self.max_drawdown,
            self.last_matured_outcome_utc,
        )
        if self.matured_selected_samples == 0:
            if any(value is not None for value in economic_outcome_fields):
                raise ValueError("unmatured selected-policy cohort has outcome evidence")
        elif any(value is None for value in economic_outcome_fields):
            raise ValueError("matured selected-policy cohort lacks outcome evidence")
        content = self.model_dump(mode="json", exclude={"cohort_id"})
        if content_sha256(content) != self.cohort_id:
            raise ValueError("selected-policy cohort identity is invalid")
        return self


class SelectedPolicyPerformanceReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract: Literal[
        "market_predictor.selected_policy_performance"
    ] = "market_predictor.selected_policy_performance"
    report_id: str = Field(pattern=SHA256_PATTERN)
    generated_at_utc: datetime
    lookback_days: int = Field(ge=1)
    minimum_matured_samples: int = Field(ge=1)
    window_start_utc: datetime
    window_end_utc: datetime
    source_intent_ids: tuple[str, ...]
    source_observation_ids: tuple[str, ...]
    source_outcome_ids: tuple[str, ...]
    source_session_record_ids: tuple[str, ...]
    source_attempt_ids: tuple[str, ...]
    session_coverage: tuple[SessionCoverage, ...]
    rows: tuple[SelectedPolicyCohort, ...]

    @field_validator(
        "generated_at_utc",
        "window_start_utc",
        "window_end_utc",
    )
    @classmethod
    def aware_report_times(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError(
                "selected-policy report timestamp must be timezone-aware"
            )
        return value.astimezone(UTC)

    @field_validator(
        "source_intent_ids",
        "source_observation_ids",
        "source_outcome_ids",
        "source_session_record_ids",
        "source_attempt_ids",
    )
    @classmethod
    def canonical_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(
            len(item) != 64
            or any(character not in "0123456789abcdef" for character in item)
            for item in value
        ):
            raise ValueError("selected-policy source identity is invalid")
        if tuple(sorted(set(value))) != value:
            raise ValueError(
                "selected-policy source identities must be unique and sorted"
            )
        return value

    @model_validator(mode="after")
    def validate_report_identity(self) -> Self:
        if (
            self.window_end_utc != self.generated_at_utc
            or self.window_start_utc
            != self.window_end_utc - timedelta(days=self.lookback_days)
        ):
            raise ValueError("selected-policy report window is inconsistent")
        route_keys = [coverage.route.key() for coverage in self.session_coverage]
        if route_keys != sorted(set(route_keys)):
            raise ValueError("report coverage routes must be sorted and unique")
        for coverage in self.session_coverage:
            expected = expected_sessions(coverage.route, start=self.window_start_utc, end=self.window_end_utc)
            if coverage.expected_sessions != expected:
                raise ValueError("report coverage does not match route activation and window")
            if not set(coverage.source_record_ids).issubset(self.source_session_record_ids):
                raise ValueError("report omits session coverage source ids")
        route_starts: dict[tuple[str, ...], datetime] = {}
        for row in self.rows:
            route = tuple(str(getattr(row, column)) for column in _IDENTITY_COLUMNS)
            route_starts[route] = min(route_starts.get(route, self.window_start_utc), row.first_decision_time_utc)
        for row in self.rows:
            route = tuple(str(getattr(row, column)) for column in _IDENTITY_COLUMNS)
            if row.window_end_utc != self.window_end_utc or row.window_start_utc != route_starts[route]:
                raise ValueError("selected-policy cohort window does not follow the route's included decisions")
        content = self.model_dump(mode="json", exclude={"report_id"})
        if content_sha256(content) != self.report_id:
            raise ValueError("selected-policy report identity is invalid")
        return self


def build_performance_cohorts(
    repository: OutcomeRepository,
    *,
    generated_at: datetime | None = None,
    minimum_samples: int = 30,
    lookback_days: int = 180,
) -> dict[str, object]:
    if minimum_samples < 1:
        raise ValueError("minimum_samples must be positive")
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    generated = _utc(generated_at or datetime.now(UTC))
    window_start = generated - timedelta(days=lookback_days)
    records: list[dict[str, object]] = []
    source_intent_ids: set[str] = set()
    source_observation_ids: set[str] = set()
    source_outcome_ids: set[str] = set()
    source_attempt_ids: set[str] = set()
    session_store = SessionRecordStore(repository.root)
    session_records = session_store.records(as_of=generated)
    committed = [record for record in session_records if record.status == "registered"]
    allowed_intents = {key: record for record in committed for key in record.intent_ids}
    source_session_record_ids = {record.record_id for record in session_records}
    record_by_observation = {key: record for record in committed for key in record.observation_ids}
    if len(record_by_observation) != sum(len(record.observation_ids) for record in committed):
        raise DataReadinessError("an observation is committed by more than one session")
    coverage = build_session_coverage(session_store.routes(), session_records, start=window_start, end=generated)
    # Maturity-aligned: a decision belongs to the window when its horizon's last session closes
    # inside it, or has not closed yet, so every horizon covers the outcomes of the same period.
    # Only the session partitions that can hold such decisions are read.
    intents: dict[str, PredictionMaturationIntent] = {}
    canonical_keys: dict[str, str] = {}
    observations: list[PredictionMonitoringObservation] = []
    for record in committed:
        session = record.decision_session
        if not _matures_in_window(session, record.route.horizon, window_start, generated):
            continue
        try:
            stored = {intent.maturation_key: intent for intent in repository.session_intents(session, intent_ids=record.intent_ids)}
            observed = repository.session_observations(session, stored, observation_ids=record.observation_ids)
        except (OSError, ValueError) as exc:
            raise DataReadinessError("committed monitoring session source evidence is unavailable") from exc
        if len({row.ticker for row in observed}) != record.members or len(stored) != record.scored:
            raise DataReadinessError("committed monitoring session population differs")
        for observation in observed:
            if (observation.snapshot_id != record.snapshot_id or observation.decision_session_et != session
                    or any(str(getattr(observation, key)) != value for key, value in record.route.identity().items())):
                raise DataReadinessError("committed monitoring observation has a different session or route identity")
        if sum(row.probability is not None for row in observed) != record.scored:
            raise DataReadinessError("committed monitoring scores differ from session counts")
        canonical_keys.update((intent.semantic_prediction_id, intent.maturation_key) for intent in stored.values())
        intents.update(stored)
        observations.extend(observed)
    route_session_ids: dict[tuple[str, ...], set[str]] = defaultdict(set)
    route_attempt_ids: dict[tuple[str, ...], set[str]] = defaultdict(set)
    route_intent_ids: dict[tuple[str, ...], set[str]] = defaultdict(set)
    route_outcome_ids: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for session_record in session_records:
        route_key = tuple(session_record.route.identity()[column] for column in _IDENTITY_COLUMNS)
        route_session_ids[route_key].add(session_record.record_id)
    route_oldest_pending = _route_oldest_pending(repository, generated=generated, allowed_intents=allowed_intents,
                                               source_attempt_ids=source_attempt_ids, route_attempt_ids=route_attempt_ids,
                                               route_intent_ids=route_intent_ids, route_outcome_ids=route_outcome_ids,
                                               cached_intents=intents)
    source_intent_ids.update(key for keys in route_intent_ids.values() for key in keys)
    source_outcome_ids.update(key for keys in route_outcome_ids.values() for key in keys)
    for observation in _canonical_observations(canonical_keys, observations):
        intent = (
            intents.get(observation.maturation_key)
            if observation.maturation_key is not None
            else None
        )
        outcome = (
            _matured_selected_outcome(
                repository,
                intent,
                generated_at=generated,
            )
            if intent is not None
            else None
        )
        resolution = (
            _unresolvable(repository, intent, generated_at=generated)
            if intent is not None and intent.actionable and outcome is None
            else None
        )
        deciding = _deciding_attempt(repository, intent, generated_at=generated) if intent is not None else None
        if deciding is not None:
            source_attempt_ids.add(deciding.attempt_id)
        fills = {fill.basis: fill for fill in resolution.sensitivity} if resolution is not None else {}
        stress = [fill for basis, fill in fills.items() if basis != "last_usable_close"]
        source_observation_ids.add(observation.observation_id)
        if intent is not None:
            source_intent_ids.add(intent.maturation_key)
        if outcome is not None:
            source_outcome_ids.add(outcome.outcome_id)
        records.append(
            {
                **_monitoring_record(observation, outcome),
                "session_record_id": record_by_observation[observation.observation_id].record_id,
                "deciding_attempt_id": deciding.attempt_id if deciding is not None else None,
                "unresolvable": resolution is not None,
                "operator_verified": resolution is not None and resolution.reasons == (OPERATOR_VERIFIED,),
                "never_entered": resolution is not None and resolution.never_entered,
                "fill_last_close_excess": (
                    fills["last_usable_close"].excess_return_vs_sector if "last_usable_close" in fills else None
                ),
                "fill_stress_excess": stress[0].excess_return_vs_sector if stress else None,
                "canonical_security_id": intent.canonical_security_id if intent is not None else None,
            }
        )
    frame = pd.DataFrame(records)
    rows: list[dict[str, object]] = []
    if not frame.empty:
        for column in (
            "decision_time_utc",
            "matured_at_utc",
            "exit_time_utc",
        ):
            frame[column] = pd.to_datetime(frame[column], errors="coerce", utc=True)
        route_starts = {
            tuple(str(value) for value in key): start.to_pydatetime()
            for key, start in frame.groupby(_IDENTITY_COLUMNS, dropna=False)["decision_time_utc"].min().items()
        }
        cohort_specs = [
            ("all", None),
            ("market_regime", "market_regime"),
            ("sector", "sector"),
            ("market_cap_bucket", "market_cap_bucket"),
            ("liquidity_bucket", "liquidity_bucket"),
            ("calibration_bin", "calibration_bin"),
        ]
        for cohort_type, cohort_column in cohort_specs:
            group_columns = [
                *_IDENTITY_COLUMNS,
                *([cohort_column] if cohort_column else []),
            ]
            for group_values, group in frame.groupby(
                group_columns,
                dropna=False,
                sort=True,
            ):
                values = (
                    group_values
                    if isinstance(group_values, tuple)
                    else (group_values,)
                )
                identity = dict(zip(group_columns, values, strict=True))
                route = tuple(str(identity[column]) for column in _IDENTITY_COLUMNS)
                row = _cohort_row(
                    group,
                    identity=identity,
                    cohort_type=cohort_type,
                    cohort_value=(
                        "all"
                        if cohort_column is None
                        else "unavailable" if pd.isna(identity[cohort_column]) else str(identity[cohort_column])
                    ),
                    minimum_samples=minimum_samples,
                    # A route's rows span its included decisions, which may precede the outcome window.
                    window_start=min(window_start, route_starts[route]),
                    window_end=generated,
                    route_oldest_pending=route_oldest_pending.get(route) if cohort_type == "all" else None,
                    route_session_ids=route_session_ids[route] if cohort_type == "all" else set(),
                    route_attempt_ids=route_attempt_ids[route] if cohort_type == "all" else set(),
                    route_intent_ids=route_intent_ids[route] if cohort_type == "all" else set(),
                    route_outcome_ids=route_outcome_ids[route] if cohort_type == "all" else set(),
                )
                rows.append(
                    SelectedPolicyCohort.model_validate(row).model_dump(
                        mode="json"
                    )
                )
    rows.sort(
        key=lambda row: (
            str(row["model_release_id"]),
            str(row["view"]),
            str(row["horizon"]),
            str(row["cohort_type"]),
            str(row["cohort_value"]),
        )
    )
    report_identity: dict[str, object] = {
        "contract": PERFORMANCE_REPORT_VERSION,
        "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
        "lookback_days": lookback_days,
        "minimum_matured_samples": minimum_samples,
        "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
        "window_end_utc": generated.isoformat().replace("+00:00", "Z"),
        "source_intent_ids": sorted(source_intent_ids),
        "source_observation_ids": sorted(source_observation_ids),
        "source_outcome_ids": sorted(source_outcome_ids),
        "source_session_record_ids": sorted(source_session_record_ids),
        "source_attempt_ids": sorted(source_attempt_ids),
        "session_coverage": [item.model_dump(mode="json") for item in sorted(coverage, key=lambda c: c.route.key())],
        "rows": rows,
    }
    report = SelectedPolicyPerformanceReport.model_validate(
        {
            **report_identity,
            "report_id": content_sha256(report_identity),
        }
    )
    return report.model_dump(mode="json")


def validate_performance_report(value: object) -> dict[str, object]:
    report = SelectedPolicyPerformanceReport.model_validate(value)
    return report.model_dump(mode="json")


def load_performance_report(path: Path) -> dict[str, object]:
    try:
        loaded = parse_strict_json_object(
            path.read_bytes(),
            label="selected-policy performance report",
        )
    except (OSError, ValueError) as exc:
        raise DataReadinessError(
            "selected-policy performance report is unreadable"
        ) from exc
    return validate_performance_report(loaded)


def write_performance_report(
    path: Path,
    report: dict[str, object],
) -> dict[str, object]:
    validated = validate_performance_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(path):
        if path.exists():
            existing = load_performance_report(path)
            existing_time = _utc(datetime.fromisoformat(str(existing["generated_at_utc"])))
            incoming_time = _utc(datetime.fromisoformat(str(validated["generated_at_utc"])))
            if existing_time > incoming_time:
                raise PredictionConflictError
            if existing_time == incoming_time:
                if existing != validated:
                    raise PredictionConflictError
                return existing
        _write_json_durable(path, validated)
    return validated


def _canonical_observations(
    canonical_keys: Mapping[str, str],
    observations: list[PredictionMonitoringObservation],
) -> list[PredictionMonitoringObservation]:
    grouped: dict[str, list[PredictionMonitoringObservation]] = {}
    for observation in observations:
        grouped.setdefault(observation.semantic_prediction_id, []).append(observation)
    canonical: list[PredictionMonitoringObservation] = []
    for semantic_id, group in grouped.items():
        canonical_key = canonical_keys.get(semantic_id)
        candidates = (
            [item for item in group if item.maturation_key == canonical_key]
            if canonical_key is not None
            else group
        )
        if not candidates:
            raise DataReadinessError(
                "semantic prediction has no canonical monitoring observation"
            )
        canonical.append(
            min(
                candidates,
                key=lambda item: (
                    item.decision_time_utc,
                    item.snapshot_id,
                    item.observation_id,
                ),
            )
        )
    return sorted(
        canonical,
        key=lambda item: (
            item.decision_time_utc,
            item.semantic_prediction_id,
        ),
    )


def _write_json_durable(path: Path, value: object) -> None:
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _matured_selected_outcome(
    repository: OutcomeRepository,
    intent: PredictionMaturationIntent,
    *,
    generated_at: datetime,
) -> MaturedOutcome | None:
    if not intent.actionable:
        return None
    if not repository.has_outcome(intent.maturation_key, intent.decision_session_et):
        return None
    # The repository checks the outcome against this intent as it loads it.
    outcome = repository.load_outcome(intent.maturation_key, intent.decision_session_et)
    return None if outcome.matured_at_utc > generated_at else outcome


def _deciding_attempt(
    repository: OutcomeRepository, intent: PredictionMaturationIntent, *, generated_at: datetime,
) -> MaturationAttempt | None:
    observed = [attempt for attempt in repository.attempts(intent.maturation_key, intent.decision_session_et)
                if attempt.observed_as_of_utc <= generated_at]
    return observed[-1] if observed else None


def _unresolvable(
    repository: OutcomeRepository, intent: PredictionMaturationIntent, *, generated_at: datetime
) -> MaturationAttempt | None:
    """The intent's latest attempt observed by the report time, when it is `unresolvable`."""
    attempt = _deciding_attempt(repository, intent, generated_at=generated_at)
    return attempt if attempt is not None and attempt.status == "unresolvable" else None


def _sensitivity_mean(matured: pd.DataFrame, unresolvable: pd.DataFrame, fill_column: str) -> float | None:
    """The mean excess return vs the sector with every entered unresolvable outcome filled from one column."""
    entered = unresolvable.loc[~unresolvable["never_entered"].astype(bool)]
    if entered.empty or entered[fill_column].isna().any():
        return None
    values = pd.concat([matured["excess_return_vs_sector"], entered[fill_column]]).astype(float)
    return float(values.mean())


def _earliest(*sessions: date | None) -> date:
    return min(session for session in sessions if session is not None)


def _matures_in_window(decision_session: date, horizon: str, window_start: datetime, generated: datetime) -> bool:
    last_close = horizon_last_close(decision_session, swing_horizon_sessions(horizon), through=generated)
    return last_close is None or last_close >= window_start


def _route_oldest_pending(
    repository: OutcomeRepository,
    *,
    generated: datetime,
    allowed_intents: dict[str, SessionRecord],
    source_attempt_ids: set[str],
    route_attempt_ids: dict[tuple[str, ...], set[str]],
    route_intent_ids: dict[tuple[str, ...], set[str]],
    route_outcome_ids: dict[tuple[str, ...], set[str]],
    cached_intents: dict[str, PredictionMaturationIntent],
) -> dict[tuple[str, ...], date]:
    """The oldest decision session of each route's selected intents that have no outcome yet."""
    oldest: dict[tuple[str, ...], date] = {}
    # The shared pending index is canonical per semantic id, including partial writes.
    # The commit inventory is authoritative here; an earlier partial snapshot must not
    # hide a committed intent. Load only intent/attempt/outcome evidence, never old
    # observation partitions. Current outcomes are checked at the report's as-of time.
    for maturation_key, record in allowed_intents.items():
        intent = cached_intents.get(maturation_key) or repository.load_intent(maturation_key, record.decision_session)
        if (intent.snapshot_id != record.snapshot_id or
                any(str(getattr(intent, key)) != value for key, value in record.route.identity().items())):
            raise DataReadinessError("committed monitoring intent has a different session or route identity")
        route = tuple(str(getattr(intent, column)) for column in _IDENTITY_COLUMNS)
        route_intent_ids[route].add(maturation_key)
        if not intent.actionable or intent.decision_time_utc > generated:
            continue
        deciding = _deciding_attempt(repository, intent, generated_at=generated)
        if deciding is not None:
            source_attempt_ids.add(deciding.attempt_id)
            route_attempt_ids[route].add(deciding.attempt_id)
        outcome = _matured_selected_outcome(repository, intent, generated_at=generated)
        if outcome is not None:
            route_outcome_ids[route].add(outcome.outcome_id)
            continue
        if deciding is not None and deciding.status == "unresolvable":
            continue
        if route not in oldest or intent.decision_session_et < oldest[route]:
            oldest[route] = intent.decision_session_et
    return oldest


def _monitoring_record(
    observation: PredictionMonitoringObservation,
    outcome: MaturedOutcome | None,
) -> dict[str, object]:
    return {
        "observation_id": observation.observation_id,
        "maturation_key": observation.maturation_key,
        "outcome_id": outcome.outcome_id if outcome is not None else None,
        "model_release_id": observation.model_release_id,
        "model_artifact_sha256": observation.model_artifact_sha256,
        "feature_artifact_sha256": observation.feature_artifact_sha256,
        "prediction_policy_sha256": observation.prediction_policy_sha256,
        "label_policy_sha256": observation.label_policy_sha256,
        "execution_policy_sha256": observation.execution_policy_sha256,
        "view": observation.view,
        "horizon": observation.horizon,
        "market_regime": observation.market_regime,
        "sector": observation.sector,
        "market_cap_bucket": observation.market_cap_bucket,
        "liquidity_bucket": observation.liquidity_bucket,
        "calibration_bin": observation.calibration_bin,
        "decision_group_id": observation.decision_group_id,
        "decision_time_utc": observation.decision_time_utc,
        "decision_session_et": observation.decision_session_et,
        "probability": observation.probability,
        "decision_score": observation.probability,
        "rank": observation.rank,
        "selection_eligible": observation.selection_eligible,
        "selected_for_policy": observation.selected_for_policy,
        "actionable": observation.actionable,
        "net_return": outcome.net_return if outcome is not None else None,
        "excess_return_vs_spy": (
            outcome.excess_return_vs_spy if outcome is not None else None
        ),
        "excess_return_vs_qqq": (
            outcome.excess_return_vs_qqq if outcome is not None else None
        ),
        "excess_return_vs_sector": (
            outcome.excess_return_vs_sector if outcome is not None else None
        ),
        "exit_time_utc": outcome.exit_time_utc if outcome is not None else None,
        "matured_at_utc": (
            outcome.matured_at_utc if outcome is not None else None
        ),
    }


def _cohort_row(
    group: pd.DataFrame,
    *,
    identity: dict[str, object],
    cohort_type: str,
    cohort_value: str,
    minimum_samples: int,
    window_start: datetime,
    window_end: datetime,
    route_oldest_pending: date | None,
    route_session_ids: set[str],
    route_attempt_ids: set[str],
    route_intent_ids: set[str],
    route_outcome_ids: set[str],
) -> dict[str, object]:
    ordered = group.sort_values(
        ["decision_time_utc", "decision_group_id", "observation_id"],
        kind="stable",
    )
    selected = ordered[ordered["actionable"].astype(bool)]
    matured = selected[selected["outcome_id"].notna()]
    unresolved = selected[selected["outcome_id"].isna()]
    unresolvable = unresolved[unresolved["unresolvable"].astype(bool)]
    pending = unresolved[~unresolved["unresolvable"].astype(bool)]
    total = len(ordered)
    eligible_count = int(ordered["selection_eligible"].astype(bool).sum())
    selected_count = int(ordered["selected_for_policy"].astype(bool).sum())
    actionable_count = len(selected)
    matured_count = len(matured)
    feature_ids = sorted(
        set(ordered["feature_artifact_sha256"].astype(str))
    )
    intent_ids = sorted(
        ordered.loc[ordered["maturation_key"].notna(), "maturation_key"].astype(str)
    )
    observation_ids = sorted(ordered["observation_id"].astype(str))
    outcome_ids = sorted(matured["outcome_id"].astype(str))
    score_metrics = _score_metrics(selected)
    outcome_metrics = _outcome_metrics(matured)
    content: dict[str, object] = {
        **{
            column: str(identity[column])
            for column in _IDENTITY_COLUMNS
        },
        "feature_artifact_set_sha256": content_sha256(feature_ids),
        "source_intent_ids_sha256": content_sha256(sorted(route_intent_ids | set(intent_ids))),
        "source_observation_ids_sha256": content_sha256(observation_ids),
        "source_outcome_ids_sha256": content_sha256(sorted(route_outcome_ids | set(outcome_ids))),
        "source_session_record_ids_sha256": content_sha256(sorted(route_session_ids | set(ordered["session_record_id"].astype(str)))),
        "source_attempt_ids_sha256": content_sha256(sorted(route_attempt_ids | set(ordered["deciding_attempt_id"].dropna().astype(str)))),
        "cohort_type": cohort_type,
        "cohort_value": cohort_value,
        "window_start_utc": window_start.isoformat().replace("+00:00", "Z"),
        "window_end_utc": window_end.isoformat().replace("+00:00", "Z"),
        "total_predictions": total,
        "eligible_predictions": eligible_count,
        "selected_predictions": selected_count,
        "actionable_predictions": actionable_count,
        "matured_selected_samples": matured_count,
        "pending_selected_samples": len(pending),
        "unresolvable_selected_samples": len(unresolvable),
        "unresolvable_selected_securities": int(unresolvable["canonical_security_id"].nunique()),
        "operator_verified_selected_samples": int(unresolvable["operator_verified"].astype(bool).sum()),
        "unresolvable_never_entered_samples": int(unresolvable["never_entered"].astype(bool).sum()),
        "sensitivity_mean_excess_last_close": _sensitivity_mean(matured, unresolvable, "fill_last_close_excess"),
        "sensitivity_mean_excess_stress": _sensitivity_mean(matured, unresolvable, "fill_stress_excess"),
        "oldest_pending_decision_time_utc": (
            _timestamp_text(pending["decision_time_utc"].min())
            if not pending.empty
            else None
        ),
        "oldest_pending_decision_session_et": (
            pending["decision_session_et"].min().isoformat()
            if not pending.empty
            else None
        ),
        "route_oldest_pending_decision_session_et": (
            _earliest(
                route_oldest_pending,
                None if pending.empty else pending["decision_session_et"].min(),
            ).isoformat()
            if cohort_type == "all" and (route_oldest_pending is not None or not pending.empty)
            else None
        ),
        "independent_decision_groups": int(
            matured["decision_group_id"].nunique()
        ),
        "evidence_status": (
            "sufficient"
            if matured_count >= minimum_samples
            else "insufficient_evidence"
        ),
        "selection_rate": selected_count / total,
        "actionable_rate": actionable_count / total,
        **score_metrics,
        **outcome_metrics,
        "first_decision_time_utc": _timestamp_text(
            ordered["decision_time_utc"].min()
        ),
        "last_decision_time_utc": _timestamp_text(
            ordered["decision_time_utc"].max()
        ),
    }
    return {**content, "cohort_id": content_sha256(content)}


def _score_metrics(selected: pd.DataFrame) -> dict[str, float | None]:
    if selected.empty:
        return {
            "mean_probability": None,
            "probability_p10": None,
            "probability_p50": None,
            "probability_p90": None,
            "mean_decision_score": None,
            "decision_score_p10": None,
            "decision_score_p50": None,
            "decision_score_p90": None,
            "mean_selected_rank": None,
            "selected_rank_p90": None,
        }
    probability = selected["probability"].to_numpy(float)
    scores = selected["decision_score"].to_numpy(float)
    ranks = pd.to_numeric(selected["rank"], errors="coerce").dropna().to_numpy(
        float
    )
    return {
        "mean_probability": float(np.mean(probability)),
        "probability_p10": _quantile(probability, 0.10),
        "probability_p50": _quantile(probability, 0.50),
        "probability_p90": _quantile(probability, 0.90),
        "mean_decision_score": float(np.mean(scores)),
        "decision_score_p10": _quantile(scores, 0.10),
        "decision_score_p50": _quantile(scores, 0.50),
        "decision_score_p90": _quantile(scores, 0.90),
        "mean_selected_rank": (
            float(np.mean(ranks)) if len(ranks) else None
        ),
        "selected_rank_p90": (
            _quantile(ranks, 0.90) if len(ranks) else None
        ),
    }


def _outcome_metrics(matured: pd.DataFrame) -> dict[str, object]:
    empty: dict[str, object] = {
        "average_net_return": None,
        "average_excess_return_vs_spy": None,
        "average_excess_return_vs_qqq": None,
        "average_excess_return_vs_sector": None,
        "cumulative_net_return": None,
        "win_rate": None,
        "max_drawdown": None,
        "last_matured_outcome_utc": None,
    }
    if matured.empty:
        return empty
    returns = matured["net_return"].to_numpy(float)
    period_returns = (
        matured.groupby(
            ["decision_time_utc", "decision_group_id"],
            sort=True,
        )["net_return"]
        .mean()
        .to_numpy(float)
    )
    equity = np.cumprod(1.0 + period_returns)
    equity_with_origin = np.concatenate(([1.0], equity))
    peak = np.maximum.accumulate(equity_with_origin)
    drawdown = 1.0 - np.divide(
        equity_with_origin,
        peak,
        out=np.ones_like(equity_with_origin),
        where=peak != 0,
    )
    result: dict[str, object] = {
        **empty,
        "average_net_return": float(np.mean(returns)),
        "average_excess_return_vs_spy": float(
            matured["excess_return_vs_spy"].mean()
        ),
        "average_excess_return_vs_qqq": float(
            matured["excess_return_vs_qqq"].mean()
        ),
        "average_excess_return_vs_sector": float(
            matured["excess_return_vs_sector"].mean()
        ),
        "cumulative_net_return": float(
            np.prod(1.0 + period_returns) - 1.0
        ),
        "win_rate": float(np.mean(returns > 0)),
        "max_drawdown": float(np.max(drawdown, initial=0.0)),
        "last_matured_outcome_utc": _timestamp_text(
            matured["matured_at_utc"].max()
        ),
    }
    return result


def _quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, quantile))


def _timestamp_text(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("performance evidence timestamp must be timezone-aware")
    text = cast(str, timestamp.tz_convert("UTC").isoformat())
    return text.replace("+00:00", "Z")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("performance report timestamp must be timezone-aware")
    return value.astimezone(UTC)
