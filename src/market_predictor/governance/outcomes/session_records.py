"""Route activation and atomic commit markers for complete monitoring decisions."""
from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Self

import exchange_calendars as xcals
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core import path_integrity
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.governance.outcomes.repository import _write_json_durable
from market_predictor.locking import file_lock

SHA = r"^[0-9a-f]{64}$"


class MonitoringRoute(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    view: Literal["swing"] = "swing"
    horizon: Literal["10b"] = "10b"
    model_release_id: str = Field(pattern=SHA)
    model_artifact_sha256: str = Field(pattern=SHA)
    prediction_policy_sha256: str = Field(pattern=SHA)
    label_policy_sha256: str = Field(pattern=SHA)
    execution_policy_sha256: str = Field(pattern=SHA)
    promoted_at_utc: datetime

    @field_validator("promoted_at_utc")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("route promotion time must be timezone-aware")
        return value.astimezone(UTC)

    def key(self) -> str:
        return content_sha256(self.model_dump(mode="json"))

    def identity(self) -> dict[str, str]:
        return {key: str(value) for key, value in self.model_dump(exclude={"promoted_at_utc"}).items()}


class SessionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    record_id: str = Field(pattern=SHA)
    route: MonitoringRoute
    decision_session: date
    status: Literal["registered", "failed"]
    recorded_at_utc: datetime
    failure_reason: Literal["exclusion_ceiling_exceeded", "inputs_unavailable", "model_unavailable", "registration_error", "not_run"] | None
    operator_id: str | None = None
    operator_reason: str | None = None
    snapshot_id: str | None = Field(default=None, pattern=SHA)
    snapshot_sha256: str | None = Field(default=None, pattern=SHA)
    member_set_sha256: str | None = Field(default=None, pattern=SHA)
    members: int = Field(default=0, ge=0)
    scored: int = Field(default=0, ge=0)
    abstentions: dict[str, int] = Field(default_factory=dict)
    observation_ids: tuple[str, ...] = ()
    intent_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def verify(self) -> Self:
        if self.recorded_at_utc.utcoffset() is None:
            raise ValueError("session recording time must be timezone-aware")
        if decision_cutoff(self.decision_session) < self.route.promoted_at_utc:
            raise ValueError("session precedes route activation")
        if self.recorded_at_utc < decision_cutoff(self.decision_session):
            raise ValueError("session record precedes decision cutoff")
        for values in (self.observation_ids, self.intent_ids):
            if values != tuple(sorted(set(values))) or any(len(v) != 64 or any(c not in "0123456789abcdef" for c in v) for v in values):
                raise ValueError("session source ids must be sorted unique hashes")
        if self.status == "registered":
            if self.failure_reason is not None or self.operator_id is not None or self.operator_reason is not None:
                raise ValueError("registered session cannot contain failure metadata")
            if not self.snapshot_id or not self.snapshot_sha256 or not self.member_set_sha256 or self.members < 1:
                raise ValueError("registered session needs complete snapshot and membership identities")
            if set(self.abstentions) - {"live_inputs_incomplete", "sector_peer_floor"} or any(v < 1 for v in self.abstentions.values()):
                raise ValueError("invalid session abstention counts")
            if self.members != self.scored + sum(self.abstentions.values()):
                raise ValueError("session membership counts do not balance")
            if len(self.observation_ids) != self.members or len(self.intent_ids) != self.scored:
                raise ValueError("session source inventory differs from member counts")
        else:
            if self.failure_reason is None or self.snapshot_id or self.snapshot_sha256 or self.member_set_sha256:
                raise ValueError("failed session cannot claim a snapshot")
            if self.members or self.scored or self.abstentions or self.observation_ids or self.intent_ids:
                raise ValueError("failed session cannot claim committed population")
            if self.failure_reason == "not_run" and (not self.operator_id or not self.operator_id.strip() or not self.operator_reason or
                                                     not self.operator_reason.strip()):
                raise ValueError("not_run requires operator identity and reason")
        if content_sha256(self.model_dump(mode="json", exclude={"record_id", "recorded_at_utc"})) != self.record_id:
            raise ValueError("session record identity does not verify")
        return self


def make_session_record(**fields: Any) -> SessionRecord:
    # Validate shape and defaults before computing identity; model_construct is never persisted.
    provisional = SessionRecord.model_construct(**fields)
    content = provisional.model_dump(mode="json", exclude={"record_id", "recorded_at_utc"})
    return SessionRecord.model_validate({**content, "recorded_at_utc": fields["recorded_at_utc"], "record_id": content_sha256(content)})


class SessionRecordStore:
    def __init__(self, root: Path):
        self.root = root

    def record_route(self, route: MonitoringRoute) -> None:
        path = self._path("monitoring_routes", f"{route.key()}.json")
        with file_lock(path):
            if path.exists():
                if self.load_route(route.key()) != route:
                    raise PredictionConflictError
            else:
                _write_json_durable(path, route.model_dump(mode="json"))

    def load_route(self, key: str) -> MonitoringRoute:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise PredictionConflictError
        try:
            route = MonitoringRoute.model_validate(parse_strict_json_object(
                self._path("monitoring_routes", f"{key}.json").read_bytes(), label="monitoring route",
            ))
        except (OSError, ValueError) as exc:
            raise PredictionConflictError from exc
        if route.key() != key:
            raise PredictionConflictError
        return route

    def routes(self) -> list[MonitoringRoute]:
        return [self.load_route(path.stem) for path in sorted(self._path("monitoring_routes").glob("*.json"))]

    def record(self, record: SessionRecord) -> SessionRecord:
        self.record_route(record.route)
        path = self._path("session_records", record.decision_session.isoformat(), f"{record.route.key()}.json")
        with file_lock(path):
            if path.exists():
                previous = self.load(record.route, record.decision_session)
                assert previous is not None
                if previous.record_id == record.record_id:
                    return previous
                if previous.status == "registered" or previous.recorded_at_utc >= record.recorded_at_utc:
                    raise PredictionConflictError
            # Keep failed attempts for as-of reporting; the current marker is written last.
            history = self._path(
                "session_record_history", record.decision_session.isoformat(), record.route.key(), f"{record.record_id}.json",
            )
            if history.exists():
                previous_version = self._read(history, record.route, record.decision_session)
                if previous_version.record_id != record.record_id:
                    raise PredictionConflictError
                record = previous_version
            else:
                _write_json_durable(history, record.model_dump(mode="json"))
            _write_json_durable(path, record.model_dump(mode="json"))
        return record

    def load(self, route: MonitoringRoute, session: date, *, as_of: datetime | None = None) -> SessionRecord | None:
        path = self._path("session_records", session.isoformat(), f"{route.key()}.json")
        if not path.exists():
            return None
        record = self._read(path, route, session)
        if as_of is None or record.recorded_at_utc <= as_of:
            return record
        # Do not infer registration from an orphan history file written before a crash.
        history = self._path("session_record_history", session.isoformat(), route.key())
        versions = [self._read(item, route, session) for item in sorted(history.glob("*.json"))]
        prior = [item for item in versions if item.recorded_at_utc <= as_of and item.recorded_at_utc < record.recorded_at_utc]
        return max(prior, key=lambda item: item.recorded_at_utc) if prior else None

    def records(self, *, as_of: datetime) -> list[SessionRecord]:
        result = []
        for directory in sorted(self._path("session_records").glob("*")):
            if not directory.is_dir():
                continue
            try:
                session = date.fromisoformat(directory.name)
            except ValueError as exc:
                raise PredictionConflictError from exc
            for path in sorted(directory.glob("*.json")):
                record = self.load(self.load_route(path.stem), session, as_of=as_of)
                if record is not None:
                    result.append(record)
        return result

    def _path(self, *parts: str) -> Path:
        return path_integrity.verify_no_reparse_ancestry(self.root.joinpath(*parts), label="monitoring session evidence")

    @staticmethod
    def _read(path: Path, route: MonitoringRoute, session: date) -> SessionRecord:
        try:
            safe = path_integrity.verify_no_reparse_ancestry(path, label="monitoring session record")
            record = SessionRecord.model_validate(parse_strict_json_object(safe.read_bytes(), label="monitoring session record"))
        except (OSError, ValueError) as exc:
            raise PredictionConflictError from exc
        if record.route != route or record.decision_session != session:
            raise PredictionConflictError
        return record


def decision_cutoff(session: date) -> datetime:
    calendar = xcals.get_calendar("XNYS")
    if not calendar.is_session(pd.Timestamp(session)):
        raise ValueError("monitoring decisions require an XNYS session")
    value: datetime = swing_prediction_cutoffs(pd.Series([session])).iloc[0].to_pydatetime()
    return value


def expected_sessions(route: MonitoringRoute, *, start: datetime, end: datetime) -> tuple[date, ...]:
    if start.utcoffset() is None or end.utcoffset() is None or start > end:
        raise ValueError("invalid monitoring coverage window")
    calendar = xcals.get_calendar("XNYS")
    lower = max(start, route.promoted_at_utc).astimezone(UTC).date()
    upper = end.astimezone(UTC).date()
    if lower > upper:
        return ()
    sessions = calendar.sessions_in_range(str(lower), str(upper))
    cutoffs = swing_prediction_cutoffs(pd.Series([session.date() for session in sessions]))
    return tuple(session.date() for session, cutoff in zip(sessions, cutoffs, strict=True)
                 if max(start, route.promoted_at_utc) <= cutoff.to_pydatetime() <= end)
