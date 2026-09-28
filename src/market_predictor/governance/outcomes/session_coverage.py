"""Coverage evidence over the decision sessions expected since release activation."""
from __future__ import annotations

from datetime import date, datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, model_validator

from market_predictor.governance.outcomes.session_records import MonitoringRoute, SessionRecord, expected_sessions


class SessionCoverage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    route: MonitoringRoute
    expected_sessions: tuple[date, ...]
    registered_sessions: tuple[date, ...]
    failed_sessions: tuple[date, ...]
    missing_sessions: tuple[date, ...]
    source_record_ids: tuple[str, ...]

    @model_validator(mode="after")
    def partition(self) -> Self:
        groups = (self.expected_sessions, self.registered_sessions, self.failed_sessions, self.missing_sessions)
        if any(values != tuple(sorted(set(values))) for values in groups):
            raise ValueError("coverage sessions must be sorted and unique")
        actual = (*self.registered_sessions, *self.failed_sessions, *self.missing_sessions)
        if len(actual) != len(set(actual)) or set(actual) != set(self.expected_sessions):
            raise ValueError("coverage does not partition the expected sessions")
        if self.source_record_ids != tuple(sorted(set(self.source_record_ids))):
            raise ValueError("coverage source ids must be sorted and unique")
        if len(self.source_record_ids) != len(self.registered_sessions) + len(self.failed_sessions):
            raise ValueError("coverage source ids do not match recorded sessions")
        if any(len(value) != 64 or any(c not in "0123456789abcdef" for c in value) for value in self.source_record_ids):
            raise ValueError("coverage source id is invalid")
        return self


def build_session_coverage(
    routes: list[MonitoringRoute], records: list[SessionRecord], *, start: datetime, end: datetime,
) -> list[SessionCoverage]:
    lookup = {(record.route.key(), record.decision_session): record for record in records}
    coverage = []
    for route in routes:
        expected = expected_sessions(route, start=start, end=end)
        registered: list[date] = []
        failed: list[date] = []
        missing: list[date] = []
        ids: list[str] = []
        for session in expected:
            record = lookup.get((route.key(), session))
            if record is None:
                missing.append(session)
            else:
                ids.append(record.record_id)
                (registered if record.status == "registered" else failed).append(session)
        coverage.append(SessionCoverage(
            route=route, expected_sessions=expected, registered_sessions=tuple(registered), failed_sessions=tuple(failed),
            missing_sessions=tuple(missing), source_record_ids=tuple(sorted(ids)),
        ))
    return coverage
