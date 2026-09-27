from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.outcomes.contracts import (
    RETIRED_INTRADAY,
    MaturationAttempt,
    MaturedOutcome,
    PredictionMaturationIntent,
    PredictionMonitoringObservation,
    content_sha256,
    monitoring_observation_from_intent,
)
from market_predictor.locking import file_lock

T = TypeVar("T", bound=BaseModel)
_PENDING = "pending"
_SESSIONS = "sessions"


class OutcomeRepository:
    """Durable immutable local repository for live prediction validation.

    Records are partitioned by decision session, so a report reads only the sessions its
    window covers. A pending index names each canonical intent that has no outcome yet,
    so maturation and overdue checks never scan history.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def record_intent(
        self,
        intent: PredictionMaturationIntent,
    ) -> PredictionMaturationIntent:
        session = intent.decision_session_et
        path = self._path(session, "intents", intent.maturation_key)
        semantic_path = self._path(session, "semantic", intent.semantic_prediction_id)
        semantic_record = {
            "schema": "market_predictor.semantic_prediction",
            "semantic_prediction_id": intent.semantic_prediction_id,
            "canonical_maturation_key": intent.maturation_key,
        }
        with file_lock(semantic_path):
            canonical = not semantic_path.exists()
            if not canonical:
                canonical_key = _semantic_record_key(
                    _load_object(semantic_path),
                    intent.semantic_prediction_id,
                )
                try:
                    canonical_intent = self.load_intent(canonical_key, session)
                except (FileNotFoundError, PredictionConflictError) as exc:
                    raise PredictionConflictError from exc
                if canonical_intent.semantic_prediction_id != intent.semantic_prediction_id:
                    raise PredictionConflictError
            self._write_idempotent(path, intent)
            self._record_horizon(session, intent.horizon)
            self.record_observation(monitoring_observation_from_intent(intent))
            if canonical:
                _write_json_durable(semantic_path, semantic_record)
                if not self.has_outcome(intent.maturation_key, session):
                    self._write_pending(intent.maturation_key, session)
        return intent

    def record_observation(
        self,
        observation: PredictionMonitoringObservation,
    ) -> PredictionMonitoringObservation:
        session = observation.decision_session_et
        if observation.maturation_key is not None:
            try:
                intent = self.load_intent(observation.maturation_key, session)
            except (FileNotFoundError, PredictionConflictError) as exc:
                raise PredictionConflictError from exc
            if monitoring_observation_from_intent(intent) != observation:
                raise PredictionConflictError
        self._write_idempotent(self._path(session, "observations", observation.observation_id), observation)
        self._record_horizon(session, observation.horizon)
        return observation

    def record_attempt(
        self,
        attempt: MaturationAttempt,
        *,
        decision_session: date,
    ) -> MaturationAttempt:
        key = _digest(attempt.maturation_key)
        path = self._partition(decision_session) / "attempts" / key / f"{_digest(attempt.attempt_id)}.json"
        self._write_idempotent(path, attempt)
        return attempt

    def record_outcome(
        self,
        intent: PredictionMaturationIntent,
        outcome: MaturedOutcome,
        *,
        evidence_rows: list[dict[str, object]],
    ) -> MaturedOutcome:
        if content_sha256(evidence_rows) != outcome.evidence_sha256:
            raise PredictionConflictError
        session = intent.decision_session_et
        try:
            stored = self.load_intent(outcome.maturation_key, session)
        except (FileNotFoundError, PredictionConflictError) as exc:
            raise PredictionConflictError from exc
        if stored != intent:
            raise PredictionConflictError
        _assert_outcome_matches_intent(outcome, intent)
        evidence_path = self._path(session, "evidence", outcome.evidence_sha256)
        outcome_path = self._path(session, "outcomes", outcome.maturation_key)
        evidence: dict[str, object] = {
            "schema": "market_predictor.outcome_evidence",
            "evidence_sha256": outcome.evidence_sha256,
            "rows": evidence_rows,
        }
        with file_lock(outcome_path):
            if outcome_path.exists():
                existing = self.load_outcome(outcome.maturation_key, session)
                if existing != outcome or not evidence_path.exists() or _load_object(evidence_path) != evidence:
                    raise PredictionConflictError
            else:
                self._write_plain_idempotent(evidence_path, evidence)
                _write_json_durable(outcome_path, outcome.model_dump(mode="json"))
            # The outcome is durable before the index entry goes; a crash in between leaves a
            # stale entry that maturation drops on its next pass.
            self._pending_path(outcome.maturation_key).unlink(missing_ok=True)
        return outcome

    def load_intent(self, maturation_key: str, session: date) -> PredictionMaturationIntent:
        intent = self._load_model(self._path(session, "intents", maturation_key), PredictionMaturationIntent)
        if intent.maturation_key != maturation_key or intent.decision_session_et != session:
            raise PredictionConflictError
        return intent

    def load_outcome(self, maturation_key: str, session: date) -> MaturedOutcome:
        outcome = self._load_model(self._path(session, "outcomes", maturation_key), MaturedOutcome)
        if outcome.maturation_key != maturation_key:
            raise PredictionConflictError
        try:
            intent = self.load_intent(maturation_key, session)
        except (FileNotFoundError, PredictionConflictError) as exc:
            raise PredictionConflictError from exc
        _assert_outcome_matches_intent(outcome, intent)
        _validate_evidence_record(
            _load_object(self._path(session, "evidence", outcome.evidence_sha256)),
            outcome.evidence_sha256,
        )
        return outcome

    def has_outcome(self, maturation_key: str, session: date) -> bool:
        return self._path(session, "outcomes", maturation_key).exists()

    def semantic_canonical_key(self, semantic_prediction_id: str, session: date) -> str | None:
        path = self._path(session, "semantic", semantic_prediction_id)
        if not path.exists():
            return None
        value = _semantic_record_key(_load_object(path), semantic_prediction_id)
        try:
            canonical = self.load_intent(value, session)
        except (FileNotFoundError, PredictionConflictError) as exc:
            raise PredictionConflictError from exc
        if canonical.semantic_prediction_id != semantic_prediction_id:
            raise PredictionConflictError
        return value

    def sessions(self) -> tuple[date, ...]:
        """Every decision session with stored records, in order."""
        root = self.root / _SESSIONS
        if not root.exists():
            return ()
        sessions = []
        for path in root.iterdir():
            if path.name.startswith("."):
                continue
            try:
                sessions.append(date.fromisoformat(path.name))
            except ValueError as exc:
                raise PredictionConflictError from exc
        return tuple(sorted(sessions))

    def session_horizons(self, session: date) -> frozenset[str]:
        path = self._partition(session) / "horizons.json"
        if not path.exists():
            return frozenset()
        loaded = _load_object(path)
        horizons = loaded.get("horizons")
        if loaded.get("schema") != "market_predictor.session_horizons" or not isinstance(horizons, list):
            raise PredictionConflictError
        return frozenset(str(horizon) for horizon in horizons)

    def session_intents(self, session: date) -> list[PredictionMaturationIntent]:
        root = self._partition(session) / "intents"
        if not root.exists():
            return []
        return [self.load_intent(path.stem, session) for path in sorted(root.glob("*.json"))]

    def session_observations(self, session: date) -> list[PredictionMonitoringObservation]:
        root = self._partition(session) / "observations"
        if not root.exists():
            return []
        observations: list[PredictionMonitoringObservation] = []
        for path in sorted(root.glob("*.json")):
            observation = self._load_model(path, PredictionMonitoringObservation)
            if observation.observation_id != path.stem or observation.decision_session_et != session:
                raise PredictionConflictError
            if observation.maturation_key is not None:
                try:
                    intent = self.load_intent(observation.maturation_key, session)
                except (FileNotFoundError, PredictionConflictError) as exc:
                    raise PredictionConflictError from exc
                if observation != monitoring_observation_from_intent(intent):
                    raise PredictionConflictError
            observations.append(observation)
        return observations

    def pending(self) -> list[tuple[str, date]]:
        """Canonical intents without an outcome, as (maturation key, decision session)."""
        root = self.root / _PENDING
        if not root.exists():
            return []
        entries = []
        for path in sorted(root.glob("*.json")):
            loaded = _load_object(path)
            key = str(loaded.get("maturation_key") or "")
            if loaded.get("schema") != "market_predictor.pending_outcome" or key != path.stem:
                raise PredictionConflictError
            try:
                session = date.fromisoformat(str(loaded.get("decision_session_et")))
            except ValueError as exc:
                raise PredictionConflictError from exc
            entries.append((_digest(key), session))
        return entries

    def drop_pending(self, maturation_key: str, session: date) -> None:
        """Remove an index entry whose outcome is already durable."""
        if not self.has_outcome(maturation_key, session):
            raise PredictionConflictError
        self._pending_path(maturation_key).unlink(missing_ok=True)

    def _write_pending(self, maturation_key: str, session: date) -> None:
        self._write_plain_idempotent(
            self._pending_path(maturation_key),
            {
                "schema": "market_predictor.pending_outcome",
                "maturation_key": maturation_key,
                "decision_session_et": session.isoformat(),
            },
        )

    def _record_horizon(self, session: date, horizon: str) -> None:
        path = self._partition(session) / "horizons.json"
        with file_lock(path):
            horizons = set(self.session_horizons(session))
            if horizon in horizons:
                return
            _write_json_durable(
                path,
                {"schema": "market_predictor.session_horizons", "horizons": sorted({*horizons, horizon})},
            )

    def _partition(self, session: date) -> Path:
        return self.root / _SESSIONS / session.isoformat()

    def _path(self, session: date, collection: str, digest: str) -> Path:
        return self._partition(session) / collection / f"{_digest(digest)}.json"

    def _pending_path(self, maturation_key: str) -> Path:
        return self.root / _PENDING / f"{_digest(maturation_key)}.json"

    def _write_idempotent(self, path: Path, value: BaseModel) -> None:
        expected = value.model_dump(mode="json")
        with file_lock(path):
            if path.exists():
                if _load_object(path) != expected:
                    raise PredictionConflictError
                return
            _write_json_durable(path, expected)

    def _write_plain_idempotent(
        self,
        path: Path,
        value: dict[str, object],
    ) -> None:
        with file_lock(path):
            if path.exists():
                if _load_object(path) != value:
                    raise PredictionConflictError
                return
            _write_json_durable(path, value)

    def _load_model(self, path: Path, model: type[T]) -> T:
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            return model.model_validate(_load_object(path))
        except ValidationError as exc:
            if any(RETIRED_INTRADAY in str(error.get("msg", "")) for error in exc.errors()):
                raise DataReadinessError(f"{RETIRED_INTRADAY}: {path}") from exc
            raise PredictionConflictError from exc


def _digest(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("repository identity must be a lowercase SHA-256 value")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    try:
        loaded = parse_strict_json_object(
            path.read_bytes(),
            label="outcome repository artifact",
        )
    except (OSError, ValueError) as exc:
        raise PredictionConflictError from exc
    if not isinstance(loaded, dict):
        raise PredictionConflictError
    return {str(key): value for key, value in loaded.items()}


def _semantic_record_key(
    loaded: dict[str, Any],
    semantic_prediction_id: str,
) -> str:
    if loaded.get("schema") != "market_predictor.semantic_prediction":
        raise PredictionConflictError
    if loaded.get("semantic_prediction_id") != semantic_prediction_id:
        raise PredictionConflictError
    value = str(loaded.get("canonical_maturation_key") or "")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise PredictionConflictError
    return value


def _assert_outcome_matches_intent(
    outcome: MaturedOutcome,
    intent: PredictionMaturationIntent,
) -> None:
    try:
        label_cost_value = intent.label_policy["round_trip_cost_bps"]
    except (KeyError, TypeError) as exc:
        raise PredictionConflictError from exc
    if isinstance(label_cost_value, bool) or not isinstance(
        label_cost_value, (int, float)
    ):
        raise PredictionConflictError
    if (
        outcome.semantic_prediction_id != intent.semantic_prediction_id
        or outcome.snapshot_id != intent.snapshot_id
        or outcome.ticker != intent.ticker
        or outcome.view != intent.view
        or outcome.horizon != intent.horizon
        or outcome.execution_policy_sha256 != intent.execution_policy_sha256
        or outcome.decision_atr != intent.decision_atr
        or outcome.label_round_trip_cost_bps != float(label_cost_value)
        or outcome.execution_participation_fraction != 0.0
    ):
        raise PredictionConflictError


def _validate_evidence_record(
    record: dict[str, Any],
    expected_sha256: str,
) -> None:
    rows = record.get("rows")
    if (
        record.get("schema") != "market_predictor.outcome_evidence"
        or record.get("evidence_sha256") != expected_sha256
        or not isinstance(rows, list)
        or content_sha256(rows) != expected_sha256
    ):
        raise PredictionConflictError


def _write_json_durable(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
