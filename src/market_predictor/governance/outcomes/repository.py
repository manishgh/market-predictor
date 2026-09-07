from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.outcomes.contracts import (
    MaturationAttemptV1,
    MaturedOutcomeV2,
    PredictionMaturationIntentV2,
    PredictionMonitoringObservationV1,
    content_sha256,
    monitoring_observation_from_intent,
)
from market_predictor.locking import file_lock

T = TypeVar("T", bound=BaseModel)


class OutcomeRepository:
    """Durable immutable local repository for live prediction validation."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def record_intent(
        self,
        intent: PredictionMaturationIntentV2,
    ) -> PredictionMaturationIntentV2:
        path = self._key_path("intents", intent.maturation_key)
        semantic_path = self._key_path(
            "semantic",
            intent.semantic_prediction_id,
        )
        semantic_record = {
            "schema": "market_predictor.semantic_prediction.v1",
            "semantic_prediction_id": intent.semantic_prediction_id,
            "canonical_maturation_key": intent.maturation_key,
        }
        with file_lock(semantic_path):
            if semantic_path.exists():
                canonical_key = _semantic_record_key(
                    _load_object(semantic_path),
                    intent.semantic_prediction_id,
                )
                try:
                    canonical_intent = self.load_intent(canonical_key)
                except (FileNotFoundError, PredictionConflictError) as exc:
                    raise PredictionConflictError from exc
                if (
                    canonical_intent.semantic_prediction_id
                    != intent.semantic_prediction_id
                ):
                    raise PredictionConflictError
                self._write_idempotent(path, intent)
                self.record_observation(monitoring_observation_from_intent(intent))
            else:
                self._write_idempotent(path, intent)
                self.record_observation(monitoring_observation_from_intent(intent))
                _write_json_durable(semantic_path, semantic_record)
        return intent

    def record_observation(
        self,
        observation: PredictionMonitoringObservationV1,
    ) -> PredictionMonitoringObservationV1:
        if observation.maturation_key is not None:
            try:
                intent = self.load_intent(observation.maturation_key)
            except (FileNotFoundError, PredictionConflictError) as exc:
                raise PredictionConflictError from exc
            if monitoring_observation_from_intent(intent) != observation:
                raise PredictionConflictError
        path = self._key_path("observations", observation.observation_id)
        self._write_idempotent(path, observation)
        return observation

    def record_attempt(
        self,
        attempt: MaturationAttemptV1,
    ) -> MaturationAttemptV1:
        path = (
            self.root
            / "attempts"
            / attempt.maturation_key[:2]
            / attempt.maturation_key
            / f"{attempt.attempt_id}.json"
        )
        self._write_idempotent(path, attempt)
        return attempt

    def record_outcome(
        self,
        outcome: MaturedOutcomeV2,
        *,
        evidence_rows: list[dict[str, object]],
    ) -> MaturedOutcomeV2:
        actual_evidence_sha = content_sha256(evidence_rows)
        if actual_evidence_sha != outcome.evidence_sha256:
            raise PredictionConflictError
        try:
            intent = self.load_intent(outcome.maturation_key)
        except (FileNotFoundError, PredictionConflictError) as exc:
            raise PredictionConflictError from exc
        _assert_outcome_matches_intent(outcome, intent)
        evidence_path = self._key_path("evidence", outcome.evidence_sha256)
        outcome_path = self._key_path("outcomes", outcome.maturation_key)
        with file_lock(outcome_path):
            if outcome_path.exists():
                existing = self.load_outcome(outcome.maturation_key)
                if existing != outcome:
                    raise PredictionConflictError
                expected_evidence = {
                    "schema": "market_predictor.outcome_evidence.v1",
                    "evidence_sha256": outcome.evidence_sha256,
                    "rows": evidence_rows,
                }
                if (
                    not evidence_path.exists()
                    or _load_object(evidence_path) != expected_evidence
                ):
                    raise PredictionConflictError
                return existing
            self._write_plain_idempotent(
                evidence_path,
                {
                    "schema": "market_predictor.outcome_evidence.v1",
                    "evidence_sha256": outcome.evidence_sha256,
                    "rows": evidence_rows,
                },
            )
            _write_json_durable(
                outcome_path,
                outcome.model_dump(mode="json"),
            )
        return outcome

    def load_intent(self, maturation_key: str) -> PredictionMaturationIntentV2:
        intent = self._load_model(
            self._key_path("intents", maturation_key),
            PredictionMaturationIntentV2,
        )
        if intent.maturation_key != maturation_key:
            raise PredictionConflictError
        return intent

    def load_outcome(self, maturation_key: str) -> MaturedOutcomeV2:
        outcome = self._load_model(
            self._key_path("outcomes", maturation_key),
            MaturedOutcomeV2,
        )
        if outcome.maturation_key != maturation_key:
            raise PredictionConflictError
        try:
            intent = self.load_intent(maturation_key)
        except (FileNotFoundError, PredictionConflictError) as exc:
            raise PredictionConflictError from exc
        _assert_outcome_matches_intent(outcome, intent)
        _validate_evidence_record(
            _load_object(self._key_path("evidence", outcome.evidence_sha256)),
            outcome.evidence_sha256,
        )
        return outcome

    def semantic_canonical_key(self, semantic_prediction_id: str) -> str | None:
        path = self._key_path("semantic", semantic_prediction_id)
        if not path.exists():
            return None
        value = _semantic_record_key(_load_object(path), semantic_prediction_id)
        try:
            canonical = self.load_intent(value)
        except (FileNotFoundError, PredictionConflictError) as exc:
            raise PredictionConflictError from exc
        if canonical.semantic_prediction_id != semantic_prediction_id:
            raise PredictionConflictError
        return value

    def intents(self) -> list[PredictionMaturationIntentV2]:
        root = self.root / "intents"
        if not root.exists():
            return []
        return [self.load_intent(path.stem) for path in sorted(root.glob("*/*.json"))]

    def observations(self) -> list[PredictionMonitoringObservationV1]:
        root = self.root / "observations"
        if not root.exists():
            return []
        observations: list[PredictionMonitoringObservationV1] = []
        for path in sorted(root.glob("*/*.json")):
            observation = self._load_model(path, PredictionMonitoringObservationV1)
            if observation.observation_id != path.stem:
                raise PredictionConflictError
            if observation.maturation_key is not None:
                try:
                    intent = self.load_intent(observation.maturation_key)
                except (FileNotFoundError, PredictionConflictError) as exc:
                    raise PredictionConflictError from exc
                if observation != monitoring_observation_from_intent(intent):
                    raise PredictionConflictError
            observations.append(observation)
        return observations

    def has_outcome(self, maturation_key: str) -> bool:
        return self._key_path("outcomes", maturation_key).exists()

    def outcomes(self) -> list[MaturedOutcomeV2]:
        root = self.root / "outcomes"
        if not root.exists():
            return []
        return [self.load_outcome(path.stem) for path in sorted(root.glob("*/*.json"))]

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
            raise PredictionConflictError from exc

    def _key_path(self, collection: str, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("repository identity must be a lowercase SHA-256 value")
        return self.root / collection / digest[:2] / f"{digest}.json"


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
    if loaded.get("schema") != "market_predictor.semantic_prediction.v1":
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
    outcome: MaturedOutcomeV2,
    intent: PredictionMaturationIntentV2,
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
        record.get("schema") != "market_predictor.outcome_evidence.v1"
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
