from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from market_predictor.locking import file_lock
from market_predictor.outcome_contracts import (
    MaturationAttemptV1,
    MaturedOutcomeV1,
    PredictionMaturationIntentV1,
    content_sha256,
)
from market_predictor.prediction_contracts import PredictionConflictError

T = TypeVar("T", bound=BaseModel)


class OutcomeRepository:
    """Durable immutable local repository for live prediction validation."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def record_intent(
        self,
        intent: PredictionMaturationIntentV1,
    ) -> PredictionMaturationIntentV1:
        path = self._key_path("intents", intent.maturation_key)
        self._write_idempotent(path, intent)
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
                existing = _load_object(semantic_path)
                if (
                    existing.get("semantic_prediction_id")
                    != intent.semantic_prediction_id
                ):
                    raise PredictionConflictError
            else:
                _write_json_durable(semantic_path, semantic_record)
        return intent

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
        outcome: MaturedOutcomeV1,
        *,
        evidence_rows: list[dict[str, object]],
    ) -> MaturedOutcomeV1:
        actual_evidence_sha = content_sha256(evidence_rows)
        if actual_evidence_sha != outcome.evidence_sha256:
            raise PredictionConflictError
        evidence_path = self._key_path("evidence", outcome.evidence_sha256)
        outcome_path = self._key_path("outcomes", outcome.maturation_key)
        with file_lock(outcome_path):
            if outcome_path.exists():
                return self.load_outcome(outcome.maturation_key)
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

    def load_intent(self, maturation_key: str) -> PredictionMaturationIntentV1:
        return self._load_model(
            self._key_path("intents", maturation_key),
            PredictionMaturationIntentV1,
        )

    def load_outcome(self, maturation_key: str) -> MaturedOutcomeV1:
        return self._load_model(
            self._key_path("outcomes", maturation_key),
            MaturedOutcomeV1,
        )

    def semantic_canonical_key(self, semantic_prediction_id: str) -> str | None:
        path = self._key_path("semantic", semantic_prediction_id)
        if not path.exists():
            return None
        loaded = _load_object(path)
        value = str(loaded.get("canonical_maturation_key") or "")
        return value or None

    def intents(self) -> list[PredictionMaturationIntentV1]:
        root = self.root / "intents"
        if not root.exists():
            return []
        return [
            PredictionMaturationIntentV1.model_validate(_load_object(path))
            for path in sorted(root.glob("*/*.json"))
        ]

    def has_outcome(self, maturation_key: str) -> bool:
        return self._key_path("outcomes", maturation_key).exists()

    def outcomes(self) -> list[MaturedOutcomeV1]:
        root = self.root / "outcomes"
        if not root.exists():
            return []
        return [
            MaturedOutcomeV1.model_validate(_load_object(path))
            for path in sorted(root.glob("*/*.json"))
        ]

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
        return model.model_validate(_load_object(path))

    def _key_path(self, collection: str, digest: str) -> Path:
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("repository identity must be a lowercase SHA-256 value")
        return self.root / collection / digest[:2] / f"{digest}.json"


def _load_object(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise PredictionConflictError
    return {str(key): value for key, value in loaded.items()}


def _write_json_durable(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
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
