from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from market_predictor.core import path_integrity
from market_predictor.core.cross_section_contracts import CrossSectionMember
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.prediction_contracts import (
    PredictionConflictError,
    PredictionDependencyError,
    PredictionNotFoundError,
    PredictionRequest,
    PredictionResponse,
    PredictionValidationError,
)
from market_predictor.locking import file_lock

SNAPSHOT_SCHEMA = "market_predictor.serving.snapshot_store"
_SNAPSHOT_ID = re.compile(r"^[0-9a-f]{64}$")


class CrossSectionReadContext(BaseModel):
    """Replay context, never a client request or a way around HTTP validation."""
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of: datetime
    tickers: tuple[str, ...]


class PredictionSnapshotStore:
    """Immutable audit snapshots, with a separate canonical cross-section identity."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def record(self, request: PredictionRequest, response: PredictionResponse) -> PredictionResponse:
        try:
            validated_request = PredictionRequest.model_validate(request.model_dump(mode="python"))
            validated_response = PredictionResponse.model_validate(response.model_dump(mode="python"))
            content = {
                "scope": "request",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "request": validated_request.model_dump(mode="json"),
                "response": validated_response.model_dump(mode="json", exclude={"snapshot_id", "snapshot_sha256"}),
            }
            return self._record(content, _content_sha256(content))
        except (TypeError, ValueError) as exc:
            raise PredictionConflictError from exc

    def record_cross_section(
        self, response: PredictionResponse, members: tuple[CrossSectionMember, ...], *,
        as_of: datetime, promoted_at: datetime,
    ) -> PredictionResponse:
        try:
            content = {
                "scope": "decision_cross_section",
                "recorded_at_utc": datetime.now(UTC).isoformat(),
                "as_of_utc": _aware(as_of).isoformat(),
                "promoted_at_utc": _aware(promoted_at).isoformat(),
                "members": [member.model_dump(mode="json") for member in members],
                "response": response.model_dump(mode="json", exclude={"snapshot_id", "snapshot_sha256"}),
            }
            _validate_cross_section(content)
            return self._record(content, _content_sha256(_canonical_cross_section(content)))
        except (TypeError, ValueError) as exc:
            raise PredictionConflictError from exc

    def _record(self, content: dict[str, Any], digest: str) -> PredictionResponse:
        envelope: dict[str, Any] = {
            "schema": SNAPSHOT_SCHEMA, "snapshot_id": digest, "content_sha256": _content_sha256(content), "content": content,
        }
        path = self.path_for(digest)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with file_lock(path):
                if path.exists():
                    # Identical decisions keep the first audit clocks and response ids.
                    existing = parse_strict_json_object(path.read_bytes(), label="prediction snapshot")
                    self._validate_envelope(existing, expected_id=digest)
                    envelope = existing
                else:
                    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
                    try:
                        with temporary.open("w", encoding="utf-8") as stream:
                            json.dump(envelope, stream, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise PredictionDependencyError from exc
        return PredictionResponse.model_validate(envelope["content"]["response"]).model_copy(
            update={"snapshot_id": digest, "snapshot_sha256": envelope["content_sha256"]},
        )

    def load(self, snapshot_id: str) -> tuple[PredictionRequest | CrossSectionReadContext, PredictionResponse, dict[str, Any]]:
        path = self.path_for(snapshot_id)
        if not path.exists():
            raise PredictionNotFoundError
        try:
            envelope: dict[str, Any] = parse_strict_json_object(path.read_bytes(), label="prediction snapshot")
            self._validate_envelope(envelope, expected_id=snapshot_id)
            content = envelope["content"]
            response = PredictionResponse.model_validate(content["response"]).model_copy(
                update={"snapshot_id": snapshot_id, "snapshot_sha256": envelope["content_sha256"]},
            )
            request: PredictionRequest | CrossSectionReadContext
            if content["scope"] == "request":
                request = PredictionRequest.model_validate(content["request"])
                if response.evidence is not None:
                    request = request.model_copy(update={"as_of": response.evidence.prediction_cutoff_utc})
            else:
                assert response.evidence is not None
                request = CrossSectionReadContext(
                    as_of=response.evidence.prediction_cutoff_utc,
                    tickers=tuple(row.ticker for row in response.predictions),
                )
            return request, response, envelope
        except PredictionConflictError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictionConflictError from exc
        except OSError as exc:
            raise PredictionDependencyError from exc

    def path_for(self, snapshot_id: str) -> Path:
        normalized = snapshot_id.strip().lower()
        if not _SNAPSHOT_ID.fullmatch(normalized):
            raise PredictionValidationError
        return path_integrity.verify_no_reparse_ancestry(
            self.root / normalized[:2] / f"{normalized}.json", label="prediction snapshot",
        )

    @staticmethod
    def _validate_envelope(envelope: dict[str, Any], *, expected_id: str) -> None:
        if set(envelope) != {"schema", "snapshot_id", "content_sha256", "content"} or envelope["schema"] != SNAPSHOT_SCHEMA:
            raise PredictionConflictError
        content = envelope["content"]
        if not isinstance(content, dict) or envelope["content_sha256"] != _content_sha256(content):
            raise PredictionConflictError
        if content.get("scope") == "request":
            if set(content) != {"scope", "recorded_at_utc", "request", "response"}:
                raise PredictionConflictError
            PredictionRequest.model_validate(content["request"])
            actual = _content_sha256(content)
        elif content.get("scope") == "decision_cross_section":
            _validate_cross_section(content)
            actual = _content_sha256(_canonical_cross_section(content))
        else:
            raise PredictionConflictError
        PredictionResponse.model_validate(content["response"])
        _aware(content["recorded_at_utc"])
        if envelope["snapshot_id"] != expected_id or actual != expected_id:
            raise PredictionConflictError


def _validate_cross_section(content: dict[str, Any]) -> None:
    if set(content) != {"scope", "recorded_at_utc", "as_of_utc", "promoted_at_utc", "members", "response"}:
        raise PredictionConflictError
    response = PredictionResponse.model_validate(content["response"])
    evidence = response.evidence
    if evidence is None or evidence.identity_status != "complete" or set(response.models) != {"swing"}:
        raise PredictionConflictError
    decision = evidence.prediction_cutoff_utc
    model = response.models["swing"]
    if (evidence.model_release_ids.get("swing") != model.release_id
            or evidence.model_artifact_sha256.get("swing") != model.artifact_sha256
            or evidence.view_prediction_policy_sha256.get("swing") != model.prediction_policy_sha256
            or evidence.view_prediction_cutoffs_utc.get("swing") != decision
            or response.resolved_horizons.get("swing") != model.resolved_horizon):
        raise PredictionConflictError
    if not _aware(content["promoted_at_utc"]) <= decision <= _aware(content["as_of_utc"]):
        raise PredictionConflictError
    members = tuple(CrossSectionMember.model_validate(value) for value in content["members"])
    tickers = {member.ticker for member in members}
    if not members or len(tickers) != len(members) or len({m.security_id for m in members}) != len(members):
        raise PredictionConflictError
    if len(response.predictions) != len(members) or {row.ticker for row in response.predictions} != tickers:
        raise PredictionConflictError
    predictions = {row.ticker: row.swing for row in response.predictions}
    rows = {row.ticker: row for row in evidence.row_feature_availability}
    if len(rows) != len(evidence.row_feature_availability):
        raise PredictionConflictError
    scored = set()
    for member in members:
        prediction = predictions[member.ticker]
        if prediction is None or member.membership_available_at_utc > decision:
            raise PredictionConflictError
        if member.abstention_reason is not None:
            if prediction.probability is not None or prediction.abstention_reasons != [member.abstention_reason]:
                raise PredictionConflictError
        else:
            row = rows.get(member.ticker)
            if prediction.probability is None or prediction.readiness.status != "valid" or row is None:
                raise PredictionConflictError
            if row.canonical_security_id != member.security_id or row.sector != member.sector or row.decision_time_utc != decision:
                raise PredictionConflictError
            if row.feature_available_at_utc > decision:
                raise PredictionConflictError
            scored.add(member.ticker)
    if set(rows) != scored:
        raise PredictionConflictError


def _canonical_cross_section(content: dict[str, Any]) -> dict[str, Any]:
    canonical = json.loads(json.dumps(content))
    for name in ("recorded_at_utc", "as_of_utc"):
        canonical.pop(name)
    canonical["members"].sort(key=lambda member: member["security_id"])
    response = canonical["response"]
    for name in ("request_id", "generated_at_utc", "snapshot_id", "snapshot_sha256"):
        response.pop(name, None)
    response["predictions"].sort(key=lambda row: row["ticker"])
    response["evidence"].pop("request_id")
    response["evidence"].pop("correlation_id")
    response["evidence"]["row_feature_availability"].sort(key=lambda row: (row["ticker"], row["view"]))
    for model in response["models"].values():
        model.pop("path", None)
    return dict(canonical)


def _aware(value: Any) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if parsed.utcoffset() is None:
        raise ValueError("snapshot clocks must be timezone-aware")
    return parsed.astimezone(UTC)


def _content_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()).hexdigest()
