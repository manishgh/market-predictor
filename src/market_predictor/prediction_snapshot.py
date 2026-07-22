from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.prediction_contracts import PredictionRequest, PredictionResponse

SNAPSHOT_SCHEMA = "market_predictor.prediction_snapshot.v1"
_SNAPSHOT_ID = re.compile(r"^[0-9a-f]{64}$")


class PredictionSnapshotStore:
    """Content-addressed, immutable persistence for served predictions."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def record(self, request: PredictionRequest, response: PredictionResponse) -> PredictionResponse:
        content = {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "request": request.model_dump(mode="json"),
            "response": response.model_dump(
                mode="json",
                exclude={"snapshot_id", "snapshot_sha256"},
            ),
        }
        digest = _content_sha256(content)
        envelope = {
            "schema": SNAPSHOT_SCHEMA,
            "snapshot_id": digest,
            "content_sha256": digest,
            "content": content,
        }
        path = self.path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(envelope, indent=2, sort_keys=True, ensure_ascii=True)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            self._validate_envelope(existing, expected_id=digest)
        else:
            temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
            temporary.write_text(encoded, encoding="utf-8")
            os.replace(temporary, path)
        return response.model_copy(update={"snapshot_id": digest, "snapshot_sha256": digest})

    def load(self, snapshot_id: str) -> tuple[PredictionRequest, PredictionResponse, dict[str, Any]]:
        path = self.path_for(snapshot_id)
        if not path.exists():
            raise FileNotFoundError(f"prediction snapshot does not exist: {snapshot_id}")
        envelope = json.loads(path.read_text(encoding="utf-8"))
        self._validate_envelope(envelope, expected_id=snapshot_id)
        content = envelope["content"]
        request = PredictionRequest.model_validate(content["request"])
        response = PredictionResponse.model_validate(content["response"]).model_copy(
            update={"snapshot_id": snapshot_id, "snapshot_sha256": snapshot_id}
        )
        return request, response, envelope

    def path_for(self, snapshot_id: str) -> Path:
        normalized = snapshot_id.strip().lower()
        if not _SNAPSHOT_ID.fullmatch(normalized):
            raise ValueError("snapshot_id must be a 64-character lowercase SHA-256 value")
        return self.root / normalized[:2] / f"{normalized}.json"

    @staticmethod
    def _validate_envelope(envelope: dict[str, Any], *, expected_id: str) -> None:
        if envelope.get("schema") != SNAPSHOT_SCHEMA:
            raise ValueError("unsupported prediction snapshot schema")
        snapshot_id = str(envelope.get("snapshot_id", ""))
        content_sha256 = str(envelope.get("content_sha256", ""))
        actual = _content_sha256(envelope.get("content"))
        if snapshot_id != expected_id or content_sha256 != expected_id or actual != expected_id:
            raise ValueError("prediction snapshot integrity check failed")


def _content_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
