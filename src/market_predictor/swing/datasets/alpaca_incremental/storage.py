"""Atomic publication of new immutable receipts; original archives are never edited."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.swing.datasets.alpaca_incremental.config import read_object


class IntegrityError(ValueError):
    """Retained successful evidence failed verification; do not repair by refetching."""


def publish(path: Path, payload: dict[str, Any], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    envelope = {"payload": payload, "sha256": json_sha256(payload)}
    try:
        write_json_object(temporary, envelope)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verified(path: Path) -> dict[str, Any]:
    try:
        envelope = read_object(path)
        payload = envelope.get("payload")
        if not isinstance(payload, dict) or envelope.get("sha256") != json_sha256(payload):
            raise IntegrityError("incremental receipt hash mismatch")
        return payload
    except (OSError, ValueError) as exc:
        raise IntegrityError("incremental receipt unreadable or altered") from exc
