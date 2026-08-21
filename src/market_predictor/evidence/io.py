"""JSON and artifact I/O utilities for edge_rebuild."""
from __future__ import annotations



import json
from pathlib import Path
from typing import Any

from market_predictor.core.errors import DataReadinessError


def read_json_object(path: Path, label: str = "JSON object") -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be a JSON object")
    # Ensuring keys are strings for strict typing
    return {str(key): item for key, item in value.items()}

def write_json_object(path: Path, value: dict[str, Any] | Any) -> None:
    if path.exists():
        raise FileExistsError(f"immutable artifact already exists: {path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

def resolve_inside_authority(root: Path, raw: object) -> Path:
    path = (root / str(raw)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DataReadinessError(f"artifact escapes authority root: {raw}") from exc
    if not path.is_file():
        raise DataReadinessError(f"authority artifact is missing: {path}")
    return path
