"""Strict validation and normalization utilities for edge_rebuild."""
from __future__ import annotations



from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError


def require_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}

def require_string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DataReadinessError(f"{label} must be an array of strings")
    return tuple(value)

def require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"{label} must be a non-empty string")
    return value

def require_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataReadinessError(f"{label} must be a non-negative integer")
    return value

def strict_bool(value: object) -> bool:
    """Strictly convert an object to boolean for swing logic."""
    return value is True or isinstance(value, np.bool_) and bool(value)

def parse_strict_bool(value: object) -> bool | None:
    """Safely cast to boolean, or return None if invalid (used by intraday normalization)."""
    return bool(value) if isinstance(value, (bool, np.bool_)) else None

def iso_timestamp(value: object) -> str:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return cast(str, parsed.tz_convert("UTC").isoformat())
