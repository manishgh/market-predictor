from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from market_predictor.v3.errors import DataReadinessError


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _finite(mapping: Mapping[str, Any], key: str) -> float:
    value = float(mapping[key])
    if not math.isfinite(value):
        raise DataReadinessError(f"selection metric {key} is not finite")
    return value
