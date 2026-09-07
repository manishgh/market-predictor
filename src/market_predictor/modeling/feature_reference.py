"""Compact numeric feature references stored with trained models."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


def build_feature_reference_profile(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, dict[str, float | int | None]]:
    """Build a numeric training reference without retaining source rows."""

    profile: dict[str, dict[str, float | int | None]] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        valid = values.dropna()
        profile[feature] = {
            "rows": int(len(values)),
            "observed": int(len(valid)),
            "missing_rate": float(values.isna().mean()),
            "mean": float(valid.mean()) if not valid.empty else None,
            "std": float(valid.std(ddof=0)) if not valid.empty else None,
        }
    return profile


def feature_reference_profile_sha256(profile: dict[str, Any]) -> str:
    encoded = json.dumps(
        profile,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def feature_reference_names_sha256(feature_names: Iterable[str]) -> str:
    names = [str(name) for name in feature_names]
    if not names or len(names) != len(set(names)) or any(not name for name in names):
        raise ValueError("feature reference names must be non-empty and unique")
    encoded = json.dumps(
        sorted(names),
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
