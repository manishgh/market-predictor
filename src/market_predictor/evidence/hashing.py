"""Hashing utilities for edge_rebuild."""
from __future__ import annotations



import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def sequence_sha256(values: Sequence[str] | pd.Series) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

def json_sha256(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
