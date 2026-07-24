from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from market_predictor.v3.errors import DataReadinessError


def canonical_policy_json(policy: Mapping[str, object]) -> str:
    return json.dumps(
        dict(policy),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def policy_sha256(policy: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_policy_json(policy).encode("utf-8")).hexdigest()


def stamped_label_policy(dataset: pd.DataFrame) -> dict[str, Any]:
    required = {"dataset_label_policy_json", "dataset_label_config_sha256"}
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise DataReadinessError(
            f"dataset is missing label policy identity: {', '.join(missing)}"
        )
    policy_values = dataset["dataset_label_policy_json"].dropna().astype(str).unique()
    hash_values = dataset["dataset_label_config_sha256"].dropna().astype(str).unique()
    if len(policy_values) != 1 or len(hash_values) != 1:
        raise DataReadinessError("dataset mixes label policies")
    try:
        loaded = json.loads(policy_values[0])
    except json.JSONDecodeError as exc:
        raise DataReadinessError("dataset label policy JSON is invalid") from exc
    if not isinstance(loaded, dict) or not loaded:
        raise DataReadinessError("dataset label policy must be a non-empty object")
    policy = {str(key): value for key, value in loaded.items()}
    if canonical_policy_json(policy) != policy_values[0]:
        raise DataReadinessError("dataset label policy JSON is not canonical")
    if policy_sha256(policy) != hash_values[0]:
        raise DataReadinessError("dataset label policy hash does not match its JSON")
    return policy
