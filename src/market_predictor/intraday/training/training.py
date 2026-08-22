"""Verified loader boundary for the immutable A4.3 intraday bar dataset."""
from __future__ import annotations



import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday.datasets.bar_dataset import (
    INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA,
    INTRADAY_BAR_DATASET_SCHEMA,
    load_complete_intraday_bar_dataset,
)
from market_predictor.intraday.features.bar_features import (
    INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
    INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    INTRADAY_BAR_MODEL_FEATURES_SHA256,
)
from market_predictor.intraday.features.bar_labels import (
    INTRADAY_BAR_LABEL_SCHEMA_VERSION,
)
from market_predictor.intraday.datasets.history import json_sha256
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    release_process_memory,
)
from market_predictor.core.errors import DataReadinessError

MODEL_FEATURE_COLUMNS: Final = INTRADAY_BAR_MODEL_FEATURE_COLUMNS
MEMORY_HARD_BUDGET_GIB: Final = 4.0
MEMORY_HEADROOM_GIB: Final = 0.75

_REQUEST_NAME: Final = "_request.json"
_MANIFEST_NAME: Final = "_manifest.json"
_AUTHORITY_NAME: Final = "_authority.json"
_SESSION_CONCAT_CHUNK_SIZE: Final = 64

_IDENTITY_COLUMNS: Final = (
    "decision_id",
    "decision_cohort_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "session_date_et",
    "sector",
    "primary_benchmark",
    "universe_snapshot_id",
    "strategy_contract_sha256",
)
_TIMESTAMP_COLUMNS: Final = (
    "decision_time_utc",
    "feature_available_at_utc",
    "entry_time_utc",
    "entry_bar_end_utc",
    "exit_time_utc",
    "exit_bar_end_utc",
    "label_available_at_utc",
)
_BOOLEAN_COLUMNS: Final = (
    "dataset_eligible",
    "feature_eligible",
    "label_eligible",
    "target_hit",
    "stop_hit",
)
_RETURN_COLUMNS: Final = (
    "gross_return",
    "cost",
    "net_return",
    "spy_return",
    "qqq_return",
    "sector_return",
    "spy_excess_return",
    "qqq_excess_return",
    "sector_excess_return",
)
_PRICE_COLUMNS: Final = (
    "entry_price",
    "stop_price",
)
_ROW_CONTRACT_COLUMNS: Final = (
    "feature_schema_version",
    "label_schema_version",
    "ordered_feature_sha256",
)
_PROJECTED_COLUMNS: Final = tuple(
    dict.fromkeys(
        (
            *_IDENTITY_COLUMNS,
            *_TIMESTAMP_COLUMNS,
            *_BOOLEAN_COLUMNS,
            *_RETURN_COLUMNS,
            *_PRICE_COLUMNS,
            *_ROW_CONTRACT_COLUMNS,
            *MODEL_FEATURE_COLUMNS,
        )
    )
)


@dataclass(frozen=True, slots=True)
class PublishedIntradayDataset:
    """Verified A4.3 rows and the immutable identities that authorize them."""

    frame: pd.DataFrame
    root: Path
    feature_columns: tuple[str, ...]
    frozen_round_trip_cost_bps: float
    dataset_sha256: str
    manifest_sha256: str
    authority_sha256: str
    request_sha256: str
    transformation_sha256: str
    session_unit_inventory_sha256: str
    ordered_feature_sha256: str
    strategy_contract_sha256: str


def load_published_intraday_dataset(directory: Path) -> PublishedIntradayDataset:
    """Replay and project one complete A4.3 bar-only training authority."""

    _guard_memory("intraday A4.3 dataset load start", peak=False)
    root = directory.resolve()
    manifest = load_complete_intraday_bar_dataset(root)
    request = _read_json_object(root / _REQUEST_NAME, "dataset request")
    authority = _read_json_object(root / _AUTHORITY_NAME, "dataset authority")
    identity = _verify_authority_identity(root, request, manifest, authority)

    summary = _object(manifest.get("summary"), "manifest.summary")
    projected_rows = _required_non_negative_int(
        summary.get("dataset_eligible_rows"),
        "manifest.summary.dataset_eligible_rows",
    )
    if projected_rows < 1:
        raise DataReadinessError("published A4.3 dataset has no eligible training rows")
    _guard_projected_memory(projected_rows)

    raw_units = manifest.get("session_units")
    if not isinstance(raw_units, list) or not raw_units:
        raise DataReadinessError("published A4.3 dataset has no verified session units")

    parts: list[pd.DataFrame] = []
    chunks: list[pd.DataFrame] = []
    loaded_rows = 0
    for index, raw_unit in enumerate(raw_units):
        unit = _object(raw_unit, f"manifest.session_units[{index}]")
        session_date = _required_string(unit.get("session_date_et"), "session unit date")
        expected_rows = _required_non_negative_int(
            unit.get("dataset_eligible_rows"),
            f"session unit {session_date} eligible rows",
        )
        physical_rows = _required_non_negative_int(
            unit.get("rows"),
            f"session unit {session_date} rows",
        )
        if physical_rows == 0:
            if expected_rows != 0:
                raise DataReadinessError(
                    f"A4.3 session {session_date} eligible row count differs"
                )
            _guard_memory(f"intraday A4.3 session {index + 1} load", peak=False)
            continue
        rows_path = _session_rows_path(root, session_date)
        schema = pq.read_schema(rows_path)  # type: ignore[no-untyped-call]
        missing = sorted(set(_PROJECTED_COLUMNS).difference(schema.names))
        if missing:
            raise DataReadinessError(
                f"A4.3 session {session_date} is missing loader columns: {missing}"
            )
        part = pd.read_parquet(
            rows_path,
            columns=list(_PROJECTED_COLUMNS),
            filters=[("dataset_eligible", "==", True)],
        )
        if len(part) != expected_rows:
            raise DataReadinessError(
                f"A4.3 session {session_date} eligible row count differs"
            )
        if not part.empty:
            _normalize_loaded_part(part, session_date=session_date)
            parts.append(part)
            loaded_rows += len(part)
        if len(parts) >= _SESSION_CONCAT_CHUNK_SIZE:
            chunks.append(pd.concat(parts, ignore_index=True))
            parts.clear()
        _guard_memory(f"intraday A4.3 session {index + 1} load", peak=False)

    if parts:
        chunks.append(pd.concat(parts, ignore_index=True))
        parts.clear()
    if loaded_rows != projected_rows or not chunks:
        raise DataReadinessError(
            "loaded A4.3 eligible rows differ from the immutable dataset summary"
        )

    frame = pd.concat(chunks, ignore_index=True)
    del chunks
    frame["dataset_row_id"] = frame["decision_id"].astype("string[pyarrow]")
    _validate_complete_frame(
        frame,
        strategy_contract_sha256=identity["strategy_contract_sha256"],
    )
    frozen_cost_bps = float(frame["cost"].iloc[0] * 10_000.0)
    release_process_memory()
    _guard_memory("intraday A4.3 dataset load", peak=True)

    return PublishedIntradayDataset(
        frame=frame,
        root=root,
        feature_columns=MODEL_FEATURE_COLUMNS,
        frozen_round_trip_cost_bps=frozen_cost_bps,
        dataset_sha256=identity["manifest_sha256"],
        manifest_sha256=identity["manifest_sha256"],
        authority_sha256=identity["authority_sha256"],
        request_sha256=identity["request_sha256"],
        transformation_sha256=identity["transformation_sha256"],
        session_unit_inventory_sha256=identity["session_unit_inventory_sha256"],
        ordered_feature_sha256=INTRADAY_BAR_MODEL_FEATURES_SHA256,
        strategy_contract_sha256=identity["strategy_contract_sha256"],
    )


def _verify_authority_identity(
    root: Path,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, str]:
    request_sha256 = _required_string(request.get("request_sha256"), "request_sha256")
    request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
    manifest_sha256 = file_sha256(root / _MANIFEST_NAME)
    authority_sha256 = file_sha256(root / _AUTHORITY_NAME)
    transformation = _object(request.get("transformation"), "request.transformation")
    transformation_sha256 = _required_string(
        request.get("transformation_sha256"),
        "request.transformation_sha256",
    )
    raw_units = manifest.get("session_units")
    if not isinstance(raw_units, list):
        raise DataReadinessError("manifest.session_units must be an array")
    inventory_sha256 = json_sha256(raw_units)
    training_contract = _object(manifest.get("training_contract"), "manifest.training_contract")
    ordered_names = _string_tuple(
        training_contract.get("ordered_feature_names"),
        "manifest.training_contract.ordered_feature_names",
    )
    request_ordered_names = _string_tuple(
        request.get("ordered_feature_names"),
        "request.ordered_feature_names",
    )
    strategy_contract_sha256 = _required_string(
        request.get("strategy_contract_sha256"),
        "request.strategy_contract_sha256",
    )
    label_columns = set(
        _string_tuple(
            training_contract.get("label_columns"),
            "manifest.training_contract.label_columns",
        )
    )
    required_labels = {
        "target_hit",
        "stop_hit",
        "entry_price",
        "stop_price",
        *_RETURN_COLUMNS,
    }
    if (
        request.get("schema") != INTRADAY_BAR_DATASET_SCHEMA
        or manifest.get("schema") != INTRADAY_BAR_DATASET_SCHEMA
        or authority.get("schema") != INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != _MANIFEST_NAME
        or authority.get("artifact_sha256") != manifest_sha256
        or request_sha256 != json_sha256(request_payload)
        or manifest.get("request_sha256") != request_sha256
        or authority.get("request_sha256") != request_sha256
        or transformation.get("sha256") != transformation_sha256
        or manifest.get("transformation_sha256") != transformation_sha256
        or manifest.get("session_unit_inventory_sha256") != inventory_sha256
        or authority.get("session_unit_inventory_sha256") != inventory_sha256
        or request.get("feature_schema_version") != INTRADAY_BAR_FEATURE_SCHEMA_VERSION
        or request.get("label_schema_version") != INTRADAY_BAR_LABEL_SCHEMA_VERSION
        or request_ordered_names != MODEL_FEATURE_COLUMNS
        or ordered_names != MODEL_FEATURE_COLUMNS
        or request.get("ordered_feature_sha256") != INTRADAY_BAR_MODEL_FEATURES_SHA256
        or training_contract.get("ordered_feature_sha256")
        != INTRADAY_BAR_MODEL_FEATURES_SHA256
        or training_contract.get("eligibility_column") != "dataset_eligible"
        or not required_labels.issubset(label_columns)
    ):
        raise DataReadinessError("A4.3 dataset loader identity differs")
    return {
        "manifest_sha256": manifest_sha256,
        "authority_sha256": authority_sha256,
        "request_sha256": request_sha256,
        "transformation_sha256": transformation_sha256,
        "session_unit_inventory_sha256": inventory_sha256,
        "strategy_contract_sha256": strategy_contract_sha256,
    }


def _normalize_loaded_part(part: pd.DataFrame, *, session_date: str) -> None:
    if not part["session_date_et"].astype(str).eq(session_date).all():
        raise DataReadinessError(f"A4.3 session {session_date} row identity differs")
    for column in MODEL_FEATURE_COLUMNS:
        values = pd.to_numeric(part[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"A4.3 model feature {column} must be finite")
        part[column] = values.astype("float32")
    for column in _RETURN_COLUMNS:
        values = pd.to_numeric(part[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"A4.3 economic column {column} must be finite")
        part[column] = values.astype("float64")
    for column in _PRICE_COLUMNS:
        values = pd.to_numeric(part[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all() or values.le(0.0).any():
            raise DataReadinessError(f"A4.3 price column {column} must be finite and positive")
        part[column] = values.astype("float64")
    for column in _TIMESTAMP_COLUMNS:
        values = pd.to_datetime(part[column], utc=True, errors="coerce")
        if values.isna().any():
            raise DataReadinessError(f"A4.3 timestamp {column} must be valid UTC")
        part[column] = values
    for column in _BOOLEAN_COLUMNS:
        values = part[column].map(_strict_bool)
        if values.isna().any():
            raise DataReadinessError(f"A4.3 boolean column {column} is invalid")
        part[column] = values.astype(bool)
    for column in (*_IDENTITY_COLUMNS, *_ROW_CONTRACT_COLUMNS):
        part[column] = part[column].astype("string[pyarrow]")


def _validate_complete_frame(
    frame: pd.DataFrame,
    *,
    strategy_contract_sha256: str,
) -> None:
    if len(frame) < 1:
        raise DataReadinessError("published A4.3 dataset has no eligible rows")
    if not frame["dataset_eligible"].all() or not frame["feature_eligible"].all() or not frame["label_eligible"].all():
        raise DataReadinessError("A4.3 loader may expose only fully eligible rows")
    for column in (*_IDENTITY_COLUMNS, "dataset_row_id"):
        values = frame[column].astype("string")
        if values.isna().any() or values.str.strip().eq("").any():
            raise DataReadinessError(f"A4.3 identity column {column} must be complete")
    if frame["decision_id"].duplicated().any():
        raise DataReadinessError("A4.3 decision_id must be globally unique")
    if not frame["strategy_contract_sha256"].astype(str).eq(strategy_contract_sha256).all():
        raise DataReadinessError("A4.3 row strategy contract differs from its authority")
    if not frame["dataset_row_id"].astype(str).eq(frame["decision_id"].astype(str)).all():
        raise DataReadinessError("dataset_row_id must alias the A4.3 decision_id")
    if not frame["feature_schema_version"].astype(str).eq(INTRADAY_BAR_FEATURE_SCHEMA_VERSION).all():
        raise DataReadinessError("A4.3 row feature schema differs")
    if not frame["label_schema_version"].astype(str).eq(INTRADAY_BAR_LABEL_SCHEMA_VERSION).all():
        raise DataReadinessError("A4.3 row label schema differs")
    if not frame["ordered_feature_sha256"].astype(str).eq(INTRADAY_BAR_MODEL_FEATURES_SHA256).all():
        raise DataReadinessError("A4.3 row ordered feature hash differs")
    group_times = frame.groupby("decision_group_id", observed=True)["decision_time_utc"].nunique()
    if group_times.ne(1).any():
        raise DataReadinessError("A4.3 decision group must map to one decision time")
    parsed_group = pd.to_datetime(frame["decision_group_id"], utc=True, errors="coerce")
    if parsed_group.isna().any() or not parsed_group.eq(frame["decision_time_utc"]).all():
        raise DataReadinessError("A4.3 decision_group_id must be the UTC decision time")
    if (
        not frame["feature_available_at_utc"].eq(frame["decision_time_utc"]).all()
        or not frame["entry_time_utc"].eq(frame["decision_time_utc"] + pd.Timedelta(minutes=1)).all()
        or frame["exit_bar_end_utc"].le(frame["entry_time_utc"]).any()
        or frame["exit_bar_end_utc"].gt(frame["label_available_at_utc"]).any()
    ):
        raise DataReadinessError("A4.3 eligible rows violate causal timestamp identity")
    if (frame["target_hit"] & frame["stop_hit"]).any():
        raise DataReadinessError("A4.3 target and stop cannot both be first")
    if frame["stop_price"].ge(frame["entry_price"]).any():
        raise DataReadinessError("A4.3 long stop price must be below its entry price")
    costs = frame["cost"].to_numpy(dtype="float64", copy=False)
    if (costs < 0.0).any() or np.unique(costs).size != 1:
        raise DataReadinessError("A4.3 dataset must use one finite non-negative cost")
    gross = frame["gross_return"].to_numpy(dtype="float64", copy=False)
    net = frame["net_return"].to_numpy(dtype="float64", copy=False)
    if not np.allclose(gross - costs, net, rtol=0.0, atol=1e-10):
        raise DataReadinessError("A4.3 net return does not apply the frozen cost exactly once")
    for benchmark in ("spy", "qqq", "sector"):
        raw = frame[f"{benchmark}_return"].to_numpy(dtype="float64", copy=False)
        excess = frame[f"{benchmark}_excess_return"].to_numpy(dtype="float64", copy=False)
        if not np.allclose(net - raw, excess, rtol=0.0, atol=1e-10):
            raise DataReadinessError(
                f"A4.3 {benchmark} excess return does not match the exact managed interval"
            )


def _guard_projected_memory(rows: int) -> None:
    numeric_bytes = (
        len(MODEL_FEATURE_COLUMNS) * 4
        + (len(_RETURN_COLUMNS) + len(_PRICE_COLUMNS)) * 8
    )
    timestamp_bytes = len(_TIMESTAMP_COLUMNS) * 8
    boolean_bytes = len(_BOOLEAN_COLUMNS)
    text_bytes = (len(_IDENTITY_COLUMNS) + len(_ROW_CONTRACT_COLUMNS) + 1) * 24
    projected_peak = rows * (numeric_bytes + timestamp_bytes + boolean_bytes + text_bytes) * 3
    safety_bytes = int((MEMORY_HARD_BUDGET_GIB - MEMORY_HEADROOM_GIB) * 1024**3)
    if projected_peak > safety_bytes:
        raise DataReadinessError(
            "projected A4.3 loader memory exceeds the 3.25 GiB safety threshold"
        )


def _session_rows_path(root: Path, session_date: str) -> Path:
    candidate = (
        root / "sessions" / f"session_date_et={session_date}" / "rows.parquet"
    ).resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise DataReadinessError("A4.3 session rows path is invalid")
    return candidate


def _guard_memory(stage: str, *, peak: bool) -> None:
    function = assert_peak_memory_budget if peak else assert_memory_budget
    function(
        hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be a JSON object")
    return {str(key): item for key, item in value.items()}


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise DataReadinessError(f"{label} must be an array of strings")
    return tuple(value)


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"{label} must be a non-empty string")
    return value


def _required_non_negative_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataReadinessError(f"{label} must be a non-negative integer")
    return value


def _strict_bool(value: object) -> bool | None:
    return bool(value) if isinstance(value, (bool, np.bool_)) else None
