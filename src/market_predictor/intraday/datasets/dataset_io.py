from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.memory import guard_memory
from market_predictor.core.validation import (
    parse_strict_bool,
    require_non_negative_int,
    require_object,
    require_string,
    require_string_tuple,
)
from market_predictor.intraday.datasets.bar_dataset import (
    INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA,
    INTRADAY_BAR_DATASET_SCHEMA,
    load_complete_intraday_bar_dataset,
)
from market_predictor.intraday.features.bar_features import (
    INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
    INTRADAY_BAR_MODEL_FEATURES_SHA256,
)
from market_predictor.intraday.features.bar_labels import (
    INTRADAY_BAR_LABEL_SCHEMA_VERSION,
)
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import read_json_object
from market_predictor.intraday.contracts.memory import MEMORY_HARD_BUDGET_GIB, MEMORY_HEADROOM_GIB
from market_predictor.intraday.datasets.dataset import PublishedIntradayDataset
from market_predictor.intraday.datasets.io import AUTHORITY_NAME, MANIFEST_NAME, REQUEST_NAME, SESSION_CONCAT_CHUNK_SIZE
from market_predictor.intraday.features.columns import (
    BOOLEAN_COLUMNS,
    IDENTITY_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    PRICE_COLUMNS,
    PROJECTED_COLUMNS,
    RETURN_COLUMNS,
    ROW_CONTRACT_COLUMNS,
    TIMESTAMP_COLUMNS,
)
from market_predictor.resources import release_process_memory

"""Verified loader boundary for the immutable A4.3 intraday bar dataset."""

def load_published_intraday_dataset(directory: Path) -> PublishedIntradayDataset:
    """Replay and project one complete A4.3 bar-only training authority."""

    guard_memory("intraday A4.3 dataset load start", peak=False, hard_budget_gib=MEMORY_HARD_BUDGET_GIB, headroom_gib=MEMORY_HEADROOM_GIB)
    root = directory.resolve()
    manifest = load_complete_intraday_bar_dataset(root)
    request = read_json_object(root / REQUEST_NAME, "dataset request")
    authority = read_json_object(root / AUTHORITY_NAME, "dataset authority")
    identity = _verify_authority_identity(root, request, manifest, authority)

    summary = require_object(manifest.get("summary"), "manifest.summary")
    projected_rows = require_non_negative_int(
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
        unit = require_object(raw_unit, f"manifest.session_units[{index}]")
        session_date = require_string(unit.get("session_date_et"), "session unit date")
        expected_rows = require_non_negative_int(
            unit.get("dataset_eligible_rows"),
            f"session unit {session_date} eligible rows",
        )
        physical_rows = require_non_negative_int(
            unit.get("rows"),
            f"session unit {session_date} rows",
        )
        if physical_rows == 0:
            if expected_rows != 0:
                raise DataReadinessError(f"A4.3 session {session_date} eligible row count differs")
            guard_memory(
                f"intraday A4.3 session {index + 1} load",
                peak=False,
                hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
                headroom_gib=MEMORY_HEADROOM_GIB,
            )
            continue
        rows_path = _session_rows_path(root, session_date)
        schema = pq.read_schema(rows_path)  # type: ignore[no-untyped-call]
        missing = sorted(set(PROJECTED_COLUMNS).difference(schema.names))
        if missing:
            raise DataReadinessError(f"A4.3 session {session_date} is missing loader columns: {missing}")
        part = pd.read_parquet(
            rows_path,
            columns=list(PROJECTED_COLUMNS),
            filters=[("dataset_eligible", "==", True)],
        )
        if len(part) != expected_rows:
            raise DataReadinessError(f"A4.3 session {session_date} eligible row count differs")
        if not part.empty:
            _normalize_loaded_part(part, session_date=session_date)
            parts.append(part)
            loaded_rows += len(part)
        if len(parts) >= SESSION_CONCAT_CHUNK_SIZE:
            chunks.append(pd.concat(parts, ignore_index=True))
            parts.clear()
        guard_memory(
            f"intraday A4.3 session {index + 1} load", peak=False, hard_budget_gib=MEMORY_HARD_BUDGET_GIB, headroom_gib=MEMORY_HEADROOM_GIB
        )

    if parts:
        chunks.append(pd.concat(parts, ignore_index=True))
        parts.clear()
    if loaded_rows != projected_rows or not chunks:
        raise DataReadinessError("loaded A4.3 eligible rows differ from the immutable dataset summary")

    frame = pd.concat(chunks, ignore_index=True)
    del chunks
    frame["dataset_row_id"] = frame["decision_id"].astype("string[pyarrow]")
    _validate_complete_frame(
        frame,
        strategy_contract_sha256=identity["strategy_contract_sha256"],
    )
    frozen_cost_bps = float(frame["cost"].iloc[0] * 10_000.0)
    release_process_memory()
    guard_memory("intraday A4.3 dataset load", peak=True, hard_budget_gib=MEMORY_HARD_BUDGET_GIB, headroom_gib=MEMORY_HEADROOM_GIB)

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
    request: dict[str, Any],
    manifest: Mapping[str, Any],
    authority: dict[str, Any],
) -> dict[str, str]:
    request_sha256 = require_string(request.get("request_sha256"), "request_sha256")
    request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
    manifest_sha256 = file_sha256(root / MANIFEST_NAME)
    authority_sha256 = file_sha256(root / AUTHORITY_NAME)
    transformation = require_object(request.get("transformation"), "request.transformation")
    transformation_sha256 = require_string(
        request.get("transformation_sha256"),
        "request.transformation_sha256",
    )
    raw_units = manifest.get("session_units")
    if not isinstance(raw_units, list):
        raise DataReadinessError("manifest.session_units must be an array")
    inventory_sha256 = json_sha256(raw_units)
    training_contract = require_object(manifest.get("training_contract"), "manifest.training_contract")
    ordered_names = require_string_tuple(
        training_contract.get("ordered_feature_names"),
        "manifest.training_contract.ordered_feature_names",
    )
    request_ordered_names = require_string_tuple(
        request.get("ordered_feature_names"),
        "request.ordered_feature_names",
    )
    strategy_contract_sha256 = require_string(
        request.get("strategy_contract_sha256"),
        "request.strategy_contract_sha256",
    )
    label_columns = set(
        require_string_tuple(
            training_contract.get("label_columns"),
            "manifest.training_contract.label_columns",
        )
    )
    required_labels = {
        "target_hit",
        "stop_hit",
        "entry_price",
        "stop_price",
        *RETURN_COLUMNS,
    }
    if (
        request.get("schema") != INTRADAY_BAR_DATASET_SCHEMA
        or manifest.get("schema") != INTRADAY_BAR_DATASET_SCHEMA
        or authority.get("schema") != INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != MANIFEST_NAME
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
        or training_contract.get("ordered_feature_sha256") != INTRADAY_BAR_MODEL_FEATURES_SHA256
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
    for column in RETURN_COLUMNS:
        values = pd.to_numeric(part[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"A4.3 economic column {column} must be finite")
        part[column] = values.astype("float64")
    for column in PRICE_COLUMNS:
        values = pd.to_numeric(part[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all() or values.le(0.0).any():
            raise DataReadinessError(f"A4.3 price column {column} must be finite and positive")
        part[column] = values.astype("float64")
    for column in TIMESTAMP_COLUMNS:
        values = pd.to_datetime(part[column], utc=True, errors="coerce")
        if values.isna().any():
            raise DataReadinessError(f"A4.3 timestamp {column} must be valid UTC")
        part[column] = values
    for column in BOOLEAN_COLUMNS:
        values = part[column].map(parse_strict_bool)
        if values.isna().any():
            raise DataReadinessError(f"A4.3 boolean column {column} is invalid")
        part[column] = values.astype(bool)
    for column in (*IDENTITY_COLUMNS, *ROW_CONTRACT_COLUMNS):
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
    for column in (*IDENTITY_COLUMNS, "dataset_row_id"):
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
            raise DataReadinessError(f"A4.3 {benchmark} excess return does not match the exact managed interval")

def _guard_projected_memory(rows: int) -> None:
    numeric_bytes = len(MODEL_FEATURE_COLUMNS) * 4 + (len(RETURN_COLUMNS) + len(PRICE_COLUMNS)) * 8
    timestamp_bytes = len(TIMESTAMP_COLUMNS) * 8
    boolean_bytes = len(BOOLEAN_COLUMNS)
    text_bytes = (len(IDENTITY_COLUMNS) + len(ROW_CONTRACT_COLUMNS) + 1) * 24
    projected_peak = rows * (numeric_bytes + timestamp_bytes + boolean_bytes + text_bytes) * 3
    safety_bytes = int((MEMORY_HARD_BUDGET_GIB - MEMORY_HEADROOM_GIB) * 1024**3)
    if projected_peak > safety_bytes:
        raise DataReadinessError("projected A4.3 loader memory exceeds the 3.25 GiB safety threshold")

def _session_rows_path(root: Path, session_date: str) -> Path:
    candidate = (root / "sessions" / f"session_date_et={session_date}" / "rows.parquet").resolve()
    if root not in candidate.parents or not candidate.is_file():
        raise DataReadinessError("A4.3 session rows path is invalid")
    return candidate
