from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from market_predictor.canonical.store import file_sha256
import market_predictor.intraday.datasets.bar_dataset as intraday_bar_dataset
from market_predictor.intraday.datasets.bar_dataset import (
    INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA,
    INTRADAY_BAR_DATASET_SCHEMA,
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
from market_predictor.intraday.training.training import (
    MODEL_FEATURE_COLUMNS,
    load_published_intraday_dataset,
)
from market_predictor.core.errors import DataReadinessError


def test_loads_real_shape_a43_units_with_exact_projected_contract(tmp_path: Path) -> None:
    frame = _training_frame(session_count=3, security_count=4)
    omitted_decision_id = str(frame.loc[0, "decision_id"])
    frame.loc[0, "dataset_eligible"] = False
    authority = _publish_dataset(
        tmp_path / "dataset",
        frame,
    )

    published = load_published_intraday_dataset(authority)

    assert MODEL_FEATURE_COLUMNS is INTRADAY_BAR_MODEL_FEATURE_COLUMNS
    assert published.feature_columns == INTRADAY_BAR_MODEL_FEATURE_COLUMNS
    assert published.ordered_feature_sha256 == INTRADAY_BAR_MODEL_FEATURES_SHA256
    assert len(published.frame) == 23
    assert omitted_decision_id not in set(published.frame["decision_id"].astype(str))
    assert published.frame["dataset_row_id"].astype(str).equals(
        published.frame["decision_id"].astype(str)
    )
    assert published.frame["dataset_row_id"].is_unique
    assert {
        "gross_return",
        "cost",
        "net_return",
        "entry_price",
        "stop_price",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "target_hit",
        "stop_hit",
        "decision_time_utc",
        "feature_available_at_utc",
        "entry_time_utc",
        "entry_bar_end_utc",
        "exit_time_utc",
        "exit_bar_end_utc",
        "label_available_at_utc",
    }.issubset(published.frame.columns)
    assert published.frame["entry_price"].dtype == np.dtype("float64")
    assert published.frame["stop_price"].dtype == np.dtype("float64")
    assert published.frame["stop_price"].lt(published.frame["entry_price"]).all()
    for benchmark in ("spy", "qqq", "sector"):
        np.testing.assert_allclose(
            published.frame["net_return"] - published.frame[f"{benchmark}_return"],
            published.frame[f"{benchmark}_excess_return"],
            rtol=0.0,
            atol=1e-10,
        )


def test_features_are_exact_order_finite_float32(tmp_path: Path) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=2, security_count=4),
    )

    published = load_published_intraday_dataset(authority)

    features = published.frame.loc[:, MODEL_FEATURE_COLUMNS]
    assert tuple(features.columns) == INTRADAY_BAR_MODEL_FEATURE_COLUMNS
    assert all(dtype == np.dtype("float32") for dtype in features.dtypes)
    assert np.isfinite(features.to_numpy(dtype="float32")).all()


@pytest.mark.parametrize("tamper", ["authority", "request", "transformation", "session_inventory", "unit_rows"])
def test_rejects_a43_authority_tamper(tmp_path: Path, tamper: str) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=2, security_count=4),
    )
    authority_path = authority / "_authority.json"
    request_path = authority / "_request.json"
    manifest_path = authority / "_manifest.json"
    authority_record = _read_json(authority_path)
    request = _read_json(request_path)
    manifest = _read_json(manifest_path)
    if tamper == "authority":
        authority_record["artifact_sha256"] = "0" * 64
        _write_json(authority_path, authority_record)
    elif tamper == "request":
        request["request_sha256"] = "1" * 64
        _write_json(request_path, request)
    elif tamper == "transformation":
        request["transformation_sha256"] = "2" * 64
        _write_json(request_path, request)
    elif tamper == "session_inventory":
        manifest["session_unit_inventory_sha256"] = "3" * 64
        _write_json(manifest_path, manifest)
        authority_record["artifact_sha256"] = file_sha256(manifest_path)
        _write_json(authority_path, authority_record)
    else:
        session_date = str(manifest["session_units"][0]["session_date_et"])
        rows_path = _rows_path(authority, session_date)
        rows = pd.read_parquet(rows_path)
        rows.loc[0, MODEL_FEATURE_COLUMNS[0]] = 999.0
        rows.to_parquet(rows_path, index=False, compression="zstd")

    with pytest.raises(DataReadinessError):
        load_published_intraday_dataset(authority)


def test_rejects_obsolete_intraday_dataset_schema(tmp_path: Path) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=1, security_count=4),
    )
    request_path = authority / "_request.json"
    request = _read_json(request_path)
    request["schema"] = "edge_rebuild.intraday_dataset.v2"
    _write_json(request_path, request)

    with pytest.raises(DataReadinessError):
        load_published_intraday_dataset(authority)


@pytest.mark.parametrize("location", ["request", "training_contract", "row"])
def test_rejects_ordered_feature_order_or_hash_drift(tmp_path: Path, location: str) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=1, security_count=4),
    )
    if location == "request":
        request_path = authority / "_request.json"
        request = _read_json(request_path)
        request["ordered_feature_names"] = list(reversed(MODEL_FEATURE_COLUMNS))
        _write_json(request_path, request)
    elif location == "training_contract":
        manifest_path = authority / "_manifest.json"
        manifest = _read_json(manifest_path)
        manifest["training_contract"]["ordered_feature_names"] = list(
            reversed(MODEL_FEATURE_COLUMNS)
        )
        _write_json(manifest_path, manifest)
        _rebind_manifest_authority(authority)
    else:
        manifest = _read_json(authority / "_manifest.json")
        session_date = str(manifest["session_units"][0]["session_date_et"])
        rows_path = _rows_path(authority, session_date)
        rows = pd.read_parquet(rows_path)
        rows["ordered_feature_sha256"] = "f" * 64
        rows.to_parquet(rows_path, index=False, compression="zstd")
        _rebind_session_and_authority(authority, session_date)

    with pytest.raises(DataReadinessError):
        load_published_intraday_dataset(authority)


@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_rejects_non_finite_features_after_full_unit_rebind(
    tmp_path: Path,
    invalid: float,
) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=1, security_count=4),
    )
    session_date = str(_read_json(authority / "_manifest.json")["session_units"][0]["session_date_et"])
    rows_path = _rows_path(authority, session_date)
    rows = pd.read_parquet(rows_path)
    rows.loc[0, MODEL_FEATURE_COLUMNS[0]] = invalid
    rows.to_parquet(rows_path, index=False, compression="zstd")
    _rebind_session_and_authority(authority, session_date)

    with pytest.raises(DataReadinessError, match="must be finite"):
        load_published_intraday_dataset(authority)


def test_identity_hashes_are_exact_and_dataclass_is_frozen(tmp_path: Path) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=2, security_count=4),
    )

    first = load_published_intraday_dataset(authority)
    second = load_published_intraday_dataset(authority)
    manifest = _read_json(authority / "_manifest.json")
    request = _read_json(authority / "_request.json")

    assert first.dataset_sha256 == first.manifest_sha256 == file_sha256(authority / "_manifest.json")
    assert first.authority_sha256 == file_sha256(authority / "_authority.json")
    assert first.request_sha256 == request["request_sha256"]
    assert first.transformation_sha256 == request["transformation_sha256"]
    assert first.session_unit_inventory_sha256 == manifest["session_unit_inventory_sha256"]
    assert first.strategy_contract_sha256 == request["strategy_contract_sha256"]
    assert first.manifest_sha256 == second.manifest_sha256
    with pytest.raises(FrozenInstanceError):
        first.dataset_sha256 = "0" * 64  # type: ignore[misc]


def test_rejects_duplicate_decision_identity_after_full_unit_rebind(tmp_path: Path) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=1, security_count=4),
    )
    session_date = str(_read_json(authority / "_manifest.json")["session_units"][0]["session_date_et"])
    rows_path = _rows_path(authority, session_date)
    rows = pd.read_parquet(rows_path)
    rows.loc[1, "decision_id"] = rows.loc[0, "decision_id"]
    rows.to_parquet(rows_path, index=False, compression="zstd")
    _rebind_session_and_authority(authority, session_date)

    with pytest.raises(DataReadinessError, match="decision_id must be globally unique"):
        load_published_intraday_dataset(authority)


def test_rejects_inexact_raw_benchmark_excess_after_full_unit_rebind(tmp_path: Path) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=1, security_count=4),
    )
    session_date = str(_read_json(authority / "_manifest.json")["session_units"][0]["session_date_et"])
    rows_path = _rows_path(authority, session_date)
    rows = pd.read_parquet(rows_path)
    rows.loc[0, "spy_return"] += 0.001
    rows.to_parquet(rows_path, index=False, compression="zstd")
    _rebind_session_and_authority(authority, session_date)

    with pytest.raises(DataReadinessError, match="spy excess return"):
        load_published_intraday_dataset(authority)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("entry_price", 0.0, "finite and positive"),
        ("stop_price", 200.0, "stop price must be below"),
    ],
)
def test_rejects_invalid_entry_or_stop_price_after_full_unit_rebind(
    tmp_path: Path,
    column: str,
    value: float,
    message: str,
) -> None:
    authority = _publish_dataset(
        tmp_path / "dataset",
        _training_frame(session_count=1, security_count=4),
    )
    session_date = str(_read_json(authority / "_manifest.json")["session_units"][0]["session_date_et"])
    rows_path = _rows_path(authority, session_date)
    rows = pd.read_parquet(rows_path)
    rows.loc[0, column] = value
    rows.to_parquet(rows_path, index=False, compression="zstd")
    _rebind_session_and_authority(authority, session_date)

    with pytest.raises(DataReadinessError, match=message):
        load_published_intraday_dataset(authority)


def _training_frame(
    *,
    session_count: int = 80,
    security_count: int = 8,
) -> pd.DataFrame:
    rng = np.random.default_rng(19)
    sessions = pd.bdate_range("2024-01-02", periods=session_count)
    strategy_contract_sha256 = "b" * 64
    rows: list[dict[str, Any]] = []
    for session_index, session in enumerate(sessions):
        market = 0.4 * math.sin(session_index / 6.0)
        for decision_index, offset_minutes in enumerate((0, 120)):
            decision = pd.Timestamp(session, tz="America/New_York") + pd.Timedelta(
                hours=11,
                minutes=offset_minutes,
            )
            decision = decision.tz_convert("UTC")
            group_id = decision.isoformat()
            cohort_id = json_sha256(
                {
                    "session_date_et": session.date().isoformat(),
                    "decision_time_utc": group_id,
                    "strategy_contract_sha256": strategy_contract_sha256,
                }
            )
            for security_index in range(security_count):
                feature_1 = float(rng.normal() + 0.3 * market + decision_index * 0.05)
                feature_2 = float(rng.normal() - 0.2 * market)
                raw = 1.1 * feature_1 - 0.65 * feature_2
                probability = 1.0 / (1.0 + math.exp(-raw))
                target_hit = bool(
                    (security_index + session_index + decision_index) % security_count
                    < round(security_count * probability)
                )
                stop_hit = not target_hit and (security_index + session_index) % 3 == 0
                gross = (0.003 if target_hit else -0.0015) + float(
                    rng.normal(0.0, 0.0002)
                )
                cost = 0.001
                net = gross - cost
                spy_return = 0.0002
                qqq_return = 0.0003
                sector_return = 0.0001
                model_features = {
                    column: (
                        feature_1 + (index % 3 - 1) * 0.05
                        if index % 2 == 0
                        else feature_2 + (index % 5 - 2) * 0.03
                    )
                    for index, column in enumerate(MODEL_FEATURE_COLUMNS)
                }
                security_id = f"SEC{security_index:02d}"
                decision_id = json_sha256(
                    {
                        "security_id": security_id,
                        "decision_time_utc": group_id,
                        "strategy_contract_sha256": strategy_contract_sha256,
                    }
                )
                entry_time = decision + pd.Timedelta(minutes=1)
                exit_time = decision + pd.Timedelta(minutes=30)
                entry_price = 100.0 + security_index
                rows.append(
                    {
                        "decision_id": decision_id,
                        "decision_cohort_id": cohort_id,
                        "decision_group_id": group_id,
                        "ticker": f"T{security_index:02d}",
                        "security_id": security_id,
                        "session_date_et": session.date(),
                        "sector": (
                            "Information Technology"
                            if security_index % 2 == 0
                            else "Health Care"
                        ),
                        "primary_benchmark": "XLK" if security_index % 2 == 0 else "XLV",
                        "universe_snapshot_id": f"snapshot-{session.date().isoformat()}",
                        "strategy_contract_sha256": strategy_contract_sha256,
                        "decision_time_utc": decision,
                        "feature_available_at_utc": decision,
                        "entry_time_utc": entry_time,
                        "entry_bar_end_utc": entry_time + pd.Timedelta(minutes=1),
                        "exit_time_utc": exit_time,
                        "exit_bar_end_utc": exit_time + pd.Timedelta(minutes=1),
                        "label_available_at_utc": exit_time + pd.Timedelta(minutes=1),
                        "dataset_eligible": True,
                        "feature_eligible": True,
                        "label_eligible": True,
                        "target_hit": target_hit,
                        "stop_hit": stop_hit,
                        "gross_return": gross,
                        "cost": cost,
                        "net_return": net,
                        "entry_price": entry_price,
                        "stop_price": entry_price * 0.985,
                        "spy_return": spy_return,
                        "qqq_return": qqq_return,
                        "sector_return": sector_return,
                        "spy_excess_return": net - spy_return,
                        "qqq_excess_return": net - qqq_return,
                        "sector_excess_return": net - sector_return,
                        "feature_schema_version": INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
                        "label_schema_version": INTRADAY_BAR_LABEL_SCHEMA_VERSION,
                        "ordered_feature_sha256": INTRADAY_BAR_MODEL_FEATURES_SHA256,
                        **model_features,
                    }
                )
    return pd.DataFrame(rows)


def _publish_dataset(directory: Path, frame: pd.DataFrame) -> Path:
    directory.mkdir(parents=True)
    prepared = frame.copy()
    prepared["session_date_et"] = pd.to_datetime(
        prepared["session_date_et"], errors="raise"
    ).dt.date
    sessions = sorted(prepared["session_date_et"].astype(str).unique())
    transformation = intraday_bar_dataset._transformation_identity()
    request_payload = {
        "schema": INTRADAY_BAR_DATASET_SCHEMA,
        "strategy_contract_sha256": str(prepared["strategy_contract_sha256"].iloc[0]),
        "feature_schema_version": INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
        "ordered_feature_names": list(INTRADAY_BAR_MODEL_FEATURE_COLUMNS),
        "ordered_feature_sha256": INTRADAY_BAR_MODEL_FEATURES_SHA256,
        "label_schema_version": INTRADAY_BAR_LABEL_SCHEMA_VERSION,
        "transformation": transformation,
        "transformation_sha256": transformation["sha256"],
        "processing_unit": "one_exchange_session",
        "decision_clock": "fixed_five_minute_cohort_after_activation",
        "planned_sessions": sessions,
    }
    request_sha256 = json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    _write_json(directory / "_request.json", request)

    unit_records: list[dict[str, Any]] = []
    for session_date, session_rows in prepared.groupby(
        prepared["session_date_et"].astype(str),
        sort=True,
        observed=True,
    ):
        unit = directory / "sessions" / f"session_date_et={session_date}"
        unit.mkdir(parents=True)
        rows_path = unit / "rows.parquet"
        session_rows.to_parquet(rows_path, index=False, compression="zstd")
        audit = {"session_date_et": str(session_date), "stock_sessions": []}
        _write_json(unit / "audit.json", audit)
        unit_record = {
            "schema": "edge_rebuild.intraday_bar_dataset_session_unit.v1",
            "state": "complete",
            "session_date_et": str(session_date),
            "request_sha256": request_sha256,
            "transformation_sha256": transformation["sha256"],
            "session_request_sha256": json_sha256(
                {"request_sha256": request_sha256, "session_date_et": str(session_date)}
            ),
            "rows": len(session_rows),
            "dataset_eligible_rows": int(session_rows["dataset_eligible"].sum()),
            "ticker_count": int(session_rows["ticker"].nunique()),
            "rows_sha256": file_sha256(rows_path),
            "audit_sha256": file_sha256(unit / "audit.json"),
            "parquet_schema": _arrow_schema_record(pq.read_schema(rows_path)),
            "parquet_schema_sha256": json_sha256(
                _arrow_schema_record(pq.read_schema(rows_path))
            ),
        }
        _write_json(unit / "_unit.json", unit_record)
        unit_records.append(unit_record)

    manifest = {
        **request,
        "state": "complete",
        "session_units": unit_records,
        "session_unit_inventory_sha256": json_sha256(unit_records),
        "summary": {
            "planned_sessions": len(sessions),
            "completed_sessions": len(unit_records),
            "rows": len(prepared),
            "dataset_eligible_rows": int(prepared["dataset_eligible"].sum()),
        },
        "training_contract": {
            "eligibility_column": "dataset_eligible",
            "ordered_feature_names": list(INTRADAY_BAR_MODEL_FEATURE_COLUMNS),
            "ordered_feature_sha256": INTRADAY_BAR_MODEL_FEATURES_SHA256,
            "label_columns": [
                "target_hit",
                "stop_hit",
                "entry_price",
                "stop_price",
                "gross_return",
                "cost",
                "net_return",
                "spy_return",
                "qqq_return",
                "sector_return",
                "spy_excess_return",
                "qqq_excess_return",
                "sector_excess_return",
            ],
        },
    }
    _write_json(directory / "_manifest.json", manifest)
    authority = {
        "schema": INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA,
        "state": "complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(directory / "_manifest.json"),
        "request_sha256": request_sha256,
        "session_unit_inventory_sha256": manifest["session_unit_inventory_sha256"],
        "sessions": len(unit_records),
        "rows": len(prepared),
    }
    _write_json(directory / "_authority.json", authority)
    return directory


def _rebind_session_and_authority(directory: Path, session_date: str) -> None:
    rows_path = _rows_path(directory, session_date)
    unit_path = rows_path.parent / "_unit.json"
    unit = _read_json(unit_path)
    rows = pd.read_parquet(rows_path)
    schema = _arrow_schema_record(pq.read_schema(rows_path))
    unit.update(
        {
            "rows_sha256": file_sha256(rows_path),
            "rows": len(rows),
            "dataset_eligible_rows": int(rows["dataset_eligible"].sum()),
            "ticker_count": int(rows["ticker"].nunique()),
            "parquet_schema": schema,
            "parquet_schema_sha256": json_sha256(schema),
        }
    )
    _write_json(unit_path, unit)
    manifest_path = directory / "_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["session_units"] = [
        unit if str(record["session_date_et"]) == session_date else record
        for record in manifest["session_units"]
    ]
    manifest["session_unit_inventory_sha256"] = json_sha256(manifest["session_units"])
    manifest["summary"]["rows"] = sum(int(record["rows"]) for record in manifest["session_units"])
    manifest["summary"]["dataset_eligible_rows"] = sum(
        int(record["dataset_eligible_rows"]) for record in manifest["session_units"]
    )
    _write_json(manifest_path, manifest)
    _rebind_manifest_authority(directory)


def _rebind_manifest_authority(directory: Path) -> None:
    manifest = _read_json(directory / "_manifest.json")
    authority_path = directory / "_authority.json"
    authority = _read_json(authority_path)
    authority["artifact_sha256"] = file_sha256(directory / "_manifest.json")
    authority["session_unit_inventory_sha256"] = manifest[
        "session_unit_inventory_sha256"
    ]
    _write_json(authority_path, authority)


def _rows_path(directory: Path, session_date: str) -> Path:
    return directory / "sessions" / f"session_date_et={session_date}" / "rows.parquet"


def _arrow_schema_record(schema: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": str(field.name),
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
