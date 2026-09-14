"""Atomic, hash-verified, research-only estimator and prediction artifacts."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.swing.training.return_estimators import FittedReturnRegressor

SCHEMA = "market_predictor.swing_return_research_model"


def _verified_bytes(path: Path, digest: str) -> bytes:
    if path.stat().st_size > 64 * 1024**2:
        raise DataReadinessError("return artifact exceeds bounded reader size")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != digest:
        raise DataReadinessError("return research artifact changed from its pinned bytes")
    return content


def verify_unit(directory: Path, *, manifest_sha256: str, request_sha256: str, unit: dict[str, Any]) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(_verified_bytes(directory / "_manifest.json", manifest_sha256))
    if (manifest.get("schema") != SCHEMA or manifest.get("request_sha256") != request_sha256
            or manifest.get("unit") != unit or manifest.get("serving_eligible") is not False
            or manifest.get("promotion_eligible") is not False):
        raise DataReadinessError("return research unit scope or request differs")
    expected = {"model.joblib", "predictions.parquet"} if unit["scope"] != "final_refit" else {"model.joblib"}
    if set(manifest["files"]) != expected:
        raise DataReadinessError("return research unit file inventory differs")
    for name, digest in manifest["files"].items():
        if file_sha256(inside(directory, name)) != digest:
            raise DataReadinessError("return research unit artifact changed")
    return manifest


def publish_unit(directory: Path, *, request_sha256: str, unit: dict[str, Any], model: FittedReturnRegressor,
    predictions: pd.DataFrame | None, metrics: dict[str, object], guard: Callable[[], None]) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=f".{directory.name}-") as temporary:
        stage = Path(temporary)
        joblib.dump(model, stage / "model.joblib", compress=3)
        files = {"model.joblib": file_sha256(stage / "model.joblib")}
        if predictions is not None:
            predictions.to_parquet(stage / "predictions.parquet", index=False)
            files["predictions.parquet"] = file_sha256(stage / "predictions.parquet")
        manifest = {"schema": SCHEMA, "request_sha256": request_sha256, "unit": unit, "files": files,
            "metrics": metrics, "serving_eligible": False, "promotion_eligible": False}
        write_json_object(stage / "_manifest.json", manifest)
        verify_unit(stage, manifest_sha256=file_sha256(stage / "_manifest.json"), request_sha256=request_sha256, unit=unit)
        guard()
        os.replace(stage, directory)
    return manifest


def load_return_model(directory: Path, *, manifest_sha256: str, purpose: str = "research") -> FittedReturnRegressor:
    if purpose != "research":
        raise DataReadinessError("research return models cannot authorize serving or promotion")
    manifest: dict[str, Any] = json.loads(_verified_bytes(directory / "_manifest.json", manifest_sha256))
    verified = verify_unit(directory, manifest_sha256=manifest_sha256,
        request_sha256=manifest["request_sha256"], unit=manifest["unit"])
    content = _verified_bytes(directory / "model.joblib", verified["files"]["model.joblib"])
    model = joblib.load(io.BytesIO(content))
    if (not isinstance(model, FittedReturnRegressor) or model.family != verified["unit"]["family"]
            or list(model.feature_names) != verified["unit"]["feature_names"]
            or model.parameters != verified["unit"]["parameters"]):
        raise DataReadinessError("return model identity differs from manifest")
    return model
