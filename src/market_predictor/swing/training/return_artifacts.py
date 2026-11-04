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
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.swing.training.return_estimators import FittedReturnRegressor

SCHEMA = "market_predictor.swing_return_research_model"
TARGET = "spy_fixed_horizon_excess_return"
FAMILIES = ("regularized_linear_return", "shallow_boosted_return")
SCOPES = ("temporal", "security_transfer")
FLAGS = ("serving_eligible", "promotion_eligible", "portfolio_evaluated", "outer_validation_opened",
    "historical_test_opened")


def _verified_bytes(path: Path, digest: str, *, limit: int = 64 * 1024**2) -> bytes:
    if path.stat().st_size > limit:
        raise DataReadinessError("return artifact exceeds bounded reader size")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise DataReadinessError("return artifact exceeds bounded reader size")
    if hashlib.sha256(content).hexdigest() != digest:
        raise DataReadinessError("return research artifact changed from its pinned bytes")
    return content


def verify_completed_return_run(directory: Path, *, manifest_sha256: str,
                                maximum_metadata_bytes: int = 8 * 1024**2) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify complete initial-fit metadata; callers separately verify consumed units.

    No current source files are repinned and no model is loaded or admitted.
    Reader limits remain caller-owned so HTTP and offline consumers keep their bounds.
    """
    manifest: dict[str, Any] = parse_strict_json_object(
        _verified_bytes(directory / "_manifest.json", manifest_sha256, limit=maximum_metadata_bytes), label="return run manifest")
    request: dict[str, Any] = parse_strict_json_object(
        _verified_bytes(directory / "_request.json", manifest["request_sha256"], limit=maximum_metadata_bytes), label="return run request")
    if (manifest.get("schema") != "market_predictor.swing_return_training"
            or manifest.get("status") != "complete_research_only"
            or request.get("schema") != "market_predictor.swing_return_training_request"
            or any(record.get(flag) is not False for record in (manifest, request) for flag in FLAGS)):
        raise DataReadinessError("completed return run requires initial-fit development evidence without promotion")
    expected = {f"{family}/{scope}/fold-{fold}" for family in FAMILIES for scope in SCOPES for fold in range(1, 5)}
    expected |= {f"{family}/final_refit" for family in FAMILIES}
    if (set(manifest["units"]) != expected or request["policy"]["target"] != TARGET
            or request["policy"]["scope"] != "initial_fit_research_only"
            or [fold["number"] for fold in request["folds"]] != [1, 2, 3, 4]):
        raise DataReadinessError("completed return run target, scope or fold inventory differs")
    return manifest, request


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
