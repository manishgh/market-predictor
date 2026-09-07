"""Strict runtime serving for promoted edge-rebuild models.

This module is deliberately independent of the legacy prediction service.  It
loads governance-verified model bundles, defines prediction outputs, and
enforces fail-closed batch/live feature parity.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from market_predictor.canonical.store import file_sha256
from market_predictor.core import path_integrity
from market_predictor.core.errors import (
    ArtifactIntegrityError,
    DataReadinessError,
    PromotionGateError,
    SchemaMismatchError,
)
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.governance.promotion import (
    bundle_contracts as _promotion_contracts,
)
from market_predictor.governance.promotion import (
    bundle_verification as _promotion_verification,
)
from market_predictor.modeling.feature_reference import (
    feature_reference_names_sha256,
    feature_reference_profile_sha256,
)
from market_predictor.modeling.strategy_contract import (
    StrategyContract,
)
from market_predictor.resources import assert_memory_budget, process_memory_snapshot
from market_predictor.swing.contracts.model_artifact import SWING_CANDIDATE_MODEL_SCHEMA

ACTIVE_GENERATION_SCHEMA: Final = "edge_rebuild.active_generation.v1"
ACTIVE_GENERATION_POINTER: Final = "active_generation.json"
GENERATION_DIRECTORY: Final = "generations"

_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FeatureParityReport(_FrozenModel):
    matched: Literal[True] = True
    row_count: int = Field(ge=1)
    feature_count: int = Field(ge=1)
    ordered_feature_sha256: str = Field(pattern=_SHA256_PATTERN)
    maximum_absolute_difference: float = Field(ge=0.0)
    maximum_relative_difference: float = Field(ge=0.0)
    relative_tolerance: float = Field(ge=0.0)
    absolute_tolerance: float = Field(ge=0.0)


@dataclass(frozen=True, slots=True)
class LoadedSwingModelGeneration:
    """One immutable, attested swing model generation held in memory."""

    generation_id: str
    pointer_sha256: str
    bundle: _promotion_contracts.PromotedSwingBundle
    model_payload: Mapping[str, object]


class SwingModelGenerationCache:
    """Load exactly one attested generation per repository and detect rollover."""

    def __init__(
        self,
        *,
        memory_budget_gib: float,
        memory_headroom_gib: float,
    ) -> None:
        self._memory_budget_gib = memory_budget_gib
        self._memory_headroom_gib = memory_headroom_gib
        self._lock = threading.Lock()
        self._loaded: dict[
            tuple[Path, str, Path, str, str, int],
            LoadedSwingModelGeneration,
        ] = {}

    def get(
        self,
        repository: Path,
        *,
        strategy_contract: StrategyContract,
        attestation_trust_store_path: Path,
        promotion_gate_policy_sha256: str,
        maximum_model_bytes: int,
        estimated_resident_gib: float,
    ) -> LoadedSwingModelGeneration:
        root = _promotion_verification.resolve_verified_bundle_root(repository)
        trust_store = path_integrity.verify_no_reparse_ancestry(
            attestation_trust_store_path,
            label="promotion attestation trust store",
        ).resolve(strict=True)
        if not trust_store.is_file():
            raise ArtifactIntegrityError("promotion attestation trust store is unavailable")
        cache_key = (
            root,
            strategy_contract.sha256(),
            trust_store,
            file_sha256(trust_store),
            promotion_gate_policy_sha256,
            maximum_model_bytes,
        )
        pointer = load_active_generation_pointer(root)
        cached = self._loaded.get(cache_key)
        if cached is not None and cached.pointer_sha256 == pointer["pointer_sha256"]:
            return cached
        with self._lock:
            pointer = load_active_generation_pointer(root)
            cached = self._loaded.get(cache_key)
            if cached is not None and cached.pointer_sha256 == pointer["pointer_sha256"]:
                return cached
            assert_memory_budget(
                hard_budget_gib=self._memory_budget_gib,
                headroom_gib=self._memory_headroom_gib,
                stage="before swing model generation load",
            )
            _assert_projected_rss(
                estimated_resident_gib,
                hard_budget_gib=self._memory_budget_gib,
                headroom_gib=self._memory_headroom_gib,
            )
            generation_root = _verified_generation_root(root, pointer["generation_id"])
            bundle_path = generation_root / "bundle.json"
            bundle_bytes = _read_verified_file_bytes(
                bundle_path,
                expected_sha256=pointer["bundle_file_sha256"],
                maximum_bytes=1024 * 1024,
                label="swing generation bundle",
            )
            try:
                raw = parse_strict_json_object(
                    bundle_bytes,
                    label="swing generation bundle",
                )
            except ValueError as exc:
                raise ArtifactIntegrityError("swing generation bundle is unreadable") from exc
            bundle = _promotion_verification.validate_file_backed_promoted_bundle(
                raw,
                bundle_root=generation_root,
                strategy_contract=strategy_contract,
                attestation_trust_store_path=attestation_trust_store_path,
                promotion_gate_policy_sha256=promotion_gate_policy_sha256,
                maximum_model_bytes=maximum_model_bytes,
                expected_mode="swing",
            )
            if bundle.sha256() != pointer["generation_id"]:
                raise ArtifactIntegrityError("active swing generation identity does not match its bundle")
            model_path = _promotion_verification.resolve_verified_bundle_artifact(
                generation_root,
                bundle.model_artifact_path,
                label="model",
            )
            payload = _load_joblib_from_verified_handle(
                model_path,
                expected_sha256=bundle.model_artifact_sha256,
                maximum_bytes=maximum_model_bytes,
            )
            if not isinstance(payload, Mapping):
                raise SchemaMismatchError("promoted swing model payload must be an object")
            _validate_swing_model_payload(payload, bundle)
            after = load_active_generation_pointer(root)
            if after["pointer_sha256"] != pointer["pointer_sha256"]:
                raise DataReadinessError("active swing model generation changed during verification")
            loaded = LoadedSwingModelGeneration(
                generation_id=pointer["generation_id"],
                pointer_sha256=pointer["pointer_sha256"],
                bundle=bundle,
                model_payload={str(key): value for key, value in payload.items()},
            )
            self._loaded[cache_key] = loaded
            assert_memory_budget(
                hard_budget_gib=self._memory_budget_gib,
                headroom_gib=self._memory_headroom_gib,
                stage="after swing model generation load",
            )
            return loaded

    def is_current(self, repository: Path, generation: LoadedSwingModelGeneration) -> bool:
        pointer = load_active_generation_pointer(_promotion_verification.resolve_verified_bundle_root(repository))
        return pointer["pointer_sha256"] == generation.pointer_sha256


def load_active_generation_pointer(root: Path) -> dict[str, str]:
    """Read and verify the single atomic pointer to an immutable generation."""

    path = _promotion_verification.resolve_verified_bundle_root(root) / ACTIVE_GENERATION_POINTER
    payload = _read_json_bytes(path, label="active generation pointer")
    expected_fields = {
        "schema",
        "generation_id",
        "bundle_file_sha256",
        "previous_generation_id",
        "activated_at_utc",
        "pointer_sha256",
    }
    if set(payload) != expected_fields or payload.get("schema") != ACTIVE_GENERATION_SCHEMA:
        raise ArtifactIntegrityError("active generation pointer schema is invalid")
    unsigned = dict(payload)
    pointer_sha = str(unsigned.pop("pointer_sha256", ""))
    if _promotion_contracts.canonical_payload_sha256(unsigned) != pointer_sha:
        raise ArtifactIntegrityError("active generation pointer hash is invalid")
    for field in ("generation_id", "bundle_file_sha256", "pointer_sha256"):
        value = str(payload.get(field, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ArtifactIntegrityError(f"active generation pointer {field} is invalid")
    _ = _strict_utc_datetime(payload.get("activated_at_utc"), "active generation activation")
    previous = payload.get("previous_generation_id")
    if previous is not None and (
        not isinstance(previous, str) or len(previous) != 64 or any(character not in "0123456789abcdef" for character in previous)
    ):
        raise ArtifactIntegrityError("active generation previous identity is invalid")
    return {str(key): str(value) if value is not None else "" for key, value in payload.items()}


def _verified_generation_root(root: Path, generation_id: str) -> Path:
    generations = root / GENERATION_DIRECTORY
    candidate = generations / generation_id
    try:
        generations_resolved = path_integrity.verify_no_reparse_ancestry(
            generations,
            label="swing generation directory",
        ).resolve(strict=True)
        resolved = path_integrity.verify_tree_containment(
            candidate,
            label="active swing generation",
        )
    except (DataReadinessError, FileNotFoundError) as exc:
        raise ArtifactIntegrityError("active swing generation is unavailable") from exc
    if not resolved.is_dir() or not resolved.is_relative_to(generations_resolved):
        raise ArtifactIntegrityError("active swing generation escapes its repository")
    return resolved


def _read_json_bytes(path: Path, *, label: str) -> dict[str, object]:
    raw = _read_verified_file_bytes(
        path,
        expected_sha256=None,
        maximum_bytes=1024 * 1024,
        label=label,
    )
    try:
        value = parse_strict_json_object(raw, label=label)
    except ValueError as exc:
        raise ArtifactIntegrityError(f"{label} is unreadable") from exc
    return value


def _read_verified_file_bytes(
    path: Path,
    *,
    expected_sha256: str | None,
    maximum_bytes: int,
    label: str,
) -> bytes:
    if path.is_symlink() or maximum_bytes < 1:
        raise ArtifactIntegrityError(f"{label} path or byte limit is invalid")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            size = int(before.st_size)
            if size < 1 or size > maximum_bytes:
                raise DataReadinessError(f"{label} byte limit exceeded")
            payload = handle.read(maximum_bytes + 1)
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError(f"{label} is unavailable") from exc
    if len(payload) != size or len(payload) > maximum_bytes:
        raise ArtifactIntegrityError(f"{label} changed while being read")
    observed = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise ArtifactIntegrityError(f"{label} SHA256 does not verify")
    return payload


def _load_joblib_from_verified_handle(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
) -> object:
    if path.is_symlink() or maximum_bytes < 1:
        raise ArtifactIntegrityError("promoted swing model path or byte limit is invalid")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            size = int(before.st_size)
            if size < 1 or size > maximum_bytes:
                raise DataReadinessError("promoted swing model byte limit exceeded")
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ArtifactIntegrityError("promoted swing model SHA256 does not verify")
            handle.seek(0)
            payload = joblib.load(handle)
            after = os.fstat(handle.fileno())
            if (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ArtifactIntegrityError("promoted swing model changed during deserialization")
            return payload
    except FileNotFoundError as exc:
        raise ArtifactIntegrityError("promoted swing model is unavailable") from exc
    except Exception as exc:
        if isinstance(exc, (ArtifactIntegrityError, DataReadinessError)):
            raise
        raise ArtifactIntegrityError("promoted swing model artifact is unreadable") from exc


def _validate_swing_model_payload(
    payload: Mapping[str, object],
    bundle: _promotion_contracts.PromotedSwingBundle,
) -> None:
    if payload.get("schema") != SWING_CANDIDATE_MODEL_SCHEMA:
        raise SchemaMismatchError("promoted swing model payload schema is unsupported")
    if payload.get("status") != "candidate" or payload.get("promotion_permitted") is not False:
        raise PromotionGateError("served model must remain an immutable candidate authorized by attestation")
    if payload.get("candidate_id") != bundle.model_id:
        raise ArtifactIntegrityError("promoted model candidate identity differs")
    if payload.get("model_family") != bundle.model_family:
        raise ArtifactIntegrityError("promoted model family differs from its bundle")
    if payload.get("strategy_contract_sha256") != bundle.strategy_contract_sha256:
        raise ArtifactIntegrityError("promoted model strategy contract binding differs")
    if payload.get("execution_policy_sha256") != bundle.execution_policy_sha256:
        raise ArtifactIntegrityError("promoted model execution policy binding differs")
    features = tuple(str(value) for value in _required_sequence(payload, "feature_columns"))
    if features != bundle.ordered_feature_columns:
        raise SchemaMismatchError("promoted model feature order differs from its bundle")
    feature_reference = payload.get("feature_reference_profile")
    if not isinstance(feature_reference, dict):
        raise SchemaMismatchError("promoted model feature reference is missing")
    if payload.get("feature_reference_profile_sha256") != (
        feature_reference_profile_sha256(feature_reference)
    ):
        raise ArtifactIntegrityError("promoted model feature reference identity differs")
    if payload.get("feature_reference_names_sha256") != (
        feature_reference_names_sha256(feature_reference)
    ):
        raise ArtifactIntegrityError("promoted model feature-name identity differs")
    if str(payload.get("ablation_profile", "")) != bundle.feature_profile:
        raise SchemaMismatchError("promoted model feature profile differs from its bundle")
    fitted_models = payload.get("fitted_models")
    if not isinstance(fitted_models, dict) or not fitted_models:
        raise SchemaMismatchError("promoted swing model is missing fitted models")
    if "classifier" not in fitted_models:
        raise SchemaMismatchError("promoted swing model is missing its classifier")
    thresholds = payload.get("probability_thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != set(fitted_models):
        raise SchemaMismatchError(
            "promoted swing model thresholds must identify every fitted model"
        )
    for name, threshold in thresholds.items():
        _finite_probability(threshold, f"{name} probability threshold")
    expected_positions = {column: index for index, column in enumerate(features)}
    for name, model in fitted_models.items():
        if getattr(model, "estimator", None) is None or getattr(model, "calibrator", None) is None:
            raise SchemaMismatchError(f"promoted swing model {name} is missing estimator or calibrator")
        model_columns = tuple(str(column) for column in getattr(model, "feature_columns", ()))
        if not model_columns or len(model_columns) != len(set(model_columns)):
            raise SchemaMismatchError(f"promoted swing model {name} has an invalid feature subset")
        if any(column not in expected_positions for column in model_columns):
            raise SchemaMismatchError(f"promoted swing model {name} feature subset is outside its bundle")
        if tuple(sorted(model_columns, key=expected_positions.__getitem__)) != model_columns:
            raise SchemaMismatchError(f"promoted swing model {name} feature subset order is invalid")


def _strict_utc_datetime(value: object, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ArtifactIntegrityError(f"{label} is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactIntegrityError(f"{label} must be timezone-aware")
    return parsed.astimezone(UTC)


def _assert_projected_rss(
    estimated_resident_gib: float,
    *,
    hard_budget_gib: float,
    headroom_gib: float,
) -> None:
    if estimated_resident_gib <= 0:
        raise DataReadinessError("swing generation resident-memory estimate is invalid")
    snapshot = process_memory_snapshot()
    if snapshot is None:
        return
    current_gib = snapshot[0] / 1024**3
    if current_gib + estimated_resident_gib > hard_budget_gib - headroom_gib:
        raise DataReadinessError("swing generation load would exceed the configured RSS safety threshold")


def validate_ordered_feature_frame(
    frame: pd.DataFrame,
    ordered_feature_columns: Sequence[str],
    *,
    frame_name: str,
) -> np.ndarray:
    """Validate an inference frame and return its finite float64 matrix."""

    expected = tuple(ordered_feature_columns)
    if not expected or len(expected) != len(set(expected)):
        raise SchemaMismatchError("expected feature contract must be non-empty and unique")
    if frame.empty:
        raise DataReadinessError(f"{frame_name} inference feature frame is empty")
    if not frame.index.is_unique:
        raise SchemaMismatchError(f"{frame_name} feature row index must be unique")
    observed = tuple(str(column) for column in frame.columns)
    if observed != expected:
        raise SchemaMismatchError(
            f"{frame_name} feature columns differ from promoted order; expected={list(expected)!r}, observed={list(observed)!r}"
        )
    invalid_types = [column for column in expected if is_bool_dtype(frame[column].dtype) or not is_numeric_dtype(frame[column].dtype)]
    if invalid_types:
        raise SchemaMismatchError(f"{frame_name} contains non-numeric model features: {invalid_types}")
    values = frame.loc[:, expected].to_numpy(dtype="float64", copy=False)
    if not bool(np.isfinite(values).all()):
        locations = np.argwhere(~np.isfinite(values))
        row, column = (int(value) for value in locations[0])
        raise DataReadinessError(f"{frame_name} contains a non-finite feature at row {row}, column {expected[column]}")
    return cast(np.ndarray[Any, np.dtype[np.float64]], values)


class SwingInferenceEngine:
    def __init__(self, generation: LoadedSwingModelGeneration):
        self.generation = generation
        self.bundle = generation.bundle
        self.payload = generation.model_payload
        _validate_swing_model_payload(self.payload, self.bundle)
        self.fitted_models = cast(dict[str, Any], self.payload.get("fitted_models") or {})
        thresholds = cast(dict[str, Any], self.payload["probability_thresholds"])
        self.threshold = _finite_probability(
            thresholds["classifier"],
            "probability_threshold",
        )

    def predict(
        self,
        feature_frame: pd.DataFrame,
        requested_models: list[str] | None = None,
    ) -> dict[str, tuple[float, ...]]:
        if requested_models:
            requested = tuple(requested_models)
            allowed = {"all", "classifier", "xgboost_regressor"}
            if (
                len(requested) != len(set(requested))
                or not set(requested).issubset(allowed)
                or ("all" in requested and len(requested) != 1)
                or ("all" not in requested and "classifier" not in requested)
            ):
                raise DataReadinessError("swing scoring requires one supported request containing the classifier")
        matrix = validate_ordered_feature_frame(
            feature_frame,
            self.bundle.ordered_feature_columns,
            frame_name="promoted swing",
        ).astype("float32", copy=False)
        feature_positions = {column: index for index, column in enumerate(self.bundle.ordered_feature_columns)}

        def _score_model(fitted: object) -> tuple[float, ...] | None:
            if fitted is None:
                return None
            estimator = getattr(fitted, "estimator", None)
            calibrator = getattr(fitted, "calibrator", None)
            if estimator is None or calibrator is None:
                return None
            model_columns = tuple(str(column) for column in getattr(fitted, "feature_columns", ()))
            if not model_columns:
                raise DataReadinessError("promoted swing model is missing its ordered feature subset")
            try:
                model_matrix = matrix[:, [feature_positions[column] for column in model_columns]]
            except KeyError as exc:
                raise DataReadinessError("promoted swing model feature subset is outside its bundle") from exc
            try:
                raw = np.asarray(estimator.predict_proba(model_matrix), dtype="float64")
                if raw.ndim != 2 or raw.shape != (len(model_matrix), 2):
                    raise ValueError("estimator probability shape is invalid")
                calibrated = np.asarray(
                    calibrator.predict_proba(raw[:, 1].reshape(-1, 1))[:, 1],
                    dtype="float64",
                )
            except (AttributeError, TypeError, ValueError) as exc:
                raise DataReadinessError("promoted swing model could not score live features") from exc
            if calibrated.shape != (len(model_matrix),) or not np.isfinite(calibrated).all():
                raise DataReadinessError("promoted swing model produced invalid probabilities")
            if bool(((calibrated < 0.0) | (calibrated > 1.0)).any()):
                raise DataReadinessError("promoted swing probabilities are outside [0, 1]")
            return tuple(float(value) for value in calibrated)

        if not requested_models or "all" in requested_models:
            # "dualhurdle" was dropped from the candidate grid; naming a family
            # that can never be fitted would silently score nothing.
            models_to_score = ["classifier", "xgboost_regressor"]
        else:
            models_to_score = [m for m in requested_models if m in self.fitted_models]

        results: dict[str, tuple[float, ...]] = {}
        for model_name in models_to_score:
            scores = _score_model(self.fitted_models.get(model_name))
            if scores is not None:
                results[model_name] = scores
        return results


def _required_sequence(payload: Mapping[str, object], field: str) -> Sequence[object]:
    value = payload.get(field)
    if not isinstance(value, (list, tuple)) or not value:
        raise SchemaMismatchError(f"promoted swing model {field} is invalid")
    return value


def _finite_probability(value: object, field: str) -> float:
    try:
        numeric = float(cast(Any, value))
    except (TypeError, ValueError) as exc:
        raise SchemaMismatchError(f"promoted swing model {field} is invalid") from exc
    if not np.isfinite(numeric) or numeric <= 0.0 or numeric >= 1.0:
        raise SchemaMismatchError(f"promoted swing model {field} is invalid")
    return numeric


def validate_batch_live_feature_parity(
    batch: pd.DataFrame,
    live: pd.DataFrame,
    ordered_feature_columns: Sequence[str],
    *,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
) -> FeatureParityReport:
    """Fail when live inference features diverge from the batch implementation."""

    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("feature parity tolerances cannot be negative")
    expected = tuple(ordered_feature_columns)
    batch_values = validate_ordered_feature_frame(
        batch,
        expected,
        frame_name="batch",
    )
    live_values = validate_ordered_feature_frame(
        live,
        expected,
        frame_name="live",
    )
    if batch_values.shape != live_values.shape:
        raise SchemaMismatchError(f"batch/live feature shapes differ; batch={batch_values.shape}, live={live_values.shape}")
    if not batch.index.equals(live.index):
        raise SchemaMismatchError("batch/live feature row identities or order differ")
    absolute_difference = np.abs(batch_values - live_values)
    denominator = np.maximum(
        np.maximum(np.abs(batch_values), np.abs(live_values)),
        np.finfo(np.float64).tiny,
    )
    relative_difference = absolute_difference / denominator
    close = np.isclose(
        batch_values,
        live_values,
        rtol=relative_tolerance,
        atol=absolute_tolerance,
    )
    if not bool(close.all()):
        row, column = (int(value) for value in np.argwhere(~close)[0])
        raise DataReadinessError(
            "batch/live feature parity failed at "
            f"row {row}, column {expected[column]}; "
            f"batch={batch_values[row, column]!r}, "
            f"live={live_values[row, column]!r}"
        )
    return FeatureParityReport(
        row_count=int(batch_values.shape[0]),
        feature_count=int(batch_values.shape[1]),
        ordered_feature_sha256=_promotion_contracts.ordered_values_sha256(expected),
        maximum_absolute_difference=(float(absolute_difference.max()) if absolute_difference.size else 0.0),
        maximum_relative_difference=(float(relative_difference.max()) if relative_difference.size else 0.0),
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
    )
