"""Read-only verification of historical model/feature bundle artifacts.

These recorded bundles use serving_bundles/ and active_serving_bundle.json.
Current inference consumes active_generation.json through swing_inference.py;
this module cannot publish, activate or roll back either format.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_predictor.core import path_integrity
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.feature_store import (
    LIVE_FEATURE_SCHEMA,
    LiveFeatureStore,
    LiveFeatureStoreConfig,
)
from market_predictor.live_features import LIVE_ARTIFACT_TYPES, LIVE_SCHEMA_VERSIONS, LiveMode
from market_predictor.registry import file_sha256
from market_predictor.release import verify_local_release

SERVING_BUNDLE_SCHEMA = "market_predictor.serving.bundle"
ACTIVE_SERVING_BUNDLE_SCHEMA = "market_predictor.active_serving_bundle"
SERVING_BUNDLE_MANIFEST = "bundle.json"
ACTIVE_SERVING_BUNDLE_POINTER = "active_serving_bundle.json"
_BUNDLE_DIRECTORY = "serving_bundles"


def verify_serving_bundle(
    root: Path,
    bundle_id: str,
    *,
    attestation_trust_store_path: Path,
    as_of: datetime | None = None,
    feature_store_config: LiveFeatureStoreConfig | None = None,
) -> dict[str, Any]:
    _require_sha256(bundle_id, "serving bundle id")
    repository = _repository_root(root)
    bundle_dir = repository / _BUNDLE_DIRECTORY / bundle_id
    _validate_bundle_directory(bundle_dir, repository / _BUNDLE_DIRECTORY)
    return _verify_bundle_directory(
        bundle_dir,
        repository=repository,
        expected_bundle_id=bundle_id,
        attestation_trust_store_path=attestation_trust_store_path,
        as_of=as_of,
        feature_store_config=feature_store_config,
    )


def load_active_serving_bundle(
    root: Path,
    *,
    attestation_trust_store_path: Path,
    as_of: datetime | None = None,
    feature_store_config: LiveFeatureStoreConfig | None = None,
) -> dict[str, Any]:
    repository = _repository_root(root)
    pointer = _load_active_pointer(repository / ACTIVE_SERVING_BUNDLE_POINTER)
    bundle_id = str(pointer["bundle_id"])
    bundle = verify_serving_bundle(
        repository,
        bundle_id,
        attestation_trust_store_path=attestation_trust_store_path,
        as_of=as_of,
        feature_store_config=feature_store_config,
    )
    manifest_path = repository / _BUNDLE_DIRECTORY / bundle_id / SERVING_BUNDLE_MANIFEST
    if file_sha256(manifest_path) != pointer["bundle_manifest_sha256"]:
        raise DataReadinessError("active serving bundle manifest changed")
    return {"pointer": pointer, "bundle": bundle}


def load_active_serving_bundle_pointer(root: Path) -> dict[str, Any]:
    return _load_active_pointer(
        _repository_root(root) / ACTIVE_SERVING_BUNDLE_POINTER
    )


def serving_bundle_asset_paths(
    root: Path,
    bundle: dict[str, Any],
) -> tuple[Path, Path]:
    repository = _repository_root(root)
    bundle_id = str(bundle.get("bundle_id") or "")
    _require_sha256(bundle_id, "serving bundle id")
    bundle_dir = repository / _BUNDLE_DIRECTORY / bundle_id
    return (
        _safe_child(bundle_dir, str(bundle["feature_path"])),
        _release_model_path(repository, bundle),
    )


def _verify_bundle_directory(
    bundle_dir: Path,
    *,
    repository: Path,
    expected_bundle_id: str,
    attestation_trust_store_path: Path,
    as_of: datetime | None,
    feature_store_config: LiveFeatureStoreConfig | None,
) -> dict[str, Any]:
    bundle = _load_json_object(
        bundle_dir / SERVING_BUNDLE_MANIFEST,
        "serving bundle manifest",
    )
    expected_fields = {
        "schema",
        "bundle_id",
        "mode",
        "horizon",
        "generated_at_utc",
        "model_release_id",
        "model_release_manifest_sha256",
        "model_artifact_sha256",
        "calibration_method",
        "calibration_identity_sha256",
        "prediction_policy_sha256",
        "label_policy_sha256",
        "execution_policy_sha256",
        "feature_path",
        "feature_manifest_path",
        "feature_manifest_sha256",
        "feature_artifact_sha256",
        "feature_schema_version",
        "feature_source_artifact_sha256",
        "feature_source_artifact_type",
        "feature_columns_sha256",
    }
    if set(bundle) != expected_fields:
        raise DataReadinessError("serving bundle manifest fields are invalid")
    if (
        bundle.get("schema") != SERVING_BUNDLE_SCHEMA
        or bundle.get("bundle_id") != expected_bundle_id
    ):
        raise DataReadinessError("serving bundle identity mismatch")
    identity = dict(bundle)
    identity.pop("bundle_id")
    if _json_sha256(identity) != expected_bundle_id:
        raise DataReadinessError("serving bundle content hash does not match its id")

    mode = _live_mode(bundle.get("mode"))
    horizon = str(bundle.get("horizon") or "").strip()
    if not horizon:
        raise DataReadinessError("serving bundle horizon is missing")
    for field in (
        "model_release_id",
        "model_release_manifest_sha256",
        "model_artifact_sha256",
        "calibration_identity_sha256",
        "prediction_policy_sha256",
        "label_policy_sha256",
        "execution_policy_sha256",
        "feature_manifest_sha256",
        "feature_artifact_sha256",
        "feature_source_artifact_sha256",
        "feature_columns_sha256",
    ):
        _require_sha256(str(bundle.get(field) or ""), f"serving bundle {field}")

    generated = _parse_utc(str(bundle.get("generated_at_utc") or ""))
    if as_of is not None and generated > _utc(as_of):
        raise DataReadinessError("serving bundle was generated after the requested time")

    release_id = str(bundle["model_release_id"])
    release = verify_local_release(
        repository,
        release_id,
        attestation_trust_store_path=attestation_trust_store_path,
    )
    release_manifest = repository / "releases" / release_id / "release.json"
    if file_sha256(release_manifest) != bundle["model_release_manifest_sha256"]:
        raise DataReadinessError("serving bundle model release manifest mismatch")
    model_manifest = _load_release_model_manifest(
        repository,
        release,
        model_release_id=release_id,
    )
    expected_model = _model_identity(model_manifest, mode=mode, horizon=horizon)
    for field, expected in expected_model.items():
        if bundle.get(field) != expected:
            raise DataReadinessError(f"serving bundle {field} does not match its model")

    feature_path = _safe_child(bundle_dir, str(bundle["feature_path"]))
    feature_manifest_path = _safe_child(
        bundle_dir,
        str(bundle["feature_manifest_path"]),
    )
    if file_sha256(feature_manifest_path) != bundle["feature_manifest_sha256"]:
        raise DataReadinessError("serving bundle feature manifest integrity failed")
    feature_manifest = _load_json_object(
        feature_manifest_path,
        "serving bundle feature manifest",
    )
    _validate_feature_manifest_identity(feature_manifest, mode=mode)
    expected_feature = _feature_identity(feature_manifest)
    for field, expected in expected_feature.items():
        if bundle.get(field) != expected:
            raise DataReadinessError(f"serving bundle {field} does not match its feature manifest")
    if file_sha256(feature_path) != bundle["feature_artifact_sha256"]:
        raise DataReadinessError("serving bundle feature artifact integrity failed")
    feature_store = LiveFeatureStore(
        Path("."),
        _staged_feature_config(mode, feature_path, feature_store_config),
    )
    feature_store.validate(mode, as_of=as_of or generated)
    return bundle


def _model_identity(
    manifest: dict[str, Any],
    *,
    mode: LiveMode,
    horizon: str,
) -> dict[str, str]:
    metrics_raw = manifest.get("metrics")
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    if manifest.get("model_type") != "canonical_swing":
        raise DataReadinessError("serving bundle model type is incompatible with its mode")
    target = str(manifest.get("target_col") or "")
    if not _target_matches_horizon(target, horizon):
        raise DataReadinessError("serving bundle model target is incompatible with its horizon")
    artifact_sha = str(manifest.get("artifact_sha256") or "")
    prediction_policy = str(
        manifest.get("prediction_policy_sha256")
        or metrics.get("prediction_policy_sha256")
        or ""
    )
    label_policy = str(
        manifest.get("dataset_label_config_sha256")
        or metrics.get("dataset_label_config_sha256")
        or ""
    )
    execution_policy = str(
        manifest.get("execution_policy_sha256")
        or metrics.get("execution_policy_sha256")
        or ""
    )
    for value, name in (
        (artifact_sha, "model artifact"),
        (prediction_policy, "prediction policy"),
        (label_policy, "label policy"),
        (execution_policy, "execution policy"),
    ):
        _require_sha256(value, name)
    calibration_method = str(metrics.get("calibration_method") or "").strip()
    if not calibration_method:
        raise DataReadinessError("serving bundle model calibration identity is missing")
    calibration_identity = _json_sha256(
        {
            "model_artifact_sha256": artifact_sha,
            "calibration_method": calibration_method,
        }
    )
    return {
        "model_artifact_sha256": artifact_sha,
        "calibration_method": calibration_method,
        "calibration_identity_sha256": calibration_identity,
        "prediction_policy_sha256": prediction_policy,
        "label_policy_sha256": label_policy,
        "execution_policy_sha256": execution_policy,
    }


def _feature_identity(manifest: dict[str, Any]) -> dict[str, str]:
    columns = manifest.get("columns")
    if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
        raise DataReadinessError("serving bundle feature columns are invalid")
    return {
        "feature_artifact_sha256": str(manifest["artifact_sha256"]),
        "feature_schema_version": str(manifest["feature_schema_version"]),
        "feature_source_artifact_sha256": str(manifest["source_artifact_sha256"]),
        "feature_source_artifact_type": str(manifest["source_artifact_type"]),
        "feature_columns_sha256": _json_sha256({"columns": columns}),
    }


def _validate_feature_manifest_identity(
    manifest: dict[str, Any],
    *,
    mode: LiveMode,
) -> None:
    if manifest.get("schema") != LIVE_FEATURE_SCHEMA or manifest.get("mode") != mode:
        raise DataReadinessError("serving bundle feature manifest mode is invalid")
    if manifest.get("feature_schema_version") != LIVE_SCHEMA_VERSIONS[mode]:
        raise DataReadinessError("serving bundle feature schema is incompatible")
    if manifest.get("source_artifact_type") != LIVE_ARTIFACT_TYPES[mode]:
        raise DataReadinessError("serving bundle feature source type is incompatible")
    for field in ("artifact_sha256", "source_artifact_sha256"):
        _require_sha256(str(manifest.get(field) or ""), f"feature manifest {field}")


def _load_release_model_manifest(
    repository: Path,
    release: dict[str, Any],
    *,
    model_release_id: str,
) -> dict[str, Any]:
    model_path = _release_model_path(
        repository,
        {
            "model_release_id": model_release_id,
            "model_path": release.get("model_path"),
        },
    )
    return _load_json_object(
        model_path.with_suffix(model_path.suffix + ".manifest.json"),
        "released model manifest",
    )


def _release_model_path(repository: Path, bundle: dict[str, Any]) -> Path:
    release_id = str(bundle.get("model_release_id") or "")
    _require_sha256(release_id, "model release id")
    release_dir = repository / "releases" / release_id
    release = _load_json_object(release_dir / "release.json", "local release manifest")
    relative = str(release.get("model_path") or bundle.get("model_path") or "")
    return _safe_child(release_dir, relative)


def _staged_feature_config(
    mode: LiveMode,
    path: Path,
    base: LiveFeatureStoreConfig | None,
) -> LiveFeatureStoreConfig:
    config = base or LiveFeatureStoreConfig()
    return LiveFeatureStoreConfig(
        swing_path=path if mode == "swing" else config.swing_path,
        intraday_path=path if mode == "intraday" else config.intraday_path,
        swing_max_age=config.swing_max_age,
        intraday_max_age=config.intraday_max_age,
        swing_feature_max_age=config.swing_feature_max_age,
        intraday_feature_max_age=config.intraday_feature_max_age,
    )


def _load_active_pointer(path: Path) -> dict[str, Any]:
    pointer = _load_json_object(path, "active serving bundle pointer")
    expected = {
        "schema",
        "bundle_id",
        "bundle_manifest_sha256",
        "previous_bundle_id",
        "activated_at_utc",
        "pointer_sha256",
    }
    if set(pointer) != expected:
        raise DataReadinessError("active serving bundle pointer fields are invalid")
    content = dict(pointer)
    pointer_sha = str(content.pop("pointer_sha256", ""))
    if content.get("schema") != ACTIVE_SERVING_BUNDLE_SCHEMA:
        raise DataReadinessError("active serving bundle pointer schema mismatch")
    _require_sha256(str(content.get("bundle_id") or ""), "active serving bundle id")
    _require_sha256(
        str(content.get("bundle_manifest_sha256") or ""),
        "active serving bundle manifest",
    )
    previous = content.get("previous_bundle_id")
    if previous is not None:
        _require_sha256(str(previous), "previous serving bundle id")
    _parse_utc(str(content.get("activated_at_utc") or ""))
    if _json_sha256(content) != pointer_sha:
        raise DataReadinessError("active serving bundle pointer integrity failed")
    return pointer


def _live_mode(value: object) -> LiveMode:
    normalized = str(value or "").strip().lower()
    if normalized == "intraday":
        raise DataReadinessError("intraday serving bundles are retired; only swing bundles are served")
    if normalized != "swing":
        raise DataReadinessError("serving bundle mode is invalid")
    return "swing"


def _target_matches_horizon(target: str, horizon: str) -> bool:
    return f"_{horizon.strip().lower()}" in target.lower()


def _safe_child(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise DataReadinessError("serving bundle path is unsafe")
    return path_integrity.resolve_existing_file_inside(
        root,
        relative,
        label="serving bundle artifact",
    )


def _validate_bundle_directory(path: Path, root: Path) -> None:
    expected = root.resolve() / path.name
    if not path.is_dir() or path.resolve() != expected:
        raise DataReadinessError("serving bundle directory escapes its repository")
    path_integrity.verify_tree_containment(path, label="serving bundle directory")


def _repository_root(root: Path) -> Path:
    absolute = path_integrity.verify_no_reparse_ancestry(
        root,
        label="serving repository",
    )
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError("serving repository is unavailable") from exc
    if not resolved.is_dir():
        raise DataReadinessError("serving repository is not a directory")
    return resolved


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        safe_path = path_integrity.verify_no_reparse_ancestry(path, label=name)
        loaded = parse_strict_json_object(safe_path.read_bytes(), label=name)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise DataReadinessError(f"{name} is unavailable or invalid") from exc
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DataReadinessError(f"{name} is not a SHA-256 digest")


def _parse_utc(value: str) -> datetime:
    try:
        return _utc(datetime.fromisoformat(value))
    except ValueError as exc:
        raise DataReadinessError("serving bundle timestamp is invalid") from exc


def _utc(value: datetime) -> datetime:
    if value.utcoffset() is None:
        raise ValueError("serving bundle timestamps must be timezone-aware")
    return value.astimezone(UTC)
