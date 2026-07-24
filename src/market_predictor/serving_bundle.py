from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.feature_store import (
    LIVE_FEATURE_SCHEMA,
    LiveFeatureStore,
    LiveFeatureStoreConfig,
)
from market_predictor.live_features import LIVE_ARTIFACT_TYPES, LIVE_SCHEMA_VERSIONS, LiveMode
from market_predictor.locking import file_lock
from market_predictor.registry import file_sha256
from market_predictor.release import verify_local_release
from market_predictor.v3.errors import DataReadinessError

SERVING_BUNDLE_SCHEMA = "market_predictor.serving_bundle.v1"
ACTIVE_SERVING_BUNDLE_SCHEMA = "market_predictor.active_serving_bundle.v1"
SERVING_BUNDLE_MANIFEST = "bundle.json"
ACTIVE_SERVING_BUNDLE_POINTER = "active_serving_bundle.json"
_BUNDLE_DIRECTORY = "serving_bundles"
_FEATURE_ASSET = "features/features.parquet"
_FEATURE_MANIFEST_ASSET = "features/features.parquet.manifest.json"


def publish_serving_bundle(
    root: Path,
    *,
    mode: LiveMode,
    horizon: str,
    model_release_id: str,
    feature_path: Path,
    attestation_trust_store_path: Path,
    activate: bool = True,
    generated_at: datetime | None = None,
    feature_store_config: LiveFeatureStoreConfig | None = None,
) -> dict[str, Any]:
    """Publish an immutable model/feature serving generation."""

    repository = root.resolve()
    generated = _utc(generated_at or datetime.now(UTC))
    release = verify_local_release(
        repository,
        model_release_id,
        attestation_trust_store_path=attestation_trust_store_path,
    )
    model_manifest = _load_release_model_manifest(
        repository,
        release,
        model_release_id=model_release_id,
    )
    model_identity = _model_identity(model_manifest, mode=mode, horizon=horizon)
    releases_root = repository / "releases"
    release_manifest_path = releases_root / model_release_id / "release.json"

    bundles_root = repository / _BUNDLE_DIRECTORY
    bundles_root.mkdir(parents=True, exist_ok=True)
    staging = bundles_root / f".staging-{uuid4().hex}"
    with file_lock(repository / ".serving-bundle-publish"):
        try:
            staging.mkdir(parents=False, exist_ok=False)
            staged_feature = staging / _FEATURE_ASSET
            staged_manifest = staging / _FEATURE_MANIFEST_ASSET
            staged_feature.parent.mkdir(parents=True, exist_ok=True)
            _copy_file_durable(feature_path.resolve(), staged_feature)
            _copy_file_durable(_feature_manifest_path(feature_path).resolve(), staged_manifest)

            feature_manifest = _load_json_object(
                staged_manifest,
                "live feature manifest",
            )
            feature_store = LiveFeatureStore(
                Path("."),
                _staged_feature_config(mode, staged_feature, feature_store_config),
            )
            feature_store.validate(mode, as_of=generated)
            _validate_feature_manifest_identity(feature_manifest, mode=mode)
            feature_identity = _feature_identity(feature_manifest)

            identity: dict[str, Any] = {
                "schema": SERVING_BUNDLE_SCHEMA,
                "mode": mode,
                "horizon": horizon,
                "generated_at_utc": generated.isoformat(),
                "model_release_id": model_release_id,
                "model_release_manifest_sha256": file_sha256(release_manifest_path),
                **model_identity,
                "feature_path": _FEATURE_ASSET,
                "feature_manifest_path": _FEATURE_MANIFEST_ASSET,
                "feature_manifest_sha256": file_sha256(staged_manifest),
                **feature_identity,
            }
            bundle_id = _json_sha256(identity)
            bundle = {**identity, "bundle_id": bundle_id}
            _write_json_atomic(staging / SERVING_BUNDLE_MANIFEST, bundle)

            destination = bundles_root / bundle_id
            if destination.exists():
                verified = verify_serving_bundle(
                    repository,
                    bundle_id,
                    attestation_trust_store_path=attestation_trust_store_path,
                    as_of=generated,
                    feature_store_config=feature_store_config,
                )
            else:
                _verify_bundle_directory(
                    staging,
                    repository=repository,
                    expected_bundle_id=bundle_id,
                    attestation_trust_store_path=attestation_trust_store_path,
                    as_of=generated,
                    feature_store_config=feature_store_config,
                )
                os.replace(staging, destination)
                _fsync_directory(bundles_root)
                verified = verify_serving_bundle(
                    repository,
                    bundle_id,
                    attestation_trust_store_path=attestation_trust_store_path,
                    as_of=generated,
                    feature_store_config=feature_store_config,
                )
        finally:
            if staging.exists():
                _remove_staging_directory(staging, bundles_root)

    result = dict(verified)
    if activate:
        result["active_pointer"] = activate_serving_bundle(
            repository,
            bundle_id,
            attestation_trust_store_path=attestation_trust_store_path,
            activated_at=generated,
            feature_store_config=feature_store_config,
        )
    return result


def verify_serving_bundle(
    root: Path,
    bundle_id: str,
    *,
    attestation_trust_store_path: Path,
    as_of: datetime | None = None,
    feature_store_config: LiveFeatureStoreConfig | None = None,
) -> dict[str, Any]:
    _require_sha256(bundle_id, "serving bundle id")
    repository = root.resolve()
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


def activate_serving_bundle(
    root: Path,
    bundle_id: str,
    *,
    attestation_trust_store_path: Path,
    activated_at: datetime | None = None,
    feature_store_config: LiveFeatureStoreConfig | None = None,
) -> dict[str, Any]:
    repository = root.resolve()
    pointer_path = repository / ACTIVE_SERVING_BUNDLE_POINTER
    with file_lock(pointer_path):
        return _activate_serving_bundle_locked(
            repository,
            bundle_id,
            pointer_path=pointer_path,
            attestation_trust_store_path=attestation_trust_store_path,
            activated_at=activated_at,
            feature_store_config=feature_store_config,
        )


def rollback_serving_bundle(
    root: Path,
    bundle_id: str,
    *,
    attestation_trust_store_path: Path,
    activated_at: datetime | None = None,
    feature_store_config: LiveFeatureStoreConfig | None = None,
) -> dict[str, Any]:
    repository = root.resolve()
    pointer_path = repository / ACTIVE_SERVING_BUNDLE_POINTER
    with file_lock(pointer_path):
        current = _load_active_pointer(pointer_path)
        if current.get("previous_bundle_id") != bundle_id:
            raise DataReadinessError(
                "serving bundle rollback target must be the immediately previous bundle"
            )
        return _activate_serving_bundle_locked(
            repository,
            bundle_id,
            pointer_path=pointer_path,
            attestation_trust_store_path=attestation_trust_store_path,
            activated_at=activated_at,
            feature_store_config=feature_store_config,
        )


def load_active_serving_bundle(
    root: Path,
    *,
    attestation_trust_store_path: Path,
    as_of: datetime | None = None,
    feature_store_config: LiveFeatureStoreConfig | None = None,
) -> dict[str, Any]:
    repository = root.resolve()
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
    return _load_active_pointer(root.resolve() / ACTIVE_SERVING_BUNDLE_POINTER)


def serving_bundle_asset_paths(
    root: Path,
    bundle: dict[str, Any],
) -> tuple[Path, Path]:
    repository = root.resolve()
    bundle_id = str(bundle.get("bundle_id") or "")
    _require_sha256(bundle_id, "serving bundle id")
    bundle_dir = repository / _BUNDLE_DIRECTORY / bundle_id
    return (
        _safe_child(bundle_dir, str(bundle["feature_path"])),
        _release_model_path(repository, bundle),
    )


def _activate_serving_bundle_locked(
    repository: Path,
    bundle_id: str,
    *,
    pointer_path: Path,
    attestation_trust_store_path: Path,
    activated_at: datetime | None,
    feature_store_config: LiveFeatureStoreConfig | None,
) -> dict[str, Any]:
    timestamp = _utc(activated_at or datetime.now(UTC))
    bundle = verify_serving_bundle(
        repository,
        bundle_id,
        attestation_trust_store_path=attestation_trust_store_path,
        as_of=timestamp,
        feature_store_config=feature_store_config,
    )
    previous_bundle_id: str | None = None
    if pointer_path.exists():
        previous_bundle_id = str(_load_active_pointer(pointer_path)["bundle_id"])
    manifest_path = repository / _BUNDLE_DIRECTORY / bundle_id / SERVING_BUNDLE_MANIFEST
    content = {
        "schema": ACTIVE_SERVING_BUNDLE_SCHEMA,
        "bundle_id": bundle_id,
        "bundle_manifest_sha256": file_sha256(manifest_path),
        "previous_bundle_id": previous_bundle_id,
        "activated_at_utc": timestamp.isoformat(),
    }
    pointer = {**content, "pointer_sha256": _json_sha256(content)}
    _write_json_atomic(pointer_path, pointer)
    verified = _load_active_pointer(pointer_path)
    if verified["bundle_id"] != bundle["bundle_id"]:
        raise DataReadinessError("active serving bundle pointer changed during activation")
    return verified


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
    expected_model_type = "canonical_swing" if mode == "swing" else "canonical_intraday"
    if manifest.get("model_type") != expected_model_type:
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
    if normalized not in {"swing", "intraday"}:
        raise DataReadinessError("serving bundle mode is invalid")
    return normalized  # type: ignore[return-value]


def _target_matches_horizon(target: str, horizon: str) -> bool:
    normalized = target.lower()
    canonical = horizon.strip().lower()
    if canonical == "5d":
        return "next_week" in normalized or "_5d" in normalized
    if canonical == "1d":
        return "next_day" in normalized or "_1d" in normalized
    return f"_{canonical}" in normalized


def _feature_manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _safe_child(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if not relative or candidate.is_absolute() or ".." in candidate.parts:
        raise DataReadinessError("serving bundle path is unsafe")
    resolved_root = root.resolve()
    resolved = (resolved_root / candidate).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise DataReadinessError("serving bundle path escapes its root")
    return resolved


def _validate_bundle_directory(path: Path, root: Path) -> None:
    expected = root.resolve() / path.name
    if not path.is_dir() or path.resolve() != expected:
        raise DataReadinessError("serving bundle directory escapes its repository")


def _remove_staging_directory(staging: Path, root: Path) -> None:
    resolved = staging.resolve()
    repository = root.resolve()
    if not resolved.is_relative_to(repository) or not staging.name.startswith(".staging-"):
        raise DataReadinessError("refusing to remove an unsafe serving bundle staging path")
    shutil.rmtree(staging)


def _copy_file_durable(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as source_handle, destination.open("xb") as target:
        shutil.copyfileobj(source_handle, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise DataReadinessError(f"{name} is unavailable or invalid") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"{name} must contain an object")
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
