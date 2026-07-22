from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol, TypedDict, cast

from market_predictor.canonical.store import file_sha256
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.intraday.contracts import INTRADAY_MODEL_SCHEMA_VERSION, INTRADAY_MODEL_TYPE
from market_predictor.live_features import LiveMode
from market_predictor.prediction_service import ServingRoute, verify_serving_model_artifact
from market_predictor.registry import manifest_path_for
from market_predictor.swing.contracts import SWING_MODEL_SCHEMA_VERSION, SWING_MODEL_TYPE
from market_predictor.v3.errors import DataReadinessError

SERVING_RELEASE_SCHEMA = "market_predictor.serving_release.v1"
ACTIVE_RELEASE_SCHEMA = "market_predictor.active_serving_release.v1"
DEFAULT_RELEASE_PREFIX = "serving/releases"
DEFAULT_ACTIVE_POINTER = "serving/_active_release.json"


class ReleaseAsset(TypedDict):
    kind: str
    mode: str
    horizon: str | None
    destination: str
    sha256: str


class ServingRelease(TypedDict):
    schema: str
    routes: list[dict[str, object]]
    assets: list[ReleaseAsset]
    release_id: str
    generated_at_utc: str


class DeploymentBlobStore(Protocol):
    def upload_file(
        self,
        local_path: Path,
        blob_relative: str | Path,
        *,
        overwrite: bool = True,
    ) -> str: ...

    def upload_bytes(
        self,
        data: bytes,
        blob_relative: str | Path,
        *,
        overwrite: bool = True,
    ) -> str: ...

    def download_file(
        self,
        blob_relative: str | Path,
        local_path: Path,
        *,
        overwrite: bool = True,
    ) -> Path: ...

    def download_bytes(self, blob_relative: str | Path) -> bytes: ...

    def blob_exists(self, blob_relative: str | Path) -> bool: ...

    def blob_sha256(self, blob_relative: str | Path) -> str | None: ...


def publish_serving_release(
    store: DeploymentBlobStore,
    *,
    root: Path,
    routes: Mapping[str, Mapping[str, ServingRoute]],
    live_feature_store: LiveFeatureStore,
    release_prefix: str = DEFAULT_RELEASE_PREFIX,
    active_pointer_blob: str = DEFAULT_ACTIVE_POINTER,
    generated_at: datetime | None = None,
) -> dict[str, object]:
    """Publish immutable model/feature assets and move the active pointer last."""

    generated = _utc(generated_at or datetime.now(UTC))
    root = root.resolve()
    assets: list[ReleaseAsset] = []
    route_rows: list[dict[str, object]] = []
    modes = sorted(routes)
    if not modes:
        raise DataReadinessError("serving release requires at least one configured route")
    for mode in modes:
        if mode not in {"swing", "intraday"}:
            raise DataReadinessError(f"unsupported serving release mode: {mode}")
        live_mode = cast(LiveMode, mode)
        feature_manifest = live_feature_store.validate(live_mode, as_of=generated)
        live_feature_store.load(live_mode, as_of=generated)
        feature_path, feature_manifest_path = live_feature_store.paths(live_mode)
        assets.extend(
            _asset_pair(
                root,
                mode=mode,
                horizon=None,
                artifact_path=feature_path,
                manifest_path=feature_manifest_path,
                artifact_kind="feature",
                expected_sha=str(feature_manifest["artifact_sha256"]),
            )
        )
        for horizon, route in sorted(routes[mode].items()):
            model_path = _resolved_under_root(root, route.model)
            model_manifest = verify_serving_model_artifact(
                model_path,
                resolved_horizon=horizon,
                expected_model_type=(SWING_MODEL_TYPE if mode == "swing" else INTRADAY_MODEL_TYPE),
                expected_schema_version=(SWING_MODEL_SCHEMA_VERSION if mode == "swing" else INTRADAY_MODEL_SCHEMA_VERSION),
            )
            model_manifest_path = manifest_path_for(model_path)
            assets.extend(
                _asset_pair(
                    root,
                    mode=mode,
                    horizon=horizon,
                    artifact_path=model_path,
                    manifest_path=model_manifest_path,
                    artifact_kind="model",
                    expected_sha=str(model_manifest["artifact_sha256"]),
                )
            )
            route_rows.append(
                {
                    "mode": mode,
                    "horizon": horizon,
                    "model_destination": _relative_destination(root, model_path),
                    "model_sha256": model_manifest["artifact_sha256"],
                    "model_type": model_manifest["model_type"],
                    "schema_version": model_manifest["schema_version"],
                    "target_col": model_manifest["target_col"],
                }
            )
    sorted_assets = sorted(assets, key=lambda item: item["destination"])
    destinations = [asset["destination"] for asset in sorted_assets]
    if len(destinations) != len(set(destinations)):
        raise DataReadinessError("serving release contains duplicate asset destinations")
    content: dict[str, object] = {
        "schema": SERVING_RELEASE_SCHEMA,
        "routes": route_rows,
        "assets": sorted_assets,
    }
    release_id = _json_sha256(content)
    prefix = f"{release_prefix.strip('/')}/{release_id}"
    release: ServingRelease = {
        "schema": SERVING_RELEASE_SCHEMA,
        "routes": route_rows,
        "assets": sorted_assets,
        "release_id": release_id,
        "generated_at_utc": generated.isoformat(),
    }
    release_bytes = _json_bytes(release)
    release_manifest_blob = f"{prefix}/release.json"
    if store.blob_exists(release_manifest_blob):
        existing = store.download_bytes(release_manifest_blob)
        existing_release = _load_release(existing, expected_release_id=release_id)
        _verify_release_assets(store, existing_release, release_manifest_blob)
        release_bytes = existing
    else:
        for asset in release["assets"]:
            local_path = _resolved_under_root(root, Path(str(asset["destination"])))
            blob = _asset_blob(release_manifest_blob, asset["destination"])
            _upload_release_asset(store, local_path=local_path, blob=blob, expected_sha=asset["sha256"])
        store.upload_bytes(release_bytes, release_manifest_blob, overwrite=False)
        if store.blob_sha256(release_manifest_blob) != _sha256(release_bytes):
            raise DataReadinessError("uploaded serving release manifest failed integrity verification")
    previous = _active_release_id(store, active_pointer_blob)
    pointer = _active_pointer(
        release_id=release_id,
        release_manifest_blob=release_manifest_blob,
        release_manifest_sha256=_sha256(release_bytes),
        previous_release_id=previous,
        activated_at=generated,
    )
    store.upload_bytes(_json_bytes(pointer), active_pointer_blob, overwrite=True)
    return pointer


def rollback_serving_release(
    store: DeploymentBlobStore,
    *,
    release_id: str,
    release_prefix: str = DEFAULT_RELEASE_PREFIX,
    active_pointer_blob: str = DEFAULT_ACTIVE_POINTER,
    activated_at: datetime | None = None,
) -> dict[str, object]:
    """Move the active pointer to a complete previously published immutable release."""

    if not _is_sha256(release_id):
        raise DataReadinessError("release_id must be a SHA-256 digest")
    manifest_blob = f"{release_prefix.strip('/')}/{release_id}/release.json"
    if not store.blob_exists(manifest_blob):
        raise DataReadinessError(f"serving release does not exist: {release_id}")
    release_bytes = store.download_bytes(manifest_blob)
    release = _load_release(release_bytes, expected_release_id=release_id)
    _verify_release_assets(store, release, manifest_blob)
    previous = _active_release_id(store, active_pointer_blob)
    pointer = _active_pointer(
        release_id=release_id,
        release_manifest_blob=manifest_blob,
        release_manifest_sha256=_sha256(release_bytes),
        previous_release_id=previous,
        activated_at=_utc(activated_at or datetime.now(UTC)),
    )
    store.upload_bytes(_json_bytes(pointer), active_pointer_blob, overwrite=True)
    return pointer


def sync_active_serving_release(
    store: DeploymentBlobStore,
    *,
    root: Path,
    active_pointer_blob: str = DEFAULT_ACTIVE_POINTER,
) -> dict[str, object]:
    """Download and verify the active release before replacing local assets manifest-last."""

    pointer_bytes = store.download_bytes(active_pointer_blob)
    pointer = _load_pointer(pointer_bytes)
    release_bytes = store.download_bytes(str(pointer["release_manifest_blob"]))
    if _sha256(release_bytes) != pointer["release_manifest_sha256"]:
        raise DataReadinessError("active release manifest integrity check failed")
    release = _load_release(release_bytes, expected_release_id=str(pointer["release_id"]))
    root = root.resolve()
    with tempfile.TemporaryDirectory(prefix="market-predictor-release-", dir=root) as temporary:
        staging = Path(temporary)
        ordered_assets = sorted(
            release["assets"],
            key=lambda item: str(item["kind"]).endswith("_manifest"),
        )
        for asset in ordered_assets:
            destination = _safe_destination(root, str(asset["destination"]))
            staged = staging / str(asset["destination"])
            store.download_file(
                _asset_blob(str(pointer["release_manifest_blob"]), asset["destination"]),
                staged,
                overwrite=True,
            )
            if file_sha256(staged) != str(asset["sha256"]):
                raise DataReadinessError(f"downloaded serving asset failed integrity: {asset['destination']}")
            destination.parent.mkdir(parents=True, exist_ok=True)
        for asset in ordered_assets:
            destination = _safe_destination(root, str(asset["destination"]))
            staged = staging / str(asset["destination"])
            os.replace(staged, destination)
    marker = root / "data/live/.active_release.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary_marker = marker.with_suffix(".tmp")
    temporary_marker.write_bytes(pointer_bytes)
    os.replace(temporary_marker, marker)
    return pointer


def _asset_pair(
    root: Path,
    *,
    mode: str,
    horizon: str | None,
    artifact_path: Path,
    manifest_path: Path,
    artifact_kind: str,
    expected_sha: str,
) -> list[ReleaseAsset]:
    resolved_artifact = _resolved_under_root(root, artifact_path)
    resolved_manifest = _resolved_under_root(root, manifest_path)
    actual_sha = file_sha256(resolved_artifact)
    if actual_sha != expected_sha:
        raise DataReadinessError(f"{artifact_kind} artifact hash changed before release publication")
    return [
        {
            "kind": artifact_kind,
            "mode": mode,
            "horizon": horizon,
            "destination": _relative_destination(root, resolved_artifact),
            "sha256": actual_sha,
        },
        {
            "kind": f"{artifact_kind}_manifest",
            "mode": mode,
            "horizon": horizon,
            "destination": _relative_destination(root, resolved_manifest),
            "sha256": file_sha256(resolved_manifest),
        },
    ]


def _active_pointer(
    *,
    release_id: str,
    release_manifest_blob: str,
    release_manifest_sha256: str,
    previous_release_id: str | None,
    activated_at: datetime,
) -> dict[str, object]:
    return {
        "schema": ACTIVE_RELEASE_SCHEMA,
        "release_id": release_id,
        "release_manifest_blob": release_manifest_blob,
        "release_manifest_sha256": release_manifest_sha256,
        "previous_release_id": previous_release_id,
        "activated_at_utc": activated_at.isoformat(),
    }


def _active_release_id(store: DeploymentBlobStore, active_pointer_blob: str) -> str | None:
    if not store.blob_exists(active_pointer_blob):
        return None
    return str(_load_pointer(store.download_bytes(active_pointer_blob))["release_id"])


def _load_pointer(data: bytes) -> dict[str, object]:
    loaded = _json_object(data, "active release pointer")
    if loaded.get("schema") != ACTIVE_RELEASE_SCHEMA:
        raise DataReadinessError("active release pointer schema mismatch")
    release_id = str(loaded.get("release_id", ""))
    manifest_sha = str(loaded.get("release_manifest_sha256", ""))
    if not _is_sha256(release_id) or not _is_sha256(manifest_sha) or not loaded.get("release_manifest_blob"):
        raise DataReadinessError("active release pointer is incomplete")
    return loaded


def _load_release(data: bytes, *, expected_release_id: str) -> ServingRelease:
    loaded = _json_object(data, "serving release")
    if loaded.get("schema") != SERVING_RELEASE_SCHEMA or loaded.get("release_id") != expected_release_id:
        raise DataReadinessError("serving release identity mismatch")
    assets = loaded.get("assets")
    routes = loaded.get("routes")
    if not isinstance(assets, list) or not assets or not isinstance(routes, list) or not routes:
        raise DataReadinessError("serving release is incomplete")
    normalized_assets: list[ReleaseAsset] = []
    for asset in assets:
        if not isinstance(asset, dict) or not all(asset.get(key) for key in ("destination", "sha256", "kind", "mode")):
            raise DataReadinessError("serving release contains an invalid asset")
        horizon = asset.get("horizon")
        if horizon is not None and not isinstance(horizon, str):
            raise DataReadinessError("serving release contains an invalid asset horizon")
        destination = str(asset["destination"])
        destination_path = PurePosixPath(destination)
        if destination_path.is_absolute() or ".." in destination_path.parts or "\\" in destination:
            raise DataReadinessError("serving release contains an unsafe asset destination")
        sha256 = str(asset["sha256"])
        if not _is_sha256(sha256):
            raise DataReadinessError("serving release contains an invalid asset hash")
        normalized_assets.append(
            {
                "kind": str(asset["kind"]),
                "mode": str(asset["mode"]),
                "horizon": horizon,
                "destination": destination,
                "sha256": sha256,
            }
        )
    destinations = [asset["destination"] for asset in normalized_assets]
    if len(destinations) != len(set(destinations)):
        raise DataReadinessError("serving release contains duplicate asset destinations")
    normalized_routes: list[dict[str, object]] = []
    for route in routes:
        if not isinstance(route, dict):
            raise DataReadinessError("serving release contains an invalid route")
        normalized_routes.append({str(key): value for key, value in route.items()})
    content: dict[str, object] = {
        "schema": str(loaded["schema"]),
        "routes": normalized_routes,
        "assets": normalized_assets,
    }
    if _json_sha256(content) != expected_release_id:
        raise DataReadinessError("serving release content hash does not match its id")
    generated_at = loaded.get("generated_at_utc")
    if not isinstance(generated_at, str):
        raise DataReadinessError("serving release is missing generated_at_utc")
    return {
        "schema": SERVING_RELEASE_SCHEMA,
        "routes": normalized_routes,
        "assets": normalized_assets,
        "release_id": expected_release_id,
        "generated_at_utc": generated_at,
    }


def _json_object(data: bytes, name: str) -> dict[str, object]:
    try:
        loaded = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{name} is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"{name} must contain an object")
    return {str(key): value for key, value in loaded.items()}


def _resolved_under_root(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise DataReadinessError(f"serving asset is outside deployment root: {path}")
    if not resolved.exists():
        raise FileNotFoundError(f"serving asset is missing: {resolved}")
    return resolved


def _safe_destination(root: Path, destination: str) -> Path:
    path = (root / destination).resolve()
    if not path.is_relative_to(root):
        raise DataReadinessError(f"release destination escapes deployment root: {destination}")
    return path


def _relative_destination(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _json_sha256(value: object) -> str:
    return _sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _asset_blob(release_manifest_blob: str, destination: str) -> str:
    release_prefix, separator, _ = release_manifest_blob.rpartition("/")
    if not separator:
        raise DataReadinessError("serving release manifest path has no parent prefix")
    return f"{release_prefix}/{destination}"


def _upload_release_asset(
    store: DeploymentBlobStore,
    *,
    local_path: Path,
    blob: str,
    expected_sha: str,
) -> None:
    if store.blob_exists(blob):
        if store.blob_sha256(blob) != expected_sha:
            raise DataReadinessError(f"existing serving release asset failed integrity: {blob}")
        return
    store.upload_file(local_path, blob, overwrite=False)
    if store.blob_sha256(blob) != expected_sha:
        raise DataReadinessError(f"uploaded serving release asset failed integrity: {blob}")


def _verify_release_assets(
    store: DeploymentBlobStore,
    release: ServingRelease,
    release_manifest_blob: str,
) -> None:
    failures: list[str] = []
    for asset in release["assets"]:
        blob = _asset_blob(release_manifest_blob, asset["destination"])
        if not store.blob_exists(blob) or store.blob_sha256(blob) != asset["sha256"]:
            failures.append(blob)
    if failures:
        raise DataReadinessError(f"serving release is incomplete or corrupt: {', '.join(failures)}")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("deployment timestamps must be timezone-aware")
    return value.astimezone(UTC)
