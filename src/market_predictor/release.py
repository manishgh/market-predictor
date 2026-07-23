from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from market_predictor.locking import file_lock
from market_predictor.promotion_attestation import (
    file_sha256,
    promotion_attestation_path_for,
    verify_promotion_attestation,
)
from market_predictor.registry import (
    MODEL_STATUS_PROMOTED,
    manifest_path_for,
    verify_model_artifact,
)
from market_predictor.v3.errors import DataReadinessError

LOCAL_RELEASE_SCHEMA = "market_predictor.local_release.v1"
ACTIVE_LOCAL_RELEASE_SCHEMA = "market_predictor.active_local_release.v1"
RELEASE_MANIFEST_NAME = "release.json"
ACTIVE_POINTER_NAME = "active_release.json"
_REQUIRED_ASSET_KINDS = {
    "model",
    "candidate_manifest",
    "promotion_attestation",
    "evidence_manifest",
}


def publish_local_release(
    root: Path,
    *,
    model_path: Path,
    evidence_manifest_path: Path,
    activate: bool = True,
    attestation_trust_store_path: Path | None = None,
) -> dict[str, Any]:
    """Publish an immutable, verified release and optionally activate it."""

    root = root.resolve()
    releases_root = root / "releases"
    releases_root.mkdir(parents=True, exist_ok=True)
    sources, identity = _source_inventory(
        model_path,
        evidence_manifest_path,
        attestation_trust_store_path=attestation_trust_store_path,
    )
    release_id = _json_sha256(identity)
    release = {
        **identity,
        "release_id": release_id,
    }
    destination = releases_root / release_id
    staging = releases_root / f".staging-{release_id}-{uuid4().hex}"
    with file_lock(root / ".release-publish"):
        if destination.exists():
            verified = verify_local_release(
                root,
                release_id,
                attestation_trust_store_path=attestation_trust_store_path,
            )
        else:
            try:
                staging.mkdir(parents=False, exist_ok=False)
                for _, source, relative in sources:
                    target = _safe_child(staging, relative)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _copy_file_durable(source, target)
                _write_json_atomic(staging / RELEASE_MANIFEST_NAME, release)
                verified = _verify_release_directory(
                    staging,
                    expected_release_id=release_id,
                    attestation_trust_store_path=attestation_trust_store_path,
                )
                os.replace(staging, destination)
                _fsync_directory(releases_root)
                verified = verify_local_release(
                    root,
                    release_id,
                    attestation_trust_store_path=attestation_trust_store_path,
                )
            finally:
                if staging.exists():
                    _remove_staging_directory(staging, releases_root)
    result = dict(verified)
    if activate:
        result["active_pointer"] = activate_local_release(
            root,
            release_id,
            attestation_trust_store_path=attestation_trust_store_path,
        )
    return result


def verify_local_release(
    root: Path,
    release_id: str,
    *,
    attestation_trust_store_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a complete immutable release, including its promotion attestation."""

    _require_sha256(release_id, "release_id")
    repository_root = root.resolve()
    releases_root = repository_root / "releases"
    release_dir = releases_root / release_id
    _validate_release_directory(release_dir, releases_root)
    return _verify_release_directory(
        release_dir,
        expected_release_id=release_id,
        attestation_trust_store_path=attestation_trust_store_path,
    )


def activate_local_release(
    root: Path,
    release_id: str,
    *,
    activated_at: datetime | None = None,
    attestation_trust_store_path: Path | None = None,
) -> dict[str, Any]:
    """Atomically switch the single active pointer to a verified release."""

    root = root.resolve()
    pointer_path = root / ACTIVE_POINTER_NAME
    with file_lock(pointer_path):
        return _activate_local_release_locked(
            root,
            release_id,
            pointer_path=pointer_path,
            activated_at=activated_at,
            attestation_trust_store_path=attestation_trust_store_path,
        )


def rollback_local_release(
    root: Path,
    release_id: str,
    *,
    activated_at: datetime | None = None,
    attestation_trust_store_path: Path | None = None,
) -> dict[str, Any]:
    """Roll back by activating a previously published, still-valid release."""

    root = root.resolve()
    pointer_path = root / ACTIVE_POINTER_NAME
    with file_lock(pointer_path):
        current = _load_active_pointer(pointer_path)
        if current.get("previous_release_id") != release_id:
            raise DataReadinessError(
                "rollback target must be the immediately previous release"
            )
        return _activate_local_release_locked(
            root,
            release_id,
            pointer_path=pointer_path,
            activated_at=activated_at,
            attestation_trust_store_path=attestation_trust_store_path,
        )


def load_active_local_release(
    root: Path,
    *,
    attestation_trust_store_path: Path | None = None,
) -> dict[str, Any]:
    """Load the active pointer and verify the referenced release before use."""

    root = root.resolve()
    pointer = _load_active_pointer(root / ACTIVE_POINTER_NAME)
    release_id = str(pointer["release_id"])
    release = verify_local_release(
        root,
        release_id,
        attestation_trust_store_path=attestation_trust_store_path,
    )
    release_manifest = root / "releases" / release_id / RELEASE_MANIFEST_NAME
    if file_sha256(release_manifest) != pointer["release_manifest_sha256"]:
        raise DataReadinessError("active pointer release manifest changed")
    return {"pointer": pointer, "release": release}


def _source_inventory(
    model_path: Path,
    evidence_manifest_path: Path,
    *,
    attestation_trust_store_path: Path | None,
) -> tuple[list[tuple[str, Path, str]], dict[str, Any]]:
    model_path = model_path.resolve()
    evidence_manifest_path = evidence_manifest_path.resolve()
    try:
        candidate = verify_model_artifact(
            model_path,
            allowed_statuses={MODEL_STATUS_PROMOTED},
        )
        attestation = verify_promotion_attestation(
            model_path,
            evidence_manifest_path=evidence_manifest_path,
            trust_store_path=attestation_trust_store_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise DataReadinessError("local release requires a valid promoted candidate") from exc
    evidence = _load_json_object(evidence_manifest_path, "training evidence manifest")
    _validate_evidence_schema(evidence, model_type=str(candidate.get("model_type") or ""))
    if evidence.get("model_artifact_sha256") != candidate.get("artifact_sha256"):
        raise DataReadinessError("release evidence does not belong to the promoted candidate")

    candidate_manifest = manifest_path_for(model_path).resolve()
    promotion_attestation = promotion_attestation_path_for(model_path).resolve()
    model_relative = f"model/{model_path.name}"
    candidate_relative = f"model/{candidate_manifest.name}"
    attestation_relative = f"model/{promotion_attestation.name}"
    evidence_manifest_relative = f"evidence/{evidence_manifest_path.name}"
    sources: list[tuple[str, Path, str]] = [
        ("model", model_path, model_relative),
        ("candidate_manifest", candidate_manifest, candidate_relative),
        ("promotion_attestation", promotion_attestation, attestation_relative),
        ("evidence_manifest", evidence_manifest_path, evidence_manifest_relative),
    ]
    files = evidence.get("files")
    if not isinstance(files, dict):
        raise DataReadinessError("training evidence manifest has no file inventory")
    evidence_root = evidence_manifest_path.parent
    for name, record in sorted(files.items()):
        if not isinstance(record, dict):
            raise DataReadinessError(f"invalid training evidence record: {name}")
        relative = _safe_relative(str(record.get("path") or ""), f"evidence file {name}")
        source = (evidence_root / Path(relative.as_posix())).resolve()
        if not source.is_relative_to(evidence_root) or not source.is_file():
            raise DataReadinessError(f"training evidence file is missing or outside its bundle: {name}")
        expected_sha = str(record.get("sha256") or "")
        _require_sha256(expected_sha, f"evidence file {name} sha256")
        if file_sha256(source) != expected_sha:
            raise DataReadinessError(f"training evidence integrity check failed: {name}")
        sources.append(("evidence_file", source, f"evidence/{relative.as_posix()}"))

    destinations = [relative for _, _, relative in sources]
    if len(destinations) != len(set(destinations)):
        raise DataReadinessError("release source inventory contains duplicate destinations")
    assets = sorted(
        (
            {
                "kind": kind,
                "destination": relative,
                "sha256": file_sha256(source),
            }
            for kind, source, relative in sources
        ),
        key=lambda record: str(record["destination"]),
    )
    identity = {
        "schema": LOCAL_RELEASE_SCHEMA,
        "model_path": model_relative,
        "candidate_manifest_path": candidate_relative,
        "promotion_attestation_path": attestation_relative,
        "evidence_manifest_path": evidence_manifest_relative,
        "attestation_id": attestation["attestation_id"],
        "assets": assets,
        "generated_at_utc": attestation["promoted_at_utc"],
    }
    return sources, identity


def _activate_local_release_locked(
    root: Path,
    release_id: str,
    *,
    pointer_path: Path,
    activated_at: datetime | None,
    attestation_trust_store_path: Path | None,
) -> dict[str, Any]:
    release = verify_local_release(
        root,
        release_id,
        attestation_trust_store_path=attestation_trust_store_path,
    )
    release_manifest = root / "releases" / release_id / RELEASE_MANIFEST_NAME
    previous_release_id: str | None = None
    if pointer_path.exists():
        previous = _load_active_pointer(pointer_path)
        previous_release_id = str(previous["release_id"])
    if previous_release_id == release_id:
        return previous
    content: dict[str, Any] = {
        "schema": ACTIVE_LOCAL_RELEASE_SCHEMA,
        "release_id": release_id,
        "release_manifest_sha256": file_sha256(release_manifest),
        "previous_release_id": previous_release_id,
        "activated_at_utc": _utc(activated_at or datetime.now(UTC)).isoformat(),
    }
    pointer = {**content, "pointer_sha256": _json_sha256(content)}
    _write_json_atomic(pointer_path, pointer)
    verified_pointer = _load_active_pointer(pointer_path)
    if verified_pointer["release_id"] != release["release_id"]:
        raise DataReadinessError("active release pointer changed during activation")
    return verified_pointer


def _verify_release_directory(
    release_dir: Path,
    *,
    expected_release_id: str,
    attestation_trust_store_path: Path | None,
) -> dict[str, Any]:
    if not release_dir.is_dir():
        raise DataReadinessError(f"local release is missing: {expected_release_id}")
    manifest_path = release_dir / RELEASE_MANIFEST_NAME
    release = _load_json_object(manifest_path, "local release manifest")
    expected_fields = {
        "schema",
        "model_path",
        "candidate_manifest_path",
        "promotion_attestation_path",
        "evidence_manifest_path",
        "attestation_id",
        "assets",
        "generated_at_utc",
        "release_id",
    }
    if set(release) != expected_fields:
        raise DataReadinessError("local release manifest fields are invalid")
    if release.get("schema") != LOCAL_RELEASE_SCHEMA or release.get("release_id") != expected_release_id:
        raise DataReadinessError("local release identity mismatch")
    assets = release.get("assets")
    if not isinstance(assets, list) or not assets:
        raise DataReadinessError("local release has no assets")
    normalized_assets: list[dict[str, str]] = []
    kinds: dict[str, list[str]] = {}
    for raw_asset in assets:
        if not isinstance(raw_asset, dict):
            raise DataReadinessError("local release contains an invalid asset")
        if set(raw_asset) != {"kind", "destination", "sha256"}:
            raise DataReadinessError("local release asset fields are invalid")
        kind = str(raw_asset.get("kind") or "")
        destination = _safe_relative(
            str(raw_asset.get("destination") or ""),
            "release asset destination",
        ).as_posix()
        sha256 = str(raw_asset.get("sha256") or "")
        _require_sha256(sha256, f"release asset {destination} sha256")
        asset_path = _safe_child(release_dir, destination)
        if not asset_path.is_file() or file_sha256(asset_path) != sha256:
            raise DataReadinessError(f"local release asset integrity failed: {destination}")
        normalized_assets.append(
            {"kind": kind, "destination": destination, "sha256": sha256}
        )
        kinds.setdefault(kind, []).append(destination)
    destinations = [asset["destination"] for asset in normalized_assets]
    if len(destinations) != len(set(destinations)):
        raise DataReadinessError("local release contains duplicate asset destinations")
    for kind in _REQUIRED_ASSET_KINDS:
        if len(kinds.get(kind, [])) != 1:
            raise DataReadinessError(f"local release requires exactly one {kind} asset")
    if set(kinds).difference(_REQUIRED_ASSET_KINDS | {"evidence_file"}):
        raise DataReadinessError("local release contains an unsupported asset kind")

    identity = {
        "schema": LOCAL_RELEASE_SCHEMA,
        "model_path": release.get("model_path"),
        "candidate_manifest_path": release.get("candidate_manifest_path"),
        "promotion_attestation_path": release.get("promotion_attestation_path"),
        "evidence_manifest_path": release.get("evidence_manifest_path"),
        "attestation_id": release.get("attestation_id"),
        "assets": normalized_assets,
        "generated_at_utc": release.get("generated_at_utc"),
    }
    generated_at_utc = str(release.get("generated_at_utc") or "")
    if not generated_at_utc:
        raise DataReadinessError("local release is missing generated_at_utc")
    try:
        generated = datetime.fromisoformat(generated_at_utc)
        _utc(generated)
    except ValueError as exc:
        raise DataReadinessError("local release generated_at_utc is invalid") from exc
    if _json_sha256(identity) != expected_release_id:
        raise DataReadinessError("local release content hash does not match its id")
    for field, kind in (
        ("model_path", "model"),
        ("candidate_manifest_path", "candidate_manifest"),
        ("promotion_attestation_path", "promotion_attestation"),
        ("evidence_manifest_path", "evidence_manifest"),
    ):
        if release.get(field) != kinds[kind][0]:
            raise DataReadinessError(f"local release {field} does not match its asset")

    model_path = _safe_child(release_dir, str(release["model_path"]))
    candidate_manifest_path = _safe_child(
        release_dir,
        str(release["candidate_manifest_path"]),
    )
    attestation_path = _safe_child(
        release_dir,
        str(release["promotion_attestation_path"]),
    )
    evidence_manifest_path = _safe_child(
        release_dir,
        str(release["evidence_manifest_path"]),
    )
    if candidate_manifest_path != manifest_path_for(model_path):
        raise DataReadinessError("candidate manifest is not adjacent to the released model")
    if attestation_path != promotion_attestation_path_for(model_path):
        raise DataReadinessError("promotion attestation is not adjacent to the released model")
    try:
        candidate = verify_model_artifact(
            model_path,
            allowed_statuses={MODEL_STATUS_PROMOTED},
        )
        attestation = verify_promotion_attestation(
            model_path,
            evidence_manifest_path=evidence_manifest_path,
            trust_store_path=attestation_trust_store_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise DataReadinessError("released candidate or attestation verification failed") from exc
    if attestation.get("attestation_id") != release.get("attestation_id"):
        raise DataReadinessError("release attestation identity mismatch")
    _verify_released_evidence(
        release_dir,
        evidence_manifest_path,
        normalized_assets,
        model_sha256=str(candidate["artifact_sha256"]),
        model_type=str(candidate["model_type"]),
    )
    return release


def _verify_released_evidence(
    release_dir: Path,
    evidence_manifest_path: Path,
    assets: list[dict[str, str]],
    *,
    model_sha256: str,
    model_type: str,
) -> None:
    evidence = _load_json_object(evidence_manifest_path, "released evidence manifest")
    _validate_evidence_schema(evidence, model_type=model_type)
    if evidence.get("model_artifact_sha256") != model_sha256:
        raise DataReadinessError("released evidence does not belong to the model")
    files = evidence.get("files")
    if not isinstance(files, dict):
        raise DataReadinessError("released evidence manifest has no file inventory")
    released_files = {
        asset["destination"]: asset["sha256"]
        for asset in assets
        if asset["kind"] == "evidence_file"
    }
    expected_files: dict[str, str] = {}
    evidence_root = evidence_manifest_path.parent
    for name, record in files.items():
        if not isinstance(record, dict):
            raise DataReadinessError(f"released evidence record is invalid: {name}")
        relative = _safe_relative(
            str(record.get("path") or ""),
            f"released evidence file {name}",
        )
        destination = f"evidence/{relative.as_posix()}"
        expected_sha = str(record.get("sha256") or "")
        _require_sha256(expected_sha, f"released evidence file {name} sha256")
        path = (evidence_root / Path(relative.as_posix())).resolve()
        if not path.is_relative_to(evidence_root) or not path.is_file():
            raise DataReadinessError(f"released evidence file is missing: {name}")
        if file_sha256(path) != expected_sha:
            raise DataReadinessError(f"released evidence integrity failed: {name}")
        expected_files[destination] = expected_sha
    if released_files != expected_files:
        raise DataReadinessError("release evidence inventory does not match its manifest")
    if not evidence_manifest_path.is_relative_to(release_dir.resolve()):
        raise DataReadinessError("released evidence manifest is outside the release")


def _load_active_pointer(path: Path) -> dict[str, Any]:
    pointer = _load_json_object(path, "active local release pointer")
    if set(pointer) != {
        "schema",
        "release_id",
        "release_manifest_sha256",
        "previous_release_id",
        "activated_at_utc",
        "pointer_sha256",
    }:
        raise DataReadinessError("active local release pointer fields are invalid")
    content = dict(pointer)
    pointer_sha = str(content.pop("pointer_sha256", ""))
    if content.get("schema") != ACTIVE_LOCAL_RELEASE_SCHEMA:
        raise DataReadinessError("active local release pointer schema mismatch")
    _require_sha256(str(content.get("release_id") or ""), "active release id")
    _require_sha256(
        str(content.get("release_manifest_sha256") or ""),
        "active release manifest sha256",
    )
    previous = content.get("previous_release_id")
    if previous is not None:
        _require_sha256(str(previous), "previous release id")
    try:
        _utc(datetime.fromisoformat(str(content.get("activated_at_utc") or "")))
    except ValueError as exc:
        raise DataReadinessError("active release activation time is invalid") from exc
    if _json_sha256(content) != pointer_sha:
        raise DataReadinessError("active local release pointer integrity failed")
    return pointer


def _validate_evidence_schema(
    evidence: dict[str, Any],
    *,
    model_type: str,
) -> None:
    expected = {
        "canonical_swing": "swing_training_evidence.v1",
        "canonical_intraday": "intraday_training_evidence.v1",
    }.get(model_type)
    if expected is None or evidence.get("schema") != expected:
        raise DataReadinessError("release evidence schema does not match the model type")


def _safe_relative(value: str, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
    ):
        raise DataReadinessError(f"{name} is unsafe")
    return path


def _safe_child(root: Path, relative: str) -> Path:
    path = _safe_relative(relative, "release path")
    root = root.resolve()
    unresolved = root / Path(path.as_posix())
    current = root
    for part in path.parts:
        current = current / part
        if current.exists() and _is_reparse_point(current):
            raise DataReadinessError("release path contains a symlink or reparse point")
    destination = unresolved.resolve()
    if not destination.is_relative_to(root):
        raise DataReadinessError("release path escapes its root")
    return destination


def _validate_release_directory(release_dir: Path, releases_root: Path) -> None:
    expected = releases_root.resolve() / release_dir.name
    if (
        not release_dir.is_dir()
        or _is_reparse_point(release_dir)
        or release_dir.resolve() != expected
    ):
        raise DataReadinessError("local release directory escapes its repository")


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    except FileNotFoundError:
        return False
    reparse_flag = getattr(os.stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & reparse_flag)


def _remove_staging_directory(staging: Path, releases_root: Path) -> None:
    resolved = staging.resolve()
    root = releases_root.resolve()
    if resolved.parent != root or not resolved.name.startswith(".staging-"):
        raise DataReadinessError("refusing to remove an unsafe release staging directory")
    shutil.rmtree(resolved)


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{name} is unavailable or invalid: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"{name} must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _copy_file_durable(source: Path, destination: Path) -> None:
    with source.open("rb") as source_handle, destination.open("xb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle, length=1024 * 1024)
        target_handle.flush()
        os.fsync(target_handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise DataReadinessError(f"{name} must be a SHA-256 digest")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("release timestamps must be timezone-aware")
    return value.astimezone(UTC)
