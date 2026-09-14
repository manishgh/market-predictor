"""Bounded, hash-pinned source archives for provenance only, never execution."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from market_predictor.core.errors import DataReadinessError

_MAX_FILE_BYTES = 2 * 1024 * 1024
_MAX_TOTAL_BYTES = 32 * 1024 * 1024
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_FILES = 512
_MANIFEST_NAME = "_manifest.json"


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise DataReadinessError("implementation snapshot requires an explicit lowercase SHA256")
    return value


def _declarations(files: Mapping[str, str], *, allow_empty: bool = False) -> dict[str, str]:
    if (not files and not allow_empty) or len(files) > _MAX_FILES:
        raise DataReadinessError("implementation snapshot requires a bounded, nonempty file mapping")
    result: dict[str, str] = {}
    folded: dict[str, str] = {}
    for name, digest in files.items():
        if not isinstance(name, str) or len(name) > 512:
            raise DataReadinessError("invalid implementation source path")
        name = name.replace("\\", "/")
        path = PurePosixPath(name)
        if (
            path.as_posix() != name
            or path.parts[:2] != ("src", "market_predictor")
            or len(path.parts) < 3
            or path.name.casefold() == "__init__.py"
            or re.fullmatch(r"[A-Za-z0-9_]+\.py", path.name) is None
            or any(re.fullmatch(r"[A-Za-z0-9_]+", part) is None for part in path.parts[2:-1])
        ):
            raise DataReadinessError(f"invalid or conflicting implementation source path: {name}")
        digest = _sha(digest)
        previous = folded.get(name.casefold())
        if previous is not None and (previous != name or result[previous] != digest):
            raise DataReadinessError(f"conflicting implementation source path: {name}")
        folded[name.casefold()] = name
        result[name] = digest
    return dict(sorted(result.items()))


def _no_links(path: Path) -> None:
    for component in (*reversed(path.parents), path):
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or (
            getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise DataReadinessError(f"implementation snapshot forbids symlinks/reparse points: {component}")


def _root(root: Path) -> Path:
    root = root.absolute()
    _no_links(root)
    if not root.is_dir():
        raise DataReadinessError(f"implementation snapshot root is missing: {root}")
    return root.resolve()


def _inside(root: Path, path: Path) -> Path:
    if ".." in path.parts:
        raise DataReadinessError(f"implementation snapshot path escapes root: {path}")
    path = path if path.is_absolute() else root / path
    if not path.is_relative_to(root):
        raise DataReadinessError(f"implementation snapshot path escapes root: {path}")
    _no_links(path)
    if not path.resolve().is_relative_to(root):
        raise DataReadinessError(f"implementation snapshot path escapes root: {path}")
    return path


def _output(root: Path, output: Path) -> Path:
    output = _inside(root, output)
    evidence = root / "data" / "evidence"
    if output == evidence or not output.is_relative_to(evidence):
        raise DataReadinessError("implementation snapshot output must be below root/data/evidence")
    return output


def _read(path: Path, limit: int) -> bytes:
    _no_links(path)
    try:
        if not path.is_file() or path.stat().st_size > limit:
            raise DataReadinessError(f"implementation snapshot file missing or exceeds byte bound: {path}")
        with path.open("rb") as stream:
            payload = stream.read(limit + 1)
    except OSError as exc:
        raise DataReadinessError(f"implementation snapshot file unreadable: {path}") from exc
    if len(payload) > limit:
        raise DataReadinessError(f"implementation snapshot exceeds byte bound: {path}")
    return payload


def _manifest_bytes(files: Mapping[str, str]) -> bytes:
    # Exact canonical bytes reject extra/duplicate keys and alternate blob paths.
    value = {
        "schema_version": 1,
        "purpose": "implementation_provenance_only",
        "files": {
            name: {"sha256": digest, "blob": f"objects/{digest}.bin"}
            for name, digest in files.items()
        },
    }
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")
    if len(payload) > _MAX_MANIFEST_BYTES:
        raise DataReadinessError("implementation snapshot manifest exceeds byte bound")
    return payload


def _manifest_files(payload: bytes) -> dict[str, str]:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
            raise ValueError("missing file mapping")
        files: dict[str, str] = {}
        for name, entry in value["files"].items():
            if not isinstance(entry, dict):
                raise ValueError("invalid file entry")
            files[name] = _sha(entry.get("sha256"))
        declared = _declarations(files)
        if payload != _manifest_bytes(declared):
            raise ValueError("noncanonical manifest")
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise DataReadinessError("invalid implementation snapshot manifest") from exc
    return declared


def _check_existing(path: Path, payload: bytes) -> None:
    _no_links(path)
    if path.exists() and _read(path, len(payload)) != payload:
        raise DataReadinessError(f"immutable implementation snapshot conflict: {path}")


def _write_once(path: Path, payload: bytes) -> None:
    _no_links(path)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
    except FileExistsError:
        _check_existing(path, payload)


def create_implementation_snapshot(root: Path, files: Mapping[str, str], output: Path) -> dict[str, str]:
    """Archive declared source bytes; return root-relative manifest_path and manifest_sha256.

    Output is a directory below data/evidence. Existing artifacts must match byte
    for byte. The manifest is written last; incomplete writes never verify.
    """
    root = _root(root)
    declared = _declarations(files)
    output = _output(root, output)
    manifest = output / _MANIFEST_NAME
    manifest_bytes = _manifest_bytes(declared)
    _check_existing(manifest, manifest_bytes)
    objects: dict[str, bytes] = {}
    total = 0
    for name, digest in declared.items():
        payload = _read(_inside(root, Path(name)), min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - total))
        total += len(payload)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise DataReadinessError(f"implementation source SHA256 mismatch: {name}")
        if digest in objects and objects[digest] != payload:
            raise DataReadinessError(f"implementation source hash conflict: {name}")
        objects[digest] = payload
    for digest, payload in objects.items():
        _check_existing(output / "objects" / f"{digest}.bin", payload)
    (output / "objects").mkdir(parents=True, exist_ok=True)
    for digest, payload in objects.items():
        _write_once(output / "objects" / f"{digest}.bin", payload)
    _write_once(manifest, manifest_bytes)
    return {
        "manifest_path": manifest.relative_to(root).as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }


def verify_implementation_snapshot(
    root: Path, manifest: Path, expected_sha256: str, files: Mapping[str, str],
) -> dict[str, str]:
    """Verify a pinned archive and requested subset; return manifest and blob pins.

    Current source files need not exist or match: archives are provenance, not an
    alternative implementation. No archived content is decoded, imported or run.
    """
    root = _root(root)
    expected_sha256 = _sha(expected_sha256)
    requested = _declarations(files, allow_empty=True)
    manifest = _inside(root, manifest)
    output = _output(root, manifest.parent)
    if manifest.name != _MANIFEST_NAME:
        raise DataReadinessError("invalid implementation snapshot manifest path")
    payload = _read(manifest, _MAX_MANIFEST_BYTES)
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DataReadinessError("implementation snapshot manifest SHA256 mismatch")
    declared = _manifest_files(payload)
    if any(declared.get(name) != digest for name, digest in requested.items()):
        raise DataReadinessError("implementation snapshot does not match requested declared files")
    result = {manifest.relative_to(root).as_posix(): expected_sha256}
    requested_hashes = set(requested.values())
    total = 0
    for digest in declared.values():
        blob = _inside(root, output / "objects" / f"{digest}.bin")
        payload = _read(blob, min(_MAX_FILE_BYTES, _MAX_TOTAL_BYTES - total))
        total += len(payload)
        if hashlib.sha256(payload).hexdigest() != digest:
            raise DataReadinessError(f"implementation snapshot blob SHA256 mismatch: {blob}")
        if digest in requested_hashes:
            result[blob.relative_to(root).as_posix()] = digest
    return result
