"""Content identity for the complete KS3 executable dependency closure."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from market_predictor.canonical.store import file_sha256
from market_predictor.v3.errors import DataReadinessError

_RUNTIME_PACKAGES = (
    "joblib",
    "numpy",
    "pandas",
    "pyarrow",
    "scikit-learn",
    "xgboost",
)


def specialist_implementation_identity() -> dict[str, object]:
    repository_root = Path(__file__).resolve().parents[3]
    source_root = repository_root / "src" / "market_predictor"
    source_paths = tuple(
        sorted(
            (
                path
                for path in source_root.rglob("*")
                if path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix not in {".pyc", ".pyo"}
            ),
            key=lambda path: path.as_posix(),
        )
    )
    lock_paths = (
        repository_root / "pyproject.toml",
        *sorted((repository_root / "requirements").glob("*.lock")),
    )
    missing = [
        str(path)
        for path in (*source_paths, *lock_paths)
        if not path.is_file()
    ]
    if missing:
        raise DataReadinessError(
            "KS3 implementation identity is missing files: "
            + ", ".join(missing)
        )
    if not source_paths:
        raise DataReadinessError("KS3 implementation source tree is empty")
    identity: dict[str, object] = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "runtime_versions": {
            package: importlib.metadata.version(package)
            for package in _RUNTIME_PACKAGES
        },
        "source_sha256": {
            path.relative_to(repository_root).as_posix(): file_sha256(path)
            for path in source_paths
        },
        "dependency_manifest_sha256": {
            path.relative_to(repository_root).as_posix(): file_sha256(path)
            for path in lock_paths
        },
    }
    identity["implementation_sha256"] = hashlib.sha256(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return identity
