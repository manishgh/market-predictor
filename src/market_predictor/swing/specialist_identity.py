"""Content identity for the complete KS3 executable dependency closure."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

from market_predictor.canonical.store import file_sha256
from market_predictor.v3.errors import DataReadinessError

_SOURCE_FILES = (
    "src/market_predictor/canonical/audits.py",
    "src/market_predictor/canonical/reconciliation.py",
    "src/market_predictor/canonical/store.py",
    "src/market_predictor/execution_policy.py",
    "src/market_predictor/prediction_policy.py",
    "src/market_predictor/regime_evidence.py",
    "src/market_predictor/registry.py",
    "src/market_predictor/resources.py",
    "src/market_predictor/swing/contracts.py",
    "src/market_predictor/swing/evaluation.py",
    "src/market_predictor/swing/specialist_contracts.py",
    "src/market_predictor/swing/specialist_dataset.py",
    "src/market_predictor/swing/specialist_experiments.py",
    "src/market_predictor/swing/specialist_identity.py",
    "src/market_predictor/swing/specialist_model.py",
    "src/market_predictor/swing/strategy_labels.py",
    "src/market_predictor/v3/calibration.py",
    "src/market_predictor/v3/errors.py",
    "src/market_predictor/v3/validation.py",
)
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
    source_paths = tuple(repository_root / path for path in _SOURCE_FILES)
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
