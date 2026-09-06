"""Identity-only verification for historical intraday security namespaces."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.path_integrity import is_reparse_point
from market_predictor.universe.sp500.membership_authority import (
    load_sp500_membership_authority_envelope,
    verify_membership_namespace_extension,
)
from market_predictor.universe.sp500.observed_membership_authority import (
    AUTHORITY_SCHEMA as OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA,
)
from market_predictor.universe.sp500.observed_membership_authority import (
    load_observed_sp500_membership_authority,
)

_DATASET_SCHEMA: Final = "edge_rebuild.intraday_bar_dataset.v1"
_AUTHORITY_SCHEMA: Final = "edge_rebuild.intraday_bar_dataset_authority.v1"
_SECURITY_NAMESPACE_SCHEMA: Final = "edge_rebuild.a43_security_identity_namespace.v1"
_REQUEST_KEYS: Final = frozenset(
    {
        "benchmark_collection_directory",
        "decision_clock",
        "feature_schema_version",
        "five_minute_projection_directory",
        "intraday_contract_lineage_file_sha256",
        "intraday_contract_lineage_path",
        "label_schema_version",
        "maximum_session_workers",
        "membership_authority_directory",
        "memory_hard_budget_gib",
        "ordered_feature_names",
        "ordered_feature_sha256",
        "parent_lineage",
        "parent_lineage_sha256",
        "planned_sessions",
        "processing_unit",
        "request_sha256",
        "schema",
        "selection_directory",
        "stock_collection_directory",
        "stock_coverage_directory",
        "strategy_contract_path",
        "strategy_contract_sha256",
        "transformation",
        "transformation_sha256",
    }
)
_MANIFEST_KEYS: Final = _REQUEST_KEYS | {
    "created_at_utc",
    "session_unit_inventory_sha256",
    "session_units",
    "state",
    "summary",
    "training_contract",
}
_AUTHORITY_KEYS: Final = frozenset(
    {
        "artifact",
        "artifact_sha256",
        "request_sha256",
        "rows",
        "schema",
        "session_unit_inventory_sha256",
        "sessions",
        "state",
    }
)
_PARENT_LINEAGE_KEYS: Final = frozenset(
    {
        "benchmark_collection_authority_sha256",
        "benchmark_collection_manifest_sha256",
        "five_minute_canonical_authority_sha256",
        "five_minute_canonical_file_inventory_sha256",
        "five_minute_canonical_manifest_sha256",
        "five_minute_projection_authority_sha256",
        "five_minute_projection_inventory_sha256",
        "five_minute_projection_manifest_sha256",
        "intraday_contract_lineage_file_sha256",
        "intraday_data_contract_sha256",
        "intraday_parent_contract_sha256",
        "membership_authority_sha256",
        "membership_manifest_sha256",
        "membership_table_sha256",
        "selection_authority_sha256",
        "selection_manifest_sha256",
        "selection_table_sha256",
        "stock_collection_authority_sha256",
        "stock_collection_manifest_sha256",
        "stock_coverage_authority_sha256",
        "stock_coverage_manifest_sha256",
        "strategy_contract_file_sha256",
        "strategy_contract_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class HistoricalSecurityNamespaceIdentity:
    """Verified hashes that identify a historical security namespace."""

    intraday_bar_authority_sha256: str
    intraday_bar_manifest_sha256: str
    intraday_bar_request_sha256: str
    intraday_bar_parent_lineage_sha256: str
    security_identity_namespace_sha256: str

    def as_dict(self) -> dict[str, str]:
        return {
            "intraday_bar_authority_sha256": self.intraday_bar_authority_sha256,
            "intraday_bar_manifest_sha256": self.intraday_bar_manifest_sha256,
            "intraday_bar_request_sha256": self.intraday_bar_request_sha256,
            "intraday_bar_parent_lineage_sha256": self.intraday_bar_parent_lineage_sha256,
            "security_identity_namespace_sha256": self.security_identity_namespace_sha256,
        }


def verify_historical_security_namespace(
    dataset_directory: Path,
    *,
    current_membership_directory: Path,
    expected_intraday_bar_authority_sha256: str,
    expected_intraday_bar_manifest_sha256: str,
    expected_intraday_bar_request_sha256: str,
    expected_intraday_bar_parent_lineage_sha256: str,
    expected_security_identity_namespace_sha256: str,
    expected_current_membership_authority_sha256: str,
    expected_current_membership_manifest_sha256: str,
    expected_current_membership_table_sha256: str,
    expected_current_membership_universe_sha256: str,
    expected_current_membership_cutoff_date: str,
) -> HistoricalSecurityNamespaceIdentity:
    """Verify immutable identity lineage without authorizing historical bar rows."""

    expected_hashes = {
        "intraday bar authority": expected_intraday_bar_authority_sha256,
        "intraday bar manifest": expected_intraday_bar_manifest_sha256,
        "intraday bar request": expected_intraday_bar_request_sha256,
        "intraday bar parent lineage": expected_intraday_bar_parent_lineage_sha256,
        "security identity namespace": expected_security_identity_namespace_sha256,
        "current membership authority": expected_current_membership_authority_sha256,
        "current membership manifest": expected_current_membership_manifest_sha256,
        "current membership table": expected_current_membership_table_sha256,
        "current membership universe": expected_current_membership_universe_sha256,
    }
    for label, value in expected_hashes.items():
        _require_sha256(value, label=label)

    root = _verified_directory(dataset_directory, label="historical intraday dataset")
    request_path = _verified_metadata_file(root, "_request.json")
    manifest_path = _verified_metadata_file(root, "_manifest.json")
    authority_path = _verified_metadata_file(root, "_authority.json")
    request = _read_strict_object(request_path)
    manifest = _read_strict_object(manifest_path)
    authority = _read_strict_object(authority_path)
    _require_exact_keys(request, _REQUEST_KEYS, label="historical intraday request")
    _require_exact_keys(manifest, _MANIFEST_KEYS, label="historical intraday manifest")
    _require_exact_keys(authority, _AUTHORITY_KEYS, label="historical intraday authority")

    request_payload = {
        str(key): value for key, value in request.items() if key != "request_sha256"
    }
    request_sha256 = _json_sha256(request_payload)
    parent_lineage = request.get("parent_lineage")
    if not isinstance(parent_lineage, dict):
        raise DataReadinessError("historical intraday parent lineage is malformed")
    _require_exact_keys(
        parent_lineage,
        _PARENT_LINEAGE_KEYS,
        label="historical intraday parent lineage",
    )
    parent_lineage_sha256 = _json_sha256(parent_lineage)
    transformation = request.get("transformation")
    sessions = request.get("planned_sessions")
    units = manifest.get("session_units")
    summary = manifest.get("summary")
    completed_sessions = _nonnegative_int(
        summary.get("completed_sessions") if isinstance(summary, Mapping) else None,
        label="historical intraday completed session count",
    )
    summary_rows = _nonnegative_int(
        summary.get("rows") if isinstance(summary, Mapping) else None,
        label="historical intraday row count",
    )
    authority_sessions = _nonnegative_int(
        authority.get("sessions"),
        label="historical intraday authority session count",
    )
    authority_rows = _nonnegative_int(
        authority.get("rows"),
        label="historical intraday authority row count",
    )
    if (
        request.get("schema") != _DATASET_SCHEMA
        or request.get("request_sha256") != request_sha256
        or request_sha256 != expected_intraday_bar_request_sha256
        or request.get("parent_lineage_sha256") != parent_lineage_sha256
        or parent_lineage_sha256 != expected_intraday_bar_parent_lineage_sha256
        or not isinstance(transformation, Mapping)
        or transformation.get("sha256") != request.get("transformation_sha256")
        or not isinstance(sessions, list)
        or not sessions
        or sessions != sorted(set(str(value) for value in sessions))
        or manifest.get("schema") != _DATASET_SCHEMA
        or manifest.get("state") != "complete"
        or any(manifest.get(key) != value for key, value in request.items())
        or manifest.get("parent_lineage") != parent_lineage
        or manifest.get("parent_lineage_sha256") != parent_lineage_sha256
        or not isinstance(units, list)
        or len(units) != len(sessions)
        or manifest.get("session_unit_inventory_sha256") != _json_sha256(units)
        or not isinstance(summary, Mapping)
        or completed_sessions != len(sessions)
        or authority.get("schema") != _AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or authority.get("session_unit_inventory_sha256")
        != manifest.get("session_unit_inventory_sha256")
        or authority_sessions != len(sessions)
        or authority_rows != summary_rows
        or file_sha256(authority_path) != expected_intraday_bar_authority_sha256
        or file_sha256(manifest_path) != expected_intraday_bar_manifest_sha256
    ):
        raise DataReadinessError("historical intraday identity authority does not verify")

    base_membership_root = _verified_directory(
        Path(str(request.get("membership_authority_directory", ""))),
        label="base membership authority",
    )
    _verify_membership_metadata_files(base_membership_root)
    base_memberships, base_parent = load_sp500_membership_authority_envelope(
        base_membership_root
    )
    expected_base = {
        "membership_authority_sha256": base_parent["authority_sha256"],
        "membership_manifest_sha256": base_parent["manifest_sha256"],
        "membership_table_sha256": base_parent["membership_table_sha256"],
    }
    if any(parent_lineage.get(key) != value for key, value in expected_base.items()):
        raise DataReadinessError(
            "historical intraday membership lineage does not match its base authority"
        )

    current_root = _verified_directory(
        current_membership_directory,
        label="current membership authority",
    )
    current_memberships, current_parent = _load_current_membership(current_root)
    expected_current = {
        "authority_sha256": expected_current_membership_authority_sha256,
        "manifest_sha256": expected_current_membership_manifest_sha256,
        "membership_table_sha256": expected_current_membership_table_sha256,
        "universe_sha256": expected_current_membership_universe_sha256,
        "cutoff_date": expected_current_membership_cutoff_date,
    }
    if any(current_parent.get(key) != value for key, value in expected_current.items()):
        raise DataReadinessError("current membership authority identity changed")
    verify_membership_namespace_extension(
        base_memberships,
        current_memberships,
        base_cutoff_date=str(base_parent["cutoff_date"]),
        current_cutoff_date=str(current_parent["cutoff_date"]),
    )

    namespace_payload = {
        "schema": _SECURITY_NAMESPACE_SCHEMA,
        **expected_base,
        "membership_universe_sha256": base_parent["universe_sha256"],
    }
    namespace_sha256 = _json_sha256(namespace_payload)
    if namespace_sha256 != expected_security_identity_namespace_sha256:
        raise DataReadinessError("historical security identity namespace changed")
    return HistoricalSecurityNamespaceIdentity(
        intraday_bar_authority_sha256=expected_intraday_bar_authority_sha256,
        intraday_bar_manifest_sha256=expected_intraday_bar_manifest_sha256,
        intraday_bar_request_sha256=request_sha256,
        intraday_bar_parent_lineage_sha256=parent_lineage_sha256,
        security_identity_namespace_sha256=namespace_sha256,
    )


def _verified_directory(path: Path, *, label: str) -> Path:
    if is_reparse_point(path):
        raise DataReadinessError(f"{label} cannot be a symlink or reparse point")
    try:
        root = path.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError(f"{label} is unavailable") from exc
    if not root.is_dir():
        raise DataReadinessError(f"{label} is not a directory")
    return root


def _verified_metadata_file(root: Path, name: str) -> Path:
    path = root / name
    if path.parent != root or is_reparse_point(path):
        raise DataReadinessError("historical identity metadata path is unsafe")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError("historical identity metadata is unavailable") from exc
    if resolved.parent != root or not resolved.is_file():
        raise DataReadinessError("historical identity metadata escapes its authority")
    return resolved


def _verify_membership_metadata_files(root: Path) -> None:
    for name in ("_request.json", "_manifest.json", "_authority.json"):
        _read_strict_object(_verified_metadata_file(root, name))


def _load_current_membership(
    root: Path,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    _verify_membership_metadata_files(root)
    authority = _read_strict_object(_verified_metadata_file(root, "_authority.json"))
    if authority.get("schema") == OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA:
        _read_strict_object(_verified_metadata_file(root, "_status.json"))
        loaded = load_observed_sp500_membership_authority(root)
        return loaded.memberships, loaded.parent
    memberships, parent = load_sp500_membership_authority_envelope(root)
    return memberships, parent


def _read_strict_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key: {key}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> None:
        raise ValueError(f"non-finite value: {value}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite value: {value}")
        return parsed

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
            parse_float=parse_finite_float,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DataReadinessError("historical identity JSON is not strict") from exc
    if not isinstance(value, dict):
        raise DataReadinessError("historical identity JSON must be an object")
    return {str(key): item for key, item in value.items()}


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str] | set[str],
    *,
    label: str,
) -> None:
    if set(value) != set(expected):
        raise DataReadinessError(f"{label} schema differs")


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DataReadinessError(f"{label} SHA-256 is invalid")


def _nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise DataReadinessError(f"{label} is invalid")
    return value


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
