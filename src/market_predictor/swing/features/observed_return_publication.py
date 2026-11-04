"""Typed registration and bounded reading of canonically constructed return inputs.

The operator-owned manifest SHA is the trust root, as for trained model bundles.
Only the offline composition publisher writes this format. The API never accepts
frames, source mappings, or an observation declaration from its clients.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.path_integrity import verify_no_reparse_ancestry
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside

SCHEMA = "market_predictor.observed_return_feature_publication"
PROFILES = ("technical", "relationships", "news")
SOURCE_ROLES = ("stock_history", "raw_decision_bars", "benchmark_history", "membership_authority",
                "cohort", "strategy_contract", "training_feature_request", "news_source_index")
CLOSED = {"serving_eligible": False, "promotion_eligible": False, "actionable": False,
          "source_coverage_admitted": False}


class ObservedSourcePin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def relative_path(cls, value: str) -> str:
        if (not value or "\\" in value or Path(value).is_absolute() or PurePosixPath(value).as_posix() != value
                or any(part in (".", "..") for part in value.split("/"))):
            raise ValueError("source paths must be repository-relative canonical paths")
        return value


class ObservedCanonicalInput(ObservedSourcePin):
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ObservedReturnPublicationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of_utc: datetime
    history_start: datetime
    stock_history: ObservedCanonicalInput
    raw_decision_bars: ObservedCanonicalInput
    benchmark_history: ObservedCanonicalInput
    membership_authority: ObservedSourcePin
    cohort: ObservedSourcePin
    strategy_contract: ObservedSourcePin
    training_feature_request: ObservedSourcePin
    news_source_index: ObservedSourcePin
    model_runs: dict[Literal["technical", "relationships", "news"], ObservedSourcePin]
    configuration_evidence: ObservedSourcePin | None = None

    @field_validator("as_of_utc", "history_start")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("publication clocks must be timezone-aware")
        return value


def checked_file(root: Path, pin: ObservedSourcePin, *, maximum_bytes: int | None = None) -> Path:
    path = verify_no_reparse_ancestry(inside(root, pin.path), label="observed return input")
    if (not path.is_file() or (maximum_bytes is not None and path.stat().st_size > maximum_bytes)
            or file_sha256(path) != pin.sha256):
        raise DataReadinessError("observed return input is missing, oversized or changed")
    return path


def checked_object(root: Path, pin: ObservedSourcePin) -> dict[str, Any]:
    path = checked_file(root, pin, maximum_bytes=16 * 1024**2)
    return parse_strict_json_object(path.read_bytes(), label="observed return authority")


def bound_canonical_input(root: Path, pin: ObservedCanonicalInput, *, artifact_type: str,
                          maximum_bytes: int, maximum_rows: int) -> tuple[pd.DataFrame, dict[str, str]]:
    path = checked_file(root, pin, maximum_bytes=maximum_bytes)
    sidecar = manifest_path_for(path)
    checked_file(root, ObservedSourcePin(path=sidecar.relative_to(root).as_posix(), sha256=pin.manifest_sha256),
                 maximum_bytes=1024**2)
    parquet = cast(Any, pq).ParquetFile(path)
    expanded = sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.metadata.num_row_groups))
    if parquet.metadata.num_rows > maximum_rows or expanded > maximum_bytes:
        raise DataReadinessError("observed source exceeds row or expanded byte budget")
    frame, metadata = load_canonical_artifact(path, expected_type=artifact_type, allow_research=True)
    audits = metadata.get("audit")
    if (not isinstance(audits, list) or not audits
            or any(item.get("status") != "pass" or item.get("failures") != 0 for item in audits)):
        raise DataReadinessError("observed canonical source has no passing source audit")
    return frame, {pin.path: pin.sha256, sidecar.relative_to(root).as_posix(): pin.manifest_sha256}


def request_source_files(request: ObservedReturnPublicationRequest) -> dict[str, str]:
    result = {getattr(request, role).path: getattr(request, role).sha256 for role in SOURCE_ROLES}
    for role in ("stock_history", "raw_decision_bars", "benchmark_history"):
        pin = getattr(request, role)
        result[str(Path(pin.path).with_suffix(Path(pin.path).suffix + ".manifest.json")).replace("\\", "/")] = pin.manifest_sha256
    result.update({pin.path: pin.sha256 for pin in request.model_runs.values()})
    if request.configuration_evidence is not None:
        result[request.configuration_evidence.path] = request.configuration_evidence.sha256
    return result


def verify_observed_return_publication(*, root: Path, directory: Path, expected_manifest_sha256: str,
        as_of: datetime, profile: str, feature_names: list[str], run_manifest_sha256: str,
        maximum_bytes: int = 67_108_864, maximum_rows: int = 2_000) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read one operator-pinned generation; never run its research producer in serving."""
    root = root.resolve()
    directory = inside(root, directory)
    metadata_path = directory / "_manifest.json"
    metadata = checked_object(root, ObservedSourcePin(path=metadata_path.relative_to(root).as_posix(),
                                                     sha256=expected_manifest_sha256))
    if (metadata.get("schema") != SCHEMA or metadata.get("status") != "complete_observed_research_inputs"
            or any(metadata.get(key) is not value for key, value in CLOSED.items())
            or profile not in PROFILES or set(metadata.get("profiles", {})) != set(PROFILES)):
        raise DataReadinessError("observed publication scope, profiles or admission differs")
    request = ObservedReturnPublicationRequest.model_validate(metadata["request"])
    run_hashes: dict[str, str] = {key: value.sha256 for key, value in request.model_runs.items()}
    if (json_sha256(request.model_dump(mode="json")) != metadata.get("request_sha256")
            or set(request.model_runs) != set(PROFILES)
            or run_hashes[profile] != run_manifest_sha256):
        raise DataReadinessError("observed publication is not bound to the requested model")
    cutoff = pd.Timestamp(as_of)
    published = pd.Timestamp(metadata["created_at_utc"])
    if cutoff.tzinfo is None or published.tzinfo is None or published > cutoff or pd.Timestamp(request.as_of_utc) > cutoff:
        raise DataReadinessError("observed publication was unavailable at the requested cutoff")
    files = metadata["source_files"]
    required = request_source_files(request)
    if not isinstance(files, dict) or any(files.get(name) != digest for name, digest in required.items()):
        raise DataReadinessError("observed publication lacks its typed source roles")
    implementations = metadata.get("implementation_files")
    if not isinstance(implementations, dict) or not implementations:
        raise DataReadinessError("observed publication has no executed implementation identity")
    for name, digest in {**files, **implementations}.items():
        checked_file(root, ObservedSourcePin(path=name, sha256=digest))
    record = metadata["profiles"][profile]
    if record["feature_names"] != feature_names or record["path"] != f"{profile}.parquet":
        raise DataReadinessError("observed publication feature ordering differs from the frozen model")
    relative = (directory / record["path"]).relative_to(root).as_posix()
    pin = ObservedCanonicalInput(path=relative, sha256=record["sha256"], manifest_sha256=record["manifest_sha256"])
    frame, _ = bound_canonical_input(root, pin, artifact_type="swing_return_live_features",
                                    maximum_bytes=maximum_bytes, maximum_rows=maximum_rows)
    if (frame.empty or frame.decision_id.duplicated().any()
            or json_sha256(sorted(frame.decision_id.tolist())) != metadata["decision_ids_sha256"]
            or json_sha256(sorted(frame.security_id.tolist())) != metadata["security_ids_sha256"]):
        raise DataReadinessError("observed publication population changed")
    return frame, {"feature_path": str(inside(root, relative)), "feature_sha256": pin.sha256,
        "feature_manifest_sha256": pin.manifest_sha256, "publication_manifest_sha256": expected_manifest_sha256,
        "created_at_utc": metadata["created_at_utc"], "source_inputs": files,
        "peer_population": metadata["peer_population"], "implementation_files": implementations}
