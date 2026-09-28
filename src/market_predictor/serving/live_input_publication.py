"""Publish pinned nightly inputs through the existing immutable-generation reader."""
from __future__ import annotations

import gc
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Self
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core import path_integrity
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.governance.outcomes.repository import _write_json_durable
from market_predictor.governance.promotion.bundle_contracts import canonical_payload_sha256
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.monitoring_lease import monitoring_lease
from market_predictor.resources import assert_memory_budget
from market_predictor.serving.swing_features import (
    SWING_LIVE_INPUT_POINTER,
    SWING_LIVE_INPUT_POINTER_SCHEMA,
    SWING_LIVE_INPUT_SCHEMA_VERSION,
    FileSwingLiveInputProvider,
    _expected_swing_decision_time,
    _load_input_pointer,
    _strict_utc_series,
    _strict_utc_value,
    build_live_swing_features,
)
from market_predictor.swing.features.catalyst_decision_authority import load_catalyst_decision_authority

SHA = r"^[0-9a-f]{64}$"


class PinnedFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    path: Path
    sha256: str = Field(pattern=SHA)

    @field_validator("path")
    @classmethod
    def absolute_path(cls, value: Path) -> Path:
        if not value.is_absolute():
            raise ValueError("publication input paths must be absolute")
        return value


class CanonicalInput(PinnedFile):
    manifest_sha256: str = Field(pattern=SHA)


class LiveInputPublication(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    as_of_utc: datetime
    stock_daily_bars: CanonicalInput
    benchmark_daily_bars: CanonicalInput
    point_in_time_memberships: CanonicalInput
    catalyst_authority: PinnedFile
    strategy_contract: PinnedFile

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        if self.as_of_utc.utcoffset() is None:
            raise ValueError("publication cutoff must be timezone-aware")
        if self.catalyst_authority.path.name != "_authority.json":
            raise ValueError("catalyst pin must name _authority.json")
        return self


def _now() -> datetime:
    return datetime.now(UTC)


def _guard(stage: str) -> None:
    assert_memory_budget(hard_budget_gib=4.0, headroom_gib=0.5, stage=stage)


def _verify(pin: PinnedFile) -> Path:
    path = path_integrity.verify_no_reparse_ancestry(pin.path, label="live publication input")
    if path.is_file() and path.suffix.lower() in {".json", ".toml"} and path.stat().st_size > 1_048_576:
        raise DataReadinessError("publication metadata byte limit exceeded")
    if not path.is_file() or file_sha256(path) != pin.sha256:
        raise DataReadinessError("live publication input pin does not verify")
    return path


def _json(path: Path) -> dict[str, object]:
    path = path_integrity.verify_no_reparse_ancestry(path, label="publication JSON")
    if path.stat().st_size > 1_048_576:
        raise DataReadinessError("publication JSON exceeds byte limit")
    return parse_strict_json_object(path.read_bytes(), label="live publication manifest")


def _copy(source: Path, target: Path, expected: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.flush()
        os.fsync(writer.fileno())
    if file_sha256(target) != expected or file_sha256(source) != expected:
        raise DataReadinessError("publication source changed during copy")


def _pointer(generation: str, previous: str | None, activated: datetime) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": SWING_LIVE_INPUT_POINTER_SCHEMA, "generation_id": generation,
        "manifest_file_sha256": generation, "previous_generation_id": previous,
        "activated_at_utc": activated.isoformat(),
    }
    return {**payload, "pointer_sha256": canonical_payload_sha256(payload)}


def publish_live_inputs(
    request: LiveInputPublication, output_directory: Path, *,
    maximum_bytes: int = 512 * 1024 * 1024, maximum_rows: int = 2_000_000,
    runtime_dir: Path | None = None,
) -> dict[str, object]:
    """Validate the whole generation before publishing its pointer, under both leases."""
    if maximum_bytes < 1 or maximum_rows < 1:
        raise ValueError("publication limits must be positive")
    with heavy_job_lease("publish-swing-live-inputs", runtime_dir=runtime_dir):
        with monitoring_lease("publish-swing-live-inputs", runtime_dir=runtime_dir):
            return _publish(request, output_directory, maximum_bytes=maximum_bytes, maximum_rows=maximum_rows)


def _publish(request: LiveInputPublication, output_directory: Path, *, maximum_bytes: int, maximum_rows: int) -> dict[str, object]:
    _guard("before nightly publication")
    now = _now()
    cutoff = pd.Timestamp(request.as_of_utc).tz_convert("UTC")
    decision = _expected_swing_decision_time(cutoff)
    if cutoff > now or cutoff < decision or decision != _expected_swing_decision_time(pd.Timestamp(now)):
        raise DataReadinessError("publication must describe the current completed nightly decision, without future clocks")
    root = path_integrity.verify_no_reparse_ancestry(output_directory, label="live publication output").resolve()
    request_id = canonical_payload_sha256(request.model_dump(mode="json"))
    old = _load_input_pointer(root) if (root / SWING_LIVE_INPUT_POINTER).exists() else None
    if old is not None:
        old_manifest_path = root / "generations" / old["generation_id"] / "_manifest.json"
        _verify(PinnedFile(path=old_manifest_path, sha256=old["manifest_file_sha256"]))
        previous_manifest = _json(old_manifest_path)
        previous_request = previous_manifest.get("publication_request")
        if (not isinstance(previous_request, dict) or
                canonical_payload_sha256(previous_request) != previous_manifest.get("publication_request_sha256")):
            raise DataReadinessError("active publication request identity does not verify")
        old_cutoff = _strict_utc_value(previous_manifest.get("observation_cutoff_utc"), "published observation cutoff")
        if cutoff < old_cutoff or pd.Timestamp(old["activated_at_utc"]) > now:
            raise DataReadinessError("publication cannot regress or precede the active generation")
    else:
        previous_manifest = {}
    pins = {name: getattr(request, name) for name in (
        "stock_daily_bars", "benchmark_daily_bars", "point_in_time_memberships",
    )}
    strategy_path = _verify(request.strategy_contract)
    contract = load_strategy_contract(strategy_path)
    authority_path = _verify(request.catalyst_authority)
    authority_root = path_integrity.verify_tree_containment(authority_path.parent, label="publication catalyst authority")
    source_files: dict[Path, str] = {strategy_path: request.strategy_contract.sha256}
    for name, pin in pins.items():
        source = _verify(pin)
        manifest_path = _verify(PinnedFile(path=manifest_path_for(source), sha256=pin.manifest_sha256))
        manifest = _json(manifest_path)
        audits = manifest.get("audit")
        if (not isinstance(audits, list) or not audits or any(not isinstance(item, dict) or item.get("status") != "pass"
                                                           or item.get("failures") != 0 for item in audits)):
            raise DataReadinessError(f"{name} canonical audit does not verify")
        if manifest.get("production_ready") is not True:
            raise DataReadinessError(f"{name} canonical input is not production-ready")
        source_files[source], source_files[manifest_path] = pin.sha256, pin.manifest_sha256
    for path in authority_root.iterdir():
        if not path.is_file():
            raise DataReadinessError("catalyst authority must have a flat immutable inventory")
        if path.suffix.lower() == ".json" and path.stat().st_size > 1_048_576:
            raise DataReadinessError("publication authority metadata byte limit exceeded")
        source_files[path] = file_sha256(path)
    authority_manifest = _json(authority_root / "_manifest.json")
    authority_request = authority_manifest.get("request")
    identity = authority_request.get("canonical_decisions") if isinstance(authority_request, dict) else None
    if identity is not None:
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256", "manifest_sha256"}:
            raise DataReadinessError("canonical catalyst decision pin is malformed")
        identity_path = _verify(PinnedFile(path=Path(str(identity["path"])), sha256=str(identity["sha256"])))
        identity_manifest = _verify(PinnedFile(path=manifest_path_for(identity_path), sha256=str(identity["manifest_sha256"])))
        source_files[identity_path] = str(identity["sha256"])
        source_files[identity_manifest] = str(identity["manifest_sha256"])
    if sum(path.stat().st_size for path in source_files) > maximum_bytes:
        raise DataReadinessError("publication aggregate byte limit exceeded")
    rows, expanded_bytes = 0, 0
    for path in source_files:
        if path.suffix != ".parquet":
            continue
        parquet = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
        rows += int(parquet.metadata.num_rows)
        expanded_bytes += sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.metadata.num_row_groups))
    if rows > maximum_rows or expanded_bytes > maximum_bytes:
        raise DataReadinessError("publication parquet row or expanded byte limit exceeded")
    # Staging has the reader's exact repository shape; validation never activates it.
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".publication-{uuid4().hex}"
    payload = staging / "payload"
    payload.mkdir(parents=True)
    try:
        files: dict[str, object] = {}
        watermarks: dict[str, str] = {}
        for name, pin in pins.items():
            target = payload / f"{name}.parquet"
            _copy(pin.path, target, pin.sha256)
            _copy(manifest_path_for(pin.path), manifest_path_for(target), pin.manifest_sha256)
            _guard(f"before canonical {name} load")
            frame, _ = load_canonical_artifact(target, expected_type="memberships" if name == "point_in_time_memberships" else "bars")
            _guard(f"after canonical {name} load")
            available = _strict_utc_series(frame["available_at_utc"], f"{name} availability")
            if frame.empty or bool(available.gt(cutoff).any()):
                raise DataReadinessError(f"{name} contains empty or future source evidence")
            watermark_key = "membership_available_at_utc" if name == "point_in_time_memberships" else f"{name}_available_at_utc"
            watermarks[watermark_key] = available.max().isoformat()
            files[name] = {"path": target.name, "sha256": pin.sha256, "rows": len(frame)}
            del frame
            gc.collect()
        for path in authority_root.iterdir():
            _copy(path, payload / "catalyst" / path.name, source_files[path])
        _guard("before publication catalyst verification")
        authority = load_catalyst_decision_authority(
            payload / "catalyst", require_production_ready=True, expected_authority_sha256=request.catalyst_authority.sha256,
        )
        _guard("after publication catalyst verification")
        alpaca = authority.coverage.loc[authority.coverage["source_family"].eq("alpaca")]
        completed = _strict_utc_series(alpaca["completed_at_utc"], "Alpaca collection completion")
        if alpaca.empty or bool(completed.gt(cutoff).any()):
            raise DataReadinessError("Alpaca collection coverage is empty or from the future")
        watermarks["alpaca_news_available_at_utc"] = completed.max().isoformat()
        del authority
        gc.collect()
        manifest = {
            "schema": SWING_LIVE_INPUT_SCHEMA_VERSION, "state": "complete", "generated_at_utc": now.isoformat(),
            "observation_cutoff_utc": cutoff.isoformat(), "publication_request_sha256": request_id,
            "publication_request": request.model_dump(mode="json"),
            "market_data_provider": "alpaca", "market_data_feed": "sip", "market_data_adjustment": "all",
            "files": files, "catalyst_authority_directory": "catalyst",
            "catalyst_authority_sha256": request.catalyst_authority.sha256, "source_watermarks": watermarks,
        }
        _write_json_durable(payload / "_manifest.json", manifest)
        generation_id = file_sha256(payload / "_manifest.json")
        staged_generation = staging / "generations" / generation_id
        staged_generation.parent.mkdir()
        os.replace(payload, staged_generation)
        _write_json_durable(staging / SWING_LIVE_INPUT_POINTER, _pointer(generation_id, None, now))
        inputs = FileSwingLiveInputProvider(staging).load(as_of_utc=now, maximum_bytes=maximum_bytes, maximum_rows=maximum_rows)
        features = build_live_swing_features(
            inputs.stock_daily_bars, inputs.benchmark_daily_bars, inputs.point_in_time_memberships,
            contract=contract, catalyst_authority_directory=inputs.catalyst_authority_directory,
            expected_catalyst_authority_sha256=inputs.catalyst_authority_sha256,
            live_manifest_path=inputs.manifest_path, expected_live_manifest_sha256=inputs.manifest_sha256,
            as_of_utc=cutoff, memory_budget_gib=4.0, memory_headroom_gib=0.5,
        )
        if features.decision_time_utc != decision:
            raise DataReadinessError("publication feature decision differs from the requested nightly session")
        del inputs, features
        gc.collect()
        for source, digest in source_files.items():
            if file_sha256(source) != digest:
                raise DataReadinessError("publication source changed during validation")
        _guard("before live generation activation")
        if previous_manifest.get("publication_request_sha256") == request_id and old is not None:
            for name in ("files", "catalyst_authority_sha256", "source_watermarks", "publication_request"):
                if previous_manifest.get(name) != manifest[name]:
                    raise DataReadinessError("active generation differs from its pinned publication request")
            # Verify stored files afresh rather than allowing a cached retry to bless tampering.
            existing_inputs = FileSwingLiveInputProvider(root).load(
                as_of_utc=_now(), maximum_bytes=maximum_bytes, maximum_rows=maximum_rows,
            )
            load_catalyst_decision_authority(existing_inputs.catalyst_authority_directory, require_production_ready=True,
                                            expected_authority_sha256=existing_inputs.catalyst_authority_sha256)
            return {"generation_id": old["generation_id"], "activated_at_utc": old["activated_at_utc"], "reused": True}
        generations = path_integrity.verify_no_reparse_ancestry(root / "generations", label="live publication generations")
        generations.mkdir(exist_ok=True)
        destination = generations / generation_id
        if destination.exists():
            raise DataReadinessError("unreferenced generation already exists; refusing to replace immutable evidence")
        os.replace(staged_generation, destination)
        activated = _now()
        if activated < now:
            raise DataReadinessError("publication clock regressed before activation")
        _write_json_durable(root / SWING_LIVE_INPUT_POINTER, _pointer(generation_id, old["generation_id"] if old else None, activated))
        return {"generation_id": generation_id, "activated_at_utc": activated.isoformat(), "reused": False}
    finally:
        # Only this invocation's verified staging child can be removed, never generations.
        safe = path_integrity.verify_tree_containment(staging, label="publication staging cleanup")
        if safe.parent == root and safe.name.startswith(".publication-"):
            shutil.rmtree(safe)
