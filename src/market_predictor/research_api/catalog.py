"""Server-owned pins for the six completed research return specifications."""
from __future__ import annotations

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.io import inside
from market_predictor.swing.training.return_artifacts import verify_unit

CATALOG_SCHEMA = "market_predictor.research_model_catalog"
FAMILIES = {"ridge": "regularized_linear_return", "xgboost": "shallow_boosted_return"}
PROFILES = {"technical": "existing_technical", "relationships": "technical_relationships", "news": "technical_relationships_news"}
TARGET = "spy_fixed_horizon_excess_return"
FLAGS = ("serving_eligible", "promotion_eligible", "portfolio_evaluated", "outer_validation_opened", "historical_test_opened")


@dataclass(frozen=True, slots=True)
class ResearchModelSpec:
    model_id: str
    label: str
    profile: str
    family: str
    artifact_directory: Path
    manifest_sha256: str


def verified_bytes(path: Path, digest: str, *, limit: int = 64 * 1024**2) -> bytes:
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise DataReadinessError("research artifact has no valid SHA256 pin")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit or hashlib.sha256(content).hexdigest() != digest:
        raise DataReadinessError("research artifact is oversized or differs from its pinned bytes")
    return content


def pinned_object(path: Path, digest: str) -> dict[str, Any]:
    return parse_strict_json_object(verified_bytes(path, digest, limit=4 * 1024**2), label="research artifact")


def load_research_model_specs(catalog_path: Path, *, repository_root: Path) -> dict[str, ResearchModelSpec]:
    with catalog_path.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema") != CATALOG_SCHEMA or set(payload) != {"schema", "profiles"}:
        raise ValueError("research catalog must contain the current return profiles")
    specs = {}
    seen = set()
    for profile in payload["profiles"]:
        name = profile["id"]
        if name not in PROFILES or name in seen or not profile["label"]:
            raise ValueError("research catalog profile is invalid or duplicated")
        relative = Path(profile["artifact_directory"])
        digest = profile["manifest_sha256"]
        if relative.is_absolute() or ".." in relative.parts or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("research catalog artifact path or pin is invalid")
        seen.add(name)
        for learner, family in FAMILIES.items():
            model_id = f"{name}_{learner}"
            specs[model_id] = ResearchModelSpec(model_id, f"{profile['label']} / {learner}", name, family,
                inside(repository_root, relative), digest)
    if seen != set(PROFILES):
        raise ValueError("research catalog requires exactly three return profiles")
    return specs


def load_run(spec: ResearchModelSpec) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = pinned_object(spec.artifact_directory / "_manifest.json", spec.manifest_sha256)
    request = pinned_object(spec.artifact_directory / "_request.json", manifest["request_sha256"])
    if (manifest.get("schema") != "market_predictor.swing_return_training"
            or manifest.get("status") != "complete_research_only"
            or request.get("schema") != "market_predictor.swing_return_training_request"
            or any(record.get(flag) is not False for record in (manifest, request) for flag in FLAGS)
            or request["policy"]["target"] != TARGET
            or request["policy"]["feature_profile"] != PROFILES[spec.profile]):
        raise DataReadinessError("completed research return scope or profile differs")
    expected = {f"{family}/{scope}/fold-{fold}" for family in FAMILIES.values()
                for scope in ("temporal", "security_transfer") for fold in range(1, 5)}
    expected |= {f"{family}/final_refit" for family in FAMILIES.values()}
    if set(manifest["units"]) != expected or [fold["number"] for fold in request["folds"]] != [1, 2, 3, 4]:
        raise DataReadinessError("research return run has an incomplete fit inventory")
    return manifest, request


def load_unit(spec: ResearchModelSpec, manifest: dict[str, Any], request: dict[str, Any],
              key: str) -> tuple[Path, dict[str, Any]]:
    record = manifest["units"][key]
    directory = inside(spec.artifact_directory, key)
    if not directory.as_posix().endswith("/" + record["path"]):
        raise DataReadinessError("research unit directory differs from the recorded path")
    unit_manifest = pinned_object(directory / "_manifest.json", record["manifest_sha256"])
    unit = unit_manifest["unit"]
    if (unit["family"] != spec.family or unit["scope"] != ("final_refit" if key.endswith("final_refit") else "temporal")
            or unit["feature_names"] != request["feature_names"] or unit["target"] != TARGET):
        raise DataReadinessError("research unit family, scope, target or feature order differs")
    verified = verify_unit(directory, manifest_sha256=record["manifest_sha256"],
        request_sha256=manifest["request_sha256"], unit=unit)
    return directory, verified


def model_state_record(spec: ResearchModelSpec) -> dict[str, Any]:
    state: dict[str, Any] = {"id": spec.model_id, "label": spec.label, "family": spec.family,
        "profile": spec.profile, "training_status": "unavailable", "integrity_verified": False,
        "historical_predictions_available": False, "live_scoring_available": False,
        "serving_eligible": False, "promotion_eligible": False, "actionable": False,
        "manifest_sha256": spec.manifest_sha256, "target": TARGET, "horizon_sessions": 10}
    try:
        manifest, request = load_run(spec)
        _, final = load_unit(spec, manifest, request, f"{spec.family}/final_refit")
        state.update(training_status=manifest["status"], integrity_verified=True,
            historical_predictions_available=True, feature_count=len(request["feature_names"]),
            final_model_sha256=final["files"]["model.joblib"],
            validation_start=request["folds"][0]["score"][0], validation_end=request["folds"][-1]["score"][-1],
            reason="Training completed. Saved temporal validation predictions are available; live release acceptance is pending.")
    except (OSError, ValueError, KeyError, TypeError, DataReadinessError):
        state["reason"] = "Pinned training artifacts are missing, modified or inconsistent."
    return state
