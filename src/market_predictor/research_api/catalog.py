from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

CATALOG_SCHEMA = "market_predictor.research_model_catalog.v1"
EXPECTED_MODEL_IDS = frozenset(
    {
        "swing_event_driven",
        "swing_baseline",
        "intraday_event_driven",
        "intraday_baseline",
    }
)


@dataclass(frozen=True, slots=True)
class ResearchModelSpec:
    model_id: str
    label: str
    mode: Literal["swing", "intraday"]
    uses_catalyst: bool
    artifact_directory: Path


@dataclass(frozen=True, slots=True)
class ResearchModelState:
    spec: ResearchModelSpec
    training_status: str
    artifact_available: bool
    integrity_verified: bool
    promotion_permitted: bool
    candidate_id: str | None
    reason: str
    artifact_sha256: str | None

    @property
    def research_scoring_available(self) -> bool:
        return (
            self.training_status == "candidate"
            and self.artifact_available
            and self.integrity_verified
        )


def load_research_model_specs(
    catalog_path: Path,
    *,
    repository_root: Path,
) -> dict[str, ResearchModelSpec]:
    with catalog_path.open("rb") as handle:
        payload = tomllib.load(handle)
    if payload.get("schema") != CATALOG_SCHEMA:
        raise ValueError("research model catalog schema is invalid")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("research model catalog must define a model list")

    specs: dict[str, ResearchModelSpec] = {}
    combinations: set[tuple[str, bool]] = set()
    root = repository_root.resolve()
    for raw in raw_models:
        if not isinstance(raw, dict):
            raise ValueError("research model catalog contains an invalid model")
        model_id = str(raw.get("id", "")).strip()
        label = str(raw.get("label", "")).strip()
        mode = str(raw.get("mode", "")).strip().lower()
        uses_catalyst = raw.get("uses_catalyst")
        relative_directory = Path(str(raw.get("artifact_directory", "")).strip())
        if (
            model_id not in EXPECTED_MODEL_IDS
            or not label
            or mode not in {"swing", "intraday"}
            or not isinstance(uses_catalyst, bool)
            or relative_directory.is_absolute()
            or ".." in relative_directory.parts
        ):
            raise ValueError(f"research model catalog entry is invalid: {model_id or '<missing>'}")
        if model_id in specs:
            raise ValueError(f"duplicate research model id: {model_id}")
        combination = (mode, uses_catalyst)
        if combination in combinations:
            raise ValueError("research model catalog duplicates a mode/catalyst combination")
        combinations.add(combination)
        specs[model_id] = ResearchModelSpec(
            model_id=model_id,
            label=label,
            mode=cast(Literal["swing", "intraday"], mode),
            uses_catalyst=uses_catalyst,
            artifact_directory=root / relative_directory,
        )
    if set(specs) != EXPECTED_MODEL_IDS:
        missing = sorted(EXPECTED_MODEL_IDS.difference(specs))
        raise ValueError(f"research model catalog is incomplete: {missing}")
    return specs


def inspect_research_model(spec: ResearchModelSpec) -> ResearchModelState:
    model_card_path = spec.artifact_directory / "model_card.json"
    candidate_path = spec.artifact_directory / "candidate.joblib"
    manifest_path = spec.artifact_directory / "_manifest.json"
    if not model_card_path.is_file():
        return ResearchModelState(
            spec=spec,
            training_status="missing",
            artifact_available=False,
            integrity_verified=False,
            promotion_permitted=False,
            candidate_id=None,
            reason="Training output is missing its model card.",
            artifact_sha256=None,
        )

    card = _read_json(model_card_path, "model card")
    status = str(card.get("status", "missing"))
    promotion_permitted = card.get("promotion_permitted") is True
    candidate_id_raw = card.get("candidate_id")
    candidate_id = str(candidate_id_raw) if candidate_id_raw else None
    if status == "no_candidate":
        return ResearchModelState(
            spec=spec,
            training_status=status,
            artifact_available=False,
            integrity_verified=manifest_path.is_file(),
            promotion_permitted=False,
            candidate_id=None,
            reason="Training completed, but no candidate passed the configured validation gates.",
            artifact_sha256=None,
        )
    if status != "candidate" or not candidate_path.is_file() or not manifest_path.is_file():
        return ResearchModelState(
            spec=spec,
            training_status=status,
            artifact_available=candidate_path.is_file(),
            integrity_verified=False,
            promotion_permitted=promotion_permitted,
            candidate_id=candidate_id,
            reason="Candidate files are incomplete or inconsistent with the model card.",
            artifact_sha256=None,
        )

    manifest = _read_json(manifest_path, "model manifest")
    files = manifest.get("files")
    candidate_record = files.get("candidate.joblib") if isinstance(files, dict) else None
    expected_sha256 = (
        str(candidate_record.get("sha256", ""))
        if isinstance(candidate_record, dict)
        else ""
    )
    actual_sha256 = _file_sha256(candidate_path)
    integrity_verified = expected_sha256 == actual_sha256
    return ResearchModelState(
        spec=spec,
        training_status=status,
        artifact_available=True,
        integrity_verified=integrity_verified,
        promotion_permitted=promotion_permitted,
        candidate_id=candidate_id,
        reason=(
            "Candidate is available for non-actionable research scoring."
            if integrity_verified
            else "Candidate artifact hash does not match its manifest."
        ),
        artifact_sha256=actual_sha256 if integrity_verified else None,
    )


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
