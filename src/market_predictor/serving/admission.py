"""Current product admission, separate from historical artifact verification."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.registry import MODEL_STATUS_PROMOTED, verify_model_artifact
from market_predictor.release import verify_local_release
from market_predictor.serving.bundle import verify_serving_bundle
from market_predictor.swing.contracts import SWING_MODEL_TYPE


def require_swing_candidate(model: Path, trust_store: Path) -> None:
    """Check provenance and product scope before publishing any new release."""
    candidate = verify_model_artifact(
        model,
        allowed_statuses={MODEL_STATUS_PROMOTED},
        attestation_trust_store_path=trust_store,
    )
    _require_swing_model(candidate)


def require_swing_release(root: Path, release_id: str, trust_store: Path) -> None:
    """Bind admission to the exact verified release, not mutable source files."""
    release = verify_local_release(
        root,
        release_id,
        attestation_trust_store_path=trust_store,
    )
    relative = str(release["candidate_manifest_path"])
    assets = [asset for asset in release["assets"] if asset["kind"] == "candidate_manifest" and asset["destination"] == relative]
    if len(assets) != 1:
        raise DataReadinessError("release must bind exactly one candidate manifest")
    # The verifier checks containment and all assets. Rehash the same bytes we
    # parse so a replacement after verification cannot change product admission.
    payload = (root.resolve() / "releases" / release_id / relative).read_bytes()
    if hashlib.sha256(payload).hexdigest() != assets[0]["sha256"]:
        raise DataReadinessError("candidate manifest changed during release admission")
    try:
        candidate = parse_strict_json_object(payload, label="released candidate manifest")
    except ValueError as exc:
        raise DataReadinessError(str(exc)) from exc
    _require_swing_model(candidate)


def require_swing_bundle(root: Path, bundle_id: str, trust_store: Path) -> None:
    """Historical verification alone cannot authorize a retired product route."""
    bundle = verify_serving_bundle(
        root,
        bundle_id,
        attestation_trust_store_path=trust_store,
    )
    if bundle["mode"] != "swing" or bundle["horizon"] != "10b":
        raise DataReadinessError("production serving requires swing with horizon 10b")


def _require_swing_model(candidate: Mapping[str, Any]) -> None:
    if candidate.get("model_type") != SWING_MODEL_TYPE:
        raise DataReadinessError("production release admission requires a swing model")
