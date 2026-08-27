from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).parents[1]


def test_research_catalog_matches_retained_behavior_named_artifacts() -> None:
    catalog = tomllib.loads(
        (REPOSITORY_ROOT / "configs/research_model_catalog.toml").read_text(
            encoding="utf-8"
        )
    )
    inventory = _inventory()
    retained = next(
        group
        for group in inventory["artifact_groups"]
        if group["action"] == "retain_and_use_for_non_actionable_research"
    )

    expected = {
        artifact["model_id"]: artifact["path"]
        for artifact in retained["artifacts"]
    }
    observed = {
        model["id"]: model["artifact_directory"] for model in catalog["models"]
    }

    assert observed == expected
    assert all("_v" not in path for path in observed.values())


def test_retirement_inventory_does_not_authorize_referenced_deletion() -> None:
    inventory = _inventory()
    destructive_actions = {
        "retire_before_old_namespace_removal",
        "retire_after_reference_and_regeneration_audit",
    }
    destructive_groups = [
        group
        for group in inventory["artifact_groups"]
        if group["action"] in destructive_actions
    ]

    assert destructive_groups
    assert all(not group["direct_references"] for group in destructive_groups)
    assert inventory["active_serving_pointer_found"] is False
    assert inventory["retired_legacy_promotion"]["status"] == "ungoverned_retired"


def test_retained_concrete_paths_have_one_retention_decision() -> None:
    inventory = _inventory()
    concrete_paths: list[str] = []
    for group in inventory["artifact_groups"]:
        concrete_paths.extend(group.get("paths", []))
        concrete_paths.extend(
            artifact["path"] for artifact in group.get("artifacts", [])
        )
    concrete_paths = [path for path in concrete_paths if "*" not in path]

    assert len(concrete_paths) == len(set(concrete_paths))


def _inventory() -> dict[str, Any]:
    payload = cast(
        dict[str, Any],
        json.loads(
            (
                REPOSITORY_ROOT / "docs/model_artifact_retention_inventory.json"
            ).read_text(encoding="utf-8")
        ),
    )
    assert payload["schema_version"] == (
        "market_predictor.model_artifact_retention_inventory.v1"
    )
    return payload
