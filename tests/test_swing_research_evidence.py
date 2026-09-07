from __future__ import annotations

import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import sequence_sha256
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.evaluation import research_evidence as evidence
from market_predictor.swing.features.panel import swing_model_feature_columns

ROOT = Path(__file__).resolve().parents[1]


def _write_inventory(path: Path, payload: dict[str, Any]) -> None:
    lines = [f"{key} = {json.dumps(value)}" for key, value in payload.items() if key != "artifacts"]
    for artifact in payload["artifacts"]:
        lines.append("\n[[artifacts]]")
        for key, value in artifact.items():
            if key == "counts":
                fields = ", ".join(f"{k} = {v}" for k, v in value.items())
                lines.append(f"counts = {{ {fields} }}")
            else:
                lines.append(f"{key} = {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def fixture_inventory(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    payload = tomllib.loads((ROOT / "configs/swing_research_evidence.toml").read_text(encoding="utf-8"))
    strategy = load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml")
    features = swing_model_feature_columns(contract=strategy, catalyst=False)
    for artifact in payload["artifacts"]:
        path = tmp_path / artifact["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if artifact["kind"] == "config":
            shutil.copyfile(ROOT / artifact["path"], path)
        elif artifact["kind"] == "access_hash_only":
            # Invalid JSON proves the exposed file is hashed without parsing.
            path.write_bytes(b"EXPOSED OUTCOME PAYLOAD: must not be parsed")
        else:
            metadata: dict[str, Any] = dict(artifact.get("counts", {}))
            metadata["files"] = [{"path": "data/raw/DO_NOT_OPEN.parquet"}]
            if artifact["kind"] == "trials":
                metadata.pop("recorded_trials")
                metadata["specialists"] = [{"experiments": [{"id": "duplicate"}] * 6}] * 2
            if artifact["id"] == "panel":
                metadata["columns_by_profile"] = {"technical_market": ["decision_id", *features, "future_return"]}
            path.write_text(json.dumps(metadata), encoding="utf-8")
        artifact["bytes"] = path.stat().st_size
        artifact["sha256"] = file_sha256(path)
    shutil.copyfile(ROOT / "configs/swing_research.toml", tmp_path / "configs/swing_research.toml")
    _write_inventory(tmp_path / "inventory.toml", payload)
    return tmp_path, payload


def _audit(root: Path) -> dict[str, object]:
    return evidence.audit_swing_research_evidence(
        root, root / "inventory.toml", root / "configs/swing_research.toml",
        root / "configs/edge_rebuild_strategy_contract.toml",
    )


def test_bounded_inventory_replays_metadata_not_payloads(fixture_inventory: tuple[Path, dict[str, Any]]) -> None:
    root, _ = fixture_inventory
    report = _audit(root)
    assert report == _audit(root)
    assert report["status"] == "metadata_verified_only"
    assert report["feature_count"] == 120
    assert report["historical_trials"] == {
        "retained_manifest_count": 5, "recorded_trials": 60,
        "duplicates_possible": True, "complete_lifetime_trial_history": False,
    }
    for key in ("training_authorized", "promotion_authorized", "source_payloads_read",
                "outcome_payloads_parsed", "full_replay_performed", "total_return_accounting_verified"):
        assert report[key] is False
    rendered = json.dumps(report)
    assert "already_exposed_research_only" in rendered
    assert "DO_NOT_OPEN" not in rendered
    assert "EXPOSED OUTCOME PAYLOAD" not in rendered
    assert "SwingPromotionConfig.max_drawdown" in rendered


def test_feature_mapping_covers_exact_order() -> None:
    inventory = tomllib.loads((ROOT / "configs/swing_research_evidence.toml").read_text(encoding="utf-8"))
    strategy = load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml")
    columns = swing_model_feature_columns(contract=strategy, catalyst=False)
    assert len(columns) == 120
    assert sequence_sha256(columns) == inventory["feature_order_sha256"]
    mapping = evidence._feature_mapping(columns)
    assert mapping["ordered_features"] == list(columns)


@pytest.mark.parametrize("mutation", ["hash", "size", "count", "feature_order", "missing_identity", "access_parse", "escape"])
def test_inventory_rejects_tampering(
    fixture_inventory: tuple[Path, dict[str, Any]], mutation: str,
) -> None:
    root, payload = fixture_inventory
    panel = next(a for a in payload["artifacts"] if a["id"] == "panel")
    if mutation == "hash":
        panel["sha256"] = "0" * 64
    elif mutation == "size":
        panel["bytes"] += 1
    elif mutation == "count":
        panel["counts"]["rows"] += 1
    elif mutation == "feature_order":
        payload["feature_order_sha256"] = "0" * 64
    elif mutation == "missing_identity":
        payload["artifacts"][0]["id"] = "unknown"
    elif mutation == "access_parse":
        next(a for a in payload["artifacts"] if a["id"] == "exposed_evaluation")["kind"] = "metadata"
    else:
        panel["path"] = "../outside/_manifest.json"
    _write_inventory(root / "inventory.toml", payload)
    with pytest.raises(DataReadinessError):
        _audit(root)


@pytest.mark.parametrize("invalid", ['{"rows": 1, "rows": 2}', '{"rows": NaN}'])
def test_strict_metadata_json_rejects_rehashed_invalid_content(
    fixture_inventory: tuple[Path, dict[str, Any]], invalid: str,
) -> None:
    root, payload = fixture_inventory
    artifact = next(a for a in payload["artifacts"] if a["id"] == "panel_request")
    path = root / artifact["path"]
    path.write_text(invalid, encoding="utf-8")
    artifact.update(bytes=path.stat().st_size, sha256=file_sha256(path))
    _write_inventory(root / "inventory.toml", payload)
    with pytest.raises(DataReadinessError):
        _audit(root)


def test_rehashed_trial_count_change_rejected(fixture_inventory: tuple[Path, dict[str, Any]]) -> None:
    root, payload = fixture_inventory
    artifact = next(a for a in payload["artifacts"] if a["kind"] == "trials")
    path = root / artifact["path"]
    path.write_text('{"specialists": [{"experiments": []}]}', encoding="utf-8")
    artifact.update(bytes=path.stat().st_size, sha256=file_sha256(path))
    _write_inventory(root / "inventory.toml", payload)
    with pytest.raises(DataReadinessError, match="count mismatch"):
        _audit(root)


def test_unknown_inventory_fields_fail_closed(fixture_inventory: tuple[Path, dict[str, Any]]) -> None:
    root, payload = fixture_inventory
    payload["training_authorized"] = True
    _write_inventory(root / "inventory.toml", payload)
    with pytest.raises(DataReadinessError):
        _audit(root)
