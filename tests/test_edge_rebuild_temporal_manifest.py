from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.temporal_manifest import (
    SWING_PANEL_AUTHORITY_SCHEMA,
    SWING_PANEL_MANIFEST_SCHEMA,
    TEMPORAL_MANIFEST_SCHEMA,
    TemporalManifestConfig,
    build_temporal_schedule,
    load_temporal_manifest_config,
    publish_temporal_manifest,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "edge_rebuild_temporal_manifest.toml"
STRATEGY = ROOT / "configs" / "edge_rebuild_strategy_contract.toml"


def test_schedule_freezes_full_year_windows_and_embargoes() -> None:
    config = load_temporal_manifest_config(POLICY)
    schedule = build_temporal_schedule(config)

    assert len(schedule.folds) == 3
    assert len(schedule.warmup_sessions) == 250
    assert len(schedule.final_refit_sessions) == 1_260
    assert len(schedule.final_embargo_sessions) == 10
    assert schedule.locked_test_sessions[0] >= config.locked_test_start
    assert schedule.locked_test_sessions[-1] <= config.locked_test_end
    for fold in schedule.folds:
        assert len(fold.train_sessions) == 1_260
        assert len(fold.embargo_sessions) == 10
        assert len(fold.validation_sessions) == 252
        assert not set(fold.train_sessions) & set(fold.validation_sessions)
        assert fold.train_sessions[-1] < fold.embargo_sessions[0]
        assert fold.embargo_sessions[-1] < fold.validation_sessions[0]
    assert not set(schedule.locked_test_sessions) & set(
        schedule.final_refit_sessions
    )


def test_complete_panel_publishes_hash_bound_assignments(tmp_path: Path) -> None:
    config = load_temporal_manifest_config(POLICY)
    schedule = build_temporal_schedule(config)
    panel = _write_panel(tmp_path / "panel", schedule.target_sessions)
    output = tmp_path / "temporal"

    manifest = publish_temporal_manifest(
        panel_directory=panel,
        policy_path=POLICY,
        strategy_contract=load_strategy_contract(STRATEGY),
        output_directory=output,
        config=config,
    )

    assert manifest["schema"] == TEMPORAL_MANIFEST_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["coverage"]["target_sessions_missing"] == 0
    assert manifest["coverage"]["outcomes_read"] is False
    assert manifest["resources"]["peak_working_set_gib"] < 4.0
    assignments = pd.read_csv(output / "session_assignments.csv")
    assert len(assignments) == len(schedule.target_sessions)
    assert assignments["session"].is_unique
    assert set(assignments["global_role"]) == {
        "warmup",
        "development",
        "locked_test",
    }
    for fold in range(1, 4):
        assert set(assignments[f"fold_{fold}_role"]) == {
            "not_used",
            "train",
            "embargo",
            "validation",
        }
    authority = json.loads((output / "_authority.json").read_text(encoding="utf-8"))
    assert authority["artifact_sha256"] == file_sha256(output / "_manifest.json")


def test_short_panel_publishes_exact_missing_history(tmp_path: Path) -> None:
    config = load_temporal_manifest_config(POLICY)
    schedule = build_temporal_schedule(config)
    retained = tuple(
        session for session in schedule.target_sessions if session >= date_from("2019-07-09")
    )
    panel = _write_panel(tmp_path / "panel", retained)

    manifest = publish_temporal_manifest(
        panel_directory=panel,
        policy_path=POLICY,
        strategy_contract=load_strategy_contract(STRATEGY),
        output_directory=tmp_path / "temporal",
        config=config,
    )

    assert manifest["status"] == "insufficient_history"
    assert manifest["coverage"]["target_sessions_missing"] > 0
    first = manifest["coverage"]["missing_ranges"][0]
    assert first["first_session"] == schedule.first_session.isoformat()
    assert first["last_session"] < "2019-07-09"


def test_partition_tampering_fails_before_publication(tmp_path: Path) -> None:
    config = load_temporal_manifest_config(POLICY)
    schedule = build_temporal_schedule(config)
    panel = _write_panel(tmp_path / "panel", schedule.target_sessions)
    partition = next((panel / "panel").rglob("*.parquet"))
    partition.write_bytes(partition.read_bytes() + b"tampered")

    with pytest.raises(DataReadinessError, match="partition hash mismatch"):
        publish_temporal_manifest(
            panel_directory=panel,
            policy_path=POLICY,
            strategy_contract=load_strategy_contract(STRATEGY),
            output_directory=tmp_path / "temporal",
            config=config,
        )


def test_contract_cannot_relax_pdf_aligned_windows() -> None:
    payload = load_temporal_manifest_config(POLICY).model_dump()
    payload["fit_sessions"] = 500
    with pytest.raises(ValidationError):
        TemporalManifestConfig.model_validate(payload)
    payload = load_temporal_manifest_config(POLICY).model_dump()
    payload["validation_folds"] = 4
    with pytest.raises(ValidationError):
        TemporalManifestConfig.model_validate(payload)


def date_from(value: str) -> date:
    return pd.Timestamp(value).date()


def _write_panel(root: Path, sessions: tuple[date, ...]) -> Path:
    root.mkdir(parents=True)
    files: list[dict[str, object]] = []
    for year in sorted({session.year for session in sessions}):
        selected = [session for session in sessions if session.year == year]
        rows = pd.DataFrame(
            {
                "session_date_et": [session for session in selected for _ in range(2)],
                "decision_group_id": [
                    session.isoformat() for session in selected for _ in range(2)
                ],
            }
        )
        path = root / "panel" / f"year={year}" / "part.parquet"
        path.parent.mkdir(parents=True)
        rows.to_parquet(path, index=False)
        files.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "year": year,
            }
        )
    manifest = {
        "schema": SWING_PANEL_MANIFEST_SCHEMA,
        "strategy_contract_sha256": load_strategy_contract(STRATEGY).sha256(),
        "sessions": len(sessions),
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "files": files,
    }
    manifest_path = root / "_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "_authority.json").write_text(
        json.dumps(
            {
                "schema": SWING_PANEL_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root
