from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.swing_materialization import materialize_swing_feature_panel
from market_predictor.edge_rebuild.temporal_manifest import (
    TEMPORAL_MANIFEST_SCHEMA,
    TemporalManifestConfig,
    build_temporal_schedule,
    load_temporal_manifest_config,
    publish_temporal_manifest,
)
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.materialization import (
    SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
    SWING_MATERIALIZATION_MANIFEST_SCHEMA,
)
from tests.test_swing_materialization import (
    _source_arguments,
)
from tests.test_swing_materialization import (
    materialization_inputs as materialization_inputs,
)

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "edge_rebuild_temporal_manifest.toml"
STRATEGY = ROOT / "configs" / "edge_rebuild_strategy_contract.toml"


def test_schedule_freezes_explicit_post_cutoff_windows_and_embargoes() -> None:
    config = load_temporal_manifest_config(POLICY)
    schedule = build_temporal_schedule(config)

    assert len(schedule.folds) == 1
    assert schedule.first_session.isoformat() == "2018-07-10"
    assert schedule.last_session.isoformat() == "2026-06-30"
    assert len(schedule.target_sessions) == 2_004
    assert len(schedule.warmup_sessions) == 250
    assert len(schedule.final_refit_sessions) == 1_493
    assert len(schedule.final_embargo_sessions) == 10
    assert len(schedule.locked_test_sessions) == 251
    assert schedule.locked_test_sessions[0] == config.locked_test_start
    assert schedule.locked_test_sessions[-1] == config.locked_test_end
    for fold in schedule.folds:
        assert len(fold.train_sessions) == 1_231
        assert len(fold.embargo_sessions) == 10
        assert len(fold.validation_sessions) == 252
        assert not set(fold.train_sessions) & set(fold.validation_sessions)
        assert fold.train_sessions[-1] < fold.embargo_sessions[0]
        assert fold.embargo_sessions[-1] < fold.validation_sessions[0]
    assert schedule.folds[0].train_sessions[0].isoformat() == "2019-07-09"
    assert schedule.folds[0].train_sessions[-1].isoformat() == "2024-05-28"
    assert schedule.folds[0].validation_sessions[0].isoformat() == "2024-06-12"
    assert schedule.folds[0].validation_sessions[-1].isoformat() == "2025-06-13"
    assert schedule.final_refit_sessions[0].isoformat() == "2019-07-09"
    assert schedule.final_refit_sessions[-1].isoformat() == "2025-06-13"
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
    for fold in range(1, 2):
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
    partition = next((panel / "final" / "panel").rglob("*.parquet"))
    partition.write_bytes(partition.read_bytes() + b"tampered")

    with pytest.raises(DataReadinessError, match="partition hash mismatch"):
        publish_temporal_manifest(
            panel_directory=panel,
            policy_path=POLICY,
            strategy_contract=load_strategy_contract(STRATEGY),
            output_directory=tmp_path / "temporal",
            config=config,
        )


def test_current_producer_is_consumed_without_reading_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _, _ = materialization_inputs
    panel = tmp_path / "panel"
    contract = load_strategy_contract(STRATEGY)
    produced = materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=panel,
        securities_per_shard=2,
    )
    assert produced["status"] == "complete"
    read_parquet = pd.read_parquet
    reads: list[Path] = []

    def read_metadata(path: Path, *, columns: list[str]) -> pd.DataFrame:
        assert columns == ["session_date_et", "decision_time_utc"]
        reads.append(path)
        return read_parquet(path, columns=columns)

    monkeypatch.setattr(pd, "read_parquet", read_metadata)
    manifest = publish_temporal_manifest(
        panel_directory=panel,
        policy_path=POLICY,
        strategy_contract=contract,
        output_directory=tmp_path / "temporal",
    )
    assert len(reads) == len(produced["files"])
    assert manifest["status"] == "insufficient_history"
    assert manifest["coverage"]["outcomes_read"] is False
    assert manifest["coverage"]["target_sessions_missing"] == 2_002


def test_flat_panel_layout_is_not_supported(tmp_path: Path) -> None:
    panel = _write_panel(tmp_path / "panel", (date(2024, 1, 2),))
    with pytest.raises(DataReadinessError, match="JSON artifact is unreadable"):
        publish_temporal_manifest(
            panel_directory=panel / "final",
            policy_path=POLICY,
            strategy_contract=load_strategy_contract(STRATEGY),
            output_directory=tmp_path / "temporal",
        )


@pytest.mark.parametrize("artifact", ["_authority.json", "_manifest.json"])
def test_noncanonical_panel_identity_is_rejected(tmp_path: Path, artifact: str) -> None:
    panel = _write_panel(tmp_path / "panel", (date(2024, 1, 2),))
    path = panel / "final" / artifact
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema"] += ".v1"
    path.write_text(json.dumps(payload), encoding="utf-8")
    if artifact == "_manifest.json":
        authority_path = panel / "final" / "_authority.json"
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
        authority["artifact_sha256"] = file_sha256(path)
        authority_path.write_text(json.dumps(authority), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="supported"):
        publish_temporal_manifest(
            panel_directory=panel,
            policy_path=POLICY,
            strategy_contract=load_strategy_contract(STRATEGY),
            output_directory=tmp_path / "temporal",
        )


@pytest.mark.parametrize("decision_time", ["invalid", "2024-01-02T01:00:00Z"])
def test_invalid_or_wrong_local_session_clock_is_rejected(
    tmp_path: Path, decision_time: str,
) -> None:
    panel = _write_panel(tmp_path / "panel", (date(2024, 1, 2),))
    manifest_path = panel / "final" / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["files"][0]
    partition = panel / "final" / record["path"]
    rows = pd.read_parquet(partition)
    rows["decision_time_utc"] = decision_time
    rows.to_parquet(partition, index=False)
    record["sha256"] = file_sha256(partition)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = panel / "final" / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="invalid session or decision timestamp"):
        publish_temporal_manifest(
            panel_directory=panel,
            policy_path=POLICY,
            strategy_contract=load_strategy_contract(STRATEGY),
            output_directory=tmp_path / "temporal",
        )


def test_contract_cannot_relax_pdf_aligned_windows() -> None:
    payload = load_temporal_manifest_config(POLICY).model_dump()
    payload["initial_fit_expected_sessions"] = 500
    with pytest.raises(ValidationError):
        TemporalManifestConfig.model_validate(payload)
    payload = load_temporal_manifest_config(POLICY).model_dump()
    payload["locked_test_expected_sessions"] = 250
    with pytest.raises(ValidationError):
        TemporalManifestConfig.model_validate(payload)


def test_explicit_range_count_is_verified_against_xnys() -> None:
    payload = load_temporal_manifest_config(POLICY).model_dump()
    payload["initial_fit_end"] = date_from("2024-05-24")
    with pytest.raises(ValidationError, match="governed temporal dates changed"):
        TemporalManifestConfig.model_validate(payload)


def test_modeled_decision_cutoff_cannot_move_earlier() -> None:
    payload = load_temporal_manifest_config(POLICY).model_dump()
    payload["modeled_decision_start"] = date_from("2019-05-28")
    payload["initial_fit_start"] = date_from("2019-05-28")

    with pytest.raises(ValidationError, match="modeled decision start"):
        TemporalManifestConfig.model_validate(payload)


def date_from(value: str) -> date:
    return pd.Timestamp(value).date()


def _write_panel(root: Path, sessions: tuple[date, ...]) -> Path:
    final = root / "final"
    final.mkdir(parents=True)
    files: list[dict[str, object]] = []
    for month in sorted({session.strftime("%Y-%m") for session in sessions}):
        selected = [session for session in sessions if session.strftime("%Y-%m") == month]
        rows = pd.DataFrame(
            {
                "session_date_et": [session for session in selected for _ in range(2)],
                "decision_id": [
                    f"security:{security}:{session.isoformat()}"
                    for session in selected for security in range(2)
                ],
                "decision_time_utc": [
                    pd.Timestamp(f"{session.isoformat()}T21:01:00Z")
                    for session in selected for _ in range(2)
                ],
            }
        )
        path = final / "panel" / "feature_profile=technical_market" / f"month={month}" / "part.parquet"
        path.parent.mkdir(parents=True)
        rows.to_parquet(path, index=False)
        files.append(
            {
                "path": str(path.relative_to(final)).replace("\\", "/"),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "partition_month": month,
                "feature_profile": "technical_market",
            }
        )
    manifest = {
        "schema": SWING_MATERIALIZATION_MANIFEST_SCHEMA,
        "strategy_contract_sha256": load_strategy_contract(STRATEGY).sha256(),
        "sessions": len(sessions),
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "files": files,
    }
    manifest_path = final / "_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (final / "_authority.json").write_text(
        json.dumps(
            {
                "schema": SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
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
