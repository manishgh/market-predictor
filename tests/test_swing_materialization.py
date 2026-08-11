from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.edge_rebuild.swing_daily_combination import (
    CombinedDailyStore,
    VerifiedCombinedInputs,
)
from market_predictor.edge_rebuild.swing_features import SWING_FEATURE_PANEL_SCHEMA
from market_predictor.edge_rebuild.swing_materialization import (
    SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
    SWING_MATERIALIZATION_MANIFEST_SCHEMA,
    SWING_MATERIALIZATION_REQUEST_SCHEMA,
    _json_sha256,
    load_complete_swing_feature_panel,
    materialize_swing_feature_panel,
)
from market_predictor.swing.contracts import MINIMUM_SWING_DECISION_DATE
from market_predictor.v3.errors import DataReadinessError


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC", "DDD"],
            "security_id": ["sec:a", "sec:b", "sec:c", "sec:d"],
            "primary_benchmark": ["XLK"] * 4,
            "effective_from_utc": [pd.Timestamp("2018-05-29T04:00:00Z")] * 4,
            "effective_to_utc": [pd.NaT] * 4,
        }
    )


def _rows(group: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for number, row in enumerate(group.itertuples(index=False)):
        for day in (2, 3):
            decision = pd.Timestamp(f"2024-01-{day:02d}T21:01:00Z")
            records.append(
                {
                    "security_id": row.security_id,
                    "ticker": row.ticker,
                    "session_date_et": pd.Timestamp(f"2024-01-{day:02d}").date(),
                    "decision_time_utc": decision,
                    "decision_id": f"{row.security_id}:{day}",
                    "available_at_utc": decision,
                    "swing_feature_panel_schema": SWING_FEATURE_PANEL_SCHEMA,
                    "feature_profile": "technical_market",
                    "feature_eligible": True,
                    "label_eligible": True,
                    "barrier_label": 1,
                    "barrier_label_available_at_utc": decision + pd.Timedelta(days=2),
                    "value": float(number + day),
                }
            )
    return pd.DataFrame.from_records(records)


def _source_arguments(root: Path) -> dict[str, Path]:
    return {
        "pre_plan_directory": root / "plan",
        "pre_collection_directory": root / "pre",
        "post_collection_directory": root / "post",
        "membership_directory": root / "membership",
        "raw_archive_directory": root / "raw",
        "event_directory": root / "events",
        "transition_directory": root / "transitions",
        "reviewed_transitions_path": root / "review.csv",
        "anchor_path": root / "anchor.csv",
    }


@pytest.fixture
def materialization_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, int]]:
    import market_predictor.edge_rebuild.swing_materialization as module

    memberships = _memberships()
    verified = VerifiedCombinedInputs(
        memberships=memberships,
        request_payload={
            "pre_collection": {"manifest_sha256": "a", "authority_sha256": "b"},
            "post_collection": {"manifest_sha256": "c"},
            "membership_authority": {"authority_sha256": "d"},
            "excluded_security_ids_sha256": "e",
            "coverage_audit_sha256": "f",
            "security_exclusions": [],
            "benchmark_coverage": [],
        },
        pre_records=(),
        post_records={},
        excluded_security_ids=(),
        benchmark_tickers=("SPY", "QQQ", "XLK"),
        coverage_audit={
            "session_gap_audit": {
                "gaps": [],
                "missing_session_count": 0,
            }
        },
        warmup_only_security_ids=("sec:warmup",),
    )
    monkeypatch.setattr(
        module,
        "verify_combined_swing_inputs",
        lambda **_kwargs: verified,
    )
    def prepare(**kwargs: Any) -> CombinedDailyStore:
        output = Path(kwargs["output_directory"])
        output.mkdir(parents=True, exist_ok=True)
        (output / "_authority.json").write_text("{}", encoding="utf-8")
        return CombinedDailyStore(
            memberships=memberships,
            artifacts={},
            manifest={},
        )

    monkeypatch.setattr(module, "prepare_combined_daily_store", prepare)
    monkeypatch.setattr(
        module,
        "load_daily_bars",
        lambda ticker, _artifacts: pd.DataFrame({"ticker": [ticker]}),
    )
    monkeypatch.setattr(
        module,
        "load_security_batch_bars",
        lambda group, _artifacts: (group[["ticker"]].copy(), 0),
    )
    calls = {"build": 0, "finalize": 0}

    def build(
        _stocks: pd.DataFrame,
        _benchmarks: pd.DataFrame,
        group: pd.DataFrame,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        calls["build"] += 1
        return _rows(group)

    def finalize(
        rows: pd.DataFrame,
        **_kwargs: Any,
    ) -> pd.DataFrame:
        calls["finalize"] += 1
        assert rows.groupby("session_date_et")["security_id"].nunique().eq(4).all()
        return rows.assign(rank_label=1, cross_section_eligible=True)

    monkeypatch.setattr(module, "build_swing_feature_rows", build)
    monkeypatch.setattr(module, "finalize_swing_feature_panel", finalize)
    return tmp_path, tmp_path, calls


def test_materialization_resumes_then_publishes_immutable_panel(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )

    first = materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
        maximum_stage_one_shards_this_run=1,
    )
    assert first["status"] == "incomplete"
    assert first["completed_stage_one_shards"] == 1
    assert calls == {"build": 1, "finalize": 0}

    complete = materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    assert complete["status"] == "complete"
    assert complete["securities"] == 4
    assert complete["stage_one_shards"] == 2
    assert complete["rows"] == 8
    assert complete["feature_profiles"] == ["technical_market"]
    assert complete["schema"] == SWING_MATERIALIZATION_MANIFEST_SCHEMA
    assert complete["swing_feature_panel_schema"] == SWING_FEATURE_PANEL_SCHEMA
    assert complete["decision_start_date"] == MINIMUM_SWING_DECISION_DATE.isoformat()
    request = json.loads((output / "_request.json").read_text(encoding="utf-8"))
    authority = json.loads(
        (output / "final" / "_authority.json").read_text(encoding="utf-8")
    )
    assert request["schema"] == SWING_MATERIALIZATION_REQUEST_SCHEMA
    assert authority["schema"] == SWING_MATERIALIZATION_AUTHORITY_SCHEMA
    assert request["decision_start_date"] == MINIMUM_SWING_DECISION_DATE.isoformat()
    assert request["modeled_security_count"] == 4
    assert request["warmup_only_security_ids"] == ["sec:warmup"]
    assert complete["warmup_only_security_ids"] == ["sec:warmup"]
    assert authority["decision_start_date"] == MINIMUM_SWING_DECISION_DATE.isoformat()
    assert {
        record["feature_profile"] for record in complete["files"]
    } == {"technical_market"}
    assert set(complete["files_by_profile"]) == {"technical_market"}
    assert calls == {"build": 2, "finalize": 1}

    replay = materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    assert replay == load_complete_swing_feature_panel(output)
    assert calls == {"build": 2, "finalize": 1}

    with pytest.raises(DataReadinessError, match="resume request differs"):
        materialize_swing_feature_panel(
            **_source_arguments(source_root),
            contract=contract,
            output_dir=output,
            securities_per_shard=1,
        )


def test_resume_refuses_a_corrupted_stage_one_shard(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
        maximum_stage_one_shards_this_run=1,
    )
    with (output / "stage1" / "shard-0000.parquet").open("ab") as handle:
        handle.write(b"corrupt")

    with pytest.raises(DataReadinessError, match="does not verify"):
        materialize_swing_feature_panel(
            **_source_arguments(source_root),
            contract=contract,
            output_dir=output,
            securities_per_shard=2,
        )


def test_materialization_refuses_pre_cutoff_stage_one_rows(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import market_predictor.edge_rebuild.swing_materialization as module

    source_root, _memberships, _calls = materialization_inputs
    original_build = module.build_swing_feature_rows

    def build_pre_cutoff(*args: Any, **kwargs: Any) -> pd.DataFrame:
        rows = original_build(*args, **kwargs)
        rows.loc[0, "session_date_et"] = pd.Timestamp("2019-07-08").date()
        return rows

    monkeypatch.setattr(module, "build_swing_feature_rows", build_pre_cutoff)
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )

    with pytest.raises(DataReadinessError, match="pre-2019-07-09"):
        materialize_swing_feature_panel(
            **_source_arguments(source_root),
            contract=contract,
            output_dir=tmp_path / "panel-pre-cutoff",
            securities_per_shard=2,
        )


def test_complete_panel_refuses_changed_request_lineage(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    request_path = output / "_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    request["profile_policy"] = "tampered"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="authority does not verify"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_stale_materialization_authority_version(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    authority_path = output / "final" / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["schema"] = "edge_rebuild.swing_panel_materialization_authority.v5"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="authority does not verify"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_changed_authority_population_fields(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    authority_path = output / "final" / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["modeled_security_count"] = 999
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="authority does not verify"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_rebound_manifest_summary_not_supported_by_partitions(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rows"] = int(manifest["rows"]) + 1
    manifest["stage_one_rows"] = int(manifest["stage_one_rows"]) + 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = final_dir / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="rows or modeled security population"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_rebound_nontechnical_profile_contract(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_profiles"] = ["catalyst_full"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = final_dir / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="authority does not verify"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_duplicate_decisions_across_rebound_partitions(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_record = manifest["files"][0]
    source_frame = pd.read_parquet(final_dir / source_record["path"])
    source_frame["session_date_et"] = pd.Timestamp("2019-08-01").date()
    relative_path = "panel/feature_profile=technical_market/month=2019-08/part.parquet"
    duplicate_path = final_dir / relative_path
    duplicate_path.parent.mkdir(parents=True)
    source_frame.to_parquet(duplicate_path, index=False)
    duplicate_record = {
        **source_record,
        "path": relative_path,
        "sha256": file_sha256(duplicate_path),
        "partition_month": "2019-08",
        "sessions": 1,
        "first_session": "2019-08-01",
        "last_session": "2019-08-01",
        "decision_ids_sha256": _json_sha256(
            sorted(source_frame["decision_id"].astype(str))
        ),
    }
    manifest["files"].append(duplicate_record)
    manifest["files_by_profile"]["technical_market"].append(
        dict(duplicate_record)
    )
    manifest["rows"] += len(source_frame)
    manifest["stage_one_rows"] += len(source_frame)
    manifest["sessions"] += 1
    manifest["last_session"] = "2019-08-01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = final_dir / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="duplicated across partitions"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_substituted_security_population_after_hash_rebinding(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed_paths: set[str] = set()
    for records in manifest["files_by_profile"].values():
        for record in records:
            relative_path = record["path"]
            if relative_path in changed_paths:
                continue
            partition_path = final_dir / relative_path
            frame = pd.read_parquet(partition_path)
            frame.loc[frame["security_id"].eq("sec:a"), "security_id"] = (
                "sec:substitute"
            )
            frame.to_parquet(partition_path, index=False)
            changed_paths.add(relative_path)
    for records in [manifest["files"], *manifest["files_by_profile"].values()]:
        for record in records:
            if record["path"] in changed_paths:
                record["sha256"] = file_sha256(final_dir / record["path"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = final_dir / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="modeled security population"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_pre_cutoff_partition_after_hash_rebinding(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = manifest["files"][0]["path"]
    partition_path = final_dir / relative_path
    frame = pd.read_parquet(partition_path)
    frame.loc[0, "session_date_et"] = pd.Timestamp("2019-07-08").date()
    frame.to_parquet(partition_path, index=False)
    rebound_sha256 = file_sha256(partition_path)
    for record in manifest["files"]:
        if record["path"] == relative_path:
            record["sha256"] = rebound_sha256
    for records in manifest["files_by_profile"].values():
        for record in records:
            if record["path"] == relative_path:
                record["sha256"] = rebound_sha256
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = final_dir / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="decision window"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_wrong_physical_profile_after_hash_rebinding(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = manifest["files"][0]["path"]
    partition_path = final_dir / relative_path
    frame = pd.read_parquet(partition_path)
    frame["feature_profile"] = "unexpected_profile"
    frame.to_parquet(partition_path, index=False)
    _rebind_partition_and_authority(final_dir, manifest, relative_path)

    with pytest.raises(DataReadinessError, match="decision window or schema"):
        load_complete_swing_feature_panel(output)


def test_replay_rejects_decision_identity_and_record_bounds(
    tmp_path: Path,
    materialization_inputs: tuple[Path, Path, dict[str, int]],
) -> None:
    source_root, _memberships, _calls = materialization_inputs
    output = tmp_path / "panel"
    contract = load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = manifest["files"][0]["path"]
    partition_path = final_dir / relative_path
    frame = pd.read_parquet(partition_path)
    frame.loc[frame.index[1], "decision_id"] = frame.loc[frame.index[0], "decision_id"]
    frame.to_parquet(partition_path, index=False)
    _rebind_partition_and_authority(final_dir, manifest, relative_path)

    with pytest.raises(DataReadinessError, match="decision window or schema"):
        load_complete_swing_feature_panel(output)

    output = tmp_path / "panel-bounds"
    materialize_swing_feature_panel(
        **_source_arguments(source_root),
        contract=contract,
        output_dir=output,
        securities_per_shard=2,
    )
    final_dir = output / "final"
    manifest_path = final_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = manifest["files"][0]["path"]
    for record in manifest["files"]:
        if record["path"] == relative_path:
            record["first_session"] = "2099-01-01"
    for records in manifest["files_by_profile"].values():
        for record in records:
            if record["path"] == relative_path:
                record["first_session"] = "2099-01-01"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = final_dir / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="identity or bounds"):
        load_complete_swing_feature_panel(output)


def _rebind_partition_and_authority(
    final_dir: Path,
    manifest: dict[str, object],
    relative_path: str,
) -> None:
    partition_sha256 = file_sha256(final_dir / relative_path)
    for record in manifest["files"]:  # type: ignore[index]
        if record["path"] == relative_path:
            record["sha256"] = partition_sha256
    for records in manifest["files_by_profile"].values():  # type: ignore[index,union-attr]
        for record in records:
            if record["path"] == relative_path:
                record["sha256"] = partition_sha256
    manifest_path = final_dir / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority_path = final_dir / "_authority.json"
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["artifact_sha256"] = file_sha256(manifest_path)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
