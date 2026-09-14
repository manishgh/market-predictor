"""Synthetic monthly authority integration; no real archive or model loads."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.datasets.initial_fit_issuer_news import (
    SavedIssuerAuthority,
    _write_frame,
    build_initial_fit_catalyst_lineage,
)
from market_predictor.swing.datasets.issuer_news_publication import (
    CONFIG_SCHEMA,
    INDEX_SCHEMA,
    _publish,
    load_monthly_catalyst_index,
    publish_monthly_catalyst_authorities,
)
from market_predictor.swing.features.catalyst_aggregates import build_swing_ablation_rows
from market_predictor.swing.features.catalyst_decision_authority import (
    _attach_verified_event_identity,
    load_catalyst_decision_authority,
    publish_catalyst_decision_authority,
)
from market_predictor.swing.features.catalyst_decision_identity import load_decision_identity
from tests.test_initial_fit_issuer_derivation import _run
from tests.test_initial_fit_issuer_derivation import saved as saved
from tests.test_swing_catalyst_lineage import _policy_text


@pytest.fixture
def monthly(saved: SavedIssuerAuthority, tmp_path: Path, request: pytest.FixtureRequest):  # type: ignore[no-untyped-def]
    derived = _run(saved, tmp_path / "derived")
    day = "2024-05-01" if getattr(request, "param", None) == "unassigned" else "2024-05-28"
    ticker = "RENAMED" if getattr(request, "param", None) == "renamed" else "AAA"
    decisions = stamp_canonical_decision_ids(pd.DataFrame({
        "security_id": ["security:a", "security:no-news"], "ticker": [ticker, "ZZZ"], "timeframe": ["1d", "1d"],
        "decision_time_utc": pd.to_datetime([day + "T22:00:00Z"] * 2, utc=True),
        "bar_start_utc": pd.to_datetime([day + "T13:30:00Z"] * 2, utc=True),
        "prediction_cutoff_policy_id": ["xnys_1800_america_new_york_v1"] * 2,
    }))
    decision_path = tmp_path / "decisions.parquet"
    _write_frame(decisions, decision_path, "decisions", {})
    policy = tmp_path / "lineage.toml"
    policy.write_text(_policy_text(), encoding="utf-8")
    namespace = "market_predictor.swing.datasets.initial_fit_issuer_news."
    with patch(namespace + "heavy_job_lease", return_value=nullcontext()), patch(namespace + "assert_system_memory_available"):
        build_initial_fit_catalyst_lineage(derived_directory=derived.directory, expected_manifest_sha256=derived.manifest_sha256,
            decisions_path=decision_path, policy_path=policy, out_dir=tmp_path / "lineage")
    source_paths = {"policy_sha256": policy, "collection_manifest_sha256": derived.directory / "collection/_manifest.json",
        "collection_audit_sha256": derived.directory / "collection/_audit.json",
        "attribution_manifest_sha256": derived.directory / "attribution/_manifest.json",
        "sentiment_manifest_sha256": derived.directory / "sentiment/_manifest.json",
        "source_collections_sha256": derived.directory / "collection/source_collections.parquet"}
    config = {"schema": CONFIG_SCHEMA, "cohort_sha256": json_sha256(sorted(decisions.security_id)), "rows": 2,
        "source_files": {path.relative_to(tmp_path).as_posix(): file_sha256(path) for path in source_paths.values()},
        "months": {"2024-05": {"decisions": {"path": "decisions.parquet", "sha256": file_sha256(decision_path),
            "manifest_sha256": file_sha256(manifest_path_for(decision_path))}, "rows": 2,
            "decision_ids_sha256": json_sha256(sorted(decisions.decision_id)),
            "lineages": [{"directory": "lineage", "manifest_sha256": file_sha256(tmp_path / "lineage/_manifest.json"),
                "source_paths": {key: path.relative_to(tmp_path).as_posix() for key, path in source_paths.items()}}]}}}
    return config, decisions


def _publish_fixture(root: Path, config: dict[str, object], **kwargs):  # type: ignore[no-untyped-def]
    path = root / "publication.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with patch("market_predictor.swing.datasets.issuer_news_publication.assert_system_memory_available"):
        return _publish(root, path, file_sha256(path), root / "monthly", **kwargs)


def test_monthly_index_external_pins_and_canonical_nullable_population(monthly, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config, decisions = monthly
    result = _publish_fixture(tmp_path, config)
    with patch("market_predictor.swing.datasets.issuer_news_publication.load_canonical_artifact", side_effect=AssertionError):
        index = load_monthly_catalyst_index(tmp_path / "monthly", expected_manifest_sha256=str(result["manifest_sha256"]))
    assert index["schema"] == INDEX_SCHEMA
    assert index["status"] == "complete_research_only"
    assert index["rows"] == index["months"]["2024-05"]["rows"] == 2
    assert index["months"]["2024-05"]["catalyst_decision_rows"] == 1
    assert index["months"]["2024-05"]["decision_ids_sha256"] == json_sha256(sorted(decisions.decision_id))
    assert all(not Path(path).is_absolute() for path in index["source_files"])
    authority = load_catalyst_decision_authority(tmp_path / "monthly/2024-05")
    decisions["feature_eligible"] = True
    decisions["label_eligible"] = True
    for frame in build_swing_ablation_rows(decisions, authority).values():
        assert len(frame) == 2
        assert set(frame.decision_id) == set(decisions.decision_id)
        unknown = frame.loc[frame.security_id.eq("security:no-news")]
        assert not unknown.catalyst_required_source_complete.any()
        assert unknown["event_count_1d"].isna().all()
    (tmp_path / "monthly/2024-05/_authority.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="pin differs"):
        load_monthly_catalyst_index(tmp_path / "monthly", expected_manifest_sha256=str(result["manifest_sha256"]))


@pytest.mark.parametrize("monthly", ["unassigned"], indirect=True)
def test_verified_unassigned_events_do_not_become_canonical_decisions(monthly, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config, decisions = monthly
    result = _publish_fixture(tmp_path, config)
    index = load_monthly_catalyst_index(tmp_path / "monthly", expected_manifest_sha256=result["manifest_sha256"])
    assert index["rows"] == len(decisions) == 2
    assert index["months"]["2024-05"]["catalyst_decision_rows"] == 0
    authority = load_catalyst_decision_authority(tmp_path / "monthly/2024-05")
    assert authority.decisions.empty


@pytest.mark.parametrize("monthly", ["renamed"], indirect=True)
def test_rename_requires_pinned_canonical_decisions_and_rechecks_on_load(monthly, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config, _ = monthly
    with pytest.raises(DataReadinessError, match="requires pinned canonical"):
        publish_catalyst_decision_authority([tmp_path / "lineage"], tmp_path / "without-proof")
    result = _publish_fixture(tmp_path, config)
    authority = load_catalyst_decision_authority(tmp_path / "monthly/2024-05")
    assert authority.decisions.ticker.tolist() == ["RENAMED"]
    proof = authority.manifest["request"]["canonical_decisions"]
    assert proof["sha256"] == config["months"]["2024-05"]["decisions"]["sha256"]
    assert result["rows"] == 2
    (tmp_path / "decisions.parquet").write_bytes(b"synthetic tamper")
    with pytest.raises(DataReadinessError, match="pin differs"):
        load_catalyst_decision_authority(tmp_path / "monthly/2024-05")


@pytest.mark.parametrize("poison", ["artifact_pin", "manifest_pin", "relative_path", "oversized", "different_decisions"])
def test_canonical_decision_proof_rejects_wrong_or_unbounded_input(monthly, tmp_path: Path, poison: str) -> None:  # type: ignore[no-untyped-def]
    config, decisions = monthly
    record = config["months"]["2024-05"]["decisions"]
    proof = {"path": str(tmp_path / "decisions.parquet"), "sha256": record["sha256"], "manifest_sha256": record["manifest_sha256"]}
    if poison == "artifact_pin":
        proof["sha256"] = "f" * 64
    elif poison == "manifest_pin":
        proof["manifest_sha256"] = "f" * 64
    elif poison == "relative_path":
        proof["path"] = "decisions.parquet"
    elif poison == "oversized":
        with patch("market_predictor.swing.features.catalyst_decision_identity.MAXIMUM_DECISION_ROWS", 1), \
                pytest.raises(DataReadinessError, match="bounded"):
            load_decision_identity(proof, production_ready=False)
        return
    else:
        changed = decisions.drop(columns="decision_id").copy()
        changed["ticker"] = "FORGED"
        changed = stamp_canonical_decision_ids(changed)
        path = tmp_path / "different.parquet"
        _write_frame(changed, path, "decisions", {})
        proof = {"path": str(path), "sha256": file_sha256(path), "manifest_sha256": file_sha256(manifest_path_for(path))}
    with pytest.raises(DataReadinessError):
        publish_catalyst_decision_authority([tmp_path / "lineage"], tmp_path / "rejected", canonical_decisions=proof)
    assert not (tmp_path / "rejected").exists()


def test_decision_proof_mutation_during_publication_prevents_output(monthly, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config, _ = monthly
    record = config["months"]["2024-05"]["decisions"]
    path = tmp_path / "decisions.parquet"
    proof = {"path": str(path), "sha256": record["sha256"], "manifest_sha256": record["manifest_sha256"]}

    def tamper(_record):  # type: ignore[no-untyped-def]
        path.write_bytes(b"synthetic mutation during publication")

    with pytest.raises(DataReadinessError, match="pin differs"):
        publish_catalyst_decision_authority([tmp_path / "lineage"], tmp_path / "unpublished",
            canonical_decisions=proof, progress=tamper)
    assert not (tmp_path / "unpublished").exists()


@pytest.mark.parametrize("stage", ["staging", "load"])
def test_proof_mutation_during_final_child_read_is_rejected(monthly, tmp_path: Path, stage: str) -> None:  # type: ignore[no-untyped-def]
    config, _ = monthly
    if stage == "load":
        _publish_fixture(tmp_path, config)

    def mutate_after_coverage(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        result = load_canonical_artifact(path, *args, **kwargs)
        if Path(path).name == "source_coverage.parquet" and kwargs.get("expected_type") == "edge_rebuild_catalyst_coverage":
            (tmp_path / "decisions.parquet").write_bytes(b"synthetic proof mutation at loader exit")
        return result

    with patch("market_predictor.swing.features.catalyst_decision_authority.load_canonical_artifact",
            side_effect=mutate_after_coverage), pytest.raises(DataReadinessError, match="pin differs"):
        if stage == "load":
            load_catalyst_decision_authority(tmp_path / "monthly/2024-05")
        else:
            _publish_fixture(tmp_path, config)
    if stage == "staging":
        assert not (tmp_path / "monthly/2024-05").exists()


@pytest.mark.parametrize("field", ["ticker", "security_id", "source_family", "feature_available_at_utc"])
def test_assignment_identity_poison_cannot_use_canonical_proof(monthly, tmp_path: Path, field: str) -> None:  # type: ignore[no-untyped-def]
    config, _ = monthly
    record = config["months"]["2024-05"]["decisions"]
    proof = {"path": str(tmp_path / "decisions.parquet"), "sha256": record["sha256"], "manifest_sha256": record["manifest_sha256"]}
    canonical = load_decision_identity(proof, production_ready=False)
    child = json.loads((tmp_path / "lineage/_manifest.json").read_text())["artifacts"][0]
    events = pd.read_parquet(child["event_path"])
    direct = events.loc[events.training_eligible & events.relation_channel.eq("direct_issuer")]
    assignments = pd.read_parquet(child["assignment_path"])
    assignments = assignments.loc[assignments.status.eq("assigned") & assignments.event_id.isin(direct.event_id)].copy()
    assert not assignments.empty
    assignments[field] = pd.Timestamp("2024-05-29T00:00:00Z") if field.endswith("_utc") else "FORGED"
    with pytest.raises(DataReadinessError):
        _attach_verified_event_identity(assignments, direct, chunk_id=child["chunk_id"], canonical_decisions=canonical)


@pytest.mark.parametrize("poison", [
    "source_pin", "decision_ids", "rows", "wrong_month", "missing_source_pin", "whole_lineage", "swapped_paths",
])
def test_monthly_rejects_unpinned_or_wrong_partition(monthly, tmp_path: Path, poison: str) -> None:  # type: ignore[no-untyped-def]
    config, decisions = monthly
    record = config["months"]["2024-05"]
    if poison == "source_pin":
        config["source_files"]["lineage.toml"] = "f" * 64
    elif poison == "decision_ids":
        record["decision_ids_sha256"] = "f" * 64
    elif poison == "rows":
        record["rows"] = 1
    elif poison == "wrong_month":
        config["months"] = {"2024-04": record}
    elif poison == "missing_source_pin":
        del config["source_files"]["lineage.toml"]
    elif poison == "swapped_paths":
        paths = record["lineages"][0]["source_paths"]
        paths["policy_sha256"], paths["collection_manifest_sha256"] = paths["collection_manifest_sha256"], paths["policy_sha256"]
    else:
        decisions = decisions.iloc[:1].copy()
        path = tmp_path / "different-decisions.parquet"
        _write_frame(decisions, path, "decisions", {})
        record["decisions"] = {"path": path.name, "sha256": file_sha256(path),
            "manifest_sha256": file_sha256(manifest_path_for(path))}
        record["rows"] = config["rows"] = 1
        record["decision_ids_sha256"] = json_sha256(sorted(decisions.decision_id))
    with pytest.raises(DataReadinessError):
        _publish_fixture(tmp_path, config)
    assert not (tmp_path / "monthly/_manifest.json").exists()


def test_monthly_lease_uses_explicit_workspace_not_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    namespace = "market_predictor.swing.datasets.issuer_news_publication."
    with patch(namespace + "heavy_job_lease", return_value=nullcontext()) as lease, \
        patch(namespace + "assert_system_memory_available"), patch(namespace + "_publish", return_value={}) as publish:
        publish_monthly_catalyst_authorities(root=workspace, config_path=Path("config.json"),
            expected_config_sha256="a" * 64, output_directory=Path("out"))
    lease.assert_called_once_with("publish-monthly-initial-fit-catalyst-authorities", runtime_dir=workspace / "data/runtime")
    assert publish.call_args.args[0] == workspace.resolve()


def test_external_checkpoint_resume_reuses_verified_month_and_completed_index(monthly, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config, _ = monthly
    progress_pins = []

    def interrupted(record):  # type: ignore[no-untyped-def]
        progress_pins.append(record["checkpoint_sha256"])
        if record["completed_months"]:
            raise RuntimeError("synthetic interruption after durable month")

    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _publish_fixture(tmp_path, config, progress=interrupted)
    checkpoint = tmp_path / "monthly/_checkpoint.json"
    assert file_sha256(checkpoint) == progress_pins[-1]
    authority = tmp_path / "monthly/2024-05/_authority.json"
    authority_pin = file_sha256(authority)
    with pytest.raises(DataReadinessError, match="external manifest/checkpoint"):
        _publish_fixture(tmp_path, config)
    with pytest.raises(DataReadinessError, match="pin differs"):
        _publish_fixture(tmp_path, config, expected_existing_manifest_sha256="f" * 64)
    publisher = "market_predictor.swing.features.catalyst_decision_authority.publish_catalyst_decision_authority"
    with patch(publisher, side_effect=AssertionError("completed month must not rebuild")):
        result = _publish_fixture(tmp_path, config, expected_existing_manifest_sha256=progress_pins[-1])
        replay = _publish_fixture(tmp_path, config, expected_existing_manifest_sha256=result["manifest_sha256"])
    assert replay == result
    assert file_sha256(authority) == authority_pin
    authority.write_text("{}", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="pin differs"):
        _publish_fixture(tmp_path, config, expected_existing_manifest_sha256=result["manifest_sha256"])
