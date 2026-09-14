"""Bounded replay orchestration, exact lineage, and migration poison regressions."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.implementation_snapshot import create_implementation_snapshot
from market_predictor.evidence.io import write_json_object
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import predictor_abstention_derivation as derivation
from market_predictor.swing.datasets import predictor_replay as owner
from market_predictor.swing.datasets import research_features as features
from market_predictor.swing.features.adjusted_source import AdjustedTechnicalSource
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from tests.test_swing_features import contract as contract
from tests.test_swing_predictor_abstention_derivation import _install_prefix_replay, _pin, _refreeze, _run
from tests.test_swing_predictor_abstention_derivation import example as example


@pytest.fixture(params=["prefix", "whole"])
def migration(example: dict[str, Any], monkeypatch: pytest.MonkeyPatch, contract: Any,
    request: pytest.FixtureRequest,
) -> dict[str, Any]:
    state = example
    root = state["root"]
    producer = root / "src/market_predictor/swing/datasets/research_features.py"
    producer.write_bytes(b"raise RuntimeError('archived code must never execute')\n")
    extra = producer.with_name("corrected_decisions.py")
    extra.write_bytes(b"# historical implementation declared only in source_files\n")
    state["request"]["implementation_files"] = {producer.relative_to(root).as_posix(): file_sha256(producer)}
    state["request"]["source_files"][extra.relative_to(root).as_posix()] = file_sha256(extra)
    _install_prefix_replay(state, monkeypatch, contract)
    if request.param == "whole":
        state["facts"]["failures"][0].update(reason_code="unverified_issuer_history", quarantine="entire_failed_group",
            first_invalid_session=None, boundary_observation_sha256=None)
        _refreeze(state)
    result = _run(state)
    published = _pin(root, state["output"] / "_manifest.json")
    published_request = json.loads((state["output"] / "_request.json").read_text())
    bound = derivation._merge(root, *(published_request[name] for name in derivation.PIN_MAPS))
    historical = {name: digest for name, digest in bound.items() if name.startswith("src/")}
    snapshot = create_implementation_snapshot(root, historical, root / "data/evidence/migration/implementation")
    bindings = root / "data/evidence/migration/_bindings.json"
    write_json_object(bindings, dict(schema="market_predictor.archive_owner_migration_bindings",
        purpose="historical_implementation_provenance_not_numerical_replay", snapshot=snapshot, publications={published.path: dict(
            manifest_sha256=published.sha256, request=(state["output"] / "_request.json").relative_to(root).as_posix(),
            request_sha256=result["request_sha256"], implementation_files=historical)}))
    for name in historical:
        (root / name).write_bytes(b"# migrated current implementation; not the preserved bytes\n")
    current = {name: file_sha256(root / name) for name in historical}
    part = result["groups"][state["key"]]
    expected_good, _ = load_canonical_artifact(state["output"] / "groups" / part["path"],
        expected_type="swing_research_technical_inputs", allow_research=True)
    decisions = state["decisions"].copy()
    decisions["source_group"] = decisions.parent_ticker
    sources = dict(source_files={state["config"].name: file_sha256(state["config"])}, request=published_request)
    context = SimpleNamespace(decisions=decisions, sources=sources, corrected_records={},
        adjusted_source_files=published_request["adjusted_source_files"], records=[dict(partition_month=month)
            for month in result["months"]])
    state.update(publication=published, published_request=published_request, published_manifest=result,
        bindings=_pin(root, bindings), snapshot=SourcePin(path=snapshot["manifest_path"], sha256=snapshot["manifest_sha256"]),
        historical=historical, current=current, expected_good=expected_good, context=context, calls=[], numeric_active=False,
        numeric_exit_failure=False, on_exit=None, mutate_frame=None, report_dir=root / "data/reports/replay",
        replay_kind=request.param)
    state["replay_calls"].clear()

    @contextmanager
    def source_context(*args: Any, **kwargs: Any) -> Any:
        with heavy_job_lease("test ordinary numeric source", runtime_dir=root / "data/runtime"):
            state["numeric_active"] = True
            try:
                yield sources
                assert not (state["report_dir"] / "_checkpoint.json").exists()
                assert not (state["report_dir"] / "_manifest.json").exists()
                if state["on_exit"]:
                    state["on_exit"]()
                if state["numeric_exit_failure"]:
                    raise DataReadinessError("ordinary source exit failed")
            finally:
                state["numeric_active"] = False

    def build(group: pd.DataFrame, prepared: Any) -> AdjustedTechnicalSource:
        assert state["numeric_active"] and prepared is context
        state["calls"].append(group.security_id.iloc[0])
        frame = expected_good.copy()
        if state["mutate_frame"]:
            state["mutate_frame"](frame)
        return AdjustedTechnicalSource(frame, part["availability_columns"])

    monkeypatch.setattr(owner, "_current_implementation", lambda root: {name: file_sha256(root / name) for name in current})
    monkeypatch.setattr(owner, "_corrected_child_pins", lambda *a: {})
    monkeypatch.setattr(owner, "heavy_job_runtime_dir", lambda: Path("data/runtime"))
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    monkeypatch.setattr(features, "_guard", lambda: None)
    monkeypatch.setattr(owner, "load_corrected_outcome_policy", derivation.load_corrected_outcome_policy)
    monkeypatch.setattr(owner, "load_corporate_action_evidence", lambda **k: SimpleNamespace(recheck=lambda root: None))
    monkeypatch.setattr(owner, "verified_corrected_research_sources", source_context)
    monkeypatch.setattr(features, "prepare_predictor_sources", lambda **k: context)
    monkeypatch.setattr(features, "build_predictor_group", build)
    return state


def replay(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return owner.replay_predictor_publication(root=state["root"], publication=state["publication"],
        migration_bindings=state["bindings"], implementation_snapshot=state["snapshot"], output=state["report_dir"], **kwargs)


def verify(state: dict[str, Any]) -> owner.VerifiedPredictorReplay:
    return owner.verify_predictor_replay(root=state["root"], publication=state["publication"],
        replay=_pin(state["root"], state["report_dir"] / "_manifest.json"))


@pytest.mark.parametrize("absolute", [False, True])
def test_snapshot_binding_accepts_equivalent_native_paths(migration: dict[str, Any], absolute: bool) -> None:
    snapshot = migration["snapshot"]
    path = Path(snapshot.path)
    if absolute:
        path = migration["root"] / path
    migration["snapshot"] = SourcePin(path=str(path), sha256=snapshot.sha256)
    assert replay(migration)["replay_complete"] is True
    assert verify(migration).historical_implementation_files == migration["historical"]


def test_feature_plan_evidence_is_bound_and_reverified(
    migration: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = migration["root"] / "data/evidence/plan-proof.json"
    marker.write_bytes(b'{"test_fixture":true}')
    pin = _pin(migration["root"], marker)
    calls: list[SourcePin | None] = []

    def evidence(root: Path, policy: Any, snapshot: SourcePin | None) -> dict[str, str]:
        calls.append(snapshot)
        assert snapshot == migration["snapshot"]
        return {pin.path: pin.sha256}

    monkeypatch.setattr(owner, "_feature_plan_evidence", evidence)
    replay(migration, feature_plan_snapshot=migration["snapshot"])
    assert verify(migration).live_evidence_files[pin.path] == pin.sha256
    assert len(calls) == 2
    marker.write_bytes(b"altered")
    with pytest.raises(DataReadinessError):
        verify(migration)


def test_all_groups_and_months_rebuilt_and_verified(migration: dict[str, Any]) -> None:
    result = replay(migration)
    assert result["replay_complete"] is True and result["rows"] == 4
    assert len(result["compared_groups"]) == 2 and len(result["compared_months"]) == 2
    assert migration["calls"] == ["issuer-AAA"]
    prefix_builds = [call for call in migration["replay_calls"] if call[0] == "build"]
    assert prefix_builds == ([("build", 1)] if migration["replay_kind"] == "prefix" else [])
    assert file_sha256(migration["root"] / migration["publication"].path) == migration["publication"].sha256
    receipt = verify(migration)
    assert dict(receipt.historical_implementation_files) == migration["historical"]
    assert all(receipt.live_evidence_files[name] == digest for name, digest in migration["current"].items())
    assert migration["snapshot"].path in receipt.live_evidence_files
    assert migration["bindings"].path in receipt.live_evidence_files
    with pytest.raises(TypeError):
        receipt.live_evidence_files["changed"] = "0" * 64  # type: ignore[index]


def test_partial_is_finalized_but_never_resumable(migration: dict[str, Any]) -> None:
    result = replay(migration, maximum_groups=1)
    assert result["replay_complete"] is False and len(result["compared_groups"]) == 1
    assert result["compared_months"] == {}
    assert not (migration["report_dir"] / "_manifest.json").exists()
    assert file_sha256(migration["report_dir"] / "_checkpoint.json") == result["checkpoint_sha256"]
    with pytest.raises(DataReadinessError, match="new directory"):
        replay(migration)


@pytest.mark.parametrize("field", [TECHNICAL_RANKING_FEATURES[0], "technical_available_at_utc", "parent_decision_id",
    "feature_eligible", "technical_missing_reasons", "daily_bar_count", "universe_snapshot_id"])
def test_completed_ordinary_group_numerical_poison_fails(migration: dict[str, Any], field: str) -> None:
    def poison(frame: pd.DataFrame) -> None:
        if field == "feature_eligible":
            frame.loc[0, field] = False
        elif field.endswith("_at_utc"):
            frame.loc[0, field] += pd.Timedelta(seconds=1)
        elif field in ("parent_decision_id", "technical_missing_reasons", "universe_snapshot_id"):
            frame.loc[0, field] = "changed"
        else:
            frame.loc[0, field] += 1
    migration["mutate_frame"] = poison
    with pytest.raises(DataReadinessError, match="replay differs|counts differ"):
        replay(migration)
    assert not (migration["report_dir"] / "_checkpoint.json").exists()


def test_comparison_uses_persisted_string_representation(migration: dict[str, Any]) -> None:
    frame = migration["expected_good"].copy()
    frame["security_id"] = frame.security_id.astype(object)
    assert frame.security_id.dtype != migration["expected_good"].security_id.dtype
    part = migration["published_manifest"]["groups"][migration["key"]]
    owner._compare_partition(migration["output"] / "groups", part, part["origin_request_sha256"],
        frame, migration["expected_good"])
    assert frame.security_id.dtype == object


@pytest.mark.parametrize("kind", ["tiny_float", "dtype", "column_order", "null", "clock_dtype"])
def test_comparison_is_exact_not_default_tolerance(migration: dict[str, Any], kind: str) -> None:
    frame = migration["expected_good"].copy()
    column = TECHNICAL_RANKING_FEATURES[0]
    if kind == "tiny_float":
        frame.loc[0, column] += 1e-12
    elif kind == "dtype":
        frame[column] = frame[column].astype("float32")
    elif kind == "column_order":
        frame = frame[frame.columns[::-1]]
    elif kind == "clock_dtype":
        frame["technical_available_at_utc"] = frame.technical_available_at_utc.dt.tz_localize(None)
    else:
        frame.loc[0, column] = float("nan")
    part = migration["published_manifest"]["groups"][migration["key"]]
    with pytest.raises(DataReadinessError, match="numerical replay differs"):
        owner._compare_partition(migration["output"] / "groups", part, part["origin_request_sha256"],
            frame, migration["expected_good"])


@pytest.mark.parametrize("kind", ["exit", "source", "implementation", "progress", "scratch"])
def test_unfinalized_progress_never_authorizes_completion(migration: dict[str, Any], kind: str) -> None:
    if kind == "exit":
        migration["numeric_exit_failure"] = True
    else:
        def mutate() -> None:
            path = {"source": migration["source"], "implementation": migration["root"] / next(iter(migration["current"])),
                "progress": migration["report_dir"] / "_progress.json",
                "scratch": migration["report_dir"] / "reconstructed_groups" / f"{migration['key']}.parquet"}[kind]
            path.write_bytes(b"changed at source exit")
        migration["on_exit"] = mutate
    with pytest.raises((DataReadinessError, json.JSONDecodeError)):
        replay(migration)
    assert not (migration["report_dir"] / "_checkpoint.json").exists()
    assert not (migration["report_dir"] / "_manifest.json").exists()


def test_active_lease_rejects_before_numeric_work(migration: dict[str, Any]) -> None:
    with heavy_job_lease("other job", runtime_dir=migration["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            replay(migration)
    assert migration["calls"] == [] and not migration["report_dir"].exists()


def test_memory_failure_is_not_recorded_as_source_abstention(migration: dict[str, Any]) -> None:
    def pressure(frame: pd.DataFrame) -> None:
        raise MemoryBudgetError("synthetic pressure")
    migration["mutate_frame"] = pressure
    with pytest.raises(MemoryBudgetError):
        replay(migration)
    assert not (migration["report_dir"] / "_checkpoint.json").exists()


@pytest.mark.parametrize("name", ["groups", "months"])
def test_foreign_population_rejected(migration: dict[str, Any], name: str) -> None:
    manifest = json.loads(json.dumps(migration["published_manifest"]))
    manifest[name]["foreign"] = next(iter(manifest[name].values()))
    with pytest.raises(DataReadinessError, match="exhaustive"):
        owner._replay_population(migration["context"], manifest, migration["published_request"])


def test_historical_metadata_helper_does_not_weaken_ordinary_validation(migration: dict[str, Any]) -> None:
    derivation.validate_historical_predictor_derivation(root=migration["root"],
        manifest=migration["published_manifest"], request=migration["published_request"])
    if migration["replay_kind"] == "prefix":
        with pytest.raises(DataReadinessError, match="conflict"):
            derivation.validate_predictor_derivation(root=migration["root"],
                manifest=migration["published_manifest"], request=migration["published_request"])


def test_no_arbitrary_src_exemption_or_alias_overwrite(tmp_path: Path) -> None:
    known = "src/market_predictor/known.py"
    unknown = "src/market_predictor/unknown.py"
    live, historical = owner._split_historical_pins(tmp_path, {known: "a" * 64, unknown: "b" * 64}, {known: "a" * 64})
    assert live == {unknown: "b" * 64} and historical == {known: "a" * 64}
    with pytest.raises(DataReadinessError, match="conflict"):
        owner._split_historical_pins(tmp_path, {known: "a" * 64, known.replace("/", "\\"): "b" * 64}, {known: "a" * 64})
    with pytest.raises(DataReadinessError, match="declarations differ"):
        owner._split_historical_pins(tmp_path, {known: "a" * 64}, {known: "b" * 64})


@pytest.mark.parametrize("kind", ["current", "source", "snapshot", "scratch", "coverage", "ancestry"])
def test_verifier_rejects_stale_tampered_or_incomplete_evidence(migration: dict[str, Any], kind: str) -> None:
    replay(migration)
    if kind == "current":
        path = migration["root"] / next(iter(migration["current"]))
    elif kind == "source":
        path = migration["source"]
    elif kind == "snapshot":
        path = migration["root"] / migration["snapshot"].path
    elif kind == "scratch":
        path = migration["report_dir"] / "reconstructed_groups" / f"{migration['key']}.parquet"
    elif kind == "ancestry":
        path = migration["parent"] / "_checkpoint.json"
    else:
        path = migration["report_dir"] / "_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["compared_groups"].pop(migration["key"])
        path.write_text(json.dumps(manifest), encoding="ascii")
        with pytest.raises(DataReadinessError, match="coverage"):
            verify(migration)
        return
    path.write_bytes(b"changed after replay")
    with pytest.raises(DataReadinessError):
        verify(migration)


def test_monthly_values_are_compared_even_with_valid_child_hashes(migration: dict[str, Any]) -> None:
    manifest = migration["published_manifest"]
    month, part = next(iter(manifest["months"].items()))
    path = migration["output"] / "months" / part["path"]
    frame, _ = load_canonical_artifact(path, expected_type="swing_research_technical_inputs", allow_research=True)
    expected = frame.copy()
    frame.loc[0, TECHNICAL_RANKING_FEATURES[0]] = 123.0
    poisoned = features._publish(frame, migration["root"] / "data/features/poison" / f"{month}.parquet",
        manifest["request_sha256"])
    with pytest.raises(DataReadinessError, match="numerical replay differs"):
        owner._compare_partition(migration["root"] / "data/features/poison", poisoned, manifest["request_sha256"],
            expected, expected)


def test_corrected_children_are_bound_through_original_authority(tmp_path: Path) -> None:
    archive = tmp_path / "data/raw/corrected"
    archive.mkdir(parents=True)
    bars = archive / "bars.parquet"
    bars.write_bytes(b"only hash; never decode")
    write_json_object(archive / "_manifest.json", {"unit_artifacts": [dict(bars_path=bars.name, bars_sha256=file_sha256(bars))]})
    write_json_object(archive / "_authority.json", {"artifact_sha256": file_sha256(archive / "_manifest.json")})
    policy = SimpleNamespace(adjusted_archive_authority=_pin(tmp_path, archive / "_authority.json"))
    pins = owner._corrected_child_pins(tmp_path, policy)
    assert pins[bars.relative_to(tmp_path).as_posix()] == file_sha256(bars)
    (archive / "_manifest.json").write_bytes(b"tampered")
    with pytest.raises(DataReadinessError):
        owner._corrected_child_pins(tmp_path, policy)
