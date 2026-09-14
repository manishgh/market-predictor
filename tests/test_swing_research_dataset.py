from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import market_predictor.swing.datasets.research_dataset as owner
from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_decisions import CorrectedDecisionProjection
from market_predictor.swing.datasets.outcome_replay import OutcomeReplayVerification
from market_predictor.swing.datasets.predictor_replay import VerifiedPredictorReplay
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from tests.test_swing_features import contract as contract
from tests.test_swing_research_ablation import _authority
from tests.test_swing_research_partition import _inputs


def _json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")
    return file_sha256(path)


def _artifact(path: Path, frame: pd.DataFrame, kind: str, inputs: dict[str, str]) -> dict[str, Any]:
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="test_fixture", status="pass",
        failures=0, rows_checked=len(frame), detail="Test-only synthetic rows."),))
    write_canonical_artifact(frame, path, artifact_type=kind, inputs=inputs, audit=audit, production_ready=False)
    return dict(path=path.name, sha256=file_sha256(path), manifest_sha256=file_sha256(manifest_path_for(path)),
        rows=len(frame), decision_ids_sha256=json_sha256(sorted(frame.decision_id)) if "decision_id" in frame else "0" * 64)


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: Any) -> dict[str, Any]:
    root = tmp_path
    package = root / "src/market_predictor"
    names = ("swing/datasets/research_dataset.py", "swing/datasets/predictor_abstention_derivation.py",
        "swing/features/research_ablation.py",
        "swing/features/research_partition.py", "swing/features/research_join.py", "swing/features/catalyst_aggregates.py",
        "swing/features/catalyst_decision_authority.py", "swing/features/catalyst_decision_identity.py",
        "swing/features/cross_sectional.py", "swing/features/pipeline.py")
    for name in names:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Test-only implementation identity.\n", encoding="ascii")
    monkeypatch.setattr(owner, "__file__", str(package / names[0]))
    configs = {name: root / f"configs/{name}.json" for name in ("decision_config", "strategy_config")}
    for path in configs.values():
        _json(path, {"fixture": path.stem})
    state: dict[str, Any] = dict(root=root, output=root / "data/features/join", active=False, calls=[],
        partitions=[], authorities={}, context_error=False, configs=configs)
    for name, path in configs.items():
        state[name] = SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))
    source_pins = {pin.path: pin.sha256 for pin in (state["decision_config"], state["strategy_config"])}
    dirs = {name: root / f"data/research/{name}" for name in ("predictors", "outcomes", "catalysts")}
    cohort = "c" * 64
    technical_request = dict(schema="market_predictor.research_predictor_request", declared_source_files=source_pins,
        source_files={}, implementation_files={}, adjusted_source_files={}, cohort_sha256=cohort,
        decision_start="2019-07-09", numeric_end="2024-05-28", expected_rows=6)
    target_request = dict(schema="market_predictor.corrected_outcome_request", lineage=dict(
        config_sha256=state["decision_config"].sha256, source_files=source_pins,
        implementation_files={}, cohort_sha256=cohort))
    tech = dict(schema="market_predictor.research_predictors", status="technical_inputs_complete_research_only",
        request_sha256=_json(dirs["predictors"] / "_request.json", technical_request), failed_groups={},
        groups={"fixture": dict(availability_columns={name: "base_available_at" for name in TECHNICAL_RANKING_FEATURES})},
        months={}, rows=6, training_eligible=False, promotion_eligible=False, exclusions_added=[])
    targets = dict(schema="market_predictor.corrected_outcomes", status="partial_research_outcomes",
        request_sha256=_json(dirs["outcomes"] / "_request.json", target_request), months={}, rows=6,
        training_eligible=False, promotion_eligible=False, managed_available=False, exclusions_added=[])
    news = dict(schema="market_predictor.initial_fit_catalyst_months", status="complete_research_only",
        cohort_sha256=cohort, rows=6, source_files=source_pins, months={})
    for day in ("2024-01-02", "2024-02-01"):
        month = day[:7]
        expected, features, outcomes = (frame.iloc[:3].copy() for frame in _inputs())
        for frame in (expected, features, outcomes):
            frame["decision_id"] = frame.decision_id + month
            frame["decision_time_utc"] = pd.Timestamp(f"{day}T23:00:00Z")
            frame["session_date_et"] = pd.Timestamp(day).date()
        features = features.drop(columns="label_eligible", errors="ignore")
        features["base_available_at"] = features.decision_time_utc - pd.Timedelta(hours=1)
        authority = _authority(expected, unknown=True)
        state["partitions"].append((month, expected))
        state["authorities"][month] = authority
        tech["months"][month] = _artifact(dirs["predictors"] / f"months/{month}.parquet", features,
            "swing_research_technical_inputs", {"research_feature_request_sha256": tech["request_sha256"]})
        target_dir = dirs["outcomes"] / month
        target_dir.mkdir(parents=True)
        outcomes.to_parquet(target_dir / "targets.parquet", index=False)
        targets["months"][month] = dict(rows=len(expected), decision_ids_sha256=json_sha256(sorted(expected.decision_id)),
            files={"targets.parquet": file_sha256(target_dir / "targets.parquet")})
        news_dir = dirs["catalysts"] / month
        artifacts = {"decisions": _artifact(news_dir / "decision_catalysts.parquet", authority.decisions,
            "test_catalysts", {}), "coverage": _artifact(news_dir / "source_coverage.parquet", authority.coverage,
            "test_coverage", {})}
        manifest_pin = _json(news_dir / "_manifest.json", dict(artifacts=artifacts))
        news["months"][month] = dict(directory=month, rows=len(expected), catalyst_decision_rows=len(authority.decisions),
            decision_ids_sha256=json_sha256(sorted(expected.decision_id)),
            authority_sha256=_json(news_dir / "_authority.json", dict(artifact_sha256=manifest_pin)))
    state.update(dirs=dirs, manifests=dict(predictors=tech, outcomes=targets, catalysts=news))
    for name in dirs:
        _repin(state, name)

    @contextmanager
    def projection(**kwargs: Any) -> Any:
        state["active"] = True
        try:
            yield CorrectedDecisionProjection(cohort, tuple(state["partitions"][0][1].security_id), 6,
                source_pins, iter((month, frame.copy()) for month, frame in state["partitions"]))
            if state["context_error"]:
                raise DataReadinessError("test context final source check failed")
        finally:
            state["active"] = False

    def load_news(path: Path, **kwargs: Any) -> Any:
        assert state["active"]
        state["calls"].append(("news", path.name))
        owner.pinned_object(path / "_authority.json", kwargs["expected_authority_sha256"])
        return state["authorities"][path.name]

    monkeypatch.setattr(owner, "verified_corrected_decision_partitions", projection)
    monkeypatch.setattr(owner, "load_catalyst_decision_authority", load_news)
    monkeypatch.setattr(owner, "load_strategy_contract", lambda *args: contract)
    monkeypatch.setattr(owner, "_guard", lambda: state["calls"].append(("guard", None)))
    monkeypatch.setattr(owner, "release_process_memory", lambda: state["calls"].append(("release", None)))
    return state


def _repin(state: dict[str, Any], name: str) -> None:
    path = state["dirs"][name] / "_manifest.json"
    state[name] = SourcePin(path=path.relative_to(state["root"]).as_posix(), sha256=_json(path, state["manifests"][name]))


def _set_news(state: dict[str, Any], month: str, authority: Any) -> None:
    state["authorities"][month] = authority
    directory = state["dirs"]["catalysts"] / month
    artifacts = {"decisions": _artifact(directory / "decision_catalysts.parquet", authority.decisions,
        "test_catalysts", {}), "coverage": _artifact(directory / "source_coverage.parquet", authority.coverage,
        "test_coverage", {})}
    manifest_pin = _json(directory / "_manifest.json", dict(artifacts=artifacts))
    state["manifests"]["catalysts"]["months"][month].update(catalyst_decision_rows=len(authority.decisions),
        authority_sha256=_json(directory / "_authority.json", dict(artifact_sha256=manifest_pin)))
    _repin(state, "catalysts")


def _run(state: dict[str, Any], resume: bool = False, **kwargs: Any) -> dict[str, Any]:
    arguments = {name: state[name] for name in ("root", "output", "decision_config", "strategy_config",
        "predictors", "outcomes", "catalysts")}
    if resume:
        arguments["expected_checkpoint_sha256"] = file_sha256(state["output"] / "_checkpoint.json")
    return owner.materialize_research_dataset(**arguments, **kwargs)


def test_replay_dependency_partition_is_exact_not_a_source_path_exemption(tmp_path: Path) -> None:
    old = "src/market_predictor/swing/features/retained.py"
    other = "src/market_predictor/swing/features/unverified.py"
    blob = "data/evidence/objects/old.bin"
    result = owner._verified_historical_dependencies(tmp_path,
        {old: "a" * 64, other: "b" * 64, "data/source.json": "c" * 64},
        {old: "a" * 64}, {blob: "a" * 64, old: "d" * 64})
    assert result == {old: "d" * 64, other: "b" * 64, "data/source.json": "c" * 64, blob: "a" * 64}
    with pytest.raises(DataReadinessError, match="declarations differ"):
        owner._verified_historical_dependencies(tmp_path, {old: "a" * 64}, {old: "b" * 64}, {})
    with pytest.raises(DataReadinessError, match="declarations differ"):
        owner._verified_historical_dependencies(tmp_path, {}, {old: "a" * 64}, {})


@pytest.mark.parametrize("source", ["predictor", "outcome"])
def test_join_propagates_invalid_replay_receipt(
    publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    def reject(**kwargs: Any) -> Any:
        assert kwargs["publication"] == publication["predictors" if source == "predictor" else "outcomes"]
        raise DataReadinessError("replay evidence is incomplete or stale")

    monkeypatch.setattr(owner, f"verify_{source}_replay", reject)
    with pytest.raises(DataReadinessError, match="incomplete or stale"):
        _run(publication, **{f"{source}_replay": SourcePin(path="data/reports/replay/_manifest.json", sha256="a" * 64)})
    assert not publication["output"].exists()


def test_successful_receipt_join_retains_verified_evidence(
    publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipts: dict[str, SourcePin] = {}
    calls: list[str] = []
    for name in ("predictor", "outcome"):
        path = publication["root"] / f"data/reports/{name}/_manifest.json"
        receipts[name] = SourcePin(path=path.relative_to(publication["root"]).as_posix(),
            sha256=_json(path, {"test_fixture": name}))

    def predictors(**kwargs: Any) -> VerifiedPredictorReplay:
        assert kwargs["publication"] == publication["predictors"]
        assert kwargs["replay"] == receipts["predictor"]
        calls.append("predictor")
        return VerifiedPredictorReplay({}, {receipts["predictor"].path: receipts["predictor"].sha256})

    def outcomes(**kwargs: Any) -> OutcomeReplayVerification:
        assert kwargs["publication"] == publication["outcomes"]
        assert kwargs["replay"] == receipts["outcome"]
        calls.append("outcome")
        return OutcomeReplayVerification({}, {receipts["outcome"].path: receipts["outcome"].sha256})

    monkeypatch.setattr(owner, "verify_predictor_replay", predictors)
    monkeypatch.setattr(owner, "verify_outcome_replay", outcomes)
    result = _run(publication, predictor_replay=receipts["predictor"], outcome_replay=receipts["outcome"])
    request = owner.pinned_object(publication["output"] / "_request.json", result["request_sha256"])
    assert calls == ["predictor", "outcome"]
    assert result["status"] == "complete_research_only"
    for pin in receipts.values():
        assert request["source_files"][pin.path] == pin.sha256


def test_publish_and_resume_preserve_nullable_population(publication: dict[str, Any]) -> None:
    result = _run(publication)
    assert result["rows"] == 6 and result["status"] == "complete_research_only"
    assert result["training_eligible"] is False and result["promotion_eligible"] is False
    assert result["exclusions_added"] == [] and not publication["active"]
    for month, expected in publication["partitions"]:
        for profile, record in result["months"][month]["profiles"].items():
            rows = pd.read_parquet(publication["output"] / record["path"])
            assert rows.decision_id.tolist() == expected.decision_id.tolist()
            assert pd.isna(rows.loc[0, "fixed_horizon_net_return"])
            assert not rows.training_eligible.any()
            if profile == "catalyst_full":
                assert rows.event_count_3d.isna().all()
    original = (publication["output"] / "_manifest.json").read_bytes()
    assert _run(publication, resume=True)["manifest_sha256"] == result["manifest_sha256"]
    assert (publication["output"] / "_manifest.json").read_bytes() == original


@pytest.mark.parametrize("name", ["predictors", "outcomes", "catalysts"])
def test_top_level_pin_tamper_rejected(publication: dict[str, Any], name: str) -> None:
    (publication["dirs"][name] / "_manifest.json").write_text("{}", encoding="ascii")
    with pytest.raises(DataReadinessError):
        _run(publication)
    assert not publication["output"].exists()


@pytest.mark.parametrize("name", ["predictors", "outcomes", "catalysts"])
def test_input_schema_rejected(publication: dict[str, Any], name: str) -> None:
    publication["manifests"][name]["schema"] = "foreign"
    _repin(publication, name)
    with pytest.raises(DataReadinessError):
        _run(publication)


@pytest.mark.parametrize("field,value", [("rows", 2), ("decision_ids_sha256", "f" * 64)])
@pytest.mark.parametrize("name", ["predictors", "outcomes", "catalysts"])
def test_month_population_attestation_rejected(publication: dict[str, Any], name: str, field: str, value: Any) -> None:
    publication["manifests"][name]["months"]["2024-01"][field] = value
    _repin(publication, name)
    with pytest.raises(DataReadinessError):
        _run(publication)


@pytest.mark.parametrize("alias", [False, True])
def test_conflicting_source_pins_rejected(publication: dict[str, Any], alias: bool) -> None:
    path = publication["dirs"]["predictors"] / "_request.json"
    request = json.loads(path.read_text())
    name = publication["decision_config"].path
    request["source_files"] = {name.replace("/", "\\") if alias else name: "f" * 64}
    publication["manifests"]["predictors"]["request_sha256"] = _json(path, request)
    _repin(publication, "predictors")
    with pytest.raises(DataReadinessError):
        _run(publication)


def test_targets_must_have_an_independent_file_pin(publication: dict[str, Any]) -> None:
    publication["manifests"]["outcomes"]["months"]["2024-01"]["files"] = {}
    _repin(publication, "outcomes")
    with pytest.raises(DataReadinessError):
        _run(publication)


@pytest.mark.parametrize("source,relative", [("predictors", "months/2024-01.parquet"),
    ("outcomes", "2024-01/targets.parquet"), ("catalysts", "2024-01/decision_catalysts.parquet")])
def test_resumed_month_rechecks_source_children(publication: dict[str, Any], source: str, relative: str) -> None:
    _run(publication)
    (publication["dirs"][source] / relative).write_bytes(b"tampered fixture")
    with pytest.raises(DataReadinessError):
        _run(publication, resume=True)


def test_final_context_failure_cannot_publish_complete_manifest(publication: dict[str, Any]) -> None:
    publication["context_error"] = True
    with pytest.raises(DataReadinessError, match="context final source"):
        _run(publication)
    assert not (publication["output"] / "_manifest.json").exists()
    assert not publication["active"]


def test_memory_pressure_stops_before_publication(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def stop() -> None:
        raise MemoryBudgetError("fixture pressure")
    monkeypatch.setattr(owner, "_guard", stop)
    with pytest.raises(MemoryBudgetError, match="pressure"):
        _run(publication)
    assert not publication["output"].exists()


def test_completed_manifest_tamper_rejected(publication: dict[str, Any]) -> None:
    _run(publication)
    path = publication["output"] / "_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["training_eligible"] = True
    _json(path, manifest)
    with pytest.raises(DataReadinessError, match="manifest changed"):
        _run(publication, resume=True)


def test_completed_join_resume_rejects_identity_helper_change(publication: dict[str, Any]) -> None:
    _run(publication)
    helper = publication["root"] / "src/market_predictor/swing/features/catalyst_decision_identity.py"
    helper.write_text("# Different test-only implementation.\n", encoding="ascii")
    with pytest.raises(DataReadinessError, match="inputs changed"):
        _run(publication, resume=True)


def test_orphan_profile_recovers_only_by_reconstruction(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner._replace_json
    def interrupt(path: Path, value: Any) -> None:
        if path.name == "_checkpoint.json" and value["months"]:
            raise RuntimeError("test interruption")
        original(path, value)
    monkeypatch.setattr(owner, "_replace_json", interrupt)
    with pytest.raises(RuntimeError, match="test interruption"):
        _run(publication)
    monkeypatch.setattr(owner, "_replace_json", original)
    assert _run(publication, resume=True)["rows"] == 6


@pytest.mark.parametrize("field,value", [("ticker", "WRONG"), ("security_id", "foreign"),
    ("decision_time_utc", pd.Timestamp("2024-01-02T23:00:01Z")),
    ("latest_event_feature_available_at_utc", pd.Timestamp("2024-01-02T23:00:01Z"))])
def test_catalyst_identity_and_future_poison_rejected(publication: dict[str, Any], field: str, value: Any) -> None:
    if field.endswith("_utc"):
        publication["authorities"]["2024-01"].decisions[field] = pd.to_datetime(
            publication["authorities"]["2024-01"].decisions[field], utc=True)
    publication["authorities"]["2024-01"].decisions.loc[0, field] = value
    with pytest.raises(DataReadinessError):
        _run(publication)
    assert not (publication["output"] / "_manifest.json").exists()


@pytest.mark.parametrize("mutation", ["schema", "decision", "strategy", "dates", "adjusted"])
def test_request_bindings_rejected(publication: dict[str, Any], mutation: str) -> None:
    path = publication["dirs"]["predictors"] / "_request.json"
    request = json.loads(path.read_text())
    if mutation == "schema":
        request["schema"] = "foreign"
    elif mutation in {"decision", "strategy"}:
        del request["declared_source_files"][publication[f"{mutation}_config"].path]
    elif mutation == "dates":
        request["numeric_end"] = "2024-05-29"
    else:
        request["adjusted_source_files"] = {publication["decision_config"].path: "f" * 64}
    publication["manifests"]["predictors"]["request_sha256"] = _json(path, request)
    _repin(publication, "predictors")
    with pytest.raises(DataReadinessError):
        _run(publication)


@pytest.mark.parametrize("mutation", ["missing", "conflict", "extra"])
def test_group_clock_maps_rejected(publication: dict[str, Any], mutation: str) -> None:
    groups = publication["manifests"]["predictors"]["groups"]
    if mutation == "missing":
        groups.clear()
    else:
        clocks = dict(groups["fixture"]["availability_columns"])
        clocks[TECHNICAL_RANKING_FEATURES[0] if mutation == "conflict" else "foreign"] = "other_clock"
        groups["other"] = dict(availability_columns=clocks)
    _repin(publication, "predictors")
    with pytest.raises(DataReadinessError):
        _run(publication)


def test_source_change_during_target_read_is_rejected(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = pd.read_parquet
    def mutate(path: Any, *args: Any, **kwargs: Any) -> pd.DataFrame:
        result = original(path, *args, **kwargs)
        if Path(path).name == "targets.parquet":
            Path(path).write_bytes(b"changed after read")
        return result
    monkeypatch.setattr(pd, "read_parquet", mutate)
    with pytest.raises(DataReadinessError, match="source changed"):
        _run(publication)
    assert not (publication["output"] / "_manifest.json").exists()


@pytest.mark.parametrize("field,value", [("schema", "foreign"), ("training_eligible", True), ("exclusions_added", ["issuer"])])
def test_checkpoint_contract_rejected(publication: dict[str, Any], field: str, value: Any) -> None:
    _run(publication)
    path = publication["output"] / "_checkpoint.json"
    checkpoint = json.loads(path.read_text())
    checkpoint[field] = value
    _json(path, checkpoint)
    with pytest.raises(DataReadinessError, match="checkpoint"):
        _run(publication, resume=True)


def test_memory_stop_mid_month_is_resumable(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner._guard
    def stop() -> None:
        if ("news", "2024-02") in publication["calls"]:
            raise MemoryBudgetError("fixture pressure")
        original()
    monkeypatch.setattr(owner, "_guard", stop)
    with pytest.raises(MemoryBudgetError, match="pressure"):
        _run(publication)
    assert not publication["active"]
    assert set(json.loads((publication["output"] / "_checkpoint.json").read_text())["months"]) == {"2024-01"}
    assert not (publication["output"] / "_manifest.json").exists()
    monkeypatch.setattr(owner, "_guard", original)
    assert _run(publication, resume=True)["rows"] == 6


@pytest.mark.parametrize("target", ["profile", "request", "checkpoint"])
def test_output_changed_before_finalization_is_rejected(
    publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    original = owner._replace_json
    def mutate(path: Path, value: Any) -> None:
        original(path, value)
        if path.name == "_checkpoint.json" and len(value["months"]) == 2:
            relative = {"profile": "2024-01/technical_market.parquet", "request": "_request.json",
                "checkpoint": "_checkpoint.json"}[target]
            (publication["output"] / relative).write_bytes(b"{}")
    monkeypatch.setattr(owner, "_replace_json", mutate)
    with pytest.raises(DataReadinessError):
        _run(publication)
    assert not (publication["output"] / "_manifest.json").exists()


@pytest.mark.parametrize("field,value", [("model_columns", []), ("availability_columns", {}),
    ("path", "2024-01/catalyst_full.parquet")])
def test_resumed_profile_contract_rejected(publication: dict[str, Any], field: str, value: Any) -> None:
    _run(publication)
    path = publication["output"] / "_checkpoint.json"
    checkpoint = json.loads(path.read_text())
    checkpoint["months"]["2024-01"]["profiles"]["technical_market"][field] = value
    _json(path, checkpoint)
    with pytest.raises(DataReadinessError, match="profile feature contract"):
        _run(publication, resume=True)


@pytest.mark.parametrize("field,value", [("sector", "Energy"), ("primary_benchmark", "XLE"),
    ("session_date_et", "2024-01-03")])
def test_outcome_context_poison_rejected(publication: dict[str, Any], field: str, value: Any) -> None:
    path = publication["dirs"]["outcomes"] / "2024-01/targets.parquet"
    rows = pd.read_parquet(path)
    rows[field] = value
    rows.to_parquet(path, index=False)
    publication["manifests"]["outcomes"]["months"]["2024-01"]["files"]["targets.parquet"] = file_sha256(path)
    _repin(publication, "outcomes")
    with pytest.raises(DataReadinessError, match="differs from frozen"):
        _run(publication)


def test_future_technical_clock_rejected(publication: dict[str, Any]) -> None:
    path = publication["dirs"]["predictors"] / "months/2024-01.parquet"
    rows = pd.read_parquet(path)
    rows["base_available_at"] = rows.decision_time_utc + pd.Timedelta(seconds=1)
    manifest = publication["manifests"]["predictors"]
    manifest["months"]["2024-01"] = _artifact(path, rows, "swing_research_technical_inputs",
        {"research_feature_request_sha256": manifest["request_sha256"]})
    _repin(publication, "predictors")
    with pytest.raises(DataReadinessError, match="unavailable at its decision"):
        _run(publication)


def test_known_zero_catalysts_remain_zero(publication: dict[str, Any]) -> None:
    for month, expected in publication["partitions"]:
        _set_news(publication, month, _authority(expected, unknown=False))
    result = _run(publication)
    for record in result["months"].values():
        rows = pd.read_parquet(publication["output"] / record["profiles"]["catalyst_full"]["path"])
        assert rows.event_count_3d.eq(0).all()
        assert rows.catalyst_required_source_complete.all()


def test_outcome_readiness_claim_is_rejected(publication: dict[str, Any]) -> None:
    path = publication["dirs"]["outcomes"] / "2024-01/targets.parquet"
    rows = pd.read_parquet(path).assign(training_eligible=True)
    rows.to_parquet(path, index=False)
    publication["manifests"]["outcomes"]["months"]["2024-01"]["files"]["targets.parquet"] = file_sha256(path)
    _repin(publication, "outcomes")
    with pytest.raises(DataReadinessError, match="cannot claim admission"):
        _run(publication)


@pytest.mark.parametrize("count", [None, True, 2])
def test_sparse_catalyst_count_attestation_rejected(publication: dict[str, Any], count: Any) -> None:
    record = publication["manifests"]["catalysts"]["months"]["2024-01"]
    record["catalyst_decision_rows"] = count
    _repin(publication, "catalysts")
    with pytest.raises(DataReadinessError, match="sparse catalyst authority row count"):
        _run(publication)


def test_sparse_catalysts_retain_unknown_full_population(publication: dict[str, Any]) -> None:
    for month, _ in publication["partitions"]:
        authority = publication["authorities"][month]
        _set_news(publication, month, replace(authority, decisions=authority.decisions.iloc[:0].copy()))
    result = _run(publication)
    assert result["rows"] == 6
    for record in result["months"].values():
        for profile in record["profiles"].values():
            rows = pd.read_parquet(publication["output"] / profile["path"])
            assert len(rows) == 3 and not rows.training_eligible.any()
            if "event_count_3d" in rows:
                assert rows.event_count_3d.isna().all()
