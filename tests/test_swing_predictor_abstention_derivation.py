from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

import market_predictor.swing.datasets.predictor_abstention_derivation as owner
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_decisions import CorrectedDecisionProjection
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from tests.test_swing_features import contract as contract
from tests.test_swing_research_dataset import _run as _run_consumer
from tests.test_swing_research_dataset import publication as publication


def _pin(root: Path, path: Path) -> SourcePin:
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))


@pytest.fixture
def example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path
    parent = root / "data/features/parent"
    parent.mkdir(parents=True)
    config = root / "decisions.toml"
    config.write_text("# synthetic decision source", encoding="ascii")
    evidence = root / "reviewed-observations.json"
    source = root / "adjusted.parquet"
    source.write_bytes(b"not a parquet: source bytes must only be hashed, never decoded")
    write_json_object(evidence, dict(schema="market_predictor.predictor_source_failure_observations.v1",
        numeric_first="2018-05-29", numeric_last="2024-05-28", combined_manifest_sha256="c" * 64,
        observations=[dict(security_id="issuer-WTW", ticker="WTW", source_path=source.name,
            source_sha256=file_sha256(source), bounded_rows=1, invalid_rows=[])]))
    implementation = root / "src/market_predictor/swing/datasets/predictor_abstention_derivation.py"
    implementation.parent.mkdir(parents=True)
    implementation.write_text("# synthetic implementation", encoding="ascii")
    producer = root / "producer.py"
    producer.write_text("# frozen producer", encoding="ascii")
    feature_config = root / "features.toml"
    feature_config.write_text("# synthetic whole-stream-only policy; never opened for numeric replay", encoding="ascii")
    monkeypatch.setattr(owner, "__file__", str(implementation))
    monkeypatch.setattr(owner, "_guard", lambda: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    monkeypatch.setattr(owner, "heavy_job_runtime_dir", lambda: Path("data/runtime"))
    decisions = pd.DataFrame([dict(decision_id=f"{symbol}-{day}", parent_decision_id=f"parent-{symbol}-{day}",
        security_id=f"issuer-{symbol}", ticker=symbol, parent_ticker=symbol, session_date_et=date.fromisoformat(day),
        decision_time_utc=pd.Timestamp(f"{day}T22:00:00Z"), sector="Technology", primary_benchmark="XLK")
        for symbol in ("AAA", "WTW") for day in ("2019-07-09", "2019-08-01")])
    request = dict(schema="market_predictor.research_predictor_request", config_sha256=file_sha256(feature_config),
        declared_source_files={config.name: file_sha256(config), feature_config.name: file_sha256(feature_config)}, source_files={},
        implementation_files={producer.name: file_sha256(producer)},
        adjusted_source_files={source.name: file_sha256(source)}, cohort_sha256="c" * 64,
        decision_start="2019-07-09", numeric_end="2024-05-28", expected_rows=len(decisions),
        historical_first_seen_proven=False, retained_security_ids=sorted(set(decisions.security_id)),
        decision_ids_sha256=json_sha256(sorted(decisions.decision_id)),
        adjusted_feature_columns=[n for n in TECHNICAL_RANKING_FEATURES if n != "dollar_volume_log"],
        raw_feature_columns=["dollar_volume_log"], separate_symbol_streams="independent_warmup_no_price_splice")
    write_json_object(parent / "_request.json", request)
    good = decisions.loc[decisions.ticker.eq("AAA")].drop(columns="parent_ticker").copy()
    good[list(TECHNICAL_RANKING_FEATURES)] = 0.25
    good["feature_eligible"] = True
    good["feature_profile"] = "technical_market"
    good["daily_bar_count"] = 300
    good["technical_missing_reasons"] = "[]"
    good["technical_available_at_utc"] = good.decision_time_utc - pd.Timedelta(hours=1)
    good["raw_dollar_volume_available_at_utc"] = good.technical_available_at_utc
    good["membership_available_at_utc"] = good.technical_available_at_utc
    good["universe_snapshot_id"] = "verified-membership"
    clocks = {n: "raw_dollar_volume_available_at_utc" if n == "dollar_volume_log" else "technical_available_at_utc"
        for n in TECHNICAL_RANKING_FEATURES}
    key = json_sha256(["issuer-AAA", "AAA"])
    fail_key = json_sha256(["issuer-WTW", "WTW"])
    part = owner._publish(good, parent / "groups" / f"{key}.parquet", file_sha256(parent / "_request.json"))
    failure = dict(security_id="issuer-WTW", symbol="WTW", error_type="DataReadinessError",
        reason="adjusted history has invalid or placeholder OHLCV/clocks", rows=2)
    checkpoint = dict(schema="market_predictor.research_predictors", status="partial_in_progress",
        request_sha256=file_sha256(parent / "_request.json"), groups={key: {**part, "availability_columns": clocks}},
        failed_groups={fail_key: failure}, months={}, training_eligible=False, promotion_eligible=False, exclusions_added=[])
    write_json_object(parent / "_checkpoint.json", checkpoint)
    fact = dict(group_key=fail_key, security_id="issuer-WTW", symbol="WTW", rows=2,
        parent_failure_sha256=json_sha256(failure), reason_code="unverified_issuer_history", quarantine="entire_failed_group",
        first_invalid_session=None, boundary_observation_sha256=None,
        source_artifacts=[_pin(root, source).model_dump()], reviewed_evidence=[_pin(root, evidence).model_dump()],
        detail="Entire unverified issuer warmup quarantined; not just its zero-volume rows.")
    facts = dict(schema_version="market_predictor.predictor_failure_facts.v1",
        parent_checkpoint_sha256=file_sha256(parent / "_checkpoint.json"), parent_request_sha256=checkpoint["request_sha256"],
        decision_config=_pin(root, config).model_dump(), parent_run_finished=True,
        feature_config=_pin(root, feature_config).model_dump(), observations=_pin(root, evidence).model_dump(),
        approval_scope="causal_prefix_replay_and_nullable_completion_only", reviewed_by="test-only independent review", failures=[fact])
    facts_path = root / "approved.json"
    write_json_object(facts_path, facts)
    state: dict[str, Any] = dict(root=root, parent=parent, output=root / "data/features/derived", facts_path=facts_path,
        facts=facts, checkpoint=checkpoint, request=request, decisions=decisions, key=key, fail_key=fail_key,
        source=source, evidence=evidence, producer=producer, config=config, feature_config=feature_config, implementation=implementation,
        exited=False, fail_exit=False, before_exit=None, partitions_read=0)

    @contextmanager
    def projection(**kwargs: Any) -> Any:
        assert kwargs["config"] == config and kwargs["expected_config_sha256"] == file_sha256(config)
        with heavy_job_lease("synthetic metadata projection", runtime_dir=root / "data/runtime"):
            def partitions() -> Any:
                for month in ("2019-07", "2019-08"):
                    state["partitions_read"] += 1
                    yield month, decisions.loc[decisions.session_date_et.map(lambda d: d.strftime("%Y-%m")).eq(month)].copy()
            yield CorrectedDecisionProjection("c" * 64, tuple(request["retained_security_ids"]), len(decisions),
                {config.name: file_sha256(config)}, partitions())
            assert not state["output"].exists(), "completion exposed before source exit checks"
            if state["before_exit"]:
                state["before_exit"]()
            if state["fail_exit"]:
                raise DataReadinessError("source exit failed")
            state["exited"] = True
    monkeypatch.setattr(owner, "verified_corrected_decision_partitions", projection)
    return state


def _run(state: dict[str, Any]) -> dict[str, Any]:
    return owner.derive_predictors_with_abstentions(root=state["root"], output=state["output"],
        parent_checkpoint=_pin(state["root"], state["parent"] / "_checkpoint.json"),
        approved_failure_facts=_pin(state["root"], state["facts_path"]))


def _refreeze(state: dict[str, Any]) -> None:
    request_path = state["parent"] / "_request.json"
    if json.loads(request_path.read_text()) != state["request"]:
        request_path.write_text(json.dumps(state["request"]), encoding="ascii")
    state["checkpoint"]["request_sha256"] = file_sha256(state["parent"] / "_request.json")
    (state["parent"] / "_checkpoint.json").write_text(json.dumps(state["checkpoint"]), encoding="ascii")
    state["facts"]["parent_checkpoint_sha256"] = file_sha256(state["parent"] / "_checkpoint.json")
    state["facts"]["parent_request_sha256"] = state["checkpoint"]["request_sha256"]
    state["facts_path"].write_text(json.dumps(state["facts"]), encoding="ascii")


def test_complete_nullable_derivation_preserves_success_bytes_and_every_identity(example: dict[str, Any]) -> None:
    parent_bytes = {p.relative_to(example["parent"]): p.read_bytes() for p in example["parent"].rglob("*") if p.is_file()}
    result = _run(example)
    assert example["exited"] and example["partitions_read"] == 2
    assert result["rows"] == 4 and result["feature_eligible_rows"] == 2 and result["unavailable_rows"] == 2
    assert result["failed_groups"] == {} and result["source_failures"] == example["checkpoint"]["failed_groups"]
    assert result["exclusions_added"] == [] and result["training_eligible"] is False and result["promotion_eligible"] is False
    assert result["status"] == "technical_inputs_complete_research_only"
    for path, payload in parent_bytes.items():
        assert (example["parent"] / path).read_bytes() == payload
        if path.parts[0] == "groups" and not path.name.endswith(".lock"):
            assert (example["output"] / path).read_bytes() == payload
    seen = []
    for part in result["months"].values():
        frame, canonical = load_canonical_artifact(example["output"] / "months" / part["path"],
            expected_type="swing_research_technical_inputs", allow_research=True)
        assert canonical["inputs"] == {"research_feature_request_sha256": result["request_sha256"]}
        seen.extend(frame.decision_id)
        bad = frame.loc[frame.ticker.eq("WTW")]
        assert bad[list(TECHNICAL_RANKING_FEATURES)].isna().all().all()
        assert bad[["technical_available_at_utc", "raw_dollar_volume_available_at_utc"]].isna().all().all()
        assert bad.membership_available_at_utc.isna().all() and bad.universe_snapshot_id.isna().all()
        assert not bad.feature_eligible.any() and bad.daily_bar_count.eq(0).all()
        assert bad.technical_missing_reasons.str.contains("unverified_issuer_history").all()
        assert frame.parent_decision_id.str.startswith("parent-").all()
        assert frame.loc[frame.ticker.eq("AAA"), "universe_snapshot_id"].eq("verified-membership").all()
    assert sorted(seen) == sorted(example["decisions"].decision_id)
    request = json.loads((example["output"] / "_request.json").read_text())
    pins = owner.validate_predictor_derivation(root=example["root"], manifest=result, request=request)
    for path in (example["facts_path"], example["parent"] / "_checkpoint.json", example["parent"] / "_request.json", example["evidence"]):
        assert pins[path.relative_to(example["root"]).as_posix()] == file_sha256(path)
    assert request["retained_security_ids"] == example["request"]["retained_security_ids"]
    assert not any("label" in name or "target" in name for name in frame)


def test_active_lease_rejects_before_decision_or_numeric_reads(example: dict[str, Any]) -> None:
    with heavy_job_lease("other job", runtime_dir=example["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            _run(example)
    assert example["partitions_read"] == 0 and not example["output"].exists()


@pytest.mark.parametrize("change", ["unfinished", "foreign", "overlap", "unapproved", "duplicate", "rows", "identity",
    "failure_hash", "source"])
def test_rejects_nonexhaustive_or_unreviewed_parent(example: dict[str, Any], change: str) -> None:
    checkpoint, facts = example["checkpoint"], example["facts"]
    if change == "unfinished":
        checkpoint["groups"].clear()
    elif change == "foreign":
        checkpoint["groups"]["f" * 64] = checkpoint["groups"][example["key"]]
    elif change == "overlap":
        checkpoint["groups"][example["fail_key"]] = checkpoint["groups"][example["key"]]
    elif change == "unapproved":
        checkpoint["failed_groups"]["e" * 64] = dict(checkpoint["failed_groups"][example["fail_key"]])
    elif change == "duplicate":
        facts["failures"].append(dict(facts["failures"][0]))
    elif change == "rows":
        facts["failures"][0]["rows"] += 1
    elif change == "identity":
        facts["failures"][0]["security_id"] = "wrong-issuer"
    elif change == "failure_hash":
        facts["failures"][0]["parent_failure_sha256"] = "0" * 64
    elif change == "source":
        facts["failures"][0]["source_artifacts"] = [_pin(example["root"], example["evidence"]).model_dump()]
    _refreeze(example)
    with pytest.raises(DataReadinessError):
        _run(example)
    assert not example["output"].exists()


@pytest.mark.parametrize("path_name", ["source", "evidence", "producer", "implementation"])
def test_source_and_implementation_tamper_fail_closed(example: dict[str, Any], path_name: str) -> None:
    if path_name == "implementation":
        example["before_exit"] = lambda: example[path_name].write_text("tampered", encoding="ascii")
    else:
        example[path_name].write_text("tampered", encoding="ascii")
    with pytest.raises(DataReadinessError, match="changed|independent file pin"):
        _run(example)
    assert not example["output"].exists()


@pytest.mark.parametrize("field,value", [("parent_run_finished", False), ("approval_scope", "source_admission"),
    ("unrecognized", True)])
def test_facts_strict_scope(example: dict[str, Any], field: str, value: Any) -> None:
    example["facts"][field] = value
    _refreeze(example)
    with pytest.raises(ValidationError):
        _run(example)


def test_source_exit_failure_never_publishes_completion(example: dict[str, Any]) -> None:
    example["fail_exit"] = True
    with pytest.raises(DataReadinessError, match="source exit failed"):
        _run(example)
    assert not example["output"].exists()
    assert not list(example["output"].parent.glob(".*.pending/_manifest.json"))


def test_inherited_artifact_tamper_rejected(example: dict[str, Any]) -> None:
    part = example["checkpoint"]["groups"][example["key"]]
    (example["parent"] / "groups" / part["path"]).write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="changed"):
        _run(example)


def test_independent_checkpoint_pin_required(example: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="independent file pin"):
        owner.derive_predictors_with_abstentions(root=example["root"], output=example["output"],
            parent_checkpoint=SourcePin(
                path=(example["parent"] / "_checkpoint.json").relative_to(example["root"]).as_posix(), sha256="0" * 64),
            approved_failure_facts=_pin(example["root"], example["facts_path"]))


def test_existing_completed_output_is_immutable(example: dict[str, Any]) -> None:
    _run(example)
    before = (example["output"] / "_manifest.json").read_bytes()
    with pytest.raises(DataReadinessError, match="new disjoint"):
        _run(example)
    assert (example["output"] / "_manifest.json").read_bytes() == before


@pytest.mark.parametrize("change", ["drop_failures", "drop_facts_pin", "drop_parent_pin", "changed_success", "unavailable_count"])
def test_consumer_rejects_missing_or_inconsistent_derivation_lineage(example: dict[str, Any], change: str) -> None:
    result = _run(example)
    request = json.loads((example["output"] / "_request.json").read_text())
    if change == "drop_failures":
        result["source_failures"] = {}
    elif change == "drop_facts_pin":
        request["source_files"].pop(example["facts_path"].relative_to(example["root"]).as_posix())
    elif change == "drop_parent_pin":
        request["source_files"].pop((example["parent"] / "_checkpoint.json").relative_to(example["root"]).as_posix())
    elif change == "changed_success":
        result["groups"][example["key"]]["sha256"] = "0" * 64
    else:
        result["unavailable_rows"] = 0
    with pytest.raises(DataReadinessError):
        owner.validate_predictor_derivation(root=example["root"], manifest=result, request=request)


def test_consumer_original_non_derived_path_unchanged() -> None:
    assert owner.validate_predictor_derivation(root=Path.cwd(), manifest={}, request={}) == {}
    with pytest.raises(DataReadinessError, match="incomplete"):
        owner.validate_predictor_derivation(root=Path.cwd(), manifest={"source_failures": {}}, request={})


def test_group_sidecar_bytes_preserved(example: dict[str, Any]) -> None:
    result = _run(example)
    part = result["groups"][example["key"]]
    assert manifest_path_for(example["parent"] / "groups" / part["path"]).read_bytes() == manifest_path_for(
        example["output"] / "groups" / part["path"]).read_bytes()
    assert part["origin_request_sha256"] == example["checkpoint"]["request_sha256"]


def test_consumer_pins_derivation_dependencies(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import market_predictor.swing.datasets.research_dataset as consumer
    state = publication
    evidence = state["root"] / "derivation-evidence.json"
    evidence.write_text('{"synthetic": true}', encoding="ascii")
    calls = []

    def validate(**kwargs: Any) -> dict[str, str]:
        calls.append(kwargs)
        return {evidence.name: file_sha256(evidence)}

    monkeypatch.setattr(consumer, "validate_predictor_derivation", validate)
    result = _run_consumer(state)
    assert result["rows"] == 6 and len(calls) == 1
    saved = json.loads((state["output"] / "_request.json").read_text())
    assert saved["source_files"][evidence.name] == file_sha256(evidence)


def test_consumer_rejects_invalid_derivation_before_join(publication: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import market_predictor.swing.datasets.research_dataset as consumer

    def reject(**kwargs: Any) -> dict[str, str]:
        raise DataReadinessError("approved facts changed")

    monkeypatch.setattr(consumer, "validate_predictor_derivation", reject)
    with pytest.raises(DataReadinessError, match="approved facts changed"):
        _run_consumer(publication)
    assert not publication["output"].exists()
    assert not any(call[0] == "news" for call in publication["calls"])


def test_pending_group_cannot_be_mistaken_for_finished_exhaustive_run(example: dict[str, Any]) -> None:
    decisions = example["decisions"]
    extra = decisions.iloc[0].copy()
    for column in ("ticker", "parent_ticker"):
        extra[column] = "CCC"
    extra["security_id"] = "issuer-CCC"
    extra["decision_id"] = "CCC-2019-07-09"
    extra["parent_decision_id"] = "parent-CCC-2019-07-09"
    decisions.loc[len(decisions)] = extra
    example["request"]["expected_rows"] = len(decisions)
    example["request"]["decision_ids_sha256"] = json_sha256(sorted(decisions.decision_id))
    example["request"]["retained_security_ids"].append("issuer-CCC")
    _refreeze(example)
    with pytest.raises(DataReadinessError, match="not exhaustive"):
        _run(example)
    assert not example["output"].exists()


def test_staged_month_tamper_cannot_publish_after_source_exit(example: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    publish = owner._publish
    def tamper(frame: pd.DataFrame, path: Path, pin: str) -> dict[str, Any]:
        result = publish(frame, path, pin)
        if path.parent.name == "months":
            path.write_bytes(b"changed after construction")
        return result
    monkeypatch.setattr(owner, "_publish", tamper)
    with pytest.raises(DataReadinessError, match="partition changed"):
        _run(example)
    assert example["exited"] and not example["output"].exists()
    assert not list(example["output"].parent.glob(".*.pending/_manifest.json"))


def test_finalization_does_not_race_another_heavy_job(example: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    @contextmanager
    def busy(*args: Any, **kwargs: Any) -> Any:
        raise HeavyJobBusyError("another job won the finalization lease")
        yield  # pragma: no cover
    monkeypatch.setattr(owner, "heavy_job_lease", busy)
    with pytest.raises(HeavyJobBusyError, match="finalization lease"):
        _run(example)
    assert example["exited"] and not example["output"].exists()
    assert not list(example["output"].parent.glob(".*.pending/_manifest.json"))


def test_independent_failure_facts_pin_required(example: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="independent file pin"):
        owner.derive_predictors_with_abstentions(root=example["root"], output=example["output"],
            parent_checkpoint=_pin(example["root"], example["parent"] / "_checkpoint.json"),
            approved_failure_facts=SourcePin(path=example["facts_path"].name, sha256="0" * 64))


def _prefix_case() -> tuple[dict[str, Any], owner.ReviewedPredictorFailure, dict[str, Any]]:
    from tests.test_swing_adjusted_source import _source_inputs
    args = _source_inputs()
    group = args["decisions"]
    group["parent_decision_id"] = "parent-" + group.decision_id
    stock = args["adjusted_bars"]
    stock.loc[stock.index[290], "volume"] = 0.0
    row = stock.iloc[290]
    day = row.bar_start_utc.tz_convert("America/New_York").date()
    invalid = dict(session_date_et=str(day), invalid_fields=["volume"],
        ohlcv={name: float(row[name]) for name in ("open", "high", "low", "close", "volume")},
        clocks={name: str(row[name]) for name in ("bar_start_utc", "bar_end_utc", "available_at_utc")})
    fact = owner.ReviewedPredictorFailure(group_key=json_sha256(["issuer-a", "AAA"]), security_id="issuer-a", symbol="AAA",
        rows=len(group), parent_failure_sha256="a" * 64, reason_code="invalid_observation_stream",
        quarantine="suffix_from_first_invalid", first_invalid_session=day, boundary_observation_sha256=json_sha256(invalid),
        source_artifacts=(SourcePin(path="synthetic.parquet", sha256="a" * 64),),
        reviewed_evidence=(SourcePin(path="synthetic.json", sha256="b" * 64),), detail="Test-only reviewed zero-volume boundary.")
    return args, fact, dict(bounded_rows=len(stock), invalid_rows=[invalid])


def _rebuild(args: dict[str, Any], fact: owner.ReviewedPredictorFailure, observation: dict[str, Any], contract: Any) -> pd.DataFrame:
    clocks = {name: "raw_dollar_volume_available_at_utc" if name == "dollar_volume_log" else "technical_available_at_utc"
        for name in TECHNICAL_RANKING_FEATURES}
    raw = args["raw_decision_bars"].loc[args["raw_decision_bars"].session_date_et.lt(fact.first_invalid_session)]
    return owner._rebuild_prefix(args["decisions"], args["adjusted_bars"], args["benchmark_bars"], args["memberships"], raw,
        fact=fact, observation=observation, expected_history_sessions=args["expected_history_sessions"], contract=contract,
        clocks=clocks).rows


def test_terminal_zero_preserves_actual_canonical_earlier_features(contract: Any) -> None:
    from market_predictor.swing.features.adjusted_source import build_adjusted_technical_source
    args, fact, observation = _prefix_case()
    clean = {**args, "adjusted_bars": args["adjusted_bars"].copy()}
    clean["adjusted_bars"].loc[clean["adjusted_bars"].index[290], "volume"] = 1_000_000.0
    baseline = build_adjusted_technical_source(**clean, contract=contract).rows
    actual = _rebuild(args, fact, observation, contract)
    before = baseline.session_date_et.lt(fact.first_invalid_session)
    columns = [*TECHNICAL_RANKING_FEATURES, "technical_available_at_utc", "raw_dollar_volume_available_at_utc",
        "feature_eligible", "daily_bar_count"]
    pd.testing.assert_frame_equal(baseline.loc[before, columns].reset_index(drop=True), actual.loc[before, columns].reset_index(drop=True))
    assert actual.loc[before, "feature_eligible"].all()
    assert len(actual) == len(baseline) and actual.loc[~before, list(TECHNICAL_RANKING_FEATURES)].isna().all().all()
    assert actual.loc[~before, ["technical_available_at_utc", "raw_dollar_volume_available_at_utc"]].isna().all().all()
    assert not actual.loc[~before, "feature_eligible"].any()
    assert actual.loc[~before, "session_date_et"].min() == fact.first_invalid_session
    assert not any(name in actual for name in ("future_net_return_10d", "return_20d_xs_rank"))


def test_future_suffix_poison_cannot_change_earlier_values_or_peer_population(contract: Any) -> None:
    from market_predictor.swing.features.research_join import rebuild_research_peer_features
    from tests.test_swing_features import _panel
    args, fact, observation = _prefix_case()
    before = _rebuild(args, fact, observation, contract)
    stock = args["adjusted_bars"]
    stock.loc[stock.index[291:], ["open", "high", "low", "close"]] *= 100
    stock.loc[stock.index[291:], "volume"] = 0
    after = _rebuild(args, fact, observation, contract)
    pd.testing.assert_frame_equal(before, after)
    panel = _panel(sessions=1, securities=60)
    cutoff = before.decision_time_utc.iloc[0]
    panel["decision_time_utc"] = cutoff
    panel["session_date_et"] = cutoff.date()
    panel["decision_id"] = panel.security_id + "-decision"
    panel["parent_decision_id"] = "parent-" + panel.decision_id
    panel["primary_benchmark"] = "XLK"
    panel["technical_available_at_utc"] = cutoff - pd.Timedelta(hours=1)
    panel["raw_dollar_volume_available_at_utc"] = cutoff - pd.Timedelta(hours=1)
    columns = [*TECHNICAL_RANKING_FEATURES, "feature_eligible", "technical_available_at_utc", "raw_dollar_volume_available_at_utc"]
    panel.loc[59, columns] = before.loc[0, columns]
    clocks = {name: "raw_dollar_volume_available_at_utc" if name == "dollar_volume_log" else "technical_available_at_utc"
        for name in TECHNICAL_RANKING_FEATURES}
    baseline, _ = rebuild_research_peer_features(panel, contract=contract, retained_security_ids=frozenset(panel.security_id),
        availability_columns=clocks)
    panel.loc[59, columns] = after.loc[0, columns]
    actual, _ = rebuild_research_peer_features(panel, contract=contract, retained_security_ids=frozenset(panel.security_id),
        availability_columns=clocks)
    pd.testing.assert_frame_equal(baseline, actual)
    assert actual.iloc[:59].sector_peer_count.eq(60).all()


@pytest.mark.parametrize("bad", ["earlier_zero", "missing_boundary", "wrong_values", "wrong_clock", "heldout", "valid_boundary"])
def test_prefix_replay_proves_first_bad_row_instead_of_trusting_date(bad: str, contract: Any) -> None:
    args, fact, observation = _prefix_case()
    bars = args["adjusted_bars"]
    if bad == "earlier_zero":
        bars.loc[bars.index[100], "volume"] = 0
    elif bad == "missing_boundary":
        args["adjusted_bars"] = bars.drop(index=bars.index[290])
    elif bad == "wrong_values":
        bars.loc[bars.index[290], "close"] += 1
    elif bad == "wrong_clock":
        bars.loc[bars.index[290], "available_at_utc"] += pd.Timedelta(hours=1)
    elif bad == "heldout":
        bars.loc[bars.index[-1], "bar_start_utc"] = pd.Timestamp("2025-01-02T14:30Z")
    else:
        bars.loc[bars.index[290], "volume"] = 1
    with pytest.raises(DataReadinessError):
        _rebuild(args, fact, observation, contract)


def test_prefix_replay_preserves_calendar_gaps_and_clean_warmup_recovery(contract: Any) -> None:
    args, fact, observation = _prefix_case()
    args["adjusted_bars"] = args["adjusted_bars"].drop(index=35)
    observation["bounded_rows"] -= 1
    actual = _rebuild(args, fact, observation, contract)
    assert actual.iloc[:5].return_20d.isna().all() and not actual.iloc[:5].feature_eligible.any()
    assert actual.iloc[5:10].return_20d.notna().all() and actual.iloc[5:10].feature_eligible.all()
    assert actual.iloc[10:][list(TECHNICAL_RANKING_FEATURES)].isna().all().all()


def test_date_only_or_blanket_observation_failure_approval_is_rejected(example: dict[str, Any]) -> None:
    fact = example["facts"]["failures"][0]
    fact["reason_code"] = "invalid_observation_stream"
    with pytest.raises(ValidationError, match="first-invalid boundary"):
        owner.ReviewedPredictorFailure.model_validate_json(json.dumps(fact))
    fact["quarantine"] = "suffix_from_first_invalid"
    fact["first_invalid_session"] = "2023-10-13"
    with pytest.raises(ValidationError, match="first-invalid boundary"):
        owner.ReviewedPredictorFailure.model_validate_json(json.dumps(fact))


def _install_prefix_replay(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch, contract: Any) -> None:
    root = state["root"]
    for name in owner.PREFIX_IMPLEMENTATION_FILES:
        path = root / "src/market_predictor" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# test-only reader implementation", encoding="ascii")
    combined = root / "combined/combined_daily"
    combined.mkdir(parents=True)
    stock = pd.DataFrame([dict(ticker="WTW", timeframe="1d", price_feed="sip", adjustment="all", schema_version="1",
        bar_start_utc=pd.Timestamp(day + "T13:30Z"), bar_end_utc=pd.Timestamp(day + "T20:00Z"),
        available_at_utc=pd.Timestamp(day + "T20:15Z"), open=10.0, high=11.0, low=9.0, close=10.0,
        volume=100.0 if day == "2019-07-09" else 0.0) for day in ("2019-07-09", "2019-08-01")])
    inventory = {}
    for symbol in ("WTW", "SPY", "QQQ", "XLK"):
        path = combined / f"{symbol}.parquet"
        path.write_bytes(f"test-only bound {symbol} source".encode())
        inventory[symbol] = dict(ticker=symbol, path=path.name, sha256=file_sha256(path))
        state["request"]["adjusted_source_files"][path.relative_to(root).as_posix()] = file_sha256(path)
    state["source"] = combined / "WTW.parquet"
    gaps = combined / "gaps.json"
    write_json_object(gaps, {"gaps": []})
    state["request"]["adjusted_source_files"][gaps.relative_to(root).as_posix()] = file_sha256(gaps)
    manifest = combined / "_manifest.json"
    write_json_object(manifest, dict(session_gap_audit=dict(path=gaps.name, sha256=file_sha256(gaps))))
    locations = dict(outcome_source_config=state["config"], combined_manifest=manifest,
        parent_request=combined.parent / "_request.json")
    for name in ("strategy_contract", "parent_manifest", "parent_authority", "adjusted_plan_authority", "adjusted_archive_authority"):
        locations[name] = root / f"{name}.json"
    for path in locations.values():
        if not path.exists():
            path.write_text("{}", encoding="ascii")
    policy = {name: _pin(root, path).model_dump() for name, path in locations.items()}
    lines = ['schema_version = "market_predictor.corrected_research_features"']
    for name, pin in policy.items():
        lines.extend([f"[{name}]", f'path = "{pin["path"]}"', f'sha256 = "{pin["sha256"]}"'])
        state["request"]["declared_source_files"][pin["path"]] = pin["sha256"]
    state["feature_config"].write_text("\n".join(lines), encoding="ascii")
    state["facts"]["feature_config"] = _pin(root, state["feature_config"]).model_dump()
    state["request"]["config_sha256"] = file_sha256(state["feature_config"])
    state["request"]["declared_source_files"][state["feature_config"].name] = file_sha256(state["feature_config"])
    row = stock.iloc[-1]
    invalid = dict(session_date_et="2019-08-01", invalid_fields=["volume"],
        ohlcv={name: float(row[name]) for name in ("open", "high", "low", "close", "volume")},
        clocks={name: str(row[name]) for name in ("bar_start_utc", "bar_end_utc", "available_at_utc")})
    report = dict(schema="market_predictor.predictor_source_failure_observations.v1", review_approval=False,
        numeric_first="2018-05-29", numeric_last="2024-05-28", combined_manifest_sha256=file_sha256(manifest),
        observations=[dict(ticker="WTW", security_id="issuer-WTW", bounded_rows=2, invalid_rows=[invalid],
            source_path=state["source"].relative_to(root).as_posix(), source_sha256=file_sha256(state["source"]))])
    state["evidence"].write_text(json.dumps(report), encoding="ascii")
    state["facts"]["observations"] = _pin(root, state["evidence"]).model_dump()
    state["facts"]["failures"][0].update(reason_code="invalid_observation_stream", quarantine="suffix_from_first_invalid",
        first_invalid_session="2019-08-01", boundary_observation_sha256=json_sha256(invalid),
        source_artifacts=[_pin(root, state["source"]).model_dump()], reviewed_evidence=[_pin(root, state["evidence"]).model_dump()])
    old_part = state["checkpoint"]["groups"][state["key"]]
    good = pd.read_parquet(state["parent"] / "groups" / old_part["path"])
    _refreeze(state)
    part = owner._publish(good, state["parent"] / "groups/rebound.parquet", state["checkpoint"]["request_sha256"])
    state["checkpoint"]["groups"][state["key"]] = {**part, "availability_columns": old_part["availability_columns"]}
    _refreeze(state)
    state.update(replay_active=False, replay_exited=False, replay_error=False, replay_calls=[], replay_before_exit=None)
    memberships = pd.DataFrame([dict(ticker="WTW", security_id="issuer-WTW", primary_benchmark="XLK",
        effective_from_utc=pd.Timestamp("2018-01-01T00:00Z"))])

    @contextmanager
    def source_context(*args: Any, **kwargs: Any) -> Any:
        assert state["exited"] and not state["replay_active"]
        with heavy_job_lease("synthetic canonical source replay", runtime_dir=root / "data/runtime"):
            state["replay_active"] = True
            try:
                yield dict(source_files={}, memberships=memberships, membership_path=root / "membership.parquet", selection={})
                if state["replay_before_exit"]:
                    state["replay_before_exit"]()
                if state["replay_error"]:
                    raise DataReadinessError("replay source exit failed")
                state["replay_exited"] = True
            finally:
                state["replay_active"] = False

    def read(directory: Path, record: dict[str, Any], **kwargs: Any) -> pd.DataFrame:
        assert state["replay_active"] and state["exited"]
        state["replay_calls"].append(("read", record["ticker"]))
        bars = stock.copy()
        bars["ticker"] = record["ticker"]
        if record["ticker"] != "WTW":
            bars["volume"] = 100.0
        return bars

    def raw(**kwargs: Any) -> pd.DataFrame:
        assert state["replay_active"] and kwargs["decisions"].session_date_et.lt(date(2019, 8, 1)).all()
        state["replay_calls"].append(("raw", len(kwargs["decisions"])))
        return pd.DataFrame()

    def build(group: pd.DataFrame, stock: pd.DataFrame, benchmarks: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        assert state["replay_active"] and group.session_date_et.lt(date(2019, 8, 1)).all()
        assert len(stock) == 1 and stock.volume.gt(0).all()
        assert kwargs["expected_history_sessions"] == (date(2019, 7, 9),)
        state["replay_calls"].append(("build", len(group)))
        rows = group.drop(columns=["source_group", "parent_ticker"]).copy()
        rows[list(TECHNICAL_RANKING_FEATURES)] = 0.25
        rows["feature_eligible"] = True
        rows["feature_profile"] = "technical_market"
        rows["daily_bar_count"] = 300
        rows["technical_available_at_utc"] = rows.decision_time_utc - pd.Timedelta(hours=1)
        rows["raw_dollar_volume_available_at_utc"] = rows.technical_available_at_utc
        rows["technical_missing_reasons"] = [()] * len(rows)
        return owner.AdjustedTechnicalSource(rows, old_part["availability_columns"])

    monkeypatch.setattr(owner, "load_corrected_outcome_policy", lambda *a: SimpleNamespace(
        action_config=SourcePin(path="actions.json", sha256="a" * 64), action_archive="actions", action_audit_sha256="a" * 64))
    monkeypatch.setattr(owner, "load_corporate_action_evidence", lambda **k: None)
    monkeypatch.setattr(owner, "verified_corrected_research_sources", source_context)
    monkeypatch.setattr(owner, "load_strategy_contract", lambda *a: contract)
    monkeypatch.setattr(owner, "load_combined_adjusted_inventory", lambda *a, **k: inventory)
    monkeypatch.setattr(owner, "_projection", lambda *a: memberships.copy())
    monkeypatch.setattr(owner, "expected_adjusted_history_sessions", lambda **k: (date(2019, 7, 9), date(2019, 8, 1)))
    monkeypatch.setattr(owner, "read_combined_adjusted_bars", read)
    monkeypatch.setattr(owner, "raw_dollar_volume_inputs", raw)
    monkeypatch.setattr(owner, "build_adjusted_technical_source", build)


def test_prefix_source_replay_and_publication_use_sequential_leases(example: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch, contract: Any,
) -> None:
    _install_prefix_replay(example, monkeypatch, contract)
    result = _run(example)
    assert example["exited"] and example["replay_exited"] and not example["replay_active"]
    assert result["rows"] == 4 and result["unavailable_rows"] == 1 and result["rebuilt_prefix_rows"] == 1
    assert result["feature_eligible_rows"] == 3 and len(result["recovered_groups"]) == 1
    assert result["source_failures"] == example["checkpoint"]["failed_groups"] and result["failed_groups"] == {}
    assert [call for call in example["replay_calls"] if call[0] == "build"] == [("build", 1)]
    assert not any(call == ("read", "AAA") for call in example["replay_calls"])
    part = result["groups"][example["key"]]
    assert (example["output"] / "groups" / part["path"]).read_bytes() == (example["parent"] / "groups" / part["path"]).read_bytes()
    request = json.loads((example["output"] / "_request.json").read_text())
    owner.validate_predictor_derivation(root=example["root"], manifest=result, request=request)
    missing_reader = json.loads(json.dumps(request))
    reader = "src/market_predictor/" + owner.PREFIX_IMPLEMENTATION_FILES[0]
    for name in ("source_files", "implementation_files"):
        missing_reader[name].pop(reader)
    with pytest.raises(DataReadinessError, match="omits inherited or approval source pins"):
        owner.validate_predictor_derivation(root=example["root"], manifest=result, request=missing_reader)
    result["groups"][example["fail_key"]]["completion_scope"]["first_invalid_session"] = "2019-07-09"
    with pytest.raises(DataReadinessError, match="prefix/suffix"):
        owner.validate_predictor_derivation(root=example["root"], manifest=result, request=request)


@pytest.mark.parametrize("failure", ["exit", "source_tamper"])
def test_prefix_source_exit_failure_cannot_publish(example: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
    contract: Any, failure: str,
) -> None:
    _install_prefix_replay(example, monkeypatch, contract)
    if failure == "exit":
        example["replay_error"] = True
    else:
        example["replay_before_exit"] = lambda: example["source"].write_bytes(b"tampered during source exit")
    with pytest.raises(DataReadinessError, match="exit failed|source changed"):
        _run(example)
    assert not example["output"].exists() and not list(example["output"].parent.glob(".*.pending"))


def test_reviewed_config_preserves_exact_four_failure_scopes() -> None:
    path = Path(__file__).resolve().parents[1] / "configs/swing_predictor_failure_facts.json"
    facts = owner.PredictorFailureFacts.model_validate_json(path.read_text(encoding="utf-8"))
    assert sum(f.rows for f in facts.failures) == 3279
    assert {fact.symbol: fact.first_invalid_session for fact in facts.failures} == {
        "WTW": None, "ATVI": date(2023, 10, 13), "INFO": date(2022, 2, 28), "SBNY": date(2023, 3, 13)}
    assert {fact.symbol for fact in facts.failures if fact.quarantine == "entire_failed_group"} == {"WTW"}
    assert facts.observations.sha256 == "70c6f85029e64ba3354ba04b5aa936c2147272f2f527ca9757517b3d31178060"
