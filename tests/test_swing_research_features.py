from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import market_predictor.swing.datasets.research_features as owner
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.swing.datasets.adjusted_history_bindings import AdjustedHistoryBindings
from market_predictor.swing.datasets.adjusted_history_source import AdjustedHistorySource, AdjustedHistoryUnit
from market_predictor.swing.features.adjusted_source import AdjustedTechnicalSource
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from tests.test_swing_features import contract as contract


@pytest.fixture
def publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contract: Any) -> dict[str, Any]:
    root = tmp_path
    package = root / "src/market_predictor"
    paths = ("swing/datasets/research_features.py", "swing/datasets/research_feature_sources.py",
        "swing/features/adjusted_source.py", "swing/features/predictors.py", "swing/dataset.py",
        "swing/features/pipeline.py", "swing/features/technical_relationships.py", "canonical/normalize.py",
        "canonical/joins.py", "canonical/cutoffs.py", "swing/contracts/research_features.py", "swing/features/panel.py",
        "swing/features/research_join.py", "swing/features/eligibility.py", "swing/datasets/corrected_outcomes.py",
        "swing/datasets/symbol_corrections.py", "swing/labels/holding_identity.py", "modeling/strategy_contract.py",
        "swing/datasets/session_requirements.py", "swing/datasets/history_archive.py",
        "evidence/io.py", "universe/symbol_correction_policy.py",
        "swing/datasets/initial_fit_raw_share_plan.py", "swing/datasets/adjusted_history_bindings.py",
        "swing/datasets/adjusted_history_source.py")
    for relative in paths:
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic implementation pin\n", encoding="ascii")
    monkeypatch.setattr(owner, "__file__", str(package / paths[0]))
    archive = root / "data/raw/adjusted"
    archive.mkdir(parents=True)
    locations = {name: root / "data/research" / name / "source.json" for name in
        ("outcome_source_config", "strategy_contract", "adjusted_plan_authority", "adjusted_archive_authority")}
    for path in locations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="ascii")
    config = root / "configs/features.toml"
    config.parent.mkdir()
    lines = ['schema_version = "market_predictor.corrected_research_features"']
    for name, path in locations.items():
        lines.extend([f"[{name}]", f'path = "{path.relative_to(root).as_posix()}"', f'sha256 = "{file_sha256(path)}"'])
    config.write_text("\n".join(lines), encoding="ascii")
    records = [dict(first_session=day, last_session=day, partition_month=day[:7]) for day in ("2019-07-09", "2019-08-01")]
    decisions = pd.DataFrame([dict(decision_id=f"{symbol}-{day}", parent_decision_id=f"old-{symbol}-{day}",
        security_id=f"issuer-{symbol}", ticker=symbol, parent_ticker=symbol, session_date_et=date.fromisoformat(day),
        decision_time_utc=pd.Timestamp(f"{day}T22:00:00Z"), sector="Technology", primary_benchmark="XLK")
        for day in ("2019-07-09", "2019-08-01") for symbol in ("AAA", "BBB")])
    memberships = pd.DataFrame([dict(ticker=symbol, security_id=f"issuer-{symbol}", sector="Technology",
        primary_benchmark="XLK", effective_from_utc=pd.Timestamp("2019-07-09T04:00:00Z"), effective_to_utc=pd.NaT)
        for symbol in ("AAA", "BBB")])
    original_memberships = memberships.copy()
    for column in owner.MEMBERSHIP_COLUMNS:
        if column not in original_memberships:
            original_memberships[column] = "synthetic"
    membership_path = root / "data/research/memberships.parquet"
    original_memberships.to_parquet(membership_path)
    sources = dict(manifest=dict(files=records), plan=dict(requirements=dict(in_window_decisions=len(decisions))),
        request=dict(cohort_sha256="c" * 64, retained_security_ids=sorted(set(decisions.security_id))),
        memberships=memberships, membership_path=membership_path,
        source_files={membership_path.relative_to(root).as_posix(): file_sha256(membership_path)}, selection={})
    state: dict[str, Any] = dict(root=root, config=config, output=root / "data/features/new",
        expected_config_sha256=file_sha256(config), calls=[], active=False, failures={}, decisions=decisions,
        sources=sources, captured_benchmarks={}, captured_sessions={})
    inventory, windows, adjusted_files = {}, {}, {}
    for symbol in ("AAA", "BBB", "SPY", "QQQ", "XLK", "XLF"):
        path = archive / f"{symbol}.parquet"
        path.write_bytes(f"synthetic bound bar bytes {symbol}".encode())
        stock = symbol in {"AAA", "BBB"}
        unit_id = f"unit-{symbol}"
        identity = dict(security_id=f"issuer-{symbol}" if stock else f"benchmark:{symbol}", ticker=symbol,
            role="stock" if stock else "benchmark", start_date="2018-05-29", end_date="2024-05-28")
        inventory[unit_id] = dict(**identity, unit_id=unit_id, bars_path=path.name, bars_sha256=file_sha256(path))
        windows[unit_id] = dict(parent={**identity, "start_date": "2019-07-09"}, query=identity)
        adjusted_files[path.relative_to(root).as_posix()] = file_sha256(path)
    adjusted = AdjustedHistorySource(root, archive, inventory, adjusted_files)
    bindings = AdjustedHistoryBindings(adjusted, windows, adjusted_files)
    state.update(inventory=inventory, archive=archive, bindings=bindings)

    @contextmanager
    def verified(*args: Any, **kwargs: Any) -> Any:
        assert not state["active"]
        state["active"] = True
        try:
            yield sources
            if state.get("exit_failure"):
                raise DataReadinessError("source context exit hash check failed")
        finally:
            state["active"] = False

    def read_bars(source: AdjustedHistorySource, unit_id: str) -> AdjustedHistoryUnit:
        assert state["active"], "numeric source read escaped the source lease"
        record = source.records[unit_id]
        if file_sha256(source.directory / record["bars_path"]) != record["bars_sha256"]:
            raise DataReadinessError("synthetic adjusted source changed")
        state["calls"].append(("read", record["ticker"]))
        bars = pd.DataFrame(dict(ticker=[record["ticker"]], bar_start_utc=[pd.Timestamp("2019-07-09T13:30Z")]))
        return AdjustedHistoryUnit(bars, (date(2019, 7, 10),), (), record)

    def build(group: pd.DataFrame, bars: pd.DataFrame, benchmarks: pd.DataFrame, *args: Any, **kwargs: Any) -> Any:
        identity = kwargs["security_id"]
        state["calls"].append(("build", identity))
        if identity in state["failures"]:
            raise state["failures"][identity]
        state["captured_benchmarks"][identity] = set(benchmarks.ticker)
        state["captured_sessions"][identity] = kwargs["expected_history_sessions"]
        frame = group.loc[:, [*owner.DECISION_KEYS, "sector", "session_date_et", "primary_benchmark"]].copy()
        frame["feature_eligible"] = True
        frame["feature_profile"] = "technical_market"
        frame["daily_bar_count"] = 300
        frame[list(TECHNICAL_RANKING_FEATURES)] = 0.25
        frame["technical_available_at_utc"] = frame.decision_time_utc - pd.Timedelta(hours=1)
        frame["technical_missing_reasons"] = [()] * len(frame)
        return AdjustedTechnicalSource(frame, {name: "technical_available_at_utc" for name in TECHNICAL_RANKING_FEATURES})

    monkeypatch.setattr(owner, "_guard", lambda: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    monkeypatch.setattr(owner, "load_strategy_contract", lambda *a, **k: contract)
    monkeypatch.setattr(owner, "load_corrected_outcome_policy", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(owner, "verified_corrected_price_sources", verified)
    monkeypatch.setattr(owner, "load_adjusted_history_bindings", lambda **k: bindings)
    monkeypatch.setattr(owner, "load_corrected_decision_partition", lambda root, sources, record, policy:
        decisions.loc[decisions.session_date_et.map(lambda day: day.strftime("%Y-%m")).eq(record["partition_month"])].copy())
    monkeypatch.setattr("market_predictor.swing.datasets.adjusted_history_bindings.read_adjusted_history_unit", read_bars)
    monkeypatch.setattr(owner, "raw_dollar_volume_inputs", lambda **k: pd.DataFrame())
    monkeypatch.setattr(owner, "build_adjusted_technical_source", build)
    return state


def _run(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return owner.materialize_research_predictors(**{name: state[name]
        for name in ("root", "config", "output", "expected_config_sha256")}, **kwargs)


def _resume(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return _run(state, expected_checkpoint_sha256=file_sha256(state["output"] / "_checkpoint.json"), **kwargs)


def test_stubbed_publication_preserves_months_population_and_lineage(publication: dict[str, Any]) -> None:
    result = _run(publication)
    assert result["status"] == "technical_inputs_complete_research_only"
    assert result["rows"] == 4 and len(result["groups"]) == 2 and len(result["months"]) == 2
    assert result["training_eligible"] is False and result["promotion_eligible"] is False
    assert result["exclusions_added"] == [] and not publication["active"]
    for record in result["months"].values():
        frame = pd.read_parquet(publication["output"] / "months" / record["path"])
        assert len(frame) == 2 and not any(name.startswith("future_") for name in frame)
        assert frame.parent_decision_id.str.startswith("old-").all()
    before = (publication["output"] / "_manifest.json").read_bytes()
    publication["calls"].clear()
    resumed = _resume(publication)
    assert resumed["manifest_sha256"] == result["manifest_sha256"]
    assert (publication["output"] / "_manifest.json").read_bytes() == before
    assert not any(kind == "build" for kind, _ in publication["calls"])


def test_bounded_resume_skips_completed_group(publication: dict[str, Any]) -> None:
    partial = _run(publication, maximum_groups_this_run=1)
    assert partial["status"] == "partial_in_progress" and len(partial["groups"]) == 1
    publication["calls"].clear()
    assert _resume(publication)["rows"] == 4
    assert [(kind, name) for kind, name in publication["calls"] if kind == "build"] == [("build", "issuer-BBB")]


def test_source_failure_isolated_and_retried_without_dropping_population(publication: dict[str, Any]) -> None:
    publication["failures"]["issuer-AAA"] = DataReadinessError("missing source")
    partial = _run(publication)
    assert partial["status"] == "partial_source_failures" and len(partial["groups"]) == 1
    assert len(partial["failed_groups"]) == 1 and not (publication["output"] / "_manifest.json").exists()
    publication["failures"].clear()
    assert _resume(publication)["rows"] == 4


def test_memory_failure_stops_instead_of_isolating(publication: dict[str, Any]) -> None:
    publication["failures"]["issuer-AAA"] = MemoryBudgetError("pressure")
    with pytest.raises(MemoryBudgetError, match="pressure"):
        _run(publication)
    assert not any(call == ("build", "issuer-BBB") for call in publication["calls"])
    assert not publication["active"]


@pytest.mark.parametrize("container", ["groups", "months", "failed_groups"])
def test_foreign_resume_population_rejected(publication: dict[str, Any], container: str) -> None:
    _run(publication, maximum_groups_this_run=1)
    path = publication["output"] / "_checkpoint.json"
    checkpoint = json.loads(path.read_text())
    checkpoint[container]["foreign"] = {}
    path.write_text(json.dumps(checkpoint), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="foreign population"):
        _resume(publication)


def test_config_pin_tamper_rejected(publication: dict[str, Any]) -> None:
    publication["config"].write_text("# tampered", encoding="ascii")
    with pytest.raises(DataReadinessError, match="independent hash"):
        _run(publication)
    assert publication["calls"] == []


def test_group_bytes_tamper_rejected_on_resume(publication: dict[str, Any]) -> None:
    partial = _run(publication, maximum_groups_this_run=1)
    record = next(iter(partial["groups"].values()))
    (publication["output"] / "groups" / record["path"]).write_bytes(b"tampered")
    with pytest.raises(DataReadinessError, match="partition changed"):
        _resume(publication)


def test_interrupted_after_artifact_before_checkpoint_reconstructs(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner._write

    def interrupt(path: Path, value: dict[str, Any]) -> None:
        if path.name == "_checkpoint.json" and value["groups"]:
            raise RuntimeError("interrupted")
        original(path, value)

    monkeypatch.setattr(owner, "_write", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        _run(publication)
    monkeypatch.setattr(owner, "_write", original)
    assert _resume(publication)["rows"] == 4


def test_reconstructed_orphan_different_values_rejected(publication: dict[str, Any]) -> None:
    frame = publication["decisions"].copy().assign(feature_eligible=True)
    path = publication["output"] / "orphan.parquet"
    owner._publish(frame, path, "a" * 64)
    frame["feature_eligible"] = False
    with pytest.raises(DataReadinessError, match="differs from reconstruction"):
        owner._publish(frame, path, "a" * 64)


def test_completed_group_source_tamper_rejected_on_resume(publication: dict[str, Any]) -> None:
    _run(publication, maximum_groups_this_run=1)
    (publication["archive"] / "AAA.parquet").write_bytes(b"source changed after checkpoint")
    with pytest.raises(DataReadinessError, match="source changed"):
        _resume(publication)


def test_historical_sector_benchmark_reaches_source_builder(publication: dict[str, Any]) -> None:
    memberships = publication["sources"]["memberships"]
    old = memberships.iloc[:1].copy()
    old["effective_from_utc"] = pd.Timestamp("2018-05-29T04:00:00Z")
    old["effective_to_utc"] = pd.Timestamp("2019-07-09T04:00:00Z")
    old["primary_benchmark"] = "XLF"
    publication["sources"]["memberships"] = pd.concat([old, memberships], ignore_index=True)
    assert _run(publication)["rows"] == 4
    assert "XLF" in publication["captured_benchmarks"]["issuer-AAA"]


def test_context_mismatch_refused() -> None:
    frame = pd.DataFrame([dict(decision_id="a", security_id="issuer", ticker="A", decision_time_utc="2019-07-09T22:00:00Z",
        sector="Technology", session_date_et=date(2019, 7, 9), primary_benchmark="XLK")])
    wrong = frame.copy().assign(primary_benchmark="XLF")
    with pytest.raises(DataReadinessError, match="frozen primary_benchmark"):
        owner._check_population(frame, wrong)


def test_original_ipo_requirements_retain_absent_session_for_warmup(publication: dict[str, Any]) -> None:
    _run(publication)
    sessions = publication["captured_sessions"]["issuer-AAA"]
    assert min(sessions) == date(2019, 7, 9)
    assert date(2019, 7, 10) in sessions


def test_price_context_requires_no_action_or_target_source(publication: dict[str, Any]) -> None:
    # The policy stub deliberately has no action_config/archive/audit fields.
    assert _run(publication)["rows"] == 4


def test_unavailable_rows_keep_frozen_months_and_ids(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner.build_adjusted_technical_source

    def unavailable(*args: Any, **kwargs: Any) -> AdjustedTechnicalSource:
        result = original(*args, **kwargs)
        result.rows["feature_eligible"] = False
        result.rows["technical_missing_reasons"] = [("stock_history_missing",)] * len(result.rows)
        return result

    monkeypatch.setattr(owner, "build_adjusted_technical_source", unavailable)
    result = _run(publication)
    assert result["rows"] == 4 and result["feature_eligible_rows"] == 0
    assert set(result["months"]) == {"2019-07", "2019-08"}
    ids = set()
    for part in result["months"].values():
        ids.update(pd.read_parquet(publication["output"] / "months" / part["path"]).decision_id)
    assert ids == set(publication["decisions"].decision_id)


@pytest.mark.parametrize("maximum_groups", [None, 1])
def test_source_exit_failure_never_publishes_complete_manifest(publication: dict[str, Any], maximum_groups: int | None) -> None:
    publication["exit_failure"] = True
    with pytest.raises(DataReadinessError, match="source context exit"):
        _run(publication, maximum_groups_this_run=maximum_groups)
    assert (publication["output"] / "_checkpoint.json").is_file()
    assert not (publication["output"] / "_manifest.json").exists()


def test_mutation_between_source_exit_and_final_lease_refused(publication: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def mutate(*args: Any, **kwargs: Any) -> Any:
        assert not publication["active"]
        (publication["archive"] / "AAA.parquet").write_bytes(b"changed after source lease")
        yield {}

    monkeypatch.setattr(owner, "heavy_job_lease", mutate)
    with pytest.raises(DataReadinessError, match="source changed"):
        _run(publication)
    assert not (publication["output"] / "_manifest.json").exists()


def test_failed_group_records_query_ticker_and_unit_identity(publication: dict[str, Any]) -> None:
    publication["failures"]["issuer-AAA"] = DataReadinessError("synthetic unavailable input")
    result = _run(publication)
    failure = next(iter(result["failed_groups"].values()))
    assert failure["source_group"] == "unit-AAA" and failure["symbol"] == "AAA"
