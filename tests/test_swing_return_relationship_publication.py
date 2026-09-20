"""Synthetic bounded authorities; no provider calls or production evidence."""
from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import history_archive
from market_predictor.swing.datasets import return_relationship_integrity as integrity
from market_predictor.swing.datasets import return_relationship_publication as owner
from market_predictor.swing.datasets.return_relationship_parent import _historical_sources
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.training.return_validation import security_transfer_mask
from tests.test_fixed_horizon_readiness import inputs
from tests.test_swing_feature_history_plan import _publish
from tests.test_swing_feature_history_plan import evidence as evidence
from tests.test_swing_history_collection import _FakeSource, _test_transport_response
from tests.test_swing_return_feature_profiles import _bars
from tests.test_swing_training_readiness import REPO, _json


def _pin(root: Path, path: Path) -> dict[str, str]:
    return dict(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))


class _FullSource(_FakeSource):
    def fetch_daily_page(self, symbol: str, start: datetime, end_exclusive: datetime, *,
        page_token: str | None, asof: date, adjustment: str,
    ) -> history_archive.SwingDailyPage:
        page = super().fetch_daily_page(symbol, start, end_exclusive, page_token=page_token, asof=asof, adjustment=adjustment)
        days = xcals.get_calendar("XNYS").sessions_in_range(start.date(), asof)
        bars = tuple(dict(t=pd.Timestamp(day.date(), tz="America/New_York").tz_convert("UTC").isoformat(),
            o=100.0 + index / 10, h=102.0 + index / 10, l=99.0 + index / 10,
            c=101.0 + index / 10, v=1000 + index) for index, day in enumerate(days))
        payload = {"bars": {symbol: list(bars)}, "next_page_token": None}
        params = dict(symbols=symbol, timeframe="1Day", start=start.isoformat(),
            end=(end_exclusive - timedelta(microseconds=1)).isoformat(), feed="sip", limit=10000,
            adjustment=adjustment, sort="asc", asof=asof.isoformat())
        return replace(page, bars=bars, raw_payload=payload, transport_response=_test_transport_response(payload, params))


def _canonical(frame: pd.DataFrame, path: Path, kind: str, request: str) -> None:
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="synthetic_fixture", status="pass", failures=0,
        rows_checked=len(frame), detail="Synthetic unit-test data only."),))
    write_canonical_artifact(frame, path, artifact_type=kind, audit=audit,
        inputs={"request_sha256": request}, production_ready=False)


def _feature_config(root: Path, values: dict[str, Any]) -> Path:
    path = root / "configs/features.toml"
    lines = ['schema_version = "market_predictor.corrected_research_features"']
    for name, value in values.items():
        lines += [f"[{name}]", f'path = "{value["path"]}"', f'sha256 = "{value["sha256"]}"']
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _combined(root: Path, strategy: Any, sessions: tuple[date, ...]) -> dict[str, Any]:
    directory = root / "data/features/source"
    combined = directory / "combined_daily"
    combined.mkdir(parents=True)
    lineage: dict[str, Any] = {"membership_authority": {"universe_sha256": "7" * 64}}
    for family in ("pre_collection", "post_collection"):
        path = root / "data/raw" / family
        _json(path / "_request.json", dict(adjustment="all", price_feed="sip"))
        _json(path / "_manifest.json", dict(schema="synthetic_test_collection"))
        lineage[family] = {"directory": str(path), "manifest_sha256": file_sha256(path / "_manifest.json")}
    combined_inputs = {**lineage, "source": "alpaca", "timeframe": "1Day", "adjustment": "all", "price_feed": "sip",
        "start_date": "2018-05-29", "pre_end_date": "2019-07-08", "post_start_date": "2019-07-09"}
    request = dict(request_sha256="a" * 64, strategy_contract_sha256=strategy.sha256(), combined_daily_inputs=combined_inputs)
    _json(directory / "_request.json", request)
    combined_sha = json_sha256({**combined_inputs, "parent_materialization_request_sha256": request["request_sha256"]})
    artifacts = []
    for ticker in ("AAA", "BBB", "SPY", "WTW"):
        frame = _bars(sessions, "AAA" if ticker != "SPY" else "SPY").drop(columns="security_id", errors="ignore")
        frame["ticker"] = ticker
        frame["high"] = frame.close * 1.01
        frame["low"] = frame.open * 0.99
        frame["schema_version"] = "market_data.v1"
        path = combined / f"{ticker}.parquet"
        _canonical(frame, path, "bars", combined_sha)
        artifacts.append(dict(ticker=ticker, path=path.name, sha256=file_sha256(path), rows=len(frame),
            canonical_manifest_sha256=file_sha256(manifest_path_for(path))))
    manifest = dict(request_sha256=combined_sha, source_lineage=lineage, artifacts=artifacts)
    combined_pin = _json(combined / "_manifest.json", manifest)
    owner_pin = _json(combined / "_authority.json", dict(state="complete", artifact_sha256=combined_pin, request_sha256=combined_sha))
    final_pin = _json(directory / "final/_manifest.json", dict(request_sha256=request["request_sha256"],
        strategy_contract_sha256=strategy.sha256(), source=dict(combined_daily_authority_sha256=owner_pin)))
    _json(directory / "final/_authority.json", dict(state="complete", artifact_sha256=final_pin,
        request_sha256=request["request_sha256"], strategy_contract_sha256=strategy.sha256()))
    return dict(directory=directory, artifacts=artifacts)


def _frames(strategy: Any) -> tuple[dict[str, pd.DataFrame], tuple[date, ...], list[str]]:
    calendar = xcals.get_calendar("XNYS")
    months = pd.period_range("2019-07", "2024-05", freq="M")
    candidates = [f"synthetic-issuer-{index}" for index in range(50)]
    mask = security_transfer_mask(pd.Series(candidates), fraction=0.2)
    identities = [candidates[int(np.flatnonzero(mask)[0])], candidates[int(np.flatnonzero(~mask)[0])]]
    frames = {}
    sessions: list[date] = []
    for month in months:
        days = tuple(stamp.date() for stamp in calendar.sessions_in_range(
            max(str(month) + "-01", "2019-07-09"), min(str(month.end_time.date()), "2024-05-28")))
        days = (days[0], days[1], days[-1])
        sessions.extend(days)
        parts = []
        for identity, ticker in zip(identities, ("AAA", "BBB"), strict=True):
            frame, _ = inputs()
            frame = frame.drop(columns=["feature", "feature_clock"])
            frame["decision_id"] = [f"{identity}-{day}" for day in days]
            frame["security_id"], frame["ticker"], frame["parent_ticker"] = identity, ticker, ticker
            frame["session_date_et"] = list(days)
            frame["decision_time_utc"] = swing_prediction_cutoffs(frame.session_date_et)
            frame["research_label_mature_at"] = pd.to_datetime([
                calendar.session_close(calendar.sessions_window(pd.Timestamp(day), 11)[-1]) for day in days], utc=True)
            terminal = frame.research_label_mature_at.gt(calendar.session_close("2024-05-28"))
            frame.loc[terminal, ["stock_source_admitted", "fixed_comparisons_complete"]] = False
            frame.loc[terminal, "research_label_mature_at"] = pd.NaT
            frame.loc[terminal, "stock_component_id"] = None
            frame.loc[terminal, "stock_missing_reasons"] = '["initial_fit_terminal_immature"]'
            frame.loc[terminal, ["fixed_horizon_gross_return", "fixed_horizon_net_return"]] = np.nan
            for role in ("spy", "qqq", "sector"):
                frame.loc[terminal, [f"{role}_horizon_gross_return", f"{role}_fixed_horizon_excess_return"]] = np.nan
                frame.loc[terminal, f"{role}_component_id"] = None
                frame.loc[terminal, f"{role}_missing_reasons"] = '["initial_fit_terminal_immature"]'
            frame["feature_profile"] = "technical_market"
            parts.append(frame)
        frames[str(month)] = pd.concat(parts, ignore_index=True)
    return frames, tuple(sessions), identities


@pytest.fixture
def publication_fixture(evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = evidence["root"]
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(root / "isolated-runtime"))
    shutil.copytree(REPO / "src/market_predictor", root / "src/market_predictor", ignore=shutil.ignore_patterns("__pycache__"))
    monkeypatch.setattr(integrity, "__file__", str(root / "src/market_predictor/swing/datasets/return_relationship_integrity.py"))
    monkeypatch.setattr(owner, "guard", lambda limit: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    monkeypatch.setattr(history_archive, "_guard", lambda stage: None)
    monkeypatch.setattr(history_archive, "_guard_peak", lambda stage: None)
    config_dir = root / "configs"
    config_dir.mkdir(exist_ok=True)
    readiness_policy: dict[str, Any] = dict(schema_version="market_predictor.swing_training_readiness",
        scope="initial_fit_fixed_horizon_diagnostics", maximum_system_used_percent=90.0)
    for field, name in (("strategy_contract", "edge_rebuild_strategy_contract.toml"),
            ("research_contract", "swing_research.toml"), ("temporal_contract", "edge_rebuild_temporal_manifest.toml")):
        shutil.copyfile(REPO / "configs" / name, config_dir / name)
        readiness_policy[field] = _pin(root, config_dir / name)
    shutil.copyfile(REPO / "configs/swing_corrected_outcomes.toml", config_dir / "swing_corrected_outcomes.toml")
    strategy = load_strategy_contract(root / readiness_policy["strategy_contract"]["path"])
    all_sessions = tuple(day.date() for day in xcals.get_calendar("XNYS").sessions_in_range("2018-05-29", "2024-05-28"))
    combined = _combined(root, strategy, all_sessions)
    plan, plan_pin = _publish(evidence)
    archive = root / "adjusted-archive"
    history_archive.collect_swing_history_plan(plan_directory=plan, output_directory=archive,
        source_factory=_FullSource, provider_symbol_for=lambda ticker: ticker, expected_plan_authority_sha256=plan_pin)
    directory = combined["directory"]
    feature = _feature_config(root, dict(outcome_source_config=_pin(root, config_dir / "swing_corrected_outcomes.toml"),
        strategy_contract=readiness_policy["strategy_contract"], parent_request=_pin(root, directory / "_request.json"),
        parent_manifest=_pin(root, directory / "final/_manifest.json"), parent_authority=_pin(root, directory / "final/_authority.json"),
        combined_manifest=_pin(root, directory / "combined_daily/_manifest.json"),
        adjusted_plan_authority=_pin(root, plan / "_authority.json"), adjusted_archive_authority=_pin(root, archive / "_authority.json")))
    observation = root / "data/reports/failure_observation.json"
    wtw = next(item for item in combined["artifacts"] if item["ticker"] == "WTW")
    wtw_pin = _pin(root, directory / "combined_daily" / wtw["path"])
    _json(observation, dict(schema="market_predictor.predictor_source_failure_observations.v1",
        numeric_first="2018-05-29", numeric_last="2024-05-28", observations=[dict(security_id="unavailable-wtw", ticker="WTW",
            source_path=wtw_pin["path"], source_sha256=wtw_pin["sha256"], invalid_rows=[])]))
    facts_path = config_dir / "failures.json"
    _json(facts_path, dict(schema_version="market_predictor.predictor_failure_facts.v1", parent_checkpoint_sha256="1" * 64,
        parent_request_sha256="2" * 64, decision_config=_pin(root, config_dir / "swing_corrected_outcomes.toml"),
        feature_config=_pin(root, feature), observations=_pin(root, observation), parent_run_finished=True,
        approval_scope="causal_prefix_replay_and_nullable_completion_only", reviewed_by="Synthetic unit test",
        failures=[dict(group_key=json_sha256(["unavailable-wtw", "WTW"]), security_id="unavailable-wtw", symbol="WTW", rows=1,
            parent_failure_sha256="3" * 64, reason_code="unverified_issuer_history", quarantine="entire_failed_group",
            first_invalid_session=None, boundary_observation_sha256=None, source_artifacts=[wtw_pin],
            reviewed_evidence=[_pin(root, observation)], detail="Synthetic absent issuer, never actual market evidence")]))
    frames, sessions, identities = _frames(strategy)
    population = pd.concat(frames.values(), ignore_index=True)
    groups = {json_sha256([identity, ticker]): dict(rows=int(population.security_id.eq(identity).sum()),
        decision_ids_sha256=json_sha256(sorted(population.loc[population.security_id.eq(identity), "decision_id"])))
        for identity, ticker in zip(identities, ("AAA", "BBB"), strict=True)}
    _json(root / "data/features/predictors/_manifest.json", dict(schema="market_predictor.research_predictors", rows=len(population),
        groups=groups, derivation=dict(approved_failure_facts=_pin(root, facts_path))))
    source_files = {path.relative_to(root).as_posix(): file_sha256(path) for path in root.rglob("*")
        if path.is_file() and not path.is_relative_to(root / "src") and not path.name.endswith(".lock")}
    joined = root / "data/features/join"
    request_pin = _json(joined / "_request.json", dict(schema="market_predictor.research_join_request", rows=len(population),
        historical_first_seen_proven=False, cohort_sha256="c" * 64, source_files=source_files, decision_source_files={}))
    months = {}
    for month, base in frames.items():
        profiles = {}
        for profile in ("technical_market", "catalyst_full"):
            names = tuple(swing_model_feature_columns(contract=strategy, catalyst=profile == "catalyst_full"))
            clocks = {name: f"available_at_{name}" for name in names}
            values = pd.DataFrame({**{name: np.full(len(base), 0.125, dtype=np.float32) for name in names},
                **{clock: base.decision_time_utc for clock in clocks.values()}})
            frame = pd.concat([base, values], axis=1)
            path = joined / month / f"{profile}.parquet"
            _canonical(frame, path, "swing_research_join", request_pin)
            profiles[profile] = dict(path=f"{month}/{profile}.parquet", sha256=file_sha256(path),
                manifest_sha256=file_sha256(manifest_path_for(path)), model_columns=list(names), availability_columns=clocks,
                audit=dict(training_eligible=False, promotion_eligible=False))
        months[month] = dict(rows=len(base), decision_ids_sha256=json_sha256(sorted(base.decision_id)), profiles=profiles)
    parent_pin = _json(joined / "_manifest.json", dict(schema="market_predictor.research_join", status="complete_research_only",
        rows=len(population), request_sha256=request_pin, months=months, exclusions_added=[],
        training_eligible=False, promotion_eligible=False))
    receipt_path = root / "data/reports/prior.json"
    _json(receipt_path, dict(manifest_sha256=parent_pin, status="passed", scope="published_join_population_clocks_original_targets",
        unique_decisions=len(population), months=59, matched_profile_population=True, original_outcome_values_exact=True,
        outcome_filtered_rows=0, training_eligible=False, promotion_eligible=False))
    config = config_dir / "relationships.json"
    _json(config, dict(schema_version="market_predictor.return_relationship_publication_config",
        parent_publication=_pin(root, joined / "_manifest.json"), parent_saved_row_verification=_pin(root, receipt_path),
        feature_config=_pin(root, feature), predictor_failure_facts=_pin(root, facts_path),
        strategy_contract=readiness_policy["strategy_contract"]))
    output = root / "data/features/relationships"
    partial = owner.materialize_return_relationships(root, config, file_sha256(config), output, maximum_groups_this_run=1)
    assert partial["status"] == "in_progress" and not (output / "_manifest.json").exists()
    checkpoint = output / "_checkpoint.json"
    checkpoint_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(checkpoint_bytes + b" ")
    with pytest.raises(DataReadinessError, match="metadata hash mismatch"):
        owner._materialize(root, output, None, None, None, {}, partial["checkpoint_sha256"], None)
    checkpoint.write_bytes(checkpoint_bytes)
    initializer = root / "src/market_predictor/__init__.py"
    initializer_bytes = initializer.read_bytes()
    initializer.write_bytes(initializer_bytes + b"\nCHANGED_INITIALIZER = True\n")
    with pytest.raises(DataReadinessError, match="resume sources or implementation changed"):
        owner.materialize_return_relationships(root, config, file_sha256(config), output,
            expected_checkpoint_sha256=partial["checkpoint_sha256"])
    initializer.write_bytes(initializer_bytes)
    stage = output / "_baseline_stage/_manifest.json"
    stage_bytes = stage.read_bytes()
    stage.write_bytes(stage_bytes + b" ")
    with pytest.raises(DataReadinessError, match="metadata hash mismatch"):
        owner.materialize_return_relationships(root, config, file_sha256(config), output,
            expected_checkpoint_sha256=partial["checkpoint_sha256"])
    stage.write_bytes(stage_bytes)
    first_group = next((output / "groups").glob("*.parquet"))
    original_group = first_group.read_bytes()
    result = owner.materialize_return_relationships(root, config, file_sha256(config), output,
        expected_checkpoint_sha256=partial["checkpoint_sha256"])
    assert first_group.read_bytes() == original_group
    publication = SourcePin(path=(output / "_manifest.json").relative_to(root).as_posix(), sha256=result["manifest_sha256"])
    readiness_policy.update(publication=_pin(root, joined / "_manifest.json"), saved_row_verification=_pin(root, receipt_path))
    return dict(root=root, publication=publication, output=output, config=SourcePin(**_pin(root, config)), sessions=sessions,
        parent=dict(policy=readiness_policy), result=result, partial=partial)


def test_real_publisher_and_metadata_verifier(publication_fixture: dict[str, Any]) -> None:
    state = publication_fixture
    verified = owner.verify_return_relationship_publication(state["root"], state["publication"])
    assert verified.manifest["rows"] == 354
    assert len(verified.months) == 59 and len(verified.model_columns) == 124
    assert verified.manifest["status"] == "complete_research_only"
    assert all(verified.manifest[name] is False for name in owner.CLOSED)
    before = {path: (file_sha256(path), path.stat().st_mtime_ns) for path in state["output"].rglob("*") if path.is_file()}
    config = state["config"]
    assert owner.materialize_return_relationships(state["root"], Path(config.path), config.sha256, state["output"],
        expected_checkpoint_sha256=state["result"]["checkpoint_sha256"]) == state["result"]
    assert before == {path: (file_sha256(path), path.stat().st_mtime_ns) for path in before}
    initializer = state["root"] / "src/market_predictor/__init__.py"
    original = initializer.read_bytes()
    try:
        initializer.write_bytes(original + b"\nCHANGED_INITIALIZER = True\n")
        with pytest.raises(DataReadinessError, match="implementation or contract changed"):
            owner.verify_return_relationship_publication(state["root"], state["publication"])
    finally:
        initializer.write_bytes(original)


def test_data_pin_cannot_be_classified_as_historical_code(tmp_path: Path) -> None:
    source = tmp_path / "data/source.py"
    source.parent.mkdir()
    source.write_bytes(b"historical data, not implementation")
    declared = {"data/source.py": file_sha256(source)}
    request = tmp_path / "data/authority/_request.json"
    _json(request, dict(schema="market_predictor.research_predictor_request", implementation_files=declared))
    declared["data/authority/_request.json"] = file_sha256(request)
    with pytest.raises(DataReadinessError, match="non-code"):
        _historical_sources(tmp_path, declared, None)  # type: ignore[arg-type]
