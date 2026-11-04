"""Synthetic UNIT sources only; no observed-data or operational admission proof."""
from datetime import UTC, datetime

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.research.observed_return_publication import _observed_bars
from market_predictor.swing.features.observed_return_publication import ObservedReturnPublicationRequest, ObservedSourcePin
from tests.test_swing_predictor_history import _inputs


def test_observed_price_consumption_checks_physical_semantics_and_ingestion() -> None:
    stocks, _ = _inputs()
    stocks["source"] = "alpaca"
    stocks["availability_policy"] = "observed"
    stocks["ingested_at_utc"] = stocks.available_at_utc
    cutoff = pd.Timestamp("2024-01-01T00:00:00Z")
    actual = _observed_bars(stocks, adjustment="all", cutoff=cutoff)
    assert len(actual) == len(stocks)
    proxy = stocks.assign(availability_policy="market_interval_close")
    with pytest.raises(DataReadinessError, match="observed inputs"):
        _observed_bars(proxy, adjustment="all", cutoff=cutoff)
    future = stocks.copy()
    future.loc[0, "ingested_at_utc"] = future.loc[0, "available_at_utc"] + pd.Timedelta(seconds=1)
    with pytest.raises(DataReadinessError, match="ingestion"):
        _observed_bars(future, adjustment="all", cutoff=cutoff)
    with pytest.raises(DataReadinessError, match="precede decision"):
        _observed_bars(stocks, adjustment="all", cutoff=pd.Timestamp("2022-01-01T00:00:00Z"))


def test_publication_request_requires_explicit_roles_and_refuses_bare_feature_pins() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ObservedReturnPublicationRequest.model_validate({"as_of_utc": datetime.now(UTC),
            "source_pins": {"unrelated.txt": "a" * 64}, "features": {"rsi": 1.0}})
    for path in ("../source.json", "C:/source.json", "nested\\source.json"):
        with pytest.raises(ValidationError):
            ObservedSourcePin(path=path, sha256="a" * 64)


def test_implementation_inventory_captures_loaded_numerical_dependency_closure() -> None:
    from pathlib import Path

    import market_predictor.live_features
    import market_predictor.swing.dataset
    from market_predictor.research import observed_return_publication as producer

    assert market_predictor.live_features.__file__ and market_predictor.swing.dataset.__file__
    root = Path(__file__).resolve().parents[1]
    files = producer._implementations(root)
    assert {"src/market_predictor/swing/dataset.py", "src/market_predictor/live_features.py",
        "src/market_predictor/research/observed_return_publication.py",
        "src/market_predictor/swing/features/observed_return_publication.py"}.issubset(files)
    assert all(name.startswith("src/market_predictor/") and name.endswith(".py") for name in files)


def test_configured_busy_publisher_lease_precedes_all_source_consumption(tmp_path, monkeypatch) -> None:
    from contextlib import contextmanager
    from pathlib import Path

    from market_predictor.heavy_jobs import HeavyJobBusyError
    from market_predictor.research import observed_return_publication as producer

    @contextmanager
    def busy(name, *, runtime_dir):
        assert runtime_dir == tmp_path / "configured-runtime"
        raise HeavyJobBusyError("UNIT busy")
        yield  # pragma: no cover

    monkeypatch.setattr(producer, "heavy_job_runtime_dir", lambda: Path("configured-runtime"))
    monkeypatch.setattr(producer, "heavy_job_lease", busy)
    monkeypatch.setattr(producer, "compose_observed_return_features", lambda **kwargs: pytest.fail("source read before lease"))
    with pytest.raises(HeavyJobBusyError):
        producer.publish_observed_return_features(root=tmp_path, request=None, output_directory=Path("output"))


@pytest.fixture
def registered_generation(tmp_path):
    """Trusted reader UNIT generation; deliberately not real source verification."""
    import json
    from pathlib import Path

    from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
    from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
    from market_predictor.evidence.hashing import json_sha256
    from market_predictor.swing.features.observed_return_publication import CLOSED, PROFILES, SCHEMA, request_source_files

    def source(name):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("UNIT source-role bytes " + name)
        return {"path": name, "sha256": file_sha256(path)}

    roles = {role: source(role + ".json") for role in
        ("membership_authority", "cohort", "strategy_contract", "training_feature_request", "news_source_index")}
    for role in ("stock_history", "raw_decision_bars", "benchmark_history"):
        pin = source(role + ".parquet")
        manifest = source(role + ".parquet.manifest.json")
        roles[role] = {**pin, "manifest_sha256": manifest["sha256"]}
    models = {profile: source(profile + "/_manifest.json") for profile in PROFILES}
    now = datetime.now(UTC)
    request = ObservedReturnPublicationRequest(as_of_utc=now, history_start=datetime(2022, 1, 1, tzinfo=UTC),
        model_runs=models, **roles)
    files = request_source_files(request)
    implementation = source("UNIT_implementation.py")
    frame = pd.DataFrame({"decision_id": ["d1", "d2"], "security_id": ["s1", "s2"], "ticker": ["AAA", "BBB"],
                          "feature_a": [0.0, float("nan")], "feature_b": [-0.1, 0.2], "forecast_eligible": [True, False]})
    directory = tmp_path / "generation"
    directory.mkdir()
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="UNIT", status="pass", failures=0,
        rows_checked=2, detail="Synthetic registered-reader fixture, no source admission"),))
    records = {}
    for profile in PROFILES:
        path = directory / (profile + ".parquet")
        write_canonical_artifact(frame, path, artifact_type="swing_return_live_features", audit=audit,
                                 inputs=files, production_ready=False)
        records[profile] = {"path": path.name, "sha256": file_sha256(path),
            "manifest_sha256": file_sha256(manifest_path_for(path)), "feature_names": ["feature_a", "feature_b"]}
    metadata = {"schema": SCHEMA, "status": "complete_observed_research_inputs", "created_at_utc": datetime.now(UTC).isoformat(),
        "request": request.model_dump(mode="json"), "request_sha256": json_sha256(request.model_dump(mode="json")),
        "source_files": files, "implementation_files": {implementation["path"]: implementation["sha256"]},
        "profiles": records, "decision_ids_sha256": json_sha256(["d1", "d2"]),
        "security_ids_sha256": json_sha256(["s1", "s2"]), "peer_population": {"security_ids": ["s1", "s2"]}, **CLOSED}
    manifest = directory / "_manifest.json"
    manifest.write_text(json.dumps(metadata))
    return dict(root=tmp_path, directory=Path("generation"), expected_manifest_sha256=file_sha256(manifest),
        as_of=datetime.now(UTC), profile="technical", feature_names=["feature_a", "feature_b"],
        run_manifest_sha256=models["technical"]["sha256"]), metadata


def test_registered_generation_reader_preserves_zero_missingness_and_population(registered_generation):
    from market_predictor.swing.features.observed_return_publication import verify_observed_return_publication

    args, _ = registered_generation
    frame, provenance = verify_observed_return_publication(**args)
    assert frame.feature_a.iloc[0] == 0 and pd.isna(frame.feature_a.iloc[1])
    assert frame.forecast_eligible.tolist() == [True, False]
    assert provenance["publication_manifest_sha256"] == args["expected_manifest_sha256"]


@pytest.mark.parametrize("poison", ["source", "model", "order", "population", "schema", "future", "bytes", "rows"])
def test_registered_generation_refuses_tampering_scope_and_read_budgets(registered_generation, poison):
    import json

    from market_predictor.canonical.store import file_sha256
    from market_predictor.swing.features.observed_return_publication import verify_observed_return_publication

    args, metadata = registered_generation
    if poison == "source":
        (args["root"] / "stock_history.parquet").write_text("changed UNIT source")
    elif poison == "model":
        args["run_manifest_sha256"] = "a" * 64
    elif poison == "order":
        args["feature_names"] = ["feature_b", "feature_a"]
    elif poison in ("population", "schema"):
        metadata["decision_ids_sha256" if poison == "population" else "schema"] = "wrong"
        manifest = args["root"] / args["directory"] / "_manifest.json"
        manifest.write_text(json.dumps(metadata))
        args["expected_manifest_sha256"] = file_sha256(manifest)
    elif poison == "future":
        args["as_of"] = datetime(2000, 1, 1, tzinfo=UTC)
    else:
        args["maximum_bytes" if poison == "bytes" else "maximum_rows"] = 1
    with pytest.raises(DataReadinessError):
        verify_observed_return_publication(**args)


def test_bare_canonical_snapshot_cannot_replace_registered_generation(registered_generation):
    from market_predictor.swing.features.observed_return_publication import verify_observed_return_publication

    args, _ = registered_generation
    args["directory"] = args["directory"] / "technical.parquet"
    with pytest.raises((DataReadinessError, OSError)):
        verify_observed_return_publication(**args)


def test_synthetic_canonical_composition_traverses_real_120_124_209_kernels(tmp_path, monkeypatch):
    """Actual numerical kernels; synthetic authority adapters establish no live proof."""
    import json
    from contextlib import contextmanager
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
    from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
    from market_predictor.modeling.strategy_contract import load_strategy_contract
    from market_predictor.research import news_decision_features as news_kernel
    from market_predictor.research import observed_return_publication as producer
    from market_predictor.research.news_source_links import SourceLinkedNewsSlice
    from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
    from market_predictor.swing.features.panel import swing_model_feature_columns
    from market_predictor.swing.training.return_artifacts import FAMILIES, FLAGS, SCOPES
    from tests.test_swing_adjusted_source import _source_inputs

    def pin(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return {"path": name, "sha256": file_sha256(path)}

    strategy_path = Path(__file__).resolve().parents[1] / "configs/edge_rebuild_strategy_contract.toml"
    (tmp_path / "strategy.toml").write_bytes(strategy_path.read_bytes())
    strategy = {"path": "strategy.toml", "sha256": file_sha256(tmp_path / "strategy.toml")}
    contract = load_strategy_contract(tmp_path / "strategy.toml")
    original = _source_inputs()
    stocks, members = [], []
    for index in range(60):
        identity, symbol = f"sec:{index:03d}", f"T{index:03d}"
        rows = original["adjusted_bars"].copy().assign(security_id=identity, ticker=symbol, source="alpaca", availability_policy="observed")
        rows["ingested_at_utc"] = rows.available_at_utc
        stocks.append(rows)
        members.append(original["memberships"].assign(security_id=identity, ticker=symbol))
    stocks = pd.concat(stocks, ignore_index=True)
    stocks["session_date_et"] = pd.to_datetime(stocks.bar_start_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    end = max(stocks.session_date_et)
    raw = stocks.loc[stocks.session_date_et.eq(end)].copy().assign(
        adjustment="raw", open=123., high=124., low=122., close=123., volume=456.)
    benchmarks = original["benchmark_bars"].copy().assign(source="alpaca", availability_policy="observed")
    benchmarks["ingested_at_utc"] = benchmarks.available_at_utc
    memberships = pd.concat(members, ignore_index=True)
    decision = original["decisions"].decision_time_utc.iloc[-1]
    payload = pin("UNIT_collection.json", {"scope": "synthetic source bytes"})
    roles = {}
    for role, frame in (("stock_history", stocks), ("raw_decision_bars", raw), ("benchmark_history", benchmarks)):
        path = tmp_path / (role + ".parquet")
        audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="UNIT", status="pass", failures=0,
            rows_checked=len(frame), detail="Synthetic observed-source test, no real-source proof"),))
        write_canonical_artifact(frame, path, artifact_type="bars", audit=audit,
            inputs={payload["path"]: payload["sha256"]}, production_ready=False)
        roles[role] = {"path": path.name, "sha256": file_sha256(path), "manifest_sha256": file_sha256(manifest_path_for(path))}
    directories = {}
    for name in ("base_membership_directory", "closed_archive_directory", "closed_event_directory"):
        target = tmp_path / name
        target.mkdir()
        directories[name] = str(target)
    pin("membership/_request.json", directories)
    roles["membership_authority"] = pin("membership/_authority.json", {"UNIT": True})
    roles["cohort"] = pin("cohort.json", {"UNIT": True})
    roles["training_feature_request"] = pin("parent_request.json", {"cohort_sha256": "c" * 64})
    roles["news_source_index"] = pin("news/_manifest.json", {"UNIT": True})
    names = {"technical": list(swing_model_feature_columns(contract=contract, catalyst=False))}
    names["relationships"] = [*names["technical"], *RETURN_RELATIONSHIP_COLUMNS]
    names["news"] = [*names["relationships"], *news_kernel.NEWS_COLUMNS]
    model_runs = {}
    for profile, feature_profile in (("technical", "existing_technical"), ("relationships", "technical_relationships"),
                                      ("news", "technical_relationships_news")):
        body = pin(f"{profile}/_request.json", {"schema": "market_predictor.swing_return_training_request",
            "policy": {"feature_profile": feature_profile, "scope": "initial_fit_research_only",
            "target": "spy_fixed_horizon_excess_return"}, **dict.fromkeys(FLAGS, False),
            "folds": [{"number": number} for number in range(1, 5)],
            "source_files": {strategy["path"]: strategy["sha256"], roles["training_feature_request"]["path"]:
                             roles["training_feature_request"]["sha256"]}, "feature_names": names[profile]})
        units = {f"{family}/{scope}/fold-{number}": {} for family in FAMILIES for scope in SCOPES for number in range(1, 5)}
        units.update({f"{family}/final_refit": {} for family in FAMILIES})
        model_runs[profile] = pin(f"{profile}/_manifest.json", {"schema": "market_predictor.swing_return_training",
            "status": "complete_research_only", "request_sha256": body["sha256"],
            "units": units, **dict.fromkeys(FLAGS, False)})
    cohort = SimpleNamespace(retained_security_ids=tuple(memberships.security_id), sha256=lambda: "c" * 64)
    authority = SimpleNamespace(memberships=memberships, parent={"observed_at_utc": decision.isoformat()})
    monkeypatch.setattr(producer, "load_swing_research_cohort", lambda *args, **kwargs: cohort)
    monkeypatch.setattr(producer, "load_observed_sp500_membership_authority", lambda *args: authority)
    monkeypatch.setattr(producer, "guard", lambda stage: None)
    monkeypatch.setattr(producer, "_implementations", lambda root: {payload["path"]: payload["sha256"]})

    class Reader:
        availability_semantics = "observed"
        decision_end_utc = decision.to_pydatetime()
        source_files = {}
        implementation_files = {}

        def for_period(self, *args, **kwargs):
            return SourceLinkedNewsSlice((), (), ())

    @contextmanager
    def observed_news(**kwargs):
        yield Reader()

    monkeypatch.setattr(producer, "open_news_source_link_index", observed_news)
    request = ObservedReturnPublicationRequest(as_of_utc=decision.to_pydatetime(),
        history_start=datetime(2022, 1, 3, tzinfo=UTC), strategy_contract=strategy, model_runs=model_runs, **roles)
    profiles, provenance = producer.compose_observed_return_features(root=tmp_path, request=request)
    assert [len(profiles[name].model_columns) for name in ("technical", "relationships", "news")] == [120, 124, 209]
    assert all(len(part.rows) == 60 for part in profiles.values())
    technical = profiles["technical"]
    np.testing.assert_allclose(technical.rows.dollar_volume_log, np.log1p(123 * 456))
    pd.testing.assert_frame_equal(technical.rows[names["technical"]], profiles["news"].rows[names["technical"]], check_exact=True)
    assert profiles["relationships"].rows.momentum_252_sessions_excluding_recent_21.notna().all()
    assert profiles["news"].rows.news_direct_earnings_present_3d.isna().all()
    assert profiles["news"].rows.news_direct_source_coverage_known_alpaca_3d.eq(0).all()
    assert len(provenance["peer_population"]["security_ids"]) == 60
