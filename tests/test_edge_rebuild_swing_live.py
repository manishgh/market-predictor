from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import market_predictor.edge_rebuild.swing_live as live_module
from market_predictor.canonical.reconciliation import (
    apply_event_assignment_features,
    event_feature_columns,
)
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.serving import (
    canonical_payload_sha256,
    validate_batch_live_feature_parity,
)
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.edge_rebuild.swing_features import (
    SWING_CATALYST_FEATURE_PROFILE,
    SWING_FEATURE_PROFILE,
    TECHNICAL_RANKING_FEATURES,
    build_swing_ablation_rows,
    finalize_swing_feature_panel,
    swing_model_feature_columns,
)
from market_predictor.edge_rebuild.swing_live import (
    SWING_LIVE_IDENTITY_COLUMNS,
    SWING_LIVE_INPUT_POINTER_SCHEMA,
    SWING_LIVE_INPUT_SCHEMA_VERSION,
    SWING_LIVE_REQUIRED_WATERMARKS,
    FileSwingLiveInputProvider,
    SwingLiveFeatureFrames,
    build_live_swing_features,
)
from market_predictor.resources import process_memory_snapshot
from market_predictor.swing.features.catalyst_decision_authority import (
    REQUIRED_MODEL_SOURCE_FAMILIES,
    TRACKED_SOURCE_FAMILIES,
    WINDOWS,
    CatalystDecisionAuthority,
)

ROOT = Path(__file__).resolve().parents[1]
DECISION_TIME = pd.Timestamp("2026-07-08T22:00:00Z")
AS_OF = pd.Timestamp("2026-07-08T22:05:00Z")
AUTHORITY_PATH = Path("verified-catalyst-authority")
SECURITY_COUNT = 60
VERIFY_LIVE_FEATURE_BINDINGS = live_module._verify_live_feature_bindings  # noqa: SLF001


@pytest.fixture(autouse=True)
def verified_live_binding_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        live_module,
        "_verify_live_feature_bindings",
        lambda **_kwargs: None,
    )


@pytest.fixture(scope="module")
def contract() -> StrategyContract:
    return load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")


def test_live_model_frame_excludes_only_stocks_below_sector_peer_floor() -> None:
    rows = pd.DataFrame(
        {
            "decision_id": ["d1", "d2"],
            "security_id": ["s1", "s2"],
            "ticker": ["AAA", "BBB"],
            "session_date_et": ["2026-07-08", "2026-07-08"],
            "decision_time_utc": [DECISION_TIME, DECISION_TIME],
            "feature_eligible": [True, True],
            "cross_section_eligible": [True, False],
            "feature": [0.2, 0.7],
        }
    )

    result = live_module._model_frame(  # noqa: SLF001
        rows,
        columns=("feature",),
        profile=SWING_FEATURE_PROFILE,
    )

    assert result.index.get_level_values("security_id").tolist() == ["s1"]


def test_live_security_exclusions_continue_through_five_percent(
    contract: StrategyContract,
) -> None:
    expected = tuple(f"sec:{index:03d}" for index in range(100))

    retained = live_module._validate_live_security_exclusions(  # noqa: SLF001
        expected,
        expected[:5],
        contract=contract,
        reason="test evidence",
    )

    assert retained == expected[:5]
    with pytest.raises(DataReadinessError, match="exceed the governed ceiling"):
        live_module._validate_live_security_exclusions(  # noqa: SLF001
            expected,
            expected[:6],
            contract=contract,
            reason="test evidence",
        )


def test_live_frames_match_the_shared_batch_construction(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _technical_rows()
    stamped = _stamp(rows)
    authority = _authority(stamped)
    calls = 0

    def build(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return rows.copy()

    monkeypatch.setattr(live_module, "build_swing_feature_rows", build)
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda path, **_kwargs: authority if path == AUTHORITY_PATH else None,
    )
    live = _build(contract)

    batch_profiles = build_swing_ablation_rows(stamped, authority)
    batch_final = {
        profile: finalize_swing_feature_panel(
            batch_profiles[profile],
            contract=contract,
            expected_security_ids=_security_ids(),
        )
        for profile in (SWING_FEATURE_PROFILE, SWING_CATALYST_FEATURE_PROFILE)
    }
    for profile, catalyst in (
        (SWING_FEATURE_PROFILE, False),
        (SWING_CATALYST_FEATURE_PROFILE, True),
    ):
        columns = swing_model_feature_columns(
            contract=contract,
            catalyst=catalyst,
        )
        expected = batch_final[profile].loc[:, columns].copy()
        expected.index = pd.MultiIndex.from_frame(
            batch_final[profile].loc[:, SWING_LIVE_IDENTITY_COLUMNS],
            names=SWING_LIVE_IDENTITY_COLUMNS,
        )
        observed = live.catalyst_full if catalyst else live.technical_market
        report = validate_batch_live_feature_parity(expected, observed, columns)
        assert report.matched

    assert calls == 1
    assert live.technical_market.index.equals(live.catalyst_full.index)
    assert live.decision_time_utc == DECISION_TIME
    assert live.as_of_utc == AS_OF


def test_feature_builder_verifies_generation_bindings_before_and_after_use(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _technical_rows()
    authority = _authority(_stamp(rows))
    checks: list[str] = []
    monkeypatch.setattr(
        live_module,
        "_verify_live_feature_bindings",
        lambda **_kwargs: checks.append("verified"),
    )
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: rows.copy(),
    )
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: authority,
    )

    _build(contract)

    assert checks == ["verified", "verified"]


def test_live_feature_binding_rejects_changed_authority_or_manifest(
    tmp_path: Path,
) -> None:
    authority = tmp_path / "catalyst"
    authority.mkdir()
    authority_path = authority / "_authority.json"
    manifest_path = tmp_path / "_manifest.json"
    authority_path.write_text('{"state":"complete"}\n', encoding="utf-8")
    manifest_path.write_text('{"state":"complete"}\n', encoding="utf-8")
    authority_sha = file_sha256(authority_path)
    manifest_sha = file_sha256(manifest_path)

    VERIFY_LIVE_FEATURE_BINDINGS(
        catalyst_authority_directory=authority,
        expected_catalyst_authority_sha256=authority_sha,
        live_manifest_path=manifest_path,
        expected_live_manifest_sha256=manifest_sha,
    )
    authority_path.write_text('{"state":"changed"}\n', encoding="utf-8")
    with pytest.raises(DataReadinessError, match="catalyst authority changed"):
        VERIFY_LIVE_FEATURE_BINDINGS(
            catalyst_authority_directory=authority,
            expected_catalyst_authority_sha256=authority_sha,
            live_manifest_path=manifest_path,
            expected_live_manifest_sha256=manifest_sha,
        )


@pytest.mark.skipif(
    os.environ.get("MARKET_PREDICTOR_RUN_MEMORY_TESTS") != "1",
    reason="set MARKET_PREDICTOR_RUN_MEMORY_TESTS=1 for production-scale RSS evidence",
)
def test_production_sized_projected_live_input_stays_below_four_gib(
    tmp_path: Path,
) -> None:
    rows = 250_000
    frame = pd.DataFrame(
        {
            "ticker": np.resize(np.asarray(["SPY", "QQQ", "XLK"]), rows),
            "timeframe": "1d",
            "bar_start_utc": pd.date_range("2020-01-02", periods=rows, freq="min", tz="UTC"),
            "bar_end_utc": pd.date_range("2020-01-02 00:01", periods=rows, freq="min", tz="UTC"),
            "available_at_utc": pd.date_range("2020-01-02 00:02", periods=rows, freq="min", tz="UTC"),
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": np.full(rows, 100.5),
            "volume": np.full(rows, 1_000_000, dtype="int64"),
            "price_feed": "sip",
            "adjustment": "all",
            "schema_version": "test",
        }
    )
    path = tmp_path / "production-sized.parquet"
    frame.to_parquet(path, index=False)
    del frame
    columns = (
        "ticker", "timeframe", "bar_start_utc", "bar_end_utc",
        "available_at_utc", "open", "high", "low", "close", "volume",
        "price_feed", "adjustment", "schema_version",
    )
    loaded, _, observed_rows = live_module._read_projected_parquet(  # noqa: SLF001
        tmp_path,
        {"path": path.name, "sha256": file_sha256(path), "rows": rows},
        columns=columns,
        maximum_bytes=512 * 1024 * 1024,
        maximum_rows=rows,
        label="production-sized benchmark bars",
        memory_budget_gib=4.0,
        memory_headroom_gib=0.5,
    )
    snapshot = process_memory_snapshot()

    assert observed_rows == rows
    assert len(loaded) == rows
    assert snapshot is not None
    assert snapshot[0] < 4 * 1024**3


def test_future_market_and_catalyst_evidence_are_rejected(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _technical_rows()
    authority = _authority(_stamp(rows))
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: rows.copy(),
    )
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: authority,
    )
    stock, benchmarks, memberships = _market_inputs()
    stock.loc[0, "available_at_utc"] = AS_OF + pd.Timedelta(seconds=1)
    with pytest.raises(DataReadinessError, match="after as_of_utc"):
        _build(
            contract,
            stock=stock,
            benchmarks=benchmarks,
            memberships=memberships,
        )

    poisoned = _authority(_stamp(rows))
    poisoned.decisions.loc[:, "latest_event_feature_available_at_utc"] = AS_OF + pd.Timedelta(seconds=1)
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: poisoned,
    )
    with pytest.raises(DataReadinessError, match="catalyst.*after as_of_utc"):
        _build(contract)


def test_incomplete_same_session_cross_section_excludes_one_security(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    incomplete = _technical_rows().iloc[:-1].copy()
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: incomplete,
    )
    authority = _authority(_stamp(incomplete))
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: authority,
    )

    result = _build(contract)

    assert len(result.excluded_security_ids) == 1
    assert len(result.catalyst_full) == SECURITY_COUNT - 1


def test_unknown_required_catalyst_coverage_abstains(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _technical_rows()
    unavailable = _authority(_stamp(rows), alpaca_available=False)
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: rows.copy(),
    )
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: unavailable,
    )

    with pytest.raises(
        DataReadinessError,
        match="exceed the governed ceiling",
    ):
        _build(contract)


def test_excludes_one_under_warm_security_and_rejects_naive_cutoff(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold = _technical_rows()
    cold.loc[0, "daily_bar_count"] = contract.swing.minimum_warmup_sessions - 1
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: cold,
    )
    authority = _authority(_stamp(cold))
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: authority,
    )
    result = _build(contract)

    assert result.excluded_security_ids == ("security-001",)

    with pytest.raises(DataReadinessError, match="timezone-aware"):
        _build(contract, as_of_utc="2026-07-08 22:05:00")


def test_rejects_stale_closed_session_and_allows_unrelated_authority_history(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _technical_rows()
    authority = _authority(_stamp(rows))
    extra = authority.decisions.iloc[[0]].copy()
    extra["decision_id"] = "unrelated-history"
    extra["decision_time_utc"] = DECISION_TIME - pd.Timedelta(days=1)
    authority = CatalystDecisionAuthority(
        directory=authority.directory,
        decisions=pd.concat([extra, authority.decisions], ignore_index=True),
        coverage=authority.coverage,
        manifest=authority.manifest,
        authority=authority.authority,
    )
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: authority,
    )
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: rows.copy(),
    )
    _build(contract)

    stale = rows.copy()
    stale["session_date_et"] = pd.Timestamp("2026-07-07").date()
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: stale.copy(),
    )
    with pytest.raises(DataReadinessError, match="decision is stale"):
        _build(contract)


def test_research_catalyst_authority_is_rejected_for_live_swing(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _technical_rows()
    authority = _authority(_stamp(rows), production_ready=False)
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: rows.copy(),
    )
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: authority,
    )

    with pytest.raises(DataReadinessError, match="production-ready"):
        _build(contract)


def test_live_swing_rejects_catalyst_collection_completed_after_cutoff(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _technical_rows()
    authority = _authority(
        _stamp(rows),
        collection_completed_at=AS_OF + pd.Timedelta(seconds=1),
    )
    monkeypatch.setattr(
        live_module,
        "build_swing_feature_rows",
        lambda *_args, **_kwargs: rows.copy(),
    )
    monkeypatch.setattr(
        live_module,
        "load_catalyst_decision_authority",
        lambda _path, **_kwargs: authority,
    )

    with pytest.raises(DataReadinessError, match="collection completion"):
        _build(contract)


def test_file_live_input_provider_verifies_atomic_manifest(tmp_path: Path) -> None:
    root = tmp_path / "live"
    root.mkdir()
    generation_staging = root / "staging"
    generation_staging.mkdir()
    files: dict[str, dict[str, object]] = {}
    bar = pd.DataFrame(
        {
            "ticker": ["MSFT"],
            "timeframe": ["1d"],
            "bar_start_utc": [pd.Timestamp("2026-07-08T13:30:00Z")],
            "bar_end_utc": [pd.Timestamp("2026-07-08T20:00:00Z")],
            "available_at_utc": [pd.Timestamp("2026-07-08T20:15:00Z")],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1_000_000],
            "price_feed": ["sip"],
            "adjustment": ["all"],
            "schema_version": ["test"],
        }
    )
    membership = pd.DataFrame(
        {
            "ticker": ["MSFT"],
            "effective_from_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
            "security_id": ["sec:msft"],
            "sector": ["Technology"],
            "industry": ["Software"],
            "market_cap_bucket": ["large"],
            "liquidity_bucket": ["liquid"],
            "primary_benchmark": ["XLK"],
            "universe_snapshot_id": ["snapshot"],
            "source": ["test"],
        }
    )
    frames = {
        "stock_daily_bars": bar,
        "benchmark_daily_bars": bar.assign(ticker="SPY"),
        "point_in_time_memberships": membership,
    }
    for name, frame in frames.items():
        path = generation_staging / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        files[name] = {
            "path": path.name,
            "sha256": file_sha256(path),
            "rows": len(frame),
        }
    catalyst = generation_staging / "catalyst"
    catalyst.mkdir()
    (catalyst / "_authority.json").write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema": SWING_LIVE_INPUT_SCHEMA_VERSION,
        "state": "complete",
        "generated_at_utc": DECISION_TIME.isoformat(),
        "market_data_provider": "alpaca",
        "market_data_feed": "sip",
        "market_data_adjustment": "all",
        "files": files,
        "catalyst_authority_directory": "catalyst",
        "catalyst_authority_sha256": file_sha256(catalyst / "_authority.json"),
        "source_watermarks": {
            key: DECISION_TIME.isoformat() for key in SWING_LIVE_REQUIRED_WATERMARKS
        },
    }
    manifest_path = generation_staging / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    generation_id = file_sha256(manifest_path)
    generations = root / "generations"
    generations.mkdir()
    generation = generations / generation_id
    generation_staging.rename(generation)
    pointer = {
        "schema": SWING_LIVE_INPUT_POINTER_SCHEMA,
        "generation_id": generation_id,
        "manifest_file_sha256": generation_id,
        "previous_generation_id": None,
        "activated_at_utc": DECISION_TIME.isoformat(),
    }
    pointer["pointer_sha256"] = canonical_payload_sha256(pointer)
    (root / "active_generation.json").write_text(
        json.dumps(pointer),
        encoding="utf-8",
    )

    loaded = FileSwingLiveInputProvider(root).load(
        as_of_utc=AS_OF.to_pydatetime(),
        maximum_bytes=10_000_000,
        maximum_rows=100,
    )

    assert loaded.generation_id == generation_id
    assert loaded.catalyst_authority_sha256 == file_sha256(
        generation / "catalyst" / "_authority.json"
    )
    with pytest.raises(DataReadinessError, match="aggregate input limit|row limit"):
        FileSwingLiveInputProvider(root).load(
            as_of_utc=AS_OF.to_pydatetime(),
            maximum_bytes=10_000_000,
            maximum_rows=2,
        )
    with pytest.raises(DataReadinessError, match="byte limit"):
        FileSwingLiveInputProvider(root).load(
            as_of_utc=AS_OF.to_pydatetime(),
            maximum_bytes=1,
            maximum_rows=100,
        )
    with pytest.raises(DataReadinessError, match="benchmark.*SIP/all"):
        live_module._require_physical_sip_all(  # noqa: SLF001
            bar.assign(price_feed="iex"),
            "benchmark daily bars",
        )
    authority_path = generation / "catalyst" / "_authority.json"
    authority_bytes = authority_path.read_bytes()
    authority_path.write_text('{"tampered":true}', encoding="utf-8")
    with pytest.raises(DataReadinessError, match="catalyst authority hash"):
        FileSwingLiveInputProvider(root).load(
            as_of_utc=AS_OF.to_pydatetime(),
            maximum_bytes=10_000_000,
            maximum_rows=100,
        )
    authority_path.write_bytes(authority_bytes)
    (generation / "stock_daily_bars.parquet").write_bytes(b"tampered")
    with pytest.raises(DataReadinessError, match="hash does not verify"):
        FileSwingLiveInputProvider(root).load(
            as_of_utc=AS_OF.to_pydatetime(),
            maximum_bytes=10_000_000,
            maximum_rows=100,
        )


def _build(
    contract: StrategyContract,
    *,
    stock: pd.DataFrame | None = None,
    benchmarks: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    as_of_utc: object = AS_OF,
) -> SwingLiveFeatureFrames:
    default_stock, default_benchmarks, default_memberships = _market_inputs()
    return build_live_swing_features(
        default_stock if stock is None else stock,
        default_benchmarks if benchmarks is None else benchmarks,
        default_memberships if memberships is None else memberships,
        contract=contract,
        catalyst_authority_directory=AUTHORITY_PATH,
        expected_catalyst_authority_sha256="a" * 64,
        live_manifest_path=Path("verified-live-manifest.json"),
        expected_live_manifest_sha256="b" * 64,
        as_of_utc=as_of_utc,
    )


def _technical_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for index, security_id in enumerate(_security_ids(), start=1):
        scale = float(index)
        record: dict[str, object] = {
            "security_id": security_id,
            "ticker": f"T{index:03d}",
            "session_date_et": DECISION_TIME.tz_convert("America/New_York").date(),
            "decision_time_utc": DECISION_TIME,
            "prediction_cutoff_policy_id": "xnys_1800_america_new_york_v1",
            "timeframe": "1d",
            "bar_start_utc": pd.Timestamp("2026-07-08T13:30:00Z"),
            "sector": "Technology",
            "feature_profile": SWING_FEATURE_PROFILE,
            "feature_eligible": True,
            "label_eligible": False,
            "daily_bar_count": 300,
            "forward_return": np.nan,
            "barrier_label": pd.NA,
        }
        for feature_number, feature in enumerate(
            TECHNICAL_RANKING_FEATURES,
            start=1,
        ):
            record[feature] = scale * float(feature_number)
        records.append(record)
    return pd.DataFrame.from_records(records)


def _stamp(rows: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=("decision_id", "latest_event_feature_available_at_utc"))
    return apply_event_assignment_features(
        rows,
        empty,
        windows={},
        source_families=(),
    ).drop(columns="latest_event_feature_available_at_utc")


def _authority(
    decisions: pd.DataFrame,
    *,
    alpaca_available: bool = True,
    production_ready: bool = True,
    collection_completed_at: pd.Timestamp = DECISION_TIME,
) -> CatalystDecisionAuthority:
    feature_columns = event_feature_columns(
        WINDOWS,
        source_families=TRACKED_SOURCE_FAMILIES,
    )
    evidence_records: list[dict[str, object]] = []
    if alpaca_available:
        for index, decision in enumerate(decisions.itertuples(index=False), start=1):
            scale = float(index)
            record: dict[str, object] = {
                "decision_id": decision.decision_id,
                "security_id": decision.security_id,
                "ticker": decision.ticker,
                "decision_time_utc": decision.decision_time_utc,
                "evidence_lineage_count": 1,
                "evidence_lineage_sha256s": '["lineage"]',
            }
            for column in feature_columns:
                if column == "latest_event_feature_available_at_utc":
                    record[column] = DECISION_TIME - pd.Timedelta(hours=1)
                elif column.startswith("source_count_alpaca_"):
                    record[column] = scale
                elif column.startswith("source_count_"):
                    record[column] = 0.0
                elif column.startswith("event_count_"):
                    record[column] = scale
                elif column.startswith("source_family_count_"):
                    record[column] = 1.0
                elif column.startswith("unknown_relevance_event_fraction_"):
                    record[column] = 0.0
                elif column.startswith("low_relevance_event_fraction_"):
                    record[column] = scale / 100.0
                elif column.startswith("sentiment_coverage_"):
                    record[column] = 0.5 + scale / 200.0
                else:
                    record[column] = scale / 100.0
            evidence_records.append(record)
    evidence = pd.DataFrame.from_records(
        evidence_records,
        columns=(
            "decision_id",
            "security_id",
            "ticker",
            "decision_time_utc",
            "evidence_lineage_count",
            "evidence_lineage_sha256s",
            *feature_columns,
        ),
    )
    coverage_records: list[dict[str, object]] = []
    for decision in decisions.itertuples(index=False):
        for family in TRACKED_SOURCE_FAMILIES:
            unknown = family == "alpaca" and not alpaca_available
            coverage_records.append(
                {
                    "security_id": decision.security_id,
                    "ticker": decision.ticker,
                    "source_family": family,
                    "requested_start_utc": DECISION_TIME - pd.Timedelta(days=4),
                    "requested_end_utc": DECISION_TIME,
                    "started_at_utc": DECISION_TIME - pd.Timedelta(minutes=1),
                    "completed_at_utc": collection_completed_at,
                    "coverage_state": (
                        "failed_or_unobserved" if unknown else ("observed_complete" if family == "alpaca" else "observed_empty")
                    ),
                    "missingness_known": not unknown,
                    "training_eligible": not unknown,
                }
            )
    return CatalystDecisionAuthority(
        directory=AUTHORITY_PATH,
        decisions=evidence,
        coverage=pd.DataFrame.from_records(coverage_records),
        manifest={
            "production_ready": production_ready,
            "completed_at_utc": DECISION_TIME.isoformat(),
            "tracked_source_families": list(TRACKED_SOURCE_FAMILIES),
            "required_model_source_families": list(REQUIRED_MODEL_SOURCE_FAMILIES),
        },
        authority={},
    )


def _security_ids() -> tuple[str, ...]:
    return tuple(f"security-{index:03d}" for index in range(1, SECURITY_COUNT + 1))


def _market_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stock = pd.DataFrame(
        {
            "bar_start_utc": [pd.Timestamp("2026-07-08T13:30:00Z")],
            "bar_end_utc": [pd.Timestamp("2026-07-08T20:00:00Z")],
            "available_at_utc": [pd.Timestamp("2026-07-08T20:15:00Z")],
        }
    )
    benchmarks = stock.copy()
    memberships = pd.DataFrame.from_records(
        [
            {
                "ticker": f"T{index:03d}",
                "security_id": security_id,
                "effective_from_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
                "effective_to_utc": pd.NaT,
                "available_at_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
            }
            for index, security_id in enumerate(_security_ids(), start=1)
        ]
    )
    return stock, benchmarks, memberships
