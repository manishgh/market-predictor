from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

import market_predictor.serving.live_input_publication as publication
from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.promotion.bundle_contracts import canonical_payload_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.serving.live_input_publication import CanonicalInput, LiveInputPublication, PinnedFile, publish_live_inputs
from market_predictor.serving.swing_features import FileSwingLiveInputProvider
from market_predictor.swing.features.catalyst_decision_authority import publish_catalyst_decision_authority
from tests import test_swing_catalyst_decision_authority as catalyst_fixture

NOW = datetime(2026, 7, 8, 22, 5, tzinfo=UTC)
DECISION = pd.Timestamp("2026-07-08T22:00:00Z")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(publication, "_now", lambda: NOW)
    monkeypatch.setattr(catalyst_fixture, "DECISION_TIME", DECISION - pd.Timedelta(hours=1))
    lineage = catalyst_fixture._lineage(tmp_path / "lineage", generation="1", production_ready=True)
    authority = publish_catalyst_decision_authority([lineage], tmp_path / "authority", production_ready=True)
    bar = pd.DataFrame({
        "ticker": ["MSFT"], "timeframe": ["1d"], "bar_start_utc": [DECISION - pd.Timedelta(hours=8.5)],
        "bar_end_utc": [DECISION - pd.Timedelta(hours=2)], "available_at_utc": [DECISION - pd.Timedelta(hours=1.75)],
        "open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0], "volume": [1_000_000],
        "price_feed": ["sip"], "adjustment": ["all"], "schema_version": ["test"],
    })
    membership = pd.DataFrame({
        "ticker": ["MSFT"], "effective_from_utc": [DECISION - pd.Timedelta(days=500)], "effective_to_utc": [pd.NaT],
        "available_at_utc": [DECISION - pd.Timedelta(minutes=5)], "security_id": ["sec:msft"], "sector": ["Technology"],
        "industry": ["Software"], "market_cap_bucket": ["large"], "liquidity_bucket": ["liquid"],
        "primary_benchmark": ["XLK"], "universe_snapshot_id": ["snapshot"], "source": ["synthetic"],
    })
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="synthetic fixture", status="pass", failures=0,
                                                           rows_checked=1, detail="unit test only"),))
    pins = {}
    for name, frame in {"stock_daily_bars": bar, "benchmark_daily_bars": bar.assign(ticker="SPY"),
                        "point_in_time_memberships": membership}.items():
        path = tmp_path / "inputs" / f"{name}.parquet"
        write_canonical_artifact(frame, path, artifact_type="memberships" if name == "point_in_time_memberships" else "bars", audit=audit)
        pins[name] = CanonicalInput(path=path, sha256=file_sha256(path), manifest_sha256=file_sha256(manifest_path_for(path)))
    strategy = ROOT / "configs/edge_rebuild_strategy_contract.toml"
    request = LiveInputPublication(as_of_utc=NOW, **pins,
                                   catalyst_authority=PinnedFile(path=authority.directory / "_authority.json",
                                                                sha256=file_sha256(authority.directory / "_authority.json")),
                                   strategy_contract=PinnedFile(path=strategy, sha256=file_sha256(strategy)))
    calls = []
    def features(*args, **kwargs):
        calls.append((args, kwargs))
        assert kwargs["as_of_utc"] == pd.Timestamp(NOW)
        assert file_sha256(kwargs["live_manifest_path"]) == kwargs["expected_live_manifest_sha256"]
        return SimpleNamespace(decision_time_utc=DECISION)
    # Feature mathematics are tested by test_swing_live_features. Here only that expensive
    # boundary is stubbed; canonical/production catalyst verification and the reader are real.
    monkeypatch.setattr(publication, "build_live_swing_features", features)
    return request, tmp_path / "live", calls


def _load(root, *, provider=None, as_of=NOW):
    return (provider or FileSwingLiveInputProvider(root)).load(as_of_utc=as_of, maximum_bytes=512 * 1024 * 1024, maximum_rows=2_000_000)


def test_publication_reader_round_trip_and_identical_retry(source, monkeypatch):
    request, root, calls = source
    first = publish_live_inputs(request, root)
    pointer = (root / "active_generation.json").read_bytes()
    loaded = _load(root)
    assert loaded.stock_daily_bars.ticker.tolist() == ["MSFT"]
    assert loaded.generation_id == first["generation_id"]
    assert loaded.source_watermarks["stock_daily_bars_available_at_utc"] == "2026-07-08T20:15:00+00:00"
    monkeypatch.setattr(publication, "_now", lambda: NOW + timedelta(seconds=10))
    retry = publish_live_inputs(request, root)
    assert retry["reused"] and retry["generation_id"] == first["generation_id"]
    assert (root / "active_generation.json").read_bytes() == pointer
    assert len(calls) == 2
    assert not list(root.glob(".publication-*"))


def test_failed_feature_validation_leaves_previous_generation_active(source, monkeypatch):
    request, root, _ = source
    publish_live_inputs(request, root)
    pointer = (root / "active_generation.json").read_bytes()
    def fail(*args, **kwargs):
        raise DataReadinessError("sector failure ceiling exceeded")
    monkeypatch.setattr(publication, "build_live_swing_features", fail)
    with pytest.raises(DataReadinessError, match="ceiling"):
        publish_live_inputs(request, root)
    assert (root / "active_generation.json").read_bytes() == pointer
    _load(root)


@pytest.mark.parametrize("kind", ["bytes", "rows", "input_hash", "research", "future"])
def test_bad_source_cannot_publish(source, kind):
    request, root, _ = source
    options = {}
    if kind == "bytes":
        options["maximum_bytes"] = 1
    elif kind == "rows":
        options["maximum_rows"] = 1
    elif kind == "input_hash":
        request.stock_daily_bars.path.write_bytes(b"changed")
    elif kind == "research":
        path = manifest_path_for(request.stock_daily_bars.path)
        value = json.loads(path.read_text())
        value["production_ready"] = False
        path.write_text(json.dumps(value))
        request = request.model_copy(update={"stock_daily_bars": request.stock_daily_bars.model_copy(
            update={"manifest_sha256": file_sha256(path)})})
    else:
        request = request.model_copy(update={"as_of_utc": NOW + timedelta(hours=1)})
    with pytest.raises(DataReadinessError):
        publish_live_inputs(request, root, **options)
    assert not (root / "active_generation.json").exists()


def test_activation_future_is_rejected_even_after_cache_load(source):
    request, root, _ = source
    publish_live_inputs(request, root)
    provider = FileSwingLiveInputProvider(root)
    _load(root, provider=provider)
    with pytest.raises(DataReadinessError, match="activated after"):
        _load(root, provider=provider, as_of=NOW - timedelta(seconds=1))


def test_crash_before_pointer_switch_preserves_original(source, monkeypatch):
    request, root, _ = source
    first = publish_live_inputs(request, root)
    original = (root / "active_generation.json").read_bytes()
    real_write = publication._write_json_durable
    def fail(path, value):
        if path == root / "active_generation.json":
            raise OSError("interrupted activation")
        return real_write(path, value)
    monkeypatch.setattr(publication, "_write_json_durable", fail)
    monkeypatch.setattr(publication, "_now", lambda: NOW + timedelta(seconds=20))
    changed = request.model_copy(update={"as_of_utc": NOW + timedelta(seconds=1)})
    # Boundary stub needs to accept the later cutoff for the same decision session.
    monkeypatch.setattr(publication, "build_live_swing_features", lambda *args, **kwargs: SimpleNamespace(decision_time_utc=DECISION))
    with pytest.raises(OSError, match="interrupted"):
        publish_live_inputs(changed, root)
    assert (root / "active_generation.json").read_bytes() == original
    assert _load(root).generation_id == first["generation_id"]


def test_heavy_lease_is_acquired_before_reading_sources(source):
    request, root, _ = source
    request.stock_daily_bars.path.unlink()
    with heavy_job_lease("existing"):
        with pytest.raises(HeavyJobBusyError):
            publish_live_inputs(request, root)


def test_cli_production_only_and_pinned_request(source, tmp_path):
    from market_predictor.cli_surface import command_names
    from market_predictor.collection_cli import app as collection
    from market_predictor.production_cli import app
    from market_predictor.research_cli import app as research
    request, root, _ = source
    assert "publish-swing-live-inputs" not in command_names(collection) | command_names(research)
    path = tmp_path / "request.json"
    path.write_text(request.model_dump_json())
    result = CliRunner().invoke(app, ["publish-swing-live-inputs", "--request-path", str(path),
                                     "--expected-request-sha256", file_sha256(path), "--output-directory", str(root)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["generation_id"] == _load(root).generation_id


def test_pointer_tampering_is_refused_before_publication(source):
    request, root, _ = source
    publish_live_inputs(request, root)
    path = root / "active_generation.json"
    value = json.loads(path.read_text())
    value["generation_id"] = "f" * 64
    unsigned = {key: item for key, item in value.items() if key != "pointer_sha256"}
    value["pointer_sha256"] = canonical_payload_sha256(unsigned)
    path.write_text(json.dumps(value))
    with pytest.raises(DataReadinessError):
        publish_live_inputs(request, root)



def test_real_feature_builder_refuses_incomplete_membership_before_activation(source, monkeypatch):
    from market_predictor.serving.swing_features import build_live_swing_features
    request, root, _ = source
    monkeypatch.setattr(publication, "build_live_swing_features", build_live_swing_features)
    with pytest.raises(DataReadinessError, match="cross-section.*minimum"):
        publish_live_inputs(request, root)
    assert not (root / "active_generation.json").exists()


def test_idempotent_retry_does_not_bless_tampered_catalyst_rows(source):
    request, root, _ = source
    first = publish_live_inputs(request, root)
    path = root / "generations" / first["generation_id"] / "catalyst" / "source_coverage.parquet"
    path.write_bytes(b"tampered")
    with pytest.raises(DataReadinessError):
        publish_live_inputs(request, root)


def test_future_bar_availability_is_rejected_with_otherwise_valid_new_pins(source):
    request, root, _ = source
    pin = request.stock_daily_bars
    frame = pd.read_parquet(pin.path)
    frame["available_at_utc"] = pd.Timestamp(NOW) + pd.Timedelta(seconds=1)
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="synthetic", status="pass", failures=0,
                                                           rows_checked=1, detail="test only"),))
    write_canonical_artifact(frame, pin.path, artifact_type="bars", audit=audit)
    changed = pin.model_copy(update={"sha256": file_sha256(pin.path), "manifest_sha256": file_sha256(manifest_path_for(pin.path))})
    request = request.model_copy(update={"stock_daily_bars": changed})
    with pytest.raises(DataReadinessError, match="future source"):
        publish_live_inputs(request, root)
    assert not (root / "active_generation.json").exists()


def test_changed_generation_invalidates_existing_reader_cache(source, monkeypatch):
    request, root, _ = source
    first = publish_live_inputs(request, root)
    provider = FileSwingLiveInputProvider(root)
    assert _load(root, provider=provider).generation_id == first["generation_id"]
    later = NOW + timedelta(seconds=20)
    monkeypatch.setattr(publication, "_now", lambda: later)
    monkeypatch.setattr(publication, "build_live_swing_features", lambda *args, **kwargs: SimpleNamespace(decision_time_utc=DECISION))
    next_request = request.model_copy(update={"as_of_utc": NOW + timedelta(seconds=1)})
    second = publish_live_inputs(next_request, root)
    assert _load(root, provider=provider, as_of=later).generation_id == second["generation_id"] != first["generation_id"]
    pointer = json.loads((root / "active_generation.json").read_text())
    assert pointer["previous_generation_id"] == first["generation_id"]
    with pytest.raises(DataReadinessError, match="regress"):
        publish_live_inputs(request, root)
