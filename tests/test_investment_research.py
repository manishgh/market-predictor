"""Synthetic unit tests only; not real-data source or model admission evidence."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.cli_surface import command_names
from market_predictor.collection_cli import app as collection_app
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import investment_features as jobs
from market_predictor.research_cli import app as research_app
from market_predictor.sources.sec import SecSource


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(jobs, "guard", lambda stage: None)
    payload = {"cik": 1, "facts": {"us-gaap": {}}}
    body = json.dumps(payload).encode()
    response = SimpleNamespace(retrieved_at_utc=datetime(2025, 6, 1, tzinfo=UTC),
        final_url="https://data.sec.gov/unit-test", body=body, body_sha256=hashlib.sha256(body).hexdigest())
    monkeypatch.setattr(jobs, "SecSource", lambda settings: SimpleNamespace(fetch_cik_company_facts=lambda cik: (payload, response)))
    output = tmp_path / "source"
    jobs.collect_investment_company_facts(root=tmp_path, cik="1", security_id="security:one", output=output)
    return output


def _request(root: Path, source: Path) -> Path:
    request = root / "request.json"
    request.write_text(json.dumps({
        "company_facts_manifest": {"path": "source/_manifest.json", "sha256": file_sha256(source / "_manifest.json")},
        "acceptance_clock_manifest": {"path": "clock.json", "sha256": "a" * 64},
        "decisions": [{"security_id": "security:one", "cik": "0000000001", "decision_time": "2025-06-02T22:00:00Z"}],
        "scope": "nonfinancial", "availability_basis": "observed"}), encoding="utf-8")
    return request


def test_missing_financials_publish_explicit_missing_values(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path, source)
    monkeypatch.setattr(jobs, "_retained_clocks", lambda *args: {})
    output = tmp_path / "features"
    result = jobs.build_investment_features(root=tmp_path, request=request, request_sha256=file_sha256(request), output=output)
    assert result["training_ready"] is False
    manifest = json.loads((output / "_manifest.json").read_text())
    assert manifest["fact_count"] == 0
    assert not any(manifest["available_counts"].values())
    snapshot = json.loads((output / "features.json").read_text())["snapshots"][0]
    assert all(value is None for value in snapshot["metrics"].values())
    assert snapshot["missing_reasons"]
    with pytest.raises(FileExistsError):
        jobs.build_investment_features(root=tmp_path, request=request, request_sha256=file_sha256(request), output=output)


def test_tampered_source_and_request_publish_nothing(source: Path, tmp_path: Path) -> None:
    request = _request(tmp_path, source)
    output = tmp_path / "features"
    with pytest.raises(DataReadinessError, match="hash differs"):
        jobs.build_investment_features(root=tmp_path, request=request, request_sha256="a" * 64, output=output)
    (source / "company_facts.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="hash differs"):
        jobs.build_investment_features(root=tmp_path, request=request, request_sha256=file_sha256(request), output=output)
    assert not output.exists()


def test_publication_rechecks_source_bytes(source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    request = _request(tmp_path, source)

    def mutate(*args: object) -> dict[str, datetime]:
        (source / "response.bin").write_bytes(b"changed")
        return {}

    monkeypatch.setattr(jobs, "_retained_clocks", mutate)
    with pytest.raises(DataReadinessError, match="changed before publication"):
        jobs.build_investment_features(root=tmp_path, request=request, request_sha256=file_sha256(request), output=tmp_path / "features")
    assert not (tmp_path / "features").exists()


def test_memory_and_lease_refusal_precede_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with heavy_job_lease("unit owner", runtime_dir=tmp_path / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            jobs.build_investment_features(root=tmp_path, request=Path("absent"), request_sha256="a" * 64, output=tmp_path / "out")

    def fail(stage: str) -> None:
        raise MemoryBudgetError("unit pressure")

    monkeypatch.setattr(jobs, "guard", fail)
    with pytest.raises(MemoryBudgetError):
        jobs.build_investment_features(root=tmp_path, request=Path("absent"), request_sha256="a" * 64, output=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_inventory_expansion_budget_precedes_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "inventory.parquet"
    pq.write_table(pa.table({"sec_cik": ["0000000001"] * (jobs.MAX_ROWS + 1)}), path)
    monkeypatch.setattr(pq, "read_table", lambda *args, **kwargs: pytest.fail("must not materialize over-budget inventory"))
    with pytest.raises(DataReadinessError, match="expanded row or byte budget"):
        jobs._bounded_inventory(path, "0000000001")


def test_inventory_expanded_bytes_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "inventory.parquet"
    pq.write_table(pa.table({"sec_cik": ["0000000001"]}), path)
    monkeypatch.setattr(jobs, "MAX_EXPANDED_BYTES", 1)
    with pytest.raises(DataReadinessError, match="expanded row or byte budget"):
        jobs._bounded_inventory(path, "0000000001")


def test_command_surfaces_and_busy_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.commands import investment_research as commands

    assert "collect-investment-company-facts" in command_names(collection_app)
    assert "collect-investment-company-facts" not in command_names(research_app)
    for name in ("build-investment-features", "evaluate-investment-predictions"):
        assert name in command_names(research_app)
        assert name not in command_names(collection_app)

    def busy(**kwargs: object) -> dict[str, object]:
        raise HeavyJobBusyError("unit owner")

    monkeypatch.setattr(commands, "build_investment_features", busy)
    result = CliRunner().invoke(research_app, ["build-investment-features", "--root", ".", "--request", "request.json",
        "--request-sha256", "a" * 64, "--output", "out"])
    assert result.exit_code == 75


def test_sec_company_facts_identity_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    source = object.__new__(SecSource)
    monkeypatch.setattr(source, "_get_archived_json", lambda url: ({"cik": 1, "facts": {}}, None))
    payload, _ = source.fetch_cik_company_facts("1")
    assert payload["cik"] == 1
    for cik in ("../1", "", "12345678901", "\u0661"):
        with pytest.raises(ValueError, match="ASCII digits"):
            source.fetch_cik_company_facts(cik)
    with pytest.raises(ValueError, match="differs"):
        source.fetch_cik_company_facts("2")
