from __future__ import annotations

import json
from contextlib import contextmanager

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.investment import cohort as population
from market_predictor.research import investment_coverage as coverage
from market_predictor.research import investment_features as artifacts
from market_predictor.swing.contracts.research_cohort import ResearchSecurityExclusion, SwingResearchCohort


def frozen_cohort():
    ids = tuple(f"s{i:03d}" for i in range(631))
    return SwingResearchCohort(
        schema_version="market_predictor.swing_research_cohort",
        scope="retrospective_development_restriction",
        price_basis_status="not_certified_by_cohort",
        combined_daily_inputs_sha256="a" * 64,
        original_security_ids=ids,
        inherited_excluded_security_ids=ids[:44],
        warmup_only_security_ids=(),
        exclusions=(ResearchSecurityExclusion(security_id=ids[44], tickers=("AAA",), reason="unavailable_trading"),),
        maximum_exclusion_bps=1000,
        cap_approval_reference="approved",
        source_files={"source": "a" * 64},
    )


def interval(security, **extra):
    return {
        "security_id": security,
        "ticker": "AAA",
        "effective_from_utc": "2019-07-09T00:00:00Z",
        "effective_to_utc": None,
        "available_at_utc": "2019-07-09T00:00:00Z",
        **extra,
    }


def test_monthly_calendar_has_truncated_last_month_and_xnys_closes():
    sessions = population.monthly_decisions()
    assert len(sessions) == 59
    assert str(sessions[0].date()) == "2019-07-31"
    assert str(sessions[-1].date()) == "2024-05-28"
    assert str(sessions[4].date()) == "2019-11-29"


def test_full_population_sector_scope_exclusions_and_missing_cik_remain_separate(monkeypatch):
    monkeypatch.setattr(population, "monthly_decisions", lambda: (pd.Timestamp("2019-07-31"),))
    members = pd.DataFrame(
        [
            interval("s045", sector="Information Technology", primary_benchmark="XLK"),
            interval("s046", sector="Financials", primary_benchmark="XLF"),
            interval("s047", sector="Health Care", primary_benchmark="XLV"),
            interval("s044", sector="Real Estate", primary_benchmark="XLRE"),
        ]
    )
    identity = pd.DataFrame([interval("s045", sec_cik="1"), interval("s046", sec_cik="2")])
    rows = population.investment_roster(frozen_cohort(), members, identity)
    assert len(rows) == 631
    indexed = {r["security_id"]: r for r in rows}
    assert indexed["s045"]["status"] == "eligible"
    assert indexed["s045"]["cik"] == "0000000001"
    assert indexed["s046"]["status"] == "out_of_scope_sector"
    assert indexed["s047"]["status"] == "identity_unknown"
    assert indexed["s044"]["status"] == "excluded_research_cohort"
    assert "out_of_scope_sector" in indexed["s044"]["qualification_reasons"]
    assert sum("excluded_research_cohort" in r["qualification_reasons"] for r in rows) == 45


def test_effective_and_availability_intervals_cannot_leak_identity(monkeypatch):
    monkeypatch.setattr(population, "monthly_decisions", lambda: (pd.Timestamp("2019-07-31"),))
    members = pd.DataFrame([interval("s045", sector="Information Technology", primary_benchmark="XLK")])
    identity = pd.DataFrame([interval("s045", sec_cik="1", available_at_utc="2019-08-01T00:00:00Z")])
    rows = population.investment_roster(frozen_cohort(), members, identity)
    assert next(r for r in rows if r["security_id"] == "s045")["cik"] is None
    identity = pd.DataFrame([interval("s045", sec_cik="1"), interval("s045", sec_cik="2")])
    with pytest.raises(DataReadinessError, match="ambiguous"):
        population.investment_roster(frozen_cohort(), members, identity)


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return {"path": str(path), "sha256": coverage.file_sha256(path)}


def job_fixture(tmp_path, monkeypatch, ciks=("0000000001", "0000000002")):
    monkeypatch.setattr(coverage, "guard", lambda *args: None)
    monkeypatch.setattr(artifacts, "guard", lambda *args: None)
    monkeypatch.setattr(coverage, "_implementation_pins", lambda: {})
    monkeypatch.setattr(artifacts, "_implementation", lambda: {})
    leases = []

    @contextmanager
    def lease(*args, **kwargs):
        leases.append(args[0])
        yield

    monkeypatch.setattr(coverage, "heavy_job_lease", lease)

    class Index:
        def __init__(self, *args):
            pass

        def get(self, cik):
            return {}

    monkeypatch.setattr(coverage, "RetainedClockIndex", Index)
    dummy = write(tmp_path / "dummy.json", {})
    request = {name: dummy for name in ("cohort", "membership", "identity", "identity_manifest", "acceptance_clock_manifest")}
    request_ref = write(tmp_path / "roster/request.json", request)
    rows = [
        {
            "decision_id": str(i),
            "security_id": f"s{i}",
            "ticker": "AAA",
            "sector": "Information Technology",
            "sector_benchmark": "XLK",
            "decision_session": "2024-05-28",
            "decision_time_utc": "2024-05-28T20:00:00Z",
            "cik": cik,
            "status": "eligible",
            "qualification_reasons": [],
        }
        for i, cik in enumerate(ciks)
    ]
    row_ref = write(tmp_path / "roster/roster.json", {"rows": rows})
    manifest = write(
        tmp_path / "roster/_manifest.json",
        {
            "schema": "market_predictor.investment_coverage_roster",
            "status": "roster_complete",
            "feature_contract_sha256": coverage.FEATURE_CONTRACT_SHA256,
            "source_files": {},
            "files": {"request.json": request_ref["sha256"], "roster.json": row_ref["sha256"]},
        },
    )
    return manifest, leases


def fake_collection(root, cik, security_id, output, settings=None):
    payload = {"cik": int(cik), "facts": {"us-gaap": {}}}
    facts = write(output / "company_facts.json", payload)
    response = write(output / "response.bin", payload)
    write(
        output / "_manifest.json",
        {
            "schema": "market_predictor.investment_company_facts",
            "status": "collected_research_only",
            "cik": cik,
            "security_id": security_id,
            "retrieved_at": "2026-11-07T00:00:00Z",
            "files": {"company_facts.json": facts["sha256"], "response.bin": response["sha256"]},
        },
    )


def recovery_roster(tmp_path, prior, sources):
    original = json.loads((tmp_path / "roster/request.json").read_text())
    request = write(tmp_path / "recovery-roster/request.json", {**original,
        "prior_coverage_manifest": prior, "reused_company_facts": sources})
    rows = write(tmp_path / "recovery-roster/roster.json", json.loads((tmp_path / "roster/roster.json").read_text()))
    manifest = json.loads((tmp_path / "roster/_manifest.json").read_text())
    manifest["files"] = {"request.json": request["sha256"], "roster.json": rows["sha256"]}
    return write(tmp_path / "recovery-roster/_manifest.json", manifest)


def test_recovery_requires_all_retained_issuers_before_any_download(tmp_path, monkeypatch):
    job_fixture(tmp_path, monkeypatch)
    prior = write(tmp_path / "prior.json", {})
    roster = recovery_roster(tmp_path, prior, [])
    monkeypatch.setattr(coverage, "collect_company_facts_unleased", lambda **_: pytest.fail("recovery attempted download"))
    with pytest.raises(DataReadinessError, match="downloads forbidden"):
        coverage.run_investment_coverage(root=tmp_path, roster_manifest=tmp_path / "recovery-roster/_manifest.json",
            roster_manifest_sha256=roster["sha256"], output=tmp_path / "recovery")


def test_recovery_verifies_prior_rows_and_reuses_sources_without_download(tmp_path, monkeypatch):
    manifest, _ = job_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(coverage, "collect_company_facts_unleased", fake_collection)
    coverage.run_investment_coverage(root=tmp_path, roster_manifest=tmp_path / "roster/_manifest.json",
        roster_manifest_sha256=manifest["sha256"], output=tmp_path / "original")
    prior_path = tmp_path / "original/final/_manifest.json"
    prior = {"path": str(prior_path), "sha256": coverage.file_sha256(prior_path)}
    sources = [{"path": str(path), "sha256": coverage.file_sha256(path)} for path in
        sorted((tmp_path / "original/sources").glob("*/_manifest.json"))]
    roster = recovery_roster(tmp_path, prior, sources)
    monkeypatch.setattr(coverage, "collect_company_facts_unleased", lambda **_: pytest.fail("recovery attempted download"))
    coverage.run_investment_coverage(root=tmp_path, roster_manifest=tmp_path / "recovery-roster/_manifest.json",
        roster_manifest_sha256=roster["sha256"], output=tmp_path / "recovery")
    final = json.loads((tmp_path / "recovery/final/_manifest.json").read_text())
    assert final["parity"]["prior_successful_issuers"] == 2
    assert final["parity"]["exact_rows"] == 2 and final["parity"]["changed_rows"] == 0
    assert final["parity"]["missing_new_issuers"] == []
    assert all((path.parent / "rejections.json").exists() for path in (tmp_path / "recovery/issuers").glob("*/_manifest.json"))


def test_recovery_parity_ignores_only_contract_hash_and_exposes_rejection_effects():
    original = [{"decision_id": "d", "security_id": "issuer", "snapshot": {"feature_contract_sha256": "old",
        "metrics": {"revenue": 1.0}, "consumed_fact_sha256": ["fact"], "missing_reasons": {}, "available_at": "clock"}}]
    new = json.loads(json.dumps(original))
    new[0]["snapshot"].update(feature_contract_sha256="new", consumed_rejection_sha256=[])
    assert coverage._snapshot_parity(original, new)["exact_rows"] == 1
    new[0]["snapshot"]["consumed_rejection_sha256"] = ["rejection"]
    new[0]["snapshot"]["metrics"]["revenue"] = None
    parity = coverage._snapshot_parity(original, new)
    assert parity["changed_rows"] == 1
    assert set(parity["row_differences"][0]["differences"]) == {"snapshot.metrics", "snapshot.consumed_rejection_sha256"}


def test_serial_failures_preserve_full_denominator_and_success_checkpoint(tmp_path, monkeypatch):
    manifest, leases = job_fixture(tmp_path, monkeypatch)
    calls = []

    def collect(**kwargs):
        calls.append(kwargs["cik"])
        if kwargs["cik"] == "0000000002":
            raise RuntimeError("issuer HTTP failure")
        fake_collection(**kwargs)

    monkeypatch.setattr(coverage, "collect_company_facts_unleased", collect)
    result = coverage.run_investment_coverage(
        root=tmp_path,
        roster_manifest=tmp_path / "roster/_manifest.json",
        roster_manifest_sha256=manifest["sha256"],
        output=tmp_path / "job",
    )
    report = json.loads((tmp_path / "job/final/_manifest.json").read_text())
    assert result["training_ready"] is False
    assert report["declared_rows"] == 2 and report["completed_issuers"] == 1 and report["failed_issuers"] == 1
    counts = json.loads((tmp_path / "job/final/coverage.json").read_text())["counts"]
    assert sum(c["rows"] for c in counts if c["feature"] == coverage.FEATURE_NAMES[0]) == 2
    assert calls == ["0000000001", "0000000002"] and leases == ["run-investment-coverage"]


def test_memory_failure_stops_without_final_then_resume_reuses_verified_issuer(tmp_path, monkeypatch):
    manifest, _ = job_fixture(tmp_path, monkeypatch)
    calls = []

    def collect(**kwargs):
        calls.append(kwargs["cik"])
        if kwargs["cik"] == "0000000002":
            raise MemoryBudgetError("memory refused")
        fake_collection(**kwargs)

    monkeypatch.setattr(coverage, "collect_company_facts_unleased", collect)
    kwargs = dict(
        root=tmp_path,
        roster_manifest=tmp_path / "roster/_manifest.json",
        roster_manifest_sha256=manifest["sha256"],
        output=tmp_path / "job",
    )
    with pytest.raises(MemoryBudgetError):
        coverage.run_investment_coverage(**kwargs)
    assert not (tmp_path / "job/final").exists()
    monkeypatch.setattr(coverage, "collect_company_facts_unleased", fake_collection)
    coverage.run_investment_coverage(**kwargs)
    assert (tmp_path / "job/final/_manifest.json").exists()
    assert calls == ["0000000001", "0000000002"]


def test_source_mutation_and_changed_resume_identity_refused(tmp_path, monkeypatch):
    manifest, _ = job_fixture(tmp_path, monkeypatch, ciks=())
    write(tmp_path / "job/_job.json", {"job_identity": "changed"})
    with pytest.raises(DataReadinessError, match="resume"):
        coverage.run_investment_coverage(
            root=tmp_path,
            roster_manifest=tmp_path / "roster/_manifest.json",
            roster_manifest_sha256=manifest["sha256"],
            output=tmp_path / "job",
        )


def test_retained_clock_index_verifies_once_and_caches_per_cik(monkeypatch, tmp_path):
    setup, decode = [], []

    def sources(*args):
        setup.append(1)
        return {"1": "utc"}, tmp_path / "archive", tmp_path / "inventory"

    monkeypatch.setattr(artifacts, "_clock_sources", sources)
    monkeypatch.setattr(artifacts, "CONVENTIONS", {"utc"})
    monkeypatch.setattr(artifacts, "_issuer_clocks", lambda *args: decode.append(args[2]) or {})
    index = artifacts.RetainedClockIndex(tmp_path, artifacts.ArtifactReference(path="clock", sha256="a" * 64), {})
    index.get("1")
    index.get("1")
    assert setup == [1] and decode == ["1"]


def test_resume_refuses_mutated_completed_issuer_source_without_final_publication(tmp_path, monkeypatch):
    manifest, _ = job_fixture(tmp_path, monkeypatch)

    def collect(**kwargs):
        if kwargs["cik"] == "0000000002":
            raise MemoryBudgetError("stop after first immutable issuer")
        fake_collection(**kwargs)

    monkeypatch.setattr(coverage, "collect_company_facts_unleased", collect)
    kwargs = dict(
        root=tmp_path,
        roster_manifest=tmp_path / "roster/_manifest.json",
        roster_manifest_sha256=manifest["sha256"],
        output=tmp_path / "job",
    )
    with pytest.raises(MemoryBudgetError):
        coverage.run_investment_coverage(**kwargs)
    (tmp_path / "job/sources/0000000001/company_facts.json").write_text("{}")
    with pytest.raises(DataReadinessError, match="checkpoint source changed"):
        coverage.run_investment_coverage(**kwargs)
    assert not (tmp_path / "job/final").exists()


def test_missing_sector_scalars_never_enter_nonfinancial_eligible_population(monkeypatch):
    monkeypatch.setattr(population, "monthly_decisions", lambda: (pd.Timestamp("2019-07-31"),))
    members = pd.DataFrame(
        [interval("s045", sector=None, primary_benchmark="SPY"), interval("s046", sector=pd.NA, primary_benchmark="SPY")]
    )
    identities = pd.DataFrame([interval("s045", sec_cik="1"), interval("s046", sec_cik="2")])
    rows = population.investment_roster(frozen_cohort(), members, identities)
    selected = [r for r in rows if r["security_id"] in {"s045", "s046"}]
    assert all(r["sector"] is None and r["status"] == "sector_unknown" for r in selected)


def test_roster_can_pin_producer_implementation_outside_data_root(tmp_path, monkeypatch):
    manifest, _ = job_fixture(tmp_path, monkeypatch, ciks=())
    producer = tmp_path.parent / f"{tmp_path.name}-producer.py"
    producer.write_text("producer implementation", encoding="utf-8")
    manifest_path = tmp_path / "roster/_manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["source_files"] = {str(producer): coverage.file_sha256(producer)}
    manifest = write(manifest_path, payload)
    result = coverage.run_investment_coverage(
        root=tmp_path, roster_manifest=manifest_path, roster_manifest_sha256=manifest["sha256"], output=tmp_path / "job"
    )
    assert result["status"] == "coverage_complete"
