"""Unit fixtures isolate extension binding; actual jobs replay provider archives separately."""
import copy
import json
from contextlib import ExitStack
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.investment import source_extensions as extension
from market_predictor.investment.label_sources import InvestmentSourceReader
from market_predictor.investment.raw_acquisition import QUARANTINE
from market_predictor.investment.stock_identity import StockIdentityUnavailable
from market_predictor.swing.contracts.holding_materialization import SourcePin


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(extension, "guard", lambda *args: None)
    parent = {"schema": "synthetic_original_selection", "audit_sha256": "unchanged", "segments": []}
    artifacts = []
    tax_rows = []
    source_files = {}
    corroborations = {}
    for symbol, identity, role in (("SPY", "benchmark:SPY", "benchmark"),
                                  ("QQQ", "benchmark:QQQ", "benchmark"),
                                  ("TEST", "issuer:test", "stock")):
        artifact = {"security_id": identity, "ticker": symbol, "provider_symbol": symbol, "role": role,
                    "bars_path": "parent.parquet", "bars_sha256": "a" * 64}
        segment = {"first_session": "2024-05-01", "last_session": "2024-05-28", "archive": "parent", "artifact": artifact}
        parent["segments"].append(segment)
        artifacts.append({**artifact, "start_date": "2024-05-29", "end_date": "2025-05-30", "status": "observed",
                          "identity_admission": QUARANTINE if role == "stock" else "captured_only", "archive": "new",
                          "missing_sessions": [], "candidate_requirements": {"candidate_parent_segment": copy.deepcopy(segment)}})
        if role == "benchmark":
            isin = "US" + symbol
            body = tmp_path / (symbol + ".json")
            body.write_text(json.dumps({"data": {"calculation": {"isin": isin, "referenceDate": 1}}}))
            source_files[str(body)] = file_sha256(body)
            tax_rows.append({"symbol": symbol, "year": 2025, "isin": isin, "status": "responses_archived",
                             "request": {"referenceDate": 1}, "sources": {"structured": {"body":
                                 {"path": body.name, "sha256": file_sha256(body)}}}})
            # Only a unit fixture stands in for the separate corporate-action source replay.
            corroborations[symbol] = SimpleNamespace(symbol=symbol, evidence=("fixture_replayed_identity",),
                tax=SimpleNamespace(isin=isin), identity=SimpleNamespace(cusip=symbol))
    pins = {key: {"path": key + ".json", "sha256": "b" * 64} for key in ("source_selection", "roster", "action_query")}
    request = extension.InvestmentRawSourceExtension.model_validate_json(json.dumps({
        "acquisition_request": {"path": "request.json", "sha256": "a" * 64}, "plan_directory": "plan",
        "plan_authority_sha256": "c" * 64, "archive_directory": "new",
        "benchmark_tax_archive": {"path": "tax.json", "sha256": "d" * 64}}))
    verified = extension.VerifiedInvestmentRawExtension(request,
        {"parent_pins": pins, "outcome_end": "2025-05-30", "identity_admission": {"stock": QUARANTINE, "benchmark": "captured_only"},
         "planned_benchmark_symbols": ["QQQ", "SPY"], "completed_artifacts": artifacts}, {"records": tax_rows}, source_files)
    return {"root": tmp_path, "extension": verified, "parent_selection": parent,
            "parent_selection_pin": SourcePin(**pins["source_selection"]), "roster_rows_pin": SourcePin(**pins["roster"]),
            "action_query_pin": SourcePin(**pins["action_query"]), "corroborations": corroborations}


def test_verified_benchmark_extension_preserves_parent_and_quarantines_stocks(scope):
    original = copy.deepcopy(scope["parent_selection"])
    result = extension.compose_investment_raw_extension(**scope)
    assert scope["parent_selection"] == original
    assert result["schema"] == "market_predictor.investment_raw_source_selection"
    assert "audit_sha256" not in result
    assert result["segments"][:3] == original["segments"]
    assert [s["source_admission"] for s in result["segments"][3:]] == [
        "benchmark_identity_verified", "benchmark_identity_verified"]
    assert result["quarantined_stock_artifact_count"] == 1
    assert result["quarantined_stock_observed_artifact_count"] == 1
    assert result["quarantined_stock_security_ids"] == ["issuer:test"]
    assert scope["extension"].acquisition["completed_artifacts"][2]["identity_admission"] == QUARANTINE
    observed = InvestmentSourceReader(scope["root"], result, "e" * 64).read(
        "issuer:test", "TEST", (date(2024, 5, 29),), benchmark=False)
    assert observed.missing_reasons == ("missing_or_cross_segment_raw_binding",)
    assert observed.frame.empty and not observed.evidence


@pytest.mark.parametrize("key", ["source_selection", "roster", "action_query"])
def test_extension_rejects_different_parent_authority(scope, key):
    scope["extension"].acquisition["parent_pins"][key]["sha256"] = "f" * 64
    with pytest.raises(DataReadinessError, match="verified parent"):
        extension.compose_investment_raw_extension(**scope)


def test_extension_requires_replayed_official_identity(scope):
    scope["corroborations"]["SPY"].evidence = ()
    with pytest.raises(DataReadinessError, match="official ticker"):
        extension.compose_investment_raw_extension(**scope)


def test_extension_rejects_terminal_year_identity_conflict(scope):
    scope["extension"].tax_archive["records"][0]["isin"] = "DIFFERENT"
    with pytest.raises(DataReadinessError, match="ISIN differs"):
        extension.compose_investment_raw_extension(**scope)


def test_extension_rejects_tax_source_mutation(scope):
    (scope["root"] / "SPY.json").write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="source changed"):
        extension.compose_investment_raw_extension(**scope)


def test_extension_rejects_overlapping_selection_even_if_bars_differ(scope):
    scope["extension"].acquisition["completed_artifacts"][0]["start_date"] = "2024-05-28"
    with pytest.raises(DataReadinessError, match="overlaps"):
        extension.compose_investment_raw_extension(**scope)


def test_extension_rejects_unreviewed_provider_transition(scope):
    scope["extension"].acquisition["completed_artifacts"][0]["provider_symbol"] = "NEW"
    with pytest.raises(DataReadinessError, match="provider/security"):
        extension.compose_investment_raw_extension(**scope)


def test_extension_cannot_grant_stock_admission(scope):
    scope["extension"].acquisition["completed_artifacts"][2]["identity_admission"] = "verified"
    with pytest.raises(DataReadinessError, match="lost quarantine"):
        extension.compose_investment_raw_extension(**scope)


def test_extension_refuses_partial_or_absent_mandatory_benchmark(scope):
    scope["extension"].acquisition["completed_artifacts"][0]["missing_sessions"] = ["2025-05-30"]
    with pytest.raises(DataReadinessError, match="sessions are incomplete"):
        extension.compose_investment_raw_extension(**scope)
    scope["extension"].acquisition["completed_artifacts"].pop(0)
    with pytest.raises(DataReadinessError, match="mandatory benchmark"):
        extension.compose_investment_raw_extension(**scope)


def test_quarantined_stock_overlap_does_not_block_verified_benchmarks(scope):
    stock = scope["extension"].acquisition["completed_artifacts"][2]
    stock["start_date"] = "2024-05-28"
    result = extension.compose_investment_raw_extension(**scope)
    assert len(result["segments"]) == 5
    assert all(item["artifact"]["role"] == "benchmark" for item in result["segments"][3:])
    assert result["quarantined_stock_artifact_count"] == 1
    assert stock["identity_admission"] == QUARANTINE


def _proof_scope(scope):
    stock = scope["extension"].acquisition["completed_artifacts"][2]
    stock.update(unit_id="synthetic-stock-unit", start_date="2024-05-29", end_date="2024-05-31", ownership_admitted=False)
    stock["candidate_requirements"]["asof"] = "2024-05-31"
    proof = SimpleNamespace(security_id=stock["security_id"], ticker=stock["ticker"])
    request = scope["extension"].request.model_copy(update={"stock_identity": SourcePin(
        path="synthetic-proof-manifest.json", sha256="e" * 64)})
    scope["extension"] = replace(scope["extension"], request=request, stock_proofs=(proof,))
    return stock


def test_positive_stock_proof_adds_only_exact_proven_sessions_and_preserves_capture(scope, monkeypatch):
    stock = _proof_scope(scope)
    monkeypatch.setattr(extension, "replay_stock_identity", lambda *args, **kwargs: SimpleNamespace(
        proven_sessions=(date(2024, 5, 29), date(2024, 5, 30)),
        missing_reasons=("stock_identity_session_unproven:2024-05-31",)))
    original = copy.deepcopy(scope["parent_selection"])
    result = extension.compose_investment_raw_extension(**scope)
    selected = [segment for segment in result["segments"][3:] if segment["artifact"]["role"] == "stock"]
    assert len(selected) == 1
    assert selected[0]["first_session"] == "2024-05-29" and selected[0]["last_session"] == "2024-05-30"
    assert selected[0]["source_admission"] == "stock_identity_reconstructed_research"
    assert result["segments"][:3] == original["segments"]
    assert stock["identity_admission"] == QUARANTINE and stock["ownership_admitted"] is False
    assert result["stock_identity_admitted_artifact_count"] == 1 and result["stock_extensions_quarantined"]


def test_observed_stock_conflicting_with_parent_owner_is_refused(scope, monkeypatch):
    stock = _proof_scope(scope)
    stock["start_date"] = "2024-05-28"
    monkeypatch.setattr(extension, "replay_stock_identity", lambda *args, **kwargs: SimpleNamespace(
        proven_sessions=(date(2024, 5, 28), date(2024, 5, 29)), missing_reasons=()))
    with pytest.raises(DataReadinessError, match="overlaps selected ownership"):
        extension.compose_investment_raw_extension(**scope)


def test_fully_admitted_stock_is_not_counted_as_current_quarantine(scope, monkeypatch):
    stock = _proof_scope(scope)
    monkeypatch.setattr(extension, "replay_stock_identity", lambda *args, **kwargs: SimpleNamespace(
        proven_sessions=(date(2024, 5, 29), date(2024, 5, 30), date(2024, 5, 31)), missing_reasons=()))
    result = extension.compose_investment_raw_extension(**scope)
    assert result["stock_identity_admitted_artifact_count"] == 1
    assert result["quarantined_stock_artifact_count"] == 0
    assert result["quarantined_stock_observed_artifact_count"] == 0
    assert result["quarantined_stock_security_ids"] == []
    assert result["stock_extensions_quarantined"] is False
    assert stock["identity_admission"] == QUARANTINE


def test_unsupported_class_is_local_but_inspected_sources_remain_bound(scope, monkeypatch):
    _proof_scope(scope)
    inspected = scope["root"] / "synthetic-inspected-source.bin"
    inspected.write_bytes(b"explicit synthetic unsupported source")

    def unsupported(*args, **kwargs):
        kwargs["cache"]["source_files"] = {str(inspected): file_sha256(inspected)}
        raise StockIdentityUnavailable("synthetic unsupported class context")

    monkeypatch.setattr(extension, "replay_stock_identity", unsupported)
    result = extension.compose_investment_raw_extension(**scope)
    assert len(result["segments"]) == 5
    assert str(inspected) in scope["extension"].source_files
    assert "stock_identity_unavailable" in result["stock_identity_admission_audit"][0]["missing_reasons"][0]


def test_extension_rejects_unknown_artifact_roles(scope):
    scope["extension"].acquisition["completed_artifacts"][2]["role"] = "unknown"
    with pytest.raises(DataReadinessError, match="artifact role is unsupported"):
        extension.compose_investment_raw_extension(**scope)


def test_custom_runtime_lease_refuses_extension_before_input_reads(scope, monkeypatch):
    root = scope["root"]
    runtime = root / "custom-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime))

    def forbidden(*args, **kwargs):
        pytest.fail("extension read inputs before obtaining the configured lease")

    monkeypatch.setattr(extension, "file_sha256", forbidden)
    with heavy_job_lease("unit custom runtime owner", runtime_dir=runtime):
        with pytest.raises(HeavyJobBusyError):
            extension.verify_investment_raw_extension(root, scope["extension"].request)


def test_custom_runtime_lease_covers_final_extension_binding(scope, monkeypatch):
    root = scope["root"]
    runtime = root / "custom-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime))
    acquisition = root / "acquisition.json"
    acquisition.write_text("{}")
    request = scope["extension"].request.model_copy(update={"acquisition_request":
        SourcePin(path=acquisition.name, sha256=file_sha256(acquisition))})
    monkeypatch.setattr(extension, "verify_investment_raw_acquisition", lambda *args, **kwargs: {"source_files": {}})

    def forbidden(*args, **kwargs):
        pytest.fail("extension binding read inputs while another job held the configured lease")

    monkeypatch.setattr(extension, "recheck_extension_sources", forbidden)
    with ExitStack() as leases:
        def tax_replay(**kwargs):
            # A competing job starts after independent verification, before binding.
            leases.enter_context(heavy_job_lease("unit next job", runtime_dir=runtime))
            return {"source_files": {}}

        monkeypatch.setattr(extension, "verify_ictax_archive", tax_replay)
        with pytest.raises(HeavyJobBusyError):
            extension.verify_investment_raw_extension(root, request)


def test_parent_extra_sector_benchmarks_do_not_expand_verified_acquisition_scope(scope):
    raw = scope["extension"].acquisition
    parent = scope["parent_selection"]["segments"]
    planned = {"SPY", "QQQ", "XLB", "XLC", "XLE", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"}
    for symbol in sorted(planned - {"SPY", "QQQ"}):
        segment = copy.deepcopy(parent[0])
        segment["artifact"].update(security_id="benchmark:" + symbol, ticker=symbol, provider_symbol=symbol)
        parent.append(segment)
        artifact = copy.deepcopy(raw["completed_artifacts"][0])
        artifact.update(security_id="benchmark:" + symbol, ticker=symbol, provider_symbol=symbol,
                        candidate_requirements={"candidate_parent_segment": copy.deepcopy(segment)})
        raw["completed_artifacts"].append(artifact)
        body = scope["root"] / (symbol + ".json")
        isin = "US" + symbol
        body.write_text(json.dumps({"data": {"calculation": {"isin": isin, "referenceDate": 1}}}))
        scope["extension"].source_files[str(body)] = file_sha256(body)
        tax_row = copy.deepcopy(scope["extension"].tax_archive["records"][0])
        tax_row.update(symbol=symbol, isin=isin)
        tax_row["sources"]["structured"]["body"] = {"path": body.name, "sha256": file_sha256(body)}
        scope["extension"].tax_archive["records"].append(tax_row)
        scope["corroborations"][symbol] = SimpleNamespace(symbol=symbol, evidence=("fixture_replayed_identity",),
            tax=SimpleNamespace(isin=isin), identity=SimpleNamespace(cusip=symbol))
    raw["planned_benchmark_symbols"] = sorted(planned)
    for symbol in ("XLF", "XLRE"):
        segment = copy.deepcopy(parent[0])
        segment["artifact"].update(security_id="benchmark:" + symbol, ticker=symbol, provider_symbol=symbol)
        parent.append(segment)
    original = copy.deepcopy(parent)
    result = extension.compose_investment_raw_extension(**scope)
    assert result["segments"][:len(original)] == original
    additions = result["segments"][len(original):]
    assert len(additions) == 11 and {item["artifact"]["ticker"] for item in additions} == planned
    raw["completed_artifacts"] = [item for item in raw["completed_artifacts"] if item["ticker"] != "XLU"]
    with pytest.raises(DataReadinessError, match="complete mandatory benchmark"):
        extension.compose_investment_raw_extension(**scope)
