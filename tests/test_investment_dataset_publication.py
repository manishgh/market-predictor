"""Synthetic pinned fixtures through dataset publication, admission and loading."""
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment import dataset as data
from market_predictor.investment.contracts import InvestmentTargetRequest, InvestmentTargetResult
from market_predictor.investment.fundamentals import FEATURE_CONTRACT_SHA256, FEATURE_NAMES, FundamentalSnapshot
from market_predictor.research import investment_dataset as publisher
from market_predictor.research import investment_features
from market_predictor.research.investment_features import FLAGS
from market_predictor.research.news_decision_features import NEWS_COLUMNS
from market_predictor.swing.contracts.holding_materialization import SourcePin
from tests.test_investment_dataset import CLOCKS, COLUMNS, decision, source_frame, source_row


def write(path: Path, value: Any) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str), encoding="utf-8")
    return {"path": str(path), "sha256": file_sha256(path)}


def bytes_pin(path: Path, content: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "sha256": file_sha256(path)}


def fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(data, "EXPECTED_ROWS", 6)
    monkeypatch.setattr(publisher, "EXPECTED_ROWS", 6)
    monkeypatch.setattr(data, "guard", lambda *_: None)
    monkeypatch.setattr(publisher, "guard", lambda *_: None)
    monkeypatch.setattr(investment_features, "guard", lambda *_: None)
    monkeypatch.setattr(publisher, "heavy_job_lease", lambda *_, **__: nullcontext())
    rows = [decision(), decision("issuer:unknown"), decision("issuer:outside", status="excluded_research_cohort")]
    rows[1]["ticker"], rows[2]["ticker"] = "UNK", "OUT"
    rows[1]["cik"] = "0000000002"
    roster_rows = write(tmp_path / "roster/roster.json", {"rows": rows})
    roster_pin = write(tmp_path / "roster/_manifest.json", {"schema": "market_predictor.investment_coverage_roster",
        "status": "roster_complete", "declared_rows": 3, "files": {"roster.json": roster_rows["sha256"]}, **FLAGS})
    target_request_pin = write(tmp_path / "target-request.json", {"roster_manifest": roster_pin})
    job = write(tmp_path / "targets/_job.json", {"job_identity": "d" * 64, "verified_at_utc": "2026-01-01T00:00:00+00:00"})
    common = write(tmp_path / "targets/_verified_sources.json", {})
    code = bytes_pin(tmp_path / "producer/src/market_predictor/investment/contracts.py", "immutable original producer")
    export = write(tmp_path / "producer/_export.json", {"commit": "a" * 40,
        "files": {"src/market_predictor/investment/contracts.py": code["sha256"]}})
    prices = write(tmp_path / "prices.json", {"fixture": "already validated original source"})
    checkpoints, origins = [], []
    calendar = xcals.get_calendar("XNYS")
    for row in rows:
        records = []
        for horizon in (63, 252):
            sessions = calendar.sessions_window(pd.Timestamp(row["decision_session"]), horizon + 1)[1:]
            entry, end = calendar.session_open(sessions[0]), calendar.session_close(sessions[-1])
            record = {"decision_id": row["decision_id"], "security_id": row["security_id"],
                "decision_session": row["decision_session"], "horizon_sessions": horizon,
                "economic_maturity_utc": end.isoformat(), "entry_time_utc": entry.isoformat(),
                "roster_status": row["status"], "membership_verified_at_utc": "2026-01-01T00:00:00+00:00",
                "label_available_at_utc": None, "request_sha256": None, "request": None, "target": None,
                "components": [], "status": "unresolved", "label_eligible": False,
                "missing_reasons": ["missing_currency" if row["status"] == "eligible" else row["status"]], **FLAGS}
            if row["security_id"] == "issuer:one":
                source = {"reference": prices["path"], "artifact_sha256": prices["sha256"], "interpretation_policy_sha256": "f" * 64}
                request = InvestmentTargetRequest.model_validate_json(json.dumps({"policy": {"horizon_sessions": horizon},
                    "decision_id": row["decision_id"], "decision_session": row["decision_session"],
                    "decision_time_utc": row["decision_time_utc"], "sector": row["sector"], "sector_membership_source": source,
                    "holdings": [{"role": role, "security_id": identity, "symbol": symbol,
                        "specification_sha256": json_sha256(role), "sources": [source]}
                        for role, identity, symbol in [("stock", row["security_id"], "ONE"),
                            ("spy", "benchmark:SPY", "SPY"), ("qqq", "benchmark:QQQ", "QQQ"), ("sector", "benchmark:XLK", "XLK")]]}))
                target = InvestmentTargetResult.model_validate_json(json.dumps({"request_sha256": request.sha256(),
                    "policy_sha256": request.policy.sha256(), "decision_id": row["decision_id"], "horizon_sessions": horizon,
                    "entry_time_utc": entry.isoformat(), "horizon_end_utc": end.isoformat(),
                    "holdings": [{"role": binding.role, "security_id": binding.security_id,
                        "specification_sha256": binding.specification_sha256, "gross_return": gross, "net_return": gross - 0.002,
                        "label_available_at_utc": "2026-01-02T00:00:00Z", "missing_reasons": []}
                        for binding, gross in zip(request.holdings, (0.102, 0.05, 0.07, 0.06), strict=True)],
                    "net_excess_vs_spy": 0.05, "net_excess_vs_qqq": 0.03, "net_excess_vs_sector": 0.04,
                    "label_available_at_utc": "2026-01-02T00:00:00Z", "label_eligible": True,
                    "status": "available", "missing_reasons": []}))
                record.update({"request": request.model_dump(mode="json"), "request_sha256": request.sha256(),
                    "target": target.model_dump(mode="json"), "label_available_at_utc": "2026-01-02T00:00:00Z",
                    "status": "source_validated", "label_eligible": True, "missing_reasons": []})
            records.append(record)
        directory = tmp_path / "targets/securities" / json_sha256(row["security_id"])
        labels = write(directory / "labels.json", {"rows": records})
        checkpoint = write(directory / "_manifest.json", {"schema": "market_predictor.investment_security_labels",
            "status": "security_complete", "security_id": row["security_id"], "job_identity": "d" * 64,
            "declared_rows": 2, "files": {"labels.json": labels["sha256"]},
            "source_files": {pin["path"]: pin["sha256"] for pin in (job, common, target_request_pin)}, **FLAGS})
        checkpoints.append(checkpoint)
        origins.append({"origin": "new", "security_id": row["security_id"], "checkpoint": checkpoint, "labels": labels,
            "request": target_request_pin, "job": job, "job_identity": "d" * 64, "verified_at_utc": "2026-01-01T00:00:00Z",
            "common_sources": common, "implementation": {code["path"]: code["sha256"]}, "producer_export": export,
            "producer_commit": "a" * 40})
    targets = write(tmp_path / "targets/final/_manifest.json", {"schema": "market_predictor.investment_label_corpus",
        "status": "labels_complete_with_explicit_gaps", "scope": "full_roster", "label_rows": 6,
        "request_sha256": target_request_pin["sha256"], "security_checkpoints": checkpoints,
        "security_checkpoint_origins": origins, "source_files": {"producer-raw-provenance-not-reopened": "b" * 64}, **FLAGS})
    issuers = []
    for row in rows[:2]:
        metrics = dict.fromkeys(FEATURE_NAMES)
        if row["security_id"] == "issuer:one":
            metrics[FEATURE_NAMES[0]] = 0.123456789123456
        snapshot = FundamentalSnapshot(security_id=row["security_id"], cik=row["cik"],
            decision_time=pd.Timestamp(row["decision_time_utc"]).to_pydatetime(), metrics=metrics,
            missing_reasons={name: "original_missing_fact" for name in FEATURE_NAMES if metrics[name] is None},
            consumed_fact_sha256=("b" * 64,), available_at=pd.Timestamp("2024-05-01T20:00:00Z").to_pydatetime())
        features = write(tmp_path / f"financial/issuers/{row['cik']}/features.json",
            {"rows": [{**row, "snapshot": snapshot.model_dump(mode="json")}]})
        issuer = write(Path(features["path"]).parent / "_manifest.json", {"schema": "market_predictor.investment_issuer_coverage",
            "status": "issuer_complete", "cik": row["cik"], "job_identity": "e" * 64, "decision_count": 1,
            "files": {"features.json": features["sha256"]}, **FLAGS})
        issuers.append({"cik": row["cik"], **issuer})
    coverage = write(tmp_path / "financial/final/coverage.json", {"issuer_artifacts": issuers, "failures": []})
    financial = write(tmp_path / "financial/final/_manifest.json", {"schema": "market_predictor.investment_coverage",
        "status": "coverage_complete", "availability_basis": "reconstructed_research", "job_identity": "e" * 64,
        "roster_manifest_sha256": roster_pin["sha256"], "roster_sha256": roster_rows["sha256"],
        "feature_contract_sha256": FEATURE_CONTRACT_SHA256, "feature_names": list(FEATURE_NAMES), "failed_issuers": 0,
        "completed_issuers": 2, "eligible_issuers": 2, "eligible_rows": 2, "files": {"coverage.json": coverage["sha256"]}, **FLAGS})
    news = news_fixture(tmp_path)
    request = {"schema": "market_predictor.investment_dataset_request", "scope": "retrospective_research",
        "targets_manifest": targets, "roster_manifest": roster_pin, "financial_manifest": financial,
        "news_manifest": news["manifest"], "news_verification": news["receipt"], "news_producer_sources": news["code"],
        "feature_lag": "previous_xnys_session_nightly", "primary_horizon": 252, "diagnostic_horizon": 63,
        "missingness": "train_partition_median_with_indicators", "evaluation_as_of_utc": datetime.now(UTC).isoformat()}
    request_pin = write(tmp_path / "request.json", request)
    return {"request": request, "request_pin": request_pin, "news": news, "roster": rows, "output": tmp_path / "output"}


def news_fixture(root: Path) -> dict[str, Any]:
    closed = dict.fromkeys(("training_eligible", "promotion_eligible", "serving_eligible",
        "source_coverage_admitted", "world_actor_verified"), False)
    code = {}
    for name in ["news_decision_features", "news_decision_projection", *(f"historical_reader_{i}" for i in range(44))]:
        code[f"src/market_predictor/research/{name}.py"] = bytes_pin(root / f"news/producer/{name}.py", f"original {name}")
    original_code = {name: pin["sha256"] for name, pin in code.items()}
    link = write(root / "news/link.json", {"schema": "market_predictor.news_source_links", "status": "complete_source_links_only",
        "coverage": "unknown", "source_files": {}, **closed})
    copy = write(root / "news/copy.json", {"schema": "market_predictor.news_sentiment_copy_evidence",
        "status": "complete_copy_identity_evidence_only", "source_index": link, "source_files": {},
        "producer_source_provenance": {}, **closed})
    policy = {"schema": "market_predictor.news_decision_publication_config", "source_link_index": link, "sentiment_copy_evidence": copy}
    config = write(root / "news/config.json", policy)
    parent = write(root / "news/parent.json", {"model_columns": list(COLUMNS[:124]), "profile_sha256": "c" * 64, "rows": 3})
    saved = write(root / "news/parent-proof.json", {"status": "passed"})
    replay = write(root / "news/original-replay.json", {"status": "passed"})
    contract_pins = {name: bytes_pin(root / f"news/{name}.toml", '[policy]\nname="original"\n')
        for name in ("strategy_contract", "research_contract", "temporal_contract")}
    config_evidence = write(root / "news/configuration-evidence.json", {"contract_pins": contract_pins})
    source = source_frame([source_row(), source_row("issuer:unknown"), source_row("issuer:outside")])
    source = pd.concat([source, pd.DataFrame(dict.fromkeys(f"missing_reason_{name}" for name in NEWS_COLUMNS),
        index=source.index)], axis=1)
    source.loc[1, list(NEWS_COLUMNS)] = float("nan")
    ids_sha = json_sha256(sorted(source.decision_id))
    copy_provenance = {"evidence": copy, "provenance_sha256": json_sha256({"build_inputs": {}, "upstream": {}})}
    original = {"schema": "market_predictor.news_decision_publication_request", "rows": 3,
        "profile": "technical_relationships_news", "model_columns": list(COLUMNS), "availability_columns": CLOCKS,
        "decision_ids_sha256": ids_sha, "config": config, "policy": policy, "source_files": {},
        "implementation_files": original_code, "historical_implementation_files": {}, "parent_profile_sha256": "c" * 64,
        "parent_publication": parent, "parent_saved_row_verification": saved, "original_snapshot_replay": replay,
        "source_index_producer_provenance": {}, "sentiment_copy_producer_provenance": copy_provenance,
        "sentiment_copy_policy": {"policy": "original"}, "uncertainty_policy": "original", **closed}
    original["profile_sha256"] = json_sha256({"profile": original["profile"], "columns": COLUMNS,
        "kernel": original_code["src/market_predictor/research/news_decision_features.py"], "parent_profile": "c" * 64,
        "source_index": link, "sentiment_copy_evidence": copy, "sentiment_copy_policy": original["sentiment_copy_policy"],
        "projection": original_code["src/market_predictor/research/news_decision_projection.py"], "uncertainty_policy": "original"})
    request = write(root / "news/publication/_request.json", original)
    path = root / "news/publication/2024-05/technical_relationships_news.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    source.to_parquet(path, index=False)
    sidecar = write(manifest_path_for(path), {"schema": "market_data.artifact_manifest", "canonical_schema_version": "market_data",
        "artifact_type": "swing_news_decision_features", "artifact_sha256": file_sha256(path), "rows": 3,
        "columns": list(source), "production_ready": False, "inputs": {"request_sha256": request["sha256"]}})
    months = {"2024-05": {"path": "2024-05/technical_relationships_news.parquet", "sha256": file_sha256(path),
        "manifest_sha256": sidecar["sha256"], "rows": 3, "decision_ids_sha256": ids_sha,
        "model_columns": list(COLUMNS), "availability_columns": CLOCKS}}
    checkpoint = write(root / "news/publication/_checkpoint.json", {"request_sha256": request["sha256"], "months": months, **closed})
    manifest = write(root / "news/publication/_manifest.json", {**original, "schema": "market_predictor.news_decision_publication",
        "status": "complete_research_only", "request_sha256": request["sha256"],
        "checkpoint_sha256": checkpoint["sha256"], "months": months})
    consumed = {pin["path"]: pin["sha256"] for pin in (manifest, request, checkpoint, config, sidecar)}
    consumed[str(path)] = file_sha256(path)
    receipt_value = {"schema": "market_predictor.news_decision_row_verification", "status": "passed",
        "scope": "complete_saved_parent_and_news_source_projection", "publication": manifest, "manifest_sha256": manifest["sha256"],
        "request_sha256": request["sha256"], "consumer_scope": "saved_receipt_consumption", "consumer_raw_sources_rechecked": False,
        "news_additions_source_replayed": True, "inherited_parent_rows_exact": True, "original_outcome_values_exact": True,
        "outcome_filtered_rows": 0, "unique_decisions": 3, "current_implementation_files": original_code,
        "source_link_index": link, "sentiment_copy_evidence": copy, "configuration_evidence": config_evidence,
        "contract_pins": contract_pins, "source_files": consumed, "publication_availability_columns": CLOCKS,
        "monthly": {"2024-05": {"rows": 3, "clock_violations": 0}},
        **{name: original[name] for name in ("rows", "profile", "profile_sha256", "model_columns", "decision_ids_sha256",
            "availability_columns", "historical_implementation_files", "parent_publication",
            "parent_saved_row_verification", "original_snapshot_replay")},
        **closed}
    receipt = write(root / "news/verification.json", receipt_value)
    return {"manifest": manifest, "receipt": receipt, "receipt_value": receipt_value, "code": code, "parquet": path,
        "source": source, "sidecar": sidecar}


def build(root: Path, value: dict[str, Any]) -> dict[str, Any]:
    return publisher.build_investment_dataset(root=root, request=Path(value["request_pin"]["path"]),
        request_sha256=value["request_pin"]["sha256"], output=value["output"])


def test_complete_publication_admission_loader_and_unknown_forecast_population(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value = fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = build(tmp_path, value)
    dataset = data.load_investment_dataset(root=tmp_path,
        manifest=SourcePin(**result["dataset"]), admission=SourcePin(**result["admission"]))
    assert len(dataset.features) == len(dataset.metadata) == 6
    assert len(dataset.model_columns["all_features"]) == 215 and len(dataset.model_columns["technical_relationships"]) == 124
    assert dataset.metadata.feature_eligible.sum() == 4 and dataset.metadata.research_eligible.sum() == 2
    unknown = dataset.metadata.security_id.eq("issuer:unknown")
    assert dataset.metadata.loc[unknown, "feature_eligible"].all()
    assert not dataset.metadata.loc[unknown, "research_eligible"].any()
    assert dataset.metadata.loc[unknown, "target_net_excess_vs_spy"].isna().all()
    assert dataset.features.loc[unknown, list(NEWS_COLUMNS)].isna().all().all()
    assert (dataset.metadata.loc[~unknown & dataset.metadata.feature_eligible, "target_net_excess_vs_spy"] == 0.05).all()
    assert dataset.metadata.source_decision_id.str.startswith("original-swing").all()
    assert all(Path(path).read_bytes() == content for path, content in before.items())
    assert not (tmp_path / "producer-raw-provenance-not-reopened").exists()
    assert json.loads(Path(result["admission"]["path"]).read_text())["research_training_allowed"] is True
    assert str(tmp_path / "news/strategy_contract.toml") in dataset.source_files


@pytest.mark.parametrize("poison", ["source_bytes", "producer_bytes", "producer_mapping", "false_news_receipt", "past_evaluation"])
def test_pinned_source_consumer_poison_refuses_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str) -> None:
    value = fixture(tmp_path, monkeypatch)
    if poison == "source_bytes":
        value["news"]["parquet"].write_bytes(b"changed parquet")
    elif poison == "producer_bytes":
        Path(next(iter(value["news"]["code"].values()))["path"]).write_text("changed original implementation")
    elif poison == "producer_mapping":
        value["request"]["news_producer_sources"].pop(next(iter(value["news"]["code"])))
    elif poison == "false_news_receipt":
        value["news"]["receipt_value"]["news_additions_source_replayed"] = False
        value["request"]["news_verification"] = write(Path(value["news"]["receipt"]["path"]), value["news"]["receipt_value"])
    elif poison == "past_evaluation":
        value["request"]["evaluation_as_of_utc"] = "2025-12-31T00:00:00Z"
    value["request_pin"] = write(Path(value["request_pin"]["path"]), value["request"])
    with pytest.raises(DataReadinessError):
        build(tmp_path, value)
    assert not (value["output"] / "admission").exists()


@pytest.mark.parametrize("poison", ["wrong_dataset", "wrong_contract", "wrong_sources", "wrong_counts", "changed_partition"])
def test_loader_rejects_wrong_admission_or_changed_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str) -> None:
    value = fixture(tmp_path, monkeypatch)
    result = build(tmp_path, value)
    path = Path(result["admission"]["path"])
    receipt = json.loads(path.read_text())
    if poison == "wrong_dataset":
        receipt["dataset"]["sha256"] = "f" * 64
    elif poison == "wrong_contract":
        receipt["feature_contract_sha256"] = "f" * 64
    elif poison == "wrong_sources":
        receipt["source_files"].pop(next(iter(receipt["source_files"])))
    elif poison == "wrong_counts":
        receipt["counts_by_horizon"]["63"]["research_eligible_rows"] += 1
    elif poison == "changed_partition":
        next((value["output"] / "dataset/months").glob("*.parquet")).write_bytes(b"tampered dataset")
    pin = write(path, receipt)
    with pytest.raises(DataReadinessError):
        data.load_investment_dataset(root=tmp_path, manifest=SourcePin(**result["dataset"]), admission=SourcePin(**pin))
