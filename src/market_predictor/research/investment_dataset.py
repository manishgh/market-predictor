"""Join immutable investment outcomes with exact lagged saved predictors."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.contracts import InvestmentTargetRequest, InvestmentTargetResult
from market_predictor.investment.dataset import (
    ADMISSION_SCHEMA,
    DATASET_SCHEMA,
    EXPECTED_ROWS,
    RETURN_COLUMNS,
    TIMESTAMPS,
    InvestmentDatasetAdmission,
    InvestmentDatasetRequest,
    add_pins,
    counts_by_horizon,
    feature_contract,
    join_lagged_features,
    model_profiles,
    pin_source_file,
    previous_session,
    read_object,
    recheck_sources,
    require,
    validate_dataset_rows,
)
from market_predictor.investment.fundamentals import FEATURE_CONTRACT_SHA256, FEATURE_NAMES, FundamentalSnapshot
from market_predictor.research.investment_features import FLAGS, _publish
from market_predictor.research.investment_label_reuse import InvestmentCheckpointOrigin
from market_predictor.research.news_decision_features import NEWS_COLUMNS
from market_predictor.swing.contracts.holding_materialization import SourcePin


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {str((package / name).resolve()): file_sha256(package / name) for name in (
        "investment/dataset.py", "research/investment_dataset.py", "investment/contracts.py", "investment/fundamentals.py",
        "research/investment_features.py", "research/investment_label_reuse.py", "canonical/store.py", "canonical/cutoffs.py",
        "evidence/hashing.py", "evidence/io.py", "core/runtime_memory.py", "resources.py")}


def _child(root: Path, parent: SourcePin, name: str, digest: str) -> SourcePin:
    return SourcePin(path=str(inside(root, parent.path).parent / name), sha256=digest)


def _closed(value: dict[str, Any]) -> bool:
    return all(value.get(key) is False for key in FLAGS)


def _roster(root: Path, request: InvestmentDatasetRequest, pins: dict[str, str]) -> list[dict[str, Any]]:
    report = read_object(root, request.roster_manifest, pins)
    require(report.get("schema") == "market_predictor.investment_coverage_roster"
        and report.get("status") == "roster_complete" and _closed(report), "investment dataset requires closed complete roster")
    rows = read_object(root, _child(root, request.roster_manifest, "roster.json", report["files"]["roster.json"]), pins)["rows"]
    require(len(rows) * 2 == EXPECTED_ROWS and report["declared_rows"] == len(rows)
        and len({row["decision_id"] for row in rows}) == len(rows), "investment roster population differs")
    for row in rows:
        require(row["decision_id"] == json_sha256({"security_id": row["security_id"], "decision_session": row["decision_session"]}),
            "investment roster decision identity differs")
    return cast(list[dict[str, Any]], rows)


def _targets(root: Path, request: InvestmentDatasetRequest, roster: list[dict[str, Any]],
    pins: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    report = read_object(root, request.targets_manifest, pins)
    require(report.get("schema") == "market_predictor.investment_label_corpus"
        and report.get("status") == "labels_complete_with_explicit_gaps" and report.get("scope") == "full_roster"
        and report.get("label_rows") == EXPECTED_ROWS and _closed(report), "investment dataset target corpus is incomplete")
    origins = [InvestmentCheckpointOrigin.model_validate_json(json.dumps(value)) for value in report["security_checkpoint_origins"]]
    require(len(origins) == len(report["security_checkpoints"]) and len({item.security_id for item in origins}) == len(origins)
        and [item.checkpoint.model_dump(mode="json") for item in origins] == report["security_checkpoints"],
        "investment target checkpoint origins differ")
    roster_rows = {row["decision_id"]: row for row in roster}
    records = []
    observed: set[tuple[str, int]] = set()
    job_values: dict[str, dict[str, Any]] = {}
    common_values: set[str] = set()
    export_values: dict[str, dict[str, Any]] = {}
    for origin in origins:
        guard("investment dataset target checkpoint")
        checkpoint = read_object(root, origin.checkpoint, pins)
        job_path = str(inside(root, origin.job.path))
        if job_path not in job_values:
            job_values[job_path] = read_object(root, origin.job, pins)
        require(pins[job_path] == origin.job.sha256, "investment target original job has conflicting pins")
        job = job_values[job_path]
        require(job["job_identity"] == origin.job_identity
            and pd.Timestamp(job["verified_at_utc"]) == origin.verified_at_utc,
            "investment target original job or clock differs")
        if origin.common_sources.path not in common_values:
            read_object(root, origin.common_sources, pins)
            common_values.add(origin.common_sources.path)
        require(pins[str(inside(root, origin.common_sources.path))] == origin.common_sources.sha256,
            "investment target original common inventory has conflicting pins")
        original_request = read_object(root, origin.request, pins)
        require(original_request["roster_manifest"] == request.roster_manifest.model_dump(mode="json")
            and origin.request.sha256 == report["request_sha256"], "investment targets bind another frozen roster/request")
        require(origin.producer_export is not None and origin.producer_commit is not None,
            "investment target origin omits immutable producer")
        assert origin.producer_export is not None
        if origin.producer_export.path not in export_values:
            export_values[origin.producer_export.path] = read_object(root, origin.producer_export, pins)
        export = export_values[origin.producer_export.path]
        require(pins[str(inside(root, origin.producer_export.path))] == origin.producer_export.sha256
            and export["commit"] == origin.producer_commit, "investment target producer commit or export pin differs")
        for name, digest in origin.implementation.items():
            path = inside(root, name)
            relative = path.relative_to(inside(root, origin.producer_export.path).parent).as_posix()
            require(export["files"].get(relative) == digest, "investment target producer export binding differs")
            add_pins(root, pins, {str(path): digest})
        require(checkpoint.get("schema") == "market_predictor.investment_security_labels"
            and checkpoint.get("status") == "security_complete" and _closed(checkpoint)
            and checkpoint.get("security_id") == origin.security_id and checkpoint.get("job_identity") == origin.job_identity
            and checkpoint.get("files") == {"labels.json": origin.labels.sha256}, "investment target checkpoint differs")
        local_pins: dict[str, str] = {}
        add_pins(root, local_pins, checkpoint["source_files"])
        require(inside(root, origin.labels.path) == inside(root, origin.checkpoint.path).parent / "labels.json"
            and all(local_pins.get(str(inside(root, pin.path))) == pin.sha256
                for pin in (origin.job, origin.common_sources, origin.request)),
            "investment target checkpoint original source bindings differ")
        rows = read_object(root, origin.labels, pins)["rows"]
        require(checkpoint["declared_rows"] == len(rows), "investment target checkpoint row count differs")
        for row in rows:
            key = (row["decision_id"], row["horizon_sessions"])
            require(key not in observed and key[0] in roster_rows and type(key[1]) is int and key[1] in (63, 252),
                "investment target row is duplicate or escapes the roster")
            observed.add(key)
            decision = roster_rows[key[0]]
            require(row["security_id"] == origin.security_id == decision["security_id"]
                and row["decision_session"] == decision["decision_session"] and row["roster_status"] == decision["status"]
                and pd.Timestamp(row["membership_verified_at_utc"]) == origin.verified_at_utc and _closed(row),
                "investment target row identity, scope or original clock differs")
            verified = row["status"] == "source_validated"
            result = {**decision, "roster_status": decision["status"], "target_status": row["status"],
                "horizon_sessions": key[1], "target_verified": verified, "target_missing_reasons": row["missing_reasons"],
                "entry_time_utc": row["entry_time_utc"], "economic_maturity_at_utc": row["economic_maturity_utc"],
                "label_available_at_utc": row["label_available_at_utc"], "target_request_sha256": row["request_sha256"],
                "target_checkpoint_path": origin.checkpoint.path, "target_checkpoint_sha256": origin.checkpoint.sha256,
                "target_origin_job_identity": origin.job_identity, "target_origin_producer_commit": origin.producer_commit,
                **dict.fromkeys(RETURN_COLUMNS)}
            if verified:
                target_request = InvestmentTargetRequest.model_validate_json(json.dumps(row["request"]))
                target = InvestmentTargetResult.model_validate_json(json.dumps(row["target"]))
                require(decision["status"] == "eligible" and target_request.sha256() == row["request_sha256"] == target.request_sha256
                    and target_request.decision_id == target.decision_id == key[0]
                    and target_request.policy.horizon_sessions == target.horizon_sessions == key[1]
                    and target.policy_sha256 == target_request.policy.sha256()
                    and target_request.decision_session.isoformat() == decision["decision_session"]
                    and target_request.decision_time_utc == pd.Timestamp(decision["decision_time_utc"])
                    and target_request.sector == decision["sector"]
                    and target_request.holdings[3].symbol == decision["sector_benchmark"]
                    and target_request.holdings[0].security_id == decision["security_id"]
                    and target.label_eligible and row["label_eligible"] is True and row["missing_reasons"] == []
                    and target.horizon_end_utc == pd.Timestamp(row["economic_maturity_utc"])
                    and target.entry_time_utc == pd.Timestamp(row["entry_time_utc"])
                    and target.label_available_at_utc == pd.Timestamp(row["label_available_at_utc"])
                    and all((binding.security_id, binding.specification_sha256) == (holding.security_id, holding.specification_sha256)
                        for binding, holding in zip(target_request.holdings, target.holdings, strict=True)),
                    "investment verified target binding or complete four-holding result differs")
                result.update({"target_net_excess_vs_spy": target.net_excess_vs_spy,
                    "stock_net_return": target.holdings[0].net_return, "spy_gross_return": target.holdings[1].gross_return,
                    "qqq_gross_return": target.holdings[2].gross_return, "sector_gross_return": target.holdings[3].gross_return})
            else:
                require(row["status"] == "unresolved" and bool(row["missing_reasons"]) and row["label_eligible"] is False
                    and row["label_available_at_utc"] is None, "investment unresolved target has no reason or claims availability")
            records.append(result)
    expected = {(row["decision_id"], horizon) for row in roster for horizon in (63, 252)}
    require(observed == expected and len(records) == EXPECTED_ROWS, "investment complete target row population differs")
    return pd.DataFrame(records), report


def _financial(root: Path, request: InvestmentDatasetRequest, roster: list[dict[str, Any]],
    pins: dict[str, str]) -> dict[str, dict[str, Any]]:
    report = read_object(root, request.financial_manifest, pins)
    roster_report = read_object(root, request.roster_manifest, pins)
    require(report.get("schema") == "market_predictor.investment_coverage"
        and report.get("status") == "coverage_complete" and _closed(report)
        and report.get("availability_basis") == "reconstructed_research"
        and report["roster_manifest_sha256"] == request.roster_manifest.sha256
        and report["roster_sha256"] == roster_report["files"]["roster.json"]
        and report["feature_contract_sha256"] == FEATURE_CONTRACT_SHA256 and report["feature_names"] == list(FEATURE_NAMES)
        and report["failed_issuers"] == 0, "investment financial coverage source or feature contract differs")
    coverage = read_object(root, _child(root, request.financial_manifest, "coverage.json", report["files"]["coverage.json"]), pins)
    expected = {row["decision_id"]: row for row in roster if row["status"] == "eligible"}
    result: dict[str, dict[str, Any]] = {}
    issuers = coverage["issuer_artifacts"]
    require(len(issuers) == report["completed_issuers"] == report["eligible_issuers"]
        and len({item["cik"] for item in issuers}) == len(issuers) and coverage["failures"] == [],
        "investment financial issuer inventory differs")
    for item in issuers:
        guard("investment dataset financial checkpoint")
        pin = SourcePin(path=item["path"], sha256=item["sha256"])
        checkpoint = read_object(root, pin, pins)
        require(checkpoint.get("schema") == "market_predictor.investment_issuer_coverage"
            and checkpoint.get("status") == "issuer_complete" and checkpoint["cik"] == item["cik"]
            and checkpoint["job_identity"] == report["job_identity"] and _closed(checkpoint),
            "investment financial issuer checkpoint differs")
        rows = read_object(root, _child(root, pin, "features.json", checkpoint["files"]["features.json"]), pins)["rows"]
        require(len(rows) == checkpoint["decision_count"], "investment financial row count differs")
        for row in rows:
            identity = row["decision_id"]
            require(identity in expected and identity not in result, "investment financial decision is duplicate or outside scope")
            expected_row = expected[identity]
            require(all(row[key] == expected_row[key] for key in ("security_id", "cik", "decision_session", "decision_time_utc")),
                "investment financial row roster metadata differs")
            snapshot = FundamentalSnapshot.model_validate_json(json.dumps(row["snapshot"]))
            require(snapshot.feature_contract_sha256 == FEATURE_CONTRACT_SHA256 and snapshot.feature_names == FEATURE_NAMES
                and set(snapshot.metrics) == set(FEATURE_NAMES), "investment financial snapshot columns differ")
            result[identity] = snapshot.model_dump(mode="json")
    require(set(result) == set(expected) and len(result) == report["eligible_rows"],
        "investment financial exact decision population differs")
    return result


def _news(root: Path, request: InvestmentDatasetRequest, pins: dict[str, str]) -> tuple[dict[str, Any], dict[str, str]]:
    """Replay the retained saved-row proof, preserving original producer provenance."""
    report = read_object(root, request.news_verification, pins)
    manifest = read_object(root, request.news_manifest, pins)
    producer_request = read_object(root, _child(root, request.news_manifest, "_request.json", manifest["request_sha256"]), pins)
    checkpoint = read_object(root, _child(root, request.news_manifest, "_checkpoint.json", manifest["checkpoint_sha256"]), pins)
    closed = ("training_eligible", "promotion_eligible", "serving_eligible", "source_coverage_admitted", "world_actor_verified")
    require(manifest.get("schema") == "market_predictor.news_decision_publication"
        and manifest.get("status") == "complete_research_only"
        and report.get("schema") == "market_predictor.news_decision_row_verification" and report.get("status") == "passed"
        and report.get("scope") == "complete_saved_parent_and_news_source_projection"
        and report.get("publication") == request.news_manifest.model_dump(mode="json")
        and report.get("manifest_sha256") == request.news_manifest.sha256
        and report.get("consumer_scope") == "saved_receipt_consumption"
        and report.get("consumer_raw_sources_rechecked") is False
        and report.get("news_additions_source_replayed") is True and report.get("inherited_parent_rows_exact") is True
        and report.get("original_outcome_values_exact") is True and report.get("outcome_filtered_rows") == 0
        and all(value.get(flag) is False for value in (report, manifest, producer_request, checkpoint) for flag in closed),
        "investment news saved-row receipt does not prove the exact closed publication")
    fields = ("rows", "profile", "profile_sha256", "model_columns", "decision_ids_sha256")
    for name in fields:
        require(manifest[name] == producer_request[name] and report[name] == manifest[name],
            "investment news receipt publication contract differs")
    for name in ("source_files", "implementation_files", "source_index_producer_provenance",
        "sentiment_copy_producer_provenance", "sentiment_copy_policy"):
        require(manifest[name] == producer_request[name], "investment news original producer declaration differs")
    require(report["request_sha256"] == manifest["request_sha256"] == checkpoint["request_sha256"]
        and checkpoint["months"] == manifest["months"] and manifest["profile"] == "technical_relationships_news"
        and tuple(manifest["model_columns"][-85:]) == NEWS_COLUMNS
        and report["unique_decisions"] == manifest["rows"]
        and report["historical_implementation_files"] == producer_request["historical_implementation_files"]
        and report["publication_availability_columns"] == manifest["availability_columns"] == producer_request["availability_columns"]
        and report["availability_columns"] == {name: manifest["availability_columns"][name] for name in manifest["model_columns"]},
        "investment news complete publication population differs")
    original_code = report["current_implementation_files"]
    require(len(original_code) == 46 and set(request.news_producer_sources) == set(original_code),
        "investment news producer source mapping is incomplete")
    for logical, pin in request.news_producer_sources.items():
        require(pin.sha256 == original_code[logical] and logical.startswith("src/market_predictor/"),
            "investment news original producer source mapping differs")
        path = inside(root, pin.path)
        require(file_sha256(path) == pin.sha256, "investment news original producer bytes changed")
        add_pins(root, pins, {str(path): pin.sha256})
    require(all(original_code.get(name) == digest for name, digest in producer_request["implementation_files"].items()),
        "investment news original publication implementation is not bound by saved proof")
    parent = None
    for name in ("parent_publication", "parent_saved_row_verification", "original_snapshot_replay"):
        require(report[name] == producer_request[name], "investment news retained parent proof differs")
        value = read_object(root, SourcePin.model_validate_json(json.dumps(report[name])), pins)
        if name == "parent_publication":
            parent = value
    require(parent is not None and parent["model_columns"] == manifest["model_columns"][:124]
        and parent["profile_sha256"] == producer_request["parent_profile_sha256"]
        and parent["rows"] == manifest["rows"], "investment news inherited 586-peer feature context differs")
    config_pin = SourcePin.model_validate_json(json.dumps(producer_request["config"]))
    policy = read_object(root, config_pin, pins)
    require(policy == producer_request["policy"], "investment news original configuration differs")
    read_object(root, SourcePin.model_validate_json(json.dumps(report["configuration_evidence"])), pins)
    proof_sources = {}
    for key in ("source_link_index", "sentiment_copy_evidence"):
        require(policy[key] == report[key], "investment news original source-evidence identity differs")
        proof_sources[key] = read_object(root, SourcePin.model_validate_json(json.dumps(policy[key])), pins)
    linked, copied = proof_sources["source_link_index"], proof_sources["sentiment_copy_evidence"]
    require(linked.get("schema") == "market_predictor.news_source_links"
        and linked.get("status") == "complete_source_links_only" and linked.get("coverage") == "unknown"
        and linked["source_files"] == producer_request["source_index_producer_provenance"]
        and copied.get("schema") == "market_predictor.news_sentiment_copy_evidence"
        and copied.get("status") == "complete_copy_identity_evidence_only" and copied["source_index"] == policy["source_link_index"]
        and all(value.get(flag) is False for value in (linked, copied) for flag in closed)
        and producer_request["sentiment_copy_producer_provenance"] == {"evidence": policy["sentiment_copy_evidence"],
            "provenance_sha256": json_sha256({"build_inputs": copied["source_files"], "upstream": copied["producer_source_provenance"]})},
        "investment news retained source or exact-copy proof differs")
    require(manifest["profile_sha256"] == json_sha256({"profile": manifest["profile"],
        "columns": tuple(manifest["model_columns"]),
        "kernel": producer_request["implementation_files"]["src/market_predictor/research/news_decision_features.py"],
        "parent_profile": producer_request["parent_profile_sha256"], "source_index": policy["source_link_index"],
        "sentiment_copy_evidence": policy["sentiment_copy_evidence"], "sentiment_copy_policy": producer_request["sentiment_copy_policy"],
        "projection": producer_request["implementation_files"]["src/market_predictor/research/news_decision_projection.py"],
        "uncertainty_policy": producer_request["uncertainty_policy"]}), "investment news original profile digest differs")
    for key, pin in report["contract_pins"].items():
        require(key in {"strategy_contract", "research_contract", "temporal_contract"}, "investment news unknown contract pin")
        pin_source_file(root, SourcePin.model_validate_json(json.dumps(pin)), pins)
    months = manifest["months"]
    require(set(report["monthly"]) == set(months) and sum(child["rows"] for child in months.values()) == manifest["rows"],
        "investment news proof monthly inventory differs")
    for month, child in months.items():
        require(report["monthly"][month]["rows"] == child["rows"] and report["monthly"][month]["clock_violations"] == 0
            and child["model_columns"] == manifest["model_columns"] and child["availability_columns"] == manifest["availability_columns"]
            and child["path"] == f"{month}/technical_relationships_news.parquet", "investment news saved month proof differs")
    original_sources: dict[str, str] = {}
    add_pins(root, original_sources, report["source_files"])
    required_sources = {str(inside(root, pin.path)): pin.sha256 for pin in (request.news_manifest, config_pin)}
    required_sources.update({str(source_parent): digest for source_parent, digest in (
        (inside(root, request.news_manifest.path).parent / "_request.json", manifest["request_sha256"]),
        (inside(root, request.news_manifest.path).parent / "_checkpoint.json", manifest["checkpoint_sha256"]))})
    for child in months.values():
        for name, digest in ((child["path"], child["sha256"]), (child["path"] + ".manifest.json", child["manifest_sha256"])):
            required_sources[str(inside(root, request.news_manifest.path).parent / name)] = digest
    require(all(original_sources.get(name) == digest for name, digest in required_sources.items()),
        "investment news saved proof omits publication or source-byte pins")
    return manifest, manifest["availability_columns"]


def build_investment_dataset(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    root, output = root.resolve(), output.resolve()
    require(not output.exists(), "immutable investment dataset output already exists")
    with heavy_job_lease("build-investment-dataset", runtime_dir=root / heavy_job_runtime_dir()):
        guard("investment dataset preparation")
        pins = _implementation()
        request_pin = SourcePin(path=str(inside(root, request)), sha256=request_sha256)
        parsed = InvestmentDatasetRequest.model_validate_json(json.dumps(read_object(root, request_pin, pins)))
        roster = _roster(root, parsed, pins)
        targets, target_report = _targets(root, parsed, roster, pins)
        financial = _financial(root, parsed, roster, pins)
        news, availability = _news(root, parsed, pins)
        profiles = model_profiles(tuple(news["model_columns"]))
        availability = {name: availability[name] for name in news["model_columns"]}
        availability.update(dict.fromkeys(FEATURE_NAMES, "financial_available_at_utc"))
        contract = feature_contract(profiles, availability)
        pieces = []
        source_parent = inside(root, parsed.news_manifest.path).parent
        columns = list(dict.fromkeys(("decision_id", "security_id", "ticker", "session_date_et", "decision_time_utc",
            "feature_eligible", "technical_missing_reasons", *news["model_columns"],
            *(availability[name] for name in news["model_columns"]), *(f"missing_reason_{name}" for name in NEWS_COLUMNS))))
        for month, decisions in pd.DataFrame(roster).groupby(pd.Series([row["decision_session"][:7] for row in roster]), sort=True):
            guard("investment dataset monthly join")
            selected_sessions = {previous_session(datetime.fromisoformat(day).date()) for day in decisions.decision_session}
            needed_months = sorted({day.isoformat()[:7] for day in selected_sessions})
            sources = []
            for source_month in needed_months:
                if source_month not in news["months"]:
                    continue
                child = news["months"][source_month]
                path = inside(source_parent, child["path"])
                require(file_sha256(path) == child["sha256"] and file_sha256(manifest_path_for(path)) == child["manifest_sha256"],
                    "investment news saved source partition changed")
                add_pins(root, pins, {str(path): child["sha256"], str(manifest_path_for(path)): child["manifest_sha256"]})
                source, sidecar = load_canonical_artifact(path, expected_type="swing_news_decision_features",
                    allow_research=True, columns=columns)
                require(sidecar["production_ready"] is False and sidecar["inputs"] == {"request_sha256": news["request_sha256"]}
                    and len(source) == child["rows"] and json_sha256(sorted(source.decision_id)) == child["decision_ids_sha256"]
                    and not source.duplicated(["security_id", "session_date_et"]).any(), "investment news saved row ownership differs")
                sources.append(source.loc[source.session_date_et.isin(selected_sessions)])
            source_rows = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame(columns=columns)
            joined = join_lagged_features(decisions=decisions.to_dict("records"), source=source_rows,
                model_columns=tuple(news["model_columns"]), availability_columns=availability, financial=financial)
            month_targets = targets.loc[targets.decision_session.str[:7].eq(month)].drop(columns=[
                name for name in decisions.columns if name != "decision_id"])
            combined = joined.merge(month_targets, on="decision_id", how="left", validate="one_to_many")
            combined["research_eligible"] = combined.feature_eligible & combined.target_verified
            for name in (*TIMESTAMPS, *set(availability.values())):
                combined[name] = pd.to_datetime(combined[name], utc=True)
            pieces.append(combined)
        frame = pd.concat(pieces, ignore_index=True).sort_values(
            ["decision_session", "security_id", "horizon_sessions"]).reset_index(drop=True)
        admitted_at = datetime.now(UTC)
        require(parsed.evaluation_as_of_utc <= admitted_at
            and pd.to_datetime(frame.loc[frame.target_verified, "label_available_at_utc"], utc=True).le(
                parsed.evaluation_as_of_utc).all(), "investment dataset labels are not available at the requested evaluation cutoff")
        validate_dataset_rows(frame, profiles, availability, admitted_at=admitted_at)
        counts = counts_by_horizon(frame)
        report: dict[str, Any] = {"schema": DATASET_SCHEMA, "status": "complete_research_dataset", "scope": "retrospective_research",
            "request": request_pin.model_dump(mode="json"), "model_columns": profiles, "availability_columns": availability,
            "feature_contract_sha256": contract, "rows": len(frame), "counts_by_horizon": counts, "months": {},
            "source_files": pins, "feature_lag": parsed.feature_lag, "historical_first_seen_proven": False,
            "source_consumption_scope": "saved_verified_artifacts_and_exact_producer_bytes",
            "raw_sources_rechecked": False, "source_admission_changed": False,
            "producer_source_provenance": {"targets": target_report["source_files"], "news": news["source_files"]}, **FLAGS}

        def save(directory: Path) -> None:
            write_json_object(directory / "request.json", parsed.model_dump(mode="json", by_alias=True))
            (directory / "months").mkdir()
            for month, partition in frame.groupby(frame.decision_session.map(str).str[:7], sort=True):
                guard("investment dataset monthly publication")
                path = directory / "months" / f"{month}.parquet"
                partition.to_parquet(path, index=False)
                report["months"][month] = {"path": f"months/{month}.parquet", "sha256": file_sha256(path), "rows": len(partition),
                    "row_keys_sha256": json_sha256(sorted(zip(partition.decision_id, partition.horizon_sessions, strict=True)))}
            recheck_sources(root, pins)

        _publish(output / "dataset", report, pins, save)
        dataset_pin = SourcePin(path=str(output / "dataset/_manifest.json"), sha256=file_sha256(output / "dataset/_manifest.json"))
        admission = InvestmentDatasetAdmission(schema=ADMISSION_SCHEMA, status="admitted_for_retrospective_research",
            scope="retrospective_research", dataset=dataset_pin,
            request=request_pin, feature_contract_sha256=contract, model_columns=profiles, admitted_at_utc=admitted_at,
            research_training_allowed=True, source_consumption_scope="saved_verified_artifacts_and_exact_producer_bytes",
            raw_sources_rechecked=False, source_admission_changed=False, historical_first_seen_proven=False, promotion_eligible=False,
            production_eligible=False, counts_by_horizon=counts, source_files=pins)
        admission_path = output / "admission"
        _publish(admission_path, admission.model_dump(mode="json", by_alias=True), {**pins, dataset_pin.path: dataset_pin.sha256})
        return {"status": "investment_research_dataset_admitted", "dataset": dataset_pin.model_dump(mode="json"),
            "admission": {"path": str(admission_path / "_manifest.json"), "sha256": file_sha256(admission_path / "_manifest.json")},
            "counts_by_horizon": counts, **FLAGS}
