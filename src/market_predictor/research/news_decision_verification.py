"""Independent monthly news projection replay; no model or source admission."""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin as NewsSourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import news_decision_publication as publication_owner
from market_predictor.research import news_parent_inputs as parent_owner
from market_predictor.research import news_runtime_memory as runtime
from market_predictor.research.news_decision_features import NEWS_COLUMNS
from market_predictor.research.news_source_links import open_news_source_link_index
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.saved_evaluation_configuration import SavedEvaluationConfigurationEvidence
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.datasets.return_relationship_integrity import pins, read_object
from market_predictor.swing.datasets.saved_evaluation_configuration import STRATEGY_RENAMES, TEMPORAL_RENAMES, _configuration
from market_predictor.swing.features.research_partition import ResearchFeaturePartition

SCHEMA = "market_predictor.news_decision_row_verification"
SCOPE = "complete_saved_parent_and_news_source_projection"
PROFILE = publication_owner.PROFILE
ARTIFACT_TYPE = publication_owner.ARTIFACT_TYPE
IMPLEMENTATION_PATHS = (
    "research/news_decision_verification.py", "research/news_runtime_memory.py",
    "swing/datasets/return_relationship_integrity.py", "swing/contracts/holding_materialization.py",
    "swing/contracts/saved_evaluation_configuration.py", "swing/contracts/return_training.py",
    "swing/contracts/training_readiness.py", "swing/datasets/saved_evaluation_configuration.py",
)
RELATIONSHIP_RUN = SourcePin(
    path="data/research/swing_relationship_return_models_initial_fit/_manifest.json",
    sha256="7a699020920024ed29b5d8547724f5355fdc2202ca176f11d38f23418bd4666e",
)


@dataclass(frozen=True)
class VerifiedNewsDecisionPublication:
    path: Path
    manifest: dict[str, Any]
    request: dict[str, Any]
    source_files: dict[str, str]
    implementation_files: dict[str, str]
    model_columns: tuple[str, ...]
    availability_columns: dict[str, str]
    months: dict[str, Any]
    receipt: dict[str, Any]
    contract_pins: dict[str, SourcePin]
    source_replay_files: dict[str, str]
    consumption_scope: str = "saved_receipt_consumption"
    raw_sources_rechecked: bool = False


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _equal(left: Any, right: Any) -> bool:
    return json_sha256(left) == json_sha256(right)


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): file_sha256(package / name) for name in IMPLEMENTATION_PATHS}


def _configuration_correspondence(root: Path, evidence_pin: SourcePin, contracts: dict[str, SourcePin],
                                  files: dict[str, str]) -> dict[str, Any]:
    """Use archived configuration bytes as data, never as permission for another fit."""
    def document(pin: SourcePin) -> dict[str, Any]:
        runtime.pin_file(root, pin.path, pin.sha256, files)
        return read_object(inside(root, pin.path), pin.sha256)

    evidence = SavedEvaluationConfigurationEvidence.model_validate_json(json.dumps(document(evidence_pin)))
    _require(set(contracts) == {"research_contract", "strategy_contract", "temporal_contract"}
             and evidence.strategy.current == contracts["strategy_contract"]
             and evidence.temporal.current == contracts["temporal_contract"]
             and RELATIONSHIP_RUN in evidence.runs, "news configuration correspondence current pins or retained run differ")
    parent = document(parent_owner.PUBLICATION)
    parent_path = inside(root, parent_owner.PUBLICATION.path)
    parent_request = document(SourcePin(path=(parent_path.parent / "_request.json").relative_to(root).as_posix(),
                                        sha256=parent["request_sha256"]))
    research = contracts["research_contract"]
    _require(parent_request["policy"]["strategy_contract"] == evidence.strategy.original_logical.model_dump(mode="json")
             and parent_request["source_files"].get(evidence.strategy.original_logical.path)
             == evidence.strategy.original_logical.sha256
             and parent_request["source_files"].get(research.path) == research.sha256,
             "news configuration correspondence does not match retained parent contracts")
    runtime.pin_file(root, research.path, research.sha256, files)
    run = document(RELATIONSHIP_RUN)
    request_pin = SourcePin(path=(inside(root, RELATIONSHIP_RUN.path).parent / "_request.json").relative_to(root).as_posix(),
                            sha256=run["request_sha256"])
    request = document(request_pin)
    _require(run.get("schema") == "market_predictor.swing_return_training" and run.get("status") == "complete_research_only"
             and request.get("schema") == "market_predictor.swing_return_training_request",
             "news temporal authority is not the retained completed research run")
    training = ReturnTrainingPolicy.model_validate(request["policy"])
    ready_config = document(training.readiness_config)
    readiness = TrainingReadinessPolicy.model_validate(ready_config)
    ready = document(training.readiness)
    _require(training.published_profile == readiness.published_profile == "technical_relationships"
             and readiness.publication == parent_owner.PUBLICATION
             and readiness.strategy_contract == evidence.strategy.original_logical
             and readiness.temporal_contract == evidence.temporal.original_logical
             and readiness.research_contract == research
             and ready.get("publication_sha256") == parent_owner.PUBLICATION.sha256
             and ready.get("status") == "diagnostic_complete"
             and ready.get("training_eligible") is False and ready.get("promotion_eligible") is False,
             "news temporal authority belongs to another parent or configuration")
    for pin in (training.readiness_config, training.readiness, evidence.strategy.original_logical, evidence.temporal.original_logical):
        _require(request["source_files"].get(pin.path) == pin.sha256, "retained run lacks configuration authority pin")
    for pin in (training.readiness_config, evidence.strategy.original_logical, evidence.temporal.original_logical):
        _require(ready["source_files"].get(pin.path) == pin.sha256, "retained readiness lacks configuration authority pin")
    _, strategy_changes = _configuration(root, evidence.strategy, STRATEGY_RENAMES, files)
    _, temporal_changes = _configuration(root, evidence.temporal, TEMPORAL_RENAMES, files)
    corroboration = document(evidence.source_configuration_report)
    entry = corroboration.get("configurations", {}).get(evidence.strategy.original_logical.path, {})
    _require(corroboration.get("schema") == "market_predictor.saved_configuration_differences"
             and corroboration.get("rewritten_historical_pins") is False
             and corroboration.get("source_equivalence_proven") is False and corroboration.get("training_eligible") is False
             and entry.get("historical_sha256") == evidence.strategy.original_logical.sha256
             and entry.get("current_sha256") == contracts["strategy_contract"].sha256
             and entry.get("changed_fields") == strategy_changes
             and isinstance(entry.get("historical_git_revision"), str)
             and len(entry["historical_git_revision"]) >= 7
             and evidence.strategy.git_commit.startswith(entry["historical_git_revision"]),
             "news strategy naming correspondence lacks existing corroboration")
    return {"configuration_evidence": evidence_pin.model_dump(mode="json"),
            "retained_relationship_run": RELATIONSHIP_RUN.model_dump(mode="json"),
            "readiness_config": training.readiness_config.model_dump(mode="json"),
            "strategy": evidence.strategy.model_dump(mode="json"), "temporal": evidence.temporal.model_dump(mode="json"),
            "changed_fields": {"strategy": strategy_changes, "temporal": temporal_changes},
            "research_contract_bytes_unchanged": True, "all_other_parsed_values_types_and_list_order_equal": True,
            "new_fit_authorized": False, "source_equivalence_proven": False}


def _metadata(root: Path, publication: SourcePin) -> VerifiedNewsDecisionPublication:
    path = inside(root, publication.path)
    _require(path.name == "_manifest.json", "news verification requires a publication manifest")
    manifest = read_object(path, publication.sha256)
    request = read_object(path.parent / "_request.json", manifest["request_sha256"])
    checkpoint = read_object(path.parent / "_checkpoint.json", manifest["checkpoint_sha256"])
    months = manifest["months"]
    _require(manifest.get("schema") == publication_owner.SCHEMA
             and manifest.get("status") == "complete_research_only"
             and request.get("schema") == publication_owner.SCHEMA + "_request"
             and checkpoint.get("schema") == publication_owner.SCHEMA + "_checkpoint"
             and all(item.get(key) is value for item in (manifest, request, checkpoint)
                     for key, value in publication_owner.CLOSED.items()), "news publication is incomplete or grants admission")
    _require(isinstance(months, dict) and tuple(sorted(months)) == parent_owner.MONTHS
             and _equal(checkpoint.get("months"), months)
             and checkpoint.get("request_sha256") == manifest["request_sha256"], "news publication checkpoint inventory differs")
    for key in ("rows", "profile", "profile_sha256", "model_columns", "availability_columns", "decision_ids_sha256",
                "source_files", "implementation_files"):
        _require(_equal(manifest.get(key), request.get(key)), f"news publication request field differs: {key}")
    _require(request["rows"] == parent_owner.EXPECTED_ROWS and type(request["rows"]) is int
             and request["decision_ids_sha256"] == parent_owner.EXPECTED_IDS
             and request["profile"] == PROFILE and manifest.get("exclusions_added") == []
             and request.get("parent_publication") == parent_owner.PUBLICATION.model_dump(mode="json")
             and request.get("parent_saved_row_verification") == parent_owner.SAVED_ROW_RECEIPT.model_dump(mode="json")
             and request.get("original_snapshot_replay") == parent_owner.ORIGINAL_REPLAY.model_dump(mode="json")
             and request.get("allowed_parent_column_changes") == ["feature_profile"]
             and request.get("availability_semantics") == "historical_proxy" and request.get("purpose") == "research_proxy"
             and request.get("uncertainty_policy") == "all_news_columns_null_for_applicable_unrepresentable_source_history",
             "news publication population, parent or semantics differ")
    columns = tuple(request["model_columns"])
    clocks = request["availability_columns"]
    _require(len(columns) == len(set(columns)) == 209 and columns[-85:] == NEWS_COLUMNS
             and set(columns).issubset(clocks)
             and all(isinstance(key, str) and isinstance(value, str) and bool(value) for key, value in clocks.items())
             and all(clocks[name] == clock for name, clock in publication_owner.NEWS_CLOCKS.items()),
             "news publication model or clock contract differs")
    config = SourcePin.model_validate(request["config"])
    policy = read_object(inside(root, config.path), config.sha256)
    _require(set(policy) == {"schema", "source_link_index"} and policy["schema"] == publication_owner.CONFIG_SCHEMA
             and _equal(policy, request["policy"]), "news publication configuration differs")
    link = SourcePin.model_validate(policy["source_link_index"])
    provenance = pins(root, request["source_files"])
    implementations = pins(root, request["implementation_files"])
    _require(provenance.get(config.path) == config.sha256 and provenance.get(link.path) == link.sha256,
             "news publication omits config or source-index pin")
    _require(request["profile_sha256"] == json_sha256({
        "profile": PROFILE, "columns": columns,
        "kernel": implementations["src/market_predictor/research/news_decision_features.py"],
        "parent_profile": request["parent_profile_sha256"], "source_index": link.model_dump(mode="json"),
        "uncertainty_policy": request["uncertainty_policy"]}), "news publication profile hash differs")
    # Consumers read saved rows and this immutable evidence, not original news
    # payloads. Preserve those original hashes separately as replay provenance.
    files = pins(root, {config.path: config.sha256, link.path: link.sha256}, {
        path.relative_to(root).as_posix(): publication.sha256,
        (path.parent / "_request.json").relative_to(root).as_posix(): manifest["request_sha256"],
        (path.parent / "_checkpoint.json").relative_to(root).as_posix(): manifest["checkpoint_sha256"],
    })
    expected_files = {"_manifest.json", "_request.json", "_checkpoint.json"}
    total = 0
    for month, child in sorted(months.items()):
        _require(child["path"] == f"{month}/{PROFILE}.parquet" and type(child["rows"]) is int and child["rows"] > 0
                 and child["model_columns"] == list(columns) and child["availability_columns"] == clocks,
                 "news publication child contract differs")
        total += child["rows"]
        for name, digest in ((child["path"], child["sha256"]), (child["path"] + ".manifest.json", child["manifest_sha256"])):
            expected_files.add(name)
            files = pins(root, files, {inside(path.parent, name).relative_to(root).as_posix(): digest})
    _require(total == parent_owner.EXPECTED_ROWS, "news publication child row total differs")
    _require({item.relative_to(path.parent).as_posix() for item in path.parent.rglob("*") if item.is_file()} == expected_files,
             "news publication contains missing or unowned files")
    current = pins(root, implementations, _implementation(root))
    consumed = pins(root, files, current)
    return VerifiedNewsDecisionPublication(path, manifest, request, consumed, current,
                                           columns, {name: clocks[name] for name in columns}, months, {}, {},
                                           pins(root, provenance, consumed))


def _identity(publication: SourcePin, verified: VerifiedNewsDecisionPublication) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "scope": SCOPE, "status": "passed", "manifest_sha256": publication.sha256,
        "publication": publication.model_dump(mode="json"), "request_sha256": verified.manifest["request_sha256"],
        "profile_sha256": verified.request["profile_sha256"], "model_columns": list(verified.model_columns),
        "availability_columns": verified.availability_columns, "rows": parent_owner.EXPECTED_ROWS,
        "publication_availability_columns": verified.request["availability_columns"],
        "unique_decisions": parent_owner.EXPECTED_ROWS, "months": len(parent_owner.MONTHS),
        "decision_ids_sha256": parent_owner.EXPECTED_IDS,
        "source_link_index": verified.request["policy"]["source_link_index"],
        "parent_publication": verified.request["parent_publication"],
        "parent_saved_row_verification": verified.request["parent_saved_row_verification"],
        "original_snapshot_replay": verified.request["original_snapshot_replay"],
        "news_additions_source_replayed": True, "inherited_parent_rows_exact": True,
        "original_outcome_values_exact": True, "baseline_numerical_replayed": False,
        "outcome_filtered_rows": 0, "allowed_parent_column_changes": ["feature_profile"],
        "source_replay_scope": "retained_source_index_versions_links_clocks_and_shared_news_kernel",
        "historical_implementation_files": verified.request["historical_implementation_files"],
        **publication_owner.CLOSED,
    }


def _compare_rows(saved: pd.DataFrame, expected: pd.DataFrame) -> None:
    # Parquet changes string storage and second-resolution (often all-null)
    # appended clocks to milliseconds. Permit only lossless physical conversion;
    # inherited parent dtypes were already checked independently by _child.
    expected = expected.copy()
    for name in saved.columns.intersection(expected.columns):
        left, right = saved[name].dtype, expected[name].dtype
        if (isinstance(left, pd.StringDtype) and isinstance(right, pd.StringDtype)
                and left.storage != right.storage
                and type(getattr(left, "na_value", pd.NA)) is type(getattr(right, "na_value", pd.NA))):
            expected[name] = expected[name].astype(left)
        if (name in publication_owner.NEWS_CLOCKS.values() and isinstance(left, pd.DatetimeTZDtype)
                and isinstance(right, pd.DatetimeTZDtype) and left != right and str(left.tz) == str(right.tz)):
            converted = expected[name].astype(left)
            try:
                pd.testing.assert_series_equal(converted.astype(right), expected[name], check_exact=True)
            except AssertionError as error:
                raise DataReadinessError("news clock storage conversion would lose timestamp precision") from error
            expected[name] = converted
    try:
        pd.testing.assert_frame_equal(saved, expected, check_exact=True)
    except AssertionError as error:
        raise DataReadinessError("news full source replay values, clocks, reasons, identities or inherited rows differ") from error


def _totals() -> dict[str, int]:
    return dict(rows=0, feature_eligible=0, complete_model_rows=0, clock_violations=0)


def verify_news_decision_rows(*, root: Path, publication: SourcePin, output: Path,
                              configuration_evidence: SourcePin, research_contract: SourcePin,
                              strategy_contract: SourcePin, temporal_contract: SourcePin,
                              progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Replay every monthly addition; keep one lease through input exits and publication."""
    root = root.resolve()
    output = inside(root, output)
    _require(output.is_relative_to(root / "data/reports"), "news receipt must be below data/reports")
    if output.exists():
        raise FileExistsError(f"immutable news verification receipt exists: {output}")
    runtime_dir = heavy_job_runtime_dir()
    with heavy_job_lease("verify-news-decision-rows", runtime_dir=runtime_dir if runtime_dir.is_absolute() else root / runtime_dir):
        runtime.guard("news full monthly verification")
        verified = _metadata(root, publication)
        contracts = dict(research_contract=research_contract, strategy_contract=strategy_contract, temporal_contract=temporal_contract)
        correspondence = _configuration_correspondence(root, configuration_evidence, contracts, verified.source_files)
        replay_files = pins(root, verified.source_replay_files, verified.source_files)
        link = NewsSourcePin.model_validate(verified.request["policy"]["source_link_index"])
        monthly: dict[str, dict[str, int]] = {}
        totals = _totals()
        with (parent_owner.verified_news_parent_inputs(root=root) as parent,
              open_news_source_link_index(root=root, index=link) as reader):
            fresh = pins(root, parent.source_files, reader.source_files,
                         {verified.request["config"]["path"]: verified.request["config"]["sha256"]})
            implementations = pins(root, publication_owner._implementation(root), parent.implementation_files,
                                   reader.implementation_files)
            _require(fresh == verified.request["source_files"] and implementations == verified.request["implementation_files"],
                     "news replay current source or implementation inventory differs")
            # These contexts bookend their own immutable source/code files. Only
            # the complementary new artifacts/proof/verifier files need another
            # pass here; never hash the full retained corpus again per month.
            owned = pins(root, parent.source_files, reader.source_files,
                         parent.implementation_files, reader.implementation_files)
            outer = {key: digest for key, digest in replay_files.items() if key not in owned}
            runtime.recheck_files(root, outer)
            _require(verified.model_columns == (*parent.model_columns, *NEWS_COLUMNS)
                     and verified.request["availability_columns"] == {**parent.availability_columns, **publication_owner.NEWS_CLOCKS}
                     and verified.request["parent_profile_sha256"] == parent.request["profile_sha256"]
                     and verified.request["cohort_sha256"] == parent.request["cohort_sha256"]
                     and verified.request["historical_implementation_files"] == parent.historical_implementation_files,
                     "news replay parent metadata differs")
            for month in parent_owner.MONTHS:
                runtime.guard("news full monthly verification")
                baseline = parent.read_month(month)
                record = verified.months[month]
                expected_parent = parent.manifest["months"][month]
                _require(record["rows"] == expected_parent["rows"]
                         and record["decision_ids_sha256"] == expected_parent["decision_ids_sha256"],
                         "news replay monthly parent ownership differs")
                publication_owner._child(verified.path.parent, record, verified.manifest["request_sha256"], baseline,
                                         verified.model_columns, verified.request["availability_columns"])
                saved, _ = load_canonical_artifact(inside(verified.path.parent, record["path"]),
                                                   expected_type=ARTIFACT_TYPE, allow_research=True)
                expected = publication_owner.build_news_month(
                    parent=ResearchFeaturePartition(baseline, parent.model_columns, parent.availability_columns, {}), reader=reader)
                _compare_rows(saved, expected.rows)
                _require(pd.api.types.is_bool_dtype(saved.feature_eligible.dtype), "news eligibility is not boolean")
                summary = dict(rows=len(saved), feature_eligible=int(saved.feature_eligible.sum()),
                               complete_model_rows=int(saved[list(verified.model_columns)].notna().all(axis=1).sum()),
                               clock_violations=0)
                monthly[month] = summary
                for key, count in summary.items():
                    totals[key] += count
                if progress is not None:
                    progress({"month": month, "completed_months": len(monthly), "rows": totals["rows"]})
                del baseline, expected, saved
        _require(totals["rows"] == parent_owner.EXPECTED_ROWS, "news replay total population differs")
        report = {**_identity(publication, verified), "source_files": verified.source_files,
                  "source_replay_files": replay_files,
                  "consumer_scope": "saved_receipt_consumption", "consumer_raw_sources_rechecked": False,
                  "current_implementation_files": verified.implementation_files,
                  "monthly": monthly, "profiles": {PROFILE: totals},
                  "configuration_evidence": configuration_evidence.model_dump(mode="json"),
                  "contract_pins": {key: pin.model_dump(mode="json") for key, pin in contracts.items()},
                  "configuration_correspondence": correspondence}
        # No receipt exists until both source contexts have completed exit checks.
        runtime.recheck_files(root, outer)
        final_files = _metadata(root, publication).source_files
        _require(all(verified.source_files.get(key) == value for key, value in final_files.items()),
                 "news publication changed at final check")
        runtime.guard("news verification receipt publication")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_object(output, report)
        return report


def validate_news_decision_receipt(root: Path, publication: SourcePin, receipt: SourcePin) -> VerifiedNewsDecisionPublication:
    """Validate pinned receipt/artifacts without numerical replay; caller owns its lease."""
    root = root.resolve()
    runtime.guard("news verification receipt validation")
    report = read_object(inside(root, receipt.path), receipt.sha256)
    verified = _metadata(root, publication)
    contracts = {key: SourcePin.model_validate(value) for key, value in report["contract_pins"].items()}
    correspondence = _configuration_correspondence(
        root, SourcePin.model_validate(report["configuration_evidence"]), contracts, verified.source_files)
    _require(_equal(correspondence, report.get("configuration_correspondence")), "news configuration correspondence differs")
    replay_files = pins(root, verified.source_replay_files, verified.source_files)
    _require(all(key in report and _equal(report[key], value) for key, value in _identity(publication, verified).items()),
             "news verification receipt identity or scope differs")
    _require(report.get("source_files") == verified.source_files
             and report.get("current_implementation_files") == verified.implementation_files
             and report.get("source_replay_files") == replay_files
             and report.get("consumer_scope") == "saved_receipt_consumption"
             and report.get("consumer_raw_sources_rechecked") is False,
             "news verification receipt source or implementation pins differ")
    monthly = report.get("monthly")
    _require(isinstance(monthly, dict) and tuple(sorted(monthly)) == parent_owner.MONTHS,
             "news verification receipt monthly inventory differs")
    assert isinstance(monthly, dict)
    totals = _totals()
    for month, summary in monthly.items():
        _require(isinstance(summary, dict) and set(summary) == set(totals)
                 and all(type(value) is int and 0 <= value <= verified.months[month]["rows"] for value in summary.values())
                 and summary["rows"] == verified.months[month]["rows"] and summary["clock_violations"] == 0,
                 "news verification receipt monthly counts differ")
        for key, value in summary.items():
            totals[key] += value
    _require(_equal(report.get("profiles"), {PROFILE: totals}), "news verification receipt totals differ")
    files = pins(root, verified.source_files, {receipt.path: receipt.sha256})
    runtime.recheck_files(root, files)
    return VerifiedNewsDecisionPublication(verified.path, verified.manifest, verified.request, files,
                                           verified.implementation_files, verified.model_columns,
                                           verified.availability_columns, verified.months, report, contracts, replay_files)
