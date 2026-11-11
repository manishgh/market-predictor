"""Recover stock/SPY outcomes without changing the fixed investment selections."""
from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal, cast

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.contracts import InvestmentSpyPairResult, InvestmentTargetRequest, InvestmentTargetResult
from market_predictor.investment.dataset import (
    DATASET_SCHEMA,
    InvestmentDatasetAdmission,
    InvestmentDatasetRequest,
    add_pins,
    pin_source_file,
    read_object,
    require,
)
from market_predictor.investment.diagnostics import CONTEXT, PREDICTION
from market_predictor.investment.feature_ablation import ablation_profiles
from market_predictor.investment.label_materialization import compile_investment_spy_pair
from market_predictor.investment.label_sources import InvestmentSourceReader
from market_predictor.investment.ranking import RANKING_SCORE
from market_predictor.investment.training import HORIZONS, SCOPES, SCORE_YEARS, investment_selection_diagnostics
from market_predictor.research.investment_diagnostics import _recheck, read_reference_run
from market_predictor.research.investment_feature_ablation import InvestmentComparisonSources, read_ablation_reference
from market_predictor.research.investment_features import ArtifactReference, _publish
from market_predictor.research.investment_label_reuse import InvestmentCheckpointOrigin, producer_origin
from market_predictor.research.investment_labels import (
    InvestmentLabelMaterializationRequest,
    _validate_membership_sources,
    prepare_investment_label_components,
    read_investment_label_inputs,
    verified_investment_label_sources,
)
from market_predictor.research.investment_labels import (
    _implementation as label_implementation,
)
from market_predictor.research.investment_ranking import FAMILY, NEWS, RANKING_PROFILES
from market_predictor.research.investment_training import FLAGS, RANKING_PROFILE
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_outcome_admission import corrected_ticker

SelectedKey = tuple[str, int]


class InvestmentSelectionOutcomeRequest(HoldingContract):
    schema_version: Literal["market_predictor.investment_selection_outcome_request"]
    scope: Literal["retrospective_research"]
    ranking_manifest: SourcePin
    selection_inventory: SourcePin
    label_corpus: SourcePin
    label_request: SourcePin
    stock_identity: SourcePin | None = None
    currency_corroborations: SourcePin | None = None


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = ("research/investment_selection_outcomes.py", "investment/dataset.py", "investment/training.py",
        "research/investment_diagnostics.py", "research/investment_feature_ablation.py", "investment/ranking.py")
    return {**label_implementation(), **{str(package / name): file_sha256(package / name) for name in names}}


def _reference(pin: SourcePin) -> ArtifactReference:
    return ArtifactReference(path=pin.path, sha256=pin.sha256)


def _same_pin(root: Path, left: SourcePin, right: SourcePin) -> bool:
    return left.sha256 == right.sha256 and inside(root, left.path) == inside(root, right.path)


def _replay_ordered_selection(scores: pd.DataFrame, score_column: str, reported: list[dict[str, Any]],
                              frozen: list[dict[str, Any]]) -> list[dict[str, Any]]:
    details = cast(list[dict[str, Any]], investment_selection_diagnostics(scores, score_column)["top_decile_by_date"])
    require(details == reported, "fixed selection differs when replayed from original scores")
    actual = [{"decision_session": row["decision_session"], "security_ids": row["selected_security_ids"]} for row in details]
    require(actual == frozen, "fixed ordered selection differs from the frozen inventory")
    return actual


def _selected_rows(root: Path, parsed: InvestmentSelectionOutcomeRequest, inventory: dict[str, Any],
                   pins: dict[str, str]) -> tuple[dict[SelectedKey, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    ranking = read_object(root, parsed.ranking_manifest, pins)
    require(ranking.get("schema") == "market_predictor.investment_ranking"
        and ranking.get("status") == "complete_research_only" and ranking.get("scope") == "retrospective_research"
        and ranking.get("fold_scope_fits") == 12 and ranking.get("final_models") == 0
        and all(ranking.get(flag) is False for flag in FLAGS),
        "selected outcomes require the closed retrospective ranking experiment")
    require(inventory.get("schema") == "fixed_investment_selection_gap_inventory"
        and _same_pin(root, SourcePin.model_validate(inventory["ranking"]), parsed.ranking_manifest)
        and inventory["selection_sha256"] == json_sha256(inventory["selections"]),
        "selected outcome inventory ranking or ordered selection hash differs")
    references = InvestmentComparisonSources.model_validate({
        name: ranking[name] for name in ("training_manifest", "training_request", "ablation_manifest", "ablation_request")})
    training = read_reference_run(root, references.training_manifest, references.training_request, pins)
    columns = ablation_profiles({name: tuple(values) for name, values in training.experiment["model_columns"].items()})
    ablation = read_ablation_reference(root, references, training, columns, pins)
    admission = InvestmentDatasetAdmission.model_validate_json(json.dumps(
        read_object(root, training.request.dataset_admission, pins)))
    dataset = read_object(root, training.request.dataset_manifest, pins)
    dataset_request = InvestmentDatasetRequest.model_validate_json(json.dumps(read_object(root, admission.request, pins)))
    require(admission.dataset == training.request.dataset_manifest and dataset.get("schema") == DATASET_SCHEMA
        and dataset.get("status") == "complete_research_dataset" and dataset["request"] == admission.request.model_dump(mode="json")
        and dataset["feature_contract_sha256"] == admission.feature_contract_sha256
        and _same_pin(root, dataset_request.targets_manifest, parsed.label_corpus),
        "selected outcomes corpus does not bind the original admitted dataset")
    label_request = read_object(root, parsed.label_request, pins)
    require(_same_pin(root, SourcePin.model_validate(label_request["roster_manifest"]), dataset_request.roster_manifest),
        "selected outcomes label request binds a different roster")
    rank_directory = inside(root, parsed.ranking_manifest.path).parent
    require(ranking["files"] == {"experiment.json": ranking["experiment_sha256"]}, "ranking experiment file inventory differs")
    experiment = read_object(root, SourcePin(path=str(rank_directory / "experiment.json"), sha256=ranking["experiment_sha256"]), pins)
    require(experiment.get("schema") == "market_predictor.investment_ranking_experiment"
        and experiment["request_sha256"] == ranking["request_sha256"]
        and experiment["training_experiment_sha256"] == training.manifest["experiment_sha256"]
        and experiment["ablation_experiment_sha256"] == ablation.manifest["experiment_sha256"]
        and experiment["dataset_manifest"] == training.request.dataset_manifest.model_dump(mode="json")
        and experiment["dataset_admission"] == training.request.dataset_admission.model_dump(mode="json"),
        "ranking experiment identity differs from its original references")
    expected_views = {f"{horizon}/{profile}/{FAMILY}/{scope}"
        for horizon in HORIZONS for profile in RANKING_PROFILES for scope in SCOPES}
    expected_units = {f"{view}/{year}" for view in expected_views for year in SCORE_YEARS}
    require(set(inventory["selections"]) == set(ranking["aggregate_diagnostics"]) == expected_views
        and set(ranking["new_units"]).isdisjoint(ranking["reused_units"])
        and set(ranking["new_units"]) | set(ranking["reused_units"]) == expected_units,
        "fixed selection view or original prediction unit inventory differs")
    selected: dict[SelectedKey, dict[str, Any]] = {}
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for view in sorted(expected_views):
        horizon_text, profile, _, _ = view.split("/")
        horizon = int(horizon_text)
        rank_model = profile == RANKING_PROFILE
        score_column = RANKING_SCORE if rank_model else PREDICTION
        frames = []
        for year in SCORE_YEARS:
            guard("investment fixed selection prediction replay")
            key = f"{view}/{year}"
            if rank_model:
                raw_pin = ranking["new_units"][key]
                producer_request, producer_experiment = ranking["request_sha256"], ranking["experiment_sha256"]
            else:
                reference = ablation if profile == NEWS else training
                raw_pin = reference.manifest["new_units" if profile == NEWS else "units"][key]
                require(raw_pin == ranking["reused_units"][key], "fixed selection control pin differs from its producer")
                producer_request, producer_experiment = reference.request_sha256, reference.manifest["experiment_sha256"]
            pin = SourcePin.model_validate(raw_pin)
            unit = read_object(root, pin, pins)
            require(unit.get("schema") == ("market_predictor.investment_ranking_model" if rank_model
                else "market_predictor.investment_research_model") and unit.get("scope") == "retrospective_research"
                and unit["request_sha256"] == producer_request
                and unit["unit"]["experiment_sha256"] == producer_experiment and unit["unit"]["profile"] == profile
                and unit["unit"]["horizon_sessions"] == horizon and unit["unit"]["score_year"] == year
                and unit["unit"]["scope"] == view.split("/")[-1] and unit["family"] == FAMILY
                and set(unit["files"]) == {"predictions.parquet", "model.joblib"}
                and all(unit.get(flag) is False for flag in FLAGS), "fixed selection prediction unit lineage differs")
            parent = inside(root, pin.path).parent
            for name, digest in unit["files"].items():
                pin_source_file(root, SourcePin(path=str(parent / name), sha256=digest), pins)
            frame = pd.read_parquet(parent / "predictions.parquet", columns=[*CONTEXT, score_column])
            require(frame.horizon_sessions.eq(horizon).all() and pd.to_datetime(frame.decision_session).dt.year.eq(year).all()
                and frame[score_column].notna().equals(frame.feature_eligible), "fixed selection saved score population differs")
            frames.append(frame)
        scores = pd.concat(frames, ignore_index=True)
        actual = _replay_ordered_selection(scores, score_column, ranking["aggregate_diagnostics"][view]["top_decile_by_date"],
            inventory["selections"][view])
        lookup = {(str(row.decision_session), row.security_id): row for row in scores.itertuples(index=False)}
        require(len(lookup) == len(scores), "fixed selection scores duplicate a date and security")
        cohorts[view] = []
        for entry in actual:
            keys = []
            for security in entry["security_ids"]:
                row = lookup[(entry["decision_session"], security)]
                selected_key = (row.decision_id, horizon)
                record = {"security_id": security, "decision_session": entry["decision_session"],
                    "originally_known": bool(row.research_eligible),
                    "stock_net_return": float(row.stock_net_return) if row.research_eligible else None,
                    "spy_gross_return": float(row.spy_gross_return) if row.research_eligible else None,
                    "net_excess_vs_spy": float(row.target_net_excess_vs_spy) if row.research_eligible else None}
                require(selected_key not in selected or selected[selected_key] == record,
                    "fixed selection views disagree on the same economic row")
                selected[selected_key] = record
                keys.append(selected_key)
            cohorts[view].append({"decision_session": entry["decision_session"], "security_ids": entry["security_ids"], "keys": keys})
    return selected, cohorts


def _original_outcomes(root: Path, parsed: InvestmentSelectionOutcomeRequest, inventory: dict[str, Any],
                       selected: dict[SelectedKey, dict[str, Any]], pins: dict[str, str]) -> dict[SelectedKey, dict[str, Any]]:
    corpus = read_object(root, parsed.label_corpus, pins)
    require(corpus.get("schema") == "market_predictor.investment_label_corpus"
        and corpus.get("status") == "labels_complete_with_explicit_gaps" and corpus.get("scope") == "full_roster"
        and corpus["request_sha256"] == parsed.label_request.sha256, "selected outcomes require the original full label corpus")
    rows = {(row["decision_id"], row["horizon_sessions"]): row for row in inventory["rows"]}
    require(len(rows) == len(inventory["rows"]) == len(selected) == inventory["unique_selected_decision_horizons"]
        and set(rows) == set(selected)
        and sum(not record["originally_known"] for record in selected.values()) == inventory["unknown_selected_decision_horizons"],
        "fixed selected decision/horizon keys or original missing count differ")
    origins = {item["security_id"]: item for item in corpus["security_checkpoint_origins"]}
    checkpoints = {str(inside(root, item["path"])): item["sha256"] for item in corpus["security_checkpoints"]}
    require(len(origins) == len(checkpoints) == len(corpus["security_checkpoint_origins"]) == len(corpus["security_checkpoints"]),
        "original label corpus checkpoint inventory is ambiguous")
    groups: dict[str, list[SelectedKey]] = {}
    for key, row in selected.items():
        groups.setdefault(row["security_id"], []).append(key)
    outputs: dict[SelectedKey, dict[str, Any]] = {}
    for security, keys in sorted(groups.items()):
        guard("investment fixed selection original labels")
        origin = InvestmentCheckpointOrigin.model_validate_json(json.dumps(origins[security]))
        checkpoint_path = inside(root, origin.checkpoint.path)
        require(checkpoints.get(str(checkpoint_path)) == origin.checkpoint.sha256
            and inside(root, origin.labels.path) == checkpoint_path.parent / "labels.json"
            and _same_pin(root, origin.request, parsed.label_request), "selected original checkpoint origin differs")
        checkpoint = read_object(root, origin.checkpoint, pins)
        require(checkpoint.get("status") == "security_complete" and checkpoint["security_id"] == security
            and checkpoint["job_identity"] == origin.job_identity and checkpoint["files"]["labels.json"] == origin.labels.sha256,
            "selected original checkpoint binding differs")
        job = read_object(root, origin.job, pins)
        require(job["job_identity"] == origin.job_identity and pd.Timestamp(job["verified_at_utc"]) == pd.Timestamp(origin.verified_at_utc),
            "selected original verification clock differs")
        prior_rows = read_object(root, origin.labels, pins)["rows"]
        prior = {(row["decision_id"], row["horizon_sessions"]): row for row in prior_rows}
        require(len(prior_rows) == len(prior) == checkpoint["declared_rows"], "selected original labels duplicate a key")
        for key in keys:
            old, frozen, prediction = prior[key], rows[key], selected[key]
            old_known = old["status"] == "source_validated"
            require(old["security_id"] == frozen["security_id"] == security and old["decision_session"] == frozen["decision_session"]
                == prediction["decision_session"] and old["status"] == frozen["target_status"]
                and old["missing_reasons"] == frozen["target_missing_reasons"]
                and old_known == frozen["target_verified"] == prediction["originally_known"]
                and _same_pin(root, origin.checkpoint,
                    SourcePin(path=frozen["target_checkpoint_path"], sha256=frozen["target_checkpoint_sha256"]))
                and pd.Timestamp(old["membership_verified_at_utc"]) == pd.Timestamp(origin.verified_at_utc),
                "fixed selection and original target status, identity, reasons or clock differ")
            record = {"decision_id": key[0], "horizon_sessions": key[1], "security_id": security, "ticker": frozen["ticker"],
                "decision_session": old["decision_session"], "entry_time_utc": old["entry_time_utc"],
                "economic_maturity_utc": old["economic_maturity_utc"], "original_status": old["status"],
                "original_missing_reasons": old["missing_reasons"], "target_checkpoint": origin.checkpoint.model_dump(mode="json"),
                "target_labels": origin.labels.model_dump(mode="json"), "target_job": origin.job.model_dump(mode="json"),
                "verified_at_utc": origin.verified_at_utc.isoformat(), "outcome_available_at_utc": None,
                "status": "unresolved", "pair_missing_reasons": [], "pair_proof": None,
                "stock_net_return": None, "spy_gross_return": None, "net_excess_vs_spy": None}
            if old_known:
                request = InvestmentTargetRequest.model_validate_json(json.dumps(old["request"]))
                target = InvestmentTargetResult.model_validate_json(json.dumps(old["target"]))
                stock, spy = target.holdings[:2]
                returns = {"stock_net_return": stock.net_return, "spy_gross_return": spy.gross_return,
                    "net_excess_vs_spy": target.net_excess_vs_spy}
                require(target.status == "available" and request.decision_id == target.decision_id == key[0]
                    and target.horizon_sessions == request.policy.horizon_sessions == key[1]
                    and request.sha256() == target.request_sha256 == old["request_sha256"]
                    and stock.security_id == security and spy.security_id == "benchmark:SPY"
                    and target.label_available_at_utc is not None
                    and old["label_available_at_utc"] == old["target"]["label_available_at_utc"]
                    and all(returns[name] == prediction[name] for name in returns),
                    "reused selected economics differs from the original label or score evidence")
                record.update(returns, status="reused_original", outcome_available_at_utc=old["label_available_at_utc"])
            outputs[key] = record
    return outputs


def _compare_cohorts(cohorts: dict[str, list[dict[str, Any]]], outcomes: dict[SelectedKey, dict[str, Any]]) -> dict[str, Any]:
    comparisons = {}
    for view, entries in cohorts.items():
        dates = []
        for cohort in entries:
            rows = [outcomes[key] for key in cohort["keys"]]
            counts = Counter(row["status"] for row in rows)
            complete = not counts["unresolved"]
            row = {"decision_session": cohort["decision_session"], "security_ids": cohort["security_ids"],
                "decision_ids": [key[0] for key in cohort["keys"]], "selected_positions": len(rows),
                "reused_positions": counts["reused_original"], "recovered_positions": counts["recovered"],
                "unresolved_positions": counts["unresolved"], "complete": complete}
            for name in ("stock_net_return", "spy_gross_return", "net_excess_vs_spy"):
                row[f"mean_{name}"] = fmean(value[name] for value in rows) if complete else None
            dates.append(row)
        complete = all(date["complete"] for date in dates)
        comparisons[view] = {"dates": dates, "date_count": len(dates), "complete_dates": sum(date["complete"] for date in dates),
            "selected_positions": sum(date["selected_positions"] for date in dates),
            "reused_positions": sum(date["reused_positions"] for date in dates),
            "recovered_positions": sum(date["recovered_positions"] for date in dates),
            "unresolved_positions": sum(date["unresolved_positions"] for date in dates),
            **{f"mean_{name}": fmean(date[f"mean_{name}"] for date in dates) if complete else None
                for name in ("stock_net_return", "spy_gross_return", "net_excess_vs_spy")}}
    return comparisons

def _amended_label_request(root: Path, request: InvestmentSelectionOutcomeRequest,
                           original: InvestmentLabelMaterializationRequest, outcomes: dict[SelectedKey, dict[str, Any]],
                           pins: dict[str, str]) -> InvestmentLabelMaterializationRequest:
    data = original.model_dump(mode="json")
    eligible_securities = {row["security_id"] for row in outcomes.values() if row["status"] == "unresolved"}
    if request.stock_identity is not None:
        require(original.raw_source_extension is not None and original.raw_source_extension.stock_identity is not None,
            "selected outcome stock identity amendment requires the original extension authority")
        assert original.raw_source_extension is not None and original.raw_source_extension.stock_identity is not None
        before = read_object(root, original.raw_source_extension.stock_identity, pins)["proofs"]
        after = read_object(root, request.stock_identity, pins)["proofs"]
        old_proofs = {(row["security_id"], row["ticker"]): row for row in before}
        new_proofs = {(row["security_id"], row["ticker"]): row for row in after}
        require(len(old_proofs) == len(before) and len(new_proofs) == len(after)
            and all(new_proofs.get(key) == proof for key, proof in old_proofs.items())
            and all(key[0] in eligible_securities for key in new_proofs.keys() - old_proofs.keys()),
            "selected outcome stock identity amendment replaces prior proof or expands beyond selected gaps")
        data["raw_source_extension"]["stock_identity"] = request.stock_identity.model_dump(mode="json")
    if request.currency_corroborations is not None:
        require(original.currency_corroboration_manifest is not None,
            "selected outcome currency amendment requires the original corroborations")
        assert original.currency_corroboration_manifest is not None
        original_pin = SourcePin(path=original.currency_corroboration_manifest.path, sha256=original.currency_corroboration_manifest.sha256)
        before = read_object(root, original_pin, pins)["corroborations"]
        after = read_object(root, request.currency_corroborations, pins)["corroborations"]
        allowed = {parts[3] for row in outcomes.values() if row["status"] == "unresolved"
            for reason in row["original_missing_reasons"] if (parts := reason.split(":", 4))[:3] == ["stock", "action", "cash_dividends"]
            and len(parts) == 5}
        additions = after[len(before):]
        require(after[:len(before)] == before and len({row["action_id"] for row in after}) == len(after)
            and all(row["action_id"] in allowed for row in additions),
            "selected outcome currency amendment replaces prior terms or expands beyond selected gaps")
        data["currency_corroboration_manifest"] = request.currency_corroborations.model_dump(mode="json")
    return InvestmentLabelMaterializationRequest.model_validate_json(json.dumps(data))


def resolve_investment_selection_outcomes(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Retain every fixed position and resolve stock/SPY pairs from verified retained evidence."""
    root = root.resolve()
    output = inside(root, output)
    if output.exists():
        raise FileExistsError(output)
    runtime = heavy_job_runtime_dir()
    request_pin = SourcePin(path=str(inside(root, request)), sha256=request_sha256)
    with heavy_job_lease("prepare-investment-selection-outcomes", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("investment fixed selection preparation")
        implementation = _implementation()
        pins = dict(implementation)
        parsed = InvestmentSelectionOutcomeRequest.model_validate_json(json.dumps(read_object(root, request_pin, pins)))
        inventory = read_object(root, parsed.selection_inventory, pins)
        add_pins(root, pins, inventory["source_files"])
        selected, cohorts = _selected_rows(root, parsed, inventory, pins)
        outcomes = _original_outcomes(root, parsed, inventory, selected, pins)
        inputs = read_investment_label_inputs(root, _reference(parsed.label_request), pins)
        amended = _amended_label_request(root, parsed, inputs.request, outcomes, pins)
        roster = {row["decision_id"]: row for row in inputs.rows}
        pending = sorted((key for key, row in outcomes.items() if row["status"] == "unresolved"),
            key=lambda key: (outcomes[key]["security_id"], outcomes[key]["decision_session"], key[1]))
        require(inputs.request.decision_ids is None and all(key[0] in roster for key in pending),
            "selected recovery requires full frozen roster membership")
        placements = sum(len(date["security_ids"]) for entries in inventory["selections"].values() for date in entries)
        require(sum(len(date["keys"]) for entries in cohorts.values() for date in entries) == placements,
            "fixed selected position placement count differs")
    with verified_investment_label_sources(root, amended, inputs.policy) as (extension, evidence, sources):
        add_pins(root, pins, sources["source_files"])
        verified_at = datetime.now(UTC)
        policy_ref = EvidenceReference(reference=parsed.label_request.path, artifact_sha256=parsed.label_request.sha256,
            record_locator="original nominal-unpaid-claim valuation and 20 bp stock cost for fixed selected stock/SPY comparison",
            interpretation_policy_sha256=request_sha256, retrieved_at=verified_at, available_at=verified_at)
        membership_ref = EvidenceReference(reference=inputs.membership.path, artifact_sha256=inputs.membership.sha256,
            record_locator="current independent verification of reconstructed historical security membership intervals",
            interpretation_policy_sha256=request_sha256, retrieved_at=verified_at, available_at=verified_at)
        actions, raw_selection = prepare_investment_label_components(root, amended, inputs.policy, inputs.roster_rows,
            extension, evidence, sources, pins, policy_ref, request_sha256)
        reader = InvestmentSourceReader(root, raw_selection, request_sha256)
        producer_export, producer_commit = producer_origin(implementation, pins)
        report: dict[str, Any] = {
            "schema": "market_predictor.investment_selection_outcomes", "scope": "retrospective_research",
            "status": "complete_with_explicit_gaps", "request": request_pin.model_dump(mode="json"),
            "request_sha256": request_sha256, **parsed.model_dump(mode="json", exclude={"schema_version", "scope"}),
            "selection_sha256": inventory["selection_sha256"], "source_files": pins, "implementation_files": implementation,
            "producer_export": None if producer_export is None else producer_export.model_dump(mode="json"),
            "producer_commit": producer_commit, "verified_at_utc": verified_at.isoformat(),
            "round_trip_stock_cost_bps": 20, "benchmark_basis": "stock_net_minus_spy_gross",
            "selection_changed": False, "models_refitted": False, "original_labels_changed": False, "portfolio_evaluated": False,
            "untouched_test_evidence": False, **FLAGS,
        }

        def save(staged: Path) -> None:
            proof_pins: dict[str, str] = {}
            previous_security = None
            for key in pending:
                row = outcomes[key]
                if row["security_id"] != previous_security:
                    guard("investment selected security outcome recovery")
                    previous_security = row["security_id"]
                guard("investment selected stock/SPY pair", reclaim_unused_memory=False)
                decision = dict(roster[key[0]])
                require(decision["status"] == "eligible" and decision["security_id"] == row["security_id"]
                    and decision["decision_session"] == row["decision_session"] and decision["ticker"] == row["ticker"],
                    "selected recovery roster identity or eligibility differs")
                decision["ticker"] = corrected_ticker(decision["security_id"], decision["ticker"],
                    pd.Timestamp(decision["decision_session"]).date(), inputs.policy.decision_corrections)
                horizon = cast(Literal[63, 252], key[1])
                pair = compile_investment_spy_pair(decision=decision, horizon=horizon, reader=reader,
                    membership=_validate_membership_sources(sources=sources, row=decision, horizon=horizon, evidence=membership_ref),
                    action_source_validation=actions.validate_sources)
                require(all(pd.Timestamp(pair[name]) == pd.Timestamp(row[name]) for name in ("entry_time_utc", "economic_maturity_utc")),
                    "selected stock/SPY recovery differs from the frozen holding interval")
                row["verified_at_utc"] = verified_at.isoformat()
                row["pair_missing_reasons"] = pair["missing_reasons"]
                row["pair_components"] = pair["components"]
                if pair["status"] != "source_validated":
                    continue
                result = InvestmentSpyPairResult.model_validate_json(json.dumps(pair["pair"]))
                require(result.status == "available" and result.decision_id == key[0] and result.horizon_sessions == key[1]
                    and result.holdings[0].security_id == row["security_id"]
                    and result.outcome_available_at_utc is not None
                    and verified_at <= result.outcome_available_at_utc <= datetime.now(UTC),
                    "recovered pair availability, identity or actual verification differs")
                relative = Path("pairs") / key[0] / f"{horizon}.json"
                path = staged / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                write_json_object(path, {"schema": "market_predictor.investment_selected_spy_pair_evidence",
                    "verified_at_utc": verified_at.isoformat(), "request": pair["request"], "specifications": pair["specifications"],
                    "result": pair["pair"]})
                digest = file_sha256(path)
                proof_pins[str(path)] = digest
                row.update(status="recovered", stock_net_return=result.holdings[0].net_return,
                    spy_gross_return=result.holdings[1].gross_return, net_excess_vs_spy=result.net_excess_vs_spy,
                    outcome_available_at_utc=pair["outcome_available_at_utc"],
                    pair_proof=SourcePin(path=str(output / relative), sha256=digest).model_dump(mode="json"))
            comparisons = _compare_cohorts(cohorts, outcomes)
            counts = Counter(row["status"] for row in outcomes.values())
            require(counts["reused_original"] == len(outcomes) - len(pending)
                and counts["recovered"] + counts["unresolved"] == len(pending),
                "fixed outcome coverage changed the original population")
            reader.recheck()
            actions.recheck()
            add_pins(root, pins, reader.source_files)
            _recheck(pins)
            _recheck(proof_pins)
            guard("investment selected outcome publication")
            write_json_object(staged / "outcomes.json", {"rows": [outcomes[key] for key in sorted(outcomes)]})
            write_json_object(staged / "comparisons.json", {"views": comparisons})
            report.update(status="complete_with_explicit_gaps" if counts["unresolved"] else "complete_research_only",
                unique_selected_keys=len(outcomes), original_unknown_keys=len(pending),
                reused_original_keys=counts["reused_original"], recovered_keys=counts["recovered"],
                unresolved_keys=counts["unresolved"], comparison_views=len(comparisons),
                comparison_cohorts=sum(view["date_count"] for view in comparisons.values()),
                selected_positions=sum(view["selected_positions"] for view in comparisons.values()),
                completed_at_utc=datetime.now(UTC).isoformat())
        result = _publish(output, report, pins, save)
        return {**result, **{name: report[name] for name in ("reused_original_keys", "recovered_keys", "unresolved_keys",
            "comparison_views", "comparison_cohorts", "selected_positions")}}
