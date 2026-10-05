"""Synthetic saved-score fixtures: no estimator loading, fitting or real labels."""
from __future__ import annotations

import json
import shutil
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.temporal_manifest import build_temporal_schedule, load_temporal_manifest_config
from market_predictor.evidence.hashing import json_sha256
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.research import swing_oof_policy_evaluation as owner
from market_predictor.swing.contracts.holding_accounting import ExecutionEvent, HoldingSpecification, KnownMark, PaymentEvent
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.funded_policy_inputs import FundedLotInput, unsimulated_managed_specification
from market_predictor.swing.evaluation.accounting import evaluate_event_aware_swing_accounting
from market_predictor.swing.evaluation.trade_simulation import simulate_ordinary_sales
from market_predictor.swing.labels.frozen_exit import compile_frozen_exit
from market_predictor.swing.training.return_comparison import FLAGS
from market_predictor.swing.training.return_validation import return_folds, security_transfer_mask
from tests.test_swing_frozen_exit import case as exit_case
from tests.test_swing_ordinary_holding import CALENDAR

REPO = Path(__file__).resolve().parents[1]


def _json(path: Path, value: dict[str, Any]) -> SourcePin:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return SourcePin(path=str(path), sha256=file_sha256(path))


def _relative(root: Path, pin: SourcePin) -> SourcePin:
    return SourcePin(path=Path(pin.path).relative_to(root).as_posix(), sha256=pin.sha256)


@pytest.fixture
def saved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> dict[str, Any]:
    monkeypatch.setattr(owner, "_guard", lambda: None)
    template = json.loads((REPO / "configs/swing_training_readiness.json").read_text())
    policy = json.loads((REPO / "configs/swing_return_training.json").read_text())
    profile = getattr(request, "param", None)
    published_profile = "technical_market"
    if isinstance(profile, dict):
        policy.update(profile)
        published_profile = profile["published_profile"]
    pins = {}
    contracts = {}
    for role in ("strategy_contract", "research_contract", "temporal_contract"):
        source = REPO / template[role]["path"]
        target = tmp_path / "configs" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        pin = SourcePin(path=target.relative_to(tmp_path).as_posix(), sha256=file_sha256(target))
        contracts[role] = pin
        template[role] = pin.model_dump(mode="json")
        pins[pin.path] = pin.sha256
    strategy = load_strategy_contract(tmp_path / contracts["strategy_contract"].path)
    temporal = load_temporal_manifest_config(tmp_path / contracts["temporal_contract"].path)
    sessions = build_temporal_schedule(temporal).folds[0].train_sessions
    names = list(owner.swing_model_feature_columns(contract=strategy, catalyst=False))
    if policy["feature_profile"] in ("technical_relationships", owner.ISSUER_REACTION_PROFILE):
        names.extend(owner.RETURN_RELATIONSHIP_COLUMNS)
    if policy["feature_profile"] == owner.ISSUER_REACTION_PROFILE:
        names.extend(owner.REACTION_COLUMNS)
    pool = pd.Series([f"issuer-{number}" for number in range(20)])
    held = security_transfer_mask(pool)
    identities = [pool[held].iloc[0], *pool[~held].iloc[:2]]
    rows = pd.DataFrame([(day, identity, ordinal) for day in sessions for ordinal, identity in enumerate(identities)],
                        columns=["session_date_et", "security_id", "ordinal"])
    rows["decision_id"] = [f"synthetic-{number}" for number in range(len(rows))]
    rows["ticker"] = rows.security_id
    rows["sector"] = rows.ordinal.map(lambda number: f"sector-{number}")
    rows["primary_benchmark"] = "XLF"
    rows["decision_time_utc"] = swing_prediction_cutoffs(rows.session_date_et)
    rows["decision_group_id"] = rows.session_date_et.map(date.isoformat)
    rows["feature_eligible"] = True
    rows["cross_section_eligible"] = True
    if isinstance(profile, str):
        rows["cross_section_eligible"] = rows.session_date_et.eq(date.fromisoformat(profile))
    rows["feature_profile"] = published_profile
    rows = rows.drop(columns="ordinal")
    publication = tmp_path / "data/research/features"
    parent_request = _json(publication / "_request.json", {"synthetic": True, "profiles": [published_profile]})
    pins[Path(parent_request.path).relative_to(tmp_path).as_posix()] = parent_request.sha256
    months = {}
    for month, part in rows.groupby(rows.session_date_et.map(lambda day: day.isoformat()[:7])):
        clocks = {name: f"available_at_{name}" for name in names}
        frame = pd.concat([part.reset_index(drop=True), pd.DataFrame({clock: part.decision_time_utc.reset_index(drop=True)
                           - pd.Timedelta(minutes=1) for clock in clocks.values()})], axis=1)
        path = publication / month / f"{published_profile}.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
        digest = file_sha256(path)
        sidecar = _json(manifest_path_for(path), {"artifact_sha256": digest, "rows": len(frame), "production_ready": False,
                                                "inputs": {"request_sha256": parent_request.sha256}})
        pins[path.relative_to(tmp_path).as_posix()] = digest
        pins[Path(sidecar.path).relative_to(tmp_path).as_posix()] = sidecar.sha256
        months[month] = {"rows": len(frame), "decision_ids_sha256": json_sha256(sorted(frame.decision_id)),
            "profiles": {published_profile: {"path": f"{month}/{published_profile}.parquet", "sha256": digest,
                "manifest_sha256": sidecar.sha256, "model_columns": names, "availability_columns": clocks}}}
    publication_pin = _relative(tmp_path, _json(publication / "_manifest.json", {
        "months": months, "request_sha256": parent_request.sha256}))
    pins[publication_pin.path] = publication_pin.sha256
    template["publication"] = publication_pin.model_dump(mode="json")
    template["published_profile"] = published_profile
    readiness_config = _relative(tmp_path, _json(tmp_path / "configs/readiness.json", template))
    readiness = _relative(tmp_path, _json(tmp_path / "data/research/readiness.json",
        {"publication_sha256": publication_pin.sha256, "published_profile": published_profile}))
    for pin in (readiness_config, readiness):
        pins[pin.path] = pin.sha256
    policy.update(readiness=readiness.model_dump(mode="json"), readiness_config=readiness_config.model_dump(mode="json"))
    folds = return_folds(sessions, count=4, minimum_train=503, embargo=10)
    training_request = {"schema": "market_predictor.swing_return_training_request", "policy": policy, "source_files": pins,
        "feature_names": names, "input_rows": len(rows), "input_decision_ids_sha256": json_sha256(sorted(rows.decision_id)),
        "holdout_security_ids": sorted(rows.loc[security_transfer_mask(rows.security_id), "security_id"].unique()),
        "folds": [{"number": fold.number, "train": [day.isoformat() for day in fold.train],
            "embargo": [day.isoformat() for day in fold.embargo], "score": [day.isoformat() for day in fold.score]} for fold in folds],
        **dict.fromkeys(FLAGS, False)}
    run = tmp_path / "data/research/saved-run"
    request_pin = _json(run / "_request.json", training_request)
    units = {f"{family}/{scope}/fold-{fold}": {"synthetic_unopened": True}
             for family in ("regularized_linear_return", "shallow_boosted_return") for scope in ("temporal", "security_transfer")
             for fold in range(1, 5)}
    units.update({f"{family}/final_refit": {"synthetic_unopened": True}
                  for family in ("regularized_linear_return", "shallow_boosted_return")})
    for fold in folds:
        frame = rows.loc[rows.session_date_et.isin(fold.score), list(owner._IDENTITIES) + ["feature_eligible"]].copy()
        frame["scope_eligible"] = True
        frame["security_holdout"] = security_transfer_mask(frame.security_id)
        frame["predicted_excess_return"] = -0.01  # Negative scores must still be selected.
        frame["prediction_status"] = "research_prediction"
        directory = run / f"regularized_linear_return/temporal/fold-{fold.number}"
        directory.mkdir(parents=True)
        frame.to_parquet(directory / "predictions.parquet", index=False)
        (directory / "model.joblib").write_bytes(b"synthetic-not-a-pickle")
        cutoff = swing_prediction_cutoffs(pd.Series([fold.score[0]])).iloc[0]
        train = rows.loc[rows.session_date_et.isin(fold.train)]
        unit = {"family": "regularized_linear_return", "scope": "temporal", "parameters": policy["linear"], "feature_names": names,
            "fit_cutoff_utc": cutoff.isoformat(), "maximum_training_label_maturity": (cutoff - pd.Timedelta(days=1)).isoformat(),
            "training_rows": len(train), "training_sessions": len(fold.train), "training_security_ids": sorted(identities),
            "training_decision_ids_sha256": json_sha256(sorted(train.decision_id)), "training_weights_sha256": "a" * 64,
            "scoring_rows": len(frame), "scope_eligible_rows": len(frame),
            "scoring_decision_ids_sha256": json_sha256(sorted(frame.decision_id)),
            "target": policy["target"], "preprocessing": policy["missingness"]}
        unit_pin = _json(directory / "_manifest.json", {"schema": "market_predictor.swing_return_research_model",
            "request_sha256": request_pin.sha256, "unit": unit, "files": {name: file_sha256(directory / name)
                for name in ("model.joblib", "predictions.parquet")}, "metrics": {},
            "serving_eligible": False, "promotion_eligible": False})
        units[f"regularized_linear_return/temporal/fold-{fold.number}"] = {
            "path": directory.relative_to(tmp_path).as_posix(), "manifest_sha256": unit_pin.sha256}
    manifest_pin = _relative(tmp_path, _json(run / "_manifest.json", {"schema": "market_predictor.swing_return_training",
        "status": "complete_research_only", "request_sha256": request_pin.sha256, "units": units,
        **dict.fromkeys(FLAGS, False)}))
    return {"root": tmp_path, "run": run, "run_manifest": manifest_pin, "feature_publication": publication_pin,
            "strategy": strategy, "strategy_pin": contracts["strategy_contract"], "research_pin": contracts["research_contract"],
            "rows": rows, "learner": "regularized_linear_return"}


def _scores(saved: dict[str, Any]) -> owner._SavedScores:
    keys = ("root", "run_manifest", "feature_publication", "strategy_pin", "research_pin", "strategy", "learner")
    return owner._saved_scores(**{name: saved[name] for name in keys}, pins={})


def test_without_proof_saved_strategy_requires_unchanged_pin(saved: dict[str, Any]) -> None:
    saved["strategy_pin"] = SourcePin(path=saved["strategy_pin"].path, sha256="b" * 64)
    with pytest.raises(DataReadinessError, match="saved readiness strategy binding differs"):
        _scores(saved)


@pytest.mark.parametrize("wrong_path", [False, True])
def test_proof_delegation_reproduces_fold_and_holdout_metadata(
    saved: dict[str, Any], monkeypatch: pytest.MonkeyPatch, wrong_path: bool,
) -> None:
    # Unit wiring only: the independent verifier's real byte/rename rejection
    # cases live in test_saved_evaluation_configuration. No real-data claim.
    from market_predictor.swing.datasets.saved_evaluation_configuration import VerifiedSavedEvaluationConfiguration

    root = saved["root"]
    training = json.loads((saved["run"] / "_request.json").read_text())
    readiness = json.loads((root / training["policy"]["readiness_config"]["path"]).read_text())
    temporal = SourcePin.model_validate(readiness["temporal_contract"])
    evidence = {"schema_version": "market_predictor.saved_evaluation_configuration",
        "scope": "saved_initial_fit_evaluation_only", "runs": [saved["run_manifest"].model_dump(mode="json"),
            {"path": "data/research/unopened-unit-run/_manifest.json", "sha256": "b" * 64}],
        "source_configuration_report": {"path": "unit-only-report.json", "sha256": "c" * 64}}
    for role, pin in (("strategy", saved["strategy_pin"]), ("temporal", temporal)):
        evidence[role] = {"original_logical": pin.model_dump(mode="json"), "original_artifact": pin.model_dump(mode="json"),
            "current": pin.model_dump(mode="json"), "git_commit": "d" * 40, "git_blob": "e" * 40}
    proof = _relative(root, _json(root / "unit-configuration-proof.json", evidence))
    calls = []

    def verify(**kwargs: Any) -> VerifiedSavedEvaluationConfiguration:
        calls.append(kwargs)
        return VerifiedSavedEvaluationConfiguration(root / ("wrong.toml" if wrong_path else saved["strategy_pin"].path),
            root / temporal.path, {"unit_case_only": True})

    monkeypatch.setattr(owner, "verify_saved_evaluation_configuration", verify)
    keys = ("root", "run_manifest", "feature_publication", "strategy_pin", "research_pin", "strategy", "learner")
    if wrong_path:
        with pytest.raises(DataReadinessError, match="verified strategy path differs"):
            owner._saved_scores(**{name: saved[name] for name in keys}, pins={}, historical_configuration_evidence=proof)
    else:
        result = owner._saved_scores(**{name: saved[name] for name in keys}, pins={}, historical_configuration_evidence=proof)
        assert result.configuration_provenance is not None
        assert result.configuration_provenance["fold_calendar_sha256"] == json_sha256(training["folds"])
        assert result.configuration_provenance["holdout_security_ids_sha256"] == json_sha256(training["holdout_security_ids"])
        assert result.configuration_provenance["complete_parent_holdout_assignment_reproduced"] is True
    assert len(calls) == 1
    assert calls[0]["original_strategy_pin"] == saved["strategy_pin"]
    assert calls[0]["original_temporal_pin"] == temporal
    assert calls[0]["evidence_pin"] == proof


def _mutate_unit(saved: dict[str, Any], change: Any, *, scores: bool = False) -> None:
    manifest = json.loads((saved["run"] / "_manifest.json").read_text())
    record = manifest["units"]["regularized_linear_return/temporal/fold-1"]
    directory = saved["root"] / record["path"]
    value = json.loads((directory / "_manifest.json").read_text())
    if scores:
        frame = pd.read_parquet(directory / "predictions.parquet")
        change(frame)
        frame.to_parquet(directory / "predictions.parquet", index=False)
        value["files"]["predictions.parquet"] = file_sha256(directory / "predictions.parquet")
    else:
        change(value["unit"])
    record["manifest_sha256"] = _json(directory / "_manifest.json", value).sha256
    saved["run_manifest"] = _relative(saved["root"], _json(saved["run"] / "_manifest.json", manifest))


def test_exact_temporal_scores_without_unpickling_or_transfer_payload(saved: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    import joblib
    monkeypatch.setattr(joblib, "load", lambda *args, **kwargs: pytest.fail("unpickle forbidden"))
    result = _scores(saved)
    assert len(result.score_sessions) == 718 and len(result.entry_sessions) == 708
    assert result.entry_sessions[-1] == "2024-05-13" and result.score_sessions[-1] == "2024-05-28"
    assert len(result.rows) == 718 * 3
    assert "spy_fixed_horizon_excess_return" not in result.rows


@pytest.mark.parametrize("saved", [{"feature_profile": owner.ISSUER_REACTION_PROFILE,
                                   "published_profile": owner.ISSUER_REACTION_PROFILE}], indirect=True)
def test_reaction_saved_run_uses_exact_126_profile_and_same_temporal_population(saved: dict[str, Any]) -> None:
    result = _scores(saved)
    request = json.loads((saved["run"] / "_request.json").read_text())
    assert len(request["feature_names"]) == 126
    assert request["feature_names"][-2:] == list(owner.REACTION_COLUMNS)
    assert len(result.rows) == 718 * 3 and len(result.entry_sessions) == 708
    assert result.rows.feature_profile.eq(owner.ISSUER_REACTION_PROFILE).all()


@pytest.mark.parametrize("field,value", [
    ("scope", "security_transfer"), ("parameters", {}), ("feature_names", ["wrong"]),
    ("fit_cutoff_utc", "2021-07-22T20:15:00Z"), ("maximum_training_label_maturity", "2026-01-01T00:00:00Z"),
    ("scoring_rows", 1), ("scoring_decision_ids_sha256", "f" * 64),
])
def test_unit_scope_parameters_cutoff_maturity_population_poison(saved: dict[str, Any], field: str, value: Any) -> None:
    _mutate_unit(saved, lambda unit: unit.update({field: value}))
    with pytest.raises(DataReadinessError):
        _scores(saved)


@pytest.mark.parametrize("field,value", [
    ("ticker", "WRONG"), ("security_id", "wrong-issuer"), ("sector", "wrong-sector"),
    ("scope_eligible", False), ("predicted_excess_return", np.inf), ("prediction_status", "wrong"),
])
def test_scores_same_ids_cannot_change_parent_identity_or_scope(saved: dict[str, Any], field: str, value: Any) -> None:
    _mutate_unit(saved, lambda frame: frame.__setitem__(field, value), scores=True)
    with pytest.raises(DataReadinessError):
        _scores(saved)


def test_selection_ignores_outcome_flags_and_negative_scores_preserves_tail(saved: dict[str, Any]) -> None:
    result = _scores(saved)
    poisoned = result.rows.assign(spy_fixed_horizon_excess_return=np.nan, fixed_horizon_supervision_available=False,
                                  research_label_mature_at=pd.NaT, training_eligible=False)
    complete, selected, counts = owner._select(poisoned, result.score_sessions, result.entry_sessions, saved["strategy"])
    assert len(selected) == 708 * 3
    assert selected.predicted_excess_return.lt(0).all()
    assert complete.terminal_excluded.sum() == 30
    assert len(counts) == 718 and all(row["selected_rows"] == 0 for row in counts[-10:])


def test_cross_section_gate_and_zero_selection_days_are_retained(saved: dict[str, Any]) -> None:
    result = _scores(saved)
    rows = result.rows.copy()
    first = date.fromisoformat(result.score_sessions[0])
    rows.loc[rows.session_date_et.eq(first), "cross_section_eligible"] = False
    _, selected, counts = owner._select(rows, result.score_sessions, result.entry_sessions, saved["strategy"])
    assert not selected.session_date_et.eq(first).any()
    assert counts[0]["rows"] == 3 and counts[0]["selected_rows"] == 0
    assert len(counts) == 718


def test_maturity_equal_to_original_cutoff_is_rejected(saved: dict[str, Any]) -> None:
    _mutate_unit(saved, lambda unit: unit.update(maximum_training_label_maturity=unit["fit_cutoff_utc"]))
    with pytest.raises(DataReadinessError, match="maturity"):
        _scores(saved)


def test_touched_artifact_mutation_is_rejected_before_publication(saved: dict[str, Any]) -> None:
    pin = saved["feature_publication"]
    path = saved["root"] / pin.path
    path.write_text(path.read_text() + " ", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="changed"):
        _scores(saved)


def _publisher_fixture(saved: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], dict[str, Any]]:
    root = saved["root"]
    package = REPO / "src/market_predictor"
    for name in owner.IMPLEMENTATION_PATHS:
        target = root / "src/market_predictor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package / name, target)
    action = {"records": {}, "unavailable": [], "source_files": {}, "replay": {"synthetic": True}}
    evidence = CorporateActionEvidence("a" * 64, "b" * 64, {}, (), {}, {"synthetic": True}, json_sha256(action))
    target = _relative(root, _json(root / "configs/synthetic-target.json", {"synthetic": True}))
    research = load_swing_research_contract(root / saved["research_pin"].path)
    state: dict[str, Any] = {"lease": False, "lots": [], "benchmarks": [], "after_yield": False}
    original_loader = owner._saved_scores

    def load_inside_lease(**kwargs: Any) -> Any:
        assert state["lease"]
        request = root / "data/research/evaluation/_request.json"
        assert request.is_file()
        return original_loader(**kwargs)

    def lot(decision_id: str) -> FundedLotInput:
        state["lots"].append(decision_id)
        security = state["selected"].set_index("decision_id").loc[decision_id, "security_id"]
        return FundedLotInput(decision_id, security, missing_reasons=("synthetic_unavailable_selected_raw_window",))

    def benchmark(ticker: str) -> FundedLotInput:
        state["benchmarks"].append(ticker)
        return FundedLotInput(f"buy-and-hold:{ticker}", f"benchmark:{ticker}", missing_reasons=("dividend_payment_evidence_unavailable",))

    @contextmanager
    def provider(**kwargs: Any) -> Any:
        assert not state["lease"]
        state["lease"] = True
        try:
            selected, calendar = kwargs["selection_loader"]()
            state["selected"] = selected
            yield SimpleNamespace(config=target, policy=SimpleNamespace(research_contract=saved["research_pin"]), research=research,
                valuation_sessions=owner.swing_valuation_sessions(calendar, 10), selected_ids=tuple(selected.decision_id),
                benchmark_tickers=("QQQ", "SPY", "XLF"), source_files={target.path: target.sha256}, simulation=None,
                lot=lot, benchmark=benchmark)
            assert (root / "data/research/evaluation/_manifest.json").is_file()
            state["after_yield"] = True
        finally:
            state["lease"] = False

    monkeypatch.setattr(owner, "_saved_scores", load_inside_lease)
    monkeypatch.setattr(owner, "verified_funded_policy_inputs", provider)
    monkeypatch.setattr(owner, "evaluate_event_aware_swing_accounting", lambda *args, **kwargs: pytest.fail("cannot evaluate missing lots"))
    arguments = {"root": root, "run_manifest": saved["run_manifest"], "feature_publication": saved["feature_publication"],
                 "strategy_contract": saved["strategy_pin"], "research_contract": saved["research_pin"], "target_config": target,
                 "learner": "regularized_linear_return", "exit_policy": "target_stop_ten_session_timeout", "evidence": evidence,
                 "output": root / "data/research/evaluation"}
    return arguments, state


def test_publication_uses_one_provider_lease_and_preserves_every_selected_gap(
    saved: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, state = _publisher_fixture(saved, monkeypatch)
    report = owner.publish_saved_oof_policy_evaluation(**arguments)
    assert state["after_yield"] and not state["lease"]
    assert report["status"] == "valuation_unavailable" and report["accounting"]["comparisons"] is None
    assert report["selected_rows"] == len(state["lots"]) == 708 * 3
    assert report["terminal_excluded_rows"] == 30
    assert len(report["sessions"]) == 718
    assert set(state["benchmarks"]) == {"SPY", "QQQ", "XLF"}
    assert len(report["accounting"]["missing_inputs"]) == 708 * 3 + 3
    assert all(report[name] is False for name in ("training_eligible", "serving_eligible", "promotion_eligible", "economic_eligible"))


@pytest.mark.parametrize("mutation", ["evidence", "resampling_implementation"])
def test_mutation_inside_provider_prevents_complete_report(
    saved: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    arguments, state = _publisher_fixture(saved, monkeypatch)
    evidence = arguments["evidence"]
    original_recheck = evidence.recheck
    checks = 0

    def recheck(root: Path) -> None:
        nonlocal checks
        checks += 1
        if checks > 1:
            if mutation == "evidence":
                evidence.records_by_symbol["synthetic"] = {}
            else:
                implementation = root / "src/market_predictor/modeling/resampling.py"
                implementation.write_text(implementation.read_text() + "\n# synthetic mutation\n", encoding="utf-8")
        original_recheck(root)

    monkeypatch.setattr(CorporateActionEvidence, "recheck", lambda self, root: recheck(root))
    with pytest.raises(DataReadinessError, match="changed"):
        owner.publish_saved_oof_policy_evaluation(**arguments)
    assert state["lots"]
    assert not (arguments["output"] / "_manifest.json").exists()


def _available_lot(decision: dict[str, Any], valuation: tuple[str, ...]) -> tuple[FundedLotInput, Any]:
    """Synthetic flat raw observations, compiled by the real exit/sale owners."""
    case = exit_case.__wrapped__()
    assert decision["session_date_et"] == case["decision_session"]
    identity, identifier = decision["security_id"], decision["decision_id"]
    refs = {day: ref.model_copy(update={"record_locator": f"security_id={identity};session_date={day}"})
            for day, ref in case["evidence"].items()}
    original = case["specification"]
    spec = HoldingSpecification.model_validate({**original.model_dump(), "decision_id": identifier,
        "security_id": identity, "sector": decision["sector"], "initial_position_id": identifier,
        "entry_evidence": (refs[next(iter(refs))],), "marks": tuple(mark.model_copy(update={
            "position_id": identifier, "evidence": (refs[day],)})
            for day, mark in zip(refs, original.marks, strict=True))})
    atr = case["decision_atr"].model_copy(update={"decision_id": identifier, "security_id": identity})
    result = compile_frozen_exit(**{**case, "specification": spec, "decision_atr": atr,
        "observations": case["observations"].assign(security_id=identity, ticker=decision["ticker"]), "evidence": refs})
    managed = unsimulated_managed_specification(result)
    ends = tuple(CALENDAR.session_close(day).to_pydatetime() for day in valuation
                 if day >= managed.initial_entry_timestamp.date().isoformat())
    extended = HoldingSpecification.model_validate({**managed.model_dump(), "session_end_timestamps": ends})
    return FundedLotInput(identifier, identity, extended, atr, result), case["simulation"]


@pytest.mark.parametrize("generated", ["event", "mark"])
def test_generated_settlement_is_rejected_but_canonical_sale_and_calendar_extension_are_allowed(generated: str) -> None:
    case = exit_case.__wrapped__()
    decision = {"decision_id": "synthetic-sale", "security_id": "synthetic-issuer", "ticker": "SYN",
                "sector": "synthetic-sector", "session_date_et": case["decision_session"]}
    valuation = tuple(day.date().isoformat() for day in CALENDAR.sessions_in_range("2024-01-03", "2024-05-28"))
    lot, _ = _available_lot(decision, valuation)
    assert owner._unsimulated_lot(lot, valuation) is lot.specification
    assert lot.specification is not None and lot.exit_result is not None and lot.exit_result.simulation is not None
    assert len([event for event in lot.specification.events if isinstance(event, ExecutionEvent)]) == 1
    replay = lot.exit_result.simulation
    if generated == "event":
        event = next(event for event in replay.specification.events if event.event_id in replay.generated_event_ids)
        spec = lot.specification.model_copy(update={"events": (*lot.specification.events, event)})
    else:
        mark = next(mark for mark in replay.specification.marks
                    if (mark.position_id, mark.mark_at) in replay.generated_mark_keys)
        spec = lot.specification.model_copy(update={"marks": (*lot.specification.marks, mark)})
    poisoned = FundedLotInput(lot.decision_id, lot.canonical_security_id, spec, lot.decision_atr, lot.exit_result)
    with pytest.raises(DataReadinessError, match="generated settlement"):
        owner._unsimulated_lot(poisoned, valuation)


@pytest.mark.parametrize("saved", ["2024-01-02"], indirect=True)
def test_available_publication_calls_real_accounting_once_and_settles_each_sale_once(
    saved: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    arguments, state = _publisher_fixture(saved, monkeypatch)
    research = load_swing_research_contract(saved["root"] / saved["research_pin"].path)
    calls = 0

    def accounting(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        assert state["lease"]
        for spec in args[1]:
            assert len([event for event in spec.events if isinstance(event, ExecutionEvent)]) == 1
            assert not any(isinstance(event, PaymentEvent) for event in spec.events)
            replay = simulate_ordinary_sales(spec, kwargs["simulation"])
            assert len([event for event in replay.specification.events if isinstance(event, PaymentEvent)]) == 1
            assert replay.outcome.snapshots[-1].net_return == pytest.approx(-0.002)
            assert replay.outcome.snapshots[-1].available_cash == pytest.approx(1.0)
        return evaluate_event_aware_swing_accounting(*args, **kwargs)

    @contextmanager
    def provider(**kwargs: Any) -> Any:
        assert not state["lease"]
        state["lease"] = True
        try:
            selected, calendar = kwargs["selection_loader"]()
            assert len(selected) == 3 and selected.session_date_et.eq(date(2024, 1, 2)).all()
            valuation = owner.swing_valuation_sessions(calendar, 10)
            lots = {}
            simulation = None
            for decision in selected.to_dict("records"):
                lot, simulation = _available_lot(decision, valuation)
                lots[lot.decision_id] = lot
            first = next(iter(lots.values())).specification
            assert first is not None
            closes = tuple(CALENDAR.session_close(day).to_pydatetime() for day in valuation)
            benchmarks = {}
            for ticker in ("SPY", "QQQ", "XLF"):
                position = f"synthetic-buy-and-hold:{ticker}"
                spec = HoldingSpecification.model_validate({**first.model_dump(), "decision_id": position,
                    "security_id": ticker, "sector": "benchmark", "initial_position_id": position,
                    "initial_entry_timestamp": CALENDAR.session_open(valuation[0]).to_pydatetime(),
                    "session_end_timestamps": closes, "policy": "fixed_horizon", "cost_prepaid_fraction": 0.0,
                    "events": (), "marks": tuple(KnownMark(position_id=position, mark_at=stamp, value_per_unit=100.0,
                        currency="USD", evidence=first.entry_evidence) for stamp in closes)})
                benchmarks[ticker] = FundedLotInput(position, f"benchmark:{ticker}", spec)
            yield SimpleNamespace(config=arguments["target_config"],
                policy=SimpleNamespace(research_contract=saved["research_pin"]), research=research,
                valuation_sessions=valuation, selected_ids=tuple(selected.decision_id), benchmark_tickers=tuple(benchmarks),
                source_files={}, simulation=simulation, lot=lots.__getitem__, benchmark=benchmarks.__getitem__)
            assert (arguments["output"] / "_manifest.json").is_file()
        finally:
            state["lease"] = False

    monkeypatch.setattr(owner, "verified_funded_policy_inputs", provider)
    monkeypatch.setattr(owner, "evaluate_event_aware_swing_accounting", accounting)
    report = owner.publish_saved_oof_policy_evaluation(**arguments)
    assert calls == 1 and not state["lease"]
    assert report["selected_rows"] == 3 and len(report["sessions"]) == 718
    ledger = report["accounting"]["base_ledger"]
    assert ledger["funded_trades"] == 3 and ledger["fully_settled"]
    assert ledger["total_cost"] == pytest.approx(0.1 * 0.002)
    assert ledger["compounded_return"] == pytest.approx(-ledger["total_cost"])
    assert report["accounting"]["stress_ledger"]["total_cost"] == pytest.approx(2 * ledger["total_cost"])
    assert report["economic_eligible"] is False
