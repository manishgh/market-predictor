"""Publish synthetic ranking units through the same estimator and immutable artifact path."""
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import investment_research
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.investment.diagnostics import PREDICTION
from market_predictor.investment.ranking import within_date_rank_targets
from market_predictor.investment.training import TARGET, investment_fold
from market_predictor.research import investment_feature_ablation, investment_training
from market_predictor.research import investment_ranking as module
from tests.test_investment_diagnostics_publication import write
from tests.test_investment_feature_ablation_publication import prepare


@pytest.mark.parametrize("relocated", [False, True])
def test_frozen_rank_experiment_publishes_dimensionless_scores_and_preserves_references(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch, relocated: bool,
) -> None:
    # The root keeps canonical nested fit paths short enough for Windows.
    base = tmp_path_factory.getbasetemp()
    root = base / ("a" if relocated else "b")
    fixture = root / "data/research"
    value, _ = prepare(fixture, monkeypatch)
    dataset = value["dataset"]
    dataset.features.iloc[:, 0] = pd.factorize(dataset.metadata.security_id)[0].astype(float)
    ablation_input, ablation_output = value["request_pin"], value["output"]
    investment_feature_ablation.run_investment_feature_ablation(root=fixture, request=Path(ablation_input["path"]),
        request_sha256=ablation_input["sha256"], output=ablation_output)
    reference_files = {path: file_sha256(path) for directory in (fixture / "run", ablation_output)
        for path in directory.rglob("*") if path.is_file()}
    ablation_manifest = ablation_output / "_manifest.json"
    request = write(root / "ranking-request.json", {"schema_version": "market_predictor.investment_ranking_request",
        "scope": "retrospective_research", "training_manifest": value["request"]["training_manifest"],
        "training_request": value["request"]["training_request"], "ablation_request": ablation_input,
        "ablation_manifest": {"path": str(ablation_manifest), "sha256": file_sha256(ablation_manifest)}})
    monkeypatch.setattr(module, "_implementation", lambda: {})
    monkeypatch.setattr(module, "guard", lambda *_: None)
    monkeypatch.setattr(investment_training, "guard", lambda *_: None)
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_, **__: nullcontext())
    monkeypatch.setattr(module, "load_investment_dataset", lambda **_: dataset)
    calls = []
    publish = module._publish_fit

    def publish_fit(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[1])
        assert kwargs["ranking_target"] is True and kwargs["family"] == module.FAMILY
        assert kwargs["columns"] == dataset.model_columns[module.TECHNICAL] + dataset.model_columns[module.FULL][124:209]
        return publish(*args, **kwargs)

    monkeypatch.setattr(module, "_publish_fit", publish_fit)
    arguments = {"root": root, "request": Path(request["path"]), "request_sha256": request["sha256"]}
    output = root / "data/research/ranking"
    alternate = base / "c" if relocated else None
    if alternate is not None:
        (root / "data").rename(alternate)
    paths = EvidencePaths(root, alternate)
    result = module.train_investment_ranking(**arguments, output=output, data_root=alternate)
    report = json.loads(paths.read(output / "_manifest.json").read_text())
    assert result["fold_scope_fits"] == len(calls) == len(set(calls)) == 12
    assert result["final_models"] == 0 and result["reused_validation_fits"] == 36
    assert len(report["unit_diagnostics"]) == 48 and len(report["aggregate_diagnostics"]) == 16
    assert all(file_sha256(paths.read(path)) == digest for path, digest in reference_files.items())
    assert not any(".investment-" in path for path in report["source_files"])
    for key, pin in report["new_units"].items():
        assert Path(pin["path"]) == output / key / "_manifest.json"
        path = paths.read(pin["path"])
        assert file_sha256(path) == pin["sha256"]
        unit = json.loads(path.read_text())
        assert unit["schema"] == "market_predictor.investment_ranking_model"
        assert unit["score_units"] == "dimensionless" and unit["score_column"] == module.RANKING_SCORE
        horizon, _, _, scope, year = key.split("/")
        fold = investment_fold(dataset.metadata, int(horizon), int(year), scope)
        train = dataset.metadata.iloc[fold.train_indices]
        digest = json_sha256(list(zip(train.decision_id, within_date_rank_targets(train).tolist(), strict=True)))
        assert unit["training_rank_targets_sha256"] == digest
        saved = pd.read_parquet(path.parent / "predictions.parquet")
        assert PREDICTION not in saved and saved[module.RANKING_SCORE].notna().equals(saved.feature_eligible)
        assert saved.loc[saved.feature_eligible & ~saved.research_eligible, module.RANKING_SCORE].notna().all()
        assert "mse" not in unit["metrics"] and "mae" not in unit["metrics"]
        assert unit["metrics"] == report["unit_diagnostics"][key]
    assert len(report["improvement_checks"]["years"]) == 12
    old_model = next(Path(pin["path"]).parent / "model.joblib" for pin in report["reused_units"].values())
    paths.read(old_model).write_bytes(b"tampered historical model")
    refused = root / "data/research/refused"
    with pytest.raises(DataReadinessError, match="source bytes changed"):
        module.train_investment_ranking(**arguments, output=refused, data_root=alternate)
    assert len(calls) == 12 and not paths.write(refused).exists()


def test_rank_publisher_fits_only_purged_training_targets_despite_outside_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = pd.DataFrame({"signal": [1.0, 2.0, 3.0, 8.0, 4.0, 5.0, 6.0, 7.0]})
    train_indices = np.array([0, 1, 2, 4, 5])
    score_indices = np.array([6, 7])
    metadata = pd.DataFrame({
        "decision_id": [f"d{index}" for index in range(8)], "security_id": [f"s{index}" for index in range(8)],
        "decision_session": pd.to_datetime(["2020-01-31"] * 4 + ["2020-02-28"] * 2 + ["2022-01-31"] * 2).date,
        "horizon_sessions": 252, "feature_eligible": True, "research_eligible": True,
        "label_available_at_utc": pd.Timestamp("2023-01-03T00:00Z"),
        "economic_maturity_at_utc": pd.Timestamp("2023-01-02T00:00Z"),
        TARGET: [-0.2, 0.3, 0.3, -0.7, -0.4, 0.4, -0.1, 0.5],
        "spy_gross_return": 0.05, "qqq_gross_return": 0.07, "sector_gross_return": 0.06,
    })
    metadata["stock_net_return"] = metadata[TARGET] + metadata.spy_gross_return
    fit = investment_training.fit_return_regressor
    calls = []

    def observed_fit(family: str, x: pd.DataFrame, y: Any, weights: Any, parameters: dict[str, Any], **kwargs: Any) -> Any:
        assert family == module.FAMILY and kwargs == {}
        pd.testing.assert_frame_equal(x, features.iloc[train_indices])
        np.testing.assert_allclose(y, [-2 / 3, 1 / 3, 1 / 3, -0.5, 0.5], rtol=0, atol=1e-15)
        np.testing.assert_allclose(weights, [5 / 6, 5 / 6, 5 / 6, 5 / 4, 5 / 4], rtol=0, atol=1e-15)
        calls.append(y.copy())
        return fit(family, x, y, weights, parameters)

    monkeypatch.setattr(investment_training, "guard", lambda *_: None)
    monkeypatch.setattr(investment_training, "fit_return_regressor", observed_fit)
    options = {"features": features, "columns": ("signal",), "family": module.FAMILY,
        "parameters": {"objective": "reg:squarederror", "tree_method": "hist", "n_estimators": 4, "n_jobs": 1, "random_state": 42},
        "train_indices": train_indices, "score_indices": score_indices, "prediction_indices": score_indices,
        "identity": {"profile": module.RANKING_PROFILE}, "request_sha256": "a" * 64,
        "admitted_at": datetime(2024, 1, 1, tzinfo=UTC), "ranking_target": True}
    first, original = investment_training._publish_fit(tmp_path, "original", metadata=metadata, **options)
    poisoned = metadata.copy()
    poisoned.loc[[3, 6, 7], TARGET] = [1e5, 3e5, -3e5]
    poisoned["stock_net_return"] = poisoned[TARGET] + poisoned.spy_gross_return
    second, observed = investment_training._publish_fit(tmp_path, "poisoned", metadata=poisoned, **options)
    assert len(calls) == 2 and original is not None and observed is not None
    assert first["training_rank_targets_sha256"] == second["training_rank_targets_sha256"]
    np.testing.assert_array_equal(original[module.RANKING_SCORE], observed[module.RANKING_SCORE])


def test_benchmark_requires_strict_improvement_and_common_date_support() -> None:
    control = {"declared_rows": 4, "scored_rows": 4, "known_scored_outcomes": 4, "unknown_scored_outcomes": 0,
        "rank_date_support": ["2022-01-31"], "auc_date_support": ["2022-01-31"],
        "mean_date_rank_correlation": 0.2, "auc": {"mean_within_date_auc": 0.6}}
    reports = {profile: dict(control) for profile in module.RANKING_PROFILES}
    candidate = reports[module.RANKING_PROFILE]
    candidate.update(mean_date_rank_correlation=0.3, auc={"mean_within_date_auc": 0.7})
    assert module._ranking_check(reports)["passes"] is True
    reports[module.FULL]["auc"] = {"mean_within_date_auc": 0.7}
    assert module._ranking_check(reports)["passes"] is False
    reports[module.TECHNICAL]["rank_date_support"] = []
    with pytest.raises(DataReadinessError, match="label-supported date"):
        module._ranking_check(reports)


def test_cli_forwards_exact_ranking_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(investment_research, "train_investment_ranking",
        lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    app = typer.Typer()
    investment_research.register_investment_research_commands(app)
    request, output = tmp_path / "request.json", tmp_path / "output"
    storage = tmp_path.with_name("ranking-storage")
    result = CliRunner().invoke(app, ["train-investment-ranking", "--root", str(tmp_path),
        "--request", str(request), "--request-sha256", "a" * 64, "--output", str(output), "--data-root", str(storage)])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "request": request, "request_sha256": "a" * 64, "output": output, "data_root": storage}]
