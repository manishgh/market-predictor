"""CLI adapters for source-bound, resumable research feature publication."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.predictor_replay import replay_predictor_publication
from market_predictor.swing.datasets.research_dataset import materialize_research_dataset
from market_predictor.swing.datasets.research_features import materialize_research_predictors


def register_research_feature_commands(app: typer.Typer) -> None:
    @app.command("replay-swing-research-predictors")
    def replay_predictors(
        publication: Path = typer.Option(...), publication_sha256: str = typer.Option(...),
        migration_bindings: Path = typer.Option(...), migration_bindings_sha256: str = typer.Option(...),
        implementation_snapshot: Path = typer.Option(...), snapshot_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
        maximum_groups: int | None = typer.Option(None, min=1),
        feature_plan_snapshot: Path | None = typer.Option(None), feature_plan_snapshot_sha256: str | None = typer.Option(None),
    ) -> None:
        """Reconstruct every technical group and month against its original publication."""
        if (feature_plan_snapshot is None) != (feature_plan_snapshot_sha256 is None):
            raise typer.BadParameter("feature-plan snapshot and SHA256 must be supplied together")
        try:
            result = replay_predictor_publication(root=root,
                publication=SourcePin(path=publication.as_posix(), sha256=publication_sha256),
                migration_bindings=SourcePin(path=migration_bindings.as_posix(), sha256=migration_bindings_sha256),
                implementation_snapshot=SourcePin(path=implementation_snapshot.as_posix(), sha256=snapshot_sha256),
                output=output, maximum_groups=maximum_groups,
                feature_plan_snapshot=None if feature_plan_snapshot is None else SourcePin(
                    path=feature_plan_snapshot.as_posix(), sha256=str(feature_plan_snapshot_sha256)))
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        typer.echo(json.dumps({key: result[key] for key in
            ("status", "replay_complete", "training_eligible", "promotion_eligible")}, sort_keys=True))
        if not result["replay_complete"]:
            raise typer.Exit(code=2)

    @app.command("materialize-swing-research-predictors")
    def predictors(
        root: Path = typer.Option(Path(".")),
        config: Path = typer.Option(Path("configs/swing_corrected_research_features.toml")),
        expected_config_sha256: str = typer.Option(...),
        output: Path = typer.Option(Path("data/features/swing_corrected_initial_fit_predictors")),
        expected_checkpoint_sha256: str | None = typer.Option(None),
        maximum_groups_this_run: int | None = typer.Option(None, min=1),
    ) -> None:
        """Rebuild predictors from saved prices, preserving all frozen decisions."""
        try:
            result = materialize_research_predictors(root=root, config=config,
                expected_config_sha256=expected_config_sha256, output=output,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                maximum_groups_this_run=maximum_groups_this_run)
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        summary = {key: result[key] for key in ("status", "rows", "feature_eligible_rows",
            "checkpoint_sha256", "manifest_sha256", "training_eligible", "promotion_eligible") if key in result}
        summary.update(groups=len(result["groups"]), failed_groups=len(result["failed_groups"]), months=len(result["months"]))
        typer.echo(json.dumps(summary, sort_keys=True))
        if result["status"] != "technical_inputs_complete_research_only":
            raise typer.Exit(code=2)

    @app.command("materialize-swing-research-dataset")
    def dataset(
        decision_config: Path = typer.Option(...),
        decision_config_sha256: str = typer.Option(...),
        strategy_config: Path = typer.Option(...),
        strategy_config_sha256: str = typer.Option(...),
        predictor_manifest: Path = typer.Option(...),
        predictor_manifest_sha256: str = typer.Option(...),
        outcome_manifest: Path = typer.Option(...),
        outcome_manifest_sha256: str = typer.Option(...),
        catalyst_manifest: Path = typer.Option(...),
        catalyst_manifest_sha256: str = typer.Option(...),
        predictor_replay: Path | None = typer.Option(None), predictor_replay_sha256: str | None = typer.Option(None),
        outcome_replay: Path | None = typer.Option(None), outcome_replay_sha256: str | None = typer.Option(None),
        root: Path = typer.Option(Path(".")),
        output: Path = typer.Option(Path("data/features/swing_corrected_initial_fit_research")),
        expected_checkpoint_sha256: str | None = typer.Option(None),
    ) -> None:
        """Publish exact technical/news ablations with independently corrected returns."""
        for path, digest in ((predictor_replay, predictor_replay_sha256), (outcome_replay, outcome_replay_sha256)):
            if (path is None) != (digest is None):
                raise typer.BadParameter("replay manifest and SHA256 must be supplied together")
        try:
            result = materialize_research_dataset(root=root,
                decision_config=SourcePin(path=decision_config.as_posix(), sha256=decision_config_sha256),
                strategy_config=SourcePin(path=strategy_config.as_posix(), sha256=strategy_config_sha256),
                predictors=SourcePin(path=predictor_manifest.as_posix(), sha256=predictor_manifest_sha256),
                outcomes=SourcePin(path=outcome_manifest.as_posix(), sha256=outcome_manifest_sha256),
                catalysts=SourcePin(path=catalyst_manifest.as_posix(), sha256=catalyst_manifest_sha256),
                output=output, expected_checkpoint_sha256=expected_checkpoint_sha256,
                predictor_replay=None if predictor_replay is None else SourcePin(
                    path=predictor_replay.as_posix(), sha256=str(predictor_replay_sha256)),
                outcome_replay=None if outcome_replay is None else SourcePin(
                    path=outcome_replay.as_posix(), sha256=str(outcome_replay_sha256)))
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        typer.echo(json.dumps({key: result[key] for key in ("status", "rows", "training_eligible",
            "promotion_eligible", "manifest_sha256", "checkpoint_sha256")}, sort_keys=True))
