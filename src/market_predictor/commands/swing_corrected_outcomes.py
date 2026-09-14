"""Offline corrected-outcome publication, checkpoint resume and specification replay."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.corrected_outcomes import materialize_corrected_outcomes
from market_predictor.swing.datasets.outcome_replay import replay_corrected_outcome_publication


def register_corrected_outcome_commands(app: typer.Typer) -> None:
    @app.command("replay-swing-corrected-outcomes")
    def replay_publication(
        config: Path = typer.Option(...), config_sha256: str = typer.Option(...),
        publication: Path = typer.Option(...), publication_sha256: str = typer.Option(...),
        implementation_snapshot: Path = typer.Option(...), snapshot_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
        maximum_months: int | None = typer.Option(None, min=1, max=60),
        expected_checkpoint_sha256: str | None = typer.Option(None),
    ) -> None:
        """Recompute targets under current code without modifying the original publication."""
        try:
            result = replay_corrected_outcome_publication(root=root,
                config=SourcePin(path=config.as_posix(), sha256=config_sha256),
                publication=SourcePin(path=publication.as_posix(), sha256=publication_sha256),
                implementation_snapshot=SourcePin(path=implementation_snapshot.as_posix(), sha256=snapshot_sha256),
                output=output, maximum_months=maximum_months,
                expected_checkpoint_sha256=expected_checkpoint_sha256)
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        typer.echo(json.dumps({key: result[key] for key in ("status", "replay_complete",
            "training_eligible", "promotion_eligible")}, sort_keys=True))
        if not result["replay_complete"]:
            raise typer.Exit(code=2)

    @app.command("materialize-swing-corrected-outcomes")
    def command(root: Path = typer.Option(Path(".")),
        config: Path = typer.Option(Path("configs/swing_corrected_outcomes.toml")),
        expected_config_sha256: str = typer.Option(...),
        output: Path = typer.Option(Path("data/labels/swing_corrected_initial_fit")),
        expected_output_sha256: str | None = typer.Option(None), replay: bool = typer.Option(False),
        maximum_months: int | None = typer.Option(None, min=1, max=60)) -> None:
        """Publish partial research targets; replay recomputes all pinned monthly specifications."""
        try:
            result = materialize_corrected_outcomes(root=root, config=config,
                expected_config_sha256=expected_config_sha256, output=output,
                expected_output_sha256=expected_output_sha256, replay=replay, maximum_months=maximum_months)
        except HeavyJobBusyError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from exc
        typer.echo(json.dumps({key: result[key] for key in ("status", "rows", "stock_source_admitted",
            "fixed_comparisons_complete", "training_eligible", "promotion_eligible", "manifest_sha256",
            "checkpoint_sha256", "run_wall_seconds", "run_resources") if key in result}, sort_keys=True))
