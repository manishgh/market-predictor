"""CLI adapters for initial-fit training and read-only retained-run integrity."""
import json
from pathlib import Path

import typer

from market_predictor.research.swing_return_training import train_swing_returns
from market_predictor.swing.training.retained_runs import verify_completed_run_integrity


def register_swing_return_training_command(app: typer.Typer) -> None:
    @app.command("train-swing-returns")
    def train(root: Path = typer.Option(...), config: Path = typer.Option(...),
        config_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        resume_checkpoint_sha256: str | None = typer.Option(None)) -> None:
        result = train_swing_returns(root=root, config=config, config_sha256=config_sha256, output=output,
            resume_checkpoint_sha256=resume_checkpoint_sha256)
        typer.echo(f"{result['status']}: {result['manifest_sha256']}")

    @app.command("verify-retained-swing-run")
    def verify(root: Path = typer.Option(...), directory: Path = typer.Option(...),
        manifest_sha256: str = typer.Option(...), request_sha256: str = typer.Option(...),
        checkpoint_sha256: str = typer.Option(...)) -> None:
        """Check historical file integrity only; never admit a model for reuse."""
        result = verify_completed_run_integrity(root=root, directory=directory, manifest_sha256=manifest_sha256,
            request_sha256=request_sha256, checkpoint_sha256=checkpoint_sha256)
        typer.echo(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
