"""CLI adapter for the strictly initial-fit swing return experiment."""
from pathlib import Path

import typer

from market_predictor.research.swing_return_training import train_swing_returns


def register_swing_return_training_command(app: typer.Typer) -> None:
    @app.command("train-swing-returns")
    def train(root: Path = typer.Option(...), config: Path = typer.Option(...),
        config_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        resume_checkpoint_sha256: str | None = typer.Option(None)) -> None:
        result = train_swing_returns(root=root, config=config, config_sha256=config_sha256, output=output,
            resume_checkpoint_sha256=resume_checkpoint_sha256)
        typer.echo(f"{result['status']}: {result['manifest_sha256']}")
