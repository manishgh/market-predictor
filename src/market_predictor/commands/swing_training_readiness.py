"""CLI adapter for the fixed-horizon input audit."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.research.swing_training_readiness import audit_swing_training_readiness


def register_training_readiness_command(app: typer.Typer) -> None:
    @app.command("audit-swing-training-readiness")
    def audit(
        config: Path = typer.Option(Path("configs/swing_training_readiness.json")),
        config_sha256: str = typer.Option(...), output: Path = typer.Option(...),
        root: Path = typer.Option(Path(".")),
    ) -> None:
        """Verify saved training inputs without fitting or granting production permission."""
        try:
            report = audit_swing_training_readiness(root=root, config=config, config_sha256=config_sha256, output=output)
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        typer.echo(json.dumps({key: report[key] for key in
            ("status", "sessions", "securities", "profiles", "training_eligible", "report_sha256")}, sort_keys=True))
