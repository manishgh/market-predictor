"""CLI adapters for immutable issuer reaction publication and independent row checks."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.issuer_reaction_publication import materialize_issuer_reactions
from market_predictor.swing.datasets.issuer_reaction_verification import verify_issuer_reaction_rows


def register_issuer_reaction_commands(app: typer.Typer) -> None:
    @app.command("materialize-swing-issuer-reactions")
    def materialize(
        config: Path = typer.Option(...), expected_config_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
        expected_checkpoint_sha256: str | None = typer.Option(None),
        maximum_months_this_run: int | None = typer.Option(None, min=1),
    ) -> None:
        """Append reviewed issuer reactions without replacing parent values or labels."""
        try:
            report = materialize_issuer_reactions(root=root, config=config,
                expected_config_sha256=expected_config_sha256, output=output,
                expected_checkpoint_sha256=expected_checkpoint_sha256,
                maximum_months_this_run=maximum_months_this_run)
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        except (DataReadinessError, OSError, ValidationError) as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=2) from error
        typer.echo(json.dumps({name: report[name] for name in (
            "status", "rows", "manifest_sha256", "checkpoint_sha256", "training_eligible",
            "promotion_eligible", "serving_eligible") if name in report}, sort_keys=True))
        if report["status"] != "complete_research_only":
            raise typer.Exit(code=2)

    @app.command("verify-swing-issuer-reactions")
    def verify(
        publication: Path = typer.Option(...), publication_sha256: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
    ) -> None:
        """Verify exact parent values, new causal inputs and retained population."""
        try:
            report = verify_issuer_reaction_rows(root=root,
                publication=SourcePin(path=publication.as_posix(), sha256=publication_sha256), output=output)
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        except (DataReadinessError, OSError, ValidationError) as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=2) from error
        typer.echo(json.dumps({name: report[name] for name in (
            "status", "manifest_sha256", "rows", "report_sha256", "training_eligible",
            "promotion_eligible", "serving_eligible")}, sort_keys=True))
        if report["status"] != "passed":
            raise typer.Exit(code=2)
