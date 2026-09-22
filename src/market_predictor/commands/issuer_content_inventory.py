"""Research CLI adapter for original saved-news content inspection."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.research.issuer_content_inventory import publish_saved_content_inventory
from market_predictor.swing.datasets.initial_fit_issuer_news import LAST_INITIAL_FIT_CUTOFF
from market_predictor.swing.datasets.issuer_news_preparation import FIRST


def register_issuer_content_commands(app: typer.Typer) -> None:
    @app.command("inspect-saved-issuer-content")
    def inspect(
        event_artifact: Path = typer.Option(...), event_sha256: str = typer.Option(...),
        event_manifest: Path = typer.Option(...), manifest_sha256: str = typer.Option(...),
        security_id: str = typer.Option(...), ticker: str = typer.Option(...),
        output: Path = typer.Option(...), root: Path = typer.Option(Path(".")),
        start_utc: str = typer.Option(FIRST.isoformat(), help="Not before the initial-fit issuer-news start."),
        cutoff_utc: str = typer.Option(LAST_INITIAL_FIT_CUTOFF.isoformat(), help="Not after the initial-fit cutoff."),
    ) -> None:
        """Verify one original query's content fields without admitting a model feature."""
        try:
            report = publish_saved_content_inventory(
                root=root, event_artifact=SourcePin(path=event_artifact.as_posix(), sha256=event_sha256),
                event_manifest=SourcePin(path=event_manifest.as_posix(), sha256=manifest_sha256),
                security_id=security_id, ticker=ticker, output=output,
                start_utc=pd.Timestamp(start_utc), cutoff_utc=pd.Timestamp(cutoff_utc),
            )
        except HeavyJobBusyError as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
        except (DataReadinessError, OSError, ValueError) as error:
            typer.echo(str(error), err=True)
            raise typer.Exit(code=2) from error
        typer.echo(json.dumps({key: report[key] for key in (
            "status", "rows", "manifest_sha256", "training_eligible", "serving_eligible")}, sort_keys=True))
