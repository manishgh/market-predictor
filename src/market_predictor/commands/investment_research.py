"""Thin command surfaces for immutable investment source and research artifacts."""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError
from market_predictor.research.investment_features import (
    build_investment_features,
    collect_investment_company_facts,
    evaluate_investment_predictions,
)


def _run(action: Callable[[], dict[str, Any]]) -> None:
    try:
        result = action()
    except HeavyJobBusyError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=HEAVY_JOB_BUSY_EXIT_CODE) from error
    except (DataReadinessError, MemoryBudgetError, OSError, ValueError, RuntimeError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps(result, sort_keys=True, allow_nan=False))


def register_investment_collection_commands(app: typer.Typer) -> None:
    @app.command("collect-investment-company-facts")
    def collect(root: Path = typer.Option(...), cik: str = typer.Option(...), security_id: str = typer.Option(...),
                output: Path = typer.Option(...)) -> None:
        """Archive one issuer's SEC financial facts; requires configured SEC_USER_AGENT."""
        _run(lambda: collect_investment_company_facts(root=root, cik=cik, security_id=security_id, output=output))


def register_investment_research_commands(app: typer.Typer) -> None:
    @app.command("build-investment-features")
    def build(root: Path = typer.Option(...), request: Path = typer.Option(...), request_sha256: str = typer.Option(...),
              output: Path = typer.Option(...)) -> None:
        """Build research-only financial snapshots from pinned SEC source and corrected clocks."""
        _run(lambda: build_investment_features(root=root, request=request, request_sha256=request_sha256, output=output))

    @app.command("evaluate-investment-predictions")
    def evaluate(root: Path = typer.Option(...), request: Path = typer.Option(...), request_sha256: str = typer.Option(...),
                 output: Path = typer.Option(...)) -> None:
        """Compare complete paired saved forecasts using mature canonical investment outcomes."""
        _run(lambda: evaluate_investment_predictions(root=root, request=request,
                                                   request_sha256=request_sha256, output=output))
