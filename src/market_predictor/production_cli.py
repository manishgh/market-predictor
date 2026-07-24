from __future__ import annotations

import typer
from rich.console import Console

from market_predictor.commands.outcomes import register_outcome_commands
from market_predictor.commands.production import register_production_commands
from market_predictor.commands.release import register_release_commands

app = typer.Typer(
    help="Operate the bounded Market Predictor production serving surface."
)
console = Console()

register_production_commands(app, console)
register_release_commands(app, console)
register_outcome_commands(app, console)
