from __future__ import annotations

from market_predictor.cli import app as source_app
from market_predictor.cli_surface import command_names, filtered_app
from market_predictor.collection_cli import COLLECTION_COMMANDS

RESEARCH_COMMANDS = command_names(source_app).difference(COLLECTION_COMMANDS)

app = filtered_app(
    source_app,
    allowed_commands=RESEARCH_COMMANDS,
    help_text="Build, train, audit, and promote Market Predictor research artifacts.",
)


if __name__ == "__main__":
    app()
