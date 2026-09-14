from __future__ import annotations

from market_predictor.cli import app as source_app
from market_predictor.cli_surface import command_names, filtered_app
from market_predictor.collection_cli import COLLECTION_COMMANDS
from market_predictor.commands.swing_research_features import register_research_feature_commands
from market_predictor.commands.swing_return_training import register_swing_return_training_command
from market_predictor.commands.swing_training_readiness import register_training_readiness_command

RESEARCH_COMMANDS = command_names(source_app).difference(COLLECTION_COMMANDS)

app = filtered_app(
    source_app,
    allowed_commands=RESEARCH_COMMANDS,
    help_text="Build, train, audit, and promote Market Predictor research artifacts.",
)

register_research_feature_commands(app)
register_training_readiness_command(app)
register_swing_return_training_command(app)


if __name__ == "__main__":
    app()
