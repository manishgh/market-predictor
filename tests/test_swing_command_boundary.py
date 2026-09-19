"""Frozen negative contract for the retired day-trading command surfaces."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from market_predictor.cli import app as combined_app
from market_predictor.cli_surface import command_names
from market_predictor.collection_cli import app as collection_app
from market_predictor.production_cli import app as production_app
from market_predictor.research_cli import app as research_app

RETIRED_COMMANDS = (
    "audit-edge-rebuild-intraday-bar-dataset",
    "audit-edge-rebuild-selected-session-one-minute",
    "audit-intraday-specialist-one-minute-coverage",
    "audit-prediction-data-readiness",
    "audit-v3-data",
    "audit-v3-development-readiness",
    "audit-v3-failure-attribution",
    "audit-v3-o1-overlay",
    "audit-v3-ranking",
    "build-intraday-dataset",
    "build-intraday-decision-report",
    "build-intraday-enriched-dataset",
    "build-intraday-live-features",
    "build-intraday-specialist-acquisition-units",
    "build-intraday-specialist-setups",
    "build-intraday-specialist-training-data",
    "build-intraday-universe",
    "build-v3-development-dataset",
    "build-v3-features",
    "build-v3-labels",
    "collect-edge-intraday-microstructure-history",
    "collect-edge-rebuild-intraday-history",
    "collect-intraday-specialist-one-minute",
    "evaluate-edge-rebuild-intraday-future-holdout",
    "materialize-edge-rebuild-intraday-history",
    "partition-v3-data",
    "plan-edge-intraday-microstructure-history",
    "plan-edge-rebuild-broad-intraday-history",
    "plan-edge-rebuild-extended-session-context",
    "plan-edge-rebuild-intraday-history",
    "plan-edge-rebuild-selected-session-benchmarks",
    "plan-edge-rebuild-selected-session-history",
    "plan-edge-rebuild-selected-session-one-minute",
    "plan-intraday-specialist-one-minute",
    "promote-intraday-model",
    "publish-edge-prospective-analyst-revision-horizon",
    "publish-edge-rebuild-intraday-bar-dataset",
    "publish-edge-rebuild-intraday-event-preflight",
    "publish-edge-rebuild-selected-session-five-minute",
    "screen-edge-rebuild-intraday-universe",
    "train-edge-rebuild-intraday-development",
    "train-intraday-model",
    "train-intraday-specialists",
    "train-v3-models",
)


@pytest.mark.parametrize("app", (combined_app, collection_app, production_app, research_app))
def test_no_retired_commands_or_aliases_are_registered(app) -> None:
    names = command_names(app)
    assert names.isdisjoint(RETIRED_COMMANDS)
    assert not any("intraday" in name or "-v3-" in name for name in names)


@pytest.mark.parametrize("command", RETIRED_COMMANDS)
def test_retired_commands_are_unknown_even_with_help(command: str) -> None:
    result = CliRunner().invoke(combined_app, [command, "--help"])
    assert result.exit_code == 2
    assert "No such command" in result.output


def test_shared_subdaily_evidence_and_swing_training_remain() -> None:
    assert {
        "collect-edge-prospective-sip-session",
        "collect-edge-prospective-broker-actions",
        "collect-edge-sec-filings",
        "collect-alpaca-news-history",
        "collect-sp500-official-source-archive",
        "export-ohlcv-artifacts",
    } <= command_names(collection_app)
    assert {
        "extract-sp500-official-events",
        "train-swing-returns",
        "audit-swing-training-readiness",
        "replay-swing-research-predictors",
    } <= command_names(research_app)
