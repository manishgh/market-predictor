from __future__ import annotations

import unittest
from pathlib import Path

from market_predictor.api import create_app
from market_predictor.cli import app


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_predictor_has_no_runtime_alert_module_or_cli_commands(self) -> None:
        package_root = Path(__file__).resolve().parents[1] / "src" / "market_predictor"
        self.assertFalse((package_root / "alerts.py").exists())

        command_names = {command.name for command in app.registered_commands}
        self.assertNotIn("monitor-alerts", command_names)
        self.assertNotIn("backtest-alerts", command_names)
        obsolete_prediction_commands = {
            "behavior",
            "build-dataset",
            "build-event-swing-datasets",
            "combine-event-swing-datasets",
            "live-once",
            "live-run",
            "live-train-event",
            "negative-reaction",
            "predict",
            "predict-watchlist",
            "rank-swing",
            "score-event-swing",
            "score-events",
            "score-swing",
            "train",
            "train-event-swing",
            "watch",
        }
        self.assertTrue(obsolete_prediction_commands.isdisjoint(command_names))
        self.assertIn("rank-sector-themes", command_names)

    def test_prediction_api_exposes_no_alert_or_execution_routes(self) -> None:
        route_paths = {route.path.lower() for route in create_app().routes}
        forbidden_tokens = ("alert", "notification", "order", "broker", "position")

        for path in route_paths:
            self.assertFalse(
                any(token in path for token in forbidden_tokens),
                f"Market Predictor route crosses the architecture boundary: {path}",
            )


if __name__ == "__main__":
    unittest.main()
