from __future__ import annotations

import unittest
from pathlib import Path

from market_predictor.api import create_app
from market_predictor.api_security import ApiSecurityConfig
from market_predictor.cli_surface import command_names
from market_predictor.production_cli import app as production_app
from market_predictor.research_cli import app as research_app


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_predictor_has_no_runtime_alert_module_or_cli_commands(self) -> None:
        package_root = Path(__file__).resolve().parents[1] / "src" / "market_predictor"
        self.assertFalse((package_root / "alerts.py").exists())
        self.assertFalse((package_root / "volatile.py").exists())

        research_commands = command_names(research_app)
        production_commands = command_names(production_app)
        self.assertNotIn("monitor-alerts", research_commands)
        self.assertNotIn("backtest-alerts", research_commands)
        obsolete_prediction_commands = {
            "behavior",
            "build-dataset",
            "build-event-swing-datasets",
            "build-volatile-dataset",
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
            "score-volatile-latest",
            "train",
            "train-event-swing",
            "train-volatile-model",
            "watch",
        }
        self.assertTrue(obsolete_prediction_commands.isdisjoint(research_commands))
        self.assertTrue({"build-swing-dataset", "train-swing-model", "promote-swing-model"}.issubset(research_commands))
        self.assertTrue(
            {
                "build-intraday-dataset",
                "train-intraday-model",
                "promote-intraday-model",
            }.issubset(research_commands)
        )
        self.assertTrue(
            {
                "build-swing-live-features",
                "build-intraday-live-features",
                "publish-live-features",
            }.issubset(research_commands | production_commands)
        )
        self.assertTrue(
            {
                "azure-publish-serving-release",
                "azure-rollback-serving-release",
                "azure-sync-serving-release",
            }.isdisjoint(research_commands | production_commands)
        )
        self.assertNotIn("azure-publish-models", research_commands)
        self.assertIn("rank-sector-themes", research_commands)
        self.assertFalse((package_root / "deployment.py").exists())

        prediction_service = (package_root / "prediction_service.py").read_text(encoding="utf-8")
        self.assertNotIn("market_predictor.entry_exit", prediction_service)

    def test_prediction_api_exposes_no_alert_or_execution_routes(self) -> None:
        route_paths = {
            route.path.lower()
            for route in create_app(
                security_config=ApiSecurityConfig(mode="disabled")
            ).routes
        }
        forbidden_tokens = ("alert", "notification", "order", "broker", "position")

        for path in route_paths:
            self.assertFalse(
                any(token in path for token in forbidden_tokens),
                f"Market Predictor route crosses the architecture boundary: {path}",
            )

    def test_container_runs_non_root_api_with_liveness_probe(self) -> None:
        root = Path(__file__).resolve().parents[1]
        dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

        self.assertEqual(
            dockerfile.splitlines()[0],
            "FROM python:3.11.15-slim-bookworm@sha256:"
            "b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba",
        )
        self.assertIn("requirements/production.lock", dockerfile)
        self.assertIn("--require-hashes --no-deps", dockerfile)
        self.assertNotIn("pip install --upgrade", dockerfile)
        self.assertNotIn("build-essential", dockerfile)
        self.assertNotIn("pip install .", dockerfile)
        self.assertIn("USER 10001:10001", dockerfile)
        self.assertIn("RUNTIME_MEMORY_BUDGET_GIB=4.0", dockerfile)
        self.assertIn("/v1/health/live", dockerfile)
        self.assertIn(
            'CMD ["python", "-m", "uvicorn", '
            '"market_predictor.api:create_app", "--factory"',
            dockerfile,
        )
        self.assertFalse(
            (root / "scripts" / "container-entrypoint.sh").exists()
        )


if __name__ == "__main__":
    unittest.main()
