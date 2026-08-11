from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from typer.main import get_command
from typer.testing import CliRunner

from market_predictor.cli_surface import command_names
from market_predictor.collection_cli import app as collection_app
from market_predictor.production_cli import app as production_app
from market_predictor.research_cli import app as research_app


class CliSurfaceTests(unittest.TestCase):
    def test_global_event_commands_have_correct_surfaces_and_options(self) -> None:
        runner = CliRunner()

        collection_help = runner.invoke(
            collection_app,
            ["collect-edge-live-global-context", "--help"],
            terminal_width=240,
        )
        self.assertEqual(collection_help.exit_code, 0, collection_help.output)

        authority_help = runner.invoke(
            research_app,
            ["publish-edge-global-event-authority", "--help"],
            terminal_width=240,
        )
        self.assertEqual(authority_help.exit_code, 0, authority_help.output)

        collection_command = get_command(collection_app).commands["collect-edge-live-global-context"]
        collection_options = {option for parameter in collection_command.params for option in getattr(parameter, "opts", ())}
        self.assertTrue({"--start", "--end", "--out-dir"}.issubset(collection_options))

        authority_command = get_command(research_app).commands["publish-edge-global-event-authority"]
        authority_options = {option for parameter in authority_command.params for option in getattr(parameter, "opts", ())}
        self.assertTrue(
            {
                "--decisions",
                "--event-artifact",
                "--coverage-artifact",
                "--required-source",
            }.issubset(authority_options)
        )

        catalyst_command = get_command(research_app).commands["publish-edge-catalyst-authority"]
        catalyst_options = {option for parameter in catalyst_command.params for option in getattr(parameter, "opts", ())}
        self.assertTrue({"--lineage-dir", "--out-dir", "--production-ready"}.issubset(catalyst_options))

    def test_global_collection_rejects_invalid_window_before_model_load(self) -> None:
        result = CliRunner().invoke(
            collection_app,
            [
                "collect-edge-live-global-context",
                "--start",
                "2026-08-01T10:00:00Z",
                "--end",
                "2026-08-01T09:00:00Z",
                "--out-dir",
                "unused-invalid-gdelt-output",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("reversed", result.output)

    def test_sec_commands_are_split_between_collection_and_research(self) -> None:
        runner = CliRunner()
        collection_help = runner.invoke(
            collection_app,
            ["collect-edge-sec-filings", "--help"],
            terminal_width=240,
        )
        authority_help = runner.invoke(
            research_app,
            ["publish-edge-sec-filing-authority", "--help"],
            terminal_width=240,
        )
        self.assertEqual(collection_help.exit_code, 0, collection_help.output)
        self.assertEqual(authority_help.exit_code, 0, authority_help.output)
        self.assertIn("--identity-relations", collection_help.output)
        self.assertIn("--collection-dir", authority_help.output)
        self.assertIn("--identity-relations", authority_help.output)
        self.assertNotIn(
            "publish-edge-sec-filing-authority",
            command_names(collection_app),
        )

    def test_command_surfaces_match_reviewed_inventory(self) -> None:
        inventory_path = Path(__file__).parent / "fixtures" / "cli_command_inventory.json"
        expected = json.loads(inventory_path.read_text(encoding="utf-8"))
        actual = {
            "production": sorted(command_names(production_app)),
            "collection": sorted(command_names(collection_app)),
            "research": sorted(command_names(research_app)),
        }

        self.assertEqual(actual, expected)
        self.assertFalse(set(actual["production"]).intersection(actual["collection"]))
        self.assertFalse(set(actual["production"]).intersection(actual["research"]))
        self.assertFalse(set(actual["collection"]).intersection(actual["research"]))

    def test_production_entrypoint_does_not_import_research_or_collection_graph(self) -> None:
        script = (
            "import json,sys; import market_predictor.production_cli; import market_predictor.api; print(json.dumps(sorted(sys.modules)))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
        )
        modules = set(json.loads(completed.stdout))
        forbidden = {
            "market_predictor.cli",
            "market_predictor.collection_cli",
            "market_predictor.research_cli",
            "market_predictor.sentiment",
            "market_predictor.sources.finviz",
            "market_predictor.intraday.promotion",
            "azure",
            "bs4",
            "torch",
            "transformers",
            "xgboost",
            "yfinance",
        }
        self.assertTrue(
            forbidden.isdisjoint(modules),
            f"production import graph contains forbidden modules: {sorted(forbidden.intersection(modules))}",
        )

    def test_project_exposes_only_explicit_split_entrypoints(self) -> None:
        pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")

        self.assertIn(
            'market-predictor-prod = "market_predictor.production_cli:app"',
            pyproject,
        )
        self.assertIn(
            'market-predictor-collect = "market_predictor.collection_cli:app"',
            pyproject,
        )
        self.assertIn(
            'market-predictor-research = "market_predictor.research_cli:app"',
            pyproject,
        )
        self.assertNotIn('market-predictor = "market_predictor.cli:app"', pyproject)


if __name__ == "__main__":
    unittest.main()
