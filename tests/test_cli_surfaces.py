from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from market_predictor.cli_surface import command_names
from market_predictor.collection_cli import app as collection_app
from market_predictor.production_cli import app as production_app
from market_predictor.research_cli import app as research_app


class CliSurfaceTests(unittest.TestCase):
    def test_command_surfaces_match_reviewed_inventory(self) -> None:
        inventory_path = (
            Path(__file__).parent / "fixtures" / "cli_command_inventory.json"
        )
        expected = json.loads(inventory_path.read_text(encoding="utf-8"))
        actual = {
            "production": sorted(command_names(production_app)),
            "collection": sorted(command_names(collection_app)),
            "research": sorted(command_names(research_app)),
        }

        self.assertEqual(actual, expected)
        self.assertFalse(
            set(actual["production"]).intersection(actual["collection"])
        )
        self.assertFalse(set(actual["production"]).intersection(actual["research"]))
        self.assertFalse(set(actual["collection"]).intersection(actual["research"]))

    def test_production_entrypoint_does_not_import_research_or_collection_graph(self) -> None:
        script = (
            "import json,sys; import market_predictor.production_cli; "
            "print(json.dumps(sorted(sys.modules)))"
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
            "market_predictor.sources",
            "market_predictor.swing.promotion",
            "market_predictor.intraday.promotion",
            "torch",
            "transformers",
            "yfinance",
        }
        self.assertTrue(
            forbidden.isdisjoint(modules),
            f"production import graph contains forbidden modules: "
            f"{sorted(forbidden.intersection(modules))}",
        )

    def test_project_exposes_only_explicit_split_entrypoints(self) -> None:
        pyproject = (
            Path(__file__).resolve().parents[1] / "pyproject.toml"
        ).read_text(encoding="utf-8")

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
