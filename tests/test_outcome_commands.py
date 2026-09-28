from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from click.utils import strip_ansi
from typer.testing import CliRunner

from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.production_cli import app
from tests.test_outcome_repository import _intent


class OutcomeCommandTests(unittest.TestCase):
    def test_monitoring_commands_are_registered(self) -> None:
        runner = CliRunner()
        invoke_options = {"color": False, "terminal_width": 240}

        report_help = runner.invoke(
            app,
            ["build-outcome-performance-report", "--help"],
            **invoke_options,
        )
        drift_help = runner.invoke(
            app,
            ["publish-drift-assessment", "--help"],
            **invoke_options,
        )

        self.assertEqual(report_help.exit_code, 0, report_help.output)
        self.assertIn("--minimum-samples", strip_ansi(report_help.output))
        self.assertEqual(drift_help.exit_code, 0, drift_help.output)
        self.assertIn("--model-release-id", strip_ansi(drift_help.output))
        collect_help = runner.invoke(app, ["collect-outcome-bars", "--help"], **invoke_options)
        self.assertEqual(collect_help.exit_code, 0, collect_help.output)
        self.assertIn("--receipt-dir", strip_ansi(collect_help.output))

    def test_maturation_fails_while_a_registration_is_unfinished(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            intent = _intent()
            OutcomeRepository(root / "outcomes").record_intent(intent)
            session = intent.decision_session_et.isoformat()
            (root / "outcomes" / "sessions" / session / "semantic" / f"{intent.semantic_prediction_id}.json").unlink()

            with mock.patch.dict(os.environ, {"MARKET_PREDICTOR_RUNTIME_DIR": str(root / "runtime")}):
                result = CliRunner().invoke(
                    app,
                    [
                        "mature-outcomes",
                        "--outcome-dir", str(root / "outcomes"),
                        "--receipt-dir", str(root / "receipts"),
                        "--observed-as-of", "2026-08-08T12:00:00+00:00",
                    ],
                    color=False,
                    terminal_width=240,
                )

            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("rerun them", strip_ansi(result.output))


if __name__ == "__main__":
    unittest.main()
