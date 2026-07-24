from __future__ import annotations

import unittest

from typer.testing import CliRunner

from market_predictor.production_cli import app


class OutcomeCommandTests(unittest.TestCase):
    def test_monitoring_commands_are_registered(self) -> None:
        runner = CliRunner()

        report_help = runner.invoke(
            app,
            ["build-outcome-performance-report", "--help"],
        )
        drift_help = runner.invoke(
            app,
            ["publish-drift-assessment", "--help"],
        )

        self.assertEqual(report_help.exit_code, 0, report_help.output)
        self.assertIn("--minimum-samples", report_help.output)
        self.assertEqual(drift_help.exit_code, 0, drift_help.output)
        self.assertIn("--model-release-id", drift_help.output)


if __name__ == "__main__":
    unittest.main()
