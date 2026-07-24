from __future__ import annotations

import unittest

from click import Context, Option
from typer.main import get_command

from market_predictor.production_cli import app


class OutcomeCommandTests(unittest.TestCase):
    def test_monitoring_commands_are_registered(self) -> None:
        root = get_command(app)
        context = Context(root)
        report = root.get_command(context, "build-outcome-performance-report")
        drift = root.get_command(context, "publish-drift-assessment")

        self.assertIsNotNone(report)
        self.assertIsNotNone(drift)
        report_options = {
            option
            for parameter in report.params
            if isinstance(parameter, Option)
            for option in parameter.opts
        }
        drift_options = {
            option
            for parameter in drift.params
            if isinstance(parameter, Option)
            for option in parameter.opts
        }
        self.assertIn("--minimum-samples", report_options)
        self.assertIn("--model-release-id", drift_options)


if __name__ == "__main__":
    unittest.main()
