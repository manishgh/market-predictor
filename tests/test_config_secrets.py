from __future__ import annotations

import unittest

from typer.testing import CliRunner

from market_predictor.collection_cli import app as collection_app
from market_predictor.config import Settings


class ConfigSecretTests(unittest.TestCase):
    def test_finviz_token_is_not_accepted_on_the_command_line(self) -> None:
        result = CliRunner().invoke(
            collection_app,
            ["download-finviz-screeners", "--help"],
        )

        self.assertEqual(result.exit_code, 0)
        self.assertNotIn("--auth", result.output)
        self.assertIn("FINVIZ_ELITE_AUTH", result.output)

    def test_provider_secrets_are_redacted_and_explicitly_unwrapped(self) -> None:
        values = {
            "ALPACA_API_SECRET_KEY": "alpaca-secret",
            "FINVIZ_ELITE_AUTH": "finviz-secret",
            "AZURE_STORAGE_CONNECTION_STRING": "azure-secret",
        }
        settings = Settings(**values)
        serialized = settings.model_dump_json()
        represented = repr(settings)

        for secret in values.values():
            self.assertNotIn(secret, serialized)
            self.assertNotIn(secret, represented)
        self.assertEqual(
            settings.alpaca_api_secret_value,
            values["ALPACA_API_SECRET_KEY"],
        )
        self.assertEqual(
            settings.azure_storage_connection_string_value,
            values["AZURE_STORAGE_CONNECTION_STRING"],
        )


if __name__ == "__main__":
    unittest.main()
