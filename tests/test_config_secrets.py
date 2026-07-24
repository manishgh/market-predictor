from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from market_predictor.collection_cli import app as collection_app
from market_predictor.config import Settings
from market_predictor.sources.seeking_alpha import SeekingAlphaRapidApiSource


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
            "REDDIT_CLIENT_SECRET": "reddit-secret",
            "REDDIT_PASSWORD": "reddit-password",
            "RAPIDAPI_KEY": "rapid-secret",
            "FINVIZ_ELITE_AUTH": "finviz-secret",
            "SEEKING_ALPHA_ACCOUNT_PASSWORD": "sa-password",
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
        self.assertEqual(settings.rapidapi_key_value, values["RAPIDAPI_KEY"])
        self.assertEqual(
            settings.azure_storage_connection_string_value,
            values["AZURE_STORAGE_CONNECTION_STRING"],
        )

    def test_access_token_cache_is_atomic_and_owner_only_on_posix(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "access_token.json"
            settings = Settings(
                SEEKING_ALPHA_ACCESS_TOKEN_CACHE_FILE=path,
            )
            source = object.__new__(SeekingAlphaRapidApiSource)
            source.settings = settings

            source._write_cached_access_token(  # noqa: SLF001
                "token-value",
                {"access_token": "token-value", "expires": 3600},
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["access_token"], "token-value")
            self.assertEqual(payload["raw_keys"], ["access_token", "expires"])
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            if os.name != "nt":
                self.assertEqual(
                    stat.S_IMODE(path.stat().st_mode),
                    stat.S_IRUSR | stat.S_IWUSR,
                )


if __name__ == "__main__":
    unittest.main()
