from __future__ import annotations

import unittest

from pydantic import ValidationError

from market_predictor.v3 import ML_V3_SCHEMA_VERSION, FrozenContract, SchemaIdentity
from market_predictor.v3.errors import DataReadinessError, MarketPredictorError


class V3FoundationTests(unittest.TestCase):
    def test_schema_identity_is_normalized_strict_and_immutable(self) -> None:
        identity = SchemaIdentity(name=" Decision_Row ")

        self.assertEqual(identity.name, "decision_row")
        self.assertEqual(identity.version, ML_V3_SCHEMA_VERSION)
        with self.assertRaises(ValidationError):
            SchemaIdentity(name="decision-row")
        with self.assertRaises(ValidationError):
            SchemaIdentity(name="decision_row", unexpected=True)  # type: ignore[call-arg]
        with self.assertRaises(ValidationError):
            identity.name = "changed"  # type: ignore[misc]

    def test_domain_errors_share_one_expected_base(self) -> None:
        self.assertTrue(issubclass(DataReadinessError, MarketPredictorError))
        self.assertTrue(issubclass(SchemaIdentity, FrozenContract))


if __name__ == "__main__":
    unittest.main()
