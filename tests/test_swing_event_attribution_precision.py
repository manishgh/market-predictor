from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.swing.event_attribution import build_event_security_relations

_EVENT_TIME = pd.Timestamp("2026-01-20T15:00:00Z")


class SwingEventAttributionPrecisionTests(unittest.TestCase):
    def test_common_short_ticker_words_do_not_match_bare_prose(self) -> None:
        cases = (
            (
                "security:it",
                "IT",
                "Gartner",
                "It remains unclear whether demand will recover",
            ),
            (
                "security:on",
                "ON",
                "ON Semiconductor",
                "Management remains focused on reducing inventory",
            ),
        )
        for security_id, ticker, company, title in cases:
            with self.subTest(ticker=ticker):
                relations = build_event_security_relations(
                    _event(
                        security_id=security_id,
                        ticker=ticker,
                        title=title,
                    ),
                    _labels(
                        _label(
                            security_id=security_id,
                            ticker=ticker,
                            company=company,
                        )
                    ),
                )

                self.assertTrue(relations.empty)

    def test_explicit_short_ticker_forms_match(self) -> None:
        for title in (
            "$IT raises its full-year outlook",
            "Gartner (IT) raises its full-year outlook",
            "NASDAQ:IT company raises its full-year outlook",
        ):
            with self.subTest(title=title):
                relations = build_event_security_relations(
                    _event(
                        security_id="security:it",
                        ticker="IT",
                        title=title,
                    ),
                    _labels(
                        _label(
                            security_id="security:it",
                            ticker="IT",
                            company="Gartner",
                        )
                    ),
                )

                self.assertEqual(
                    relations["relation_channel"].tolist(),
                    ["direct_issuer"],
                )
                self.assertIn(
                    "ticker_text",
                    relations.loc[0, "relation_basis"],
                )

    def test_ambiguous_longer_tickers_require_explicit_notation(
        self,
    ) -> None:
        for ticker in ("APP", "ALL", "FOR", "NOW", "ARE"):
            with self.subTest(ticker=ticker):
                bare = build_event_security_relations(
                    _event(
                        security_id=f"security:{ticker.lower()}",
                        ticker=ticker,
                        title=f"{ticker} investors should review the market",
                    ),
                    _labels(
                        _label(
                            security_id=f"security:{ticker.lower()}",
                            ticker=ticker,
                            company="Unrelated Company",
                        )
                    ),
                )
                explicit = build_event_security_relations(
                    _event(
                        security_id=f"security:{ticker.lower()}",
                        ticker=ticker,
                        title=f"${ticker} raises its outlook",
                    ),
                    _labels(
                        _label(
                            security_id=f"security:{ticker.lower()}",
                            ticker=ticker,
                            company="Unrelated Company",
                        )
                    ),
                )

                self.assertTrue(bare.empty)
                self.assertEqual(
                    explicit["relation_channel"].tolist(),
                    ["direct_issuer"],
                )

    def test_generic_company_token_does_not_establish_identity(self) -> None:
        relations = build_event_security_relations(
            _event(
                security_id="security:ibm",
                ticker="IBM",
                title="The business outlook remains uncertain",
            ),
            _labels(
                _label(
                    security_id="security:ibm",
                    ticker="IBM",
                    company="International Business Machines",
                )
            ),
        )

        self.assertTrue(relations.empty)

    def test_ambiguous_single_company_names_require_full_legal_name(
        self,
    ) -> None:
        target_labels = _labels(
            _label(
                security_id="security:tgt",
                ticker="TGT",
                company="Target Corporation",
            )
        )
        target_prose = build_event_security_relations(
            _event(
                security_id="security:tgt",
                ticker="TGT",
                title="Analysts raise the price target",
            ),
            target_labels,
        )
        target_full_name = build_event_security_relations(
            _event(
                security_id="security:tgt",
                ticker="TGT",
                title="Target Corporation raises its outlook",
            ),
            target_labels,
        )
        pool_prose = build_event_security_relations(
            _event(
                security_id="security:pool",
                ticker="POOL",
                title="The liquidity pool remains stable",
            ),
            _labels(
                _label(
                    security_id="security:pool",
                    ticker="POOL",
                    company="Pool",
                )
            ),
        )

        self.assertTrue(target_prose.empty)
        self.assertEqual(
            target_full_name["relation_channel"].tolist(),
            ["direct_issuer"],
        )
        self.assertTrue(pool_prose.empty)

    def test_full_normalized_company_name_establishes_identity(self) -> None:
        relations = build_event_security_relations(
            _event(
                security_id="security:ibm",
                ticker="IBM",
                title=("International, Business Machines announces quarterly results"),
            ),
            _labels(
                _label(
                    security_id="security:ibm",
                    ticker="IBM",
                    company="International Business Machines",
                )
            ),
        )

        self.assertEqual(
            relations["relation_channel"].tolist(),
            ["direct_issuer"],
        )
        self.assertIn("company_text", relations.loc[0, "relation_basis"])

    def test_current_label_cannot_attribute_historical_news(self) -> None:
        relations = build_event_security_relations(
            _event(
                security_id="security:market",
                ticker="MARKET",
                title="Enterprise storage demand accelerates",
                feature_available_at=pd.Timestamp("2024-06-10T15:00:00Z"),
            ),
            _labels(
                _label(
                    security_id="security:stx",
                    ticker="STX",
                    company="Seagate Technology",
                    match_terms=["enterprise storage"],
                    effective_from=pd.Timestamp("2020-01-01T00:00:00Z"),
                    available_at=pd.Timestamp("2026-07-26T00:00:00Z"),
                )
            ),
        )

        self.assertTrue(relations.empty)

    def test_company_identity_works_without_business_tags(self) -> None:
        empty_labels = _labels(
            _label(
                security_id="security:placeholder",
                ticker="ZZZZ",
                company="Placeholder Company",
            )
        ).iloc[0:0]
        identities = pd.DataFrame(
            {
                "security_id": ["security:unknown"],
                "ticker": ["UNK"],
                "company": ["Unknown Industries"],
                "effective_from_utc": [
                    pd.Timestamp("2025-01-01T00:00:00Z")
                ],
                "effective_to_utc": [pd.NaT],
                "available_at_utc": [
                    pd.Timestamp("2025-01-01T00:00:00Z")
                ],
            }
        )

        relations = build_event_security_relations(
            _event(
                security_id="security:unknown",
                ticker="UNK",
                title="Unknown Industries reports quarterly results",
            ),
            empty_labels,
            identities,
        )

        self.assertEqual(
            relations["relation_channel"].tolist(),
            ["direct_issuer"],
        )
        self.assertIn(
            "company_text",
            relations.loc[0, "relation_basis"],
        )


def _event(
    *,
    security_id: str,
    ticker: str,
    title: str,
    feature_available_at: pd.Timestamp = _EVENT_TIME,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["precision_event_0001"],
            "security_id": [security_id],
            "ticker": [ticker],
            "feature_available_at_utc": [feature_available_at],
            "title": [title],
            "summary": [""],
            "text": [""],
        }
    )


def _labels(*records: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _label(
    *,
    security_id: str,
    ticker: str,
    company: str,
    match_terms: list[str] | None = None,
    effective_from: pd.Timestamp = pd.Timestamp("2025-01-01T00:00:00Z"),
    available_at: pd.Timestamp = pd.Timestamp("2025-01-01T00:00:00Z"),
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "ticker": ticker,
        "company": company,
        "business_tag": "enterprise_storage",
        "label_type": "offering",
        "match_terms": match_terms or ["enterprise storage"],
        "tag_rank": 1,
        "confidence": 0.9,
        "relation_use": "exposure",
        "effective_from_utc": effective_from,
        "effective_to_utc": None,
        "available_at_utc": available_at,
    }


if __name__ == "__main__":
    unittest.main()
