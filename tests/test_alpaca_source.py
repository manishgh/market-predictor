from __future__ import annotations

import json
import unittest
from datetime import UTC, date, datetime
from unittest.mock import Mock

from market_predictor.config import Settings
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.http import HttpByteResponse


class AlpacaSourceTests(unittest.TestCase):
    def test_prospective_news_page_preserves_exact_http_body(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        payload = {
            "news": [
                {
                    "id": 1,
                    "created_at": "2026-08-15T10:00:00Z",
                    "updated_at": "2026-08-15T10:05:00Z",
                    "headline": "Broker action",
                    "source": "benzinga",
                    "symbols": ["AAPL"],
                }
            ],
            "next_page_token": "next",
        }
        body = json.dumps(payload, sort_keys=True).encode()
        final_url = "https://data.alpaca.markets/v1beta1/news?symbols=AAPL"
        redirect_chain = (
            "https://data.alpaca.markets/news-redirect",
            final_url,
        )
        client = Mock()
        client.get_bytes_with_metadata.return_value = _byte_response(
            body,
            "https://data.alpaca.markets/v1beta1/news",
            final_url=final_url,
            redirect_chain=redirect_chain,
        )
        source.client = client

        page = source.fetch_news_page_observed(
            "AAPL",
            datetime(2026, 8, 15, 9, tzinfo=UTC),
            datetime(2026, 8, 15, 11, tzinfo=UTC),
        )

        self.assertEqual(page.raw_body, body)
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.next_page_token, "next")
        self.assertEqual(page.news[0]["id"], 1)
        self.assertEqual(page.final_url, final_url)
        self.assertEqual(page.redirect_chain, redirect_chain)

    def test_prospective_asset_snapshot_preserves_asset_id_and_http_body(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        payload = [
            {
                "id": "asset-aapl",
                "symbol": "AAPL",
                "status": "active",
                "exchange": "NASDAQ",
                "tradable": True,
                "marginable": True,
            }
        ]
        body = json.dumps(payload, sort_keys=True).encode()
        final_url = "https://api.alpaca.markets/v2/assets?status=active"
        redirect_chain = (
            "https://api.alpaca.markets/assets-redirect",
            final_url,
        )
        client = Mock()
        client.get_bytes_with_metadata.return_value = _byte_response(
            body,
            "https://api.alpaca.markets/v2/assets",
            final_url=final_url,
            redirect_chain=redirect_chain,
        )
        source.client = client

        snapshot = source.fetch_assets_snapshot()

        self.assertEqual(snapshot.raw_body, body)
        self.assertEqual(snapshot.assets.loc[0, "id"], "asset-aapl")
        self.assertNotIn("marginable", snapshot.assets.columns)
        self.assertEqual(snapshot.final_url, final_url)
        self.assertEqual(snapshot.redirect_chain, redirect_chain)

    def test_multi_symbol_bar_page_preserves_tokens_and_rate_headers(
        self,
    ) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        client.get_json_with_headers.return_value = (
            {
                "bars": {
                    "AAPL": [
                        {
                            "t": "2026-07-01T13:30:00Z",
                            "o": 100.0,
                            "h": 101.0,
                            "l": 99.0,
                            "c": 100.5,
                            "v": 1000,
                        }
                    ]
                },
                "next_page_token": "next",
            },
            {"X-RateLimit-Remaining": "199"},
        )
        source.client = client

        page = source.fetch_bars_page(
            ("AAPL", "MSFT"),
            datetime(2026, 7, 1, 13, 30, tzinfo=UTC),
            datetime(2026, 7, 1, 14, 30, tzinfo=UTC),
            timeframe="1Min",
            asof=date(2026, 7, 1),
        )

        self.assertEqual(page.next_page_token, "next")
        self.assertEqual(len(page.bars["AAPL"]), 1)
        self.assertEqual(
            page.response_headers["X-RateLimit-Remaining"],
            "199",
        )
        params = client.get_json_with_headers.call_args.kwargs["params"]
        self.assertEqual(params["symbols"], "AAPL,MSFT")
        self.assertEqual(params["feed"], "sip")
        self.assertEqual(params["adjustment"], "all")
        self.assertEqual(params["sort"], "asc")
        self.assertEqual(params["asof"], "2026-07-01")

    def test_multi_symbol_bar_page_rejects_unexpected_symbol(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        client.get_json_with_headers.return_value = (
            {"bars": {"TSLA": []}, "next_page_token": None},
            {},
        )
        source.client = client

        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            source.fetch_bars_page(
                ("AAPL",),
                datetime(2026, 7, 1, 13, 30, tzinfo=UTC),
                datetime(2026, 7, 1, 14, 30, tzinfo=UTC),
                timeframe="1Min",
            )

    def test_trade_page_preserves_raw_market_identity_and_request_bounds(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        payload = {
            "trades": {
                "AAPL": [
                    {
                        "t": "2026-07-01T13:30:00.123456789Z",
                        "x": "V",
                        "p": 100.25,
                        "s": 25,
                        "c": ["@"],
                        "i": 42,
                        "z": "C",
                    }
                ]
            },
            "next_page_token": "trade-next",
        }
        client.get_json_with_headers.return_value = (
            payload,
            {"X-RateLimit-Remaining": "198"},
        )
        source.client = client
        start = datetime(2026, 7, 1, 13, 30, tzinfo=UTC)
        end = datetime(2026, 7, 1, 13, 31, tzinfo=UTC)

        page = source.fetch_trades_page(
            ("AAPL",),
            start,
            end,
            asof=date(2026, 7, 1),
        )

        self.assertEqual(page.trades["AAPL"][0]["i"], 42)
        self.assertEqual(page.next_page_token, "trade-next")
        self.assertEqual(page.raw_payload, payload)
        params = client.get_json_with_headers.call_args.kwargs["params"]
        self.assertEqual(params["start"], start.isoformat())
        self.assertEqual(params["end"], end.isoformat())
        self.assertEqual(params["feed"], "sip")
        self.assertEqual(params["sort"], "asc")
        self.assertNotIn("adjustment", params)

    def test_quote_page_preserves_nbbo_fields_and_pagination(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        payload = {
            "quotes": {
                "MSFT": [
                    {
                        "t": "2026-07-01T13:30:00.100000000Z",
                        "ax": "Q",
                        "ap": 500.02,
                        "as": 4,
                        "bx": "P",
                        "bp": 500.00,
                        "bs": 7,
                        "c": ["R"],
                        "z": "C",
                    }
                ]
            },
            "next_page_token": None,
        }
        client.get_json_with_headers.return_value = (payload, {})
        source.client = client

        page = source.fetch_quotes_page(
            ("MSFT",),
            datetime(2026, 7, 1, 13, 30, tzinfo=UTC),
            datetime(2026, 7, 1, 13, 31, tzinfo=UTC),
            page_token="quote-page",
        )

        self.assertEqual(page.request_page_token, "quote-page")
        self.assertIsNone(page.next_page_token)
        self.assertEqual(page.quotes["MSFT"][0]["bp"], 500.00)
        params = client.get_json_with_headers.call_args.kwargs["params"]
        self.assertEqual(params["page_token"], "quote-page")

    def test_market_event_page_rejects_unexpected_symbol_and_naive_time(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        client.get_json_with_headers.return_value = (
            {"quotes": {"TSLA": []}, "next_page_token": None},
            {},
        )
        source.client = client
        start = datetime(2026, 7, 1, 13, 30, tzinfo=UTC)
        end = datetime(2026, 7, 1, 13, 31, tzinfo=UTC)

        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            source.fetch_quotes_page(("AAPL",), start, end)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            source.fetch_trades_page(
                ("AAPL",),
                datetime(2026, 7, 1, 13, 30),
                end,
            )

    def test_news_pages_preserve_provider_timestamps_and_page_tokens(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        client.get_json.side_effect = [
            {
                "news": [
                    {
                        "id": 1,
                        "created_at": "2026-07-01T12:00:00Z",
                        "updated_at": "2026-07-01T12:05:00Z",
                        "headline": "First",
                        "source": "benzinga",
                    }
                ],
                "next_page_token": "page-2",
            },
            {
                "news": [
                    {
                        "id": 2,
                        "created_at": "2026-07-01T13:00:00Z",
                        "updated_at": "2026-07-01T13:02:00Z",
                        "headline": "Second",
                        "source": "benzinga",
                    }
                ],
                "next_page_token": None,
            },
        ]
        source.client = client

        events = source.fetch_news(
            "MSFT",
            datetime(2026, 7, 1, tzinfo=UTC),
            datetime(2026, 7, 2, tzinfo=UTC),
        )

        self.assertEqual([event.title for event in events], ["First", "Second"])
        self.assertEqual(events[0].timestamp, datetime(2026, 7, 1, 12, tzinfo=UTC))
        self.assertEqual(
            client.get_json.call_args_list[1].kwargs["params"]["page_token"],
            "page-2",
        )

    def test_news_pagination_rejects_repeated_token(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        client.get_json.return_value = {
            "news": [],
            "next_page_token": "same-token",
        }
        source.client = client

        with self.assertRaisesRegex(RuntimeError, "repeated"):
            source.fetch_news(
                "MSFT",
                datetime(2026, 7, 1, tzinfo=UTC),
                datetime(2026, 7, 2, tzinfo=UTC),
            )

    def test_security_transitions_normalize_renames_and_deduplicate_mergers(self) -> None:
        source = AlpacaSource(Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret"))
        client = Mock()
        client.get_json.return_value = {
            "corporate_actions": {
                "name_changes": [
                    {
                        "id": "rename",
                        "process_date": "2022-02-17",
                        "old_symbol": "VIAC",
                        "new_symbol": "PARA",
                        "old_cusip": "92556H206",
                        "new_cusip": "92556H206",
                    }
                ],
                "cash_mergers": [
                    {
                        "id": "cash",
                        "process_date": "2025-08-07",
                        "effective_date": "2025-08-07",
                        "acquiree_symbol": "PARA",
                        "acquirer_symbol": "PSKY",
                        "acquiree_cusip": "92556H206",
                        "acquirer_cusip": "69932A204",
                    }
                ],
                "stock_mergers": [
                    {
                        "id": "stock",
                        "process_date": "2025-08-07",
                        "effective_date": "2025-08-07",
                        "acquiree_symbol": "PARA",
                        "acquirer_symbol": "PSKY",
                        "acquiree_cusip": "92556H206",
                        "acquirer_cusip": "69932A204",
                    }
                ],
            },
            "next_page_token": None,
        }
        source.client = client

        frame = source.fetch_security_transitions(date(2022, 1, 1), date(2026, 1, 1))

        self.assertEqual(list(frame["old_symbol"]), ["VIAC", "PARA"])
        self.assertEqual(list(frame["new_symbol"]), ["PARA", "PSKY"])
        self.assertEqual(list(frame["identity_continuity"]), [True, False])
        self.assertEqual(list(frame["membership_continuity"]), [True, False])


def _byte_response(
    body: bytes,
    url: str,
    *,
    final_url: str | None = None,
    redirect_chain: tuple[str, ...] = (),
) -> HttpByteResponse:
    return HttpByteResponse(
        body=body,
        requested_url=url,
        final_url=final_url or url,
        redirect_chain=redirect_chain,
        status_code=200,
        retrieved_at_utc=datetime(2026, 8, 15, 12, 0, 1, tzinfo=UTC),
        content_type="application/json",
        content_encoding=None,
        etag=None,
        last_modified=None,
        body_length=len(body),
        sha256="0" * 64,
        body_representation="http_entity_encoded",
        safe_headers=(("content-type", "application/json"),),
    )


if __name__ == "__main__":
    unittest.main()
