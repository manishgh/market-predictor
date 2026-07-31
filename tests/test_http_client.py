from __future__ import annotations

import unittest
from datetime import UTC
from hashlib import sha256
from io import BytesIO
from typing import Any
from unittest.mock import Mock, patch

import requests

from market_predictor.sources.http import HttpClient, _retry_delay


class HttpClientTests(unittest.TestCase):
    def test_get_bytes_preserves_non_utf8_and_crlf_body(self) -> None:
        body = b"\xff\xfeheader\r\nvalue\x80\r\n"
        client = HttpClient()
        client.session = Mock()
        client.session.get.return_value = _Response(
            200,
            body=body,
            headers={
                "Content-Type": "text/plain; charset=windows-1252",
                "Content-Encoding": "identity",
                "ETag": '"v1"',
                "Last-Modified": "Wed, 30 Jul 2026 20:00:00 GMT",
                "Set-Cookie": "secret=true",
                "X-Untrusted": "ignore-me",
            },
        )

        result = client.get_bytes_with_metadata("https://example.test/raw")

        self.assertEqual(result.body, body)
        self.assertEqual(result.body_length, len(body))
        self.assertEqual(result.sha256, sha256(body).hexdigest())
        self.assertEqual(
            result.content_type,
            "text/plain; charset=windows-1252",
        )
        self.assertEqual(result.content_encoding, "identity")
        self.assertEqual(result.etag, '"v1"')
        self.assertEqual(
            result.last_modified,
            "Wed, 30 Jul 2026 20:00:00 GMT",
        )
        self.assertEqual(result.retrieved_at_utc.tzinfo, UTC)
        self.assertNotIn("set-cookie", dict(result.safe_headers))
        self.assertNotIn("x-untrusted", dict(result.safe_headers))

    def test_get_bytes_records_redirect_metadata(self) -> None:
        first = _Response(
            301,
            url="https://example.test/source",
            headers={"Location": "https://cdn.example.test/archive"},
        )
        final = _Response(
            200,
            body=b"archive",
            url="https://cdn.example.test/archive",
            history=[first],
        )
        client = HttpClient()
        client.session = Mock()
        client.session.get.return_value = final

        result = client.get_bytes_with_metadata(
            "https://example.test/source",
        )

        self.assertEqual(result.requested_url, "https://example.test/source")
        self.assertEqual(result.final_url, "https://cdn.example.test/archive")
        self.assertEqual(
            result.redirect_chain,
            (
                "https://example.test/source",
                "https://cdn.example.test/archive",
            ),
        )
        self.assertEqual(result.status_code, 200)

    def test_get_bytes_records_prepared_url_with_query_parameters(self) -> None:
        response = _Response(200, body=b"archive")
        response.request = requests.Request(
            "GET",
            "https://example.test/archive",
            params={"o": "100", "l": "100"},
        ).prepare()
        client = HttpClient()
        client.session = Mock()
        client.session.get.return_value = response

        result = client.get_bytes_with_metadata(
            "https://example.test/archive",
            params={"o": "100", "l": "100"},
        )

        self.assertEqual(
            result.requested_url,
            "https://example.test/archive?o=100&l=100",
        )

    def test_get_bytes_preserves_content_encoded_entity_bytes(self) -> None:
        encoded = b"encoded-gzip-representation"
        client = HttpClient()
        client.session = Mock()
        client.session.get.return_value = _Response(
            200,
            body=encoded,
            headers={"Content-Encoding": "gzip"},
        )

        result = client.get_bytes_with_metadata("https://example.test/raw")

        self.assertEqual(result.body, encoded)
        self.assertEqual(result.content_encoding, "gzip")
        self.assertEqual(result.body_representation, "http_entity_encoded")

    def test_get_bytes_rejects_oversized_stream(self) -> None:
        client = HttpClient()
        client.session = Mock()
        client.session.get.return_value = _Response(200, body=b"12345")

        with self.assertRaisesRegex(RuntimeError, "exceeds maximum_body_bytes"):
            client.get_bytes_with_metadata(
                "https://example.test/raw",
                maximum_body_bytes=4,
            )

    @patch("market_predictor.sources.http.time.sleep")
    def test_get_bytes_retries_408_429_and_all_server_errors(
        self,
        sleep: Mock,
    ) -> None:
        client = HttpClient()
        client.session = Mock()
        client.session.get.side_effect = [
            _Response(408, headers={"Retry-After": "0"}),
            _Response(429, headers={"Retry-After": "0"}),
            _Response(599, headers={"Retry-After": "0"}),
            _Response(200, body=b"ok"),
        ]

        result = client.get_bytes_with_metadata(
            "https://example.test/raw",
            retries=4,
        )

        self.assertEqual(result.body, b"ok")
        self.assertEqual(client.session.get.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [0.0] * 3,
        )

    @patch("market_predictor.sources.http.time.sleep")
    def test_get_bytes_does_not_retry_terminal_client_error(
        self,
        sleep: Mock,
    ) -> None:
        client = HttpClient()
        client.session = Mock()
        client.session.get.return_value = _Response(404)

        with self.assertRaisesRegex(RuntimeError, "status=404"):
            client.get_bytes_with_metadata(
                "https://example.test/missing",
                retries=5,
            )

        self.assertEqual(client.session.get.call_count, 1)
        sleep.assert_not_called()

    @patch("market_predictor.sources.http.time.sleep")
    def test_retry_after_controls_429_delay(self, sleep: Mock) -> None:
        client = HttpClient()
        client.session = Mock()
        client.session.get.side_effect = [
            _Response(429, headers={"Retry-After": "2"}),
            _Response(200, payload={"ok": True}),
        ]

        payload, _ = client.get_json_with_headers(
            "https://example.test",
            retries=3,
        )

        self.assertEqual(payload, {"ok": True})
        sleep.assert_called_once_with(2.0)
        self.assertEqual(client.session.get.call_count, 2)

    @patch("market_predictor.sources.http.random.uniform", return_value=0.0)
    @patch("market_predictor.sources.http.time.sleep")
    def test_server_error_uses_exponential_retry(
        self,
        sleep: Mock,
        random_uniform: Mock,
    ) -> None:
        del random_uniform
        client = HttpClient()
        client.session = Mock()
        client.session.get.side_effect = [
            _Response(503),
            _Response(503),
            _Response(200, payload={"ok": True}),
        ]

        payload, _ = client.get_json_with_headers(
            "https://example.test",
            retries=3,
            pause=1.0,
        )

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(
            [call.args[0] for call in sleep.call_args_list],
            [1.0, 2.0],
        )

    @patch("market_predictor.sources.http.time.sleep")
    def test_authentication_failure_is_not_retried(
        self,
        sleep: Mock,
    ) -> None:
        client = HttpClient()
        client.session = Mock()
        client.session.get.return_value = _Response(401)

        with self.assertRaisesRegex(RuntimeError, "status=401"):
            client.get_json_with_headers(
                "https://example.test",
                retries=5,
            )

        self.assertEqual(client.session.get.call_count, 1)
        sleep.assert_not_called()

    @patch("market_predictor.sources.http.time.sleep")
    def test_post_authentication_failure_is_not_retried(
        self,
        sleep: Mock,
    ) -> None:
        client = HttpClient()
        client.session = Mock()
        client.session.post.return_value = _Response(403)

        with self.assertRaisesRegex(RuntimeError, "status=403"):
            client.post_json_with_headers(
                "https://example.test",
                retries=5,
            )

        self.assertEqual(client.session.post.call_count, 1)
        sleep.assert_not_called()

    def test_retry_delay_caps_untrusted_provider_value(self) -> None:
        response = _Response(429, headers={"Retry-After": "999999"})

        self.assertEqual(
            _retry_delay(response, attempt=0, pause=1.0),
            120.0,
        )


class _Response(requests.Response):
    def __init__(
        self,
        status_code: int,
        *,
        payload: object | None = None,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
        url: str = "https://example.test",
        history: list[requests.Response] | None = None,
    ) -> None:
        super().__init__()
        self.status_code = status_code
        self._payload = payload
        self.headers = requests.structures.CaseInsensitiveDict(headers or {})
        self._content = body if body is not None else b"test response"
        self.raw = _RawBody(self._content)
        self.url = url
        self.history = history or []

    def json(self, **kwargs: Any) -> object:
        del kwargs
        return self._payload


class _RawBody(BytesIO):
    decode_content: bool = False


if __name__ == "__main__":
    unittest.main()
