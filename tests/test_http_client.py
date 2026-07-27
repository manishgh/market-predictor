from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import requests

from market_predictor.sources.http import HttpClient, _retry_delay


class HttpClientTests(unittest.TestCase):
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
        _: Mock,
    ) -> None:
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


class _Response:
    def __init__(
        self,
        status_code: int,
        *,
        payload: object | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = "test response"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response = requests.Response()
            response.status_code = self.status_code
            response._content = self.text.encode()
            raise requests.HTTPError(response=response)

    def json(self) -> object:
        return self._payload


if __name__ == "__main__":
    unittest.main()
