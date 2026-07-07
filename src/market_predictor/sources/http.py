from __future__ import annotations

import time
from typing import Any

import requests


def _http_error_message(method: str, url: str, error: Exception | None) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        response = error.response
        body = response.text[:300].replace("\n", " ").replace("\r", " ")
        return f"{method} failed: {url} status={response.status_code} body={body}"
    if error is not None:
        return f"{method} failed: {url} error={error}"
    return f"{method} failed: {url}"


class HttpClient:
    def __init__(self, user_agent: str = "market-predictor/0.1", timeout: int = 30) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        pause: float = 1.0,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 429 and attempt < retries - 1:
                    time.sleep(pause * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(pause * (attempt + 1))
        raise RuntimeError(_http_error_message("GET", url, last_error)) from last_error

    def get_json_with_headers(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        pause: float = 1.0,
    ) -> tuple[Any, dict[str, str]]:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 429 and attempt < retries - 1:
                    time.sleep(pause * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json(), dict(response.headers)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(pause * (attempt + 1))
        raise RuntimeError(_http_error_message("GET", url, last_error)) from last_error

    def post_json_with_headers(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        pause: float = 1.0,
    ) -> tuple[Any, dict[str, str]]:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.post(
                    url,
                    json=payload or {},
                    headers=headers,
                    timeout=self.timeout,
                )
                if response.status_code == 429 and attempt < retries - 1:
                    time.sleep(pause * (attempt + 1))
                    continue
                response.raise_for_status()
                return response.json(), dict(response.headers)
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(pause * (attempt + 1))
        raise RuntimeError(_http_error_message("POST", url, last_error)) from last_error
