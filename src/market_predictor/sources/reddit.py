from __future__ import annotations

import re
from datetime import UTC, datetime

import pandas as pd
import requests

from market_predictor.config import Settings
from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpClient


class RedditSource:
    token_url = "https://www.reddit.com/api/v1/access_token"
    api_base_url = "https://oauth.reddit.com"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.subreddits = settings.reddit_subreddit_list
        self.client = HttpClient(user_agent=settings.reddit_user_agent)
        self._token: str | None = None

    def _access_token(self) -> str:
        if self._token:
            return self._token
        if not self.settings.has_reddit:
            raise ValueError(
                "Reddit API credentials are required: REDDIT_CLIENT_ID, "
                "REDDIT_CLIENT_SECRET, REDDIT_USERNAME, REDDIT_PASSWORD."
            )
        response = requests.post(
            self.token_url,
            auth=(self.settings.reddit_client_id or "", self.settings.reddit_client_secret or ""),
            data={
                "grant_type": "password",
                "username": self.settings.reddit_username,
                "password": self.settings.reddit_password,
            },
            headers={"User-Agent": self.settings.reddit_user_agent},
            timeout=30,
        )
        response.raise_for_status()
        self._token = response.json()["access_token"]
        return self._token

    def fetch_mentions(self, ticker: str, start: datetime, limit_per_subreddit: int | None = None) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        limit = limit_per_subreddit or self.settings.reddit_limit_per_subreddit
        terms = f'"${ticker.upper()}" OR "{ticker.upper()} stock"'
        headers = {"Authorization": f"bearer {self._access_token()}"}
        for subreddit in self.subreddits:
            url = f"{self.api_base_url}/r/{subreddit}/search"
            payload = self.client.get_json(
                url,
                params={
                    "q": terms,
                    "restrict_sr": "1",
                    "sort": "new",
                    "t": self.settings.reddit_search_time_filter,
                    "limit": limit,
                },
                headers=headers,
                retries=2,
                pause=2.0,
            )
            for child in payload.get("data", {}).get("children", []):
                data = child.get("data", {})
                created = pd.to_datetime(data.get("created_utc"), unit="s", utc=True).to_pydatetime()
                if created < start.astimezone(UTC):
                    continue
                events.append(
                    NewsEvent(
                        ticker=ticker.upper(),
                        timestamp=created,
                        source=f"reddit:post:r/{subreddit}",
                        title=data.get("title") or "",
                        url=f"https://www.reddit.com{data.get('permalink', '')}",
                        summary=data.get("selftext")[:1000] if data.get("selftext") else None,
                        text=" ".join([data.get("title") or "", data.get("selftext") or ""]).strip(),
                        engagement_score=float(data.get("score") or 0),
                        engagement_comments=float(data.get("num_comments") or 0),
                        engagement_upvote_ratio=float(data.get("upvote_ratio") or 0),
                        raw={
                            "kind": "post",
                            "id": data.get("id"),
                            "score": data.get("score"),
                            "num_comments": data.get("num_comments"),
                            "upvote_ratio": data.get("upvote_ratio"),
                        },
                    )
                )
                if self.settings.reddit_include_post_comments and data.get("id"):
                    events.extend(
                        self._fetch_matching_comments(
                            ticker=ticker,
                            subreddit=subreddit,
                            post_id=str(data["id"]),
                            start=start,
                            headers=headers,
                        )
                    )
        return events

    def _fetch_matching_comments(
        self,
        *,
        ticker: str,
        subreddit: str,
        post_id: str,
        start: datetime,
        headers: dict[str, str],
    ) -> list[NewsEvent]:
        url = f"{self.api_base_url}/r/{subreddit}/comments/{post_id}"
        payload = self.client.get_json(
            url,
            params={
                "limit": self.settings.reddit_comments_per_post,
                "sort": "top",
                "raw_json": 1,
            },
            headers=headers,
            retries=2,
            pause=2.0,
        )
        comments_listing = payload[1] if isinstance(payload, list) and len(payload) > 1 else {}
        events: list[NewsEvent] = []
        for child in comments_listing.get("data", {}).get("children", []):
            data = child.get("data", {})
            body = data.get("body") or ""
            if not body or not self._mentions_ticker(body, ticker):
                continue
            created = pd.to_datetime(data.get("created_utc"), unit="s", utc=True).to_pydatetime()
            if created < start.astimezone(UTC):
                continue
            events.append(
                NewsEvent(
                    ticker=ticker.upper(),
                    timestamp=created,
                    source=f"reddit:comment:r/{subreddit}",
                    title=f"Reddit comment mentioning {ticker.upper()}",
                    url=f"https://www.reddit.com{data.get('permalink', '')}",
                    summary=body[:1000],
                    text=body,
                    engagement_score=float(data.get("score") or 0),
                    engagement_comments=0.0,
                    engagement_upvote_ratio=None,
                    raw={
                        "kind": "comment",
                        "id": data.get("id"),
                        "parent_id": data.get("parent_id"),
                        "score": data.get("score"),
                    },
                )
            )
        return events

    def _mentions_ticker(self, text: str, ticker: str) -> bool:
        ticker_upper = ticker.upper()
        if re.search(rf"(?i)(^|\W)\${re.escape(ticker_upper)}($|\W)", text):
            return True
        if ticker_upper in self.settings.reddit_ticker_false_positive_stoplist:
            return False
        return bool(re.search(rf"(?<![A-Za-z]){re.escape(ticker_upper)}(?![A-Za-z])", text))
