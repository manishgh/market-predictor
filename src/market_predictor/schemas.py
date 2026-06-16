from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class NewsEvent:
    ticker: str
    timestamp: datetime
    source: str
    title: str
    url: str | None = None
    summary: str | None = None
    text: str | None = None
    engagement_score: float | None = None
    engagement_comments: float | None = None
    engagement_upvote_ratio: float | None = None
    raw: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
