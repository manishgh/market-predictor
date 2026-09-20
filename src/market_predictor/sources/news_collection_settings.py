"""Local deployment configuration for the shared raw-news collection owner."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class NewsCollectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    owner: Literal["market_predictor"]
    coordination: Literal["single_host"]
    storage_root: str = Field(min_length=1)
    maximum_system_memory_percent: float = Field(gt=0, le=90)


def load_news_collection_settings(path: Path) -> tuple[NewsCollectionSettings, Path]:
    """Resolve one shared local root relative to its deployment config, not a plan."""
    with path.open("rb") as handle:
        raw = handle.read(16_385)
    if len(raw) > 16_384:
        raise ValueError("news collection config exceeds 16 KiB")
    settings = NewsCollectionSettings.model_validate(tomllib.loads(raw.decode("utf-8")))
    root = (path.resolve().parent / settings.storage_root).resolve()
    if root.as_posix().startswith("//"):
        raise ValueError("shared news collection requires a local single-host filesystem")
    if root == Path(root.anchor) or path.resolve().is_relative_to(root):
        raise ValueError("news storage must be a dedicated directory separate from its config")
    return settings, root
