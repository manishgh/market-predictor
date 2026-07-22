from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("configs/default.toml")


@lru_cache
def load_app_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def config_get(config: dict[str, Any], dotted_path: str, default: Any) -> Any:
    current: Any = config
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
