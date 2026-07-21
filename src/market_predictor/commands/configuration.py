from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import TypeVar

import typer
from pydantic import BaseModel, ValidationError

ConfigModel = TypeVar("ConfigModel", bound=BaseModel)


def load_typed_config(path: Path | None, model: type[ConfigModel]) -> ConfigModel:
    if path is None:
        return model()
    if not path.exists():
        raise typer.BadParameter(f"configuration file does not exist: {path}")
    try:
        if path.suffix.lower() == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix.lower() in {".toml", ".tml"}:
            loaded = tomllib.loads(path.read_text(encoding="utf-8"))
        else:
            raise typer.BadParameter("configuration must be JSON or TOML")
        if not isinstance(loaded, dict):
            raise typer.BadParameter("configuration must contain an object/table")
        return model.model_validate(loaded)
    except (json.JSONDecodeError, tomllib.TOMLDecodeError, ValidationError) as exc:
        raise typer.BadParameter(f"invalid {model.__name__} configuration: {exc}") from exc
