from __future__ import annotations

from collections.abc import Collection
from typing import Any

import typer


def filtered_app(
    source: typer.Typer,
    *,
    allowed_commands: Collection[str],
    help_text: str,
) -> typer.Typer:
    allowed = set(allowed_commands)
    target = typer.Typer(help=help_text)
    discovered: set[str] = set()
    for command in source.registered_commands:
        callback = command.callback
        name = command.name or (
            callback.__name__.replace("_", "-")
            if callback is not None
            else ""
        )
        if name in allowed:
            target.registered_commands.append(command)
            discovered.add(name)
    missing = allowed.difference(discovered)
    if missing:
        raise RuntimeError(f"CLI surface references unknown commands: {sorted(missing)}")
    return target


def command_names(app: typer.Typer) -> frozenset[str]:
    names: set[str] = set()
    for command in app.registered_commands:
        callback: Any = command.callback
        name = command.name or (
            callback.__name__.replace("_", "-")
            if callback is not None
            else ""
        )
        if name:
            names.add(name)
    return frozenset(names)
