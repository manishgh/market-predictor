"""Bounded integrity and memory primitives shared by publisher and verifier."""
from __future__ import annotations

import ast
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.system_memory import system_memory_snapshot
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.contracts.holding_materialization import SourcePin


def pins(root: Path, *groups: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    folded: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise DataReadinessError("relationship source pins require a mapping")
        for name, digest in group.items():
            pin = SourcePin(path=name, sha256=digest)
            key = inside(root, pin.path).relative_to(root).as_posix()
            if (key in result and result[key] != digest) or folded.get(key.casefold(), key) != key:
                raise DataReadinessError(f"relationship source pins conflict: {key}")
            result[key] = digest
            folded[key.casefold()] = key
    return dict(sorted(result.items()))


def check_files(root: Path, files: Mapping[str, str]) -> None:
    for name, digest in pins(root, files).items():
        path = inside(root, name)
        if not path.is_file() or file_sha256(path) != digest:
            raise DataReadinessError(f"relationship source changed: {name}")


def read_object(path: Path, digest: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size > 32 * 1024**2:
        raise DataReadinessError(f"relationship metadata missing or exceeds bound: {path}")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise DataReadinessError(f"relationship metadata hash mismatch: {path}")
    return parse_strict_json_object(payload, label=str(path))


def load_policy(root: Path, config: Path, digest: str) -> Any:
    from market_predictor.swing.contracts.return_relationship_publication import ReturnRelationshipPublicationPolicy

    value = read_object(inside(root, config), digest)
    return ReturnRelationshipPublicationPolicy.model_validate_json(json.dumps(value))


def current_implementation(root: Path) -> dict[str, str]:
    """Pin the local import closure without importing or executing archived code."""
    package = Path(__file__).resolve().parents[2]
    pending = [package / "swing/datasets/return_relationship_publication.py"]
    visited: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if not path.is_relative_to(root) or not path.is_file():
            raise DataReadinessError("relationship implementation must execute from the bound repository")
        visited.add(path)
        directory = path.parent
        while directory.is_relative_to(package):
            initializer = directory / "__init__.py"
            if initializer.is_file() and initializer not in visited:
                pending.append(initializer)
            if directory == package:
                break
            directory = directory.parent
        for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    anchor = path.parent
                    for _ in range(node.level - 1):
                        anchor = anchor.parent
                    if not anchor.is_relative_to(package):
                        raise DataReadinessError("relationship implementation relative import escapes package")
                    prefix = ".".join(("market_predictor", *anchor.relative_to(package).parts))
                    module = prefix + ("." + node.module if node.module else "")
                else:
                    module = node.module or ""
                modules = [module, *(module + "." + alias.name for alias in node.names if alias.name != "*")]
            else:
                modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            for module in modules:
                if module == "market_predictor" or module.startswith("market_predictor."):
                    child = package.joinpath(*module.split(".")[1:])
                    candidate = child.with_suffix(".py")
                    if candidate.is_file():
                        pending.append(candidate)
                    elif (child / "__init__.py").is_file():
                        pending.append(child / "__init__.py")
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(visited)}


def guard(maximum_system_used_percent: float) -> None:
    assert_memory_budget(stage="return relationship publication", hard_budget_gib=5.0, headroom_gib=0.75)
    memory = system_memory_snapshot()
    if memory is None or memory.used_percent >= maximum_system_used_percent:
        raise MemoryBudgetError("relationship publication requires system memory below the configured percentage")


def replace_checkpoint(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    try:
        write_json_object(temporary, value)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def assert_closed(value: Mapping[str, Any]) -> None:
    if any(value.get(name) is not False for name in ("training_eligible", "promotion_eligible", "serving_eligible")):
        raise DataReadinessError("relationship publication cannot grant model admission")
