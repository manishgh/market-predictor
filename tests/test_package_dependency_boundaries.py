from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "market_predictor"
PRODUCTION_PACKAGES = (
    "core",
    "sources",
    "evidence",
    "universe",
    "catalysts",
    "modeling",
    "swing",
    "intraday",
    "governance",
    "serving",
)
FORBIDDEN_DEPENDENCIES = ("market_predictor.research", "market_predictor.commands")


def test_production_packages_do_not_depend_on_research_or_command_adapters() -> None:
    violations: list[str] = []
    for package_name in PRODUCTION_PACKAGES:
        package = PACKAGE_ROOT / package_name
        if not package.exists():
            continue
        for path in package.rglob("*.py"):
            violations.extend(_forbidden_imports(path))

    assert not violations, "Production dependency violations:\n" + "\n".join(sorted(violations))


def test_chronology_named_v3_package_is_absent() -> None:
    assert not (PACKAGE_ROOT / "v3").exists()


def _forbidden_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        pytest.fail(f"Cannot inspect invalid Python module {path}: {exc}")

    violations: list[str] = []
    for node in ast.walk(tree):
        imported_names = _imported_names(node)
        for imported_name in imported_names:
            if imported_name.startswith(FORBIDDEN_DEPENDENCIES):
                relative_path = path.relative_to(PACKAGE_ROOT.parent)
                violations.append(f"{relative_path}:{node.lineno}: {imported_name}")
    return violations


def _imported_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if module == "market_predictor":
            return tuple(f"{module}.{alias.name}" for alias in node.names)
        return (module,)
    return ()
