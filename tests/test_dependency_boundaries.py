from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_PACKAGES = {
    "azure-identity",
    "azure-storage-blob",
    "beautifulsoup4",
    "torch",
    "transformers",
    "xgboost",
    "yfinance",
}


def _declared_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0].lower()


def _locked_names(path: Path) -> set[str]:
    return set(
        re.findall(
            r"^([a-z0-9][a-z0-9._-]*)==",
            path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


class DependencyBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.configuration = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )
        cls.project = cls.configuration["project"]

    def test_build_backend_is_exactly_pinned(self) -> None:
        self.assertEqual(
            self.configuration["build-system"],
            {
                "requires": ["hatchling==1.31.0"],
                "build-backend": "hatchling.build",
            },
        )

    def test_production_dependencies_exclude_collection_and_training_packages(
        self,
    ) -> None:
        runtime = {
            _declared_name(requirement)
            for requirement in self.project["dependencies"]
        }

        self.assertTrue(OPTIONAL_PACKAGES.isdisjoint(runtime))

    def test_optional_dependency_surfaces_are_explicit(self) -> None:
        extras = {
            name: {_declared_name(requirement) for requirement in requirements}
            for name, requirements in self.project["optional-dependencies"].items()
        }
        collection = {
            "azure-identity",
            "azure-storage-blob",
            "beautifulsoup4",
            "yfinance",
        }
        training = {"torch", "transformers", "xgboost"}

        self.assertEqual(extras["collection"], collection)
        self.assertEqual(extras["training"], training)
        self.assertEqual(extras["ranking"], {"xgboost"})
        self.assertEqual(extras["research"], collection | training)
        self.assertEqual(
            extras["dev"],
            {
                "editables",
                "hatchling",
                "httpx",
                "mypy",
                "pip-audit",
                "pip-licenses",
                "ruff",
                "uv",
            },
        )

    def test_hash_locks_match_dependency_surfaces(self) -> None:
        lock_directory = ROOT / "requirements"
        production = _locked_names(lock_directory / "production.lock")
        collection = _locked_names(lock_directory / "collection.lock")
        training = _locked_names(lock_directory / "training.lock")
        validation = _locked_names(lock_directory / "validation.lock")
        development = _locked_names(lock_directory / "development.lock")

        self.assertTrue(OPTIONAL_PACKAGES.isdisjoint(production))
        self.assertTrue(
            {
                "azure-identity",
                "azure-storage-blob",
                "beautifulsoup4",
                "yfinance",
            }.issubset(collection)
        )
        self.assertTrue({"torch", "transformers", "xgboost"}.issubset(training))
        self.assertTrue(
            {
                "azure-identity",
                "azure-storage-blob",
                "beautifulsoup4",
                "editables",
                "httpx",
                "hatchling",
                "mypy",
                "pip-audit",
                "pip-licenses",
                "ruff",
                "uv",
                "xgboost",
                "yfinance",
            }.issubset(validation)
        )
        self.assertTrue({"torch", "transformers"}.isdisjoint(validation))
        self.assertTrue(
            OPTIONAL_PACKAGES.union(
                {
                    "editables",
                    "hatchling",
                    "httpx",
                    "mypy",
                    "pip-audit",
                    "pip-licenses",
                    "ruff",
                    "uv",
                }
            ).issubset(development)
        )

    def test_every_locked_requirement_has_a_hash(self) -> None:
        for lock_path in sorted((ROOT / "requirements").glob("*.lock")):
            text = lock_path.read_text(encoding="utf-8")
            self.assertIn(
                "#    python scripts/lock_dependencies.py",
                text,
                lock_path.name,
            )
            requirement_starts = list(
                re.finditer(
                    r"^[a-z0-9][a-z0-9._-]*==",
                    text,
                    flags=re.MULTILINE,
                )
            )
            self.assertGreater(len(requirement_starts), 0, lock_path.name)
            for index, start in enumerate(requirement_starts):
                end = (
                    requirement_starts[index + 1].start()
                    if index + 1 < len(requirement_starts)
                    else len(text)
                )
                self.assertIn("--hash=sha256:", text[start.start() : end])


if __name__ == "__main__":
    unittest.main()
