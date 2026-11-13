from __future__ import annotations

from pathlib import Path

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.paths import EvidencePaths


@pytest.fixture
def roots(tmp_path: Path) -> tuple[Path, Path]:
    project = tmp_path / "project"
    storage = tmp_path / "alternate data"
    (project / "data/raw").mkdir(parents=True)
    (project / "configs").mkdir()
    (storage / "raw").mkdir(parents=True)
    return project, storage


def test_relocated_evidence_preserves_logical_identity(roots: tuple[Path, Path]) -> None:
    project, storage = roots
    paths = EvidencePaths(project, storage)
    logical = project / "data/raw/bars.json"
    physical = storage / "raw/bars.json"
    logical.write_text("old copy", encoding="utf-8")
    physical.write_text("active copy", encoding="utf-8")
    config = project / "configs" / "policy.toml"
    config.touch()

    for spelling in ("data/raw/bars.json", r"data\raw\bars.json", logical, physical):
        assert paths.identity(spelling) == logical
        assert paths.relative(spelling) == Path("data/raw/bars.json")
        assert paths.read(spelling) == physical
    assert paths.read(config) == config
    assert paths.directory("data/raw") == storage / "raw"
    assert paths.write("data/research/results/outcomes.json") == storage / "research/results/outcomes.json"
    assert not (storage / "research/results").exists()

    physical.unlink()
    with pytest.raises(DataReadinessError, match="missing"):
        paths.read(logical)
    physical.parent.rmdir()
    storage.rmdir()
    with pytest.raises(DataReadinessError, match="unavailable"):
        paths.read(logical)


def test_default_uses_project_data(roots: tuple[Path, Path]) -> None:
    project, _ = roots
    source = project / "data/raw/bars.json"
    source.touch()
    assert EvidencePaths(project).read("data/raw/bars.json") == source
    assert EvidencePaths(project, project / "data").read("data/raw/bars.json") == source
    assert EvidencePaths(project).write("data/research/report.json") == project / "data/research/report.json"


@pytest.mark.parametrize("namespace", ["raw", "features", "research", "canonical", "external", "labels", "cache"])
def test_selected_data_namespaces_move(roots: tuple[Path, Path], namespace: str) -> None:
    project, storage = roots
    paths = EvidencePaths(project, storage)
    destination = storage / namespace / "artifact.json"
    assert paths.write(f"data/{namespace}/artifact.json") == destination
    assert paths.identity(destination) == project / "data" / namespace / "artifact.json"


@pytest.mark.parametrize("namespace", [
    "models", "artifacts", "reports", "evidence", "runtime", "universe", "usage",
    "research/implementation_snapshots", "unknown",
])
def test_other_namespaces_remain_local(roots: tuple[Path, Path], namespace: str) -> None:
    project, storage = roots
    paths = EvidencePaths(project, storage)
    local = project / "data" / namespace / "artifact.json"
    assert paths.write(f"data/{namespace}/artifact.json") == local
    local.parent.mkdir(parents=True)
    local.touch()
    assert paths.read(local) == local
    with pytest.raises(DataReadinessError, match="outside the relocated dataset namespaces"):
        paths.identity(storage / namespace / "artifact.json")


@pytest.mark.parametrize("raw", [
    "data/../configs/policy.toml",
    r"data\..\configs\policy.toml",
    r"C:data\bars.json",
    r"\data\bars.json",
    r"\\server\share\bars.json",
    r"\\?\C:\data\bars.json",
    "data/bars.json:alternate",
    "data/NUL.txt",
    "data/COM¹",
    "data/bars.json ",
    "data/...",
    "data/a\x00b",
])
def test_unsafe_path_spellings_are_rejected(roots: tuple[Path, Path], raw: str) -> None:
    with pytest.raises(DataReadinessError, match="unsafe"):
        EvidencePaths(*roots).identity(raw)


def test_rejects_unrelated_paths_and_overlapping_authorities(roots: tuple[Path, Path]) -> None:
    project, storage = roots
    paths = EvidencePaths(project, storage)
    with pytest.raises(DataReadinessError, match="outside the project"):
        paths.identity(storage.parent / "unrelated.json")
    with pytest.raises(DataReadinessError, match="outside the relocated dataset namespaces"):
        paths.write(storage)
    for overlap in (project.parent, project, project / "configs"):
        with pytest.raises(DataReadinessError, match="outside the project"):
            EvidencePaths(project, overlap)


def test_rejects_linked_root_and_child_paths(roots: tuple[Path, Path]) -> None:
    project, storage = roots
    outside = storage.parent / "outside"
    outside.mkdir()
    (outside / "bars.json").touch()
    linked = storage / "raw/linked"
    try:
        linked.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"cannot create a directory symlink: {exc}")

    with pytest.raises(DataReadinessError, match="symlink or reparse"):
        EvidencePaths(project, linked)
    paths = EvidencePaths(project, storage)
    with pytest.raises(DataReadinessError, match="symlink or reparse"):
        paths.read("data/raw/linked/bars.json")
    with pytest.raises(DataReadinessError, match="symlink or reparse"):
        paths.directory("data/raw/linked")
    with pytest.raises(DataReadinessError, match="symlink or reparse"):
        paths.write("data/raw/linked/results.json")
