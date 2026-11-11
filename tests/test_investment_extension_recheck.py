"""Synthetic files verify streaming hashes and unchanged per-file resource refusal."""
from pathlib import Path

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.investment import source_extensions as extension


def source_files(root: Path, sizes: list[int]) -> dict[str, str]:
    files = {}
    for index, size in enumerate(sizes):
        path = root / f"source-{index}.bin"
        with path.open("wb") as stream:
            stream.truncate(size)
        files[path.name] = file_sha256(path)
    return files


@pytest.mark.parametrize("sizes,cleanups", [
    ([1] * 33, [True] + [False] * 31 + [True]),
    ([5 * 1024**2, 3 * 1024**2, 1], [True, False, True]),
    ([1, 9 * 1024**2, 1], [True, True, True]),
    ([], []),
])
def test_every_hash_has_guard_and_bounded_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    sizes: list[int], cleanups: list[bool]) -> None:
    files = source_files(tmp_path, sizes)
    events = []

    def guard(stage: str, *, reclaim_unused_memory: bool = True) -> None:
        assert stage == "investment extension source hash verification"
        events.append(("guard", reclaim_unused_memory))

    def hashed(path: Path) -> str:
        events.append(("hash", path.name))
        return file_sha256(path)

    monkeypatch.setattr(extension, "guard", guard)
    monkeypatch.setattr(extension, "file_sha256", hashed)
    extension.recheck_extension_sources(tmp_path, files)
    assert events == [event for name, cleanup in zip(files, cleanups, strict=True)
        for event in (("guard", cleanup), ("hash", name))]


@pytest.mark.parametrize("failure", ["tamper", "memory"])
def test_lightweight_guard_path_still_refuses_before_later_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    failure: str) -> None:
    files = source_files(tmp_path, [1, 1, 1])
    if failure == "tamper":
        (tmp_path / "source-1.bin").write_bytes(b"x")
    events = []

    def guard(stage: str, *, reclaim_unused_memory: bool = True) -> None:
        events.append(("guard", reclaim_unused_memory))
        if failure == "memory" and not reclaim_unused_memory:
            raise MemoryBudgetError("synthetic lightweight-path pressure")

    def hashed(path: Path) -> str:
        events.append(("hash", path.name))
        return file_sha256(path)

    monkeypatch.setattr(extension, "guard", guard)
    monkeypatch.setattr(extension, "file_sha256", hashed)
    expected = DataReadinessError if failure == "tamper" else MemoryBudgetError
    with pytest.raises(expected):
        extension.recheck_extension_sources(tmp_path, files)
    assert events == [("guard", True), ("hash", "source-0.bin"), ("guard", False)] + (
        [("hash", "source-1.bin")] if failure == "tamper" else [])


def test_outside_path_cannot_be_hashed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("No hash is permitted outside the source root")

    monkeypatch.setattr(extension, "file_sha256", forbidden)
    with pytest.raises(DataReadinessError):
        extension.recheck_extension_sources(tmp_path, {str(tmp_path.parent / "outside.bin"): "a" * 64})
